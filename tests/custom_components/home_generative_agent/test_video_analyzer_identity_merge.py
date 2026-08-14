# ruff: noqa: S101
"""
Tests for batch-level identity consolidation (issue #543).

Face recognition flaps between a known name and "Unknown Person" across frames
of one human, so the summarizer reports a phantom second person. The merge in
VideoAnalyzer._merge_unknown_faces renames batch-local unknowns to the batch's
single known identity, strictly:
- exactly one known name in the batch,
- no frame with two or more detected faces (genuine-companion guard),
- the known person is the unknown embedding's NEAREST gallery match, within
  VIDEO_ANALYZER_FACE_MERGE_THRESHOLD.

Also covers: the nearest_match DAO method, the three-site
frame_descriptions/frame_hits alignment, propagation of the merged list, and
the Sentinel regression guarantee (unknown-person rules key on
recognized_people being EMPTY, so the merge cannot change rule firing).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.agent.tools import VLM_ERROR_CAPTION
from custom_components.home_generative_agent.const import (
    VIDEO_ANALYZER_FACE_MERGE_THRESHOLD,
)
from custom_components.home_generative_agent.core.person_gallery import (
    FACE_EMBEDDING_DIMS,
    FACE_RECOGNITION_THRESHOLD,
    PersonGalleryDAO,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    FaceHit,
    VideoAnalyzer,
)
from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_night_home import (
    UnknownPersonAtNightWhileHomeRule,
)
from custom_components.home_generative_agent.snapshot.schema import (
    CameraActivity,
    FullStateSnapshot,
    validate_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant

# ---------------------------------------------------------------------------
# Override autouse fixtures from pytest-homeassistant-custom-component
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks explicitly awaited; no HA resources to clean up."""


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CAMERA = "camera.playroomdoor"
_KNOWN = "Nico"
_EMB = [0.25, 0.5, 0.75]
_GOOD_DISTANCE = VIDEO_ANALYZER_FACE_MERGE_THRESHOLD - 0.3
_BAD_DISTANCE = VIDEO_ANALYZER_FACE_MERGE_THRESHOLD + 0.1


@pytest.fixture
def entry() -> MagicMock:
    e = MagicMock()
    e.runtime_data.options = {}
    e.runtime_data.person_gallery = None
    return e


@pytest.fixture
def va(entry: MagicMock) -> VideoAnalyzer:
    return VideoAnalyzer(MagicMock(), entry)


def _dao(distance: object = None) -> MagicMock:
    """
    Gallery DAO stub whose nearest_match resolves (or raises).

    A float is wrapped as (known-name, distance); a tuple passes through as
    a (name, distance) pair; None means an empty gallery; an exception (any
    BaseException — CancelledError included) becomes the side effect.
    """
    dao = MagicMock()
    if isinstance(distance, BaseException):
        dao.nearest_match = AsyncMock(side_effect=distance)
    elif isinstance(distance, float):
        dao.nearest_match = AsyncMock(return_value=(_KNOWN, distance))
    else:
        dao.nearest_match = AsyncMock(return_value=distance)
    return dao


def _stub_snapshots(
    va: VideoAnalyzer,
    replies: Sequence[tuple[dict[str, list[str]], list[FaceHit]]],
) -> None:
    """Replace _process_snapshot with canned (caption-dict, hits) tuples."""
    reply_iter = iter(replies)

    async def fake_process(
        path: Path,  # noqa: ARG001
        camera_id: str,  # noqa: ARG001
        prev_text: str | None = None,  # noqa: ARG001
    ) -> tuple[dict[str, list[str]], list[FaceHit]]:
        return next(reply_iter)

    va._process_snapshot = AsyncMock(side_effect=fake_process)  # type: ignore[method-assign]


def _frame(
    caption: str, hits: list[FaceHit]
) -> tuple[dict[str, list[str]], list[FaceHit]]:
    """Build a _process_snapshot reply matching its production contract."""
    return {caption: [h.name for h in hits]}, hits


def _ordered(n: int) -> list[tuple[Path, int]]:
    return [(Path(f"snap_{i}.jpg"), 1000 + 8 * i) for i in range(n)]


def _identities(descs: list[dict[str, list[str]]]) -> list[list[str]]:
    return [v for d in descs for v in d.values()]


def _known_hit() -> FaceHit:
    return FaceHit(name=_KNOWN, embedding=list(_EMB))


def _unknown_hit() -> FaceHit:
    return FaceHit(name="Unknown Person", embedding=list(_EMB))


# ---------------------------------------------------------------------------
# Consolidation decision table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_within_distance(va: VideoAnalyzer, entry: MagicMock) -> None:
    """AC1: single known + unknown within bound merges into the known name."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN]
    assert _identities(descs) == [[_KNOWN], [_KNOWN]]
    entry.runtime_data.person_gallery.nearest_match.assert_awaited_once_with(_EMB)
    assert va._metrics[_CAMERA].unknown_merged == 1


@pytest.mark.asyncio
async def test_refused_when_distance_beyond_bound(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC2: distance at/over the bound keeps both identities unchanged."""
    entry.runtime_data.person_gallery = _dao(_BAD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert _identities(descs) == [[_KNOWN], ["Unknown Person"]]
    assert va._metrics[_CAMERA].unknown_merged == 0
    assert va._metrics[_CAMERA].unknown_merge_refused_distance == 1


@pytest.mark.asyncio
async def test_refused_on_same_frame_cooccurrence(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC3: a frame with two detected faces is the companion signal; no merge."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame(
                "Two people stand at the entrance.",
                [_known_hit(), _unknown_hit()],
            ),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert "Unknown Person" in _identities(descs)[0]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_cooccurrence == 1
    assert va._metrics[_CAMERA].unknown_merged == 0


@pytest.mark.asyncio
async def test_refused_with_two_known_names(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC4: two known names plus an unknown refuses (closest-match is v2)."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame(
                "A woman crosses the yard.", [FaceHit(name="Anna", embedding=[0.1])]
            ),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == ["Anna", _KNOWN, "Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_multi_known == 1
    assert va._metrics[_CAMERA].unknown_merged == 0


@pytest.mark.asyncio
async def test_refused_with_zero_known_names(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """
    Unknown-only batches have no merge target; output unchanged, no count.

    Stranger-only batches are the common case; counting them as refusals
    would drown the tuning counters (review decision D2).
    """
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(1))

    assert recognized == ["Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_multi_known == 0
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 0


@pytest.mark.asyncio
async def test_all_indeterminate_increments_nothing(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC5: no unknowns present -> zero DAO calls and zero merge counters."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("An empty driveway.", [FaceHit(name="Indeterminate")]),
            _frame("A cat crosses the driveway.", [FaceHit(name="Indeterminate")]),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == ["Indeterminate"]
    assert _identities(descs) == [["Indeterminate"], ["Indeterminate"]]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    m = va._metrics[_CAMERA]
    assert m.unknown_merged == 0
    assert m.unknown_merge_refused_cooccurrence == 0
    assert m.unknown_merge_refused_distance == 0
    assert m.unknown_merge_refused_multi_known == 0
    assert m.unknown_merge_refused_no_lookup == 0


@pytest.mark.asyncio
async def test_refused_when_person_unenrolled_mid_batch(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC6: nearest_match returning None (empty gallery) refuses."""
    entry.runtime_data.person_gallery = _dao(None)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 1
    assert va._metrics[_CAMERA].unknown_merge_refused_distance == 0


@pytest.mark.asyncio
async def test_dao_error_keeps_unknown_and_batch_survives(
    va: VideoAnalyzer,
    entry: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A DAO/DB exception keeps 'Unknown Person'; the batch must not fail."""
    entry.runtime_data.person_gallery = _dao(RuntimeError("db down"))
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 1
    assert "nearest_match failed" in caplog.text


@pytest.mark.asyncio
async def test_multiple_unknown_frames_all_merge(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """Distance checks and the merged counter are per unknown face."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
            _frame("A man leaves through the gate.", [_unknown_hit()]),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == [_KNOWN]
    assert _identities(descs) == [[_KNOWN], [_KNOWN], [_KNOWN]]
    assert va._metrics[_CAMERA].unknown_merged == 2


# ---------------------------------------------------------------------------
# Three-site frame/hit alignment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alignment_vlm_error_fallback_frame_carries_embedding(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """A VLM-error frame kept via the person fallback still merges correctly."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    unknown_emb = [0.9, 0.8, 0.7]
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame(
                VLM_ERROR_CAPTION,
                [FaceHit(name="Unknown Person", embedding=list(unknown_emb))],
            ),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN]
    assert _identities(descs) == [[_KNOWN], [_KNOWN]]
    entry.runtime_data.person_gallery.nearest_match.assert_awaited_once_with(
        unknown_emb
    )


@pytest.mark.asyncio
async def test_alignment_sentinel_keep_frame_carries_embedding(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """A sentinel frame kept for its detected face still merges correctly."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    unknown_emb = [0.6, 0.5, 0.4]
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame(
                "Scene unchanged.",
                [FaceHit(name="Unknown Person", embedding=list(unknown_emb))],
            ),
        ],
    )

    descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN]
    assert _identities(descs) == [[_KNOWN], [_KNOWN]]
    entry.runtime_data.person_gallery.nearest_match.assert_awaited_once_with(
        unknown_emb
    )


# ---------------------------------------------------------------------------
# Propagation: merged list feeds summary input and listener dispatch source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merged_identities_reach_summary_and_last_recognized(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """AC7: summary input and _last_recognized (sensor/notify source) merge."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )
    va._summarize = AsyncMock(return_value="Nico walks toward the entrance.")  # type: ignore[method-assign]
    va._finalize = AsyncMock()  # type: ignore[method-assign]

    await va._analyze_and_finalize(_CAMERA, _ordered(2))

    assert va._last_recognized[_CAMERA] == [_KNOWN]
    assert va._summarize.await_args is not None
    summary_descs = va._summarize.await_args.args[1]
    assert _identities(summary_descs) == [[_KNOWN], [_KNOWN]]


# ---------------------------------------------------------------------------
# Sentinel regression: rules key on recognized_people being EMPTY
# ---------------------------------------------------------------------------


def _activity(recognized_people: list[str]) -> CameraActivity:
    return {
        "camera_entity_id": _CAMERA,
        "area": "Playroom",
        "last_activity": "2026-02-01T02:00:00+00:00",
        "motion_entities": ["binary_sensor.playroomdoor_motion"],
        "vmd_entities": [],
        "snapshot_summary": "A person stands at the door.",
        "recognized_people": recognized_people,
        "latest_path": None,
    }


def _sentinel_snapshot(
    recognized_people: list[str], *, anyone_home: bool
) -> FullStateSnapshot:
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2026-02-01T02:00:00+00:00",
            "entities": [],
            "camera_activity": [_activity(recognized_people)],
            "derived": {
                "now": "2026-02-01T02:00:00+00:00",
                "timezone": "UTC",
                "is_night": True,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


@pytest.mark.parametrize(
    ("recognized_people", "fires"),
    [
        ([], True),
        ([_KNOWN], False),  # post-merge shape
        ([_KNOWN, "Unknown Person"], False),  # refused-merge shape
    ],
)
def test_builtin_unknown_person_rule_firing_unchanged(
    recognized_people: list[str],
    *,
    fires: bool,
) -> None:
    """AC8: merge output shapes cannot change the built-in rule's firing."""
    snapshot = _sentinel_snapshot(recognized_people, anyone_home=True)
    findings = UnknownPersonAtNightWhileHomeRule().evaluate(snapshot)
    assert bool(findings) is fires


@pytest.mark.parametrize(
    ("recognized_people", "fires"),
    [
        ([], True),
        ([_KNOWN], False),  # post-merge shape
        ([_KNOWN, "Unknown Person"], False),  # refused-merge shape
    ],
)
def test_dynamic_unknown_person_rule_firing_unchanged(
    recognized_people: list[str],
    *,
    fires: bool,
) -> None:
    """AC8: merge output shapes cannot change dynamic-rule firing either."""
    snapshot = _sentinel_snapshot(recognized_people, anyone_home=True)
    rules = [
        {
            "rule_id": "unknown_rule_1",
            "template_id": "unknown_person_camera_when_home",
            "params": {"camera_entity_id": _CAMERA},
            "severity": "medium",
            "confidence": 0.6,
            "is_sensitive": False,
            "suggested_actions": [],
        }
    ]
    findings = evaluate_dynamic_rules(snapshot, rules)
    assert bool(findings) is fires


# ---------------------------------------------------------------------------
# PersonGalleryDAO.nearest_match
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, row: dict[str, object] | None) -> None:
        self.row = row
        self.executed: tuple[str, tuple[object, ...]] | None = None

    async def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed = (sql, params)

    async def fetchone(self) -> dict[str, object] | None:
        return self.row


class _FakeAcm:
    def __init__(self, obj: object) -> None:
        self._obj = obj

    async def __aenter__(self) -> object:
        return self._obj

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connection(self) -> _FakeAcm:
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=_FakeAcm(self._cursor))
        return _FakeAcm(conn)


def _valid_embedding() -> list[float]:
    return [1.0] + [0.0] * 511


@pytest.mark.asyncio
async def test_nearest_match_returns_name_and_distance(hass: HomeAssistant) -> None:
    """Rows present: the globally nearest (name, distance) pair comes back."""
    cursor = _FakeCursor({"name": _KNOWN, "distance": 0.42})
    dao = PersonGalleryDAO(cast("Any", _FakePool(cursor)), hass)

    nearest = await dao.nearest_match(_valid_embedding())

    assert nearest is not None
    assert nearest[0] == _KNOWN
    assert nearest[1] == pytest.approx(0.42)
    assert cursor.executed is not None
    sql, _params = cursor.executed
    assert "ORDER BY distance" in sql


@pytest.mark.asyncio
async def test_nearest_match_none_when_gallery_empty(hass: HomeAssistant) -> None:
    """An empty gallery yields None instead of raising."""
    cursor = _FakeCursor(None)
    dao = PersonGalleryDAO(cast("Any", _FakePool(cursor)), hass)

    assert await dao.nearest_match(_valid_embedding()) is None


@pytest.mark.asyncio
async def test_nearest_match_none_when_distance_null(hass: HomeAssistant) -> None:
    """A NULL distance (defensive) also yields None instead of raising."""
    cursor = _FakeCursor({"name": _KNOWN, "distance": None})
    dao = PersonGalleryDAO(cast("Any", _FakePool(cursor)), hass)

    assert await dao.nearest_match(_valid_embedding()) is None


@pytest.mark.asyncio
async def test_nearest_match_rejects_bad_dimensions(hass: HomeAssistant) -> None:
    """Dimension validation propagates, matching recognize_person's contract."""
    cursor = _FakeCursor({"name": _KNOWN, "distance": 0.1})
    dao = PersonGalleryDAO(cast("Any", _FakePool(cursor)), hass)

    with pytest.raises(ValueError, match="dims"):
        await dao.nearest_match([1.0, 2.0])


# ---------------------------------------------------------------------------
# Coverage-audit additions: boundary, guards, and modified existing behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refused_at_exact_threshold(va: VideoAnalyzer, entry: MagicMock) -> None:
    """Distance == threshold must refuse (strict less-than semantics)."""
    entry.runtime_data.person_gallery = _dao(VIDEO_ANALYZER_FACE_MERGE_THRESHOLD)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert va._metrics[_CAMERA].unknown_merge_refused_distance == 1
    assert va._metrics[_CAMERA].unknown_merged == 0


@pytest.mark.asyncio
async def test_misaligned_outer_lists_refuse_merge(
    va: VideoAnalyzer,
    entry: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame/hit outer-length mismatch skips the merge and mutates nothing."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    descs: list[dict[str, list[str]]] = [
        {"c1": [_KNOWN]},
        {"c2": ["Unknown Person"]},
    ]
    hits = [[_known_hit()]]

    await va._merge_unknown_faces(_CAMERA, descs, hits)

    assert descs[1]["c2"] == ["Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert "misalignment" in caplog.text


@pytest.mark.asyncio
async def test_misaligned_inner_lists_refuse_merge(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """A per-frame names/hits length mismatch also refuses outright."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    descs: list[dict[str, list[str]]] = [
        {"c1": [_KNOWN]},
        {"c2": ["Unknown Person"]},
    ]
    hits = [[_known_hit()], [_unknown_hit(), _unknown_hit()]]

    await va._merge_unknown_faces(_CAMERA, descs, hits)

    assert descs[1]["c2"] == ["Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_dao_keeps_unknown(va: VideoAnalyzer, entry: MagicMock) -> None:
    """Gallery unavailable at merge time: refuse cleanly, batch survives."""
    entry.runtime_data.person_gallery = None
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 1


@pytest.mark.asyncio
async def test_embeddingless_unknown_keeps_unknown(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """An unknown hit without an embedding refuses without a DAO call."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [FaceHit(name="Unknown Person")]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 1


@pytest.mark.asyncio
async def test_two_unknowns_in_one_frame_refuse(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """Two unknown faces sharing a frame are two people; never merge."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("Two people near the door.", [_unknown_hit(), _unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert "Unknown Person" in recognized
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_cooccurrence == 1


@pytest.mark.asyncio
async def test_known_plus_indeterminate_frame_still_merges(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """Indeterminate placeholders are not detected faces; merge proceeds."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame(
                "A man and a shadow.", [_known_hit(), FaceHit(name="Indeterminate")]
            ),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == ["Indeterminate", _KNOWN]
    assert va._metrics[_CAMERA].unknown_merged == 1


@pytest.mark.asyncio
async def test_cancelled_error_propagates(va: VideoAnalyzer, entry: MagicMock) -> None:
    """CancelledError from the DAO must escape the merge, not be swallowed."""
    entry.runtime_data.person_gallery = _dao(asyncio.CancelledError())
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    with pytest.raises(asyncio.CancelledError):
        await va._process_batch(_CAMERA, _ordered(2))


def test_m_inc_unknown_key_and_non_int_field_ignored(va: VideoAnalyzer) -> None:
    """_m_inc increments int counters only; other keys silently no-op."""
    va._m_inc(_CAMERA, "semaphore_timeouts")
    assert va._metrics[_CAMERA].semaphore_timeouts == 1

    va._m_inc(_CAMERA, "no_such_counter")  # must not raise or create fields
    va._m_inc(_CAMERA, "lat_ms")  # non-int field must be left intact
    assert isinstance(va._metrics[_CAMERA].lat_ms, deque)
    assert not hasattr(va._metrics[_CAMERA], "no_such_counter")


@pytest.mark.asyncio
async def test_metrics_flush_logs_and_resets_merge_counters(
    va: VideoAnalyzer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The hourly flush reports the five merge counters and zeroes them."""
    m = va._metrics[_CAMERA]
    m.unknown_merged = 3
    m.unknown_merge_refused_cooccurrence = 2
    m.unknown_merge_refused_distance = 1
    m.unknown_merge_refused_multi_known = 4
    m.unknown_merge_refused_no_lookup = 5

    with caplog.at_level(logging.INFO):
        await va._metrics_flush_report(cast("Any", None))

    assert "unknown_merged=3" in caplog.text
    assert "unknown_merge_refused_cooccurrence=2" in caplog.text
    assert "unknown_merge_refused_distance=1" in caplog.text
    assert "unknown_merge_refused_multi_known=4" in caplog.text
    assert "unknown_merge_refused_no_lookup=5" in caplog.text
    assert m.unknown_merged == 0
    assert m.unknown_merge_refused_cooccurrence == 0
    assert m.unknown_merge_refused_distance == 0
    assert m.unknown_merge_refused_multi_known == 0
    assert m.unknown_merge_refused_no_lookup == 0


@pytest.mark.asyncio
async def test_nearest_match_rejects_zero_vector(hass: HomeAssistant) -> None:
    """Zero vectors raise from _normalize instead of returning a match."""
    cursor = _FakeCursor({"name": _KNOWN, "distance": 0.1})
    dao = PersonGalleryDAO(cast("Any", _FakePool(cursor)), hass)

    with pytest.raises(ValueError, match="Zero vector"):
        await dao.nearest_match([0.0] * 512)


# ---------------------------------------------------------------------------
# recognize_faces FaceHit contract (production embedding source for the merge)
# ---------------------------------------------------------------------------


class _FaceApiResp:
    def __init__(self, faces: list[dict[str, object]]) -> None:
        self._faces = faces

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"faces": self._faces}


class _FaceApiClient:
    def __init__(self, faces: list[dict[str, object]]) -> None:
        self._faces = faces

    async def post(self, *_args: object, **_kwargs: object) -> _FaceApiResp:
        return _FaceApiResp(self._faces)


def _analyzer_with_api(
    hass: HomeAssistant,
    *,
    faces: list[dict[str, object]],
    person_gallery: object,
    face_recognition: bool = True,
) -> VideoAnalyzer:
    runtime_data = SimpleNamespace(
        face_recognition=face_recognition,
        face_api_url="http://face-api",
        person_gallery=person_gallery,
    )
    analyzer = VideoAnalyzer(
        hass, cast("Any", SimpleNamespace(runtime_data=runtime_data))
    )
    analyzer._httpx_client = cast("Any", _FaceApiClient(faces))
    return analyzer


@pytest.mark.asyncio
async def test_recognize_faces_enrolled_path_carries_embedding(
    hass: HomeAssistant,
) -> None:
    """
    Regression: the enrolled path must keep the embedding on the FaceHit.

    This is the merge's only production embedding source — dropping it would
    silently no-op the whole feature via refused_distance.
    """
    emb = [0.1] * FACE_EMBEDDING_DIMS
    dao = MagicMock()
    dao.recognize_person = AsyncMock(return_value=_KNOWN)
    analyzer = _analyzer_with_api(hass, faces=[{"embedding": emb}], person_gallery=dao)

    hits = await analyzer.recognize_faces(b"not-an-image", "camera.test")

    assert hits == [FaceHit(name=_KNOWN, embedding=emb)]
    dao.recognize_person.assert_awaited_once_with(emb)


@pytest.mark.asyncio
async def test_recognize_faces_no_faces_returns_indeterminate_hit(
    hass: HomeAssistant,
) -> None:
    """No faces in the frame: a single embedding-less Indeterminate hit."""
    dao = MagicMock()
    dao.recognize_person = AsyncMock()
    analyzer = _analyzer_with_api(hass, faces=[], person_gallery=dao)

    hits = await analyzer.recognize_faces(b"not-an-image", "camera.test")

    assert hits == [FaceHit(name="Indeterminate", embedding=None)]
    dao.recognize_person.assert_not_awaited()


@pytest.mark.asyncio
async def test_recognize_faces_disabled_returns_empty(hass: HomeAssistant) -> None:
    """Face recognition disabled: an empty hit list, no API call."""
    analyzer = _analyzer_with_api(
        hass, faces=[{"embedding": [0.1]}], person_gallery=None, face_recognition=False
    )

    assert await analyzer.recognize_faces(b"img", "camera.test") == []


# ---------------------------------------------------------------------------
# Red-team hardening: dropped-frame evidence, DAO short-circuit, API trust
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dropped_two_person_frame_refuses_merge(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """
    A VLM-dropped frame holding known+unknown still trips the guard.

    Face recognition ran before the VLM call failed, so the companion
    evidence exists and must refuse the merge even though the frame never
    reaches the summary.
    """
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            ({}, [_known_hit(), _unknown_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == [_KNOWN, "Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_cooccurrence == 1


@pytest.mark.asyncio
async def test_dropped_second_known_name_refuses_merge(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """A second known name seen only in a VLM-dropped frame refuses the merge."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            ({}, [FaceHit(name="Anna", embedding=[0.9])]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == [_KNOWN, "Unknown Person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    assert va._metrics[_CAMERA].unknown_merge_refused_multi_known == 1


@pytest.mark.asyncio
async def test_dropped_single_unknown_does_not_refuse(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """A lone unknown in a dropped frame is the same flapping person; merge."""
    entry.runtime_data.person_gallery = _dao(_GOOD_DISTANCE)
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            ({}, [_unknown_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == [_KNOWN]
    assert va._metrics[_CAMERA].unknown_merged == 1


@pytest.mark.asyncio
async def test_dao_failure_short_circuits_remaining_lookups(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """After the first DAO failure, later unknown faces skip the lookup."""
    entry.runtime_data.person_gallery = _dao(RuntimeError("db down"))
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
            _frame("A man leaves through the gate.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(3))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert entry.runtime_data.person_gallery.nearest_match.await_count == 1
    assert va._metrics[_CAMERA].unknown_merge_refused_no_lookup == 2


def test_merge_threshold_looser_than_recognition_threshold() -> None:
    """The merge bound must stay above the base recognition bound."""
    assert VIDEO_ANALYZER_FACE_MERGE_THRESHOLD > FACE_RECOGNITION_THRESHOLD


@pytest.mark.asyncio
async def test_recognize_faces_malformed_entries_degrade_to_indeterminate(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed face-API entries become Indeterminate, not a batch failure."""
    good = [0.2] * FACE_EMBEDDING_DIMS
    dao = MagicMock()
    dao.recognize_person = AsyncMock(return_value=_KNOWN)
    analyzer = _analyzer_with_api(
        hass,
        faces=[
            {"embedding": [0.1] * (FACE_EMBEDDING_DIMS - 1)},  # ragged
            {"no_embedding": True},  # missing key
            {"embedding": [float("nan")] * FACE_EMBEDDING_DIMS},  # non-finite
            {"embedding": good},
        ],
        person_gallery=dao,
    )

    hits = await analyzer.recognize_faces(b"not-an-image", "camera.test")

    assert [h.name for h in hits] == [
        "Indeterminate",
        "Indeterminate",
        "Indeterminate",
        _KNOWN,
    ]
    dao.recognize_person.assert_awaited_once_with(good)
    assert "malformed face entry" in caplog.text


@pytest.mark.asyncio
async def test_enroll_refuses_reserved_identity_labels(hass: HomeAssistant) -> None:
    """Reserved pipeline labels cannot be enrolled as person names."""
    dao = PersonGalleryDAO(cast("Any", _FakePool(_FakeCursor(None))), hass)
    dao._client = MagicMock()

    for name in ("Unknown Person", " indeterminate ", "None", ""):
        assert await dao.enroll_from_image("http://face-api", name, b"img") is False
    dao._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_refused_when_nearest_match_is_different_person(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """
    A face nearest to a DIFFERENT enrolled person must never merge.

    Codex adversarial finding: within-0.85 distance to the batch's known
    person is not enough — if the gallery's nearest match is someone else,
    relabeling the face as the known person is misattribution.
    """
    entry.runtime_data.person_gallery = _dao(("Anna", 0.4))
    _stub_snapshots(
        va,
        [
            _frame("A man walks toward the entrance.", [_known_hit()]),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == [_KNOWN, "Unknown Person"]
    assert va._metrics[_CAMERA].unknown_merge_refused_distance == 1
    assert va._metrics[_CAMERA].unknown_merged == 0


@pytest.mark.asyncio
async def test_legacy_reserved_label_row_is_not_a_merge_target(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    """
    Pre-guard gallery rows named like reserved labels never count as known.

    Codex structured-review finding: earlier versions enrolled arbitrary
    names, so "unknown person" (lowercase) can exist as a gallery identity.
    """
    entry.runtime_data.person_gallery = _dao(("unknown person", 0.2))
    _stub_snapshots(
        va,
        [
            _frame(
                "A man walks by.", [FaceHit(name="unknown person", embedding=[0.1])]
            ),
            _frame("A man stands near the doorway.", [_unknown_hit()]),
        ],
    )

    _descs, recognized, _, _sole = await va._process_batch(_CAMERA, _ordered(2))

    assert recognized == ["Unknown Person", "unknown person"]
    entry.runtime_data.person_gallery.nearest_match.assert_not_awaited()
    # Reserved-labeled rows leave the known set empty -> silent no-merge.
    assert va._metrics[_CAMERA].unknown_merge_refused_multi_known == 0
    assert va._metrics[_CAMERA].unknown_merged == 0
