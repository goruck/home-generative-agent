# ruff: noqa: S101
"""
Tests for the repeated-scene sentinel handling in VideoAnalyzer._process_batch.

Issue #493: the VLM is prompted to reply "Scene unchanged." for static repeated
scenes. The analyzer must:
- drop sentinel frames from the summary input,
- keep prev_text anchored on the last full description (never a sentinel),
- keep a sentinel frame when face recognition found someone the VLM missed,
- treat a sentinel-looking reply on the first frame (no prev context) as a
  normal description.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_generative_agent.core.video_analyzer import (
    VideoAnalyzer,
    _caption_mentions_person,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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

_FULL_DESC = "An empty driveway with a parked car and a small tree nearby."
_CAMERA = "camera.front_porch"


@pytest.fixture
def entry() -> MagicMock:
    e = MagicMock()
    e.runtime_data.options = {}
    return e


@pytest.fixture
def va(entry: MagicMock) -> VideoAnalyzer:
    return VideoAnalyzer(MagicMock(), entry)


def _stub_snapshots(
    va: VideoAnalyzer,
    replies: Sequence[dict[str, list[str]] | None],
) -> list[str | None]:
    """Replace _process_snapshot; return the list capturing prev_text per call."""
    prev_texts: list[str | None] = []
    reply_iter = iter(replies)

    async def fake_process(
        path: Path,  # noqa: ARG001
        camera_id: str,  # noqa: ARG001
        prev_text: str | None = None,
    ) -> dict[str, list[str]] | None:
        prev_texts.append(prev_text)
        return next(reply_iter)

    va._process_snapshot = AsyncMock(side_effect=fake_process)  # type: ignore[method-assign]
    return prev_texts


def _ordered(n: int) -> list[tuple[Path, int]]:
    return [(Path(f"snap_{i}.jpg"), 1000 + 8 * i) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_frames_dropped_and_prev_text_stays_anchored(
    va: VideoAnalyzer,
) -> None:
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"Scene unchanged.": ["Indeterminate"]},
            {"Nothing has changed.": ["Indeterminate"]},
        ],
    )

    descs, recognized, _ = await va._process_batch(_CAMERA, _ordered(3))

    assert descs == [{f"t+0s. {_FULL_DESC}": ["Indeterminate"]}]
    # Pre-existing behavior: only "None" is filtered here; "Indeterminate"
    # is filtered downstream (_has_real_subject, format_subject).
    assert recognized == ["Indeterminate"]
    # Frames 2 and 3 must both be compared against the full description,
    # never against a sentinel.
    assert prev_texts == [None, _FULL_DESC, _FULL_DESC]


@pytest.mark.asyncio
async def test_full_description_after_sentinel_reanchors(va: VideoAnalyzer) -> None:
    person_desc = "A person walks up the driveway."
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"Scene unchanged.": ["Indeterminate"]},
            {person_desc: ["Indeterminate"]},
            {"Scene unchanged.": ["Indeterminate"]},
        ],
    )

    descs, _, _ = await va._process_batch(_CAMERA, _ordered(4))

    assert descs == [
        {f"t+0s. {_FULL_DESC}": ["Indeterminate"]},
        {f"t+16s. {person_desc}": ["Indeterminate"]},
    ]
    assert prev_texts == [None, _FULL_DESC, _FULL_DESC, person_desc]


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["Alice", "Unknown Person"])
async def test_sentinel_frame_with_detected_person_is_kept(
    va: VideoAnalyzer, identity: str
) -> None:
    """Known names AND unenrolled persons must both defeat the sentinel drop."""
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"Scene unchanged.": [identity]},
        ],
    )

    descs, recognized, _ = await va._process_batch(_CAMERA, _ordered(2))

    # Frame kept so the detected identity survives, but the sentinel text
    # still must not become the prev_text anchor.
    assert descs == [
        {f"t+0s. {_FULL_DESC}": ["Indeterminate"]},
        {"t+8s. Scene unchanged.": [identity]},
    ]
    assert identity in recognized
    assert prev_texts == [None, _FULL_DESC]


@pytest.mark.asyncio
async def test_error_caption_never_becomes_anchor(va: VideoAnalyzer) -> None:
    """A VLM failure caption must not anchor a later sentinel comparison."""
    err = "Error analyzing image with VLM model."
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {err: []},
            {"Scene unchanged.": ["Indeterminate"]},
        ],
    )

    descs, _, _ = await va._process_batch(_CAMERA, _ordered(3))

    # The error caption is skipped entirely (like the return-{} error paths)
    # and the sentinel on frame 3 is validated against the real description.
    assert descs == [{f"t+0s. {_FULL_DESC}": ["Indeterminate"]}]
    assert prev_texts == [None, _FULL_DESC, _FULL_DESC]


@pytest.mark.asyncio
async def test_error_caption_with_detected_person_is_kept(
    va: VideoAnalyzer,
) -> None:
    """A VLM failure must not erase a successful face detection."""
    err = "Error analyzing image with VLM model."
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {err: ["Unknown Person"]},
        ],
    )

    descs, recognized, _ = await va._process_batch(_CAMERA, _ordered(2))

    # The frame survives with a neutral caption — the raw error text must not
    # leak into summaries/notifications, and the identity must be preserved.
    assert descs == [
        {f"t+0s. {_FULL_DESC}": ["Indeterminate"]},
        {"t+8s. A person is present; scene analysis unavailable.": ["Unknown Person"]},
    ]
    assert "Unknown Person" in recognized
    assert all(err not in text for d in descs for text in d)
    # The error caption still must not become the comparison anchor.
    assert prev_texts == [None, _FULL_DESC]


@pytest.mark.asyncio
async def test_empty_caption_is_skipped(va: VideoAnalyzer) -> None:
    """An empty VLM reply must not produce a junk entry or clear the anchor."""
    prev_texts = _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"": ["Indeterminate"]},
            {"Scene unchanged.": ["Indeterminate"]},
        ],
    )

    descs, _, _ = await va._process_batch(_CAMERA, _ordered(3))

    assert descs == [{f"t+0s. {_FULL_DESC}": ["Indeterminate"]}]
    assert prev_texts == [None, _FULL_DESC, _FULL_DESC]


@pytest.mark.asyncio
async def test_sentinel_reply_on_first_frame_is_a_description(
    va: VideoAnalyzer,
) -> None:
    """
    First-frame sentinel is kept as a description but never anchors.

    Without prev context the reply is kept as-is, but a sentinel-shaped text
    must still never become the comparison anchor — otherwise the whole batch
    could collapse against contentless context.
    """
    prev_texts = _stub_snapshots(
        va,
        [
            {"Scene unchanged.": ["Indeterminate"]},
            {_FULL_DESC: ["Indeterminate"]},
        ],
    )

    descs, _, _ = await va._process_batch(_CAMERA, _ordered(2))

    assert descs == [
        {"t+0s. Scene unchanged.": ["Indeterminate"]},
        {f"t+8s. {_FULL_DESC}": ["Indeterminate"]},
    ]
    assert prev_texts == [None, None]


# ---------------------------------------------------------------------------
# Notification reference frame selection (issue: post-#495 image/text mismatch)
#
# The summary text is generated only from frames that survive the sentinel and
# error drops, so the notification reference image must be chosen from those
# same frames — never from the middle of the raw batch, which after #495 is
# dominated by "Scene unchanged." frames showing an empty scene.
# ---------------------------------------------------------------------------

_WALK_DESC = "A man walks down the stairs."
_PAUSE_DESC = "The man pauses on the porch near pink flowers."
_SENTINEL = {"Scene unchanged.": ["Indeterminate"]}


@pytest.mark.asyncio
async def test_process_batch_notify_frame_prefers_person_frames(
    va: VideoAnalyzer,
) -> None:
    """The chosen frame must show the person the summary describes."""
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {_WALK_DESC: ["Indeterminate"]},
            {_PAUSE_DESC: ["Indeterminate"]},
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(7))

    # Person frames are snap_1/snap_2 (middle of the person pool is snap_2).
    # The raw batch middle, snap_3, is a dropped sentinel frame.
    assert notify_frame == Path("snap_2.jpg")


@pytest.mark.asyncio
async def test_process_batch_notify_frame_falls_back_to_middle_kept_frame(
    va: VideoAnalyzer,
) -> None:
    """With no person in any kept frame, use the middle kept frame."""
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"A delivery truck idles by the curb.": ["Indeterminate"]},
            {"The truck drives away, leaving the curb empty.": ["Indeterminate"]},
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(7))

    # Kept frames are snap_0..snap_2; middle is snap_1. Batch middle (snap_3)
    # is a sentinel frame and must never be chosen.
    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.asyncio
async def test_process_batch_notify_frame_counts_recognized_faces(
    va: VideoAnalyzer,
) -> None:
    """A sentinel frame kept for a recognized face qualifies as a person frame."""
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"Scene unchanged.": ["Alice"]},
            dict(_SENTINEL),
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(3))

    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.asyncio
async def test_process_batch_empty_batch_has_no_notify_frame(
    va: VideoAnalyzer,
) -> None:
    descs, recognized, notify_frame = await va._process_batch(_CAMERA, [])

    assert descs == []
    assert recognized == []
    assert notify_frame is None


@pytest.mark.asyncio
async def test_notification_image_matches_summary_not_batch_middle(
    va: VideoAnalyzer,
) -> None:
    """
    End-to-end regression for the post-#495 image/text mismatch.

    A person walks through early in the batch; the tail is sentinel frames.
    The pushed notification image must be one of the person frames, not the
    batch-middle sentinel frame (an empty scene).
    """
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {_WALK_DESC: ["Indeterminate"]},
            {_PAUSE_DESC: ["Indeterminate"]},
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
            dict(_SENTINEL),
        ],
    )
    va._summarize = AsyncMock(  # type: ignore[method-assign]
        return_value="A man walks down the stairs, then pauses on the porch."
    )
    va._store_results = AsyncMock()  # type: ignore[method-assign]
    va._prune_old_snapshots = AsyncMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.latest_target",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer."
            "publish_latest_atomic",
            new_callable=AsyncMock,
        ) as mock_publish,
        patch(
            "custom_components.home_generative_agent.core.video_analyzer."
            "dispatch_on_loop"
        ),
    ):
        await va._analyze_and_finalize(_CAMERA, _ordered(7))

    va._send_notification.assert_awaited_once()
    assert va._send_notification.await_args is not None
    notify_img = va._send_notification.await_args.args[2]
    assert Path(notify_img).name in {"snap_1.jpg", "snap_2.jpg"}
    # The "latest" image entity must show the same representative frame.
    assert mock_publish.await_args is not None
    assert mock_publish.await_args.args[1].name in {"snap_1.jpg", "snap_2.jpg"}


@pytest.mark.asyncio
async def test_process_batch_all_frames_dropped_yields_no_notify_frame(
    va: VideoAnalyzer,
) -> None:
    """When every frame errors out, there is nothing to attach to a notification."""
    _stub_snapshots(va, [None, None, None])

    descs, recognized, notify_frame = await va._process_batch(_CAMERA, _ordered(3))

    assert descs == []
    assert recognized == []
    assert notify_frame is None


@pytest.mark.asyncio
async def test_process_batch_person_fallback_frame_can_be_notify_frame(
    va: VideoAnalyzer,
) -> None:
    """A frame kept via the person-fallback caption keeps its path and is chosen."""
    err = "Error analyzing image with VLM model."
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {err: ["Unknown Person"]},
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(2))

    # snap_1 survives with the neutral person caption; it is the only person
    # frame, so it must be the notification reference image.
    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.asyncio
async def test_process_batch_notify_frame_respects_last_8_cap(
    va: VideoAnalyzer,
) -> None:
    """Paths must stay aligned with descriptions through the cap-to-last-8 slice."""
    _stub_snapshots(
        va,
        [
            {f"A bird lands on the feeder, event {i}.": ["Indeterminate"]}
            for i in range(10)
        ],
    )

    descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(10))

    # All 10 captions are distinct and non-human; the cap keeps snap_2..snap_9
    # and the fallback picks the middle kept frame: index 4 of 8 -> snap_6.
    assert len(descs) == 8
    assert notify_frame == Path("snap_6.jpg")


@pytest.mark.asyncio
async def test_handle_notification_without_notify_frame_uses_batch_middle(
    va: VideoAnalyzer,
) -> None:
    """Legacy callers that pass no notify_frame keep the middle-of-batch image."""
    batch = [Path(f"/media/local/camera_front_porch/snap_{i}.jpg") for i in range(3)]
    va._send_notification = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.latest_target",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer."
            "publish_latest_atomic",
            new_callable=AsyncMock,
        ) as mock_publish,
        patch(
            "custom_components.home_generative_agent.core.video_analyzer."
            "dispatch_on_loop"
        ),
    ):
        await va._handle_notification(_CAMERA, "a quiet porch", batch)

    assert mock_publish.await_args is not None
    assert mock_publish.await_args.args[1] == batch[1]
    va._send_notification.assert_awaited_once()
    assert va._send_notification.await_args is not None
    assert Path(va._send_notification.await_args.args[2]).name == "snap_1.jpg"


@pytest.mark.asyncio
async def test_negated_person_caption_does_not_pollute_person_pool(
    va: VideoAnalyzer,
) -> None:
    """
    "No people visible" must not count as a person frame.

    Cross-model review finding: substring/negation-blind matching let a
    negated caption late in the batch drag the person pool's middle onto an
    empty-scene frame — reintroducing the image/text mismatch this fix removes.
    """
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {_WALK_DESC: ["Indeterminate"]},
            {"No people are visible near the door.": ["Indeterminate"]},
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(3))

    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.asyncio
async def test_substring_human_terms_do_not_count_as_person(
    va: VideoAnalyzer,
) -> None:
    """'man' inside 'manicured' must not make a lawn caption a person frame."""
    _stub_snapshots(
        va,
        [
            {_FULL_DESC: ["Indeterminate"]},
            {"A manicured lawn in front of the house.": ["Indeterminate"]},
            {"A round bush beside the paved path.": ["Indeterminate"]},
        ],
    )

    _descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(3))

    # No person frames at all: fall back to the middle kept frame.
    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.asyncio
async def test_duplicate_run_keeps_path_of_recognized_face_frame(
    va: VideoAnalyzer,
) -> None:
    """
    Merged duplicate runs keep the path of the recognized-face frame.

    The notification names the recognized person, so the attached image must
    be the frame where they are actually identifiable.
    """
    desc = "A person stands at the door."
    _stub_snapshots(
        va,
        [
            {desc: ["Indeterminate"]},
            {desc: ["Alice"]},
        ],
    )

    _descs, recognized, notify_frame = await va._process_batch(_CAMERA, _ordered(2))

    assert "Alice" in recognized
    assert notify_frame == Path("snap_1.jpg")


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("A man walks down the stairs.", True),
        ("Two men stand near the porch.", True),
        ("Several women chat on the paved path.", True),
        ("A person is present; scene analysis unavailable.", True),
        ("A child plays in the yard, nobody else around.", True),
        ("There are no people in the scene.", False),
        ("No person is present.", False),
        ("No people are visible near the door.", False),
        ("A manicured lawn in front of the house.", False),
        ("A German shepherd rests on the porch.", False),
        ("An empty driveway with a parked car.", False),
    ],
)
def test_caption_mentions_person(caption: str, *, expected: bool) -> None:
    """Word-bounded, plural-aware, negation-scrubbed person detection."""
    assert _caption_mentions_person(caption) is expected


@pytest.mark.asyncio
async def test_pick_notify_frame_failure_degrades_to_batch_middle_fallback(
    va: VideoAnalyzer,
) -> None:
    """A selection bug must cost image accuracy, never the alert itself."""
    _stub_snapshots(va, [{_WALK_DESC: ["Indeterminate"]}])

    with patch(
        "custom_components.home_generative_agent.core.video_analyzer."
        "_pick_notify_frame",
        side_effect=ValueError("misaligned"),
    ):
        descs, _, notify_frame = await va._process_batch(_CAMERA, _ordered(1))

    # Descriptions survive so the alert still goes out; frame falls back.
    assert descs == [{f"t+0s. {_WALK_DESC}": ["Indeterminate"]}]
    assert notify_frame is None
