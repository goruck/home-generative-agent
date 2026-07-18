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
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.core.video_analyzer import VideoAnalyzer

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

    descs, recognized = await va._process_batch(_CAMERA, _ordered(3))

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

    descs, _ = await va._process_batch(_CAMERA, _ordered(4))

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

    descs, recognized = await va._process_batch(_CAMERA, _ordered(2))

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

    descs, _ = await va._process_batch(_CAMERA, _ordered(3))

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

    descs, recognized = await va._process_batch(_CAMERA, _ordered(2))

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

    descs, _ = await va._process_batch(_CAMERA, _ordered(3))

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

    descs, _ = await va._process_batch(_CAMERA, _ordered(2))

    assert descs == [
        {"t+0s. Scene unchanged.": ["Indeterminate"]},
        {f"t+8s. {_FULL_DESC}": ["Indeterminate"]},
    ]
    assert prev_texts == [None, None]
