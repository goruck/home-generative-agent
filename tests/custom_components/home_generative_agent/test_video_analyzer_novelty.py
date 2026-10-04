# ruff: noqa: S101, ERA001
"""
Regression tests for VideoAnalyzer caption novelty / deduplication logic.

Covers:
- _normalize_caption: punctuation, whitespace, hyphen canonicalization
- _in_artifact_bucket: artifact term detection
- _has_real_subject: subject detection, negated-human handling, face names
- _has_action: active-motion verb detection
- _is_caption_novel: stale_snapshot, no_match, score_none, score_below_threshold,
  score_above_threshold, recent_match, artifact_bucket, stale_match, store_timeout
- _handle_notification: novelty decision drives notify vs suppress
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import ANY, AsyncMock, MagicMock, call

import homeassistant.util.dt as dt_util
import pytest

from custom_components.home_generative_agent.const import (
    VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    CaptionNoveltyDecision,
    VideoAnalyzer,
    _BatchNotifyContext,
    _cosine_similarity,
    _has_action,
    _has_real_subject,
    _in_artifact_bucket,
    _normalize_caption,
)

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
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    content: str,
    score: float | None,
    age_seconds: int = 60,
    camera: str | None = None,
) -> MagicMock:
    """Build a minimal SearchItem mock; it belongs to any camera unless one is named."""
    item = MagicMock()
    item.namespace = ("video_analysis", camera) if camera else ANY
    item.score = score
    item.value = {"content": content}
    item.created_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    return item


def _fresh_snapshot_name() -> str:
    """Return a snapshot filename whose timestamp is well within the time-offset window."""
    ts = dt_util.now().strftime("%Y%m%d_%H%M%S")
    return f"snapshot_{ts}.jpg"


def _stale_snapshot_name(offset_minutes: int = 20) -> str:
    """Return a snapshot filename whose timestamp is beyond VIDEO_ANALYZER_TIME_OFFSET."""
    ts = (dt_util.now() - timedelta(minutes=offset_minutes)).strftime("%Y%m%d_%H%M%S")
    return f"snapshot_{ts}.jpg"


@pytest.fixture
def entry() -> MagicMock:
    e = MagicMock()
    e.runtime_data.options = {}
    e.runtime_data.store.asearch = AsyncMock(return_value=[])
    e.runtime_data.store.aget = AsyncMock(return_value=None)
    return e


@pytest.fixture
def va(entry: MagicMock) -> VideoAnalyzer:
    return VideoAnalyzer(MagicMock(), entry)


# ---------------------------------------------------------------------------
# _normalize_caption
# ---------------------------------------------------------------------------


def test_normalize_lowercase() -> None:
    assert _normalize_caption("A Bright Light.") == "a bright light"


def test_normalize_collapses_whitespace() -> None:
    assert "  " not in _normalize_caption("lots   of   spaces")


def test_normalize_strips_punctuation() -> None:
    result = _normalize_caption("blur, glare; streak!")
    assert "," not in result
    assert ";" not in result
    assert "!" not in result


def test_normalize_hyphen_canonicalization() -> None:
    assert "black and white" in _normalize_caption("black-and-white scene")


# ---------------------------------------------------------------------------
# _in_artifact_bucket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caption",
    [
        "a bright light appears near the driveway",
        "a bright horizontal blur streaks across the walkway",
        "light streak visible in the frame",
        "there is some glare on the lens",
        "a blurry image of the gate",
        "monochrome view of the walkway",
        "a black and white scene at night",
        "no people visible in the scene",
        "night scene with no activity",
    ],
)
def test_in_artifact_bucket_true(caption: str) -> None:
    assert _in_artifact_bucket(_normalize_caption(caption))


@pytest.mark.parametrize(
    "caption",
    [
        "a person walks up the path",
        "a white vehicle is parked beyond the fences",
        "package left on the doorstep",
        "a quiet walkway with trees",  # no artifact term
    ],
)
def test_in_artifact_bucket_false(caption: str) -> None:
    assert not _in_artifact_bucket(_normalize_caption(caption))


# ---------------------------------------------------------------------------
# _has_real_subject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caption",
    [
        "a person walks up the path",
        "two people are visible near the gate",
        "a white vehicle is parked beyond the fences in the monochrome scene",
        "a package was left on the doorstep",
        "a truck drives past",
        "a car is parked in the driveway",
        "a child runs across the walkway",
        "a white SUV is parked on a paved driveway next to a white picket fence",
        "a van pulls into the driveway",
        "a deer crosses the walkway",
        "a dog is visible near the gate",
        "a dark-furred animal stands on the walkway near the house entrance",
        "a dark animal stands on a grassy area near a large bush",
    ],
)
def test_has_real_subject_true(caption: str) -> None:
    assert _has_real_subject(_normalize_caption(caption), [])


@pytest.mark.parametrize(
    "caption",
    [
        "no people visible in the black and white scene",
        "no person visible near the gate",
        "no one visible on the walkway",
        "nobody visible in the frame",
        "no people are visible in the scene",
        "a bright light streaks across the walkway at night",
        "a quiet monochrome walkway",
    ],
)
def test_has_real_subject_false(caption: str) -> None:
    assert not _has_real_subject(_normalize_caption(caption), [])


def test_has_real_subject_recognized_face() -> None:
    caption = _normalize_caption("a blur near the fence")
    assert _has_real_subject(caption, ["Alice"])


def test_has_real_subject_unknown_person_counts() -> None:
    """'Unknown Person' means a face was seen but not recognized — counts as a subject."""
    caption = _normalize_caption("a blur near the fence")
    assert _has_real_subject(caption, ["Unknown Person"])


def test_has_real_subject_indeterminate_does_not_count() -> None:
    """'Indeterminate' means face recognition found nothing — must not count as a subject."""
    caption = _normalize_caption("a porch with white railings and a metal bench")
    assert not _has_real_subject(caption, ["Indeterminate"])


def test_has_real_subject_negated_then_vehicle() -> None:
    """'No people visible but a car drove past' — vehicle is a real subject."""
    caption = _normalize_caption("no people visible but a car drove past")
    assert _has_real_subject(caption, [])


# ---------------------------------------------------------------------------
# _has_action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caption",
    [
        "a person walks on the sidewalk in the background, then disappears",
        "a white SUV pulls up to the paved driveway",
        "a deer crosses the walkway",
        "a person approaches the front gate",
        "a car drives past the fence",
        "a person steps onto the porch",
        "a person leaves through the front gate",
        # presence verbs — subject is observable even without motion
        "a person and a dog stand on the sidewalk near a white SUV",
        "a person stands near the front door",
        "two people sit on the porch steps",
        "a person is sitting on the bench",
        "a person waits near the gate",
        "a child is watching from the window",
        "a man watches from the corner",
    ],
)
def test_has_action_true(caption: str) -> None:
    assert _has_action(_normalize_caption(caption))


@pytest.mark.parametrize(
    "caption",
    [
        "a house with beige siding and white trim features a front porch with white railings",
        "a white SUV is parked on a paved driveway next to a white picket fence",
        "the porch remains empty with a white SUV parked nearby",
        "a paved driveway with a white picket gate is visible, bordered by a white fence",
        "no people visible in the black and white scene",
        "a quiet monochrome walkway",
        # inanimate verb false positives — scrubbed by _STATIC_CONTEXT_RE
        "a white SUV sits parked on the driveway next to the picket fence",
        "a white SUV sits parked on a brick driveway behind a white picket gate",
        "a paved walkway runs alongside the fence",
        "a gravel path runs between the house and the wooden fence",
        "the driveway runs toward the gate",
        "stepping stones are visible in the yard",
        "a view from a porch shows a grassy yard with stepping stones",
    ],
)
def test_has_action_false(caption: str) -> None:
    assert not _has_action(_normalize_caption(caption))


# ---------------------------------------------------------------------------
# _is_caption_novel: stale_snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_snapshot_notifies(va: VideoAnalyzer) -> None:
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur across the walkway", _stale_snapshot_name(), []
    )
    assert decision == CaptionNoveltyDecision(notify=True, reason="stale_snapshot")


@pytest.mark.asyncio
async def test_stale_snapshot_skips_store_search(va: VideoAnalyzer) -> None:
    await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "msg", _stale_snapshot_name(), []
    )
    va.entry.runtime_data.store.asearch.assert_not_called()


@pytest.mark.asyncio
async def test_recording_frame_name_is_parsed(va: VideoAnalyzer) -> None:
    """An event-recording frame (`_rNN` suffix) heading the batch must not raise."""
    name = _stale_snapshot_name().replace(".jpg", "_r00.jpg")
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "a car passes the door", name, []
    )
    assert decision == CaptionNoveltyDecision(notify=True, reason="stale_snapshot")


@pytest.mark.asyncio
async def test_fresh_recording_frame_reaches_store_search(va: VideoAnalyzer) -> None:
    name = _fresh_snapshot_name().replace(".jpg", "_r07.jpg")
    await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "a car passes the door", name, []
    )
    va.entry.runtime_data.store.asearch.assert_called_once()


# ---------------------------------------------------------------------------
# _is_caption_novel: store_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_timeout_notifies(va: VideoAnalyzer) -> None:
    va.entry.runtime_data.store.asearch = AsyncMock(side_effect=TimeoutError)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur", _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "store_timeout"


# ---------------------------------------------------------------------------
# _is_caption_novel: no_match / score_none
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_search_results_notifies(va: VideoAnalyzer) -> None:
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=[])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur", _fresh_snapshot_name(), []
    )
    assert decision == CaptionNoveltyDecision(notify=True, reason="no_match")


@pytest.mark.asyncio
async def test_all_scores_none_notifies(va: VideoAnalyzer) -> None:
    results = [_make_search_result("prior caption", score=None)]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur", _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "score_none"


# ---------------------------------------------------------------------------
# _is_caption_novel: score_above_threshold suppresses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_score_suppresses(va: VideoAnalyzer) -> None:
    results = [_make_search_result("bright blur across the walkway", score=0.95)]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur streaks", _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"
    assert decision.best_score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_one_high_score_among_low_scores_suppresses(va: VideoAnalyzer) -> None:
    """Best score above threshold suppresses even when other scores are low."""
    results = [
        _make_search_result("bright blur", score=0.92),
        _make_search_result("quiet walkway", score=0.40),
        _make_search_result("light streak", score=0.35),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur", _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"
    assert decision.best_score == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_score_at_exact_threshold_suppresses(va: VideoAnalyzer) -> None:
    """Score == threshold uses >=, so it must suppress (not notify)."""
    results = [
        _make_search_result(
            "bright blur across the walkway", score=VIDEO_ANALYZER_SIMILARITY_THRESHOLD
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "bright blur streaks", _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_score_just_below_threshold_notifies(va: VideoAnalyzer) -> None:
    """Score just below threshold must notify."""
    score = VIDEO_ANALYZER_SIMILARITY_THRESHOLD - 0.001
    results = [_make_search_result("quiet empty walkway", score=score)]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "package left on step", _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "score_below_threshold"


# ---------------------------------------------------------------------------
# _is_caption_novel: score_below_threshold notifies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_score_notifies(va: VideoAnalyzer) -> None:
    results = [_make_search_result("person at the front door", score=0.50)]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "package left on step", _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "score_below_threshold"


# ---------------------------------------------------------------------------
# _is_caption_novel: artifact_bucket suppresses within window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_bucket_suppresses_within_window(va: VideoAnalyzer) -> None:
    results = [
        _make_search_result(
            "no people visible in the black and white scene",
            score=0.75,
            age_seconds=300,
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a bright horizontal blur streaks across the walkway",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "artifact_bucket"


@pytest.mark.asyncio
async def test_artifact_bucket_notifies_when_subject_present(va: VideoAnalyzer) -> None:
    """Vehicle in the current caption prevents artifact suppression."""
    results = [
        _make_search_result(
            "bright light near the driveway", score=0.75, age_seconds=60
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a white vehicle is parked beyond the fences in the monochrome scene",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "score_below_threshold"


@pytest.mark.asyncio
async def test_artifact_bucket_notifies_when_matched_has_subject(
    va: VideoAnalyzer,
) -> None:
    """Matched caption containing a real subject prevents artifact suppression."""
    results = [
        _make_search_result(
            "a car is parked on the monochrome driveway", score=0.75, age_seconds=60
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "bright light near the top of the frame",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "score_below_threshold"


@pytest.mark.asyncio
async def test_negated_human_does_not_block_artifact_suppression(
    va: VideoAnalyzer,
) -> None:
    """'No people visible' counts as human absence, not presence."""
    results = [
        _make_search_result(
            "a bright light streaks across the walkway at night",
            score=0.75,
            age_seconds=60,
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "No people are visible in the black and white scene.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "artifact_bucket"


# ---------------------------------------------------------------------------
# _is_caption_novel: stale_match notifies outside dedupe window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_bucket_stale_match_notifies(va: VideoAnalyzer) -> None:
    results = [
        _make_search_result(
            "no people visible in the black and white scene",
            score=0.75,
            age_seconds=7200,  # 2 hours > VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "bright blur across the walkway",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_artifact_bucket_at_exact_window_suppresses(va: VideoAnalyzer) -> None:
    """Age == window uses <=, so it must suppress (not notify as stale_match)."""
    results = [
        _make_search_result(
            "no people visible in the black and white scene",
            score=0.75,
            age_seconds=VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC,
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a bright horizontal blur streaks across the walkway",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "artifact_bucket"


@pytest.mark.asyncio
async def test_artifact_bucket_one_second_over_window_notifies(
    va: VideoAnalyzer,
) -> None:
    """One second past the window boundary must fire stale_match."""
    results = [
        _make_search_result(
            "no people visible in the black and white scene",
            score=0.75,
            age_seconds=VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC + 1,
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a bright horizontal blur streaks across the walkway",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


# ---------------------------------------------------------------------------
# _is_caption_novel: artifact fast-path scans all candidates, not just best
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_bucket_recent_lower_score_suppresses(
    va: VideoAnalyzer,
) -> None:
    """
    Recent artifact match at lower score suppresses even if best result is old.

    best_result (score=0.80) is an artifact from 2 hours ago — stale.
    A second result (score=0.70) is a recent artifact from 5 minutes ago.
    The fast path must scan all candidates and suppress based on the recent one.
    """
    results = [
        _make_search_result(
            "a bright horizontal blur streaks across the walkway",
            score=0.80,
            age_seconds=7200,  # 2 hours — outside window
        ),
        _make_search_result(
            "glare and blur visible near the fence",
            score=0.70,
            age_seconds=300,  # 5 minutes — inside window
        ),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a bright light streaks across the walkway at night",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "artifact_bucket"
    assert decision.matched_age_seconds == 300


@pytest.mark.asyncio
async def test_artifact_bucket_all_old_notifies_stale_match(
    va: VideoAnalyzer,
) -> None:
    """When every artifact candidate is outside the window, stale_match fires."""
    results = [
        _make_search_result(
            "a bright horizontal blur streaks across the walkway",
            score=0.80,
            age_seconds=3600,  # 1 hour
        ),
        _make_search_result(
            "glare and blur visible near the fence",
            score=0.70,
            age_seconds=2400,  # 40 minutes
        ),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "a bright light streaks across the walkway at night",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.matched_age_seconds == 2400  # most recent artifact match


# ---------------------------------------------------------------------------
# _is_caption_novel: person appears after no-people caption notifies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_person_after_no_people_notifies(va: VideoAnalyzer) -> None:
    results = [
        _make_search_result(
            "No people are visible in the black and white scene.",
            score=0.70,
            age_seconds=120,
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "A person walks up the path toward the gate.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True


# ---------------------------------------------------------------------------
# _is_caption_novel: recognized face bypasses artifact suppression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recognized_face_bypasses_artifact_suppression(
    va: VideoAnalyzer,
) -> None:
    results = [
        _make_search_result(
            "bright blur across the walkway", score=0.75, age_seconds=60
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "bright light near the top of the frame",
        _fresh_snapshot_name(),
        recognized_names=["Alice"],
    )
    assert decision.notify is True
    assert decision.reason == "score_below_threshold"


# ---------------------------------------------------------------------------
# _is_caption_novel: stale high-score match with real subject notifies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_score_stale_match_with_subject_notifies(va: VideoAnalyzer) -> None:
    """A person-event caption matched against a days-old record should notify."""
    results = [
        _make_search_result(
            "A person walks on the sidewalk in the background, then disappears from view.",
            score=0.992,
            age_seconds=247826,  # ~2.87 days
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontporch",
        "A person walks on the sidewalk in the background, then disappears.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.best_score == pytest.approx(0.992)


@pytest.mark.asyncio
async def test_person_dog_standing_stale_notifies(va: VideoAnalyzer) -> None:
    """
    Person+dog standing outside matched against 69-day-old record should notify.

    Regression: 'stand' was not in _ACTION_RE so the stale_match guard never
    fired and the event was silently suppressed for ~69 days.
    """
    results = [
        _make_search_result(
            "A person stands near the SUV. Later, they walk with a dog on the sidewalk.",
            score=0.910,
            age_seconds=5946760,  # ~68.8 days
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontporch",
        "A person and a dog stand on the sidewalk near a white SUV. Later, they remain visible.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.best_score == pytest.approx(0.910)


@pytest.mark.asyncio
async def test_suv_identical_stale_suppresses(va: VideoAnalyzer) -> None:
    """Parked SUV with no action verb suppresses even when the match is days old."""
    caption = "The porch remains empty with a white SUV parked nearby."
    results = [
        _make_search_result(caption, score=1.000, age_seconds=324412)  # ~3.75 days
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontporch", caption, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_unknown_person_identical_stale_notifies(va: VideoAnalyzer) -> None:
    """
    Unknown person on the porch must notify even when the caption is identical.

    Regression: the score=1.0 guard prevented stale_match from firing for an
    exact repeat of 'An unknown person stands on the porch...', silently
    suppressing a security-significant event 1.93 days after the prior record.
    """
    caption = (
        "An unknown person stands on the porch of a beige house, then remains there."
    )
    results = [
        _make_search_result(caption, score=1.000, age_seconds=167144)  # ~1.93 days
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "backyard", caption, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_suv_sits_parked_identical_stale_suppresses(va: VideoAnalyzer) -> None:
    """
    'SUV sits parked' must suppress even when score=1.0 and match is days old.

    Regression: 'sits' was added to _ACTION_RE for person-presence detection, but
    'A white SUV sits parked...' also matched, causing parked-car scenes to fire
    stale_match notifications.  _STATIC_CONTEXT_RE now scrubs 'vehicle sits' before
    _ACTION_RE is applied, so _has_action returns False for these captions.
    """
    caption = "A white SUV sits parked on the driveway next to the picket fence."
    results = [
        _make_search_result(caption, score=1.000, age_seconds=182595)  # ~2.1 days
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", caption, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_suv_near_identical_stale_notifies(va: VideoAnalyzer) -> None:
    """Score < 1.0 means the scene changed — stale vehicle match should notify."""
    results = [
        _make_search_result(
            "A white SUV is parked on a paved driveway next to a white picket fence.",
            score=0.992,
            age_seconds=140709,  # ~1.6 days
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate",
        "A white SUV pulls up to the paved driveway next to the white picket fence.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_high_score_stale_match_no_subject_suppresses(va: VideoAnalyzer) -> None:
    """A static/artifact caption matched against a days-old record should still suppress."""
    results = [
        _make_search_result(
            "A porch with white railings and a metal bench.",
            score=1.000,
            age_seconds=501486,  # ~5.8 days
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontporch",
        "A porch with white railings and columns features a metal bench and chair.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_high_score_recent_match_with_subject_suppresses(
    va: VideoAnalyzer,
) -> None:
    """A person event matched within the dedupe window is still suppressed."""
    results = [
        _make_search_result(
            "A person walks on the sidewalk, then disappears.",
            score=0.992,
            age_seconds=300,  # 5 minutes — within window
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontporch",
        "A person walks on the sidewalk in the background, then disappears.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_generic_animal_stale_notifies(va: VideoAnalyzer) -> None:
    """
    'dark animal stands' must notify when the match is days old.

    Regression: 'animal' was absent from _SUBJECT_RE so _has_real_subject
    returned False and stale_match never fired — the event was silently
    suppressed even when the nearest match was 102 days old.
    """
    results = [
        _make_search_result(
            "A dark animal stands near bushes beside a house door.",
            score=0.906,
            age_seconds=8810154,  # ~102 days
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "playroomdoor",
        "A dark-furred animal stands on a paved walkway near a house entrance.",
        _fresh_snapshot_name(),
        [],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.best_score == pytest.approx(0.906)


# ---------------------------------------------------------------------------
# _is_caption_novel: decision fields are populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_carries_matched_caption_and_score(va: VideoAnalyzer) -> None:
    results = [_make_search_result("prior caption text", score=0.60, age_seconds=90)]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "frontgate", "current caption", _fresh_snapshot_name(), []
    )
    assert decision.best_score == pytest.approx(0.60)
    assert decision.matched_caption == "prior caption text"
    assert decision.matched_age_seconds is not None
    assert decision.matched_age_seconds >= 90


# ---------------------------------------------------------------------------
# _handle_notification: novelty decision drives notify vs suppress
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

_NO_NAMES = _BatchNotifyContext(recognized=[], batch_names=[])


def _make_batch() -> list[Path]:
    """Minimal batch: one snapshot path with 3+ parts so chosen.parts[-3:] works."""
    snap_name = _fresh_snapshot_name()
    return [Path("/media/local/camera_frontporch") / snap_name]


@pytest.mark.asyncio
async def test_handle_notification_notifies_when_decision_is_notify(
    va: VideoAnalyzer,
) -> None:
    """notify=True decision must call protect_notify_image and _send_notification."""
    va.entry.runtime_data.options = {"video_analyzer_mode": "notify_on_anomaly"}
    va._is_caption_novel = AsyncMock(  # type: ignore[method-assign]
        return_value=CaptionNoveltyDecision(notify=True, reason="score_below_threshold")
    )
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.latest_target",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.publish_latest_atomic",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.dispatch_on_loop"
        ),
    ):
        await va._handle_notification(  # type: ignore[attr-defined]
            "camera.frontporch",
            "a person walks up the path",
            _make_batch(),
            context=_NO_NAMES,
        )

    va.protect_notify_image.assert_called_once()
    va._send_notification.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_notification_suppresses_when_decision_is_no_notify(
    va: VideoAnalyzer,
) -> None:
    """notify=False decision must not call protect_notify_image or _send_notification."""
    va.entry.runtime_data.options = {"video_analyzer_mode": "notify_on_anomaly"}
    va._is_caption_novel = AsyncMock(  # type: ignore[method-assign]
        return_value=CaptionNoveltyDecision(
            notify=False, reason="score_above_threshold"
        )
    )
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.latest_target",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.publish_latest_atomic",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.dispatch_on_loop"
        ),
    ):
        await va._handle_notification(  # type: ignore[attr-defined]
            "camera.frontporch",
            "empty porch scene",
            _make_batch(),
            context=_NO_NAMES,
        )

    va.protect_notify_image.assert_not_called()
    va._send_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_notification_always_notifies_outside_anomaly_mode(
    va: VideoAnalyzer,
) -> None:
    """Mode != 'notify_on_anomaly' must always notify regardless of caption novelty."""
    va.entry.runtime_data.options = {"video_analyzer_mode": "always_notify"}
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.latest_target",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.publish_latest_atomic",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.home_generative_agent.core.video_analyzer.dispatch_on_loop"
        ),
    ):
        await va._handle_notification(  # type: ignore[attr-defined]
            "camera.frontporch",
            "empty porch scene",
            _make_batch(),
            context=_NO_NAMES,
        )

    va.protect_notify_image.assert_called_once()
    va._send_notification.assert_awaited_once()


# ---------------------------------------------------------------------------
# _is_caption_novel: a recent near-duplicate outranked by an older caption (#704)
# ---------------------------------------------------------------------------

_WHITE_SHIRT_NOW = (
    "A person in a white shirt walks toward the house. A dark gray Tesla."
)
_WHITE_SHIRT_RECENT = (
    "A person in a white shirt walks toward the house. Later, a person stands near."
)
_WHITE_SHIRT_OLD = "A person in a white shirt walks toward the house."
_THREE_DAYS_S = 255602


def _recency_aware_asearch(
    by_score: list[MagicMock], newest_first: list[MagicMock]
) -> AsyncMock:
    """Answer the similarity search and the newest-first scan separately."""

    async def _asearch(*_args: object, **kwargs: object) -> list[MagicMock]:
        return by_score if kwargs.get("query") else newest_first

    return AsyncMock(side_effect=_asearch)


@pytest.mark.asyncio
async def test_recent_match_outranked_by_older_caption_suppresses(
    va: VideoAnalyzer,
) -> None:
    """A 0.88 match from a minute ago suppresses despite a 0.95 match days old."""
    results = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S),
        _make_search_result(_WHITE_SHIRT_RECENT, score=0.88, age_seconds=64),
    ]
    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(return_value=results)
    store.embeddings.aembed_documents = AsyncMock()
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.best_score == pytest.approx(0.88)
    # The match was among the search results: no scan, nothing embedded.
    store.asearch.assert_awaited_once()
    store.embeddings.aembed_documents.assert_not_awaited()
    assert decision.matched_caption == _WHITE_SHIRT_RECENT
    assert decision.matched_age_seconds is not None
    assert decision.matched_age_seconds <= VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC


@pytest.mark.asyncio
async def test_recent_match_below_threshold_does_not_suppress(
    va: VideoAnalyzer,
) -> None:
    """A recent caption scoring under the threshold leaves the stale match alone."""
    results = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S),
        _make_search_result(
            "A dog runs across the lawn.",
            score=VIDEO_ANALYZER_SIMILARITY_THRESHOLD - 0.01,
            age_seconds=64,
        ),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    # The newest-first scan re-scores the in-window caption: still dissimilar.
    va.entry.runtime_data.store.embeddings.aembed_documents = AsyncMock(
        return_value=[[1.0, 0.0], [0.0, 1.0]]
    )
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.best_score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_recent_match_missing_from_top_results_suppresses(
    va: VideoAnalyzer,
) -> None:
    """Old near-duplicates fill the top results; the newest-first scan finds it."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S + i)
        for i in range(10)
    ]
    newest_first = [
        _make_search_result("A dog runs across the lawn.", score=None, age_seconds=30),
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64),
        _make_search_result(_WHITE_SHIRT_OLD, score=None, age_seconds=_THREE_DAYS_S),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    # Current caption, then the two in-window captions, newest first.
    store.embeddings.aembed_documents = AsyncMock(
        return_value=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    )
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.matched_caption == _WHITE_SHIRT_RECENT
    assert decision.matched_age_seconds is not None
    assert decision.matched_age_seconds <= VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC
    assert decision.best_score == pytest.approx(0.9939, abs=1e-3)
    # Only in-window captions are embedded, alongside the current one.
    store.embeddings.aembed_documents.assert_awaited_once_with(
        [_WHITE_SHIRT_NOW, "A dog runs across the lawn.", _WHITE_SHIRT_RECENT]
    )
    # The scan reads this camera's captions, newest first, one page.
    assert store.asearch.await_count == 2
    assert store.asearch.await_args_list[1] == call(
        ("video_analysis", "side"),
        filter={"notified": True},
        limit=50,
        offset=0,
        refresh_ttl=False,
    )


@pytest.mark.asyncio
async def test_recent_scan_dissimilar_captions_still_stale_match(
    va: VideoAnalyzer,
) -> None:
    """Recent captions that do not resemble the current one do not suppress."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        _make_search_result("A dog runs across the lawn.", score=None, age_seconds=30),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_lone_old_match_skips_embedding(va: VideoAnalyzer) -> None:
    """With nothing stored inside the window there is nothing to embed."""
    results = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(return_value=results)
    store.embeddings.aembed_documents = AsyncMock()
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    store.embeddings.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("embedding down")])
async def test_recent_scan_failure_notifies(
    va: VideoAnalyzer, error: Exception
) -> None:
    """A failed recent scan must not swallow the notification."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock(side_effect=error)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_recent_scan_without_embeddings_notifies(va: VideoAnalyzer) -> None:
    """A store with no embedding function cannot score; the stale match stands."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings = None
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("pool closed")])
async def test_recent_scan_search_failure_notifies(
    va: VideoAnalyzer, error: Exception
) -> None:
    """The scan's own store read failing must not swallow the notification."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]

    async def _asearch(*_args: object, **kwargs: object) -> list[MagicMock]:
        if kwargs.get("query"):
            return by_score
        raise error

    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(side_effect=_asearch)
    store.embeddings.aembed_documents = AsyncMock()
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    store.embeddings.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        [],  # nothing came back
        [[1.0, 0.0]],  # one vector short
        [[1.0, 0.0], [1.0]],  # right count, ragged dimensions
        [[1.0, 0.0], None],  # right count, not a vector
    ],
)
async def test_recent_scan_malformed_embeddings_notify(
    va: VideoAnalyzer, vectors: list[object]
) -> None:
    """A malformed embedding reply costs the dedup, never the notification."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock(return_value=vectors)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_recent_scan_skips_items_without_caption_text(va: VideoAnalyzer) -> None:
    """Only items with caption text are embedded, so vectors stay aligned."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    no_value = _make_search_result("x", score=None, age_seconds=10)
    no_value.value = None
    newest_first = [
        no_value,
        _make_search_result("", score=None, age_seconds=20),
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64),
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=90),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.matched_caption == _WHITE_SHIRT_RECENT
    # A repeated caption is embedded once and reported at its newest age.
    assert decision.matched_age_seconds is not None
    assert decision.matched_age_seconds < 90
    store.embeddings.aembed_documents.assert_awaited_once_with(
        [_WHITE_SHIRT_NOW, _WHITE_SHIRT_RECENT]
    )


@pytest.mark.asyncio
async def test_other_cameras_recent_caption_does_not_suppress(
    va: VideoAnalyzer,
) -> None:
    """
    The store matches namespaces by prefix: "side" also returns "side_gate".

    A recent caption from the other camera must not silence this one, in the
    search results or in the scan.
    """
    by_score = [
        _make_search_result(
            _WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S, camera="side"
        ),
        _make_search_result(
            _WHITE_SHIRT_RECENT, score=0.99, age_seconds=64, camera="side_gate"
        ),
    ]
    newest_first = [
        _make_search_result(
            _WHITE_SHIRT_RECENT, score=None, age_seconds=64, camera="side_gate"
        ),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock()
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert decision.best_score == pytest.approx(0.95)
    store.embeddings.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_other_cameras_captions_is_no_match(va: VideoAnalyzer) -> None:
    """Search results that all belong to a prefix-sibling camera count as none."""
    results = [
        _make_search_result(
            _WHITE_SHIRT_RECENT, score=0.99, age_seconds=64, camera="side_gate"
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "no_match"


@pytest.mark.asyncio
async def test_recent_scan_pages_past_a_busy_window(va: VideoAnalyzer) -> None:
    """More than one page of captions inside the window: the scan reads on."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    page_one = [
        _make_search_result(f"A dog runs across the lawn {i}.", None, age_seconds=i)
        for i in range(50)
    ]
    page_two = [
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=600),
        _make_search_result(_WHITE_SHIRT_OLD, score=None, age_seconds=_THREE_DAYS_S),
    ]

    async def _asearch(*_args: object, **kwargs: object) -> list[MagicMock]:
        if kwargs.get("query"):
            return by_score
        return page_one if kwargs["offset"] == 0 else page_two

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], *([[0.0, 1.0]] * (len(texts) - 2)), [1.0, 0.0]]

    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(side_effect=_asearch)
    store.embeddings.aembed_documents = AsyncMock(side_effect=_embed)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.matched_caption == _WHITE_SHIRT_RECENT
    assert store.asearch.await_args_list[2] == call(
        ("video_analysis", "side"),
        filter={"notified": True},
        limit=50,
        offset=50,
        refresh_ttl=False,
    )
    assert store.asearch.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "age_seconds", "reason"),
    [
        (VIDEO_ANALYZER_SIMILARITY_THRESHOLD, 64, "recent_match"),
        (0.9, VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC - 30, "recent_match"),
        (0.9, VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC + 60, "stale_match"),
    ],
)
async def test_recent_match_boundaries(
    va: VideoAnalyzer, score: float, age_seconds: int, reason: str
) -> None:
    """At the threshold and inside the window suppress; outside it does not."""
    results = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S),
        _make_search_result(_WHITE_SHIRT_RECENT, score=score, age_seconds=age_seconds),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.reason == reason
    assert decision.notify is (reason == "stale_match")


@pytest.mark.asyncio
async def test_best_of_several_recent_matches_is_reported(va: VideoAnalyzer) -> None:
    """With several matches inside the window, the highest-scoring is reported."""
    results = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S),
        _make_search_result("A person walks to the house.", score=0.87, age_seconds=30),
        _make_search_result(_WHITE_SHIRT_RECENT, score=0.91, age_seconds=64),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.best_score == pytest.approx(0.91)
    assert decision.matched_caption == _WHITE_SHIRT_RECENT


@pytest.mark.asyncio
async def test_static_caption_with_old_match_never_scans(va: VideoAnalyzer) -> None:
    """A caption with no subject in action suppresses without the extra scan."""
    results = [
        _make_search_result(
            "A dark SUV is parked in the driveway.", 0.97, age_seconds=_THREE_DAYS_S
        )
    ]
    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", "A dark SUV is parked in the driveway.", _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"
    store.asearch.assert_awaited_once()


def test_cosine_similarity() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="same length"):
        _cosine_similarity([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# Only a caption that was itself notified suppresses a real subject in action
# ---------------------------------------------------------------------------


def _unnotified(item: MagicMock) -> MagicMock:
    """Mark a stored caption as one that was suppressed, not sent."""
    item.value["notified"] = False
    return item


@pytest.mark.asyncio
async def test_recent_best_match_that_was_suppressed_does_not_suppress(
    va: VideoAnalyzer,
) -> None:
    """
    A suppressed caption does not renew the window.

    The scene's last notification is outside the window, so it notifies again
    although a near-identical caption was stored two minutes ago.
    """
    results = [
        _unnotified(_make_search_result(_WHITE_SHIRT_RECENT, 0.97, age_seconds=120)),
        _make_search_result(
            _WHITE_SHIRT_OLD,
            score=0.9,
            age_seconds=VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC + 60,
        ),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "renotify"
    assert decision.best_score == pytest.approx(0.97)


@pytest.mark.asyncio
async def test_suppressed_best_match_with_notified_match_in_window_suppresses(
    va: VideoAnalyzer,
) -> None:
    """The window is anchored on the caption that notified, not the latest one."""
    results = [
        _unnotified(_make_search_result(_WHITE_SHIRT_RECENT, 0.97, age_seconds=120)),
        _make_search_result(_WHITE_SHIRT_OLD, score=0.9, age_seconds=600),
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.matched_caption == _WHITE_SHIRT_OLD


@pytest.mark.asyncio
async def test_recent_scan_ignores_suppressed_captions(va: VideoAnalyzer) -> None:
    """The newest-first scan embeds only captions that were notified."""
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        _unnotified(_make_search_result(_WHITE_SHIRT_RECENT, None, age_seconds=64)),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock()
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    store.embeddings.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_static_caption_suppressed_by_unnotified_match(va: VideoAnalyzer) -> None:
    """A static scene still suppresses on any match; the anchor is for action."""
    results = [
        _unnotified(
            _make_search_result(
                "A dark SUV is parked in the driveway.", 0.97, age_seconds=120
            )
        )
    ]
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", "A dark SUV is parked in the driveway.", _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


# ---------------------------------------------------------------------------
# A batch with an unknown face is suppressed only by another unknown-face batch
# ---------------------------------------------------------------------------


def _unknown(item: MagicMock) -> MagicMock:
    """Mark a stored caption as coming from a batch that held an unknown face."""
    item.value["unknown_person"] = True
    return item


async def _novel_with_unknown_face(
    va: VideoAnalyzer, results: list[MagicMock]
) -> CaptionNoveltyDecision:
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=results)
    return await va._is_caption_novel(  # type: ignore[attr-defined]
        "side",
        _WHITE_SHIRT_NOW,
        _fresh_snapshot_name(),
        ["Unknown Person"],
        unknown_person_seen=True,
    )


@pytest.mark.asyncio
async def test_unknown_person_not_suppressed_by_a_residents_recent_match(
    va: VideoAnalyzer,
) -> None:
    """Caption wording cannot tell a stranger from the resident seen a minute ago."""
    decision = await _novel_with_unknown_face(
        va,
        [
            _make_search_result(_WHITE_SHIRT_OLD, 0.95, age_seconds=_THREE_DAYS_S),
            _make_search_result(_WHITE_SHIRT_RECENT, score=0.88, age_seconds=64),
        ],
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"


@pytest.mark.asyncio
async def test_unknown_person_not_suppressed_by_a_residents_best_match(
    va: VideoAnalyzer,
) -> None:
    """Even the closest match, notified a minute ago, does not silence a stranger."""
    decision = await _novel_with_unknown_face(
        va, [_make_search_result(_WHITE_SHIRT_RECENT, score=0.95, age_seconds=64)]
    )
    assert decision.notify is True
    assert decision.reason == "renotify"


@pytest.mark.asyncio
async def test_unknown_person_suppressed_by_an_unknown_face_notification(
    va: VideoAnalyzer,
) -> None:
    """A stranger who lingers is deduplicated against their own notification."""
    decision = await _novel_with_unknown_face(
        va,
        [
            _unnotified(
                _unknown(_make_search_result(_WHITE_SHIRT_RECENT, 0.97, age_seconds=60))
            ),
            _unknown(_make_search_result(_WHITE_SHIRT_OLD, 0.9, age_seconds=600)),
        ],
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.matched_caption == _WHITE_SHIRT_OLD


@pytest.mark.asyncio
async def test_unknown_person_best_match_from_an_unknown_face_batch_suppresses(
    va: VideoAnalyzer,
) -> None:
    decision = await _novel_with_unknown_face(
        va, [_unknown(_make_search_result(_WHITE_SHIRT_RECENT, 0.95, age_seconds=64))]
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"


@pytest.mark.asyncio
async def test_unknown_person_scan_asks_for_unknown_face_notifications(
    va: VideoAnalyzer,
) -> None:
    by_score = [
        _make_search_result(_WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S)
    ]
    newest_first = [
        # A resident's notification: returned by a fake store, dropped here.
        _make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=30),
        _unknown(_make_search_result(_WHITE_SHIRT_RECENT, score=None, age_seconds=64)),
    ]
    store = va.entry.runtime_data.store
    store.asearch = _recency_aware_asearch(by_score, newest_first)
    store.embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0], [1.0, 0.0]])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side",
        _WHITE_SHIRT_NOW,
        _fresh_snapshot_name(),
        ["Unknown Person"],
        unknown_person_seen=True,
    )
    assert decision.notify is False
    assert decision.reason == "recent_match"
    assert decision.matched_age_seconds is not None
    assert decision.matched_age_seconds >= 60
    assert store.asearch.await_args_list[1].kwargs["filter"] == {
        "notified": True,
        "unknown_person": True,
    }


# ---------------------------------------------------------------------------
# Any failure of the similarity search notifies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_error_notifies(va: VideoAnalyzer) -> None:
    """An embedding or database error must not drop the notification."""
    va.entry.runtime_data.store.asearch = AsyncMock(
        side_effect=RuntimeError("embedding provider down")
    )
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "store_error"


# ---------------------------------------------------------------------------
# _handle_notification / _finalize / _store_results: the notified flag
# ---------------------------------------------------------------------------


async def _run_handle_notification(
    va: VideoAnalyzer, decision: CaptionNoveltyDecision, context: _BatchNotifyContext
) -> bool:
    va.entry.runtime_data.options = {"video_analyzer_mode": "notify_on_anomaly"}
    va._is_caption_novel = AsyncMock(return_value=decision)  # type: ignore[method-assign]
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock(return_value=True)  # type: ignore[method-assign]
    module = "custom_components.home_generative_agent.core.video_analyzer"
    with (
        patch(f"{module}.latest_target", return_value=MagicMock()),
        patch(f"{module}.publish_latest_atomic", new_callable=AsyncMock),
        patch(f"{module}.dispatch_on_loop"),
    ):
        return await va._handle_notification(  # type: ignore[attr-defined]
            "camera.frontporch",
            "a person walks up the path",
            _make_batch(),
            context=context,
        )


@pytest.mark.asyncio
async def test_handle_notification_reports_whether_it_sent(va: VideoAnalyzer) -> None:
    sent = await _run_handle_notification(
        va, CaptionNoveltyDecision(notify=True, reason="stale_match"), _NO_NAMES
    )
    assert sent is True
    withheld = await _run_handle_notification(
        va, CaptionNoveltyDecision(notify=False, reason="recent_match"), _NO_NAMES
    )
    assert withheld is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recognized", "batch_names", "expected"),
    [
        ([], [], False),
        (["Lindo"], ["Lindo"], False),
        (["Unknown Person"], ["Unknown Person"], True),
        # Seen only in a frame the summary dropped.
        (["Lindo"], ["Lindo", "Unknown Person"], True),
    ],
)
async def test_handle_notification_passes_unknown_person_seen(
    va: VideoAnalyzer, recognized: list[str], batch_names: list[str], expected: object
) -> None:
    await _run_handle_notification(
        va,
        CaptionNoveltyDecision(notify=True, reason="stale_match"),
        _BatchNotifyContext(recognized=recognized, batch_names=batch_names),
    )
    kwargs = va._is_caption_novel.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["unknown_person_seen"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("notified", "batch_names", "unknown_person"),
    [
        (True, [], False),
        (False, [], False),
        (True, ["Lindo", "Unknown Person"], True),
    ],
)
async def test_finalize_stores_whether_the_caption_notified(
    va: VideoAnalyzer, notified: object, batch_names: list[str], unknown_person: object
) -> None:
    va._handle_notification = AsyncMock(return_value=notified)  # type: ignore[method-assign]
    store = va.entry.runtime_data.store
    store.aput = AsyncMock()
    batch = _make_batch()
    await va._finalize(  # type: ignore[attr-defined]
        "camera.frontporch",
        batch,
        "a person walks up the path",
        context=_BatchNotifyContext(recognized=[], batch_names=batch_names),
    )
    store.aput.assert_awaited_once_with(
        namespace=("video_analysis", "frontporch"),
        key=batch[0].name,
        value={
            "content": "a person walks up the path",
            "snapshots": [str(batch[0])],
            "notified": notified,
            "unknown_person": unknown_person,
        },
    )


@pytest.mark.asyncio
async def test_withheld_caption_does_not_erase_a_notified_one_under_the_same_key(
    va: VideoAnalyzer,
) -> None:
    """Two batches can share a key (one-second snapshot names)."""
    store = va.entry.runtime_data.store
    earlier = MagicMock()
    earlier.value = {"content": "first", "notified": True, "unknown_person": True}
    store.aget = AsyncMock(return_value=earlier)
    store.aput = AsyncMock()
    batch = _make_batch()
    await va._store_results(  # type: ignore[attr-defined]
        "camera.frontporch", batch, "second", notified=False
    )
    value = store.aput.await_args_list[0].kwargs["value"]
    assert value["notified"] is True
    assert value["unknown_person"] is True
    assert value["content"] == "second"


@pytest.mark.asyncio
async def test_notified_caption_is_stored_without_reading_first(
    va: VideoAnalyzer,
) -> None:
    store = va.entry.runtime_data.store
    store.aput = AsyncMock()
    await va._store_results(  # type: ignore[attr-defined]
        "camera.frontporch", _make_batch(), "first", notified=True
    )
    store.aget.assert_not_awaited()


# ---------------------------------------------------------------------------
# Cameras sharing a name prefix, scan paging, the sent flag, the finalize lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_other_cameras_captions_do_not_crowd_out_own_match(
    va: VideoAnalyzer,
) -> None:
    """Ten "side_gate" captions outrank the camera's own; its own is still found."""
    foreign = [
        _make_search_result(
            _WHITE_SHIRT_RECENT, 0.99, age_seconds=i, camera="side_gate"
        )
        for i in range(10)
    ]
    weaker = _make_search_result("A person walks.", 0.86, age_seconds=30, camera="side")
    own = _make_search_result(_WHITE_SHIRT_RECENT, 0.9, age_seconds=64, camera="side")
    store = va.entry.runtime_data.store
    # Returned out of score order: the camera's own rows are ranked here.
    store.asearch = AsyncMock(return_value=[*foreign, weaker, own])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is False
    assert decision.reason == "score_above_threshold"
    assert decision.best_score == pytest.approx(0.9)
    store.asearch.assert_awaited_once_with(
        ("video_analysis", "side"),
        query=_WHITE_SHIRT_NOW,
        limit=50,
        refresh_ttl=False,
    )


@pytest.mark.asyncio
async def test_search_keeps_the_ten_best_own_captions(va: VideoAnalyzer) -> None:
    """Rows beyond the ten best of the camera's own are not considered."""
    rows = [
        _make_search_result(f"A dog {i}.", 0.5 - i / 100, age_seconds=_THREE_DAYS_S)
        for i in range(10)
    ]
    eleventh = _make_search_result(_WHITE_SHIRT_RECENT, 0.1, age_seconds=64)
    va.entry.runtime_data.store.asearch = AsyncMock(return_value=[*rows, eleventh])
    own = await va._search_own_captions(  # type: ignore[attr-defined]
        ("video_analysis", "side"), _WHITE_SHIRT_NOW
    )
    assert own == rows


def _scan_store(va: VideoAnalyzer, pages: list[list[MagicMock]]) -> MagicMock:
    """Store whose similarity search finds one old match and whose scan pages."""
    by_score = [
        _make_search_result(
            _WHITE_SHIRT_OLD, score=0.95, age_seconds=_THREE_DAYS_S, camera="side"
        )
    ]

    async def _asearch(*_args: object, **kwargs: object) -> list[MagicMock]:
        if kwargs.get("query"):
            return by_score
        index = int(str(kwargs["offset"])) // 50
        return pages[index] if index < len(pages) else []

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], *([[0.0, 1.0]] * (len(texts) - 1))]

    store = va.entry.runtime_data.store
    store.asearch = AsyncMock(side_effect=_asearch)
    store.embeddings.aembed_documents = AsyncMock(side_effect=_embed)
    return store


def _scan_calls(store: MagicMock) -> int:
    return sum(1 for c in store.asearch.await_args_list if "query" not in c.kwargs)


@pytest.mark.asyncio
async def test_recent_scan_stops_at_the_page_reaching_past_the_window(
    va: VideoAnalyzer,
) -> None:
    """A full page whose oldest row is outside the window ends the scan."""
    page = [
        _make_search_result(f"A dog runs {i}.", None, age_seconds=i, camera="side")
        for i in range(49)
    ]
    page.append(
        _make_search_result("old", None, age_seconds=_THREE_DAYS_S, camera="side")
    )
    store = _scan_store(va, [page, page])
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.reason == "stale_match"
    assert _scan_calls(store) == 1


@pytest.mark.asyncio
async def test_recent_scan_is_capped(va: VideoAnalyzer) -> None:
    """Full pages of in-window rows from another camera: four pages, then stop."""
    page = [
        _make_search_result(f"A dog {i}.", None, age_seconds=i, camera="side_gate")
        for i in range(50)
    ]
    store = _scan_store(va, [page] * 10)
    decision = await va._is_caption_novel(  # type: ignore[attr-defined]
        "side", _WHITE_SHIRT_NOW, _fresh_snapshot_name(), []
    )
    assert decision.notify is True
    assert decision.reason == "stale_match"
    assert _scan_calls(store) == 4
    store.embeddings.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_notification_without_a_notify_service_is_not_notified(
    va: VideoAnalyzer,
) -> None:
    """A push with nowhere to go must not anchor the dedupe window."""
    va.entry.runtime_data.options = {"video_analyzer_mode": "notify_on_anomaly"}
    va._is_caption_novel = AsyncMock(  # type: ignore[method-assign]
        return_value=CaptionNoveltyDecision(notify=True, reason="no_match")
    )
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._send_notification = AsyncMock(return_value=False)  # type: ignore[method-assign]
    module = "custom_components.home_generative_agent.core.video_analyzer"
    with (
        patch(f"{module}.latest_target", return_value=MagicMock()),
        patch(f"{module}.publish_latest_atomic", new_callable=AsyncMock),
        patch(f"{module}.dispatch_on_loop"),
    ):
        sent = await va._handle_notification(  # type: ignore[attr-defined]
            "camera.frontporch", "a person walks", _make_batch(), context=_NO_NAMES
        )
    assert sent is False


@pytest.mark.asyncio
async def test_unknown_person_spelling_variant_is_seen(va: VideoAnalyzer) -> None:
    await _run_handle_notification(
        va,
        CaptionNoveltyDecision(notify=True, reason="stale_match"),
        _BatchNotifyContext(recognized=[], batch_names=[" unknown person "]),
    )
    kwargs = va._is_caption_novel.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["unknown_person_seen"] is True


@pytest.mark.asyncio
async def test_finalize_is_one_step_per_camera(va: VideoAnalyzer) -> None:
    """A second batch's novelty check waits for the first batch's caption store."""
    events: list[str] = []
    first_checking = asyncio.Event()
    release = asyncio.Event()

    async def _handle(_camera: str, msg: str, *_a: object, **_k: object) -> bool:
        events.append(f"check {msg}")
        if msg == "first":
            first_checking.set()
            await release.wait()
        return True

    async def _store(_camera: str, _batch: object, msg: str, **_k: object) -> None:
        events.append(f"store {msg}")

    va._handle_notification = _handle  # type: ignore[method-assign]
    va._store_results = _store  # type: ignore[method-assign]
    batch = _make_batch()
    first = asyncio.create_task(
        va._finalize("camera.side", batch, "first", context=_NO_NAMES)  # type: ignore[attr-defined]
    )
    await first_checking.wait()
    second = asyncio.create_task(
        va._finalize("camera.side", batch, "second", context=_NO_NAMES)  # type: ignore[attr-defined]
    )
    other = asyncio.create_task(
        va._finalize("camera.back", batch, "other", context=_NO_NAMES)  # type: ignore[attr-defined]
    )
    await other  # another camera is not held up
    assert events == ["check first", "check other", "store other"]
    release.set()
    await asyncio.gather(first, second)
    assert events[3:] == ["store first", "check second", "store second"]
