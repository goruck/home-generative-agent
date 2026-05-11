# ruff: noqa: S101, ERA001
"""
Regression tests for VideoAnalyzer caption novelty / deduplication logic.

Covers:
- _normalize_caption: punctuation, whitespace, hyphen canonicalization
- _in_artifact_bucket: artifact term detection
- _has_real_subject: subject detection, negated-human handling, face names
- _has_action: active-motion verb detection
- _is_caption_novel: stale_snapshot, no_match, score_none, score_below_threshold,
  score_above_threshold, artifact_bucket, stale_match, store_timeout
- _handle_notification: novelty decision drives notify vs suppress
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import homeassistant.util.dt as dt_util
import pytest

from custom_components.home_generative_agent.const import (
    VIDEO_ANALYZER_CAPTION_DEDUPE_WINDOW_SEC,
    VIDEO_ANALYZER_SIMILARITY_THRESHOLD,
)
from custom_components.home_generative_agent.core.video_analyzer import (
    CaptionNoveltyDecision,
    VideoAnalyzer,
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
) -> MagicMock:
    """Build a minimal SearchItem mock."""
    item = MagicMock()
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

    best_result (score=0.85) is an artifact from 2 hours ago — stale.
    A second result (score=0.70) is a recent artifact from 5 minutes ago.
    The fast path must scan all candidates and suppress based on the recent one.
    """
    results = [
        _make_search_result(
            "a bright horizontal blur streaks across the walkway",
            score=0.85,
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
            score=0.85,
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
            "camera.frontporch", "a person walks up the path", _make_batch()
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
            "camera.frontporch", "empty porch scene", _make_batch()
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
            "camera.frontporch", "empty porch scene", _make_batch()
        )

    va.protect_notify_image.assert_called_once()
    va._send_notification.assert_awaited_once()
