# ruff: noqa: S101
"""
Tests for the video analyzer's per-camera notification cooldown (issue #672).

The pure decision table is covered in test_notify_cooldown.py. These tests
cover what the analyzer adds around it: the notify payloads, the per-camera
lock, failure handling, the non-companion-app gate, and the batch-local names
handed to the sensor/image signals and the novelty check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.const import (
    CONF_NOTIFY_SERVICE,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S,
)
from custom_components.home_generative_agent.core import video_analyzer as va_mod
from custom_components.home_generative_agent.core.notify_cooldown import HumanLevel
from custom_components.home_generative_agent.core.video_analyzer import (
    CaptionNoveltyDecision,
    VideoAnalyzer,
    _BatchNotifyContext,
)


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure-asyncio tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: all tasks explicitly awaited; no HA resources to clean up."""


_CAMERA = "camera.frontporch"
_NAME = "frontporch"
_OTHER_CAMERA = "camera.side"
_IMG = Path("/media/local/snapshots/camera_frontporch/snapshot_20261002_120000.jpg")
_SERVICE = "notify.mobile_app_phone"
_COOLDOWN = 120
_QUIET_PUSH = {"interruption-level": "passive", "sound": "none"}


class _Clock:
    """Controllable stand-in for the analyzer's monotonic and wall clocks."""

    def __init__(self) -> None:
        self.mono = 1000.0
        self.wall = 1_790_000_000.0

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@pytest.fixture
def clock() -> Any:
    c = _Clock()
    with (
        patch.object(va_mod, "monotonic", lambda: c.mono),
        patch.object(va_mod, "time", lambda: c.wall),
    ):
        yield c


@pytest.fixture
def entry() -> MagicMock:
    e = MagicMock()
    e.runtime_data.options = {
        CONF_NOTIFY_SERVICE: _SERVICE,
        CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S: _COOLDOWN,
    }
    e.runtime_data.store.asearch = AsyncMock(return_value=[])
    e.runtime_data.store.aput = AsyncMock()
    return e


@pytest.fixture
def va(entry: MagicMock) -> VideoAnalyzer:
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    return VideoAnalyzer(hass, entry)


def _service(va: VideoAnalyzer) -> AsyncMock:
    return cast("AsyncMock", va.hass.services.async_call)


def _calls(va: VideoAnalyzer) -> list[Any]:
    return _service(va).await_args_list


def _data(call: Any) -> dict[str, Any]:
    return call.args[2]["data"]


async def _push(
    va: VideoAnalyzer,
    msg: str = "A person stands at the door.",
    *,
    names: list[str] | None = None,
    capture_ts: float = 2000.0,
    camera_id: str = _CAMERA,
) -> None:
    await va._dispatch_notification(
        camera_id,
        camera_id.rsplit(".", maxsplit=1)[-1],
        msg,
        _IMG,
        batch_names=names or [],
        capture_ts=capture_ts,
    )


# ---------------------------------------------------------------------------
# Option off: the regression contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, None, "", "abc", -5])
@pytest.mark.asyncio
async def test_cooldown_off_sends_the_unchanged_payload(
    va: VideoAnalyzer, entry: MagicMock, value: object
) -> None:
    """With the option at 0 (or malformed) the call is exactly the old one."""
    entry.runtime_data.options[CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S] = value

    await _push(va, "A **person** stands at the `door`.")
    await _push(va, "A person stands at the door again.")

    assert len(_calls(va)) == 2
    first = _calls(va)[0]
    assert first.args == (
        "notify",
        "mobile_app_phone",
        {
            "message": "A person stands at the door.",
            "title": f"Camera Alert from {_NAME}!",
            "data": {"image": str(_IMG)},
        },
    )
    assert first.kwargs == {"blocking": False}
    assert _calls(va)[1].kwargs == {"blocking": False}
    assert va._notify_windows == {}


# ---------------------------------------------------------------------------
# Window: open, quiet update, expiry, camera isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_push_sounds_and_later_ones_replace_the_card_quietly(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va, capture_ts=2000.0)
    clock.advance(30)
    await _push(va, "A person still stands at the door.", capture_ts=2030.0)

    alert, update = _calls(va)
    tag = f"hga_camera_{_NAME}_1790000000"
    assert _data(alert) == {"image": str(_IMG), "tag": tag}
    assert alert.kwargs == {"blocking": True}
    assert _data(update) == {
        "image": str(_IMG),
        "tag": tag,
        "alert_once": True,
        "push": _QUIET_PUSH,
    }
    assert update.kwargs == {"blocking": False}
    assert update.args[2]["message"] == "A person still stands at the door."


@pytest.mark.asyncio
async def test_quiet_updates_do_not_extend_the_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A lingering subject sounds again once the fixed window has run out."""
    await _push(va, capture_ts=2000.0)
    for step in range(1, 4):
        clock.advance(39)
        await _push(va, capture_ts=2000.0 + 39 * step)
    clock.advance(3)  # 120 s after the opening push
    await _push(va, capture_ts=2120.0)

    calls = _calls(va)
    assert [c.kwargs["blocking"] for c in calls] == [True, False, False, False, True]
    assert "alert_once" not in _data(calls[-1])
    assert _data(calls[-1])["tag"] != _data(calls[0])["tag"]


@pytest.mark.asyncio
async def test_another_camera_is_never_affected(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va)
    clock.advance(5)
    await _push(va, camera_id=_OTHER_CAMERA)

    other = _calls(va)[1]
    assert other.kwargs == {"blocking": True}
    assert "alert_once" not in _data(other)
    assert _data(other)["tag"].startswith("hga_camera_side_")


# ---------------------------------------------------------------------------
# Bypass: only a human-level escalation sounds inside a window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "second_sounds"),
    [
        # Recognized resident, then an explicit unknown face: escalates.
        (("Lindo walks in.", ["Lindo"]), ("A man waits.", ["Unknown Person"]), True),
        # Recognized resident, then a caption human with no face: same person.
        (("Lindo walks in.", ["Lindo"]), ("A person walks away.", []), False),
        # Unrecognized resident, then an unknown face: already unidentified.
        (("A person walks in.", []), ("A man waits.", ["Unknown Person"]), False),
        # Animals, vehicles and packages never escalate.
        (("A person walks past.", []), ("A person with a dog.", []), False),
        # No human, then the first human: escalates.
        (("A car is parked.", []), ("Two men approach.", []), True),
        # Known limit: human words outside the detector's list stay quiet.
        (("A car is parked.", []), ("A visitor approaches.", []), False),
        # A second enrolled person is not an escalation.
        (("Lindo walks in.", ["Lindo"]), ("Sam walks in.", ["Sam"]), False),
        # A negated human phrase is not a human.
        (("A car is parked.", []), ("No people are visible.", []), False),
        # A non-English human caption is not recognized as a human.
        (("A car is parked.", []), ("Osoba stoji u dveri.", []), False),
    ],
)
@pytest.mark.asyncio
async def test_only_an_escalation_sounds_inside_a_window(
    va: VideoAnalyzer,
    clock: _Clock,
    first: tuple[str, list[str]],
    second: tuple[str, list[str]],
    second_sounds: bool,  # noqa: FBT001
) -> None:
    await _push(va, first[0], names=first[1], capture_ts=2000.0)
    clock.advance(20)
    await _push(va, second[0], names=second[1], capture_ts=2020.0)

    follow_up = _calls(va)[1]
    assert follow_up.kwargs == {"blocking": second_sounds}
    assert ("alert_once" in _data(follow_up)) is not second_sounds
    # A sounding push gets its own card; a quiet one reuses the window's.
    same_tag = _data(follow_up)["tag"] == _data(_calls(va)[0])["tag"]
    assert same_tag is not second_sounds


@pytest.mark.asyncio
async def test_improving_face_evidence_on_one_subject_stays_quiet(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """Caption human, then Unknown Person, then an enrolled name: one buzz."""
    await _push(va, "A person approaches.", names=[], capture_ts=2000.0)
    clock.advance(10)
    await _push(va, "A man waits.", names=["Unknown Person"], capture_ts=2010.0)
    clock.advance(10)
    await _push(va, "Lindo waits.", names=["Lindo"], capture_ts=2020.0)

    assert [c.kwargs["blocking"] for c in _calls(va)] == [True, False, False]
    assert va._notify_windows[_CAMERA].level is HumanLevel.UNIDENTIFIED


@pytest.mark.asyncio
async def test_bypass_restarts_the_window(va: VideoAnalyzer, clock: _Clock) -> None:
    await _push(va, "Lindo walks in.", names=["Lindo"], capture_ts=2000.0)
    clock.advance(100)
    await _push(va, "A man waits.", names=["Unknown Person"], capture_ts=2100.0)
    clock.advance(100)  # 200 s after the open, 100 s after the bypass
    await _push(va, "A man still waits.", names=["Unknown Person"], capture_ts=2200.0)

    assert [c.kwargs["blocking"] for c in _calls(va)] == [True, True, False]


# ---------------------------------------------------------------------------
# Stale quiet update (an older batch finishing last)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_older_batch_does_not_overwrite_a_newer_card(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va, capture_ts=2000.0)
    clock.advance(10)
    await _push(va, "Newer scene: a person leaves.", capture_ts=2050.0)
    clock.advance(1)
    await _push(va, "Older scene: a person arrives.", capture_ts=2020.0)

    assert len(_calls(va)) == 2
    assert va._notify_windows[_CAMERA].last_capture == 2050.0


@pytest.mark.asyncio
async def test_an_older_batch_still_sounds_when_it_escalates(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A sounding push is never withheld, however old its batch."""
    await _push(va, "Lindo walks in.", names=["Lindo"], capture_ts=2050.0)
    clock.advance(5)
    await _push(va, "A man waits.", names=["Unknown Person"], capture_ts=2020.0)

    assert [c.kwargs["blocking"] for c in _calls(va)] == [True, True]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_alert_opens_no_window_and_falls_back_for_a_minute(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    _service(va).side_effect = [
        HomeAssistantError("offline"),
        None,
        None,
    ]

    await _push(va)  # sounding attempt fails
    assert _CAMERA not in va._notify_windows

    clock.advance(30)
    await _push(va)  # inside the fallback: exactly the cooldown-off call
    fallback = _calls(va)[1]
    assert _data(fallback) == {"image": str(_IMG)}
    assert fallback.kwargs == {"blocking": False}
    assert _CAMERA not in va._notify_windows

    clock.advance(31)
    await _push(va)  # fallback over: the cooldown is tried again
    retry = _calls(va)[2]
    assert retry.kwargs == {"blocking": True}
    assert "tag" in _data(retry)
    assert _CAMERA in va._notify_windows


@pytest.mark.asyncio
async def test_alert_that_hangs_times_out_and_opens_no_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    async def _hang(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.sleep(3600)

    _service(va).side_effect = _hang
    with patch.object(va_mod, "_NOTIFY_ALERT_TIMEOUT_SEC", 0.01):
        await _push(va)

    assert _CAMERA not in va._notify_windows
    assert va._notify_fallback_until[_CAMERA] == clock.mono + 60


@pytest.mark.asyncio
async def test_failed_bypass_leaves_the_window_unchanged(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va, "Lindo walks in.", names=["Lindo"], capture_ts=2000.0)
    opened = va._notify_windows[_CAMERA]
    _service(va).side_effect = HomeAssistantError("offline")

    clock.advance(10)
    await _push(va, "A man waits.", names=["Unknown Person"], capture_ts=2010.0)

    assert va._notify_windows[_CAMERA] == opened


@pytest.mark.asyncio
async def test_no_notify_service_changes_no_state(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    del entry.runtime_data.options[CONF_NOTIFY_SERVICE]
    with patch.object(va_mod, "discover_mobile_notify_service", return_value=None):
        await _push(va)

    assert _calls(va) == []
    assert va._notify_windows == {}
    assert va._notify_fallback_until == {}


@pytest.mark.asyncio
async def test_discovered_companion_app_service_gets_the_cooldown(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    del entry.runtime_data.options[CONF_NOTIFY_SERVICE]
    with patch.object(
        va_mod, "discover_mobile_notify_service", return_value="mobile_app_found"
    ):
        await _push(va)

    assert _calls(va)[0].args[:2] == ("notify", "mobile_app_found")
    assert _calls(va)[0].kwargs == {"blocking": True}


@pytest.mark.asyncio
async def test_non_companion_target_is_sent_as_if_the_cooldown_were_off(
    va: VideoAnalyzer,
    entry: MagicMock,
    clock: _Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entry.runtime_data.options[CONF_NOTIFY_SERVICE] = "notify.family_group"

    with caplog.at_level("INFO"):
        await _push(va)
        clock.advance(5)
        await _push(va)

    for call in _calls(va):
        assert call.args[:2] == ("notify", "family_group")
        assert _data(call) == {"image": str(_IMG)}
        assert call.kwargs == {"blocking": False}
    assert va._notify_windows == {}
    notices = [r for r in caplog.records if "does not apply" in r.getMessage()]
    assert len(notices) == 1


@pytest.mark.asyncio
async def test_overlapping_batches_on_one_camera_sound_once(
    va: VideoAnalyzer,
) -> None:
    """The lock makes the second batch wait and become a quiet update."""
    release = asyncio.Event()
    started = asyncio.Event()

    async def _slow_first(*_args: Any, **kwargs: Any) -> None:
        if kwargs["blocking"]:
            started.set()
            await release.wait()

    _service(va).side_effect = _slow_first

    first = asyncio.create_task(_push(va, capture_ts=2000.0))
    await started.wait()
    second = asyncio.create_task(_push(va, capture_ts=2008.0))
    await asyncio.sleep(0)
    assert len(_calls(va)) == 1  # the second is parked on the lock
    release.set()
    await asyncio.gather(first, second)

    assert [c.kwargs["blocking"] for c in _calls(va)] == [True, False]
    assert _data(_calls(va)[1])["alert_once"] is True


# ---------------------------------------------------------------------------
# Through _finalize / _handle_notification
# ---------------------------------------------------------------------------


def _batch() -> list[Path]:
    return [_IMG]


def _handle_patches() -> Any:
    return (
        patch.object(va_mod, "latest_target", return_value=MagicMock()),
        patch.object(va_mod, "publish_latest_atomic", new_callable=AsyncMock),
        patch.object(va_mod, "dispatch_on_loop"),
    )


@pytest.mark.asyncio
async def test_cooldown_bug_does_not_skip_storage(va: VideoAnalyzer) -> None:
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MODE] = "always_notify"
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._dispatch_with_cooldown = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("bug")
    )
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3:
        await va._finalize(_CAMERA, _batch(), "A person stands at the door.")

    va.entry.runtime_data.store.aput.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_still_propagates_through_the_cooldown(
    va: VideoAnalyzer,
) -> None:
    va._dispatch_with_cooldown = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await _push(va)


@pytest.mark.asyncio
async def test_novelty_gate_still_runs_before_the_cooldown(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A caption dedup rejects sends nothing, even inside an open window."""
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MODE] = "notify_on_anomaly"
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    decisions = [
        CaptionNoveltyDecision(notify=True, reason="no_match"),
        CaptionNoveltyDecision(notify=False, reason="score_above_threshold"),
    ]
    va._is_caption_novel = AsyncMock(side_effect=decisions)  # type: ignore[method-assign]
    context = _BatchNotifyContext(
        recognized=["Lindo"], batch_names=["Lindo"], capture_ts=2000.0
    )
    stranger = _BatchNotifyContext(
        recognized=["Unknown Person"],
        batch_names=["Unknown Person"],
        capture_ts=2010.0,
    )
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3:
        await va._handle_notification(
            _CAMERA, "Lindo walks in.", _batch(), None, context
        )
        clock.advance(10)
        await va._handle_notification(_CAMERA, "A man waits.", _batch(), None, stranger)

    assert len(_calls(va)) == 1
    assert va._notify_windows[_CAMERA].level is HumanLevel.KNOWN


@pytest.mark.asyncio
async def test_signals_and_novelty_use_the_batch_names_not_the_shared_slot(
    va: VideoAnalyzer,
) -> None:
    """Another batch overwriting _last_recognized must not leak into this one."""
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MODE] = "notify_on_anomaly"
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._is_caption_novel = AsyncMock(  # type: ignore[method-assign]
        return_value=CaptionNoveltyDecision(notify=False, reason="no_match")
    )
    va._last_recognized[_CAMERA] = ["Someone Else"]
    context = _BatchNotifyContext(
        recognized=["Lindo"], batch_names=["Lindo"], capture_ts=2000.0
    )
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3 as dispatch:
        await va._handle_notification(
            _CAMERA, "Lindo walks in.", _batch(), None, context
        )

    latest, recognized = dispatch.call_args_list
    assert latest.args[5] == ["Lindo"]
    assert recognized.args[3] == ["Lindo"]
    novelty_call = va._is_caption_novel.await_args
    assert novelty_call is not None
    assert novelty_call.kwargs["recognized_names"] == ["Lindo"]


@pytest.mark.asyncio
async def test_analyze_and_finalize_passes_full_batch_names_to_the_cooldown(
    va: VideoAnalyzer,
) -> None:
    """Names the summary cap sliced off still reach the cooldown."""
    va._process_batch = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            [{"t+0s. Lindo works in the yard.": ["Lindo"]}],
            ["Lindo"],
            _IMG,
            None,
            ["Lindo", "Unknown Person"],
        )
    )
    va._summarize = AsyncMock(return_value="Lindo works.")  # type: ignore[method-assign]
    va._finalize = AsyncMock()  # type: ignore[method-assign]

    await va._analyze_and_finalize(_CAMERA, [(_IMG, 2000), (_IMG, 2016)])

    finalize_call = va._finalize.await_args
    assert finalize_call is not None
    context = finalize_call.args[4]
    assert context == _BatchNotifyContext(
        recognized=["Lindo"],
        batch_names=["Lindo", "Unknown Person"],
        capture_ts=2016,
    )
