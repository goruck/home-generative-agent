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
from homeassistant.exceptions import ServiceNotFound

from custom_components.home_generative_agent.const import (
    CONF_NOTIFY_SERVICE,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S,
    VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_MAX_S,
)
from custom_components.home_generative_agent.core import video_analyzer as va_mod
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


def _tag(call: Any) -> str:
    return _data(call)["tag"]


def _sounds(call: Any) -> bool:
    return "alert_once" not in _data(call)


def _context(
    names: list[str] | None = None, *, batch_names: list[str] | None = None
) -> _BatchNotifyContext:
    """Names on the displayed frames; `batch_names` when the whole batch differs."""
    shown = names or []
    return _BatchNotifyContext(
        recognized=shown, batch_names=shown if batch_names is None else batch_names
    )


async def _push(
    va: VideoAnalyzer,
    msg: str = "A person stands at the door.",
    *,
    names: list[str] | None = None,
    batch_names: list[str] | None = None,
    camera_id: str = _CAMERA,
) -> None:
    await va._dispatch_notification(
        camera_id, msg, _IMG, _context(names, batch_names=batch_names)
    )


# ---------------------------------------------------------------------------
# Option off: the regression contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 0.0, None, "", "abc", -5, "inf", "1e999"])
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
    assert _data(_calls(va)[1]) == {"image": str(_IMG)}
    assert va._notify_windows == {}


@pytest.mark.parametrize("value", [120, 120.0, "120", 1.0])
@pytest.mark.asyncio
async def test_cooldown_is_on_for_the_values_the_options_flow_stores(
    va: VideoAnalyzer, entry: MagicMock, clock: _Clock, value: object
) -> None:
    """The number selector stores a float; a string survives a restore."""
    entry.runtime_data.options[CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S] = value

    await _push(va)

    assert "tag" in _data(_calls(va)[0])
    assert va._notify_windows[_CAMERA].started == clock.mono


def test_cooldown_is_clamped_to_the_documented_maximum(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    entry.runtime_data.options[CONF_VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_S] = 999_999

    assert va._notification_cooldown_s() == VIDEO_ANALYZER_NOTIFICATION_COOLDOWN_MAX_S


# ---------------------------------------------------------------------------
# Window: open, quiet update, expiry, camera isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_push_sounds_and_later_ones_replace_the_card_quietly(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va)
    clock.advance(30)
    await _push(va, "A person still stands at the door.")

    alert, update = _calls(va)
    assert _data(alert) == {"image": str(_IMG), "tag": _tag(alert)}
    assert _tag(alert).endswith("_1790000000")
    assert _data(update) == {
        "image": str(_IMG),
        "tag": _tag(alert),
        "alert_once": True,
        "push": _QUIET_PUSH,
    }
    # Sent exactly like today's push: fire-and-forget, never awaited.
    assert alert.kwargs == update.kwargs == {"blocking": False}
    assert update.args[2]["message"] == "A person still stands at the door."


@pytest.mark.asyncio
async def test_quiet_updates_do_not_extend_the_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A lingering subject sounds again once the fixed window has run out."""
    await _push(va)
    for _ in range(3):
        clock.advance(39)
        await _push(va)
    clock.advance(3)  # 120 s after the opening push
    await _push(va)

    calls = _calls(va)
    assert [_sounds(c) for c in calls] == [True, False, False, False, True]
    assert _tag(calls[-1]) != _tag(calls[0])


@pytest.mark.asyncio
async def test_another_camera_is_never_affected(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va)
    clock.advance(5)
    await _push(va, camera_id=_OTHER_CAMERA)

    first, other = _calls(va)
    assert _sounds(other)
    assert _tag(other) != _tag(first)


# ---------------------------------------------------------------------------
# Unknown-face bypass, quiet update, follow-up card
# ---------------------------------------------------------------------------

_UNKNOWN = ["Unknown Person"]
_SOUND = "sound"
_QUIET = "quiet"
_FOLLOW_UP = "follow_up"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # The first face-confirmed unknown person in a window sounds...
        (("Lindo walks in.", ["Lindo"]), ("A man waits.", _UNKNOWN), _SOUND),
        (("A person walks in.", []), ("A man waits.", _UNKNOWN), _SOUND),
        (("A car is parked.", []), ("A man waits.", _UNKNOWN), _SOUND),
        # ...and nothing else does. Captions decide nothing, whatever they say.
        (("A car is parked.", []), ("Two men approach.", []), _QUIET),
        (("A car is parked.", []), ("An intruder climbs the fence.", []), _QUIET),
        (("A car is parked.", []), ("No people are visible.", []), _QUIET),
        (("A person walks past.", []), ("A person with a dog.", []), _QUIET),
        (("Lindo walks in.", ["Lindo"]), ("A person walks away.", []), _QUIET),
        (("Lindo walks in.", ["Lindo"]), ("Sam walks in.", ["Sam"]), _QUIET),
        (("Lindo walks in.", ["Lindo"]), ("A car is parked.", []), _QUIET),
        # A window that opened on an unknown face has had its one alert.
        (("A man waits.", _UNKNOWN), ("A man still waits.", _UNKNOWN), _QUIET),
        # After an unknown-person alert, a push that does not show an unknown
        # face must not replace that alert's card.
        (("A man waits.", _UNKNOWN), ("A car is parked.", []), _FOLLOW_UP),
        (("A man waits.", _UNKNOWN), ("Lindo walks in.", ["Lindo"]), _FOLLOW_UP),
    ],
)
@pytest.mark.asyncio
async def test_what_the_second_push_in_a_window_does(
    va: VideoAnalyzer,
    clock: _Clock,
    first: tuple[str, list[str]],
    second: tuple[str, list[str]],
    expected: str,
) -> None:
    await _push(va, first[0], names=first[1])
    clock.advance(20)
    await _push(va, second[0], names=second[1])

    alert, follow_up = _calls(va)
    assert _sounds(follow_up) is (expected == _SOUND)
    if expected == _QUIET:
        assert _tag(follow_up) == _tag(alert)
    elif expected == _FOLLOW_UP:
        assert _tag(follow_up) == _tag(alert) + "_more"
    else:
        # A sounding push gets its own card.
        assert not _tag(follow_up).startswith(_tag(alert))


@pytest.mark.asyncio
async def test_unknown_face_sounds_once_per_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A lingering unknown visitor is one alert, then quiet updates."""
    await _push(va, "A car is parked.")
    clock.advance(10)
    await _push(va, "A man waits.", names=_UNKNOWN)
    clock.advance(10)
    await _push(va, "A man still waits.", names=_UNKNOWN)
    clock.advance(10)
    await _push(va, "A man knocks.", names=_UNKNOWN)

    calls = _calls(va)
    assert [_sounds(c) for c in calls] == [True, True, False, False]
    assert _tag(calls[2]) == _tag(calls[3]) == _tag(calls[1])
    assert va._notify_windows[_CAMERA].unknown_face


@pytest.mark.asyncio
async def test_unknown_face_alert_does_not_restart_the_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """The window is a fixed length from its opening push, whatever sounds in it."""
    await _push(va, "Lindo walks in.", names=["Lindo"])
    opened_at = va._notify_windows[_CAMERA].started
    clock.advance(100)
    await _push(va, "A man waits.", names=_UNKNOWN)
    assert va._notify_windows[_CAMERA].started == opened_at
    clock.advance(20)  # 120 s after the open: the window is over
    await _push(va, "A man still waits.", names=_UNKNOWN)

    assert [_sounds(c) for c in _calls(va)] == [True, True, True]


@pytest.mark.asyncio
async def test_unknown_face_in_a_frame_the_summary_dropped_still_sounds(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """Escalation reads the whole batch, not only the frames that are shown."""
    await _push(va, "Lindo works in the yard.", names=["Lindo"])
    clock.advance(30)
    await _push(
        va,
        "Lindo keeps working in the yard.",
        names=["Lindo"],
        batch_names=["Lindo", "Unknown Person"],
    )

    assert [_sounds(c) for c in _calls(va)] == [True, True]


@pytest.mark.asyncio
async def test_push_not_showing_the_unknown_cannot_replace_the_alert_card(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """
    Which card a quiet update lands on follows what it displays.

    The batch still carries an unknown face in an early frame, but the summary
    and image show an empty scene, so it must go to the follow-up card.
    """
    await _push(va, "A man waits.", names=_UNKNOWN)
    clock.advance(30)
    await _push(va, "The porch is empty.", names=[], batch_names=_UNKNOWN)

    alert, update = _calls(va)
    assert not _sounds(update)
    assert _tag(update) == _tag(alert) + "_more"


@pytest.mark.asyncio
async def test_unknown_face_alert_in_the_same_second_gets_a_new_card(
    va: VideoAnalyzer,
) -> None:
    """A batch parked on the lock can dispatch within the opening second."""
    await _push(va, "Lindo walks in.", names=["Lindo"])
    await _push(va, "A man waits.", names=_UNKNOWN)

    first, second = _calls(va)
    assert _sounds(second)
    assert _tag(second) != _tag(first)


@pytest.mark.asyncio
async def test_follow_up_updates_share_one_card_and_leave_the_alert_card(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va, "A man waits.", names=_UNKNOWN)
    clock.advance(10)
    await _push(va, "A car is parked.")
    clock.advance(10)
    await _push(va, "The porch is quiet.")
    clock.advance(10)
    await _push(va, "A man returns.", names=_UNKNOWN)

    alert, calm_1, calm_2, same = _calls(va)
    assert _tag(calm_1) == _tag(calm_2) == _tag(alert) + "_more"
    assert not _sounds(calm_1)
    assert _tag(same) == _tag(alert)
    assert not _sounds(same)


@pytest.mark.asyncio
async def test_a_different_notify_target_starts_its_own_window(
    va: VideoAnalyzer, entry: MagicMock, clock: _Clock
) -> None:
    """A phone that never got the sounding push must not get a quiet one."""
    await _push(va)
    entry.runtime_data.options[CONF_NOTIFY_SERVICE] = "notify.mobile_app_other"
    clock.advance(10)
    await _push(va)

    first, second = _calls(va)
    assert second.args[:2] == ("notify", "mobile_app_other")
    assert _sounds(second)
    assert _tag(second) != _tag(first)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_the_service_rejects_opens_no_window(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    """A push that cannot even be handed over must not start a quiet window."""
    _service(va).side_effect = [
        ServiceNotFound("notify", "mobile_app_phone"),
        None,
    ]

    await _push(va)  # must not raise
    assert _CAMERA not in va._notify_windows

    clock.advance(5)
    await _push(va)
    assert _sounds(_calls(va)[1])
    assert _CAMERA in va._notify_windows


@pytest.mark.asyncio
async def test_failed_unknown_face_alert_and_quiet_update_leave_the_window_unchanged(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    await _push(va, "Lindo walks in.", names=["Lindo"])
    opened = va._notify_windows[_CAMERA]
    _service(va).side_effect = ServiceNotFound("notify", "mobile_app_phone")

    clock.advance(10)
    await _push(va, "A man waits.", names=_UNKNOWN)  # failed sounding push
    clock.advance(10)
    await _push(va, "Lindo sits down.", names=["Lindo"])  # failed quiet update

    assert va._notify_windows[_CAMERA] == opened


@pytest.mark.asyncio
async def test_failure_on_one_camera_does_not_touch_another(
    va: VideoAnalyzer, clock: _Clock
) -> None:
    _service(va).side_effect = [ServiceNotFound("notify", "mobile_app_phone"), None]

    await _push(va)
    clock.advance(5)
    await _push(va, camera_id=_OTHER_CAMERA)

    assert _CAMERA not in va._notify_windows
    assert _OTHER_CAMERA in va._notify_windows


@pytest.mark.asyncio
async def test_cooldown_bug_still_sends_the_push_without_the_cooldown(
    va: VideoAnalyzer, caplog: pytest.LogCaptureFixture
) -> None:
    """A bug in the decision costs the convenience, never the alert."""
    with patch.object(va_mod, "decide", side_effect=RuntimeError("bug")):
        await _push(va)

    (call,) = _calls(va)
    assert _data(call) == {"image": str(_IMG)}
    assert va._notify_windows == {}
    assert "Notification cooldown failed" in caplog.text


@pytest.mark.asyncio
async def test_no_notify_service_changes_no_state(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    del entry.runtime_data.options[CONF_NOTIFY_SERVICE]
    with patch.object(va_mod, "discover_mobile_notify_service", return_value=None):
        await _push(va)

    assert _calls(va) == []
    assert va._notify_windows == {}


@pytest.mark.asyncio
async def test_discovered_companion_app_service_gets_the_cooldown(
    va: VideoAnalyzer, entry: MagicMock
) -> None:
    del entry.runtime_data.options[CONF_NOTIFY_SERVICE]
    with patch.object(
        va_mod, "discover_mobile_notify_service", return_value="mobile_app_found"
    ) as discover:
        await _push(va)

    assert _calls(va)[0].args[:2] == ("notify", "mobile_app_found")
    assert "tag" in _data(_calls(va)[0])
    # Resolved once: the service the decision was made for gets the push.
    discover.assert_called_once()


@pytest.mark.asyncio
async def test_non_companion_target_is_sent_as_if_the_cooldown_were_off(
    va: VideoAnalyzer,
    entry: MagicMock,
    clock: _Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entry.runtime_data.options[CONF_NOTIFY_SERVICE] = "notify.family_group"

    await _push(va)
    clock.advance(5)
    await _push(va)

    for call in _calls(va):
        assert call.args[:2] == ("notify", "family_group")
        assert _data(call) == {"image": str(_IMG)}
        assert call.kwargs == {"blocking": False}
    assert va._notify_windows == {}
    # Visible at Home Assistant's default log level, and only once.
    notices = [
        r
        for r in caplog.records
        if "does not apply" in r.getMessage() and r.levelname == "WARNING"
    ]
    assert len(notices) == 1


@pytest.mark.asyncio
async def test_overlapping_batches_on_one_camera_sound_once(
    va: VideoAnalyzer,
) -> None:
    """The lock makes the second batch wait and become a quiet update."""
    release = asyncio.Event()
    started = asyncio.Event()

    async def _slow_first(*_args: Any, **_kwargs: Any) -> None:
        if not started.is_set():
            started.set()
            await release.wait()

    _service(va).side_effect = _slow_first

    first = asyncio.create_task(_push(va))
    await started.wait()
    second = asyncio.create_task(_push(va))
    await asyncio.sleep(0)
    assert len(_calls(va)) == 1  # the second is parked on the lock
    release.set()
    await asyncio.gather(first, second)

    assert [_sounds(c) for c in _calls(va)] == [True, False]


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
async def test_failed_push_with_the_cooldown_on_does_not_skip_storage(
    va: VideoAnalyzer,
) -> None:
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MODE] = "always_notify"
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    _service(va).side_effect = ServiceNotFound("notify", "mobile_app_phone")
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3:
        await va._finalize(
            _CAMERA, _batch(), "A person stands at the door.", context=_context()
        )

    va.entry.runtime_data.store.aput.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_still_propagates_through_the_cooldown(
    va: VideoAnalyzer,
) -> None:
    _service(va).side_effect = asyncio.CancelledError

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
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3:
        await va._handle_notification(
            _CAMERA, "Lindo walks in.", _batch(), context=_context(["Lindo"])
        )
        clock.advance(10)
        await va._handle_notification(
            _CAMERA, "A man waits.", _batch(), context=_context(["Unknown Person"])
        )

    assert len(_calls(va)) == 1
    assert not va._notify_windows[_CAMERA].unknown_face


@pytest.mark.asyncio
async def test_signals_and_novelty_use_the_batch_names(va: VideoAnalyzer) -> None:
    """Sensor, image entity and dedup all get the names this batch carried."""
    va.entry.runtime_data.options[CONF_VIDEO_ANALYZER_MODE] = "notify_on_anomaly"
    va.protect_notify_image = MagicMock()  # type: ignore[method-assign]
    va._is_caption_novel = AsyncMock(  # type: ignore[method-assign]
        return_value=CaptionNoveltyDecision(notify=False, reason="no_match")
    )
    p1, p2, p3 = _handle_patches()

    with p1, p2, p3 as dispatch:
        await va._handle_notification(
            _CAMERA, "Lindo walks in.", _batch(), context=_context(["Lindo"])
        )

    latest, recognized = dispatch.call_args_list
    assert latest.args[5] == ["Lindo"]
    assert recognized.args[3] == ["Lindo"]
    novelty_call = va._is_caption_novel.await_args
    assert novelty_call is not None
    assert novelty_call.kwargs["recognized_names"] == ["Lindo"]


@pytest.mark.asyncio
async def test_overlapping_analyses_each_keep_their_own_names(
    va: VideoAnalyzer,
) -> None:
    """
    Two analyses of one camera must not swap names (the shared-slot race).

    The first batch is parked in its summary call while a second batch for the
    same camera runs to completion. Each must reach _finalize with the names
    its own frames produced.
    """
    release = asyncio.Event()
    batches = {
        2000: (["Lindo"], ["Lindo"]),
        3000: (["Sam"], ["Sam", "Unknown Person"]),
    }

    async def _process(_camera: str, ordered: list[tuple[Path, int]]) -> Any:
        recognized, batch_names = batches[ordered[0][1]]
        return [{"t+0s. caption": recognized}], recognized, _IMG, None, batch_names

    async def _summarize(_camera: str, descs: Any, _sole: Any) -> str:
        if descs[0]["t+0s. caption"] == ["Lindo"]:
            await release.wait()
        return "summary"

    va._process_batch = AsyncMock(side_effect=_process)  # type: ignore[method-assign]
    va._summarize = AsyncMock(side_effect=_summarize)  # type: ignore[method-assign]
    va._finalize = AsyncMock()  # type: ignore[method-assign]

    first = asyncio.create_task(va._analyze_and_finalize(_CAMERA, [(_IMG, 2000)]))
    await asyncio.sleep(0)
    await va._analyze_and_finalize(_CAMERA, [(_IMG, 3000)])
    release.set()
    await first

    contexts = [c.kwargs["context"] for c in va._finalize.await_args_list]
    assert contexts == [
        _BatchNotifyContext(recognized=["Sam"], batch_names=["Sam", "Unknown Person"]),
        _BatchNotifyContext(recognized=["Lindo"], batch_names=["Lindo"]),
    ]
