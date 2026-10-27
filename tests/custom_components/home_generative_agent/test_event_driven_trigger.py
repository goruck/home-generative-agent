# ruff: noqa: S101
"""Tests for event-driven triggering — Issue #257."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest

from custom_components.home_generative_agent.sentinel.engine import (
    SentinelEngine,
    _anomaly_type_for_state,
    _parse_rule_entity_exclusions,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.explain.llm_explain import (
        LLMExplainer,
    )


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()
        self._read_only = False

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        """Return state."""
        return self._state

    @property
    def is_read_only(self) -> bool:
        """Return read-only flag."""
        return self._read_only

    async def async_save(self) -> None:  # type: ignore[override]
        """No-op save."""
        return


class DummyNotifier:
    """Notification dispatcher stub."""

    def set_silence_check(self, check: object) -> None:
        """Accept the engine's level-0 check; unused by this stub."""
        del check

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_notify(self, finding, snapshot, explanation) -> None:  # type: ignore[no-untyped-def]
        """Record notification call."""
        self.calls.append({"finding": finding})


class DummyAudit:
    """Audit store stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_append_finding(  # type: ignore[no-untyped-def]
        self, snapshot, finding, explanation, **kwargs: Any
    ) -> None:
        """Record audit call."""
        self.calls.append({"finding": finding})


def _make_engine(hass: Any, **options: Any) -> SentinelEngine:
    return SentinelEngine(
        hass=cast("HomeAssistant", hass),
        options={
            **options,
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
        },
        suppression=DummySuppression(),
        notifier=cast("Any", DummyNotifier()),
        audit_store=cast("Any", DummyAudit()),
        explainer=None,
        entry_id="test_entry",
    )


def _make_state(entity_id: str, domain: str, device_class: str | None = None) -> Any:
    """Create a minimal mock State object."""
    state = MagicMock()
    state.entity_id = entity_id
    state.domain = domain
    attrs: dict[str, Any] = {}
    if device_class is not None:
        attrs["device_class"] = device_class
    state.attributes = attrs
    return state


# ---------------------------------------------------------------------------
# _anomaly_type_for_state unit tests
# ---------------------------------------------------------------------------


def test_lock_domain_maps_to_unlocked_lock_at_night() -> None:
    state = _make_state("lock.front_door", "lock")
    assert _anomaly_type_for_state("lock.front_door", state) == "unlocked_lock_at_night"


def test_camera_domain_maps_to_camera_entry_unsecured() -> None:
    state = _make_state("camera.front_porch", "camera")
    assert (
        _anomaly_type_for_state("camera.front_porch", state) == "camera_entry_unsecured"
    )


def test_person_domain_maps_to_open_entry_while_away() -> None:
    state = _make_state("person.alice", "person")
    assert _anomaly_type_for_state("person.alice", state) == "open_entry_while_away"


def test_binary_sensor_door_maps_correctly() -> None:
    state = _make_state(
        "binary_sensor.front_door", "binary_sensor", device_class="door"
    )
    assert (
        _anomaly_type_for_state("binary_sensor.front_door", state)
        == "open_entry_while_away"
    )


def test_binary_sensor_window_maps_correctly() -> None:
    state = _make_state(
        "binary_sensor.kitchen_window", "binary_sensor", device_class="window"
    )
    assert (
        _anomaly_type_for_state("binary_sensor.kitchen_window", state)
        == "open_entry_while_away"
    )


def test_binary_sensor_gate_maps_correctly() -> None:
    state = _make_state("binary_sensor.gate", "binary_sensor", device_class="gate")
    assert (
        _anomaly_type_for_state("binary_sensor.gate", state) == "open_entry_while_away"
    )


def test_binary_sensor_motion_maps_correctly() -> None:
    state = _make_state(
        "binary_sensor.hallway_motion", "binary_sensor", device_class="motion"
    )
    assert (
        _anomaly_type_for_state("binary_sensor.hallway_motion", state)
        == "camera_entry_unsecured"
    )


def test_binary_sensor_occupancy_maps_correctly() -> None:
    state = _make_state(
        "binary_sensor.living_room_occ", "binary_sensor", device_class="occupancy"
    )
    result = _anomaly_type_for_state("binary_sensor.living_room_occ", state)
    assert result == "unknown_person_camera_no_home"


def test_image_domain_maps_to_unknown_person_rule() -> None:
    """
    HGA image entity updates (recognition results landing) wake the engine.

    The unknown-person rules gate on a 10-minute sighting freshness window;
    the "Unknown Person" label arrives on the image entity minutes after the
    motion event, so without this mapping a sighting could expire between
    polls on installs with a long sentinel_interval_seconds.
    """
    state = _make_state("image.backyard_last_event", "image")
    state.attributes = {
        "camera_id": "camera.backyard",
        "recognized_people": ["Unknown Person"],
    }
    result = _anomaly_type_for_state("image.backyard_last_event", state)
    assert result == "unknown_person_camera_no_home"


def test_third_party_image_entity_does_not_wake_engine() -> None:
    """
    Non-HGA image entities (vacuum maps, radar) must not enqueue triggers.

    An unqualified image-domain match would let a chatty third-party image
    entity drive near-continuous full sentinel evaluations.
    """
    state = _make_state("image.roborock_map", "image")
    assert _anomaly_type_for_state("image.roborock_map", state) is None


def test_image_trigger_respects_camera_exclusion() -> None:
    """
    An image wake-up is suppressed when its CAMERA is excluded.

    Users configure unknown-person exclusions against camera.* ids; the
    wake-up arrives from image.<slug>_last_event, so the engine must resolve
    the image entity to its camera before the exclusion check (#481 contract:
    excluded entities do not wake the engine).
    """
    hass = MagicMock()
    engine = _make_engine(hass)
    engine._rule_entity_exclusions = _parse_rule_entity_exclusions(
        {"unknown_person_camera_no_home": ["camera.backyard"]}
    )

    image_state = _make_state("image.backyard_last_event", "image")
    image_state.attributes = {
        "camera_id": "camera.backyard",
        "recognized_people": ["Unknown Person"],
    }
    event = MagicMock()
    event.data = {"entity_id": "image.backyard_last_event", "new_state": image_state}

    engine._on_state_changed(event)

    assert engine._trigger_scheduler.queue_depth == 0


def test_irrelevant_domain_returns_none() -> None:
    state = _make_state("sensor.temperature", "sensor")
    assert _anomaly_type_for_state("sensor.temperature", state) is None


def test_binary_sensor_with_unknown_device_class_returns_none() -> None:
    state = _make_state("binary_sensor.smoke", "binary_sensor", device_class="smoke")
    assert _anomaly_type_for_state("binary_sensor.smoke", state) is None


# ---------------------------------------------------------------------------
# Event handler tests
# ---------------------------------------------------------------------------


def test_state_change_enqueues_trigger() -> None:
    """Relevant state change enqueues a trigger in the scheduler."""
    hass = MagicMock()
    engine = _make_engine(hass)

    lock_state = _make_state("lock.front", "lock")
    event = MagicMock()
    event.data = {"entity_id": "lock.front", "new_state": lock_state}

    engine._on_state_changed(event)

    assert engine._trigger_scheduler.queue_depth == 1


def test_state_change_removed_entity_ignored() -> None:
    """State changes with new_state=None (entity removed) are ignored."""
    hass = MagicMock()
    engine = _make_engine(hass)

    event = MagicMock()
    event.data = {"entity_id": "lock.front", "new_state": None}

    engine._on_state_changed(event)

    assert engine._trigger_scheduler.queue_depth == 0


def test_irrelevant_entity_not_enqueued() -> None:
    """State changes for irrelevant entities do not enqueue triggers."""
    hass = MagicMock()
    engine = _make_engine(hass)

    temp_state = _make_state("sensor.temperature", "sensor")
    event = MagicMock()
    event.data = {"entity_id": "sensor.temperature", "new_state": temp_state}

    engine._on_state_changed(event)

    assert engine._trigger_scheduler.queue_depth == 0


def test_duplicate_event_within_coalesce_window_produces_one_trigger() -> None:
    """Two rapid state changes for the same anomaly type are coalesced to one."""
    hass = MagicMock()
    engine = _make_engine(hass)

    lock_state = _make_state("lock.front", "lock")
    event = MagicMock()
    event.data = {"entity_id": "lock.front", "new_state": lock_state}

    engine._on_state_changed(event)
    engine._on_state_changed(event)  # second event — same type within 5 s

    assert engine._trigger_scheduler.queue_depth == 1


# ---------------------------------------------------------------------------
# start() / stop() subscription lifecycle
# ---------------------------------------------------------------------------


def test_start_registers_event_listener() -> None:
    """start() subscribes to EVENT_STATE_CHANGED."""
    hass = MagicMock()
    # async_listen returns an unsubscribe callable
    hass.bus.async_listen.return_value = MagicMock()
    # Close the coroutine to prevent "coroutine was never awaited" warnings.
    _task_mock = MagicMock()
    hass.async_create_task.side_effect = lambda coro, **_kw: (coro.close(), _task_mock)[
        1
    ]

    engine = _make_engine(hass)
    engine.start()

    hass.bus.async_listen.assert_called_once()
    assert len(engine._event_unsubscribers) == 1


@pytest.mark.asyncio
async def test_stop_unsubscribes_event_listener() -> None:
    """stop() calls all unsubscribe callbacks."""
    hass = MagicMock()
    unsub = MagicMock()
    hass.bus.async_listen.return_value = unsub

    task = MagicMock()
    task.cancel = MagicMock()

    # Close the coroutine to prevent "coroutine was never awaited" warnings.
    hass.async_create_task.side_effect = lambda coro, **_kw: (coro.close(), task)[1]

    engine = _make_engine(hass)
    engine.start()

    # Manually cancel task simulation
    engine._task = MagicMock()
    engine._task.cancel = MagicMock()

    with suppress(Exception):
        engine._stop_event.set()
        with patch.object(asyncio, "CancelledError", Exception):
            await engine.stop()

    # Unsubscribers must have been called and cleared
    unsub.assert_called_once()
    assert len(engine._event_unsubscribers) == 0


# ---------------------------------------------------------------------------
# Polling still runs when no events queued
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polling_runs_when_no_triggers_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no triggers are queued, run_once_if_triggered returns False and polling runs."""
    snapshot: FullStateSnapshot = validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )

    hass = MagicMock()
    engine = _make_engine(hass)

    # No triggers in queue.
    assert engine._trigger_scheduler.queue_depth == 0

    # run_once_if_triggered returns False when queue is empty.
    triggered = await engine._trigger_scheduler.run_once_if_triggered(engine._run_once)
    assert triggered is False

    # Polling path works normally.
    await engine._trigger_scheduler.run_polling(engine._run_once)
    # Engine ran with empty snapshot — no findings, no notifications.
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    assert notifier.calls == []


# ---------------------------------------------------------------------------
# Each run renews the triage call budget (#866)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_run_renews_the_triage_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run_once tells the triage service a run begins, before any finding."""
    snapshot: FullStateSnapshot = validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )

    events: list[str] = []

    async def _fake_build(_hass: HomeAssistant, **_kwargs: Any) -> FullStateSnapshot:
        events.append("build")
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )
    engine = _make_engine(MagicMock())
    triage = MagicMock()
    triage.begin_run.side_effect = lambda: events.append("begin")
    engine._triage_service = triage
    engine._explain_deferred_this_run = True
    engine._explain_calls_this_run = 3

    await engine._run_once("poll")
    assert engine._explain_deferred_this_run is False
    assert engine._explain_calls_this_run == 0
    assert engine._explain_limit == 5  # the option's default
    await engine._run_once("event")

    # The budget is renewed before the snapshot, so before any finding of
    # the run can reach triage().
    assert events == ["begin", "build", "begin", "build"]
    triage.triage.assert_not_called()


# ---------------------------------------------------------------------------
# Explanations stay inside the run's model budget (#866)
# ---------------------------------------------------------------------------


class _FakeExplainer:
    def __init__(
        self, *, defer_on_call: int | None = None, deployment: str = "edge"
    ) -> None:
        self.calls = 0
        self.last_call_deferred = False
        self.deployment = deployment
        self._defer_on_call = defer_on_call

    async def async_explain(self, _finding: Any) -> str | None:
        self.calls += 1
        self.last_call_deferred = self.calls == self._defer_on_call
        return None if self.last_call_deferred else f"explanation {self.calls}"


@pytest.mark.asyncio
async def test_a_pass_through_finding_gets_no_explanation_call() -> None:
    """Findings past the triage limit, or deferred, do not fan out to explain."""
    engine = _make_engine(MagicMock())
    explainer = _FakeExplainer()
    explainer_arg = cast("LLMExplainer", explainer)
    finding = MagicMock()

    assert await engine._async_explanation(explainer_arg, finding, "triage_cap") is None
    assert (
        await engine._async_explanation(explainer_arg, finding, "triage_deferred")
        is None
    )
    assert explainer.calls == 0
    assert await engine._async_explanation(explainer_arg, finding, "llm_notify") == (
        "explanation 1"
    )
    assert (
        await engine._async_explanation(explainer_arg, finding, None) == "explanation 2"
    )


@pytest.mark.asyncio
async def test_a_deferred_explanation_stops_the_runs_explanations() -> None:
    """Once chat held the model, the rest of the run is skipped; the next run tries again."""
    engine = _make_engine(MagicMock())
    explainer = _FakeExplainer(defer_on_call=2)
    explainer_arg = cast("LLMExplainer", explainer)
    finding = MagicMock()

    assert (
        await engine._async_explanation(explainer_arg, finding, None) == "explanation 1"
    )
    assert (
        await engine._async_explanation(explainer_arg, finding, None) is None
    )  # deferred
    assert (
        await engine._async_explanation(explainer_arg, finding, None) is None
    )  # skipped
    assert explainer.calls == 2

    engine._explain_deferred_this_run = False  # what _run_once does
    assert (
        await engine._async_explanation(explainer_arg, finding, None) == "explanation 3"
    )


@pytest.mark.asyncio
async def test_explanation_calls_are_capped_per_run() -> None:
    """The same limit bounds explanation calls, triage on or off."""
    engine = _make_engine(MagicMock())
    engine._explain_limit = 2
    explainer = _FakeExplainer()
    explainer_arg = cast("LLMExplainer", explainer)
    finding = MagicMock()

    assert (
        await engine._async_explanation(explainer_arg, finding, None) == "explanation 1"
    )
    assert (
        await engine._async_explanation(explainer_arg, finding, None) == "explanation 2"
    )
    assert await engine._async_explanation(explainer_arg, finding, None) is None
    assert explainer.calls == 2


@pytest.mark.asyncio
async def test_the_explanation_limit_comes_from_the_option() -> None:
    """The user's cap reaches the engine, clamped and coerced like the rest."""
    import custom_components.home_generative_agent.sentinel.engine as engine_module  # noqa: PLC0415

    key = engine_module.CONF_SENTINEL_TRIAGE_MAX_PER_RUN
    for raw, expected in ((2, 2), (0, 1), ("abc", 5), (None, 5)):
        engine = _make_engine(MagicMock(), **({key: raw} if raw is not None else {}))
        with (
            patch.object(engine, "_network_context", return_value=None),
            patch(
                "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
                side_effect=ValueError("stop here"),
            ),
        ):
            await engine._run_once("poll")
        assert engine._explain_limit == expected, raw


@pytest.mark.asyncio
async def test_an_explanation_waits_out_the_gap_after_the_last_sentinel_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import custom_components.home_generative_agent.core.utils as utils_mod  # noqa: PLC0415
    import custom_components.home_generative_agent.sentinel.engine as engine_module  # noqa: PLC0415

    monkeypatch.setattr(engine_module, "SENTINEL_TRIAGE_CALL_GAP_S", 0.2)
    utils_mod._sentinel_last_call_end = time.monotonic()
    engine = _make_engine(MagicMock())
    explainer = _FakeExplainer()
    start = time.monotonic()
    assert (
        await engine._async_explanation(
            cast("LLMExplainer", explainer), MagicMock(), None
        )
        == "explanation 1"
    )
    # gstack-timing-ok: lower bound only
    assert time.monotonic() - start >= 0.19


@pytest.mark.asyncio
async def test_a_cloud_explainer_is_never_capped() -> None:
    engine = _make_engine(MagicMock())
    engine._explain_limit = 1
    engine._explain_deferred_this_run = True
    explainer = _FakeExplainer(deployment="cloud")
    arg = cast("LLMExplainer", explainer)
    assert (
        await engine._async_explanation(arg, MagicMock(), "triage_cap")
        == "explanation 1"
    )
    assert await engine._async_explanation(arg, MagicMock(), None) == "explanation 2"
