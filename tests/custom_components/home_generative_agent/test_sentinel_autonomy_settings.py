# ruff: noqa: S101
"""
Sentinel's autonomy level and auto-execution are settings a user can set (#783).

The engine read a configured level, an override duration and the auto-execution
guardrails, but no form or resolver ever supplied them: the level was always 1,
an override lasted an hour, and levels 2 and 3 could neither act nor report what
they would do. A finding's suggested action may now run on its own, never a
critical one, and dry-run mode names it in the notification instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_PROMPT_USER,
    CONF_CRITICAL_ACTIONS,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_AUTONOMY_LEVEL,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.flows.sentinel_subentry_flow import (
    _default_payload,
)
from custom_components.home_generative_agent.sentinel.execution import (
    SentinelExecutionService,
)
from custom_components.home_generative_agent.sentinel.notifier import canary_line

from .test_execution_service import _make_finding, _make_snapshot
from .test_sentinel_autonomy_level import _make_engine
from .test_sentinel_notifier import (
    _finding_with_severity,
    _make_notifier,
    _minimal_snapshot,
)
from .test_subentries import DummyEntry, DummySubentry, _make_sentinel_flow_for_mode

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_NOW = datetime(2025, 1, 1, 1, 0, 0, tzinfo=UTC)
_RECENT = "2025-01-01T00:59:50+00:00"

# Every guardrail set so that only the action gate decides.
_ACTING: dict[str, Any] = {
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.5,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 10,
}


def _evaluate(
    actions: list[str],
    allowed: list[str],
    *,
    entities: list[str] | None = None,
    **options: Any,
) -> Any:
    svc = SentinelExecutionService(
        {
            **_ACTING,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: allowed,
            **options,
        }
    )
    targets = entities or ["lock.front_door"]
    finding = _make_finding(
        confidence=0.9, suggested_actions=actions, triggering_entities=targets
    )
    snapshot = _make_snapshot(entity_id=targets[0], last_changed=_RECENT)
    return svc.evaluate(finding, snapshot, 2, _NOW)


# ---------------------------------------------------------------------------
# Execution: what may run on its own
# ---------------------------------------------------------------------------


def test_an_allowlisted_safe_action_runs() -> None:
    """Locking a door is not a critical action, so an allowed lock.lock runs."""
    result = _evaluate(["lock.lock"], ["lock"])
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


@pytest.mark.parametrize(
    ("actions", "allowed", "entities"),
    [
        # A Critical Action PIN rule covers it.
        (["lock.unlock"], ["lock"], ["lock.front_door"]),
        (["cover.open_cover"], ["cover"], ["cover.garage_door"]),
        # An alarm panel guards itself; Sentinel never changes one.
        (
            ["alarm_control_panel.alarm_disarm"],
            ["alarm_control_panel"],
            ["alarm_control_panel.home"],
        ),
        (
            ["alarm_control_panel.alarm_arm_away"],
            ["alarm_control_panel.alarm_arm_away"],
            ["alarm_control_panel.home"],
        ),
        # Starts configuration whose steps are not read here.
        (["script.turn_on"], ["script"], ["script.anything"]),
        (["automation.trigger"], ["automation"], ["automation.anything"]),
    ],
)
def test_a_critical_action_never_runs_whatever_the_allowlist(
    actions: list[str], allowed: list[str], entities: list[str]
) -> None:
    """Allowlisting a critical action leaves it to the user."""
    result = _evaluate(actions, allowed, entities=entities)
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "critical_action"


def test_a_users_own_pin_rule_bars_an_action() -> None:
    """The user's Critical Action PIN rules, not only the defaults, apply."""
    result = _evaluate(
        ["light.turn_off"],
        ["light"],
        entities=["light.porch"],
        **{CONF_CRITICAL_ACTIONS: [{"domain": "light", "service": "turn_off"}]},
    )
    assert result.block_reason == "critical_action"


def test_advice_beside_an_action_does_not_stop_it() -> None:
    """`lock_entity` is advice, never called, so it needs no allowlist entry."""
    result = _evaluate(["lock.lock", "lock_entity"], ["lock"])
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


def test_advice_alone_is_left_to_the_user() -> None:
    """With nothing callable, nothing runs."""
    result = _evaluate(["check_camera"], ["lock"])
    assert result.block_reason == "no_executable_action"


def test_an_unlisted_action_does_not_run() -> None:
    """An empty allowlist runs nothing."""
    result = _evaluate(["lock.lock"], [])
    assert result.block_reason == "service_not_allowlisted"


def test_dry_run_reports_without_the_execution_switch() -> None:
    """Canary mode says what would run even with live execution off."""
    options: dict[str, Any] = {
        **_ACTING,
        CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False,
        CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["lock"],
    }
    svc = SentinelExecutionService(options)
    finding = _make_finding(
        confidence=0.9,
        suggested_actions=["lock.lock"],
        triggering_entities=["lock.front_door"],
    )
    snapshot = _make_snapshot(entity_id="lock.front_door", last_changed=_RECENT)
    canary = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert canary.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
    live = svc.evaluate(finding, snapshot, 2, _NOW)
    assert live.block_reason == "auto_execution_disabled"


# ---------------------------------------------------------------------------
# Dry-run line in the notification
# ---------------------------------------------------------------------------


def test_the_dry_run_line_names_the_action_and_the_device() -> None:
    """Device names come from Home Assistant when it knows them."""
    finding = _make_finding(
        suggested_actions=["lock.lock"], triggering_entities=["lock.front_door"]
    )
    line = canary_line(finding, ["lock.lock"], None)
    assert line == "Dry run: Sentinel would have run lock.lock on Front Door."


@pytest.mark.asyncio
async def test_the_dry_run_line_is_added_and_never_held() -> None:
    """Like the also line, it reaches the user in the push that carries it."""
    notifier, hass, _s, _a = _make_notifier(
        {CONF_NOTIFY_SERVICE: "notify.mobile_app_phone"}
    )
    snapshot = _minimal_snapshot()
    for i in range(3):
        await notifier.async_notify(
            _finding_with_severity("low", anomaly_id=f"burst{i}"),
            snapshot,  # type: ignore[arg-type]
            f"Burst {i}.",
        )
    line = "Dry run: Sentinel would have run lock.lock on Front Door."
    sent = await notifier.async_notify(
        _finding_with_severity("low", anomaly_id="canary"),
        snapshot,  # type: ignore[arg-type]
        "Front door is unlocked.",
        canary_line=line,
    )
    assert sent is True
    assert notifier._held_batch == []
    assert hass.services.calls[-1]["data"]["message"].endswith(line)


# ---------------------------------------------------------------------------
# Engine: the configured level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "level"),
    [(2, 2), ("3", 3), (0, 0), (7, RECOMMENDED_SENTINEL_AUTONOMY_LEVEL), (None, 1)],
)
def test_the_engine_runs_at_the_configured_level(
    configured: object, level: int
) -> None:
    """A value outside 0-3 reads as the default."""
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: configured})
    assert engine.get_autonomy_level("test-entry-1") == level


# ---------------------------------------------------------------------------
# Resolver: the settings reach the engine
# ---------------------------------------------------------------------------


def test_the_settings_reach_the_engine() -> None:
    """Every autonomy setting stored in the subentry is in the runtime options."""
    stored: dict[str, Any] = {
        CONF_SENTINEL_ENABLED: True,
        CONF_SENTINEL_AUTONOMY_LEVEL: 2,
        CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 1440,
        CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
        CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False,
        CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE: 0.8,
        CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 3,
        CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["lock.lock"],
    }
    entry = DummyEntry(options={})
    sentinel = DummySubentry("sentinel1", SUBENTRY_TYPE_SENTINEL, "Sentinel", stored)
    entry.subentries[sentinel.subentry_id] = sentinel
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    for key, value in stored.items():
        assert options[key] == value


# ---------------------------------------------------------------------------
# Options form
# ---------------------------------------------------------------------------


def _settings(**changes: Any) -> dict[str, Any]:
    payload = _default_payload()
    payload[CONF_SENTINEL_AUTONOMY_LEVEL] = str(payload[CONF_SENTINEL_AUTONOMY_LEVEL])
    payload.update(changes)
    return payload


@pytest.mark.asyncio
async def test_the_form_stores_the_settings_as_the_engine_reads_them(
    hass: HomeAssistant,
) -> None:
    """The level is stored as a number and the allowlist normalized."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    result = await flow.async_step_settings(
        _settings(
            **{
                CONF_SENTINEL_AUTONOMY_LEVEL: "2",
                CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 1440.0,
                CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
                CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: [
                    " Lock ",
                    "lock.lock",
                    "lock",
                ],
            }
        )
    )
    assert result.get("type") == "create_entry"
    data = result.get("data") or {}
    assert data[CONF_SENTINEL_AUTONOMY_LEVEL] == 2
    assert data[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 1440
    assert data[CONF_SENTINEL_AUTO_EXEC_CANARY_MODE] is True
    assert data[CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES] == ["lock", "lock.lock"]


@pytest.mark.asyncio
async def test_the_form_refuses_an_allowlist_entry_it_cannot_match(
    hass: HomeAssistant,
) -> None:
    """Only a domain or a domain.service can match a suggested action."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    result = await flow.async_step_settings(
        _settings(**{CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["lock unlock"]})
    )
    assert result.get("type") == "form"
    assert result.get("errors") == {"base": "invalid_allowed_services"}


def test_the_defaults_change_nothing() -> None:
    """A fresh setup runs at level 1 with execution and dry-run off."""
    payload = _default_payload()
    assert payload[CONF_SENTINEL_AUTONOMY_LEVEL] == 1
    assert payload[CONF_SENTINEL_AUTO_EXECUTION_ENABLED] is False
    assert payload[CONF_SENTINEL_AUTO_EXEC_CANARY_MODE] is False
    assert payload[CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES] == []
