# ruff: noqa: S101
"""
Sentinel's autonomy level and override duration are settings (#783).

The engine read a configured level and an override duration that no form or
resolver supplied: the level was always 1 and an override lasted an hour. The
auto-execution guardrails and the dry-run line are in place but not offered in
the options yet (#789), so nothing here can act.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

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
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_QUIET_HOURS_START,
    CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.flows.sentinel_subentry_flow import (
    _default_payload,
    _normalize_autonomy,
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
        # A cover can be a garage whose id names no door; never on its own.
        (["cover.close_cover"], ["cover"], ["cover.ratgdo_main"]),
        # Through the generic domain, the floor reads the targets' domains.
        (["homeassistant.toggle"], ["homeassistant"], ["cover.ratgdo_main"]),
        (["homeassistant.turn_off"], ["homeassistant"], ["lock.front_door"]),
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


def test_an_unlock_never_runs_without_the_lock_rules() -> None:
    """A user who removed the lock rules still never gets an unattended unlock."""
    rules: dict[str, Any] = {
        CONF_CRITICAL_ACTIONS: [{"domain": "light", "service": "turn_off"}]
    }
    result = _evaluate(["lock.unlock"], ["lock"], **rules)
    assert result.block_reason == "critical_action"


def test_prose_with_a_dot_is_advice_not_an_action() -> None:
    """Only a `domain.service` slug pair is ever called."""
    result = _evaluate(["Check the router. Then reboot it."], ["lock"])
    assert result.block_reason == "no_executable_action"


def test_levels_two_and_three_drop_nothing_while_no_action_can_run() -> None:
    """With auto-execution off, an unavailable device does not drop a finding."""
    svc = SentinelExecutionService({CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False})
    finding = _make_finding(severity="medium", triggering_entities=["sensor.gone"])
    snapshot = _make_snapshot(entity_id="lock.front_door", last_changed=_RECENT)
    for evaluate in (svc.evaluate, svc.evaluate_canary):
        result = evaluate(finding, snapshot, 2, _NOW)
        assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
        assert result.block_reason == "auto_execution_disabled"


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


def _resolved(stored: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    entry = DummyEntry(options=parent)
    sentinel = DummySubentry("sentinel1", SUBENTRY_TYPE_SENTINEL, "Sentinel", stored)
    entry.subentries[sentinel.subentry_id] = sentinel
    return resolve_runtime_options(entry)  # type: ignore[arg-type]


def test_the_settings_reach_the_engine() -> None:
    """The level and override duration stored in the subentry are used."""
    options = _resolved(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 2,
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 1440,
        },
        {},
    )
    assert options[CONF_SENTINEL_AUTONOMY_LEVEL] == 2
    assert options[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 1440


def test_an_older_installs_level_is_kept() -> None:
    """A level stored in the entry options survives a subentry without one."""
    options = _resolved(
        {CONF_SENTINEL_ENABLED: True},
        {
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 120,
        },
    )
    assert options[CONF_SENTINEL_AUTONOMY_LEVEL] == 0
    assert options[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 120


def test_an_older_setting_cannot_turn_automatic_actions_on() -> None:
    """A value left in the entry options does not enable them (#789)."""
    options = _resolved(
        {CONF_SENTINEL_ENABLED: True},
        {
            CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
            CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
        },
    )
    assert options[CONF_SENTINEL_AUTO_EXECUTION_ENABLED] is False
    assert options[CONF_SENTINEL_AUTO_EXEC_CANARY_MODE] is False


def test_auto_execution_is_not_set_by_the_subentry() -> None:
    """Automatic actions are not offered yet, so nothing turns them on."""
    options = _resolved(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
            CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: True,
        },
        {},
    )
    assert not options.get(CONF_SENTINEL_AUTO_EXECUTION_ENABLED)
    assert not options.get(CONF_SENTINEL_AUTO_EXEC_CANARY_MODE)


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
    """The level and the duration are stored as numbers."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    result = await flow.async_step_settings(
        _settings(
            **{
                CONF_SENTINEL_AUTONOMY_LEVEL: "2",
                CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 1440.0,
            }
        )
    )
    assert result.get("type") == "create_entry"
    data = result.get("data") or {}
    assert data[CONF_SENTINEL_AUTONOMY_LEVEL] == 2
    assert data[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 1440


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [1, 2, 0])
async def test_the_form_accepts_its_own_suggested_values(
    hass: HomeAssistant, stored: int
) -> None:
    """
    Saving the form untouched passes its own schema.

    The level is stored as a number and the select takes text: pre-filled
    with the number, every untouched save was refused.
    """
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {**_default_payload(), CONF_SENTINEL_AUTONOMY_LEVEL: stored},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    form = await flow.async_step_settings()
    schema = cast("Any", form).get("data_schema")
    assert schema is not None
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested[CONF_SENTINEL_AUTONOMY_LEVEL] == str(stored)
    validated = schema(suggested)
    assert validated[CONF_SENTINEL_AUTONOMY_LEVEL] == str(stored)


def test_the_defaults_change_nothing() -> None:
    """A fresh setup runs at level 1 with no automatic actions."""
    payload = _default_payload()
    assert payload[CONF_SENTINEL_AUTONOMY_LEVEL] == 1
    assert CONF_SENTINEL_AUTO_EXECUTION_ENABLED not in payload
    assert CONF_SENTINEL_AUTO_EXEC_CANARY_MODE not in payload


@pytest.mark.asyncio
async def test_the_form_shows_and_keeps_an_older_installs_level(
    hass: HomeAssistant,
) -> None:
    """A level in the entry options is the one in force, so the form keeps it."""
    existing = DummySubentry(
        "sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {CONF_SENTINEL_ENABLED: True}
    )
    entry = DummyEntry(options={CONF_SENTINEL_AUTONOMY_LEVEL: 0})
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    form = await flow.async_step_settings()
    schema = cast("Any", form).get("data_schema")
    assert schema is not None
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested[CONF_SENTINEL_AUTONOMY_LEVEL] == "0"


@pytest.mark.asyncio
async def test_basic_setup_keeps_the_level(hass: HomeAssistant) -> None:
    """A Sentinel set to 0 stays silent when Basic setup runs again."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True, CONF_SENTINEL_AUTONOMY_LEVEL: 0},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    updates: list[Any] = []
    flow.async_update_and_abort = lambda *_a, **kw: (  # type: ignore[assignment]
        updates.append(kw) or {"type": "abort", "reason": "reconfigure_successful"}
    )
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )
    assert updates[0]["data"][CONF_SENTINEL_AUTONOMY_LEVEL] == 0


@pytest.mark.parametrize(
    ("given", "level", "ttl"),
    [
        ({CONF_SENTINEL_AUTONOMY_LEVEL: "9"}, 1, 60),
        ({CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: True}, 1, 60),
        ({CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: "x"}, 1, 60),
        (
            {
                CONF_SENTINEL_AUTONOMY_LEVEL: "3",
                CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 90.0,
            },
            3,
            90,
        ),
    ],
)
def test_unreadable_values_are_stored_as_the_defaults(
    given: dict[str, Any], level: int, ttl: int
) -> None:
    """The engine never sees a level or duration it cannot read."""
    data = dict(given)
    _normalize_autonomy(data)
    assert data[CONF_SENTINEL_AUTONOMY_LEVEL] == level
    assert data[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == ttl


@pytest.mark.asyncio
async def test_the_form_after_an_error_accepts_its_own_level(
    hass: HomeAssistant,
) -> None:
    """The redisplayed form shows the level as the select's text, as on first show."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    form = await flow.async_step_settings(
        _settings(
            **{
                CONF_SENTINEL_AUTONOMY_LEVEL: "2",
                CONF_SENTINEL_QUIET_HOURS_START: "22",
            }
        )
    )
    assert cast("Any", form).get("errors") == {"base": "quiet_hours_incomplete"}
    schema = cast("Any", form).get("data_schema")
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested[CONF_SENTINEL_AUTONOMY_LEVEL] == "2"
    assert schema(suggested)[CONF_SENTINEL_AUTONOMY_LEVEL] == "2"
