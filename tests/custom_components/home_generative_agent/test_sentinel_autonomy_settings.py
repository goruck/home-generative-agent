# ruff: noqa: S101
"""
Sentinel's autonomy level and override duration are settings (#783).

The engine read a configured level and an override duration that no form or
resolver supplied: the level was always 1 and an override lasted an hour. The
auto-execution guardrails and the dry-run line are in place but not offered in
the options yet (#789), so nothing here can act.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_PROMPT_USER,
    CONF_CRITICAL_ACTIONS,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
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
from custom_components.home_generative_agent.sentinel.models import FindingAction
from custom_components.home_generative_agent.sentinel.notifier import canary_line

from .test_audit_exposed_relays_898 import _wrap as wrap_switch
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
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR: 10,
}


def _evaluate(
    actions: list[str],
    allowed: list[str],
    *,
    entities: list[str] | None = None,
    hass: HomeAssistant | None = None,
    **options: Any,
) -> Any:
    svc = SentinelExecutionService(
        {
            **_ACTING,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: allowed,
            **options,
        },
        hass=hass,
    )
    targets = entities or ["lock.front_door"]
    finding = _make_finding(
        confidence=0.9, suggested_actions=actions, triggering_entities=targets
    )
    snapshot = _make_snapshot(entity_id=targets[0], last_changed=_RECENT)
    return svc.evaluate_canary(finding, snapshot, 2, _NOW)


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
        # A button, valve or siren can be an opener, a water line, a sounder.
        (["button.press"], ["button"], ["button.gate"]),
        (["input_button.press"], ["input_button"], ["input_button.gate"]),
        (["valve.close_valve"], ["valve"], ["valve.water_main"]),
        (["siren.turn_off"], ["siren"], ["siren.hall"]),
        # The generic domain never runs: it reaches any domain's service.
        (["homeassistant.turn_off"], ["homeassistant"], ["switch.drier"]),
        (["homeassistant.toggle"], ["homeassistant"], ["cover.ratgdo_main"]),
        (["homeassistant.turn_off"], ["homeassistant"], ["lock.front_door"]),
        # Starts configuration whose steps are not read here.
        (["script.turn_on"], ["script"], ["script.anything"]),
        (["automation.trigger"], ["automation"], ["automation.anything"]),
        # The generic domain on an alarm panel, or on a script.
        (
            ["homeassistant.toggle"],
            ["homeassistant"],
            ["alarm_control_panel.home"],
        ),
        (["homeassistant.turn_on"], ["homeassistant"], ["script.open_garage"]),
        # One critical action in a list holds the whole finding.
        (["light.turn_off", "lock.unlock"], ["light", "lock"], ["lock.front_door"]),
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
    rules: dict[str, Any] = {
        CONF_CRITICAL_ACTIONS: [{"domain": "light", "service": "turn_off"}]
    }
    result = _evaluate(["light.turn_off"], ["light"], entities=["light.porch"], **rules)
    assert result.block_reason == "critical_action"


async def test_a_pin_rule_reaches_entities_by_name_and_group(
    hass: HomeAssistant,
) -> None:
    """
    Sentinel extends the rules as the PIN gate does (#773).

    A user rule for `garage` on switches holds the relay named "Garage
    Opener" and a switch group holding it, though neither id says garage.
    """
    hass.states.async_set("switch.relay_1", "off", {"friendly_name": "Garage Opener"})
    hass.states.async_set("switch.porch", "off", {"friendly_name": "Porch"})
    hass.states.async_set("switch.outside", "off", {"entity_id": ["switch.relay_1"]})
    rules: dict[str, Any] = {
        CONF_CRITICAL_ACTIONS: [
            {"domain": "switch", "service": "turn_on", "entity_match": "garage"}
        ]
    }
    for entity in ("switch.relay_1", "switch.outside"):
        result = _evaluate(
            ["switch.turn_on"], ["switch"], entities=[entity], hass=hass, **rules
        )
        assert result.block_reason == "critical_action", entity
    result = _evaluate(
        ["switch.turn_on"], ["switch"], entities=["switch.porch"], hass=hass, **rules
    )
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


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
    result = svc.evaluate_canary(finding, snapshot, 2, _NOW)
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
    svc = SentinelExecutionService(
        {**options, CONF_SENTINEL_AUTO_EXEC_CANARY_MODE: False}
    )
    off = svc.evaluate_canary(finding, snapshot, 2, _NOW)
    assert off.block_reason == "auto_execution_disabled"


# Explicit actions (#789)
# ---------------------------------------------------------------------------


def _evaluate_actions(
    actions: tuple[FindingAction, ...],
    allowed: list[str],
    *,
    advice: list[str] | None = None,
    hass: HomeAssistant | None = None,
    **options: Any,
) -> Any:
    """Evaluate a finding a sensor triggered, carrying *actions*."""
    svc = SentinelExecutionService(
        {
            **_ACTING,
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: allowed,
            **options,
        },
        hass=hass,
    )
    trigger = "binary_sensor.front_door"
    finding = replace(
        _make_finding(
            confidence=0.9, suggested_actions=advice, triggering_entities=[trigger]
        ),
        actions=actions,
    )
    snapshot = _make_snapshot(entity_id=trigger, last_changed=_RECENT)
    return svc.evaluate_canary(finding, snapshot, 2, _NOW)


def test_a_service_named_in_the_advice_never_runs() -> None:
    """A stored or model-written `lock.lock` string is advice, never a call."""
    result = _evaluate_actions((), ["lock"], advice=["lock.lock", "switch.turn_off"])
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "no_executable_action"


def test_an_action_on_its_own_target_runs() -> None:
    """The target need not be what triggered the finding."""
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),), ["switch.turn_off"]
    )
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


@pytest.mark.parametrize(
    "action",
    [
        FindingAction("switch.turn_off", ()),
        FindingAction("switch", ("switch.drier",)),
        FindingAction("switch.turn_off", ("all",)),
        FindingAction("switch.turn_off", ("switch.drier", "")),
        FindingAction("switch.turn_off", ("Switch.Drier",)),
        FindingAction("switch.turn_off", ("switch.",)),
        FindingAction("switch.turn_off", ("switch.bad id",)),
        FindingAction("switch.turn_off", ("switch.a.b",)),
        FindingAction("switch.turn_off", ("light.hall",)),
        FindingAction("homeassistant.toggle", ("Cover.garage",)),
    ],
)
def test_a_malformed_action_stops_every_action(action: FindingAction) -> None:
    """No target, no service or a non-entity target: nothing runs."""
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.kettle",)), action),
        ["switch"],
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "malformed_action"


@pytest.mark.parametrize(
    "action",
    [
        FindingAction("homeassistant.turn_on", ("cover.garage",)),
        FindingAction("homeassistant.turn_off", ("switch.drier", "lock.front_door")),
        FindingAction("lock.unlock", ("lock.front_door",)),
        FindingAction("cover.close_cover", ("cover.blind",)),
    ],
)
def test_the_floor_reads_the_targets_of_the_action(action: FindingAction) -> None:
    """The finding's trigger is a door sensor; the floor judges what is called."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    result = _evaluate_actions(
        (action,), ["homeassistant", "lock", "cover"], **no_pin_rules
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "critical_action"


def test_a_pin_rule_reads_the_targets_of_the_action() -> None:
    """A sensor triggers; the PIN rule is matched against what is called."""
    rules: dict[str, Any] = {
        CONF_CRITICAL_ACTIONS: [
            {"domain": "switch", "service": "turn_on", "entity_match": "garage"}
        ]
    }
    held = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.porch", "switch.garage_relay")),),
        ["switch"],
        **rules,
    )
    assert held.block_reason == "critical_action"
    free = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.porch",)),), ["switch"], **rules
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_what_the_target_starts_is_read_too(hass: HomeAssistant) -> None:
    """A sensor triggers; the switch called is followed to what it opens (#831)."""
    hass.states.async_set("switch.porch", "off")
    hass.states.async_set("cover.garage_door", "closed", {"device_class": "garage"})
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": {
                        "name": "Opener",
                        "state": "{{ false }}",
                        "turn_on": [
                            {
                                "action": "cover.open_cover",
                                "target": {"entity_id": "cover.garage_door"},
                            }
                        ],
                        "turn_off": [],
                    }
                }
            ]
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("switch.opener") is not None

    held = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.porch", "switch.opener")),),
        ["switch"],
        hass=hass,
    )
    assert held.block_reason == "critical_action"
    free = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.porch",)),), ["switch"], hass=hass
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_a_switch_that_drives_a_guarded_relay_never_runs(
    hass: HomeAssistant,
) -> None:
    """A template switch calling the relay under a blind moves the blind."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    wrap_switch(hass, "switch.relay_9", "cover.study_blind", attributes={})
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": {
                        "name": "Blind helper",
                        "state": "{{ false }}",
                        "turn_on": [
                            {
                                "action": "switch.turn_on",
                                "target": {"entity_id": "switch.relay_9"},
                            }
                        ],
                        "turn_off": [],
                    }
                }
            ]
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("switch.blind_helper") is not None

    held = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.blind_helper",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert held.block_reason == "critical_action"


@pytest.mark.parametrize(
    "wrapper", ["cover.side_gate", "lock.shed", "valve.water", "siren.hall"]
)
async def test_the_relay_under_a_wrapped_switch_never_runs(
    hass: HomeAssistant, wrapper: str
) -> None:
    """A switch wrapped as a cover, lock or valve is that device's relay."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    hass.states.async_set("switch.drier", "on")
    wrap_switch(hass, "switch.relay_4", wrapper, attributes={})

    held = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier", "switch.relay_4")),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert held.block_reason == "critical_action"
    free = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_whatever_reaches_a_guarded_relay_never_runs(
    hass: HomeAssistant,
) -> None:
    """A group holding the relay, or a light wrapper of it, moves the gate too."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    wrap_switch(hass, "switch.relay_4", "cover.side_gate", attributes={})
    wrap_switch(hass, "switch.relay_4", "light.gate_lamp", attributes={})
    hass.states.async_set("light.gate_lamp", "off")
    hass.states.async_set(
        "switch.all_plugs", "on", {"entity_id": ["switch.drier", "switch.relay_4"]}
    )
    hass.states.async_set("switch.drier", "on")

    for action in (
        FindingAction("switch.turn_off", ("switch.all_plugs",)),
        FindingAction("light.turn_on", ("light.gate_lamp",)),
    ):
        held = _evaluate_actions(
            (action,), ["switch", "light"], hass=hass, **no_pin_rules
        )
        assert held.block_reason == "critical_action", action
    free = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_an_unreadable_wrapper_stops_every_action(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay under an entry that cannot be read is not known: nothing runs."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    hass.states.async_set("switch.drier", "on")
    wrap_switch(hass, "switch.relay_4", "cover.side_gate", attributes={})

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        msg = "registry unreadable"
        raise RuntimeError(msg)

    monkeypatch.setattr(er, "async_entries_for_config_entry", _boom)
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "relay_map_unreadable"


async def _template_cover(hass: HomeAssistant, target: Any) -> None:
    """Set up a template blind whose open action turns a switch on."""
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "cover": {
                        "name": "Attic blind",
                        "state": "{{ 'closed' }}",
                        "open_cover": [
                            {
                                "action": "switch.turn_on",
                                "target": {"entity_id": target},
                            }
                        ],
                        "close_cover": [],
                    }
                }
            ]
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("cover.attic_blind") is not None


async def test_what_a_template_cover_drives_never_runs(hass: HomeAssistant) -> None:
    """The switch a template blind's action turns on is the blind's relay."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    hass.states.async_set("switch.relay_7", "off")
    hass.states.async_set("switch.drier", "on")
    await _template_cover(hass, "switch.relay_7")

    held = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.relay_7",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert held.block_reason == "critical_action"
    free = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_a_template_that_cannot_be_read_stops_every_action(
    hass: HomeAssistant,
) -> None:
    """A templated target could be any switch, so nothing runs (#789)."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    hass.states.async_set("switch.drier", "on")
    await _template_cover(hass, "{{ states('input_text.which') }}")

    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "relay_map_unreadable"


async def test_a_member_of_a_wrapped_switch_group_never_runs(
    hass: HomeAssistant,
) -> None:
    """The wrapper wraps a group of switches: each member moves the gate."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    wrap_switch(hass, "switch.gate_pair", "cover.side_gate", attributes={})
    hass.states.async_set(
        "switch.gate_pair", "off", {"entity_id": ["switch.gate_a", "switch.gate_b"]}
    )
    hass.states.async_set("switch.gate_a", "off")
    hass.states.async_set("switch.gate_b", "off")
    hass.states.async_set("switch.drier", "on")

    held = _evaluate_actions(
        (FindingAction("switch.turn_on", ("switch.gate_a",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert held.block_reason == "critical_action"
    free = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert free.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_a_switch_wrapped_as_a_light_is_not_held(hass: HomeAssistant) -> None:
    """Only a wrapper of a guarded domain makes its switch a relay."""
    no_pin_rules: dict[str, Any] = {CONF_CRITICAL_ACTIONS: []}
    wrap_switch(hass, "switch.lamp_plug", "light.lamp", attributes={})
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.lamp_plug",)),),
        ["switch"],
        hass=hass,
        **no_pin_rules,
    )
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


def test_a_check_that_cannot_be_made_stops_the_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure while reading the rules is a refusal, not an error or a pass."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        msg = "registry unreadable"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "custom_components.home_generative_agent.agent.rule_entities."
        "expand_critical_rules",
        _boom,
    )
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.drier",)),), ["switch"]
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "critical_action"


@pytest.mark.parametrize("state", [None, "unavailable"])
async def test_a_target_home_assistant_would_skip_stops_the_action(
    hass: HomeAssistant, state: str | None
) -> None:
    """It would drop the entity and return: a call recorded as done, not made."""
    hass.states.async_set("switch.kettle", "on")
    if state is not None:
        hass.states.async_set("switch.drier", state)
    result = _evaluate_actions(
        (FindingAction("switch.turn_off", ("switch.kettle", "switch.drier")),),
        ["switch"],
        hass=hass,
    )
    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "target_unavailable"


def test_the_dry_run_line_keeps_each_service_with_its_own_devices() -> None:
    """Two services on different devices do not read as each on every device."""
    actions = [
        FindingAction("switch.turn_off", ("switch.drier",)),
        FindingAction("light.turn_on", ("light.porch",)),
    ]
    assert canary_line(actions, None) == (
        "Dry run: Sentinel would have run switch.turn_off on Drier. "
        "Dry run: Sentinel would have run light.turn_on on Porch."
    )


def test_the_dry_run_line_names_the_targets_not_the_trigger() -> None:
    """A dry run says which device would have been switched, once each."""
    actions = [
        FindingAction("switch.turn_off", ("switch.drier", "switch.iron")),
        FindingAction("switch.turn_off", ("switch.drier",)),
    ]
    assert canary_line(actions, None) == (
        "Dry run: Sentinel would have run switch.turn_off on Drier, Iron."
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dry-run line in the notification
# ---------------------------------------------------------------------------


def test_the_dry_run_line_names_the_action_and_the_device() -> None:
    """Device names come from Home Assistant when it knows them."""
    finding = _make_finding(
        suggested_actions=["lock.lock"], triggering_entities=["lock.front_door"]
    )
    line = canary_line(list(finding.actions), None)
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


@pytest.mark.asyncio
async def test_basic_setup_keeps_an_older_installs_level(
    hass: HomeAssistant,
) -> None:
    """A level 0 kept in the entry options stays 0 through a Basic re-run."""
    existing = DummySubentry(
        "sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {CONF_SENTINEL_ENABLED: True}
    )
    entry = DummyEntry(options={CONF_SENTINEL_AUTONOMY_LEVEL: 0})
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [None, "x", 0, True])
async def test_the_form_opens_with_an_unreadable_duration(
    hass: HomeAssistant, stored: object
) -> None:
    """A duration an older install left unreadable shows as the default."""
    existing = DummySubentry(
        "sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {CONF_SENTINEL_ENABLED: True}
    )
    entry = DummyEntry(options={CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: stored})
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    form = await flow.async_step_settings()
    schema = cast("Any", form).get("data_schema")
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 60
    schema(suggested)


def test_data_quality_is_not_claimed_when_it_was_not_checked() -> None:
    """With no action to decide, the audit says data quality was not read."""
    svc = SentinelExecutionService({CONF_SENTINEL_AUTO_EXECUTION_ENABLED: False})
    finding = _make_finding(severity="medium", triggering_entities=["sensor.gone"])
    snapshot = _make_snapshot(entity_id="lock.front_door", last_changed=_RECENT)
    for level in (1, 2):
        result = svc.evaluate_canary(finding, snapshot, level, _NOW)
        assert result.data_quality == "not_evaluated", level


@pytest.mark.asyncio
async def test_basic_setup_keeps_the_override_duration(hass: HomeAssistant) -> None:
    """The duration is the user's choice too: a Basic re-run keeps it."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True, CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 600},
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
    assert updates[0]["data"][CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 600


@pytest.mark.asyncio
async def test_basic_setup_keeps_an_older_installs_duration(
    hass: HomeAssistant,
) -> None:
    existing = DummySubentry(
        "sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {CONF_SENTINEL_ENABLED: True}
    )
    entry = DummyEntry(options={CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 240})
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
    assert updates[0]["data"][CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 240


@pytest.mark.asyncio
async def test_the_form_shows_legacy_values_as_they_run(hass: HomeAssistant) -> None:
    """
    A legacy level 0.0 runs as 0 and a duration "5" as five minutes.

    The form shows them so, so saving it unchanged changes nothing.
    """
    existing = DummySubentry(
        "sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {CONF_SENTINEL_ENABLED: True}
    )
    entry = DummyEntry(
        options={
            CONF_SENTINEL_AUTONOMY_LEVEL: 0.0,
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: "5",
        }
    )
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    form = await flow.async_step_settings()
    schema = cast("Any", form).get("data_schema")
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested[CONF_SENTINEL_AUTONOMY_LEVEL] == "0"
    assert suggested[CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES] == 5
