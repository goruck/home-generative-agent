# ruff: noqa: S101
"""
A change is followed to the entities computed from it (#820).

Since #814 a change the agent makes is checked against the automations
watching that entity or a group holding it. An automation watching a
template sensor or a helper computed from it (min/max, threshold,
switch_as_x...) could still unlock the door with no PIN. Home Assistant's own
tracking says which entities follow which: what each template re-renders on,
and which entities subscribe to a state change.
"""

from __future__ import annotations

import logging
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import Entity
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.agent import template_actions
from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.const import RECOMMENDED_CRITICAL_ACTIONS

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS
_UNLOCK = [{"action": "lock.unlock", "target": {"entity_id": "lock.front_door"}}]


def _held(hass: HomeAssistant, entity_id: str, service: str = "turn_on") -> bool:
    domain = entity_id.split(".", 1)[0]
    args = {"domain": domain, "service": service, "entity_id": entity_id}
    return _is_critical_action(args, RULES, "mcp__call_service", hass)


async def _setup(hass: HomeAssistant, watched: str, **components: Any) -> None:
    """
    Set up helpers, the given components, and an unlock on `watched`.

    Components are set up in the order given: `sensor` before `template`,
    whose platforms would set `sensor` up without the YAML.
    """
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("sensor.indoor", "20")
    assert await async_setup_component(
        hass,
        "input_boolean",
        {"input_boolean": {"let_in": {"name": "Let In"}, "gate": {"name": "Gate"}}},
    )
    assert await async_setup_component(
        hass, "input_number", {"input_number": {"level": {"min": 0, "max": 10}}}
    )
    for domain, config in components.items():
        assert await async_setup_component(hass, domain, {domain: config})
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "alias": "Unlock on the derived one",
                    "triggers": [{"trigger": "state", "entity_id": watched}],
                    "actions": _UNLOCK,
                }
            ]
        },
    )
    await hass.async_block_till_done()


async def test_a_template_sensor_reading_the_helper(hass: HomeAssistant) -> None:
    """The issue's example: an automation on a template sensor of the helper."""
    await _setup(
        hass,
        "binary_sensor.let_in_derived",
        template=[
            {
                "binary_sensor": [
                    {
                        "name": "Let In Derived",
                        "state": "{{ is_state('input_boolean.let_in', 'on') }}",
                    }
                ]
            }
        ],
    )
    assert hass.states.get("binary_sensor.let_in_derived") is not None
    assert _held(hass, "input_boolean.let_in")
    # A template that does not read the gate is not changed by it.
    assert not _held(hass, "input_boolean.gate")


@pytest.mark.parametrize(("gate", "held"), [("off", False), ("on", True)])
async def test_a_conditional_template_follows_what_it_reads_now(
    hass: HomeAssistant, gate: str, *, held: bool
) -> None:
    """A template reading the helper only while the gate is on."""
    await _setup(
        hass,
        "binary_sensor.gated",
        template=[
            {
                "binary_sensor": [
                    {
                        "name": "Gated",
                        "state": (
                            "{% if is_state('input_boolean.gate', 'on') %}"
                            "{{ is_state('input_boolean.let_in', 'on') }}"
                            "{% else %}false{% endif %}"
                        ),
                    }
                ]
            }
        ],
    )
    hass.states.async_set("input_boolean.gate", gate)
    await hass.async_block_till_done()
    assert _held(hass, "input_boolean.let_in") is held


async def test_a_template_reading_a_whole_domain(hass: HomeAssistant) -> None:
    """`states.input_boolean | ...` re-renders on any input boolean."""
    await _setup(
        hass,
        "sensor.helpers_on",
        template=[
            {
                "sensor": [
                    {
                        "name": "Helpers On",
                        "state": (
                            "{{ states.input_boolean | selectattr('state', 'eq',"
                            " 'on') | list | count }}"
                        ),
                    }
                ]
            }
        ],
    )
    assert _held(hass, "input_boolean.gate")


async def test_a_trigger_based_template_sensor(hass: HomeAssistant) -> None:
    """A trigger-based template entity changes when its trigger fires."""
    await _setup(
        hass,
        "sensor.last_let_in",
        template=[
            {
                "triggers": [{"trigger": "state", "entity_id": "input_boolean.let_in"}],
                "sensor": [{"name": "Last Let In", "state": "{{ now() }}"}],
            }
        ],
    )
    assert hass.states.get("sensor.last_let_in") is not None
    assert _held(hass, "input_boolean.let_in")
    assert not _held(hass, "input_boolean.gate")


async def test_a_threshold_helper(hass: HomeAssistant) -> None:
    """A YAML threshold binary sensor on the input number."""
    await _setup(
        hass,
        "binary_sensor.level_high",
        binary_sensor=[
            {
                "platform": "threshold",
                "name": "Level High",
                "entity_id": "input_number.level",
                "upper": 5,
            }
        ],
    )
    assert hass.states.get("binary_sensor.level_high") is not None
    args = {
        "domain": "input_number",
        "service": "set_value",
        "entity_id": "input_number.level",
        "value": 7,
    }
    assert _is_critical_action(args, RULES, "mcp__call_service", hass)


async def test_a_min_max_helper_of_a_derived_one(hass: HomeAssistant) -> None:
    """Two steps: the helper, a template on it, and min/max on the template."""
    await _setup(
        hass,
        "sensor.max_of_both",
        sensor=[
            {
                "platform": "min_max",
                "name": "Max Of Both",
                "type": "max",
                "entity_ids": ["sensor.let_in_number", "sensor.indoor"],
            }
        ],
        template=[
            {
                "sensor": [
                    {
                        "name": "Let In Number",
                        "state": "{{ 1 if is_state('input_boolean.let_in', 'on') else 0 }}",
                    }
                ]
            }
        ],
    )
    assert hass.states.get("sensor.max_of_both") is not None
    assert _held(hass, "input_boolean.let_in")


async def test_an_unrelated_change_is_not_held(hass: HomeAssistant) -> None:
    """Following derived entities does not hold what nothing derives from."""
    await _setup(
        hass,
        "binary_sensor.let_in_derived",
        template=[
            {
                "binary_sensor": [
                    {
                        "name": "Let In Derived",
                        "state": "{{ is_state('input_boolean.let_in', 'on') }}",
                    }
                ]
            }
        ],
    )
    hass.states.async_set("light.porch", "off")
    assert not _held(hass, "light.porch")


async def test_unreadable_subscriptions_are_warned_about(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """If HA renames its subscription registry, the gate says so."""
    await _setup(
        hass,
        "binary_sensor.level_high",
        binary_sensor=[
            {
                "platform": "threshold",
                "name": "Level High",
                "entity_id": "input_number.level",
                "upper": 5,
            }
        ],
    )
    monkeypatch.setattr(template_actions, "_TRACK_STATE_CHANGE_KEY", "_renamed")
    monkeypatch.setattr(template_actions, "_warned_unreadable", set())
    with caplog.at_level(logging.WARNING):
        _held(hass, "input_number.level", "set_value")
    assert "state-change subscriptions" in caplog.text


async def test_a_switch_as_x_wrapper(hass: HomeAssistant) -> None:
    """A light made from a switch changes with the switch."""
    # switch_as_x copies the switch's Assist exposure, kept by this component.
    assert await async_setup_component(hass, "homeassistant", {})
    registry = er.async_get(hass)
    registry.async_get_or_create("switch", "test", "relay", suggested_object_id="relay")
    hass.states.async_set("switch.relay", "off")
    entry = MockConfigEntry(
        domain="switch_as_x",
        options={
            "entity_id": "switch.relay",
            "invert": False,
            "target_domain": "light",
        },
        title="Relay Light",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    wrapper = next(
        e.entity_id for e in er.async_entries_for_config_entry(registry, entry.entry_id)
    )
    await _setup(hass, wrapper)
    assert hass.states.get(wrapper) is not None
    assert _held(hass, "switch.relay")


async def test_a_bayesian_template_observation(hass: HomeAssistant) -> None:
    """Template observations subscribe through a template tracker, not the entity."""
    await _setup(
        hass,
        "binary_sensor.someone_waiting",
        binary_sensor=[
            {
                "platform": "bayesian",
                "name": "Someone Waiting",
                "prior": 0.2,
                "probability_threshold": 0.5,
                "observations": [
                    {
                        "platform": "template",
                        "value_template": "{{ is_state('input_boolean.let_in', 'on') }}",
                        "prob_given_true": 0.9,
                        "prob_given_false": 0.1,
                    }
                ],
            }
        ],
    )
    assert hass.states.get("binary_sensor.someone_waiting") is not None
    assert _held(hass, "input_boolean.let_in")


@pytest.mark.parametrize("length", [7, 14])
async def test_a_long_chain_of_templates_is_held(
    hass: HomeAssistant, length: int
) -> None:
    """Followed to its end; past the depth bound, held rather than passed."""
    sensors = [
        {
            "name": f"Step {index}",
            "state": (
                "{{ is_state('input_boolean.let_in', 'on') }}"
                if index == 0
                else f"{{{{ states('sensor.step_{index - 1}') }}}}"
            ),
        }
        for index in range(length)
    ]
    await _setup(hass, f"sensor.step_{length - 1}", template=[{"sensor": sensors}])
    assert hass.states.get(f"sensor.step_{length - 1}") is not None
    assert _held(hass, "input_boolean.let_in")


async def test_an_error_reading_what_follows_holds_the_change(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected shape in HA's tracking holds the call, not fails the turn."""
    await _setup(hass, "sensor.indoor")

    def _broken(*_args: Any) -> set[str]:
        msg = "unexpected shape"
        raise TypeError(msg)

    monkeypatch.setattr(template_actions.DerivedEntities, "of", _broken)
    assert _held(hass, "input_boolean.gate")


async def test_a_new_tracker_may_join_a_zone(hass: HomeAssistant) -> None:
    """`device_tracker.see` can create a tracker inside a zone no list names yet."""
    hass.states.async_set("zone.home", "0", {"persons": []})
    await _setup(hass, "zone.home")
    args = {"domain": "device_tracker", "service": "see", "dev_id": "brand_new"}
    assert _is_critical_action(args, RULES, "mcp__call_service", hass)


class _Template:
    def __init__(self, text: str) -> None:
        self.template = text


class _Owner(Entity):
    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id

    def handle(self, *_args: Any) -> None:
        """Stand in for a state listener."""


def test_every_entity_a_callback_reaches_counts() -> None:
    """A partial's argument, a closure over two entities, a coordinator's listener."""
    owner, other = _Owner("sensor.owner"), _Owner("sensor.aaa_other")

    def handler(_event: Any, entity: Any) -> None:
        """Stand in."""
        assert entity is not None

    def make() -> Any:
        another, me = other, owner

        def listener(_event: Any) -> None:
            assert another is not None
            assert me is not None

        return listener

    coordinator = SimpleNamespace(_listeners={1: (owner.handle, None)})
    assert template_actions._job_entities(partial(handler, entity=owner)) == {
        "sensor.owner"
    }
    assert template_actions._job_entities(make()) == {
        "sensor.owner",
        "sensor.aaa_other",
    }
    assert template_actions._job_entities(coordinator.__dict__["_listeners"]) == {
        "sensor.owner"
    }
    # A template tracker's job leading to the coordinator.
    assert template_actions._job_entities(SimpleNamespace(_job=coordinator)) == {
        "sensor.owner"
    }


def test_a_template_not_rendered_yet_is_read_from_its_text() -> None:
    """Before its first render, what its text names; with a pending render, its domains."""
    template = _Template("{{ is_state('input_boolean.let_in', 'on') }}")
    waiting = SimpleNamespace(
        _template_result_info=None, _template_attrs={template: []}
    )
    derived = template_actions.DerivedEntities.__new__(template_actions.DerivedEntities)
    derived.hass = MagicMock()
    derived.by_entity, derived.by_domain, derived.every = {}, {}, set()
    derived._triggered, derived._trigger_index = [], None
    derived._read_template("binary_sensor.waiting", waiting)
    domain_text = _Template("{{ states.input_boolean | list | count }}")
    pending = SimpleNamespace(
        _template_result_info=SimpleNamespace(listeners={"entities": set()}),
        _template_attrs={domain_text: []},
    )
    derived._read_template("sensor.count", pending)
    registry = MagicMock()
    registry.async_get.return_value = None
    with patch.object(template_actions.er, "async_get", return_value=registry):
        assert derived.of("input_boolean.let_in") == {
            "binary_sensor.waiting",
            "sensor.count",
        }
