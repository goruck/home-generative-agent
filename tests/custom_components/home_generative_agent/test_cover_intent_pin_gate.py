# ruff: noqa: S101
"""
Cover and valve intents must reach the Critical Action PIN gate (#770).

Home Assistant's on/off intents carry no service: HA picks it per matched
entity (on opens a cover or a valve, off closes it), and HassSetPosition
sets its position. The gate needed a domain and a service, so "open the
garage door" through `HassTurnOn` ran with no PIN even with the recommended
garage/door/gate rules.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.helpers import (
    IntentServiceCall,
    intent_service_calls,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a garage door, a barn door, blinds, a water valve and a lock."""
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    # Registered like an integration's entity, so HA matches its computed name.
    for entity_id, name, area_name in (
        ("cover.garage_door", "Garage Door", "Garage"),
        ("cover.living_room_blinds", "Living Room Blinds", "Living Room"),
    ):
        area = areas.async_get_or_create(area_name)
        domain, object_id = entity_id.split(".")
        entry = entities.async_get_or_create(
            domain,
            "test",
            object_id,
            suggested_object_id=object_id,
            original_name=name,
        )
        assert entry.entity_id == entity_id
        entities.async_update_entity(entity_id, area_id=area.id)

    hass.states.async_set(
        "cover.garage_door", "closed", {"friendly_name": "Garage Door"}
    )
    # Named without door/gate/garage: only its entity id matches a rule.
    hass.states.async_set("cover.barn_door", "closed", {"friendly_name": "Big Barn"})
    hass.states.async_set(
        "cover.living_room_blinds", "closed", {"friendly_name": "Living Room Blinds"}
    )
    hass.states.async_set("valve.main_water", "closed", {"friendly_name": "Main Water"})
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})

    return hass


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        # The two shapes from the issue.
        ("HassTurnOn", {"domain": ["cover"], "name": "Garage Door"}),
        ("intent__HassTurnOn", {"entity_id": "cover.garage_door"}),
        # Name only: no domain, so only HA's matcher knows it is a cover.
        ("intent__HassTurnOn", {"name": "Garage Door"}),
        # The name matches no rule; the entity id it resolves to does.
        ("intent__HassTurnOn", {"name": "Big Barn"}),
        # Every cover in the area, which includes the garage door.
        ("intent__HassTurnOn", {"domain": ["cover"], "area": "Garage"}),
        # Opening a door part way is still opening it.
        ("intent__HassSetPosition", {"name": "Garage Door", "position": 100}),
    ],
)
async def test_opening_a_door_cover_is_critical(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    assert _is_critical_action(args, RULES, tool_name, home)


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        # Closing is not an opening service.
        ("intent__HassTurnOff", {"domain": ["cover"], "name": "Garage Door"}),
        # A cover whose name and entity id match no entity_match stays ungated.
        ("intent__HassTurnOn", {"domain": ["cover"], "name": "Living Room Blinds"}),
        ("intent__HassTurnOn", {"domain": ["cover"], "area": "Living Room"}),
        ("intent__HassSetPosition", {"name": "Living Room Blinds", "position": 50}),
        # Not a cover at all.
        ("intent__HassTurnOn", {"name": "Porch"}),
        # Nothing matches, so HA acts on nothing.
        ("intent__HassTurnOn", {"name": "No Such Thing"}),
    ],
)
async def test_other_calls_stay_ungated(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    assert not _is_critical_action(args, RULES, tool_name, home)


async def test_a_lock_named_without_lock_or_a_domain_is_critical(
    home: HomeAssistant,
) -> None:
    """Same gap for locks: "Front Door" with no domain unlocked with no PIN."""
    assert _is_critical_action({"name": "Front Door"}, RULES, "HassTurnOff", home)


async def test_a_valve_rule_gates_the_on_off_intent(home: HomeAssistant) -> None:
    rules = [{"domain": "valve", "service": "open_valve"}]
    assert _is_critical_action({"name": "Main Water"}, rules, "HassTurnOn", home)
    assert not _is_critical_action({"name": "Main Water"}, rules, "HassTurnOff", home)


async def test_a_turn_on_rule_for_another_domain_gates_the_intent(
    home: HomeAssistant,
) -> None:
    """A user rule on `turn_on` now sees the service HA calls for the intent."""
    rules = [{"domain": "light", "service": "turn_on", "entity_match": "porch"}]
    assert _is_critical_action({"name": "Porch"}, rules, "HassTurnOn", home)


async def test_the_matcher_reads_the_payload_ha_is_sent(home: HomeAssistant) -> None:
    """An entity id with a domain is sent as the name, and resolves to it."""
    calls = intent_service_calls(
        home, "HassTurnOn", {"domain": ["cover"], "entity_id": "cover.garage_door"}
    )
    assert calls == [IntentServiceCall("cover", "open_cover", "cover.garage_door")]


def test_a_matcher_that_raises_fails_closed() -> None:
    """A matcher HA changed under us must gate every call, not none."""
    hass = MagicMock()
    hass.states.async_all.side_effect = RuntimeError("boom")
    assert _is_critical_action({"name": "Anything"}, RULES, "HassTurnOn", hass)
    assert not _is_critical_action({"name": "Anything"}, [], "HassTurnOn", hass)


@pytest.mark.parametrize(
    "args",
    [
        {"domain": ["cover"], "name": "Garage Door"},
        {"entity_id": "cover.garage_door"},
    ],
)
def test_without_hass_the_declared_domain_is_checked(args: dict[str, Any]) -> None:
    """With no `hass`, an unnamed cover target matches every entity_match rule."""
    assert _is_critical_action(args, RULES, "HassTurnOn")
    assert not _is_critical_action(args, RULES, "HassTurnOff")


async def test_the_tool_path_holds_the_call_for_a_pin(home: HomeAssistant) -> None:
    """End to end through argument preparation: the call is held, not run."""
    pending: dict[str, dict[str, Any]] = {}
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=home,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RULES,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions=pending,
        ),
    )
    redirect, _ = _prepare_tool_args(
        "intent__HassTurnOn",
        {"domain": ["cover"], "name": "Garage Door"},
        ctx,
        {"id": "call1"},
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert len(pending) == 1


@pytest.mark.parametrize(
    ("tool_name", "args", "line"),
    [
        # HA unlocks a lock on HassTurnOff, whatever the name says.
        ("intent__HassTurnOff", {"name": "Front Door"}, 'unlock "Front Door"'),
        ("intent__HassTurnOn", {"name": "Garage Door"}, 'open "Garage Door"'),
        ("intent__HassTurnOn", {"name": "Big Barn"}, 'open "Big Barn"'),
        (
            "intent__HassSetPosition",
            {"name": "Garage Door", "position": 100},
            'set "Garage Door" to 100%',
        ),
    ],
)
async def test_the_pin_line_names_what_ha_will_run(
    home: HomeAssistant, tool_name: str, args: dict[str, Any], line: str
) -> None:
    """The line the user confirms must not say "turn off" for an unlock."""
    action = {"tool_name": tool_name, "tool_args": args}
    assert describe_pending_action(home, action) == line
