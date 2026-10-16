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

import pytest
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr

from custom_components.home_generative_agent.agent import helpers
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
    """
    Set up a small house.

    Garage (on floor "1"): the garage door, a light and an opener button.
    Living Room: blinds. Porch: a lock and a lamp. Shed: a lock and a gate.
    Plus a barn door named
    without door/gate/garage, a gate named "7", a water valve, a front door
    lock, a porch light and a script.
    """
    areas = ar.async_get(hass)
    entities = er.async_get(hass)
    floor = fr.async_get(hass).async_create("1")
    garage = areas.async_get_or_create("Garage")
    areas.async_update(garage.id, floor_id=floor.floor_id)
    # Registered like an integration's entity, so HA matches its computed name.
    for entity_id, name, area_name in (
        ("cover.garage_door", "Garage Door", "Garage"),
        ("light.garage", "Garage Light", "Garage"),
        ("button.garage_opener", "Garage Opener", "Garage"),
        ("cover.living_room_blinds", "Living Room Blinds", "Living Room"),
        ("lock.porch_door", "Porch Door", "Porch"),
        ("light.porch_lamp", "Porch Lamp", "Porch"),
        ("lock.shed_door", "Shed Door", "Shed"),
        ("cover.shed_gate", "Shed Gate", "Shed"),
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
        hass.states.async_set(entity_id, "off", {"friendly_name": name})

    hass.states.async_set(
        "cover.garage_door", "closed", {"friendly_name": "Garage Door"}
    )
    # Named without door/gate/garage: only its entity id matches a rule.
    hass.states.async_set("cover.barn_door", "closed", {"friendly_name": "Big Barn"})
    hass.states.async_set("cover.gate_7", "closed", {"friendly_name": "7"})
    hass.states.async_set("valve.main_water", "closed", {"friendly_name": "Main Water"})
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    hass.states.async_set("script.open_garage", "off", {"friendly_name": "Open Garage"})

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
        # "all" is HA's word for no name: every cover, the garage door too.
        ("intent__HassTurnOn", {"name": "all", "domain": ["cover"]}),
        # Everything in the area, of every kind, the garage door among it.
        ("intent__HassTurnOn", {"area": "Garage"}),
        # HA reads a number as text: floor 1, and the gate named "7".
        ("intent__HassTurnOn", {"floor": 1}),
        ("intent__HassTurnOn", {"name": 7}),
        # A lock and a lamp: turning the area off unlocks the lock.
        ("intent__HassTurnOff", {"area": "Porch"}),
        # A lock domain is held even when nothing matches it yet.
        ("intent__HassTurnOff", {"domain": ["lock"], "name": "No Such Lock"}),
        # A slot HA would refuse or read some other way: held, not guessed.
        ("intent__HassTurnOn", {"area": ["Garage"]}),
        ("intent__HassTurnOn", {"name": {"text": "Garage Door"}}),
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
        # The floor resolves; closing, a light off and a button HA refuses.
        ("intent__HassTurnOff", {"floor": 1}),
        # Padded, it is a name HA looks up, not its "all".
        ("intent__HassTurnOn", {"name": " all ", "domain": ["cover"]}),
        # No target at all: HA refuses it, so there is nothing to hold.
        ("intent__HassSetPosition", {"name": "all", "position": 50}),
        # Stopping is not opening.
        ("intent__HassStopMoving", {"name": "Garage Door"}),
        # HA acts on the named blinds and ignores the entity id.
        (
            "intent__HassTurnOn",
            {
                "domain": ["cover"],
                "name": "Living Room Blinds",
                "entity_id": "cover.garage_door",
            },
        ),
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


async def test_a_button_press_rule_gates_turning_the_button_on(
    home: HomeAssistant,
) -> None:
    """HA presses a button on HassTurnOn and refuses to turn one off."""
    rules = [{"domain": "button", "service": "press", "entity_match": "garage"}]
    assert _is_critical_action({"name": "Garage Opener"}, rules, "HassTurnOn", home)
    assert not _is_critical_action(
        {"name": "Garage Opener"}, rules, "HassTurnOff", home
    )


async def test_a_rule_for_a_scripts_own_service_gates_turning_it_on(
    home: HomeAssistant,
) -> None:
    """Turning a script on runs it, the same as calling `script.open_garage`."""
    rules = [{"domain": "script", "service": "open_garage"}]
    assert _is_critical_action({"name": "Open Garage"}, rules, "HassTurnOn", home)


async def test_a_stop_rule_gates_the_stop_intent(home: HomeAssistant) -> None:
    """HassStopMoving stops a cover or a valve; a rule on stop_valve applies."""
    rules = [{"domain": "valve", "service": "stop_valve"}]
    assert _is_critical_action({"name": "Main Water"}, rules, "HassStopMoving", home)


def test_without_hass_an_unreadable_slot_is_held() -> None:
    args = {"entity_id": {"x": 1}, "domain": "lock"}
    assert _is_critical_action(args, RULES, "HassTurnOff")


async def test_the_matcher_reads_the_payload_ha_is_sent(home: HomeAssistant) -> None:
    """An entity id with a domain is sent as the name, and resolves to it."""
    calls = intent_service_calls(
        home, "HassTurnOn", {"domain": ["cover"], "entity_id": "cover.garage_door"}
    )
    assert calls == [IntentServiceCall("cover", "open_cover", "cover.garage_door")]


async def test_a_matcher_that_raises_fails_closed(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matcher HA changed under us must gate every call, rules or none."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError

    monkeypatch.setattr(helpers.intent, "async_match_targets", boom)
    assert intent_service_calls(home, "HassTurnOn", {"name": "Porch"}) is None
    assert _is_critical_action({"name": "Porch"}, RULES, "HassTurnOn", home)
    assert _is_critical_action({"name": "Front Door"}, [], "HassTurnOff", home)


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


def test_without_hass_a_declared_lock_is_critical() -> None:
    args = {"domain": ["lock"], "name": "Front Door"}
    assert _is_critical_action(args, RULES, "HassTurnOff")


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
        # The position HA runs: 0 is shown, and "1_0" runs as 10.
        (
            "intent__HassSetPosition",
            {"name": "Garage Door", "position": 0},
            'set "Garage Door" to 0%',
        ),
        (
            "intent__HassSetPosition",
            {"name": "Garage Door", "position": "1_0"},
            'set "Garage Door" to 10%',
        ),
        # HA ignores the entity id and unlocks the lock the name matches.
        (
            "intent__HassTurnOff",
            {"name": "Front Door", "entity_id": "light.porch"},
            'unlock "Front Door"',
        ),
        # Several kinds: every action named, most sensitive first, place kept.
        (
            "intent__HassTurnOn",
            {"area": "Garage"},
            (
                "open every cover in Garage, press every button device in Garage, "
                "turn on every light device in Garage"
            ),
        ),
        (
            "intent__HassTurnOff",
            {"area": "Porch"},
            "unlock every lock in Porch, turn off every light device in Porch",
        ),
        (
            "intent__HassTurnOn",
            {"floor": 1},
            (
                "open every cover on floor 1, press every button device on floor 1, "
                "turn on every light device on floor 1"
            ),
        ),
        # Locking the shed also opens its gate: the line must say both.
        (
            "intent__HassTurnOn",
            {"area": "Shed"},
            "lock every lock in Shed, open every cover in Shed",
        ),
        ("intent__HassTurnOn", {"name": "Garage Opener"}, 'press "Garage Opener"'),
        ("intent__HassStopMoving", {"name": "Garage Door"}, 'stop "Garage Door"'),
        # Every cover and valve in the home: "every device", never "a device".
        (
            "intent__HassTurnOn",
            {"domain": ["cover", "valve"], "name": "all"},
            "open every device",
        ),
        (
            "intent__HassSetPosition",
            {"domain": ["cover", "valve"], "name": "all", "position": 100},
            "set every device to 100%",
        ),
        # Floor 0 is kept: dropping it would unlock every lock in the home.
        (
            "intent__HassTurnOff",
            {"domain": ["lock"], "floor": 0},
            "unlock every lock on floor 0",
        ),
        # A position HA cannot read is shown as written, not a crash.
        (
            "intent__HassSetPosition",
            {"name": "Garage Door", "position": float("inf")},
            'set "Garage Door" to inf%',
        ),
    ],
)
async def test_the_pin_line_names_what_ha_will_run(
    home: HomeAssistant, tool_name: str, args: dict[str, Any], line: str
) -> None:
    """The line the user confirms must not say "turn off" for an unlock."""
    action = {"tool_name": tool_name, "tool_args": args}
    assert describe_pending_action(home, action) == line


def test_a_lock_call_keeps_floor_zero_in_the_payload() -> None:
    """HA reads floor 0 as the floor "0"; without it every lock would act."""
    payload = helpers.minimal_payload_for_domain({"domain": ["lock"], "floor": 0})
    assert payload["floor"] == 0
