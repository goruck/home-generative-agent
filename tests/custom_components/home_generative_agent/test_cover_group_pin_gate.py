# ruff: noqa: S101
"""
Garage covers are held by device class and through groups (#773).

The recommended cover rules match `door`, `gate` or `garage` in the entity id
only. A cover whose device class is garage but whose id is plain, and a group
(cover group or old-style group) holding a garage door, opened with no PIN.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.automation_pin import (
    find_critical_automation_calls,
)
from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
from custom_components.home_generative_agent.agent.indirect_pin import (
    runs_critical_action,
)
from custom_components.home_generative_agent.agent.rule_entities import (
    expand_critical_rules,
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
    Set up covers a name match misses.

    A garage opener with a plain id, a gate whose device class is only in the
    registry, blinds, a cover group and old-style groups (one nested) holding
    the opener, a group of blinds only, and scenes opening the groups.
    """
    hass.states.async_set(
        "cover.msg100_abc", "closed", {"device_class": "garage", "friendly_name": "Q"}
    )
    hass.states.async_set(
        "cover.living_room_blinds", "closed", {"device_class": "shade"}
    )
    hass.states.async_set("cover.study_blinds", "closed")
    entry = er.async_get(hass).async_get_or_create(
        "cover",
        "test",
        "yard",
        suggested_object_id="yard_1",
        original_device_class="gate",
    )
    hass.states.async_set(entry.entity_id, "closed")
    hass.states.async_set(
        "cover.all_covers",
        "closed",
        {
            "entity_id": ["cover.living_room_blinds", "cover.msg100_abc"],
            "friendly_name": "All Covers",
        },
    )
    hass.states.async_set(
        "cover.all_blinds",
        "closed",
        {"entity_id": ["cover.living_room_blinds", "cover.study_blinds"]},
    )
    hass.states.async_set(
        "group.openers", "closed", {"entity_id": ["cover.msg100_abc"]}
    )
    hass.states.async_set("group.outside", "closed", {"entity_id": ["group.openers"]})
    hass.states.async_set("light.porch", "off")
    assert await async_setup_component(
        hass,
        "scene",
        {
            "scene": [
                {"name": "Open Everything", "entities": {"cover.all_covers": "open"}},
                {"name": "Open Blinds", "entities": {"cover.all_blinds": "open"}},
            ]
        },
    )
    await hass.async_block_till_done()
    return hass


def _open(entity_id: str) -> dict[str, Any]:
    return {"domain": "cover", "service": "open_cover", "entity_id": entity_id}


@pytest.mark.parametrize(
    ("entity_id", "held_before"),
    [
        # Device class garage in the state, id with no rule word.
        ("cover.msg100_abc", False),
        # Device class gate only in the registry.
        ("cover.yard_1", False),
        # A cover group.
        ("cover.all_covers", False),
        # An old-style group, and a group of groups: already held, since the
        # screen cannot tell what a `group.*` target reaches.
        ("group.openers", True),
        ("group.outside", True),
    ],
)
async def test_opening_a_garage_cover_however_named_is_held(
    home: HomeAssistant, entity_id: str, *, held_before: bool
) -> None:
    rules = expand_critical_rules(home, RULES)
    assert _is_critical_action(_open(entity_id), rules, "mcp__call_service", home)
    assert (
        _is_critical_action(_open(entity_id), RULES, "mcp__call_service", home)
        is held_before
    )


@pytest.mark.parametrize(
    "entity_id", ["cover.living_room_blinds", "cover.study_blinds", "cover.all_blinds"]
)
async def test_blinds_and_a_group_of_blinds_are_not_held(
    home: HomeAssistant, entity_id: str
) -> None:
    rules = expand_critical_rules(home, RULES)
    assert not _is_critical_action(_open(entity_id), rules, "mcp__call_service", home)


async def test_turning_on_a_garage_cover_group_by_name_is_held(
    home: HomeAssistant,
) -> None:
    """The agent's usual intent reaches the group through the same rules."""
    rules = expand_critical_rules(home, RULES)
    args = {"name": "All Covers"}
    assert _is_critical_action(args, rules, "intent__HassTurnOn", home)
    assert not _is_critical_action(args, RULES, "intent__HassTurnOn", home)


async def test_closing_a_garage_cover_group_is_not_held(home: HomeAssistant) -> None:
    rules = expand_critical_rules(home, RULES)
    args = {**_open("cover.all_covers"), "service": "close_cover"}
    assert not _is_critical_action(args, rules, "mcp__call_service", home)


async def test_a_scene_opening_a_group_with_a_garage_door_is_held(
    home: HomeAssistant,
) -> None:
    """The scene screen checks the group by the extended rules too."""
    rules = expand_critical_rules(home, RULES)
    assert runs_critical_action(home, "scene.open_everything", rules)
    assert not runs_critical_action(home, "scene.open_blinds", rules)


async def test_an_automation_opening_the_group_is_held(home: HomeAssistant) -> None:
    rules = expand_critical_rules(home, RULES)
    config = {
        "actions": [
            {"action": "cover.open_cover", "target": {"entity_id": "cover.all_covers"}}
        ]
    }
    assert find_critical_automation_calls(config, rules)
    assert not find_critical_automation_calls(config, RULES)


async def test_the_gate_holds_a_garage_cover_for_a_pin(home: HomeAssistant) -> None:
    """End to end: the tool-call gate extends the rules itself."""
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
        "mcp__call_service", _open("cover.all_covers"), ctx, {"id": "call1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert len(pending) == 1


async def test_a_user_rule_reaches_groups_holding_its_entity(
    home: HomeAssistant,
) -> None:
    """The workaround rule from the issue also covers the groups."""
    rules = expand_critical_rules(
        home, [{"domain": "cover", "service": "open_cover", "entity_match": "msg100"}]
    )
    assert _is_critical_action(
        _open("cover.all_covers"), rules, "mcp__call_service", home
    )


async def test_rules_without_entity_match_are_unchanged(home: HomeAssistant) -> None:
    rules = [{"domain": "lock", "service": "unlock"}]
    assert expand_critical_rules(home, rules) == rules


def test_without_hass_the_rules_are_copied_unchanged() -> None:
    expanded = expand_critical_rules(None, RULES)
    assert expanded == RULES
    assert expanded is not RULES
    assert all(a is not b for a, b in zip(expanded, RULES, strict=True))


async def test_an_entity_already_named_by_the_rule_adds_nothing(
    hass: HomeAssistant,
) -> None:
    hass.states.async_set("cover.garage_door", "closed", {"device_class": "garage"})
    assert expand_critical_rules(hass, RULES) == RULES
