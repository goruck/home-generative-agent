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
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

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
from custom_components.home_generative_agent.agent.tools import (
    _maybe_require_pin_for_automation,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.core.utils import hash_pin
from custom_components.home_generative_agent.snapshot.network import (
    AutomationInput,
    HaNativeInputs,
    NetworkBuildContext,
    _collect_critical_actions,
    ha_native_adapter,
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


def _reaches(hass: HomeAssistant, rules: Any = RULES) -> set[str]:
    """Return the entity ids the extended rules name beyond the given ones."""
    return {r["entity_match"] for r in expand_critical_rules(hass, rules)[len(rules) :]}


async def test_a_garage_cover_not_loaded_yet_is_held(hass: HomeAssistant) -> None:
    """At startup a registered cover has no state; an automation outlives that."""
    er.async_get(hass).async_get_or_create(
        "cover",
        "test",
        "opener",
        suggested_object_id="q1",
        original_device_class="garage",
    )
    assert "cover.q1" in _reaches(hass)


async def test_a_user_set_device_class_and_mixed_case_count(
    hass: HomeAssistant,
) -> None:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create("cover", "test", "a", suggested_object_id="a1")
    entities.async_update_entity(entry.entity_id, device_class="gate")
    hass.states.async_set(entry.entity_id, "closed")
    hass.states.async_set("cover.b1", "closed", {"device_class": "Garage"})
    assert {"cover.a1", "cover.b1"} <= _reaches(hass)


async def test_device_class_must_equal_the_word(hass: HomeAssistant) -> None:
    """A custom rule is not stretched over every device class containing it."""
    hass.states.async_set("switch.plug", "off", {"device_class": "outlet"})
    rules = [{"domain": "switch", "service": "turn_on", "entity_match": "out"}]
    assert _reaches(hass, rules) == set()


@pytest.mark.parametrize(
    ("name", "held"),
    [
        ("Garage Door", True),
        ("Back gate", True),
        ("Indoor Blinds", False),
        ("Garagekeeper", False),
    ],
)
async def test_a_cover_named_with_the_word_is_held(
    hass: HomeAssistant, name: str, *, held: bool
) -> None:
    """A relay cover with no device class is found by its name, as HA finds it."""
    hass.states.async_set("cover.shelly1_abc", "closed", {"friendly_name": name})
    assert ("cover.shelly1_abc" in _reaches(hass)) is held


async def test_an_alias_counts_as_a_name(hass: HomeAssistant) -> None:
    entities = er.async_get(hass)
    entry = entities.async_get_or_create("cover", "test", "r", suggested_object_id="r1")
    entities.async_update_entity(entry.entity_id, aliases=["garage"])
    hass.states.async_set(entry.entity_id, "closed")
    assert "cover.r1" in _reaches(hass)


async def test_a_group_listing_group_entities_is_held(hass: HomeAssistant) -> None:
    """MQTT and ZHA groups list their members as `group_entities`."""
    hass.states.async_set("cover.garage_door", "closed")
    hass.states.async_set(
        "cover.mqtt_group", "unavailable", {"group_entities": ["cover.garage_door"]}
    )
    assert "cover.mqtt_group" in _reaches(hass)


async def test_an_unavailable_group_helper_is_read_from_its_config(
    hass: HomeAssistant,
) -> None:
    """An unavailable group writes no members; its config entry still lists them."""
    entities = er.async_get(hass)
    for options, object_id in (
        ({"entities": ["cover.garage_door"]}, "garage_group"),
        ({"entities": ["cover.study_blinds"]}, "blind_group"),
    ):
        entry = MockConfigEntry(domain="group", options=options)
        entry.add_to_hass(hass)
        reg = entities.async_get_or_create(
            "cover",
            "group",
            object_id,
            config_entry=entry,
            suggested_object_id=object_id,
        )
        hass.states.async_set(reg.entity_id, "unavailable")
    hass.states.async_set("cover.garage_door", "closed")
    hass.states.async_set("cover.study_blinds", "closed")
    reached = _reaches(hass)
    assert "cover.garage_group" in reached
    assert "cover.blind_group" not in reached


async def test_a_group_helper_with_unreadable_members_is_held(
    hass: HomeAssistant,
) -> None:
    """With no member list anywhere, the group could hold the garage door."""
    reg = er.async_get(hass).async_get_or_create(
        "cover", "group", "x", suggested_object_id="mystery"
    )
    hass.states.async_set(reg.entity_id, "unavailable")
    assert "cover.mystery" in _reaches(hass)


async def test_groups_in_a_loop_and_a_long_chain_are_followed(
    hass: HomeAssistant,
) -> None:
    hass.states.async_set("cover.msg100_abc", "closed", {"device_class": "garage"})
    hass.states.async_set(
        "group.a", "on", {"entity_id": ["group.b", "cover.msg100_abc"]}
    )
    hass.states.async_set("group.b", "on", {"entity_id": ["group.a"]})
    # Outermost first, the order a level-by-level scan handles worst.
    for i in range(300):
        hass.states.async_set(
            f"group.chain{i}", "on", {"entity_id": [f"group.chain{i + 1}"]}
        )
    hass.states.async_set("group.chain300", "on", {"entity_id": ["cover.msg100_abc"]})
    reached = _reaches(hass)
    assert {"group.a", "group.b", "group.chain0", "group.chain300"} <= reached


async def test_scenes_and_scripts_are_not_followed_as_groups(
    home: HomeAssistant,
) -> None:
    """A scene lists its entities too; it is screened by what it sets instead."""
    assert not {e for e in _reaches(home) if e.startswith(("scene.", "script."))}


async def test_a_rule_with_no_domain_reaches_every_domain(hass: HomeAssistant) -> None:
    hass.states.async_set("switch.relay", "off", {"friendly_name": "Garage"})
    rules = [{"service": "turn_on", "entity_match": "garage"}]
    assert "switch.relay" in _reaches(hass, rules)


async def test_add_automation_holds_an_automation_opening_the_group(
    home: HomeAssistant,
) -> None:
    """The add_automation gate extends the rules itself."""
    hashed, salt = hash_pin("1234")
    pending: dict[str, Any] = {}
    cfg = cast(
        "Any",
        {
            "hass": home,
            "options": {
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": pending,
            "thread_id": "t1",
        },
    )
    validated = {
        "actions": [
            {"action": "cover.open_cover", "target": {"entity_id": "cover.all_covers"}}
        ]
    }
    out = _maybe_require_pin_for_automation(
        cfg=cfg, validated_config=validated, ha_automation_config=dict(validated)
    )
    assert out is not None
    assert "requires_pin" in out
    assert len(pending) == 1


async def test_the_audit_flags_a_webhook_automation_opening_the_group(
    home: HomeAssistant,
) -> None:
    context = NetworkBuildContext(options={})
    automation = {
        "triggers": [{"trigger": "webhook", "webhook_id": "x", "local_only": False}],
        "actions": [
            {"action": "cover.open_cover", "target": {"entity_id": "cover.all_covers"}}
        ],
    }
    inputs = HaNativeInputs(
        now=dt_util.utcnow(),
        automations=[AutomationInput("automation.open_all", automation)],
        critical_actions=_collect_critical_actions(home, context),
    )
    ha = ha_native_adapter(inputs, [], context).ha_security
    assert ha["webhook_automations_critical"] == ["automation.open_all"]
