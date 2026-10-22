# ruff: noqa: S101
"""
Only Home Assistant's intents are reduced to an intent payload (#836).

Every tool call was rebuilt with `minimal_payload_for_domain` before it ran.
A script tool given `domain: lock` lost its other arguments: a script that
unlocks `{{ door | default('lock.garage') }}`, called for the front door,
unlocked the garage. A script, intent script or service tool now runs with
the arguments it was given, on the normal path and on PIN confirmation, and
the PIN line names them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import _prepare_tool_args
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    _request_key,
    minimal_payload_for_domain,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.agent.tools import (
    confirm_sensitive_action,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

confirm_tool = cast("Any", confirm_sensitive_action).coroutine

SCRIPT_TOOL = "script__unlock_door"
SCRIPT_ARGS = {"domain": "lock", "door": "lock.front_door"}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up two locks and a script that unlocks the door it is given."""
    assert await async_setup_component(hass, "homeassistant", {})
    for entity_id, name in {
        "lock.front_door": "Front Door",
        "lock.garage": "Garage",
    }.items():
        hass.states.async_set(entity_id, "locked", {"friendly_name": name})
        async_expose_entity(hass, "conversation", entity_id, should_expose=True)
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "unlock_door": {
                    "alias": "Unlock Door",
                    "fields": {
                        "domain": {"selector": {"text": {}}},
                        "door": {"selector": {"entity": {"domain": "lock"}}},
                    },
                    "sequence": [
                        {
                            "action": "lock.unlock",
                            "target": {
                                "entity_id": "{{ door | default('lock.garage') }}"
                            },
                        }
                    ],
                }
            }
        },
    )
    await hass.async_block_till_done()
    return hass


def _ctx(hass: HomeAssistant, *, pin: bool) -> Any:
    return cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RECOMMENDED_CRITICAL_ACTIONS,
            pin_enabled=pin,
            pin_hash="hash" if pin else None,
            pin_salt="salt" if pin else None,
            pending_actions={},
        ),
    )


class _RecordingAPI:
    """Stand-in for the HA LLM API that records what would run."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call_tool(self, tool_input: Any) -> dict[str, object]:
        self.calls.append(dict(tool_input.tool_args))
        return {"success": True}


async def _confirm(home: HomeAssistant, ctx: Any) -> dict[str, Any]:
    """Confirm the one held action with the right PIN; return what HA got."""
    ((action_id, action),) = ctx.pending_actions.items()
    action.update(user="u1", thread_id="conv1", reply_turn="turn2")
    hashed, salt = hash_pin("1234")
    api = _RecordingAPI()
    config = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": ctx.pending_actions,
            "user_id": "u1",
            "thread_id": "conv1",
            "pin_reply": PinReply("1234", turn_id="turn2"),
            "ha_llm_api": api,
            "hass": home,
        }
    }
    result = await confirm_tool(action_id, "1234", config=config, store=None)
    assert json.loads(result).get("status") == "completed"
    (sent,) = api.calls
    return sent


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        (SCRIPT_TOOL, SCRIPT_ARGS),
        # A service tool: no lock is looked up and filled in for it.
        ("mcp__call_service", {"domain": "lock", "name": "Front Door"}),
        ("mcp__call_service", {"domain": "alarm_control_panel", "level": 2}),
        ("intent_script__OpenDoor", {"domain": "lock", "door": "front"}),
    ],
)
async def test_a_tool_that_is_not_an_intent_runs_as_given(
    home: HomeAssistant, tool_name: str, args: dict[str, Any]
) -> None:
    """With no PIN asked, the tool gets exactly its arguments."""
    assert minimal_payload_for_domain(dict(args), tool_name) == args
    redirect, out = _prepare_tool_args(
        tool_name, dict(args), _ctx(home, pin=False), {"id": "c1"}
    )
    if args.get("domain") == "alarm_control_panel":
        return  # sent to the alarm tool, as before
    assert redirect is None
    assert out == args


async def test_a_held_script_runs_on_the_door_it_was_given(
    home: HomeAssistant,
) -> None:
    """The PIN line and the confirmed call both keep the script's door."""
    ctx = _ctx(home, pin=True)
    redirect, _ = _prepare_tool_args(SCRIPT_TOOL, dict(SCRIPT_ARGS), ctx, {"id": "c1"})
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    (action,) = ctx.pending_actions.values()
    assert "Front Door" in describe_pending_action(home, action)
    assert await _confirm(home, ctx) == SCRIPT_ARGS


async def test_an_intent_is_still_reduced_to_its_slots(home: HomeAssistant) -> None:
    """HA's intents ignore an entity id, so it is still sent as the name."""
    args = {"domain": "lock", "entity_id": "lock.front_door", "extra": 1}
    for tool_name in ("HassTurnOn", "intent__HassTurnOn"):
        payload = minimal_payload_for_domain(dict(args), tool_name)
        assert payload["name"] == "lock.front_door"
        assert "extra" not in payload
    _, out = _prepare_tool_args(
        "intent__HassTurnOn", dict(args), _ctx(home, pin=False), {"id": "c1"}
    )
    assert out["domain"] == ["lock"]
    assert out["name"] == "lock.front_door"


@pytest.fixture
async def intents(home: HomeAssistant) -> HomeAssistant:
    """Add intent scripts: one unlocking what it matches, one taking a code."""
    home.states.async_set("alarm_control_panel.home", "armed_away")
    assert await async_setup_component(
        home,
        "intent_script",
        {
            "intent_script": {
                "UnlockDoor": {
                    "action": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "{{ targets.entities }}"},
                        }
                    ]
                },
                # Takes over a name shaped like a built-in intent's.
                "HassUnlockFront": {
                    "action": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "lock.front_door"},
                        }
                    ]
                },
            }
        },
    )
    await home.async_block_till_done()
    return home


async def test_an_intent_script_keeps_its_slots_and_is_narrowed(
    intents: HomeAssistant,
) -> None:
    """HA matches an intent script's targets by name: the entity id is sent so."""
    args = {"domain": "lock", "entity_id": "lock.front_door", "speed": "slow"}
    _, out = _prepare_tool_args(
        "intent_script__UnlockDoor", dict(args), _ctx(intents, pin=False), {}
    )
    assert out == {**args, "name": "lock.front_door"}
    ctx = _ctx(intents, pin=True)
    redirect, _ = _prepare_tool_args(
        "intent_script__UnlockDoor", dict(args), ctx, {"id": "c1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert await _confirm(intents, ctx) == {**args, "name": "lock.front_door"}


async def test_an_intent_script_with_a_built_in_shaped_name_keeps_its_slots(
    intents: HomeAssistant,
) -> None:
    """An intent script named like a built-in intent is not reduced as one."""
    args = {"domain": "lock", "name": "Front Door", "device_class": ["front"]}
    for tool_name in ("intent_script__HassUnlockFront", "HassUnlockFront"):
        ctx = _ctx(intents, pin=True)
        _prepare_tool_args(tool_name, dict(args), ctx, {"id": "c1"})
        assert await _confirm(intents, ctx) == args


async def test_a_script_code_is_not_read_as_an_alarm_code(
    intents: HomeAssistant,
) -> None:
    """Only HA's own intents are sent to the alarm tool for a numeric code."""
    redirect, out = _prepare_tool_args(
        "script__gate_keypad", {"code": "1234"}, _ctx(intents, pin=False), {}
    )
    assert redirect is None
    assert out == {"code": "1234"}
    redirect, _ = _prepare_tool_args(
        "intent__HassTurnOff", {"code": "1234"}, _ctx(intents, pin=False), {}
    )
    assert redirect is not None


async def test_a_blank_domain_entry_does_not_hide_a_lock(home: HomeAssistant) -> None:
    """A tool's list of domains is screened without its blank entries."""
    args = {"domain": ["", "lock"], "service": "unlock", "name": "Front Door"}
    ctx = _ctx(home, pin=True)
    redirect, out = _prepare_tool_args(
        "mcp__call_service", dict(args), ctx, {"id": "c1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert out == args


def test_a_repeated_request_matches_only_what_runs_the_same() -> None:
    """Two calls a tool runs differently are not the same held request."""

    def key(tool_name: str, args: dict[str, Any]) -> Any:
        return _request_key({"tool_name": tool_name, "tool_args": args})

    as_text = {"domain": "lock", "name": "Bob"}
    as_list = {"domain": ["lock"], "name": "bob"}
    assert key(SCRIPT_TOOL, as_text) != key(SCRIPT_TOOL, as_list)
    assert key("HassTurnOff", as_text) == key("HassTurnOff", as_list)
