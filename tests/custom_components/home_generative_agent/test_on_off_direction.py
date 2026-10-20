# ruff: noqa: S101
"""
HassTurnOn / HassTurnOff go the way the user asked (#797).

Field test 2026-10-18: "Yes" to "Would you like me to open the study shade?"
ran HassTurnOff (close) and the reply said "the study shade is now open";
"Open All Shades" ran HassTurnOff the same way. Both tools were bound, the
model picked the wrong one, and HA's result did not say what was done.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.helpers import PinReply
from custom_components.home_generative_agent.agent.on_off_direction import (
    requested_direction,
    with_action_performed,
    wrong_direction_refusal,
)
from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Open All Shades", "on"),
        ("open the study shade", "on"),
        ("Close the garage door", "off"),
        ("shut the blinds", "off"),
        ("turn on the kitchen light", "on"),
        ("turn the porch light off", "off"),
        ("switch off the fan", "off"),
        ("lock the front door", "on"),
        ("unlock the front door", "off"),
        ("press the doorbell button", "on"),
        ("Please close it.", "off"),
        ("Can you open the blinds?", "on"),
        ("Would you like me to open the study shade?", "on"),
        ("Should I close the garage?", "off"),
        ("OK, close the shade", "off"),
        ("Open the shades, I can\u2019t see anything", "on"),
        ("Why didn't you open the shades? Open them now", "on"),
        # A direction word that is a state or a name is not a command.
        ("The garage door is open and I'm leaving", None),
        ("the blinds are open, it is too bright", None),
        ("kill the open plan lights", None),
        # Incidental words are not a second command.
        ("Open the lower shades", "on"),
        ("Open all shades before I start work", "on"),
        ("I can open the shade. Would that help?", "on"),
        # Both directions, another action, long, none, another language.
        ("close the shade, it is open", "off"),
        ("turn off the light and open the blinds", None),
        ("Close the shade and start the coffee maker", None),
        ("stop the open garage script", None),
        # Both on and off, or a "turn" not read: not checked.
        ("turn the light on the porch off", None),
        ("turn the router off and on again", None),
        ("turn the tv on and the lights off", None),
        ("Close the shade and turn the downstairs living room ceiling fan on", None),
        ("set the shade to 40 percent", None),
        ("Open " + "x " * 200, None),
        ("otev\u0159i \u017ealuzie", None),
        ("", None),
        ("the light was turned on by someone", None),
    ],
)
def test_the_direction_a_request_asks_for(text: str, expected: str | None) -> None:
    found = requested_direction(text)
    assert (found[0] if found else None) == expected


_COVER = {"name": "Study Shade", "domain": ["cover"]}
_SHADES = {"name": "All Shades", "domain": ["cover"]}


@pytest.mark.parametrize(
    ("tool", "args", "text", "refused"),
    [
        # The field cases.
        ("intent__HassTurnOff", _SHADES, "Open All Shades", True),
        ("intent__HassTurnOn", _SHADES, "Open All Shades", False),
        ("intent__HassTurnOn", _COVER, "close the shade", True),
        ("HassTurnOff", {"name": "Front Door"}, "lock the front door", True),
        ("HassTurnOn", {"name": "Front Door"}, "unlock the front door", True),
        ("HassTurnOff", {"name": "Front Door"}, "unlock the front door", False),
        # Not an on/off tool, or nothing to read.
        ("intent__HassSetPosition", _COVER, "open the shade", False),
        ("intent__HassTurnOff", _COVER, "set the shade to 40 percent", False),
        # A script or scene is run, not turned; a name holding a direction.
        (
            "HassTurnOn",
            {"name": "Close Garage", "domain": ["script"]},
            "run the close garage script",
            False,
        ),
        ("HassTurnOn", {"name": "script.close_garage"}, "close garage", False),
        (
            "HassTurnOff",
            {"name": "Open Plan Lights", "domain": ["light"]},
            "kill the open plan lights",
            False,
        ),
        (
            "HassTurnOff",
            {"area": "Open Plan", "domain": ["light"]},
            "open plan lights off",
            False,
        ),
        ("HassTurnOn", {"name": "Lock Up", "domain": "scene"}, "lock up", False),
        # A name is taken out, not a reason to skip the check.
        (
            "HassTurnOff",
            {"name": "Front Door Lock", "domain": ["lock"]},
            "Lock the front door lock",
            True,
        ),
        ("HassTurnOff", {"name": "lock.front_door"}, "Lock the front door", True),
        # Opening a lock unlocks it; open/close need a known non-lock target.
        (
            "HassTurnOff",
            {"name": "Front Door", "domain": ["lock"]},
            "Open the front door lock",
            False,
        ),
        ("HassTurnOff", {"name": "All Shades"}, "Open All Shades", False),
    ],
)
def test_a_call_against_the_request_is_refused(
    tool: str, args: dict[str, Any], text: str, *, refused: bool
) -> None:
    assert (wrong_direction_refusal(tool, args, text) is not None) is refused


def test_the_refusal_names_the_right_tool() -> None:
    refusal = wrong_direction_refusal("intent__HassTurnOff", _SHADES, "Open All Shades")
    assert refusal is not None
    assert "intent__HassTurnOn" in refusal
    assert "'Open'" in refusal


def _ha_result(*targets: tuple[str, str]) -> dict[str, Any]:
    return {
        "speech": {},
        "response_type": "action_done",
        "data": {
            "targets": [],
            "success": [
                {"name": name, "type": "entity", "id": entity_id}
                for name, entity_id in targets
            ],
            "failed": [],
        },
    }


@pytest.mark.parametrize(
    ("tool", "entity_id", "done"),
    [
        (
            "intent__HassTurnOff",
            "cover.study_shade",
            "close command sent to Study Shade",
        ),
        ("intent__HassTurnOn", "cover.study_shade", "open command sent to Study Shade"),
        ("HassTurnOff", "lock.study_shade", "unlock command sent to Study Shade"),
        ("HassTurnOn", "valve.study_shade", "open command sent to Study Shade"),
        ("HassTurnOn", "light.study_shade", "turn on command sent to Study Shade"),
        ("HassTurnOn", "button.study_shade", "press command sent to Study Shade"),
    ],
)
def test_the_result_says_what_was_done(tool: str, entity_id: str, done: str) -> None:
    result = with_action_performed(tool, _ha_result(("Study Shade", entity_id)))
    assert result["action_performed"] == [done]


def test_other_results_are_left_as_they_are() -> None:
    data = {"success": True, "result": "Kitchen light: on"}
    assert with_action_performed("HassTurnOff", data) is data
    assert with_action_performed("GetLiveContext", _ha_result()) == _ha_result()
    area = {"data": {"success": [{"name": "Study", "type": "area", "id": "study"}]}}
    assert "action_performed" not in with_action_performed("HassTurnOn", area)


class _FakeAPI:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[str] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.calls.append(tool_input.tool_name)
        return self.result


def _ctx(api: _FakeAPI, *messages: Any) -> graph.ToolExecutionContext:
    return graph.ToolExecutionContext(
        hass=MagicMock(),
        store=MagicMock(),
        config=cast("Any", {"configurable": {}}),
        ha_llm_api=api,
        critical_actions=[],
        pin_enabled=False,
        pin_hash=None,
        pin_salt=None,
        pending_actions={},
        state=cast("Any", {"messages": list(messages)}),
    )


_ARGS = {"name": "Study Shade", "domain": ["cover"]}
_OFFER = [
    HumanMessage("Ask me if I want the study shade opened"),
    AIMessage("The study shade is closed. Would you like me to open the study shade?"),
    HumanMessage("Yes"),
]


@pytest.mark.parametrize(
    ("messages", "tool", "runs"),
    [
        # The first field case: a "Yes" to the agent's offer.
        (_OFFER, "intent__HassTurnOff", False),
        (_OFFER, "intent__HassTurnOn", True),
        # The second: a direct request.
        ([HumanMessage("Open All Shades")], "intent__HassTurnOff", False),
        ([HumanMessage("Open All Shades")], "intent__HassTurnOn", True),
        # A later tool round still reads the user's request.
        (
            [
                HumanMessage("Close the study shade"),
                AIMessage("", tool_calls=[]),
            ],
            "intent__HassTurnOn",
            False,
        ),
        # Nothing to read: it runs.
        ([HumanMessage("Zavři žaluzie")], "intent__HassTurnOn", True),
        # "Please close it" gives its own command, not the earlier offer's.
        (
            [
                HumanMessage("open the study shade"),
                AIMessage("Should I also open the other shades?"),
                HumanMessage("Please close it."),
            ],
            "intent__HassTurnOff",
            True,
        ),
        # Every question of the offer is read.
        (
            [
                AIMessage("Would you like me to open the shade? Is that okay?"),
                HumanMessage("Yes"),
            ],
            "intent__HassTurnOff",
            False,
        ),
        # A state, not a command.
        (
            [HumanMessage("The study shade is open and I'm leaving")],
            "intent__HassTurnOff",
            True,
        ),
    ],
)
async def test_a_call_against_the_request_does_not_run(
    messages: list[Any], tool: str, *, runs: bool
) -> None:
    api = _FakeAPI(_ha_result(("Study Shade", "cover.study_shade")))

    message = await graph._run_ha_tool(
        tool, dict(_ARGS), _ctx(api, *messages), {"id": "call1"}
    )

    assert (api.calls == [tool]) is runs
    assert (message.status == "error") is not runs


async def test_the_model_is_told_what_was_done() -> None:
    """Whatever ran, the reply can no longer say the opposite."""
    api = _FakeAPI(_ha_result(("Study Shade", "cover.study_shade")))

    message = await graph._run_ha_tool(
        "intent__HassTurnOff",
        dict(_ARGS),
        _ctx(api, HumanMessage("Zavři žaluzie")),
        {"id": "call1"},
    )

    assert json.loads(cast("str", message.content))["action_performed"] == [
        "close command sent to Study Shade"
    ]


async def test_a_pin_confirmed_call_says_what_was_done() -> None:
    hashed, salt = hash_pin("1234")
    api = _FakeAPI(_ha_result(("Front Door", "lock.front_door")))
    config = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": {
                "aid": {
                    "tool_name": "HassTurnOff",
                    "tool_args": {"domain": ["lock"], "name": "Front Door"},
                    "user": "u1",
                    "thread_id": "conv1",
                    "reply_turn": "turn2",
                }
            },
            "user_id": "u1",
            "thread_id": "conv1",
            "pin_reply": PinReply("1234", turn_id="turn2"),
            "ha_llm_api": api,
            "hass": None,
        }
    }
    confirm = cast("Any", confirm_sensitive_action).coroutine

    reply = json.loads(await confirm("aid", "1234", config=config, store=None))

    assert reply["status"] == "completed"
    assert reply["result"]["action_performed"] == ["unlock command sent to Front Door"]
