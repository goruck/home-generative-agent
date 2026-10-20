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
        # Both directions, a negation, none, or another language: not read.
        ("close the shade, it is open", None),
        ("turn off the light and open the blinds", None),
        ("don't open the garage", None),
        ("do not turn off the TV", None),
        ("set the shade to 40 percent", None),
        ("otevři žaluzie", None),
        ("", None),
        # "turned" is a state, not a request.
        ("the light was turned on by someone", None),
    ],
)
def test_the_direction_a_request_asks_for(text: str, expected: str | None) -> None:
    found = requested_direction(text)
    assert (found[0] if found else None) == expected


@pytest.mark.parametrize(
    ("tool", "text", "refused"),
    [
        # The field cases.
        ("intent__HassTurnOff", "Open All Shades", True),
        ("intent__HassTurnOn", "Open All Shades", False),
        ("intent__HassTurnOn", "close the shade", True),
        ("HassTurnOff", "lock the front door", True),
        ("HassTurnOn", "unlock the front door", True),
        ("HassTurnOff", "unlock the front door", False),
        # Not an on/off tool, or nothing to read.
        ("intent__HassSetPosition", "open the shade", False),
        ("intent__HassTurnOff", "set the shade to 40 percent", False),
    ],
)
def test_a_call_against_the_request_is_refused(
    tool: str, text: str, *, refused: bool
) -> None:
    assert (wrong_direction_refusal(tool, text) is not None) is refused


def test_the_refusal_names_the_right_tool() -> None:
    refusal = wrong_direction_refusal("intent__HassTurnOff", "Open All Shades")
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
        ("intent__HassTurnOff", "cover.study_shade", "closed Study Shade"),
        ("intent__HassTurnOn", "cover.study_shade", "opened Study Shade"),
        ("HassTurnOff", "lock.study_shade", "unlocked Study Shade"),
        ("HassTurnOn", "valve.study_shade", "opened Study Shade"),
        ("HassTurnOn", "light.study_shade", "turned on Study Shade"),
        ("HassTurnOn", "button.study_shade", "pressed Study Shade"),
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
        "closed Study Shade"
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
    assert reply["result"]["action_performed"] == ["unlocked Front Door"]
