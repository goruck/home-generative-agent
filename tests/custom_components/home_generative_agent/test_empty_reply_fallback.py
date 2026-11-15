# ruff: noqa: S101
"""
An empty reply after a tool round says "Done." only when the tools acted (#1013).

Qwen3 extended thinking leaves `content=''` once Ollama strips the reasoning.
After a tool round that used to become "Done." whatever the tool returned: a
failed tool read as done, and a critical action held for the PIN read as done
with the fixed PIN line asking for the PIN to do it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent.automation_targets import (
    AUTOMATION_ADDED_PREFIX,
    AUTOMATION_REFUSAL_PREFIX,
)
from custom_components.home_generative_agent.agent.graph import _should_continue
from custom_components.home_generative_agent.agent.helpers import (
    register_pending_action,
    start_pin_turn,
)
from custom_components.home_generative_agent.const import (
    NO_REPLY_APOLOGY,
    NO_REPLY_PARTIAL,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _call(call_id: str, name: str = "HassTurnOff") -> dict[str, Any]:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def _result(
    content: Any, call_id: str = "c1", name: str = "HassTurnOff", **kwargs: Any
) -> ToolMessage:
    text = content if isinstance(content, str) else json.dumps(content)
    return ToolMessage(content=text, tool_call_id=call_id, name=name, **kwargs)


def _state(*results: ToolMessage) -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content="turn it off"),
            AIMessage(content="", tool_calls=[_call(r.tool_call_id) for r in results]),
            *results,
        ],
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }


def _config(hass: HomeAssistant, **configurable: Any) -> dict[str, Any]:
    return {
        "configurable": {
            "chat_model": MagicMock(),
            "user_id": "user-test",
            "hass": hass,
            "options": {},
            "chat_model_options": {},
            "prompt": "You are a helpful assistant.",
            "langchain_tools": {},
            "ha_llm_api": None,
            "pending_actions": {},
            **configurable,
        }
    }


@pytest.fixture
def empty_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make every model call return no text and no tool call."""

    async def _fake_invoke_model(*_args: object, **_kwargs: object) -> AIMessage:
        return AIMessage(content="", tool_calls=[])

    async def _fake_trim(
        messages: list[object], *_args: object, **_kwargs: object
    ) -> list[object]:
        return messages

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    monkeypatch.setattr(agent_graph, "_trim_messages_for_model", _fake_trim)
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])
    return store


async def _reply(
    state: dict[str, Any], config: dict[str, Any], store: MagicMock
) -> Any:
    call_model: Any = agent_graph._call_model
    result = await call_model(state, config, store=store)
    return result["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        pytest.param([_result({"success": True})], id="success_true"),
        pytest.param([_result(["a", "b"])], id="json_list"),
        # A tool's own status object is not one of ours, and must not raise.
        pytest.param([_result({"status": {"state": "on"}})], id="status_object"),
        pytest.param([_result({"status": []})], id="status_list"),
        pytest.param(
            [_result(f"{AUTOMATION_ADDED_PREFIX}01X", name="add_automation")],
            id="automation_added",
        ),
        # Another server's tool of the name is read like any other.
        pytest.param(
            [_result({"success": True}, name="srv__add_automation")],
            id="namesake_add_automation",
        ),
        pytest.param([_result(json.dumps(json.dumps({"ok": 1})))], id="json_string"),
        pytest.param(
            [_result({"success": True}), _result("ok", call_id="c2")], id="two_ok"
        ),
    ],
)
async def test_done_after_a_round_that_acted(
    hass: HomeAssistant, empty_model: MagicMock, results: list[ToolMessage]
) -> None:
    """Every result in the round succeeded: "Done." is still the reply."""
    reply = await _reply(_state(*results), _config(hass), empty_model)

    assert isinstance(reply, AIMessage)
    assert reply.content == "Done."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        pytest.param([_result("Error: no such device", status="error")], id="status"),
        pytest.param([_result({"error": "No history found."})], id="error_field"),
        pytest.param([_result({"success": False})], id="success_false"),
        pytest.param([_result({"warning": "Alarm did not arm."})], id="warning"),
        pytest.param([_result({"status": "error", "reason": "x"})], id="status_error"),
        pytest.param(
            [_result({"response_type": "action_done", "failure": "No device acted"})],
            id="none_acted",
        ),
        # add_automation says it did not act in plain text (#1013 review).
        pytest.param(
            [_result(f"{AUTOMATION_REFUSAL_PREFIX}: x", name="add_automation")],
            id="automation_refused",
        ),
        pytest.param(
            [_result("Invalid automation configuration x", name="add_automation")],
            id="automation_invalid",
        ),
        pytest.param(
            [_result("No PIN is set.", name="add_automation")],
            id="automation_needs_a_pin",
        ),
        pytest.param(
            [_result(f"{AUTOMATION_REFUSAL_PREFIX}: x", name="Add_Automation")],
            id="automation_refused_any_case",
        ),
        # An HA API tool's JSON-string result arrives encoded twice.
        pytest.param(
            [_result(json.dumps(json.dumps({"error": "Lock jammed"})))],
            id="json_string_error",
        ),
        pytest.param(
            [_result(json.dumps(json.dumps({"status": "error"})))],
            id="json_string_status_error",
        ),
        pytest.param(
            # Held for the PIN, with no fixed line to follow (the prompt is gone).
            [_result({"status": "requires_pin", "action_id": "a1"})],
            id="held_without_a_prompt",
        ),
    ],
)
async def test_apology_after_a_round_that_did_not_act(
    hass: HomeAssistant, empty_model: MagicMock, results: list[ToolMessage]
) -> None:
    """A failed, half-done or held result never reads as "Done."."""
    reply = await _reply(_state(*results), _config(hass), empty_model)

    assert isinstance(reply, AIMessage)
    assert reply.content == NO_REPLY_APOLOGY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        pytest.param(
            [_result({"success": True}), _result({"error": "x"}, call_id="c2")],
            id="one_of_two_failed",
        ),
        pytest.param(
            [
                _result(f"{AUTOMATION_ADDED_PREFIX}01X", name="add_automation"),
                _result(f"{AUTOMATION_REFUSAL_PREFIX}: x", "c2", "add_automation"),
            ],
            id="one_automation_added_one_refused",
        ),
        # One call that acted on some of its devices only (#1013 review).
        pytest.param(
            [_result({"response_type": "action_done", "partial_failure": "1 failed"})],
            id="half_done",
        ),
    ],
)
async def test_a_round_that_partly_acted_does_not_ask_to_try_again(
    hass: HomeAssistant, empty_model: MagicMock, results: list[ToolMessage]
) -> None:
    """Asking to try again would repeat what ran: an automation added twice."""
    reply = await _reply(_state(*results), _config(hass), empty_model)

    assert isinstance(reply, AIMessage)
    assert reply.content == NO_REPLY_PARTIAL
    assert "try again" not in str(reply.content).lower()


@pytest.mark.asyncio
async def test_only_the_last_round_counts(
    hass: HomeAssistant, empty_model: MagicMock
) -> None:
    """An earlier round's failure was the model's to correct; this round acted."""
    state = _state(_result({"error": "wrong name"}))
    state["messages"] += [
        AIMessage(content="", tool_calls=[_call("c2")]),
        _result("action_done", call_id="c2"),
    ]

    reply = await _reply(state, _config(hass), empty_model)

    assert reply.content == "Done."


def _prompt_turn() -> dict[str, Any]:
    pending: dict[str, Any] = {}
    pin_reply = start_pin_turn(pending, "conv1", "unlock the front door")
    register_pending_action(
        pending,
        {
            "tool_name": "HassTurnOff",
            "tool_args": {"name": "Front Door Lock", "domain": ["lock"]},
            "user": "robot",
            "thread_id": "conv1",
            "turn_id": pin_reply.turn_id,
        },
        pin_reply=pin_reply,
    )
    return {"pending_actions": pending, "pin_reply": pin_reply}


@pytest.mark.asyncio
async def test_a_held_action_adds_nothing_before_the_pin_line(
    hass: HomeAssistant, empty_model: MagicMock
) -> None:
    """
    The fixed PIN line is the reply, and asks for the PIN.

    "Done." in front of "say your PIN to unlock the front door" said both. The
    turn ends on the tool round, as the round-limit guard's does, and the
    conversation adds the line alone (`_async_finish_reply`).
    """
    state = _state(_result({"status": "requires_pin", "action_id": "a1"}))

    reply = await _reply(state, _config(hass, **_prompt_turn()), empty_model)

    assert reply == []
    assert _should_continue(cast("Any", state)) == "summarize_and_remove_messages"


@pytest.mark.asyncio
async def test_a_confirmed_action_adds_nothing_after_its_outcome_line(
    hass: HomeAssistant, empty_model: MagicMock
) -> None:
    """A PIN ran the action: "PIN accepted: ..." is the reply, even when it failed."""
    pending: dict[str, Any] = {}
    pin_reply = start_pin_turn(pending, "conv1", "1234")
    pin_reply.confirmed = 'unlock "Front Door Lock"'
    state = _state(_result({"success": False}))

    reply = await _reply(
        state,
        _config(hass, pending_actions=pending, pin_reply=pin_reply),
        empty_model,
    )

    assert reply == []
