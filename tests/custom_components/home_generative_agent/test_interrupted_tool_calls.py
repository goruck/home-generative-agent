# ruff: noqa: S101
"""
A cancelled turn's tool calls get a result in what the model is sent (#947).

A turn cancelled after the model asked for tools and before their results
were saved left the call without a result in the checkpoint. OpenAI and
Anthropic reject that history, so every later turn of the conversation failed
until trimming dropped the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent.graph import (
    _with_interrupted_results,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _call(call_id: str, name: str = "HassTurnOn") -> dict[str, Any]:
    return {"name": name, "args": {"name": "Porch"}, "id": call_id}


def _result(call_id: str, content: str = "Done") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name="HassTurnOn")


def _cancelled_history() -> list[Any]:
    """Return a history whose earlier turn ended on a tool call, then a new one."""
    return [
        HumanMessage(content="Turn on the porch and garage lights", id="h1"),
        AIMessage(content="", tool_calls=[_call("call-a"), _call("call-b")], id="a1"),
        HumanMessage(content="Is the porch light on?", id="h2"),
    ]


def _results_follow_their_calls(messages: list[Any]) -> None:
    """Every tool call is followed by its result before the next other message."""
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        following: set[str] = set()
        for later in messages[index + 1 :]:
            if not isinstance(later, ToolMessage):
                break
            following.add(later.tool_call_id)
        expected = {call["id"] for call in message.tool_calls}
        assert expected <= following, f"unanswered calls after {message.id}"


def test_each_unanswered_call_gets_a_result_right_after_it() -> None:
    repaired = _with_interrupted_results(_cancelled_history())

    _results_follow_their_calls(repaired)
    assert [m.id for m in repaired[:2]] == ["h1", "a1"]
    assert repaired[-1].id == "h2", "the user's new message stays last"
    answers = [m for m in repaired if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in answers] == ["call-a", "call-b"]
    for answer in answers:
        assert answer.status == "success", "may have run: not marked failed"
        assert "may or may not have run" in str(answer.content)
        assert "Do not call it again" in str(answer.content)


def test_only_the_missing_results_are_added() -> None:
    history = _cancelled_history()
    history.insert(2, _result("call-a"))

    repaired = _with_interrupted_results(history)

    _results_follow_their_calls(repaired)
    answers = [m for m in repaired if isinstance(m, ToolMessage)]
    assert (answers[0].tool_call_id, answers[0].content) == ("call-a", "Done")
    assert [m.tool_call_id for m in answers] == ["call-a", "call-b"]


def test_a_reused_id_answered_in_an_earlier_turn_still_gets_a_result() -> None:
    """Some servers number calls from call_0 in every response."""
    history = [
        HumanMessage(content="Turn on the porch light", id="h1"),
        AIMessage(content="", tool_calls=[_call("call_0")], id="a1"),
        _result("call_0"),
        AIMessage(content="The porch light is on.", id="a2"),
        HumanMessage(content="And the garage", id="h2"),
        AIMessage(content="", tool_calls=[_call("call_0")], id="a3"),
        HumanMessage(content="Is it on?", id="h3"),
    ]

    repaired = _with_interrupted_results(history)

    _results_follow_their_calls(repaired)
    assert len(repaired) == len(history) + 1


def test_a_complete_history_is_left_alone() -> None:
    history = [
        HumanMessage(content="Turn on the porch light", id="h1"),
        AIMessage(content="", tool_calls=[_call("call-a")], id="a1"),
        _result("call-a"),
        AIMessage(content="The porch light is on.", id="a2"),
        HumanMessage(content="Thanks", id="h2"),
    ]

    assert _with_interrupted_results(history) == history


@pytest.mark.asyncio
async def test_the_model_is_sent_the_repaired_history_and_the_state_is_not_changed(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []

    async def _fake_invoke_model(
        _model: Any, messages: Any, *_a: Any, **_k: Any
    ) -> AIMessage:
        sent.extend(messages)
        return AIMessage(content="It is on.")

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[])
    history = _cancelled_history()
    state: Any = {
        "messages": list(history),
        "selected_tools": [],
        "summary": "",
        "tool_routing_map": {},
        "messages_to_remove": [],
        "chat_model_usage_metadata": {},
    }
    config: Any = {
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
        }
    }

    result = await agent_graph._call_model(state, config, store=store)

    # Through the real trimming: the call is kept, and its results follow it.
    _results_follow_their_calls(sent)
    assert sum(isinstance(m, ToolMessage) for m in sent) == 2
    assert not any(isinstance(m, ToolMessage) for m in result["messages"]), (
        "the results go only to the model, not into the saved history"
    )
