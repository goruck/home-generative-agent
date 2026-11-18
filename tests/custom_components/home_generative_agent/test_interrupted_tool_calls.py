# ruff: noqa: S101
"""
A cancelled turn's tool calls get a result at the next turn (#947).

A turn cancelled after the model asked for tools and before their results
were saved left the call without a result in the checkpoint. OpenAI and
Anthropic reject that history, so every later turn of the conversation failed
until trimming dropped the call.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from custom_components.home_generative_agent.agent import graph as graph_module
from custom_components.home_generative_agent.agent.graph import (
    _answer_interrupted_calls,
    _retrieve_tools,
)

from .test_tool_inclusions import _config, _search_item, _store


def _call(call_id: str, name: str = "HassTurnOn") -> dict[str, Any]:
    return {"name": name, "args": {"name": "Porch"}, "id": call_id}


def _cancelled_history() -> list[Any]:
    """Earlier turn ended on a tool call; the user has just spoken again."""
    return [
        HumanMessage(content="Turn on the porch and garage lights", id="h1"),
        AIMessage(
            content="",
            tool_calls=[_call("call-a"), _call("call-b")],
            id="a1",
        ),
        HumanMessage(content="Is the porch light on?", id="h2"),
    ]


def _apply(history: list[Any], update: dict[str, Any]) -> list[Any]:
    """Merge a node's update into the history the way the graph does."""
    return list(add_messages(history, update["messages"]))


def _results_follow_their_calls(messages: list[Any]) -> None:
    """Every tool call is followed by its result before the next other message."""
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        expected = {call["id"] for call in message.tool_calls}
        following: set[str] = set()
        for later in messages[index + 1 :]:
            if not isinstance(later, ToolMessage):
                break
            following.add(later.tool_call_id)
        assert expected <= following, f"unanswered calls after {message.id}"


def test_each_unanswered_call_gets_a_result_right_after_it() -> None:
    history = _cancelled_history()

    repaired = _apply(history, _answer_interrupted_calls(history))

    _results_follow_their_calls(repaired)
    assert [m.id for m in repaired[:2]] == ["h1", "a1"]
    assert repaired[-1].id == "h2", "the user's new message stays last"
    answers = [m for m in repaired if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in answers] == ["call-a", "call-b"]
    for answer in answers:
        assert answer.status == "error"
        assert "may or may not have run" in str(answer.content)
        assert "Do not call it again" in str(answer.content)


def test_only_the_missing_results_are_added() -> None:
    history = [
        HumanMessage(content="Turn on the porch and garage lights", id="h1"),
        AIMessage(content="", tool_calls=[_call("call-a"), _call("call-b")], id="a1"),
        ToolMessage(content="Done", tool_call_id="call-a", name="HassTurnOn", id="t1"),
        HumanMessage(content="Is the porch light on?", id="h2"),
    ]

    repaired = _apply(history, _answer_interrupted_calls(history))

    _results_follow_their_calls(repaired)
    answers = [m for m in repaired if isinstance(m, ToolMessage)]
    assert sorted(m.tool_call_id for m in answers) == ["call-a", "call-b"]
    assert next(m for m in answers if m.tool_call_id == "call-a").content == "Done"


def test_a_complete_history_is_left_alone() -> None:
    history = [
        HumanMessage(content="Turn on the porch light", id="h1"),
        AIMessage(content="", tool_calls=[_call("call-a")], id="a1"),
        ToolMessage(content="Done", tool_call_id="call-a", name="HassTurnOn", id="t1"),
        AIMessage(content="The porch light is on.", id="a2"),
        HumanMessage(content="Thanks", id="h2"),
    ]

    assert _answer_interrupted_calls(history) == {}


@pytest.mark.asyncio
async def test_the_turn_start_repairs_the_history_it_was_given() -> None:
    """``_retrieve_tools`` runs first in every turn, so it carries the repair."""
    store = _store([_search_item("HassTurnOn", score=0.6)], {})
    history = _cancelled_history()
    state: Any = {
        "messages": history,
        "summary": "",
        "chat_model_usage_metadata": {},
        "messages_to_remove": [],
        "selected_tools": [],
        "tool_routing_map": {},
    }

    result = await _retrieve_tools(
        state,
        _config(options={"llm_hass_api": ["assist"]}, live={"assist": ["HassTurnOn"]}),
        store=store,
    )

    _results_follow_their_calls(_apply(history, result))


def test_the_turn_start_node_is_retrieve_tools() -> None:
    """The repair lives in the node every turn enters first."""
    assert ("__start__", "retrieve_tools") in graph_module.workflow.edges
