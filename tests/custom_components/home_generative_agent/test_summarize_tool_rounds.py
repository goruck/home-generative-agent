# ruff: noqa: S101
"""
Summarizing a conversation that used tools sends no tool calls (#1045).

The summarization request kept the assistant messages that asked for tools
and dropped every tool result. OpenAI and Anthropic reject a tool call with
no result after it, so once a conversation that used tools was long enough to
be summarized, every later turn of it failed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.agent.graph import (
    _messages_as_text_for_summary,
)


def _call(call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": "HassTurnOn", "args": args or {"name": "Porch"}, "id": call_id}


def _tool_round() -> list[Any]:
    """Return one turn that used a tool: request, call, result, reply."""
    return [
        HumanMessage(content="Turn on the porch light", id="h1"),
        AIMessage(content="", tool_calls=[_call("call_1")], id="a1"),
        ToolMessage(
            content="Turned on Porch", tool_call_id="call_1", name="HassTurnOn", id="t1"
        ),
        AIMessage(content="The porch light is on.", id="a2"),
    ]


async def _summarize(
    monkeypatch: pytest.MonkeyPatch, removed: list[Any], summary: str = ""
) -> tuple[list[Any], dict[str, Any]]:
    """Run the real node and return what the model was sent and the result."""
    sent: list[Any] = []

    async def _fake_invoke_model(
        _model: Any, messages: Any, *_a: Any, **_k: Any
    ) -> AIMessage:
        sent.extend(messages)
        return AIMessage(content="The user turned on the porch light.")

    monkeypatch.setattr(agent_graph, "_invoke_model", _fake_invoke_model)
    state: Any = {"messages": [], "summary": summary, "messages_to_remove": removed}
    config: Any = {"configurable": {"summarization_model": object()}}
    result = await agent_graph._summarize_and_remove_messages(state, config)
    return sent, result


def _assert_text_only(messages: list[Any]) -> None:
    for message in messages:
        assert not isinstance(message, ToolMessage)
        assert not getattr(message, "tool_calls", None), "a tool call was sent"
        assert isinstance(message.content, str)
        assert message.content.strip(), "an empty message was sent"


@pytest.mark.asyncio
async def test_the_summary_request_holds_no_tool_call_or_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent, result = await _summarize(monkeypatch, _tool_round())

    _assert_text_only(sent)
    assert [type(m) for m in sent] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assistant = sent[2].content
    assert 'HassTurnOn with {"name": "Porch"}' in assistant
    assert "returned: Turned on Porch" in assistant
    assert assistant.endswith("The porch light is on.")
    # The whole round, results included, still leaves the history.
    assert [m.id for m in result["messages"]] == ["h1", "a1", "t1", "a2"]
    assert all(isinstance(m, RemoveMessage) for m in result["messages"])
    assert result["summary"] == "The user turned on the porch light."


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic"])
async def test_the_request_body_has_no_unanswered_tool_call(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """The body each provider's client would send, built without a network."""
    # A call whose result was not trimmed with it: the case no pairing fixes.
    removed = [*_tool_round(), HumanMessage(content="And the garage", id="h2")]
    removed.append(AIMessage(content="", tool_calls=[_call("call_2")], id="a3"))
    sent, _ = await _summarize(monkeypatch, removed)

    model: Any
    if provider == "anthropic":
        model = ChatAnthropic(model_name="claude-haiku-4-5", api_key="test")  # type: ignore[call-arg]
    else:
        model = ChatOpenAI(
            model="gpt-5-nano",
            api_key="test",  # type: ignore[arg-type]
            use_responses_api=provider == "openai_responses",
        )
    body = json.dumps(model._get_request_payload(sent))

    for marker in ("tool_calls", "tool_use", "function_call", "call_1", "call_2"):
        assert marker not in body, f"{marker} is in the {provider} request"
    assert "Turned on Porch" in body


def test_the_pin_and_alarm_code_are_left_out_of_the_text() -> None:
    history = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "confirm_sensitive_action",
                    "args": {"action_id": "abc", "pin": "4321"},
                    "id": "c1",
                },
                {
                    "name": "alarm_control",
                    "args": {"code": "9876", "state": "disarm"},
                    "id": "c2",
                },
                {"name": "GetLiveContext", "args": {}, "id": "c3"},
            ],
        )
    ]

    (message,) = _messages_as_text_for_summary(history)

    assert "4321" not in message.content
    assert "9876" not in message.content
    assert 'confirm_sensitive_action with {"action_id": "abc"}' in message.content
    assert 'alarm_control with {"state": "disarm"}' in message.content
    assert "(I used the tool GetLiveContext.)" in message.content


def test_long_arguments_and_results_are_cut_to_one_short_line() -> None:
    history = [
        AIMessage(content="", tool_calls=[_call("c1", name="x" * 5000)]),
        ToolMessage(content="line one\n\nline two " + "y" * 5000, tool_call_id="c1"),
    ]

    (message,) = _messages_as_text_for_summary(history)

    call_line, result_line = str(message.content).split("\n")
    assert len(call_line) < 260
    assert len(result_line) < 360
    assert "line one line two" in result_line
    assert "The tool HassTurnOn returned" in result_line, "named from its call"


def test_a_failed_result_and_a_result_without_its_call_are_still_text() -> None:
    history = [
        ToolMessage(content="", tool_call_id="gone", name="HassTurnOff"),
        ToolMessage(
            content=[{"type": "text", "text": "No such device"}],
            tool_call_id="c9",
            name="HassTurnOn",
            status="error",
        ),
        HumanMessage(content="Thanks"),
    ]

    rendered = _messages_as_text_for_summary(history)

    _assert_text_only(rendered)
    assert [type(m) for m in rendered] == [AIMessage, HumanMessage]
    assert "The tool HassTurnOff returned: nothing" in rendered[0].content
    assert "HassTurnOn failed and returned: No such device" in rendered[0].content


@pytest.mark.asyncio
async def test_a_conversation_without_tools_is_sent_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = [
        HumanMessage(content="Hello", id="h1"),
        AIMessage(content="Hi there.", id="a1"),
    ]

    sent, _ = await _summarize(monkeypatch, removed, summary="Earlier talk.")

    assert sent[1] is removed[0]
    assert (type(sent[2]), sent[2].content) == (AIMessage, "Hi there.")
    assert "Earlier talk." in sent[3].content
