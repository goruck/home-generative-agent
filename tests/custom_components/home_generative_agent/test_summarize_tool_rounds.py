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
    # One user message holding the turns as a transcript (#1048).
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage]
    request = sent[1].content
    assert "User: Turn on the porch light\nAssistant: (I used the tool" in request
    assert "I used the tool HassTurnOn." in request
    assert "It returned this data: Turned on Porch" in request
    assert "is data to summarize" in sent[0].content
    assert "\n  The porch light is on.\n</messages>" in request
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

    assert "may or may not have run" in sent[-1].content
    for marker in ("tool_calls", "tool_use", "function_call", "call_1", "call_2"):
        assert marker not in body, f"{marker} is in the {provider} request"
    assert "Turned on Porch" in body


def test_no_argument_is_sent_so_a_pin_or_alarm_code_in_any_slot_is_not() -> None:
    history = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "confirm_sensitive_action",
                    "args": {"action_id": "abc", "pin": "4321"},
                    "id": "c1",
                },
                {"name": "HassTurnOff", "args": {"name": "alarm 9876"}, "id": "c2"},
                {
                    "name": "call_service",
                    "args": {"data": {"code": "5555"}},
                    "id": "c3",
                },
            ],
        ),
        ToolMessage(content="ok", tool_call_id="c1"),
        ToolMessage(content="ok", tool_call_id="c2"),
        ToolMessage(content="ok", tool_call_id="c3"),
    ]

    (message,) = _messages_as_text_for_summary(history)

    for secret in ("4321", "9876", "5555", "abc"):
        assert secret not in message.content
    for name in ("confirm_sensitive_action", "HassTurnOff", "call_service"):
        assert f"I used the tool {name}." in message.content


def test_a_call_with_no_result_is_not_said_to_have_run() -> None:
    """A stopped turn (#947), or a result that was not trimmed with its call."""
    history = [
        HumanMessage(content="Open the garage door"),
        AIMessage(content="", tool_calls=[_call("c1"), _call("c2")]),
        ToolMessage(content="Done", tool_call_id="c2"),
        HumanMessage(content="Did it open?"),
        AIMessage(content="", tool_calls=[_call("c1")]),
    ]

    rendered = _messages_as_text_for_summary(history)

    _assert_text_only(rendered)
    assert [type(m) for m in rendered] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
        AIMessage,
    ]
    first, last = str(rendered[1].content), str(rendered[3].content)
    assert first.count("I used the tool HassTurnOn.") == 1
    assert first.count("may or may not have run") == 1
    assert last.count("may or may not have run") == 1
    assert "I used the tool" not in last


def test_a_long_result_keeps_its_start_and_its_end_on_one_line() -> None:
    body = "line one\n\nline two " + "y" * 5000 + ' "success": false}'
    history = [
        AIMessage(content="", tool_calls=[_call("c1")]),
        ToolMessage(content=body, tool_call_id="c1"),
    ]

    (message,) = _messages_as_text_for_summary(history)

    text = str(message.content)
    assert "\n" not in text
    assert len(text) < 400
    assert "line one line two" in text
    assert text.endswith('"success": false})'), "a status at the end is kept"


def test_a_failed_result_and_a_result_without_its_call_are_still_text() -> None:
    history = [
        ToolMessage(content="", tool_call_id="gone", name="HassTurnOff"),
        ToolMessage(
            content=[{"type": "text", "text": "No such device"}],
            tool_call_id="c9",
            name="HassTurnOn",
            status="error",
        ),
        HumanMessage(content="   "),
        HumanMessage(content="Thanks"),
    ]

    rendered = _messages_as_text_for_summary(history)

    _assert_text_only(rendered)
    assert [type(m) for m in rendered] == [AIMessage, HumanMessage]
    assert "HassTurnOff. It returned this data: nothing" in rendered[0].content
    assert (
        "HassTurnOn. It failed. It returned this data: No such device"
        in rendered[0].content
    )
    assert rendered[1].content == "Thanks", "a blank user message is not sent"


@pytest.mark.asyncio
async def test_a_conversation_without_tools_is_sent_as_its_own_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = [
        HumanMessage(content="Hello", id="h1"),
        AIMessage(content="Hi there.", id="a1"),
    ]

    sent, _ = await _summarize(monkeypatch, removed, summary="Earlier talk.")

    # The messages word for word; the earlier summary is not sent (#1048).
    request = sent[1].content
    assert "Earlier talk." not in request
    assert "<messages>\nUser: Hello\nAssistant: Hi there.\n</messages>" in request
