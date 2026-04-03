"""Tests for Assist reasoning trace builder (ChatLog thinking_content source)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from custom_components.home_generative_agent.core.assist_reasoning_trace import (
    MAX_TOOL_MESSAGE_CHARS,
    build_assist_reasoning_trace,
)


def test_build_trace_truncates_tool_message() -> None:
    huge = "B" * (MAX_TOOL_MESSAGE_CHARS + 500)
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="call tool"),
            ToolMessage(content=huge, name="t1", tool_call_id="id1"),
            AIMessage(content="done"),
        ],
        "selected_tools": ["a"],
        "selected_instructions": [],
        "summary": "",
        "redacted_thinking_chunks": [],
    }
    trace = build_assist_reasoning_trace(state)
    assert "B" * 100 in trace
    assert len(trace) < len(huge)
    assert "… [truncated]" in trace or "[truncated]" in trace


def test_build_trace_includes_invalid_and_extra_kw_tool_calls() -> None:
    """Trace must show valid, invalid, and provider additional_kwargs tool payloads."""
    msg = AIMessage(
        content="calling",
        tool_calls=[
            {"name": "light.turn_on", "args": {}, "id": "call-1", "type": "tool_call"}
        ],
        invalid_tool_calls=[
            {
                "name": "broken",
                "args": "{}",
                "id": "bad-1",
                "error": "JSON decode error",
            }
        ],
        additional_kwargs={
            "tool_calls": [{"id": "raw-openai", "function": {"name": "x"}}]
        },
    )
    state = {
        "messages": [HumanMessage(content="go"), msg],
        "selected_tools": [],
        "selected_instructions": [],
        "summary": "",
        "redacted_thinking_chunks": [],
    }
    trace = build_assist_reasoning_trace(state)
    assert "Tool calls:" in trace
    assert "light.turn_on" in trace
    assert "Invalid tool calls:" in trace
    assert "JSON decode error" in trace
    assert "additional_kwargs" in trace
    assert "raw-openai" in trace


def test_build_trace_includes_thinking_chunks() -> None:
    state = {
        "messages": [HumanMessage(content="q"), AIMessage(content="a")],
        "selected_tools": [],
        "selected_instructions": [],
        "summary": "",
        "redacted_thinking_chunks": ["step one", "step two"],
    }
    trace = build_assist_reasoning_trace(state)
    assert "step one" in trace
    assert "step two" in trace
    assert "Model thinking" in trace
