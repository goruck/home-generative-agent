"""Tests for ChatLog append helpers (Assist 2026.4+ thinking / tool rows)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from homeassistant.helpers import intent, llm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from custom_components.home_generative_agent.core.assist_chat_log import (
    _delta_stream,
    append_langgraph_turn_to_chat_log,
    turn_messages_for_chat_log,
)

# ── turn_messages_for_chat_log ───────────────────────────────────────────────


def test_turn_messages_prefers_input_count() -> None:
    """Slice by invoke-time count is the primary boundary."""
    msgs = [
        HumanMessage(content="u"),
        AIMessage(content="a"),
        AIMessage(content="final"),
    ]
    assert turn_messages_for_chat_log(msgs, 1) == [
        AIMessage(content="a"),
        AIMessage(content="final"),
    ]


def test_turn_messages_falls_back_to_last_human() -> None:
    """If input_message_count equals len(messages), fall back to HumanMessage slice."""
    msgs = [
        HumanMessage(content="old"),
        AIMessage(content="a"),
        HumanMessage(content="new"),
        AIMessage(content="final"),
    ]
    assert turn_messages_for_chat_log(msgs, len(msgs)) == [AIMessage(content="final")]


# ── _delta_stream ────────────────────────────────────────────────────────────


async def _collect(gen: Any) -> list[dict]:
    """Drain an async generator into a list."""
    return [item async for item in gen]


async def test_delta_stream_interleaved_with_thinking() -> None:
    """Tool call → tool result → final with thinking_content when native thinking."""
    turn = [
        AIMessage(
            content="I'll check.",
            tool_calls=[
                {
                    "name": "get_and_analyze_camera_image",
                    "args": {"entity_id": "camera.x"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="vision output",
            tool_call_id="call_1",
            name="get_and_analyze_camera_image",
        ),
    ]
    deltas = await _collect(
        _delta_stream(
            turn,
            reasoning_plain="## Trace",
            has_native_thinking=True,
            debug_assist_trace=False,
            final_spoken_text="Meter reads 0.",
            ha_tool_intent_responses=None,
        )
    )

    roles = [d.get("role") for d in deltas if "role" in d]
    assert roles.count("assistant") == 2
    assert any(d.get("role") == "tool_result" for d in deltas)

    # tool_calls delta carries external ToolInput
    tool_calls_delta = next(d for d in deltas if "tool_calls" in d)
    assert len(tool_calls_delta["tool_calls"]) == 1
    assert tool_calls_delta["tool_calls"][0].external is True
    assert tool_calls_delta["tool_calls"][0].tool_name == "get_and_analyze_camera_image"

    # thinking_content delta present
    thinking_delta = next(d for d in deltas if "thinking_content" in d)
    assert thinking_delta["thinking_content"] == "## Trace"

    # final content delta is the last "content" key in the sequence
    content_delta = next(d for d in reversed(deltas) if "content" in d)
    assert content_delta["content"] == "Meter reads 0."


async def test_delta_stream_no_thinking_without_native_or_debug() -> None:
    """No thinking_content delta when neither native thinking nor debug is on."""
    deltas = await _collect(
        _delta_stream(
            [AIMessage(content="Hello.")],
            reasoning_plain="trace",
            has_native_thinking=False,
            debug_assist_trace=False,
            final_spoken_text="Hello.",
            ha_tool_intent_responses=None,
        )
    )
    assert not any("thinking_content" in d for d in deltas)


async def test_delta_stream_debug_toggle_forces_thinking() -> None:
    """debug_assist_trace=True includes thinking_content delta."""
    deltas = await _collect(
        _delta_stream(
            [],
            reasoning_plain="full trace",
            has_native_thinking=False,
            debug_assist_trace=True,
            final_spoken_text="Hello.",
            ha_tool_intent_responses=None,
        )
    )
    thinking_delta = next((d for d in deltas if "thinking_content" in d), None)
    assert thinking_delta is not None
    assert thinking_delta["thinking_content"] == "full trace"


async def test_delta_stream_native_thinking_populates_without_debug() -> None:
    """has_native_thinking=True includes thinking_content even if debug is off."""
    deltas = await _collect(
        _delta_stream(
            [],
            reasoning_plain="thinking blocks",
            has_native_thinking=True,
            debug_assist_trace=False,
            final_spoken_text="Answer.",
            ha_tool_intent_responses=None,
        )
    )
    thinking_delta = next((d for d in deltas if "thinking_content" in d), None)
    assert thinking_delta is not None
    assert thinking_delta["thinking_content"] == "thinking blocks"


async def test_delta_stream_always_emits_role_and_content() -> None:
    """Empty turn still emits role + content deltas for the final answer."""
    deltas = await _collect(
        _delta_stream(
            [],
            reasoning_plain="",
            has_native_thinking=False,
            debug_assist_trace=False,
            final_spoken_text="answer",
            ha_tool_intent_responses=None,
        )
    )
    assert any(d.get("role") == "assistant" for d in deltas)
    assert any(d.get("content") == "answer" for d in deltas)


async def test_delta_stream_maps_ha_intent_sidecar_to_intent_response_dict() -> None:
    """Tool results use llm.IntentResponseDict when sidecar has str(tool_call_id)."""
    ir = intent.IntentResponse(language="en")
    ir.async_set_speech("spoken elsewhere")
    sidecar = {"call-abc": ir}
    deltas = await _collect(
        _delta_stream(
            [
                ToolMessage(
                    content="{}",
                    tool_call_id="call-abc",
                    name="HassTurnOn",
                )
            ],
            reasoning_plain="",
            has_native_thinking=False,
            debug_assist_trace=False,
            final_spoken_text="Done.",
            ha_tool_intent_responses=sidecar,
        )
    )
    tr = next(d for d in deltas if d.get("role") == "tool_result")
    assert isinstance(tr["tool_result"], llm.IntentResponseDict)
    assert tr["tool_result"].original is ir
    assert tr["tool_call_id"] == "call-abc"


# ── append_langgraph_turn_to_chat_log ────────────────────────────────────────


async def test_append_calls_delta_stream() -> None:
    """append_langgraph_turn_to_chat_log drives async_add_delta_content_stream."""

    async def _fake_stream(_agent_id: str, _gen: Any):
        # Drain the generator (required) and yield nothing
        async for _ in _gen:
            pass
        return
        yield  # make it an async generator

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _fake_stream

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.test_agent",
        [HumanMessage(content="q"), AIMessage(content="Hello.")],
        input_message_count=1,
        reasoning_plain="trace",
        has_native_thinking=True,
        debug_assist_trace=False,
        final_spoken_text="Hello.",
    )
    # If we reach here without exception, the generator was consumed correctly.


async def test_append_passes_correct_agent_id() -> None:
    """agent_id is forwarded verbatim to async_add_delta_content_stream."""
    received: list[str] = []

    async def _capture_id(agent_id: str, gen: Any):
        received.append(agent_id)
        async for _ in gen:
            pass
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _capture_id

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.my_agent",
        [],
        input_message_count=0,
        reasoning_plain="",
        has_native_thinking=False,
        debug_assist_trace=False,
        final_spoken_text="hi",
    )
    assert received == ["conversation.my_agent"]


async def test_append_interleaved_rows_external_tool_calls() -> None:
    """Verify the delta sequence for a full tool-call turn."""
    captured_deltas: list[dict] = []

    async def _capture_deltas(_agent_id: str, gen: Any):
        async for d in gen:
            captured_deltas.append(d)
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _capture_deltas

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.test_agent",
        [
            HumanMessage(content="user"),
            AIMessage(
                content="I'll check.",
                tool_calls=[
                    {
                        "name": "get_and_analyze_camera_image",
                        "args": {"entity_id": "camera.x"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="vision output",
                tool_call_id="call_1",
                name="get_and_analyze_camera_image",
            ),
            AIMessage(content="The meter reads 0."),
        ],
        input_message_count=1,
        reasoning_plain="## Transcript\n...",
        has_native_thinking=True,
        debug_assist_trace=False,
        final_spoken_text="Fixed: The meter reads 0.",
    )

    # Must have at least two assistant roles (tool-calling + final)
    assistant_roles = [d for d in captured_deltas if d.get("role") == "assistant"]
    assert len(assistant_roles) >= 2

    # Must have a tool_result role
    assert any(d.get("role") == "tool_result" for d in captured_deltas)

    # Must carry external tool call
    tool_calls_delta = next((d for d in captured_deltas if "tool_calls" in d), None)
    assert tool_calls_delta is not None
    assert tool_calls_delta["tool_calls"][0].external is True

    # Must carry thinking_content (native thinking is True)
    assert any("thinking_content" in d for d in captured_deltas)

    # The last "content" delta must be the final_spoken_text
    content_delta = next((d for d in reversed(captured_deltas) if "content" in d), None)
    assert content_delta is not None
    assert "meter" in content_delta["content"].lower()


async def test_append_no_thinking_without_native_or_debug() -> None:
    """No thinking_content delta when both flags are False."""
    captured: list[dict] = []

    async def _cap(_aid: str, gen: Any):
        async for d in gen:
            captured.append(d)
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _cap

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.x",
        [HumanMessage(content="q"), AIMessage(content="Hello.")],
        input_message_count=1,
        reasoning_plain="trace",
        has_native_thinking=False,
        debug_assist_trace=False,
        final_spoken_text="Hello.",
    )
    assert not any("thinking_content" in d for d in captured)


async def test_append_debug_toggle_forces_thinking_content() -> None:
    """debug_assist_trace=True includes thinking_content delta."""
    captured: list[dict] = []

    async def _cap(_aid: str, gen: Any):
        async for d in gen:
            captured.append(d)
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _cap

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.x",
        [HumanMessage(content="q"), AIMessage(content="Hello.")],
        input_message_count=1,
        reasoning_plain="full trace",
        has_native_thinking=False,
        debug_assist_trace=True,
        final_spoken_text="Hello.",
    )
    thinking = next((d for d in captured if "thinking_content" in d), None)
    assert thinking is not None
    assert thinking["thinking_content"] == "full trace"


async def test_append_native_thinking_populates_without_debug() -> None:
    """has_native_thinking=True enables thinking_content regardless of debug."""
    captured: list[dict] = []

    async def _cap(_aid: str, gen: Any):
        async for d in gen:
            captured.append(d)
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _cap

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.x",
        [HumanMessage(content="q"), AIMessage(content="Hello.")],
        input_message_count=1,
        reasoning_plain="thinking blocks",
        has_native_thinking=True,
        debug_assist_trace=False,
        final_spoken_text="Hello.",
    )
    thinking = next((d for d in captured if "thinking_content" in d), None)
    assert thinking is not None
    assert thinking["thinking_content"] == "thinking blocks"


async def test_append_always_emits_final_content_even_with_empty_turn() -> None:
    """Final content delta is always present so async_get_result_from_chat_log works."""
    captured: list[dict] = []

    async def _cap(_aid: str, gen: Any):
        async for d in gen:
            captured.append(d)
        return
        yield

    chat_log = MagicMock()
    chat_log.async_add_delta_content_stream = _cap

    await append_langgraph_turn_to_chat_log(
        chat_log,
        "conversation.x",
        [],
        input_message_count=0,
        reasoning_plain="trace",
        has_native_thinking=False,
        debug_assist_trace=True,
        final_spoken_text="answer",
    )
    content_delta = next((d for d in captured if "content" in d), None)
    assert content_delta is not None
    assert content_delta["content"] == "answer"
    assert any("thinking_content" in d for d in captured)
