"""
Append LangGraph turn output to Home Assistant ChatLog (Assist 2026.4+ UI).

## How "Show details" actually works

The Assist frontend (`ha-assist-chat.ts`) builds the chat message from
``intent-progress`` pipeline events carrying ``chat_log_delta`` payloads.
``message.thinking`` and ``message.tool_calls`` are only populated from those
deltas — NOT from the static ``CONTENT_ADDED`` chat-log subscriber events.

The pipeline fires ``intent-progress`` through a ``chat_log_delta_listener``
that is attached to ``ChatLog.delta_listener``.  That listener is invoked
by ``ChatLog.async_add_delta_content_stream``, not by
``async_add_assistant_content_without_tools``.

Therefore the correct path is:

  1. Synthesise an async-generator of delta dicts from LangGraph output.
  2. Drive it through ``chat_log.async_add_delta_content_stream``.
  3. Consume the generator (we discard the yielded content objects — they are
     already stored on ``chat_log.content`` internally).

This ensures every delta reaches the pipeline listener and the frontend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import intent, llm
from homeassistant.util.ulid import ulid_now
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    from homeassistant.components.conversation.chat_log import ChatLog

_LOGGER = logging.getLogger(__name__)


# ── Turn slicing ────────────────────────────────────────────────────────────


def turn_messages_for_chat_log(
    messages: Sequence[AnyMessage], input_message_count: int
) -> list[AnyMessage]:
    """
    Return messages added by this invoke (stable against summarisation).

    Uses the pre-invoke message count as the primary boundary. If that yields
    nothing (e.g. the checkpointer shrank the list), falls back to messages
    after the last ``HumanMessage``.
    """
    if input_message_count < len(messages):
        sliced = list(messages[input_message_count:])
        if sliced:
            return sliced
    last_human = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human = i
    if last_human >= 0:
        return list(messages[last_human + 1 :])
    return []


# ── LangChain → HA type helpers ─────────────────────────────────────────────


def _tool_call_to_external_input(tc: Any) -> llm.ToolInput:
    """Map a LangChain tool call dict/object to an external ``ToolInput``."""
    if isinstance(tc, dict):
        name = str(tc.get("name") or "unknown")
        raw_args = tc.get("args")
        tc_id = tc.get("id")
    else:
        name = str(getattr(tc, "name", None) or "unknown")
        raw_args = getattr(tc, "args", None)
        tc_id = getattr(tc, "id", None)
    args = raw_args if isinstance(raw_args, dict) else {}
    call_id = str(tc_id) if tc_id else ulid_now()
    return llm.ToolInput(
        tool_name=name,
        tool_args=args,
        id=call_id,
        external=True,
    )


def _ai_text(msg: AIMessage) -> str | None:
    """Return stripped text content of an AIMessage, or None if empty."""
    content = msg.content
    if isinstance(content, str):
        return content.strip() or None
    return str(content).strip() or None


def _coerce_tool_result(content: Any) -> Any:
    """Return content in a JSON-safe form for ``ToolResultContent.tool_result``."""
    if isinstance(content, (dict, list, str, int, float, bool)) or content is None:
        return content
    return str(content)


# ── Delta synthesis ─────────────────────────────────────────────────────────


async def _delta_stream(  # noqa: PLR0912, PLR0913
    turn: Sequence[AnyMessage],
    *,
    reasoning_plain: str,
    has_native_thinking: bool,
    debug_assist_trace: bool,
    final_spoken_text: str,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse] | None,
) -> AsyncGenerator[dict[str, Any]]:
    """
    Yield ``AssistantContentDeltaDict`` / ``ToolResultContentDeltaDict`` items.

    Row order (mirrors the OpenAI entity streaming pattern):
      - Per-tool-call ``AssistantContent`` rows  (external ``tool_calls``).
      - ``ToolResultContent`` rows for each tool response.
      - Final ``AssistantContent`` with ``content`` + optional ``thinking_content``.

    Yielding a ``{"role": "assistant"}`` signals the start of a new assistant
    message to ``async_add_delta_content_stream``.
    """
    show_trace = has_native_thinking or debug_assist_trace

    started_first_message = False

    for msg in turn:
        if isinstance(msg, AIMessage):
            tcalls = getattr(msg, "tool_calls", None) or []
            if not tcalls:
                continue
            inputs = [_tool_call_to_external_input(tc) for tc in tcalls]
            if started_first_message:
                yield {"role": "assistant"}
            else:
                yield {"role": "assistant"}
                started_first_message = True
            text = _ai_text(msg)
            if text:
                yield {"content": text}
            yield {"tool_calls": inputs}

        elif isinstance(msg, ToolMessage):
            tid_key = str(getattr(msg, "tool_call_id", "") or "")
            if ha_tool_intent_responses and tid_key in ha_tool_intent_responses:
                tool_result_payload: Any = llm.IntentResponseDict(
                    ha_tool_intent_responses[tid_key]
                )
            else:
                tool_result_payload = _coerce_tool_result(msg.content)
            yield {
                "role": "tool_result",
                "tool_call_id": tid_key,
                "tool_name": msg.name or "tool",
                "tool_result": tool_result_payload,
            }

    # Final assistant message: answer text + optional thinking_content.
    if started_first_message:
        yield {"role": "assistant"}
    else:
        yield {"role": "assistant"}

    if show_trace and reasoning_plain.strip():
        yield {"thinking_content": reasoning_plain.strip()}

    content_str = final_spoken_text.strip() or None
    if content_str:
        yield {"content": content_str}


# ── Public entry point ───────────────────────────────────────────────────────


async def append_langgraph_turn_to_chat_log(  # noqa: PLR0913
    chat_log: ChatLog,
    agent_id: str,
    graph_messages: Sequence[AnyMessage],
    *,
    input_message_count: int,
    reasoning_plain: str,
    has_native_thinking: bool,
    debug_assist_trace: bool,
    final_spoken_text: str,
    ha_tool_intent_responses: Mapping[str, intent.IntentResponse] | None = None,
) -> None:
    """
    Drive LangGraph output through ChatLog so the Assist UI receives deltas.

    Uses ``chat_log.async_add_delta_content_stream`` which:
      - Calls ``ChatLog.delta_listener`` for every delta →
        pipeline fires ``intent-progress chat_log_delta`` →
        frontend ``ha-assist-chat`` populates ``message.thinking`` /
        ``message.tool_calls`` → "Show details" button appears.
      - Stores each ``AssistantContent`` / ``ToolResultContent`` on
        ``chat_log.content`` for persistence and
        ``async_get_result_from_chat_log``.

    ``thinking_content`` is only set when:
    - The model produced native thinking blocks (``has_native_thinking``), OR
    - ``debug_assist_trace`` is enabled (the "Debug: populate Assist Show
      details" option in HGA settings).
    """
    turn = turn_messages_for_chat_log(graph_messages, input_message_count)

    stream = _delta_stream(
        turn,
        reasoning_plain=reasoning_plain,
        has_native_thinking=has_native_thinking,
        debug_assist_trace=debug_assist_trace,
        final_spoken_text=final_spoken_text,
        ha_tool_intent_responses=ha_tool_intent_responses,
    )

    async for _ in chat_log.async_add_delta_content_stream(agent_id, stream):  # type: ignore[arg-type]
        pass
