"""

Build a single reasoning transcript string from LangGraph state for Assist ChatLog.

Home Assistant 2026.4+ reads thinking and tool history from the **conversation
ChatLog** (``AssistantContent.thinking_content``, external ``tool_calls``, and
``ToolResultContent``), not from ``IntentResponse`` speech ``extra_data``.

This module produces the **thinking_content** body (plus merged transcript, RAG
context, etc.) passed to ``append_langgraph_turn_to_chat_log``. Structured tool
calls and results are also represented as separate ChatLog entries there.

``IntentResponse.async_set_speech`` is used only for the final spoken answer text.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

# Hard caps so large tool payloads cannot overwhelm the HA frontend.
MAX_REASONING_TOTAL_CHARS = 24_000
MAX_TOOL_MESSAGE_CHARS = 2_000
MAX_ASSISTANT_TEXT_CHARS = 8_000
MAX_USER_TEXT_CHARS = 8_000
_TRUNC_SUFFIX = "\n… [truncated]"


def _truncate(text: str, max_chars: int) -> str:
    """Return text trimmed to max_chars with a clear suffix when truncated."""
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    budget = max_chars - len(_TRUNC_SUFFIX)
    if budget <= 0:
        return _TRUNC_SUFFIX.strip()
    return stripped[:budget].rstrip() + _TRUNC_SUFFIX


def _json_section(label: str, payload: object) -> str:
    """Serialize payload to JSON for trace, with fallback and truncation."""
    try:
        dumped = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return f"{label}\n{_truncate(str(payload), MAX_ASSISTANT_TEXT_CHARS)}"
    body = _truncate(dumped, MAX_ASSISTANT_TEXT_CHARS)
    return f"{label}\n{body}"


def _append_ai_tool_trace(parts: list[str], msg: AIMessage) -> None:
    """Append tool-call related lines (valid, invalid, provider raw kwargs)."""
    if msg.tool_calls:
        parts.append(_json_section("Tool calls:", msg.tool_calls))
    if msg.invalid_tool_calls:
        parts.append(_json_section("Invalid tool calls:", msg.invalid_tool_calls))
    raw_kw = msg.additional_kwargs or {}
    extra_tc = raw_kw.get("tool_calls")
    if extra_tc:
        parts.append(_json_section("Tool calls (additional_kwargs):", extra_tc))


def _format_message_line(msg: AnyMessage, index: int) -> str:
    """Format a single LangChain message for the transcript section."""
    header = f"[{index}] "
    if isinstance(msg, HumanMessage):
        body = _truncate(str(msg.content), MAX_USER_TEXT_CHARS)
        return f"{header}User\n{body}"
    if isinstance(msg, AIMessage):
        parts: list[str] = [f"{header}Assistant"]
        text = _truncate(str(msg.content), MAX_ASSISTANT_TEXT_CHARS)
        if text:
            parts.append(text)
        _append_ai_tool_trace(parts, msg)
        return "\n".join(parts)
    if isinstance(msg, ToolMessage):
        name = msg.name or "tool"
        tid = getattr(msg, "tool_call_id", "") or ""
        prefix = f"{header}Tool `{name}`"
        if tid:
            prefix += f" (id={tid})"
        body = _truncate(str(msg.content), MAX_TOOL_MESSAGE_CHARS)
        status = getattr(msg, "status", None)
        if status:
            return f"{prefix} [status={status}]\n{body}"
        return f"{prefix}\n{body}"
    fallback = _truncate(str(msg.content), MAX_ASSISTANT_TEXT_CHARS)
    return f"{header}{type(msg).__name__}\n{fallback}"


def build_assist_reasoning_trace(state: dict[str, Any]) -> str:
    """Assemble the reasoning / details string for ChatLog ``thinking_content``."""
    sections: list[str] = []

    thinking_chunks = state.get("redacted_thinking_chunks") or []
    if thinking_chunks:
        blocks: list[str] = []
        for i, chunk in enumerate(thinking_chunks, 1):
            if not (chunk and str(chunk).strip()):
                continue
            blocks.append(
                f"### Block {i}\n{_truncate(str(chunk), MAX_ASSISTANT_TEXT_CHARS)}"
            )
        if blocks:
            sections.append(
                "## Model thinking (redacted blocks)\n" + "\n\n".join(blocks)
            )

    selected_tools = state.get("selected_tools") or []
    selected_instructions = state.get("selected_instructions") or []
    if selected_tools or selected_instructions:
        lines = ["## Retrieval"]
        if selected_tools:
            lines.append("Tools: " + ", ".join(sorted(str(t) for t in selected_tools)))
        if selected_instructions:
            lines.append(
                "Instructions: "
                + ", ".join(sorted(str(i) for i in selected_instructions))
            )
        sections.append("\n".join(lines))

    summary = state.get("summary") or ""
    if isinstance(summary, str) and summary.strip():
        sections.append(
            "## Conversation summary context\n"
            + _truncate(summary, MAX_ASSISTANT_TEXT_CHARS)
        )

    messages = state.get("messages") or []
    if messages:
        transcript = "\n\n".join(
            _format_message_line(m, i + 1) for i, m in enumerate(messages)
        )
        sections.append("## Transcript\n" + transcript)

    combined = "\n\n".join(s for s in sections if s.strip())
    return _truncate(combined, MAX_REASONING_TOTAL_CHARS)
