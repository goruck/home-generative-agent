# ruff: noqa: S101, E402
"""Unit tests for _stream_langgraph_to_ha generator in conversation.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolCallChunk,
    ToolMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from homeassistant.components.conversation import (
        AssistantContentDeltaDict,
        ToolResultContentDeltaDict,
    )


def _stub_ha_conversation() -> None:
    """Stub HA conversation module for testing."""
    if "homeassistant.components.conversation" in sys.modules:
        return

    # Build real (empty) base classes.
    class _AbstractConversationAgent:
        pass

    class _ConversationEntity:
        pass

    conv_mod: Any = types.ModuleType("homeassistant.components.conversation")
    conv_mod.AssistantContentDeltaDict = dict
    conv_mod.ToolResultContentDeltaDict = dict
    conv_mod.trace = MagicMock()
    conv_mod.ConversationEntity = _ConversationEntity

    # conversation.models submodule
    models_mod: Any = types.ModuleType("homeassistant.components.conversation.models")
    models_mod.AbstractConversationAgent = _AbstractConversationAgent
    conv_mod.models = models_mod

    sys.modules["homeassistant.components.conversation"] = conv_mod
    sys.modules["homeassistant.components.conversation.models"] = models_mod
    sys.modules["home_assistant_intents"] = types.ModuleType("home_assistant_intents")
    sys.modules["hassil"] = types.ModuleType("hassil")
    sys.modules["hassil.recognize"] = types.ModuleType("hassil.recognize")


_stub_ha_conversation()

from custom_components.home_generative_agent import conversation as hga_conversation
from custom_components.home_generative_agent.agent.helpers import chat_log_tool_result
from custom_components.home_generative_agent.conversation import (
    _get_synthetic_rejections,
    _handle_on_chain_end,
    _normalize_tool_result,
    _populate_chat_log_from_response,
    _reply_entry,
    _sanitize_tool_result_dict,
    _stream_langgraph_to_ha,
    _voice_tool_acknowledgement,
    _with_tool_acknowledgement,
)
from custom_components.home_generative_agent.core.utils import extract_final


def _tool_data(delta: Any) -> Any:
    """Return a tool-result delta's data in either HA chat-log shape."""
    if "tool_result" in delta:
        return delta["tool_result"]
    return delta["result"].data


async def _streamed_text(event_stream: AsyncGenerator[dict[str, Any]]) -> str:
    """
    Accumulate deltas the way HA's chat_log does.

    ChatLog.async_add_delta_content_stream builds the committed AssistantContent
    with `current_content += delta_content` — verbatim concatenation, no
    separator and no stripping.
    """
    text = ""
    async for delta in _stream_langgraph_to_ha(event_stream, "agent_1"):
        if chunk := cast("dict[str, Any]", delta).get("content"):
            text += chunk
    return text


@pytest.mark.asyncio
async def test_stream_text_only() -> None:
    """Test streaming plain text tokens."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Hello")},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=" world")},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="Hello world")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    assert len(deltas) == 3
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Hello"}
    assert deltas[2] == {"content": " world"}


@pytest.mark.asyncio
async def test_stream_nonstreaming_text_only() -> None:
    """Providers that don't emit on_chat_model_stream yield text from on_chat_model_end."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        # No on_chat_model_stream events (non-streaming model)
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="Here is the answer.")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    assert len(deltas) == 2
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Here is the answer."}


@pytest.mark.asyncio
async def test_stream_nonstreaming_tool_then_text() -> None:
    """Non-streaming multi-turn: tool call followed by text-only response."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        # Turn 1: tool call
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[{"name": "get_state", "args": {}, "id": "call_1"}],
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="on", tool_call_id="call_1", name="get_state"
                        )
                    ]
                }
            },
        }
        # Turn 2: text-only response (non-streaming)
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="The light is on.")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # role, tool_calls, tool_result, role, content
    assert len(deltas) == 5
    assert deltas[0] == {"role": "assistant"}
    assert "tool_calls" in deltas[1]
    assert deltas[2].get("role") == "tool_result"
    assert deltas[3] == {"role": "assistant"}
    assert deltas[4] == {"content": "The light is on."}


@pytest.mark.asyncio
async def test_stream_anthropic_text_delta_chunks() -> None:
    """Anthropic streaming chunks with type='text_delta' yield text correctly."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "chunk": AIMessageChunk(
                    content=[{"type": "text_delta", "text": "Hello"}]
                )
            },
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "chunk": AIMessageChunk(
                    content=[{"type": "text_delta", "text": " world"}]
                )
            },
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="Hello world")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    assert len(deltas) == 3
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Hello"}
    assert deltas[2] == {"content": " world"}


@pytest.mark.asyncio
async def test_stream_filters_thinking() -> None:
    """Test that Claude 'thinking' blocks are filtered out."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "chunk": AIMessageChunk(
                    content=[
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "The answer is 42."},
                    ]
                )
            },
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="The answer is 42.")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Thinking blocks are intentionally filtered in the MVP streaming path.
    # Piping thinking_content through AssistantContentDeltaDict is a deferred feature.
    assert len(deltas) == 2
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "The answer is 42."}


@pytest.mark.asyncio
async def test_stream_recursive_loops() -> None:
    """Test that role='assistant' is yielded at every agent start to support loops."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Thinking...")},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="Thinking...",
                    tool_calls=[{"name": "tool", "args": {}, "id": "call_1"}],
                )
            },
        }
        # Simulate Action node finishing
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="Result", tool_call_id="call_1", name="tool"
                        )
                    ]
                }
            },
        }
        # Loop back to Agent
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Done.")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Should have TWO {"role": "assistant"} blocks
    roles = [d for d in deltas if d.get("role") == "assistant"]
    assert len(roles) == 2
    assert deltas[0] == {"role": "assistant"}
    assert deltas[4] == {"role": "assistant"}


@pytest.mark.asyncio
async def test_stream_malformed_chunks() -> None:
    """Test that malformed chunks are skipped gracefully."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        # Correct
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Hello")},
        }
        # Missing data/chunk
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {},
        }
        # Content is list but blocks are missing text
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "text"}])},
        }
        # Not a chunk object at all (should be skipped by isinstance check)
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": {"completely": "wrong"}},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # 1. role="assistant"
    # 2. content="Hello"
    assert len(deltas) == 2
    assert deltas[1] == {"content": "Hello"}


@pytest.mark.asyncio
async def test_stream_partial_results() -> None:
    """Test that missing tool results trigger synthetic failures."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "tool_1", "args": {}, "id": "call_1"},
                        {"name": "tool_2", "args": {}, "id": "call_2"},
                    ],
                )
            },
        }
        # Only one result arrives
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="Result 1", tool_call_id="call_1", name="tool_1"
                        )
                    ]
                }
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # 1. role="assistant"
    # 2. tool_calls=[...]
    # 3. role="tool_result" for call_1
    # 4. role="tool_result" (synthetic) for call_2
    assert len(deltas) == 4
    assert cast("Any", deltas[2])["tool_call_id"] == "call_1"
    assert cast("Any", deltas[3])["role"] == "tool_result"
    assert cast("Any", deltas[3])["tool_call_id"] == "call_2"
    tool_result = _tool_data(deltas[3])
    assert "rejected by routing policy" in tool_result["error"]


@pytest.mark.asyncio
async def test_stream_with_tool_calls() -> None:
    """Test streaming with tool calls and results (successful mapping)."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="Calling tool...",
                    tool_calls=[
                        {
                            "name": "turn_on",
                            "args": {"entity_id": "light.test"},
                            "id": "call_1",
                        }
                    ],
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="Light turned on",
                            tool_call_id="call_1",
                            name="turn_on",
                        )
                    ]
                }
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    assert len(deltas) == 3
    assert deltas[0] == {"role": "assistant"}
    assert "tool_calls" in deltas[1]
    assert deltas[2] == {
        "role": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "turn_on",
        **chat_log_tool_result({"result": "Light turned on"}),
    }


@pytest.mark.asyncio
async def test_stream_orphaned_tool_calls() -> None:
    """Test synthetic rejection for tool calls that yield NO results at all."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "agent"},
        }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "turn_on",
                            "args": {"entity_id": "light.test"},
                            "id": "call_1",
                        }
                    ],
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {"output": {"messages": []}},  # Rejected by guard
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Note: Synthetic rejection fires on ACTION node completion, not generator end
    assert len(deltas) == 3
    assert deltas[0] == {"role": "assistant"}
    assert "tool_calls" in deltas[1]
    assert deltas[2] == {
        "role": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "turn_on",
        **chat_log_tool_result(
            {"error": "Tool execution rejected by routing policy."}, error=True
        ),
    }


@pytest.mark.asyncio
async def test_stream_cancellation() -> None:
    """Test that the generator handles asyncio.CancelledError gracefully."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        raise asyncio.CancelledError

    gen = _stream_langgraph_to_ha(event_stream(), "agent_1")

    # First item should arrive (the assistant role)
    delta = await anext(gen)
    assert delta == {"role": "assistant"}

    # Next iteration should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await anext(gen)


# ---------------------------------------------------------------------------
# _normalize_tool_result unit tests
# ---------------------------------------------------------------------------


def test_normalize_tool_result_plain_string() -> None:
    """Plain (non-JSON) strings are wrapped under 'result'."""
    result = _normalize_tool_result("light turned on")
    assert result == {"result": "light turned on"}


def test_normalize_tool_result_json_string_parsed_to_dict() -> None:
    """
    JSON-encoded dicts are deserialized.

    _parse_tool_response produces json.dumps output; Show Details must show
    clean key/value pairs, not escaped JSON.
    """
    content = '{"timezone": "Europe/London", "datetime": "2026-04-17T12:00:00"}'
    result = _normalize_tool_result(content)
    assert result == {"timezone": "Europe/London", "datetime": "2026-04-17T12:00:00"}


def test_normalize_tool_result_nested_content_blocks() -> None:
    """HA tools that return Anthropic-style content blocks are also deserialized."""
    content = '{"content": [{"type": "text", "text": "The time is noon."}]}'
    result = _normalize_tool_result(content)
    assert result == {"content": [{"type": "text", "text": "The time is noon."}]}


def test_normalize_tool_result_list_of_blocks() -> None:
    """List content (Anthropic content blocks) is joined into a single string."""
    content = [
        {"type": "text", "text": "Part one."},
        {"type": "text", "text": "Part two."},
    ]
    result = _normalize_tool_result(content)
    assert result == {"result": "Part one.\nPart two."}


def test_normalize_tool_result_non_dict_json_stays_wrapped() -> None:
    """JSON arrays and scalars are NOT returned as-is; they stay wrapped."""
    result = _normalize_tool_result("[1, 2, 3]")
    assert result == {"result": "[1, 2, 3]"}


# ---------------------------------------------------------------------------
# on_chain_end state-event guard test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ignores_state_chain_end() -> None:
    """
    LangGraph emits a second on_chain_end for 'action' with a list output.

    The output is the full messages list (not a dict). The generator must skip
    it without error.
    """

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[{"name": "turn_on", "args": {}, "id": "call_1"}],
                )
            },
        }
        # (a) node function event — dict output
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="done", tool_call_id="call_1", name="turn_on"
                        )
                    ]
                }
            },
        }
        # (b) graph-level state event — list output (must be ignored)
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": [
                    ToolMessage(content="done", tool_call_id="call_1", name="turn_on")
                ]
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Exactly one tool result — the state event must not add a duplicate.
    tool_results = cast(
        "list[ToolResultContentDeltaDict]",
        [d for d in deltas if isinstance(d, dict) and d.get("role") == "tool_result"],
    )
    assert len(tool_results) == 1
    assert cast("Any", tool_results[0])["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_idless_tool_calls() -> None:
    """Test correlation for models that emit tool calls without IDs (e.g. Ollama)."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "test_tool", "args": {}, "id": None},  # ID is None
                    ],
                )
            },
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="Success", tool_call_id="", name="test_tool"
                        )
                    ]
                }
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # 1. role="assistant"
    # 2. AssistantContentDeltaDict with generated ULID
    # 3. ToolResultContentDeltaDict with matching ULID
    assert len(deltas) == 3
    assistant_delta = cast("AssistantContentDeltaDict", deltas[1])
    tool_calls = assistant_delta.get("tool_calls")
    assert tool_calls is not None
    generated_id = tool_calls[0].id
    assert generated_id.startswith("01")  # Valid ULID start

    result_delta = cast("ToolResultContentDeltaDict", deltas[2])
    assert result_delta.get("tool_call_id") == generated_id
    assert _tool_data(result_delta) == {"result": "Success"}


@pytest.mark.asyncio
async def test_stream_graph_error_persistence_guard() -> None:
    """Test that orphaned tool calls are flushed if the graph crashes."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="",
                    tool_calls=[{"name": "tool_1", "args": {}, "id": "call_1"}],
                )
            },
        }
        # Simulate a crash before the action node completes
        msg = "Graph exploded"
        raise ValueError(msg)

    deltas = []
    with contextlib.suppress(ValueError):
        async for delta in _stream_langgraph_to_ha(event_stream(), "agent_1"):
            deltas.append(delta)  # noqa: PERF401

    # Should have yielded the synthetic rejection in the except block
    assert len(deltas) == 3
    assert deltas[2]["role"] == "tool_result"
    result_delta = cast("ToolResultContentDeltaDict", deltas[2])
    assert result_delta.get("tool_call_id") == "call_1"
    tool_result = cast("dict[str, Any]", _tool_data(result_delta))
    assert "failed mid-stream" in tool_result["error"]


@pytest.mark.asyncio
async def test_stream_drops_orphaned_tool_results() -> None:
    """Test that tool results without a preceding tool call are safely dropped."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(content="ok", tool_call_id="", name="tool")
                    ]
                }
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]
    # Result for an unknown call should be dropped, leaving only the agent start.
    assert len(deltas) == 1
    assert deltas[0].get("role") == "assistant"


@pytest.mark.asyncio
async def test_stream_mixed_text_and_tools() -> None:
    """Test that text is not swallowed when mixed with tool call chunks."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                # Mixed chunk: has both content and tool_call_chunks
                "chunk": AIMessageChunk(
                    content="Thinking... ",
                    tool_call_chunks=[
                        ToolCallChunk(name="tool", args="{}", id="call_1", index=0)
                    ],
                )
            },
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Should have yielded the text despite the presence of tool_call_chunks
    assert len(deltas) == 2
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Thinking... "}


@pytest.mark.asyncio
async def test_stream_role_tracking() -> None:
    """Test that redundant role deltas are suppressed."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        # Simulate a second start (e.g. from a loop without an intervening role change)
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content="Hi")},
        }

    deltas = [d async for d in _stream_langgraph_to_ha(event_stream(), "agent_1")]

    # Should only have ONE role="assistant" delta
    roles = [d for d in deltas if d.get("role") == "assistant"]
    assert len(roles) == 1
    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Hi"}


# ---------------------------------------------------------------------------
# _sanitize_tool_result_dict unit tests
# ---------------------------------------------------------------------------


def test_sanitize_passthrough_plain_dict() -> None:
    """Dicts without response_type are returned unchanged."""
    d = {"timezone": "Europe/London", "datetime": "2026-04-17T12:00:00"}
    assert _sanitize_tool_result_dict(d) == d


def test_sanitize_ha_action_done_with_speech() -> None:
    """HassTurnOn-style response is flattened; speech text becomes 'result'."""
    d = {
        "response_type": "action_done",
        "speech": {"plain": {"speech": "Turned on Garage Light", "extra_data": None}},
        "data": {
            "success": [{"name": "Garage Light", "type": "domain"}],
            "failed": [],
        },
    }
    result = _sanitize_tool_result_dict(d)
    assert result["result"] == "Turned on Garage Light"
    assert result["success"] == [{"name": "Garage Light", "type": "domain"}]
    assert result["failed"] == []
    assert "response_type" not in result
    assert "speech" not in result


def test_sanitize_ha_action_done_empty_speech_falls_back_to_response_type() -> None:
    """When speech text is absent the response_type string is used as result."""
    d = {
        "response_type": "action_done",
        "speech": {},
        "data": {"success": [], "failed": []},
    }
    result = _sanitize_tool_result_dict(d)
    assert result["result"] == "action_done"
    assert "response_type" not in result


def test_normalize_tool_result_ha_intent_string() -> None:
    """JSON-encoded HA intent responses are sanitized by _normalize_tool_result."""
    content = json.dumps(
        {
            "response_type": "action_done",
            "speech": {
                "plain": {"speech": "Turned on Garage Light", "extra_data": None}
            },
            "data": {
                "success": [{"name": "Garage Light", "type": "domain"}],
                "failed": [],
            },
        }
    )
    result = _normalize_tool_result(content)
    assert result["result"] == "Turned on Garage Light"
    assert "response_type" not in result
    assert "speech" not in result


# ---------------------------------------------------------------------------
# Issue #628: the streamed copy and the graph-state copy must be byte-identical
# ---------------------------------------------------------------------------
# _async_run_astream compares chat_log's committed text against the final
# AIMessage from graph state and, when they differ, assumes a mid-stream model
# fallback produced a new answer and replaces the chat_log entry. _call_model
# builds that graph-state text with extract_final(..., collapse_whitespace=False).
# These tests pin both sides of that comparison: nothing in the suite covered it,
# so a normalization change on either side would silently reintroduce #628.


@pytest.mark.asyncio
async def test_streamed_text_matches_graph_state_for_multiline_reply() -> None:
    """A streamed markdown reply must equal its extract_final graph-state copy."""
    reply = (
        "Here's your security audit:\n\n"
        "**High severity:**\n\n"
        "1. **Add-ons** - text.\n"
        "2. **Locks** - text.\n"
    )

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        # Providers split a reply at arbitrary points, including inside a line.
        for piece in (reply[:20], reply[20:47], reply[47:]):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "agent"},
                "data": {"chunk": AIMessageChunk(content=piece)},
            }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content=reply)},
        }

    streamed = await _streamed_text(event_stream())
    graph_state = extract_final(reply, collapse_whitespace=False)

    assert streamed == reply
    assert streamed == graph_state


@pytest.mark.asyncio
async def test_streamed_text_matches_graph_state_for_anthropic_blocks() -> None:
    """Multiple content blocks must join the same way on both sides."""
    blocks: list[Any] = [
        {"type": "thinking", "thinking": "reasoning"},
        {"type": "text", "text": "First paragraph.\n\n"},
        {"type": "text", "text": "Second paragraph.\n"},
    ]

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        for block in blocks:
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "agent"},
                "data": {"chunk": AIMessageChunk(content=[block])},
            }
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content=blocks)},
        }

    streamed = await _streamed_text(event_stream())
    graph_state = extract_final(blocks, collapse_whitespace=False)

    assert streamed == "First paragraph.\n\nSecond paragraph.\n"
    assert streamed == graph_state


@pytest.mark.asyncio
async def test_nonstreaming_text_matches_graph_state_for_multiline_reply() -> None:
    """The on_chat_model_end fallback path must agree with graph state too."""
    reply = "Line one.\n\n- bullet\n- bullet\n"

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content=reply)},
        }

    streamed = await _streamed_text(event_stream())

    assert streamed == extract_final(reply, collapse_whitespace=False)


def _chunked_turn(
    *chunks: str, tool_calls: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """One streamed model turn: start, one stream event per chunk, end."""
    events: list[dict[str, Any]] = [
        {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
    ]
    events.extend(
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": AIMessageChunk(content=chunk)},
        }
        for chunk in chunks
    )
    events.append(
        {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {
                "output": AIMessage(
                    content="".join(chunks), tool_calls=tool_calls or []
                )
            },
        }
    )
    return events


async def _replay(events: list[dict[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_stream_drops_think_block_split_across_deltas() -> None:
    """Inline reasoning never reaches the chat_log, so a voice pipeline never speaks it."""
    chunks = ("<thi", "nk>The user wants the lig", "ht.</th", "ink>\n\nIt is ", "on.")
    streamed = await _streamed_text(_replay(_chunked_turn(*chunks)))

    assert streamed == "It is on."
    assert streamed == extract_final("".join(chunks), collapse_whitespace=False)


@pytest.mark.asyncio
async def test_stream_drops_unclosed_think_block() -> None:
    """A block the model never closes runs to the end of the turn, like extract_final."""
    streamed = await _streamed_text(
        _replay(_chunked_turn("Sure. ", "<think>still reasoning", " about it"))
    )

    assert streamed == "Sure. "


@pytest.mark.asyncio
async def test_stream_keeps_angle_brackets_that_are_not_tags() -> None:
    """A held tag prefix that never completes is released at the end of the turn."""
    streamed = await _streamed_text(_replay(_chunked_turn("If a < b then ", "use <th")))

    assert streamed == "If a < b then use <th"


@pytest.mark.asyncio
async def test_stream_releases_held_text_before_tool_calls() -> None:
    """Text held as a possible tag is emitted ahead of the turn's tool calls."""
    events = _chunked_turn(
        "Let me check <",
        tool_calls=[{"name": "get_state", "args": {}, "id": "call_1"}],
    )
    deltas = [d async for d in _stream_langgraph_to_ha(_replay(events), "agent_1")]

    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Let me check "}
    assert deltas[2] == {"content": "<"}
    assert "tool_calls" in deltas[3]


@pytest.mark.asyncio
async def test_nonstreaming_text_drops_think_block() -> None:
    """The on_chat_model_end fallback path strips inline reasoning too."""
    reply = "<think>reasoning</think>\n\nThe door is locked."

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content=reply)},
        }

    assert await _streamed_text(event_stream()) == "The door is locked."


@pytest.mark.asyncio
async def test_nonstreaming_think_only_reply_streams_nothing() -> None:
    """A reply that is all reasoning is not delivered as the answer."""

    async def event_stream() -> AsyncGenerator[dict[str, Any]]:
        yield {"event": "on_chat_model_start", "metadata": {"langgraph_node": "agent"}}
        yield {
            "event": "on_chat_model_end",
            "metadata": {"langgraph_node": "agent"},
            "data": {"output": AIMessage(content="<think>only reasoning")},
        }

    assert await _streamed_text(event_stream()) == ""


_TOOL_CALL = [{"name": "get_state", "args": {}, "id": "call_1"}]


def _tool_then_answer(*first_turn_chunks: str) -> list[dict[str, Any]]:
    """Build a tool-calling turn, its result, then a text answer."""
    return [
        *_chunked_turn(*first_turn_chunks, tool_calls=_TOOL_CALL),
        {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "action"},
            "data": {
                "output": {
                    "messages": [
                        ToolMessage(
                            content="off", tool_call_id="call_1", name="get_state"
                        )
                    ]
                }
            },
        },
        *_chunked_turn("The landing light is off."),
    ]


async def _acked(events: list[dict[str, Any]], ack: str) -> list[dict[str, Any]]:
    stream = _with_tool_acknowledgement(
        _stream_langgraph_to_ha(_replay(events), "agent_1"), ack
    )
    return [cast("dict[str, Any]", d) async for d in stream]


@pytest.mark.asyncio
async def test_acknowledgement_precedes_first_tool_call() -> None:
    """Text then a tool call is what makes the pipeline start speaking (#671)."""
    deltas = await _acked(_tool_then_answer(), "Let me check.")

    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Let me check. "}
    assert "tool_calls" in deltas[2]
    assert [d.get("content") for d in deltas if d.get("content")] == [
        "Let me check. ",
        "The landing light is off.",
    ]


@pytest.mark.asyncio
async def test_acknowledgement_skipped_when_model_speaks_first() -> None:
    """A model that says something before its tool call is not talked over."""
    deltas = await _acked(_tool_then_answer("One moment."), "Let me check.")

    contents = [d.get("content") for d in deltas if d.get("content")]
    assert contents == ["One moment.", "The landing light is off."]


@pytest.mark.asyncio
async def test_acknowledgement_said_once_per_turn() -> None:
    """A second tool round does not repeat it."""
    events = [
        *_tool_then_answer()[:-3],
        *_tool_then_answer(),
    ]
    deltas = await _acked(events, "Let me check.")

    assert [d.get("content") for d in deltas].count("Let me check. ") == 1


@pytest.mark.asyncio
async def test_no_acknowledgement_on_text_only_turn_or_when_off() -> None:
    """An answer with no tool call, or an empty option, adds nothing."""
    text_only = await _acked(_chunked_turn("It is noon."), "Let me check.")
    assert [d.get("content") for d in text_only if d.get("content")] == ["It is noon."]

    off = await _acked(_tool_then_answer(), "")
    assert all(d.get("content") != " " for d in off)
    assert [d.get("content") for d in off if d.get("content")] == [
        "The landing light is off."
    ]


@pytest.mark.parametrize(
    ("satellite_id", "device_id", "expected"),
    [
        ("assist_satellite.voice_pe", None, "Let me check."),
        ("assist_satellite.voice_pe", "voice_device", "Let me check."),
        # A typed pipeline run may carry a device id for area context.
        (None, "voice_device", ""),
        (None, None, ""),
    ],
)
def test_acknowledgement_only_on_satellite_turns(
    satellite_id: str | None, device_id: str | None, expected: str
) -> None:
    """A device id alone does not mean anyone is listening."""
    user_input = cast(
        "Any", types.SimpleNamespace(satellite_id=satellite_id, device_id=device_id)
    )
    options = {"voice_tool_acknowledgement": "  Let me check.  "}

    assert _voice_tool_acknowledgement(options, user_input).text == expected
    assert _voice_tool_acknowledgement({}, user_input).text == ""


@pytest.mark.parametrize(
    ("configured", "spoken"),
    [
        ("One moment", "One moment."),
        ("Let me check!", "Let me check!"),
        ("Momentík…", "Momentík…"),
        ('Checking "now."', 'Checking "now."'),
    ],
)
def test_acknowledgement_gets_a_sentence_end(configured: str, spoken: str) -> None:
    """A streaming TTS engine only speaks a phrase early once it ends a sentence."""
    user_input = cast("Any", types.SimpleNamespace(satellite_id="s", device_id=None))

    assert (
        _voice_tool_acknowledgement(
            {"voice_tool_acknowledgement": configured}, user_input
        ).text
        == spoken
    )


@pytest.mark.asyncio
async def test_whitespace_before_tool_call_does_not_use_up_acknowledgement() -> None:
    """A model that emits only a newline before its tool call said nothing."""
    deltas = await _acked(_tool_then_answer("\n\n"), "Let me check.")

    assert "Let me check. " in [d.get("content") for d in deltas]


@pytest.mark.asyncio
async def test_think_filter_survives_characters_that_lengthen_when_lowered() -> None:
    """Turkish "İ" lowers to two characters; the answer must not lose any."""
    streamed = await _streamed_text(
        _replay(_chunked_turn("<think>İyi İstanbul</think>", "Merhaba dünya"))
    )

    assert streamed == "Merhaba dünya"


@pytest.mark.parametrize(
    "chunks",
    [
        ("Hello<think>x</think> world",),
        ("Sure. <think>x</think> ok",),
        ("<think>x</think>Answer\n",),
        ("<thi", "nk>x</think>\n\nLine one.\n\n", "Line two.\n"),
        ("Intro ", "<think>a</think>", " middle ", "<THINK>b</think>", " end\n"),
    ],
)
@pytest.mark.asyncio
async def test_think_filter_whitespace_matches_extract_final(
    chunks: tuple[str, ...],
) -> None:
    """Only the reply's outer whitespace goes; words are never glued together."""
    streamed = await _streamed_text(_replay(_chunked_turn(*chunks)))

    assert streamed == extract_final("".join(chunks), collapse_whitespace=False)


def test_acknowledgement_timing_defaults_to_turn_start() -> None:
    """Unset timing means turn start; a stored choice is honored."""
    user_input = cast("Any", types.SimpleNamespace(satellite_id="s", device_id=None))
    options: dict[str, Any] = {"voice_tool_acknowledgement": "One moment."}

    assert _voice_tool_acknowledgement(options, user_input).timing == "turn_start"
    options["voice_tool_acknowledgement_timing"] = "tool_call"
    assert _voice_tool_acknowledgement(options, user_input).timing == "tool_call"


def test_unknown_acknowledgement_timing_falls_back_to_the_default() -> None:
    """A stale or hand-edited timing value never silently changes the trigger."""
    user_input = cast("Any", types.SimpleNamespace(satellite_id="s", device_id=None))
    options = {
        "voice_tool_acknowledgement": "One moment.",
        "voice_tool_acknowledgement_timing": "start",
    }

    assert _voice_tool_acknowledgement(options, user_input).timing == "turn_start"


def test_reply_entry_ignores_only_a_lone_acknowledgement() -> None:
    """Recovery must not read a turn-start acknowledgement as the reply."""
    only_ack = ["user request", "One moment."]
    assert _reply_entry(only_ack, 1) is None
    assert _reply_entry([*only_ack, "The light is off."], 1) == "The light is off."
    assert _reply_entry(["user request", "reply"], None) == "reply"
    assert _reply_entry([], None) is None


# ---------------------------------------------------------------------------
# HA 2026.10 renamed the chat log's tool result to ``result: llm.ToolResult``
# (issue #735). ``tool_result=`` is a TypeError on ToolResultContent there and a
# deprecated delta key; 2026.9 has no llm.ToolResult and wants ``tool_result``.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ToolResult2610:
    data: Any
    error: bool = False


@dataclass(frozen=True)
class _ToolResultContent2610:
    agent_id: str
    tool_call_id: str
    tool_name: str
    result: _ToolResult2610


@pytest.fixture
def ha_2026_10(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give llm and conversation HA 2026.10's tool result shapes."""
    from homeassistant.helpers import llm  # noqa: PLC0415

    monkeypatch.setattr(llm, "ToolResult", _ToolResult2610, raising=False)
    monkeypatch.setattr(
        hga_conversation.conversation,
        "ToolResultContent",
        _ToolResultContent2610,
        raising=False,
    )


def _tool_message() -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"success": True}), tool_call_id="c1", name="HassTurnOn"
    )


@pytest.mark.usefixtures("ha_2026_10")
def test_schema_first_backfill_builds_a_2026_10_tool_result() -> None:
    """
    Field-shaped case: schema-first turns crashed after the device acted.

    "ToolResultContent.__init__() got an unexpected keyword argument
    'tool_result'".
    """
    chat_log = MagicMock()

    _populate_chat_log_from_response(chat_log, "conversation.hga", [_tool_message()])

    added = chat_log.async_add_assistant_content_without_tools.call_args.args[0]
    assert added.result == _ToolResult2610(data={"success": True})


@pytest.mark.usefixtures("ha_2026_10")
def test_streamed_tool_result_uses_the_2026_10_result_key() -> None:
    pending: dict[str, Any] = {"c1": {"name": "HassTurnOn", "args": {}, "id": "c1"}}

    deltas = _handle_on_chain_end(
        {"output": {"messages": [_tool_message()]}}, pending, deque()
    )

    delta = cast("dict[str, Any]", deltas[0])
    assert "tool_result" not in delta
    assert delta["result"] == _ToolResult2610(data={"success": True})


@pytest.mark.usefixtures("ha_2026_10")
def test_synthetic_rejection_is_flagged_as_an_error() -> None:
    pending: dict[str, Any] = {"c1": {"name": "HassTurnOn", "args": {}, "id": "c1"}}

    delta = cast("dict[str, Any]", _get_synthetic_rejections(pending, "Rejected.")[0])

    assert delta["result"].error is True
    assert delta["result"].data == {"error": "Rejected."}


def test_streamed_tool_result_keeps_tool_result_before_2026_10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from homeassistant.helpers import llm  # noqa: PLC0415

    monkeypatch.delattr(llm, "ToolResult", raising=False)
    pending: dict[str, Any] = {"c1": {"name": "HassTurnOn", "args": {}, "id": "c1"}}

    deltas = _handle_on_chain_end(
        {"output": {"messages": [_tool_message()]}}, pending, deque()
    )

    delta = cast("dict[str, Any]", deltas[0])
    assert delta["tool_result"] == {"success": True}
    assert "result" not in delta


@pytest.mark.usefixtures("ha_2026_10")
def test_failed_tool_message_is_flagged_as_an_error_in_the_chat_log() -> None:
    """A tool that failed must not show as a success in HA's Show Details."""
    failed = ToolMessage(
        content=json.dumps({"error": "Lock jammed"}),
        tool_call_id="c1",
        name="HassTurnOn",
        status="error",
    )
    pending: dict[str, Any] = {"c1": {"name": "HassTurnOn", "args": {}, "id": "c1"}}
    chat_log = MagicMock()

    deltas = _handle_on_chain_end({"output": {"messages": [failed]}}, pending, deque())
    _populate_chat_log_from_response(chat_log, "conversation.hga", [failed])

    assert cast("dict[str, Any]", deltas[0])["result"].error is True
    added = chat_log.async_add_assistant_content_without_tools.call_args.args[0]
    assert added.result.error is True
