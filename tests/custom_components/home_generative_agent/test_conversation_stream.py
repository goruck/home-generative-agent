# ruff: noqa: S101, E402
"""Unit tests for _stream_langgraph_to_ha generator in conversation.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types
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

from custom_components.home_generative_agent.conversation import (
    _normalize_tool_result,
    _sanitize_tool_result_dict,
    _stream_langgraph_to_ha,
)


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
    """Non-streaming providers (Anthropic/OpenAI streaming=False) yield text from on_chat_model_end."""

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
    tool_result = cast("Any", deltas[3])["tool_result"]
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
        "tool_result": {"result": "Light turned on"},
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
        "tool_result": {"error": "Tool execution rejected by routing policy."},
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
    assert result_delta.get("tool_result") == {"result": "Success"}


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
    tool_result = cast("dict[str, Any]", result_delta.get("tool_result"))
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
