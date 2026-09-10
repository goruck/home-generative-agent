# ruff: noqa: S101
"""
Tests for core/prompt_cache.py — issue #617.

The system prompt is carried as two text blocks (stable prefix, volatile
tail). Anthropic must receive an explicit cache breakpoint on the stable
block; every other provider must receive the plain string it always did, so
the Anthropic-only key never reaches an API that rejects unknown fields.
"""

from __future__ import annotations

import types
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_message_to_dict
from pydantic import Field

import custom_components.home_generative_agent.agent.graph as agent_graph
from custom_components.home_generative_agent.core.fallback import FallbackChatModel
from custom_components.home_generative_agent.core.prompt_cache import (
    adapt_system_message_for_model,
    build_system_message,
    concrete_chat_model,
    is_anthropic_chat_model,
)

_TOOL = {
    "type": "function",
    "function": {
        "name": "HassTurnOn",
        "description": "Turn on",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _two_block_messages() -> list[Any]:
    return [
        build_system_message("STABLE instructions", "Current time is 15:09:20."),
        HumanMessage(content="hi"),
    ]


def _anthropic() -> ChatAnthropic:
    return ChatAnthropic(  # type: ignore[call-arg]
        model="claude-sonnet-4-6",  # type: ignore[call-arg]
        anthropic_api_key="test-key",  # type: ignore[call-arg]
        model_kwargs={"cache_control": {"type": "ephemeral"}},
    )


def _openai() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", api_key="test-key")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_system_message
# ---------------------------------------------------------------------------


def test_build_system_message_two_blocks_when_volatile() -> None:
    msg = build_system_message("stable", "volatile")
    assert msg.content == [
        {"type": "text", "text": "stable"},
        {"type": "text", "text": "volatile"},
    ]


def test_build_system_message_plain_string_without_volatile() -> None:
    assert build_system_message("stable", "").content == "stable"


# ---------------------------------------------------------------------------
# Unwrapping the runnable stack the graph actually builds
# ---------------------------------------------------------------------------


def test_concrete_chat_model_unwraps_configurable_bound_and_config() -> None:
    """configurable_fields → bind_tools → with_config all resolve to the model."""
    base = _anthropic()
    configurable = cast(
        "Any", base.configurable_fields(model=ConfigurableField(id="model"))
    )
    wrapped = configurable.bind_tools([_TOOL]).with_config(
        configurable={"model": "claude-haiku-4-5"}
    )
    assert isinstance(wrapped, BaseChatModel) is False
    concrete = concrete_chat_model(wrapped)
    assert isinstance(concrete, ChatAnthropic)
    assert is_anthropic_chat_model(wrapped)


def test_is_anthropic_false_for_openai_and_unresolvable() -> None:
    assert not is_anthropic_chat_model(_openai().bind_tools([_TOOL]))
    assert not is_anthropic_chat_model(object())
    assert concrete_chat_model(FallbackChatModel(chain=[])) is None


# ---------------------------------------------------------------------------
# adapt_system_message_for_model
# ---------------------------------------------------------------------------


def test_adapt_marks_stable_block_for_anthropic() -> None:
    messages = _two_block_messages()
    adapted = adapt_system_message_for_model(_anthropic().bind_tools([_TOOL]), messages)
    system = adapted[0]
    assert isinstance(system, SystemMessage)
    assert system.content == [
        {
            "type": "text",
            "text": "STABLE instructions",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Current time is 15:09:20."},
    ]
    assert adapted[1] is messages[1]


def test_adapt_collapses_to_string_for_openai() -> None:
    """No Anthropic-only key may reach OpenAI: it rejects unknown part fields."""
    adapted = adapt_system_message_for_model(_openai(), _two_block_messages())
    assert adapted[0].content == "STABLE instructions\nCurrent time is 15:09:20."
    assert "cache_control" not in repr(adapted[0])


def test_adapt_leaves_unresolvable_model_untouched() -> None:
    """A fallback chain shapes per member; the outer call must not pre-collapse."""
    messages = _two_block_messages()
    chain = FallbackChatModel(chain=[(_anthropic(), "cloud", "a")])
    assert adapt_system_message_for_model(chain, messages) is messages
    assert adapt_system_message_for_model(object(), messages) is messages


def test_adapt_does_not_mutate_input_and_is_idempotent() -> None:
    messages = _two_block_messages()
    original = [dict(b) for b in messages[0].content]  # type: ignore[union-attr]
    model = _anthropic()
    once = adapt_system_message_for_model(model, messages)
    twice = adapt_system_message_for_model(model, once)
    assert messages[0].content == original, "input system message was mutated"
    assert twice[0].content == once[0].content


def test_adapt_ignores_plain_string_system_and_non_list_input() -> None:
    plain = [SystemMessage(content="just text"), HumanMessage(content="hi")]
    assert adapt_system_message_for_model(_anthropic(), plain) is plain
    assert adapt_system_message_for_model(_anthropic(), "raw prompt") == "raw prompt"


# ---------------------------------------------------------------------------
# The pinned langchain-anthropic client really emits both breakpoints
# ---------------------------------------------------------------------------


def test_anthropic_request_payload_carries_both_breakpoints() -> None:
    """
    Pin the wire shape against the pinned langchain-anthropic.

    The stable block must carry the explicit breakpoint, the volatile block
    must not, and the client's top-level (automatic) cache_control must
    survive alongside it — Anthropic documents the two as compatible.
    """
    model = _anthropic()
    adapted = adapt_system_message_for_model(model, _two_block_messages())
    payload = model._get_request_payload(adapted)
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert payload["system"] == [
        {
            "type": "text",
            "text": "STABLE instructions",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Current time is 15:09:20."},
    ]
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_request_never_sees_cache_control() -> None:
    adapted = adapt_system_message_for_model(_openai(), _two_block_messages())
    assert _convert_message_to_dict(adapted[0]) == {
        "role": "system",
        "content": "STABLE instructions\nCurrent time is 15:09:20.",
    }


# ---------------------------------------------------------------------------
# FallbackChatModel shapes the system message per member
# ---------------------------------------------------------------------------


class _RecordingChat(BaseChatModel):
    """Minimal chat model that records what it was invoked with."""

    llm_type: str
    fail_with: Any = None
    seen: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return self.llm_type

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        if self.fail_with is not None:
            raise self.fail_with
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.llm_type))]
        )


@pytest.mark.asyncio
async def test_fallback_chain_shapes_system_message_per_member() -> None:
    """Anthropic primary fails over to OpenAI: each sees its own shape."""
    anthropic = _RecordingChat(
        llm_type="anthropic-chat",
        fail_with=httpx.ConnectError("down"),
    )
    openai = _RecordingChat(llm_type="openai-chat")
    chain = FallbackChatModel(chain=[(anthropic, "cloud", "a"), (openai, "cloud", "o")])

    result = await chain.ainvoke(_two_block_messages())

    assert result.content == "openai-chat"
    assert anthropic.seen[0][0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert openai.seen[0][0].content == "STABLE instructions\nCurrent time is 15:09:20."


@pytest.mark.parametrize(
    "content",
    [
        [],
        [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ],
        [{"type": "text", "text": 1}],
        ["bare string block"],
    ],
)
def test_adapt_passes_through_system_content_that_is_not_all_text_blocks(
    content: list[Any],
) -> None:
    messages = [SystemMessage(content=content), HumanMessage(content="hi")]
    assert adapt_system_message_for_model(_anthropic(), messages) is messages


def test_concrete_chat_model_gives_up_on_overdeep_wrappers() -> None:
    node: Any = types.SimpleNamespace(bound=None)
    for _ in range(12):
        node = types.SimpleNamespace(bound=node)
    assert concrete_chat_model(node) is None
    loop = types.SimpleNamespace()
    loop.bound = loop
    assert concrete_chat_model(loop) is None


@pytest.mark.asyncio
async def test_graph_invoke_model_shapes_system_message_for_concrete_model() -> None:
    """
    The single-provider wiring: graph._invoke_model must adapt before calling.

    Without this line Anthropic would receive two plain blocks (valid request,
    no breakpoint, cache_read 0 — the #617 symptom) and nothing would go red.
    """
    anthropic = _RecordingChat(llm_type="anthropic-chat")
    openai = _RecordingChat(llm_type="openai-chat")
    messages = _two_block_messages()

    await agent_graph._invoke_model(anthropic, messages, {})
    await agent_graph._invoke_model(openai, messages, {}, drop_unsupported=False)

    assert anthropic.seen[0][0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert openai.seen[0][0].content == "STABLE instructions\nCurrent time is 15:09:20."
    assert messages[0].content[0] == {"type": "text", "text": "STABLE instructions"}


@pytest.mark.asyncio
async def test_graph_invoke_model_leaves_fallback_chain_input_untouched() -> None:
    """The chain shapes per member; the outer call must not pre-collapse."""
    openai = _RecordingChat(llm_type="openai-chat")
    anthropic = _RecordingChat(llm_type="anthropic-chat")
    chain = FallbackChatModel(chain=[(openai, "cloud", "o"), (anthropic, "cloud", "a")])
    await agent_graph._invoke_model(chain, _two_block_messages(), {})
    assert openai.seen[0][0].content == "STABLE instructions\nCurrent time is 15:09:20."


class _RejectingSamplingChat(_RecordingChat):
    """Rejects sampling params once so the chain takes the rebind path."""

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(400, request=request)
        body = {
            "message": "Unsupported value: 'temperature' does not support 0.2.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_value",
        }
        raise openai.BadRequestError(body["message"], response=response, body=body)


@pytest.mark.asyncio
async def test_fallback_rebind_path_shapes_system_message_for_member() -> None:
    """The sampling-param rebind retry is a second injection point; cover it."""
    rejecting = _RejectingSamplingChat(llm_type="anthropic-chat")
    rebound = _RecordingChat(llm_type="anthropic-chat")
    stripped = MagicMock()
    stripped.bind_tools.return_value = rebound
    src = MagicMock()
    src.bind_tools.return_value = rejecting
    src.with_config.return_value = stripped

    chain = FallbackChatModel(chain=[(src, "cloud", "a")]).bind_tools([_TOOL])
    result = await chain.ainvoke(_two_block_messages())

    assert result.content == "anthropic-chat"
    assert rebound.seen[0][0].content[0]["cache_control"] == {"type": "ephemeral"}
