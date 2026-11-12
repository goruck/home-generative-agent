"""
OpenAI chat model that sends no reasoning items back to the Responses API.

GPT-6 and the other Responses-only models return their reasoning as reasoning
items, encrypted when ``store=False`` (OpenAI sends the encrypted copy whether
or not ``include`` asks for it), and langchain-openai replays every reasoning
item it finds in the history on the next turn.

The integration keeps a reply in langchain's ``v0`` output format, so a
history checkpointed on OpenAI still converts for an Ollama, Anthropic or
Gemini fallback. That format holds one reasoning item per message, and GPT-6
often returns several in one reply (four to eight on a multi-step tool task,
seen live 2026-11-11). Streaming concatenates their encrypted contents into
one string, and OpenAI rejects the replay with 400 ``invalid_encrypted_content``
on the next tool turn.

This subclass leaves reasoning items out of every Responses request, so the
model reasons afresh after each tool result, as it does on Chat Completions.
It also drops them from the reply: unused, they would only grow every
checkpoint, which rewrites the whole message list, by 10-40 KB a turn.
Replaying reasoning needs langchain's ``responses/v1`` format, which keeps
the items apart, plus a step that strips them before the history reaches a
non-OpenAI provider (TODOS.md; upstream langchain-ai/langchain#41807).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.outputs import ChatGenerationChunk, ChatResult

_REASONING_ITEM = "reasoning"
# Where langchain-openai's v0 output format puts a reply's reasoning item.
_REASONING_KWARG = "reasoning"


def drop_reasoning_items(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a Responses API payload without reasoning input items."""
    items = payload.get("input")
    if not isinstance(items, list):
        # A Chat Completions payload (``messages``) or a bare string input.
        return payload
    kept = [
        item
        for item in items
        if not (isinstance(item, dict) and item.get("type") == _REASONING_ITEM)
    ]
    if len(kept) == len(items):
        return payload
    return {**payload, "input": kept}


class StatelessReasoningChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that never replays reasoning items on the Responses API."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return drop_reasoning_items(payload)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for chunk in super()._stream(*args, **kwargs):
            chunk.message.additional_kwargs.pop(_REASONING_KWARG, None)
            yield chunk

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in super()._astream(*args, **kwargs):
            chunk.message.additional_kwargs.pop(_REASONING_KWARG, None)
            yield chunk

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _without_reasoning(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _without_reasoning(await super()._agenerate(*args, **kwargs))


def _without_reasoning(result: ChatResult) -> ChatResult:
    for generation in result.generations:
        generation.message.additional_kwargs.pop(_REASONING_KWARG, None)
    return result
