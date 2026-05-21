# ruff: noqa: S101
"""Tests for core fallback wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from custom_components.home_generative_agent.core.fallback import (
    CircuitBreaker,
    FallbackChatModel,
    FallbackEmbeddings,
    FallbackVLM,
    _is_retryable,
)


class FakeAIMessage:
    """Minimal fake AIMessage for testing."""

    def __init__(self, content: str) -> None:
        self.content = content


def test_is_retryable_home_assistant_error() -> None:
    """HomeAssistantError should be retryable."""
    assert _is_retryable(HomeAssistantError("oops")) is True


def test_is_retryable_timeout() -> None:
    """TimeoutError should be retryable."""
    assert _is_retryable(TimeoutError("timed out")) is True


def test_is_retryable_rate_limit() -> None:
    """Rate limit string should be retryable."""
    assert _is_retryable(RuntimeError("Rate limit exceeded")) is True


def test_is_retryable_non_retryable() -> None:
    """Generic ValueError should NOT be retryable."""
    assert _is_retryable(ValueError("bad input")) is False


@pytest.mark.asyncio
async def test_fallback_chat_primary_succeeds() -> None:
    """When primary succeeds, fallback is never tried."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage("primary")
    fallback = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    result = await model.ainvoke(["hello"])
    assert result.content == "primary"
    primary.ainvoke.assert_awaited_once()
    fallback.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_chat_primary_fails() -> None:
    """When primary fails with retryable error, fallback is used."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("primary down")
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    result = await model.ainvoke(["hello"])
    assert result.content == "fallback"
    assert primary.ainvoke.await_count == 1
    assert fallback.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_fallback_chat_all_fail() -> None:
    """When all providers fail, HomeAssistantError is raised."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("primary down")
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = HomeAssistantError("fallback down")

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await model.ainvoke(["hello"])


@pytest.mark.asyncio
async def test_fallback_chat_non_retryable_raises_immediately() -> None:
    """Non-retryable exceptions should not trigger fallback."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = ValueError("bad input")
    fallback = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    with pytest.raises(ValueError, match="bad input"):
        await model.ainvoke(["hello"])
    fallback.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_chat_stream_primary_succeeds() -> None:
    """Streaming: when primary succeeds, fallback is never tried."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage("primary")
    primary.astream = _fake_astream(FakeAIMessage("chunk1"))
    fallback = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    chunks = [c async for c in model.astream(["hello"])]
    assert len(chunks) == 1
    assert chunks[0].content == "chunk1"
    fallback.astream.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_chat_stream_primary_fails() -> None:
    """Streaming: when primary fails mid-stream, fallback stream is used."""
    _err = HomeAssistantError("primary down")

    async def _failing_astream(
        *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[FakeAIMessage]:
        raise _err
        yield  # type: ignore[unreachable]

    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("primary down")
    primary.astream = _failing_astream
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    fallback.astream = _fake_astream(FakeAIMessage("fb_chunk"))

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    chunks = [c async for c in model.astream(["hello"])]
    assert len(chunks) == 1
    assert chunks[0].content == "fb_chunk"


@pytest.mark.asyncio
async def test_fallback_vlm_primary_succeeds() -> None:
    """VLM fallback not used when primary succeeds."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage("vlm primary")
    fallback = AsyncMock()

    model = FallbackVLM(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await model.ainvoke(["image"])
    assert result.content == "vlm primary"
    fallback.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_vlm_primary_fails() -> None:
    """VLM fallback used when primary fails."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("vlm down")
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("vlm fallback")

    model = FallbackVLM(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await model.ainvoke(["image"])
    assert result.content == "vlm fallback"


@pytest.mark.asyncio
async def test_fallback_embeddings_primary_succeeds() -> None:
    """Embedding fallback not used when primary succeeds."""
    primary = AsyncMock()
    primary.aembed_documents.return_value = [[0.1, 0.2]]
    fallback = AsyncMock()

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await emb.aembed_documents(["hello"])
    assert result == [[0.1, 0.2]]
    fallback.aembed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_embeddings_primary_fails() -> None:
    """Embedding fallback used when primary fails."""
    primary = AsyncMock()
    primary.aembed_documents.side_effect = ConnectionError("emb down")
    fallback = AsyncMock()
    fallback.aembed_documents.return_value = [[0.3, 0.4]]

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await emb.aembed_documents(["hello"])
    assert result == [[0.3, 0.4]]


@pytest.mark.asyncio
async def test_fallback_embeddings_query() -> None:
    """Embedding query fallback works."""
    primary = AsyncMock()
    primary.aembed_query.side_effect = ConnectionError("emb down")
    fallback = AsyncMock()
    fallback.aembed_query.return_value = [0.5, 0.6]

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await emb.aembed_query("hello")
    assert result == [0.5, 0.6]


def test_circuit_breaker_opens_after_threshold() -> None:
    """Circuit breaker should disable provider after threshold failures."""
    cb = CircuitBreaker(threshold=2, window_seconds=10, cooldown_seconds=60)
    cb.record_failure("p1")
    assert cb.is_available("p1") is True
    cb.record_failure("p1")
    assert cb.is_available("p1") is False
    cb.record_failure("p1")
    assert cb.is_available("p1") is False


def test_circuit_breaker_success_resets() -> None:
    """Circuit breaker should reset on success."""
    cb = CircuitBreaker(threshold=2, window_seconds=10, cooldown_seconds=60)
    cb.record_failure("p1")
    cb.record_failure("p1")
    assert cb.is_available("p1") is False
    cb.record_success("p1")
    assert cb.is_available("p1") is True


@pytest.mark.asyncio
async def test_fallback_chat_circuit_breaker_skips_broken() -> None:
    """Circuit-broken provider should be skipped."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("primary down")
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")

    cb = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=60)
    cb.record_failure("p1")

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=cb,
    )
    result = await model.ainvoke(["hello"])
    assert result.content == "fallback"
    primary.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_chat_bind_tools() -> None:
    """bind_tools should propagate to each model in chain."""
    primary = MagicMock()
    primary.bind_tools.return_value = AsyncMock()
    fallback = MagicMock()
    fallback.bind_tools.return_value = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    bound = model.bind_tools(["tool1"])
    assert isinstance(bound, FallbackChatModel)
    assert len(bound.chain) == 2
    primary.bind_tools.assert_called_once_with(["tool1"])
    fallback.bind_tools.assert_called_once_with(["tool1"])


@pytest.mark.asyncio
async def test_fallback_chat_with_config() -> None:
    """with_config should propagate to each model and store config."""
    primary = MagicMock()
    primary.with_config.return_value = AsyncMock()
    primary.config = {"configurable": {"model": "gpt-4o"}}
    fallback = MagicMock()
    fallback.with_config.return_value = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        config={"configurable": {"temperature": 0.5}},
    )
    wrapped = model.with_config(config={"configurable": {"top_p": 1.0}})
    assert isinstance(wrapped, FallbackChatModel)
    assert len(wrapped.chain) == 2
    primary.with_config.assert_called_once()
    fallback.with_config.assert_called_once()
    assert "top_p" in wrapped.config.get("configurable", {})


def _fake_astream(
    *items: FakeAIMessage,
) -> Any:
    """Return an async generator function that yields the given items."""

    async def _gen(*_args: Any, **_kwargs: Any) -> AsyncIterator[FakeAIMessage]:
        for item in items:
            yield item

    return _gen


def test_fallback_embeddings_sync_raises() -> None:
    """Synchronous embedding methods must raise NotImplementedError."""
    primary = AsyncMock()
    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1")])
    with pytest.raises(NotImplementedError, match="Use aembed_documents"):
        emb.embed_documents(["hello"])
    with pytest.raises(NotImplementedError, match="Use aembed_query"):
        emb.embed_query("hello")
