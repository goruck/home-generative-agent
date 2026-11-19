# ruff: noqa: S101
"""
An embedding provider that never answers gives up, and the next one is tried (#969).

A server whose process is frozen (`kill -STOP`, a host swapping hard, a
suspended VM) still completes the TCP handshake: the kernel accepts the
connection into its backlog, and nothing is ever read or written. The Ollama
client limits only the connect, so an embedding call to it waited forever, and
with it the conversation turn searching its tools and memories.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from langchain_ollama import OllamaEmbeddings

import custom_components.home_generative_agent.core.fallback as fallback_module
from custom_components.home_generative_agent import _ollama_httpx_client_kwargs
from custom_components.home_generative_agent.core.fallback import (
    CircuitBreaker,
    FallbackEmbeddings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# The test's own limit: the call must end well before it.
_OUTER_LIMIT_S = 10.0


@pytest.fixture
def frozen_server(socket_enabled: None) -> Iterator[str]:  # noqa: ARG001
    """Return the URL of a server that takes connections and never answers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)  # Never accepted: what a stopped process looks like.
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
    finally:
        server.close()


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fallback_module, "EMBEDDING_CALL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(fallback_module, "EMBEDDING_TIMEOUT_PER_TEXT_S", 0.0)


def _frozen_ollama(url: str) -> OllamaEmbeddings:
    """Build Ollama embeddings the way setup does."""
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=url,
        sync_client_kwargs=_ollama_httpx_client_kwargs(),
        async_client_kwargs=_ollama_httpx_client_kwargs(),
    )


def _answering(vector: list[float]) -> MagicMock:
    model = MagicMock()
    model.aembed_query = AsyncMock(return_value=vector)
    model.aembed_documents = AsyncMock(return_value=[vector])
    return model


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_lone_frozen_provider_fails_the_call(frozen_server: str) -> None:
    emb = FallbackEmbeddings([(_frozen_ollama(frozen_server), "edge", "p1")])

    async with asyncio.timeout(_OUTER_LIMIT_S):
        with pytest.raises(HomeAssistantError, match="No answer within"):
            await emb.aembed_query("turn on the porch light")


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["aembed_query", "aembed_documents"])
async def test_a_frozen_primary_hands_the_call_to_the_fallback(
    frozen_server: str, method: str
) -> None:
    breaker = CircuitBreaker()
    fallback = _answering([0.1, 0.2])
    emb = FallbackEmbeddings(
        [(_frozen_ollama(frozen_server), "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    arg = "porch light" if method == "aembed_query" else ["porch light"]

    async with asyncio.timeout(_OUTER_LIMIT_S):
        result = await getattr(emb, method)(arg)

    assert result in ([0.1, 0.2], [[0.1, 0.2]])
    assert breaker._failures["p1"], "the timeout counts toward the breaker"


@pytest.mark.asyncio
async def test_more_texts_get_more_time(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def _record(call: object, timeout_s: float) -> object:
        waits.append(timeout_s)
        return await call  # type: ignore[misc]

    monkeypatch.setattr(fallback_module, "_within", _record)
    emb = FallbackEmbeddings([(_answering([0.1]), "edge", "p1")])

    await emb.aembed_query("one")
    await emb.aembed_documents(["text"] * 200)

    assert waits[0] < fallback_module.EMBEDDING_CALL_TIMEOUT_S + 1
    assert waits[1] > waits[0] + 40


@pytest.mark.asyncio
async def test_the_providers_own_timeout_keeps_its_cause() -> None:
    cause = OSError("read timed out")

    async def _times_out() -> None:
        msg = "provider timeout"
        raise TimeoutError(msg) from cause

    with pytest.raises(TimeoutError, match="provider timeout") as caught:
        await fallback_module._within(_times_out(), 5.0)
    assert caught.value.__cause__ is cause


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_stopped_caller_is_not_taken_for_a_frozen_provider(
    frozen_server: str,
) -> None:
    """The caller's cancel stops the call; the next provider is not tried."""
    fallback = _answering([0.1])
    emb = FallbackEmbeddings(
        [(_frozen_ollama(frozen_server), "edge", "p1"), (fallback, "cloud", "p2")]
    )
    task = asyncio.create_task(emb.aembed_query("porch light"))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    fallback.aembed_query.assert_not_called()
