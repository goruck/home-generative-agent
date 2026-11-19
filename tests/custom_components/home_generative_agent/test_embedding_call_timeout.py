# ruff: noqa: S101
"""
An embedding provider that never answers gives up, and the next one is tried (#969).

A server whose process is frozen (`kill -STOP`, a host swapping hard, a
suspended VM) still completes the TCP handshake: the kernel accepts the
connection into its backlog, and nothing is ever read or written. The Ollama
client limits only the connect, so an embedding call to it waited forever, and
with it the conversation turn searching its tools and memories.

A provider that timed out is then skipped for a while, so a turn's later
searches, and the next turns, do not each wait it out again.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from langchain_ollama import OllamaEmbeddings

import custom_components.home_generative_agent as hga_component
import custom_components.home_generative_agent.core.fallback as fallback_module
from custom_components.home_generative_agent.agent.graph import _search_memories
from custom_components.home_generative_agent.core.fallback import (
    CircuitBreaker,
    EmbeddingsTimedOutError,
    EmbeddingsUnavailableError,
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
    client_kwargs = cast("Any", hga_component)._ollama_httpx_client_kwargs
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=url,
        sync_client_kwargs=client_kwargs(),
        async_client_kwargs=client_kwargs(),
    )


def _answering(vector: list[float]) -> MagicMock:
    model = MagicMock()
    model.aembed_query = AsyncMock(return_value=vector)
    model.aembed_documents = AsyncMock(return_value=[vector])
    return model


class _Frozen:
    """An in-process provider that never answers, and says when it was called."""

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()

    async def aembed_query(self, _text: str) -> list[float]:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError  # pragma: no cover - never reached

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(text) for text in texts]


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_lone_frozen_provider_fails_the_call(frozen_server: str) -> None:
    emb = FallbackEmbeddings([(_frozen_ollama(frozen_server), "edge", "p1")])

    async with asyncio.timeout(_OUTER_LIMIT_S):
        with pytest.raises(EmbeddingsTimedOutError, match="p1 timed out"):
            await emb.aembed_query("turn on the porch light")


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "arg", "expected"),
    [
        ("aembed_query", "porch light", [0.1, 0.2]),
        ("aembed_documents", ["porch light"], [[0.1, 0.2]]),
    ],
)
async def test_a_frozen_primary_hands_the_call_to_the_fallback(
    frozen_server: str, method: str, arg: object, expected: object
) -> None:
    breaker = CircuitBreaker()
    fallback = _answering([0.1, 0.2])
    emb = FallbackEmbeddings(
        [(_frozen_ollama(frozen_server), "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )

    async with asyncio.timeout(_OUTER_LIMIT_S):
        result = await getattr(emb, method)(arg)

    assert result == expected
    getattr(fallback, method).assert_awaited_once()
    assert breaker._failures["p1"], "the timeout counts toward the breaker"


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_lone_provider_that_timed_out_is_skipped_meanwhile() -> None:
    """A turn's later searches fail at once instead of each waiting it out."""
    frozen = _Frozen()
    emb = FallbackEmbeddings([(frozen, "edge", "p1")])

    with pytest.raises(EmbeddingsTimedOutError):
        await emb.aembed_query("turn on the porch light")
    async with asyncio.timeout(0.2):  # Well under the provider's 0.5 s.
        with pytest.raises(EmbeddingsTimedOutError, match="p1 timed out"):
            await emb.aembed_query("and close the garage door")

    assert frozen.calls == 1


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_timed_out_primary_is_skipped_for_the_fallback() -> None:
    """The breaker needs three failures in 60 s; timeouts 30 s apart never get there."""
    frozen = _Frozen()
    fallback = _answering([0.3])
    emb = FallbackEmbeddings(
        [(frozen, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=CircuitBreaker(),
    )

    assert await emb.aembed_query("porch light") == [0.3]
    async with asyncio.timeout(0.2):
        assert await emb.aembed_query("garage door") == [0.3]

    assert frozen.calls == 1


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_timed_out_provider_is_tried_again_after_the_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _Frozen()
    emb = FallbackEmbeddings([(frozen, "edge", "p1")])
    now = [1000.0]
    monkeypatch.setattr(fallback_module, "monotonic", lambda: now[0])

    with pytest.raises(EmbeddingsTimedOutError):
        await emb.aembed_query("porch light")
    now[0] += fallback_module.EMBEDDING_TIMEOUT_HOLD_S + 1
    with pytest.raises(EmbeddingsTimedOutError):
        await emb.aembed_query("porch light")

    assert frozen.calls == 2


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_another_failure_is_not_reported_as_a_timeout() -> None:
    """A timeout beside a provider that failed another way is not 'unavailable'."""
    broken = MagicMock()
    broken.aembed_query = AsyncMock(side_effect=ConnectionError("refused"))
    emb = FallbackEmbeddings(
        [(_Frozen(), "edge", "p1"), (broken, "cloud", "p2")],
        circuit_breaker=CircuitBreaker(),
    )

    with pytest.raises(HomeAssistantError) as caught:
        await emb.aembed_query("porch light")

    assert not isinstance(caught.value, EmbeddingsUnavailableError)


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


@pytest.mark.asyncio
async def test_a_cancel_the_client_reports_as_its_own_error_is_a_timeout() -> None:
    """Some clients turn the cancel into a read error (#985)."""

    async def _converts_the_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            msg = "connection reset"
            raise OSError(msg) from None

    with pytest.raises(TimeoutError, match="No answer within"):
        await fallback_module._within(_converts_the_cancel(), 0.05)


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_stopped_caller_is_not_taken_for_a_frozen_provider() -> None:
    """The caller's cancel stops the call; the next provider is not tried."""
    frozen = _Frozen()
    fallback = _answering([0.1])
    emb = FallbackEmbeddings([(frozen, "edge", "p1"), (fallback, "cloud", "p2")])
    task = asyncio.create_task(emb.aembed_query("porch light"))
    await frozen.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    fallback.aembed_query.assert_not_called()
    assert not emb._timed_out("p1"), "a stop does not hold the provider"


@pytest.mark.asyncio
async def test_memory_search_uses_recent_memories_after_a_timeout() -> None:
    recent = [MagicMock()]

    async def _asearch(_namespace: tuple[str, ...], **kwargs: object) -> list:
        if kwargs.get("query"):
            msg = "No embedding provider answered in time"
            raise EmbeddingsTimedOutError(msg)
        return recent

    store = MagicMock()
    store.asearch = AsyncMock(side_effect=_asearch)

    assert await _search_memories(store, "user-1", "how do I take my coffee") == (
        recent
    )


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_a_chain_of_timeouts_longer_than_the_hold_is_still_timed_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first member's hold can run out before the last one times out."""
    monkeypatch.setattr(fallback_module, "EMBEDDING_TIMEOUT_HOLD_S", 0.01)
    emb = FallbackEmbeddings(
        [(_Frozen(), "edge", "p1"), (_Frozen(), "edge", "p2")],
        circuit_breaker=CircuitBreaker(),
    )

    with pytest.raises(EmbeddingsTimedOutError, match="p1, p2 timed out"):
        await emb.aembed_query("porch light")


def test_a_timed_out_member_does_not_keep_the_others_from_a_half_open_try() -> None:
    """Held after a timeout, it is left out like a member held for its build."""
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure("p2")
    emb = FallbackEmbeddings(
        [(_Frozen(), "edge", "p1"), (_answering([0.1]), "cloud", "p2")],
        circuit_breaker=breaker,
    )
    emb._timed_out_until["p1"] = fallback_module.monotonic() + 60

    assert emb._probe_candidates(emb._ordered_chain()) == ["p2"]


def test_removing_a_member_drops_its_hold() -> None:
    emb = FallbackEmbeddings(
        [(_Frozen(), "edge", "p1"), (_answering([0.1]), "cloud", "p2")]
    )
    emb._timed_out_until["p1"] = fallback_module.monotonic() + 60

    emb.remove_provider("p1")

    assert "p1" not in emb._timed_out_until


def test_marking_the_tool_index_stale_drops_the_retry_wait() -> None:
    """Another embedder answers now: its rebuild does not wait (#969)."""
    from custom_components.home_generative_agent.core.utils import (  # noqa: PLC0415
        mark_tool_index_stale,
    )

    rd = MagicMock()
    rd.tool_content_hashes = {}
    rd.tool_index_task = None
    rd.tool_index_retry_after = 1e12

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "custom_components.home_generative_agent.core.utils.async_dispatcher_send",
            MagicMock(),
        )
        mark_tool_index_stale(MagicMock(), rd)

    assert rd.tool_index_retry_after == 0.0


@pytest.mark.usefixtures("short_timeout")
@pytest.mark.asyncio
async def test_each_member_is_held_from_the_end_of_an_all_timed_out_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first member's hold does not run out before the call ends."""
    now = [1000.0]
    monkeypatch.setattr(fallback_module, "monotonic", lambda: now[0])
    emb = FallbackEmbeddings(
        [(_Frozen(), "edge", "p1"), (_Frozen(), "edge", "p2")],
        circuit_breaker=CircuitBreaker(),
    )
    real_within = fallback_module._within

    async def _ticking(call: object, timeout_s: float) -> object:
        now[0] += 70.0  # Longer than a hold: p1's own would run out.
        return await real_within(call, timeout_s)  # type: ignore[arg-type]

    monkeypatch.setattr(fallback_module, "_within", _ticking)

    with pytest.raises(EmbeddingsTimedOutError):
        await emb.aembed_query("porch light")

    assert emb._timed_out("p1"), "held from the end, not from its own timeout"
