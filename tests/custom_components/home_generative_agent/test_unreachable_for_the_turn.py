# ruff: noqa: S101
"""
A provider a turn cannot connect to is tried once in that turn (#1027).

A turn that uses a tool calls the model at least twice. With a sleeping
Ollama server as the primary, each call waited out the 8 s connect timeout
before the fallback answered (reported on #1022).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import httpx2
import openai
import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.core.fallback import (
    CircuitBreaker,
    FallbackChatModel,
    FallbackVLM,
    never_connected,
    skipping_unreachable_providers,
)


class _Reply:
    content = "ok"
    response_metadata: dict[str, Any] = {}  # noqa: RUF012
    tool_calls: list[Any] = []  # noqa: RUF012


def _chain(
    primary_error: Exception,
    wrapper: Any = FallbackChatModel,
    breaker: CircuitBreaker | None = None,
) -> tuple[Any, AsyncMock, AsyncMock]:
    primary, fallback = AsyncMock(), AsyncMock()
    primary.ainvoke.side_effect = primary_error
    fallback.ainvoke.return_value = _Reply()
    model = wrapper(
        chain=[(primary, "edge", "air"), (fallback, "edge", "mini")],
        circuit_breaker=breaker,
    )
    return model, primary, fallback


def _raise(err: Exception) -> Any:
    raise err


def _api_connection_error(cause: Exception) -> Exception:
    err = openai.APIConnectionError(request=httpx2.Request("POST", "https://api"))
    err.__cause__ = cause
    return err


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (httpx.ConnectTimeout(""), True),
        (httpx.ConnectError("All connection attempts failed"), True),
        (httpx2.ConnectTimeout(""), True),
        # What the Ollama client raises for a refused connection.
        (ConnectionError("Failed to connect to Ollama."), True),
        (ConnectionRefusedError(111, "Connection refused"), True),
        (_api_connection_error(httpx2.ConnectError("refused")), True),
        # The Gemini client connects through aiohttp.
        (aiohttp.ConnectionTimeoutError(), True),
        (
            aiohttp.ClientConnectorError(MagicMock(), OSError(113, "No route to host")),
            True,
        ),
        (aiohttp.ServerDisconnectedError(), False),
        (aiohttp.SocketTimeoutError(), False),
        # Connected, then went quiet or dropped: not evidence it is unreachable.
        (httpx.ReadTimeout(""), False),
        (httpx.RemoteProtocolError("Server disconnected"), False),
        (ConnectionResetError(104, "reset"), False),
        (httpx.PoolTimeout(""), False),
        (TimeoutError(), False),
        (_api_connection_error(httpx2.ReadError("dropped")), False),
        (RuntimeError("Rate limit exceeded"), False),
    ],
)
def test_only_a_connection_never_made_counts(err: Exception, *, expected: bool) -> None:
    assert never_connected(err) is expected


@pytest.mark.parametrize("wrapper", [FallbackChatModel, FallbackVLM])
@pytest.mark.asyncio
async def test_the_second_call_of_a_turn_does_not_try_it_again(wrapper: Any) -> None:
    model, primary, fallback = _chain(httpx.ConnectTimeout(""), wrapper)

    with skipping_unreachable_providers():
        await model.ainvoke(["first"])
        await model.ainvoke(["second"])
        await model.ainvoke(["third"])

    assert primary.ainvoke.await_count == 1
    assert fallback.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_the_next_turn_tries_it_again() -> None:
    model, primary, _fallback = _chain(httpx.ConnectTimeout(""))

    for _turn in range(2):
        with skipping_unreachable_providers():
            await model.ainvoke(["first"])
            await model.ainvoke(["second"])

    assert primary.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_a_call_outside_a_turn_is_unchanged() -> None:
    """Sentinel, camera analysis and summaries run outside a conversation turn."""
    model, primary, _fallback = _chain(httpx.ConnectTimeout(""))

    await model.ainvoke(["first"])
    await model.ainvoke(["second"])

    assert primary.ainvoke.await_count == 2


@pytest.mark.parametrize(
    "err",
    [httpx.ReadTimeout(""), RuntimeError("Rate limit exceeded"), TimeoutError()],
)
@pytest.mark.asyncio
async def test_a_provider_that_was_reached_is_tried_again(err: Exception) -> None:
    model, primary, _fallback = _chain(err)

    with skipping_unreachable_providers():
        await model.ainvoke(["first"])
        await model.ainvoke(["second"])

    assert primary.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_it_is_tried_last_when_the_fallback_cannot_answer() -> None:
    """A fallback that fails costs nothing new: the primary is still tried."""
    model, primary, fallback = _chain(httpx.ConnectTimeout(""))
    order: list[str] = []
    primary.ainvoke.side_effect = lambda *_a, **_k: (
        order.append("air") or _raise(httpx.ConnectTimeout(""))
    )
    fallback.ainvoke.side_effect = lambda *_a, **_k: (
        order.append("mini") or _raise(httpx.ReadTimeout(""))
    )

    with skipping_unreachable_providers():
        for _call in range(2):
            with pytest.raises(HomeAssistantError):
                await model.ainvoke(["hello"])

    assert order == ["air", "mini", "mini", "air"]


@pytest.mark.asyncio
async def test_an_answer_puts_it_back_in_its_place() -> None:
    """It woke up mid-turn and answered when the fallback failed."""
    model, primary, fallback = _chain(httpx.ConnectTimeout(""))

    with skipping_unreachable_providers():
        await model.ainvoke(["first"])
        primary.ainvoke.side_effect = None
        primary.ainvoke.return_value = _Reply()
        fallback.ainvoke.side_effect = httpx.ReadTimeout("")
        await model.ainvoke(["second"])  # mini fails, air answers
        fallback.ainvoke.reset_mock(side_effect=True)
        fallback.ainvoke.return_value = _Reply()
        await model.ainvoke(["third"])

    assert primary.ainvoke.await_count == 3
    assert fallback.ainvoke.await_count == 0


@pytest.mark.asyncio
async def test_a_lone_provider_is_tried_on_every_call() -> None:
    primary = AsyncMock()
    primary.ainvoke.side_effect = httpx.ConnectTimeout("")
    model = FallbackChatModel(chain=[(primary, "edge", "air")])

    with skipping_unreachable_providers():
        for _call in range(2):
            with pytest.raises(httpx.ConnectTimeout):
                await model.ainvoke(["hello"])

    assert primary.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_another_chain_of_the_turn_skips_it_only_with_a_fallback() -> None:
    """The summary model may use the same server, with or without a fallback."""
    chat, chat_primary, _ = _chain(httpx.ConnectTimeout(""))
    summary, summary_primary, summary_fallback = _chain(httpx.ConnectTimeout(""))
    lone = AsyncMock()
    lone.ainvoke.side_effect = httpx.ConnectTimeout("")
    alone = FallbackChatModel(chain=[(lone, "edge", "air")])

    with skipping_unreachable_providers():
        await chat.ainvoke(["hello"])
        await summary.ainvoke(["summarize"])
        with pytest.raises(httpx.ConnectTimeout):
            await alone.ainvoke(["summarize"])

    assert chat_primary.ainvoke.await_count == 1
    assert summary_primary.ainvoke.await_count == 0
    assert summary_fallback.ainvoke.await_count == 1
    assert lone.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_calls_made_in_tasks_of_the_turn_share_what_was_learned() -> None:
    """LangGraph runs each node in a task, which copies the context."""
    model, primary, fallback = _chain(httpx.ConnectTimeout(""))

    with skipping_unreachable_providers():
        await asyncio.create_task(model.ainvoke(["first"]))
        await asyncio.create_task(model.ainvoke(["second"]))

    assert primary.ainvoke.await_count == 1
    assert fallback.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_two_turns_at_once_do_not_share_it() -> None:
    model, primary, _fallback = _chain(httpx.ConnectTimeout(""))

    async def _turn() -> None:
        with skipping_unreachable_providers():
            await model.ainvoke(["first"])
            await model.ainvoke(["second"])

    await asyncio.gather(_turn(), _turn())

    assert primary.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_a_call_not_made_is_not_counted_by_any_breaker() -> None:
    """One failed connection is one failure, for the chain that saw it."""
    chat_breaker = CircuitBreaker(threshold=3, window_seconds=60, cooldown_seconds=120)
    vlm_breaker = CircuitBreaker(threshold=1, window_seconds=60, cooldown_seconds=120)
    chat, primary, _ = _chain(httpx.ConnectTimeout(""), breaker=chat_breaker)
    vlm, vlm_primary, _ = _chain(httpx.ConnectTimeout(""), FallbackVLM, vlm_breaker)

    with skipping_unreachable_providers():
        for _call in range(4):
            await chat.ainvoke(["hello"])
            await vlm.ainvoke(["frame"])

    assert primary.ainvoke.await_count == 1
    assert vlm_primary.ainvoke.await_count == 0
    assert chat_breaker.is_available("air")
    assert vlm_breaker.is_available("air")


@pytest.mark.asyncio
async def test_a_task_that_outlives_the_turn_tries_it_again() -> None:
    """A camera worker started by a tool's state change keeps the context."""
    model, primary, _fallback = _chain(httpx.ConnectTimeout(""))
    turn_over = asyncio.Event()

    async def _worker() -> None:
        await turn_over.wait()
        await model.ainvoke(["frame"])
        await model.ainvoke(["frame"])

    with skipping_unreachable_providers():
        await model.ainvoke(["first"])
        worker = asyncio.create_task(_worker())
    turn_over.set()
    await worker

    assert primary.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_half_open_tries_keep_the_order_the_breaker_chose() -> None:
    """With every breaker open, the primary is still the first one tried (#881)."""
    breaker = CircuitBreaker(
        threshold=1, window_seconds=60, cooldown_seconds=120, probe_interval_seconds=0
    )
    model, primary, fallback = _chain(httpx.ConnectTimeout(""), breaker=breaker)
    fallback.ainvoke.side_effect = httpx.ReadTimeout("")
    order: list[str] = []

    with skipping_unreachable_providers():
        with pytest.raises(HomeAssistantError):
            await model.ainvoke(["first"])  # opens both breakers
        primary.ainvoke.side_effect = lambda *_a, **_k: order.append("air") or _Reply()
        fallback.ainvoke.side_effect = lambda *_a, **_k: (
            order.append("mini") or _Reply()
        )
        await model.ainvoke(["second"])

    assert order == ["air"]
