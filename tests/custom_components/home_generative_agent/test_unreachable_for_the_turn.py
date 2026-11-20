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
from unittest.mock import AsyncMock

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
async def test_a_chain_is_never_emptied() -> None:
    """With every member unreachable, each is still tried on the next call."""
    model, primary, fallback = _chain(httpx.ConnectTimeout(""))
    fallback.ainvoke.side_effect = httpx.ConnectError("refused")
    fallback.ainvoke.return_value = None

    with skipping_unreachable_providers():
        for _call in range(2):
            with pytest.raises(HomeAssistantError):
                await model.ainvoke(["hello"])

    assert primary.ainvoke.await_count == 2
    assert fallback.ainvoke.await_count == 2


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
async def test_a_skipped_call_still_counts_toward_the_breaker() -> None:
    """Three calls of one turn opened the breaker before; they still do."""
    breaker = CircuitBreaker(threshold=3, window_seconds=60, cooldown_seconds=120)
    model, primary, _fallback = _chain(httpx.ConnectTimeout(""), breaker=breaker)

    with skipping_unreachable_providers():
        await model.ainvoke(["first"])
        await model.ainvoke(["second"])
        assert breaker.is_available("air")
        await model.ainvoke(["third"])

    assert primary.ainvoke.await_count == 1
    assert not breaker.is_available("air")
    # The breaker now holds it out of the next turn, as it did before.
    with skipping_unreachable_providers():
        await model.ainvoke(["next turn"])
    assert primary.ainvoke.await_count == 1
