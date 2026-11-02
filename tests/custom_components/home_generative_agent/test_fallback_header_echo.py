# ruff: noqa: S101
"""
An API key echoed by h11 never reaches a log line, an error or a traceback.

h11 rejects a header value with a stray newline or space and puts the value,
"Bearer <key>", in its error message; httpcore and httpx re-raise it with the
same text. #880 covered an SDK's APIConnectionError in a chain of two or more.
This covers the rest: a lone provider's error, which used to be re-raised
as-is (#828), and a transport error raised unwrapped, as langchain-ollama
does with httpx's.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import httpx2
import openai
import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.core import utils as core_utils
from custom_components.home_generative_agent.core.fallback import (
    FallbackChatModel,
    FallbackEmbeddings,
    FallbackVLM,
    echoes_request_headers,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_KEY = "sk-FAKE-KEY-0123456789"
# This integration's log lines. httpcore's own DEBUG trace prints the error
# too; that is third-party logging, off unless a user turns it on.
_OUR_LOGGER = "custom_components.home_generative_agent"


def _our_log(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.name.startswith(_OUR_LOGGER)
    )


@pytest.fixture
async def local_url(socket_enabled: None) -> AsyncIterator[str]:  # noqa: ARG001
    """Serve a listening socket that never answers; h11 fails before it reads."""

    async def _handle(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1/chat"
    finally:
        server.close()
        await server.wait_closed()


async def _real_header_error(url: str) -> Exception:
    """Return what httpx raises for a key ending in a newline, as h11 built it."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, headers={"Authorization": f"Bearer {_KEY}\n"})
        except httpx.LocalProtocolError as err:
            return err
    msg = "httpx accepted a header value h11 should reject"
    raise AssertionError(msg)


def _printed(err: BaseException) -> str:
    """Return the traceback a caller's LOGGER.exception would print."""
    return "".join(traceback.format_exception(err))


@pytest.mark.asyncio
async def test_the_probe_reproduces_the_echo(local_url: str) -> None:
    """Guard the fixture: the raw error does carry the key."""
    err = await _real_header_error(local_url)
    assert _KEY in _printed(err)


def _sdk_error(cause: Exception) -> Exception:
    err = openai.APIConnectionError(request=httpx2.Request("POST", "https://api"))
    err.__cause__ = cause
    return err


@pytest.mark.parametrize("wrapped", [False, True], ids=["unwrapped", "sdk"])
@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.parametrize("wrapper", [FallbackChatModel, FallbackVLM])
@pytest.mark.asyncio
async def test_a_header_echo_never_leaves_the_model_chain(
    local_url: str,
    caplog: pytest.LogCaptureFixture,
    wrapper: type[FallbackChatModel | FallbackVLM],
    members: int,
    *,
    wrapped: bool,
) -> None:
    raw = await _real_header_error(local_url)
    failure = _sdk_error(raw) if wrapped else raw
    chain: list[tuple[Any, str, str]] = []
    for i in range(members):
        provider = AsyncMock()
        provider.ainvoke.side_effect = failure
        chain.append((provider, "cloud", f"p{i}"))
    model = wrapper(chain=chain)

    # A lone provider's error is wrapped too, so no caller prints its causes.
    with caplog.at_level(logging.DEBUG), pytest.raises(HomeAssistantError) as info:
        await model.ainvoke(["hello"])

    assert _KEY not in str(info.value)
    assert _KEY not in _printed(info.value)
    assert _KEY not in _our_log(caplog)
    assert "LocalProtocolError" in _our_log(caplog)
    # The failure is still named by type.
    assert "LocalProtocolError" in str(info.value)


@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.asyncio
async def test_a_header_echo_never_leaves_the_embedding_chain(
    local_url: str, caplog: pytest.LogCaptureFixture, members: int
) -> None:
    raw = await _real_header_error(local_url)
    chain: list[tuple[Any, str, str]] = []
    for i in range(members):
        provider = AsyncMock()
        provider.aembed_query.side_effect = raw
        chain.append((provider, "cloud", f"p{i}"))
    emb = FallbackEmbeddings(chain=chain)

    with caplog.at_level(logging.DEBUG), pytest.raises(HomeAssistantError) as info:
        await emb.aembed_query("hello")

    assert _KEY not in _printed(info.value)
    assert _KEY not in _our_log(caplog)
    assert "LocalProtocolError" in _our_log(caplog)
    assert "LocalProtocolError" in str(info.value)


@pytest.mark.asyncio
async def test_a_lone_providers_other_errors_keep_their_type() -> None:
    """#828: without a header echo, a lone provider's own error is re-raised."""
    provider = AsyncMock()
    provider.ainvoke.side_effect = ValueError("bad request: max_tokens too large")
    model = FallbackChatModel(chain=[(provider, "cloud", "p0")])
    with pytest.raises(ValueError, match="max_tokens too large"):
        await model.ainvoke(["hello"])


def _wrapped_in_context(raw: Exception) -> Exception:
    """Return an error a library raised while handling `raw`, without `from`."""
    try:
        raise raw
    except type(raw):
        try:
            msg = "the client gave up"
            raise RuntimeError(msg)  # noqa: TRY301
        except RuntimeError as err:
            return err
    msg = "unreachable"
    raise AssertionError(msg)


@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.asyncio
async def test_an_error_not_retried_does_not_carry_a_header_echo_out(
    local_url: str, caplog: pytest.LogCaptureFixture, members: int
) -> None:
    """
    A failure the chain does not retry is re-raised, not wrapped.

    When it holds a header echo, here only in ``__context__`` (raised while
    handling it, without ``from``), a type-only error replaces it: graph.py
    copies the text into the chat reply and the checkpoint.
    """
    failure = _wrapped_in_context(await _real_header_error(local_url))
    assert failure.__cause__ is None
    assert _KEY in _printed(failure)
    chain: list[tuple[Any, str, str]] = []
    for i in range(members):
        provider = AsyncMock()
        provider.ainvoke.side_effect = failure
        chain.append((provider, "cloud", f"p{i}"))

    with caplog.at_level(logging.DEBUG), pytest.raises(HomeAssistantError) as info:
        await FallbackChatModel(chain=chain).ainvoke(["hello"])

    assert _KEY not in str(info.value)
    assert _KEY not in _printed(info.value)
    assert _KEY not in _our_log(caplog)
    assert "RuntimeError" in str(info.value)


@pytest.mark.asyncio
async def test_a_timeout_that_chains_a_header_echo_drops_it(local_url: str) -> None:
    """A timeout keeps its cause in a chain error, unless that echoes a header."""
    failure = TimeoutError("request timed out")
    failure.__cause__ = await _real_header_error(local_url)
    chain: list[tuple[Any, str, str]] = []
    for i in range(2):
        provider = AsyncMock()
        provider.ainvoke.side_effect = failure
        chain.append((provider, "cloud", f"p{i}"))

    with pytest.raises(HomeAssistantError) as info:
        await FallbackChatModel(chain=chain).ainvoke(["hello"])

    assert info.value.__cause__ is None
    assert _KEY not in _printed(info.value)


def test_a_chain_too_deep_to_walk_counts_as_an_echo() -> None:
    """A deep wrapper must not hide an echo by depth; past the cap, fail closed."""
    err: Exception = ValueError("innermost")
    for _ in range(40):
        outer = RuntimeError("wrapper")
        outer.__cause__ = err
        err = outer
    assert echoes_request_headers(err)
    assert not echoes_request_headers(RuntimeError("shallow"))


def test_a_cyclic_chain_ends() -> None:
    """A cause cycle does not loop forever, and holds no echo."""
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a
    assert not echoes_request_headers(a)


@pytest.mark.parametrize(
    "validator",
    [
        core_utils.validate_openai_key,
        core_utils.validate_gemini_key,
        core_utils.validate_anthropic_key,
    ],
)
@pytest.mark.asyncio
async def test_a_setup_check_logs_no_rejected_header(
    hass: Any,
    local_url: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    validator: Any,
) -> None:
    """The setup checks log their transport error by type when it is a header echo."""
    raw = await _real_header_error(local_url)
    client = AsyncMock()
    client.get.side_effect = raw
    monkeypatch.setattr(core_utils, "get_async_client", lambda _hass: client)
    with caplog.at_level(logging.DEBUG), pytest.raises(core_utils.CannotConnectError):
        await validator(hass, _KEY + "\n")
    assert "connectivity exception: LocalProtocolError" in _our_log(caplog)
    assert _KEY not in _our_log(caplog)


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (TimeoutError("timed out"), "TimeoutError: timed out"),
        (httpx.ConnectError("Connection refused"), "ConnectError: Connection refused"),
        (
            httpx.LocalProtocolError(f"Illegal header value b'Bearer {_KEY}'"),
            "LocalProtocolError",
        ),
        (ValueError(f"Bearer {_KEY}"), "ValueError"),
    ],
)
def test_safe_error_text_shows_text_only_for_network_failures(
    err: Exception, expected: str
) -> None:
    assert core_utils.safe_error_text(err) == expected
