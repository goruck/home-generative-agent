# ruff: noqa: S101
"""Tests for core fallback wrappers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import openai
import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.const import EMBEDDING_MODEL_DIMS
from custom_components.home_generative_agent.core.fallback import (
    DROPPABLE_SAMPLING_PARAMS,
    CircuitBreaker,
    EmbeddingsUnavailableError,
    FallbackChatModel,
    FallbackEmbeddings,
    FallbackVLM,
    _is_retryable,
    ainvoke_dropping_unsupported_params,
    invoke_dropping_unsupported_params,
    is_model_not_found,
    unsupported_sampling_param,
    unsupported_sampling_param_in_chain,
)


class FakeAIMessage:
    """Minimal fake AIMessage for testing."""

    def __init__(
        self,
        content: str,
        *,
        response_metadata: dict[str, Any] | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.response_metadata = response_metadata or {}
        self.tool_calls = tool_calls or []


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


def test_is_retryable_httpx_connect_error() -> None:
    """httpx.ConnectError (Ollama network failure) should be retryable."""
    assert _is_retryable(httpx.ConnectError("All connection attempts failed")) is True


def test_is_retryable_httpx2_connect_error() -> None:
    """The SDKs on httpx2 (openai 3, anthropic 2) fail over the same way (#878)."""
    assert _is_retryable(httpx2.ConnectError("All connection attempts failed")) is True


def _sdk_connection_error(sdk: Any, *, cause: bool = True) -> Exception:
    """Return the SDK's 'Connection error.' as its client raises it (#880)."""
    err = sdk.APIConnectionError(request=httpx2.Request("POST", "https://api"))
    if cause:
        err.__cause__ = httpx2.ConnectError("[Errno 111] Connection refused")
    return err


@pytest.mark.parametrize("cause", [True, False], ids=["with_cause", "bare"])
@pytest.mark.parametrize("sdk_name", ["openai", "anthropic"])
def test_is_retryable_sdk_connection_error(sdk_name: str, *, cause: bool) -> None:
    """A refused connection to OpenAI or Anthropic moves the chain on (#880)."""
    sdk = pytest.importorskip(sdk_name)
    assert _is_retryable(_sdk_connection_error(sdk, cause=cause)) is True


def test_is_retryable_wrapped_transport_error() -> None:
    """A provider wrapper that keeps the transport error as its cause fails over."""
    err = RuntimeError("model call failed")
    err.__cause__ = httpx2.ConnectError("[Errno 111] Connection refused")
    assert _is_retryable(err) is True


def _bad_input_raised_in_a_connection_error_handler() -> Exception:
    """Return a ValueError raised while a connection error was handled."""
    try:
        try:
            raise _sdk_connection_error(openai)
        except openai.APIConnectionError:
            msg = "bad input"
            raise ValueError(msg) from None
    except ValueError as err:
        return err


def test_is_retryable_connection_error_only_as_context() -> None:
    """A connection error that is only context, not the cause, does not count."""
    err = _bad_input_raised_in_a_connection_error_handler()
    assert err.__context__ is not None
    assert _is_retryable(err) is False


@pytest.mark.asyncio
async def test_fallback_chat_sdk_connection_error_uses_fallback() -> None:
    """The chain answers from the fallback when the primary refuses (#880)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _sdk_connection_error(openai)
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")

    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")]
    )
    result = await model.ainvoke(["hello"])
    assert result.content == "fallback"
    assert fallback.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_fallback_chat_sdk_connection_error_counts_for_the_breaker() -> None:
    """A refused connection is a failure the circuit breaker counts (#880)."""
    import anthropic  # noqa: PLC0415

    breaker = CircuitBreaker(threshold=1)
    primary = AsyncMock()
    primary.ainvoke.side_effect = _sdk_connection_error(anthropic)
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")

    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=breaker,
    )
    await model.ainvoke(["hello"])
    assert not breaker.is_available("p1")
    assert breaker.is_available("p2")


@pytest.mark.asyncio
async def test_fallback_vlm_sdk_connection_error_uses_fallback() -> None:
    """A camera frame goes to the fallback when the primary refuses (#880)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _sdk_connection_error(openai)
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("vlm fallback")

    model = FallbackVLM(chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")])
    result = await model.ainvoke(["image"])
    assert result.content == "vlm fallback"


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_sdk_connection_error_uses_fallback(
    method: str,
) -> None:
    """An embedding call goes to the fallback when the primary refuses (#880)."""
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).side_effect = _sdk_connection_error(openai)
    getattr(fallback, method).return_value = [[0.3]]

    emb = FallbackEmbeddings(
        chain=[(primary, "cloud", "p1"), (fallback, "cloud", "p2")]
    )
    assert await _call(emb, method) == [[0.3]]


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_sdk_connection_error_keeps_a_held_fallback_out(
    method: str,
) -> None:
    """A refused primary never lets in a fallback held until its build is checked."""
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).side_effect = _sdk_connection_error(openai)
    getattr(fallback, method).return_value = [[0.3]]

    emb = FallbackEmbeddings(chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")])
    emb.hold_until_verified(["p2"], AsyncMock(return_value=False))
    with pytest.raises(EmbeddingsUnavailableError):
        await _call(emb, method)
    getattr(fallback, method).assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_chat_sdk_connection_error_logs_its_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK says only 'Connection error.', so the log names the cause (#880)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _sdk_connection_error(openai)
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")

    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")]
    )
    await model.ainvoke(["hello"])
    assert "APIConnectionError: Connection error. (cause: ConnectError)" in caplog.text


@pytest.mark.parametrize("cause", [True, False], ids=["with_cause", "bare"])
@pytest.mark.asyncio
async def test_all_providers_refused_message_names_the_cause_type(
    *, cause: bool
) -> None:
    """The error a user sees names the cause's type, never its text."""
    p1, p2 = AsyncMock(), AsyncMock()
    p1.ainvoke.side_effect = _sdk_connection_error(openai, cause=cause)
    p2.ainvoke.side_effect = _sdk_connection_error(openai, cause=cause)
    model = FallbackChatModel(chain=[(p1, "cloud", "p1"), (p2, "edge", "p2")])
    with pytest.raises(HomeAssistantError) as info:
        await model.ainvoke(["hello"])
    text = str(info.value)
    assert "APIConnectionError: Connection error." in text
    assert ("(cause: ConnectError)" in text) is cause
    assert "Errno" not in text


@pytest.mark.asyncio
async def test_connection_error_cause_text_never_reaches_logs_or_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A key in a rejected header (h11 echoes it) is never logged or shown (#880)."""
    leaked = "sk-FAKE-KEY-0123456789"
    err = openai.APIConnectionError(request=httpx2.Request("POST", "https://api"))
    err.__cause__ = httpx2.LocalProtocolError(
        f"Illegal header value b'Bearer {leaked}\\n'"
    )
    p1, p2 = AsyncMock(), AsyncMock()
    p1.ainvoke.side_effect = err
    p2.ainvoke.side_effect = err
    model = FallbackChatModel(chain=[(p1, "cloud", "p1"), (p2, "edge", "p2")])
    with pytest.raises(HomeAssistantError) as info:
        await model.ainvoke(["hello"])
    assert "(cause: LocalProtocolError)" in str(info.value)
    assert leaked not in str(info.value)
    assert leaked not in caplog.text


_LEAKED_KEY = "sk-FAKE-KEY-0123456789"


def _key_in_header_error() -> Exception:
    """Return the SDK error for a key h11 rejected, its value in the cause."""
    err = openai.APIConnectionError(request=httpx2.Request("POST", "https://api"))
    err.__cause__ = httpx2.LocalProtocolError(
        f"Illegal header value b'Bearer {_LEAKED_KEY}\\n'"
    )
    return err


async def _chain_failure(kind: str) -> BaseException:
    """Return what a two-member chain of `kind` raises when both refuse."""
    p1, p2 = AsyncMock(), AsyncMock()
    if kind == "embeddings":
        p1.aembed_query.side_effect = _key_in_header_error()
        p2.aembed_query.side_effect = _key_in_header_error()
        emb = FallbackEmbeddings(chain=[(p1, "cloud", "p1"), (p2, "cloud", "p2")])
        call = emb.aembed_query("hello")
    else:
        p1.ainvoke.side_effect = _key_in_header_error()
        p2.ainvoke.side_effect = _key_in_header_error()
        wrapper = FallbackVLM if kind == "vlm" else FallbackChatModel
        model = wrapper(chain=[(p1, "cloud", "p1"), (p2, "edge", "p2")])
        call = model.ainvoke(["hello"])
    with pytest.raises(HomeAssistantError) as info:
        await call
    return info.value


@pytest.mark.parametrize("kind", ["chat", "vlm", "embeddings"])
@pytest.mark.asyncio
async def test_a_logged_traceback_of_a_refused_chain_holds_no_key(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A caller's LOGGER.exception prints the cause chain: it must not hold the key."""
    err = await _chain_failure(kind)
    logger = logging.getLogger("test.caller")
    with caplog.at_level(logging.DEBUG):
        try:
            raise err
        except HomeAssistantError:
            logger.exception("model call failed")
    assert "LocalProtocolError" in caplog.text
    assert _LEAKED_KEY not in caplog.text


@pytest.mark.asyncio
async def test_transport_error_summary_does_not_repeat_itself(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ollama's own transport error already says what failed: no cause suffix."""
    inner = httpx2.ConnectError("All connection attempts failed")
    outer = httpx2.ConnectError("All connection attempts failed")
    outer.__cause__ = inner
    primary = AsyncMock()
    primary.ainvoke.side_effect = outer
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )
    await model.ainvoke(["hello"])
    assert "ConnectError: All connection attempts failed" in caplog.text
    assert "(cause:" not in caplog.text


@pytest.mark.asyncio
async def test_fallback_chat_sdk_connection_error_is_not_stall_evidence() -> None:
    """A refused connection fails over but never marks the provider stalled."""
    tracker = MagicMock()
    tracker.is_stalled.return_value = False
    primary = AsyncMock()
    primary.ainvoke.side_effect = _sdk_connection_error(openai)
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "edge", "p2")],
        stall_tracker=tracker,
    )
    with patch(
        "custom_components.home_generative_agent.core.fallback.counts_as_stall_evidence",
        return_value=True,
    ):
        result = await model.ainvoke(["hello"])
    assert result.content == "fallback"
    tracker.record_timeout.assert_not_called()


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
async def test_fallback_chat_primary_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
    assert "Model call failed for provider p1 (provider 1/2)" in caplog.text
    assert "(attempt 1/2)" not in caplog.text


@pytest.mark.asyncio
async def test_fallback_chat_primary_empty_length_response() -> None:
    """Empty length-limited responses should trigger fallback."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage(
        "",
        response_metadata={"done_reason": "length", "model_name": "gpt-oss"},
    )
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
async def test_fallback_chat_empty_tool_call_response_does_not_fallback() -> None:
    """Empty tool-call responses are valid and should not trigger fallback."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage(
        "",
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[{"name": "tool"}],
    )
    fallback = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")]
    )

    result = await model.ainvoke(["hello"])

    assert cast("FakeAIMessage", result).tool_calls == [{"name": "tool"}]
    fallback.ainvoke.assert_not_awaited()


def test_fallback_chat_with_config_preserves_bound_model_config() -> None:
    """Per-call fallback config preserves the configured provider model."""
    primary = MagicMock()
    primary.config = {
        "configurable": {
            "model": "qwen-summary",
            "temperature": 0.2,
            "num_ctx": 32000,
        }
    }
    primary.with_config.return_value = MagicMock()

    model = FallbackChatModel(chain=[(primary, "edge", "p1")])
    configured = model.with_config(
        config={"configurable": {"num_predict": 128, "reasoning": False}}
    )

    assert isinstance(configured, FallbackChatModel)
    call_config = primary.with_config.call_args.args[0]
    assert call_config["configurable"] == {
        "model": "qwen-summary",
        "temperature": 0.2,
        "num_ctx": 32000,
        "num_predict": 128,
        "reasoning": False,
    }


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
async def test_fallback_chat_astream_raises_not_implemented() -> None:
    """Astream is disabled; callers must use ainvoke."""
    primary = AsyncMock()
    model = FallbackChatModel(chain=[(primary, "edge", "p1")])
    with pytest.raises(NotImplementedError, match="Use ainvoke"):
        async for _ in model.astream(["hello"]):
            pass


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
async def test_fallback_vlm_primary_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """VLM fallback used when primary fails."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("vlm down")
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("vlm fallback")

    model = FallbackVLM(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    result = await model.ainvoke(["image"])
    assert result.content == "vlm fallback"
    assert "VLM call failed for provider p1 (provider 1/2)" in caplog.text
    assert "(attempt 1/2)" not in caplog.text


@pytest.mark.asyncio
async def test_fallback_vlm_alert_callback_includes_fallback_deployment() -> None:
    """VLM fallback alert callback receives provider and deployment details."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("vlm down")
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("vlm fallback")
    alert_callback = MagicMock()

    model = FallbackVLM(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        alert_callback=alert_callback,
    )

    assert (await model.ainvoke(["image"])).content == "vlm fallback"
    alert_callback.assert_called_once()
    assert alert_callback.call_args.args[:2] == ("p1", primary.ainvoke.side_effect)
    assert alert_callback.call_args.kwargs == {
        "fallback_to": "p2",
        "fallback_deployment": "cloud",
    }


def test_fallback_vlm_with_config_preserves_bound_model_config() -> None:
    """Video per-call VLM config preserves the configured provider model."""
    primary = MagicMock()
    primary.config = {
        "configurable": {
            "model": "qwen-vlm",
            "temperature": 0.1,
            "num_ctx": 32000,
        }
    }
    primary.with_config.return_value = MagicMock()

    model = FallbackVLM(chain=[(primary, "edge", "p1")])
    configured = model.with_config(
        config={"configurable": {"num_predict": 256, "reasoning": False}}
    )

    assert isinstance(configured, FallbackVLM)
    call_config = primary.with_config.call_args.args[0]
    assert call_config["configurable"] == {
        "model": "qwen-vlm",
        "temperature": 0.1,
        "num_ctx": 32000,
        "num_predict": 256,
        "reasoning": False,
    }


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
async def test_fallback_embeddings_primary_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Embedding fallback used when primary fails."""
    primary = AsyncMock()
    primary.aembed_documents.side_effect = ConnectionError("emb down")
    fallback = AsyncMock()
    fallback.aembed_documents.return_value = [[0.3, 0.4]]

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    caplog.set_level(logging.WARNING)
    result = await emb.aembed_documents(["hello"])
    assert result == [[0.3, 0.4]]
    assert (
        "Embedding fallback activated: provider p2 succeeded after failed "
        "provider(s): p1"
    ) in caplog.text
    assert "Embedding provider switched from p1 to p2" in caplog.text


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


def test_fallback_embeddings_remove_provider() -> None:
    """A member is dropped by its provider id; an unknown id changes nothing."""
    primary, fallback = AsyncMock(), AsyncMock()
    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")])

    assert emb.remove_provider("p2")
    assert emb.chain == [(primary, "edge", "p1")]
    assert not emb.remove_provider("p2")
    assert emb.chain == [(primary, "edge", "p1")]


@pytest.mark.asyncio
async def test_fallback_embeddings_removing_the_active_provider_hands_back() -> None:
    """The sticky fallback removed: the primary is first again, with no switch."""
    primary, fallback = AsyncMock(), AsyncMock()
    primary.aembed_documents.side_effect = [ConnectionError("emb down"), [[0.1]]]
    fallback.aembed_documents.return_value = [[0.9]]
    switch_callback, removed_callback = MagicMock(), MagicMock()
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        switch_callback=switch_callback,
        removed_callback=removed_callback,
    )
    assert await emb.aembed_documents(["hello"]) == [[0.9]]
    switch_callback.assert_called_once_with("p1", "p2")
    assert "p2" in emb.served

    assert emb.remove_provider("p2")

    # A "fell back" notice would be wrong: nothing failed, the chain went back.
    removed_callback.assert_called_once_with("p2", "p1")
    switch_callback.assert_called_once()
    assert await emb.aembed_documents(["hello"]) == [[0.1]]
    assert fallback.aembed_documents.await_count == 1
    switch_callback.assert_called_once()


def _call(emb: FallbackEmbeddings, method: str) -> Any:
    """Start an embed call of either kind; the store saves through documents."""
    if method == "aembed_documents":
        return asyncio.create_task(emb.aembed_documents(["hello"]))
    return asyncio.create_task(emb.aembed_query("hello"))


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_a_call_in_flight_skips_a_removed_provider(
    method: str,
) -> None:
    """Removed while the primary was still answering: the fallback is not tried."""
    gate = asyncio.Event()
    primary, fallback = AsyncMock(), AsyncMock()

    async def _slow_failure(*_args: Any) -> Any:
        await gate.wait()
        msg = "emb down"
        raise ConnectionError(msg)

    getattr(primary, method).side_effect = _slow_failure
    getattr(fallback, method).return_value = [[0.9]]
    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")])
    call = _call(emb, method)
    await asyncio.sleep(0)

    assert emb.remove_provider("p2")
    gate.set()

    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await call
    getattr(fallback, method).assert_not_awaited()
    assert emb._active_provider_id == "p1"


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_an_answer_from_a_removed_provider_is_dropped(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Removed while it answered: its vectors are another build's, so not used."""
    gate = asyncio.Event()
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).side_effect = ConnectionError("emb down")

    async def _slow_answer(*_args: Any) -> Any:
        await gate.wait()
        return [[0.9]]

    getattr(fallback, method).side_effect = _slow_answer
    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")])
    call = _call(emb, method)
    await asyncio.sleep(0)

    assert emb.remove_provider("p2")
    gate.set()

    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await call
    assert "removed from the embedding chain while it answered" in caplog.text
    assert emb._active_provider_id == "p1"
    assert "p2" not in emb.served


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_a_discarded_answer_is_not_called_a_failure(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The next success does not list the removed provider as one that failed."""
    gate = asyncio.Event()
    primary, fallback, third = AsyncMock(), AsyncMock(), AsyncMock()
    getattr(primary, method).side_effect = ConnectionError("emb down")

    async def _slow_answer(*_args: Any) -> Any:
        await gate.wait()
        return [[0.9]]

    getattr(fallback, method).side_effect = _slow_answer
    getattr(third, method).return_value = [[0.7]]
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2"), (third, "edge", "p3")]
    )
    call = _call(emb, method)
    await asyncio.sleep(0)
    assert emb.remove_provider("p2")
    gate.set()

    assert await call == [[0.7]]
    assert "after failed provider(s): p1" in caplog.text
    assert "p2" not in caplog.text.split("after failed provider(s):")[1].splitlines()[0]


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_fallback_embeddings_a_lone_member_ignores_its_open_breaker(
    method: str,
) -> None:
    """Down to one provider, the chain is the unwrapped provider setup builds."""
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).return_value = [[0.1]]
    breaker = MagicMock()
    breaker.is_available.return_value = False
    # Tried a moment ago: no half-open try this call (#881).
    breaker.due_for_probe.return_value = []
    breaker.next_probe_in.return_value = 5.0
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=breaker,
    )
    with pytest.raises(HomeAssistantError, match="circuit breaker is open for p1, p2"):
        await _call(emb, method)

    assert emb.remove_provider("p2")

    assert await _call(emb, method) == [[0.1]]
    breaker.record_success.assert_called_once_with("p1")


@pytest.mark.asyncio
async def test_fallback_embeddings_an_emptied_chain_says_so() -> None:
    """Removing every member is allowed by the method; a call then names it."""
    primary, fallback = AsyncMock(), AsyncMock()
    removed_callback = MagicMock()
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        removed_callback=removed_callback,
    )

    assert emb.remove_provider("p1")
    assert emb.remove_provider("p2")

    assert emb.chain == []
    assert emb._active_provider_id is None
    removed_callback.assert_called_once_with("p1", "p2")
    with pytest.raises(HomeAssistantError, match="No embedding provider is left"):
        await emb.aembed_query("hello")


@pytest.mark.asyncio
async def test_fallback_embeddings_switch_callback_and_sticky_provider(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding fallback switches provider and uses it first afterward."""

    class FakeGeminiEmbeddings:
        """Fake Gemini embeddings model for dimensionality-normalization coverage."""

        def __init__(self) -> None:
            """Initialize fake model."""
            self.aembed_documents = AsyncMock(return_value=[[0.3, 0.4]])

    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.fallback."
        "GoogleGenerativeAIEmbeddings",
        FakeGeminiEmbeddings,
    )

    primary = AsyncMock()
    primary.aembed_documents.side_effect = ConnectionError("emb down")
    primary.aembed_query.return_value = [0.1, 0.2]
    fallback = FakeGeminiEmbeddings()
    switch_callback = MagicMock()

    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        switch_callback=switch_callback,
    )

    assert await emb.aembed_documents(["hello"]) == [[0.3, 0.4]]
    fallback.aembed_documents.assert_awaited_once_with(
        ["hello"], output_dimensionality=EMBEDDING_MODEL_DIMS
    )
    switch_callback.assert_called_once_with("p1", "p2")

    fallback.aembed_documents.reset_mock()
    fallback.aembed_documents.return_value = [[0.5, 0.6]]

    assert await emb.aembed_query("hello") == [0.5, 0.6]
    primary.aembed_query.assert_not_awaited()
    fallback.aembed_documents.assert_awaited_once_with(
        ["hello"], output_dimensionality=EMBEDDING_MODEL_DIMS
    )
    assert "Using sticky embedding provider p2 first" in caplog.text
    assert "configured primary provider p1 will not be retried" in caplog.text


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
async def test_fallback_chat_all_circuit_broken_reports_no_provider() -> None:
    """After a half-open try, a refused call names the open breakers (#881)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError()
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = TimeoutError()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    # The first call tries both, in chain order.
    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await model.ainvoke(["hello"])
    primary.ainvoke.assert_awaited_once()
    fallback.ainvoke.assert_awaited_once()
    # Then neither is due: the call is refused at once, and says why.
    with pytest.raises(
        HomeAssistantError,
        match=r"No LLM provider was tried: p1, p2 failed repeatedly, so their "
        r"circuit breakers are open\. One is tried again in \d+ s\.",
    ):
        await model.ainvoke(["hello"])
    assert primary.ainvoke.await_count == 1
    assert fallback.ainvoke.await_count == 1


def _all_open(probe_interval: float = 10.0) -> CircuitBreaker:
    """Return a breaker open for p1 and p2, both due a half-open try."""
    breaker = CircuitBreaker(
        threshold=1,
        window_seconds=10,
        cooldown_seconds=120,
        probe_interval_seconds=probe_interval,
    )
    breaker.record_failure("p1")
    breaker.record_failure("p2")
    breaker._last_probe.clear()  # Opened over an interval ago.
    return breaker


async def _hang(*_args: Any, **_kwargs: Any) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_chain_back_after_a_network_drop_answers_at_once() -> None:
    """Every breaker open, the network back: the first call is answered (#881)."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage("back")
    fallback = AsyncMock()
    breaker = _all_open()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )

    result = await model.ainvoke(["hello"])

    assert result.content == "back"
    fallback.ainvoke.assert_not_awaited()
    # Its answer closes its breaker: the chain is back to normal.
    assert breaker.is_available("p1")
    assert not breaker.is_available("p2")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Drive the breaker's clock by hand."""
    now = [1000.0]
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.fallback.monotonic",
        lambda: now[0],
    )
    return now


@pytest.mark.asyncio
async def test_a_half_open_try_is_retried_after_its_interval(
    clock: list[float],
) -> None:
    """Between tries a call fails fast; after the interval one is tried again."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = [TimeoutError(), FakeAIMessage("back")]
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = TimeoutError()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await model.ainvoke(["hello"])
    with pytest.raises(HomeAssistantError, match="One is tried again in 10 s"):
        await model.ainvoke(["hello"])

    clock[0] += 10.0

    assert (await model.ainvoke(["hello"])).content == "back"


def test_a_breaker_that_just_opened_waits_an_interval_before_its_try(
    clock: list[float],
) -> None:
    """A provider that just failed repeatedly is not tried again at once."""
    breaker = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=120)
    breaker.record_failure("p1")

    assert breaker.due_for_probe(["p1"]) == []
    clock[0] += 10.0
    assert breaker.due_for_probe(["p1"]) == ["p1"]


@pytest.mark.asyncio
async def test_a_stalled_member_is_not_tried_half_open() -> None:
    """Stalled and circuit-broken: the stall tracker's probe owns it (#828)."""
    primary = AsyncMock()
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    stall = MagicMock()
    stall.is_stalled.side_effect = lambda provider_id: provider_id == "p1"
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
        stall_tracker=stall,
    )

    assert (await model.ainvoke(["hello"])).content == "fallback"
    primary.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refusal_names_providers_by_their_readable_names() -> None:
    """The reply names the providers the user configured, not their ids."""
    stall = MagicMock()
    stall.is_stalled.return_value = False
    stall.name.side_effect = {"p1": "OpenAI", "p2": "Anthropic"}.get
    breaker = _all_open(probe_interval=60.0)
    for provider_id in ("p1", "p2"):
        assert breaker.begin_probe(provider_id)
        breaker.end_probe(provider_id)
    model = FallbackChatModel(
        chain=[(AsyncMock(), "cloud", "p1"), (AsyncMock(), "cloud", "p2")],
        circuit_breaker=breaker,
        stall_tracker=stall,
    )

    with pytest.raises(HomeAssistantError, match="tried: OpenAI, Anthropic failed"):
        await model.ainvoke(["hello"])


@pytest.mark.asyncio
async def test_a_vlm_chain_back_after_a_network_drop_answers_at_once() -> None:
    """The camera path recovers the same way."""
    primary = AsyncMock()
    primary.ainvoke.return_value = FakeAIMessage("a car")
    model = FallbackVLM(
        chain=[(primary, "edge", "p1"), (AsyncMock(), "cloud", "p2")],
        circuit_breaker=_all_open(),
    )

    assert (await model.ainvoke(["image"])).content == "a car"


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_an_embedding_chain_back_after_a_network_drop_answers(
    method: str,
) -> None:
    """The memory store recovers the same way, through its first member."""
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).return_value = (
        [[0.1]] if method == "aembed_documents" else [0.1]
    )
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=_all_open(),
    )

    assert await _call(emb, method)
    getattr(fallback, method).assert_not_awaited()


@pytest.mark.asyncio
async def test_a_recovered_fallback_is_not_starved_by_a_failing_primary(
    clock: list[float],
) -> None:
    """Every due member is tried, primary first, once old failures age out."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError()
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    breaker = CircuitBreaker()
    for provider_id in ("p1", "p2"):
        for _ in range(3):
            breaker.record_failure(provider_id)
        clock[0] += 1.0
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    clock[0] += 70.0  # The failures that opened both are out of the window.

    assert (await model.ainvoke(["hello"])).content == "fallback"
    primary.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_slow_try_of_the_primary_does_not_hold_back_the_fallback() -> None:
    """Each member is reserved only as its own try starts."""
    answered = asyncio.Event()

    async def slow(*_args: Any, **_kwargs: Any) -> FakeAIMessage:
        await answered.wait()
        return FakeAIMessage("back")

    primary = AsyncMock()
    primary.ainvoke.side_effect = slow
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    first = asyncio.create_task(model.ainvoke(["hello"]))
    await asyncio.sleep(0)

    # The primary is being tried: this call goes on to the fallback.
    assert (await model.ainvoke(["hello"])).content == "fallback"
    primary.ainvoke.assert_awaited_once()
    answered.set()
    assert (await first).content == "back"


@pytest.mark.asyncio
async def test_the_interval_runs_from_the_end_of_a_try(clock: list[float]) -> None:
    """A try that waits out its limit is not repeated by the next call at once."""

    async def slow_failure(*_args: Any, **_kwargs: Any) -> None:
        clock[0] += 30.0
        raise TimeoutError

    primary = AsyncMock()
    primary.ainvoke.side_effect = slow_failure
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = TimeoutError()
    breaker = _all_open()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await model.ainvoke(["hello"])

    assert breaker.due_for_probe(["p1"]) == []
    clock[0] += 10.0
    assert breaker.due_for_probe(["p1"]) == ["p1"]


@pytest.mark.asyncio
async def test_a_failed_try_leaves_the_cooldown_as_it_was(
    clock: list[float],
) -> None:
    """Being tried does not keep a provider skipped longer (#881)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError()
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = TimeoutError()
    breaker = _all_open()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    clock[0] += 100.0
    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await model.ainvoke(["hello"])

    clock[0] += 20.0  # 120 s after both opened.
    assert breaker.is_available("p1")
    assert breaker.is_available("p2")


@pytest.mark.asyncio
async def test_a_refusal_during_running_tries_says_one_is_being_tried() -> None:
    """No wait is promised while every open member is being tried."""
    primary, fallback = AsyncMock(), AsyncMock()
    primary.ainvoke.side_effect = _hang
    fallback.ainvoke.side_effect = _hang
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    tries = [asyncio.create_task(model.ainvoke(["hello"])) for _ in range(2)]
    await asyncio.sleep(0)

    with pytest.raises(HomeAssistantError, match=r"One is being tried now\.$"):
        await model.ainvoke(["hello"])
    for task in tries:
        task.cancel()
    await asyncio.gather(*tries, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_cancelled_half_open_try_releases_its_member(
    clock: list[float],
) -> None:
    """A caller giving up on the call does not leave the member held forever."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _hang
    breaker = _all_open()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (AsyncMock(), "cloud", "p2")],
        circuit_breaker=breaker,
    )
    call = asyncio.create_task(model.ainvoke(["hello"]))
    await asyncio.sleep(0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    clock[0] += 10.0

    assert breaker.due_for_probe(["p1", "p2"]) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_a_cancelled_embedding_try_releases_its_member(
    clock: list[float],
) -> None:
    """The embedding chain releases a cancelled try the same way."""
    primary = AsyncMock()
    primary.aembed_query.side_effect = _hang
    breaker = _all_open()
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (AsyncMock(), "edge", "p2")],
        circuit_breaker=breaker,
    )
    call = asyncio.create_task(emb.aembed_query("hello"))
    await asyncio.sleep(0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    clock[0] += 10.0

    assert breaker.due_for_probe(["p1", "p2"]) == ["p1", "p2"]


def test_the_wait_quoted_ends_with_the_cooldown_if_that_comes_first(
    clock: list[float],
) -> None:
    """A breaker closing before the next try is due is the wait to quote."""
    breaker = CircuitBreaker(
        threshold=1, window_seconds=10, cooldown_seconds=5, probe_interval_seconds=10
    )
    breaker.record_failure("p1")

    assert breaker.next_probe_in(["p1"]) == 5.0
    clock[0] += 3.0
    assert breaker.next_probe_in(["p1"]) == 2.0


def test_the_breaker_opening_is_logged_once_per_opening(
    clock: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """A failure while open says nothing new; a later opening is said again."""
    breaker = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=60)
    with caplog.at_level(logging.WARNING):
        breaker.record_failure("p1")
        breaker.record_failure("p1")
    assert caplog.text.count("Circuit breaker opened for provider p1") == 1

    clock[0] += 61.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        breaker.record_failure("p1")
    assert "Circuit breaker opened for provider p1" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_half_open_try_is_not_logged_as_a_new_outage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The breaker opening was logged once; each failed try is debug only."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError()
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = TimeoutError()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    caplog.clear()

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(HomeAssistantError, match="All LLM providers failed"),
    ):
        await model.ainvoke(["hello"])

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert "half-open try of provider p1 failed" in caplog.text


@pytest.mark.parametrize("method", ["aembed_documents", "aembed_query"])
@pytest.mark.asyncio
async def test_an_embedding_refusal_says_when_one_is_tried_again(
    method: str, clock: list[float]
) -> None:
    """Both tried once; the next call is refused and says when (#881)."""
    primary, fallback = AsyncMock(), AsyncMock()
    getattr(primary, method).side_effect = TimeoutError()
    getattr(fallback, method).side_effect = TimeoutError()
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=_all_open(),
    )
    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await _call(emb, method)
    clock[0] += 4.0

    with pytest.raises(
        HomeAssistantError,
        match=r"circuit breaker is open for p1, p2 \(one is tried again in 6 s\)",
    ):
        await _call(emb, method)
    assert getattr(primary, method).await_count == 1
    assert getattr(fallback, method).await_count == 1


@pytest.mark.asyncio
async def test_an_embedding_fallback_held_for_its_build_does_not_block_a_try(
    clock: list[float],
) -> None:
    """A primary whose breaker opened beside a held fallback is tried (#873)."""
    primary, fallback = AsyncMock(), AsyncMock()
    primary.aembed_query.return_value = [0.1]
    breaker = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=120)
    breaker.record_failure("p1")
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=breaker,
    )
    emb.hold_until_verified(["p2"], AsyncMock(return_value=False))
    clock[0] += 10.0

    assert await emb.aembed_query("hello") == [0.1]
    fallback.aembed_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_member_closed_by_another_call_is_tried_as_usual() -> None:
    """A fallback another call's try just closed is not refused as taken."""
    failed = asyncio.Event()

    async def slow_failure(*_args: Any, **_kwargs: Any) -> None:
        await failed.wait()
        raise TimeoutError

    primary = AsyncMock()
    primary.ainvoke.side_effect = slow_failure
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    breaker = _all_open()
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    first = asyncio.create_task(model.ainvoke(["hello"]))
    await asyncio.sleep(0)
    # Meanwhile another call tries the fallback, which answers.
    assert (await model.ainvoke(["hello"])).content == "fallback"
    assert breaker.is_available("p2")

    failed.set()

    assert (await first).content == "fallback"
    assert fallback.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_a_member_another_call_is_trying_is_skipped() -> None:
    """Reserved by another call meanwhile: this call does not try it too."""
    failed = asyncio.Event()

    async def slow_failure(*_args: Any, **_kwargs: Any) -> None:
        await failed.wait()
        raise TimeoutError

    primary = AsyncMock()
    primary.ainvoke.side_effect = slow_failure
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = _hang
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )
    first = asyncio.create_task(model.ainvoke(["hello"]))
    await asyncio.sleep(0)
    second = asyncio.create_task(model.ainvoke(["hello"]))  # Tries p2.
    await asyncio.sleep(0)
    failed.set()

    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
        await first
    fallback.ainvoke.assert_awaited_once()
    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


def test_a_member_not_yet_due_holds_back_those_after_it(
    clock: list[float],
) -> None:
    """Tries go from the primary down, never to a fallback alone (#881)."""
    breaker = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=120)
    breaker.record_failure("p1")
    clock[0] += 8.0
    breaker.record_failure("p2")
    clock[0] += 2.0

    assert breaker.due_for_probe(["p1", "p2"]) == ["p1"]
    assert breaker.begin_probe("p1")
    breaker.end_probe("p1")
    clock[0] += 8.0  # p2 is due, p1 is not.
    assert breaker.due_for_probe(["p1", "p2"]) == []
    clock[0] += 2.0
    assert breaker.due_for_probe(["p1", "p2"]) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_an_embedding_fallback_waits_for_the_active_members_try() -> None:
    """A slow try of the store's provider does not let another take over."""
    primary, fallback = AsyncMock(), AsyncMock()
    primary.aembed_query.side_effect = _hang
    fallback.aembed_query.return_value = [0.2]
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=_all_open(),
    )
    first = asyncio.create_task(emb.aembed_query("hello"))
    await asyncio.sleep(0)

    with pytest.raises(HomeAssistantError, match="one is being tried now"):
        await emb.aembed_query("hello")
    fallback.aembed_query.assert_not_awaited()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_failed_embedding_try_leaves_the_cooldown_as_it_was(
    clock: list[float],
) -> None:
    """As for the chat chain: being tried does not keep it skipped longer."""
    primary, fallback = AsyncMock(), AsyncMock()
    primary.aembed_query.side_effect = TimeoutError()
    fallback.aembed_query.side_effect = TimeoutError()
    breaker = _all_open()
    emb = FallbackEmbeddings(
        chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")],
        circuit_breaker=breaker,
    )
    clock[0] += 100.0
    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await emb.aembed_query("hello")

    clock[0] += 20.0
    assert breaker.is_available("p1")
    assert breaker.is_available("p2")


@pytest.mark.asyncio
async def test_a_half_open_try_without_the_model_is_still_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing model stays visible, even on a half-open try (#752)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _model_not_found_errors()[0]
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=_all_open(),
    )

    with caplog.at_level(logging.WARNING):
        assert (await model.ainvoke(["hello"])).content == "fallback"

    assert "does not have the configured model" in caplog.text


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


@pytest.mark.asyncio
async def test_fallback_vlm_with_config() -> None:
    """with_config should propagate to VLM models and store config."""
    primary = MagicMock()
    primary.with_config.return_value = AsyncMock()
    fallback = MagicMock()
    fallback.with_config.return_value = AsyncMock()

    model = FallbackVLM(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        config={"configurable": {"temperature": 0.5}},
    )

    wrapped = model.with_config(config={"configurable": {"num_predict": 128}})

    assert isinstance(wrapped, FallbackVLM)
    assert len(wrapped.chain) == 2
    primary.with_config.assert_called_once()
    fallback.with_config.assert_called_once()
    assert "num_predict" in wrapped.config.get("configurable", {})


def test_fallback_embeddings_sync_raises() -> None:
    """Synchronous embedding methods must raise NotImplementedError."""
    primary = AsyncMock()
    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1")])
    with pytest.raises(NotImplementedError, match="Use aembed_documents"):
        emb.embed_documents(["hello"])
    with pytest.raises(NotImplementedError, match="Use aembed_query"):
        emb.embed_query("hello")


@pytest.mark.asyncio
async def test_fallback_vlm_all_providers_fail() -> None:
    """When all VLM providers fail, HomeAssistantError is raised."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = HomeAssistantError("vlm primary down")
    fallback = AsyncMock()
    fallback.ainvoke.side_effect = HomeAssistantError("vlm fallback down")

    model = FallbackVLM(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    with pytest.raises(HomeAssistantError, match="All VLM providers failed"):
        await model.ainvoke(["image"])


@pytest.mark.asyncio
async def test_fallback_embeddings_aembed_documents_all_fail() -> None:
    """When all embedding providers fail on documents, HomeAssistantError is raised."""
    primary = AsyncMock()
    primary.aembed_documents.side_effect = HomeAssistantError("emb primary down")
    fallback = AsyncMock()
    fallback.aembed_documents.side_effect = HomeAssistantError("emb fallback down")

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await emb.aembed_documents(["hello"])


@pytest.mark.asyncio
async def test_fallback_embeddings_aembed_query_all_fail() -> None:
    """When all embedding providers fail on query, HomeAssistantError is raised."""
    primary = AsyncMock()
    primary.aembed_query.side_effect = HomeAssistantError("emb primary down")
    fallback = AsyncMock()
    fallback.aembed_query.side_effect = HomeAssistantError("emb fallback down")

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")])
    with pytest.raises(HomeAssistantError, match="All embedding providers failed"):
        await emb.aembed_query("hello")


def test_circuit_breaker_window_expiry() -> None:
    """Failures outside the window are not counted toward the threshold."""
    cb = CircuitBreaker(threshold=2, window_seconds=10.0, cooldown_seconds=60.0)

    with patch(
        "custom_components.home_generative_agent.core.fallback.monotonic"
    ) as mock_time:
        mock_time.return_value = 0.0
        cb.record_failure("p1")

        mock_time.return_value = 15.0
        cb.record_failure("p1")

        assert cb.is_available("p1") is True


# ---------------------------------------------------------------------------
# Unsupported sampling parameter retry (issue #502)
# ---------------------------------------------------------------------------


def _bad_request_error(
    param: str | None = "temperature", code: str | None = "unsupported_value"
) -> Exception:
    """Build a real openai.BadRequestError like the API returns for temperature."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "message": (
            f"Unsupported value: '{param}' does not support 0.2 with this model. "
            "Only the default (1) value is supported."
        ),
        "type": "invalid_request_error",
        "param": param,
        "code": code,
    }
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}",
        response=response,  # pyright: ignore[reportArgumentType]  # httpx vs openai 3's httpx2
        body=body,
    )


def test_unsupported_sampling_param_detects_temperature() -> None:
    """The OpenAI unsupported-temperature 400 is detected as droppable."""
    assert unsupported_sampling_param(_bad_request_error()) == "temperature"


def test_unsupported_sampling_param_detects_top_p() -> None:
    """The OpenAI unsupported-top_p 400 is detected as droppable."""
    assert unsupported_sampling_param(_bad_request_error(param="top_p")) == "top_p"


def test_unsupported_sampling_param_ignores_other_params() -> None:
    """A 400 for a non-sampling param must not trigger a retry."""
    assert unsupported_sampling_param(_bad_request_error(param="messages")) is None


def test_unsupported_sampling_param_ignores_other_codes() -> None:
    """A temperature 400 with an unrelated code must not trigger a retry."""
    err = _bad_request_error(code="invalid_type")
    assert unsupported_sampling_param(err) is None


def test_unsupported_sampling_param_ignores_other_exceptions() -> None:
    """Non-OpenAI exceptions must not trigger a retry."""
    assert unsupported_sampling_param(ValueError("temperature")) is None


@pytest.mark.asyncio
async def test_ainvoke_drops_rejected_temperature() -> None:
    """A model rejecting temperature is retried once with temperature nulled."""
    model = AsyncMock()
    model.ainvoke.side_effect = [_bad_request_error(), FakeAIMessage("ok")]

    result = await ainvoke_dropping_unsupported_params(model, ["hi"])

    assert result.content == "ok"
    assert model.ainvoke.await_count == 2
    retry_config = model.ainvoke.await_args_list[1].args[1]
    assert retry_config["configurable"]["temperature"] is None


@pytest.mark.asyncio
async def test_ainvoke_drop_preserves_existing_config() -> None:
    """The retry keeps callbacks and other configurable entries intact."""
    model = AsyncMock()
    model.ainvoke.side_effect = [_bad_request_error(), FakeAIMessage("ok")]
    config = {"callbacks": ["cb"], "configurable": {"user_id": "u1"}}

    await ainvoke_dropping_unsupported_params(model, ["hi"], config)

    first_config = model.ainvoke.await_args_list[0].args[1]
    assert first_config is config
    retry_config = model.ainvoke.await_args_list[1].args[1]
    assert retry_config["callbacks"] == ["cb"]
    assert retry_config["configurable"] == {"user_id": "u1", "temperature": None}


@pytest.mark.asyncio
async def test_ainvoke_drops_temperature_then_top_p() -> None:
    """Sequential rejections of temperature and top_p are both dropped."""
    model = AsyncMock()
    model.ainvoke.side_effect = [
        _bad_request_error(),
        _bad_request_error(param="top_p"),
        FakeAIMessage("ok"),
    ]

    result = await ainvoke_dropping_unsupported_params(model, ["hi"])

    assert result.content == "ok"
    assert model.ainvoke.await_count == 3
    final_config = model.ainvoke.await_args_list[2].args[1]
    assert final_config["configurable"] == {"temperature": None, "top_p": None}


@pytest.mark.asyncio
async def test_ainvoke_reraises_when_drop_does_not_help() -> None:
    """A second rejection of an already-dropped param is re-raised, not looped."""
    model = AsyncMock()
    model.ainvoke.side_effect = [_bad_request_error(), _bad_request_error()]

    with pytest.raises(openai.BadRequestError):
        await ainvoke_dropping_unsupported_params(model, ["hi"])
    assert model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_ainvoke_reraises_unrelated_errors() -> None:
    """Errors that are not droppable sampling-param 400s propagate unchanged."""
    model = AsyncMock()
    model.ainvoke.side_effect = ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await ainvoke_dropping_unsupported_params(model, ["hi"])
    assert model.ainvoke.await_count == 1


def test_sync_invoke_drops_rejected_temperature() -> None:
    """The sync variant (Sentinel executor path) retries without temperature."""
    model = MagicMock()
    model.invoke.side_effect = [_bad_request_error(), FakeAIMessage("ok")]

    result = invoke_dropping_unsupported_params(model, ["hi"])

    assert result.content == "ok"
    assert model.invoke.call_count == 2
    retry_config = model.invoke.call_args_list[1].args[1]
    assert retry_config["configurable"]["temperature"] is None


@pytest.mark.asyncio
async def test_fallback_chat_drops_temperature_without_switching_provider() -> None:
    """A temperature 400 is retried on the same provider, not failed over."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = [_bad_request_error(), FakeAIMessage("primary")]
    fallback = AsyncMock()

    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "cloud", "p2")]
    )
    result = await model.ainvoke(["hello"])

    assert result.content == "primary"
    assert primary.ainvoke.await_count == 2
    fallback.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_vlm_drops_temperature_without_switching_provider() -> None:
    """FallbackVLM also retries the same provider on a temperature 400."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = [_bad_request_error(), FakeAIMessage("caption")]
    fallback = AsyncMock()

    vlm = FallbackVLM(chain=[(primary, "cloud", "p1"), (fallback, "cloud", "p2")])
    result = await vlm.ainvoke({"image": "x"})

    assert result.content == "caption"
    assert primary.ainvoke.await_count == 2
    fallback.ainvoke.assert_not_awaited()


def test_unsupported_sampling_param_in_chain_direct_and_wrapped() -> None:
    """The chain walker finds the rejected param through HomeAssistantError causes."""
    direct = _bad_request_error()
    assert unsupported_sampling_param_in_chain(direct) == "temperature"

    wrapped = HomeAssistantError(f"Model invocation failed: {direct}")
    wrapped.__cause__ = direct
    assert unsupported_sampling_param_in_chain(wrapped) == "temperature"

    assert unsupported_sampling_param_in_chain(ValueError("boom")) is None


@pytest.mark.asyncio
async def test_ainvoke_helper_passes_through_fallback_wrappers() -> None:
    """The outer helper must not merge drops into a fallback chain's shared config."""
    provider = AsyncMock()
    provider.ainvoke.side_effect = [_bad_request_error(), _bad_request_error()]
    wrapper = FallbackChatModel(chain=[(provider, "cloud", "p1")])

    # A lone provider's own error comes back unwrapped (#828).
    with pytest.raises(openai.BadRequestError):
        await ainvoke_dropping_unsupported_params(wrapper, ["hi"])

    # Inner per-provider drop retried once (2 calls); the outer helper must NOT
    # re-run the chain with a contaminated shared config (which would be 4).
    assert provider.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_fallback_chat_surviving_sampling_400_falls_over() -> None:
    """A provider still rejecting after the drop fails over to the next provider."""

    def _always_temp_400(*_args: Any, **_kwargs: Any) -> None:
        raise _bad_request_error()

    primary = AsyncMock()
    primary.ainvoke.side_effect = _always_temp_400
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("fallback")
    breaker = CircuitBreaker(threshold=3)

    model = FallbackChatModel(
        chain=[(primary, "cloud", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    result = await model.ainvoke(["hello"])

    assert result.content == "fallback"
    # Drop retry on p1 (2 calls), then fail over to p2 with p1's failure recorded.
    assert primary.ainvoke.await_count == 2
    assert len(breaker._failures.get("p1", [])) == 1


@pytest.mark.asyncio
async def test_fallback_chat_tool_bound_rebinds_same_provider() -> None:
    """
    A tool-bound provider rejecting sampling params is rebound, not failed over.

    bind_tools bakes sampling params into a concrete binding, so the
    config-drop retry cannot help; the wrapper must rebuild that provider
    from its pre-bind source with default sampling and retry it in place.
    """
    bound_p1 = AsyncMock()
    bound_p1.ainvoke.side_effect = [_bad_request_error(), _bad_request_error()]
    rebound_p1 = AsyncMock()
    rebound_p1.ainvoke.return_value = FakeAIMessage("rebound")
    stripped_p1 = MagicMock()
    stripped_p1.bind_tools.return_value = rebound_p1
    src_p1 = MagicMock()
    src_p1.bind_tools.return_value = bound_p1
    src_p1.with_config.return_value = stripped_p1

    bound_p2 = AsyncMock()
    src_p2 = MagicMock()
    src_p2.bind_tools.return_value = bound_p2

    breaker = CircuitBreaker(threshold=3)
    model = FallbackChatModel(
        chain=[(src_p1, "cloud", "p1"), (src_p2, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    bound = model.bind_tools([{"name": "t"}])

    result = await bound.ainvoke(["hello"])

    assert result.content == "rebound"
    # p1 was retried in place with default sampling; p2 never entered.
    bound_p2.ainvoke.assert_not_awaited()
    strip_config = src_p1.with_config.call_args.kwargs["config"]
    assert strip_config == {"configurable": dict.fromkeys(DROPPABLE_SAMPLING_PARAMS)}
    # The rebind success cleared p1 instead of recording a breaker failure.
    assert breaker._failures.get("p1") in (None, [])


@pytest.mark.asyncio
async def test_fallback_chat_rebind_failure_still_falls_over() -> None:
    """When the rebound provider also fails, normal provider fallback proceeds."""
    bound_p1 = AsyncMock()
    bound_p1.ainvoke.side_effect = [_bad_request_error(), _bad_request_error()]
    rebound_p1 = AsyncMock()
    rebound_p1.ainvoke.side_effect = HomeAssistantError("still down")
    stripped_p1 = MagicMock()
    stripped_p1.bind_tools.return_value = rebound_p1
    src_p1 = MagicMock()
    src_p1.bind_tools.return_value = bound_p1
    src_p1.with_config.return_value = stripped_p1

    bound_p2 = AsyncMock()
    bound_p2.ainvoke.return_value = FakeAIMessage("fallback")
    src_p2 = MagicMock()
    src_p2.bind_tools.return_value = bound_p2

    model = FallbackChatModel(chain=[(src_p1, "cloud", "p1"), (src_p2, "cloud", "p2")])
    bound = model.bind_tools([{"name": "t"}])

    result = await bound.ainvoke(["hello"])
    assert result.content == "fallback"


def test_reasoning_effort_is_droppable() -> None:
    """A 400 rejecting reasoning_effort must be retried without it (issue #580)."""
    assert "reasoning_effort" in DROPPABLE_SAMPLING_PARAMS


# ---------------------------------------------------------------------------
# A provider without the model moves the chain on (#752, #759)
# ---------------------------------------------------------------------------


def _http_error(cls: Any, status: int, message: str) -> Exception:
    """Build an OpenAI/Anthropic SDK status error for an HTTP response."""
    request = httpx.Request("POST", "https://api.example.invalid/v1/x")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


def _http_404(cls: Any, message: str) -> Exception:
    """Build an OpenAI/Anthropic SDK status error for a 404 response."""
    return _http_error(cls, 404, message)


def _wrapped_gemini_404() -> Exception:
    """langchain-google-genai's embeddings keep Google's 404 only as the cause."""
    from google.api_core.exceptions import NotFound  # noqa: PLC0415
    from langchain_google_genai._common import (  # noqa: PLC0415
        GoogleGenerativeAIError,
    )

    try:
        try:
            not_found = "models/not-a-model is not found"
            raise NotFound(not_found)  # noqa: TRY301
        except NotFound as err:
            msg = f"Error embedding content: {err}"
            raise GoogleGenerativeAIError(msg) from err
    except GoogleGenerativeAIError as wrapped:
        return wrapped


def _model_not_found_errors() -> list[Exception]:
    import anthropic  # noqa: PLC0415
    import ollama  # noqa: PLC0415
    from google.api_core.exceptions import NotFound  # noqa: PLC0415

    return [
        ollama.ResponseError(
            'model "nomic-embed-text" not found, try pulling it first', 404
        ),
        _http_404(openai.NotFoundError, "The model `gpt-x` does not exist"),
        _http_404(anthropic.NotFoundError, "model: claude-x"),
        NotFound("models/gemini-x is not found"),
        _wrapped_gemini_404(),
    ]


@pytest.mark.parametrize(
    "err", _model_not_found_errors(), ids=lambda e: type(e).__name__
)
def test_model_not_found_is_retryable(err: Exception) -> None:
    """Each provider's 404 for a missing model moves the chain on."""
    assert is_model_not_found(err)
    assert _is_retryable(err) is True


def test_other_client_errors_still_stop_the_chain() -> None:
    """A rejected key or a bad request is not a missing model."""
    import ollama  # noqa: PLC0415

    auth = _http_error(openai.AuthenticationError, 401, "Incorrect API key")
    bad = ollama.ResponseError("invalid input", 400)
    for err in (auth, bad):
        assert not is_model_not_found(err)
        assert _is_retryable(err) is False


def _auth_error_raised_in_a_404_handler() -> Exception:
    """Return a 401 raised while a 404 was handled: the 404 is only context."""
    import ollama  # noqa: PLC0415

    try:
        try:
            not_found = "model not found"
            raise ollama.ResponseError(not_found, 404)  # noqa: TRY301
        except ollama.ResponseError:
            raise _http_error(
                openai.AuthenticationError, 401, "Incorrect API key"
            ) from None
    except openai.AuthenticationError as err:
        return err


def test_error_raised_while_handling_a_404_is_not_a_missing_model() -> None:
    """Only an explicit cause counts: a 401 raised in a 404 handler still stops."""
    err = _auth_error_raised_in_a_404_handler()
    assert err.__context__ is not None
    assert not is_model_not_found(err)
    assert _is_retryable(err) is False


def _ollama_missing(model: str = "not-pulled") -> Exception:
    import ollama  # noqa: PLC0415

    return ollama.ResponseError(f'model "{model}" not found, try pulling it first', 404)


@pytest.mark.asyncio
async def test_chat_primary_without_its_model_fails_over(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing primary chat model is logged at ERROR and the fallback answers."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _ollama_missing()
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("hi")
    breaker = CircuitBreaker(threshold=1)
    alert = MagicMock()

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
        alert_callback=alert,
    )
    caplog.set_level(logging.WARNING)
    result = await model.ainvoke(["hello"])

    assert result.content == "hi"
    assert not breaker.is_available("p1")
    alert.assert_called_once()
    assert alert.call_args.kwargs["fallback_to"] == "p2"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("does not have the configured model" in r.getMessage() for r in errors)


@pytest.mark.asyncio
async def test_vlm_fallback_without_its_model_moves_on() -> None:
    """A fallback missing its model hands over to the next one (#752)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError("primary timed out")
    middle = AsyncMock()
    middle.ainvoke.side_effect = _ollama_missing("qwen2.5vl:7b")
    last = AsyncMock()
    last.ainvoke.return_value = FakeAIMessage("a cat")

    vlm = FallbackVLM(
        chain=[(primary, "cloud", "p1"), (middle, "edge", "p2"), (last, "cloud", "p3")]
    )
    result = await vlm.ainvoke(["image"])

    assert result.content == "a cat"
    last.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedding_primary_without_its_model_fails_over() -> None:
    """The #759 case: the primary's model was never pulled; the fallback embeds."""
    primary = AsyncMock()
    primary.aembed_query.side_effect = _ollama_missing("mxbai-embed-large")
    fallback = AsyncMock()
    fallback.aembed_query.return_value = [0.5, 0.6]

    emb = FallbackEmbeddings(chain=[(primary, "edge", "p1"), (fallback, "edge", "p2")])
    assert await emb.aembed_query("hello") == [0.5, 0.6]


@pytest.mark.asyncio
async def test_embedding_fallback_without_its_model_moves_on() -> None:
    """The #752 case: a fallback missing its model does not end the chain."""
    primary = AsyncMock()
    primary.aembed_documents.side_effect = ConnectionError("primary down")
    middle = AsyncMock()
    middle.aembed_documents.side_effect = _ollama_missing()
    last = AsyncMock()
    last.aembed_documents.return_value = [[0.7, 0.8]]

    emb = FallbackEmbeddings(
        chain=[(primary, "cloud", "p1"), (middle, "edge", "p2"), (last, "edge", "p3")]
    )
    assert await emb.aembed_documents(["hello"]) == [[0.7, 0.8]]


@pytest.mark.asyncio
async def test_missing_model_opens_the_circuit_breaker() -> None:
    """Repeated 404s are recorded, so later calls skip the provider."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = _ollama_missing()
    fallback = AsyncMock()
    fallback.ainvoke.return_value = FakeAIMessage("ok")
    breaker = CircuitBreaker(threshold=3)

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=breaker,
    )
    for _ in range(4):
        await model.ainvoke(["hello"])

    assert primary.ainvoke.await_count == 3
