# ruff: noqa: S101
"""Tests for core fallback wrappers."""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.const import EMBEDDING_MODEL_DIMS
from custom_components.home_generative_agent.core.fallback import (
    DROPPABLE_SAMPLING_PARAMS,
    CircuitBreaker,
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
    """All-skipped providers should raise a clean HomeAssistantError."""
    primary = AsyncMock()
    fallback = AsyncMock()
    cb = CircuitBreaker(threshold=1, window_seconds=10, cooldown_seconds=60)
    cb.record_failure("p1")
    cb.record_failure("p2")

    model = FallbackChatModel(
        chain=[(primary, "edge", "p1"), (fallback, "cloud", "p2")],
        circuit_breaker=cb,
    )
    with pytest.raises(HomeAssistantError, match="No provider was available"):
        await model.ainvoke(["hello"])
    primary.ainvoke.assert_not_awaited()
    fallback.ainvoke.assert_not_awaited()


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

    with pytest.raises(HomeAssistantError, match="All LLM providers failed"):
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
