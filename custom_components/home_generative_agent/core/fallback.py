"""Fallback wrappers for LLM/VLM/embedding model chains."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage

LOGGER = logging.getLogger(__name__)

FALLBACK_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    HomeAssistantError,
)

try:
    from langchain_core.exceptions import LangChainException

    FALLBACK_RETRYABLE_EXCEPTIONS += (LangChainException,)
except ImportError:
    pass


def _is_retryable(err: Exception) -> bool:
    """Return True if the exception should trigger a fallback attempt."""
    if isinstance(err, FALLBACK_RETRYABLE_EXCEPTIONS):
        return True
    # Also retry on rate-limit / API errors from common providers
    err_str = str(err).lower()
    return any(
        keyword in err_str
        for keyword in (
            "rate limit",
            "ratelimit",
            "too many requests",
            "timeout",
            "connection",
            "unavailable",
            "internal server error",
            "bad gateway",
            "service unavailable",
        )
    )


class CircuitBreaker:
    """Simple in-memory circuit breaker for model providers."""

    def __init__(
        self,
        threshold: int = 3,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 120.0,
    ) -> None:
        """Initialize circuit breaker with thresholds."""
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[str, list[float]] = {}
        self._disabled_until: dict[str, float] = {}

    def record_failure(self, provider_id: str) -> None:
        """Record a failure for the given provider."""
        now = monotonic()
        self._failures.setdefault(provider_id, []).append(now)
        self._failures[provider_id] = [
            t
            for t in self._failures[provider_id]
            if now - t <= self.window_seconds
        ]
        if len(self._failures[provider_id]) >= self.threshold:
            self._disabled_until[provider_id] = now + self.cooldown_seconds
            LOGGER.warning(
                "Circuit breaker opened for provider %s (disabled for %.0fs)",
                provider_id,
                self.cooldown_seconds,
            )

    def record_success(self, provider_id: str) -> None:
        """Clear failures for a provider that succeeded."""
        self._failures.pop(provider_id, None)
        if provider_id in self._disabled_until:
            del self._disabled_until[provider_id]

    def is_available(self, provider_id: str) -> bool:
        """Return True if the provider is not currently circuit-broken."""
        until = self._disabled_until.get(provider_id, 0.0)
        return monotonic() >= until


def _default_alert(
    provider_id: str, err: Exception, *, fallback_to: str | None = None
) -> None:
    """Default alert callback that logs a warning."""
    if fallback_to:
        LOGGER.warning(
            "Fallback activated: provider %s failed (%s), switching to %s",
            provider_id,
            err,
            fallback_to,
        )
    else:
        LOGGER.error(
            "All providers failed. Last provider %s error: %s",
            provider_id,
            err,
        )


class FallbackChatModel:
    """Chainable chat model that falls back through a list of providers."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
        alert_callback: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize fallback chat model.

        :param chain: List of (model_instance, deployment, provider_id).
        :param circuit_breaker: Optional CircuitBreaker instance.
        :param alert_callback: Callable(provider_id, error, fallback_to=None).
        :param config: Internal config dict (used for with_config chaining).
        """
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.alert_callback = alert_callback or _default_alert
        self._config = config or {}

    @property
    def config(self) -> dict[str, Any]:
        """Return internal config dict (for video_analyzer compat)."""
        return self._config

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FallbackChatModel:
        """Return a new FallbackChatModel with tools bound to each model."""
        bound_chain: list[tuple[Any, str, str]] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "bind_tools"):
                bound_chain.append(
                    (model.bind_tools(tools, **kwargs), deployment, provider_id)
                )
            else:
                bound_chain.append((model, deployment, provider_id))
        return FallbackChatModel(
            bound_chain,
            self.circuit_breaker,
            self.alert_callback,
            dict(self._config),
        )

    def with_config(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> FallbackChatModel:
        """Return a new FallbackChatModel with merged config."""
        merged: dict[str, Any] = dict(self._config)
        if config:
            merged.update(config)
        wrapped_chain: list[tuple[Any, str, str]] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "with_config"):
                wrapped_chain.append(
                    (model.with_config(config, **kwargs), deployment, provider_id)
                )
            else:
                wrapped_chain.append((model, deployment, provider_id))
        return FallbackChatModel(
            wrapped_chain,
            self.circuit_breaker,
            self.alert_callback,
            merged,
        )

    async def ainvoke(
        self,
        input_data: LanguageModelInput,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke the chat model chain, falling back on retryable failures."""
        last_err: Exception | None = None
        for i, (model, _deployment, provider_id) in enumerate(self.chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                LOGGER.debug("Provider %s is circuit-broken; skipping.", provider_id)
                continue
            try:
                result = await model.ainvoke(input_data, config, **kwargs)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "Model call failed for provider %s (attempt %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    err,
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                self.alert_callback(
                    provider_id,
                    err,
                    fallback_to=(
                        self.chain[i + 1][2] if i + 1 < len(self.chain) else None
                    ),
                )
                continue
            else:
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                return result

        msg = f"All LLM providers failed. Last error: {last_err}"
        raise HomeAssistantError(msg) from last_err

    async def astream(
        self,
        input_data: LanguageModelInput,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Stream from the first available model in the chain."""
        last_err: Exception | None = None
        for i, (model, _deployment, provider_id) in enumerate(self.chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                async for chunk in model.astream(input_data, config, **kwargs):
                    yield chunk
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "Model stream failed for provider %s (attempt %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    err,
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                self.alert_callback(
                    provider_id,
                    err,
                    fallback_to=(
                        self.chain[i + 1][2] if i + 1 < len(self.chain) else None
                    ),
                )
                continue
            else:
                return

        msg = f"All LLM providers failed. Last error: {last_err}"
        raise HomeAssistantError(msg) from last_err


class FallbackVLM:
    """Fallback wrapper for VLM image analysis."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize fallback VLM wrapper."""
        self.chain = chain
        self.circuit_breaker = circuit_breaker

    async def ainvoke(
        self,
        input_data: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke the VLM chain, falling back on retryable failures."""
        last_err: Exception | None = None
        for i, (model, _deployment, provider_id) in enumerate(self.chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                return await model.ainvoke(input_data, config, **kwargs)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "VLM call failed for provider %s (attempt %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    err,
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue

        msg = f"All VLM providers failed. Last error: {last_err}"
        raise HomeAssistantError(msg) from last_err


class FallbackEmbeddings:
    """Fallback wrapper for embedding models."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize fallback embeddings wrapper."""
        self.chain = chain
        self.circuit_breaker = circuit_breaker

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents through the fallback chain."""
        last_err: Exception | None = None
        for i, (model, _deployment, provider_id) in enumerate(self.chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                return await model.aembed_documents(texts)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "Embedding call failed for provider %s (attempt %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    err,
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue

        msg = f"All embedding providers failed. Last error: {last_err}"
        raise HomeAssistantError(msg) from last_err

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query through the fallback chain."""
        last_err: Exception | None = None
        for i, (model, _deployment, provider_id) in enumerate(self.chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                return await model.aembed_query(text)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "Embedding query failed for provider %s (attempt %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    err,
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue

        msg = f"All embedding providers failed. Last error: {last_err}"
        raise HomeAssistantError(msg) from last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Run synchronous document embedding via the async fallback chain."""
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Run synchronous query embedding via the async fallback chain."""
        return asyncio.run(self.aembed_query(text))
