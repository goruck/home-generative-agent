"""Fallback wrappers for LLM/VLM/embedding model chains."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from homeassistant.exceptions import HomeAssistantError
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..const import EMBEDDING_MODEL_DIMS  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

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

try:
    import httpx

    FALLBACK_RETRYABLE_EXCEPTIONS += (httpx.TransportError,)
except ImportError:
    pass

try:
    from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
    OpenAIBadRequestError = None  # type: ignore[assignment, misc]

# Sampling parameters that can be safely omitted when a model rejects them
# (e.g. OpenAI reasoning-style models only accept the default temperature=1).
DROPPABLE_SAMPLING_PARAMS = frozenset({"temperature", "top_p"})

_UNSUPPORTED_PARAM_CODES = frozenset({"unsupported_value", "unsupported_parameter"})


def unsupported_sampling_param(err: Exception) -> str | None:
    """Return the droppable sampling param an OpenAI 400 rejected, if any."""
    if OpenAIBadRequestError is None or not isinstance(err, OpenAIBadRequestError):
        return None
    param = getattr(err, "param", None)
    code = getattr(err, "code", None)
    if param in DROPPABLE_SAMPLING_PARAMS and code in _UNSUPPORTED_PARAM_CODES:
        return param
    return None


def _config_dropping(config: Any, dropped: dict[str, Any]) -> Any:
    """Merge dropped sampling params into a Runnable config."""
    if not dropped:
        return config
    return _merge_config(config, {"configurable": dropped})


def unsupported_sampling_param_in_chain(
    err: BaseException, max_depth: int = 10
) -> str | None:
    """Return the droppable param rejected anywhere in an exception cause chain."""
    current: BaseException | None = err
    for _ in range(max_depth):
        if current is None:
            return None
        if isinstance(current, Exception) and (
            param := unsupported_sampling_param(current)
        ):
            return param
        current = current.__cause__
    return None


def _record_dropped_param(err: Exception, dropped: dict[str, Any]) -> None:
    """Register the rejected param for the retry, re-raising if not droppable."""
    param = unsupported_sampling_param(err)
    if param is None or param in dropped:
        raise err
    dropped[param] = None
    LOGGER.warning(
        "Model rejected unsupported parameter %r (%s); retrying without it. "
        "Configure the model's %s to its supported default to avoid this retry.",
        param,
        _safe_err_summary(err),
        param,
    )


async def ainvoke_dropping_unsupported_params(
    model: Any,
    input_data: Any,
    config: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Invoke a model, dropping sampling params its provider rejects with a 400.

    Some OpenAI models (o-series / gpt-5-style reasoning models and certain
    mini deployments) only accept the default temperature/top_p and reject any
    other value with ``unsupported_value``. Configurable-field models omit a
    param entirely when it is None, so retry with the rejected param nulled.

    NOTE: the config-merge retry only works on configurable-field models. It
    is a no-op on tool-bound models (bind_tools bakes params into a concrete
    binding) — the chat path handles those by rebinding in agent/graph.py.
    """
    if isinstance(model, (FallbackChatModel, FallbackVLM)):
        # The wrapper already drops rejected params per provider; merging
        # drops into the shared config here would silently null sampling
        # params for every provider in the chain.
        return await model.ainvoke(input_data, config, **kwargs)
    dropped: dict[str, Any] = {}
    while True:
        # Omit a None config entirely: duck-typed models (e.g. Sentinel test
        # doubles) may only accept ainvoke(messages).
        call_config = _config_dropping(config, dropped)
        try:
            if call_config is None:
                return await model.ainvoke(input_data, **kwargs)
            return await model.ainvoke(input_data, call_config, **kwargs)
        except Exception as err:  # noqa: BLE001 — re-raised unless droppable
            _record_dropped_param(err, dropped)


def invoke_dropping_unsupported_params(
    model: Any,
    input_data: Any,
    config: Any = None,
    **kwargs: Any,
) -> Any:
    """Sync variant of ainvoke_dropping_unsupported_params for executor calls."""
    dropped: dict[str, Any] = {}
    while True:
        call_config = _config_dropping(config, dropped)
        try:
            if call_config is None:
                return model.invoke(input_data, **kwargs)
            return model.invoke(input_data, call_config, **kwargs)
        except Exception as err:  # noqa: BLE001 — re-raised unless droppable
            _record_dropped_param(err, dropped)


def _is_retryable(err: Exception) -> bool:
    """Return True if the exception should trigger a fallback attempt."""
    if isinstance(err, FALLBACK_RETRYABLE_EXCEPTIONS):
        return True
    if unsupported_sampling_param(err) is not None:
        # A sampling-param 400 that survived the in-provider drop retry:
        # treat the provider as misconfigured — record the failure (so the
        # circuit breaker can open) and let the chain try the next provider.
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
            "internal server error",
            "bad gateway",
            "service unavailable",
        )
    )


def _safe_err_summary(err: Exception | None) -> str:
    """Return a truncated, safe error string for logging."""
    if err is None:
        return "No provider was available"
    return f"{type(err).__name__}: {str(err)[:200]}"


def _merge_config(
    base: dict[str, Any] | None, override: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge Runnable config dicts while preserving existing configurable values."""
    merged = dict(base or {})
    if not override:
        return merged

    for key, value in override.items():
        if key == "configurable" and isinstance(value, dict):
            existing = merged.get("configurable")
            merged["configurable"] = {
                **(existing if isinstance(existing, dict) else {}),
                **value,
            }
        else:
            merged[key] = value
    return merged


def _merge_model_config(model: Any, config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge new config with any config already bound to a model."""
    existing = getattr(model, "config", None)
    return _merge_config(existing if isinstance(existing, dict) else None, config)


def _retryable_empty_response(result: Any) -> HomeAssistantError | None:
    """Return retryable error when a model hit length limit with empty content."""
    content = getattr(result, "content", None)
    if content:
        return None
    if getattr(result, "tool_calls", None):
        return None

    metadata_raw: Any = getattr(result, "response_metadata", {}) or {}
    if not isinstance(metadata_raw, dict):
        return None
    metadata = cast("dict[str, Any]", metadata_raw)
    finish_reason = metadata.get("finish_reason") or metadata.get("done_reason")
    if str(finish_reason).lower() not in {"length", "max_tokens"}:
        return None

    model_name = metadata.get("model_name") or metadata.get("model")
    msg = f"Empty model content after length-limited response from {model_name}"
    return HomeAssistantError(msg)


def _raise_for_retryable_empty_response(result: Any) -> None:
    """Raise when a result should be retried through fallback."""
    if retry_err := _retryable_empty_response(result):
        raise retry_err


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
            t for t in self._failures[provider_id] if now - t <= self.window_seconds
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
    provider_id: str,
    err: Exception,
    *,
    fallback_to: str | None = None,
    fallback_deployment: str | None = None,
) -> None:
    """Default alert callback that logs a warning."""
    del fallback_deployment
    summary = _safe_err_summary(err)
    if fallback_to:
        LOGGER.warning(
            "Fallback activated: provider %s failed (%s), switching to %s",
            provider_id,
            summary,
            fallback_to,
        )
    else:
        LOGGER.error(
            "All providers failed. Last provider %s error: %s",
            provider_id,
            summary,
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
        # Parallel to chain after bind_tools: (pre-bind model, tools, kwargs)
        # per provider, so a sampling-param 400 can rebind THAT provider with
        # default sampling instead of failing over (bind_tools bakes sampling
        # params into a concrete binding the config-drop retry can't reach).
        self._rebind_sources: list[tuple[Any, list[Any], dict[str, Any]] | None] = []

    @property
    def config(self) -> dict[str, Any]:
        """Return internal config dict (for video_analyzer compat)."""
        return self._config

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FallbackChatModel:
        """Return a new FallbackChatModel with tools bound to each model."""
        bound_chain: list[tuple[Any, str, str]] = []
        rebind_sources: list[tuple[Any, list[Any], dict[str, Any]] | None] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "bind_tools"):
                bound_chain.append(
                    (model.bind_tools(tools, **kwargs), deployment, provider_id)
                )
                rebind_sources.append((model, list(tools), dict(kwargs)))
            else:
                bound_chain.append((model, deployment, provider_id))
                rebind_sources.append(None)
        bound = FallbackChatModel(
            bound_chain,
            self.circuit_breaker,
            self.alert_callback,
            dict(self._config),
        )
        bound._rebind_sources = rebind_sources
        return bound

    async def _retry_with_default_sampling(
        self,
        index: int,
        err: Exception,
        input_data: LanguageModelInput,
        config: dict[str, Any] | None,
        **kwargs: Any,
    ) -> BaseMessage | None:
        """
        Rebind one provider without rejected sampling params and retry it.

        Returns the result on success, or None when this error is not a
        sampling-param rejection, the provider has no pre-bind source, or the
        rebound call failed (callers fall through to normal fallback).
        """
        if unsupported_sampling_param_in_chain(err) is None:
            return None
        source = (
            self._rebind_sources[index] if index < len(self._rebind_sources) else None
        )
        if source is None:
            return None
        src_model, tools, tool_kwargs = source
        provider_id = self.chain[index][2]

        def _rebuild() -> Any:
            stripped = src_model.with_config(
                config={"configurable": dict.fromkeys(DROPPABLE_SAMPLING_PARAMS)}
            )
            return stripped.bind_tools(tools, **tool_kwargs)

        try:
            rebound = await asyncio.to_thread(_rebuild)
            result = await rebound.ainvoke(input_data, config, **kwargs)
            _raise_for_retryable_empty_response(result)
        except Exception:
            LOGGER.exception(
                "Sampling-param rebind retry failed for provider %s", provider_id
            )
            return None
        LOGGER.warning(
            "Provider %s rejected an unsupported sampling parameter with tools "
            "bound; retried with provider-default sampling parameters.",
            provider_id,
        )
        return result

    def with_config(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> FallbackChatModel:
        """Return a new FallbackChatModel with merged config."""
        merged = _merge_config(self._config, config)
        wrapped_chain: list[tuple[Any, str, str]] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "with_config"):
                wrapped_chain.append(
                    (
                        model.with_config(_merge_model_config(model, config), **kwargs),
                        deployment,
                        provider_id,
                    )
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
                result = await ainvoke_dropping_unsupported_params(
                    model, input_data, config, **kwargs
                )
                _raise_for_retryable_empty_response(result)
            except Exception as err:
                rebind_result = await self._retry_with_default_sampling(
                    i, err, input_data, config, **kwargs
                )
                if rebind_result is not None:
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success(provider_id)
                    return rebind_result
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "Model call failed for provider %s (provider %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    _safe_err_summary(err),
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                next_provider = self.chain[i + 1] if i + 1 < len(self.chain) else None
                self.alert_callback(
                    provider_id,
                    err,
                    fallback_to=next_provider[2] if next_provider else None,
                    fallback_deployment=next_provider[1] if next_provider else None,
                )
                continue
            else:
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                return result

        msg = f"All LLM providers failed. Last error: {_safe_err_summary(last_err)}"
        raise HomeAssistantError(msg) from last_err

    async def astream(
        self,
        _input_data: LanguageModelInput,
        _config: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Raise: astream cannot safely retry mid-stream — use ainvoke."""
        msg = "Use ainvoke; astream is not safe with fallback chains"
        raise NotImplementedError(msg)
        yield  # type: ignore[unreachable]


class FallbackVLM:
    """Fallback wrapper for VLM image analysis."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
        alert_callback: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize fallback VLM wrapper."""
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.alert_callback = alert_callback or _default_alert
        self._config = config or {}

    @property
    def config(self) -> dict[str, Any]:
        """Return internal config dict."""
        return self._config

    def with_config(
        self, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> FallbackVLM:
        """Return a new FallbackVLM with config applied to each model."""
        merged = _merge_config(self._config, config)
        wrapped_chain: list[tuple[Any, str, str]] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "with_config"):
                wrapped_chain.append(
                    (
                        model.with_config(_merge_model_config(model, config), **kwargs),
                        deployment,
                        provider_id,
                    )
                )
            else:
                wrapped_chain.append((model, deployment, provider_id))
        return FallbackVLM(
            wrapped_chain,
            self.circuit_breaker,
            self.alert_callback,
            merged,
        )

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
                return await ainvoke_dropping_unsupported_params(
                    model, input_data, config, **kwargs
                )
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                LOGGER.warning(
                    "VLM call failed for provider %s (provider %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(self.chain),
                    _safe_err_summary(err),
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                next_provider = self.chain[i + 1] if i + 1 < len(self.chain) else None
                self.alert_callback(
                    provider_id,
                    err,
                    fallback_to=next_provider[2] if next_provider else None,
                    fallback_deployment=next_provider[1] if next_provider else None,
                )
                continue

        msg = f"All VLM providers failed. Last error: {_safe_err_summary(last_err)}"
        raise HomeAssistantError(msg) from last_err


class FallbackEmbeddings:
    """Fallback wrapper for embedding models."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
        switch_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize fallback embeddings wrapper."""
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.switch_callback = switch_callback
        self._active_provider_id = chain[0][2] if chain else None

    def _ordered_chain(self) -> list[tuple[Any, str, str]]:
        """Return chain with the active provider first."""
        if self._active_provider_id is None:
            return self.chain
        active = [item for item in self.chain if item[2] == self._active_provider_id]
        inactive = [item for item in self.chain if item[2] != self._active_provider_id]
        if inactive and active and active[0][2] != self.chain[0][2]:
            LOGGER.debug(
                "Using sticky embedding provider %s first; configured primary "
                "provider %s will not be retried until the integration is reloaded.",
                active[0][2],
                self.chain[0][2],
            )
        return [*active, *inactive]

    def _record_success(self, provider_id: str) -> None:
        """Record the provider used successfully for future embedding calls."""
        previous_provider_id = self._active_provider_id
        self._active_provider_id = provider_id
        if previous_provider_id and previous_provider_id != provider_id:
            LOGGER.warning(
                "Embedding provider switched from %s to %s. Existing vector "
                "indexes may be stale and should be rebuilt.",
                previous_provider_id,
                provider_id,
            )
            if self.switch_callback:
                self.switch_callback(previous_provider_id, provider_id)

    async def _aembed_documents(
        self, model: Any, texts: list[str]
    ) -> list[list[float]]:
        """Embed documents with provider-specific dimensionality normalization."""
        if isinstance(model, GoogleGenerativeAIEmbeddings):
            return await model.aembed_documents(
                texts, output_dimensionality=EMBEDDING_MODEL_DIMS
            )
        return await model.aembed_documents(texts)

    async def _aembed_query(self, model: Any, text: str) -> list[float]:
        """Embed a query with provider-specific dimensionality normalization."""
        if isinstance(model, GoogleGenerativeAIEmbeddings):
            return (await self._aembed_documents(model, [text]))[0]
        return await model.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents through the fallback chain."""
        last_err: Exception | None = None
        failed_providers: list[str] = []
        chain = self._ordered_chain()
        for i, (model, _deployment, provider_id) in enumerate(chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                result = await self._aembed_documents(model, texts)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                failed_providers.append(provider_id)
                LOGGER.warning(
                    "Embedding call failed for provider %s (provider %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(chain),
                    _safe_err_summary(err),
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue
            else:
                if failed_providers:
                    LOGGER.warning(
                        "Embedding fallback activated: provider %s succeeded "
                        "after failed provider(s): %s",
                        provider_id,
                        ", ".join(failed_providers),
                    )
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                self._record_success(provider_id)
                return result

        msg = (
            f"All embedding providers failed. Last error: {_safe_err_summary(last_err)}"
        )
        raise HomeAssistantError(msg) from last_err

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query through the fallback chain."""
        last_err: Exception | None = None
        failed_providers: list[str] = []
        chain = self._ordered_chain()
        for i, (model, _deployment, provider_id) in enumerate(chain):
            if self.circuit_breaker and not self.circuit_breaker.is_available(
                provider_id
            ):
                continue
            try:
                result = await self._aembed_query(model, text)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                failed_providers.append(provider_id)
                LOGGER.warning(
                    "Embedding query failed for provider %s (provider %d/%d): %s",
                    provider_id,
                    i + 1,
                    len(chain),
                    _safe_err_summary(err),
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue
            else:
                if failed_providers:
                    LOGGER.warning(
                        "Embedding fallback activated: provider %s succeeded "
                        "after failed provider(s): %s",
                        provider_id,
                        ", ".join(failed_providers),
                    )
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                self._record_success(provider_id)
                return result

        msg = (
            f"All embedding providers failed. Last error: {_safe_err_summary(last_err)}"
        )
        raise HomeAssistantError(msg) from last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Raise: synchronous embedding is not supported in async HA context."""
        msg = "Use aembed_documents in async context"
        raise NotImplementedError(msg)

    def embed_query(self, text: str) -> list[float]:
        """Raise: synchronous embedding is not supported in async HA context."""
        msg = "Use aembed_query in async context"
        raise NotImplementedError(msg)
