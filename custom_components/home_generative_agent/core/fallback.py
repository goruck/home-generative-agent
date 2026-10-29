"""Fallback wrappers for LLM/VLM/embedding model chains."""

from __future__ import annotations

import asyncio
import importlib
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from homeassistant.exceptions import HomeAssistantError
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..const import EMBEDDING_MODEL_DIMS  # noqa: TID252
from .prompt_cache import adapt_system_message_for_model
from .provider_stall import (
    budget_is_split,
    counts_as_stall_evidence,
    member_timeout,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage

    from .provider_stall import ProviderStallTracker

LOGGER = logging.getLogger(__name__)

# How long an embedding fallback whose build could not be checked is skipped
# before its build is read again (#857).
_VERIFY_RETRY_S = 60.0

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

# The provider SDKs are moving from httpx to httpx2 (openai 3, anthropic 2;
# langchain-ollama and the health checks stay on httpx), and neither
# package's exceptions subclass the other's, so match both (#878).
_HTTP_TRANSPORT_ERRORS: tuple[type[Exception], ...] = ()
_HTTP_TIMEOUTS: tuple[type[Exception], ...] = ()
_HTTP_POOL_TIMEOUTS: tuple[type[Exception], ...] = ()
for _http_module in ("httpx", "httpx2"):
    try:
        _http = importlib.import_module(_http_module)
    except ImportError:
        continue
    _HTTP_TRANSPORT_ERRORS += (_http.TransportError,)
    _HTTP_TIMEOUTS += (_http.TimeoutException,)
    _HTTP_POOL_TIMEOUTS += (_http.PoolTimeout,)
FALLBACK_RETRYABLE_EXCEPTIONS += _HTTP_TRANSPORT_ERRORS

try:
    from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
    OpenAIBadRequestError = None  # type: ignore[assignment, misc]

# Sampling parameters that can be safely omitted when a model rejects them
# (e.g. OpenAI reasoning-style models only accept the default temperature=1,
# and non-reasoning models reject reasoning_effort outright). Each name must
# match a registered ConfigurableField id so the retry can null it.
DROPPABLE_SAMPLING_PARAMS = frozenset({"temperature", "top_p", "reasoning_effort"})

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


_HTTP_NOT_FOUND = 404
# How far down __cause__ to look: provider wrappers re-raise the SDK's error
# once or twice with ``raise ... from`` (langchain-google-genai wraps Gemini's).
_MAX_CAUSE_DEPTH = 5


def is_model_not_found(err: BaseException) -> bool:
    """
    Return True if the provider answered 404: it does not have this model.

    Ollama says 'model "x" not found, try pulling it first'; OpenAI,
    Anthropic and Gemini return their own 404s for an unknown model name, and
    an OpenAI-compatible server at the wrong path does the same. Each SDK
    carries the status differently (``status_code``, or ``code`` for Google),
    and some wrappers keep the SDK error only as the cause, so the causes are
    walked. ``__context__`` is not: an error raised while some caller was
    handling a 404 (a rejected key, say) is not a missing model.
    """
    seen: BaseException | None = err
    for _ in range(_MAX_CAUSE_DEPTH):
        if seen is None:
            return False
        for attr in ("status_code", "code"):
            value = getattr(seen, attr, None)
            # HTTPStatus and plain ints compare equal to 404; grpc and str
            # codes do not.
            if isinstance(value, int) and value == _HTTP_NOT_FOUND:
                return True
        seen = seen.__cause__
    return False


def _log_provider_failure(
    kind: str, provider_id: str, position: int, total: int, err: Exception
) -> None:
    """
    Log a provider the chain is moving past.

    A missing model is a configuration problem rather than an outage, so it
    is logged at ERROR and says so: the chain still moves on (#752, #759), and
    this line is how a typo or an unpulled model on the primary stays visible
    while a fallback covers for it.
    """
    if is_model_not_found(err):
        LOGGER.error(
            "%s failed for provider %s (provider %d/%d): the provider "
            "does not have the configured model; pull it or fix the model "
            "name. Trying the next provider. %s",
            kind,
            provider_id,
            position,
            total,
            _safe_err_summary(err),
        )
        return
    LOGGER.warning(
        "%s failed for provider %s (provider %d/%d): %s",
        kind,
        provider_id,
        position,
        total,
        _safe_err_summary(err),
    )


# SDK timeout errors that do not subclass TimeoutError: openai/anthropic
# APITimeoutError (an APIConnectionError) and google-api-core's
# DeadlineExceeded.
_TIMEOUT_ERROR_NAMES = frozenset({"APITimeoutError", "DeadlineExceeded"})


def is_timeout_error(err: BaseException) -> bool:
    """Return True if the provider took too long to answer (#828)."""
    seen: BaseException | None = err
    for _ in range(_MAX_CAUSE_DEPTH):
        if seen is None:
            return False
        if isinstance(seen, TimeoutError) or type(seen).__name__ in (
            _TIMEOUT_ERROR_NAMES
        ):
            return True
        if isinstance(seen, _HTTP_TIMEOUTS):
            return True
        seen = seen.__cause__
    return False


def _is_retryable(err: Exception) -> bool:
    """Return True if the exception should trigger a fallback attempt."""
    if isinstance(err, FALLBACK_RETRYABLE_EXCEPTIONS):
        return True
    if is_timeout_error(err):
        return True
    if is_model_not_found(err):
        # The next provider may have the model; a missing one at any position
        # moves the chain on and is recorded, so the circuit breaker can skip
        # it (#752, #759). _log_provider_failure keeps it loud.
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


class EmbeddingsUnavailableError(HomeAssistantError):
    """
    No embedding provider can answer for the memory store (#751, #857).

    Raised when none could be built at setup, and by the chain when the only
    members it could try are held until their build is checked: memory
    search falls back to recent memories, and a re-embed stops with every
    vector kept.
    """


class ModelChainTimeoutError(HomeAssistantError, TimeoutError):
    """
    Every provider tried timed out.

    Still a TimeoutError, so callers keep their timeout handling (triage's
    timeout reason, the video analyzer's timeout count) now that the timeout
    fires inside the chain instead of around it (#828).
    """


class ProviderStalledError(HomeAssistantError):
    """
    Every usable provider in the chain is stalled; nothing was called.

    Not a TimeoutError: the call failed at once, and its message names the
    stalled provider so the user is told what is wrong (#828).
    """


_PROBE_PROMPT = "Reply with OK."


def _probe_call(model: Any) -> Callable[[], Awaitable[Any]]:
    """
    Return the stall probe for a provider: a tiny real request.

    It goes through the same sampling-param drop as live calls, so a model
    that rejects a configured parameter can still pass its probe.
    """

    def _probe() -> Awaitable[Any]:
        return ainvoke_dropping_unsupported_params(
            model, [HumanMessage(content=_PROBE_PROMPT)]
        )

    return _probe


def _chain_failed_error(kind: str, last_err: Exception | None) -> HomeAssistantError:
    """Return the error for a chain where every provider tried failed."""
    msg = f"All {kind} providers failed. Last error: {_safe_err_summary(last_err)}"
    if last_err is not None and is_timeout_error(last_err):
        return ModelChainTimeoutError(msg)
    return HomeAssistantError(msg)


def _chain_error(
    chain: list[tuple[Any, str, str]], kind: str, last_err: Exception | None
) -> Exception:
    """
    Return the error to raise when no member answered.

    A lone provider's own error, as before every model was wrapped (#828):
    callers keep its type and message. A real chain wraps the last error.
    """
    if len(chain) == 1 and last_err is not None:
        return last_err
    error = _chain_failed_error(kind, last_err)
    error.__cause__ = last_err
    return error


def _callable_members(
    chain: list[tuple[Any, str, str]],
    circuit_breaker: CircuitBreaker | None,
    stall_tracker: ProviderStallTracker | None,
) -> list[int]:
    """Return the chain positions to try: not circuit-broken, not stalled."""
    members: list[int] = []
    for i, (_model, _deployment, provider_id) in enumerate(chain):
        if circuit_breaker and not circuit_breaker.is_available(provider_id):
            LOGGER.debug("Provider %s is circuit-broken; skipping.", provider_id)
            continue
        if stall_tracker and stall_tracker.is_stalled(provider_id):
            LOGGER.debug("Provider %s is stalled; skipping.", provider_id)
            continue
        members.append(i)
    return members


def _stalled_chain_error(
    chain: list[tuple[Any, str, str]],
    stall_tracker: ProviderStallTracker | None,
) -> ProviderStalledError | None:
    """
    Return the error for a chain whose only usable providers are stalled.

    Raised at once instead of waiting out another timeout, and it names the
    provider so the reply says what is wrong (#828).
    """
    if stall_tracker is None:
        return None
    names = [
        stall_tracker.name(provider_id)
        for _model, _deployment, provider_id in chain
        if stall_tracker.is_stalled(provider_id)
    ]
    if not names:
        return None
    msg = (
        f"Model provider {', '.join(names)} stopped answering. It is checked "
        "every minute and used again as soon as it answers."
    )
    return ProviderStalledError(msg)


def _budget_used_up(timeout_s: float | None) -> bool:
    """Return True when the caller's deadline leaves no time for a provider."""
    return timeout_s is not None and timeout_s <= 0


def _deadline(timeout_s: float | None) -> float | None:
    """Return the loop time a member's share of the budget runs out, or None."""
    if timeout_s is None:
        return None
    return asyncio.get_running_loop().time() + timeout_s


def _is_client_pool_timeout(err: BaseException) -> bool:
    """Return True for a timeout waiting on our own connection pool."""
    seen: BaseException | None = err
    for _ in range(_MAX_CAUSE_DEPTH):
        if seen is None:
            return False
        if isinstance(seen, _HTTP_POOL_TIMEOUTS):
            return True
        seen = seen.__cause__
    return False


def _probe_model(wrapper: FallbackChatModel | FallbackVLM, index: int) -> Any:
    """
    Return the model a stall probe should call for one chain member.

    A tool-bound chat member probes its pre-bind source: parameters baked
    into the binding cannot be dropped, so a model that rejects one could
    otherwise never pass its probe.
    """
    if isinstance(wrapper, FallbackChatModel) and index < len(
        wrapper._rebind_sources  # noqa: SLF001
    ):
        source = wrapper._rebind_sources[index]  # noqa: SLF001
        if source is not None:
            return source[0]
    return wrapper.chain[index][0]


def _handle_member_failure(  # noqa: PLR0913
    wrapper: FallbackChatModel | FallbackVLM,
    kind: str,
    members: list[int],
    pos: int,
    err: Exception,
    *,
    started: float,
    deadline: float | None,
    split: bool,
) -> bool:
    """
    Log and count a failed member; announce the provider tried next.

    Returns True when the call should stop here: the budget was not split and
    this member used it up, so the next provider would get only the margin.
    """
    i = members[pos]
    _model, _deployment, provider_id = wrapper.chain[i]
    lone = len(wrapper.chain) == 1
    now = asyncio.get_running_loop().time()
    spent = not split and deadline is not None and now >= deadline
    if lone:
        # A lone provider's caller reports the error, as it did before every
        # model was wrapped (#828); there is no next provider to announce.
        LOGGER.debug("%s failed for provider %s: %s", kind, provider_id, err)
    else:
        _log_provider_failure(kind, provider_id, i + 1, len(wrapper.chain), err)
    if wrapper.circuit_breaker:
        wrapper.circuit_breaker.record_failure(provider_id)
    # Only a long wait is evidence of a hang: a slow model missing a short
    # Sentinel deadline, or our own connection pool timing out, is not a
    # stalled provider.
    if (
        wrapper.stall_tracker is not None
        and is_timeout_error(err)
        and not _is_client_pool_timeout(err)
        and counts_as_stall_evidence(now - started)
    ):
        wrapper.stall_tracker.record_timeout(
            provider_id, _probe_call(_probe_model(wrapper, i))
        )
    if lone or spent:
        return spent
    next_provider = wrapper.chain[members[pos + 1]] if pos + 1 < len(members) else None
    wrapper.alert_callback(
        provider_id,
        err,
        fallback_to=next_provider[2] if next_provider else None,
        fallback_deployment=next_provider[1] if next_provider else None,
    )
    return False


def _record_answer(
    circuit_breaker: CircuitBreaker | None,
    stall_tracker: ProviderStallTracker | None,
    provider_id: str,
) -> None:
    """Clear failure state for a provider that answered."""
    if circuit_breaker:
        circuit_breaker.record_success(provider_id)
    if stall_tracker:
        stall_tracker.record_success(provider_id)


class FallbackChatModel:
    """Chainable chat model that falls back through a list of providers."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
        alert_callback: Any | None = None,
        config: dict[str, Any] | None = None,
        stall_tracker: ProviderStallTracker | None = None,
    ) -> None:
        """
        Initialize fallback chat model.

        :param chain: List of (model_instance, deployment, provider_id).
        :param circuit_breaker: Optional CircuitBreaker instance.
        :param alert_callback: Callable(provider_id, error, fallback_to=None).
        :param config: Internal config dict (used for with_config chaining).
        :param stall_tracker: Shared per-provider stall state (#828).
        """
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.stall_tracker = stall_tracker
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

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tools_for_member: Callable[[Any, list[Any]], list[Any]] | None = None,
        **kwargs: Any,
    ) -> FallbackChatModel:
        """
        Return a new FallbackChatModel with tools bound to each model.

        Members can be different providers with different schema rules, so
        ``tools_for_member(model, tools)`` may reshape the list per member.
        """
        bound_chain: list[tuple[Any, str, str]] = []
        rebind_sources: list[tuple[Any, list[Any], dict[str, Any]] | None] = []
        for model, deployment, provider_id in self.chain:
            if hasattr(model, "bind_tools"):
                member_tools = (
                    tools_for_member(model, tools) if tools_for_member else tools
                )
                bound_chain.append(
                    (model.bind_tools(member_tools, **kwargs), deployment, provider_id)
                )
                rebind_sources.append((model, list(member_tools), dict(kwargs)))
            else:
                bound_chain.append((model, deployment, provider_id))
                rebind_sources.append(None)
        bound = FallbackChatModel(
            bound_chain,
            self.circuit_breaker,
            self.alert_callback,
            dict(self._config),
            self.stall_tracker,
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
        sampling-param rejection or the provider has no pre-bind source. A
        rebound call that fails raises its own error, so the chain handles
        what actually happened (a timeout counts toward a stall) instead of
        the stale rejection.
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

        rebound = await asyncio.to_thread(_rebuild)
        result = await rebound.ainvoke(
            adapt_system_message_for_model(rebound, input_data), config, **kwargs
        )
        _raise_for_retryable_empty_response(result)
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
            self.stall_tracker,
        )

    async def ainvoke(
        self,
        input_data: LanguageModelInput,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        Invoke the chat model chain, falling back on retryable failures.

        Each provider gets its share of the caller's budget (see
        ``member_timeout``), so a hung one times out here, is counted toward a
        stall, and the next provider still has time to answer (#828).
        """
        last_err: Exception | None = None
        members = _callable_members(
            self.chain, self.circuit_breaker, self.stall_tracker
        )
        if not members and (
            stalled := _stalled_chain_error(self.chain, self.stall_tracker)
        ):
            raise stalled
        split = budget_is_split()
        for pos, i in enumerate(members):
            model, _deployment, provider_id = self.chain[i]
            timeout_s = member_timeout(last=pos == len(members) - 1)
            if _budget_used_up(timeout_s):
                break
            # The sampling-param retry below shares this member's deadline.
            deadline = _deadline(timeout_s)
            started = asyncio.get_running_loop().time()
            try:
                async with asyncio.timeout_at(deadline):
                    result = await ainvoke_dropping_unsupported_params(
                        # Members can be different providers; shape the system
                        # message for the one actually being called.
                        model,
                        adapt_system_message_for_model(model, input_data),
                        config,
                        **kwargs,
                    )
                _raise_for_retryable_empty_response(result)
            except Exception as err:
                failure: Exception = err
                try:
                    async with asyncio.timeout_at(deadline):
                        rebind_result = await self._retry_with_default_sampling(
                            i, err, input_data, config, **kwargs
                        )
                except Exception as retry_err:  # noqa: BLE001
                    # The rebound request's own failure, not the rejection
                    # that prompted it (#828).
                    rebind_result = None
                    failure = retry_err
                if rebind_result is not None:
                    _record_answer(
                        self.circuit_breaker, self.stall_tracker, provider_id
                    )
                    return rebind_result
                last_err = failure
                if not _is_retryable(failure):
                    if failure is err:
                        raise
                    raise failure from err
                if _handle_member_failure(
                    self,
                    "Model call",
                    members,
                    pos,
                    failure,
                    started=started,
                    deadline=deadline,
                    split=split,
                ):
                    break
                continue
            else:
                _record_answer(self.circuit_breaker, self.stall_tracker, provider_id)
                return result

        raise _chain_error(self.chain, "LLM", last_err)

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
        stall_tracker: ProviderStallTracker | None = None,
    ) -> None:
        """Initialize fallback VLM wrapper."""
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.stall_tracker = stall_tracker
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
            self.stall_tracker,
        )

    async def ainvoke(
        self,
        input_data: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke the VLM chain, falling back on retryable failures (see chat)."""
        last_err: Exception | None = None
        members = _callable_members(
            self.chain, self.circuit_breaker, self.stall_tracker
        )
        if not members and (
            stalled := _stalled_chain_error(self.chain, self.stall_tracker)
        ):
            raise stalled
        split = budget_is_split()
        for pos, i in enumerate(members):
            model, _deployment, provider_id = self.chain[i]
            timeout_s = member_timeout(last=pos == len(members) - 1)
            if _budget_used_up(timeout_s):
                break
            deadline = _deadline(timeout_s)
            started = asyncio.get_running_loop().time()
            try:
                async with asyncio.timeout_at(deadline):
                    result = await ainvoke_dropping_unsupported_params(
                        model, input_data, config, **kwargs
                    )
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                if _handle_member_failure(
                    self,
                    "VLM call",
                    members,
                    pos,
                    err,
                    started=started,
                    deadline=deadline,
                    split=split,
                ):
                    break
                continue
            else:
                # The breaker's success was never recorded for VLM calls; the
                # stall tracker needs it to clear a provider that answers.
                _record_answer(self.circuit_breaker, self.stall_tracker, provider_id)
                return result

        raise _chain_error(self.chain, "VLM", last_err)


def with_member_configurable(
    model: Any,
    configurable_for: Callable[[Any, str], dict[str, Any]],
    deployment: str,
) -> Any:
    """
    Bind per-call configurable values, member by member for a chain.

    ``configurable_for(member, member_deployment)`` returns the values for one
    model. A chain wrapper's own ``config`` is empty, so values read from it
    would miss each member's settings (its model name, whether it has a
    ``reasoning`` field); this reads and binds each member's own instead.
    A bare model uses *deployment*.
    """
    if isinstance(model, (FallbackChatModel, FallbackVLM)):
        chain = [
            (
                member.with_config(
                    config={"configurable": configurable_for(member, member_deployment)}
                ),
                member_deployment,
                provider_id,
            )
            for member, member_deployment, provider_id in model.chain
        ]
        if isinstance(model, FallbackVLM):
            return FallbackVLM(
                chain,
                model.circuit_breaker,
                model.alert_callback,
                dict(model.config),
                model.stall_tracker,
            )
        wrapped = FallbackChatModel(
            chain,
            model.circuit_breaker,
            model.alert_callback,
            dict(model.config),
            model.stall_tracker,
        )
        # Keep the sampling-param rebind working: each pre-bind source gets
        # the values its bound member got.
        wrapped._rebind_sources = [  # noqa: SLF001
            None
            if source is None
            else (
                source[0].with_config(
                    config={
                        "configurable": configurable_for(
                            source[0], model.chain[index][1]
                        )
                    }
                ),
                source[1],
                source[2],
            )
            for index, source in enumerate(model._rebind_sources)  # noqa: SLF001
        ]
        return wrapped
    return model.with_config(
        config={"configurable": configurable_for(model, deployment)}
    )


class FallbackEmbeddings:
    """Fallback wrapper for embedding models."""

    def __init__(
        self,
        chain: list[tuple[Any, str, str]],
        circuit_breaker: CircuitBreaker | None = None,
        switch_callback: Callable[[str, str], None] | None = None,
        removed_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """
        Initialize fallback embeddings wrapper.

        ``switch_callback(previous, current)`` runs when a call succeeds on
        another provider than the last one; ``removed_callback(removed,
        current)`` when `remove_provider` drops the active one (#860), where
        a "fell back" notice would be wrong.
        """
        self.chain = chain
        self.circuit_breaker = circuit_breaker
        self.switch_callback = switch_callback
        self.removed_callback = removed_callback
        self._active_provider_id = chain[0][2] if chain else None
        # Providers that answered an embedding call: the store embeds a save
        # and a search alike, so any of them may have written vectors (#860).
        # In memory only; a member held by `hold_until_verified` answers
        # nothing before its build is checked, so a reload loses no write of
        # an unchecked build (#857).
        self.served: set[str] = set()
        # Members whose build nobody compared with the store's: skipped until
        # the verifier admits them (#857).
        self._unverified: set[str] = set()
        self._verifier: Callable[[str], Awaitable[bool]] | None = None
        self._admitted_callback: Callable[[str], None] | None = None
        self._verify_locks: dict[str, asyncio.Lock] = {}
        self._verify_after: dict[str, float] = {}
        # Admitted against the store's record, the primary's build unknown.
        self._by_record: set[str] = set()
        # Bumped each time a member is held again, for the calls on it.
        self._admissions: dict[str, int] = {}

    def _is_member(self, provider_id: str) -> bool:
        return any(item[2] == provider_id for item in self.chain)

    def hold_until_verified(
        self,
        provider_ids: Iterable[str],
        verifier: Callable[[str], Awaitable[bool]],
        *,
        admitted_callback: Callable[[str], None] | None = None,
    ) -> None:
        """
        Keep `provider_ids` from answering until `verifier` admits each (#857).

        For Ollama embedding fallbacks whose build setup could not compare
        with the primary's: one holding another build would write its vectors
        into the store unnoticed. Before such a member's first call the
        verifier reads its build; True admits it for good, False skips it for
        this call and, unless it was removed meanwhile, for the next
        `_VERIFY_RETRY_S`. `admitted_callback(provider_id)` runs once per
        member each time it is admitted, whoever admits it: again after
        `hold_again`, so it must not mind a second call.
        """
        self._verifier = verifier
        self._admitted_callback = admitted_callback
        self._unverified.update(p for p in provider_ids if self._is_member(p))

    def verify_provider(self, provider_id: str, *, by_record: bool = False) -> None:
        """
        Admit a held member whose build was found to be the store's (#857).

        `by_record`: compared with the build the store's record names, the
        primary's being unknown; `revoke_record_admissions` holds it again
        once the primary's build is known.
        """
        if provider_id not in self._unverified:
            return
        self._unverified.discard(provider_id)
        if by_record:
            self._by_record.add(provider_id)
        else:
            self._by_record.discard(provider_id)
        if self._admitted_callback is not None and self._is_member(provider_id):
            self._admitted_callback(provider_id)

    def revoke_record_admissions(self) -> None:
        """Hold again the members admitted against the store's record (#857)."""
        for provider_id in tuple(self._by_record):
            self.hold_again(provider_id)
        self._by_record.clear()

    def hold_again(self, provider_id: str) -> None:
        """
        Hold an admitted member again until its build is checked anew (#857).

        For one admitted against a reference that was since replaced (the
        store's record, before the primary's build was known) and whose build
        cannot be read now to compare with the new one.
        """
        if self._verifier is None or not self._is_member(provider_id):
            return
        self._unverified.add(provider_id)
        self._verify_after.pop(provider_id, None)
        # A call already on it discards its answer (see `_held_meanwhile`).
        self._admissions[provider_id] = self._admissions.get(provider_id, 0) + 1

    @property
    def unverified(self) -> frozenset[str]:
        """Members held out until their build is checked."""
        return frozenset(self._unverified)

    async def _admitted(self, provider_id: str) -> bool:
        """Return True if `provider_id` may answer; check a held one's build first."""
        if provider_id not in self._unverified:
            return True
        if monotonic() < self._verify_after.get(provider_id, 0.0):
            # Refused a moment ago: no wait on a check in flight either.
            return False
        lock = self._verify_locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            if provider_id not in self._unverified:
                return self._is_member(provider_id)
            if self._verifier is None or monotonic() < self._verify_after.get(
                provider_id, 0.0
            ):
                return False
            try:
                admitted = await self._verifier(provider_id)
            except Exception as err:  # noqa: BLE001 - held, never let in blind
                LOGGER.warning(
                    "Could not check the build of embedding provider %s (%s); it "
                    "stays held.",
                    provider_id,
                    _safe_err_summary(err),
                )
                admitted = False
            if admitted and self._is_member(provider_id):
                self.verify_provider(provider_id)
                return True
            self._verify_after[provider_id] = monotonic() + _VERIFY_RETRY_S
            return False

    def _is_open(self, provider_id: str) -> bool:
        """
        Return True if the circuit breaker holds `provider_id` out of a call.

        A chain down to one member is the unwrapped provider setup would have
        built (#860): it is tried every time, breaker or not, so a removal
        does not leave a recovered primary skipped for the cooldown.
        """
        return (
            len(self.chain) > 1
            and self.circuit_breaker is not None
            and not self.circuit_breaker.is_available(provider_id)
        )

    @staticmethod
    def _all_failed(
        last_err: Exception | None, skipped: list[str], held: list[str]
    ) -> str:
        held_reason = (
            "the build of the model on "
            + ", ".join(held)
            + " is not yet known to match the memory store's"
        )
        if last_err is not None:
            return (
                "All embedding providers failed. Last error: "
                f"{_safe_err_summary(last_err)}"
                + (f"; not tried because {held_reason}" if held else "")
            )
        reasons = []
        if skipped:
            reasons.append("the circuit breaker is open for " + ", ".join(skipped))
        if held:
            reasons.append(held_reason)
        if reasons:
            return "No embedding provider was tried: " + "; ".join(reasons)
        return "No embedding provider is left in the chain"

    @classmethod
    def _chain_error(
        cls, last_err: Exception | None, skipped: list[str], held: list[str]
    ) -> HomeAssistantError:
        """Return the error for a call no member answered."""
        msg = cls._all_failed(last_err, skipped, held)
        if held or (last_err is None and not skipped):
            # A member held until its build is checked, or none left: the
            # store's embeddings are unavailable, not failing.
            return EmbeddingsUnavailableError(msg)
        return HomeAssistantError(msg)

    def _held_meanwhile(self, label: str, provider_id: str, admission: int) -> bool:
        """Return True, logging it, if the member was held again meanwhile (#857)."""
        if self._admissions.get(provider_id, 0) == admission:
            return False
        LOGGER.warning(
            "%s on embedding provider %s discarded: it was held again while "
            "the call ran, until its build is checked anew.",
            label,
            provider_id,
        )
        return True

    def _removed_meanwhile(self, label: str, provider_id: str) -> bool:
        """Return True, logging it, if the provider left the chain meanwhile (#860)."""
        if self._is_member(provider_id):
            return False
        LOGGER.warning(
            "%s: provider %s was removed from the embedding chain while it "
            "answered; its result is discarded.",
            label,
            provider_id,
        )
        return True

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

    def remove_provider(self, provider_id: str) -> bool:
        """
        Drop a provider from the chain; True if it was in it.

        For an embedding fallback found, after setup, to hold another build
        of the model (#860). If it was the active provider, the chain goes
        back to its first remaining member and the removed callback runs, so
        the tool index its vectors filled is rebuilt. A call in flight on it
        skips it from here on and discards an answer it already gave.
        """
        kept = [item for item in self.chain if item[2] != provider_id]
        if len(kept) == len(self.chain):
            return False
        self.chain = kept
        self._unverified.discard(provider_id)
        self._by_record.discard(provider_id)
        if self._active_provider_id == provider_id:
            self._active_provider_id = kept[0][2] if kept else None
            if self._active_provider_id is not None and self.removed_callback:
                self.removed_callback(provider_id, self._active_provider_id)
        return True

    def _record_success(self, provider_id: str) -> None:
        """Record the provider used successfully for future embedding calls."""
        self.served.add(provider_id)
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
        skipped: list[str] = []
        held: list[str] = []
        chain = self._ordered_chain()
        for i, (model, _deployment, provider_id) in enumerate(chain):
            # The chain can lose a member while this call runs (#860).
            if not self._is_member(provider_id):
                continue
            if self._is_open(provider_id):
                skipped.append(provider_id)
                continue
            if not await self._admitted(provider_id):
                held.append(provider_id)
                continue
            admission = self._admissions.get(provider_id, 0)
            try:
                result = await self._aembed_documents(model, texts)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                failed_providers.append(provider_id)
                _log_provider_failure(
                    "Embedding call", provider_id, i + 1, len(chain), err
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue
            else:
                if self._removed_meanwhile("Embedding call", provider_id):
                    continue
                if self._held_meanwhile("Embedding call", provider_id, admission):
                    held.append(provider_id)
                    continue
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

        raise self._chain_error(last_err, skipped, held) from last_err

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a query through the fallback chain."""
        last_err: Exception | None = None
        failed_providers: list[str] = []
        skipped: list[str] = []
        held: list[str] = []
        chain = self._ordered_chain()
        for i, (model, _deployment, provider_id) in enumerate(chain):
            if not self._is_member(provider_id):
                continue
            if self._is_open(provider_id):
                skipped.append(provider_id)
                continue
            if not await self._admitted(provider_id):
                held.append(provider_id)
                continue
            admission = self._admissions.get(provider_id, 0)
            try:
                result = await self._aembed_query(model, text)
            except Exception as err:
                last_err = err
                if not _is_retryable(err):
                    raise
                failed_providers.append(provider_id)
                _log_provider_failure(
                    "Embedding query", provider_id, i + 1, len(chain), err
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)
                continue
            else:
                if self._removed_meanwhile("Embedding query", provider_id):
                    continue
                if self._held_meanwhile("Embedding query", provider_id, admission):
                    held.append(provider_id)
                    continue
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

        raise self._chain_error(last_err, skipped, held) from last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Raise: synchronous embedding is not supported in async HA context."""
        msg = "Use aembed_documents in async context"
        raise NotImplementedError(msg)

    def embed_query(self, text: str) -> list[float]:
        """Raise: synchronous embedding is not supported in async HA context."""
        msg = "Use aembed_query in async context"
        raise NotImplementedError(msg)
