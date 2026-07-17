"""Utility functions for Home Generative Agent integration."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import re
import secrets
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
import psycopg
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.httpx_client import get_async_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..const import (  # noqa: TID252
    CONF_OLLAMA_URL,
    EMBEDDING_MODEL_DIMS,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_UNAUTHORIZED,
    OLLAMA_BOOL_HINT_TAGS,
    OLLAMA_CATEGORY_URL_KEYS,
    OLLAMA_GPT_EFFORT,
    OLLAMA_OSS_TAG,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Coroutine,
        Mapping,
        MutableMapping,
        Sequence,
    )

    from homeassistant.core import HomeAssistant


LOGGER = logging.getLogger(__name__)


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)

# ---------------------------
# Exceptions
# ---------------------------


class CannotConnectError(Exception):
    """Network/resource unreachable or bad response."""


class InvalidAuthError(Exception):
    """Credentials present but rejected."""


# ---------------------------
# Helpers
# ---------------------------


def dispatch_on_loop(hass: HomeAssistant, signal: str, *args: Any) -> None:
    """Thread-safe dispatcher send; always run on HA's loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # not in any loop
        loop = None
    if loop is hass.loop:
        async_dispatcher_send(hass, signal, *args)
    else:
        hass.loop.call_soon_threadsafe(async_dispatcher_send, hass, signal, *args)


def ensure_http_url(url: str) -> str:
    """Ensure a URL has an explicit scheme."""
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


def _normalized_url(value: Any) -> str | None:
    """Return a cleaned HTTP(S) URL or None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return ensure_http_url(str(value))


def normalize_openai_compatible_base_url(url: str) -> str:
    """
    Return an OpenAI-compatible base URL that ends with /v1.

    The OpenAI SDK appends bare paths (/chat/completions, /embeddings) to the
    base URL, and servers such as llama.cpp only guarantee OpenAI-format
    responses under the /v1 prefix — its bare /embeddings route returns a raw
    JSON list that crashes the SDK parser. Accepts URLs entered with or
    without a trailing /v1.
    """
    base = ensure_http_url(url.strip()).rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def ollama_url_for_category(
    options: Mapping[str, Any],
    category: str,
    *,
    fallback: str | None = None,
) -> str | None:
    """Pick the Ollama URL for a model category, falling back to the global URL."""
    specific_key = OLLAMA_CATEGORY_URL_KEYS.get(category)
    if specific_key:
        url = _normalized_url(options.get(specific_key))
        if url:
            return url

    if fallback:
        return _normalized_url(fallback)

    return _normalized_url(options.get(CONF_OLLAMA_URL))


def configured_ollama_urls(
    options: Mapping[str, Any], *, fallback: str | None = None
) -> list[str]:
    """Return a de-duplicated list of configured Ollama URLs."""
    urls: list[str] = []

    base_url = _normalized_url(options.get(CONF_OLLAMA_URL)) or _normalized_url(
        fallback
    )
    if base_url:
        urls.append(base_url)

    for cat in OLLAMA_CATEGORY_URL_KEYS:
        url = ollama_url_for_category(options, cat, fallback=base_url)
        if url:
            urls.append(url)

    seen: set[str] = set()
    deduped = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


# ---------------------------------------------------------------------------
# Deployment-aware admission control
# ---------------------------------------------------------------------------
# Priority order (highest to lowest):
#   1. Chat (edge)      — marks the full turn active via local_chat_session.
#                         Cancels in-flight Sentinel LLM tasks on entry.
#   2. Video (edge)     — marks video model calls active via local_video_session.
#                         Defers Sentinel LLM tasks for the entire duration:
#                         chat-wait + semaphore-wait + model-call.  This is
#                         intentional — video reserves its slot from entry so
#                         Sentinel cannot interleave with a pending video turn.
#   3. Sentinel (edge)  — calls sentinel_admission before every LLM invocation;
#                         defers if chat OR video is active.
#   Cloud providers     — bypass all local gates.
# ---------------------------------------------------------------------------

_chat_idle = asyncio.Event()
_chat_idle.set()  # "no chat active" by default
_chat_active_count: int = 0

_video_idle = asyncio.Event()
_video_idle.set()  # "no video active" by default
_video_active_count: int = 0

_sentinel_last_run: float = 0.0  # monotonic timestamp of last admitted run
_sentinel_first_defer: float = 0.0
_sentinel_defer_count: int = 0
_sentinel_admissions_deferred_chat: int = 0
_sentinel_admissions_deferred_video: int = 0
_SENTINEL_STARVATION_WARN_S: float = 300.0
_SENTINEL_CANCEL_WAIT_S: float = 2.0
_sentinel_llm_tasks: set[asyncio.Future[Any]] = set()

# Public constant: maximum seconds a Sentinel LLM call waits for chat or video
# to become idle before deferring.  All callers (triage, discovery, explain)
# use the same value so policy changes need only one edit.
SENTINEL_ADMISSION_TIMEOUT_S: float = 2.0


class SentinelLLMDeferredError(Exception):
    """Raised when a background Sentinel LLM call is deferred or interrupted."""

    def __init__(self, category: str, reason: str) -> None:
        """Initialize with the Sentinel LLM category and deferral reason."""
        super().__init__(f"Sentinel {category} {reason}.")


def is_edge_deployment(deployment: str | None) -> bool:
    """Return True for edge (local) deployments subject to resource gating."""
    return deployment == "edge"


@contextlib.asynccontextmanager
async def local_chat_session(
    deployment: str, *, category: str = "chat"
) -> AsyncIterator[None]:
    """
    Mark a local chat turn active for its full duration.

    Clears _chat_idle for the entire turn so Sentinel LLM invocations can
    check for active chat and defer.  No-op for cloud deployments.
    The try/finally guarantees the counter resets on cancellation or exception.
    """
    if not is_edge_deployment(deployment):
        yield
        return
    global _chat_active_count  # noqa: PLW0603
    _chat_active_count += 1
    _chat_idle.clear()
    try:
        await _cancel_active_sentinel_llm_tasks(category)
        LOGGER.debug(
            "Local chat resource gate active (configured_deployment=%s, category=%s).",
            deployment,
            category,
        )
        yield
    finally:
        _chat_active_count -= 1
        if _chat_active_count == 0:
            _chat_idle.set()
        LOGGER.debug(
            "Local chat resource gate ended (configured_deployment=%s, category=%s).",
            deployment,
            category,
        )


@contextlib.asynccontextmanager
async def local_video_session(deployment: str) -> AsyncIterator[None]:
    """
    Mark a local video model call active for its duration.

    Clears _video_idle so Sentinel admission can detect and defer.
    Reference-counted: deferral lifts only when the last concurrent video
    context exits.  No-op when CONF_MODEL_PROVIDER_UNCONTENDED is True
    (caller is responsible for checking) or for cloud deployments.
    The try/finally guarantees the flag resets on cancellation or exception.
    """
    if not is_edge_deployment(deployment):
        yield
        return
    global _video_active_count  # noqa: PLW0603
    _video_active_count += 1
    _video_idle.clear()
    try:
        LOGGER.debug("Video session active (deployment=%s).", deployment)
        yield
    finally:
        _video_active_count -= 1
        if _video_active_count == 0:
            _video_idle.set()
        LOGGER.debug("Video session ended (deployment=%s).", deployment)


def video_is_active() -> bool:
    """Return True if any video model call is currently in progress."""
    return not _video_idle.is_set()


async def wait_for_chat_idle(timeout_s: float) -> bool:
    """Wait for chat to become idle. Returns True when idle, False on timeout."""
    if _chat_idle.is_set():
        return True
    try:
        async with asyncio.timeout(timeout_s):
            await _chat_idle.wait()
    except TimeoutError:
        return False
    else:
        return True


async def _cancel_active_sentinel_llm_tasks(category: str) -> None:
    """Cancel in-flight Sentinel LLM HTTP calls before foreground chat proceeds."""
    tasks = [task for task in _sentinel_llm_tasks if not task.done()]
    if not tasks:
        return
    LOGGER.warning(
        "Cancelling %d active Sentinel LLM call(s) for foreground %s.",
        len(tasks),
        category,
    )
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_SENTINEL_CANCEL_WAIT_S,
        )
    except TimeoutError:
        LOGGER.warning(
            "Sentinel LLM task(s) did not cancel within %.1fs; "
            "GPU contention may persist briefly.",
            _SENTINEL_CANCEL_WAIT_S,
        )


async def sentinel_admission(
    deployment: str,
    *,
    category: str,
    timeout_s: float,
    health_stats: MutableMapping[str, Any] | None = None,
) -> bool:
    """
    Return True if a Sentinel LLM call may proceed.

    Edge: waits up to timeout_s for both chat and video to become idle; returns
    False if either is still active so the caller can defer the pass.
    Cloud: always admitted.
    Tracks last-admitted timestamp and consecutive deferral count for health
    monitoring; logs a WARNING when Sentinel has been starved beyond the
    starvation threshold.
    """
    global _sentinel_first_defer, _sentinel_last_run, _sentinel_defer_count  # noqa: PLW0603
    global _sentinel_admissions_deferred_chat, _sentinel_admissions_deferred_video  # noqa: PLW0603
    if not is_edge_deployment(deployment):
        _sentinel_last_run = time.monotonic()
        _sentinel_first_defer = 0.0
        _sentinel_defer_count = 0
        if health_stats is not None:
            health_stats["sentinel_admission_degraded"] = False
            health_stats["sentinel_admission_degraded_category"] = None
            health_stats["sentinel_admission_consecutive_deferrals"] = 0
            health_stats["sentinel_admission_starved_for_s"] = 0
        return True

    try:
        async with asyncio.timeout(timeout_s):
            # Both events must be set (idle) before Sentinel may proceed.
            await asyncio.gather(_chat_idle.wait(), _video_idle.wait())
    except TimeoutError:
        # Re-evaluate after timeout: both could still be blocking.
        chat_blocking = not _chat_idle.is_set()
        video_blocking = not _video_idle.is_set()
        _sentinel_defer_count += 1
        if chat_blocking:
            _sentinel_admissions_deferred_chat += 1
        if video_blocking:
            _sentinel_admissions_deferred_video += 1
        now = time.monotonic()
        if _sentinel_first_defer <= 0:
            _sentinel_first_defer = now
        last_activity = _sentinel_last_run or _sentinel_first_defer
        starved_for_s = now - last_activity
        degraded = starved_for_s > _SENTINEL_STARVATION_WARN_S
        if health_stats is not None:
            health_stats["sentinel_admission_degraded"] = degraded
            health_stats["sentinel_admission_degraded_category"] = (
                category if degraded else None
            )
            health_stats["sentinel_admission_consecutive_deferrals"] = (
                _sentinel_defer_count
            )
            health_stats["sentinel_admission_starved_for_s"] = int(starved_for_s)
        if degraded:
            LOGGER.warning(
                "Sentinel %s deferred %d consecutive time(s); "
                "last successful run or first deferral %.0fs ago. "
                "Consider reducing chat load or tuning sentinel intervals.",
                category,
                _sentinel_defer_count,
                starved_for_s,
            )
        return False
    _sentinel_last_run = time.monotonic()
    _sentinel_first_defer = 0.0
    _sentinel_defer_count = 0
    if health_stats is not None:
        health_stats["sentinel_admission_degraded"] = False
        health_stats["sentinel_admission_degraded_category"] = None
        health_stats["sentinel_admission_consecutive_deferrals"] = 0
        health_stats["sentinel_admission_starved_for_s"] = 0
    return True


def sentinel_admission_counters() -> dict[str, int]:
    """Return and reset Sentinel admission deferral counters since last call."""
    global _sentinel_admissions_deferred_chat, _sentinel_admissions_deferred_video  # noqa: PLW0603
    result = {
        "deferred_chat": _sentinel_admissions_deferred_chat,
        "deferred_video": _sentinel_admissions_deferred_video,
    }
    _sentinel_admissions_deferred_chat = 0
    _sentinel_admissions_deferred_video = 0
    return result


async def run_sentinel_llm_call(  # noqa: PLR0913
    call_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    deployment: str,
    category: str,
    admission_timeout_s: float,
    call_timeout_s: float,
    health_stats: MutableMapping[str, Any] | None = None,
) -> Any:
    """Run a Sentinel LLM call with chat-priority admission and interruption."""
    if not await sentinel_admission(
        deployment,
        category=category,
        timeout_s=admission_timeout_s,
        health_stats=health_stats,
    ):
        raise SentinelLLMDeferredError(category, "deferred; chat is active")

    task = asyncio.create_task(call_factory())
    _sentinel_llm_tasks.add(task)
    try:
        return await asyncio.wait_for(task, timeout=call_timeout_s)
    except asyncio.CancelledError as err:
        raise SentinelLLMDeferredError(
            category, "interrupted by foreground chat"
        ) from err
    finally:
        _sentinel_llm_tasks.discard(task)


async def run_sentinel_llm_call_in_executor(  # noqa: PLR0913
    call_factory: Callable[[], Any],
    *,
    deployment: str,
    category: str,
    admission_timeout_s: float,
    call_timeout_s: float,
    health_stats: MutableMapping[str, Any] | None = None,
) -> Any:
    """Run a blocking Sentinel LLM call off the event loop."""
    if not await sentinel_admission(
        deployment,
        category=category,
        timeout_s=admission_timeout_s,
        health_stats=health_stats,
    ):
        raise SentinelLLMDeferredError(category, "deferred; chat is active")

    task = asyncio.create_task(asyncio.to_thread(call_factory))
    _sentinel_llm_tasks.add(task)
    try:
        return await asyncio.wait_for(task, timeout=call_timeout_s)
    except asyncio.CancelledError as err:
        raise SentinelLLMDeferredError(
            category, "interrupted by foreground chat"
        ) from err
    finally:
        _sentinel_llm_tasks.discard(task)


async def run_sentinel_model_call(  # noqa: PLR0913
    model: Any,
    messages: Any,
    *,
    deployment: str,
    category: str,
    admission_timeout_s: float,
    call_timeout_s: float,
    health_stats: MutableMapping[str, Any] | None = None,
) -> Any:
    """Run a Sentinel model call, preferring sync ``invoke`` off the event loop."""
    invoke = getattr(model, "invoke", None)
    if callable(invoke):
        return await run_sentinel_llm_call_in_executor(
            lambda: invoke(messages),
            deployment=deployment,
            category=category,
            admission_timeout_s=admission_timeout_s,
            call_timeout_s=call_timeout_s,
            health_stats=health_stats,
        )
    return await run_sentinel_llm_call(
        lambda: model.ainvoke(messages),
        deployment=deployment,
        category=category,
        admission_timeout_s=admission_timeout_s,
        call_timeout_s=call_timeout_s,
        health_stats=health_stats,
    )


async def generate_embeddings(
    emb: Any,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings for the given texts.

    Embedding models are expected to be small enough to run concurrently with
    LLM work, so no chat-priority gate is applied regardless of provider.
    For Gemini embeddings, output_dimensionality is forced to EMBEDDING_MODEL_DIMS
    to match the pgvector index.
    """
    texts_list = [t for t in texts if t]
    if not texts_list:
        return []
    if isinstance(emb, GoogleGenerativeAIEmbeddings):
        return await emb.aembed_documents(
            texts_list, output_dimensionality=EMBEDDING_MODEL_DIMS
        )
    return await emb.aembed_documents(texts_list)


def discover_mobile_notify_service(hass: HomeAssistant) -> str | None:
    """Return the name of a mobile_app notify service if available."""
    # Returns just the service *name* (e.g., "mobile_app_lindos_iphone")
    services = hass.services.async_services().get("notify", {})
    # `services` is a dict mapping service_name -> Service object
    for svc_name in services:
        if svc_name.startswith("mobile_app_"):
            return svc_name
    return None


def list_mobile_notify_services(hass: HomeAssistant) -> list[str]:
    """Return a sorted list like ['notify.mobile_app_xxx', ...]."""
    services = hass.services.async_services().get("notify", {}) or {}
    return sorted(
        f"notify.{name}" for name in services if name.startswith("mobile_app_")
    )


def default_mobile_notify_service(hass: HomeAssistant) -> str | None:
    """Best default mobile_app notify service name or None."""
    services = list_mobile_notify_services(hass)
    return services[0] if services else None


def hash_pin(pin: str, *, salt: str | None = None) -> tuple[str, str]:
    """Return (hash, salt) for a numeric PIN using PBKDF2-HMAC."""
    if not salt:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), salt.encode("utf-8"), 120_000
    )
    return digest.hex(), salt


def verify_pin(pin: str, *, hashed: str, salt: str) -> bool:
    """Compare a PIN against the stored hash+salt."""
    digest, _ = hash_pin(pin, salt=salt)
    return hmac.compare_digest(digest, hashed)


EXCLUSION_ENTRY_MAX_LEN = 256

# Exclusion entries may only use the entity-ID alphabet (lowercase letters,
# digits, underscore, the domain separator dot) plus the two simple glob
# wildcards. Character classes ("[...]") are rejected outright: their syntax
# characters defeat any "has a literal character" check — e.g. "[!.]*.*"
# reads as having literals but matches every valid entity ID — and no
# realistic entity-ID pattern needs them.
_EXCLUSION_ENTRY_ALLOWED_RE = re.compile(r"[a-z0-9_.*?]+\Z")
_EXCLUSION_ENTRY_LITERAL_RE = re.compile(r"[a-z0-9_]")


def valid_exclusion_entry(entry: str) -> bool:
    """
    Return True when *entry* is a usable sentinel exclusion entry.

    Entries are entity IDs or glob patterns over them (``*`` and ``?`` only).
    Four rules keep a security engine from failing open on a typo: the entry
    must use only the entity-ID alphabet plus ``*``/``?`` (no ``[...]``
    character classes — see the comment above), must contain a dot (entity
    IDs are ``domain.object``), must contain at least one literal
    ``[a-z0-9_]`` character (rejecting match-everything spellings like ``*``
    and ``*.*``), and must fit in ``EXCLUSION_ENTRY_MAX_LEN``. Shared by the
    engine option parser and the config-flow field validator so both stay in
    lockstep.
    """
    if "." not in entry or len(entry) > EXCLUSION_ENTRY_MAX_LEN:
        return False
    if _EXCLUSION_ENTRY_ALLOWED_RE.fullmatch(entry) is None:
        return False
    return _EXCLUSION_ENTRY_LITERAL_RE.search(entry) is not None


# ---------------------------
# Health checks
# ---------------------------


async def ollama_healthy(
    hass: HomeAssistant, base_url: str, timeout_s: float = 2.0
) -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    try:
        await validate_ollama_url(hass, base_url, timeout_s)
    except CannotConnectError as err:
        LOGGER.warning(
            "Ollama health check failed (%s): %s", ensure_http_url(base_url), err
        )
        return False
    else:
        return True


async def openai_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Return True if OpenAI API is reachable, False otherwise."""
    if not api_key:
        LOGGER.warning("OpenAI health check skipped: missing API key.")
        return False
    try:
        await validate_openai_key(hass, api_key, timeout_s)
    except (CannotConnectError, InvalidAuthError) as err:
        LOGGER.warning("OpenAI health check failed: %s", err)
        return False
    else:
        return True


async def gemini_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Return True if Gemini API is reachable, False otherwise."""
    if not api_key:
        LOGGER.warning("Gemini health check skipped: missing API key.")
        return False
    try:
        await validate_gemini_key(hass, api_key, timeout_s)
    except (CannotConnectError, InvalidAuthError) as err:
        LOGGER.warning("Gemini health check failed: %s", err)
        return False
    else:
        return True


# ---------------------------
# Validators
# ---------------------------


async def validate_ollama_url(
    hass: HomeAssistant, base_url: str, timeout_s: float = 10.0
) -> None:
    """Validate that the Ollama endpoint is reachable."""
    if not base_url:
        return
    base_url = ensure_http_url(base_url)
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(urljoin(base_url.rstrip("/") + "/", "api/tags"))
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("Ollama connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def list_ollama_models(  # noqa: PLR0911
    hass: HomeAssistant, base_url: str, timeout_s: float = 5.0
) -> list[str]:
    """Return available Ollama model names for the given base URL."""
    if not base_url:
        return []
    base_url = ensure_http_url(base_url)
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(urljoin(base_url.rstrip("/") + "/", "api/tags"))
    except (TimeoutError, httpx.RequestError):
        return []
    if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
        return []
    try:
        payload = resp.json()
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


async def validate_openai_key(
    hass: HomeAssistant, api_key: str, timeout_s: float = 10.0
) -> None:
    """Validate that an OpenAI API key is authorized and reachable."""
    if not api_key:
        return
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("OpenAI connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code == HTTP_STATUS_UNAUTHORIZED:
            raise InvalidAuthError
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def validate_openai_compatible_url(
    hass: HomeAssistant,
    base_url: str,
    api_key: str | None = None,
    timeout_s: float = 10.0,
) -> None:
    """Validate an OpenAI-compatible endpoint by calling its /v1/models path."""
    if not base_url:
        raise CannotConnectError
    url = normalize_openai_compatible_base_url(base_url) + "/models"
    headers: dict[str, str] = {}
    if api_key and api_key != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(url, headers=headers)
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("OpenAI-compatible connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code == HTTP_STATUS_UNAUTHORIZED:
            raise InvalidAuthError
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def openai_compatible_healthy(
    hass: HomeAssistant,
    base_url: str | None,
    api_key: str | None = None,
    timeout_s: float = 2.0,
) -> bool:
    """Return True if an OpenAI-compatible endpoint is reachable, False otherwise."""
    if not base_url:
        LOGGER.warning("OpenAI-compatible health check skipped: missing base URL.")
        return False
    try:
        await validate_openai_compatible_url(hass, base_url, api_key, timeout_s)
    except (CannotConnectError, InvalidAuthError) as err:
        LOGGER.warning("OpenAI-compatible health check failed (%s): %s", base_url, err)
        return False
    else:
        return True


async def validate_gemini_key(
    hass: HomeAssistant, api_key: str, timeout_s: float = 10.0
) -> None:
    """Validate that a Gemini API key is authorized and reachable."""
    if not api_key:
        return
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
            )
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("Gemini connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code == HTTP_STATUS_UNAUTHORIZED:
            raise InvalidAuthError
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def validate_anthropic_key(
    hass: HomeAssistant, api_key: str, timeout_s: float = 10.0
) -> None:
    """Validate that an Anthropic API key is authorized and reachable."""
    if not api_key:
        return
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("Anthropic connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code == HTTP_STATUS_UNAUTHORIZED:
            raise InvalidAuthError
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def anthropic_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Return True if Anthropic API is reachable, False otherwise."""
    if not api_key:
        LOGGER.debug("Anthropic health check skipped: not configured.")
        return False
    try:
        await validate_anthropic_key(hass, api_key, timeout_s)
    except (CannotConnectError, InvalidAuthError) as err:
        LOGGER.warning("Anthropic health check failed: %s", err)
        return False
    else:
        return True


async def validate_face_api_url(
    hass: HomeAssistant, base_url: str, timeout_s: float = 10.0
) -> None:
    """Validate that the face-recognition API endpoint is reachable."""
    if not base_url:
        return
    base_url = ensure_http_url(base_url)
    client = get_async_client(hass)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await client.get(urljoin(base_url.rstrip("/") + "/", "status"))
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("Face API connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def validate_db_uri(hass: HomeAssistant, db_uri: str) -> None:
    """Validate PostgreSQL connection URI by executing a trivial query."""
    if not db_uri:
        return

    parsed = urlparse(db_uri)
    if not parsed.scheme.startswith("postgres"):
        raise CannotConnectError
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise CannotConnectError

    def _psycopg_ping() -> None:
        try:
            with (
                psycopg.connect(db_uri, connect_timeout=5) as conn,
                conn.cursor() as cur,
            ):
                cur.execute("SELECT 1")
                cur.fetchone()
        except psycopg.Error as err:
            raise CannotConnectError from err

    await hass.async_add_executor_job(_psycopg_ping)


# ---------------------------
# Ollama "thinking"/reasoning
# ---------------------------


# 'reasoning' may be boolean or an effort level required by gpt-oss.
type ReasoningValue = bool | Literal["low", "medium", "high"]


@lru_cache(maxsize=128)
def _guess_ollama_reasoning(model: str) -> tuple[bool, ReasoningValue]:
    """
    Zero-network heuristic to decide if a model probably supports reasoning.

    Returns:
        (True, OLLAMA_GPT_EFFORT)  for gpt-oss models
        (True, True)               for boolean-style models
        (False, False)             for others

    """
    m = model.lower()

    # Normalize: drop digest, tag, and any registry/user prefix
    base = m.split("@", 1)[0].split(":", 1)[0].rsplit("/", 1)[-1]

    # Special-case: qwen3* can reason EXCEPT qwen3-vl*
    if base.startswith("qwen3") and not base.startswith("qwen3-vl"):
        return True, True

    if OLLAMA_OSS_TAG in m:
        return True, OLLAMA_GPT_EFFORT

    if any(tag in m for tag in OLLAMA_BOOL_HINT_TAGS):
        return True, True

    return False, False


def reasoning_field(
    *,
    model: str,
    enabled: bool,
) -> dict[str, ReasoningValue]:
    """
    Return {'reasoning': value} for models that support thinking.

    Explicitly passes False when disabled so ChatOllama does not fall back to
    the model's default (which for Qwen3-family is thinking-on).
    """
    supported, value = _guess_ollama_reasoning(model)
    if not supported:
        return {}
    return {"reasoning": value if enabled else False}


def extract_final(raw: str | list[Any], max_chars: int | None = None) -> str:
    """Return plain text with <think> blocks removed."""
    if not raw:
        return ""
    if isinstance(raw, list):
        raw = " ".join(
            part["text"]
            for part in raw
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    # Remove any leaked reasoning
    s = _THINK_BLOCK.sub("", raw)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Char-limit (if specified)
    if max_chars is None:
        return s
    if len(s) <= max_chars:
        return s
    segment = s[:max_chars]
    last_space = segment.rfind(" ")
    if last_space > 0:
        return segment[:last_space].rstrip(" ,;")
    return segment.rstrip(" ,;")


async def gather_store_puts_in_chunks(
    tasks: list[Any],
    *,
    chunk_size: int = 4,
    sleep_s: float = 0.01,
) -> None:
    """Await store.aput coroutines in sequential chunks (embedding provider limits)."""
    if not tasks:
        return
    n = len(tasks)
    for i in range(0, n, chunk_size):
        await asyncio.gather(*tasks[i : i + chunk_size])
        await asyncio.sleep(sleep_s)
