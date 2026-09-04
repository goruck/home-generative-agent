"""
Shared runtime for the OpenAI-backed audio platforms (STT and TTS).

Both platforms read the same subentry credential shape, written by the shared
flow helper (``flows/openai_compatible_endpoint.py``)::

    openai -> {"api_key": str | None, <provider_id_key>: str | None}
    local  -> {"base_url": "<...>/v1", "api_key": str | None, ...}

and turn it into one OpenAI SDK client. This module is the single owner of the
rules for that: a linked OpenAI model provider is authoritative for the key, a
keyless local server gets a placeholder key plus a stripped Authorization
header, and the client is built on Home Assistant's shared httpx client with a
pinned timeout, no retries, and a cache keyed on the two configurable inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from homeassistant.helpers.httpx_client import get_async_client
from openai import AsyncOpenAI, Omit

from ..const import (  # noqa: TID252
    CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL,
    LOCAL_KEYLESS_API_KEY,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

# Connect timeout shared by both platforms; the read timeout is per platform.
CONNECT_TIMEOUT_S = 5.0


class OpenAIConnectionError(Exception):
    """The subentry cannot produce a usable connection (message says why)."""


_MISSING_BASE_URL = "base URL missing"
_MISSING_API_KEY = "API key missing"


@dataclass(frozen=True, slots=True)
class OpenAIConnection:
    """Resolved credentials for one request."""

    api_key: str
    base_url: str | None
    keyless: bool

    def apply_to_request(self, request: dict[str, Any]) -> None:
        """
        Strip the Authorization header for a keyless local server.

        The placeholder key exists only because the SDK constructor refuses an
        empty one (openai>=2.45); the SDK's ``Omit`` sentinel drops the header
        the client would otherwise add from it, so the wire shape matches the
        flow's keyless validation request.
        """
        if self.keyless:
            request["extra_headers"] = {"Authorization": Omit()}


def load_model_settings(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the subentry's ``model`` mapping as a plain dict."""
    model_data = data.get("model", {})
    return dict(model_data) if isinstance(model_data, Mapping) else {}


def _settings(data: Mapping[str, Any]) -> dict[str, Any]:
    settings = data.get("settings", {})
    return dict(settings) if isinstance(settings, Mapping) else {}


def resolve_openai_api_key(
    entry: ConfigEntry, data: Mapping[str, Any], *, provider_id_key: str
) -> str | None:
    """
    Return the API key for an OpenAI provider subentry, or ``None``.

    A linked OpenAI model-provider subentry is authoritative: its key is used
    and there is never a fallback to the platform-level key, which the flow
    blanks out when linking. Without a link, the platform-level key is used.
    """
    settings = _settings(data)
    provider_id = settings.get(provider_id_key)
    if provider_id:
        provider = entry.subentries.get(provider_id)
        if provider and provider.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER:
            provider_settings = provider.data.get("settings", {})
            if isinstance(provider_settings, Mapping):
                provider_key = dict(provider_settings).get("api_key")
                if isinstance(provider_key, str) and provider_key:
                    return provider_key
                return None
    api_key = settings.get("api_key")
    return api_key if isinstance(api_key, str) and api_key else None


def resolve_openai_connection(
    entry: ConfigEntry,
    provider_type: str,
    data: Mapping[str, Any],
    *,
    provider_id_key: str,
) -> OpenAIConnection:
    """
    Return the connection for a provider subentry.

    ``local`` requires the stored base URL and substitutes the keyless
    placeholder when no key is configured; ``openai`` requires a key (own or
    linked) and uses the SDK's default endpoint. Raises
    ``OpenAIConnectionError`` with a short reason when the subentry is unusable.
    """
    if provider_type == "local":
        settings = _settings(data)
        base_url = settings.get(CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL)
        if not isinstance(base_url, str) or not base_url:
            raise OpenAIConnectionError(_MISSING_BASE_URL)
        configured_key = settings.get("api_key")
        if isinstance(configured_key, str) and configured_key:
            return OpenAIConnection(configured_key, base_url, keyless=False)
        return OpenAIConnection(LOCAL_KEYLESS_API_KEY, base_url, keyless=True)
    api_key = resolve_openai_api_key(entry, data, provider_id_key=provider_id_key)
    if not api_key:
        raise OpenAIConnectionError(_MISSING_API_KEY)
    return OpenAIConnection(api_key, None, keyless=False)


class OpenAIClientCache:
    """
    One cached ``AsyncOpenAI`` per entity, rebuilt when its inputs change.

    The client is built on Home Assistant's shared httpx client so the SDK
    never creates its own SSL context — a blocking read of the certifi bundle —
    on the event loop, and so back-to-back requests can reuse a pooled
    connection instead of repeating the TLS handshake.

    The httpx client belongs to Home Assistant. Do not close it from here. HA
    also blocks that mistake — it swaps in a warn-only ``aclose`` and only its
    own shutdown listener holds the real one — so a stray close would warn
    rather than tear down the shared pool. Dropping a superseded cached client
    is safe for the same reason: the SDK installs its close-on-GC finalizer only
    on a client it built itself, never on one we supply.

    The timeout is pinned rather than inherited. The SDK adopts a supplied
    client's timeout only when it differs from the httpx default, so leaving it
    off would silently retime every request if a future Home Assistant release
    set one on the shared client. Two other SDK defaults are deliberately given
    up with the swap and left as HA has them: ``follow_redirects`` (httpx's
    ``False``, so a 3xx from a proxy surfaces as an error) and the SDK's
    connection limits (HA's pool keeps connections alive for 15s, which bounds
    the reuse win to back-to-back requests).

    Retries are off. The SDK default of two would make a wedged server cost
    three timeouts of silence before the pipeline hears an error; a voice turn
    should fail fast instead, so the timeout means one attempt.

    The cache is keyed on ``(api_key, base_url)`` — the two configurable inputs.
    ``base_url`` is ``None`` for the OpenAI provider (SDK default endpoint) and
    the configured endpoint for ``local`` providers, so switching a subentry
    between provider types, repointing a local server, or rotating a key
    rebuilds the client on the next request.
    """

    def __init__(self, timeout_s: float) -> None:
        """Remember the platform's read timeout."""
        self._timeout_s = timeout_s
        self._client: AsyncOpenAI | None = None
        self._cache_key: tuple[str, str | None] | None = None

    @property
    def client(self) -> AsyncOpenAI | None:
        """The currently cached client, if one has been built."""
        return self._client

    @property
    def cache_key(self) -> tuple[str, str | None] | None:
        """The ``(api_key, base_url)`` the cached client was built for."""
        return self._cache_key

    def get(
        self, hass: HomeAssistant, api_key: str, base_url: str | None
    ) -> AsyncOpenAI:
        """Return the cached client for these inputs, building it if needed."""
        cache_key = (api_key, base_url)
        if self._client is not None and self._cache_key == cache_key:
            return self._client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=get_async_client(hass),
            timeout=httpx.Timeout(self._timeout_s, connect=CONNECT_TIMEOUT_S),
            max_retries=0,
        )
        self._client = client
        self._cache_key = cache_key
        return client
