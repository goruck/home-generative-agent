"""Utility functions for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import async_timeout
import httpx
import psycopg
from homeassistant.helpers.httpx_client import get_async_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..const import (  # noqa: TID252
    EMBEDDING_MODEL_DIMS,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_UNAUTHORIZED,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant
    from langchain_ollama import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings

LOGGER = logging.getLogger(__name__)

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


def ensure_http_url(url: str) -> str:
    """Ensure a URL has an explicit scheme."""
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


async def generate_embeddings(
    emb: OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings from a list of text.

    If Gemini: force output_dimensionality to match index size.
    """
    texts_list = [t for t in texts if t]  # drop empties defensively
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
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(urljoin(base_url.rstrip("/") + "/", "api/tags"))
    except (TimeoutError, httpx.RequestError) as err:
        LOGGER.debug("Ollama connectivity exception: %s", err)
        raise CannotConnectError from err
    else:
        if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
            raise CannotConnectError


async def validate_openai_key(
    hass: HomeAssistant, api_key: str, timeout_s: float = 10.0
) -> None:
    """Validate that an OpenAI API key is authorized and reachable."""
    if not api_key:
        return
    client = get_async_client(hass)
    try:
        async with async_timeout.timeout(timeout_s):
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


async def validate_gemini_key(
    hass: HomeAssistant, api_key: str, timeout_s: float = 10.0
) -> None:
    """Validate that a Gemini API key is authorized and reachable."""
    if not api_key:
        return
    client = get_async_client(hass)
    try:
        async with async_timeout.timeout(timeout_s):
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


async def validate_face_api_url(
    hass: HomeAssistant, base_url: str, timeout_s: float = 10.0
) -> None:
    """Validate that the face-recognition API endpoint is reachable."""
    if not base_url:
        return
    base_url = ensure_http_url(base_url)
    client = get_async_client(hass)
    try:
        async with async_timeout.timeout(timeout_s):
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
