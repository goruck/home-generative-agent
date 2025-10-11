"""Utility functions for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import async_timeout
from homeassistant.helpers.httpx_client import get_async_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..const import EMBEDDING_MODEL_DIMS, HTTP_STATUS_BAD_REQUEST  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant
    from langchain_ollama import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings

LOGGER = logging.getLogger(__name__)


def ensure_http_url(url: str) -> str:
    """Ensure a URL has an explicit scheme."""
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


async def ollama_healthy(
    hass: HomeAssistant, base_url: str, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for Ollama."""
    url = ensure_http_url(base_url).rstrip("/") + "/api/tags"
    client = get_async_client(hass)
    try:
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(url)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("Ollama health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Ollama health check failed at %s: %s", url, err)
    return False


async def openai_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for OpenAI."""
    if not api_key:
        LOGGER.warning("OpenAI health check skipped: missing API key.")
        return False
    client = get_async_client(hass)
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with async_timeout.timeout(timeout_s):
            resp = await client.get("https://api.openai.com/v1/models", headers=headers)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("OpenAI health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("OpenAI health check failed: %s", err)
    return False


async def gemini_healthy(
    hass: HomeAssistant, api_key: str | None, timeout_s: float = 2.0
) -> bool:
    """Quick reachability check for Gemini."""
    if not api_key:
        LOGGER.warning("Gemini health check skipped: missing API key.")
        return False
    client = get_async_client(hass)
    try:
        # Public models list endpoint requires key query param
        url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(url)
        if resp.status_code < HTTP_STATUS_BAD_REQUEST:
            return True
        LOGGER.warning("Gemini health check HTTP %s: %s", resp.status_code, resp.text)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Gemini health check failed: %s", err)
    return False


async def generate_embeddings(
    emb: OpenAIEmbeddings | OllamaEmbeddings | GoogleGenerativeAIEmbeddings,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Generate embeddings from a list of text.

    Note: Gemini supports custom output dimensionality; force 1024 to match our index.
    """
    texts_list = list(texts)
    # If it's Gemini, ask for 1024-d explicitly
    if isinstance(emb, GoogleGenerativeAIEmbeddings):
        return await emb.aembed_documents(
            texts_list, output_dimensionality=EMBEDDING_MODEL_DIMS
        )
    # OpenAI / Ollama paths unchanged
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
