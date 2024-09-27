"""Home Generative Agent Initalization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.helpers.httpx_client import get_async_client
from langchain_openai import ChatOpenAI

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[ChatOpenAI]

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home generative Agent from a config entry."""
    api_key = entry.data.get(CONF_API_KEY)
    model_name = entry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
    temperature = entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
    max_tokens = entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
    client = ChatOpenAI( #TODO: fix blocking call
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        http_async_client=get_async_client(hass)
    )

    # Cache current platform data which gets added to each request (caching done by library)
    #_ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.bind(timeout=10).get_name)
    # TODO: improve error handling.
    except Exception:
        _LOGGER.exception("Unexpected exception")
        return False

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
