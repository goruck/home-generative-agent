"""Home Generative Agent Initalization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[ChatOpenAI]

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home generative Agent from a config entry."""
    model = ChatOpenAI( #TODO: fix blocking call
        api_key=entry.data.get(CONF_API_KEY),
        cache=True,
        timeout=10,
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model_name=ConfigurableField(id="model_name"),
        temperature=ConfigurableField(id="temperature"),
        max_tokens=ConfigurableField(id="max_tokens"),
        top_p=ConfigurableField(id="top_p"),
    )

    try:
        await hass.async_add_executor_job(model.get_name)
    # TODO: improve error handling.
    except Exception:
        _LOGGER.exception("Unexpected exception")
        return False

    entry.runtime_data = model

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
