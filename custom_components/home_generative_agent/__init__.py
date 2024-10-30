"""Home Generative Agent Initalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.httpx_client import get_async_client
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .const import OLLAMA_RECOMMENDED_BASE_URL, OLLAMA_RECOMMENDED_MODEL

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[HGAData]

@dataclass
class HGAData:
    """Data for Home Generative Assistant."""

    model: ChatOpenAI
    edge_model: ChatOllama

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home generative Agent from a config entry."""
    model = ChatOpenAI( #TODO: fix blocking call
        api_key=entry.data.get(CONF_API_KEY),
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
    except HomeAssistantError as err:
        LOGGER.error("Error setting up ChatOpenAI: %s", err)
        return False

    entry.model = model

    edge_model = ChatOllama(
        model=OLLAMA_RECOMMENDED_MODEL,
        base_url=OLLAMA_RECOMMENDED_BASE_URL,
        format="json",
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        top_k=ConfigurableField(id="top_k"),
        num_predict=ConfigurableField(id="num_predict"),
    )

    try:
        await hass.async_add_executor_job(edge_model.get_name)
    except HomeAssistantError as err:
        LOGGER.error("Error setting up Ollama: %s", err)
        return False

    entry.edge_model = edge_model

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
