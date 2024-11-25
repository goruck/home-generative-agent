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

from .const import RECOMMENDED_VLM, VLM_URL

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = (Platform.CONVERSATION,)

type HGAConfigEntry = ConfigEntry[HGAData]

@dataclass
class HGAData:
    """Data for Home Generative Assistant."""

    chat_model: ChatOpenAI
    vision_model: ChatOllama

async def async_setup_entry(hass: HomeAssistant, entry: HGAConfigEntry) -> bool:
    """Set up Home generative Agent from a config entry."""
    chat_model = ChatOpenAI( #TODO: fix blocking call
        api_key=entry.data.get(CONF_API_KEY),
        timeout=10,
        http_async_client=get_async_client(hass),
    ).configurable_fields(
        model_name=ConfigurableField(id="model_name"),
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
    )

    try:
        await hass.async_add_executor_job(chat_model.get_name)
    except HomeAssistantError as err:
        LOGGER.error("Error setting up ChatOpenAI: %s", err)
        return False

    entry.chat_model = chat_model

    vision_model = ChatOllama(
        model=RECOMMENDED_VLM,
        base_url=VLM_URL,
        http_async_client=get_async_client(hass)
    ).configurable_fields(
        model=ConfigurableField(id="model"),
        format=ConfigurableField(id="format"),
        temperature=ConfigurableField(id="temperature"),
        num_predict=ConfigurableField(id="num_predict"),
    )

    try:
        await hass.async_add_executor_job(vision_model.get_name)
    except HomeAssistantError as err:
        LOGGER.error("Error setting up Ollama: %s", err)
        return False

    entry.vision_model = vision_model

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Home Generative Agent."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
