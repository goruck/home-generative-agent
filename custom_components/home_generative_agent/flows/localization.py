"""Shared localization helpers for config subentry flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.translation import async_get_translations

from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


async def async_common_translation(hass: HomeAssistant, key: str, fallback: str) -> str:
    """
    Return a translated string from the integration's "common" section.

    Flow handlers only know the server-configured language, not the
    browsing user's frontend language, so the returned text follows the
    server locale. Languages without the key fall back to English via the
    translation cache; `fallback` covers a missing key or load failure
    (a corrupt translation file raises from the loader).
    """
    try:
        translations = await async_get_translations(
            hass, hass.config.language, "common", {DOMAIN}
        )
    except HomeAssistantError:
        return fallback
    return translations.get(f"component.{DOMAIN}.common.{key}", fallback)
