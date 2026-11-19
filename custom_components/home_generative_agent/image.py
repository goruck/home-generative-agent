"""Set up one ImageEntity per discovered camera."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.core import callback

from .core.image_entity import LastEventImage
from .core.lifecycle import async_track_cameras

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .core.runtime import HGAConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HGAConfigEntry,
    async_add_entities: Any,
) -> None:
    """Set up one ImageEntity per camera, including cameras added later."""

    @callback
    def _add_cameras(cams: list[str]) -> None:
        async_add_entities([LastEventImage(hass, cam) for cam in cams])

    async_track_cameras(hass, entry, _add_cameras)
