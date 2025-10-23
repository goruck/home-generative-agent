"""Set up one ImageEntity per discovered camera."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

from .core.image_entity import LastEventImage

if TYPE_CHECKING:
    from homeassistant.core import Event, HomeAssistant


def _discover_cameras(hass: HomeAssistant) -> list[str]:
    """Return all current camera entity_ids."""
    return [s.entity_id for s in hass.states.async_all("camera")]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: Any,  # noqa: ARG001
    async_add_entities: Any,
) -> None:
    """
    Set up one ImageEntity per discovered camera.

    If no cameras exist yet, wait for Home Assistant to finish starting,
    then try discovery again.
    """
    cams = _discover_cameras(hass)
    if cams:
        async_add_entities([LastEventImage(hass, cam) for cam in cams])
        return

    async def _on_started(_: Event) -> None:
        new_cams = _discover_cameras(hass)
        if not new_cams:
            return
        async_add_entities([LastEventImage(hass, cam) for cam in new_cams])

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _on_started)
