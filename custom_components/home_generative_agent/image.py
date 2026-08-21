"""Set up one ImageEntity per discovered camera."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState

from .core.image_entity import LastEventImage
from .core.lifecycle import defer_start_until_hass_started

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .core.runtime import HGAConfigEntry


def _discover_cameras(hass: HomeAssistant) -> list[str]:
    """Return all current camera entity_ids."""
    return [s.entity_id for s in hass.states.async_all("camera")]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HGAConfigEntry,
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

    def _add_discovered_cameras() -> None:
        # The deferred-start cancel cannot close the window between
        # async_unload_entry starting and HA running the on-unload callbacks
        # (see defer_start_until_hass_started). The engines close it with a
        # _stopped latch; the platform equivalent is this entry-state guard —
        # adding entities to a platform that has begun unloading strands them
        # on a dead entry. SETUP_IN_PROGRESS stays allowed because a platform
        # set up while HA is still starting defers into exactly that state.
        if entry.state not in (
            ConfigEntryState.LOADED,
            ConfigEntryState.SETUP_IN_PROGRESS,
        ):
            return
        new_cams = _discover_cameras(hass)
        if new_cams:
            async_add_entities([LastEventImage(hass, cam) for cam in new_cams])

    defer_start_until_hass_started(hass, entry, _add_discovered_cameras)
