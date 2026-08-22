"""Set up one recognized-people sensor per camera and the sentinel health sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState

from .core.lifecycle import defer_start_until_hass_started
from .core.recognized_sensor import RecognizedPeopleSensor
from .core.sentinel_health_sensor import SentinelHealthSensor
from .core.tool_index_sensor import ToolIndexSensor

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
    Set up sensors for this config entry.

    Registers the Sentinel health sensor, the tool index diagnostic sensor,
    and one recognized-people sensor per camera.
    If no cameras exist yet, waits for HA startup before registering them.
    """
    data = entry.runtime_data
    async_add_entities(
        [
            SentinelHealthSensor(
                hass,
                data.options,
                data.audit_store,
                data.sentinel,
                entry.entry_id,
                baseline_updater=data.baseline_updater,
                discovery_engine=data.discovery_engine,
                proposal_store=data.proposal_store,
            ),
            ToolIndexSensor(hass, entry.entry_id),
        ]
    )

    cams = _discover_cameras(hass)
    if cams:
        async_add_entities([RecognizedPeopleSensor(hass, cam) for cam in cams])
        return

    def _add_discovered_cameras() -> None:
        # Same entry-state guard as image.py: the deferred-start cancel cannot
        # close the window between async_unload_entry starting and HA running
        # the on-unload callbacks (see defer_start_until_hass_started), and
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
            async_add_entities([RecognizedPeopleSensor(hass, cam) for cam in new_cams])

    defer_start_until_hass_started(hass, entry, _add_discovered_cameras)
