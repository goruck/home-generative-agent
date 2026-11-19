"""Set up one recognized-people sensor per camera and the sentinel health sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.core import callback

from .core.lifecycle import async_track_cameras
from .core.recognized_sensor import RecognizedPeopleSensor
from .core.sentinel_health_sensor import SentinelHealthSensor
from .core.tool_index_sensor import ToolIndexSensor

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .core.runtime import HGAConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HGAConfigEntry,
    async_add_entities: Any,
) -> None:
    """
    Set up sensors for this config entry.

    Registers the Sentinel health sensor, the tool index diagnostic sensor,
    and one recognized-people sensor per camera, including cameras added
    later.
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
                stall_tracker=data.stall_tracker,
            ),
            ToolIndexSensor(hass, entry.entry_id),
        ]
    )

    @callback
    def _add_cameras(cams: list[str]) -> None:
        async_add_entities([RecognizedPeopleSensor(hass, cam) for cam in cams])

    async_track_cameras(hass, entry, _add_cameras)
