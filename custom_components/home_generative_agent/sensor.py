"""Set up one recognized-people sensor per camera and the sentinel health sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

from .core.recognized_sensor import RecognizedPeopleSensor
from .core.sentinel_health_sensor import SentinelHealthSensor
from .core.tool_index_sensor import ToolIndexSensor

if TYPE_CHECKING:
    from homeassistant.core import Event, HomeAssistant

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

    async def _on_started(_: Event) -> None:
        new_cams = _discover_cameras(hass)
        if not new_cams:
            return
        async_add_entities([RecognizedPeopleSensor(hass, cam) for cam in new_cams])

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, _on_started)
