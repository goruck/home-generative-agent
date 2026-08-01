"""Diagnostic sensor that exposes the RAG tool-index state."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util

from ..const import SIGNAL_TOOL_INDEX_UPDATED  # noqa: TID252

LOGGER = logging.getLogger(__name__)

_DESC = SensorEntityDescription(
    key="tool_index_status",
    icon="mdi:database-search",
    entity_category=EntityCategory.DIAGNOSTIC,
)

# Possible state values
_STATE_UNKNOWN = "unknown"
_STATE_INDEXING = "indexing"
_STATE_READY = "ready"
_STATE_FAILED = "failed"


class ToolIndexSensor(SensorEntity):
    """Diagnostic sensor showing whether the RAG tool index is ready."""

    entity_description: SensorEntityDescription = _DESC
    _attr_has_entity_name = True
    _attr_icon = "mdi:database-search"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the tool index sensor."""
        self.hass = hass
        self._attr_name = "Tool Index Status"
        self._attr_unique_id = f"tool_index_status::{entry_id}"
        self._attr_native_value = _STATE_UNKNOWN
        self._tools_indexed: int = 0
        self._last_updated: str | None = None
        self._attr_extra_state_attributes: dict[str, Any] = {}

    async def async_added_to_hass(self) -> None:
        """Subscribe to tool-index update signal."""

        @callback
        def _on_updated(state: str, tools_indexed: int) -> None:
            self._attr_native_value = state
            if state == _STATE_READY:
                self._tools_indexed = tools_indexed
                self._last_updated = dt_util.now().isoformat()
            self._attr_extra_state_attributes = {
                "tools_indexed": self._tools_indexed,
                "last_updated": self._last_updated,
            }
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(self.hass, SIGNAL_TOOL_INDEX_UPDATED, _on_updated)
        )
