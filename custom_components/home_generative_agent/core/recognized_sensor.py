"""Per-camera sensor exposing recognized names and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.restore_state import RestoreEntity as _RestoreEntity

from ..const import SIGNAL_HGA_RECOGNIZED  # noqa: TID252

# During type checking, avoid mixed-base attribute conflicts.
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    RestoreBase = object
else:
    RestoreBase = _RestoreEntity

RECOGNIZED_DESC = SensorEntityDescription(
    key="recognized_people",
    icon="mdi:account-group",
)


def _truncate_state(people: list[str], *, max_chars: int = 255) -> str:
    """Short state string; full list goes in attributes."""
    if not people:
        return "None"
    joined = ", ".join(people)
    return joined if len(joined) <= max_chars else f"{len(people)} recognized"


class RecognizedPeopleSensor(SensorEntity, RestoreBase):
    """Per-camera sensor exposing recognized names and metadata."""

    # Match SensorEntity base types EXACTLY:
    entity_description: SensorEntityDescription = RECOGNIZED_DESC
    _attr_device_class: SensorDeviceClass | None = None  # no 'str' here
    _attr_state_class: SensorStateClass | str | None = None  # include 'str'
    _attr_native_unit_of_measurement: str | None = None
    _attr_has_entity_name = True
    _attr_icon = "mdi:account-group"

    def __init__(self, hass: HomeAssistant, camera_id: str) -> None:
        """Initialize a sensor for a specific camera."""
        self.hass = hass
        self._camera_id = camera_id
        slug = camera_id.split(".", 1)[1]
        self._attr_name = f"{slug} Recognized People"
        self._attr_unique_id = f"recognized::{camera_id}"
        self._people: list[str] = []
        self._summary: str | None = None
        self._last_event_iso: str | None = None
        self._latest_path: str | None = None
        self._attr_native_value = "None"
        self._attrs: dict[str, Any] = {
            "recognized_people": self._people,
            "count": 0,
            "summary": None,
            "last_event": None,
            "latest_path": None,
            "camera_id": self._camera_id,
        }
        self._attr_extra_state_attributes = self._attrs

    async def async_added_to_hass(self) -> None:
        """Restore prior state and subscribe to recognition updates."""
        # Grab RestoreEntity's coroutine if present at runtime;
        # stays type-safe in TYPE_CHECKING.
        get_last: Callable[[], Awaitable[Any]] | None = getattr(
            self, "async_get_last_state", None
        )
        last: Any | None
        if callable(get_last):
            last = await get_last()
        else:
            last = None

        if last is not None:
            self._people = list(last.attributes.get("recognized_people", []))
            self._summary = last.attributes.get("summary")
            self._last_event_iso = last.attributes.get("last_event")
            self._latest_path = last.attributes.get("latest_path")
            self._attr_native_value = _truncate_state(self._people)
            self._attrs.update(
                {
                    "recognized_people": self._people,
                    "count": len(self._people),
                    "summary": self._summary,
                    "last_event": self._last_event_iso,
                    "latest_path": self._latest_path,
                }
            )

        @callback
        def _on_recognized(
            camera_id: str,
            people: list[str],
            summary: str | None = None,
            last_event_iso: str | None = None,
            latest_path: str | None = None,
        ) -> None:
            """Update sensor with new recognition results."""
            if camera_id != self._camera_id:
                return
            if (
                people == self._people
                and summary == self._summary
                and last_event_iso == self._last_event_iso
                and latest_path == self._latest_path
            ):
                return
            self._people = people
            self._summary = summary
            self._last_event_iso = last_event_iso
            self._latest_path = latest_path
            self._attr_native_value = _truncate_state(self._people)
            self._attrs.update(
                {
                    "recognized_people": self._people,
                    "count": len(self._people),
                    "summary": self._summary,
                    "last_event": self._last_event_iso,
                    "latest_path": self._latest_path,
                }
            )
            self.async_write_ha_state()

        remove = async_dispatcher_connect(
            self.hass, SIGNAL_HGA_RECOGNIZED, _on_recognized
        )
        self.async_on_remove(remove)
