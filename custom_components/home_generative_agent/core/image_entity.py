"""Expose the latest analyzed frame plus its AI metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from homeassistant.components.image import ImageEntity
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    SIGNAL_HGA_NEW_LATEST,
    SIGNAL_HGA_RECOGNIZED,
    VIDEO_ANALYZER_SNAPSHOT_ROOT,
)
from custom_components.home_generative_agent.core.video_helpers import latest_target

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

# -------------------------
# Constants (no magic vals)
# -------------------------
ATTR_SUMMARY: Final = "summary"
ATTR_RECOGNIZED_PEOPLE: Final = "recognized_people"
ATTR_COUNT: Final = "count"
ATTR_LAST_EVENT: Final = "last_event"
ATTR_LATEST_PATH: Final = "latest_path"
ATTR_CAMERA_ID: Final = "camera_id"

BUS_EVT_LAST_EVENT_FRAME: Final = "hga_last_event_frame"
BUS_KEY_CAMERA_ID: Final = "camera_id"
BUS_KEY_LATEST: Final = "latest"
BUS_KEY_SUMMARY: Final = "summary"

MIME_JPEG: Final = "image/jpeg"

UNIQUE_PREFIX: Final = "last_event::"
NAME_SUFFIX: Final = " Last Event"

# Positional index constants
IDX_CAMERA_ID: Final = 0
IDX_LATEST_PATH: Final = 1
IDX_SUMMARY: Final = 2
IDX_PEOPLE: Final = 3
IDX_LAST_EVENT_ISO: Final = 4


@dataclass(slots=True)
class UpdateBundle:
    """Normalized update payload across all sources."""

    latest_path: Path | None = None
    summary: str | None = None
    people: list[str] | None = None
    last_event_iso: str | None = None


class LastEventImage(ImageEntity):
    """ImageEntity exposing the latest analyzed frame plus its AI metadata."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, camera_id: str) -> None:
        """Initialize a per-camera image entity."""
        super().__init__(hass)
        self.hass = hass
        self._camera_id = camera_id
        self._path: Path | None = None

        attrs: dict[str, Any] = {
            ATTR_SUMMARY: None,
            ATTR_RECOGNIZED_PEOPLE: [],
            ATTR_COUNT: 0,
            ATTR_LAST_EVENT: None,
            ATTR_LATEST_PATH: None,
            ATTR_CAMERA_ID: camera_id,
        }
        self._attrs = attrs
        self._attr_extra_state_attributes = self._attrs

        slug = camera_id.split(".", 1)[1] if "." in camera_id else camera_id
        self._attr_name = f"{slug}{NAME_SUFFIX}"
        self._attr_unique_id = f"{UNIQUE_PREFIX}{camera_id}"

    # -------------------------
    # HA lifecycle
    # -------------------------
    async def async_added_to_hass(self) -> None:
        """Subscribe to updates and seed from disk if available."""
        self._seed_from_disk()

        rm_new = async_dispatcher_connect(
            self.hass, SIGNAL_HGA_NEW_LATEST, self._on_new_latest
        )
        rm_rec = async_dispatcher_connect(
            self.hass, SIGNAL_HGA_RECOGNIZED, self._on_recognized
        )
        rm_evt = self.hass.bus.async_listen(
            BUS_EVT_LAST_EVENT_FRAME, self._on_last_event_frame
        )

        for remover in (rm_new, rm_rec, cast("CALLBACK_TYPE", rm_evt)):
            self.async_on_remove(remover)

    # -------------------------
    # Seeds & helpers
    # -------------------------
    def _seed_from_disk(self) -> None:
        """Initialize state from the latest snapshot on disk, if present."""
        seed = latest_target(Path(VIDEO_ANALYZER_SNAPSHOT_ROOT), self._camera_id)
        if seed.exists():
            self._apply_update(UpdateBundle(latest_path=seed))
            self.async_write_ha_state()

    def _apply_update(self, upd: UpdateBundle) -> None:
        """Apply an UpdateBundle to internal state & attributes."""
        if upd.latest_path is not None:
            self._path = upd.latest_path
            self._attrs[ATTR_LATEST_PATH] = str(upd.latest_path)
            self._attr_image_last_updated = dt_util.utcnow()

        if upd.summary is not None:
            self._attrs[ATTR_SUMMARY] = upd.summary

        if upd.people is not None:
            ppl = list(upd.people)
            self._attrs[ATTR_RECOGNIZED_PEOPLE] = ppl
            self._attrs[ATTR_COUNT] = len(ppl)

        if upd.last_event_iso is not None:
            self._attrs[ATTR_LAST_EVENT] = upd.last_event_iso

    @staticmethod
    def _list_or_none(seq: Iterable[str] | None) -> list[str] | None:
        return list(seq) if seq is not None else None

    # -------------------------
    # Signal/event handlers
    # -------------------------
    @callback
    def _on_new_latest(self, *args: Any) -> None:
        """Handle SIGNAL_HGA_NEW_LATEST."""
        if not args or cast("str", args[IDX_CAMERA_ID]) != self._camera_id:
            return

        latest_path = (
            cast("str | None", args[IDX_LATEST_PATH])
            if len(args) > IDX_LATEST_PATH
            else None
        )
        summary = (
            cast("str | None", args[IDX_SUMMARY]) if len(args) > IDX_SUMMARY else None
        )
        people = (
            cast("Sequence[str] | None", args[IDX_PEOPLE])
            if len(args) > IDX_PEOPLE
            else None
        )
        last_event_iso = (
            cast("str | None", args[IDX_LAST_EVENT_ISO])
            if len(args) > IDX_LAST_EVENT_ISO
            else None
        )

        upd = UpdateBundle(
            latest_path=Path(latest_path) if latest_path else None,
            summary=summary,
            people=self._list_or_none(people),
            last_event_iso=last_event_iso,
        )
        self._apply_update(upd)
        self.async_write_ha_state()

    @callback
    def _on_recognized(self, *args: Any) -> None:
        """Handle SIGNAL_HGA_RECOGNIZED."""
        if not args or cast("str", args[IDX_CAMERA_ID]) != self._camera_id:
            return

        people = (
            cast("Sequence[str] | None", args[IDX_LATEST_PATH])
            if len(args) > IDX_LATEST_PATH
            else None
        )
        summary = (
            cast("str | None", args[IDX_SUMMARY]) if len(args) > IDX_SUMMARY else None
        )
        last_event_iso = (
            cast("str | None", args[IDX_PEOPLE]) if len(args) > IDX_PEOPLE else None
        )
        latest_path = (
            cast("str | None", args[IDX_LAST_EVENT_ISO])
            if len(args) > IDX_LAST_EVENT_ISO
            else None
        )

        upd = UpdateBundle(
            latest_path=Path(latest_path) if latest_path else None,
            summary=summary,
            people=self._list_or_none(people),
            last_event_iso=last_event_iso,
        )
        self._apply_update(upd)
        self.async_write_ha_state()

    @callback
    def _on_last_event_frame(self, event: Event) -> None:
        """Backstop: take summary/path from the bus event."""
        data = event.data or {}
        if data.get(BUS_KEY_CAMERA_ID) != self._camera_id:
            return

        latest_any = data.get(BUS_KEY_LATEST)
        latest_path = Path(str(latest_any)) if latest_any else None
        summary_any = data.get(BUS_KEY_SUMMARY)
        summary = str(summary_any) if summary_any is not None else None

        upd = UpdateBundle(
            latest_path=latest_path,
            summary=summary,
            people=None,
            last_event_iso=dt_util.utcnow().isoformat(),
        )
        self._apply_update(upd)
        self.async_write_ha_state()

    # -------------------------
    # ImageEntity API
    # -------------------------
    @cached_property
    def content_type(self) -> str:
        """Return MIME type."""
        return MIME_JPEG

    async def async_image(self) -> bytes | None:
        """Return image bytes, or None if unavailable."""
        path = self._path
        if path is None:
            return None
        try:
            return await self.hass.async_add_executor_job(path.read_bytes)
        except OSError:
            return None
