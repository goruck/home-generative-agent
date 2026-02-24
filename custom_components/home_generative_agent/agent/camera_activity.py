"""Shared camera activity retrieval helpers for agent flows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)


async def get_recent_camera_activity(
    hass: HomeAssistant, store: BaseStore
) -> list[dict[str, dict[str, str]]]:
    """Get most recent camera activity from stored video analysis."""
    camera_activity: list[dict[str, dict[str, str]]] = []
    for entity_id in hass.states.async_entity_ids():
        if not entity_id.startswith("camera."):
            continue

        camera = entity_id.split(".")[-1]
        results = await store.asearch(("video_analysis", camera), limit=1)
        if results and (content := results[0].value.get("content")):
            camera_activity.append(
                {
                    camera: {
                        "last activity": content,
                        "date_time": results[0].updated_at.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                }
            )

    if camera_activity:
        LOGGER.debug("Recent camera activity: %s", camera_activity)
        return camera_activity

    LOGGER.debug("No recent camera activity found.")
    return []


def get_camera_last_events_from_states(
    hass: HomeAssistant, camera_entity_id: str | None = None
) -> list[dict[str, Any]]:
    """Collect latest camera event attributes exposed by image entities."""
    entity_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)

    results: list[dict[str, Any]] = []
    for image_state in hass.states.async_all("image"):
        cam_id = image_state.attributes.get("camera_id")
        if not isinstance(cam_id, str):
            continue
        if camera_entity_id is not None and cam_id != camera_entity_id:
            continue

        area_name: str | None = None
        entity_entry = entity_reg.async_get(cam_id)
        if entity_entry and entity_entry.area_id:
            area_entry = area_reg.async_get_area(entity_entry.area_id)
            if area_entry:
                area_name = area_entry.name

        results.append(
            {
                "camera_entity_id": cam_id,
                "area": area_name,
                "last_event": image_state.attributes.get("last_event"),
                "summary": image_state.attributes.get("summary"),
                "recognized_people": image_state.attributes.get(
                    "recognized_people", []
                ),
            }
        )

    return results
