"""Tests for shared camera activity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from custom_components.home_generative_agent.agent.camera_activity import (
    get_camera_last_events_from_states,
    get_recent_camera_activity,
)


@dataclass
class _FakeRecord:
    value: dict[str, Any]
    updated_at: datetime


class _FakeStore:
    def __init__(self, data: dict[str, list[_FakeRecord]]) -> None:
        self._data = data

    async def asearch(
        self, namespace: tuple[str, str], limit: int = 1
    ) -> list[_FakeRecord]:
        _ = limit
        return self._data.get(namespace[1], [])


@pytest.mark.asyncio
async def test_get_recent_camera_activity_reads_store_by_camera_slug(hass) -> None:
    hass.states.async_set("camera.front_door", "idle")
    hass.states.async_set("camera.backyard", "idle")

    store = _FakeStore(
        {
            "front_door": [
                _FakeRecord(
                    value={"content": "Person seen at front door"},
                    updated_at=datetime(2026, 2, 1, 12, 30, 0, tzinfo=UTC),
                )
            ]
        }
    )

    result = await get_recent_camera_activity(hass, store)  # type: ignore[arg-type]

    assert result == [
        {
            "front_door": {
                "last activity": "Person seen at front door",
                "date_time": "2026-02-01 12:30:00",
            }
        }
    ]


def test_get_camera_last_events_from_states_filters_by_camera(hass) -> None:
    hass.states.async_set(
        "image.front_door_last_event",
        "ok",
        {
            "camera_id": "camera.front_door",
            "last_event": "2026-02-01T12:30:00+00:00",
            "summary": "Delivery person at door",
            "recognized_people": ["Alice"],
        },
    )
    hass.states.async_set(
        "image.backyard_last_event",
        "ok",
        {
            "camera_id": "camera.backyard",
            "last_event": "2026-02-01T13:00:00+00:00",
            "summary": "No activity",
        },
    )

    result = get_camera_last_events_from_states(hass, "camera.front_door")

    assert result == [
        {
            "camera_entity_id": "camera.front_door",
            "area": None,
            "last_event": "2026-02-01T12:30:00+00:00",
            "summary": "Delivery person at door",
            "recognized_people": ["Alice"],
        }
    ]
