# ruff: noqa: S101
"""Tests for shared camera activity helpers."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.camera_activity import (
    get_camera_last_events_from_states,
)


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
