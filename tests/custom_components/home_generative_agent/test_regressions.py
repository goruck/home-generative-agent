# ruff: noqa: S101
"""Regression tests for previously fixed issues."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.home_generative_agent.agent import tools as agent_tools
from custom_components.home_generative_agent.const import SIGNAL_HGA_RECOGNIZED
from custom_components.home_generative_agent.core.image_entity import LastEventImage
from custom_components.home_generative_agent.core.person_gallery import PersonGalleryDAO
from custom_components.home_generative_agent.core.recognized_sensor import (
    RecognizedPeopleSensor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from homeassistant.core import HomeAssistant


HistoryTool = Any
history_tool = cast(
    "HistoryTool", cast("Any", agent_tools.get_entity_history).coroutine
)


@pytest.mark.asyncio
async def test_get_entity_history_pairs_zip_warns(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched name/domain lengths should be paired best-effort."""
    calls: list[tuple[str, str]] = []

    async def _fake_get_existing_entity_id(
        name: str, hass_arg: object, domain: str
    ) -> str:
        _ = hass_arg
        calls.append((name, domain))
        return f"{domain}.{name.lower().replace(' ', '_')}"

    async def _fake_fetch(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(
        agent_tools, "_get_existing_entity_id", _fake_get_existing_entity_id
    )
    monkeypatch.setattr(agent_tools, "_fetch_data_from_history", _fake_fetch)
    monkeypatch.setattr(agent_tools, "_fetch_data_from_long_term_stats", _fake_fetch)

    config = {"configurable": {"hass": hass}}
    with caplog.at_level(logging.WARNING):
        result = await history_tool(
            ["Front Door", "Living Room Light"],
            ["binary_sensor"],
            "2025-01-01T00:00:00+0000",
            "2025-01-02T00:00:00+0000",
            config=config,
        )

    assert calls == [("Front Door", "binary_sensor")]
    assert any("pairing best-effort" in rec.message for rec in caplog.records)
    assert result == {}


def test_filter_data_total_increasing_empty(hass: HomeAssistant) -> None:
    """Return zero value when total_increasing data is non-numeric."""
    hass.states.async_set(
        "sensor.energy",
        "unknown",
        {
            "state_class": "total_increasing",
            "unit_of_measurement": "kWh",
        },
    )

    result = agent_tools._filter_data(
        "sensor.energy", [{"state": "unknown"}], hass
    )
    assert result["value"] == 0.0
    assert result["units"] == "kWh"


@pytest.mark.asyncio
async def test_recognized_sensor_attributes_update(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extra_state_attributes should reflect current people list."""
    sensor = RecognizedPeopleSensor(hass, "camera.test")
    monkeypatch.setattr(sensor, "async_write_ha_state", lambda: None)
    await sensor.async_added_to_hass()

    attrs_initial = dict(sensor.extra_state_attributes or {})
    assert attrs_initial is not None
    async_dispatcher_send(
        hass, SIGNAL_HGA_RECOGNIZED, "camera.test", ["Alice"], None, None, None
    )
    attrs_updated = sensor.extra_state_attributes
    assert attrs_updated is not None

    assert attrs_initial["count"] == 0
    assert attrs_updated["count"] == 1


def test_last_event_image_recognized_mapping(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SIGNAL_HGA_RECOGNIZED args should map to the right fields."""
    entity = LastEventImage(hass, "camera.test")
    monkeypatch.setattr(entity, "async_write_ha_state", lambda: None)

    latest_path = tmp_path / "latest.jpg"
    entity._on_recognized(
        "camera.test",
        ["Alice"],
        "Porch activity",
        "2025-01-01T00:00:00+0000",
        str(latest_path),
    )

    attrs = entity._attrs
    assert attrs["recognized_people"] == ["Alice"]
    assert attrs["summary"] == "Porch activity"
    assert attrs["last_event"] == "2025-01-01T00:00:00+0000"
    assert attrs["latest_path"] == str(latest_path)


@pytest.mark.asyncio
async def test_person_gallery_invalid_embedding(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid embeddings should not trigger DB inserts."""

    class _FakeResp:
        def __init__(self) -> None:
            self._payload: dict[str, object] = {"faces": [{"embedding": [0.1, 0.2]}]}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        async def post(self, *_args: object, **_kwargs: object) -> _FakeResp:
            return _FakeResp()

    dao = PersonGalleryDAO(cast("Any", object()), hass)
    dao._client = _FakeClient()  # type: ignore[assignment]

    async def _fail_add_person(*_args: object, **_kwargs: object) -> None:
        msg = "add_person should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(dao, "add_person", _fail_add_person)

    result = await dao.enroll_from_image("http://face-api", "Alice", b"img")
    assert result is False
