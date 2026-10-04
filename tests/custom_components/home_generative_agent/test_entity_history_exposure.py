# ruff: noqa: S101
"""get_entity_history resolves only entities exposed to Assist (issue #716)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.tools import (
    _get_existing_entity_id,
    get_entity_history,
    resolve_entity_ids,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@pytest.fixture(autouse=True)
async def _home(hass: HomeAssistant) -> None:
    """Build a home with exposed, hidden and user-exposed entities."""
    assert await async_setup_component(hass, "homeassistant", {})
    # Lights are auto-exposed to Assist.
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch Light"})
    # Locks are never auto-exposed.
    hass.states.async_set("lock.front", "locked", {"friendly_name": "Front Door"})
    # Moisture sensors are not auto-exposed; the user exposed this one only.
    hass.states.async_set(
        "binary_sensor.sink",
        "off",
        {"friendly_name": "Sink Leak", "device_class": "moisture"},
    )
    async_expose_entity(hass, "conversation", "binary_sensor.sink", should_expose=True)
    hass.states.async_set(
        "binary_sensor.basement",
        "off",
        {"friendly_name": "Basement Leak", "device_class": "moisture"},
    )
    # Two entities share a name; the user hid one of them.
    hass.states.async_set("light.desk", "on", {"friendly_name": "Desk Lamp"})
    hass.states.async_set("light.desk_old", "off", {"friendly_name": "Desk Lamp"})
    async_expose_entity(hass, "conversation", "light.desk_old", should_expose=False)


@pytest.mark.asyncio
async def test_exposed_entities_resolve(hass: HomeAssistant) -> None:
    assert await _get_existing_entity_id("Porch Light", hass, "light") == "light.porch"
    assert (
        await _get_existing_entity_id("Sink Leak", hass, "binary_sensor")
        == "binary_sensor.sink"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "domain"),
    [
        ("Front Door", "lock"),
        ("Basement Leak", "binary_sensor"),
    ],
)
async def test_hidden_entity_reads_as_nonexistent(
    hass: HomeAssistant, name: str, domain: str
) -> None:
    """A hidden entity's error is the nonexistent one, so it isn't confirmed."""
    with pytest.raises(ValueError, match="entity found") as hidden:
        await _get_existing_entity_id(name, hass, domain)
    with pytest.raises(ValueError, match="entity found") as missing:
        await _get_existing_entity_id(f"{name} X", hass, domain)
    assert str(hidden.value) == str(missing.value).replace(f"{name} X", name)


@pytest.mark.asyncio
async def test_hidden_duplicate_does_not_make_a_name_ambiguous(
    hass: HomeAssistant,
) -> None:
    assert await _get_existing_entity_id("Desk Lamp", hass, "light") == "light.desk"


@pytest.mark.asyncio
async def test_no_exposure_registry_resolves_nothing(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: nothing can be shown to be exposed without the registry."""
    monkeypatch.delitem(hass.data, DATA_EXPOSED_ENTITIES)
    with pytest.raises(ValueError, match="entity found"):
        await _get_existing_entity_id("Porch Light", hass, "light")


@pytest.mark.asyncio
async def test_history_tool_refuses_a_hidden_entity_without_reading_it(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issue's case: a hidden lock named at a satellite returns not-found."""
    fetch = AsyncMock(return_value={})
    monkeypatch.setattr(tools_module, "_fetch_data_from_history", fetch)
    monkeypatch.setattr(tools_module, "_fetch_data_from_long_term_stats", fetch)
    config: dict[str, Any] = {"configurable": {"hass": hass}}

    result = await get_entity_history.ainvoke(
        {
            "friendly_names": ["Front Door"],
            "domains": ["lock"],
            "local_start_time": "2026-10-04T00:00:00-0700",
            "local_end_time": "2026-10-04T12:00:00-0700",
        },
        config=cast("Any", config),
    )

    assert result == {"error": "No 'lock' entity found with friendly name 'Front Door'"}
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_entity_ids_fuzzy_match_skips_hidden_entities(
    hass: HomeAssistant,
) -> None:
    """A near-miss id must not reveal a hidden device's real id."""
    config: dict[str, Any] = {"configurable": {"hass": hass}}
    result = await resolve_entity_ids.ainvoke(
        {"entity_ids": ["lock.front_door", "light.porch_light", "lock.front"]},
        config=cast("Any", config),
    )
    # Hidden lock: the fuzzy match is withheld, the guess comes back as given.
    assert result["lock.front_door"] == "lock.front_door"
    # Exposed light: still corrected.
    assert result["light.porch_light"] == "light.porch"
    # An exact id is returned unchanged, as an unknown id is.
    assert result["lock.front"] == "lock.front"
