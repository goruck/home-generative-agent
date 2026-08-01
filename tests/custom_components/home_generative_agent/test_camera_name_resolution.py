# ruff: noqa: S101
"""Tests for camera name → entity_id resolution in agent tools (issue #502)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import ATTR_FRIENDLY_NAME

from custom_components.home_generative_agent.agent.tools import (
    _available_camera_names,
    _normalize_camera_name,
    _resolve_camera_entity_id,
    get_and_analyze_camera_image,
)


class FakeState:
    """Minimal stand-in for a Home Assistant State object."""

    def __init__(self, entity_id: str, friendly_name: str | None = None) -> None:
        self.entity_id = entity_id
        self.attributes: dict[str, Any] = {}
        if friendly_name is not None:
            self.attributes[ATTR_FRIENDLY_NAME] = friendly_name


class FakeHass:
    """Minimal stand-in for HomeAssistant with a camera state registry."""

    def __init__(self, states: list[FakeState]) -> None:
        self._states = {state.entity_id: state for state in states}
        self.states = self

    def get(self, entity_id: str) -> FakeState | None:
        return self._states.get(entity_id)

    def async_all(self, domain: str) -> list[FakeState]:
        prefix = f"{domain}."
        return [
            state
            for entity_id, state in self._states.items()
            if entity_id.startswith(prefix)
        ]


def _hass_with_czech_cameras() -> FakeHass:
    return FakeHass(
        [
            FakeState("camera.kamera4", "kamera 4"),
            FakeState("camera.kamera_obyvak", "kamera obývák"),
            FakeState("camera.kamera_obyvak_2", "kamera obývák 2"),
            FakeState("camera.front_porch", "Front Porch"),
            FakeState("sensor.kamera_obyvak_battery", "kamera obývák battery"),
        ]
    )


def test_normalize_strips_diacritics_spaces_and_case() -> None:
    """Diacritics, spaces, underscores, and case all normalize away."""
    assert _normalize_camera_name("Kamera obývák 2") == "kameraobyvak2"
    assert _normalize_camera_name("kamera_obyvak_2") == "kameraobyvak2"
    assert _normalize_camera_name("kamera 4") == "kamera4"


def test_resolve_literal_entity_id() -> None:
    """A full entity_id from the LLM resolves as-is."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "camera.front_porch")  # type: ignore[arg-type]
    assert resolved == "camera.front_porch"


def test_resolve_legacy_lowercase_slug() -> None:
    """The legacy lowercased-name behavior still resolves exact slugs."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "Front_Porch")  # type: ignore[arg-type]
    assert resolved == "camera.front_porch"


def test_resolve_friendly_name_with_space() -> None:
    """'kamera 4' resolves to camera.kamera4 via its friendly name."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "kamera 4")  # type: ignore[arg-type]
    assert resolved == "camera.kamera4"


def test_resolve_friendly_name_with_diacritics() -> None:
    """'kamera obývák 2' resolves to the diacritics-free slug entity."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "kamera obývák 2")  # type: ignore[arg-type]
    assert resolved == "camera.kamera_obyvak_2"


def test_resolve_prefix_is_not_cross_matched() -> None:
    """'kamera obývák' must resolve to its own camera, not the '2' variant."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "kamera obývák")  # type: ignore[arg-type]
    assert resolved == "camera.kamera_obyvak"


def test_resolve_diacritics_input_against_slug() -> None:
    """A name matching only the entity_id slug (no friendly match) resolves."""
    hass = FakeHass([FakeState("camera.zahrada_vychod")])
    resolved = _resolve_camera_entity_id(hass, "Zahrada východ")  # type: ignore[arg-type]
    assert resolved == "camera.zahrada_vychod"


def test_resolve_ignores_non_camera_domains() -> None:
    """Matching never leaves the camera domain."""
    hass = FakeHass([FakeState("sensor.kamera_obyvak_battery", "kamera obývák")])
    assert _resolve_camera_entity_id(hass, "kamera obývák") is None  # type: ignore[arg-type]


def test_resolve_unknown_name_returns_none() -> None:
    """A name with no camera match returns None instead of a bogus entity_id."""
    hass = _hass_with_czech_cameras()
    assert _resolve_camera_entity_id(hass, "garage camera") is None  # type: ignore[arg-type]


def test_resolve_empty_name_returns_none() -> None:
    """Punctuation-only input cannot match every camera via empty normalization."""
    hass = _hass_with_czech_cameras()
    assert _resolve_camera_entity_id(hass, "???") is None  # type: ignore[arg-type]


def test_available_camera_names_lists_friendly_names() -> None:
    """The not-found hint lists camera friendly names only."""
    names = _available_camera_names(_hass_with_czech_cameras())  # type: ignore[arg-type]
    assert names == "Front Porch, kamera 4, kamera obývák, kamera obývák 2"


@pytest.mark.asyncio
async def test_camera_tool_resolves_friendly_name_before_capture() -> None:
    """The chat tool captures from the resolved entity_id, not the raw name."""
    hass = _hass_with_czech_cameras()
    config = {"configurable": {"hass": hass, "vlm_model": MagicMock()}}

    capture = AsyncMock(return_value=None)
    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=capture,
    ):
        result = await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="kamera obývák 2", detection_keywords=None, config=config
        )

    capture.assert_awaited_once_with(hass, "camera.kamera_obyvak_2")
    assert result == "Error getting image from camera."


@pytest.mark.asyncio
async def test_camera_tool_reports_unknown_camera_with_available_names() -> None:
    """An unresolvable name returns a message the LLM can use to self-correct."""
    hass = _hass_with_czech_cameras()
    config = {"configurable": {"hass": hass, "vlm_model": MagicMock()}}

    capture = AsyncMock()
    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=capture,
    ):
        result = await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="garage camera", detection_keywords=None, config=config
        )

    capture.assert_not_awaited()
    assert result.startswith("Camera 'garage camera' was not found.")
    assert "kamera obývák 2" in result


def test_resolve_capitalized_domain_prefix() -> None:
    """A 'Camera.'-prefixed name still resolves despite entity_ids being lowercase."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "Camera.Front_Porch")  # type: ignore[arg-type]
    assert resolved == "camera.front_porch"


def test_resolve_domain_prefixed_friendly_name() -> None:
    """A domain-prefixed friendly name is matched with the prefix stripped."""
    hass = _hass_with_czech_cameras()
    resolved = _resolve_camera_entity_id(hass, "camera.kamera obývák 2")  # type: ignore[arg-type]
    assert resolved == "camera.kamera_obyvak_2"


def test_available_camera_names_falls_back_to_entity_id() -> None:
    """A camera without a friendly name is listed by its entity_id."""
    names = _available_camera_names(FakeHass([FakeState("camera.nameless")]))  # type: ignore[arg-type]
    assert names == "camera.nameless"


@pytest.mark.asyncio
async def test_camera_tool_reports_when_no_cameras_exist() -> None:
    """With no camera entities at all, the tool says no cameras are available."""
    hass = FakeHass([FakeState("sensor.only_sensor", "Only Sensor")])
    config = {"configurable": {"hass": hass, "vlm_model": MagicMock()}}

    capture = AsyncMock()
    with patch(
        "custom_components.home_generative_agent.agent.tools._get_camera_image",
        new=capture,
    ):
        result = await get_and_analyze_camera_image.coroutine(  # type: ignore[misc]
            camera_name="garage camera", detection_keywords=None, config=config
        )

    capture.assert_not_awaited()
    assert result == "Camera 'garage camera' was not found. No cameras are available."


def test_resolve_ambiguous_friendly_names_returns_none() -> None:
    """Two cameras normalizing to the same name must not resolve arbitrarily."""
    hass = FakeHass(
        [
            FakeState("camera.front_door_1", "Front Door"),
            FakeState("camera.front_door_2", "Front-Door"),
        ]
    )
    assert _resolve_camera_entity_id(hass, "front door") is None  # type: ignore[arg-type]


def test_resolve_none_friendly_name_does_not_crash() -> None:
    """A camera whose friendly_name attribute is None must not break resolution."""
    broken = FakeState("camera.broken")
    broken.attributes[ATTR_FRIENDLY_NAME] = None
    hass = FakeHass([broken, FakeState("camera.kamera4", "kamera 4")])
    assert _resolve_camera_entity_id(hass, "kamera 4") == "camera.kamera4"  # type: ignore[arg-type]


def test_resolve_non_latin_friendly_name() -> None:
    """Cyrillic names resolve via the casefolded unicode fallback."""
    hass = FakeHass([FakeState("camera.ulice", "Камера улица")])
    resolved = _resolve_camera_entity_id(hass, "камера улица")  # type: ignore[arg-type]
    assert resolved == "camera.ulice"


def test_resolve_non_latin_does_not_match_latin() -> None:
    """The unicode fallback never cross-matches unrelated Latin names."""
    hass = FakeHass([FakeState("camera.kamera4", "kamera 4")])
    assert _resolve_camera_entity_id(hass, "камера") is None  # type: ignore[arg-type]


def test_available_camera_names_caps_long_lists() -> None:
    """The not-found hint is capped so it cannot overflow the model context."""
    hass = FakeHass(
        [FakeState(f"camera.cam_{i:03d}", f"Cam {i:03d}") for i in range(40)]
    )
    names = _available_camera_names(hass)  # type: ignore[arg-type]
    assert names.endswith("and 15 more")
    assert names.count(",") == 25


def test_resolve_mixed_script_name_with_digits() -> None:
    """Mixed Cyrillic-plus-digit names keep their script during matching."""
    hass = FakeHass(
        [
            FakeState("camera.ulice_1", "Камера улица 1"),
            FakeState("camera.genkan_1", "玄関 1"),
        ]
    )
    assert _resolve_camera_entity_id(hass, "камера улица 1") == "camera.ulice_1"  # type: ignore[arg-type]
    assert _resolve_camera_entity_id(hass, "玄関 1") == "camera.genkan_1"  # type: ignore[arg-type]


def test_mixed_script_names_do_not_collapse_to_digits() -> None:
    """The ASCII fold must not reduce foreign names to their digits alone."""
    assert _normalize_camera_name("Камера улица 1") != "1"
    assert _normalize_camera_name("Камера улица 1") != _normalize_camera_name("玄関 1")
