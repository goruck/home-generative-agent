# ruff: noqa: S101
"""Tests for loading the HGA dashboard cards automatically."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import custom_components.home_generative_agent as hga
from custom_components.home_generative_agent.const import HGA_FRONTEND_CARDS

COMPONENT_DIR = Path(hga.__file__).parent


def test_every_card_file_is_loaded() -> None:
    """A card added to www/ must not need a manual Resources entry."""
    on_disk = {path.name for path in (COMPONENT_DIR / "www").glob("hga-*-card.js")}
    assert set(HGA_FRONTEND_CARDS) == on_disk


def test_frontend_is_a_declared_dependency() -> None:
    """add_extra_js_url fails unless the frontend is set up first."""
    manifest = json.loads((COMPONENT_DIR / "manifest.json").read_text())
    assert "frontend" in manifest["dependencies"]


def test_card_registers_once_when_loaded_twice() -> None:
    """A leftover Resources entry evaluates the file a second time."""
    for card in HGA_FRONTEND_CARDS:
        tag = card.removesuffix(".js")
        source = (COMPONENT_DIR / "www" / card).read_text()
        guard = source.index(f'if (!customElements.get("{tag}"))')
        assert source.index("customElements.define(", guard) > guard
        assert source.index("window.customCards.push(") > guard


async def test_cards_load_with_the_release_version() -> None:
    """The version in the URL replaces a card cached by an earlier release."""
    hass = MagicMock()
    with (
        patch.object(
            hga,
            "async_get_integration",
            AsyncMock(return_value=SimpleNamespace(version="1.2.3")),
        ),
        patch.object(hga, "add_extra_js_url") as add_url,
    ):
        await hga._async_load_frontend_cards(hass)

    assert [call.args for call in add_url.call_args_list] == [
        (hass, f"/hga-card/{card}?v=1.2.3") for card in HGA_FRONTEND_CARDS
    ]
