# ruff: noqa: S101
"""Tests for loading the HGA dashboard cards automatically."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_homeassistant.common import MockConfigEntry

import custom_components.home_generative_agent as hga
from custom_components.home_generative_agent.const import DOMAIN, HGA_FRONTEND_CARDS

from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies

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


def _http_spy(hass: Any) -> AsyncMock:
    """Give ``hass`` an http object that records static path registration."""
    register = AsyncMock()
    hass.http = SimpleNamespace(
        async_register_static_paths=register, register_view=MagicMock()
    )
    return register


async def test_domain_setup_serves_and_loads_the_cards(hass: Any) -> None:
    """The card files and URLs are registered with no config entry at all."""
    register = _http_spy(hass)
    with patch.object(hga, "add_extra_js_url") as add_url:
        assert await hga.async_setup(hass, {}) is True

    assert register.await_args is not None
    (paths,) = register.await_args.args
    assert [config.url_path for config in paths] == ["/hga-card", "/hga-enroll-card"]
    assert all(Path(config.path) == COMPONENT_DIR / "www" for config in paths)
    assert len(add_url.call_args_list) == len(HGA_FRONTEND_CARDS)
    # The enroll card's endpoint, so its upload is not a bare 404.
    (view,) = hass.http.register_view.call_args.args
    assert view.url == "/api/home_generative_agent/enroll"


async def test_cards_load_when_entry_setup_fails(hass: Any) -> None:
    """
    An entry that cannot be set up no longer takes the cards with it.

    Entry setup used to register the card files near its end, so a database or
    provider that was unreachable left dashboards with "Custom element doesn't
    exist". Entry setup now registers none of it.
    """
    register = _http_spy(hass)
    with patch.object(hga, "add_extra_js_url") as add_url:
        await hga.async_setup(hass, {})
        loaded = len(add_url.call_args_list)
        entry = MockConfigEntry(domain=DOMAIN, data={})
        entry.add_to_hass(hass)
        with (
            patch.object(
                hga,
                "build_database_uri_from_entry",
                side_effect=RuntimeError("database unreachable"),
            ),
            pytest.raises(RuntimeError, match="database unreachable"),
        ):
            await hga.async_setup_entry(hass, entry)

    assert loaded == len(HGA_FRONTEND_CARDS)
    assert len(add_url.call_args_list) == loaded
    assert register.await_count == 1


async def test_entry_setup_registers_nothing_a_second_time(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A route registered twice raises, which would fail a healthy entry.

    Domain setup owns the card files and the enroll endpoint; a complete entry
    setup after it adds neither again.
    """
    register = _http_spy(hass)
    with patch.object(hga, "add_extra_js_url"):
        await hga.async_setup(hass, {})
        entry = MockConfigEntry(domain=DOMAIN, data={})
        entry.add_to_hass(hass)
        _patch_setup_dependencies(hass, monkeypatch, _fallback_setup_data())
        assert await hga.async_setup_entry(hass, entry) is True

    assert register.await_count == 1
    assert hass.http.register_view.call_count == 1


async def test_a_card_failure_does_not_stop_the_integration(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A False from domain setup would keep every entry from loading."""
    register = _http_spy(hass)
    register.side_effect = RuntimeError("route already registered")

    assert await hga.async_setup(hass, {}) is True
    assert "Could not register the HGA dashboard cards" in caplog.text


async def test_missing_card_files_are_reported(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A partial install says why the cards are missing."""
    register = _http_spy(hass)
    with patch.object(hga, "_resolve_www_dir", return_value=None):
        assert await hga.async_setup(hass, {}) is True

    register.assert_not_awaited()
    assert "dashboard card files were not found" in caplog.text
