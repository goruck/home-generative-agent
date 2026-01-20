# ruff: noqa: S101
"""Tests for setup-time pool failure handling."""

from __future__ import annotations

from typing import Any, cast

import pytest
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN


class DummyPool:
    """AsyncConnectionPool stand-in that tracks close calls."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the dummy pool."""
        self.open_called = False
        self.close_called = False

    async def open(self) -> None:
        """Simulate pool open."""
        self.open_called = True

    async def close(self) -> None:
        """Simulate pool close."""
        self.close_called = True


class DummyStore:
    """Minimal AsyncPostgresStore stand-in."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the dummy store."""


class DummySaver:
    """Minimal AsyncPostgresSaver stand-in."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the dummy saver."""


@pytest.mark.asyncio
async def test_setup_closes_pool_on_migration_failure(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pool should close when setup fails after it is opened."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)

    pool_holder: dict[str, DummyPool] = {}

    def _make_pool(*args: Any, **kwargs: Any) -> DummyPool:
        pool = DummyPool(*args, **kwargs)
        pool_holder["pool"] = pool
        return pool

    async def _noop_bootstrap(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _fail_migrate(*_args: Any, **_kwargs: Any) -> None:
        msg = "pool exhausted"
        raise PoolTimeout(msg)

    async def _false_health(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(hga_component, "AsyncConnectionPool", _make_pool)
    monkeypatch.setattr(hga_component, "AsyncPostgresStore", DummyStore)
    monkeypatch.setattr(hga_component, "AsyncPostgresSaver", DummySaver)
    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _noop_bootstrap)
    monkeypatch.setattr(hga_component, "_bootstrap_vectors_once", _noop_bootstrap)
    monkeypatch.setattr(hga_component, "migrate_person_gallery", _fail_migrate)
    monkeypatch.setattr(
        hga_component, "build_database_uri_from_entry", lambda _entry: "postgres://db"
    )
    monkeypatch.setattr(hga_component, "resolve_runtime_options", lambda _entry: {})
    monkeypatch.setattr(
        hga_component, "configured_ollama_urls", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(hga_component, "openai_healthy", _false_health)
    monkeypatch.setattr(hga_component, "gemini_healthy", _false_health)
    monkeypatch.setattr(hga_component, "ollama_healthy", _false_health)
    monkeypatch.setattr(hga_component, "_register_services", lambda *_args: None)
    monkeypatch.setattr(
        hga_component, "_ensure_default_feature_subentries", lambda *_args: None
    )
    monkeypatch.setattr(
        hga_component, "_assign_first_provider_if_needed", lambda *_args: None
    )

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is False
    assert pool_holder["pool"].open_called is True
    assert pool_holder["pool"].close_called is True
