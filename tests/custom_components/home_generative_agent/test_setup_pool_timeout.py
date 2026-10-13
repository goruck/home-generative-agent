# ruff: noqa: S101
"""Tests for setup-time pool failure handling."""

from __future__ import annotations

from typing import Any, cast

import pytest
from homeassistant.exceptions import ConfigEntryNotReady
from psycopg import OperationalError
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core.database_guard import (
    DATABASE_UNREACHABLE,
    database_unreachable,
)


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


def _patch_setup(
    monkeypatch: pytest.MonkeyPatch,
    pool_holder: dict[str, DummyPool],
    migrate: Any,
    reachable: Any = None,
) -> None:
    """Stub setup down to the database block, with ``migrate`` as the failure."""

    def _make_pool(*args: Any, **kwargs: Any) -> DummyPool:
        pool = DummyPool(*args, **kwargs)
        pool_holder["pool"] = pool
        return pool

    async def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _false_health(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(hga_component, "AsyncConnectionPool", _make_pool)
    monkeypatch.setattr(hga_component, "AsyncPostgresStore", DummyStore)
    monkeypatch.setattr(hga_component, "AsyncPostgresSaver", DummySaver)
    monkeypatch.setattr(
        hga_component, "async_check_database_reachable", reachable or _noop
    )
    # DummyPool has no connection(); without this stub the pgvector check
    # (#756) fails first and the failure under test is never reached.
    monkeypatch.setattr(hga_component, "async_ensure_pgvector", _noop)
    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _noop)
    monkeypatch.setattr(hga_component, "_bootstrap_vectors_once", _noop)
    monkeypatch.setattr(hga_component, "migrate_person_gallery", migrate)
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
    monkeypatch.setattr(hga_component, "openai_compatible_healthy", _false_health)
    monkeypatch.setattr(hga_component, "_register_services", lambda *_args: None)
    monkeypatch.setattr(
        hga_component, "_ensure_default_feature_subentries", lambda *_args: None
    )
    monkeypatch.setattr(
        hga_component, "_assign_first_provider_if_needed", lambda *_args: None
    )


@pytest.mark.asyncio
async def test_database_lost_mid_setup_retries_and_closes_the_pool(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool timeout after the preflight retries setup instead of failing it."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    pool_holder: dict[str, DummyPool] = {}

    async def _fail_migrate(*_args: Any, **_kwargs: Any) -> None:
        msg = "couldn't get a connection after 30.00 sec"
        raise PoolTimeout(msg)

    _patch_setup(monkeypatch, pool_holder, _fail_migrate)

    with pytest.raises(ConfigEntryNotReady) as exc_info:
        await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert exc_info.value.translation_key == DATABASE_UNREACHABLE
    assert exc_info.value.translation_placeholders == {
        "error": "couldn't get a connection after 30.00 sec"
    }
    assert pool_holder["pool"].open_called is True
    assert pool_holder["pool"].close_called is True


@pytest.mark.asyncio
async def test_non_connection_failure_still_fails_setup(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in the schema step is not retried; the pool is still closed."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    pool_holder: dict[str, DummyPool] = {}

    async def _fail_migrate(*_args: Any, **_kwargs: Any) -> None:
        msg = "migration bug"
        raise RuntimeError(msg)

    _patch_setup(monkeypatch, pool_holder, _fail_migrate)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is False
    assert pool_holder["pool"].close_called is True


@pytest.mark.asyncio
async def test_unreachable_database_retries_before_building_the_pool(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight's retry fires before any pool exists to leak."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    pool_holder: dict[str, DummyPool] = {}

    async def _unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise database_unreachable(OperationalError("connection refused"))

    async def _never(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError

    _patch_setup(monkeypatch, pool_holder, _never, reachable=_unreachable)

    with pytest.raises(ConfigEntryNotReady):
        await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert "pool" not in pool_holder
