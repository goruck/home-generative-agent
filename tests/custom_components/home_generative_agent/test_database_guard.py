# ruff: noqa: S101
"""
Tests for the no-database guard (#615).

An entry with no Database subentry loads on in-memory storage. That must be
visible (a log warning and a repair issue), and the two face-enrollment
paths must say what is missing instead of dereferencing a ``None`` gallery.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.core import CoreState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import pipeline_guard
from custom_components.home_generative_agent.core.database_guard import (
    ISSUE_DATABASE_NOT_CONFIGURED,
    _issue_id,
    async_clear_database_issue,
    async_sync_database_issue,
)

from .test_deferred_start_listener import _UnreachablePool
from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies


def _issue(hass: Any, entry_id: str) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _issue_id(entry_id))


def _prepare_entry(hass: Any, monkeypatch: pytest.MonkeyPatch) -> MockConfigEntry:
    """Create the entry and patch the fallback harness (no database URI)."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    _patch_setup_dependencies(hass, monkeypatch, _fallback_setup_data())
    return entry


async def _setup_without_database(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> MockConfigEntry:
    """Set up through the fallback harness, which resolves no database URI."""
    entry = _prepare_entry(hass, monkeypatch)
    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    assert entry.runtime_data.person_gallery is None
    return entry


class _ReachablePool(_UnreachablePool):
    """A pool whose one connection answers the two setup queries."""

    def connection(self) -> Any:
        cursor = MagicMock()
        cursor.execute = _async_none
        cursor.fetchone = _async_row
        cursor.__aenter__ = _async_self
        cursor.__aexit__ = _async_none
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cursor)
        conn.__aenter__ = _async_self
        conn.__aexit__ = _async_none
        return conn


async def _async_none(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _async_self(self: Any, *_args: Any, **_kwargs: Any) -> Any:
    return self


async def _async_row(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "db": "ha_db",
        "usr": "ha_user",
        "host": "127.0.0.1",
        "port": 5432,
        "schemas": ["public"],
        "search_path": "public",
        "total": 0,
    }


def _patch_configured_database(monkeypatch: pytest.MonkeyPatch, pool_cls: type) -> None:
    """Undo the harness default of "no database" so the DB branch is taken."""
    monkeypatch.setattr(
        hga_component, "build_database_uri_from_entry", lambda _e: "postgresql://x/y"
    )
    monkeypatch.setattr(hga_component, "AsyncConnectionPool", pool_cls)
    # The real langgraph stores spawn background batch tasks; stub them.
    monkeypatch.setattr(hga_component, "AsyncPostgresStore", MagicMock())
    monkeypatch.setattr(hga_component, "AsyncPostgresSaver", MagicMock())
    monkeypatch.setattr(hga_component, "_bootstrap_vectors_once", _async_none)
    monkeypatch.setattr(hga_component, "migrate_person_gallery", _async_none)
    monkeypatch.setattr(hga_component, "PersonGalleryDAO", MagicMock())


# ---------------------------------------------------------------------------
# setup raises / clears the issue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_without_database_warns_and_raises_repair_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The silent fallback branch must log at WARNING and surface an issue."""
    entry = await _setup_without_database(hass, monkeypatch)

    records = [r for r in caplog.records if "No database is configured" in r.message]
    assert records
    assert records[0].levelno == logging.WARNING
    assert "Run + Setup" in records[0].message
    issue = _issue(hass, entry.entry_id)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.translation_key == ISSUE_DATABASE_NOT_CONFIGURED
    assert issue.is_persistent is False


@pytest.mark.asyncio
async def test_fresh_entry_without_a_provider_gets_no_repair_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A just-added entry has neither a provider nor a database.

    That is onboarding (README steps 5-6), not misconfiguration: a repair
    issue seconds after "Add Integration" would train users to ignore it.
    The log line stays, at INFO.
    """
    caplog.set_level(logging.INFO)
    entry = _prepare_entry(hass, monkeypatch)
    monkeypatch.setattr(
        hga_component, "resolve_model_provider_configs", lambda *_args: {}
    )

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    assert _issue(hass, entry.entry_id) is None
    records = [r for r in caplog.records if "No database is configured" in r.message]
    assert records
    assert records[0].levelno == logging.INFO


@pytest.mark.asyncio
async def test_setup_with_a_database_clears_a_stale_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Adding the database and reloading drops the issue via the real wiring."""
    entry = _prepare_entry(hass, monkeypatch)
    async_sync_database_issue(hass, entry.entry_id, database_missing=True)
    _patch_configured_database(monkeypatch, _ReachablePool)
    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _async_none)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    assert entry.runtime_data.person_gallery is not None
    assert _issue(hass, entry.entry_id) is None
    assert "No database is configured" not in caplog.text


@pytest.mark.asyncio
async def test_unreachable_database_still_clears_the_not_configured_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    "Configured" is a property of the entry, not of the connection.

    The #615 user follows the repair text, adds the Database subentry while
    the add-on is still down, and reloads: setup fails. The issue must not
    survive that and keep claiming the subentry does not exist.
    """
    hass.set_state(CoreState.not_running)
    entry = _prepare_entry(hass, monkeypatch)
    async_sync_database_issue(hass, entry.entry_id, database_missing=True)
    _patch_configured_database(monkeypatch, _UnreachablePool)

    async def _unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise PoolTimeout

    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _unreachable)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is False
    assert _issue(hass, entry.entry_id) is None


# ---------------------------------------------------------------------------
# unload / removal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unload_drops_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled or reloading entry is not "running without a database"."""
    entry = await _setup_without_database(hass, monkeypatch)
    assert _issue(hass, entry.entry_id) is not None

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is True
    assert _issue(hass, entry.entry_id) is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_custom_integrations")
async def test_removing_the_entry_through_ha_clears_both_issues(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Removal goes through Home Assistant's dispatch, not a direct call.

    ``ConfigEntry.async_remove`` finds the hook by name on the integration
    module; a rename would pass a direct-call test and clear nothing in
    production. Custom integrations must be enabled for that lookup, or HA
    swallows ``IntegrationNotFound`` and never reaches the hook. The sibling
    PIN issue shares the per-entry id scheme and was previously left behind
    on removal.
    """
    entry = await _setup_without_database(hass, monkeypatch)
    pin_issue_id = pipeline_guard._issue_id(entry.entry_id)
    ir.async_create_issue(
        hass,
        DOMAIN,
        pin_issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=pipeline_guard.ISSUE_PIN_BYPASSED,
        translation_placeholders={"pipelines": "Kitchen"},
    )
    assert _issue(hass, entry.entry_id) is not None

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert _issue(hass, entry.entry_id) is None
    assert ir.async_get(hass).async_get_issue(DOMAIN, pin_issue_id) is None


@pytest.mark.asyncio
async def test_sync_database_issue_is_idempotent(hass: Any) -> None:
    """Raise twice, clear twice: no error, no duplicate."""
    async_sync_database_issue(hass, "entry-a", database_missing=True)
    async_sync_database_issue(hass, "entry-a", database_missing=True)
    assert _issue(hass, "entry-a") is not None

    async_sync_database_issue(hass, "entry-a", database_missing=False)
    assert _issue(hass, "entry-a") is None
    async_sync_database_issue(hass, "entry-a", database_missing=False)
    assert _issue(hass, "entry-a") is None


@pytest.mark.asyncio
async def test_issue_is_per_entry(hass: Any) -> None:
    """Two entries do not clobber each other's issue."""
    async_sync_database_issue(hass, "entry-a", database_missing=True)
    async_sync_database_issue(hass, "entry-b", database_missing=True)

    async_clear_database_issue(hass, "entry-a")
    assert _issue(hass, "entry-a") is None
    assert _issue(hass, "entry-b") is not None


# ---------------------------------------------------------------------------
# enroll_person service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enroll_person_service_refuses_before_reading_any_media(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A clear error, not AttributeError on None — and no I/O first.

    An entry with no gallery must not read a caller-chosen file or fetch a
    media source on behalf of a call that cannot succeed.
    """
    await _setup_without_database(hass, monkeypatch)

    async def _must_not_read(*_args: Any, **_kwargs: Any) -> bytes:
        msg = "media was read before the gallery check"
        raise AssertionError(msg)

    monkeypatch.setattr(hga_component, "_read_enroll_image_bytes", _must_not_read)

    with pytest.raises(HomeAssistantError, match="configured database"):
        await hass.services.async_call(
            DOMAIN,
            "enroll_person",
            {"name": "Alice", "file_path": "/media/faces/alice.jpg"},
            blocking=True,
        )
