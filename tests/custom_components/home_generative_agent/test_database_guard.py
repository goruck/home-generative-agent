# ruff: noqa: S101
"""
Tests for the no-database guard (#615) and the pgvector setup error (#756).

An entry with no Database subentry loads on in-memory storage. That must be
visible (a log warning and a repair issue), and the two face-enrollment
paths must say what is missing instead of dereferencing a ``None`` gallery.
A configured database that cannot use pgvector must fail setup with a
message naming the fix, not the raw PostgreSQL error.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import CoreState
from homeassistant.exceptions import (
    ConfigEntryError,
    ConfigEntryNotReady,
    HomeAssistantError,
)
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers import translation
from psycopg import OperationalError
from psycopg import errors as pg_errors
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import pipeline_guard
from custom_components.home_generative_agent.core.database_guard import (
    DATABASE_UNREACHABLE,
    ISSUE_DATABASE_NOT_CONFIGURED,
    PGVECTOR_NOT_AVAILABLE,
    PGVECTOR_NOT_PERMITTED,
    _issue_id,
    async_check_database_reachable,
    async_clear_database_issue,
    async_ensure_pgvector,
    async_sync_database_issue,
    database_unreachable,
    is_connection_failure,
)

from .test_deferred_start_listener import _UnreachablePool
from .test_fallback_setup import _fallback_setup_data, _patch_setup_dependencies

# A libpq key/value DSN for a port nothing listens on. It carries a password
# so the tests can show the message never echoes it.
_DOWN_DSN = "host=127.0.0.1 port=1 dbname=ha_db user=ha_user password=s3cret"


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
    monkeypatch.setattr(hga_component, "async_ensure_pgvector", _async_none)
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

    with pytest.raises(ConfigEntryNotReady):
        await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert _issue(hass, entry.entry_id) is None


# ---------------------------------------------------------------------------
# unreachable database retries setup (#748)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_reports_the_servers_reason() -> None:
    """A refused connection becomes a retry that carries libpq's summary."""
    with pytest.raises(ConfigEntryNotReady) as exc_info:
        await async_check_database_reachable(_DOWN_DSN)

    err = exc_info.value
    assert err.translation_key == DATABASE_UNREACHABLE
    detail = (err.translation_placeholders or {})["error"]
    assert "127.0.0.1" in detail
    assert "s3cret" not in detail


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_custom_integrations")
async def test_home_assistant_retries_an_entry_whose_database_is_down(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through HA's own setup: SETUP_RETRY with the reason, not SETUP_ERROR."""
    entry = _prepare_entry(hass, monkeypatch)
    _patch_configured_database(monkeypatch, _ReachablePool)
    monkeypatch.setattr(
        hga_component,
        "build_database_uri_from_entry",
        lambda _e: _DOWN_DSN,
    )
    # The real preflight, against a port nothing listens on.
    monkeypatch.setattr(
        hga_component, "async_check_database_reachable", async_check_database_reachable
    )

    # ConfigEntry.async_setup is where HA turns the exception into a state and
    # a scheduled retry; the manager's wrapper would first set up the
    # conversation and recorder dependencies, which the test env lacks.
    # Component setup loads the integration's translations and marks it loaded
    # before any entry is set up (homeassistant/setup.py); the card's reason is
    # rendered from them.
    await translation.async_load_integrations(hass, {DOMAIN})
    hass.config.components.add(DOMAIN)
    async with entry.setup_lock:
        await entry.async_setup(hass)

    assert entry.state is ConfigEntryState.SETUP_RETRY
    assert entry.reason is not None
    assert entry.reason.startswith("Cannot reach the PostgreSQL database yet:")
    assert "127.0.0.1" in entry.reason
    assert "s3cret" not in entry.reason


def test_connection_failures_are_told_from_bugs() -> None:
    """Only a failure to reach the server is retried."""
    assert is_connection_failure(OperationalError("down"))
    assert is_connection_failure(PoolTimeout("timeout"))
    assert not is_connection_failure(RuntimeError("bug"))
    assert not is_connection_failure(pg_errors.UndefinedTable("no table"))


def test_unreachable_message_keeps_only_the_first_line() -> None:
    """The card shows the libpq summary line, not its hints."""
    err = database_unreachable(
        OperationalError('connection to server at "db", port 5432 failed\n\tIs it up?')
    )
    assert err.translation_placeholders == {
        "error": 'connection to server at "db", port 5432 failed'
    }


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


# ---------------------------------------------------------------------------
# pgvector check before the schema setup (#756)
# ---------------------------------------------------------------------------


class _PgvectorPool:
    """A pool whose one cursor answers the pgvector state query."""

    def __init__(
        self, *, enabled: bool, available: bool, create_error: Exception | None
    ) -> None:
        self.row = {
            "database": "ha_db",
            "username": "ha_user",
            "enabled": enabled,
            "available": available,
        }
        self.create_error = create_error
        self.statements: list[str] = []

    def connection(self) -> Any:
        pool = self

        async def _execute(sql: str, *_args: Any) -> None:
            pool.statements.append(" ".join(sql.split()))
            if sql.startswith("CREATE EXTENSION") and pool.create_error:
                raise pool.create_error

        async def _fetchone() -> dict[str, Any]:
            return pool.row

        cursor = MagicMock()
        cursor.execute = _execute
        cursor.fetchone = _fetchone
        cursor.__aenter__ = _async_self
        cursor.__aexit__ = _async_none
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cursor)
        conn.__aenter__ = _async_self
        conn.__aexit__ = _async_none
        return conn


def _created_extension(pool: _PgvectorPool) -> bool:
    return "CREATE EXTENSION IF NOT EXISTS vector" in pool.statements


@pytest.mark.asyncio
async def test_enabled_pgvector_passes_without_creating_it() -> None:
    """The usual case: one read-only query, nothing else."""
    pool = _PgvectorPool(enabled=True, available=True, create_error=None)

    await async_ensure_pgvector(cast("Any", pool))

    assert len(pool.statements) == 1
    assert not _created_extension(pool)


@pytest.mark.asyncio
async def test_available_pgvector_is_enabled_when_the_user_may() -> None:
    """
    Enabled up front, embeddings or not.

    langgraph's store enables it only with an embedding model configured, and
    the person gallery's VECTOR column needs it either way.
    """
    pool = _PgvectorPool(enabled=False, available=True, create_error=None)

    await async_ensure_pgvector(cast("Any", pool))

    assert _created_extension(pool)


@pytest.mark.asyncio
async def test_pgvector_the_user_may_not_enable_names_the_fix() -> None:
    """The #756 case: pgvector on the server, not enabled here, no superuser."""
    denied = pg_errors.InsufficientPrivilege(
        'permission denied to create extension "vector"'
    )
    pool = _PgvectorPool(enabled=False, available=True, create_error=denied)

    with pytest.raises(ConfigEntryError) as excinfo:
        await async_ensure_pgvector(cast("Any", pool))

    assert excinfo.value.translation_domain == DOMAIN
    assert excinfo.value.translation_key == PGVECTOR_NOT_PERMITTED
    assert excinfo.value.translation_placeholders == {
        "database": "ha_db",
        "user": "ha_user",
    }
    assert excinfo.value.__cause__ is denied


@pytest.mark.asyncio
async def test_server_without_pgvector_names_the_fix() -> None:
    """No pgvector on the server: say so, and do not try to create it."""
    pool = _PgvectorPool(enabled=False, available=False, create_error=None)

    with pytest.raises(ConfigEntryError) as excinfo:
        await async_ensure_pgvector(cast("Any", pool))

    assert excinfo.value.translation_key == PGVECTOR_NOT_AVAILABLE
    assert excinfo.value.translation_placeholders == {
        "database": "ha_db",
        "user": "ha_user",
    }
    assert not _created_extension(pool)


@pytest.mark.asyncio
async def test_other_errors_creating_pgvector_propagate() -> None:
    """Only a permission error has the superuser fix; others stay generic."""
    pool = _PgvectorPool(
        enabled=False,
        available=True,
        create_error=pg_errors.UndefinedFile("could not load library"),
    )

    with pytest.raises(pg_errors.UndefinedFile):
        await async_ensure_pgvector(cast("Any", pool))


class _TrackedPool(_ReachablePool):
    """A reachable pool that remembers every instance setup builds."""

    instances: list[_TrackedPool] = []  # noqa: RUF012

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _TrackedPool.instances.append(self)


@pytest.mark.asyncio
async def test_pgvector_failure_fails_setup_before_the_schema(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Setup raises the translated error, closes the pool, touches no schema.

    The check runs before the bootstrap and the gallery migration, so neither
    can fail first with a bare error (the gallery's VECTOR column fails with
    `type "vector" does not exist` when no embedding model is configured).
    """
    entry = _prepare_entry(hass, monkeypatch)
    _TrackedPool.instances = []
    _patch_configured_database(monkeypatch, _TrackedPool)
    error = ConfigEntryError(
        translation_domain=DOMAIN,
        translation_key=PGVECTOR_NOT_PERMITTED,
        translation_placeholders={"database": "ha_db", "user": "ha_user"},
    )

    async def _ensure(*_args: Any) -> None:
        raise error

    schema_steps: list[str] = []

    async def _step(*_args: Any, **_kwargs: Any) -> None:
        schema_steps.append("ran")

    monkeypatch.setattr(hga_component, "async_ensure_pgvector", _ensure)
    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _step)
    monkeypatch.setattr(hga_component, "_bootstrap_vectors_once", _step)
    monkeypatch.setattr(hga_component, "migrate_person_gallery", _step)

    with pytest.raises(ConfigEntryError) as excinfo:
        await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert excinfo.value is error
    assert schema_steps == []
    assert [pool.close_calls for pool in _TrackedPool.instances] == [1]


@pytest.mark.asyncio
async def test_other_schema_errors_keep_the_generic_failure(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Anything that is not pgvector still logs and returns False, as before."""
    entry = _prepare_entry(hass, monkeypatch)
    _TrackedPool.instances = []
    _patch_configured_database(monkeypatch, _TrackedPool)

    denied = pg_errors.InsufficientPrivilege("permission denied for table store")

    async def _bootstrap(*_args: Any, **_kwargs: Any) -> None:
        raise denied

    monkeypatch.setattr(hga_component, "_bootstrap_db_once", _bootstrap)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is False
    assert "Postgresql setup failed" in caplog.text
    assert [pool.close_calls for pool in _TrackedPool.instances] == [1]
