"""
Surface an entry that is running without its PostgreSQL database.

A config entry with no Database subentry loads successfully: setup falls back
to in-memory conversation state, a null long-term store, and no person
gallery. Nothing else fails loudly, so the user sees a working chat and only
discovers the gap when memory does not persist across restarts, semantic
recall returns nothing, or face enrollment fails.

Two paths lead there (#615): ``+ Setup`` was never run after the entry was
created, or the Advanced wizard was closed at its final Database step (it
writes the feature subentries *before* that step, so a "cannot connect" error
dismissed there leaves every feature configured and no database). This module
raises a repair issue for that state and clears it once a database is
configured, so the cause is visible in Settings → Repairs instead of only in a
warning buried in the log.

The issue is deliberately not persistent (the registry default): the entry
re-syncs it on every setup, and a persistent issue would outlive an
uninstall-before-remove, where Home Assistant never reaches the integration's
``async_remove_entry`` hook (``ConfigEntry.async_remove`` returns on
``IntegrationNotFound`` first).

It also checks, before the schema is set up, that a configured database can
use pgvector (#756). The ``vector`` extension must be enabled in the
integration's own database: langgraph's store enables it only when an
embedding model is configured, and the person gallery's ``VECTOR`` column
needs it either way. On a database the user brought themselves it is often
missing (enabled in another database, or the server has no pgvector), and
enabling it takes a superuser. Setup then failed with a bare PostgreSQL
error; it now fails with a message that says which problem it is and how to
fix it.

A configured database the integration cannot reach when it sets up -- the
pgvector app still starting on a cold boot, a database restart during a
reload -- raises ``ConfigEntryNotReady`` so Home Assistant retries with
backoff (#748). It used to fail setup for good until someone reloaded the
entry by hand. The connection pool cannot say why a connection failed: a
server that is down, still starting, or rejecting the password all surface as
the same ``PoolTimeout`` after the pool's 30 s wait. So setup first connects
once directly, which fails in seconds with the server's own message, and that
message is what the integration card shows. Every connection failure is
retried, a wrong password included: telling them apart means matching server
text, which PostgreSQL translates, and a cold boot's "the database system is
starting up" reads like a permanent refusal. Only errors about the
connection itself count: psycopg raises the same ``OperationalError`` for
server-side failures such as an index over pgvector's dimension limit, and
retrying those would hide a permanent error behind a "cannot reach" message.
Settings libpq rejects before it connects (an unknown connection option) fail
setup with their own message, since no retry can fix them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
from homeassistant.core import callback
from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers import issue_registry as ir
from psycopg import errors as pg_errors
from psycopg_pool import PoolTimeout

from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

ISSUE_DATABASE_NOT_CONFIGURED = "database_not_configured"

PGVECTOR_NOT_PERMITTED = "pgvector_not_permitted"
PGVECTOR_NOT_AVAILABLE = "pgvector_not_available"
DATABASE_UNREACHABLE = "database_unreachable"
DATABASE_INVALID_SETTINGS = "database_invalid_settings"

# SQLSTATEs a server sends when it drops or refuses a connection: class 08
# (connection exception) is matched by prefix; these are the server shutting
# down or still starting (57P01 admin_shutdown, 57P02 crash_shutdown, 57P03
# cannot_connect_now) and running out of connection slots (53300).
_CONNECTION_SQLSTATES = frozenset({"57P01", "57P02", "57P03", "53300"})

# Long enough for a slow network, short enough that a database that is down
# does not hold Home Assistant's startup for long on each retry.
_CONNECT_TIMEOUT_S = 10


def is_connection_failure(err: BaseException) -> bool:
    """
    Whether ``err`` means the database could not be reached, not a bug.

    A failure on the client side (refused, timed out, bad password at
    connect) carries no SQLSTATE; a server that answered with an error
    carries one, and only the connection ones above count.
    """
    if isinstance(err, PoolTimeout):
        return True
    if not isinstance(err, psycopg.OperationalError):
        return False
    sqlstate = err.sqlstate
    return (
        sqlstate is None
        or sqlstate.startswith("08")
        or sqlstate in _CONNECTION_SQLSTATES
    )


def _first_line(err: BaseException) -> str:
    """Return the first line of ``err`` (libpq's summary), or its type."""
    text = str(err).strip()
    return text.splitlines()[0] if text else type(err).__name__


def database_unreachable(err: BaseException) -> ConfigEntryNotReady:
    """Return the retryable setup error for an unreachable database."""
    # e.g. 'connection failed: connection to server at "10.0.0.2", port 5432
    # failed: Connection refused'. It names the host and port, not the
    # password: build_postgres_uri percent-encodes the user and password, so
    # neither can spill into the host libpq reports.
    return ConfigEntryNotReady(
        translation_domain=DOMAIN,
        translation_key=DATABASE_UNREACHABLE,
        translation_placeholders={"error": _first_line(err)},
    )


async def async_check_database_reachable(db_uri: str) -> None:
    """
    Connect once to ``db_uri``, or raise a setup error saying why.

    Runs before the pool is built, so the card shows the server's reason
    within seconds instead of a bare pool timeout after 30 s. A connection
    failure raises ``ConfigEntryNotReady`` (retried); settings libpq rejects
    before connecting raise ``ConfigEntryError`` (not retried).
    """
    try:
        conn = await psycopg.AsyncConnection.connect(
            db_uri, connect_timeout=_CONNECT_TIMEOUT_S
        )
    except psycopg.OperationalError as err:
        raise database_unreachable(err) from err
    except psycopg.Error as err:
        raise ConfigEntryError(
            translation_domain=DOMAIN,
            translation_key=DATABASE_INVALID_SETTINGS,
            translation_placeholders={"error": _first_line(err)},
        ) from err
    await conn.close()


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_DATABASE_NOT_CONFIGURED}_{entry_id}"


@callback
def async_sync_database_issue(
    hass: HomeAssistant, entry_id: str, *, database_missing: bool
) -> None:
    """
    Raise or clear the repair issue for an entry with no database.

    Idempotent: safe to call on every setup and reload. Deleting an issue that
    does not exist is a no-op in the issue registry, and re-creating an
    unchanged issue neither writes storage nor fires an event, so repeated
    setups are quiet.
    """
    issue_id = _issue_id(entry_id)

    if not database_missing:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_DATABASE_NOT_CONFIGURED,
    )


@callback
def async_clear_database_issue(hass: HomeAssistant, entry_id: str) -> None:
    """
    Drop the issue outside of setup: on unload and on entry removal.

    An unloaded (disabled, reloading, or failed) entry is not "running without
    a database", so the issue must not stand while it is down; the next
    successful setup re-raises it if the database is still missing.
    """
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))


# Always one row: the names the fix has to be run against, whether vector is
# enabled in this database, and whether the server has pgvector at all.
_PGVECTOR_STATE_SQL = """
    SELECT
        current_database() AS database,
        current_user AS username,
        EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS enabled,
        EXISTS (
            SELECT 1 FROM pg_available_extensions WHERE name = 'vector'
        ) AS available
"""


async def async_ensure_pgvector(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    """
    Make sure the connected database can use pgvector, or fail setup clearly.

    Enables the extension when the server has it and the user may create it
    (the same statement langgraph's setup would run). Raises a translated
    ``ConfigEntryError`` naming the database and user, as the server reports
    them, when the server has no pgvector or the user may not enable it.
    Other database errors propagate unchanged.
    """
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_PGVECTOR_STATE_SQL)
        row = await cur.fetchone()
        if not row or row["enabled"]:
            return
        names = {"database": str(row["database"]), "user": str(row["username"])}
        if not row["available"]:
            raise ConfigEntryError(
                translation_domain=DOMAIN,
                translation_key=PGVECTOR_NOT_AVAILABLE,
                translation_placeholders=names,
            )
        try:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except pg_errors.InsufficientPrivilege as err:
            raise ConfigEntryError(
                translation_domain=DOMAIN,
                translation_key=PGVECTOR_NOT_PERMITTED,
                translation_placeholders=names,
            ) from err
