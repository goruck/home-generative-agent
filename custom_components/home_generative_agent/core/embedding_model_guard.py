"""
Surface an Ollama embedding model that was never pulled onto its server.

The server answers its health check, so setup builds the embeddings client
and loads normally; the missing model only shows up later as a 404 from
``/api/embed``. Long-term memory search then fails open on every turn and the
tool index never builds, which drops tool retrieval to its keyword fallback
(issue #757). Both log errors, but nothing names the cause where a user looks
for it, so this module raises a repair issue that does, and clears it on the
next setup once the model is installed.

Every reachable Ollama server that embeds is checked: the primary and any
Ollama fallback in the embedding chain, which embed with the same configured
model (#749). A server that has the model is also asked for its vector size.
The memory store's vector column gets its size once, when the table is first
created, and a model that produces another size (nomic-embed-text's 768
against the default 1024) fails every write in the same quiet way (#755). The
size is read from that column, not from the configuration, because a store
first created by another provider keeps that provider's size. A wrong primary
and a wrong fallback get separate issues, since the fix differs.

Like the other per-entry issues they are not persistent: setup re-syncs them
every time, and unload and removal clear them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import psycopg
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..const import CONF_VECTORS_BOOTSTRAPPED, DOMAIN  # noqa: TID252
from .utils import (
    fetch_ollama_models,
    format_embedding_dims,
    ollama_embedding_dims,
    ollama_model_installed,
    redact_url,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

LOGGER = logging.getLogger(__name__)

ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING = "ollama_embedding_model_missing"
ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH = "ollama_embedding_dims_mismatch"
ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH = (
    "ollama_fallback_embedding_dims_mismatch"
)

# Setup schedules the check as a background task, and HA runs the entry's
# unload hook (which clears the issue) before it cancels those tasks. A check
# whose reply lands in that gap must not re-raise the issue for an entry that
# is no longer running.
ACTIVE_ENTRY_STATES = frozenset(
    {ConfigEntryState.SETUP_IN_PROGRESS, ConfigEntryState.LOADED}
)

# The table and column langgraph's PostgresStore creates for vectors
# (VECTOR_MIGRATIONS in langgraph.store.postgres.base); a test pins them to
# that migration so an upgrade that renames them fails loudly.
STORE_VECTORS_TABLE = "store_vectors"
STORE_VECTORS_COLUMN = "embedding"

# Always one row. pgvector stores a vector column's size as its type modifier
# (-1 when declared without one); dims is NULL when the column is missing.
_STORE_VECTOR_DIMS_SQL = """
    SELECT
        to_regclass(%(table)s) IS NOT NULL AS has_table,
        (
            SELECT a.atttypmod
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass(%(table)s)
              AND a.attname = %(column)s
              AND NOT a.attisdropped
        ) AS dims
"""

_STORE_DIMS_TIMEOUT_S = 5.0


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING}_{entry_id}"


def _dims_issue_id(entry_id: str) -> str:
    """Per-entry id of the primary's vector-size issue."""
    return f"{ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH}_{entry_id}"


def _fallback_dims_issue_id(entry_id: str) -> str:
    """Per-entry id of the fallbacks' vector-size issue."""
    return f"{ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH}_{entry_id}"


def _entry_issue_ids(entry_id: str) -> tuple[str, str, str]:
    """Every issue this module raises for an entry: missing, size, fallback size."""
    return (
        _issue_id(entry_id),
        _dims_issue_id(entry_id),
        _fallback_dims_issue_id(entry_id),
    )


async def async_store_vector_dims(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None,
    *,
    default: int | None,
) -> int | None:
    """
    Return the vector size embeddings must have to fit the memory store.

    The size of the store's vector column; ``default`` while the table does
    not exist yet, which is the size a new store will be created with, or
    None once setup has already created the store (a table missing then is a
    broken store, not a new one). None when it cannot be told: no pool, a
    database that cannot be asked in time (closed, busy or stalled), or a
    column without a fixed size. Callers skip the size check on None.
    """
    if pool is None:
        return None
    try:
        # One deadline for the whole read: the pool's own timeout only bounds
        # getting a connection, not a query on one that has stopped replying.
        async with (
            asyncio.timeout(_STORE_DIMS_TIMEOUT_S),
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                _STORE_VECTOR_DIMS_SQL,
                {"table": STORE_VECTORS_TABLE, "column": STORE_VECTORS_COLUMN},
            )
            row = await cur.fetchone()
    except (TimeoutError, psycopg.Error) as err:
        LOGGER.debug("Could not read the memory store's vector size: %s", err)
        return None
    if not row:
        return None
    if not row.get("has_table"):
        return default
    dims = row.get("dims")
    if isinstance(dims, int) and not isinstance(dims, bool) and dims > 0:
        return dims
    return None


async def async_sync_ollama_embedding_model_issue(  # noqa: PLR0913
    hass: HomeAssistant,
    entry_id: str,
    *,
    urls: Sequence[str],
    primary_url: str | None,
    model: str | None,
    configured_dims: int,
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None,
) -> None:
    """
    Raise or clear the issues for the entry's Ollama embedding servers.

    ``urls`` are the reachable Ollama servers that embed for this entry, and
    ``primary_url`` is the one among them that is the primary provider (None
    when Ollama is not; it then has no fallbacks either, #853). No servers
    (embeddings are not on Ollama, or there is no database to use them)
    clears every issue. A server that
    cannot list its models is a "can't tell", not a "missing"; one that
    answers with an empty list has no models at all. The size check holds
    servers that have the model to the store column's size, or to
    ``configured_dims`` before setup has created the store. Each server is
    then right, wrong or unknown (no listing, or no size reported), and an
    unknown never stands in for either: when the store's size or a server the
    decision rests on is unknown, the size issues are left as they are.
    """
    missing_id, dims_id, fallback_dims_id = _entry_issue_ids(entry_id)
    if not urls or not model:
        for issue_id in (missing_id, dims_id, fallback_dims_id):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    entry = hass.config_entries.async_get_entry(entry_id)
    bootstrapped = bool(entry and entry.data.get(CONF_VECTORS_BOOTSTRAPPED))
    # The store read does not depend on the servers: run them side by side.
    expected, listings = await asyncio.gather(
        async_store_vector_dims(
            pool, default=None if bootstrapped else configured_dims
        ),
        asyncio.gather(*(fetch_ollama_models(hass, url) for url in urls)),
    )
    missing: list[str] = []
    present: list[str] = []
    for url, installed in zip(urls, listings, strict=True):
        if installed is None:
            continue
        if ollama_model_installed(model, installed):
            present.append(url)
        else:
            missing.append(redact_url(url))
    sizes = (
        await ollama_embedding_dims(hass, present, model)
        if expected is not None
        else {}
    )

    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.state not in ACTIVE_ENTRY_STATES:
        return
    if expected is not None:
        _sync_dims_issues(
            hass,
            (dims_id, fallback_dims_id),
            model=model,
            expected=expected,
            sizes=sizes,
            # A server without the model gets the missing issue instead; one
            # that could not list its models is unknown.
            fallback_urls=[
                url
                for url in urls
                if url != primary_url and redact_url(url) not in missing
            ],
            primary_url=primary_url,
        )
    if not missing:
        ir.async_delete_issue(hass, DOMAIN, missing_id)
        return

    servers = ", ".join(missing)
    LOGGER.error(
        "Ollama embedding model %s is not installed on %s, so embeddings from "
        "that server fail. Pull it there (ollama pull %s) or choose an installed "
        "embedding model, then reload the integration.",
        model,
        servers,
        model,
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        missing_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING,
        translation_placeholders={"model": model, "url": servers},
    )


def _sync_dims_issues(  # noqa: PLR0913
    hass: HomeAssistant,
    issue_ids: tuple[str, str],
    *,
    model: str,
    expected: int,
    sizes: dict[str, int],
    fallback_urls: list[str],
    primary_url: str | None,
) -> None:
    """
    Raise the primary or the fallback vector-size issue, or clear them.

    ``sizes`` holds the servers that were measured. A primary measured wrong
    means the configured model itself is the wrong size, and choosing
    another one fixes every server, so its issue lists them all. The
    fallback issue needs a primary measured right: only then is a wrong
    fallback a different build of the model. Ollama fallbacks exist only
    behind an Ollama primary (#751, #853). An Ollama primary
    that was not measured (down at boot, no listing, no such model, no size
    reported) decides nothing, so both issues stay as they are. The fallback
    issue is cleared only once every fallback server was measured.
    """
    dims_id, fallback_dims_id = issue_ids
    wrong = {url: dims for url, dims in sizes.items() if dims != expected}
    if primary_url is not None and primary_url not in sizes:
        return
    if primary_url in wrong:
        _sync_dims_issue(
            hass, dims_id, ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH, model, expected, wrong
        )
        ir.async_delete_issue(hass, DOMAIN, fallback_dims_id)
        return
    ir.async_delete_issue(hass, DOMAIN, dims_id)
    if not wrong and any(url not in sizes for url in fallback_urls):
        return
    _sync_dims_issue(
        hass,
        fallback_dims_id,
        ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH,
        model,
        expected,
        wrong,
    )


def _sync_dims_issue(  # noqa: PLR0913
    hass: HomeAssistant,
    issue_id: str,
    translation_key: str,
    model: str,
    expected: int,
    wrong: dict[str, int],
) -> None:
    """Raise one vector-size issue for ``wrong`` servers, or clear it."""
    if not wrong:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    servers, dims = format_embedding_dims(wrong)
    LOGGER.error(
        "Ollama embedding model %s produces %s-dimension vectors on %s, but the "
        "memory store holds %s-dimension vectors, so embedding writes from "
        "there fail.",
        model,
        dims,
        servers,
        expected,
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=translation_key,
        translation_placeholders={
            "model": model,
            "dims": dims,
            "expected": str(expected),
            "url": servers,
        },
    )


@callback
def async_clear_ollama_embedding_model_issue(
    hass: HomeAssistant, entry_id: str
) -> None:
    """Drop every issue on unload and on entry removal."""
    for issue_id in _entry_issue_ids(entry_id):
        ir.async_delete_issue(hass, DOMAIN, issue_id)
