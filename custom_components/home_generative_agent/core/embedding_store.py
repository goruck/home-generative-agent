"""
Keep the memory store's tables present and its vector size matched (#760).

The store's vector column (``store_vectors.embedding``, created by langgraph's
``PostgresStore``) gets its size once, when the table is first created. Two
setups then fail every embedding write with nothing raised where a user looks:

- **Tables gone after setup created them.** Setup runs the store and
  checkpointer migrations once, and remembers it in the entry's data. A
  database that was swapped for a fresh one, or a table that was dropped
  (``store_vectors`` is the obvious one to drop to change its size), was never
  set up again. ``async_store_tables`` tells setup which are gone, and
  ``async_forget_migrations`` lets them be created again: langgraph's
  ``setup()`` skips every migration its record table lists, and each record
  is cleared only when a table its migrations create is gone. All of them are
  ``IF NOT EXISTS``, so running them again rebuilds exactly what is missing.
- **A provider whose vectors are another size.** OpenAI and Gemini are asked
  for ``EMBEDDING_MODEL_DIMS``; an OpenAI-compatible server returns its
  model's own size, so it is measured with one embedding. Ollama servers have
  their own issues (``embedding_model_guard``), which name the model to pull.
  A wrong primary and wrong fallbacks get separate issues, since the fix
  differs: only the store's size can follow the primary.

Like the other per-entry issues they are not persistent: setup re-syncs them
every time, and unload and removal clear them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psycopg
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from psycopg import sql

from ..const import (  # noqa: TID252
    DOMAIN,
    EMBEDDING_MODEL_DIMS,
    PROVIDER_TYPE_LABELS,
)
from .embedding_model_guard import (
    ACTIVE_ENTRY_STATES,
    STORE_VECTORS_TABLE,
    async_store_vector_dims,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

LOGGER = logging.getLogger(__name__)

ISSUE_EMBEDDING_DIMS_MISMATCH = "embedding_dims_mismatch"
# A fallback can only be of the primary's type (#751), and until each
# OpenAI-compatible provider gets a client of its own (#853) one shares the
# primary's, so this issue has nothing to report yet; it is kept for then.
ISSUE_FALLBACK_EMBEDDING_DIMS_MISMATCH = "fallback_embedding_dims_mismatch"
ISSUE_EMBEDDING_FALLBACK_OTHER_MODEL = "embedding_fallback_other_model"

# Tables langgraph's PostgresStore and AsyncPostgresSaver create, and the
# record table of each one's migrations.
STORE_TABLES = ("store",)
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
VECTOR_TABLES = (STORE_VECTORS_TABLE,)
STORE_MIGRATIONS = "store_migrations"
CHECKPOINT_MIGRATIONS = "checkpoint_migrations"
VECTOR_MIGRATIONS = "vector_migrations"
MIGRATION_RECORDS: dict[str, tuple[str, ...]] = {
    STORE_MIGRATIONS: STORE_TABLES,
    CHECKPOINT_MIGRATIONS: CHECKPOINT_TABLES,
    VECTOR_MIGRATIONS: VECTOR_TABLES,
}
_SCHEMA_TABLES = tuple(
    name for record, created in MIGRATION_RECORDS.items() for name in (record, *created)
)
_STORE_TABLES_SQL = """
    SELECT name FROM unnest(%(tables)s::text[]) AS name
    WHERE to_regclass(name) IS NOT NULL
"""
_SCHEMA_TIMEOUT_S = 5.0

# Providers asked for EMBEDDING_MODEL_DIMS (OpenAI's `dimensions`, Gemini's
# `output_dimensionality`); an OpenAI-compatible server is not asked for one.
_FIXED_SIZE_PROVIDERS = frozenset({"openai", "gemini"})
OPENAI_COMPATIBLE = "openai_compatible"
OLLAMA = "ollama"
_PROBE_TEXT = "embedding size check"
_PROBE_TIMEOUT_S = 15.0


async def async_store_tables(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> frozenset[str] | None:
    """Return which of the store's tables exist, or None if it cannot be told."""
    try:
        async with (
            asyncio.timeout(_SCHEMA_TIMEOUT_S),
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(_STORE_TABLES_SQL, {"tables": list(_SCHEMA_TABLES)})
            rows = await cur.fetchall()
    except (TimeoutError, psycopg.Error) as err:
        LOGGER.warning(
            "Could not read which memory store tables exist, so missing ones "
            "are not created again this time: %s",
            err,
        )
        return None
    return frozenset(str(row["name"]) for row in rows)


def missing_tables(tables: frozenset[str] | None, names: Iterable[str]) -> list[str]:
    """Return the `names` known to be gone; none when it cannot be told."""
    if tables is None:
        return []
    return [name for name in names if name not in tables]


async def async_forget_migrations(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    tables: frozenset[str],
    records: Iterable[str],
) -> None:
    """
    Forget the migrations of `records` whose tables are gone.

    So the next ``setup()`` runs them all again. A record table that is gone
    itself is skipped: ``setup()`` creates it empty.
    """
    stale = [
        record
        for record in records
        if record in tables and missing_tables(tables, MIGRATION_RECORDS[record])
    ]
    if not stale:
        return
    async with pool.connection() as conn, conn.cursor() as cur:
        for record in stale:
            await cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(record)))


@dataclass(frozen=True)
class EmbeddingProvider:
    """One provider in the embedding chain, and the client it embeds with."""

    key: str
    name: str
    provider_type: str
    embeddings: Any
    primary: bool


def embedding_chain_providers(
    chain: Sequence[Any],
    instance_for: Callable[[Any], Any],
    *,
    primary_type: str,
    primary_embeddings: Any,
) -> list[EmbeddingProvider]:
    """
    Return the embedding chain's providers the size check measures.

    ``chain`` is the resolved chain of provider configs, primary first, and
    ``instance_for`` gives each one's client. Ollama servers have their own
    check, which names the model to pull. A provider whose client was not
    built (its health check failed at boot) is kept: OpenAI's and Gemini's
    size is known without it, and an OpenAI-compatible one is then unknown,
    which leaves its issue as it is. A provider whose client is an earlier
    one's (one client per type but Ollama, #853) is that provider again, so it
    is measured once. Without a chain, the primary alone.
    """
    if chain:
        providers: list[EmbeddingProvider] = []
        for index, p in enumerate(chain):
            if p.provider_type == OLLAMA:
                continue
            instance = instance_for(p)
            if instance is not None and any(
                instance is q.embeddings for q in providers
            ):
                continue
            providers.append(
                EmbeddingProvider(
                    p.entry_id, p.name, p.provider_type, instance, primary=index == 0
                )
            )
        return providers
    if primary_embeddings is None or primary_type == OLLAMA:
        return []
    return [
        EmbeddingProvider(
            primary_type,
            PROVIDER_TYPE_LABELS.get(primary_type, primary_type),
            primary_type,
            primary_embeddings,
            primary=True,
        )
    ]


async def _provider_size(provider: EmbeddingProvider) -> int | None:
    """Return the size of a provider's vectors, or None if it cannot be told."""
    if provider.provider_type in _FIXED_SIZE_PROVIDERS:
        return EMBEDDING_MODEL_DIMS
    if provider.provider_type != OPENAI_COMPATIBLE or provider.embeddings is None:
        return None
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            vector = await provider.embeddings.aembed_query(_PROBE_TEXT)
    except Exception as err:  # noqa: BLE001
        LOGGER.debug(
            "Could not measure the embedding size of %s: %s",
            provider.name,
            type(err).__name__,
        )
        return None
    return len(vector) if isinstance(vector, list) and vector else None


def _issue_ids(entry_id: str) -> tuple[str, str]:
    """Per-entry ids of the primary's and the fallbacks' issues."""
    return (
        f"{ISSUE_EMBEDDING_DIMS_MISMATCH}_{entry_id}",
        f"{ISSUE_FALLBACK_EMBEDDING_DIMS_MISMATCH}_{entry_id}",
    )


def _format_wrong(wrong: list[tuple[str, int]]) -> tuple[str, str]:
    """Spell the wrong providers and their sizes for the issue placeholders."""
    sizes = sorted({dims for _, dims in wrong})
    if len(sizes) == 1:
        return ", ".join(name for name, _ in wrong), str(sizes[0])
    providers = ", ".join(f"{name} ({dims})" for name, dims in wrong)
    return providers, ", ".join(str(size) for size in sizes)


def _sync_issue(
    hass: HomeAssistant,
    issue_id: str,
    translation_key: str,
    sized: list[tuple[EmbeddingProvider, int | None]],
    expected: int,
) -> None:
    """
    Raise one issue for the wrong providers in `sized`, or clear it.

    An unknown size never stands in for right or wrong, so with none wrong and
    one unknown the issue is left as it is.
    """
    wrong = [(p.name, s) for p, s in sized if s is not None and s != expected]
    if not wrong:
        if all(s is not None for _, s in sized):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    names, dims = _format_wrong(wrong)
    LOGGER.error(
        "Embedding provider %s produces %s-dimension vectors, but the memory "
        "store holds %s-dimension vectors, so its embedding writes fail.",
        names,
        dims,
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
            "providers": names,
            "dims": dims,
            "expected": str(expected),
            "default_dims": str(EMBEDDING_MODEL_DIMS),
        },
    )


async def async_sync_embedding_size_issue(
    hass: HomeAssistant,
    entry_id: str,
    *,
    providers: Sequence[EmbeddingProvider],
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None,
) -> None:
    """
    Raise or clear the issues for embedding providers of the wrong size.

    ``providers`` are the embedding chain's non-Ollama providers, each right,
    wrong or unknown (a probe that failed). An unknown store size (no
    database, no vector table) decides nothing.
    """
    primary_id, fallback_id = _issue_ids(entry_id)
    if not providers or pool is None:
        for issue_id in (primary_id, fallback_id):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    expected, sizes = await asyncio.gather(
        async_store_vector_dims(pool, default=None),
        asyncio.gather(*(_provider_size(p) for p in providers)),
    )
    entry = hass.config_entries.async_get_entry(entry_id)
    if expected is None or entry is None or entry.state not in ACTIVE_ENTRY_STATES:
        return
    sized = list(zip(providers, sizes, strict=True))
    _sync_issue(
        hass,
        primary_id,
        ISSUE_EMBEDDING_DIMS_MISMATCH,
        [(p, s) for p, s in sized if p.primary],
        expected,
    )
    _sync_issue(
        hass,
        fallback_id,
        ISSUE_FALLBACK_EMBEDDING_DIMS_MISMATCH,
        [(p, s) for p, s in sized if not p.primary],
        expected,
    )


class EmbeddingsUnavailableError(HomeAssistantError):
    """No embedding provider could be built at setup (#751)."""


class UnavailableEmbeddings:
    """
    An embedding client that fails at once, for a provider that is down (#751).

    Keeps the memory store's index while no provider in the embedding chain
    could be built at setup, so a write fails instead of being saved without
    a vector, and a search fails open without waiting on the provider.
    """

    def __init__(self, provider: str) -> None:
        """Remember which provider is unavailable, for the error."""
        self.provider = provider

    def _error(self) -> EmbeddingsUnavailableError:
        return EmbeddingsUnavailableError(
            f"Embedding provider {self.provider} is not available; memory "
            "search and saving are off until it answers."
        )

    async def aembed_documents(self, _texts: list[str]) -> list[list[float]]:
        """Fail: the provider is unavailable."""
        raise self._error()

    async def aembed_query(self, _text: str) -> list[float]:
        """Fail: the provider is unavailable."""
        raise self._error()

    def embed_documents(self, _texts: list[str]) -> list[list[float]]:
        """Fail: the provider is unavailable."""
        raise self._error()

    def embed_query(self, _text: str) -> list[float]:
        """Fail: the provider is unavailable."""
        raise self._error()


def _other_model_issue_id(entry_id: str) -> str:
    """Per-entry id of the left-out fallbacks' issue."""
    return f"{ISSUE_EMBEDDING_FALLBACK_OTHER_MODEL}_{entry_id}"


@callback
def async_sync_other_model_fallback_issue(
    hass: HomeAssistant, entry_id: str, *, primary: str, left_out: Sequence[str]
) -> None:
    """
    Raise or clear the issue naming chosen fallbacks left out of the chain (#751).

    ``left_out`` are the embedding fallbacks the user picked whose provider
    type differs from the primary's, so they would embed with another model.
    """
    issue_id = _other_model_issue_id(entry_id)
    if not left_out:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_EMBEDDING_FALLBACK_OTHER_MODEL,
        translation_placeholders={
            "providers": ", ".join(left_out),
            "primary": primary,
        },
    )


@callback
def async_clear_embedding_size_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the size issues on unload and on entry removal."""
    for issue_id in _issue_ids(entry_id):
        ir.async_delete_issue(hass, DOMAIN, issue_id)


@callback
def async_clear_other_model_fallback_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the left-out fallback issue on a disable and on entry removal."""
    ir.async_delete_issue(hass, DOMAIN, _other_model_issue_id(entry_id))
