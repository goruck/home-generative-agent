"""
Record which embedding model filled the memory store (#855).

The store's vectors (memories, camera captions, the tool index) are comparable
only with vectors of the same model, and the size check (#760) catches a model
change only when the size changes too. So the database keeps one row naming
the model that filled the store: provider type, model name, size, and for
Ollama the build's weights digest (two servers, or two pulls, can hold the
same tag in different builds). Setup compares it with the current embedding
primary:

- **No record, or no vectors yet:** the current primary is recorded. An
  install upgrading to this has no record, and its store is taken to be the
  current model's, so nobody gets an issue for a store that was never mixed.
- **Same model:** nothing to do. A digest the record lacks is filled in.
- **Another model:** a fixable repair issue names both, and the row notes
  that the current model now writes into the store too, so switching back
  later does not clear the issue: only a finished re-embed does. The fix
  re-embeds every stored vector with the primary's own client in the
  background, pausing after each batch as long as it took. Each batch commits
  with its place in the row, so a restart picks it up where it stopped; the
  run holds an advisory lock and sets its target only under it; the row is
  read and written under a row lock, so a setup check never undoes a run's
  progress; a row the store rewrites meanwhile keeps its own new vector; and
  a run that stops brings the issue back.

An unknown Ollama digest (the server was down, or does not list the model)
never stands in for a different build, nor for the same one while the record
names a build: the record, a re-embed in flight and the issue are then left
as they are, and no re-embed starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import psycopg
from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir
from langgraph.store.base.embed import get_text_at_path, tokenize_path
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..const import DOMAIN, PROVIDER_TYPE_LABELS  # noqa: TID252
from .embedding_model_guard import ACTIVE_ENTRY_STATES, STORE_VECTORS_TABLE
from .embedding_store import (
    GEMINI,
    OLLAMA,
    EmbeddingsUnavailableError,
    UnavailableEmbeddings,
)
from .utils import (
    builds_differ,
    fetch_ollama_model_digest,
    generate_embeddings,
    ollama_model_key,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from psycopg import AsyncConnection, AsyncCursor
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

LOGGER = logging.getLogger(__name__)

ISSUE_EMBEDDING_MODEL_CHANGED = "embedding_model_changed"
ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD = "embedding_fallback_other_build"

_CREATE_RECORD_SQL = """
    CREATE TABLE IF NOT EXISTS hga_embedding_model (
        id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
        model JSONB NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        reembed_target JSONB,
        reembed_cursor JSONB,
        mixed_with JSONB
    )
"""
_READ_RECORD_SQL = """
    SELECT model, reembed_target, reembed_cursor, mixed_with
    FROM hga_embedding_model WHERE id = 1
"""
# Read under a row lock inside a transaction: a run's batch commit and a
# setup check's write then never interleave.
_READ_RECORD_FOR_UPDATE_SQL = _READ_RECORD_SQL + " FOR UPDATE"
_WRITE_RECORD_SQL = """
    INSERT INTO hga_embedding_model (id, model) VALUES (1, %(model)s)
    ON CONFLICT (id) DO UPDATE
    SET model = EXCLUDED.model, recorded_at = now(),
        reembed_target = NULL, reembed_cursor = NULL, mixed_with = NULL
"""
_MARK_MIXED_SQL = """
    UPDATE hga_embedding_model
    SET mixed_with = %(mixed)s, reembed_cursor = %(cursor)s
    WHERE id = 1
"""
# A repair confirmed again for the run's own target keeps its place.
_START_REEMBED_SQL = """
    UPDATE hga_embedding_model
    SET reembed_cursor = CASE WHEN reembed_target = %(target)s
            THEN reembed_cursor END,
        reembed_target = %(target)s
    WHERE id = 1
"""
# Guarded on the place the run saved last: a setup check that noted another
# writer resets it, and the run must then stop, not write its place back.
_SAVE_CURSOR_SQL = """
    UPDATE hga_embedding_model SET reembed_cursor = %(cursor)s
    WHERE id = 1 AND reembed_target = %(target)s
        AND reembed_cursor IS NOT DISTINCT FROM %(previous)s
"""
# Guarded on the writers the run started with: one noted since stays on
# record, and the run must not certify the store without it.
_FINISH_REEMBED_SQL = """
    UPDATE hga_embedding_model
    SET model = reembed_target, recorded_at = now(),
        reembed_target = NULL, reembed_cursor = NULL, mixed_with = NULL
    WHERE id = 1 AND reembed_target = %(target)s
        AND mixed_with IS NOT DISTINCT FROM %(mixed)s
"""
# A build pulled during the run wrote into rows already done: the next run
# starts over, and the other build counts as a writer.
_BUILD_CHANGED_SQL = """
    UPDATE hga_embedding_model
    SET reembed_cursor = NULL, mixed_with = %(mixed)s
    WHERE id = 1 AND reembed_target = %(target)s
"""
# A run whose first batch could not be written (vectors of another size)
# changed nothing: drop its target so setup does not restart it forever.
_CLEAR_UNSTARTED_TARGET_SQL = """
    UPDATE hga_embedding_model SET reembed_target = NULL
    WHERE id = 1 AND reembed_target = %(target)s AND reembed_cursor IS NULL
"""
_RECORDED_MODEL_SQL = (
    "SELECT model, reembed_target FROM hga_embedding_model WHERE id = 1"
)
_HAS_VECTOR_TABLE_SQL = "SELECT to_regclass(%(table)s) IS NOT NULL AS has_table"
_HAS_VECTORS_SQL = "SELECT EXISTS (SELECT 1 FROM store_vectors) AS has_vectors"
_COUNT_VECTORS_SQL = "SELECT count(*) AS total FROM store_vectors"
# Two statements, not one with an "IS NULL OR" cursor: a prepared statement's
# generic plan cannot fold that away, so the keyset turned into a filter and
# every batch rescanned the rows done before it. The store side of the next
# batch is bounded too: a generic plan does not know the LIMIT and may
# merge-join, which would otherwise walk the store's index from the start.
_BATCH_COLUMNS = """
    SELECT sv.prefix, sv.key, sv.field_name, sv.updated_at, s.value
    FROM store_vectors sv
    JOIN store s ON s.prefix = sv.prefix AND s.key = sv.key
"""
_FIRST_BATCH_SQL = (
    _BATCH_COLUMNS
    + """
    ORDER BY sv.prefix, sv.key, sv.field_name
    LIMIT %(limit)s
"""
)
_NEXT_BATCH_SQL = (
    _BATCH_COLUMNS
    + """
    WHERE (sv.prefix, sv.key, sv.field_name) > (%(prefix)s, %(key)s, %(field)s)
        AND (s.prefix, s.key) >= (%(prefix)s, %(key)s)
    ORDER BY sv.prefix, sv.key, sv.field_name
    LIMIT %(limit)s
"""
)
# A row the store rewrote while its batch was embedding already holds the
# current model's vector of its new text: left alone (updated_at moved).
_UPDATE_VECTOR_SQL = """
    UPDATE store_vectors SET embedding = %s
    WHERE prefix = %s AND key = %s AND field_name = %s AND updated_at = %s
"""
# A vector whose text is gone from its item would keep the old model's vector
# under a store recorded as the new model's.
_DELETE_VECTOR_SQL = """
    DELETE FROM store_vectors
    WHERE prefix = %s AND key = %s AND field_name = %s AND updated_at = %s
"""
# Any fixed number unique to this lock; it only has to match across entries.
_REEMBED_LOCK_KEY = 855_855
_REEMBED_LOCK_SQL = "SELECT pg_try_advisory_lock(%(key)s) AS locked"
_REEMBED_UNLOCK_SQL = "SELECT pg_advisory_unlock(%(key)s) AS unlocked"
# The run holds a pooled connection for hours: a stalled statement or a row
# lock held elsewhere must end it, not pin the lock forever. Reset before the
# connection goes back to the pool.
# One statement each: the pool prepares every statement, and a prepared one
# holds a single command.
_RUN_TIMEOUTS_SQL = (
    "SET statement_timeout = '120s'",
    "SET lock_timeout = '30s'",
)
_RESET_TIMEOUTS_SQL = ("RESET statement_timeout", "RESET lock_timeout")

_RECORD_TIMEOUT_S = 10.0
# Quoted in the repair text (strings.json and translations) and the docs.
_BATCH_SIZE = 64
_PROGRESS_LOG_S = 300.0
# One embedding call; a server that hangs is retried like one that fails.
_EMBED_TIMEOUT_S = 180.0
# Waits after a failed embedding call before the run stops; the repair issue
# comes back so it can be started again.
_RETRY_DELAYS_S = (60.0, 120.0, 300.0)
_DIGEST_READ_TIMEOUT_S = 3.0
_DIGEST_FETCH_TIMEOUT_S = 5.0
# Waits before each re-read of a build setup could not read: setup reads it
# with the short health-check timeout, which Home Assistant's busy startup
# often runs past.
_BUILD_REREAD_DELAYS_S = (0.0, 10.0, 30.0)
_GEMINI_MODEL_PREFIX = "models/"
_WHOLE_DOCUMENT = "$"


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """The embedding model a store's vectors come from."""

    provider_type: str
    model: str
    dims: int | None = None
    digest: str | None = None

    def same_name(self, other: EmbeddingFingerprint) -> bool:
        """Return True for the same provider type and model name."""
        return (self.provider_type, self.model) == (other.provider_type, other.model)

    def same_model(self, other: EmbeddingFingerprint) -> bool:
        """Return True if both embed alike; an unknown digest decides nothing."""
        return self.same_name(other) and not builds_differ(self.digest, other.digest)

    @property
    def undetermined(self) -> bool:
        """An Ollama model whose build could not be read."""
        return self.provider_type == OLLAMA and self.digest is None

    @property
    def label(self) -> str:
        """Name the model for a user: provider type, model, short digest."""
        name = f"{PROVIDER_TYPE_LABELS.get(self.provider_type, self.provider_type)} "
        name += self.model
        return f"{name} ({self.digest[:12]})" if self.digest else name

    def as_json(self) -> Jsonb:
        """Return the fingerprint as a JSONB parameter."""
        return Jsonb(asdict(self))

    @classmethod
    def from_json(cls, raw: Any) -> EmbeddingFingerprint | None:
        """Read a stored fingerprint; None for anything that is not one."""
        if not isinstance(raw, dict):
            return None
        provider_type, model = raw.get("provider_type"), raw.get("model")
        if not isinstance(provider_type, str) or not isinstance(model, str):
            return None
        dims, digest = raw.get("dims"), raw.get("digest")
        return cls(
            provider_type,
            model,
            dims if isinstance(dims, int) else None,
            digest if isinstance(digest, str) and digest else None,
        )


def embedding_fingerprint(
    provider_type: str, model: str, *, dims: int | None, digest: str | None
) -> EmbeddingFingerprint:
    """
    Fingerprint a configured model, its name spelled the way its server reads it.

    Ollama fills in a default tag, and Gemini takes a model with or without
    its ``models/`` prefix.
    """
    name = model.strip()
    if provider_type == OLLAMA:
        name = ollama_model_key(name)
    elif provider_type == GEMINI:
        name = name.removeprefix(_GEMINI_MODEL_PREFIX)
    return EmbeddingFingerprint(provider_type, name, dims, digest or None)


@dataclass(frozen=True)
class _Record:
    model: EmbeddingFingerprint | None
    target: EmbeddingFingerprint | None
    cursor: list[str] | None
    # Other models that wrote into the store since it was last whole, and the
    # stored value they were read from (the finish is guarded on it).
    mixed: tuple[EmbeddingFingerprint, ...] = ()
    mixed_json: Any = None

    @property
    def partly_reembedded(self) -> bool:
        """A re-embed to `target` committed at least one batch."""
        return self.target is not None and self.cursor is not None

    def holders(self) -> list[EmbeddingFingerprint]:
        """Every model whose vectors the store may hold, the recorded one first."""
        held: list[EmbeddingFingerprint] = [self.model] if self.model else []
        extra = [self.target] if self.partly_reembedded and self.target else []
        for fingerprint in (*extra, *self.mixed):
            if not any(fingerprint.same_model(h) for h in held):
                held.append(fingerprint)
        return held

    @property
    def held_label(self) -> str:
        """Name what the store holds: its model, plus any that wrote into it."""
        return " + ".join(f.label for f in self.holders())


def _fingerprints(raw: Any) -> tuple[EmbeddingFingerprint, ...]:
    if not isinstance(raw, list):
        return ()
    parsed = (EmbeddingFingerprint.from_json(item) for item in raw)
    return tuple(f for f in parsed if f is not None)


async def _create_record_table(cur: AsyncCursor[DictRow]) -> None:
    # Two entries starting together can both run it; IF NOT EXISTS is not
    # safe against that, and the loser's table exists all the same.
    with contextlib.suppress(psycopg.errors.UniqueViolation):
        await cur.execute(_CREATE_RECORD_SQL)


async def _read_record(
    cur: AsyncCursor[DictRow], *, for_update: bool = False
) -> _Record | None:
    await cur.execute(_READ_RECORD_FOR_UPDATE_SQL if for_update else _READ_RECORD_SQL)
    row = await cur.fetchone()
    if row is None:
        return None
    cursor = row["reembed_cursor"]
    valid_cursor = (
        isinstance(cursor, list)
        and len(cursor) == 3  # noqa: PLR2004
        and all(isinstance(part, str) for part in cursor)
    )
    return _Record(
        EmbeddingFingerprint.from_json(row["model"]),
        EmbeddingFingerprint.from_json(row["reembed_target"]),
        cursor if valid_cursor else None,
        _fingerprints(row.get("mixed_with")),
        row.get("mixed_with"),
    )


def _changed_issue_id(entry_id: str) -> str:
    return f"{ISSUE_EMBEDDING_MODEL_CHANGED}_{entry_id}"


def _other_build_issue_id(entry_id: str) -> str:
    return f"{ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD}_{entry_id}"


async def _has_vectors(cur: AsyncCursor[DictRow]) -> bool | None:
    """Return whether the store holds any vector; None without a vector table."""
    await cur.execute(_HAS_VECTOR_TABLE_SQL, {"table": STORE_VECTORS_TABLE})
    row = await cur.fetchone()
    if not row or not row["has_table"]:
        return None
    await cur.execute(_HAS_VECTORS_SQL)
    row = await cur.fetchone()
    return bool(row and row["has_vectors"])


class _Decision(StrEnum):
    MATCH = "match"
    WAIT = "wait"
    RESUME = "resume"
    MISMATCH = "mismatch"


def _decide(record: _Record, current: EmbeddingFingerprint) -> _Decision:
    """
    Compare a readable record holding vectors with the current model.

    An Ollama build that could not be read never stands in for a match or a
    difference while the record names a build of that model: the record,
    any re-embed in flight and the issue are left as they are (WAIT).
    """
    model, target = record.model, record.target
    if model is None:
        return _Decision.MISMATCH
    named = [model, *([target] if target else []), *record.mixed]
    if current.undetermined and any(
        f.same_name(current) and f.digest is not None for f in named
    ):
        return _Decision.WAIT
    if target is not None and target.same_model(current):
        return _Decision.RESUME
    if model.same_model(current) and len(record.holders()) == 1:
        return _Decision.MATCH
    return _Decision.MISMATCH


async def _async_mark_mixed(
    cur: AsyncCursor[DictRow], record: _Record, current: EmbeddingFingerprint
) -> _Record:
    """
    Note that `current` writes into a store recorded as another model's.

    A re-embed to a third model that already committed batches loses its
    place: `current`'s writes can land before its cursor, so it starts over
    when resumed, and its target counts as a writer too.
    """
    mixed = list(record.mixed)
    cursor = record.cursor
    if record.partly_reembedded and record.target is not None:
        if not any(record.target.same_model(m) for m in mixed):
            mixed.append(record.target)
        cursor = None
    if not any(current.same_model(f) for f in (*filter(None, [record.model]), *mixed)):
        mixed.append(current)
    if tuple(mixed) == record.mixed and cursor == record.cursor:
        return record
    await cur.execute(
        _MARK_MIXED_SQL,
        {
            "mixed": Jsonb([asdict(f) for f in mixed]),
            "cursor": Jsonb(cursor) if cursor is not None else None,
        },
    )
    return replace(
        record,
        cursor=cursor,
        mixed=tuple(mixed),
        mixed_json=[asdict(f) for f in mixed],
    )


async def _async_check_record(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    current: EmbeddingFingerprint,
) -> tuple[_Decision, _Record | None]:
    """
    Decide what the store's record means for `current`; record what it must.

    A store with no vector table has nothing to record (MATCH, no record).
    One with no record, an unreadable one or no vectors records `current`.
    Back on the recorded model with nothing else written, a re-embed to
    another model that never committed a batch is dropped, and a build the
    record lacks is filled in. On a MISMATCH the row notes that `current`
    writes into the store from now on.
    """
    async with (
        asyncio.timeout(_RECORD_TIMEOUT_S),
        pool.connection() as conn,
        conn.cursor() as cur,
    ):
        await _create_record_table(cur)
        async with conn.transaction():
            record = await _read_record(cur, for_update=True)
            has_vectors = await _has_vectors(cur)
            if has_vectors is None:
                return _Decision.MATCH, None
            if (
                (record is None or record.model is None)
                and has_vectors
                and current.undetermined
            ):
                # Nothing on record yet and the build is unread: a fallback
                # may be writing, so record nothing until the build is known.
                return _Decision.WAIT, None
            if record is None or record.model is None or not has_vectors:
                if record is not None and record.model is None and has_vectors:
                    LOGGER.warning(
                        "The memory store's embedding model record could not be "
                        "read; recording %s.",
                        current.label,
                    )
                await cur.execute(_WRITE_RECORD_SQL, {"model": current.as_json()})
                LOGGER.info(
                    "Recorded %s as the memory store's embedding model.",
                    current.label,
                )
                return _Decision.MATCH, None
            decision = _decide(record, current)
            if decision is _Decision.MATCH and (
                record.target is not None
                or (record.model.digest is None and current.digest is not None)
            ):
                await cur.execute(_WRITE_RECORD_SQL, {"model": current.as_json()})
            elif decision is _Decision.MISMATCH:
                record = await _async_mark_mixed(cur, record, current)
            return decision, record


async def async_sync_embedding_record(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """
    Compare the store's model record with the current primary, at setup.

    Records the current model when there is no record or no vector yet, and
    raises the model-changed issue when they differ, unless a re-embed to the
    current model was started, which this resumes.
    """
    rd = entry.runtime_data
    pool, current = rd.pool, rd.embedding_fingerprint
    if pool is None or current is None or entry.state not in ACTIVE_ENTRY_STATES:
        return
    if current.undetermined and rd.embedding_digest_source is not None:
        current = await _async_learn_primary_build(hass, entry, current)
        if entry.state not in ACTIVE_ENTRY_STATES:
            return
    issue_id = _changed_issue_id(entry.entry_id)
    try:
        decision, record = await _async_check_record(pool, current)
    except (TimeoutError, psycopg.Error) as err:
        LOGGER.warning(
            "Could not read which embedding model filled the memory store: %s", err
        )
        return
    if entry.state not in ACTIVE_ENTRY_STATES or decision is _Decision.WAIT:
        return
    if decision is _Decision.MATCH or record is None:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    if decision is _Decision.RESUME:
        if _start_reembed_task(hass, entry, start=False):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
        else:
            _raise_changed_issue(hass, entry, record.held_label, current.label)
        return
    LOGGER.warning(
        "The memory store holds vectors of %s, but the embedding model is now %s, "
        "so memory search returns wrong results until the store is re-embedded.",
        record.held_label,
        current.label,
    )
    _raise_changed_issue(hass, entry, record.held_label, current.label)


@callback
def _raise_changed_issue(
    hass: HomeAssistant, entry: ConfigEntry, recorded: str, current: str
) -> None:
    ir.async_create_issue(
        hass,
        DOMAIN,
        _changed_issue_id(entry.entry_id),
        is_fixable=True,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_EMBEDDING_MODEL_CHANGED,
        translation_placeholders={"recorded": recorded, "current": current},
        data={"entry_id": entry.entry_id},
    )


def _can_reembed(entry: ConfigEntry) -> bool:
    """
    Return True when this entry can re-embed with its configured primary.

    The run embeds with the primary's own client, never through the fallback
    chain, whose members' builds are not all known. Not for an Ollama primary
    whose build is unknown: the run would be recorded as a build nobody read.
    """
    rd = getattr(entry, "runtime_data", None)
    return (
        entry.state in ACTIVE_ENTRY_STATES
        and rd is not None
        and rd.pool is not None
        and rd.embedding_fingerprint is not None
        and not rd.embedding_fingerprint.undetermined
        and rd.primary_embeddings is not None
        and not isinstance(rd.primary_embeddings, UnavailableEmbeddings)
    )


async def _async_primary_build(hass: HomeAssistant, entry: ConfigEntry) -> str | None:
    """Read the Ollama primary's build now; None if it is not Ollama or unread."""
    source = entry.runtime_data.embedding_digest_source
    if source is None:
        return None
    url, model = source
    return await fetch_ollama_model_digest(
        hass, url, model, timeout_s=_DIGEST_FETCH_TIMEOUT_S
    )


async def _async_learn_primary_build(
    hass: HomeAssistant, entry: ConfigEntry, current: EmbeddingFingerprint
) -> EmbeddingFingerprint:
    """Read the Ollama primary's build setup missed; unchanged if still unread."""
    for delay in _BUILD_REREAD_DELAYS_S:
        if delay:
            await asyncio.sleep(delay)
        if entry.state not in ACTIVE_ENTRY_STATES:
            return current
        build = await _async_primary_build(hass, entry)
        if build is not None:
            learned = replace(current, digest=build)
            entry.runtime_data.embedding_fingerprint = learned
            return learned
    LOGGER.warning(
        "Could not read which build of %s the embedding primary serves; "
        "the store's model record is checked at the next reload.",
        current.model,
    )
    return current


async def async_start_reembed(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """
    Start re-embedding the store with the current model, from the repair.

    The Ollama primary's build is read again first: one unread at setup is
    learned now, and one changed since setup means the target is stale, so
    the repair waits for a reload. Returns False when it cannot start: the
    entry is not loaded, it has no database, or its primary is unavailable.
    The run records its target once it holds the lock, so a run another
    entry holds is never redirected.
    """
    rd = getattr(entry, "runtime_data", None)
    if rd is None or rd.embedding_fingerprint is None:
        return False
    fingerprint = rd.embedding_fingerprint
    if rd.embedding_digest_source is not None:
        build = await _async_primary_build(hass, entry)
        if build is not None and fingerprint.digest is None:
            rd.embedding_fingerprint = replace(fingerprint, digest=build)
        elif builds_differ(build, fingerprint.digest):
            LOGGER.warning(
                "The embedding primary now serves another build of %s than at "
                "startup; reload Home Generative Agent, then start the repair.",
                fingerprint.model,
            )
            return False
    return _start_reembed_task(hass, entry, start=True)


def _start_reembed_task(
    hass: HomeAssistant, entry: ConfigEntry, *, start: bool
) -> bool:
    """Start the run unless one is going; False when it cannot run."""
    rd = entry.runtime_data
    task = rd.reembed_task
    if task is not None and not task.done():
        return True
    if not _can_reembed(entry):
        LOGGER.warning(
            "Re-embedding the memory store waits for its embedding primary to "
            "answer; start the repair again once it does."
        )
        return False
    target = rd.embedding_fingerprint

    @callback
    def _stopped(recorded: str) -> None:
        # The fix flow deleted the issue when the run started: bring it back
        # so a stopped run can be started again.
        if entry.state in ACTIVE_ENTRY_STATES:
            _raise_changed_issue(hass, entry, recorded, target.label)

    async def _current_build() -> str | None:
        # Read before the run records its target: a build pulled during the
        # run would otherwise be certified as the one read at the start.
        return await _async_primary_build(hass, entry)

    rd.reembed_task = entry.async_create_background_task(
        hass,
        _async_reembed(
            rd.pool,
            rd.primary_embeddings,
            target,
            start=start,
            stopped=_stopped,
            current_build=_current_build,
        ),
        name="hga_embedding_reembed",
    )
    return True


def text_at_field(value: Any, field_name: str) -> str | None:
    """
    Return the text a stored vector was embedded from, or None.

    ``field_name`` is the path langgraph embedded, with ``.<n>`` appended when
    the path held several texts (langgraph's ``_prepare_batch_PUT_queries``);
    ``$`` is the whole document.
    """
    if field_name == _WHOLE_DOCUMENT:
        texts = get_text_at_path(value, _WHOLE_DOCUMENT)
        return texts[0] if texts and texts[0] else None
    texts = get_text_at_path(value, tokenize_path(field_name))
    if len(texts) == 1:
        return texts[0] or None
    base, _, index = field_name.rpartition(".")
    if base and index.isdigit():
        texts = get_text_at_path(value, tokenize_path(base))
        if int(index) < len(texts):
            return texts[int(index)] or None
    return None


async def _embed_with_retry(
    embeddings: Any, texts: Sequence[str]
) -> tuple[list[list[float]], float]:
    """
    Embed `texts`, retrying a failed or hung call a few times.

    Also returns how long the successful call took, retries and waits aside,
    which is how long the run pauses after the batch.
    """
    for delay in (*_RETRY_DELAYS_S, None):
        started = time.monotonic()
        try:
            async with asyncio.timeout(_EMBED_TIMEOUT_S):
                vectors = await generate_embeddings(embeddings, texts)
        except EmbeddingsUnavailableError:
            raise
        except Exception as err:
            if delay is None:
                raise
            LOGGER.warning(
                "Re-embedding the memory store: embedding failed (%s); "
                "trying again in %d s.",
                type(err).__name__,
                delay,
            )
            await asyncio.sleep(delay)
            continue
        took = time.monotonic() - started
        if len(vectors) != len(texts):
            msg = f"got {len(vectors)} vectors for {len(texts)} texts"
            raise ValueError(msg)
        return vectors, took
    raise AssertionError  # unreachable: the last attempt returns or raises


class _SupersededError(Exception):
    """The record no longer names this run's target."""


class _RecordGoneError(Exception):
    """The store has no readable model record to re-embed."""


class _BuildChangedError(Exception):
    """The primary serves another build than the run's target, or none read."""

    def __init__(self, build: str | None) -> None:
        super().__init__(build)
        self.build = build


async def _async_prepare_run(
    conn: AsyncConnection[DictRow], target: EmbeddingFingerprint, *, start: bool
) -> tuple[_Record | None, str]:
    """
    Under the lock: record the run's target (`start`) or check it is still set.

    Returns the record as the run should continue it (None when there is
    nothing to re-embed) and the label of what the store holds now.
    """
    async with conn.cursor() as cur:
        await _create_record_table(cur)
        async with conn.transaction():
            record = await _read_record(cur, for_update=True)
            if record is None or record.model is None:
                raise _RecordGoneError
            recorded = record.held_label
            if start:
                if (
                    record.target is None
                    and len(record.holders()) == 1
                    and record.model.same_model(target)
                ):
                    LOGGER.info(
                        "The memory store already holds %s; nothing to re-embed.",
                        target.label,
                    )
                    return None, recorded
                await cur.execute(_START_REEMBED_SQL, {"target": target.as_json()})
                record = await _read_record(cur)
            elif record.target is None or not record.target.same_model(target):
                raise _SupersededError
    if record is None or record.target is None:
        raise _RecordGoneError
    return record, recorded


async def _async_reembed(  # noqa: PLR0913
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    embeddings: Any,
    target: EmbeddingFingerprint,
    *,
    start: bool,
    stopped: Callable[[str], None],
    current_build: Callable[[], Awaitable[str | None]],
) -> None:
    """
    Re-embed every stored vector with `target`'s model, then record it.

    Holds the advisory lock throughout and records `target` (``start``), or
    checks the record still names it, only once it holds it. `stopped`
    brings the repair back when the run ends without finishing, except on a
    cancel (the unload) and when another run took over.
    """
    recorded = target.label
    try:
        async with pool.connection() as conn:
            row = await (
                await conn.execute(_REEMBED_LOCK_SQL, {"key": _REEMBED_LOCK_KEY})
            ).fetchone()
            if not row or not row["locked"]:
                async with conn.cursor() as cur:
                    with contextlib.suppress(psycopg.Error):
                        held = await _read_record(cur)
                        if held is not None and held.model is not None:
                            recorded = held.held_label
                LOGGER.warning(
                    "Another Home Generative Agent run is re-embedding the memory "
                    "store; this one does not start. Start the repair again once "
                    "that run ends."
                )
                stopped(recorded)
                return
            try:
                for statement in _RUN_TIMEOUTS_SQL:
                    await conn.execute(statement)
                await _async_run_locked(
                    conn, embeddings, target, start=start, current_build=current_build
                )
            except asyncio.CancelledError:
                raise
            except _StoppedError as err:
                if err.recorded is not None:
                    stopped(err.recorded)
            finally:
                try:
                    await conn.execute(_REEMBED_UNLOCK_SQL, {"key": _REEMBED_LOCK_KEY})
                    for statement in _RESET_TIMEOUTS_SQL:
                        await conn.execute(statement)
                except psycopg.Error:
                    # A pooled connection still holding the lock would refuse
                    # every later run: close it so the pool drops it.
                    await conn.close()
    except asyncio.CancelledError:
        raise
    except Exception as err:  # noqa: BLE001 - the connection or lock failed
        LOGGER.warning(
            "Re-embedding the memory store with %s could not start: %s. Start "
            "the repair again.",
            target.label,
            err,
        )
        stopped(recorded)


class _StoppedError(Exception):
    """The run ended without finishing; `recorded` brings the repair back."""

    def __init__(self, recorded: str | None) -> None:
        super().__init__(recorded)
        self.recorded = recorded


async def _async_run_locked(
    conn: AsyncConnection[DictRow],
    embeddings: Any,
    target: EmbeddingFingerprint,
    *,
    start: bool,
    current_build: Callable[[], Awaitable[str | None]],
) -> None:
    """Run under the lock; every way it ends unfinished becomes _StoppedError."""
    recorded = target.label
    stored: EmbeddingFingerprint | None = None
    record_mixed: tuple[EmbeddingFingerprint, ...] = ()
    try:
        record, recorded = await _async_prepare_run(conn, target, start=start)
        if record is None or record.target is None:
            return
        stored, record_mixed = record.target, record.mixed
        await _async_reembed_rows(conn, embeddings, record, current_build)
    except asyncio.CancelledError:
        raise
    except _SupersededError:
        LOGGER.warning(
            "Re-embedding the memory store with %s stopped: another run "
            "changed its target.",
            target.label,
        )
        raise _StoppedError(None) from None
    except _RecordGoneError:
        LOGGER.warning(
            "The memory store has no embedding model record to re-embed; "
            "reload Home Generative Agent to record one."
        )
        raise _StoppedError(None) from None
    except _BuildChangedError as err:
        if err.build is not None and stored is not None:
            # Rows already done may hold the new build: start over next time.
            other = replace(stored, digest=err.build)
            mixed = [*(asdict(f) for f in record_mixed), asdict(other)]
            with contextlib.suppress(psycopg.Error):
                await conn.execute(
                    _BUILD_CHANGED_SQL,
                    {"target": stored.as_json(), "mixed": Jsonb(mixed)},
                )
        LOGGER.warning(
            "Re-embedding the memory store with %s stopped: the embedding "
            "primary %s. Reload Home Generative Agent, then start the repair "
            "again.",
            target.label,
            "now serves another build"
            if err.build is not None
            else "did not say which build it serves",
        )
        raise _StoppedError(recorded) from None
    except psycopg.DataError as err:
        if stored is not None:
            with contextlib.suppress(psycopg.Error):
                await conn.execute(
                    _CLEAR_UNSTARTED_TARGET_SQL, {"target": stored.as_json()}
                )
        LOGGER.error(  # noqa: TRY400
            "Re-embedding the memory store with %s stopped: %s. Its vectors "
            "are another size than the store's; see the embedding size "
            "repair issue.",
            target.label,
            err,
        )
        raise _StoppedError(recorded) from None
    except EmbeddingsUnavailableError as err:
        LOGGER.error(  # noqa: TRY400
            "Re-embedding the memory store with %s stopped: %s. Start the "
            "repair again once the provider answers; it resumes where it "
            "stopped.",
            target.label,
            err,
        )
        raise _StoppedError(recorded) from None
    except Exception:
        LOGGER.exception(
            "Re-embedding the memory store with %s stopped. Start the "
            "repair again; it resumes where it stopped.",
            target.label,
        )
        raise _StoppedError(recorded) from None


async def _async_reembed_rows(
    conn: AsyncConnection[DictRow],
    embeddings: Any,
    record: _Record,
    current_build: Callable[[], Awaitable[str | None]],
) -> None:
    if record.target is None:
        raise _SupersededError
    target, cursor = record.target, record.cursor
    total_row = await (await conn.execute(_COUNT_VECTORS_SQL)).fetchone()
    total = int(total_row["total"]) if total_row else 0
    LOGGER.info(
        "Re-embedding the memory store's %d vectors with %s%s. It pauses after "
        "each batch for as long as the batch took, leaving the server time for "
        "conversations.",
        total,
        target.label,
        ", resuming where it stopped" if cursor else "",
    )
    done = 0
    last_log = time.monotonic()
    target_json = target.as_json()
    while True:
        params: dict[str, Any] = {"limit": _BATCH_SIZE}
        if cursor is not None:
            params.update(zip(("prefix", "key", "field"), cursor, strict=True))
        query = _FIRST_BATCH_SQL if cursor is None else _NEXT_BATCH_SQL
        rows = await (await conn.execute(query, params)).fetchall()
        if not rows:
            break
        found: list[tuple[Any, str]] = []
        gone: list[Any] = []
        for row in rows:
            text = text_at_field(row["value"], row["field_name"])
            if text is None:
                gone.append(row)
            else:
                found.append((row, text))
        vectors, took = (
            await _embed_with_retry(embeddings, [text for _, text in found])
            if found
            else ([], 0.0)
        )
        last = rows[-1]
        next_cursor = [last["prefix"], last["key"], last["field_name"]]
        # One commit per batch, its place saved first: a run that stops never
        # leaves rewritten vectors the record does not know about, and one
        # whose target was taken writes nothing.
        async with conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                _SAVE_CURSOR_SQL,
                {
                    "cursor": Jsonb(next_cursor),
                    "target": target_json,
                    "previous": Jsonb(cursor) if cursor is not None else None,
                },
            )
            if cur.rowcount == 0:
                raise _SupersededError
            if found:
                await cur.executemany(
                    _UPDATE_VECTOR_SQL,
                    [
                        (v, r["prefix"], r["key"], r["field_name"], r["updated_at"])
                        for (r, _), v in zip(found, vectors, strict=True)
                    ],
                )
            if gone:
                await cur.executemany(
                    _DELETE_VECTOR_SQL,
                    [
                        (r["prefix"], r["key"], r["field_name"], r["updated_at"])
                        for r in gone
                    ],
                )
        cursor = next_cursor
        done += len(rows)
        if time.monotonic() - last_log >= _PROGRESS_LOG_S:
            last_log = time.monotonic()
            LOGGER.info(
                "Re-embedding the memory store: %d of about %d vectors done.",
                done,
                total,
            )
        await asyncio.sleep(took)
    await _async_finish(conn, record, current_build, done)


async def _async_finish(
    conn: AsyncConnection[DictRow],
    record: _Record,
    current_build: Callable[[], Awaitable[str | None]],
    done: int,
) -> None:
    """Record the run's target, once its build and its writers still hold."""
    if record.target is None:
        raise _SupersededError
    target = record.target
    target_json = target.as_json()
    # An Ollama target is finished only on a build read now and found the
    # same: an unread one may be the build pulled during the run.
    if target.provider_type == OLLAMA:
        build = await current_build()
        if build is None or builds_differ(build, target.digest):
            raise _BuildChangedError(build)
    mixed = Jsonb(record.mixed_json) if record.mixed_json is not None else None
    finished = await conn.execute(
        _FINISH_REEMBED_SQL, {"target": target_json, "mixed": mixed}
    )
    if finished.rowcount == 0:
        raise _SupersededError
    LOGGER.info(
        "Re-embedded the memory store with %s (%d vectors); memory search uses "
        "it from now on.",
        target.label,
        done,
    )


async def async_recorded_ollama_digest(db_uri: str, model: str) -> str | None:
    """
    Return the Ollama build the store's record names for `model`.

    Read at setup, before the pool opens, to judge Ollama fallbacks while the
    primary is down (#855). A re-embed to a build of `model` names the build
    the store is heading to; otherwise the recorded model's. None when it
    cannot be told: no record, another model, no digest, or a database that
    does not answer quickly.
    """
    try:
        async with (
            asyncio.timeout(_DIGEST_READ_TIMEOUT_S),
            await psycopg.AsyncConnection.connect(db_uri, autocommit=True) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(_RECORDED_MODEL_SQL)
            row = await cur.fetchone()
    except (TimeoutError, psycopg.Error, OSError) as err:
        LOGGER.debug("Could not read the store's recorded embedding build: %s", err)
        return None
    if not row:
        return None
    current = embedding_fingerprint(OLLAMA, model, dims=None, digest=None)
    for raw in (row.get("reembed_target"), row.get("model")):
        recorded = EmbeddingFingerprint.from_json(raw)
        if recorded is not None and recorded.same_name(current):
            return recorded.digest
    return None


@callback
def async_clear_embedding_model_changed_issue(
    hass: HomeAssistant, entry_id: str
) -> None:
    """Drop the model-changed issue on a disable and on entry removal."""
    ir.async_delete_issue(hass, DOMAIN, _changed_issue_id(entry_id))


@callback
def async_sync_other_build_fallback_issue(
    hass: HomeAssistant,
    entry_id: str,
    *,
    primary: str,
    model: str,
    left_out: Sequence[str],
) -> None:
    """
    Raise or clear the issue naming Ollama fallbacks left out for their build.

    ``left_out`` are the embedding fallbacks whose server holds another build
    (digest) of the configured model than the primary's.
    """
    issue_id = _other_build_issue_id(entry_id)
    if not left_out:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD,
        translation_placeholders={
            "providers": ", ".join(left_out),
            "primary": primary,
            "model": model,
        },
    )


@callback
def async_clear_other_build_fallback_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the other-build fallback issue on a disable and on entry removal."""
    ir.async_delete_issue(hass, DOMAIN, _other_build_issue_id(entry_id))
