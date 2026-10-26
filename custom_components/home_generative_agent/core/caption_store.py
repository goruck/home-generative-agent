"""Where camera captions live in the store, and how one camera's are read."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Final

import psycopg

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

LOGGER = logging.getLogger(__name__)

CAPTION_NAMESPACE: Final = "video_analysis"
# The store matches a namespace as a text prefix (prefix LIKE 'video_analysis.
# side%'), so a read for "side" alone also returns "side_gate", and "_" in a
# camera name is itself a LIKE wildcard. Each caption therefore names its
# camera in this value field, and the recent-caption scan filters on it in
# the database, before its page limit (#706). The similarity search does not:
# see VideoAnalyzer._search_own_captions.
CAPTION_CAMERA_FIELD: Final = "camera"

# Backfill of the field into captions stored before it existed. One run per
# database (an advisory lock; any fixed number unique to it), walking the
# primary key a batch at a time and pausing as long as each batch took, so
# the database keeps about half its time for conversations and cameras.
_BACKFILL_BATCH: Final = 2000
_BACKFILL_LOCK_KEY: Final = 706_706
_BACKFILL_RETRIES: Final = 3
_BACKFILL_RETRY_BACKOFF_SEC: Final = 10.0
# A run that finds the lock held waits and looks again, so the rest is not
# stranded when the run holding it stops (its entry unloaded, retries spent).
_LOCK_WAIT_SEC: Final = 60.0
_LOCK_SQL: Final = "SELECT pg_try_advisory_lock(%(key)s) AS locked"
_UNLOCK_SQL: Final = "SELECT pg_advisory_unlock(%(key)s) AS unlocked"
# A row lock held elsewhere or a stalled statement ends the batch (retried),
# not the run's hold on its connection. One statement each: the pool prepares
# every statement, and a prepared one holds a single command.
_RUN_TIMEOUTS_SQL: Final = (
    "SET statement_timeout = '60s'",
    "SET lock_timeout = '10s'",
)
_RESET_TIMEOUTS_SQL: Final = ("RESET statement_timeout", "RESET lock_timeout")
# Every caption row; the namespace's "_" escaped so it is not a wildcard.
_CAPTION_PREFIX_PATTERN: Final = CAPTION_NAMESPACE.replace("_", r"\_") + ".%"
# substr() is 1-based: the camera starts right after "video_analysis.".
_CAMERA_START: Final = len(CAPTION_NAMESPACE) + 2

_PENDING_SQL: Final = """
    SELECT 1 AS pending FROM store
    WHERE prefix LIKE %(pattern)s AND NOT (value ? %(field)s)
    LIMIT 1
"""
_BATCH_SQL: Final = """
    WITH batch AS (
        SELECT prefix, key FROM store
        WHERE prefix LIKE %(pattern)s
          AND (prefix, key) > (%(after_prefix)s, %(after_key)s)
        ORDER BY prefix, key
        LIMIT %(batch)s
    ), updated AS (
        UPDATE store AS s
        SET value = s.value || jsonb_build_object(
            %(field)s::text, substr(s.prefix, %(start)s)
        )
        FROM batch
        WHERE s.prefix = batch.prefix AND s.key = batch.key
          AND NOT (s.value ? %(field)s)
        RETURNING 1
    ), last AS (
        SELECT prefix, key FROM batch ORDER BY prefix DESC, key DESC LIMIT 1
    )
    SELECT
        (SELECT count(*) FROM batch) AS seen,
        (SELECT count(*) FROM updated) AS updated,
        (SELECT prefix FROM last) AS last_prefix,
        (SELECT key FROM last) AS last_key
"""


def caption_namespace(camera: str) -> tuple[str, str]:
    """Return the store namespace a camera's captions are written under."""
    return (CAPTION_NAMESPACE, camera)


def caption_filter(camera: str, **fields: Any) -> dict[str, Any]:
    """Return the store filter that keeps one camera's captions."""
    return {CAPTION_CAMERA_FIELD: camera, **fields}


async def async_backfill_caption_cameras(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> None:
    """
    Add the camera field to captions stored without it (#706).

    The camera is the namespace's second segment (camera object ids have no
    dot), and updated_at is left as it is: it orders the newest caption
    reads. Each start first looks for a caption without the field (one scan
    of the caption rows), so captions written meanwhile by an older version
    (a downgrade then an upgrade, or another install on the same database
    still running one) are picked up by the next start. Until a caption has
    the field, its camera's recent-caption scan does not find it, so a
    repeated scene may notify once more. While another entry on the same
    database runs it, this one looks again every minute until nothing is
    left; a failure leaves the rest to the next start.
    """
    params: dict[str, Any] = {
        "pattern": _CAPTION_PREFIX_PATTERN,
        "field": CAPTION_CAMERA_FIELD,
    }
    try:
        while True:
            async with pool.connection() as conn:
                total = await _async_attempt(conn, params)
            if total is not None:
                break
            LOGGER.debug("Another run is adding the camera to stored captions.")
            await asyncio.sleep(_LOCK_WAIT_SEC)
    except asyncio.CancelledError:
        raise
    except Exception:
        LOGGER.warning(
            "Could not add the camera to the stored camera captions; the next "
            "start resumes. Until then, older captions are not found by the "
            "recent-caption scan.",
            exc_info=True,
        )
        return
    if total:
        LOGGER.info("Added the camera to %s stored camera captions.", total)


async def _async_attempt(
    conn: AsyncConnection[DictRow], params: dict[str, Any]
) -> int | None:
    """
    Make one attempt on one pooled connection.

    Returns how many captions were updated (0 when none lacked the field),
    or None when another run holds the lock. The timeouts are set before
    the first statement, so a table lock held elsewhere ends the attempt
    instead of pinning the connection. The connection goes back to the pool
    only after a clean finish; any interruption (a cancel, even one landing
    while the lock is granted, or a failed statement) closes it instead, so
    no later user inherits the lock or the timeouts. The server then ends
    that session, and releases its lock, once its statement stops, which
    the timeouts bound.
    """
    finished = False
    try:
        for statement in _RUN_TIMEOUTS_SQL:
            await conn.execute(statement)
        result: int | None = 0
        if await (await conn.execute(_PENDING_SQL, params)).fetchone():
            row = await (
                await conn.execute(_LOCK_SQL, {"key": _BACKFILL_LOCK_KEY})
            ).fetchone()
            if not row or not row["locked"]:
                result = None
            else:
                try:
                    result = await _async_run_batches(conn, params)
                finally:
                    await conn.execute(_UNLOCK_SQL, {"key": _BACKFILL_LOCK_KEY})
        for statement in _RESET_TIMEOUTS_SQL:
            await conn.execute(statement)
        finished = True
        return result
    finally:
        if not finished:
            await conn.close()


async def _async_run_batches(
    conn: AsyncConnection[DictRow], params: dict[str, Any]
) -> int:
    """Walk every caption row in key order; return how many were updated."""
    after = ("", "")
    total = 0
    retries = 0
    while True:
        started = time.monotonic()
        try:
            row = await (
                await conn.execute(
                    _BATCH_SQL,
                    {
                        **params,
                        "after_prefix": after[0],
                        "after_key": after[1],
                        "batch": _BACKFILL_BATCH,
                        "start": _CAMERA_START,
                    },
                )
            ).fetchone()
        except (
            psycopg.errors.LockNotAvailable,
            psycopg.errors.QueryCanceled,
            psycopg.errors.DeadlockDetected,
            psycopg.errors.SerializationFailure,
        ):
            # A caption locked by (or deadlocked with) a live write, or a slow
            # batch: retried from the same place after a pause.
            retries += 1
            if retries > _BACKFILL_RETRIES:
                raise
            await asyncio.sleep(_BACKFILL_RETRY_BACKOFF_SEC * retries)
            continue
        retries = 0
        if not row:
            return total
        total += row["updated"]
        if row["seen"] < _BACKFILL_BATCH or row["last_prefix"] is None:
            return total
        after = (row["last_prefix"], row["last_key"])
        # As long as the batch took, so the database keeps half its time.
        await asyncio.sleep(time.monotonic() - started)
