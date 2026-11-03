# ruff: noqa: S101
"""
Camera captions not written for 30 days are deleted, a batch at a time (#709).

Nothing ever deleted a stored caption, so each camera's caption namespace (and
its vectors) grew for as long as the integration ran. On one install that was
465,000 rows, 92% of them older than 30 days.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import psycopg
import pytest

from custom_components.home_generative_agent.const import CAPTION_RETENTION_DAYS
from custom_components.home_generative_agent.core import caption_store
from custom_components.home_generative_agent.core.caption_store import (
    CaptionPruner,
    async_prune_captions,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


class _Result:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    async def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _Conn:
    """Answers the prune's statements: the lock, then each batch in turn."""

    def __init__(
        self, batches: Sequence[dict[str, Any] | Exception], *, locked: bool = True
    ) -> None:
        self.batches: list[dict[str, Any] | Exception] = list(batches)
        self.locked = locked
        self.params: list[dict[str, Any]] = []
        self.statements: list[str] = []
        self.transactions = 0
        self.closed = False

    async def execute(self, query: str, params: Any = None) -> _Result:
        self.statements.append(query)
        if query is caption_store._LOCK_SQL:
            return _Result({"locked": self.locked})
        if query is caption_store._PRUNE_SQL:
            self.params.append(dict(params))
            answer = self.batches.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return _Result(answer)
        return _Result({"unlocked": True})

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        self.transactions += 1
        yield

    async def close(self) -> None:
        self.closed = True


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[_Conn]:
        yield self.conn


def _batch(seen: int, deleted: int, last: str | None) -> dict[str, Any]:
    return {
        "seen": seen,
        "deleted": deleted,
        "last_prefix": None if last is None else "video_analysis.porch",
        "last_key": last,
    }


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small batches and no real pauses."""
    monkeypatch.setattr(caption_store, "_PRUNE_BATCH", 4)
    monkeypatch.setattr(caption_store, "_PRUNE_MIN_BATCH", 1)
    monkeypatch.setattr(caption_store, "_BACKFILL_RETRY_BACKOFF_SEC", 0.0)
    real_sleep = asyncio.sleep

    async def _no_pause(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(caption_store.asyncio, "sleep", _no_pause)


async def _prune(conn: _Conn) -> int | None:
    return await async_prune_captions(cast("Any", _Pool(conn)))


@pytest.mark.asyncio
async def test_the_walk_goes_by_key_until_a_batch_is_short() -> None:
    """Each batch starts after the last key read, so none rescans a deleted row."""
    conn = _Conn([_batch(4, 3, "k4"), _batch(4, 0, "k8"), _batch(2, 2, "k10")])
    assert await _prune(conn) == 5
    assert [p["after_key"] for p in conn.params] == ["", "k4", "k8"]
    assert conn.transactions == 3
    assert conn.statements[-1] is caption_store._UNLOCK_SQL
    assert not conn.closed


@pytest.mark.asyncio
async def test_a_batch_deleting_little_does_not_end_the_walk() -> None:
    """Ending on a short delete stopped early when another writer held rows."""
    conn = _Conn([_batch(4, 0, "k4"), _batch(4, 1, "k8"), _batch(0, 0, None)])
    assert await _prune(conn) == 1
    assert len(conn.params) == 3


def test_only_old_captions_of_this_namespace_are_deleted() -> None:
    sql = caption_store._PRUNE_SQL
    # Checked on the row deleted too, so a caption rewritten meanwhile stays.
    assert "s.updated_at < NOW() - make_interval(days => %(days)s)" in sql
    assert caption_store._CAPTION_PREFIX_PATTERN == r"video\_analysis.%"
    assert CAPTION_RETENTION_DAYS == 30


@pytest.mark.asyncio
async def test_another_entry_pruning_the_database_is_left_to_it() -> None:
    conn = _Conn([], locked=False)
    assert await _prune(conn) is None
    assert conn.params == []
    assert not conn.closed


@pytest.mark.asyncio
async def test_a_slow_batch_is_retried_smaller_from_the_same_place() -> None:
    conn = _Conn(
        [
            _batch(4, 4, "k4"),
            psycopg.errors.QueryCanceled(),
            _batch(2, 2, "k6"),
            _batch(1, 1, "k7"),
        ]
    )
    assert await _prune(conn) == 7
    assert [(p["after_key"], p["batch"]) for p in conn.params] == [
        ("", 4),
        ("k4", 4),
        ("k4", 2),
        ("k6", 2),
    ]


@pytest.mark.asyncio
async def test_a_batch_that_keeps_failing_closes_the_connection() -> None:
    """The lock goes with the session, so no later user of the pool inherits it."""
    conn = _Conn([psycopg.errors.LockNotAvailable()] * 4)
    with pytest.raises(psycopg.errors.LockNotAvailable):
        await _prune(conn)
    assert conn.closed


@pytest.mark.asyncio
async def test_the_pruner_logs_a_failing_streak_once_and_quiet_runs_at_debug(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    outcomes: list[int | Exception | None] = [
        RuntimeError("down"),
        RuntimeError("down"),
        5000,
        12,
        None,
    ]

    async def _one_run(_pool: Any, _days: int = 30) -> int | None:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(caption_store, "async_prune_captions", _one_run)
    pruner = CaptionPruner(cast("Any", None))
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            await pruner.async_run()
    by_level = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert [m for _, m in by_level].count(
        "Could not delete old camera captions; retrying every hour."
    ) == 1
    assert (logging.INFO, "Deleting old camera captions works again.") in by_level
    # The big first run at INFO, the hourly ones after it at DEBUG.
    assert (logging.INFO, "Deleted 5000 camera captions older than 30 days.") in (
        by_level
    )
    assert (logging.DEBUG, "Deleted 12 camera captions older than 30 days.") in by_level
    assert (logging.DEBUG, "Another entry is deleting old camera captions.") in by_level
