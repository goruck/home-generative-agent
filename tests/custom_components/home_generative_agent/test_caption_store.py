# ruff: noqa: S101
"""Camera captions name their camera; older ones get it in the background (#706)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import psycopg
import pytest

from custom_components.home_generative_agent.core import caption_store
from custom_components.home_generative_agent.core.caption_store import (
    async_backfill_caption_cameras,
    caption_filter,
    caption_namespace,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


class _Result:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    async def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _Conn:
    """Answers each statement the backfill sends, and records them."""

    def __init__(
        self,
        *,
        pending: bool | list[bool] = True,
        locked: bool | list[bool] = True,
        batches: Sequence[dict[str, Any] | Exception] | None = None,
    ) -> None:
        self.pending = pending
        self.locked = locked
        self.batches: list[dict[str, Any] | Exception] = list(batches or [])
        self.statements: list[str] = []
        self.batch_params: list[dict[str, Any]] = []
        self.closed = False

    async def execute(self, query: str, params: Any = None) -> _Result:
        self.statements.append(" ".join(query.split())[:40])
        if query is caption_store._PENDING_SQL:
            return _Result({"pending": 1} if _next(self, "pending") else None)
        if query is caption_store._LOCK_SQL:
            return _Result({"locked": _next(self, "locked")})
        if query is caption_store._BATCH_SQL:
            self.batch_params.append(dict(params))
            answer = self.batches.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return _Result(answer)
        return _Result({"unlocked": True})

    async def close(self) -> None:
        self.closed = True


def _next(conn: _Conn, name: str) -> bool:
    """Answer from a list one call at a time (the last answer repeats)."""
    value = getattr(conn, name)
    if isinstance(value, list):
        return value.pop(0) if len(value) > 1 else value[0]
    return value


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[_Conn]:
        yield self.conn


def _batch(seen: int, updated: int, last: tuple[str, str] | None) -> dict[str, Any]:
    return {
        "seen": seen,
        "updated": updated,
        "last_prefix": last[0] if last else None,
        "last_key": last[1] if last else None,
    }


@pytest.fixture(autouse=True)
def _small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_store, "_BACKFILL_BATCH", 2)
    monkeypatch.setattr(caption_store, "_BACKFILL_RETRY_BACKOFF_SEC", 0)
    monkeypatch.setattr(caption_store, "_LOCK_WAIT_SEC", 0)


async def _run(conn: _Conn) -> None:
    await async_backfill_caption_cameras(cast("Any", _Pool(conn)))


def test_reads_filter_on_the_camera() -> None:
    """The namespace alone matches a prefix; the filter names the camera."""
    assert caption_namespace("side") == ("video_analysis", "side")
    assert caption_filter("side", notified=True) == {
        "camera": "side",
        "notified": True,
    }


@pytest.mark.parametrize("camera", ["side", "side_gate", "front_door"])
def test_the_camera_is_cut_from_the_prefix_at_the_right_place(camera: str) -> None:
    """substr() is 1-based; the pattern escapes the namespace's "_"."""
    prefix = f"video_analysis.{camera}"
    assert prefix[caption_store._CAMERA_START - 1 :] == camera
    assert caption_store._CAPTION_PREFIX_PATTERN == r"video\_analysis.%"


@pytest.mark.asyncio
async def test_a_done_store_costs_one_check() -> None:
    """Nothing lacks the field: one read inside the timeouts, no lock, no batch."""
    conn = _Conn(pending=False)

    await _run(conn)

    assert [s.split()[0] for s in conn.statements] == [
        "SET",
        "SET",
        "SELECT",
        "RESET",
        "RESET",
    ]
    assert conn.batch_params == []
    assert not conn.closed


@pytest.mark.asyncio
async def test_batches_walk_the_keys_until_one_is_short(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each batch starts after the last key of the one before; then unlocks."""
    caplog.set_level("INFO")
    conn = _Conn(
        batches=[
            _batch(2, 2, ("video_analysis.side", "k2")),
            _batch(2, 1, ("video_analysis.side_gate", "k1")),
            _batch(1, 1, ("video_analysis.side_gate", "k9")),
        ]
    )

    await _run(conn)

    assert [(p["after_prefix"], p["after_key"]) for p in conn.batch_params] == [
        ("", ""),
        ("video_analysis.side", "k2"),
        ("video_analysis.side_gate", "k1"),
    ]
    assert conn.batch_params[0]["start"] == len("video_analysis.") + 1
    assert any("pg_advisory_unlock" in s for s in conn.statements)
    assert any(s.startswith("RESET lock_timeout") for s in conn.statements)
    assert "Added the camera to 4 stored camera captions" in caplog.text


@pytest.mark.asyncio
async def test_another_run_holding_the_lock_is_waited_for() -> None:
    """A second entry waits; once that run is done there is nothing left."""
    conn = _Conn(pending=[True, True, False], locked=False)

    await _run(conn)

    assert conn.batch_params == []
    assert not any("pg_advisory_unlock" in s for s in conn.statements)


@pytest.mark.asyncio
async def test_a_run_that_stopped_is_taken_over() -> None:
    """The run holding the lock went away with captions left: this one runs."""
    conn = _Conn(
        locked=[False, True], batches=[_batch(1, 1, ("video_analysis.a", "k"))]
    )

    await _run(conn)

    assert len(conn.batch_params) == 1
    assert any("pg_advisory_unlock" in s for s in conn.statements)


@pytest.mark.asyncio
async def test_a_locked_caption_is_retried_from_the_same_place() -> None:
    """A row lock held by a live write ends the batch, not the run."""
    conn = _Conn(
        batches=[
            psycopg.errors.LockNotAvailable("locked"),
            _batch(1, 1, ("video_analysis.side", "k1")),
        ]
    )

    await _run(conn)

    assert [(p["after_prefix"], p["after_key"]) for p in conn.batch_params] == [
        ("", ""),
        ("", ""),
    ]


@pytest.mark.asyncio
async def test_a_failure_leaves_the_rest_to_the_next_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retries run out: logged, the lock is released, nothing raises."""
    conn = _Conn(batches=[psycopg.errors.QueryCanceled("slow") for _ in range(4)])

    await _run(conn)

    assert len(conn.batch_params) == 4
    assert any("pg_advisory_unlock" in s for s in conn.statements)
    assert "the next start resumes" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_unlock_closes_the_connection() -> None:
    """A pooled connection still holding the lock would block later runs."""

    class _FailingUnlock(_Conn):
        async def execute(self, query: str, params: Any = None) -> _Result:
            if query is caption_store._UNLOCK_SQL:
                msg = "connection lost"
                raise psycopg.OperationalError(msg)
            return await super().execute(query, params)

    conn = _FailingUnlock(batches=[_batch(0, 0, None)])

    await _run(conn)

    assert conn.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["batch", "pause", "unlock", "lock"])
async def test_a_cancel_mid_run_gives_the_connection_back_clean(
    where: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The unload's cancel propagates and the connection is closed, not pooled.

    Wherever it lands (in a batch, the pause, the release, or while the lock
    is being granted), the session may still hold the lock or the timeouts.
    """
    gate = asyncio.Event()

    class _Hang(_Conn):
        async def execute(self, query: str, params: Any = None) -> _Result:
            if (
                (where == "batch" and query is caption_store._BATCH_SQL)
                or (where == "unlock" and query is caption_store._UNLOCK_SQL)
                or (where == "lock" and query is caption_store._LOCK_SQL)
            ):
                gate.set()
                await asyncio.sleep(3600)
            return await super().execute(query, params)

    conn = _Hang(
        batches=[_batch(2, 2, ("video_analysis.a", f"k{i}")) for i in range(3)]
    )
    if where == "pause":
        real_sleep = asyncio.sleep

        async def _slow(_delay: float) -> None:
            gate.set()
            await real_sleep(3600)

        monkeypatch.setattr(caption_store.asyncio, "sleep", _slow)
    if where == "unlock":
        conn.batches = [_batch(0, 0, None)]
    task = asyncio.get_running_loop().create_task(_run(conn))
    await gate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert conn.closed
    if where in ("batch", "pause"):
        assert any("pg_advisory_unlock" in s for s in conn.statements)


@pytest.mark.asyncio
async def test_a_deadlock_with_a_live_write_is_retried() -> None:
    """A caption write and a batch deadlocking: the batch is tried again."""
    conn = _Conn(
        batches=[
            psycopg.errors.DeadlockDetected("deadlock"),
            _batch(1, 1, ("video_analysis.side", "k1")),
        ]
    )

    await _run(conn)

    assert len(conn.batch_params) == 2
    assert not conn.closed
