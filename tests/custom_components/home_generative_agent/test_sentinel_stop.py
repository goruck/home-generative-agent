# ruff: noqa: S101
"""
Sentinel stop paths (#985).

stop() of the engine, the discovery engine and the baseline updater must not
wait without a bound on a task that caught its cancel, and must not swallow a
cancel aimed at its own caller. The baseline wait must not leave a waiter task
behind each interval.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

import custom_components.home_generative_agent.core.utils as utils_mod
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
)
from custom_components.home_generative_agent.sentinel.baseline import (
    SentinelBaselineUpdater,
)
from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine

if TYPE_CHECKING:
    from collections.abc import Callable


def _engine() -> SentinelEngine:
    return SentinelEngine(
        MagicMock(),
        {"sentinel_interval_seconds": 60},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        rule_registry=MagicMock(),
        entry_id="entry",
    )


def _discovery() -> SentinelDiscoveryEngine:
    return SentinelDiscoveryEngine(
        hass=MagicMock(),
        options={},
        model=MagicMock(),
        store=MagicMock(),
        rule_registry=MagicMock(),
        proposal_store=MagicMock(),
    )


def _baseline() -> SentinelBaselineUpdater:
    return SentinelBaselineUpdater(MagicMock(), MagicMock(), {})


_PARTS: list[Callable[[], Any]] = [_engine, _discovery, _baseline]
_IDS = ["engine", "discovery", "baseline"]


async def _stubborn(release: asyncio.Event) -> None:
    """Stand in for a library that catches every cancel until released."""
    while not release.is_set():
        with suppress(asyncio.CancelledError):
            await release.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", _PARTS, ids=_IDS)
async def test_stop_gives_up_on_a_task_that_ignores_every_cancel(
    make: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop() must return even if the task never ends; it logs and moves on."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    part = make()
    release = asyncio.Event()
    task = asyncio.create_task(_stubborn(release))
    await asyncio.sleep(0)
    part._task = task

    stopping = asyncio.create_task(part.stop())
    try:
        done, _ = await asyncio.wait({stopping}, timeout=2)
        assert stopping in done, "stop() kept waiting on a task that ignores cancel"
        assert part._task is None
    finally:
        release.set()
        await task
        await stopping


@pytest.mark.asyncio
@pytest.mark.parametrize("make", _PARTS, ids=_IDS)
async def test_stop_lets_a_cancel_of_its_caller_through(
    make: Callable[[], Any],
) -> None:
    """A cancel of whoever called stop() (shutdown, a reload) must reach it."""
    part = make()
    release = asyncio.Event()
    task = asyncio.create_task(_stubborn(release))
    await asyncio.sleep(0)
    part._task = task

    stopping = asyncio.create_task(part.stop())
    await asyncio.sleep(0)
    stopping.cancel()
    try:
        done, _ = await asyncio.wait({stopping}, timeout=1)
        assert stopping in done, "stop() swallowed its caller's cancel"
        assert stopping.cancelled()
    finally:
        release.set()
        await task
        with suppress(asyncio.CancelledError):
            await stopping


@pytest.mark.asyncio
async def test_stop_ends_a_run_that_caught_the_cancel_and_kept_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that catches the first cancel and goes on must not hold stop()."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    engine = _engine()
    started = asyncio.Event()
    stages: list[str] = []

    async def _run(_source: str = "poll") -> None:
        started.set()
        with suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        stages.append("caught")
        # Keeps working, as a run does after a library ate the cancel; the
        # repeated cancel must end it here.
        await asyncio.Event().wait()
        stages.append("finished")

    engine._timed_run = _run  # type: ignore[method-assign]
    loop_task = asyncio.create_task(engine._run_loop())
    engine._task = loop_task
    await asyncio.wait_for(started.wait(), timeout=1)

    stopping = asyncio.create_task(engine.stop())
    try:
        done, _ = await asyncio.wait({stopping}, timeout=2)
        assert stopping in done, "stop() waited on a run that caught the cancel"
        assert loop_task.done()
        assert stages == ["caught"]
    finally:
        loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await loop_task
        await stopping


def _event_waiters() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done() and t.get_coro().__qualname__ == "Event.wait"  # type: ignore[union-attr]
    ]


@pytest.mark.asyncio
async def test_baseline_wait_leaves_no_waiter_behind_each_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each interval timeout must not leave a pending stop-event waiter task."""
    updater = SentinelBaselineUpdater(
        MagicMock(), MagicMock(), {CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES: 0}
    )
    cycles = 0
    five_cycles = asyncio.Event()

    async def _snapshot(_hass: Any) -> dict[str, Any]:
        return {}

    async def _update(_snapshot: Any) -> None:
        nonlocal cycles
        cycles += 1
        if cycles >= 5:
            five_cycles.set()

    async def _write() -> None:
        return

    monkeypatch.setattr(
        "custom_components.home_generative_agent.snapshot.builder."
        "async_build_full_state_snapshot",
        _snapshot,
    )
    updater._update_baselines = _update  # type: ignore[method-assign]
    updater._write_offered_counters = _write  # type: ignore[method-assign]

    before = len(_event_waiters())
    loop_task = asyncio.create_task(updater._run_loop())
    try:
        await asyncio.wait_for(five_cycles.wait(), timeout=2)
        # One waiter can be live for the wait in progress; none may pile up.
        assert len(_event_waiters()) - before <= 1
    finally:
        updater._stop_event.set()
        await asyncio.wait_for(loop_task, timeout=1)
