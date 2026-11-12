# ruff: noqa: S101
"""
A Sentinel run that outlives its bounded stop writes, alerts and acts no more.

#985 bounded every Sentinel stop, so a run that keeps catching its cancel is
left behind. #988: once the stop stops waiting, everything that run could
change outside itself is refused. #986: an on-demand run never starts on a
stopped engine.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import custom_components.home_generative_agent.core.utils as utils_mod
from custom_components.home_generative_agent.audit.store import AuditStore
from custom_components.home_generative_agent.core.stop_latch import StopLatch
from custom_components.home_generative_agent.sentinel.auth_inventory import (
    AuthInventory,
)
from custom_components.home_generative_agent.sentinel.baseline import (
    SentinelBaselineUpdater,
)
from custom_components.home_generative_agent.sentinel.discovery_engine import (
    SentinelDiscoveryEngine,
)
from custom_components.home_generative_agent.sentinel.discovery_store import (
    DiscoveryStore,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
)
from custom_components.home_generative_agent.sentinel.notifier import (
    SentinelNotifier,
)
from custom_components.home_generative_agent.sentinel.proposal_store import (
    ProposalStore,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _with_mock_store[T](obj: T) -> T:
    """Replace the object's HA Store so writes are recorded, not made."""
    store = MagicMock()
    store.async_save = AsyncMock()
    store.async_remove = AsyncMock()
    store.async_delay_save = MagicMock()
    obj._store = store  # type: ignore[attr-defined]
    return obj


class _Parts:
    """A real engine with its real collaborators, writes going to mocks."""

    def __init__(self) -> None:
        self.hass = MagicMock()
        self.hass.services.async_call = AsyncMock()
        hass = cast("HomeAssistant", self.hass)
        self.suppression = _with_mock_store(SuppressionManager(hass))
        self.audit = _with_mock_store(AuditStore(hass))
        self.auth = _with_mock_store(AuthInventory(hass))
        self.network = _with_mock_store(NetworkInventory(hass))
        self.notifier = SentinelNotifier(hass, {}, self.suppression, MagicMock())
        self.engine = SentinelEngine(
            hass,
            {"sentinel_interval_seconds": 60},
            self.suppression,
            self.notifier,
            self.audit,
            None,
            rule_registry=MagicMock(),
            entry_id="entry",
            auth_inventory=self.auth,
            network_inventory=self.network,
        )

    def stores(self) -> list[MagicMock]:
        return [
            cast("Any", part)._store
            for part in (self.suppression, self.audit, self.auth, self.network)
        ]


async def _left_behind(release: asyncio.Event, then: Any) -> None:
    """Catch every cancel until released, then run *then* (a run left behind)."""
    while not release.is_set():
        with suppress(asyncio.CancelledError):
            await release.wait()
    await then()


@pytest.mark.asyncio
async def test_a_run_left_behind_by_stop_writes_alerts_and_acts_no_more(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Everything the left-behind run tries after stop() returned is refused."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    parts = _Parts()
    results: dict[str, Any] = {}

    async def _effects() -> None:
        await parts.suppression.async_save()
        results["sent"] = await parts.notifier.async_notify(
            MagicMock(), cast("Any", {}), None
        )
        results["outcome"] = await parts.engine._auto_execute(MagicMock(), "exec-1")
        await parts.audit.async_append_finding(
            cast("Any", {}), MagicMock(), None, suppression_reason_code="x"
        )
        results["auth_saved"] = await parts.auth.async_save()
        results["network_saved"] = await parts.network.async_save()

    release = asyncio.Event()
    task = asyncio.create_task(_left_behind(release, _effects))
    await asyncio.sleep(0)
    parts.engine._task = task

    with caplog.at_level(logging.WARNING):
        await parts.engine.stop()
        assert not task.done(), "the run should have outlived the bounded stop"
        release.set()
        await task

    assert results["sent"] is False
    assert results["outcome"]["status"] == "stopped"
    assert results["auth_saved"] is False
    assert results["network_saved"] is False
    parts.hass.services.async_call.assert_not_awaited()
    for store in parts.stores():
        store.async_save.assert_not_awaited()
        store.async_delay_save.assert_not_called()
    assert parts.audit._records == []
    assert "issue #988" in caplog.text


@pytest.mark.asyncio
async def test_a_run_that_ends_within_the_wait_still_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The latch closes after the wait: a run that ends in time keeps its save."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 1.0)
    parts = _Parts()
    started = asyncio.Event()

    async def _run() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await parts.suppression.async_save()

    task = asyncio.create_task(_run())
    await started.wait()
    parts.engine._task = task

    await parts.engine.stop()

    cast("Any", parts.suppression)._store.async_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_engine_never_started_still_closes_on_stop() -> None:
    """A stop with no run loop (a deferred start not yet run) still closes."""
    parts = _Parts()

    await parts.engine.stop()

    assert await parts.auth.async_save() is False
    assert await parts.notifier.async_notify(MagicMock(), cast("Any", {}), None) is (
        False
    )


@pytest.mark.asyncio
async def test_removing_sentinel_still_deletes_the_closed_inventories() -> None:
    """The stop closes the inventories; the reset that follows still removes them."""
    parts = _Parts()
    await parts.engine.stop()

    await parts.auth.async_reset()
    assert await parts.network.async_reset() is True

    cast("Any", parts.auth)._store.async_remove.assert_awaited_once()
    cast("Any", parts.network)._store.async_remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_now_on_a_stopped_engine_runs_nothing() -> None:
    """Approving a rule during a reload or shutdown runs no cycle (#986)."""
    parts = _Parts()
    timed_run = AsyncMock()
    parts.engine._timed_run = timed_run  # type: ignore[method-assign]

    assert await parts.engine.async_run_now() is True
    timed_run.assert_awaited_once_with("on_demand")

    await parts.engine.stop()
    timed_run.reset_mock()

    assert await parts.engine.async_run_now() is False
    timed_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_now_rechecks_the_stop_under_the_lock() -> None:
    """A stop that lands after the first check still stops the run (#986)."""
    parts = _Parts()
    timed_run = AsyncMock()
    parts.engine._timed_run = timed_run  # type: ignore[method-assign]
    scheduler = parts.engine._trigger_scheduler
    real_run_now = scheduler.run_now

    async def _stop_then_run(run_once: Any) -> bool:
        parts.engine._stopped = True  # stop() latches while the call waits
        return await real_run_now(run_once)

    scheduler.run_now = _stop_then_run  # type: ignore[method-assign]

    assert await parts.engine.async_run_now() is False
    timed_run.assert_not_awaited()


def _discovery() -> tuple[SentinelDiscoveryEngine, DiscoveryStore, ProposalStore]:
    hass = cast("HomeAssistant", MagicMock())
    store = _with_mock_store(DiscoveryStore(hass, max_records=10))
    proposals = _with_mock_store(ProposalStore(hass))
    engine = SentinelDiscoveryEngine(
        hass=hass,
        options={},
        model=MagicMock(),
        store=store,
        rule_registry=MagicMock(),
        proposal_store=proposals,
    )
    return engine, store, proposals


@pytest.mark.asyncio
async def test_a_discovery_cycle_left_behind_saves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Records and proposals of a cycle that outlived the stop are not saved."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    engine, store, proposals = _discovery()

    async def _effects() -> None:
        await store.async_append({"candidates": []})
        await proposals.async_append({"candidate_id": "c1"})

    release = asyncio.Event()
    task = asyncio.create_task(_left_behind(release, _effects))
    await asyncio.sleep(0)
    engine._task = task

    await engine.stop()
    release.set()
    await task

    cast("Any", store)._store.async_save.assert_not_awaited()
    cast("Any", proposals)._store.async_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_never_started_still_closes_and_refuses_run_now() -> None:
    """A manual discovery run on a stopped engine runs no cycle (#986)."""
    engine, store, _proposals = _discovery()
    cycle = AsyncMock()
    engine._run_once = cycle  # type: ignore[method-assign]

    await engine.stop()

    assert await engine.async_run_now() is False
    cycle.assert_not_awaited()
    await store.async_save()
    cast("Any", store)._store.async_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_baseline_write_after_the_stop_is_rolled_back() -> None:
    """A database write released after the stop commits nothing (#988)."""
    updater = SentinelBaselineUpdater(MagicMock(), MagicMock(), {})
    conn = MagicMock()
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()

    assert await updater._commit(conn) is True
    conn.commit.assert_awaited_once()

    await updater.stop()
    conn.commit.reset_mock()

    assert await updater._commit(conn) is False
    conn.commit.assert_not_awaited()
    conn.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_closed_audit_store_keeps_late_records_out_of_its_final_write() -> None:
    """A record from a run left behind is not in the unload's final write."""
    audit = _with_mock_store(AuditStore(cast("HomeAssistant", MagicMock())))
    audit.close()

    await audit.async_append_finding(
        cast("Any", {}), MagicMock(), None, suppression_reason_code="x"
    )
    await audit.async_close()

    cast("Any", audit)._store.async_save.assert_awaited_once_with([])


def test_the_latch_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """A run left behind is worth one warning, not one per refused write."""
    latch = StopLatch("Sentinel")
    assert latch.refuses("save") is False
    latch.close()

    with caplog.at_level(logging.WARNING):
        assert latch.refuses("save") is True
        assert latch.refuses("notification") is True

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "issue #988" in warnings[0].getMessage()
