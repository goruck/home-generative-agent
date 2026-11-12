# ruff: noqa: S101
"""
A Sentinel run that outlives its bounded stop writes and acts no more.

#985 bounded every Sentinel stop, so a run that keeps catching its cancel is
left behind. #988: once the stop stops waiting, that run's saves and actions
are refused (its pushes still go out: its cooldown was saved before them), and
an on-demand run is stopped with the loop. #986: an on-demand run never starts
on a stopped engine.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import custom_components.home_generative_agent as hga_component
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
from custom_components.home_generative_agent.sentinel.engine import (
    SentinelEngine,
    _auto_execute_finding,
)
from custom_components.home_generative_agent.sentinel.network_inventory import (
    NetworkInventory,
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
    """A real engine with its real stores, writes going to mocks."""

    def __init__(self) -> None:
        self.hass = MagicMock()
        self.hass.services.async_call = AsyncMock()
        hass = cast("HomeAssistant", self.hass)
        self.suppression = _with_mock_store(SuppressionManager(hass))
        self.audit = _with_mock_store(AuditStore(hass))
        self.auth = _with_mock_store(AuthInventory(hass))
        self.network = _with_mock_store(NetworkInventory(hass))
        self.notifier = MagicMock()
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


def _finding(actions: list[str]) -> Any:
    return SimpleNamespace(
        anomaly_id="f1",
        suggested_actions=actions,
        triggering_entities=["lock.front_door"],
    )


@pytest.mark.asyncio
async def test_a_run_left_behind_by_stop_writes_and_acts_no_more(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every save and action the left-behind run tries after stop() is refused."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    parts = _Parts()
    results: dict[str, Any] = {}

    async def _effects() -> None:
        await parts.suppression.async_save()
        await parts.audit.async_append_finding(
            cast("Any", {}), MagicMock(), None, suppression_reason_code="x"
        )
        results["auth_saved"] = await parts.auth.async_save()
        results["network_saved"] = await parts.network.async_save()
        try:
            await parts.engine._auto_execute(
                cast("Any", _finding(["lock.lock"])), "exec-1"
            )
        except asyncio.CancelledError:
            results["ended"] = True

    release = asyncio.Event()
    task = asyncio.create_task(_left_behind(release, _effects))
    await asyncio.sleep(0)
    parts.engine._task = task

    with caplog.at_level(logging.WARNING):
        await parts.engine.stop()
        assert not task.done(), "the run should have outlived the bounded stop"
        release.set()
        await task

    assert results["ended"] is True  # no action ran: the run ends there
    assert results["auth_saved"] is False
    assert results["network_saved"] is False
    parts.hass.services.async_call.assert_not_awaited()
    for store in parts.stores():
        store.async_save.assert_not_awaited()
        store.async_delay_save.assert_not_called()
    assert parts.audit._records == []
    # The engine's collaborators share its latch: one stop, one warning.
    warnings = [r for r in caplog.records if "issue #988" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_stop_unsubscribes_the_buttons_when_it_closes_the_stores() -> None:
    """A button press is no longer taken rather than half-saved (#988)."""
    parts = _Parts()

    await parts.engine.stop()

    parts.notifier.stop.assert_called_once()
    parts.notifier.close.assert_not_called()  # pushes are not latched


@pytest.mark.asyncio
async def test_a_failing_notifier_stop_still_closes_everything() -> None:
    """The notifier is stopped last and contained: the latch is never skipped."""
    parts = _Parts()
    parts.notifier.stop.side_effect = RuntimeError("unsubscribed twice")

    await parts.engine.stop()

    assert await parts.auth.async_save() is False
    await parts.suppression.async_save()
    cast("Any", parts.suppression)._store.async_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_actions_are_refused_from_the_start_of_the_stop() -> None:
    """A run still acting while stop() waits acts on options being replaced."""
    parts = _Parts()
    parts.engine._stopped = True  # stop() has begun; its wait is not over

    # Every action refused: the run ends before the finding's cooldown is
    # saved, so the reloaded engine runs the action.
    with pytest.raises(asyncio.CancelledError):
        await parts.engine._auto_execute(cast("Any", _finding(["lock.lock"])), "exec-1")

    parts.hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stop_the_service_turned_into_an_error_runs_no_more_actions() -> None:
    """A cancel a service call converted to an error stops the next action."""
    hass = MagicMock()

    async def _call(*_args: Any, **_kwargs: Any) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()  # the stop's cancel lands during the call...
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(0)  # ...and the integration catches it
        msg = "service failed"
        raise RuntimeError(msg)

    hass.services.async_call = AsyncMock(side_effect=_call)

    outcome = await _auto_execute_finding(
        hass, cast("Any", _finding(["lock.lock", "switch.turn_off"])), "exec-1"
    )

    # The call may have acted: it is recorded as attempted, so the run keeps
    # its cooldown and audit; the next action does not run.
    assert hass.services.async_call.await_count == 1
    assert [a["status"] for a in outcome["actions"]] == ["error", "stopped"]
    assert outcome["status"] == "error"


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
    parts.notifier.stop.assert_called_once()


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
async def test_each_action_checks_the_stop() -> None:
    """A stop during one action's blocking call stops the next one (#988)."""
    hass = MagicMock()
    latch = StopLatch("Sentinel")

    async def _call(*_args: Any, **_kwargs: Any) -> None:
        latch.close()  # the engine stops while this call runs

    hass.services.async_call = AsyncMock(side_effect=_call)

    outcome = await _auto_execute_finding(
        hass,
        cast("Any", _finding(["lock.lock", "switch.turn_off"])),
        "exec-1",
        refuses=latch.refuses,
    )

    assert hass.services.async_call.await_count == 1
    assert [a["status"] for a in outcome["actions"]] == ["ok", "stopped"]
    assert outcome["status"] == "partial"


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


@pytest.mark.asyncio
async def test_stop_ends_an_on_demand_run_too() -> None:
    """An on-demand run is cancelled and waited for like the loop's (#988)."""
    parts = _Parts()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _slow(_source: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    parts.engine._timed_run = _slow  # type: ignore[method-assign]
    run = asyncio.create_task(parts.engine.async_run_now())
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(parts.engine.stop(), timeout=2)

    assert cancelled.is_set()
    assert await asyncio.wait_for(run, timeout=1) is False


@pytest.mark.asyncio
async def test_an_on_demand_error_still_reaches_the_caller() -> None:
    """Running in its own task does not swallow the run's error."""
    parts = _Parts()
    failing = AsyncMock(side_effect=RuntimeError("boom"))
    parts.engine._timed_run = failing  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="boom"):
        await parts.engine.async_run_now()


@pytest.mark.asyncio
async def test_run_now_is_busy_while_a_run_is_still_tracked() -> None:
    """A run whose caller was cancelled may still be going: one at a time."""
    parts = _Parts()
    timed_run = AsyncMock()
    parts.engine._timed_run = timed_run  # type: ignore[method-assign]
    parts.engine._run_now_task = cast("Any", MagicMock())

    assert await parts.engine.async_run_now() is False
    timed_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_cancels_an_on_demand_run_before_waiting_for_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The on-demand run does not act on old options during the loop's wait."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.5)
    parts = _Parts()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _slow(_source: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    parts.engine._timed_run = _slow  # type: ignore[method-assign]
    run = asyncio.create_task(parts.engine.async_run_now())
    await asyncio.wait_for(started.wait(), timeout=1)
    release = asyncio.Event()
    loop = asyncio.create_task(_left_behind(release, AsyncMock()))
    await asyncio.sleep(0)
    parts.engine._task = loop

    stopping = asyncio.create_task(parts.engine.stop())
    await asyncio.wait_for(cancelled.wait(), timeout=0.3)  # before the loop wait

    assert not stopping.done()
    release.set()
    await stopping
    await loop
    assert await run is False


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
    """Records of a cycle that outlived the stop are not saved."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    engine, store, _proposals = _discovery()

    async def _effects() -> None:
        await store.async_append({"candidates": []})

    release = asyncio.Event()
    task = asyncio.create_task(_left_behind(release, _effects))
    await asyncio.sleep(0)
    engine._task = task

    await engine.stop()
    release.set()
    await task

    cast("Any", store)._store.async_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_rule_approval_after_discovery_stops_is_still_saved() -> None:
    """Proposals are the user's: approving one during a reload is kept."""
    engine, _store, proposals = _discovery()
    await proposals.async_append({"candidate_id": "c1", "status": "proposed"})
    cast("Any", proposals)._store.async_save.reset_mock()

    await engine.stop()

    assert await proposals.async_update_status("c1", "approved") is True
    cast("Any", proposals)._store.async_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_runs_now_until_it_stops() -> None:
    """A manual run runs while discovery runs, and not once it has stopped (#986)."""
    engine, _store, _proposals = _discovery()
    cycle = AsyncMock()
    engine._run_once = cycle  # type: ignore[method-assign]

    assert await engine.async_run_now() is True
    cycle.assert_awaited_once()

    await engine.stop()
    cycle.reset_mock()

    assert engine.stopped
    assert await engine.async_run_now() is False
    cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_rechecks_the_stop_under_its_lock() -> None:
    """A stop that lands while the manual run waits for the lock runs nothing."""
    engine, _store, _proposals = _discovery()
    cycle = AsyncMock()
    engine._run_once = cycle  # type: ignore[method-assign]

    await engine._run_lock.acquire()
    run = asyncio.create_task(engine._run_once_guarded())
    await asyncio.sleep(0)
    engine._stopped = True
    engine._run_lock.release()

    assert await run is False
    cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_ends_a_manual_discovery_run_too() -> None:
    """A manual discovery run is cancelled and waited for like the loop's."""
    engine, _store, _proposals = _discovery()
    started = asyncio.Event()

    async def _slow() -> None:
        started.set()
        await asyncio.Event().wait()

    engine._run_once = _slow  # type: ignore[method-assign]
    run = asyncio.create_task(engine.async_run_now())
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(engine.stop(), timeout=2)

    assert await asyncio.wait_for(run, timeout=1) is False


@pytest.mark.asyncio
async def test_a_stopped_discovery_engine_reports_unavailable_not_busy() -> None:
    """The run-discovery service says why nothing ran (#986)."""
    engine, _store, _proposals = _discovery()
    await engine.stop()
    entry = SimpleNamespace(runtime_data=SimpleNamespace(discovery_engine=engine))

    result = await hga_component._trigger_sentinel_discovery(cast("Any", entry))

    assert result == {"status": "unavailable"}


@pytest.mark.asyncio
async def test_a_cancelled_caller_leaves_a_stubborn_manual_run_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery keeps tracking a manual run that caught its cancel."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    engine, _store, _proposals = _discovery()
    release = asyncio.Event()
    started = asyncio.Event()

    async def _stubborn() -> bool:
        started.set()
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await release.wait()
        return True

    engine._run_once_guarded = _stubborn  # type: ignore[method-assign]
    caller = asyncio.create_task(engine.async_run_now())
    await asyncio.wait_for(started.wait(), timeout=1)
    run_task = engine._run_now_task

    caller.cancel()
    with suppress(asyncio.CancelledError):
        await caller

    assert run_task is not None
    assert engine._run_now_task is run_task  # still tracked: busy
    assert await engine.async_run_now() is False
    release.set()
    await run_task
    await asyncio.sleep(0)
    assert engine._run_now_task is None


class _Transaction:
    """Stand in for psycopg's transaction(): it rolls back when its block raises."""

    def __init__(self, *, fail_commit: bool = False) -> None:
        self.outcome: str | None = None
        self._fail_commit = fail_commit

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, *_args: object) -> bool:
        if exc_type is None and self._fail_commit:
            self.outcome = "rolled back"
            msg = "db down"
            raise OSError(msg)
        self.outcome = "rolled back" if exc_type is not None else "committed"
        return False


def _pool(*, fail_commit: bool = False) -> tuple[MagicMock, _Transaction]:
    transaction = _Transaction(fail_commit=fail_commit)
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value={"sample_count": 1})
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.transaction = MagicMock(return_value=transaction)
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, transaction


@pytest.mark.asyncio
async def test_a_baseline_write_after_the_stop_is_rolled_back() -> None:
    """
    A write released after the stop commits nothing (#988).

    The pool autocommits, so only the write's own transaction can undo it.
    """
    pool, transaction = _pool()
    updater = SentinelBaselineUpdater(MagicMock(), pool, {})

    await updater._prune_entity_rows(["sensor.a"])
    assert transaction.outcome == "committed"
    assert updater._router_rows_pruned is True

    updater._router_rows_pruned = False
    await updater.stop()
    await updater._prune_entity_rows(["sensor.a"])

    assert transaction.outcome == "rolled back"
    assert updater._router_rows_pruned is False


@pytest.mark.asyncio
async def test_a_closed_audit_store_keeps_late_changes_out_of_its_final_write() -> None:
    """A record or response after the close is not in the unload's final write."""
    audit = _with_mock_store(AuditStore(cast("HomeAssistant", MagicMock())))
    await audit.async_append_finding(
        cast("Any", {}),
        MagicMock(as_dict=MagicMock(return_value={"anomaly_id": "a1"})),
        None,
        suppression_reason_code="not_suppressed",
    )
    before = [dict(record) for record in audit._records]
    audit.close()

    await audit.async_append_finding(
        cast("Any", {}), MagicMock(), None, suppression_reason_code="x"
    )
    await audit.async_update_response("a1", {"action": "dismiss"}, None)

    assert audit._records == before


@pytest.mark.asyncio
async def test_a_cancelled_caller_leaves_a_stubborn_run_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that caught its cancel stays tracked, so stop() still ends it."""
    monkeypatch.setattr(utils_mod, "CANCEL_WAIT_S", 0.01)
    parts = _Parts()
    release = asyncio.Event()
    started = asyncio.Event()

    async def _stubborn(_source: str) -> None:
        started.set()
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await release.wait()

    parts.engine._timed_run = _stubborn  # type: ignore[method-assign]
    caller = asyncio.create_task(parts.engine.async_run_now())
    await asyncio.wait_for(started.wait(), timeout=1)
    run_task = parts.engine._run_now_task

    caller.cancel()
    with suppress(asyncio.CancelledError):
        await caller

    assert run_task is not None
    assert not run_task.done()
    assert parts.engine._run_now_task is run_task  # still tracked
    release.set()
    await run_task
    await asyncio.sleep(0)
    assert parts.engine._run_now_task is None  # forgotten once it ended


@pytest.mark.asyncio
async def test_two_manual_discovery_runs_at_once_run_one_cycle() -> None:
    """The second of two concurrent triggers is told discovery is busy."""
    engine, _store, _proposals = _discovery()
    cycle = AsyncMock()
    engine._run_once = cycle  # type: ignore[method-assign]

    results = await asyncio.gather(engine.async_run_now(), engine.async_run_now())

    assert sorted(results) == [False, True]
    cycle.assert_awaited_once()


class _CycleStoppedError(Exception):
    """Ends a discovery cycle in a test once it passed the proposal cleanup."""


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True], ids=["running", "stopped"])
async def test_a_stopped_discovery_cycle_cleans_up_no_proposals(
    monkeypatch: pytest.MonkeyPatch,
    stopped: bool,  # noqa: FBT001
) -> None:
    """Proposal cleanup is a discovery write: refused once discovery stopped."""
    engine, _store, proposals = _discovery()
    cleanup = AsyncMock(return_value=0)
    proposals.cleanup_unsupported_ttl = cleanup  # type: ignore[method-assign]
    module = "custom_components.home_generative_agent.sentinel.discovery_engine"
    monkeypatch.setattr(
        f"{module}.async_build_full_state_snapshot", AsyncMock(return_value={})
    )
    monkeypatch.setattr(f"{module}.reduce_snapshot_for_discovery", lambda s: s)
    monkeypatch.setattr(f"{module}.redact_network_identifiers", lambda s: s)
    engine._existing_semantic_context = AsyncMock(  # type: ignore[method-assign]
        side_effect=_CycleStoppedError
    )
    if stopped:
        engine._latch.close()

    with pytest.raises(_CycleStoppedError):
        await engine._run_once()

    assert cleanup.await_count == (0 if stopped else 1)


def _snapshot(entity_id: str, value: str) -> Any:
    return {
        "entities": [
            {
                "entity_id": entity_id,
                "state": value,
                "domain": "sensor",
                "attributes": {"device_class": "temperature"},
            }
        ],
        "derived": {},
    }


@pytest.mark.asyncio
async def test_a_baseline_sample_after_the_stop_is_rolled_back() -> None:
    """The main write rolls back, and memory keeps matching the database."""
    pool, transaction = _pool()
    updater = SentinelBaselineUpdater(
        MagicMock(), pool, {"sentinel_baseline_min_samples": 1}
    )
    await updater._write_samples({"sensor.a": 21.0}, None)
    assert transaction.outcome == "committed"
    assert "sensor.a" in updater._established  # once committed

    await updater.stop()
    dow = dict(updater._dow_state)
    await updater._write_samples({"sensor.b": 21.0}, None)

    assert transaction.outcome == "rolled back"
    assert "sensor.b" not in updater._established
    assert updater._dow_state == dow


@pytest.mark.asyncio
async def test_a_failed_baseline_commit_restores_the_weekly_sums() -> None:
    """A rolled-back write leaves no day-of-week sample in memory (#988 review)."""
    pool, transaction = _pool()
    updater = SentinelBaselineUpdater(
        MagicMock(),
        pool,
        {
            "sentinel_baseline_weekly_patterns": True,
            "sentinel_baseline_min_samples": 1,
        },
    )
    await updater._write_samples({"sensor.a": 20.0}, None)
    assert transaction.outcome == "committed"
    dow = dict(updater._dow_state)
    assert dow, "the write should have advanced a day-of-week slot"

    failing, failed = _pool(fail_commit=True)
    updater._pool = failing
    await updater._write_samples({"sensor.a": 30.0, "sensor.b": 5.0}, None)

    assert failed.outcome == "rolled back"
    assert updater._dow_state == dow
    assert "sensor.b" not in updater._established  # never committed


@pytest.mark.asyncio
async def test_a_counter_prune_after_the_stop_is_rolled_back() -> None:
    """The once-a-day counter prune is refused like the other writes."""
    pool, transaction = _pool()
    updater = SentinelBaselineUpdater(MagicMock(), pool, {})
    await updater.stop()

    await updater._write_offered_counters()

    assert transaction.outcome == "rolled back"


@pytest.mark.asyncio
async def test_a_reset_during_the_commit_is_not_undone() -> None:
    """An entity a reset overtook while the write committed is not established."""
    pool, transaction = _pool()
    updater = SentinelBaselineUpdater(
        MagicMock(), pool, {"sentinel_baseline_min_samples": 1}
    )
    real_exit = _Transaction.__aexit__

    async def _reset_meanwhile(self: _Transaction, *args: Any) -> bool:
        updater._reset_generation += 1  # async_reset_baseline committed
        return await real_exit(self, *args)

    with patch.object(_Transaction, "__aexit__", _reset_meanwhile):
        await updater._write_samples({"sensor.a": 21.0}, None)

    assert transaction.outcome == "committed"
    assert "sensor.a" not in updater._established


@pytest.mark.asyncio
async def test_stopping_the_notifier_sends_its_held_batch() -> None:
    """Held alerts were counted as sent: stop() pushes them, never drops them."""
    from custom_components.home_generative_agent.sentinel.notifier import (  # noqa: PLC0415
        SentinelNotifier,
    )

    notifier = SentinelNotifier(
        cast("HomeAssistant", MagicMock()), {}, MagicMock(), MagicMock()
    )
    notifier._held_batch.append((MagicMock(), None, None))
    flush = MagicMock()
    notifier._async_flush_batch = flush  # type: ignore[method-assign]

    notifier.stop()

    flush.assert_called_once()
    assert notifier._stopped


def test_the_latch_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """A run left behind is worth one warning, not one per refused write."""
    latch = StopLatch("Sentinel")
    assert latch.refuses("save") is False
    latch.close()

    with caplog.at_level(logging.WARNING):
        assert latch.refuses("save") is True
        assert latch.refuses("auto-execute action") is True

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "issue #988" in warnings[0].getMessage()
