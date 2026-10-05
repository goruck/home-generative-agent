# ruff: noqa: S101
"""
Regression tests for the config-entry lifecycle P2 batch.

Three leaks in one area, all follow-ups from the deferred-start ship:

* ``image.py`` and ``sensor.py`` registered bare ``EVENT_HOMEASSISTANT_STARTED``
  listeners that were never wired into ``entry.async_on_unload``, so a deferred
  ``async_add_entities`` could run for an entry that had already unloaded.
* ``async_unload_entry`` was a run of bare awaits: one raising step put the
  entry in ``FAILED_UNLOAD`` — which skips the on-unload callbacks — leaving
  every deferred-start listener armed and every engine below the raising line
  un-stopped and un-latched.
* the seventeen ``hass.services.async_register`` calls were never removed, so
  service handlers closing over a dead generation's ``runtime_data`` stayed a
  live API surface after unload.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState
from langgraph.store.base.batch import AsyncBatchedBaseStore
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent import image as image_platform
from custom_components.home_generative_agent import sensor as sensor_platform
from custom_components.home_generative_agent.const import DOMAIN

from .test_deferred_start_listener import (
    REMOVE_LISTENER_ERROR,
    _setup_with_deferred_sentinel_start,
)

# ---------------------------------------------------------------------------
# image.py / sensor.py deferred entity adds
# ---------------------------------------------------------------------------


def _platform_entry(hass: Any) -> MockConfigEntry:
    """Build an added entry whose runtime_data satisfies the sensor platform."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    entry.runtime_data = MagicMock()
    return entry


@pytest.mark.asyncio
async def test_image_platform_adds_cameras_discovered_at_startup(hass: Any) -> None:
    """
    Positive control: the deferred add must still work for a loaded entry.

    The cancellation tests below assert only that nothing was added, which
    would pass just as happily if the deferred add were never registered at
    all. This proves the wiring fires for the normal case.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await image_platform.async_setup_entry(hass, entry, added.extend)
    assert added == [], "no cameras exist yet, nothing should be added"

    entry.mock_state(hass, ConfigEntryState.LOADED)
    hass.states.async_set("camera.front", "idle")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert len(added) == 1, "startup-discovered camera was not added"


@pytest.mark.asyncio
async def test_image_platform_deferred_add_is_cancelled_on_unload(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Unloading before HA finishes starting must cancel the deferred add.

    The entry is put back into LOADED before the event fires, because that is
    what a real reload does — HA reuses the SAME ConfigEntry object, so a
    leaked listener's closure would see the reloaded entry pass the state
    guard and duplicate-add entities. Without this the guard alone keeps the
    assertion green and the cancel is untested.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await image_platform.async_setup_entry(hass, entry, added.extend)

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

        entry.mock_state(hass, ConfigEntryState.LOADED)
        hass.states.async_set("camera.front", "idle")
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
        await hass.async_block_till_done()

    assert added == [], "orphaned listener added entities after unload"
    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_image_platform_refuses_the_add_while_the_entry_is_unloading(
    hass: Any,
) -> None:
    """
    The residual window: STARTED landing mid-unload must not add entities.

    The cancel runs only after ``async_unload_entry`` returns, so the listener
    stays armed for that whole function. The engines close this window with a
    ``_stopped`` latch; the platform equivalent is the entry-state guard.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await image_platform.async_setup_entry(hass, entry, added.extend)

    entry.mock_state(hass, ConfigEntryState.UNLOAD_IN_PROGRESS)
    hass.states.async_set("camera.front", "idle")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert added == [], "entities were added to a platform that is unloading"


@pytest.mark.asyncio
async def test_sensor_platform_deferred_add_is_cancelled_on_unload(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Same leak, sensor platform: the camera-sensor add must cancel on unload.

    LOADED is restored before the event fires for the same reason as the
    image test above — a real reload reuses the same ConfigEntry object, so
    only the cancel (not the state guard) protects this path.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await sensor_platform.async_setup_entry(hass, entry, added.extend)
    # The health and tool-index sensors are added immediately; only the
    # per-camera sensors defer.
    immediate = len(added)

    with caplog.at_level(logging.ERROR):
        await entry._async_process_on_unload(hass)

        entry.mock_state(hass, ConfigEntryState.LOADED)
        hass.states.async_set("camera.front", "idle")
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
        await hass.async_block_till_done()

    assert len(added) == immediate, "orphaned listener added sensors after unload"
    assert REMOVE_LISTENER_ERROR not in caplog.text


@pytest.mark.asyncio
async def test_sensor_platform_adds_cameras_discovered_at_startup(hass: Any) -> None:
    """Positive control for the sensor platform's deferred camera add."""
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await sensor_platform.async_setup_entry(hass, entry, added.extend)
    immediate = len(added)

    entry.mock_state(hass, ConfigEntryState.LOADED)
    hass.states.async_set("camera.front", "idle")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert len(added) == immediate + 1, "startup-discovered camera sensor not added"


@pytest.mark.asyncio
async def test_sensor_platform_refuses_the_add_while_the_entry_is_unloading(
    hass: Any,
) -> None:
    """
    Same residual window as image.py, sensor platform.

    The guard is duplicated in each platform rather than shared, so the image
    test cannot stand in for this one: a regression in sensor.py's copy would
    strand camera sensors on an unloading entry with the image suite green.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await sensor_platform.async_setup_entry(hass, entry, added.extend)
    immediate = len(added)

    entry.mock_state(hass, ConfigEntryState.UNLOAD_IN_PROGRESS)
    hass.states.async_set("camera.front", "idle")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert len(added) == immediate, "sensors were added to a platform mid-unload"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform", [image_platform, sensor_platform], ids=["image", "sensor"]
)
async def test_platform_allows_the_add_while_setup_is_still_in_progress(
    hass: Any, platform: Any
) -> None:
    """
    SETUP_IN_PROGRESS must pass the entry-state guard.

    A platform set up while HA is still starting defers into exactly that
    state, so tightening the guard to LOADED-only would silently drop every
    startup-discovered camera. The LOADED positive controls cannot catch that
    regression; this pins the second allowed state.
    """
    hass.set_state(CoreState.not_running)
    entry = _platform_entry(hass)
    added: list[Any] = []

    await platform.async_setup_entry(hass, entry, added.extend)
    immediate = len(added)

    entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
    hass.states.async_set("camera.front", "idle")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await hass.async_block_till_done()

    assert len(added) == immediate + 1, (
        "startup-discovered camera was refused while setup was in progress"
    )


# ---------------------------------------------------------------------------
# async_unload_entry failure containment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unload_closes_the_audit_store(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Audit saves are delayed, so unload must write the store and close it.

    Otherwise a reload loads the file before the old store's pending write
    lands, and a late write from the old store overwrites the new one's.
    """
    entry, _sentinel, _discovery, _client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    class _RecordingAudit:
        def __init__(self) -> None:
            self.close_calls = 0

        async def async_close(self) -> None:
            self.close_calls += 1

    audit = _RecordingAudit()
    entry.runtime_data.audit_store = audit

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is True
    assert audit.close_calls == 1


@pytest.mark.asyncio
async def test_a_raising_teardown_step_does_not_abort_the_unload(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    One raising step must not skip the remaining stops or fail the unload.

    An exception out of ``async_unload_entry`` puts the entry in
    ``FAILED_UNLOAD`` — non-recoverable — and Home Assistant returns WITHOUT
    running the on-unload callbacks, so the deferred-start listeners stay
    armed and every engine below the raising line is left un-stopped: exactly
    the orphan-start the teardown exists to prevent.
    """
    entry, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    async def _boom() -> None:
        msg = "video analyzer teardown exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(entry.runtime_data.video_analyzer, "stop", _boom)

    # A SYNC step that raises before any awaitable exists must be contained
    # by the same wrapper (the raise happens at `run()`, not at the await),
    # and steps after it must still run. The harness leaves notifier and pool
    # as None, so stubs are attached to exercise those legs too.
    class _SyncBoomNotifier:
        def stop(self) -> None:
            msg = "notifier teardown exploded synchronously"
            raise RuntimeError(msg)

    class _RecordingPool:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    pool = _RecordingPool()
    entry.runtime_data.notifier = _SyncBoomNotifier()
    entry.runtime_data.pool = pool

    with caplog.at_level(logging.ERROR):
        result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is True, "a contained teardown failure must not fail the unload"
    assert sentinel.stop_calls == 1, "steps after the raising one were skipped"
    assert discovery.stop_calls == 1, "steps after the raising one were skipped"
    assert pool.close_calls == 1, "steps after the sync-raising one were skipped"
    assert "video_analyzer.stop" in caplog.text, "the async failure was not logged"
    assert "notifier.stop" in caplog.text, "the sync failure was not logged"

    # And because the unload succeeded, the on-unload callbacks still run —
    # closing the client the FAILED_UNLOAD path would have leaked.
    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_unload_cancels_the_tool_index_write_before_the_pool_closes(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The index write is cancelled and awaited before the pool closes.

    A reload used to close the pool under the background index write, which
    died with PoolClosed and was logged as a failed index on every options
    change that raced it.
    """
    entry, _sentinel, _discovery, _client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )
    order: list[str] = []
    started = asyncio.Event()

    async def _index_write() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("index cancelled")
            raise

    class _RecordingPool:
        async def close(self) -> None:
            order.append("pool closed")

    async def _batch_worker() -> None:
        # langgraph's store runs each put batch in a worker of its own.
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("store worker cancelled")
            raise

    rd = entry.runtime_data
    rd.pool = _RecordingPool()
    rd.tool_index_task = hass.async_create_task(_index_write())
    rd.store = MagicMock(_task=hass.async_create_task(_batch_worker()))
    await started.wait()

    with caplog.at_level(logging.ERROR):
        assert await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert order == ["index cancelled", "store worker cancelled", "pool closed"]
    assert rd.tool_index_task is None
    assert rd.store._task.done()
    assert "tool_index" not in caplog.text


@pytest.mark.asyncio
async def test_unload_with_no_index_write_in_flight_is_a_noop(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    entry, _sentinel, _discovery, _client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    async def _finished() -> None:
        return None

    rd = entry.runtime_data
    rd.tool_index_task = hass.async_create_task(_finished())
    await hass.async_block_till_done()
    assert rd.tool_index_task.done()
    rd.store = MagicMock(_task=None)

    with caplog.at_level(logging.ERROR):
        assert await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert rd.tool_index_task is None
    assert "tool_index" not in caplog.text


@pytest.mark.asyncio
async def test_a_refused_platform_unload_still_aborts_before_teardown(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The containment must not swallow the platform-unload abort at the top."""
    entry, sentinel, discovery, client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )

    async def _refuse(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(hass.config_entries, "async_unload_platforms", _refuse)

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is False
    assert sentinel.stop_calls == 0
    assert discovery.stop_calls == 0
    assert client.close_calls == 0


# ---------------------------------------------------------------------------
# service deregistration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_services_are_removed_when_the_entry_unloads(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Unload must remove the domain services registered by setup.

    Service handlers close over the entry's ``runtime_data``; after unload
    they were still callable and reached closed pools and stopped engines
    (e.g. ``hga.sentinel_get_baselines``). The positive half asserts the
    services exist first, so deleting the registrations outright cannot leave
    this green.
    """
    entry, *_ = await _setup_with_deferred_sentinel_start(hass, monkeypatch)

    registered = set(hass.services.async_services().get(DOMAIN, {}))
    assert "enroll_person" in registered
    assert "get_audit_records" in registered
    assert "sentinel_get_baselines" in registered

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)
    assert result is True
    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()

    assert not hass.services.async_services().get(DOMAIN, {}), (
        "services survived the unload and still pin the dead generation"
    )


@pytest.mark.asyncio
async def test_register_services_wires_save_and_analyze_for_removal(
    hass: Any,
) -> None:
    """
    The 17th service — the one the setup harness stubs — must deregister too.

    ``save_and_analyze_snapshot`` is registered by ``_register_services``,
    which ``_setup_with_deferred_sentinel_start`` replaces with a no-op, so
    the full-setup removal test above never sees it. Reverting that one call
    site to a bare ``hass.services.async_register`` would leak it across
    unloads with the rest of the suite green.
    """
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)

    cast("Any", hga_component)._register_services(hass, entry)
    assert "save_and_analyze_snapshot" in hass.services.async_services().get(
        DOMAIN, {}
    ), "service was not registered"

    await entry._async_process_on_unload(hass)

    assert "save_and_analyze_snapshot" not in hass.services.async_services().get(
        DOMAIN, {}
    ), "save_and_analyze_snapshot survived the unload"


class _CancelEatingStore(AsyncBatchedBaseStore):
    """A batched store whose in-flight batch turns a cancel into an error."""

    def __init__(self) -> None:
        super().__init__()
        self.in_batch = asyncio.Event()

    def batch(self, ops: Any) -> list[Any]:
        raise NotImplementedError

    async def abatch(self, ops: Any) -> list[Any]:
        self.in_batch.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # What a database driver does when its own cleanup fails while
            # the cancel is passing through it.
            msg = "connection lost during cancel"
            raise RuntimeError(msg) from None
        return [None for _ in ops]


@pytest.mark.asyncio
async def test_cancel_and_wait_ends_a_batch_worker_that_absorbed_the_first_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The unload must not wait forever on langgraph's batch worker.

    The worker hands any ordinary exception from the in-flight batch to its
    callers and returns to its queue, so a cancel converted into one leaves
    it alive. An unbounded wait on it left the entry in UNLOAD_IN_PROGRESS
    and every entity unavailable until a restart.
    """
    monkeypatch.setattr(hga_component, "_CANCEL_WAIT_S", 0.05)
    store = _CancelEatingStore()
    put = asyncio.ensure_future(store.aput(("tools",), "k", {"content": "x"}))
    await store.in_batch.wait()
    worker = store._task
    assert worker is not None

    await asyncio.wait_for(
        cast("Any", hga_component)._cancel_and_wait(worker, "store batch worker"),
        timeout=5,
    )

    assert worker.done(), "the batch worker outlived the unload's cancel"
    with pytest.raises(RuntimeError):
        await put


@pytest.mark.asyncio
async def test_cancel_and_wait_gives_up_on_a_task_that_never_dies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A task that ignores every cancel is logged and left, not waited on."""
    monkeypatch.setattr(hga_component, "_CANCEL_WAIT_S", 0.01)
    release = asyncio.Event()
    cancels = 0

    async def _immortal() -> None:
        nonlocal cancels
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancels += 1

    task = asyncio.ensure_future(_immortal())
    await asyncio.sleep(0)
    attempts = cast("Any", hga_component)._CANCEL_ATTEMPTS

    # try/finally: a failure here must still release the task, or the event
    # loop's teardown waits on it forever and the run hangs instead of failing.
    try:
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(
                cast("Any", hga_component)._cancel_and_wait(task, "test task"),
                timeout=5,
            )

        assert not task.done()
        assert cancels == attempts
        assert (
            f"The test task ({task.get_name()}) did not stop after {attempts}"
            in caplog.text
        )
    finally:
        release.set()
        await task


# Generated by /ship test-coverage audit.
# Value: protects=an unload racing an index write whose store worker absorbs
#   the first cancel still returns True and closes the pool after the worker
#   ends; fails_when=async_unload_entry waits on the store worker without a
#   bound or a repeat cancel (inlined, or _cancel_tool_index stops routing
#   through _cancel_and_wait); why_new=the helper tests call _cancel_and_wait
#   directly and the unload-level test uses a worker that dies on the first
#   cancel; seam=none (no production hook added)
@pytest.mark.asyncio
async def test_unload_completes_when_the_store_worker_absorbs_the_first_cancel(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole unload, not just the helper, survives the stubborn worker."""
    monkeypatch.setattr(hga_component, "_CANCEL_WAIT_S", 0.05)
    entry, _sentinel, _discovery, _client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )
    store = _CancelEatingStore()
    worker_done_at_pool_close: list[bool] = []

    class _RecordingPool:
        async def close(self) -> None:
            worker = store._task
            worker_done_at_pool_close.append(worker is not None and worker.done())

    rd = entry.runtime_data
    rd.pool = _RecordingPool()
    rd.store = store
    rd.tool_index_task = hass.async_create_task(
        store.aput(("tools",), "k", {"content": "x"})
    )
    await store.in_batch.wait()

    result = await asyncio.wait_for(
        cast("Any", hga_component).async_unload_entry(hass, entry), timeout=5
    )

    assert result is True
    assert rd.tool_index_task is None
    assert worker_done_at_pool_close == [True], (
        "the pool closed under a live store worker, or never closed"
    )


# Generated by /ship test-coverage audit.
# Value: protects=a cancellation of the unload itself (shutdown, a cancelled
#   reload) propagates out of _cancel_and_wait while it waits on a stubborn
#   task; fails_when=the bounded wait is rewritten to await the task under a
#   CancelledError suppress or a shield; why_new=no test cancels the waiter,
#   only the waited-on task; seam=none
@pytest.mark.asyncio
async def test_cancel_and_wait_lets_a_cancel_of_the_unload_itself_through() -> None:
    """Cancelling the caller mid-wait must not be swallowed by the helper."""
    release = asyncio.Event()

    async def _immortal() -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    task = asyncio.ensure_future(_immortal())
    await asyncio.sleep(0)
    waiter = asyncio.ensure_future(
        cast("Any", hga_component)._cancel_and_wait(task, "test task")
    )
    try:
        await asyncio.sleep(0.01)
        assert not waiter.done()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=5)
    finally:
        release.set()
        await task


def test_the_shipped_cancel_budget_is_finite_and_short() -> None:
    """The tests above patch the wait; this pins the values that ship."""
    wait_s = cast("Any", hga_component)._CANCEL_WAIT_S
    attempts = cast("Any", hga_component)._CANCEL_ATTEMPTS
    assert isinstance(wait_s, (int, float))
    assert 0 < wait_s * attempts <= 10


@pytest.mark.asyncio
async def test_unload_gives_up_on_a_store_worker_that_never_dies(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The unload still returns, closes the pool and says which task it left."""
    monkeypatch.setattr(hga_component, "_CANCEL_WAIT_S", 0.01)
    entry, _sentinel, _discovery, _client = await _setup_with_deferred_sentinel_start(
        hass, monkeypatch
    )
    release = asyncio.Event()

    async def _immortal() -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    worker = asyncio.ensure_future(_immortal())
    await asyncio.sleep(0)
    worker_done_at_pool_close: list[bool] = []

    class _RecordingPool:
        async def close(self) -> None:
            worker_done_at_pool_close.append(worker.done())

    rd = entry.runtime_data
    rd.pool = _RecordingPool()
    rd.store = SimpleNamespace(_task=worker)
    rd.tool_index_task = None
    try:
        with caplog.at_level(logging.WARNING):
            result = await asyncio.wait_for(
                cast("Any", hga_component).async_unload_entry(hass, entry), timeout=5
            )

        assert result is True
        assert worker_done_at_pool_close == [False]
        assert "The store batch worker (" in caplog.text
    finally:
        release.set()
        await worker
