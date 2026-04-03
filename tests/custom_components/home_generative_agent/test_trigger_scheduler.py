"""Tests for SentinelTriggerScheduler (issue #256)."""

from __future__ import annotations

import asyncio
import time

import pytest

from custom_components.home_generative_agent.sentinel.trigger_scheduler import (
    COALESCE_WINDOW_SECONDS,
    QUEUE_MAX_SIZE,
    TRIGGER_TTL_SECONDS,
    SentinelTriggerScheduler,
    TriggerRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(anomaly_type: str, age_seconds: float = 0.0) -> TriggerRecord:
    """Return a TriggerRecord with an artificially adjusted enqueued_at."""
    record = TriggerRecord(anomaly_type=anomaly_type)
    # Shift enqueued_at backwards to simulate a record that is `age_seconds` old.
    record.enqueued_at = time.monotonic() - age_seconds
    return record


# ---------------------------------------------------------------------------
# Coalescing tests
# ---------------------------------------------------------------------------


def test_coalescing_same_type_within_window() -> None:
    """Two triggers of the same type within the window produce only one queue entry."""
    scheduler = SentinelTriggerScheduler()
    r1 = TriggerRecord(anomaly_type="open_entry_while_away")
    r2 = TriggerRecord(anomaly_type="open_entry_while_away")

    scheduler.enqueue(r1)
    scheduler.enqueue(r2)

    assert scheduler.queue_depth == 1


def test_coalescing_different_types_not_merged() -> None:
    """Two triggers of different types are both enqueued even within the window."""
    scheduler = SentinelTriggerScheduler()
    r1 = TriggerRecord(anomaly_type="open_entry_while_away")
    r2 = TriggerRecord(anomaly_type="unlocked_lock_at_night")

    scheduler.enqueue(r1)
    scheduler.enqueue(r2)

    assert scheduler.queue_depth == 2


def test_coalescing_same_type_outside_window_not_merged() -> None:
    """A trigger that is outside the coalescing window is enqueued as a new entry."""
    scheduler = SentinelTriggerScheduler()

    # Add a record that is just older than the coalescing window.
    old_record = _make_record(
        "open_entry_while_away", age_seconds=COALESCE_WINDOW_SECONDS + 1.0
    )
    scheduler._queue.append(old_record)

    new_record = TriggerRecord(anomaly_type="open_entry_while_away")
    scheduler.enqueue(new_record)

    assert scheduler.queue_depth == 2


@pytest.mark.asyncio
async def test_coalescing_two_triggers_produce_one_run() -> None:
    """Two coalesced triggers result in _run_once being called only once."""
    scheduler = SentinelTriggerScheduler()
    call_count = 0

    async def fake_run_once() -> None:
        nonlocal call_count
        call_count += 1

    scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))
    scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))  # coalesced

    await scheduler.run_once_if_triggered(fake_run_once)
    # A second call should find the queue empty.
    await scheduler.run_once_if_triggered(fake_run_once)

    assert call_count == 1


# ---------------------------------------------------------------------------
# Queue-full / drop-policy tests
# ---------------------------------------------------------------------------


def test_queue_full_drops_oldest_non_security_critical() -> None:
    """When full, the oldest non-security-critical item is evicted."""
    scheduler = SentinelTriggerScheduler()

    # Fill the queue with non-security-critical items of different ages.
    for i in range(QUEUE_MAX_SIZE):
        # Vary age: item i is `i` seconds old (item 0 is newest, last item is oldest)
        record = _make_record("appliance_power_duration", age_seconds=float(i))
        scheduler._queue.append(record)

    # The queue is now at capacity.  Enqueue a security-critical trigger.
    security_trigger = TriggerRecord(anomaly_type="unlocked_lock_at_night")
    scheduler.enqueue(security_trigger)

    # Queue depth stays at max (one item was evicted and one added).
    assert scheduler.queue_depth == QUEUE_MAX_SIZE

    # The security-critical trigger must be in the queue.
    types_in_queue = [r.anomaly_type for r in scheduler._queue]
    assert "unlocked_lock_at_night" in types_in_queue


def test_queue_full_incoming_non_security_dropped_when_all_critical() -> None:
    """Non-security-critical incoming trigger is dropped when all queued items are critical."""
    scheduler = SentinelTriggerScheduler()

    # Fill the queue entirely with security-critical items.
    for _ in range(QUEUE_MAX_SIZE):
        scheduler._queue.append(TriggerRecord(anomaly_type="unlocked_lock_at_night"))

    # Now try to enqueue a non-critical trigger - it should be dropped.
    scheduler.enqueue(TriggerRecord(anomaly_type="appliance_power_duration"))

    assert scheduler.queue_depth == QUEUE_MAX_SIZE
    types_in_queue = [r.anomaly_type for r in scheduler._queue]
    assert "appliance_power_duration" not in types_in_queue


def test_queue_full_oldest_non_security_is_evicted_not_newest() -> None:
    """Drop policy evicts the *oldest* non-security-critical item, not a newer one."""
    scheduler = SentinelTriggerScheduler()

    # Insert QUEUE_MAX_SIZE - 1 items; the first one (index 0) is the oldest.
    oldest = _make_record("appliance_power_duration", age_seconds=100.0)
    newer = _make_record("appliance_power_duration", age_seconds=1.0)

    scheduler._queue.append(oldest)
    # Fill the rest with the newer record repeated.
    for _ in range(QUEUE_MAX_SIZE - 1):
        scheduler._queue.append(newer)

    # Now add a new trigger that forces a drop.
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))

    assert scheduler.queue_depth == QUEUE_MAX_SIZE
    # oldest should have been removed (only one item with age 100 was inserted).
    ages = [r.enqueued_at for r in scheduler._queue]
    assert oldest.enqueued_at not in ages


# ---------------------------------------------------------------------------
# TTL tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_expired_trigger_is_discarded() -> None:
    """Triggers older than TRIGGER_TTL_SECONDS are discarded without calling _run_once."""
    scheduler = SentinelTriggerScheduler()
    call_count = 0

    async def fake_run_once() -> None:
        nonlocal call_count
        call_count += 1

    # Insert a record that is well past its TTL directly into the queue.
    expired = _make_record(
        "open_entry_while_away", age_seconds=TRIGGER_TTL_SECONDS + 5.0
    )
    scheduler._queue.append(expired)

    result = await scheduler.run_once_if_triggered(fake_run_once)

    assert result is False
    assert call_count == 0


@pytest.mark.asyncio
async def test_ttl_valid_trigger_is_processed() -> None:
    """A trigger within TTL is not discarded."""
    scheduler = SentinelTriggerScheduler()
    call_count = 0

    async def fake_run_once() -> None:
        nonlocal call_count
        call_count += 1

    # Fresh record (age 0).
    scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))

    result = await scheduler.run_once_if_triggered(fake_run_once)

    assert result is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_ttl_expired_records_skipped_until_valid_one_found() -> None:
    """Multiple expired items are skipped and the first valid item is processed."""
    scheduler = SentinelTriggerScheduler()
    call_count = 0

    async def fake_run_once() -> None:
        nonlocal call_count
        call_count += 1

    # Two expired items followed by one valid item.
    for _ in range(2):
        expired = _make_record(
            "appliance_power_duration", age_seconds=TRIGGER_TTL_SECONDS + 10.0
        )
        scheduler._queue.append(expired)

    valid = TriggerRecord(anomaly_type="unlocked_lock_at_night")
    scheduler._queue.append(valid)

    result = await scheduler.run_once_if_triggered(fake_run_once)

    assert result is True
    assert call_count == 1
    # The two expired ones were consumed; only the valid one triggered a run.
    assert scheduler.queue_depth == 0


# ---------------------------------------------------------------------------
# Single-flight lock tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_prevents_concurrent_runs() -> None:
    """A second run_once_if_triggered call returns False while the first is running."""
    scheduler = SentinelTriggerScheduler()
    started = asyncio.Event()
    allow_finish = asyncio.Event()
    concurrent_result: list[bool] = []

    async def slow_run_once() -> None:
        started.set()
        await allow_finish.wait()

    # Enqueue two records so two calls to run_once_if_triggered would normally fire.
    scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))
    scheduler.enqueue(TriggerRecord(anomaly_type="unlocked_lock_at_night"))

    # Start the first triggered run in the background.
    task = asyncio.create_task(scheduler.run_once_if_triggered(slow_run_once))

    # Wait until slow_run_once has started (lock is held).
    await started.wait()

    # Attempt a second triggered run - it should skip because the lock is held.
    result = await scheduler.run_once_if_triggered(slow_run_once)
    concurrent_result.append(result)

    # Unblock the first run.
    allow_finish.set()
    first_result = await task

    assert first_result is True
    assert concurrent_result == [False]


@pytest.mark.asyncio
async def test_lock_prevents_polling_concurrent_with_triggered_run() -> None:
    """run_polling waits for the lock; it does not run concurrently with a triggered run."""
    scheduler = SentinelTriggerScheduler()
    order: list[str] = []

    barrier = asyncio.Event()

    async def triggered_run_once() -> None:
        order.append("triggered_start")
        await barrier.wait()
        order.append("triggered_end")

    async def polling_run_once() -> None:
        order.append("polling")

    # Enqueue a trigger.
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))

    # Start the triggered run (will block at barrier).
    triggered_task = asyncio.create_task(
        scheduler.run_once_if_triggered(triggered_run_once)
    )
    # Give the triggered run a chance to acquire the lock.
    await asyncio.sleep(0)

    # Start the polling run in the background.  It should wait behind the lock.
    polling_task = asyncio.create_task(scheduler.run_polling(polling_run_once))

    # Release the barrier.
    barrier.set()
    await triggered_task
    await polling_task

    # The triggered run must complete entirely before polling starts.
    assert order == ["triggered_start", "triggered_end", "polling"]


# ---------------------------------------------------------------------------
# wait_for_trigger wakeup tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_trigger_wakes_on_enqueue() -> None:
    """wait_for_trigger returns as soon as a trigger is enqueued."""
    scheduler = SentinelTriggerScheduler()

    async def _enqueue_after_delay() -> None:
        await asyncio.sleep(0.05)
        scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))

    task = asyncio.create_task(_enqueue_after_delay())
    # Should complete well within 1 s once the enqueue fires.
    await asyncio.wait_for(scheduler.wait_for_trigger(), timeout=1.0)
    await task
    # Event is cleared after wait_for_trigger returns.
    assert not scheduler._trigger_available.is_set()


@pytest.mark.asyncio
async def test_wait_for_trigger_times_out_when_no_enqueue() -> None:
    """wait_for_trigger raises TimeoutError if no trigger arrives within the deadline."""
    scheduler = SentinelTriggerScheduler()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(scheduler.wait_for_trigger(), timeout=0.05)


@pytest.mark.asyncio
async def test_enqueue_sets_trigger_available_event() -> None:
    """Successful enqueue signals _trigger_available."""
    scheduler = SentinelTriggerScheduler()
    assert not scheduler._trigger_available.is_set()
    scheduler.enqueue(TriggerRecord(anomaly_type="unlocked_lock_at_night"))
    assert scheduler._trigger_available.is_set()


# ---------------------------------------------------------------------------
# Trigger scheduler statistics
# ---------------------------------------------------------------------------


def test_stats_initial_values_are_zero() -> None:
    """All counters start at zero."""
    scheduler = SentinelTriggerScheduler()
    s = scheduler.stats
    assert s["triggers_coalesced"] == 0
    assert s["triggers_dropped_incoming"] == 0
    assert s["triggers_dropped_queued"] == 0
    assert s["triggers_ttl_expired"] == 0


def test_stats_coalesced_increments_on_coalesce() -> None:
    """triggers_coalesced increments when a trigger is merged into a pending one."""
    scheduler = SentinelTriggerScheduler()
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))
    # Second enqueue of same type within coalesce window → coalesced.
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))

    assert scheduler.stats["triggers_coalesced"] == 1
    assert scheduler.queue_depth == 1  # only one in queue


def test_stats_dropped_incoming_increments_when_queue_all_critical() -> None:
    """triggers_dropped_incoming increments when incoming trigger is dropped."""
    scheduler = SentinelTriggerScheduler()
    # Pre-fill queue with security-critical items by injecting directly (bypassing
    # coalescing) so we can use more than one record per type.
    _critical_types = [
        "camera_entry_unsecured",
        "unknown_person_camera_no_home",
        "open_entry_while_away",
        "unlocked_lock_at_night",
    ]
    # All injected records must be older than COALESCE_WINDOW_SECONDS (5 s) so the
    # incoming enqueue below cannot coalesce with any of them.
    for i in range(QUEUE_MAX_SIZE):
        scheduler._queue.append(
            _make_record(_critical_types[i % 4], age_seconds=(i + 1) * 10.0)
        )
    initial_depth = scheduler.queue_depth

    # Enqueueing one more item when queue is full of critical items → incoming dropped.
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))

    assert scheduler.stats["triggers_dropped_incoming"] == 1
    assert scheduler.queue_depth == initial_depth  # nothing evicted


def test_stats_dropped_queued_increments_when_item_evicted() -> None:
    """triggers_dropped_queued increments when a queued item is bumped out."""
    scheduler = SentinelTriggerScheduler()
    # Fill queue with non-critical items aged far apart so coalescing doesn't trigger.
    for i in range(QUEUE_MAX_SIZE):
        scheduler.enqueue(
            _make_record(f"open_entry_while_away_{i}", age_seconds=i * 10.0)
        )
    # Enqueue a security-critical item → oldest non-critical item evicted.
    scheduler.enqueue(TriggerRecord(anomaly_type="camera_entry_unsecured"))

    assert scheduler.stats["triggers_dropped_queued"] == 1
    assert scheduler.queue_depth == QUEUE_MAX_SIZE


def test_stats_ttl_expired_increments_in_pop_next_valid() -> None:
    """triggers_ttl_expired increments when an expired trigger is discarded at dequeue."""
    scheduler = SentinelTriggerScheduler()
    # Directly inject an expired record bypassing the coalescing check.
    expired = _make_record(
        "camera_entry_unsecured", age_seconds=TRIGGER_TTL_SECONDS + 5.0
    )
    scheduler._queue.append(expired)

    result = scheduler._pop_next_valid()

    assert result is None
    assert scheduler.stats["triggers_ttl_expired"] == 1


def test_stats_returns_snapshot_not_reference() -> None:
    """Stats property returns a copy — mutations do not affect internal counters."""
    scheduler = SentinelTriggerScheduler()
    snapshot = scheduler.stats
    snapshot["triggers_coalesced"] = 999

    assert scheduler._stats["triggers_coalesced"] == 0
