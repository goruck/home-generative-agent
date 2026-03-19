"""
Trigger scheduler for the sentinel engine.

Provides ``SentinelTriggerScheduler``, which sits between external event
sources and the sentinel evaluation loop.  Key properties:

* **Bounded queue** - at most 10 pending trigger records.
* **Coalescing window** - a new trigger that arrives within 5 s of an
  already-pending trigger of the same *type* is merged (not enqueued again).
* **TTL** - triggers older than 30 s are silently discarded when the
  scheduler tries to hand one off to the engine.
* **Drop policy** - when the queue is full the *lowest-priority* item is
  dropped to make room; security-critical triggers are considered
  higher-priority and are therefore preferred.
* **Single-flight lock** - an ``asyncio.Lock`` prevents a second concurrent
  ``_run_once()`` call from being issued while a previous one is still
  running.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tuneable constants
# --------------------------------------------------------------------------- #

QUEUE_MAX_SIZE: int = 10
COALESCE_WINDOW_SECONDS: float = 5.0
TRIGGER_TTL_SECONDS: float = 30.0

# --------------------------------------------------------------------------- #
# Priority helpers
# --------------------------------------------------------------------------- #

# Anomaly types that are considered security-critical (high-priority).
_SECURITY_CRITICAL_TYPES: frozenset[str] = frozenset(
    {
        "unknown_person_camera_no_home",
        "camera_entry_unsecured",
        "open_entry_while_away",
        "unlocked_lock_at_night",
    }
)


def _is_security_critical(anomaly_type: str) -> bool:
    """Return True when *anomaly_type* is classified as security-critical."""
    return anomaly_type in _SECURITY_CRITICAL_TYPES


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class TriggerRecord:
    """
    A single scheduler entry.

    Attributes:
        anomaly_type: The anomaly type string from the trigger source.
            Used for coalescing and priority decisions.
        enqueued_at: Monotonic clock timestamp (``time.monotonic()``) of
            when the record was created.  Used for TTL checks.
        is_security_critical: Cached flag so priority checks do not
            re-compute the set membership every time.

    """

    anomaly_type: str
    enqueued_at: float = field(default_factory=time.monotonic)
    is_security_critical: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Populate computed fields after dataclass initialization."""
        self.is_security_critical = _is_security_critical(self.anomaly_type)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def is_expired(self, now: float | None = None) -> bool:
        """Return True when this record has exceeded its TTL."""
        if now is None:
            now = time.monotonic()
        return (now - self.enqueued_at) > TRIGGER_TTL_SECONDS

    def age(self, now: float | None = None) -> float:
        """Return the age of this record in seconds."""
        if now is None:
            now = time.monotonic()
        return now - self.enqueued_at


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


class SentinelTriggerScheduler:
    """
    Scheduler that feeds triggers into the sentinel evaluation loop.

    Usage pattern inside the engine::

        scheduler = SentinelTriggerScheduler()

        # From an event listener / state-change callback:
        scheduler.enqueue(TriggerRecord(anomaly_type="open_entry_while_away"))

        # Inside the run-loop, the engine calls:
        await scheduler.run_once_if_triggered(engine._run_once)

    The ``run_once_if_triggered`` method drains expired items, pops the next
    valid trigger, and executes the provided coroutine under a single-flight
    lock so that at most one evaluation runs at a time.  If no scheduler-
    driven trigger is ready the method returns ``False`` and the caller
    should fall back to its normal polling schedule.
    """

    def __init__(self) -> None:
        """Initialize the scheduler with an empty queue and a new lock."""
        # The pending queue is a plain Python list used as an ordered deque.
        # Using a list (rather than asyncio.Queue) gives us random-access for
        # the coalescing look-ahead and the priority-drop policy.
        self._queue: list[TriggerRecord] = []
        self._lock = asyncio.Lock()
        # Signalled whenever a trigger is successfully enqueued so the run
        # loop can wake up immediately rather than waiting for the next poll.
        self._trigger_available = asyncio.Event()
        # Cumulative trigger statistics; read by SentinelEngine via .stats.
        self._stats: dict[str, int] = {
            "triggers_coalesced": 0,
            "triggers_dropped_incoming": 0,
            "triggers_dropped_queued": 0,
            "triggers_ttl_expired": 0,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enqueue(self, record: TriggerRecord) -> None:
        """
        Add *record* to the pending queue.

        The method applies three policies before inserting:

        1. **Coalescing** - if a pending record with the same
           ``anomaly_type`` was enqueued less than
           ``COALESCE_WINDOW_SECONDS`` ago the new record is silently
           dropped (the existing one already represents this event).

        2. **Drop policy** - if the queue is already at
           ``QUEUE_MAX_SIZE`` the *lowest-priority* item is removed to
           make room.  Security-critical records are higher-priority than
           non-security ones.  Within the same priority tier the *oldest*
           record is dropped (FIFO eviction).

        3. The record is appended.
        """
        now = time.monotonic()

        # --- 1. Coalescing window check ---
        for pending in self._queue:
            if (
                pending.anomaly_type == record.anomaly_type
                and pending.age(now) < COALESCE_WINDOW_SECONDS
            ):
                LOGGER.debug(
                    "Trigger coalesced: type=%s age=%.1fs",
                    record.anomaly_type,
                    pending.age(now),
                )
                self._stats["triggers_coalesced"] += 1
                return

        # --- 2. Drop policy when full ---
        if len(self._queue) >= QUEUE_MAX_SIZE:
            drop_idx = self._select_drop_index()
            if drop_idx is None:
                # All items are security-critical; the incoming record loses.
                LOGGER.warning(
                    "Trigger queue full (all security-critical); dropping incoming "
                    "trigger type=%s.",
                    record.anomaly_type,
                )
                self._stats["triggers_dropped_incoming"] += 1
                return
            dropped = self._queue.pop(drop_idx)
            LOGGER.warning(
                "Trigger queue full; dropped lowest-priority item type=%s "
                "(security_critical=%s) to make room for type=%s.",
                dropped.anomaly_type,
                dropped.is_security_critical,
                record.anomaly_type,
            )
            self._stats["triggers_dropped_queued"] += 1

        # --- 3. Enqueue ---
        self._queue.append(record)
        self._trigger_available.set()
        LOGGER.debug(
            "Trigger enqueued: type=%s queue_depth=%d",
            record.anomaly_type,
            len(self._queue),
        )

    async def run_once_if_triggered(
        self,
        run_once: Callable[[], Coroutine[Any, Any, None]],
    ) -> bool:
        """
        Execute *run_once* for the next valid trigger, if any.

        Expired triggers are discarded.  The provided coroutine is
        protected by a single-flight ``asyncio.Lock`` - if the lock is
        already held by a concurrent call this method returns immediately
        with ``False`` rather than waiting.

        Returns:
            ``True``  - a trigger was consumed and *run_once* was called.
            ``False`` - no valid trigger was available (queue empty or all
                        expired) *or* a concurrent run was already in progress.



        """
        record = self._pop_next_valid()
        if record is None:
            return False

        if self._lock.locked():
            LOGGER.debug(
                "Single-flight lock held; skipping trigger-driven run for type=%s.",
                record.anomaly_type,
            )
            # Re-queue so the trigger is not lost permanently.
            self._queue.insert(0, record)
            return False

        async with self._lock:
            LOGGER.debug(
                "Executing trigger-driven _run_once for type=%s.",
                record.anomaly_type,
            )
            await run_once()

        return True

    async def run_now(
        self,
        run_once: Callable[[], Coroutine[Any, Any, None]],
    ) -> bool:
        """Execute *run_once* immediately under the single-flight lock."""
        if self._lock.locked():
            LOGGER.debug("Single-flight lock held; skipping immediate run request.")
            return False
        async with self._lock:
            LOGGER.debug("Executing immediate _run_once request.")
            await run_once()
        return True

    async def wait_for_trigger(self) -> None:
        """
        Wait until a trigger is enqueued, then clear the signal.

        Intended to be used with ``asyncio.wait_for`` so the run loop wakes
        up immediately when a new trigger arrives rather than sleeping for the
        full polling interval.
        """
        await self._trigger_available.wait()
        self._trigger_available.clear()

    async def run_polling(
        self,
        run_once: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Execute *run_once* under the single-flight lock (polling path).

        Used by the engine's polling timer when no scheduler-driven trigger
        has fired.  Acquiring the same lock as ``run_once_if_triggered``
        ensures that a trigger-driven run and a polling run can never
        execute concurrently.
        """
        async with self._lock:
            LOGGER.debug("Executing polling-driven _run_once.")
            await run_once()

    @property
    def queue_depth(self) -> int:
        """Return the number of items currently waiting in the queue."""
        return len(self._queue)

    @property
    def stats(self) -> dict[str, int]:
        """Return a snapshot of cumulative trigger scheduler statistics."""
        return dict(self._stats)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _pop_next_valid(self) -> TriggerRecord | None:
        """
        Remove and return the first non-expired record.

        Expired items encountered before a valid one are discarded with a
        debug log entry.
        """
        now = time.monotonic()
        while self._queue:
            candidate = self._queue.pop(0)
            if candidate.is_expired(now):
                LOGGER.debug(
                    "Trigger TTL expired; discarding type=%s age=%.1fs.",
                    candidate.anomaly_type,
                    candidate.age(now),
                )
                self._stats["triggers_ttl_expired"] += 1
                continue
            return candidate
        return None

    def _select_drop_index(self) -> int | None:
        """
        Return the index of the lowest-priority item to evict.

        Priority rules (lowest first = evicted first):

        1. Non-security-critical items are lower priority than
           security-critical ones.
        2. Within the same tier, the *oldest* (smallest ``enqueued_at``)
           item is evicted.

        Returns ``None`` when *every* item is security-critical (caller
        should then decide whether to drop the incoming record instead).
        """
        # Prefer to drop the oldest non-security-critical item.
        best_idx: int | None = None
        best_age: float = -1.0
        now = time.monotonic()

        for idx, item in enumerate(self._queue):
            if item.is_security_critical:
                continue
            age = item.age(now)
            if age > best_age:
                best_age = age
                best_idx = idx

        return best_idx
