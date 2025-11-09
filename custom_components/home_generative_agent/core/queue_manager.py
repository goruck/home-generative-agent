"""Queue management utilities for async task coordination."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class QueueManager(Generic[T]):
    """Manages per-camera processing queues with worker tasks."""

    def __init__(
        self,
        hass: HomeAssistant,
        worker_factory: Callable[[str], Coroutine],
        max_size: int = 50,
    ) -> None:
        """Initialize queue manager.

        Args:
            hass: Home Assistant instance
            worker_factory: Function that creates a worker coroutine for a given camera_id
            max_size: Maximum queue size per camera
        """
        self.hass = hass
        self.worker_factory = worker_factory
        self.max_size = max_size
        self._queues: dict[str, asyncio.Queue[T]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def get_or_create(self, camera_id: str) -> asyncio.Queue[T]:
        """Get existing queue or create new one with worker.

        Args:
            camera_id: Camera identifier

        Returns:
            Queue for this camera
        """
        if camera_id not in self._queues:
            queue: asyncio.Queue[T] = asyncio.Queue(maxsize=self.max_size)
            self._queues[camera_id] = queue

            # Start worker task
            task = self.hass.async_create_task(self.worker_factory(camera_id))
            self._tasks[camera_id] = task
            LOGGER.debug("[%s] Created queue and started worker", camera_id)

        return self._queues[camera_id]

    def get_queue(self, camera_id: str) -> asyncio.Queue[T] | None:
        """Get queue for camera if it exists.

        Args:
            camera_id: Camera identifier

        Returns:
            Queue or None if not found
        """
        return self._queues.get(camera_id)

    @staticmethod
    def drain(queue: asyncio.Queue[T]) -> list[T]:
        """Drain all items from queue into a list.

        Args:
            queue: Queue to drain

        Returns:
            List of all queued items
        """
        items: list[T] = []
        try:
            while True:
                items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return items

    @staticmethod
    async def put_with_backpressure(queue: asyncio.Queue[T], item: T) -> None:
        """Put item into queue, dropping oldest if full.

        Args:
            queue: Queue to add to
            item: Item to enqueue
        """
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = queue.get_nowait()
                queue.task_done()
            LOGGER.debug("Queue full; dropped oldest item to enqueue new one")
        await queue.put(item)

    async def stop_all(self, timeout: float = 5.0) -> None:
        """Cancel all worker tasks and clear queues.

        Args:
            timeout: Maximum time to wait for tasks to cancel
        """
        # Cancel all tasks
        for camera_id, task in self._tasks.items():
            LOGGER.debug("[%s] Cancelling worker task", camera_id)
            task.cancel()

        # Wait for cancellation
        if self._tasks:
            _, pending = await asyncio.wait(
                list(self._tasks.values()), timeout=timeout
            )
            for task in pending:
                LOGGER.warning("Task did not cancel in time: %s", task)

        # Clear state
        self._queues.clear()
        self._tasks.clear()
        LOGGER.debug("All queues and tasks cleared")

    def get_active_cameras(self) -> list[str]:
        """Get list of camera IDs with active queues.

        Returns:
            List of camera IDs
        """
        return list(self._queues.keys())
