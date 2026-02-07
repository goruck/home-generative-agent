"""Persistent store for LLM discovery suggestions."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_discovery"


class DiscoveryStore:
    """Persist discovery candidates (advisory only)."""

    def __init__(self, hass: HomeAssistant, max_records: int) -> None:
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._records: list[dict[str, Any]] = []
        self._max_records = max_records

    async def async_load(self) -> None:
        """Load discovery records from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, list):
            self._records = list(data)

    async def async_save(self) -> None:
        """Persist discovery records."""
        try:
            await self._store.async_save(self._records)
        except (HomeAssistantError, OSError, ValueError):
            return

    async def async_append(self, payload: dict[str, Any]) -> None:
        """Append discovery payload and enforce max_records."""
        self._records.append(payload)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]
        await self.async_save()

    async def async_get_latest(self, limit: int) -> list[dict[str, Any]]:
        """Return latest discovery payloads, newest first."""
        if limit <= 0:
            return []
        return list(reversed(self._records[-limit:]))

    def find_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Find a candidate by ID across stored payloads."""
        for payload in reversed(self._records):
            for candidate in payload.get("candidates", []):
                if candidate.get("candidate_id") == candidate_id:
                    return candidate
        return None
