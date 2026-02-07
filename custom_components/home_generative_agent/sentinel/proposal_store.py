"""Persistent store for rule proposal drafts (from discovery)."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_proposals"


class ProposalStore:
    """Persist discovery-derived proposal drafts."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._records: list[dict[str, Any]] = []

    async def async_load(self) -> None:
        """Load proposal drafts from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, list):
            self._records = list(data)

    async def async_save(self) -> None:
        """Persist proposal drafts."""
        try:
            await self._store.async_save(self._records)
        except (HomeAssistantError, OSError, ValueError):
            return

    async def async_append(self, draft: dict[str, Any]) -> None:
        """Append a proposal draft."""
        self._records.append(draft)
        await self.async_save()

    async def async_get_latest(self, limit: int) -> list[dict[str, Any]]:
        """Return latest proposal drafts, newest first."""
        if limit <= 0:
            return []
        return list(reversed(self._records[-limit:]))

    def find_by_candidate_id(self, candidate_id: str) -> dict[str, Any] | None:
        """Return the most recent record for a candidate id."""
        for record in reversed(self._records):
            if record.get("candidate_id") == candidate_id:
                return record
        return None

    def find_by_rule_id(self, rule_id: str) -> dict[str, Any] | None:
        """Return the most recent record for a rule id."""
        for record in reversed(self._records):
            if record.get("rule_id") == rule_id:
                return record
        return None

    async def async_update_status(
        self,
        candidate_id: str,
        status: str,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Update status for a proposal draft by candidate_id."""
        for record in reversed(self._records):
            if record.get("candidate_id") != candidate_id:
                continue
            record["status"] = status
            if notes is not None:
                record["review_notes"] = notes
            if extra:
                record.update(extra)
            await self.async_save()
            return True
        return False
