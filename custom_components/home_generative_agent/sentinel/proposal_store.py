"""Persistent store for rule proposal drafts (from discovery)."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_proposals"

# Unsupported proposals older than this are pruned each discovery cycle.
_UNSUPPORTED_TTL_DAYS = 30

LOGGER = logging.getLogger(__name__)


class ProposalStore:
    """Persist discovery-derived proposal drafts."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize persistent proposal storage."""
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

    async def cleanup_unsupported_ttl(self) -> int:
        """
        Remove unsupported proposals older than _UNSUPPORTED_TTL_DAYS days.

        Returns the number of records removed. Iterates a snapshot of
        self._records so that list mutation is safe. Records with a missing or
        unparseable created_at are left in place rather than silently dropped.
        """
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=_UNSUPPORTED_TTL_DAYS
        )
        keep: list[dict[str, Any]] = []
        removed = 0
        for record in list(self._records):
            if record.get("status") != "unsupported":
                keep.append(record)
                continue
            try:
                created_raw = record.get("created_at")
                if created_raw is None:
                    keep.append(record)
                    continue
                created_at = datetime.datetime.fromisoformat(str(created_raw))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.UTC)
            except (ValueError, TypeError):
                keep.append(record)
                continue
            if created_at < cutoff:
                removed += 1
            else:
                keep.append(record)
        if removed:
            self._records = keep
            LOGGER.debug(
                "ProposalStore TTL: removed %d expired unsupported proposal(s).",
                removed,
            )
            await self.async_save()
        return removed

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
