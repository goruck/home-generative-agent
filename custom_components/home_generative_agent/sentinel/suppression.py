"""Suppression and cooldown handling for sentinel findings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from homeassistant.core import HomeAssistant

    from .models import AnomalyFinding

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_suppression"

LOGGER = logging.getLogger(__name__)


@dataclass
class SuppressionState:
    """Persistent suppression state."""

    last_by_type: dict[str, str] = field(default_factory=dict)
    last_by_entity: dict[str, dict[str, str]] = field(default_factory=dict)
    pending_prompts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuppressionState:
        """Create suppression state from persisted data."""
        return cls(
            last_by_type=dict(data.get("last_by_type", {})),
            last_by_entity={
                key: dict(value)
                for key, value in data.get("last_by_entity", {}).items()
            },
            pending_prompts=dict(data.get("pending_prompts", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert suppression state to persisted data."""
        return {
            "last_by_type": dict(self.last_by_type),
            "last_by_entity": {
                key: dict(value) for key, value in self.last_by_entity.items()
            },
            "pending_prompts": dict(self.pending_prompts),
        }


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return dt_util.parse_datetime(value)


def should_suppress(
    state: SuppressionState,
    finding: AnomalyFinding,
    now: datetime,
    cooldown_type: timedelta,
    cooldown_entity: timedelta,
) -> bool:
    """Determine whether a finding should be suppressed."""
    if finding.anomaly_id in state.pending_prompts:
        LOGGER.debug(
            "Suppressing %s (%s): pending prompt exists.",
            finding.anomaly_id,
            finding.type,
        )
        return True

    last_type = _parse_dt(state.last_by_type.get(finding.type))
    if last_type is not None and now - last_type < cooldown_type:
        LOGGER.debug(
            "Suppressing %s (%s): type cooldown active (last=%s).",
            finding.anomaly_id,
            finding.type,
            state.last_by_type.get(finding.type),
        )
        return True

    for entity_id in finding.triggering_entities:
        last_entity = _parse_dt(
            state.last_by_entity.get(entity_id, {}).get(finding.type)
        )
        if last_entity is not None and now - last_entity < cooldown_entity:
            LOGGER.debug(
                "Suppressing %s (%s): entity cooldown active for %s (last=%s).",
                finding.anomaly_id,
                finding.type,
                entity_id,
                state.last_by_entity.get(entity_id, {}).get(finding.type),
            )
            return True

    return False


def register_finding(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> None:
    """Register a finding occurrence for cooldown tracking."""
    iso = dt_util.as_utc(now).isoformat()
    state.last_by_type[finding.type] = iso
    for entity_id in finding.triggering_entities:
        per_entity = state.last_by_entity.setdefault(entity_id, {})
        per_entity[finding.type] = iso


def register_prompt(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> None:
    """Register an active prompt to avoid duplicates."""
    state.pending_prompts[finding.anomaly_id] = dt_util.as_utc(now).isoformat()


def resolve_prompt(state: SuppressionState, anomaly_id: str) -> None:
    """Resolve an active prompt once user responds."""
    state.pending_prompts.pop(anomaly_id, None)


class SuppressionManager:
    """Manage suppression persistence for sentinel findings."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize persistent suppression state management."""
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._state = SuppressionState()

    @property
    def state(self) -> SuppressionState:
        """Return current suppression state."""
        return self._state

    async def async_load(self) -> None:
        """Load suppression state from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, dict):
            self._state = SuppressionState.from_dict(data)

    async def async_save(self) -> None:
        """Persist suppression state to storage."""
        try:
            await self._store.async_save(self._state.as_dict())
        except (HomeAssistantError, OSError, ValueError):
            return
