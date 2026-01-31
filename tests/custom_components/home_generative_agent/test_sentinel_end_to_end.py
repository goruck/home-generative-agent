# ruff: noqa: S101
"""End-to-end sentinel test with synthetic snapshot."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)


class DummySuppression(SuppressionManager):
    """Suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class DummyNotifier:
    """Notification dispatcher stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_notify(self, finding, snapshot, explanation):  # type: ignore[no-untyped-def]
        self.calls.append({"finding": finding, "snapshot": snapshot})


class DummyAudit:
    """Audit store stub."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def async_append_finding(self, snapshot, finding, explanation):  # type: ignore[no-untyped-def]
        self.calls.append({"finding": finding, "snapshot": snapshot})


@pytest.mark.asyncio
async def test_sentinel_end_to_end(monkeypatch) -> None:
    """Sentinel processes a snapshot and emits a finding."""
    snapshot = {
        "schema_version": 1,
        "generated_at": "2025-01-01T00:00:00+00:00",
        "entities": [
            {
                "entity_id": "binary_sensor.front_door",
                "domain": "binary_sensor",
                "state": "on",
                "friendly_name": "Front Door",
                "area": "Front",
                "attributes": {"device_class": "door"},
                "last_changed": "2025-01-01T00:00:00+00:00",
                "last_updated": "2025-01-01T00:00:00+00:00",
            }
        ],
        "camera_activity": [],
        "derived": {
            "now": "2025-01-01T00:00:00+00:00",
            "timezone": "UTC",
            "is_night": False,
            "anyone_home": False,
            "last_motion_by_area": {},
        },
    }

    async def _fake_build(_hass):
        return snapshot

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine.async_build_full_state_snapshot",
        _fake_build,
    )

    engine = SentinelEngine(
        hass=object(),
        options={
            "sentinel_cooldown_minutes": 0,
            "sentinel_entity_cooldown_minutes": 0,
            "sentinel_interval_seconds": 60,
            "explain_enabled": False,
        },
        suppression=DummySuppression(),
        notifier=DummyNotifier(),
        audit_store=DummyAudit(),
        explainer=None,
    )

    await engine._run_once()

    assert engine._notifier.calls  # type: ignore[attr-defined]
    assert engine._audit_store.calls  # type: ignore[attr-defined]
