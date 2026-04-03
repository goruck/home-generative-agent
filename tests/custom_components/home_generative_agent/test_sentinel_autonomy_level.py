"""Tests for sentinel_set_autonomy_level service and engine autonomy-level logic."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_AUTONOMY_LEVEL,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
)
from custom_components.home_generative_agent.core.utils import hash_pin
from custom_components.home_generative_agent.sentinel import engine as engine_module
from custom_components.home_generative_agent.sentinel.engine import (
    _AUTONOMY_OVERRIDES,
    SentinelEngine,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionManager,
    SuppressionState,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class DummySuppression(SuppressionManager):
    """Minimal suppression manager stub."""

    def __init__(self) -> None:  # type: ignore[override]
        self._state = SuppressionState()

    @property
    def state(self) -> SuppressionState:  # type: ignore[override]
        return self._state

    async def async_save(self) -> None:  # type: ignore[override]
        return None


class DummyNotifier:
    """Minimal notification dispatcher stub."""

    async def async_notify(self, finding: Any, snapshot: Any, explanation: Any) -> None:
        pass


class DummyAudit:
    """Minimal audit store stub."""

    async def async_append_finding(
        self, snapshot: Any, finding: Any, explanation: Any, **kwargs: Any
    ) -> None:
        pass


def _make_engine(options: dict[str, object] | None = None) -> SentinelEngine:
    """Construct a SentinelEngine with optional options and stub dependencies."""
    return SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options=options or {},
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
        entry_id="test-entry-1",
    )


# ---------------------------------------------------------------------------
# Helper: clear the module-level overrides dict between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_overrides() -> Generator[None]:
    """Ensure _AUTONOMY_OVERRIDES is clean before every test."""
    _AUTONOMY_OVERRIDES.clear()
    yield
    _AUTONOMY_OVERRIDES.clear()


# ---------------------------------------------------------------------------
# Tests: get_autonomy_level falls back to config default
# ---------------------------------------------------------------------------


def test_get_autonomy_level_returns_config_default_when_no_override() -> None:
    """get_autonomy_level returns the config-supplied value when no override exists."""
    engine = _make_engine(options={CONF_SENTINEL_AUTONOMY_LEVEL: 2})
    assert engine.get_autonomy_level("test-entry-1") == 2


def test_get_autonomy_level_returns_recommended_default_when_no_config() -> None:
    """get_autonomy_level returns the recommended default when options is empty."""
    engine = _make_engine(options={})
    assert (
        engine.get_autonomy_level("test-entry-1") == RECOMMENDED_SENTINEL_AUTONOMY_LEVEL
    )


# ---------------------------------------------------------------------------
# Tests: set_autonomy_level persists the override
# ---------------------------------------------------------------------------


def test_set_autonomy_level_stores_override() -> None:
    """set_autonomy_level sets the in-memory override for the given entry."""
    engine = _make_engine()
    engine.set_autonomy_level("test-entry-1", 3)
    assert engine.get_autonomy_level("test-entry-1") == 3


def test_set_autonomy_level_level_decrease_never_requires_pin() -> None:
    """A level decrease must succeed even when require_pin is True."""
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 3,
        }
    )
    # Pre-set the override to level 3 so current level is 3.
    engine.set_autonomy_level("test-entry-1", 3)
    # Decrease to 1 - should NOT raise.
    engine.set_autonomy_level("test-entry-1", 1)
    assert engine.get_autonomy_level("test-entry-1") == 1


def test_set_autonomy_level_increase_blocked_without_pin() -> None:
    """Level increase is blocked when require_pin is True and no PIN is provided."""
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
        }
    )
    with pytest.raises(HomeAssistantError, match="PIN is required"):
        engine.set_autonomy_level("test-entry-1", 2)


def test_set_autonomy_level_increase_allowed_with_pin() -> None:
    """Level increase is allowed when require_pin is True and a PIN is provided."""
    pin_hash, pin_salt = hash_pin("1234")
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: pin_hash,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: pin_salt,
        }
    )
    engine.set_autonomy_level("test-entry-1", 2, pin="1234")
    assert engine.get_autonomy_level("test-entry-1") == 2


def test_set_autonomy_level_increase_rejects_wrong_pin() -> None:
    """Wrong PIN blocks the autonomy level increase."""
    pin_hash, pin_salt = hash_pin("1234")
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: pin_hash,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: pin_salt,
        }
    )
    with pytest.raises(HomeAssistantError, match="Invalid PIN"):
        engine.set_autonomy_level("test-entry-1", 2, pin="9999")


def test_set_autonomy_level_increase_allowed_when_pin_not_required() -> None:
    """Level increase is allowed without a PIN when require_pin is False."""
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
        }
    )
    engine.set_autonomy_level("test-entry-1", 3)
    assert engine.get_autonomy_level("test-entry-1") == 3


# ---------------------------------------------------------------------------
# Tests: TTL expiry reverts to config default
# ---------------------------------------------------------------------------


def test_ttl_expiry_reverts_to_config_default() -> None:
    """When the TTL has expired the override is discarded and the config default is used."""
    engine = _make_engine(
        options={
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 30,
            CONF_SENTINEL_AUTONOMY_LEVEL: 1,
        }
    )
    engine.set_autonomy_level("test-entry-1", 3)
    assert engine.get_autonomy_level("test-entry-1") == 3

    # Simulate the clock advancing beyond the TTL.
    future = dt_util.utcnow() + timedelta(minutes=31)
    with patch.object(engine_module.dt_util, "utcnow", return_value=future):
        level = engine.get_autonomy_level("test-entry-1")

    assert level == 1  # config default
    # Override entry should have been cleaned up.
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES


def test_override_within_ttl_is_not_discarded() -> None:
    """Within the TTL window the override is still returned."""
    engine = _make_engine(
        options={
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 60,
            CONF_SENTINEL_AUTONOMY_LEVEL: 1,
        }
    )
    engine.set_autonomy_level("test-entry-1", 2)

    near_expiry = dt_util.utcnow() + timedelta(minutes=59)
    with patch.object(engine_module.dt_util, "utcnow", return_value=near_expiry):
        level = engine.get_autonomy_level("test-entry-1")

    assert level == 2  # override still active


# ---------------------------------------------------------------------------
# Tests: multiple entry_ids are isolated
# ---------------------------------------------------------------------------


def test_override_is_scoped_to_entry_id() -> None:
    """Overrides for different entry_ids do not interfere with each other."""
    engine_a = _make_engine(options={CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    engine_b = _make_engine(options={CONF_SENTINEL_AUTONOMY_LEVEL: 0})

    engine_a.set_autonomy_level("entry-a", 3)
    # entry-b should still report the config default (0).
    assert engine_b.get_autonomy_level("entry-b") == 0
    assert engine_a.get_autonomy_level("entry-a") == 3
