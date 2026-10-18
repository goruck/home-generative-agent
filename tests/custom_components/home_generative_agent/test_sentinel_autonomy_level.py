# ruff: noqa: S101
"""Tests for sentinel_set_autonomy_level service and engine autonomy-level logic."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

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

    def set_silence_check(self, check: object) -> None:
        """Accept the engine's level-0 check; unused by this stub."""
        del check

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
    engine_module._OVERRIDE_BASES.clear()
    yield
    _AUTONOMY_OVERRIDES.clear()
    engine_module._OVERRIDE_BASES.clear()


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
    pin_hash, pin_salt = hash_pin("1234")
    engine = _make_engine(
        options={
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: pin_hash,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: pin_salt,
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


def test_an_override_ends_when_the_configured_level_changes() -> None:
    """An override is relative to the level it was set over (#783)."""
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 3})
    engine_module._AUTONOMY_OVERRIDES["test-entry-1"] = (
        3,
        dt_util.utcnow() + timedelta(days=30),
    )
    engine_module._OVERRIDE_BASES["test-entry-1"] = 3
    assert engine.get_autonomy_level("test-entry-1") == 3
    lowered = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    assert lowered.get_autonomy_level("test-entry-1") == 1
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES


# ---------------------------------------------------------------------------
# Override bookkeeping through the real paths (#783 review)
# ---------------------------------------------------------------------------


def test_an_override_from_the_service_ends_when_the_setting_is_lowered() -> None:
    """The level an override was set over is recorded by the service path."""
    _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 3}).set_autonomy_level(
        "test-entry-1", 3
    )
    lowered = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    assert lowered.get_autonomy_level("test-entry-1") == 1
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES


def test_a_mute_from_the_service_ends_when_the_setting_is_raised() -> None:
    _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1}).set_autonomy_level(
        "test-entry-1", 0
    )
    raised = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 2})
    assert raised.get_autonomy_level("test-entry-1") == 2


def test_the_status_shows_an_override_and_when_it_ends() -> None:
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    engine.set_autonomy_level("test-entry-1", 0)
    expires = _AUTONOMY_OVERRIDES["test-entry-1"][1]
    assert engine.autonomy_status() == {
        "autonomy_level": 0,
        "autonomy_override_expires": expires.isoformat(),
    }


def test_an_engine_without_an_entry_has_no_status() -> None:
    engine = SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={},
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
    )
    assert engine.autonomy_status() == {
        "autonomy_level": None,
        "autonomy_override_expires": None,
    }


@pytest.mark.asyncio
async def test_a_reload_clears_the_record_of_a_raised_override() -> None:
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    engine.set_autonomy_level("test-entry-1", 3)
    await engine.stop()
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES
    assert "test-entry-1" not in engine_module._OVERRIDE_BASES


def test_setting_level_zero_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    with caplog.at_level("INFO"):
        engine.set_autonomy_level("test-entry-1", 0)
    assert [r.levelname for r in caplog.records if "level set to 0" in r.message] == [
        "WARNING"
    ]


@pytest.mark.parametrize(
    ("stored", "minutes"),
    [(0, 60), (-5, 60), ("5", 5), (5.0, 5), (99_999_999_999, 43200), ("x", 60)],
)
def test_the_engine_bounds_the_override_duration(stored: object, minutes: int) -> None:
    """The engine reads the duration as the form does (1 minute to 30 days)."""
    engine = _make_engine(
        {
            CONF_SENTINEL_AUTONOMY_LEVEL: 1,
            CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: stored,
        }
    )
    before = dt_util.utcnow()
    engine.set_autonomy_level("test-entry-1", 0)
    expires = _AUTONOMY_OVERRIDES["test-entry-1"][1]
    assert abs((expires - before) - timedelta(minutes=minutes)) < timedelta(seconds=5)


@pytest.mark.parametrize(("stored", "level"), [(0.0, 0), ("2", 2), (2.5, 1), (7, 1)])
def test_the_engine_reads_a_legacy_level_as_the_form_does(
    stored: object, level: int
) -> None:
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: stored})
    assert engine.get_autonomy_level("test-entry-1") == level


def test_a_stopped_engine_leaves_the_new_engines_override_alone() -> None:
    """During a reload the old engine must not drop the new one's override."""
    old = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    old._stopped = True
    new = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 2})
    new.set_autonomy_level("test-entry-1", 0)
    assert old.get_autonomy_level("test-entry-1") == 1
    assert "test-entry-1" in _AUTONOMY_OVERRIDES
    assert new.get_autonomy_level("test-entry-1") == 0


def test_a_level_zero_kept_through_a_reload_is_logged_at_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A service-set 0 survives a reload; the new engine says so (#783)."""
    _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1}).set_autonomy_level(
        "test-entry-1", 0
    )
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    cast("Any", engine)._hass = MagicMock()
    with caplog.at_level("WARNING"):
        engine.start()
    assert any("level 0" in r.getMessage() for r in caplog.records)


def test_no_start_warning_above_level_zero(caplog: pytest.LogCaptureFixture) -> None:
    engine = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    cast("Any", engine)._hass = MagicMock()
    with caplog.at_level("WARNING"):
        engine.start()
    assert not [r for r in caplog.records if "level 0" in r.getMessage()]


def test_the_status_shows_no_expiry_for_an_override_that_no_longer_applies() -> None:
    """A stopped engine keeps a stale override; its expiry is not shown."""
    _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 2}).set_autonomy_level(
        "test-entry-1", 0
    )
    old = _make_engine({CONF_SENTINEL_AUTONOMY_LEVEL: 1})
    old._stopped = True
    assert old.autonomy_status() == {
        "autonomy_level": 1,
        "autonomy_override_expires": None,
    }


def test_the_real_notifier_is_told_when_sentinel_is_silent() -> None:
    """The engine wires its level-0 check into the notifier it is given."""
    from custom_components.home_generative_agent.sentinel.notifier import (  # noqa: PLC0415
        SentinelNotifier as RealNotifier,
    )

    notifier = RealNotifier(
        hass=cast("HomeAssistant", MagicMock()),
        options={},
        suppression=cast("Any", MagicMock()),
        action_handler=cast("Any", MagicMock()),
    )
    SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options={CONF_SENTINEL_AUTONOMY_LEVEL: 0},
        suppression=DummySuppression(),
        notifier=notifier,
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
        entry_id="test-entry-1",
    )
    assert cast("Any", notifier)._silenced() is True
