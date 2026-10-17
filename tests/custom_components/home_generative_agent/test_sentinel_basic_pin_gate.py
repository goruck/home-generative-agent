# ruff: noqa: S101
"""
Regression tests for #762: a level-increase PIN set in Basic setup was ignored.

The Basic form hashed and stored the PIN but left
``sentinel_require_pin_for_level_increase`` at its ``False`` default, and
``SentinelEngine.set_autonomy_level`` gates the whole check on that flag, so
``sentinel_set_autonomy_level`` raised the level with no PIN at all.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_SENTINEL_AUTONOMY_LEVEL,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.core.utils import hash_pin
from custom_components.home_generative_agent.sentinel import engine as engine_module
from custom_components.home_generative_agent.sentinel.engine import (
    _AUTONOMY_OVERRIDES,
    SentinelEngine,
)
from custom_components.home_generative_agent.sentinel.execution import (
    level_increase_pin_required,
)

from .test_sentinel_autonomy_level import DummyAudit, DummyNotifier, DummySuppression
from .test_silent_config_failures import _sentinel_flow_with_existing
from .test_subentries import DummyEntry, DummySubentry, _make_sentinel_flow_for_mode

if TYPE_CHECKING:
    from collections.abc import Generator

    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.sentinel.notifier import (
        SentinelNotifier,
    )

_BASIC_INPUT: dict[str, Any] = {
    CONF_SENTINEL_ENABLED: True,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
    CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
}


@pytest.fixture(autouse=True)
def _clear_overrides() -> Generator[None]:
    _AUTONOMY_OVERRIDES.clear()
    engine_module._OVERRIDE_BASES.clear()
    yield
    _AUTONOMY_OVERRIDES.clear()
    engine_module._OVERRIDE_BASES.clear()


def _engine_for(data: dict[str, Any]) -> SentinelEngine:
    """Build an engine from Sentinel subentry data the way setup resolves it."""
    entry = DummyEntry(options={})
    sentinel = DummySubentry("sentinel1", SUBENTRY_TYPE_SENTINEL, "Sentinel", data)
    entry.subentries[sentinel.subentry_id] = sentinel
    return SentinelEngine(
        hass=cast("HomeAssistant", object()),
        options=resolve_runtime_options(entry),  # type: ignore[arg-type]
        suppression=DummySuppression(),
        notifier=cast("SentinelNotifier", DummyNotifier()),
        audit_store=cast("AuditStore", DummyAudit()),
        explainer=None,
        entry_id="test-entry-1",
    )


@pytest.mark.asyncio
async def test_basic_setup_pin_is_enforced_end_to_end(hass: HomeAssistant) -> None:
    """A PIN typed into Basic setup blocks an increase without it."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(
        {**_BASIC_INPUT, CONF_CRITICAL_ACTION_PIN: "4321"}
    )

    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True

    engine = _engine_for({**data, CONF_SENTINEL_AUTONOMY_LEVEL: 0})
    with pytest.raises(HomeAssistantError, match="PIN is required"):
        engine.set_autonomy_level("test-entry-1", 2)
    with pytest.raises(HomeAssistantError, match="Invalid PIN"):
        engine.set_autonomy_level("test-entry-1", 2, pin="1111")
    engine.set_autonomy_level("test-entry-1", 2, pin="4321")
    assert engine.get_autonomy_level("test-entry-1") == 2


@pytest.mark.asyncio
async def test_basic_setup_without_a_pin_leaves_the_gate_off(
    hass: HomeAssistant,
) -> None:
    """A blank PIN box on a fresh install keeps the recommended default."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(dict(_BASIC_INPUT))

    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is False
    assert CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH not in data

    engine = _engine_for({**data, CONF_SENTINEL_AUTONOMY_LEVEL: 0})
    engine.set_autonomy_level("test-entry-1", 2)
    assert engine.get_autonomy_level("test-entry-1") == 2


def test_engine_enforces_a_pre_fix_basic_pin() -> None:
    """A hash stored with the flag off (pre-#762 Basic setup) is still enforced."""
    pin_hash, pin_salt = hash_pin("4321")
    engine = _engine_for(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_AUTONOMY_LEVEL: 0,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: pin_hash,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: pin_salt,
        }
    )
    with pytest.raises(HomeAssistantError, match="PIN is required"):
        engine.set_autonomy_level("test-entry-1", 2)
    engine.set_autonomy_level("test-entry-1", 2, pin="4321")


@pytest.mark.asyncio
async def test_basic_rerun_keeps_a_pre_fix_basic_pin(hass: HomeAssistant) -> None:
    """Re-running Basic with a blank PIN box must not drop a pre-#762 PIN."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: "stored-hash",
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: "stored-salt",
        },
    )
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(dict(_BASIC_INPUT))

    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] == "stored-hash"
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT] == "stored-salt"


@pytest.mark.asyncio
async def test_advanced_pin_with_the_box_unchecked_is_enforced(
    hass: HomeAssistant,
) -> None:
    """A PIN saved from the Advanced form turns the gate on, as in Basic."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_CRITICAL_ACTION_PIN: "4321",
        }
    )

    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True


@pytest.mark.asyncio
async def test_advanced_form_shows_a_pre_fix_basic_pin_as_required(
    hass: HomeAssistant,
) -> None:
    """Saving the Advanced form unchanged must not disarm a pre-#762 PIN."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: "stored-hash",
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: "stored-salt",
        },
    )
    form = cast("dict[str, Any]", await flow.async_step_settings())
    schema = form["data_schema"].schema
    key: Any = next(
        k for k in schema if k == CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE
    )
    assert key.description["suggested_value"] is True

    # Submit what the form suggested, as a user saving it unchanged would.
    await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: key.description[
                "suggested_value"
            ],
        }
    )
    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] == "stored-hash"
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True


def _gated(level: int = 0, **extra: Any) -> dict[str, Any]:
    """Sentinel data with the gate on through a stored PIN (flag off, pre-#762)."""
    pin_hash, pin_salt = hash_pin("4321")
    return {
        CONF_SENTINEL_ENABLED: True,
        CONF_SENTINEL_AUTONOMY_LEVEL: level,
        CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
        CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: pin_hash,
        CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: pin_salt,
        **extra,
    }


def test_gate_helper_agrees_with_the_engine() -> None:
    """The execution policy hash and the engine use one definition of gate-on."""
    assert level_increase_pin_required({CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: "h"})
    assert level_increase_pin_required(
        {CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True}
    )
    assert not level_increase_pin_required({})


def test_pre_fix_pin_allows_a_decrease_and_rejects_a_wrong_pin() -> None:
    """With only a stored hash, lowering needs no PIN and a wrong PIN fails."""
    engine = _engine_for(_gated(level=2))
    engine.set_autonomy_level("test-entry-1", 1)
    assert engine.get_autonomy_level("test-entry-1") == 1
    with pytest.raises(HomeAssistantError, match="Invalid PIN"):
        engine.set_autonomy_level("test-entry-1", 3, pin="0000")


def test_same_level_call_without_pin_does_not_renew_a_raised_level() -> None:
    """Re-requesting the raised level without the PIN keeps the original expiry."""
    engine = _engine_for(_gated(**{CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 60}))
    base_level = engine.get_autonomy_level("test-entry-1")
    engine.set_autonomy_level("test-entry-1", 3, pin="4321")
    _, first_expiry = _AUTONOMY_OVERRIDES["test-entry-1"]

    later = dt_util.utcnow() + timedelta(minutes=59)
    with patch.object(engine_module.dt_util, "utcnow", return_value=later):
        engine.set_autonomy_level("test-entry-1", 3)
        engine.set_autonomy_level("test-entry-1", 2)
    assert _AUTONOMY_OVERRIDES["test-entry-1"][1] == first_expiry

    expired = first_expiry + timedelta(seconds=1)
    with patch.object(engine_module.dt_util, "utcnow", return_value=expired):
        assert engine.get_autonomy_level("test-entry-1") == base_level


def test_same_level_call_with_the_pin_renews_a_raised_level() -> None:
    """The PIN still buys a fresh TTL; a wrong one is refused."""
    engine = _engine_for(_gated(**{CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES: 60}))
    engine.set_autonomy_level("test-entry-1", 3, pin="4321")
    _, first_expiry = _AUTONOMY_OVERRIDES["test-entry-1"]

    later = dt_util.utcnow() + timedelta(minutes=59)
    with patch.object(engine_module.dt_util, "utcnow", return_value=later):
        with pytest.raises(HomeAssistantError, match="Invalid PIN"):
            engine.set_autonomy_level("test-entry-1", 3, pin="0000")
        engine.set_autonomy_level("test-entry-1", 3, pin="4321")
    assert _AUTONOMY_OVERRIDES["test-entry-1"][1] > first_expiry


@pytest.mark.asyncio
async def test_engine_stop_drops_a_raised_level() -> None:
    """A reload starts at the configured level, so arming the PIN ends a raise."""
    engine = _engine_for(_gated())
    engine.set_autonomy_level("test-entry-1", 3, pin="4321")
    await engine.stop()
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES


@pytest.mark.parametrize("stored_flag", [True, False])
@pytest.mark.asyncio
async def test_advanced_uncheck_and_blank_pin_removes_the_pin(
    hass: HomeAssistant,
    stored_flag: bool,  # noqa: FBT001
) -> None:
    """The documented way to turn the gate off still works."""
    data_in = _gated()
    data_in[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] = stored_flag
    flow, update_calls = _sentinel_flow_with_existing(hass, data_in)
    await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
        }
    )

    data = update_calls[0]["data"]
    assert CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH not in data
    assert CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT not in data
    engine = _engine_for({**data, CONF_SENTINEL_AUTONOMY_LEVEL: 0})
    engine.set_autonomy_level("test-entry-1", 2)
    assert engine.get_autonomy_level("test-entry-1") == 2


@pytest.mark.asyncio
async def test_advanced_gate_on_without_a_pin_is_refused(
    hass: HomeAssistant,
) -> None:
    """Checking the box with no PIN typed or stored asks for the PIN."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        }
    )

    assert result.get("type") == "form"
    assert result.get("errors") == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_advanced_retry_after_another_error_asks_for_the_pin_again(
    hass: HomeAssistant,
) -> None:
    """A redisplay blanks the PIN box; resubmitting must not save a PIN-less gate."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    await flow.async_step_user()
    first = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_CRITICAL_ACTION_PIN: "4321",
            "sentinel_quiet_hours_start": "22:00",
        }
    )
    assert first.get("type") == "form"
    assert first.get("errors") == {"base": "quiet_hours_incomplete"}

    # The user fixes quiet hours; the PIN box came back blank, the box checked.
    second = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        }
    )
    assert second.get("type") == "form"
    assert second.get("errors") == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_basic_rerun_drops_a_gate_with_no_stored_pin(
    hass: HomeAssistant,
) -> None:
    """A flag with no hash refuses every increase; Basic does not carry it."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        },
    )
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(dict(_BASIC_INPUT))

    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is False


def test_a_stopped_engine_refuses_level_changes() -> None:
    """A call during the unload window cannot write under the old options."""
    engine = _engine_for({CONF_SENTINEL_ENABLED: True})
    engine._stopped = True
    with pytest.raises(HomeAssistantError, match="reloading"):
        engine.set_autonomy_level("test-entry-1", 3)
    assert "test-entry-1" not in _AUTONOMY_OVERRIDES


def test_lowering_with_a_stale_pin_still_lowers() -> None:
    """A wrong PIN on a decrease is ignored rather than blocking it."""
    engine = _engine_for(_gated())
    engine.set_autonomy_level("test-entry-1", 3, pin="4321")
    engine.set_autonomy_level("test-entry-1", 2, pin="0000")
    assert engine.get_autonomy_level("test-entry-1") == 2


def test_returning_to_the_configured_level_needs_no_pin() -> None:
    """Going back up to the base level after lowering is not an increase."""
    engine = _engine_for(_gated())
    base_level = engine.get_autonomy_level("test-entry-1")
    engine.set_autonomy_level("test-entry-1", base_level - 1)
    engine.set_autonomy_level("test-entry-1", base_level)
    assert engine.get_autonomy_level("test-entry-1") == base_level
    with pytest.raises(HomeAssistantError, match="PIN is required"):
        engine.set_autonomy_level("test-entry-1", base_level + 1)


@pytest.mark.asyncio
async def test_engine_stop_keeps_a_lowered_level() -> None:
    """A reload must not undo a temporary reduction."""
    engine = _engine_for(_gated())
    base_level = engine.get_autonomy_level("test-entry-1")
    engine.set_autonomy_level("test-entry-1", base_level - 1)
    await engine.stop()
    assert _AUTONOMY_OVERRIDES["test-entry-1"][0] == base_level - 1


@pytest.mark.asyncio
async def test_advanced_gate_with_a_hash_but_no_salt_is_refused(
    hass: HomeAssistant,
) -> None:
    """A stored hash without its salt cannot verify any PIN."""
    flow, _ = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: "stored-hash",
        },
    )
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        }
    )
    assert result.get("errors") == {"base": "pin_required"}


@pytest.mark.parametrize("pin", [None, "4321"])
def test_gate_with_no_stored_pin_says_how_to_set_one(pin: str | None) -> None:
    """With the flag on and no PIN stored, both calls name the fix."""
    engine = _engine_for(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        }
    )
    with pytest.raises(HomeAssistantError, match=r"none is set\. Set one in Sentinel"):
        engine.set_autonomy_level("test-entry-1", 3, pin=pin)
