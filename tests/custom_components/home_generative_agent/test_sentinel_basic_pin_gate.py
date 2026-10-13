# ruff: noqa: S101
"""
Regression tests for #762: a level-increase PIN set in Basic setup was ignored.

The Basic form hashed and stored the PIN but left
``sentinel_require_pin_for_level_increase`` at its ``False`` default, and
``SentinelEngine.set_autonomy_level`` gates the whole check on that flag, so
``sentinel_set_autonomy_level`` raised the level with no PIN at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_SENTINEL_AUTONOMY_LEVEL,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.core.utils import hash_pin
from custom_components.home_generative_agent.sentinel.engine import (
    _AUTONOMY_OVERRIDES,
    SentinelEngine,
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
    yield
    _AUTONOMY_OVERRIDES.clear()


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

    await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
        }
    )
    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] == "stored-hash"
