# ruff: noqa: S101
"""
Sentinel auto-execution follows what an action starts (#831).

A template switch whose action opens the garage, or an automation started by
a switch turning on, runs when Sentinel turns that switch on. The PIN gate
follows these (#780, #814); Sentinel's own gate checked only the call, so an
allowlisted `switch.turn_on` could open the garage with nobody there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent import indirect_pin
from custom_components.home_generative_agent.const import ACTION_POLICY_AUTO_EXECUTE

from .test_sentinel_autonomy_settings import _evaluate

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _call(action: str, entity_id: str) -> dict[str, Any]:
    return {"action": action, "target": {"entity_id": entity_id}}


@pytest.fixture
async def garage(hass: HomeAssistant) -> HomeAssistant:
    """Set up a garage door, a template opener, and a switch an automation chains."""
    hass.states.async_set(
        "cover.garage_door",
        "closed",
        {"device_class": "garage", "friendly_name": "Garage Door"},
    )
    hass.states.async_set("switch.porch", "off")
    hass.states.async_set("switch.chained", "off")
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": [
                        {
                            "name": "Opener",
                            "unique_id": "opener",
                            "turn_on": [_call("cover.open_cover", "cover.garage_door")],
                            "turn_off": [],
                        }
                    ]
                }
            ]
        },
    )
    assert await async_setup_component(
        hass,
        "automation",
        {
            "automation": [
                {
                    "id": "chain",
                    "alias": "Open when chained turns on",
                    "triggers": [
                        {
                            "trigger": "state",
                            "entity_id": "switch.chained",
                            "to": "on",
                        }
                    ],
                    "actions": [_call("cover.open_cover", "cover.garage_door")],
                }
            ]
        },
    )
    await hass.async_block_till_done()
    return hass


@pytest.mark.parametrize("entity", ["switch.opener", "switch.chained"])
async def test_a_switch_that_opens_the_garage_never_runs_alone(
    garage: HomeAssistant, entity: str
) -> None:
    """Its template action or the automation it starts opens the door."""
    result = _evaluate(["switch.turn_on"], ["switch"], entities=[entity], hass=garage)

    assert result.block_reason == "critical_action"


async def test_an_ordinary_switch_still_runs(garage: HomeAssistant) -> None:
    result = _evaluate(
        ["switch.turn_on"], ["switch"], entities=["switch.porch"], hass=garage
    )

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_locking_a_door_still_runs(garage: HomeAssistant) -> None:
    """The check only follows what an action starts; lock.lock starts nothing."""
    garage.states.async_set("lock.front_door", "unlocked")

    result = _evaluate(
        ["lock.lock"], ["lock"], entities=["lock.front_door"], hass=garage
    )

    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


async def test_a_failed_check_blocks_the_action(
    garage: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If what it starts cannot be checked, it does not run."""

    def _broken(*_args: Any) -> list[Any]:
        msg = "registry unreadable"
        raise RuntimeError(msg)

    monkeypatch.setattr(indirect_pin, "stored_action_calls", _broken)

    result = _evaluate(
        ["switch.turn_on"], ["switch"], entities=["switch.porch"], hass=garage
    )

    assert result.block_reason == "critical_action"
