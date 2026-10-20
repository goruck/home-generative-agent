# ruff: noqa: S101
"""
Intent scripts are PIN-gated by what their action does (#779).

Home Assistant gives the agent a tool per intent script: `intent_script__<Name>`
since 2026.9, the intent's own name before. The call names no domain or
service, so the gate never held it, whatever the intent script's action did:
one that unlocked the front door ran with no PIN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import _is_critical_action
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

RULES = RECOMMENDED_CRITICAL_ACTIONS


def _call(action: str, entity_id: str) -> dict[str, Any]:
    return {"action": action, "target": {"entity_id": entity_id}}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a front door lock, a light and intent scripts."""
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("light.porch", "off")
    assert await async_setup_component(hass, "intent", {})
    assert await async_setup_component(
        hass,
        "intent_script",
        {
            "intent_script": {
                "OpenFrontDoor": {
                    "description": "Let a visitor in",
                    "action": [_call("lock.unlock", "lock.front_door")],
                    "speech": {"text": "Done"},
                },
                "Porch Lights On": {
                    "action": [_call("light.turn_on", "light.porch")],
                    "speech": {"text": "Done"},
                },
                "SayHello": {"speech": {"text": "Hello"}},
                # The door to open comes from the model's slot.
                "OpenDoor": {
                    "action": [
                        {
                            "action": "lock.unlock",
                            "target": {"entity_id": "{{ door }}"},
                        }
                    ],
                },
            }
        },
    )
    await hass.async_block_till_done()
    return hass


@pytest.mark.parametrize(
    ("tool_name", "critical"),
    [
        # Home Assistant 2026.9 and later, alone or under a merged API.
        ("intent_script__OpenFrontDoor", True),
        ("assist__intent_script__OpenFrontDoor", True),
        # Before 2026.9 the tool was named by the intent.
        ("OpenFrontDoor", True),
        # A slot the model fills cannot be read ahead: held.
        ("intent_script__OpenDoor", True),
        ("intent_script__Porch_Lights_On", False),
        ("Porch Lights On", False),
        # Speech only: nothing runs.
        ("intent_script__SayHello", False),
        # An intent script tool whose handler is gone: held.
        ("intent_script__Removed", True),
    ],
)
async def test_an_intent_script_is_held_by_what_it_runs(
    home: HomeAssistant, tool_name: str, *, critical: bool
) -> None:
    """The example from the issue, in every tool name HA has used."""
    assert _is_critical_action({}, RULES, tool_name, home) is critical


async def test_a_tool_sharing_no_intent_s_name_is_not_affected(
    home: HomeAssistant,
) -> None:
    """A tool that is not an intent script is checked as before."""
    assert not _is_critical_action({}, RULES, "GetLiveContext", home)


async def test_the_pin_line_names_the_intent(home: HomeAssistant) -> None:
    """The line says what the PIN allows, from Home Assistant."""
    line = describe_pending_action(
        home, {"tool_name": "intent_script__OpenFrontDoor", "tool_args": {}}
    )
    assert "OpenFrontDoor" in line
