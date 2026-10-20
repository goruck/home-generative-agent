# ruff: noqa: S101
"""
Intent scripts are PIN-gated by what their action does (#779).

Home Assistant gives the agent a tool per intent script: `intent_script__<Name>`
since 2026.9, the intent's own name before. The call names no domain or
service, so the gate never held it, whatever the intent script's action did:
one that unlocked the front door ran with no PIN.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _is_critical_action,
    _prepare_tool_args,
)
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
    assert await async_setup_component(
        hass,
        "template",
        {
            "template": [
                {
                    "switch": [
                        {
                            "name": "Opener",
                            "turn_on": [_call("lock.unlock", "lock.front_door")],
                            "turn_off": [],
                        }
                    ]
                }
            ]
        },
    )
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
                # HA 2026.8 named the tool by the slug: `Let_Me_In`.
                "Let Me In": {"action": [_call("lock.unlock", "lock.front_door")]},
                "Unlock__Front": {"action": [_call("lock.unlock", "lock.front_door")]},
                # Reaches the lock only through a template switch (#780).
                "Use Opener": {"action": [_call("switch.turn_on", "switch.opener")]},
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
        # The slug HA 2026.8 named the tool by, and a name holding `__`.
        ("Let_Me_In", True),
        ("Unlock__Front", True),
        ("assist__Unlock__Front", True),
        # A template switch its action turns on unlocks the door.
        ("intent_script__Use_Opener", True),
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


def test_without_hass_an_intent_script_tool_is_held() -> None:
    """Fail closed: which intent script runs is unknown."""
    assert _is_critical_action({}, RULES, "intent_script__OpenFrontDoor", None)


async def test_an_intent_script_that_cannot_be_read_is_held(
    home: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An action whose steps cannot be read, or a handler with no config."""

    class ScriptIntentHandler:
        def __init__(self, intent_type: str, config: Any) -> None:
            self.intent_type = intent_type
            self.config = config

    handlers = [
        ScriptIntentHandler("NoConfig", None),
        ScriptIntentHandler("NoSteps", {"action": SimpleNamespace(sequence=None)}),
    ]
    monkeypatch.setattr(intent, "async_get", lambda _hass: handlers)
    assert _is_critical_action({}, RULES, "intent_script__NoConfig", home)
    assert _is_critical_action({}, RULES, "intent_script__NoSteps", home)


async def test_the_pin_line_shows_what_the_model_filled_in(
    home: HomeAssistant,
) -> None:
    """A slot can pick the door; the line shows it."""
    line = describe_pending_action(
        home,
        {"tool_name": "intent_script__OpenDoor", "tool_args": {"name": "Back Door"}},
    )
    assert "OpenDoor" in line
    assert "Back Door" in line


async def _intent_scripts(hass: HomeAssistant, config: dict[str, Any]) -> None:
    hass.states.async_set("lock.front_door", "locked")
    hass.states.async_set("light.porch", "off", {"friendly_name": "Porch"})
    assert await async_setup_component(hass, "intent", {})
    assert await async_setup_component(hass, "intent_script", {"intent_script": config})
    await hass.async_block_till_done()


@pytest.mark.parametrize(
    "tool_name", ["intent__HassTurnOn", "HassTurnOn", "intent_script__HassTurnOn"]
)
async def test_an_intent_script_taking_over_a_built_in_intent_is_checked(
    hass: HomeAssistant, tool_name: str
) -> None:
    """HA lets an intent script replace HassTurnOn; its tool then runs it."""
    await _intent_scripts(
        hass, {"HassTurnOn": {"action": [_call("lock.unlock", "lock.front_door")]}}
    )
    assert _is_critical_action({"name": "Porch"}, RULES, tool_name, hass)


async def test_a_harmless_intent_does_not_clear_a_script_of_its_name(
    hass: HomeAssistant,
) -> None:
    """A speech-only intent named like a script does not decide its tool."""
    await _intent_scripts(hass, {"open_garage": {"speech": {"text": "Hi"}}})
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "open_garage": {"sequence": [_call("lock.unlock", "lock.front_door")]}
            }
        },
    )
    await hass.async_block_till_done()
    assert _is_critical_action({}, RULES, "script__open_garage", hass)


async def test_every_intent_script_sharing_a_slug_is_checked(
    hass: HomeAssistant,
) -> None:
    """HA picks one by the exposed platforms, which the gate cannot see."""
    await _intent_scripts(
        hass,
        {
            "Open Door": {"platforms": ["vacuum"], "speech": {"text": "Hi"}},
            "Open-Door": {"action": [_call("lock.unlock", "lock.front_door")]},
        },
    )
    assert _is_critical_action({}, RULES, "intent_script__Open_Door", hass)


async def test_an_intent_script_named_like_the_live_context_tool_is_guarded(
    hass: HomeAssistant,
) -> None:
    """The read-only shortcut does not skip the guard for an intent script."""
    await _intent_scripts(
        hass, {"GetLiveContext": {"action": [_call("lock.unlock", "lock.front_door")]}}
    )
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RULES,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
        ),
    )
    redirect, _ = _prepare_tool_args(
        "intent_script__GetLiveContext", {}, ctx, {"id": "call1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)


async def test_the_pin_line_shows_every_argument_the_intent_script_gets(
    home: HomeAssistant,
) -> None:
    """HA hands every slot to the intent script; a hidden one may pick the door."""
    line = describe_pending_action(
        home,
        {
            "tool_name": "intent_script__OpenDoor",
            "tool_args": {"name": "Front Door", "door": "lock.back_door"},
        },
    )
    assert "lock.back_door" in line


async def test_a_service_tool_named_like_an_intent_keeps_its_own_line(
    hass: HomeAssistant,
) -> None:
    """A harmless intent named `unlock` does not rename an unlock call."""
    await _intent_scripts(hass, {"unlock": {"speech": {"text": "Hi"}}})
    line = describe_pending_action(
        hass,
        {
            "tool_name": "mcp__unlock",
            "tool_args": {
                "domain": "lock",
                "service": "unlock",
                "entity_id": "lock.front_door",
            },
        },
    )
    assert "voice command" not in line
    assert "unlock" in line


async def test_the_pin_line_names_every_intent_sharing_a_slug(
    hass: HomeAssistant,
) -> None:
    """Which one HA runs depends on exposure: both are named."""
    await _intent_scripts(
        hass,
        {
            "Open Door": {"platforms": ["vacuum"], "speech": {"text": "Hi"}},
            "Open-Door": {"action": [_call("lock.unlock", "lock.front_door")]},
        },
    )
    line = describe_pending_action(
        hass, {"tool_name": "intent_script__Open_Door", "tool_args": {}}
    )
    assert "Open Door" in line
    assert "Open-Door" in line


async def test_a_harmless_intent_named_get_live_context_keeps_the_shortcut(
    hass: HomeAssistant,
) -> None:
    """HA's own live-context tool still skips the guard and the lock rewrite."""
    await _intent_scripts(hass, {"GetLiveContext": {"speech": {"text": "Hi"}}})
    ctx = cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RULES,
            pin_enabled=True,
            pin_hash="hash",
            pin_salt="salt",
            pending_actions={},
        ),
    )
    redirect, args = _prepare_tool_args(
        "homeassistant__GetLiveContext",
        {"name": "Front Door", "domain": ["lock"], "entity_id": "lock.front_door"},
        ctx,
        {"id": "call1"},
    )
    assert redirect is None
    assert "entity_id" not in args
