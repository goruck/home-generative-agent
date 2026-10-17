# ruff: noqa: S101
"""
A device is a lock by what Home Assistant matches, never by its letters (#782).

Field test 2026-10-17: a scene named "HGA Test Unlock" was turned into a call
on the garage door lock, and a unit check showed "turn off the Kitchen Clock"
unlocking the front door: "lock" inside the name made it a lock call, and the
nearest lock by name was filled in when none matched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.agent.graph import _prepare_tool_args
from custom_components.home_generative_agent.agent.helpers import (
    maybe_fill_lock_entity,
    normalize_intent_for_lock,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.const import (
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@pytest.fixture
def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up a front door lock, a clock switch, an "Unlock" scene, a namesake."""
    hass.states.async_set(
        "lock.front_door", "locked", {"friendly_name": "Front Door Lock"}
    )
    hass.states.async_set(
        "switch.kitchen_clock", "on", {"friendly_name": "Kitchen Clock"}
    )
    hass.states.async_set("scene.unlock", "unknown", {"friendly_name": "Unlock"})
    hass.states.async_set("scene.lock_up", "unknown", {"friendly_name": "Lock Up"})
    # A lock and a switch sharing a name: HA acts on both, so not a lock call.
    hass.states.async_set("lock.shed", "locked", {"friendly_name": "Shed"})
    hass.states.async_set("switch.shed", "off", {"friendly_name": "Shed"})
    return hass


def _ctx(hass: HomeAssistant, *, pin: bool) -> Any:
    return cast(
        "Any",
        SimpleNamespace(
            hass=hass,
            config={"configurable": {"thread_id": "t1"}},
            critical_actions=RECOMMENDED_CRITICAL_ACTIONS,
            pin_enabled=pin,
            pin_hash="hash" if pin else None,
            pin_salt="salt" if pin else None,
            pending_actions={},
        ),
    )


@pytest.mark.parametrize(
    ("tool_name", "tool_args"),
    [
        ("intent__HassTurnOff", {"name": "Kitchen Clock"}),
        ("intent__HassTurnOff", {"name": "Kitchen Clock", "domain": ["switch"]}),
        ("intent__HassTurnOn", {"name": "Unlock"}),
        ("intent__HassTurnOn", {"name": "Lock Up"}),
    ],
)
@pytest.mark.parametrize("pin", [True, False])
def test_a_device_named_with_lock_is_not_sent_to_a_lock(
    home: HomeAssistant, tool_name: str, tool_args: dict[str, Any], *, pin: bool
) -> None:
    """The call HA gets names what the model asked for, with no lock added."""
    redirect, out = _prepare_tool_args(
        tool_name, dict(tool_args), _ctx(home, pin=pin), {"id": "c1"}
    )
    assert redirect is None
    assert out["name"] == tool_args["name"]
    assert "lock" not in out.get("domain", [])
    assert "entity_id" not in out
    assert "service" not in out


def test_a_name_shared_with_a_lock_is_held_but_not_rewritten(
    home: HomeAssistant,
) -> None:
    """HA turns off both "Shed"s, the lock too, so the gate holds the call as is."""
    ctx = _ctx(home, pin=True)
    redirect, out = _prepare_tool_args(
        "intent__HassTurnOff", {"name": "Shed"}, ctx, {"id": "c1"}
    )
    assert redirect is not None
    assert out == {"name": "Shed", "domain": []}


def test_a_lock_named_in_full_is_still_a_lock_and_held(home: HomeAssistant) -> None:
    """HA's matcher finds only the lock, so the unlock is held for the PIN."""
    ctx = _ctx(home, pin=True)
    redirect, out = _prepare_tool_args(
        "intent__HassTurnOff", {"name": "Front Door Lock"}, ctx, {"id": "c1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert out["domain"] == ["lock"]
    assert out["service"] == "unlock"
    (action,) = ctx.pending_actions.values()
    assert describe_pending_action(home, action) == 'unlock "Front Door Lock"'


def test_a_lock_name_without_the_word_lock_is_completed(home: HomeAssistant) -> None:
    """A lock-domain call for "Front Door" is the lock "Front Door Lock"."""
    out = maybe_fill_lock_entity({"name": "Front Door", "domain": ["lock"]}, home)
    assert out["entity_id"] == "lock.front_door"
    assert out["name"] == "Front Door Lock"


@pytest.mark.parametrize("name", ["Garage", "Front", "Kitchen Clock"])
def test_a_lock_is_never_guessed_from_a_near_name(
    home: HomeAssistant, name: str
) -> None:
    """With no lock of that name, nothing is filled in for HA to act on."""
    out = maybe_fill_lock_entity({"name": name, "domain": ["lock"]}, home)
    assert "entity_id" not in out
    assert out["name"] == name


def test_without_hass_only_a_declared_lock_domain_counts() -> None:
    """No matcher to ask: the letters of the name are not evidence."""
    args = {"name": "Front Door Lock"}
    assert normalize_intent_for_lock("intent__HassTurnOff", dict(args)) == args
    declared = normalize_intent_for_lock(
        "intent__HassTurnOff", {"name": "Front Door", "domain": "lock"}
    )
    assert declared["domain"] == ["lock"]
    assert declared["service"] == "unlock"
