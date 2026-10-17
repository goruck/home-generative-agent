# ruff: noqa: S101
"""
A device is a lock, or an alarm, by what Home Assistant matches (#782).

Field test 2026-10-17: a scene named "HGA Test Unlock" was turned into a call
on the garage door lock, and a unit check showed "turn off the Kitchen Clock"
unlocking the front door: "lock" inside the name made it a lock call, and the
nearest lock by name was filled in when none matched. Review found the same
fill sent a call for one non-Latin lock name to another ("Задняя дверь" and
"Входная дверь" both reduced to an empty ASCII slug), and the alarm normalizer
read "security" in "Security Door Lock" as an alarm panel.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component

from custom_components.home_generative_agent.agent.graph import (
    _prepare_tool_args,
    _route_alarm_intent_to_tool,
)
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    _same_request,
    maybe_fill_lock_entity,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.agent.tools import (
    confirm_sensitive_action,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    RECOMMENDED_CRITICAL_ACTIONS,
)
from custom_components.home_generative_agent.core.utils import hash_pin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

confirm_tool = cast("Any", confirm_sensitive_action).coroutine

_EXPOSED = {
    "lock.front_door": "Front Door Lock",
    "lock.porch": "Porch",
    "lock.back": "Задняя дверь",
    "lock.entry": "Входная дверь",
    "lock.security_door": "Security Door Lock",
    "lock.side_a": "Side Door Lock",
    "lock.side_b": "Side Door Lock",
    # A lock and a switch sharing a name: HA acts on both.
    "lock.shed": "Shed",
    "switch.shed": "Shed",
    "switch.kitchen_clock": "Kitchen Clock",
    "light.security_light": "Security Light",
    "scene.unlock": "Unlock",
    "scene.lock_up": "Lock Up",
    "alarm_control_panel.home": "Home Alarm",
}
# Locks are never exposed to Assist by default; this one the user left hidden.
_HIDDEN = {"lock.hidden": "Hidden Door Lock"}


@pytest.fixture
async def home(hass: HomeAssistant) -> HomeAssistant:
    """Set up locks, look-alike devices and an alarm panel, as Assist sees them."""
    assert await async_setup_component(hass, "homeassistant", {})
    for entity_id, name in {**_EXPOSED, **_HIDDEN}.items():
        hass.states.async_set(entity_id, "on", {"friendly_name": name})
    for entity_id in _EXPOSED:
        async_expose_entity(hass, "conversation", entity_id, should_expose=True)
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
        ("intent__HassTurnOn", {"name": "Security Light"}),
    ],
)
@pytest.mark.parametrize("pin", [True, False])
def test_a_device_named_with_lock_or_alarm_keeps_its_own_target(
    home: HomeAssistant, tool_name: str, tool_args: dict[str, Any], *, pin: bool
) -> None:
    """The call HA gets names what the model asked for, as a lock or alarm never."""
    redirect, out = _prepare_tool_args(
        tool_name, dict(tool_args), _ctx(home, pin=pin), {"id": "c1"}
    )
    assert redirect is None
    assert out["name"] == tool_args["name"]
    assert not {"lock", "alarm_control_panel"} & set(out.get("domain", []))
    assert "entity_id" not in out
    assert "service" not in out


def test_a_name_shared_with_a_lock_is_held_but_not_rewritten(
    home: HomeAssistant,
) -> None:
    """HA turns off both "Shed"s, the lock too, so the gate holds the call as is."""
    redirect, out = _prepare_tool_args(
        "intent__HassTurnOff", {"name": "Shed"}, _ctx(home, pin=True), {"id": "c1"}
    )
    assert redirect is not None
    assert out == {"name": "Shed", "domain": []}


@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("Front Door Lock", 'unlock "Front Door Lock"'),
        # Not taken for an alarm panel by "security" in its name.
        ("Security Door Lock", 'unlock "Security Door Lock"'),
        # Not taken for another lock by a name reduced to ASCII.
        ("Задняя дверь", 'unlock "Задняя дверь"'),
    ],
)
def test_a_lock_named_as_ha_knows_it_is_held(
    home: HomeAssistant, name: str, line: str
) -> None:
    """HA's matcher finds only the lock, so the unlock is held for that lock."""
    ctx = _ctx(home, pin=True)
    redirect, out = _prepare_tool_args(
        "intent__HassTurnOff", {"name": name}, ctx, {"id": "c1"}
    )
    assert redirect is not None
    assert "requires_pin" in str(redirect.content)
    assert out["domain"] == ["lock"]
    assert out["service"] == "unlock"
    assert out["name"] == name
    (action,) = ctx.pending_actions.values()
    assert describe_pending_action(home, action) == line


def test_with_no_pin_a_named_lock_is_sent_as_that_one_lock(
    home: HomeAssistant,
) -> None:
    """The payload HA runs names the lock, so it never widens to every lock."""
    redirect, out = _prepare_tool_args(
        "intent__HassTurnOff", {"name": "Front Door Lock"}, _ctx(home, pin=False), {}
    )
    assert redirect is None
    assert out["domain"] == ["lock"]
    assert out["name"] == "Front Door Lock"


@pytest.mark.parametrize(
    ("name", "entity_id", "filled"),
    [
        # The name with "lock" added or taken off finds exactly one lock.
        ("Front Door", "lock.front_door", "Front Door Lock"),
        ("Porch Lock", "lock.porch", "Porch"),
        ("front door", "lock.front_door", "Front Door Lock"),
    ],
)
def test_a_lock_name_with_or_without_lock_is_completed(
    home: HomeAssistant, name: str, entity_id: str, filled: str
) -> None:
    """A declared lock call for "Front Door" is the lock "Front Door Lock"."""
    out = maybe_fill_lock_entity({"name": name, "domain": ["lock"]}, home)
    assert out["entity_id"] == entity_id
    assert out["name"] == filled


@pytest.mark.parametrize(
    "name",
    [
        # HA matches these as given: nothing to fill, nothing replaced.
        "Задняя дверь",
        "Front Door Lock",
        "lock.porch",
        # No lock of that name: never a near guess.
        "Garage",
        "Front",
        "Kitchen Clock",
        "Боковая дверь",
        "裏口",
        "!!!",
        # Two locks answer: no guess between them.
        "Side Door",
        # Hidden from Assist: never named, never filled.
        "Hidden Door",
    ],
)
def test_a_lock_is_never_guessed(home: HomeAssistant, name: str) -> None:
    """Only a name that finds exactly one exposed lock is ever filled in."""
    out = maybe_fill_lock_entity({"name": name, "domain": ["lock"]}, home)
    assert "entity_id" not in out
    assert out["name"] == name


def test_a_hidden_lock_is_not_a_lock_by_name(home: HomeAssistant) -> None:
    """A name only a hidden lock answers to is left for HA, which refuses it."""
    args = {"name": "Hidden Door Lock"}
    assert normalize_intent_for_lock("intent__HassTurnOff", dict(args), home) == args


@pytest.mark.parametrize("name", ["all", "   ", None, 123])
def test_a_name_that_cannot_be_matched_is_not_a_lock(
    home: HomeAssistant, name: object
) -> None:
    """No name, every device, or a non-text name: nothing to read a lock from."""
    args = {"name": name}
    assert normalize_intent_for_lock("intent__HassTurnOff", dict(args), home) == args
    assert normalize_intent_for_alarm("intent__HassTurnOff", dict(args), home) == args


def test_a_matcher_error_leaves_the_call_and_is_logged(
    home: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The gate's own matcher fails closed on the same error; this one says so."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError

    monkeypatch.setattr(intent, "async_match_targets", _raise)
    args = {"name": "Front Door Lock"}
    with caplog.at_level(logging.WARNING):
        out = normalize_intent_for_lock("intent__HassTurnOff", dict(args), home)
    assert out == args
    assert "Could not match the name" in caplog.text
    filled = maybe_fill_lock_entity({"name": "Front Door", "domain": ["lock"]}, home)
    assert "entity_id" not in filled


@pytest.mark.parametrize(
    ("domain", "name", "is_lock"),
    [
        # A declared domain decides, whatever the name matches.
        (["switch"], "Front Door Lock", False),
        ("  ", "Kitchen Clock", False),
        (["lock", "switch"], "Front Door", True),
        # A blank domain is no domain: the name decides.
        ([""], "Front Door Lock", True),
    ],
)
def test_a_declared_domain_decides_before_the_name(
    home: HomeAssistant, domain: object, name: str, *, is_lock: bool
) -> None:
    """The model's declared domain is never overwritten by a name match."""
    out = normalize_intent_for_lock(
        "intent__HassTurnOff", {"name": name, "domain": domain}, home
    )
    assert (out.get("service") == "unlock") is is_lock


def test_an_alarm_panel_named_as_ha_knows_it_is_routed_to_the_alarm_tool(
    home: HomeAssistant,
) -> None:
    """A name HA resolves to the alarm panel only is still an alarm call."""
    out = normalize_intent_for_alarm("intent__HassTurnOn", {"name": "Home Alarm"}, home)
    assert out["domain"] == ["alarm_control_panel"]
    assert out["service"] == "alarm_arm_home"
    assert _route_alarm_intent_to_tool("intent__HassTurnOn", out, home, {}) is not None


@pytest.mark.parametrize(
    ("tool_args", "redirected"),
    [
        # A declared non-alarm domain is respected, code or not.
        ({"name": "Security Light", "domain": ["light"]}, False),
        ({"name": "Front Door Lock", "domain": ["lock"], "code": "1234"}, False),
        # No domain and a numeric code, with a panel in the home: the hint stays.
        ({"name": "Downstairs", "code": "1234"}, True),
    ],
)
def test_the_alarm_redirect_respects_a_declared_domain(
    home: HomeAssistant, tool_args: dict[str, Any], *, redirected: bool
) -> None:
    """Never an alarm by the letters of a name; a code hints only with no domain."""
    result = _route_alarm_intent_to_tool("intent__HassTurnOff", tool_args, home, {})
    assert (result is not None) is redirected


def test_without_hass_only_a_declared_domain_counts() -> None:
    """No matcher to ask: the letters of the name are not evidence."""
    args = {"name": "Front Door Lock"}
    assert normalize_intent_for_lock("intent__HassTurnOff", dict(args), None) == args
    alarm = {"name": "Security Alarm"}
    assert normalize_intent_for_alarm("intent__HassTurnOn", dict(alarm), None) == alarm
    declared = normalize_intent_for_lock(
        "intent__HassTurnOff", {"name": "Front Door", "domain": "lock"}, None
    )
    assert declared["domain"] == ["lock"]
    assert declared["service"] == "unlock"


def test_a_reissued_lock_request_is_the_same_request(home: HomeAssistant) -> None:
    """Two holds of the same name-only unlock are recognized as one request."""
    held = []
    for call_id in ("c1", "c2"):
        ctx = _ctx(home, pin=True)
        _prepare_tool_args(
            "intent__HassTurnOff", {"name": "Front Door Lock"}, ctx, {"id": call_id}
        )
        held.extend(ctx.pending_actions.values())
    assert _same_request(held[0], held[1])


class _RecordingAPI:
    """Stand-in for the HA LLM API that records what would run."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call_tool(self, tool_input: Any) -> dict[str, object]:
        self.calls.append(dict(tool_input.tool_args))
        return {"success": True}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Front Door Lock", {"domain": ["lock"], "name": "Front Door Lock"}),
        # Held as given, and confirmed as given: never turned into a lock call.
        ("Shed", {"name": "Shed"}),
    ],
)
async def test_the_pin_confirms_the_call_that_was_held(
    home: HomeAssistant, name: str, expected: dict[str, Any]
) -> None:
    """Confirmation re-reads the held call with hass and runs the same target."""
    ctx = _ctx(home, pin=True)
    _prepare_tool_args("intent__HassTurnOff", {"name": name}, ctx, {"id": "c1"})
    ((action_id, action),) = ctx.pending_actions.items()
    action.update(user="u1", thread_id="conv1", reply_turn="turn2")
    hashed, salt = hash_pin("1234")
    api = _RecordingAPI()
    config = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": ctx.pending_actions,
            "user_id": "u1",
            "thread_id": "conv1",
            "pin_reply": PinReply("1234", turn_id="turn2"),
            "ha_llm_api": api,
            "hass": home,
        }
    }
    result = await confirm_tool(action_id, "1234", config=config, store=None)
    assert json.loads(result).get("status") == "completed"
    (sent,) = api.calls
    assert {key: sent.get(key) for key in expected} == expected
    if "domain" not in expected:
        assert "lock" not in (sent.get("domain") or [])
        assert "service" not in sent
