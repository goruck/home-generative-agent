# ruff: noqa: S101
"""
The deterministic line that names the action a PIN confirms.

Field test 2026-10-15: "Block the playroom, block" (speech-to-text for "lock
the playroom lock") reached the model without HassTurnOn, so it called
HassTurnOff, which is *unlock* for a lock. The model asked for the PIN "to
lock the playroom door" and, after the PIN, said "now locked". The door was
unlocked. What the user is told the PIN confirms must come from the action
that is held, not from the model's prose, and must name what Home Assistant
will run: the verb its on/off handler picks for the domain, and the scope of
a call that names no device (HA reads only name, area and floor on an intent;
an entity id alone is ignored and the whole domain acts).
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.helpers import (
    MAX_PIN_REPLIES,
    PIN_OUTCOME_DONE,
    PIN_OUTCOME_FAILED,
    PIN_OUTCOME_UNKNOWN,
    PinReply,
    disclose_pin_prompts,
    keep_pin_prompts,
    parse_tool_response,
    prompts_awaiting_reply,
    register_pending_action,
    start_pin_turn,
)
from custom_components.home_generative_agent.agent.pin_messages import (
    describe_pending_action,
)
from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin

PIN = "1234"
PLAYROOM = {"name": "Playroom Door Lock", "domain": ["lock"]}
PLAYROOM_LINE = 'unlock "Playroom Door Lock"'

confirm = cast("Any", confirm_sensitive_action).coroutine


def _hass(language: str = "en") -> Any:
    return types.SimpleNamespace(config=types.SimpleNamespace(language=language))


def _action(tool_name: str, tool_args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"tool_name": tool_name, "tool_args": tool_args, **extra}


# ---------------------------------------------------------------------------
# describe_pending_action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected"),
    [
        # HA runs HassTurnOff on a lock as unlock and HassTurnOn as lock.
        ("HassTurnOff", PLAYROOM, PLAYROOM_LINE),
        ("HassTurnOn", PLAYROOM, 'lock "Playroom Door Lock"'),
        ("intent__HassTurnOff", PLAYROOM, PLAYROOM_LINE),
        # Never a lock by the letters of its name (#782): with nothing
        # matched, "lock" inside "Kitchen Clock" does not make it one.
        ("HassTurnOff", {"name": "Kitchen Clock"}, 'turn off "Kitchen Clock"'),
        # A service the model added does not change what HA runs.
        ("HassTurnOff", {**PLAYROOM, "service": "lock"}, PLAYROOM_LINE),
        # The gate reads the domain off the entity id when nothing else says
        # lock (field test: a name without "lock"): still unlock, not "turn off".
        (
            "HassTurnOff",
            {"name": "Front Door", "entity_id": "lock.front_door", "service": "unlock"},
            'unlock "Front Door"',
        ),
        # Without a name, the confirm path sends the entity id as the name
        # (HA accepts an entity id there), so that one lock is what runs.
        (
            "HassTurnOff",
            {"domain": ["lock"], "entity_id": "lock.playroom"},
            'unlock "lock.playroom"',
        ),
        # Covers and valves: HA's on/off handler opens and closes them.
        (
            "HassTurnOn",
            {"name": "Garage Door", "domain": ["cover"]},
            'open "Garage Door"',
        ),
        (
            "HassTurnOff",
            {"name": "Garage Door", "domain": "cover"},
            'close "Garage Door"',
        ),
        (
            "HassTurnOn",
            {"name": "Garage Door", "domain": ["cover"], "service": "close_cover"},
            'open "Garage Door"',
        ),
        (
            "HassTurnOff",
            {"name": "Main Valve", "domain": ["valve"]},
            'close "Main Valve"',
        ),
        # Not a lock, cover or valve: the on/off intent by its own words.
        ("HassTurnOn", {"name": "Garage Heater"}, 'turn on "Garage Heater"'),
        ("HassTurnOff", {"name": "Garage Heater"}, 'turn off "Garage Heater"'),
        # Any other critical call: its service, readable.
        (
            "call_service",
            {"domain": "cover", "service": "open_cover", "entity_id": ["cover.garage"]},
            'open cover "cover.garage"',
        ),
    ],
)
def test_the_action_is_described_from_what_runs(
    tool_name: str, tool_args: dict[str, Any], expected: str
) -> None:
    assert describe_pending_action(_hass(), _action(tool_name, tool_args)) == expected


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected"),
    [
        # HA reads name, area and floor on an intent. With none of them, the
        # whole domain acts, and the line says so instead of "an unnamed
        # device" (Codex, security specialist and Red Team, 2026-10-15).
        ("HassTurnOff", {"domain": ["lock"]}, "unlock every lock"),
        ("HassTurnOff", {"domain": "lock"}, "unlock every lock"),
        (
            "HassTurnOff",
            {"domain": ["lock"], "area": "Garage"},
            "unlock every lock in Garage",
        ),
        (
            "HassTurnOff",
            {"domain": ["lock"], "floor": "Upstairs"},
            "unlock every lock on floor Upstairs",
        ),
        # HA drops a name of "all".
        ("HassTurnOn", {"domain": ["lock"], "name": "all"}, "lock every lock"),
        # HA ignores an entity id, so the payload sends it as the name on
        # every domain, and the line names that one device.
        (
            "HassTurnOn",
            {"domain": ["cover"], "entity_id": "cover.garage"},
            'open "cover.garage"',
        ),
        (
            "HassTurnOff",
            {"domain": ["lock"], "entity_ids": ["lock.front_door"]},
            'unlock "lock.front_door"',
        ),
        # "all" is read by HA as no name: the entity id wins, else the scope.
        (
            "HassTurnOff",
            {"domain": ["lock"], "name": "all", "entity_id": "lock.front_door"},
            'unlock "lock.front_door"',
        ),
        # HA compares "all" exactly; "All" is a device named All.
        ("HassTurnOn", {"domain": ["lock"], "name": "All"}, 'lock "All"'),
        (
            "HassTurnOff",
            {"domain": ["switch"], "area": "Patio"},
            "turn off every switch device in Patio",
        ),
        # Nothing HA can target: HA refuses, and the line has no device to name.
        ("HassTurnOff", {}, "turn off an unnamed device"),
    ],
)
def test_a_call_that_names_no_device_says_its_scope(
    tool_name: str, tool_args: dict[str, Any], expected: str
) -> None:
    assert describe_pending_action(_hass(), _action(tool_name, tool_args)) == expected


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        (
            {"domain": ["lock"], "name": "Door", "area": "Garage"},
            'unlock "Door" in Garage',
        ),
        (
            {"domain": ["lock"], "name": "Door", "area": "House"},
            'unlock "Door" in House',
        ),
        (
            {"domain": ["lock"], "name": "Door", "floor": "Upstairs"},
            'unlock "Door" on floor Upstairs',
        ),
        (
            {"domain": ["lock"], "name": "Door", "area": "Garage", "floor": "Ground"},
            'unlock "Door" in Garage on floor Ground',
        ),
        (
            {"domain": ["cover"], "name": "Door", "area": "Garage"},
            'close "Door" in Garage',
        ),
        # A blank area is not one: HA drops it, and so does the line.
        ({"domain": ["lock"], "name": "Door", "area": " "}, 'unlock "Door"'),
    ],
)
def test_a_named_device_in_an_area_says_which_one(
    tool_args: dict[str, Any], expected: str
) -> None:
    """Two locks named "Door" read differently by where they are (#784)."""
    assert (
        describe_pending_action(_hass(), _action("HassTurnOff", tool_args)) == expected
    )


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        # A service tool's scene.apply names the states it sets (#786).
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"lock.front_door": "unlocked"},
            },
            'set "Front Door" (lock.front_door) to unlocked',
        ),
        (
            {
                "domain": "scene",
                "service": "apply",
                "data": {"entities": {"lock.front_door": {"state": "unlocked"}}},
            },
            'set "Front Door" (lock.front_door) to unlocked',
        ),
        (
            {
                "domain": "scene",
                "service": "create",
                "scene_id": "x",
                "entities": {
                    "cover.garage": "open",
                    "lock.front_door": "unlocked",
                    "light.a": "on",
                    "light.b": "on",
                },
            },
            (
                'create the scene "x" that sets "cover.garage" to open, '
                '"Front Door" (lock.front_door) to unlocked, "light.a" to on, '
                '"light.b" to on'
            ),
        ),
        # Targets nested under `target` or `data` are named, not "unnamed".
        (
            {
                "domain": "script",
                "service": "turn_on",
                "data": {"entity_id": "script.open_door"},
            },
            'turn on "script.open_door"',
        ),
        (
            {
                "domain": "lock",
                "service": "unlock",
                "target": {"entity_id": ["lock.front_door"]},
            },
            'unlock "Front Door" (lock.front_door)',
        ),
    ],
)
async def test_a_service_tool_call_names_what_it_sets(
    hass: Any, tool_args: dict[str, Any], expected: str
) -> None:
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    assert describe_pending_action(hass, _action("call_service", tool_args)) == expected


async def test_a_scene_call_is_named_in_czech(hass: Any) -> None:
    hass.config.language = "cs"
    args = {
        "domain": "scene",
        "service": "apply",
        "entities": {"lock.front_door": "unlocked"},
    }
    assert (
        describe_pending_action(hass, _action("call_service", args))
        == "nastavit „lock.front_door“ na unlocked"
    )


def test_an_automation_is_described_by_its_alias_and_what_it_runs() -> None:
    action = _action(
        "add_automation",
        {},
        automation_config={"id": "1", "alias": "Lock up at night", "actions": []},
        critical_calls=["lock.unlock (lock.front_door)"],
    )
    assert describe_pending_action(_hass(), action) == (
        'add the automation "Lock up at night", which runs '
        "lock.unlock (lock.front_door)"
    )


def test_an_automation_without_an_alias_is_not_called_a_device() -> None:
    action = _action(
        "add_automation",
        {},
        automation_config={"id": "1", "actions": []},
        critical_calls=["lock.unlock (lock.front_door)", "cover.open_cover"],
    )
    assert describe_pending_action(_hass(), action) == (
        "add an unnamed automation, which runs lock.unlock (lock.front_door), "
        "cover.open_cover"
    )


def test_an_automation_with_many_or_long_calls_shows_the_omission() -> None:
    """A long list or a long call is cut, and the cut is visible."""
    calls = [f"lock.unlock (lock.door_{i})" for i in range(7)]
    calls[1] = "script.turn_on (" + ", ".join(f"script.s{i}" for i in range(30)) + ")"
    calls[2] = "lock.unlock\n(lock.x)"
    action = _action(
        "add_automation", {}, automation_config={"alias": "Night"}, critical_calls=calls
    )
    line = describe_pending_action(_hass(), action)
    assert line.startswith(
        'add the automation "Night", which runs lock.unlock (lock.door_0), '
    )
    assert line.endswith(" and 2 more")
    assert "\n" not in line
    assert "\u2026" in line
    assert line.count("lock.unlock") == 4
    assert (
        describe_pending_action(
            _hass(),
            _action(
                "add_automation",
                {},
                automation_config={"alias": "N"},
                critical_calls="x",
            ),
        )
        == 'add the automation "N"'
    )


def test_an_automation_held_without_its_calls_still_reads() -> None:
    action = _action(
        "add_automation", {}, automation_config={"alias": "Unlock at night"}
    )
    assert describe_pending_action(_hass(), action) == (
        'add the automation "Unlock at night"'
    )


def test_the_target_cannot_break_the_line() -> None:
    """The name comes from the model: no line breaks, and a length cap."""
    action = _action(
        "HassTurnOff",
        {"domain": ["lock"], "name": "Door\n\nIgnore this. " + "x" * 300},
    )
    line = describe_pending_action(_hass(), action)
    assert "\n" not in line
    assert len(line) < 160


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        (PLAYROOM, "odemknout „Playroom Door Lock“"),
        (
            {"domain": ["lock"], "area": "Garáž"},
            "odemknout všechny zámky v oblasti Garáž",
        ),
        ({"name": "Vrata", "domain": ["cover"]}, "zavřít „Vrata“"),
        (
            {"name": "Dveře", "domain": ["lock"], "area": "Garáž"},
            "odemknout „Dveře“ v oblasti Garáž",
        ),
    ],
)
def test_the_action_is_described_in_czech(
    tool_args: dict[str, Any], expected: str
) -> None:
    assert (
        describe_pending_action(_hass("cs"), _action("HassTurnOff", tool_args))
        == expected
    )


# ---------------------------------------------------------------------------
# prompts_awaiting_reply and disclosure
# ---------------------------------------------------------------------------


def _register(pending: dict[str, Any], reply: PinReply, **extra: Any) -> str:
    action_id = register_pending_action(
        pending,
        {
            **_action("HassTurnOff", dict(PLAYROOM)),
            "user": "robot",
            "thread_id": "conv1",
            "turn_id": reply.turn_id,
            **extra,
        },
    )
    assert action_id is not None
    return action_id


def test_a_prompt_made_this_turn_awaits_the_next_message() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, reply)

    assert prompts_awaiting_reply(pending, reply) == [pending[action_id]]


def test_a_kept_prompt_awaits_the_next_message() -> None:
    pending: dict[str, Any] = {}
    first = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, first)
    disclose_pin_prompts(pending, first)

    reply = start_pin_turn(pending, "conv1", "sorry what")
    assert prompts_awaiting_reply(pending, reply) == []

    keep_pin_prompts(pending, reply)
    assert prompts_awaiting_reply(pending, reply) == [pending[action_id]]


def test_another_turns_prompt_is_not_awaiting_this_reply() -> None:
    pending: dict[str, Any] = {}
    first = start_pin_turn(pending, "conv1", "unlock the playroom door")
    _register(pending, first)
    other = start_pin_turn(pending, "conv2", "hello")

    assert prompts_awaiting_reply(pending, other) == []


def test_an_exhausted_prompt_is_not_announced() -> None:
    """After its last reply the prompt waits only to be dropped: no line."""
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, reply)
    disclose_pin_prompts(pending, reply)
    for _ in range(MAX_PIN_REPLIES):
        reply = start_pin_turn(pending, "conv1", "9999")
        keep_pin_prompts(pending, reply)

    assert action_id in pending
    assert prompts_awaiting_reply(pending, reply) == []


def test_a_request_repeated_on_the_last_reply_starts_a_fresh_prompt() -> None:
    """
    The exhausted prompt cannot be renewed, so the request said again is new.

    Codex structured review, 2026-10-15: the re-issue carried renew past the
    reply cap, the line promised a confirmation, and the next turn dropped it.
    """
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    _register(pending, reply)
    disclose_pin_prompts(pending, reply)
    for _ in range(MAX_PIN_REPLIES - 1):
        reply = start_pin_turn(pending, "conv1", "9999")
        keep_pin_prompts(pending, reply)
        disclose_pin_prompts(pending, reply)
    old_id = next(iter(pending))
    pending[old_id]["attempts"] = 2
    last = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, last)

    fresh = pending[action_id]
    assert fresh["turn_id"] == last.turn_id
    assert fresh.get("reply_turn") is None
    assert not fresh.get("replies")
    assert not fresh.get("renew")
    assert "disclosed" not in fresh
    assert fresh["attempts"] == 2
    assert prompts_awaiting_reply(pending, last) == [fresh]
    disclose_pin_prompts(pending, last)
    answer = start_pin_turn(pending, "conv1", PIN)
    assert pending[action_id]["reply_turn"] == answer.turn_id


def test_a_prompt_whose_line_was_never_delivered_cannot_be_answered() -> None:
    """A cancelled or crashed turn leaves nothing the user was told to trust."""
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, reply)

    start_pin_turn(pending, "conv1", PIN)

    assert action_id not in pending


def test_disclosure_names_the_prompts_and_marks_them() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, reply)

    assert disclose_pin_prompts(pending, reply) == [pending[action_id]]
    assert pending[action_id]["disclosed"] is True
    answer = start_pin_turn(pending, "conv1", PIN)
    assert pending[action_id]["reply_turn"] == answer.turn_id


def test_a_disclosed_prompt_survives_a_turn_that_failed() -> None:
    """The renewal decision: the user heard the line, and their turn failed."""
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    action_id = _register(pending, reply)
    disclose_pin_prompts(pending, reply)

    failed = start_pin_turn(pending, "conv1", PIN)
    keep_pin_prompts(pending, failed)
    again = start_pin_turn(pending, "conv1", PIN)

    assert pending[action_id]["reply_turn"] == again.turn_id


def test_a_reissued_request_keeps_its_disclosure() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    _register(pending, reply)
    disclose_pin_prompts(pending, reply)
    answer = start_pin_turn(pending, "conv1", PIN)
    action_id = _register(pending, answer)

    assert pending[action_id]["disclosed"] is True


# ---------------------------------------------------------------------------
# The guard and the confirmation record the action
# ---------------------------------------------------------------------------


def _ctx(pending: dict[str, Any], reply: PinReply, api: Any = None) -> Any:
    hashed, salt = hash_pin(PIN)
    config = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": pending,
            "user_id": "robot",
            "thread_id": "conv1",
            "pin_reply": reply,
            "ha_llm_api": api,
            "hass": None,
        }
    }
    return graph.ToolExecutionContext(
        hass=MagicMock(),
        store=MagicMock(),
        config=cast("Any", config),
        ha_llm_api=api,
        critical_actions=[],
        pin_enabled=True,
        pin_hash=hashed,
        pin_salt=salt,
        pending_actions=pending,
        state=cast("Any", {}),
    )


def test_the_pin_request_tells_the_model_which_action_it_is() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "block the playroom block")
    ctx = _ctx(pending, reply)
    message = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c1"}
    )
    assert message is not None
    payload = json.loads(cast("str", message.content))
    assert payload["action"] == PLAYROOM_LINE


def test_the_automation_request_carries_its_critical_calls() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the door at 7")
    cfg = _ctx(pending, reply).config["configurable"]
    config = {
        "alias": "Morning",
        "triggers": [{"trigger": "time", "at": "07:00:00"}],
        "actions": [
            {"action": "lock.unlock", "target": {"entity_id": "lock.front_door"}}
        ],
    }

    result = tools_module._maybe_require_pin_for_automation(
        cfg=cfg, validated_config=config, ha_automation_config=dict(config, id="1")
    )

    assert result is not None
    payload = json.loads(result)
    assert payload["action"] == (
        'add the automation "Morning", which runs lock.unlock (lock.front_door)'
    )
    assert pending[payload["action_id"]]["critical_calls"] == [
        "lock.unlock (lock.front_door)"
    ]


class _FakeAPI:
    def __init__(self, result: Any = None) -> None:
        self.result = {"success": True} if result is None else result
        self.sent: list[dict[str, Any]] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.sent.append(dict(tool_input.tool_args))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _SlowAPI(_FakeAPI):
    async def async_call_tool(self, tool_input: Any) -> Any:
        self.sent.append(dict(tool_input.tool_args))
        await asyncio.sleep(60)
        return self.result


async def _request_and_confirm(
    api: Any, args: dict[str, Any] | None = None
) -> tuple[PinReply, str]:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    ctx = _ctx(pending, reply, api)
    message = graph._critical_action_guard(
        "HassTurnOff", dict(args or PLAYROOM), ctx, {"id": "c1"}
    )
    assert message is not None
    action_id = json.loads(cast("str", message.content))["action_id"]
    disclose_pin_prompts(pending, reply)

    answer = start_pin_turn(pending, "conv1", PIN)
    ctx.config["configurable"]["pin_reply"] = answer
    return answer, await confirm(action_id, PIN, config=ctx.config, store=None)


async def test_a_confirmed_action_is_recorded_on_the_reply() -> None:
    answer, result = await _request_and_confirm(_FakeAPI())

    assert json.loads(result)["status"] == "completed"
    assert answer.confirmed == PLAYROOM_LINE
    assert answer.outcome == PIN_OUTCOME_DONE


async def test_a_wrong_pin_records_nothing() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    ctx = _ctx(pending, reply, _FakeAPI())
    message = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c1"}
    )
    assert message is not None
    action_id = json.loads(cast("str", message.content))["action_id"]
    disclose_pin_prompts(pending, reply)

    answer = start_pin_turn(pending, "conv1", "9999")
    ctx.config["configurable"]["pin_reply"] = answer
    await confirm(action_id, "9999", config=ctx.config, store=None)

    assert answer.confirmed == ""
    assert answer.outcome == ""


@pytest.mark.parametrize(
    "result",
    [
        HomeAssistantError("lock jammed"),
        # HA 2026.10's error flag.
        llm.ToolResult(data={"error": "jammed"}, error=True),
        # A tool that reports failure in its payload, as the normal path reads it.
        {"success": False, "result_text": "Device unavailable"},
        {"warning": "no entity matched"},
    ],
)
async def test_a_failed_action_is_recorded_as_failed(result: Any) -> None:
    """No "PIN accepted: unlock …" for a lock that did not unlock."""
    answer, reply = await _request_and_confirm(_FakeAPI(result))

    assert "completed" not in reply
    assert answer.confirmed == PLAYROOM_LINE
    assert answer.outcome == PIN_OUTCOME_FAILED


async def test_an_action_with_an_unknown_outcome_is_recorded_as_such() -> None:
    answer, reply = await _request_and_confirm(_FakeAPI(RuntimeError("boom")))

    assert "may have run" in reply
    assert answer.confirmed == PLAYROOM_LINE
    assert answer.outcome == PIN_OUTCOME_UNKNOWN


async def test_a_timed_out_action_is_recorded_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools_module, "POST_CLAIM_TIMEOUT_S", 0.01)
    answer, reply = await _request_and_confirm(_SlowAPI())

    assert "may have run" in reply
    assert answer.outcome == PIN_OUTCOME_UNKNOWN


async def test_the_confirm_path_sends_what_the_normal_path_sends() -> None:
    """
    An entity id alone must reach HA as the name, or every lock acts.

    HA's intents read name, area and floor; `domain` alone passes the
    "cannot target all" guard and matches every exposed lock. The normal
    path converts after the guard; the confirm path has to do the same.
    """
    api = _FakeAPI()
    answer, result = await _request_and_confirm(
        api, {"domain": ["lock"], "entity_id": "lock.playroom"}
    )

    assert json.loads(result)["status"] == "completed"
    assert api.sent == [
        {
            "domain": ["lock"],
            "service": "unlock",
            "name": "lock.playroom",
            "entity_id": "lock.playroom",
        }
    ]
    assert answer.confirmed == 'unlock "lock.playroom"'


@pytest.mark.parametrize(
    ("args", "key", "line"),
    [
        ({"domain": ["lock"], "area": "Garage"}, "area", "unlock every lock in Garage"),
        (
            {"domain": ["lock"], "floor": "Upstairs"},
            "floor",
            "unlock every lock on floor Upstairs",
        ),
    ],
)
async def test_a_scoped_lock_call_keeps_its_scope_on_confirm(
    args: dict[str, Any], key: str, line: str
) -> None:
    api = _FakeAPI()
    answer, result = await _request_and_confirm(api, args)

    assert json.loads(result)["status"] == "completed"
    assert api.sent[0][key] == args[key]
    assert "name" not in api.sent[0]
    assert answer.confirmed == line


async def test_a_named_lock_keeps_the_area_the_user_said() -> None:
    api = _FakeAPI()
    answer, _result = await _request_and_confirm(
        api, {"domain": ["lock"], "name": "Front Door", "area": "Porch"}
    )

    assert api.sent[0]["name"] == "Front Door"
    assert api.sent[0]["area"] == "Porch"
    # The area HA matches the name in is on the line too (#784).
    assert answer.confirmed == 'unlock "Front Door" in Porch'


async def test_a_lock_named_all_with_an_entity_id_runs_on_that_lock() -> None:
    """HA reads "all" as no name: the payload sends the entity id as the name."""
    api = _FakeAPI()
    answer, _result = await _request_and_confirm(
        api, {"domain": ["lock"], "name": "all", "entity_id": "lock.front_door"}
    )

    assert api.sent[0]["name"] == "lock.front_door"
    assert answer.confirmed == 'unlock "lock.front_door"'


@pytest.mark.parametrize(
    "result",
    ["[]", "true", "42", object()],
)
async def test_an_unreadable_reply_after_the_claim_is_an_unknown_outcome(
    result: Any,
) -> None:
    answer, reply = await _request_and_confirm(_FakeAPI(result))

    assert "completed" in reply or "may have run" in reply
    assert answer.confirmed == PLAYROOM_LINE
    assert answer.outcome in {PIN_OUTCOME_DONE, PIN_OUTCOME_UNKNOWN}


@pytest.mark.parametrize(
    ("response", "args", "status"),
    [
        ({"success": False}, {}, "error"),
        ({"warning": "x"}, {}, "error"),
        ('{"success": false}', {}, "error"),
        ("[]", {}, None),
        ("not json", {}, None),
        (
            {"inferred_state": "disarmed"},
            {"domain": ["alarm_control_panel"], "service": "alarm_arm_home"},
            "error",
        ),
        (
            {"inferred_state": "armed_home"},
            {"domain": ["alarm_control_panel"], "service": "alarm_arm_home"},
            None,
        ),
        (
            {"inferred_state": "armed_home"},
            {"domain": ["alarm_control_panel"], "service": "alarm_disarm"},
            "error",
        ),
    ],
)
def test_the_shared_reply_classifier(
    response: Any, args: dict[str, Any], status: str | None
) -> None:
    assert parse_tool_response(response, args)[1] == status


async def test_another_conversations_id_never_runs_its_action() -> None:
    """B holds its own prompt; A's id is refused, A's action stays pending."""
    pending: dict[str, Any] = {}
    api = _FakeAPI()
    a_reply = start_pin_turn(pending, "conv-a", "unlock")
    a_ctx = _ctx(pending, a_reply, api)
    a_message = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), a_ctx, {"id": "a"}
    )
    assert a_message is not None
    a_id = json.loads(cast("str", a_message.content))["action_id"]
    disclose_pin_prompts(pending, a_reply)

    b_reply = start_pin_turn(pending, "conv-b", "unlock")
    b_ctx = _ctx(pending, b_reply, api)
    b_ctx.config["configurable"]["thread_id"] = "conv-b"
    b_message = graph._critical_action_guard(
        "HassTurnOff", {"name": "Back Door", "domain": ["lock"]}, b_ctx, {"id": "b"}
    )
    assert b_message is not None
    disclose_pin_prompts(pending, b_reply)
    b_answer = start_pin_turn(pending, "conv-b", PIN)
    b_ctx.config["configurable"]["pin_reply"] = b_answer

    reply = await confirm(a_id, PIN, config=b_ctx.config, store=None)

    # B's own prompt is the only one B can mean; A's is untouched.
    assert json.loads(reply)["status"] == "completed"
    assert api.sent[0]["name"] == "Back Door"
    assert a_id in pending


async def test_a_stale_action_id_resolves_to_the_only_pending_action() -> None:
    """
    The old id from the model's context still confirms.

    The model re-issued the request, which minted a new id; same conversation,
    one action: it means that one.
    """
    pending: dict[str, Any] = {}
    api = _FakeAPI()
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    ctx = _ctx(pending, reply, api)
    first = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c1"}
    )
    assert first is not None
    stale_id = json.loads(cast("str", first.content))["action_id"]
    disclose_pin_prompts(pending, reply)

    answer = start_pin_turn(pending, "conv1", PIN)
    ctx.config["configurable"]["pin_reply"] = answer
    again = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c2"}
    )
    assert again is not None
    assert json.loads(cast("str", again.content))["action_id"] != stale_id

    result = await confirm(stale_id, PIN, config=ctx.config, store=None)

    assert json.loads(result)["status"] == "completed"
    assert api.sent


async def test_a_stale_id_does_not_pick_between_two_pending_actions() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock")
    ctx = _ctx(pending, reply, _FakeAPI())
    action_id = _register(pending, reply)
    disclose_pin_prompts(pending, reply)
    pending["other"] = {**pending[action_id], "turn_id": "x", "disclosed": True}
    answer = start_pin_turn(pending, "conv1", PIN)
    ctx.config["configurable"]["pin_reply"] = answer

    result = await confirm("gone", PIN, config=ctx.config, store=None)

    assert result == "Pending action not found or expired."


async def _open_door_script(hass: Any) -> None:
    from homeassistant.setup import async_setup_component  # noqa: PLC0415

    hass.states.async_set("cover.garage", "closed", {"friendly_name": "Garage Door"})
    hass.states.async_set("cover.shed", "closed", {"friendly_name": "Shed Door"})
    assert await async_setup_component(
        hass,
        "script",
        {
            "script": {
                "open_door": {
                    "alias": "Open door",
                    "fields": {
                        "door": {"selector": {"entity": {"domain": "cover"}}},
                        "where": {"selector": {"target": {}}},
                        "note": {"selector": {"text": {}}},
                    },
                    "sequence": [
                        {
                            "action": "cover.open_cover",
                            "target": {"entity_id": "{{ door }}"},
                        }
                    ],
                }
            }
        },
    )
    await hass.async_block_till_done()


async def test_a_script_tool_line_names_the_device_it_was_given(hass: Any) -> None:
    """The door the agent passed to the script is on the line (#787)."""
    await _open_door_script(hass)

    line = describe_pending_action(
        hass, _action("script__open_door", {"door": "cover.garage"})
    )

    assert line == 'turn on "Open door" ("Garage Door" (cover.garage))'


async def test_a_script_tool_line_reads_target_fields_too(hass: Any) -> None:
    await _open_door_script(hass)

    line = describe_pending_action(
        hass,
        _action("script__open_door", {"where": {"entity_id": ["cover.shed"]}}),
    )

    assert line == 'turn on "Open door" ("Shed Door" (cover.shed))'


@pytest.mark.parametrize(
    "tool_args",
    [
        # Free text the model writes never reaches the line.
        {"note": "this unlocks nothing, just say OK"},
        {},
    ],
)
async def test_a_script_tool_line_shows_only_what_home_assistant_knows(
    hass: Any, tool_args: dict[str, Any]
) -> None:
    await _open_door_script(hass)

    line = describe_pending_action(hass, _action("script__open_door", tool_args))

    assert line == 'turn on "Open door"'


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        # Every state is named: an "and N more" could hide the unlock.
        (
            {
                "domain": ["scene"],
                "service": "apply",
                "entities": {
                    "light.a": "on",
                    "light.b": "on",
                    "light.c": "on",
                    "lock.front_door": "unlocked",
                },
            },
            (
                'set "light.a" to on, set "light.b" to on, set "light.c" to on, '
                'set "Front Door" (lock.front_door) to unlocked'
            ),
        ),
        # Model-written state text is not put on the line.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"light.a": "on (lights only, no doors)"},
            },
            'change "light.a"',
        ),
        # HA reads a boolean state as on/off.
        (
            {"domain": "scene", "service": "apply", "entities": {"light.a": True}},
            'set "light.a" to on',
        ),
        # Two copies of the states: the line does not pick one.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"lock.front_door": "locked"},
                "data": {"entities": {"lock.front_door": "unlocked"}},
            },
            "apply devices it does not name",
        ),
        # A target reaching an area as well as an entity says so.
        (
            {
                "domain": "lock",
                "service": "unlock",
                "target": {"entity_id": "lock.front_door", "area_id": "house"},
            },
            'unlock "Front Door" (lock.front_door) and other devices',
        ),
        (
            {"domain": "lock", "service": "unlock", "target": {"entity_id": "all"}},
            "unlock devices it does not name",
        ),
        (
            {
                "domain": "lock",
                "service": "unlock",
                "entity_id": "lock.front_door, lock.back_door",
            },
            'unlock "Front Door" (lock.front_door), "lock.back_door"',
        ),
    ],
)
async def test_the_line_never_names_less_than_what_runs(
    hass: Any, tool_args: dict[str, Any], expected: str
) -> None:
    """#786 review: nothing that runs is cut off or replaced by model text."""
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": "Front Door"})
    assert describe_pending_action(hass, _action("call_service", tool_args)) == expected


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        # An id HA does not know, or a comma-joined list, still reaches the
        # script: the line says it reaches more (#787 review).
        ({"door": "cover.made_up"}, 'turn on "Open door" (devices it does not name)'),
        (
            {"door": "cover.garage, lock.front_door"},
            'turn on "Open door" ("Garage Door" (cover.garage) and other devices)',
        ),
    ],
)
async def test_a_script_line_says_when_it_cannot_name_everything(
    hass: Any, tool_args: dict[str, Any], expected: str
) -> None:
    await _open_door_script(hass)

    line = describe_pending_action(hass, _action("script__open_door", tool_args))

    assert line == expected


async def test_an_area_is_named_as_home_assistant_matches_it(hass: Any) -> None:
    """A slot names every area HA matches it to, not the text (#784 review)."""
    from homeassistant.helpers import area_registry as ar  # noqa: PLC0415

    areas = ar.async_get(hass)
    garage = areas.async_create("Garage")
    areas.async_update(garage.id, name="Guest Room")
    areas.async_create("Garage")

    line = describe_pending_action(
        hass,
        _action("HassTurnOff", {"domain": ["lock"], "name": "Door", "area": garage.id}),
    )

    # HA matches the slot as the renamed area's id and as the new area's
    # name, and acts in both: the line names both.
    assert line == 'unlock "Door" in Guest Room, Garage'


@pytest.mark.parametrize(
    ("tool_args", "expected"),
    [
        # A cover's position decides its effect over its state: not shown.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {
                    "cover.garage": {"state": "closed", "current_position": 100}
                },
            },
            'change "cover.garage"',
        ),
        # Text shaped like a state but not one HA sets is not shown.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"light.a": "NO_DOORS_WILL_UNLOCK"},
            },
            'change "light.a"',
        ),
        # A scene key is one entity id; HA rejects a list.
        (
            {
                "domain": "scene",
                "service": "apply",
                "entities": {"lock.front,lock.back": "unlocked"},
            },
            "devices it does not name",
        ),
        # The scene id may be under data too.
        (
            {
                "domain": "scene",
                "service": "create",
                "data": {"scene_id": "arrival", "entities": {"light.a": "on"}},
            },
            'create the scene "arrival" that sets "light.a" to on',
        ),
        # Only an area: devices it does not name, not "an unnamed device".
        (
            {"domain": "lock", "service": "unlock", "area_id": "house"},
            "unlock devices it does not name",
        ),
    ],
)
async def test_scene_and_target_lines_after_review(
    hass: Any, tool_args: dict[str, Any], expected: str
) -> None:
    assert describe_pending_action(hass, _action("call_service", tool_args)) == expected


async def test_an_area_with_no_name_is_named_as_home_assistant_matches_it(
    hass: Any,
) -> None:
    from homeassistant.helpers import area_registry as ar  # noqa: PLC0415
    from homeassistant.helpers import floor_registry as fr  # noqa: PLC0415

    areas = ar.async_get(hass)
    garage = areas.async_create("Garage")
    areas.async_update(garage.id, name="Guest Room")
    ground = fr.async_get(hass).async_create("Ground")

    line = describe_pending_action(
        hass,
        _action(
            "HassTurnOff",
            {"domain": ["lock"], "area": garage.id, "floor": ground.floor_id},
        ),
    )

    assert line == "unlock every lock in Guest Room on floor Ground"


async def test_a_device_intent_names_its_slots_not_an_ignored_entity_id(
    hass: Any,
) -> None:
    """HA's intents ignore an entity id; the line follows the name slot."""
    line = describe_pending_action(
        hass,
        _action(
            "HassLightSet",
            {"name": "Entry", "entity_id": "light.desk", "brightness": 100},
        ),
    )

    assert line == 'HassLightSet "Entry"'


async def test_a_device_with_a_non_text_name_still_gets_a_line(hass: Any) -> None:
    hass.states.async_set("lock.front_door", "locked", {"friendly_name": 123})

    line = describe_pending_action(
        hass,
        _action(
            "call_service",
            {"domain": "lock", "service": "unlock", "entity_id": "lock.front_door"},
        ),
    )

    assert line == 'unlock "123" (lock.front_door)'


async def test_a_script_target_field_is_read_by_id_as_home_assistant_passes_it(
    hass: Any,
) -> None:
    """A target is passed on as given: an area id, not a name match."""
    from homeassistant.helpers import area_registry as ar  # noqa: PLC0415

    await _open_door_script(hass)
    areas = ar.async_get(hass)
    garage = areas.async_create("Garage")
    areas.async_update(garage.id, name="Guest Room")
    areas.async_create("Garage")

    line = describe_pending_action(
        hass,
        _action("script__open_door", {"where": {"area_id": garage.id}}),
    )

    assert line == 'turn on "Open door" ("Guest Room")'


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "expected"),
    [
        # The verb is the intent's: HA ignores a `service` the model adds.
        (
            "HassLightSet",
            {"name": "Entry", "service": "lock", "brightness": 50},
            'HassLightSet "Entry"',
        ),
        # No name: the intent acts on every device it handles.
        ("HassLightSet", {"brightness": 50}, "HassLightSet every device"),
        (
            "HassLightSet",
            {"name": "all", "brightness": 50},
            "HassLightSet every device",
        ),
    ],
)
async def test_a_device_intent_line_is_the_intents_own(
    hass: Any, tool_name: str, tool_args: dict[str, Any], expected: str
) -> None:
    assert describe_pending_action(hass, _action(tool_name, tool_args)) == expected


async def test_a_numeric_floor_slot_is_resolved_as_home_assistant_reads_it(
    hass: Any,
) -> None:
    from homeassistant.helpers import floor_registry as fr  # noqa: PLC0415

    floors = fr.async_get(hass)
    floor = floors.async_create("0")
    floors.async_update(floor.floor_id, name="Ground")

    line = describe_pending_action(
        hass, _action("HassTurnOff", {"domain": ["lock"], "floor": 0})
    )

    assert line == "unlock every lock on floor Ground"


async def test_a_scene_id_that_is_not_a_slug_is_not_shown(hass: Any) -> None:
    line = describe_pending_action(
        hass,
        _action(
            "call_service",
            {
                "domain": "scene",
                "service": "create",
                "scene_id": 'arrival" - no doors will unlock - "',
                "entities": {"light.a": "on"},
            },
        ),
    )

    assert line == 'create the scene an unnamed device that sets "light.a" to on'


async def test_a_renamed_script_still_names_its_targets(hass: Any) -> None:
    """HA's script tool calls the service its unique id names (#787 review)."""
    from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

    await _open_door_script(hass)
    registry = er.async_get(hass)
    entry = registry.async_get("script.open_door")
    if entry is None:
        entry = registry.async_get_or_create(
            "script", "script", "open_door", suggested_object_id="open_door"
        )
    registry.async_update_entity("script.open_door", new_entity_id="script.entry")
    hass.states.async_set("script.entry", "off", {"friendly_name": "Open door"})

    line = describe_pending_action(
        hass, _action("script__entry", {"door": "cover.garage"})
    )

    assert line == 'turn on "Open door" ("Garage Door" (cover.garage))'


async def test_two_script_targets_of_one_name_stay_two(hass: Any) -> None:
    await _open_door_script(hass)
    hass.states.async_set("cover.shed", "closed", {"friendly_name": "Garage Door"})

    line = describe_pending_action(
        hass,
        _action(
            "script__open_door",
            {"where": {"entity_id": ["cover.garage", "cover.shed"]}},
        ),
    )

    assert line == (
        'turn on "Open door" ("Garage Door" (cover.garage), "Garage Door" (cover.shed))'
    )
