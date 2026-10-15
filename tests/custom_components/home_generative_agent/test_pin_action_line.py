# ruff: noqa: S101
"""
The deterministic line that names the action a PIN confirms.

Field test 2026-10-15: "Block the playroom, block" (speech-to-text for "lock
the playroom lock") reached the model without HassTurnOn, so it called
HassTurnOff, which is *unlock* for a lock. The model asked for the PIN "to
lock the playroom door" and, after the PIN, said "now locked". The door was
unlocked. What the user is told the PIN confirms must come from the action
that is held, not from the model's prose.
"""

from __future__ import annotations

import json
import types
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.helpers import (
    PinReply,
    keep_pin_prompts,
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
        ("HassTurnOff", PLAYROOM, "unlock Playroom Door Lock"),
        ("HassTurnOn", PLAYROOM, "lock Playroom Door Lock"),
        ("intent__HassTurnOff", PLAYROOM, "unlock Playroom Door Lock"),
        # The lock is recognized by name too, as the gate does.
        ("HassTurnOff", {"name": "Front Door Lock"}, "unlock Front Door Lock"),
        # A service the model added does not change what HA runs.
        ("HassTurnOff", {**PLAYROOM, "service": "lock"}, "unlock Playroom Door Lock"),
        # Without a name, the entity it runs on.
        (
            "HassTurnOff",
            {"domain": ["lock"], "entity_id": "lock.playroom"},
            "unlock lock.playroom",
        ),
        # Not a lock: the on/off intent by its own words.
        ("HassTurnOn", {"name": "Garage Heater"}, "turn on Garage Heater"),
        ("HassTurnOff", {"name": "Garage Heater"}, "turn off Garage Heater"),
        # Any other critical call: its service, readable.
        (
            "call_service",
            {"domain": "cover", "service": "open_cover", "entity_id": ["cover.garage"]},
            "open cover cover.garage",
        ),
    ],
)
def test_the_action_is_described_from_what_runs(
    tool_name: str, tool_args: dict[str, Any], expected: str
) -> None:
    assert describe_pending_action(_hass(), _action(tool_name, tool_args)) == expected


def test_an_automation_is_described_by_its_alias() -> None:
    action = _action(
        "add_automation",
        {},
        automation_config={"id": "1", "alias": "Unlock at night", "actions": []},
    )
    assert describe_pending_action(_hass(), action) == (
        'add the automation "Unlock at night"'
    )


def test_a_target_without_a_name_still_reads_as_a_sentence() -> None:
    assert describe_pending_action(_hass(), _action("HassTurnOff", {"domain": "lock"}))
    assert describe_pending_action(_hass(), _action("HassTurnOff", {})) == (
        "turn off an unnamed device"
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


def test_the_action_is_described_in_czech() -> None:
    assert (
        describe_pending_action(_hass("cs"), _action("HassTurnOff", PLAYROOM))
        == "odemknout Playroom Door Lock"
    )


# ---------------------------------------------------------------------------
# prompts_awaiting_reply
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
    assert payload["action"] == "unlock Playroom Door Lock"


class _FakeAPI:
    async def async_call_tool(self, tool_input: Any) -> Any:
        return {"success": True}


async def test_a_confirmed_action_is_recorded_on_the_reply() -> None:
    pending: dict[str, Any] = {}
    api = _FakeAPI()
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    ctx = _ctx(pending, reply, api)
    message = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c1"}
    )
    assert message is not None
    action_id = json.loads(cast("str", message.content))["action_id"]

    answer = start_pin_turn(pending, "conv1", PIN)
    ctx.config["configurable"]["pin_reply"] = answer
    result = json.loads(await confirm(action_id, PIN, config=ctx.config, store=None))

    assert result["status"] == "completed"
    assert answer.confirmed == "unlock Playroom Door Lock"


async def test_a_wrong_pin_records_nothing() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv1", "unlock the playroom door")
    ctx = _ctx(pending, reply, _FakeAPI())
    message = graph._critical_action_guard(
        "HassTurnOff", dict(PLAYROOM), ctx, {"id": "c1"}
    )
    assert message is not None
    action_id = json.loads(cast("str", message.content))["action_id"]

    answer = start_pin_turn(pending, "conv1", "9999")
    ctx.config["configurable"]["pin_reply"] = answer
    await confirm(action_id, "9999", config=ctx.config, store=None)

    assert answer.confirmed == ""
