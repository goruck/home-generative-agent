# ruff: noqa: S101
"""A PIN confirmation is accepted only from a turn after the PIN prompt."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.helpers import (
    PIN_AWAITING_USER_REPLY,
)
from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin

PIN = "1234"
UNLOCK_ARGS: dict[str, Any] = {"domain": ["lock"], "name": "Front Door"}

confirm = cast("Any", confirm_sensitive_action).coroutine


class _FakeAPI:
    def __init__(self, *, error: bool = False) -> None:
        self.error = error
        self.calls: list[str] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.calls.append(tool_input.tool_name)
        if self.error:
            return {"error": "Lock jammed"}
        return {"success": True}


def _setup(api: _FakeAPI) -> tuple[dict[str, Any], graph.ToolExecutionContext]:
    """Return a turn's config and the graph context that shares its store."""
    hashed, salt = hash_pin(PIN)
    pending: dict[str, Any] = {}
    config: dict[str, Any] = {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": pending,
            "user_id": "u1",
            "turn_id": "turn-1",
            "ha_llm_api": api,
            "hass": None,
        }
    }
    ctx = graph.ToolExecutionContext(
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
    return config, ctx


def _request_unlock(ctx: graph.ToolExecutionContext) -> str:
    """Call the critical tool and return the action id of its PIN prompt."""
    message = graph._critical_action_guard(
        "HassTurnOff", dict(UNLOCK_ARGS), ctx, {"id": "call1"}
    )
    assert message is not None
    payload = json.loads(cast("str", message.content))
    assert payload["status"] == "requires_pin"
    return payload["action_id"]


async def test_confirm_in_the_request_turn_is_refused_without_an_attempt() -> None:
    """A PIN in the same turn as the prompt came from the model, not the user."""
    api = _FakeAPI()
    config, ctx = _setup(api)
    action_id = _request_unlock(ctx)

    reply = json.loads(await confirm(action_id, PIN, config=config, store=None))

    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert reply["action_id"] == action_id
    assert api.calls == []
    pending = config["configurable"]["pending_actions"][action_id]
    assert pending["attempts"] == 0


async def test_the_users_reply_in_the_next_turn_confirms() -> None:
    """After a same-turn refusal, the user's PIN in the next turn still works."""
    api = _FakeAPI()
    config, ctx = _setup(api)
    action_id = _request_unlock(ctx)
    await confirm(action_id, PIN, config=config, store=None)

    config["configurable"]["turn_id"] = "turn-2"
    reply = json.loads(await confirm(action_id, PIN, config=config, store=None))

    assert reply["status"] == "completed"
    assert api.calls == ["HassTurnOff"]


async def test_reissued_action_is_not_confirmed_with_the_pin_from_history() -> None:
    """
    A failed action cannot be re-run in the same turn by re-issuing the tool.

    The user confirmed in turn 2 and the device failed. The model then calls the
    tool again and confirms the new prompt with the PIN already in the
    conversation; that second confirmation must not reach the device.
    """
    api = _FakeAPI(error=True)
    config, ctx = _setup(api)
    first_id = _request_unlock(ctx)
    config["configurable"]["turn_id"] = "turn-2"
    await confirm(first_id, PIN, config=config, store=None)

    second_id = _request_unlock(ctx)
    reply = json.loads(await confirm(second_id, PIN, config=config, store=None))

    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == ["HassTurnOff"]


async def test_confirm_without_a_turn_id_is_refused() -> None:
    """With no turn id the PIN's source cannot be told, so it is refused."""
    api = _FakeAPI()
    config, ctx = _setup(api)
    action_id = _request_unlock(ctx)
    del config["configurable"]["turn_id"]

    reply = json.loads(await confirm(action_id, PIN, config=config, store=None))

    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == []


def test_a_same_turn_refusal_keeps_the_pin_flow_pending() -> None:
    """The confirm tool is still offered when the user replies with the PIN."""
    prompt = ToolMessage(
        content=json.dumps({"status": "requires_pin", "action_id": "a1"}),
        tool_call_id="tc1",
        name="HassTurnOff",
    )
    refusal = ToolMessage(
        content=json.dumps({"status": PIN_AWAITING_USER_REPLY, "action_id": "a1"}),
        tool_call_id="tc2",
        name="confirm_sensitive_action",
    )

    assert graph._has_pending_pin_confirmation([prompt, refusal]) is True
