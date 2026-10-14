# ruff: noqa: S101
"""A PIN confirmation is accepted only from the user's current message."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers import llm
from langchain_core.messages import ToolMessage

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent.helpers import (
    PIN_AWAITING_USER_REPLY,
    PinReply,
    pin_in_text,
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


@dataclass(slots=True)
class _ToolResult:
    """Same shape as HA 2026.10's ``llm.ToolResult``."""

    data: Any
    error: bool = False


@pytest.fixture
def tool_result_cls(monkeypatch: pytest.MonkeyPatch) -> type:
    """Return ``llm.ToolResult``, installing a stand-in on HA < 2026.10."""
    existing = getattr(llm, "ToolResult", None)
    if existing is not None:
        return existing
    monkeypatch.setattr(llm, "ToolResult", _ToolResult, raising=False)
    return _ToolResult


class _FakeAPI:
    def __init__(self, result: Any = None) -> None:
        self.result = {"success": True} if result is None else result
        self.calls: list[str] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.calls.append(tool_input.tool_name)
        # Yield so a second confirmation in the same batch runs meanwhile.
        await asyncio.sleep(0)
        return self.result


class _Turns:
    """One conversation: the graph context and a config per user message."""

    def __init__(self, api: _FakeAPI, thread_id: str = "conv1") -> None:
        hashed, salt = hash_pin(PIN)
        self.pending: dict[str, Any] = {}
        self.config: dict[str, Any] = {
            "configurable": {
                "options": {
                    CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                    CONF_CRITICAL_ACTION_PIN_SALT: salt,
                },
                "pending_actions": self.pending,
                "user_id": "robot",
                "thread_id": thread_id,
                "pin_reply": PinReply("unlock the front door"),
                "ha_llm_api": api,
                "hass": None,
            }
        }
        self.ctx = graph.ToolExecutionContext(
            hass=MagicMock(),
            store=MagicMock(),
            config=cast("Any", self.config),
            ha_llm_api=api,
            critical_actions=[],
            pin_enabled=True,
            pin_hash=hashed,
            pin_salt=salt,
            pending_actions=self.pending,
            state=cast("Any", {}),
        )

    def say(self, text: str) -> None:
        """Start the next turn with the user's message."""
        self.config["configurable"]["pin_reply"] = PinReply(text)

    def request_unlock(self) -> str:
        """Call the critical tool and return the action id of its PIN prompt."""
        message = graph._critical_action_guard(
            "HassTurnOff", dict(UNLOCK_ARGS), self.ctx, {"id": "call1"}
        )
        assert message is not None
        payload = json.loads(cast("str", message.content))
        assert payload["status"] == "requires_pin"
        return payload["action_id"]

    async def confirm(self, action_id: str, pin: str = PIN) -> dict[str, Any] | str:
        reply = await confirm(action_id, pin, config=self.config, store=None)
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return reply


async def test_a_pin_the_user_did_not_write_is_refused_without_an_attempt() -> None:
    """The model confirming in the request turn supplies a PIN the user never gave."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == []
    assert turns.pending[action_id]["attempts"] == 0


async def test_the_users_pin_reply_confirms() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"
    assert api.calls == ["HassTurnOff"]


async def test_a_pin_from_history_does_not_confirm_on_an_unrelated_reply() -> None:
    """
    The user gave the PIN earlier, then answered the new prompt with "never mind".

    The PIN is still in the conversation, but not in this message.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    turns.say(PIN)
    first = turns.request_unlock()
    await turns.confirm(first)

    turns.say("lock it again")
    second = turns.request_unlock()
    turns.say("never mind")
    reply = await turns.confirm(second)

    assert isinstance(reply, dict)
    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == ["HassTurnOff"]


async def test_reissuing_the_tool_in_the_pin_reply_still_completes() -> None:
    """A model that calls the tool again in the reply turn is not locked out."""
    api = _FakeAPI()
    turns = _Turns(api)
    turns.request_unlock()

    turns.say(f"it's {PIN}")
    reissued = turns.request_unlock()
    reply = await turns.confirm(reissued)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"
    assert api.calls == ["HassTurnOff"]


async def test_one_pin_message_runs_one_action_even_after_a_failure(
    tool_result_cls: type,
) -> None:
    """
    A failed action cannot be re-run from the same message by re-issuing it.

    A tool that half-acted before reporting failure would otherwise act twice.
    """
    api = _FakeAPI(tool_result_cls(data={"error": "Lock jammed"}, error=True))
    turns = _Turns(api)
    first = turns.request_unlock()
    turns.say(PIN)
    failed = await turns.confirm(first)
    assert isinstance(failed, str)
    assert "Lock jammed" in failed

    reissued = turns.request_unlock()
    reply = await turns.confirm(reissued)

    assert isinstance(reply, dict)
    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert "already confirmed" in reply["reason"]
    assert api.calls == ["HassTurnOff"]


async def test_two_actions_confirmed_together_from_one_message_run_once() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    first = turns.request_unlock()
    second = turns.request_unlock()

    turns.say(PIN)
    replies = await asyncio.gather(turns.confirm(first), turns.confirm(second))

    assert api.calls == ["HassTurnOff"]
    assert sum(isinstance(r, dict) and r["status"] == "completed" for r in replies) == 1


async def test_a_pin_in_the_request_itself_confirms() -> None:
    """The user (or an automation) wrote the PIN in the request; it is theirs."""
    api = _FakeAPI()
    turns = _Turns(api)
    turns.say(f"unlock the front door, PIN {PIN}")
    action_id = turns.request_unlock()

    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_another_conversation_cannot_confirm_the_action() -> None:
    """Satellites and automations share one user, so the conversation decides."""
    api = _FakeAPI()
    satellite_a = _Turns(api, thread_id="conv-a")
    action_id = satellite_a.request_unlock()

    satellite_b = _Turns(api, thread_id="conv-b")
    satellite_b.pending.update(satellite_a.pending)
    satellite_b.say(PIN)
    by_id = await satellite_b.confirm(action_id)
    without_id = await satellite_b.confirm("")

    assert by_id == "Pending action not found or expired."
    assert without_id == "Pending action not found or expired."
    assert api.calls == []


async def test_without_an_id_the_conversations_only_action_is_confirmed() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    turns.request_unlock()

    turns.say(PIN)
    reply = await turns.confirm("")

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_without_an_id_another_conversations_action_is_ignored() -> None:
    """Satellite A's pending unlock does not make B's id-less confirm ambiguous."""
    api = _FakeAPI()
    satellite_a = _Turns(api, thread_id="conv-a")
    satellite_a.request_unlock()
    satellite_b = _Turns(api, thread_id="conv-b")
    satellite_b.pending.update(satellite_a.pending)
    own = satellite_b.request_unlock()

    satellite_b.say(PIN)
    reply = await satellite_b.confirm("")

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"
    assert reply["action_id"] == own


async def test_a_confirmation_without_the_users_message_is_refused() -> None:
    """With no message to check the PIN against, nothing is confirmed."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    del turns.config["configurable"]["pin_reply"]

    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == []


async def test_wrong_user_is_reported_before_the_missing_pin() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.config["configurable"]["user_id"] = "someone_else"

    reply = await turns.confirm(action_id)

    assert isinstance(reply, str)
    assert "different user" in reply


async def test_an_unexpected_error_still_uses_up_the_confirmation() -> None:
    """The action is claimed before the call, so it is never run a second time."""

    class _BoomAPI(_FakeAPI):
        async def async_call_tool(self, tool_input: Any) -> Any:
            self.calls.append(tool_input.tool_name)
            msg = "integration bug"
            raise RuntimeError(msg)

    api = _BoomAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(PIN)

    with pytest.raises(RuntimeError):
        await turns.confirm(action_id)

    assert action_id not in turns.pending
    assert api.calls == ["HassTurnOff"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", True),
        ("my PIN is 1234.", True),
        ("1, 2, 3, 4", True),
        ("1-2-3-4", True),
        ("one two three four", True),
        ("One, two, three, four.", True),
        ("jedna dva tři čtyři", True),
        ("jedna dve tri ctyri", True),
        ("unlock 2 doors, PIN 1234", True),
        ("door 1, 1234", True),
        ("PIN 1 2 3 4, 2 doors please", True),
        ("My PIN is one two three four. One door only.", True),
        ("\u0661\u0662\u0663\u0664", True),  # Arabic-Indic digits
        ("\uff11\uff12\uff13\uff14", True),  # full-width digits
        ("51234", False),
        ("12345", False),
        ("1 2 3", False),
        ("12 34", False),
        ("never mind", False),
        ("", False),
        # Ordinary numbers in a message are not a PIN (review of PR #769).
        ("Cancel the unlock. The invoice total is 12.34 dollars.", False),
        ("Do not unlock; the serial is abc1234xyz.", False),
        ("Use tracking number 1234-5678.", False),
        ("call 555-1234", False),
        ("at 12.34 please", False),
    ],
)
def test_pin_in_text(text: str, *, expected: bool) -> None:
    assert pin_in_text(PIN, text) is expected


@pytest.mark.parametrize(
    ("pin", "text"),
    [
        ("2024", "remind me on 2024-10-14"),
        ("1014", "on 10-14 at 9"),
        ("1230", "at 12.30 please"),
        ("2130", "set the thermostat to 21.30"),
        ("2021", "set bedroom to 20, 21"),
        ("2026", "Alert at 2026-10-14T12:34:56"),
    ],
)
def test_dates_times_and_lists_are_not_a_pin(pin: str, text: str) -> None:
    assert pin_in_text(pin, text) is False


def test_the_users_message_is_not_in_the_repr() -> None:
    assert PIN not in repr(PinReply(f"my PIN is {PIN}"))


async def test_a_correct_pin_refused_for_not_being_in_the_message_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A voice user whose PIN is transcribed unreadably is not left in silence."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say("twelve thirty-four")

    with caplog.at_level(logging.WARNING):
        await turns.confirm(action_id)

    assert "Refused a correct critical-action PIN" in caplog.text
    assert PIN not in caplog.text


def test_a_refusal_then_completion_ends_the_pin_flow() -> None:
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
    completed = ToolMessage(
        content=json.dumps({"status": "completed", "action_id": "a1"}),
        tool_call_id="tc3",
        name="confirm_sensitive_action",
    )

    assert graph._has_pending_pin_confirmation([prompt, refusal]) is True
    assert graph._has_pending_pin_confirmation([prompt, refusal, completed]) is False
