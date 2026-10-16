# ruff: noqa: S101
"""A PIN confirmation is accepted only from the user's current message."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from langchain_core.messages import ToolMessage

from custom_components.home_generative_agent.agent import graph
from custom_components.home_generative_agent.agent import tools as tools_module
from custom_components.home_generative_agent.agent.helpers import (
    PENDING_ACTION_TTL,
    PIN_AWAITING_USER_REPLY,
    PIN_REFUSAL_ENDS_TURN,
    PinReply,
    disclose_pin_prompts,
    keep_pin_prompts,
    normalize_pin,
    pin_digits,
    pin_in_text,
    register_pending_action,
    start_pin_turn,
    thread_has_pending_action,
)
from custom_components.home_generative_agent.agent.pin_messages import pin_msg
from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentOptionsFlow,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin, verify_pin

PIN = "1234"
UNLOCK_ARGS: dict[str, Any] = {"domain": ["lock"], "name": "Front Door"}
BACK_DOOR_ARGS: dict[str, Any] = {"domain": ["lock"], "name": "Back Door"}

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
        self.sent_names: list[Any] = []

    async def async_call_tool(self, tool_input: Any) -> Any:
        self.calls.append(tool_input.tool_name)
        self.sent_names.append(tool_input.tool_args.get("name"))
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
                "pin_reply": start_pin_turn(
                    self.pending, thread_id, "unlock the front door"
                ),
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
        configurable = self.config["configurable"]
        configurable["pin_reply"] = start_pin_turn(
            self.pending, configurable["thread_id"], text
        )

    def request_unlock(self, args: dict[str, Any] | None = None) -> str:
        """Call the critical tool and return the action id of its PIN prompt."""
        message = graph._critical_action_guard(
            "HassTurnOff", dict(args or UNLOCK_ARGS), self.ctx, {"id": "call1"}
        )
        assert message is not None
        payload = json.loads(cast("str", message.content))
        assert payload["status"] == "requires_pin"
        # The turn ends by naming the prompt, as conversation.py does; a
        # prompt whose line was never delivered cannot be answered.
        disclose_pin_prompts(self.pending, self.config["configurable"]["pin_reply"])
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


async def test_only_the_request_turn_refusal_ends_the_turn() -> None:
    """
    Before the user was asked, the reply asks in fixed words and the turn ends.

    In the reply turn the model is told what to say (say the digits one by
    one; the PIN was already used), so the turn goes on. So does a turn whose
    PIN already ran an action and whose model then re-issued the request.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    before_the_prompt = await turns.confirm(action_id)
    assert isinstance(before_the_prompt, dict)
    assert before_the_prompt[PIN_REFUSAL_ENDS_TURN] is True

    turns.say("yes, go ahead")
    not_in_message = await turns.confirm(action_id)
    assert isinstance(not_in_message, dict)
    assert not_in_message["status"] == PIN_AWAITING_USER_REPLY
    assert PIN_REFUSAL_ENDS_TURN not in not_in_message

    turns.say(PIN)
    done = await turns.confirm(action_id)
    assert isinstance(done, dict)
    assert done["status"] == "completed"
    reissued = turns.request_unlock()
    used = await turns.confirm(reissued)
    assert isinstance(used, dict)
    assert used["status"] == PIN_AWAITING_USER_REPLY
    assert PIN_REFUSAL_ENDS_TURN not in used
    assert api.calls == ["HassTurnOff"]


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
    first = turns.request_unlock()
    turns.say(PIN)
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
    assert api.calls == ["HassTurnOff"]


async def test_a_second_protected_request_in_one_turn_is_refused() -> None:
    """
    One protected request at a time: the second is refused, not silently kept.

    Both would otherwise have prompts out while only one could be confirmed.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    first = turns.request_unlock()
    second = graph._critical_action_guard(
        "HassTurnOff", dict(BACK_DOOR_ARGS), turns.ctx, {"id": "call2"}
    )

    assert second is not None
    assert second.status == "error"
    assert "already waiting" in cast("str", second.content)
    assert list(turns.pending) == [first]


async def test_confirmations_in_one_batch_run_the_action_once() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(PIN)
    replies = await asyncio.gather(turns.confirm(action_id), turns.confirm(action_id))

    assert api.calls == ["HassTurnOff"]
    assert sum(isinstance(r, dict) and r["status"] == "completed" for r in replies) == 1


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
    """
    The action is claimed before the call, so it is never run a second time.

    Any error after the claim gets an honest "may have run" reply instead of
    escaping the turn: the device may already have acted.
    """

    class _BoomAPI(_FakeAPI):
        async def async_call_tool(self, tool_input: Any) -> Any:
            self.calls.append(tool_input.tool_name)
            msg = "request failed: https://device/?token=SECRET"
            raise RuntimeError(msg)

    api = _BoomAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(PIN)

    reply = await turns.confirm(action_id)

    assert isinstance(reply, str)
    assert "may have run" in reply
    assert "RuntimeError" in reply
    assert "SECRET" not in reply  # the error's text stays out of the model
    assert action_id not in turns.pending
    assert api.calls == ["HassTurnOff"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", True),
        ("my PIN is 1234.", True),
        ("1,234", True),
        ("12 34", True),
        ("1, 2, 3, 4", True),
        ("1,2,3,4", True),
        ("1, 2, 3 and 4", True),
        ("1-2-3-4", True),
        ("1 - 2 - 3 - 4", True),
        ("1\u20132\u20133\u20134", True),  # en dashes
        ("1. 2. 3. 4.", True),
        ("one two three four", True),
        ("One, two, three, four.", True),
        ("jedna dva tři čtyři", True),
        ("jedna dve tři ctyri", True),
        ("unlock 2 doors, PIN 1234", True),
        ("door 1, 1234", True),
        ("My PIN is one two three four. One door only.", True),
        ("\u0661\u0662\u0663\u0664", True),  # Arabic-Indic digits
        ("\uff11\uff12\uff13\uff14", True),  # full-width digits
        ("51234", False),
        ("12345", False),
        ("1 2 3", False),
        ("1 2 3 4 5", False),
        ("never mind", False),
        ("", False),
        # A longer number that contains the PIN, and compounds (review of
        # PR #769).
        ("nine one two three four five", False),
        ("555 1234", False),
        ("call 555-1234", False),
        ("+44 1234 567890", False),
        ("Use tracking number 1234-5678.", False),
        ("12-34", False),
        ("1234,56", False),
        ("1.2.3.4", False),
        ("12:34", False),
        ("Cancel the unlock. The invoice total is 12.34 dollars.", False),
        ("at 12.34 please", False),
        ("Do not unlock; the serial is abc1234xyz.", False),
        ("one234", False),
        ("123four", False),
        # The PIN read out with a door number after it reads as one longer
        # number and is refused; the user is asked again.
        ("PIN 1 2 3 4, 2 doors please", False),
        # Fourth review pass of PR #769.
        ("1,2,3,4, 5", False),
        ("555 - 1234", False),
        ("1234 - 5678", False),
        ("one two three four oh", False),
        ("1,2,3 and 4", True),
        ("one,2,3,4", True),
        ("Oh, 1234", True),
        # A spoken digit before a written number joins it: asked again.
        ("unlock door one 1234", False),
        # Pass 2 of the fourth review: never a fragment of a longer number,
        # a unit, a decimal, a range or a date.
        ("1234 five", False),
        ("12 34 oh", False),
        ("call 555 one two three four", False),
        ("The meter reads 1234.56kWh", False),
        ("\u0661\u0662\u0663\u0664\u066b\u0665\u0666", False),  # Arabic decimal
        ("1234 - 56.78", False),
        ("1234 56.78", False),
        ("12 : 34", False),
        ("1 / 2 / 3 / 4", False),
        # Compatibility forms fold to plain digits (NFKC), as setup stores them.
        ("\u2460\u2461\u2462\u2463", True),  # circled digits
        ("1\u200b234", True),  # zero-width space
        (unicodedata.normalize("NFD", "jedna dva tři čtyři"), True),
        # Pass 3 of the fourth review: a full-width decimal, a Unicode minus,
        # a unit glued to the number, and "oh and ..." are not the PIN.
        ("\uff11\uff12\uff13\uff14\uff0e\uff15\uff16", False),
        ("1234\u221256", False),
        ("1234\u00b0C", False),
        ("1234\u2103", False),
        ("1234%", False),
        ("1234\u2030", False),
        ("oh and one two three four", False),
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
        ("1111", "ping 1.1.1.1"),
        ("2026", "on 2026 / 10 / 15"),
        # "pet" and "tri" are not read as Czech digits: they are English words.
        ("5234", "feed the pet, 2, 3, 4"),
        ("3234", "tri 2 3 4"),
    ],
)
def test_dates_times_and_lists_are_not_a_pin(pin: str, text: str) -> None:
    assert pin_in_text(pin, text) is False


@pytest.mark.parametrize(
    ("pin", "text"),
    [
        ("1004", "one oh oh four"),
        ("1230", "one two three oh"),
        ("0123", "oh one two three"),
        ("123456", "123 456"),
    ],
)
def test_spoken_and_grouped_pins(pin: str, text: str) -> None:
    assert pin_in_text(pin, text) is True


def test_oh_after_the_pin_makes_a_longer_number() -> None:
    """A trailing "oh" is a zero, so "1234 oh" is 12340: asked again."""
    assert pin_in_text(PIN, f"{PIN} oh wait") is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", "1234"),
        ("1 2 3 4", "1234"),
        ("one two three four", "1234"),
        ("1, 2, 3 and 4", "1234"),
        ("PIN 1234", "1234"),
        ("12:34", None),
        ("12 and 34", None),
    ],
)
def test_pin_digits(text: str, expected: str | None) -> None:
    assert pin_digits(text) == expected


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


async def test_the_pin_as_the_user_said_it_confirms() -> None:
    """A model passing "1 2 3 4" or words is not refused in a loop."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("one two three four")
    reply = await turns.confirm(action_id, pin="one two three four")

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_a_wrong_pin_uses_up_the_message() -> None:
    """One message is one attempt: a looping model cannot spend all five."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("9999")
    first = await turns.confirm(action_id, pin="9999")
    second = await turns.confirm(action_id, pin="9999")

    assert isinstance(first, str)
    assert first.startswith("Incorrect PIN")
    assert isinstance(second, dict)
    assert second["status"] == PIN_AWAITING_USER_REPLY
    assert turns.pending[action_id]["attempts"] == 1
    assert api.calls == []


async def test_a_wrong_pin_refused_for_not_being_in_the_message_is_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say("never mind")

    with caplog.at_level(logging.WARNING):
        await turns.confirm(action_id, pin="9999")

    assert "Refused a correct critical-action PIN" not in caplog.text


async def test_the_pin_hash_runs_once_per_message_for_the_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model looping on a refused PIN does not hash it on the event loop again."""
    calls: list[str] = []

    def _verify(pin: str, **_: Any) -> bool:
        calls.append(pin)
        return True

    monkeypatch.setattr(tools_module, "verify_pin", _verify)
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say("never mind")

    for _ in range(3):
        await turns.confirm(action_id)

    assert calls == [PIN]


async def test_wrong_conversation_is_reported_before_wrong_user() -> None:
    """Another conversation's action reads as missing, whoever asks."""
    api = _FakeAPI()
    satellite_a = _Turns(api, thread_id="conv-a")
    action_id = satellite_a.request_unlock()
    satellite_b = _Turns(api, thread_id="conv-b")
    satellite_b.pending.update(satellite_a.pending)
    satellite_b.config["configurable"]["user_id"] = "someone_else"
    satellite_b.say(PIN)

    reply = await satellite_b.confirm(action_id)

    assert reply == "Pending action not found or expired."


async def test_an_action_without_a_tool_name_is_dropped() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    del turns.pending[action_id]["tool_name"]

    turns.say(PIN)
    await turns.confirm(action_id)

    assert action_id not in turns.pending
    assert api.calls == []


async def test_an_action_that_cannot_run_does_not_use_up_the_message() -> None:
    """With the HA API unavailable, nothing ran, so the user can retry."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.config["configurable"]["ha_llm_api"] = None

    turns.say(PIN)
    await turns.confirm(action_id)

    assert action_id in turns.pending
    assert turns.config["configurable"]["pin_reply"].spent is False


class _ToolStore:
    async def aget(self, _namespace: Any, key: str) -> Any:
        assert key == "hga_local::confirm_sensitive_action"
        return MagicMock(
            value={
                "name": "confirm_sensitive_action",
                "api_id": "hga_local",
                "description": "Confirm.",
                "parameters": {},
            }
        )


async def test_a_pending_action_keeps_the_pin_tool_past_the_history() -> None:
    """
    The store, not only the history, keeps the confirm tool available.

    The history is trimmed to a lookback window, so a pending challenge can
    fall out of it while the action is still waiting in the store.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    turns.request_unlock()
    messages: list[Any] = []
    store = cast("Any", _ToolStore())

    assert await graph._get_pending_pin_tools(messages, store) == []
    tools = await graph._get_pending_pin_tools(
        messages, store, cast("Any", turns.config)
    )
    assert [tool["name"] for tool in tools] == ["confirm_sensitive_action"]

    other = _Turns(api, thread_id="conv-other")
    other.pending.update(turns.pending)
    assert (
        await graph._get_pending_pin_tools(messages, store, cast("Any", other.config))
        == []
    )


async def test_a_new_protected_request_replaces_the_conversations_older_one() -> None:
    """
    The user's PIN never runs a request they moved on from.

    "Unlock the back door", then "no, the front door": only the newest
    request is waiting, so the back door cannot be confirmed. An id the
    conversation no longer holds means its only pending action (the one the
    fixed line named), and the PIN runs it once.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    stale = turns.request_unlock(BACK_DOOR_ARGS)
    turns.say("no, the front door")
    newest = turns.request_unlock()

    assert stale not in turns.pending

    turns.say(PIN)
    stale_reply = await turns.confirm(stale)
    newest_reply = await turns.confirm(newest)

    assert isinstance(stale_reply, dict)
    assert stale_reply["status"] == "completed"
    assert stale_reply["action_id"] == newest
    assert newest_reply == "Pending action not found or expired."
    assert api.calls == ["HassTurnOff"]
    assert api.sent_names == ["Front Door"]


async def test_another_conversations_pending_action_is_not_replaced() -> None:
    api = _FakeAPI()
    satellite_a = _Turns(api, thread_id="conv-a")
    kept = satellite_a.request_unlock()
    satellite_b = _Turns(api, thread_id="conv-b")
    satellite_b.pending.update(satellite_a.pending)
    satellite_b.request_unlock()

    assert kept in satellite_b.pending


@pytest.mark.parametrize(
    "said", ["\uff11\uff12\uff13\uff14", "\u0661\u0662\u0663\u0664"]
)
async def test_the_pin_in_other_digit_scripts_confirms(said: str) -> None:
    """A correct PIN typed in full-width or Arabic-Indic digits is not "wrong"."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(said)
    reply = await turns.confirm(action_id, pin=said)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"
    assert api.calls == ["HassTurnOff"]


async def test_a_junk_first_refusal_does_not_silence_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning is checked once per message, but only when it can run."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say("twelve thirty-four")

    with caplog.at_level(logging.WARNING):
        await turns.confirm(action_id, pin="PIN")
        await turns.confirm(action_id)

    assert "Refused a correct critical-action PIN" in caplog.text


@pytest.mark.parametrize(
    "created_at",
    [
        (dt_util.utcnow() - PENDING_ACTION_TTL - timedelta(seconds=1)).isoformat(),
        None,
        "not-a-date",
        "2026-10-14T12:00:00",  # timezone-naive
    ],
)
def test_expired_or_malformed_actions_are_not_pending(created_at: Any) -> None:
    pending = {"a1": {"thread_id": "conv1", "created_at": created_at}}
    assert thread_has_pending_action(pending, "conv1") is False


def test_a_fresh_action_is_pending_only_in_its_conversation() -> None:
    pending = {"a1": {"thread_id": "conv1", "created_at": dt_util.utcnow().isoformat()}}
    assert thread_has_pending_action(pending, "conv1") is True
    assert thread_has_pending_action(pending, "conv2") is False
    assert thread_has_pending_action(pending, None) is False


async def test_a_prompt_is_answered_by_the_next_message_only() -> None:
    """A user who moved on cannot confirm later: "which door?", then the PIN."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("which door was that?")
    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert reply == "Pending action not found or expired."
    assert api.calls == []


async def test_a_request_started_during_the_pin_reply_cannot_take_the_pin() -> None:
    """
    Text the model reads after the user's PIN reply starts another action.

    It replaces the waiting one, but was created in this turn, so this
    message's PIN cannot confirm it: nothing runs, and the user asks again.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    waiting = turns.request_unlock()

    turns.say(PIN)
    injected = turns.request_unlock(BACK_DOOR_ARGS)
    by_id = await turns.confirm(injected)
    without_id = await turns.confirm("")

    assert waiting not in turns.pending
    assert isinstance(by_id, dict)
    assert by_id["status"] == PIN_AWAITING_USER_REPLY
    assert isinstance(without_id, dict)
    assert without_id["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == []


async def test_failed_attempts_carry_over_to_a_replacing_request() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    first = turns.request_unlock()
    turns.say("9999")
    await turns.confirm(first, pin="9999")

    second = turns.request_unlock(BACK_DOOR_ARGS)

    assert turns.pending[second]["attempts"] == 1


async def test_another_users_request_in_the_conversation_does_not_replace() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    kept = turns.request_unlock()
    turns.config["configurable"]["user_id"] = "someone_else"
    turns.request_unlock(BACK_DOOR_ARGS)

    assert kept in turns.pending


def test_actions_without_a_conversation_are_not_replaced() -> None:
    pending: dict[str, Any] = {}
    first = register_pending_action(pending, {"tool_name": "a", "thread_id": None})
    second = register_pending_action(pending, {"tool_name": "b", "thread_id": None})

    assert first is not None
    assert second is not None
    assert {first, second} <= pending.keys()


def test_a_new_turn_only_ages_its_own_conversation() -> None:
    pending: dict[str, Any] = {}
    reply = start_pin_turn(pending, "conv-a", "unlock")
    action_id = register_pending_action(
        pending, {"tool_name": "a", "thread_id": "conv-a", "turn_id": reply.turn_id}
    )
    assert action_id is not None
    disclose_pin_prompts(pending, reply)

    for _ in range(3):
        start_pin_turn(pending, "conv-b", "hello")
    assert action_id in pending

    start_pin_turn(pending, "conv-a", "1234")
    assert pending[action_id]["reply_turn"] is not None
    start_pin_turn(pending, "conv-a", "never mind")
    assert action_id not in pending


async def test_a_value_that_never_reaches_the_hash_does_not_use_up_the_message() -> (
    None
):
    """A wrong-format value (the door number) leaves the reply usable."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(f"door 2, {PIN}")

    first = await turns.confirm(action_id, pin="2")
    second = await turns.confirm(action_id)

    assert isinstance(first, str)
    assert "Invalid PIN" in first
    assert isinstance(second, dict)
    assert second["status"] == "completed"


async def test_a_slow_action_after_the_claim_says_it_may_have_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout after the claim gets the honest reply, inside the tool."""

    class _SlowAPI(_FakeAPI):
        async def async_call_tool(self, tool_input: Any) -> Any:
            self.calls.append(tool_input.tool_name)
            await asyncio.sleep(5)
            return self.result

    monkeypatch.setattr(tools_module, "POST_CLAIM_TIMEOUT_S", 0.01)
    api = _SlowAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(PIN)

    reply = await turns.confirm(action_id)

    assert isinstance(reply, str)
    assert "may have run" in reply
    assert action_id not in turns.pending
    assert api.calls == ["HassTurnOff"]


async def test_only_the_ascii_form_of_the_pin_is_checked() -> None:
    """
    A PIN an earlier version stored as typed in other digits must be re-entered.

    Confirmation hashes the ASCII digits only (CHANGELOG), so the stored
    full-width form no longer matches; the user saves the PIN again.
    """
    said = "\uff11\uff12\uff13\uff14"
    hashed, salt = hash_pin(said)
    api = _FakeAPI()
    turns = _Turns(api)
    turns.config["configurable"]["options"] = {
        CONF_CRITICAL_ACTION_PIN_HASH: hashed,
        CONF_CRITICAL_ACTION_PIN_SALT: salt,
    }
    action_id = turns.request_unlock()

    turns.say(said)
    reply = await turns.confirm(action_id, pin=said)

    assert isinstance(reply, str)
    assert reply.startswith("Incorrect PIN")


def test_a_pin_is_stored_as_ascii_digits() -> None:
    assert normalize_pin(" \uff11\uff12\uff13\uff14 ") == "1234"
    assert normalize_pin("\u0661\u0662\u0663\u0664") == "1234"


async def test_a_labelled_pin_from_the_model_is_read() -> None:
    """The model passing "PIN 1234" is not refused in a loop."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(PIN)
    reply = await turns.confirm(action_id, pin=f"PIN {PIN}")

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_the_refused_pin_warning_hashes_off_the_event_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    jobs: list[Any] = []

    async def _executor_job(target: Any, *args: Any) -> Any:
        jobs.append(target)
        return target(*args)

    api = _FakeAPI()
    turns = _Turns(api)
    turns.config["configurable"]["hass"] = SimpleNamespace(
        async_add_executor_job=_executor_job, config=SimpleNamespace(language="en")
    )
    action_id = turns.request_unlock()
    turns.say("twelve thirty-four")

    with caplog.at_level(logging.WARNING):
        await turns.confirm(action_id)

    assert len(jobs) == 1
    assert "Refused a correct critical-action PIN" in caplog.text


def test_the_options_flow_stores_a_full_width_pin_as_ascii() -> None:
    """A PIN typed in full-width digits is enrolled as the ASCII PIN."""
    options: dict[str, Any] = {}
    error = HomeGenerativeAgentOptionsFlow._maybe_edit_pin(
        cast("Any", None),
        options,
        {
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "\uff11\uff12\uff13\uff14",
        },
    )

    assert error is None
    assert verify_pin(
        PIN,
        hashed=options[CONF_CRITICAL_ACTION_PIN_HASH],
        salt=options[CONF_CRITICAL_ACTION_PIN_SALT],
    )


async def test_circled_digits_read_as_the_pin() -> None:
    """Folded to plain digits (NFKC) in the message and in the model's value."""
    said = "\u2460\u2461\u2462\u2463"
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(said)
    reply = await turns.confirm(action_id, pin=said)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


def test_the_options_flow_refuses_digits_that_do_not_fold_to_ascii() -> None:
    """
    Setup stores only what confirmation can read back.

    Ethiopic digits are digits to str.isdigit but not decimal, so they never
    fold to ASCII and no message could ever confirm such a PIN.
    """
    options: dict[str, Any] = {}
    error = HomeGenerativeAgentOptionsFlow._maybe_edit_pin(
        cast("Any", None),
        options,
        {
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "\u1369\u136a\u136b\u136c",
        },
    )

    assert error == "invalid_pin"
    assert CONF_CRITICAL_ACTION_PIN_HASH not in options


def test_a_pin_is_stored_with_compatibility_digits_folded() -> None:
    assert normalize_pin("\u2460\u2461\u2462\u2463") == "1234"
    assert normalize_pin("\u00b9\u00b2\u00b3\u2074") == "1234"


async def test_a_wrong_pin_keeps_the_prompt_for_one_more_reply() -> None:
    """A typo is not a dead end: the next message can still answer the prompt."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("9999")
    wrong = await turns.confirm(action_id, pin="9999")
    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert isinstance(wrong, str)
    assert wrong.startswith("Incorrect PIN")
    assert isinstance(reply, dict)
    assert reply["status"] == "completed"
    assert api.calls == ["HassTurnOff"]


async def test_an_unreadable_pin_keeps_the_prompt_for_one_more_reply() -> None:
    """A PIN the matcher could not read is asked for again, and the reply counts."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("twelve thirty-four")
    refused = await turns.confirm(action_id)
    turns.say("1 2 3 4")
    reply = await turns.confirm(action_id)

    assert isinstance(refused, dict)
    assert refused["status"] == PIN_AWAITING_USER_REPLY
    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_a_repeated_request_in_the_reply_turn_gets_a_fresh_reply() -> None:
    """
    "Unlock the front door" said again re-prompts, and that prompt is answerable.

    Codex structured review, pass 3: the re-issued prompt inherited a reply
    turn that was already used, so the PIN that answered it was refused.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    turns.request_unlock()

    turns.say("unlock the front door")
    reissued = turns.request_unlock()
    turns.say(PIN)
    reply = await turns.confirm(reissued)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_a_failed_turn_keeps_the_prompt_for_one_more_reply() -> None:
    """A PIN reply whose turn failed (model error) can be sent again."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say(PIN)
    keep_pin_prompts(turns.pending, turns.config["configurable"]["pin_reply"])
    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_a_prompt_is_kept_for_at_most_three_replies() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    for guess in ("9999", "9998", "9997"):
        turns.say(guess)
        await turns.confirm(action_id, pin=guess)
    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert reply == "Pending action not found or expired."
    assert api.calls == []


async def test_an_unrelated_message_after_a_wrong_pin_still_cancels() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("9999")
    await turns.confirm(action_id, pin="9999")
    turns.say("what's the weather?")
    turns.say(PIN)
    reply = await turns.confirm(action_id)

    assert reply == "Pending action not found or expired."
    assert api.calls == []


async def test_a_renewed_prompt_still_allows_one_attempt_per_message() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    turns.say("9999")
    await turns.confirm(action_id, pin="9999")
    again = await turns.confirm(action_id, pin="9999")

    assert isinstance(again, dict)
    assert again["status"] == PIN_AWAITING_USER_REPLY
    assert turns.pending[action_id]["attempts"] == 1


async def test_an_expired_prompt_is_dropped_at_the_next_turn() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.pending[action_id]["created_at"] = (
        dt_util.utcnow() - PENDING_ACTION_TTL - timedelta(seconds=1)
    ).isoformat()

    turns.say(PIN)

    assert action_id not in turns.pending


async def test_reissuing_an_expired_request_does_not_revive_its_reply_turn() -> None:
    """
    An expired prompt re-issued in the late reply turn needs a fresh reply.

    Pass 3 (both Codex reviews): the re-issue carried the reply turn and reset
    the clock, so a PIN sent after the TTL could still run the action.
    """
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(f"unlock the front door, PIN {PIN}")
    turns.pending[action_id]["created_at"] = (
        dt_util.utcnow() - PENDING_ACTION_TTL - timedelta(seconds=1)
    ).isoformat()

    reissued = turns.request_unlock()
    reply = await turns.confirm(reissued)

    assert isinstance(reply, dict)
    assert reply["status"] == PIN_AWAITING_USER_REPLY
    assert api.calls == []


async def test_a_reissue_that_differs_only_in_case_is_the_same_request() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    turns.request_unlock()

    turns.say(PIN)
    reissued = turns.request_unlock({"domain": ["lock"], "name": "front door"})
    reply = await turns.confirm(reissued)

    assert isinstance(reply, dict)
    assert reply["status"] == "completed"


async def test_the_claim_spends_the_users_message() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(PIN)

    await turns.confirm(action_id)

    assert turns.config["configurable"]["pin_reply"].spent is True


async def test_the_deadline_counts_from_the_start_of_the_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Time spent before the claim (the PIN hash) counts toward the deadline."""
    real_verify = tools_module.verify_pin

    def _slow_verify(*args: Any, **kwargs: Any) -> bool:
        time.sleep(0.05)
        return real_verify(*args, **kwargs)

    class _SlowAPI(_FakeAPI):
        async def async_call_tool(self, tool_input: Any) -> Any:
            self.calls.append(tool_input.tool_name)
            await asyncio.sleep(0.04)
            return self.result

    monkeypatch.setattr(tools_module, "verify_pin", _slow_verify)
    monkeypatch.setattr(tools_module, "POST_CLAIM_TIMEOUT_S", 0.06)
    api = _SlowAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say(PIN)

    reply = await turns.confirm(action_id)

    assert isinstance(reply, str)
    assert "may have run" in reply


async def test_the_refusal_asks_for_the_digits_one_by_one() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()
    turns.say("twelve thirty-four")

    reply = await turns.confirm(action_id)

    assert isinstance(reply, dict)
    assert "one by one" in reply["reason"]


async def test_the_last_reply_does_not_promise_another_one() -> None:
    """After the third reply the prompt is gone: say so, not "say it again"."""
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    replies = []
    for said, pin in (("9999", "9999"), ("9998", "9998"), ("twelve thirty-four", PIN)):
        turns.say(said)
        replies.append(await turns.confirm(action_id, pin=pin))

    assert isinstance(replies[0], str)
    assert "next message" in replies[0]
    assert isinstance(replies[2], dict)
    assert "next message" not in replies[2]["reason"]
    assert pin_msg(None, "pin_request_again") in replies[2]["reason"]


async def test_a_wrong_pin_on_the_last_reply_asks_for_a_new_request() -> None:
    api = _FakeAPI()
    turns = _Turns(api)
    action_id = turns.request_unlock()

    replies = []
    for guess in ("9999", "9998", "9997"):
        turns.say(guess)
        replies.append(await turns.confirm(action_id, pin=guess))

    assert pin_msg(None, "pin_retry_next_message") in replies[1]
    assert pin_msg(None, "pin_request_again") in replies[2]
    assert pin_msg(None, "pin_retry_next_message") not in replies[2]
