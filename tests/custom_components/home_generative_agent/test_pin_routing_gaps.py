# ruff: noqa: S101
"""
A PIN confirms only the request its speaker was told about (#1001).

The gaps the #987 re-review found: a re-asked locked-out request, a refusal
that did not name the waiting request, voices told apart by a user id they
share, and a PIN reply in progress cut off by its owner's next turn.
"""

from __future__ import annotations

from typing import Any

from custom_components.home_generative_agent.agent.helpers import (
    MAX_PIN_ATTEMPTS,
    PIN_REFUSED_LOCKED,
    PIN_REFUSED_OTHER_VOICE,
    PinReply,
    disclose_pin_prompts,
    end_pin_turn,
    newest_pin_prompt,
    register_pending_action,
    start_pin_turn,
    waiting_pin_prompt,
)
from custom_components.home_generative_agent.agent.pin_messages import pin_msg

ANNA = "conv1:anna"
BORIS = "conv1:boris"


def _unlock(
    thread_id: str,
    user: str,
    name: str,
    reply: PinReply,
    *,
    voice_thread: str | None = None,
) -> dict[str, Any]:
    """Return an unlock request as a voice's turn holds it."""
    return {
        "tool_name": "HassTurnOff",
        "tool_args": {"domain": ["lock"], "name": name},
        "user": user,
        "thread_id": thread_id,
        "voice_conversation": "conv1",
        # The thread of the voice that spoke; the turn may have followed
        # another voice's prompt onto that one's thread.
        "voice_thread": voice_thread or thread_id,
        "turn_id": reply.turn_id,
    }


def _hold(pending: dict[str, Any], thread_id: str, user: str, name: str) -> str | None:
    """Make the request in a whole turn: hold it, deliver its line, end the turn."""
    reply = start_pin_turn(pending, thread_id, f"unlock the {name}")
    action_id = register_pending_action(
        pending, _unlock(thread_id, user, name, reply), pin_reply=reply
    )
    disclose_pin_prompts(pending, reply)
    end_pin_turn(pending, reply.turn_id)
    return action_id


def test_a_locked_out_request_asked_again_is_refused_not_announced() -> None:
    """
    Anna, locked out, asks again in her reply turn: refused, nothing to confirm.

    Before the fix the request was held again with its five wrong PINs and
    announced ("Your PIN will confirm: unlock Front Door"); as it no longer
    blocked, Boris's request was held beside it and took her next PIN.
    """
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    reply = start_pin_turn(pending, ANNA, "0000")
    pending[anna]["attempts"] = MAX_PIN_ATTEMPTS

    again = register_pending_action(
        pending, _unlock(ANNA, "anna", "Front Door", reply), pin_reply=reply
    )

    assert again is None
    assert reply.refusal == PIN_REFUSED_LOCKED
    # Nothing new is waiting for her message, and the locked prompt is not
    # one a PIN said next can be followed to.
    assert newest_pin_prompt(pending, "conv1") is None
    assert waiting_pin_prompt(pending, "conv1", "1234") is None


def test_the_refusal_names_the_waiting_request() -> None:
    """Boris is told which request waits, so his PIN does not run it unheard."""
    pending: dict[str, Any] = {}
    assert _hold(pending, ANNA, "anna", "Front Door") is not None
    reply = start_pin_turn(pending, BORIS, "open the garage")

    held = register_pending_action(
        pending, _unlock(BORIS, "boris", "Garage Door", reply), pin_reply=reply
    )

    assert held is None
    assert reply.refusal == PIN_REFUSED_OTHER_VOICE
    assert reply.waiting_action is not None
    assert reply.waiting_action["tool_args"]["name"] == "Front Door"
    line = pin_msg(None, "pin_other_voice_waiting_line", action="unlock Front Door")
    assert "unlock Front Door" in line
    assert not line.rstrip().endswith("?")


def test_voices_sharing_a_user_are_still_told_apart() -> None:
    """
    Two voices with no linked person both run as `robot`.

    Boris's turn follows Anna's prompt onto her thread; his different request
    must be refused, not taken for Anna's own and swapped in for her prompt.
    """
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "robot", "Front Door")
    assert anna is not None
    reply = start_pin_turn(pending, ANNA, "open the garage, 1234")

    held = register_pending_action(
        pending,
        _unlock(ANNA, "robot", "Garage Door", reply, voice_thread=BORIS),
        pin_owner="robot",
        pin_reply=reply,
    )

    assert held is None
    assert reply.refusal == PIN_REFUSED_OTHER_VOICE
    assert pending[anna]["tool_args"]["name"] == "Front Door"


def test_a_pin_reply_in_progress_survives_its_owners_next_turn() -> None:
    """
    A PIN reply in progress is not cut off by its prompt owner's next turn.

    Boris's correct PIN for Anna's prompt is still being handled when Anna
    speaks again: her turn must not drop the prompt his reply is confirming.
    """
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    boris_reply = start_pin_turn(pending, ANNA, "1234")
    assert pending[anna]["reply_turn"] == boris_reply.turn_id

    anna_turn = start_pin_turn(pending, ANNA, "what time is it")

    assert anna in pending
    assert pending[anna]["reply_turn"] == boris_reply.turn_id
    # Nor does a new request from Anna in that turn replace it.
    held = register_pending_action(
        pending, _unlock(ANNA, "anna", "Back Door", anna_turn), pin_reply=anna_turn
    )
    assert held is None
    assert pending[anna]["tool_args"]["name"] == "Front Door"
    end_pin_turn(pending, anna_turn.turn_id)
    assert anna in pending


def test_a_locked_out_prompt_is_not_followed() -> None:
    """
    A PIN said next is not followed to a prompt no PIN can confirm.

    A prompt kept for another reply after its last wrong PIN is disclosed and
    waiting, but locked out: it no longer blocks another voice's request, so
    following it would leave that request's speaker unheard.
    """
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    assert newest_pin_prompt(pending, "conv1") is pending[anna]

    pending[anna]["attempts"] = MAX_PIN_ATTEMPTS

    assert newest_pin_prompt(pending, "conv1") is None
