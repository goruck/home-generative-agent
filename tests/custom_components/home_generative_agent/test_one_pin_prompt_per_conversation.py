# ruff: noqa: S101
"""A satellite conversation holds one PIN prompt across its voices (#974)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.agent.helpers import (
    MAX_PIN_ATTEMPTS,
    PENDING_ACTION_TTL,
    PinReply,
    another_voice_waiting,
    disclose_pin_prompts,
    end_pin_turn,
    keep_pin_prompts,
    register_pending_action,
    start_pin_turn,
    waiting_pin_prompt,
)

ANNA = "conv1:anna"
BORIS = "conv1:boris"
VOICES = {ANNA, BORIS, "conv2:anna"}
_COMPONENT_DIR = (
    Path(__file__).resolve().parents[3] / "custom_components" / "home_generative_agent"
)


def _unlock(thread_id: str, user: str, name: str, reply: PinReply) -> dict[str, Any]:
    """Return an unlock request as the handler holds it in the reply's turn."""
    return {
        "tool_name": "HassTurnOff",
        "tool_args": {"domain": ["lock"], "name": name},
        "user": user,
        "thread_id": thread_id,
        # Only a voice's turn names its conversation.
        "voice_conversation": (
            thread_id.split(":", 1)[0] if thread_id in VOICES else None
        ),
        "turn_id": reply.turn_id,
    }


def _hold(pending: dict[str, Any], thread_id: str, user: str, name: str) -> str | None:
    """Make the request in a whole turn: hold it, deliver its line, end the turn."""
    reply = start_pin_turn(pending, thread_id, f"unlock the {name}")
    action_id = register_pending_action(pending, _unlock(thread_id, user, name, reply))
    disclose_pin_prompts(pending, reply)
    end_pin_turn(pending, reply.turn_id)
    return action_id


def test_another_voices_live_prompt_refuses_a_new_request() -> None:
    """The first request wins, so the PIN said next confirms what Anna was told."""
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None

    assert _hold(pending, BORIS, "boris", "Garage Door") is None

    assert list(pending) == [anna]
    prompt = waiting_pin_prompt(pending, "conv1", "1234")
    assert prompt is not None
    assert prompt["tool_args"]["name"] == "Front Door"


def test_a_prompt_whose_line_is_on_its_way_refuses_too() -> None:
    """Overlapping turns: Anna's line is not out yet when Boris asks (review 1)."""
    pending: dict[str, Any] = {}
    anna_turn = start_pin_turn(pending, ANNA, "unlock the front door")
    anna = register_pending_action(
        pending, _unlock(ANNA, "anna", "Front Door", anna_turn)
    )
    assert anna is not None

    assert _hold(pending, BORIS, "boris", "Garage Door") is None

    disclose_pin_prompts(pending, anna_turn)
    end_pin_turn(pending, anna_turn.turn_id)
    assert list(pending) == [anna]


def test_a_pin_reply_in_progress_is_not_cut_off() -> None:
    """Anna's correct PIN must still confirm her request (review 2)."""
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    reply = start_pin_turn(pending, ANNA, "1234")
    assert pending[anna]["reply_turn"] == reply.turn_id

    assert _hold(pending, BORIS, "boris", "Garage Door") is None

    assert pending[anna]["reply_turn"] == reply.turn_id


def test_a_prompt_from_a_failed_turn_does_not_block() -> None:
    """Never delivered, so it is dropped as its turn ends (review 3)."""
    pending: dict[str, Any] = {}
    boris_turn = start_pin_turn(pending, BORIS, "unlock the garage door")
    register_pending_action(pending, _unlock(BORIS, "boris", "Garage Door", boris_turn))
    end_pin_turn(pending, boris_turn.turn_id)

    assert pending == {}
    assert _hold(pending, ANNA, "anna", "Front Door") is not None


def test_a_prompt_answered_without_being_kept_does_not_block() -> None:
    """Anna moved on, so her prompt ends with that turn."""
    pending: dict[str, Any] = {}
    _hold(pending, ANNA, "anna", "Front Door")
    reply = start_pin_turn(pending, ANNA, "what time is it")
    end_pin_turn(pending, reply.turn_id)

    assert pending == {}
    assert _hold(pending, BORIS, "boris", "Garage Door") is not None


def test_a_prompt_kept_for_another_reply_still_refuses() -> None:
    """A turn that failed keeps its prompt, so it is still live."""
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    reply = start_pin_turn(pending, ANNA, "1234")
    keep_pin_prompts(pending, reply)
    end_pin_turn(pending, reply.turn_id)

    assert list(pending) == [anna]
    assert _hold(pending, BORIS, "boris", "Garage Door") is None


def test_a_locked_out_prompt_does_not_block_or_carry_its_attempts() -> None:
    """Boris's correct PIN must not get "too many attempts" from Anna's lockout."""
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    pending[anna]["attempts"] = MAX_PIN_ATTEMPTS

    boris = _hold(pending, BORIS, "boris", "Garage Door")

    assert boris is not None
    assert pending[boris]["attempts"] == 0
    assert "sentence_checks" not in pending[boris]


def test_another_voices_expired_prompt_does_not_block() -> None:
    pending: dict[str, Any] = {}
    anna = _hold(pending, ANNA, "anna", "Front Door")
    assert anna is not None
    old = dt_util.utcnow() - PENDING_ACTION_TTL - timedelta(seconds=1)
    pending[anna]["created_at"] = old.isoformat()

    boris = _hold(pending, BORIS, "boris", "Garage Door")

    assert list(pending) == [boris]


def test_the_same_voice_still_replaces_its_own_prompt() -> None:
    """The per-user rule is unchanged: a voice's new request replaces its own."""
    pending: dict[str, Any] = {}
    first = _hold(pending, ANNA, "anna", "Front Door")
    assert first is not None
    pending[first]["attempts"] = 2

    second = _hold(pending, ANNA, "anna", "Garage Door")

    assert second is not None
    assert list(pending) == [second]
    assert pending[second]["attempts"] == 2


def test_other_conversations_keep_their_prompts() -> None:
    pending: dict[str, Any] = {}
    other_room = _hold(pending, "conv2:anna", "anna", "Back Door")
    typed = _hold(pending, "conv1", "anna", "Shed")
    boris = _hold(pending, BORIS, "boris", "Garage Door")

    assert set(pending) == {other_room, typed, boris}


def test_a_typed_conversation_is_not_refused() -> None:
    """Without voices there is no other voice: the per-user rule applies as before."""
    pending: dict[str, Any] = {}
    _hold(pending, "conv1", "anna", "Front Door")
    boris = _hold(pending, "conv1", "boris", "Garage Door")

    assert boris is not None
    assert boris in pending


def test_automation_ids_with_colons_are_not_voices() -> None:
    """conversation.process keeps a caller's id, so "door:front" is a conversation."""
    pending: dict[str, Any] = {}
    front = _hold(pending, "door:front", "automation", "Front Door")
    back = _hold(pending, "door:back", "automation", "Back Door")

    assert set(pending) == {front, back}
    assert waiting_pin_prompt(pending, "door", "1234") is None


def test_another_voice_waiting_ignores_a_voices_own_prompt() -> None:
    pending: dict[str, Any] = {}
    _hold(pending, ANNA, "anna", "Front Door")
    reply = start_pin_turn(pending, ANNA, "unlock the garage door")
    assert not another_voice_waiting(
        pending, _unlock(ANNA, "anna", "Garage Door", reply)
    )


def test_the_handler_names_the_conversation_only_for_a_voice() -> None:
    """Read as text: importing conversation.py needs hassil."""
    src = (_COMPONENT_DIR / "conversation.py").read_text()
    assert '"voice_conversation": conversation_id if speaker else None' in src


def test_the_handler_ends_every_turn() -> None:
    """Dead prompts are dropped however the turn returns or fails."""
    src = (_COMPONENT_DIR / "conversation.py").read_text()
    assert "finally:\n            end_pin_turn(runtime_data.pending_actions" in src
