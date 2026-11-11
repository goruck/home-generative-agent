# ruff: noqa: S101
"""A satellite conversation holds one PIN prompt across its voices (#974)."""

from __future__ import annotations

from typing import Any

from custom_components.home_generative_agent.agent.helpers import (
    disclose_pin_prompts,
    register_pending_action,
    start_pin_turn,
    waiting_pin_prompt,
)

ANNA = "conv1:anna"
BORIS = "conv1:boris"
VOICES = {ANNA, BORIS, "conv2:anna"}


def _hold(pending: dict[str, Any], thread_id: str, user: str, name: str) -> str | None:
    """Hold an unlock in a new turn on the thread and deliver its prompt."""
    voice = thread_id.split(":", 1)[0] if thread_id in VOICES else None
    reply = start_pin_turn(pending, thread_id, f"unlock the {name}")
    action_id = register_pending_action(
        pending,
        {
            "tool_name": "HassTurnOff",
            "tool_args": {"domain": ["lock"], "name": name},
            "user": user,
            "thread_id": thread_id,
            # As the handler sets it: only a voice's turn names one.
            "voice_conversation": voice,
            "turn_id": reply.turn_id,
        },
    )
    disclose_pin_prompts(pending, reply)
    return action_id


def test_a_voices_new_prompt_drops_another_voices_prompt() -> None:
    """Two prompts from overlapping turns: the PIN could confirm the wrong one."""
    pending: dict[str, Any] = {}
    _hold(pending, ANNA, "anna", "Front Door")
    newest = _hold(pending, BORIS, "boris", "Garage Door")

    assert list(pending) == [newest]
    prompt = waiting_pin_prompt(pending, "conv1", "1234")
    assert prompt is not None
    assert prompt["tool_args"]["name"] == "Garage Door"


def test_another_voices_request_does_not_reset_the_attempts() -> None:
    pending: dict[str, Any] = {}
    first = _hold(pending, ANNA, "anna", "Front Door")
    assert first is not None
    pending[first]["attempts"] = 4

    second = _hold(pending, BORIS, "boris", "Front Door")
    assert second is not None
    assert pending[second]["attempts"] == 4

    # And back again: the limit still holds.
    third = _hold(pending, ANNA, "anna", "Front Door")
    assert third is not None
    assert pending[third]["attempts"] == 4


def test_the_higher_attempts_win() -> None:
    """A request of the same voice carries its own count; another's may be higher."""
    pending: dict[str, Any] = {}
    first = _hold(pending, ANNA, "anna", "Front Door")
    assert first is not None
    pending[first]["attempts"] = 1
    second = _hold(pending, BORIS, "boris", "Garage Door")
    assert second is not None
    pending[second]["attempts"] = 3

    third = _hold(pending, ANNA, "anna", "Front Door")

    assert third is not None
    assert list(pending) == [third]
    assert pending[third]["attempts"] == 3


def test_another_voices_request_does_not_reset_the_sentence_checks() -> None:
    """Another voice's request in between gives no fresh hash-checked guesses."""
    pending: dict[str, Any] = {}
    first = _hold(pending, ANNA, "anna", "Front Door")
    assert first is not None
    pending[first]["sentence_checks"] = 4

    second = _hold(pending, BORIS, "boris", "Garage Door")
    assert second is not None
    assert pending[second]["sentence_checks"] == 4

    third = _hold(pending, ANNA, "anna", "Front Door")
    assert third is not None
    assert pending[third]["sentence_checks"] == 4


def test_other_conversations_keep_their_prompts() -> None:
    pending: dict[str, Any] = {}
    other_room = _hold(pending, "conv2:anna", "anna", "Back Door")
    typed = _hold(pending, "conv1", "anna", "Shed")
    newest = _hold(pending, BORIS, "boris", "Garage Door")

    assert set(pending) == {other_room, typed, newest}


def test_a_typed_conversation_keeps_the_per_user_rule() -> None:
    """Without voices, two users' prompts in one conversation are left alone."""
    pending: dict[str, Any] = {}
    anna = _hold(pending, "conv1", "anna", "Front Door")
    boris = _hold(pending, "conv1", "boris", "Garage Door")

    assert set(pending) == {anna, boris}


def test_automation_ids_with_colons_are_not_voices() -> None:
    """conversation.process keeps a caller's id, so "door:front" is a conversation."""
    pending: dict[str, Any] = {}
    front = _hold(pending, "door:front", "automation", "Front Door")
    back = _hold(pending, "door:back", "automation", "Back Door")

    assert set(pending) == {front, back}


def test_the_handler_names_the_conversation_only_for_a_voice() -> None:
    import inspect  # noqa: PLC0415

    from custom_components.home_generative_agent.conversation import (  # noqa: PLC0415
        HGAConversationEntity,
    )

    src = inspect.getsource(HGAConversationEntity._async_handle_message_active)
    assert '"voice_conversation": conversation_id if speaker else None' in src
