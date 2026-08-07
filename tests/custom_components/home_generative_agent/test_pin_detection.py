# ruff: noqa: S101
"""Unit tests for _has_pending_pin_confirmation."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from custom_components.home_generative_agent.agent.graph import (
    _has_pending_pin_confirmation,
)


def _requires_pin_msg(action_id: str = "act1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"status": "requires_pin", "action_id": action_id}),
        tool_call_id="tc1",
        name="HassLockLock",
    )


def _confirm_msg(content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id="tc2",
        name="confirm_sensitive_action",
    )


def _ai_pin_ask() -> AIMessage:
    return AIMessage(content="Please provide your PIN to confirm the lock action.")


class TestHasPendingPinConfirmation:
    """Tests for _has_pending_pin_confirmation."""

    def test_no_messages(self) -> None:
        assert _has_pending_pin_confirmation([]) is False

    def test_requires_pin_no_confirmation(self) -> None:
        msgs = [_requires_pin_msg()]
        assert _has_pending_pin_confirmation(msgs) is True

    def test_requires_pin_then_success(self) -> None:
        success_content = json.dumps({"status": "completed", "action_id": "act1"})
        msgs = [_requires_pin_msg(), _confirm_msg(success_content)]
        assert _has_pending_pin_confirmation(msgs) is False

    def test_requires_pin_then_wrong_pin_still_pending(self) -> None:
        """Core regression: wrong PIN should NOT clear the pending state."""
        msgs = [
            _requires_pin_msg(),
            _confirm_msg("Incorrect PIN. Action not executed."),
        ]
        assert _has_pending_pin_confirmation(msgs) is True

    def test_requires_pin_then_wrong_pin_then_correct(self) -> None:
        """After wrong PIN, correct PIN resolves the flow."""
        success_content = json.dumps({"status": "completed", "action_id": "act1"})
        msgs = [
            _requires_pin_msg(),
            _confirm_msg("Incorrect PIN. Action not executed."),
            _confirm_msg(success_content),
        ]
        assert _has_pending_pin_confirmation(msgs) is False

    def test_requires_pin_then_too_many_attempts(self) -> None:
        """Too many attempts resolves the flow (action expired, user must restart)."""
        msgs = [
            _requires_pin_msg(),
            _confirm_msg("Too many incorrect attempts; please re-run the request."),
        ]
        assert _has_pending_pin_confirmation(msgs) is False

    def test_multiple_wrong_pins_still_pending(self) -> None:
        """Multiple wrong PINs — flow remains pending until correct or expired."""
        msgs = [
            _requires_pin_msg(),
            _confirm_msg("Incorrect PIN. Action not executed."),
            _confirm_msg("Incorrect PIN. Action not executed."),
        ]
        assert _has_pending_pin_confirmation(msgs) is True


class TestAddAutomationChallenge:
    """
    The add_automation gate depends on this tool-name-agnostic scan.

    `add_automation` is a LangChain tool, not an HA actuation tool, so it is
    absent from ACTUATION_LANGCHAIN_TOOLS and the degraded-flow fallback never
    applies to it. If the requires_pin scan ever started filtering by tool
    name, a held automation would become unconfirmable — the model would never
    get `confirm_sensitive_action` force-injected on the next turn.
    """

    @staticmethod
    def _automation_challenge() -> ToolMessage:
        return ToolMessage(
            content=json.dumps({"status": "requires_pin", "action_id": "auto1"}),
            tool_call_id="tc9",
            name="add_automation",
        )

    def test_add_automation_challenge_keeps_the_flow_pending(self) -> None:
        assert _has_pending_pin_confirmation([self._automation_challenge()]) is True

    def test_add_automation_challenge_resolves_on_completion(self) -> None:
        msgs = [
            self._automation_challenge(),
            _confirm_msg(json.dumps({"status": "completed", "action_id": "auto1"})),
        ]
        assert _has_pending_pin_confirmation(msgs) is False

    def test_add_automation_challenge_survives_a_wrong_pin(self) -> None:
        msgs = [
            self._automation_challenge(),
            _confirm_msg("Incorrect PIN. Action not executed."),
        ]
        assert _has_pending_pin_confirmation(msgs) is True
