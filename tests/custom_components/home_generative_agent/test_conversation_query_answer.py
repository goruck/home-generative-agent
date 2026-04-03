"""Tests for ConversationResult ``response_type`` (intent_output / API parity)."""

from __future__ import annotations

from homeassistant.components import conversation
from homeassistant.helpers import intent

from custom_components.home_generative_agent.conversation import (
    _apply_query_answer_when_no_ha_intent,
)


def test_apply_query_answer_sets_query_answer_when_no_ha_intent() -> None:
    """Pure Q&A turns should report QUERY_ANSWER, not default ACTION_DONE."""
    ir = intent.IntentResponse(language="en")
    assert ir.response_type == intent.IntentResponseType.ACTION_DONE
    result = conversation.ConversationResult(response=ir, conversation_id="c1")
    _apply_query_answer_when_no_ha_intent(
        result, ha_had_intent_response_from_tool=False
    )
    assert result.response.response_type == intent.IntentResponseType.QUERY_ANSWER


def test_apply_query_answer_preserves_action_done_when_ha_intent_ran() -> None:
    """When an HA intent tool ran, do not override ACTION_DONE from the tool."""
    ir = intent.IntentResponse(language="en")
    result = conversation.ConversationResult(response=ir, conversation_id="c1")
    _apply_query_answer_when_no_ha_intent(result, ha_had_intent_response_from_tool=True)
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE


def test_apply_query_answer_does_not_override_error() -> None:
    """Errors must stay ERROR."""
    ir = intent.IntentResponse(language="en")
    ir.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "failed")
    result = conversation.ConversationResult(response=ir, conversation_id="c1")
    _apply_query_answer_when_no_ha_intent(
        result, ha_had_intent_response_from_tool=False
    )
    assert result.response.response_type == intent.IntentResponseType.ERROR
