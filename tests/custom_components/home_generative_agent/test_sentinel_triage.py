"""Tests for SentinelTriageService — sentinel/triage.py (Issue #262)."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.triage import (
    TRIAGE_NOTIFY,
    TRIAGE_SUPPRESS,
    SentinelTriageService,
    TriageDecision,
    _build_prompt,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Configurable LLM stub for triage tests."""

    def __init__(
        self,
        response_json: dict[str, Any] | None = None,
        raise_exc: Exception | None = None,
        timeout: bool = False,
    ) -> None:
        self.response_json = response_json
        self.raise_exc = raise_exc
        self.timeout = timeout
        self.last_messages: list[Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.last_messages = messages
        if self.timeout:
            await asyncio.sleep(9999)
        if self.raise_exc:
            raise self.raise_exc
        content = json.dumps(self.response_json) if self.response_json else ""
        return type("Result", (), {"content": content})()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(
    ftype: str = "open_entry_while_away",
    confidence: float = 0.8,
    is_sensitive: bool = False,
    evidence: dict[str, Any] | None = None,
) -> AnomalyFinding:
    return AnomalyFinding(
        anomaly_id="test-id-1",
        type=ftype,
        severity="medium",
        confidence=confidence,
        triggering_entities=["binary_sensor.front_door"],
        evidence=evidence or {},
        suggested_actions=["lock_door"],
        is_sensitive=is_sensitive,
    )


def _snapshot(is_night: bool = False, anyone_home: bool = False) -> FullStateSnapshot:
    return cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T10:00:00+00:00",
                "timezone": "UTC",
                "is_night": is_night,
                "anyone_home": anyone_home,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        },
    )


# ---------------------------------------------------------------------------
# 1. Input allowlist enforced
# ---------------------------------------------------------------------------


def test_prompt_allowlist_contains_required_structural_fields() -> None:
    """Prompt must include all required structural fields from the allowlist."""
    finding = _finding()
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "type:" in prompt
    assert "severity:" in prompt
    assert "confidence:" in prompt
    assert "is_sensitive:" in prompt
    assert "entity_count:" in prompt
    assert "suggested_actions_count:" in prompt


def test_prompt_allowlist_excludes_entity_ids() -> None:
    """Raw entity IDs from triggering_entities must not appear in the prompt."""
    finding = _finding()
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "binary_sensor.front_door" not in prompt


def test_prompt_allowlist_excludes_person_names_replaces_with_count() -> None:
    """Person names from recognized_people must be replaced with a count."""
    evidence = {"recognized_people": ["Alice", "Bob"]}
    finding = _finding(evidence=evidence)
    snapshot = _snapshot(is_night=True, anyone_home=False)

    prompt = _build_prompt(finding, snapshot)

    assert "Alice" not in prompt
    assert "Bob" not in prompt
    assert "recognized_people_count: 2" in prompt


def test_prompt_allowlist_excludes_unlisted_evidence_keys() -> None:
    """Evidence keys not on the allowlist must never appear in the prompt."""
    evidence = {
        "entity_id": "binary_sensor.front_door",
        "state": "on",
        "area": "Front Entrance",
        "raw_attribute": "some_value",
        "last_changed_age_seconds": 60,
    }
    finding = _finding(evidence=evidence)
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "entity_id" not in prompt
    assert "Front Entrance" not in prompt
    assert "raw_attribute" not in prompt
    assert "some_value" not in prompt
    # The one allowed key is still present.
    assert "last_changed_age_seconds: 60" in prompt


def test_prompt_entity_count_and_actions_count_match_finding() -> None:
    """entity_count and suggested_actions_count must equal the finding's list lengths."""
    finding = AnomalyFinding(
        anomaly_id="multi-entity",
        type="open_entry_while_away",
        severity="high",
        confidence=0.9,
        triggering_entities=[
            "binary_sensor.front_door",
            "binary_sensor.back_door",
            "binary_sensor.garage_door",
        ],
        evidence={},
        suggested_actions=["lock_door", "alert_user"],
        is_sensitive=False,
    )
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "entity_count: 3" in prompt
    assert "suggested_actions_count: 2" in prompt


def test_prompt_without_evidence_has_no_evidence_block() -> None:
    """When no allowed evidence fields are present the evidence block is omitted."""
    finding = _finding(evidence={})
    minimal_snapshot = cast(
        "FullStateSnapshot",
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {},
        },
    )

    prompt = _build_prompt(finding, minimal_snapshot)

    assert "evidence:" not in prompt
    assert "is_night" not in prompt
    assert "anyone_home" not in prompt


def test_prompt_allowed_sanitized_evidence_in_prompt() -> None:
    """All allowed evidence fields appear in the prompt; person names do not."""
    evidence = {
        "recognized_people": ["Alice"],
        "last_changed_age_seconds": 120,
    }
    finding = _finding(ftype="unknown_person_camera_no_home", evidence=evidence)
    snapshot = _snapshot(is_night=True, anyone_home=False)

    prompt = _build_prompt(finding, snapshot)

    assert "is_night: True" in prompt
    assert "anyone_home: False" in prompt
    assert "recognized_people_count: 1" in prompt
    assert "last_changed_age_seconds: 120" in prompt
    assert "Alice" not in prompt


def test_prompt_last_changed_age_seconds_non_numeric_excluded() -> None:
    """Non-numeric last_changed_age_seconds must be silently excluded from prompt."""
    evidence = {"last_changed_age_seconds": "not_a_number"}
    finding = _finding(evidence=evidence)
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "last_changed_age_seconds" not in prompt


def test_prompt_recognized_people_non_list_excluded() -> None:
    """Non-list recognized_people must be silently excluded from the prompt."""
    evidence = {"recognized_people": "Alice"}  # string, not a list
    finding = _finding(evidence=evidence)
    snapshot = _snapshot()

    prompt = _build_prompt(finding, snapshot)

    assert "recognized_people_count" not in prompt
    assert "Alice" not in prompt


# ---------------------------------------------------------------------------
# 2. Timeout fails-open to notify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_fails_open_to_notify() -> None:
    """LLM timeout must produce decision=notify with reason_code=triage_timeout."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(MockLLM(timeout=True), timeout_seconds=0)
    result = await svc.triage(finding, snapshot)

    assert isinstance(result, TriageDecision)
    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "triage_timeout"
    assert result.triage_confidence is None


# ---------------------------------------------------------------------------
# 3. LLM error fails-open to notify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_runtime_error_fails_open_to_notify() -> None:
    """RuntimeError from the LLM must fail-open to notify."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(MockLLM(raise_exc=RuntimeError("backend unavailable")))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "triage_error"
    assert result.triage_confidence is None


@pytest.mark.asyncio
async def test_llm_value_error_fails_open_to_notify() -> None:
    """ValueError from the LLM must fail-open to notify."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(MockLLM(raise_exc=ValueError("serialisation error")))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "triage_error"


@pytest.mark.asyncio
async def test_llm_os_error_fails_open_to_notify() -> None:
    """OSError from the LLM must fail-open to notify."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(MockLLM(raise_exc=OSError("connection refused")))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "triage_error"


# ---------------------------------------------------------------------------
# 4. triage_confidence written to TriageDecision; finding.confidence unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_confidence_written_to_decision_not_to_finding() -> None:
    """triage_confidence appears in TriageDecision but must NOT mutate the finding."""
    original_confidence = 0.8
    finding = _finding(confidence=original_confidence)
    snapshot = _snapshot()

    llm_response = {
        "decision": "notify",
        "reason_code": "security_concern",
        "triage_confidence": 0.87,
        "summary": "Suspicious activity detected.",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.triage_confidence == pytest.approx(0.87)
    assert finding.confidence == pytest.approx(original_confidence)


@pytest.mark.asyncio
async def test_triage_confidence_clamped_above_one() -> None:
    """triage_confidence values above 1.0 are clamped to 1.0."""
    finding = _finding()
    snapshot = _snapshot()

    llm_response = {
        "decision": "notify",
        "reason_code": "security_concern",
        "triage_confidence": 1.5,
        "summary": "Very confident.",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.triage_confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_triage_confidence_missing_from_response_is_none() -> None:
    """When triage_confidence is absent from LLM response it must be None."""
    finding = _finding()
    snapshot = _snapshot()

    llm_response = {
        "decision": "notify",
        "reason_code": "security_concern",
        "summary": "Concerning.",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.triage_confidence is None


# ---------------------------------------------------------------------------
# 5. Suppress decision with LLM reason code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suppress_decision_preserves_llm_reason_code() -> None:
    """LLM suppress response must produce decision=suppress with the exact reason_code."""
    finding = _finding()
    snapshot = _snapshot()

    llm_response = {
        "decision": "suppress",
        "reason_code": "routine_state",
        "triage_confidence": 0.95,
        "summary": "Door is routinely opened at this time.",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_SUPPRESS
    assert result.reason_code == "routine_state"
    assert result.triage_confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# 6. LLM returns notify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_notify_response_produces_notify_decision() -> None:
    """LLM notify response must produce decision=notify with correct fields."""
    finding = _finding()
    snapshot = _snapshot()

    llm_response = {
        "decision": "notify",
        "reason_code": "security_concern",
        "triage_confidence": 0.9,
        "summary": "high risk",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "security_concern"
    assert result.triage_confidence == pytest.approx(0.9)
    assert result.summary == "high risk"


# ---------------------------------------------------------------------------
# 7. Disabled triage (model=None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_triage_model_none_returns_notify() -> None:
    """SentinelTriageService(None) must pass all findings through as notify."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(None)
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == "triage_disabled"
    assert result.triage_confidence is None


# ---------------------------------------------------------------------------
# 8. Malformed JSON response fails-open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_json_response_fails_open_to_notify() -> None:
    """Non-JSON LLM response must fail-open and return decision=notify."""
    finding = _finding()
    snapshot = _snapshot()

    class BrokenLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": "This is not JSON at all!"})()

    svc = SentinelTriageService(BrokenLLM())
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY


@pytest.mark.asyncio
async def test_empty_response_fails_open_to_notify() -> None:
    """Empty LLM response must fail-open and return decision=notify."""
    finding = _finding()
    snapshot = _snapshot()

    class EmptyLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": ""})()

    svc = SentinelTriageService(EmptyLLM())
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY


@pytest.mark.asyncio
async def test_markdown_wrapped_json_is_parsed_correctly() -> None:
    """JSON wrapped in markdown code fences must be parsed without error."""
    finding = _finding()
    snapshot = _snapshot()

    fenced_content = (
        "```json\n"
        '{"decision": "suppress", "reason_code": "routine_state", '
        '"triage_confidence": 0.8, "summary": "Routine."}\n'
        "```"
    )

    class FencedLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": fenced_content})()

    svc = SentinelTriageService(FencedLLM())
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_SUPPRESS
    assert result.reason_code == "routine_state"


@pytest.mark.asyncio
async def test_think_block_before_json_is_stripped() -> None:
    """<think> block preceding JSON (qwen3/qwen3.5 output) must be stripped before parsing."""
    finding = _finding()
    snapshot = _snapshot()

    think_content = (
        "<think>evaluating risk level for open entry</think>\n"
        '{"decision": "suppress", "reason_code": "routine_state", '
        '"triage_confidence": 0.9, "summary": "Routine door open."}'
    )

    class ThinkingLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": think_content})()

    svc = SentinelTriageService(ThinkingLLM())
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_SUPPRESS
    assert result.reason_code == "routine_state"
    assert result.triage_confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_unknown_decision_value_defaults_to_notify() -> None:
    """Unrecognised decision tokens in LLM response default to notify."""
    finding = _finding()
    snapshot = _snapshot()

    llm_response = {
        "decision": "maybe",  # not a valid token
        "reason_code": "uncertain",
        "triage_confidence": 0.5,
        "summary": "Not sure.",
    }
    svc = SentinelTriageService(MockLLM(response_json=llm_response))
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY


# ---------------------------------------------------------------------------
# 9. LLM receives correct message structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_receives_system_and_human_messages() -> None:
    """The LLM must be invoked with exactly two messages: SystemMessage then HumanMessage."""
    finding = _finding()
    snapshot = _snapshot()

    llm = MockLLM(
        response_json={
            "decision": "notify",
            "reason_code": "security_concern",
            "triage_confidence": 0.8,
            "summary": "Alert.",
        }
    )
    svc = SentinelTriageService(llm)
    await svc.triage(finding, snapshot)

    assert llm.last_messages is not None
    assert len(llm.last_messages) == 2

    assert isinstance(llm.last_messages[0], SystemMessage)
    assert isinstance(llm.last_messages[1], HumanMessage)

    # The human message must contain the finding type so the LLM has context.
    assert "open_entry_while_away" in llm.last_messages[1].content
