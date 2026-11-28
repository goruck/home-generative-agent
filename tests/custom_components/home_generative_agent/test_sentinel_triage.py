# ruff: noqa: S101
"""Tests for SentinelTriageService — sentinel/triage.py (Issue #262)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.const import (
    SENTINEL_TRIAGE_BATTERY_FLOOR_PCT,
)
from custom_components.home_generative_agent.core.utils import SentinelLLMDeferredError
from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.triage import (
    TRIAGE_NOTIFY,
    TRIAGE_REASON_BATTERY_FLOOR,
    TRIAGE_REASON_CAP,
    TRIAGE_REASON_DEFERRED,
    TRIAGE_REASON_LLM_NOTIFY,
    TRIAGE_REASON_LLM_SUPPRESS,
    TRIAGE_REASONS_NOT_RUN,
    TRIAGE_SUPPRESS,
    SentinelTriageService,
    TriageDecision,
    _build_prompt,
    triage_memo_key,
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
        timeout: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.response_json = response_json
        self.raise_exc = raise_exc
        self.timeout = timeout
        self.last_messages: list[Any] | None = None
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.last_messages = messages
        self.calls += 1
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
    is_sensitive: bool = False,  # noqa: FBT001, FBT002
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


def _snapshot(is_night: bool = False, anyone_home: bool = False) -> FullStateSnapshot:  # noqa: FBT001, FBT002
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


def test_prompt_counts_enrolled_only_and_flags_unknown_person() -> None:
    """
    Reserved labels must not inflate the count the triage LLM reasons on.

    An unknown-person finding always carries "Unknown Person" (often with
    "Indeterminate"); counting those would read as "a known person was
    recognized" and invite the LLM to suppress the stranger alert. The
    stranger signal is surfaced explicitly instead.
    """
    evidence = {"recognized_people": ["Indeterminate", "Unknown Person"]}
    finding = _finding(ftype="unknown_person_camera_no_home", evidence=evidence)
    snapshot = _snapshot(is_night=True, anyone_home=False)

    prompt = _build_prompt(finding, snapshot)

    assert "recognized_people_count: 0" in prompt
    assert "unknown_person_present: true" in prompt


def test_prompt_no_unknown_person_flag_for_enrolled_only() -> None:
    """The stranger flag appears only when the label is actually present."""
    evidence = {"recognized_people": ["Alice"]}
    finding = _finding(evidence=evidence)
    snapshot = _snapshot(is_night=True, anyone_home=False)

    prompt = _build_prompt(finding, snapshot)

    assert "recognized_people_count: 1" in prompt
    assert "unknown_person_present" not in prompt


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


def _battery_finding(
    levels: dict[str, Any],
    triggering: list[str],
    threshold: Any = 40.0,
    template_id: str = "low_battery_sensors",
) -> AnomalyFinding:
    # Shaped as dynamic_rules builds it: typed by rule id, template in evidence.
    return AnomalyFinding(
        anomaly_id="battery-id",
        type="low_battery_rule",
        severity="medium",
        confidence=0.8,
        triggering_entities=triggering,
        evidence={
            "rule_id": "low_battery_rule",
            "template_id": template_id,
            "sensor_entity_ids": list(levels),
            "sensor_states": {k: str(v) for k, v in levels.items()},
            "sensor_levels": levels,
            "threshold": threshold,
        },
        suggested_actions=[],
        is_sensitive=False,
    )


def test_low_battery_prompt_from_a_real_rule_finding() -> None:
    """A finding built by the rule evaluator carries the level (#1085)."""
    snapshot = _snapshot()
    snapshot["entities"] = [
        {
            "entity_id": "sensor.front_lock_battery",
            "domain": "sensor",
            "state": "3",
            "friendly_name": "Front lock battery",
            "area": None,
            "attributes": {"device_class": "battery", "unit_of_measurement": "%"},
            "last_changed": "2025-01-01T10:00:00+00:00",
            "last_updated": "2025-01-01T10:00:00+00:00",
        }
    ]
    rule = {
        "rule_id": "low_battery_front_lock_v1",
        "template_id": "low_battery_sensors",
        "params": {"sensor_entity_ids": ["sensor.front_lock_battery"], "threshold": 40},
        "severity": "medium",
        "confidence": 0.7,
    }
    findings = evaluate_dynamic_rules(snapshot, [rule])
    assert len(findings) == 1
    assert findings[0].type == "low_battery_front_lock_v1"

    prompt = _build_prompt(findings[0], snapshot)

    assert "battery_level_pct: 0" in prompt
    assert "battery_threshold_pct: 40" in prompt
    assert "front_lock" not in prompt.split("evidence:")[1]


def test_low_battery_prompt_carries_level_and_threshold() -> None:
    """The model is told how low the lowest triggering battery is (#1085)."""
    finding = _battery_finding(
        {"sensor.lock_battery": 3.0, "sensor.door_battery": 38.0},
        ["sensor.lock_battery", "sensor.door_battery"],
    )
    prompt = _build_prompt(finding, _snapshot())

    assert "battery_level_pct: 0" in prompt
    assert "battery_threshold_pct: 40" in prompt
    assert "lock_battery" not in prompt
    assert "door_battery" not in prompt


def test_low_battery_level_ignores_sensors_above_threshold() -> None:
    """Only triggering sensors count toward the reported level."""
    finding = _battery_finding(
        {"sensor.a": 37.0, "sensor.b": 2.0},
        ["sensor.a"],
    )
    prompt = _build_prompt(finding, _snapshot())

    assert "battery_level_pct: 35" in prompt


def test_low_battery_small_drop_keeps_the_triage_memo() -> None:
    """A 1% lower reading is the same question; a much lower one is new."""
    snapshot = _snapshot()
    at_39 = _battery_finding({"sensor.a": 39.0}, ["sensor.a"])
    at_36 = _battery_finding({"sensor.a": 36.0}, ["sensor.a"])
    at_3 = _battery_finding({"sensor.a": 3.0}, ["sensor.a"])

    assert triage_memo_key(at_39, snapshot) == triage_memo_key(at_36, snapshot)
    assert triage_memo_key(at_39, snapshot) != triage_memo_key(at_3, snapshot)


@pytest.mark.parametrize(
    ("level", "threshold"),
    [(float("nan"), float("inf")), (True, "40"), ("12", None)],
)
def test_low_battery_non_finite_or_non_numeric_values_are_left_out(
    level: Any, threshold: Any
) -> None:
    """Only finite numbers reach the prompt."""
    finding = _battery_finding({"sensor.a": level}, ["sensor.a"], threshold)
    prompt = _build_prompt(finding, _snapshot())

    assert "battery_level_pct" not in prompt
    assert "battery_threshold_pct" not in prompt


def test_low_battery_out_of_range_values_are_bounded() -> None:
    """A reading or threshold outside 0-100 is clamped."""
    finding = _battery_finding({"sensor.a": -7.0}, ["sensor.a"], 250.0)
    prompt = _build_prompt(finding, _snapshot())

    assert "battery_level_pct: 0" in prompt
    assert "battery_threshold_pct: 100" in prompt


def test_battery_fields_only_for_low_battery_findings() -> None:
    """Another rule's threshold or levels never reach the prompt."""
    finding = _battery_finding(
        {"sensor.a": 5.0}, ["sensor.a"], template_id="sensor_threshold_condition"
    )
    prompt = _build_prompt(finding, _snapshot())

    assert "battery_level_pct" not in prompt
    assert "battery_threshold_pct" not in prompt


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


@pytest.mark.parametrize(
    "payload",
    ["[]", "null", '"notify"', "42", "true"],
)
@pytest.mark.asyncio
async def test_valid_json_that_is_not_an_object_fails_open(payload: str) -> None:
    """
    JSON that decodes to a non-mapping must fail open, not raise.

    ``json.loads`` accepts all of these, so the decode succeeds and the old code
    then called ``.get()`` on a list/None/str and raised ``AttributeError``.
    That is neither ``JSONDecodeError`` nor ``ValueError``, so it escaped the
    parser's handler *and* ``triage()``'s fail-open block, propagated through
    ``_dispatch_item`` -> ``_run_once`` -> ``_timed_run`` -> ``_run_loop``
    (none of which catch), and killed the engine task. ``start()`` then refused
    to replace the non-``None`` task, so Sentinel stayed dead until the config
    entry was reloaded -- from one bad model response.
    """
    finding = _finding()
    snapshot = _snapshot()

    class NonObjectLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": payload})()

    svc = SentinelTriageService(NonObjectLLM())
    result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY


@pytest.mark.asyncio
async def test_parse_failure_cannot_escape_the_fail_open_boundary() -> None:
    """Any parser exception must be contained, not propagated to the caller."""
    finding = _finding()
    snapshot = _snapshot()

    class OkLLM:
        async def ainvoke(self, messages: list[Any]) -> Any:
            return type("Result", (), {"content": '{"decision": "notify"}'})()

    svc = SentinelTriageService(OkLLM())
    with patch(
        "custom_components.home_generative_agent.sentinel.triage._parse_response",
        side_effect=AttributeError("boom"),
    ):
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
@pytest.mark.parametrize(
    "reasoning",
    [
        "<think>evaluating risk level for open entry</think>\n",
        # The template opened <think>, so only the close is in the output (#1017).
        "evaluating risk level for open entry\n</think>\n",
    ],
)
async def test_think_block_before_json_is_stripped(reasoning: str) -> None:
    """<think> block preceding JSON (qwen3/qwen3.5 output) must be stripped before parsing."""
    finding = _finding()
    snapshot = _snapshot()

    think_content = (
        reasoning + '{"decision": "suppress", "reason_code": "routine_state", '
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


# ---------------------------------------------------------------------------
# 10. Deferred triage when chat is active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_deferred_when_chat_active() -> None:
    """SentinelLLMDeferredError from the model call must produce a deferred decision."""
    finding = _finding()
    snapshot = _snapshot()

    svc = SentinelTriageService(MockLLM())
    with patch(
        "custom_components.home_generative_agent.sentinel.triage.run_sentinel_model_call",
        side_effect=SentinelLLMDeferredError("triage", "chat is active"),
    ):
        result = await svc.triage(finding, snapshot)

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == TRIAGE_REASON_DEFERRED
    assert result.triage_confidence is None


# ---------------------------------------------------------------------------
# 11. Per-run call budget on a shared model (#866)
# ---------------------------------------------------------------------------

_SUPPRESS_REPLY = {
    "decision": "suppress",
    "reason_code": "routine",
    "triage_confidence": 0.9,
    "summary": "Routine.",
}


@pytest.mark.asyncio
async def test_findings_past_the_run_limit_pass_through_untriaged() -> None:
    """A run sends at most max_per_run findings to the model; the rest notify."""
    llm = MockLLM(_SUPPRESS_REPLY)
    health: dict[str, Any] = {}
    svc = SentinelTriageService(llm, max_per_run=2, call_gap_s=0.0, health_stats=health)
    svc.begin_run()

    results = [await svc.triage(_finding(), _snapshot()) for _ in range(4)]

    assert [r.decision for r in results] == [
        TRIAGE_SUPPRESS,
        TRIAGE_SUPPRESS,
        TRIAGE_NOTIFY,
        TRIAGE_NOTIFY,
    ]
    assert results[2].reason_code == results[3].reason_code == TRIAGE_REASON_CAP
    assert results[2].triage_confidence is None
    assert llm.calls == 2
    assert health["triage_calls_last_run"] == 2
    assert health["triage_passed_through_last_run"] == 2
    assert TRIAGE_REASON_CAP in TRIAGE_REASONS_NOT_RUN


@pytest.mark.asyncio
async def test_begin_run_renews_the_call_budget() -> None:
    """The next run triages again, and the health counters start from zero."""
    llm = MockLLM(_SUPPRESS_REPLY)
    health: dict[str, Any] = {}
    svc = SentinelTriageService(llm, max_per_run=1, call_gap_s=0.0, health_stats=health)
    svc.begin_run()
    await svc.triage(_finding(), _snapshot())
    capped = await svc.triage(_finding(), _snapshot())
    assert capped.reason_code == TRIAGE_REASON_CAP

    svc.begin_run()
    assert health["triage_calls_last_run"] == 0
    assert health["triage_passed_through_last_run"] == 0
    again = await svc.triage(_finding(), _snapshot())

    assert again.decision == TRIAGE_SUPPRESS
    assert again.reason_code == "routine"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_a_limit_below_one_still_triages_exactly_one_finding() -> None:
    """The limit is clamped to at least one call per run."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=0, call_gap_s=0.0)
    svc.begin_run()
    first = await svc.triage(_finding(), _snapshot())
    second = await svc.triage(_finding(), _snapshot())
    assert first.decision == TRIAGE_SUPPRESS
    assert second.reason_code == TRIAGE_REASON_CAP
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_consecutive_calls_in_a_run_are_spaced_by_the_gap() -> None:
    """The first call starts at once; the second waits out the gap."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=5, call_gap_s=0.2)
    svc.begin_run()

    await svc.triage(_finding(), _snapshot())
    first_done = time.monotonic()
    await svc.triage(_finding(), _snapshot())
    second_done = time.monotonic()

    # gstack-timing-ok: lower bound only; a slow runner can only lengthen the gap
    assert second_done - first_done >= 0.19
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_the_gap_also_spaces_the_first_call_of_the_next_run() -> None:
    """An event run right after a poll run does not restart the burst."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=5, call_gap_s=0.2)
    svc.begin_run()
    await svc.triage(_finding(), _snapshot())
    first_done = time.monotonic()

    svc.begin_run()
    await svc.triage(_finding(), _snapshot())
    second_done = time.monotonic()

    # gstack-timing-ok: lower bound only; a slow runner can only lengthen the gap
    assert second_done - first_done >= 0.19
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_a_deferred_call_stops_triage_for_the_rest_of_the_run() -> None:
    """Once chat took the model, no further call is started that run."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=5, call_gap_s=0.0)
    svc.begin_run()

    with patch(
        "custom_components.home_generative_agent.sentinel.triage.run_sentinel_model_call",
        side_effect=SentinelLLMDeferredError("triage", "chat is active"),
    ) as call:
        first = await svc.triage(_finding(), _snapshot())
        second = await svc.triage(_finding(), _snapshot())

    assert first.reason_code == second.reason_code == TRIAGE_REASON_DEFERRED
    assert first.decision == second.decision == TRIAGE_NOTIFY
    assert call.call_count == 1
    # The model made no decision for either: both are pass-throughs.
    assert svc._run_calls == 0
    assert svc._run_skipped == 2

    # The next run tries the model again.
    svc.begin_run()
    result = await svc.triage(_finding(), _snapshot())
    assert result.decision == TRIAGE_SUPPRESS
    assert result.reason_code == "routine"


@pytest.mark.asyncio
async def test_an_llm_suppress_reply_is_counted_as_a_call() -> None:
    """A suppress with an empty reason code reads llm_suppress, as before."""
    llm = MockLLM({"decision": "suppress", "reason_code": ""})
    svc = SentinelTriageService(llm, call_gap_s=0.0)
    svc.begin_run()
    result = await svc.triage(_finding(), _snapshot())
    assert result.reason_code == TRIAGE_REASON_LLM_SUPPRESS
    assert svc._run_calls == 1
    assert svc._run_skipped == 0


@pytest.mark.asyncio
async def test_an_admission_deferred_finding_is_a_pass_through_not_a_call() -> None:
    """The health counters match the reason code: the model saw nothing."""
    health: dict[str, Any] = {}
    svc = SentinelTriageService(
        MockLLM(_SUPPRESS_REPLY), max_per_run=5, call_gap_s=0.0, health_stats=health
    )
    svc.begin_run()
    with patch(
        "custom_components.home_generative_agent.sentinel.triage.run_sentinel_model_call",
        side_effect=SentinelLLMDeferredError("triage", "deferred; chat is active"),
    ):
        result = await svc.triage(_finding(), _snapshot())

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == TRIAGE_REASON_DEFERRED
    assert health["triage_calls_last_run"] == 0
    assert health["triage_passed_through_last_run"] == 1


@pytest.mark.asyncio
async def test_a_model_reply_cannot_fake_a_deferral() -> None:
    """A reserved reason code from the model is the model's decision, not a deferral."""
    llm = MockLLM({"decision": "notify", "reason_code": "triage_deferred"})
    svc = SentinelTriageService(llm, max_per_run=5, call_gap_s=0.0)
    svc.begin_run()
    first = await svc.triage(_finding(), _snapshot())
    second = await svc.triage(_finding(), _snapshot())

    assert llm.calls == 2  # the second finding still reached the model
    assert first.reason_code == second.reason_code == TRIAGE_REASON_LLM_NOTIFY
    assert svc._run_calls == 2
    assert svc._run_deferred is False


@pytest.mark.asyncio
async def test_a_reserved_suppress_code_keeps_the_suppress_decision() -> None:
    """Normalising the reason code never changes what the model decided."""
    llm = MockLLM({"decision": "suppress", "reason_code": "triage_cap"})
    svc = SentinelTriageService(llm, call_gap_s=0.0)
    svc.begin_run()
    result = await svc.triage(_finding(), _snapshot())
    assert result.decision == TRIAGE_SUPPRESS
    assert result.reason_code == TRIAGE_REASON_LLM_SUPPRESS


@pytest.mark.asyncio
async def test_a_call_interrupted_in_flight_stays_counted_as_a_call() -> None:
    """The model got the request, so the load counter keeps it; the run still stops."""
    health: dict[str, Any] = {}
    svc = SentinelTriageService(
        MockLLM(_SUPPRESS_REPLY), max_per_run=5, call_gap_s=0.0, health_stats=health
    )
    svc.begin_run()
    with patch(
        "custom_components.home_generative_agent.sentinel.triage.run_sentinel_model_call",
        side_effect=SentinelLLMDeferredError(
            "triage", "interrupted by foreground chat", started=True
        ),
    ):
        result = await svc.triage(_finding(), _snapshot())

    assert result.reason_code == TRIAGE_REASON_DEFERRED
    assert health["triage_calls_last_run"] == 1
    assert health["triage_passed_through_last_run"] == 1
    assert svc._run_deferred is True


@pytest.mark.asyncio
async def test_a_cloud_model_is_never_capped_or_gapped() -> None:
    """The limit and the gap exist for a GPU shared with chat; a cloud model has none."""
    import custom_components.home_generative_agent.core.utils as utils_mod  # noqa: PLC0415

    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=1, call_gap_s=0.5, deployment="cloud")
    svc.begin_run()
    utils_mod._sentinel_last_call_end = time.monotonic()
    start = time.monotonic()
    first = await svc.triage(_finding(), _snapshot())
    second = await svc.triage(_finding(), _snapshot())

    assert first.decision == second.decision == TRIAGE_SUPPRESS
    assert llm.calls == 2
    # gstack-timing-ok: the alternative is a 0.5 s gap; 0.3 s is a generous bound
    assert time.monotonic() - start < 0.3


# ---------------------------------------------------------------------------
# Low-battery floor (#1085)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [0.0, 3.0, SENTINEL_TRIAGE_BATTERY_FLOOR_PCT])
async def test_a_battery_at_or_below_the_floor_notifies_without_the_model(
    level: float,
) -> None:
    """A nearly empty battery is never sent to triage, so it cannot be suppressed."""
    llm = MockLLM(_SUPPRESS_REPLY)
    health: dict[str, Any] = {}
    svc = SentinelTriageService(llm, call_gap_s=0.0, health_stats=health)
    svc.begin_run()

    result = await svc.triage(
        _battery_finding({"sensor.a": level}, ["sensor.a"]), _snapshot()
    )

    assert result.decision == TRIAGE_NOTIFY
    assert result.reason_code == TRIAGE_REASON_BATTERY_FLOOR
    assert result.triage_confidence is None
    assert llm.calls == 0
    assert health["triage_calls_last_run"] == 0
    assert health["triage_passed_through_last_run"] == 0
    # The health sensor leaves it out of the suppress rate; a model may not
    # claim it.
    assert TRIAGE_REASON_BATTERY_FLOOR in TRIAGE_REASONS_NOT_RUN


@pytest.mark.asyncio
async def test_a_battery_above_the_floor_is_still_triaged() -> None:
    """Above the floor the model decides, and may suppress."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, call_gap_s=0.0)
    svc.begin_run()

    result = await svc.triage(
        _battery_finding(
            {"sensor.a": SENTINEL_TRIAGE_BATTERY_FLOOR_PCT + 1}, ["sensor.a"]
        ),
        _snapshot(),
    )

    assert result.decision == TRIAGE_SUPPRESS
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_the_floor_uses_the_lowest_triggering_sensor_only() -> None:
    """A low sensor that did not trigger the finding does not skip triage."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, call_gap_s=0.0)
    svc.begin_run()

    result = await svc.triage(
        _battery_finding({"sensor.a": 30.0, "sensor.b": 2.0}, ["sensor.a"]),
        _snapshot(),
    )

    assert result.decision == TRIAGE_SUPPRESS
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_a_floor_battery_notifies_past_the_run_limit_and_uses_no_slot() -> None:
    """The floor is checked before the run's limit and does not spend it."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, max_per_run=1, call_gap_s=0.0)
    svc.begin_run()

    floor = await svc.triage(
        _battery_finding({"sensor.a": 2.0}, ["sensor.a"]), _snapshot()
    )
    other = await svc.triage(_finding(), _snapshot())

    assert floor.reason_code == TRIAGE_REASON_BATTERY_FLOOR
    assert other.decision == TRIAGE_SUPPRESS
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_the_floor_applies_only_to_low_battery_findings() -> None:
    """Another template's low reading is triaged as before."""
    llm = MockLLM(_SUPPRESS_REPLY)
    svc = SentinelTriageService(llm, call_gap_s=0.0)
    svc.begin_run()

    result = await svc.triage(
        _battery_finding(
            {"sensor.a": 1.0}, ["sensor.a"], template_id="sensor_threshold_condition"
        ),
        _snapshot(),
    )

    assert result.decision == TRIAGE_SUPPRESS
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_triage_off_still_reports_disabled_for_a_floor_battery() -> None:
    """With triage off the decision stays triage_disabled."""
    svc = SentinelTriageService(None)

    result = await svc.triage(
        _battery_finding({"sensor.a": 1.0}, ["sensor.a"]), _snapshot()
    )

    assert result.reason_code == "triage_disabled"
