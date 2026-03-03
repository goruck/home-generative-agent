# ruff: noqa: S101
"""Tests for LambdaRuleRegistry and AST validation — Issue #266."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_generative_agent.sentinel.dynamic_rules import (
    evaluate_dynamic_rules,
)
from custom_components.home_generative_agent.sentinel.lambda_registry import (
    STATUS_ACTIVE,
    STATUS_PENDING,
    STATUS_REJECTED,
    LambdaRuleRegistry,
    validate_lambda_expression,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hass() -> tuple[MagicMock, MagicMock]:
    hass = MagicMock()
    store_mock = MagicMock()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)
    with patch(
        "custom_components.home_generative_agent.sentinel.lambda_registry.Store",
        return_value=store_mock,
    ):
        return hass, store_mock


async def _make_registry() -> tuple[LambdaRuleRegistry, MagicMock]:
    hass, store_mock = _make_hass()
    with patch(
        "custom_components.home_generative_agent.sentinel.lambda_registry.Store",
        return_value=store_mock,
    ):
        registry = LambdaRuleRegistry(hass)
    return registry, store_mock


def _base_rule(rule_id: str = "r1", expression: str = "1 == 1") -> dict:
    return {
        "rule_id": rule_id,
        "expression": expression,
        "severity": "low",
        "confidence": 0.5,
        "is_sensitive": False,
        "suggested_actions": [],
        "description": "test rule",
    }


def _minimal_snapshot() -> FullStateSnapshot:
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": "2025-01-01T00:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2025-01-01T00:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": False,
                "people_home": [],
                "people_away": [],
                "last_motion_by_area": {},
            },
        }
    )


# ---------------------------------------------------------------------------
# AST validation tests
# ---------------------------------------------------------------------------


def test_ast_accept_simple_comparison() -> None:
    """Simple comparison expression passes validation."""
    validate_lambda_expression("x == 1")


def test_ast_accept_boolean_ops() -> None:
    """Boolean operators are allowed."""
    validate_lambda_expression("x > 0 and y < 10")


def test_ast_accept_attribute_access() -> None:
    """Attribute access is allowed."""
    validate_lambda_expression("entities['lock.front_door']['state'] == 'locked'")


def test_ast_accept_arithmetic() -> None:
    """Arithmetic operators are allowed."""
    validate_lambda_expression("x + y > 5")


def test_ast_reject_function_call() -> None:
    """Expression with function call is rejected."""
    with pytest.raises(HomeAssistantError, match="disallowed AST node type"):
        validate_lambda_expression("os.system('rm -rf /')")


def test_ast_reject_import() -> None:
    """Expression with __import__ call is rejected."""
    with pytest.raises(HomeAssistantError, match="disallowed AST node type"):
        validate_lambda_expression("__import__('os')")


def test_ast_reject_nested_lambda() -> None:
    """Expression with nested lambda is rejected."""
    with pytest.raises(HomeAssistantError, match="disallowed AST node type"):
        validate_lambda_expression("(lambda x: x)(1)")


def test_ast_reject_list_comprehension() -> None:
    """List comprehension is rejected."""
    with pytest.raises(HomeAssistantError, match="disallowed AST node type"):
        validate_lambda_expression("[x for x in range(10)]")


def test_ast_reject_syntax_error() -> None:
    """Syntax error is rejected."""
    with pytest.raises(HomeAssistantError, match="syntax error"):
        validate_lambda_expression("x ==")


def test_ast_reject_empty_expression() -> None:
    """Empty expression is rejected."""
    with pytest.raises(HomeAssistantError, match="non-empty string"):
        validate_lambda_expression("   ")


# ---------------------------------------------------------------------------
# LambdaRuleRegistry lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_valid_rule_sets_pending() -> None:
    """Submitting a valid rule sets status=pending."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    ok, reason = await registry.async_receive(_base_rule())

    assert ok is True
    assert reason == STATUS_PENDING
    rules = registry.list_pending()
    assert len(rules) == 1
    assert rules[0]["rule_id"] == "r1"
    assert rules[0]["status"] == STATUS_PENDING
    assert rules[0]["template_id"] == "lambda"


@pytest.mark.asyncio
async def test_receive_invalid_expression_returns_false() -> None:
    """Submitting a rule with a disallowed expression is rejected at receipt."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    ok, reason = await registry.async_receive(
        _base_rule(expression="os.system('id')")
    )

    assert ok is False
    assert "disallowed" in reason
    assert registry.list_pending() == []


@pytest.mark.asyncio
async def test_receive_duplicate_rule_id_rejected() -> None:
    """Duplicate rule_id is rejected without AST validation."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(_base_rule())
    ok, reason = await registry.async_receive(_base_rule())

    assert ok is False
    assert "duplicate" in reason


@pytest.mark.asyncio
async def test_approve_transitions_pending_to_active() -> None:
    """Approving a pending rule moves it to active."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(_base_rule())
    assert registry.list_active() == []

    ok = await registry.async_approve("r1")

    assert ok is True
    active = registry.list_active()
    assert len(active) == 1
    assert active[0]["status"] == STATUS_ACTIVE
    assert active[0]["approved_at"] is not None
    assert registry.list_pending() == []


@pytest.mark.asyncio
async def test_approve_nonexistent_rule_returns_false() -> None:
    """Approving an unknown rule_id returns False."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    ok = await registry.async_approve("does_not_exist")

    assert ok is False


@pytest.mark.asyncio
async def test_reject_transitions_pending_to_rejected() -> None:
    """Explicitly rejecting a rule sets status=rejected."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(_base_rule())
    ok = await registry.async_reject("r1")

    assert ok is True
    rule = registry.find_rule("r1")
    assert rule is not None
    assert rule["status"] == STATUS_REJECTED
    assert registry.list_pending() == []
    assert registry.list_active() == []


@pytest.mark.asyncio
async def test_remove_deletes_rule() -> None:
    """Removing a rule deletes it entirely."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(_base_rule())
    ok = await registry.async_remove("r1")

    assert ok is True
    assert registry.find_rule("r1") is None
    assert registry.list_all() == []


# ---------------------------------------------------------------------------
# Integration with evaluate_dynamic_rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_rule_skipped_by_dynamic_evaluator() -> None:
    """Pending lambda rules produce no findings when evaluated."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(_base_rule(expression="True"))
    pending_rules = registry.list_pending()

    snapshot = _minimal_snapshot()
    findings = evaluate_dynamic_rules(snapshot, pending_rules)

    assert findings == []


@pytest.mark.asyncio
async def test_active_truthy_lambda_rule_produces_finding() -> None:
    """An active lambda rule whose expression evaluates truthy produces a finding."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(
        _base_rule(rule_id="active_r1", expression="True")
    )
    await registry.async_approve("active_r1")

    active_rules = registry.list_active()
    snapshot = _minimal_snapshot()
    findings = evaluate_dynamic_rules(snapshot, active_rules)

    assert len(findings) == 1
    assert findings[0].type == "active_r1"


@pytest.mark.asyncio
async def test_active_falsy_lambda_rule_produces_no_findings() -> None:
    """An active lambda rule whose expression evaluates falsy produces no findings."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    await registry.async_receive(
        _base_rule(rule_id="falsy_r1", expression="False")
    )
    await registry.async_approve("falsy_r1")

    active_rules = registry.list_active()
    snapshot = _minimal_snapshot()
    findings = evaluate_dynamic_rules(snapshot, active_rules)

    assert findings == []


@pytest.mark.asyncio
async def test_lambda_rule_cannot_call_builtins() -> None:
    """Lambda rule expression with __import__ is blocked at AST validation."""
    registry, store_mock = await _make_registry()
    store_mock.async_load = AsyncMock(return_value=None)
    store_mock.async_save = AsyncMock(return_value=None)

    ok, _reason = await registry.async_receive(
        _base_rule(rule_id="bad_r1", expression="__import__('os')")
    )

    assert ok is False
    assert registry.list_pending() == []
