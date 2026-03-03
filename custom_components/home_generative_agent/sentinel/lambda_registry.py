"""
Lambda rule registry with AST validation — Issue #266.

Stores user-submitted lambda/expression rules with a three-state lifecycle:

  pending  → Active after operator approval via ``sentinel_approve_lambda_rule``.
  active   → Evaluated by the sentinel engine.
  rejected → Rejected at receipt (bad AST) or explicitly rejected by operator.

Only ``active`` rules are fed to the engine.  ``pending`` and ``rejected``
rules are never evaluated.

AST Safety
----------
Expressions are validated at receipt using :mod:`ast`.  The allowlist permits
pure data-access and comparison constructs.  Any node type not in the
allowlist — including ``Call``, ``Import``, ``FunctionDef``, and nested
``Lambda`` — causes immediate rejection with an explanatory reason string.
"""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_lambda_registry"

STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"

# ---------------------------------------------------------------------------
# AST allowlist
# ---------------------------------------------------------------------------

# Node *types* allowed in submitted expressions.  Any node whose type is not
# in this set causes immediate rejection.
_ALLOWED_NODE_TYPES: frozenset[type[ast.AST]] = frozenset(
    {
        # Structural
        ast.Expression,
        ast.Module,
        # Literals
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        # Names
        ast.Name,
        ast.Load,
        ast.Store,
        # Attribute access (read-only use; write-side still guarded by eval sandbox)
        ast.Attribute,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        # Expressions
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.Compare,
        # Boolean operators
        ast.And,
        ast.Or,
        ast.Not,
        # Comparison operators
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        # Arithmetic operators (non-mutating)
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        # Statement wrapper for ``ast.parse(expr, mode="eval")``
        ast.Expr,
    }
)

# Node types that are explicitly disallowed (listed for documentation).
_DISALLOWED_NODE_TYPES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Call,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.Lambda,
        ast.GeneratorExp,
        ast.DictComp,
        ast.SetComp,
        ast.ListComp,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Raise,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Assert,
        ast.Pass,
        ast.Break,
        ast.Continue,
    }
)


def _validate_expression(expression: str) -> str | None:
    """
    Parse and walk *expression* for disallowed AST node types.

    Returns ``None`` on success, or a human-readable rejection reason string
    on failure.  Never raises.
    """
    if not isinstance(expression, str) or not expression.strip():
        return "expression must be a non-empty string"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type in _DISALLOWED_NODE_TYPES:
            return f"disallowed AST node type: {node_type.__name__}"
        if node_type not in _ALLOWED_NODE_TYPES:
            return f"unrecognized AST node type: {node_type.__name__}"

    return None


def validate_lambda_expression(expression: str) -> None:
    """
    Validate *expression* for use as a lambda rule condition.

    Raises :class:`~homeassistant.exceptions.HomeAssistantError` if the
    expression fails AST validation.  Call this at rule-receipt time.
    """
    reason = _validate_expression(expression)
    if reason is not None:
        msg = f"Lambda rule rejected: {reason}"
        raise HomeAssistantError(msg)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LambdaRuleRegistry:
    """
    Persist and manage lambda rules with approval lifecycle.

    Each stored rule dict contains at minimum:
    - ``rule_id``: str — unique identifier
    - ``expression``: str — validated Python expression
    - ``status``: ``"pending"`` | ``"active"`` | ``"rejected"``
    - ``created_at``: ISO 8601 UTC timestamp
    - ``approved_at``: ISO 8601 UTC timestamp or ``None``
    - ``severity``: str
    - ``confidence``: float
    - ``is_sensitive``: bool
    - ``suggested_actions``: list[str]
    - ``description``: str (optional, human-readable)
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the registry."""
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._rules: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    async def async_load(self) -> None:
        """Load persisted rules from HA storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Lambda registry: failed to load stored rules.")
            return
        if isinstance(data, list):
            self._rules = list(data)
        LOGGER.debug("Lambda registry loaded %d rule(s).", len(self._rules))

    async def async_save(self) -> None:
        """Persist current rules to HA storage."""
        try:
            await self._store.async_save(self._rules)
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Lambda registry: failed to save rules.")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def list_pending(self) -> list[dict[str, Any]]:
        """Return all rules with status 'pending'."""
        return [r for r in self._rules if r.get("status") == STATUS_PENDING]

    def list_active(self) -> list[dict[str, Any]]:
        """Return all rules with status 'active'."""
        return [r for r in self._rules if r.get("status") == STATUS_ACTIVE]

    def list_all(self) -> list[dict[str, Any]]:
        """Return all rules regardless of status."""
        return list(self._rules)

    def find_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Return a rule by ``rule_id``, or ``None`` if not found."""
        for rule in self._rules:
            if rule.get("rule_id") == rule_id:
                return rule
        return None

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    async def async_receive(self, rule: dict[str, Any]) -> tuple[bool, str]:
        """
        Receive a new lambda rule submission.

        Performs AST validation on ``rule["expression"]``.  On success,
        sets ``status="pending"`` and persists the rule.

        Returns ``(True, "pending")`` on success or
        ``(False, rejection_reason)`` on failure.  Duplicate ``rule_id``
        values are rejected without AST validation.
        """
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            return False, "rule_id is required"

        if self.find_rule(rule_id):
            return False, f"duplicate rule_id: {rule_id}"

        expression = str(rule.get("expression", "")).strip()
        reason = _validate_expression(expression)
        if reason is not None:
            LOGGER.info(
                "Lambda registry rejected rule %s: %s", rule_id, reason
            )
            return False, reason

        stored: dict[str, Any] = {
            "rule_id": rule_id,
            "expression": expression,
            "status": STATUS_PENDING,
            "created_at": _now_iso(),
            "approved_at": None,
            "severity": str(rule.get("severity", "low")),
            "confidence": float(rule.get("confidence", 0.5)),
            "is_sensitive": bool(rule.get("is_sensitive", False)),
            "suggested_actions": list(rule.get("suggested_actions") or []),
            "description": str(rule.get("description", "")),
            # Keep template_id so evaluate_dynamic_rules can dispatch to the
            # lambda evaluator.
            "template_id": "lambda",
        }
        self._rules.append(stored)
        await self.async_save()
        LOGGER.info("Lambda registry: rule %s received, status=pending.", rule_id)
        return True, STATUS_PENDING

    async def async_approve(self, rule_id: str) -> bool:
        """
        Approve a pending rule, transitioning it to ``active``.

        Returns ``True`` if the rule was found and transitioned, ``False``
        otherwise.  Already-active rules return ``True`` without mutation.
        """
        for rule in self._rules:
            if rule.get("rule_id") != rule_id:
                continue
            current_status = rule.get("status")
            if current_status == STATUS_ACTIVE:
                return True
            if current_status != STATUS_PENDING:
                LOGGER.warning(
                    "Lambda registry: cannot approve rule %s in status %s.",
                    rule_id,
                    current_status,
                )
                return False
            rule["status"] = STATUS_ACTIVE
            rule["approved_at"] = _now_iso()
            await self.async_save()
            LOGGER.info("Lambda registry: rule %s approved (active).", rule_id)
            return True
        return False

    async def async_reject(self, rule_id: str) -> bool:
        """
        Explicitly reject a pending rule.

        Returns ``True`` if the rule was found and marked rejected.
        """
        for rule in self._rules:
            if rule.get("rule_id") != rule_id:
                continue
            rule["status"] = STATUS_REJECTED
            await self.async_save()
            LOGGER.info("Lambda registry: rule %s rejected.", rule_id)
            return True
        return False

    async def async_remove(self, rule_id: str) -> bool:
        """Remove a rule from the registry entirely."""
        for idx, rule in enumerate(self._rules):
            if rule.get("rule_id") == rule_id:
                self._rules.pop(idx)
                await self.async_save()
                return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
