"""
Sentinel execution service — Issue #259.

Provides ``SentinelExecutionService``, which evaluates the action-policy gate
for each finding.  All auto-execution guardrails live here so that both the
mobile-action flow and the autonomous-execution flow share a single policy
evaluation path.

Guardrail evaluation order
--------------------------
1. Autonomy level < 2                 → ``prompt_user``
2. Data quality != "fresh"            → ``prompt_user`` (stale) /
                                        ``blocked`` (unavailable)
3. Sensitivity flag                   → ``handoff``
4. Auto-execution disabled            → ``prompt_user`` (the canary
                                        evaluator still runs on in canary mode)
5. Confidence below threshold         → ``prompt_user``
6. Actions: none callable, one not
   allowlisted, or one critical (a
   PIN rule, a script/scene/automation
   it starts, an alarm panel or cover,
   a lock action but lock.lock)       → ``prompt_user``
7. Rate limit exceeded                → ``prompt_user``
8. Idempotency duplicate              → ``prompt_user``
9. All checks passed                  → ``auto_execute``
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_BLOCKED,
    ACTION_POLICY_HANDOFF,
    ACTION_POLICY_PROMPT_USER,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
    DATA_QUALITY_FRESH,
    DATA_QUALITY_STALE,
    DATA_QUALITY_UNAVAILABLE,
    RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
    RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
    RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
    RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
    RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)

# Minimum autonomy level required for auto-execution.
_MIN_AUTO_EXECUTE_LEVEL: int = 2
# A suggested action Sentinel can call; anything else is advice to the user.
_SERVICE_ACTION = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")
# Domains Sentinel never acts on by itself, whatever the rules or allowlist:
# an alarm panel guards itself with its own code, and a cover can be a door,
# gate or garage whose entity id names none of them.
_NEVER_AUTO_DOMAINS = frozenset({"alarm_control_panel", "cover"})
# The only lock service Sentinel may call by itself.
_SAFE_LOCK_SERVICE = "lock.lock"


def executable_actions(finding: AnomalyFinding) -> list[str]:
    """Return the suggested actions Sentinel can call: ``domain.service`` ones."""
    return [a for a in finding.suggested_actions if _SERVICE_ACTION.fullmatch(a)]


def level_increase_pin_required(options: Mapping[str, Any]) -> bool:
    """
    Return whether raising the autonomy level needs the PIN.

    A stored PIN hash turns the gate on even with the flag off: before #762, a
    PIN saved from Sentinel setup could be stored without the flag.
    """
    return bool(
        options.get(
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
            RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
        )
    ) or bool(options.get(CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH))


@dataclass(frozen=True)
class ActionPolicyResult:
    """Result of action-policy evaluation for a single finding."""

    action_policy_path: str
    data_quality: str
    data_quality_details: dict[str, Any]
    execution_id: str | None
    block_reason: str | None


class SentinelExecutionService:
    """
    Stateful execution policy evaluator for Sentinel.

    Holds in-memory state for rate limiting and idempotency.  A single
    instance should be created per config entry and reused across engine
    cycles.
    """

    def __init__(self, options: dict[str, Any]) -> None:
        """Initialize with the resolved runtime options dict."""
        self._options = options
        # UTC datetimes of recent successful auto-execute approvals (for rate limiting).
        self._recent_action_times: list[datetime] = []
        # Idempotency keys seen within the current time window.
        self._idempotency_seen: set[str] = set()
        # Monotonic timestamp of last idempotency window reset.
        self._idempotency_window_reset_at: float = time.monotonic()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def commit_auto_execute(self, execution_id: str, now: datetime) -> None:
        """
        Record a successful live auto-execution after the service call completes.

        This is used by the engine's live path so rate-limit and idempotency
        state reflect actual completed actions rather than optimistic approval.
        """
        window_start = now - timedelta(hours=1)
        self._recent_action_times = [
            t for t in self._recent_action_times if t > window_start
        ]
        self._refresh_idempotency_window()
        self._recent_action_times.append(now)
        self._idempotency_seen.add(execution_id)
        LOGGER.info("Recorded auto-execute completion (execution_id=%s).", execution_id)

    def evaluate(  # noqa: PLR0911
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        autonomy_level: int,
        now: datetime,
    ) -> ActionPolicyResult:
        """
        Evaluate execution policy for *finding* and return an ``ActionPolicyResult``.

        The returned ``action_policy_path`` is one of:

        - ``"prompt_user"`` — notify and ask the user to decide
        - ``"handoff"``    — delegate to conversation agent
        - ``"auto_execute"`` — all guardrails passed; execute automatically
        - ``"blocked"``    — finding is suppressed at the execution layer

        This method is **not** a coroutine; it performs no I/O.
        """
        # 1. Autonomy level gate — auto-execute requires level 2+.
        if autonomy_level < _MIN_AUTO_EXECUTE_LEVEL:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=DATA_QUALITY_FRESH,
                data_quality_details={},
                execution_id=None,
                block_reason="autonomy_level_below_2",
            )

        # 2. Data quality gate.
        dq, dq_details = self._compute_data_quality(finding, snapshot, now)
        if dq == DATA_QUALITY_UNAVAILABLE:
            high_sev = finding.severity == "high"
            path = ACTION_POLICY_PROMPT_USER if high_sev else ACTION_POLICY_BLOCKED
            return ActionPolicyResult(
                action_policy_path=path,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="data_quality_unavailable",
            )
        if dq == DATA_QUALITY_STALE:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="data_quality_stale",
            )

        # 3. Sensitivity gate.
        if finding.is_sensitive:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_HANDOFF,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="finding_sensitive",
            )

        # 4. Auto-execution enabled gate.
        auto_enabled = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
                RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
            )
        )
        if not auto_enabled:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="auto_execution_disabled",
            )

        # 5. Confidence gate.
        min_confidence = float(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
            )
        )
        if finding.confidence < min_confidence:
            LOGGER.debug(
                "Auto-execute blocked for %s: confidence %.2f < threshold %.2f.",
                finding.anomaly_id,
                finding.confidence,
                min_confidence,
            )
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="confidence_below_threshold",
            )

        # 6. Action gate: something to run, allowlisted, never critical.
        if (block_reason := self._actions_block_reason(finding)) is not None:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason=block_reason,
            )

        # 7. Rate limit gate.
        if self._is_rate_limited(now):
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="rate_limit_exceeded",
            )

        # 8. Idempotency gate.
        policy_version = self._compute_policy_version(autonomy_level, min_confidence)
        execution_id = self._compute_execution_id(
            finding.anomaly_id, ACTION_POLICY_AUTO_EXECUTE, policy_version
        )
        self._refresh_idempotency_window()
        if execution_id in self._idempotency_seen:
            LOGGER.debug(
                "Auto-execute blocked for %s: idempotency duplicate (execution_id=%s).",
                finding.anomaly_id,
                execution_id,
            )
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=execution_id,
                block_reason="idempotency_duplicate",
            )

        # All guardrails passed — approve auto-execute.
        self._recent_action_times.append(now)
        self._idempotency_seen.add(execution_id)
        LOGGER.debug(
            "Auto-execute approved for finding %s (execution_id=%s).",
            finding.anomaly_id,
            execution_id,
        )
        return ActionPolicyResult(
            action_policy_path=ACTION_POLICY_AUTO_EXECUTE,
            data_quality=dq,
            data_quality_details=dq_details,
            execution_id=execution_id,
            block_reason=None,
        )

    def evaluate_canary(  # noqa: PLR0911
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        autonomy_level: int,
        now: datetime,
    ) -> ActionPolicyResult:
        """
        Evaluate execution policy **without mutating any instance state**.

        Identical logic to ``evaluate()`` but uses snapshot copies of the
        rate-limit and idempotency state so that canary decisions leave no
        side effects.  Returns the hypothetical ``ActionPolicyResult`` that
        *would* have been returned if live execution were enabled.
        """
        # 1. Autonomy level gate.
        if autonomy_level < _MIN_AUTO_EXECUTE_LEVEL:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=DATA_QUALITY_FRESH,
                data_quality_details={},
                execution_id=None,
                block_reason="autonomy_level_below_2",
            )

        # 2. Data quality gate.
        dq, dq_details = self._compute_data_quality(finding, snapshot, now)
        if dq == DATA_QUALITY_UNAVAILABLE:
            high_sev = finding.severity == "high"
            path = ACTION_POLICY_PROMPT_USER if high_sev else ACTION_POLICY_BLOCKED
            return ActionPolicyResult(
                action_policy_path=path,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="data_quality_unavailable",
            )
        if dq == DATA_QUALITY_STALE:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="data_quality_stale",
            )

        # 3. Sensitivity gate.
        if finding.is_sensitive:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_HANDOFF,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="finding_sensitive",
            )

        # 4. Auto-execution enabled gate.
        auto_enabled = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
                RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
            )
        )
        # Canary mode reports what would run whether or not live execution
        # is on, so a dry-run review needs no switch that could act.
        canary_mode = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
                RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
            )
        )
        if not auto_enabled and not canary_mode:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="auto_execution_disabled",
            )

        # 5. Confidence gate.
        min_confidence = float(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
            )
        )
        if finding.confidence < min_confidence:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="confidence_below_threshold",
            )

        # 6. Action gate: something to run, allowlisted, never critical.
        if (block_reason := self._actions_block_reason(finding)) is not None:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason=block_reason,
            )

        # 7. Rate limit gate — use a read-only snapshot of the action-time list.
        window_start = now - timedelta(hours=1)
        recent_snapshot = [t for t in self._recent_action_times if t > window_start]
        max_per_hour = int(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
            )
        )
        if len(recent_snapshot) >= max_per_hour:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=None,
                block_reason="rate_limit_exceeded",
            )

        # 8. Idempotency gate — use a read-only snapshot of the seen set.
        policy_version = self._compute_policy_version(autonomy_level, min_confidence)
        execution_id = self._compute_execution_id(
            finding.anomaly_id, ACTION_POLICY_AUTO_EXECUTE, policy_version
        )
        # Refresh window check against monotonic clock without mutating state.
        window_minutes = int(
            self._options.get(
                CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
                RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
            )
        )
        window_seconds = window_minutes * 60
        elapsed = time.monotonic() - self._idempotency_window_reset_at
        idempotency_snapshot = (
            set() if elapsed >= window_seconds else set(self._idempotency_seen)
        )
        if execution_id in idempotency_snapshot:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=execution_id,
                block_reason="idempotency_duplicate",
            )

        # All guardrails pass — hypothetical auto-execute (no state mutation).
        return ActionPolicyResult(
            action_policy_path=ACTION_POLICY_AUTO_EXECUTE,
            data_quality=dq,
            data_quality_details=dq_details,
            execution_id=execution_id,
            block_reason=None,
        )

    # ---------------------------------------------------------------------- #
    # Data quality / staleness
    # ---------------------------------------------------------------------- #

    def _compute_data_quality(
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        now: datetime,
    ) -> tuple[str, dict[str, Any]]:
        """
        Determine the data quality for the entities triggering *finding*.

        Returns a ``(quality, details)`` tuple where quality is one of
        ``"fresh"``, ``"stale"``, or ``"unavailable"``.
        """
        threshold_seconds = int(
            self._options.get(
                CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
                RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS,
            )
        )
        threshold = timedelta(seconds=threshold_seconds)

        entity_map: dict[str, Any] = {e["entity_id"]: e for e in snapshot["entities"]}

        stale_entities: list[str] = []
        unavailable_entities: list[str] = []
        fresh_entities: list[str] = []

        for entity_id in finding.triggering_entities:
            entity = entity_map.get(entity_id)
            if entity is None:
                unavailable_entities.append(entity_id)
                continue
            last_changed_raw = entity.get("last_changed")
            last_changed = (
                dt_util.parse_datetime(last_changed_raw) if last_changed_raw else None
            )
            if last_changed is None:
                stale_entities.append(entity_id)
                continue
            age = now - dt_util.as_utc(last_changed)
            if age > threshold:
                stale_entities.append(entity_id)
            else:
                fresh_entities.append(entity_id)

        details: dict[str, Any] = {
            "threshold_seconds": threshold_seconds,
            "stale_entities": stale_entities,
            "unavailable_entities": unavailable_entities,
            "fresh_entities": fresh_entities,
        }

        if unavailable_entities:
            return DATA_QUALITY_UNAVAILABLE, details
        if stale_entities:
            return DATA_QUALITY_STALE, details
        return DATA_QUALITY_FRESH, details

    # ---------------------------------------------------------------------- #
    # Allowlist
    # ---------------------------------------------------------------------- #

    def _actions_block_reason(self, finding: AnomalyFinding) -> str | None:
        """
        Return why *finding*'s suggested actions cannot run on their own, or None.

        Only ``domain.service`` actions are ever called (advisory text such as
        ``check_camera`` is not), so only those must be on the allowlist; with
        none, there is nothing to run and the user decides. An empty allowlist
        permits nothing. A critical action never runs without a person,
        whatever the allowlist says: one a Critical Action PIN rule covers,
        one the automation screener holds (a script, scene or automation it
        would start), any alarm panel or cover action, and any lock action
        but ``lock.lock``.
        """
        actions = executable_actions(finding)
        if not actions:
            return "no_executable_action"
        allowed: list[str] = list(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
            )
            or []
        )
        for action in actions:
            if action not in allowed and action.split(".", 1)[0] not in allowed:
                LOGGER.debug(
                    "Auto-execute blocked: service %s not in allowlist %s.",
                    action,
                    allowed,
                )
                return "service_not_allowlisted"
        if self._is_critical(finding, actions):
            return "critical_action"
        return None

    def _is_critical(self, finding: AnomalyFinding, actions: list[str]) -> bool:
        """Return True if any action needs a person: a PIN-gated or alarm call."""
        from custom_components.home_generative_agent.agent.automation_pin import (  # noqa: PLC0415
            find_critical_automation_calls,
        )
        from custom_components.home_generative_agent.agent.helpers import (  # noqa: PLC0415
            resolve_critical_action_policy,
        )

        for action in actions:
            domain = action.split(".", 1)[0]
            if domain in _NEVER_AUTO_DOMAINS or (
                domain == "lock" and action != _SAFE_LOCK_SERVICE
            ):
                LOGGER.debug("Auto-execute blocked: %s needs a person.", action)
                return True
        rules = resolve_critical_action_policy(self._options).critical_actions
        target = {"entity_id": list(finding.triggering_entities)}
        steps = [{"action": action, "target": target} for action in actions]
        found = find_critical_automation_calls({"actions": steps}, rules)
        if found:
            LOGGER.debug(
                "Auto-execute blocked: critical action %s.",
                "; ".join(call.describe() for call in found),
            )
        return bool(found)

    # ---------------------------------------------------------------------- #
    # Rate limiting
    # ---------------------------------------------------------------------- #

    def _is_rate_limited(self, now: datetime) -> bool:
        """
        Return True if the per-hour action rate limit has been reached.

        Prunes the ``_recent_action_times`` list to the rolling 1-hour window
        before comparing against ``CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR``.
        """
        window_start = now - timedelta(hours=1)
        self._recent_action_times = [
            t for t in self._recent_action_times if t > window_start
        ]
        max_per_hour = int(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
            )
        )
        return len(self._recent_action_times) >= max_per_hour

    # ---------------------------------------------------------------------- #
    # Idempotency
    # ---------------------------------------------------------------------- #

    def _refresh_idempotency_window(self) -> None:
        """Reset the seen-set when the current time bucket rolls over."""
        window_minutes = int(
            self._options.get(
                CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
                RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
            )
        )
        window_seconds = window_minutes * 60
        now_monotonic = time.monotonic()
        elapsed = now_monotonic - self._idempotency_window_reset_at
        if elapsed >= window_seconds:
            self._idempotency_seen.clear()
            self._idempotency_window_reset_at = now_monotonic

    def _compute_execution_id(
        self,
        anomaly_id: str,
        action_policy_path: str,
        policy_version: str,
    ) -> str:
        """
        Compute a stable idempotency key for an auto-execute decision.

        ``sha256(anomaly_id + ":" + action_policy_path + ":" + policy_version
                 + ":" + window_bucket)``
        """
        window_minutes = int(
            self._options.get(
                CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
                RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
            )
        )
        window_bucket = int(time.time() / (window_minutes * 60))
        raw = f"{anomaly_id}:{action_policy_path}:{policy_version}:{window_bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _compute_policy_version(
        self, autonomy_level: int, effective_rule_threshold: float
    ) -> str:
        """
        Compute a hash that captures the current execution policy.

        Any change in execution-relevant config automatically produces a
        different hash, invalidating existing idempotency keys.
        """
        allowed: list[str] = sorted(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
            )
        )
        policy: dict[str, Any] = {
            "allowed_services": allowed,
            "autonomy_level": autonomy_level,
            "min_confidence": float(
                self._options.get(
                    CONF_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
                    RECOMMENDED_SENTINEL_AUTO_EXECUTE_DEFAULT_MIN_CONFIDENCE,
                )
            ),
            "max_actions_per_hour": int(
                self._options.get(
                    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                    RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                )
            ),
            "require_pin_for_increase": level_increase_pin_required(self._options),
            "effective_rule_threshold": effective_rule_threshold,
        }
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
