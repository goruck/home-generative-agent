"""
Sentinel execution service — Issue #259.

Provides ``SentinelExecutionService``, which evaluates the action-policy gate
for each finding.  All auto-execution guardrails live here so that both the
mobile-action flow and the autonomous-execution flow share a single policy
evaluation path.

Guardrail evaluation order
--------------------------
1. Autonomy level < 2                 → ``prompt_user``
2. Auto-execution disabled            → ``prompt_user`` (the canary
                                        evaluator still runs on in canary mode)
3. The finding carries no action
   (its suggested actions are advice,
   never called)                      → ``prompt_user``
4. Data quality != "fresh"            → ``prompt_user`` (stale) /
                                        ``blocked`` (unavailable)
5. Actions: one is not a targeted
   ``domain.service`` call on entities
   of that domain, a target is missing
   or unavailable, one is not
   allowlisted, or one is critical
   (see ``_is_critical``)             → ``prompt_user``
6. Rate limit exceeded                → ``prompt_user``
7. Idempotency duplicate              → ``prompt_user``
8. All checks passed                  → ``auto_execute``

Only rule code decides whether a finding may act, by giving it actions.
Neither ``is_sensitive`` (which governs what reaches a model or a
notification) nor ``confidence`` (which a model chooses for a discovered
rule) is read here, and the engine asks no model about a finding that
carries an action (#789).
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

from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import valid_entity_id
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_BLOCKED,
    ACTION_POLICY_PROMPT_USER,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
    DATA_QUALITY_FRESH,
    DATA_QUALITY_NOT_EVALUATED,
    DATA_QUALITY_STALE,
    DATA_QUALITY_UNAVAILABLE,
    RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
    RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
    RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
    RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
    RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS,
)
from custom_components.home_generative_agent.snapshot.schema import entity_last_heard

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
        FindingAction,
    )
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)

# Minimum autonomy level required for auto-execution.
_MIN_AUTO_EXECUTE_LEVEL: int = 2
# The form of a service Sentinel can call.
_SERVICE_ACTION = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")
# Domains Sentinel never acts on by itself, whatever the rules or allowlist:
# an alarm panel guards itself with its own code, and a cover can be a door,
# gate or garage whose entity id names none of them. A button, valve or siren
# can be the same opener, a water or gas line, or an alarm sounder, and
# nothing in its entity id has to say so.
_NEVER_AUTO_DOMAINS = frozenset(
    {"alarm_control_panel", "button", "cover", "input_button", "siren", "valve"}
)
# The only lock service Sentinel may call by itself.
_SAFE_LOCK_SERVICE = "lock.lock"
_GENERIC_DOMAIN = "homeassistant"
# The same floor as rules, for what an action starts (#831): a template
# switch that disarms the alarm needs a person as much as a disarm does.
_FLOOR_RULES: tuple[dict[str, str], ...] = (
    *({"domain": domain, "service": ""} for domain in sorted(_NEVER_AUTO_DOMAINS)),
    {"domain": "lock", "service": "unlock"},
    {"domain": "lock", "service": "open"},
)


def executable_actions(finding: AnomalyFinding) -> list[FindingAction]:
    """
    Return the actions Sentinel can call for *finding*.

    Only the finding's own ``actions``, which rule code builds with their
    targets. Its ``suggested_actions`` are advice and are never called,
    whatever they look like (#789).
    """
    return list(finding.actions)


def is_callable_action(action: FindingAction) -> bool:
    """
    Return True for a ``domain.service`` action on entities of that domain.

    Every target must be a whole, lower-case entity id, as Home Assistant
    writes them: it lower-cases what it is given, so ``Cover.garage`` would
    slip past a floor that reads domains and still move the cover. An entity
    service acts only on its own domain, so a target of another domain is a
    call that does nothing and reads as done; only ``homeassistant.*`` takes
    any domain, and the floor judges its targets.
    """
    service = action.service
    targets = action.entity_ids
    if not isinstance(service, str) or _SERVICE_ACTION.fullmatch(service) is None:
        return False
    if not isinstance(targets, tuple) or not targets:
        return False
    domain = service.split(".", 1)[0]
    for entity_id in targets:
        if not isinstance(entity_id, str) or not valid_entity_id(entity_id):
            return False
        if domain not in (_GENERIC_DOMAIN, entity_id.split(".", 1)[0]):
            return False
    return True


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

    def __init__(
        self, options: dict[str, Any], *, hass: HomeAssistant | None = None
    ) -> None:
        """Initialize with the resolved runtime options dict."""
        self._options = options
        # Read to extend the Critical Action rules to device classes, names
        # and groups, as the PIN gate does (#773).
        self._hass = hass
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

    def evaluate(
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
        - ``"auto_execute"`` — all guardrails passed; execute automatically
        - ``"blocked"``    — finding is suppressed at the execution layer

        An approval is counted toward the hourly limit and remembered for the
        idempotency window. This method is **not** a coroutine; it performs
        no I/O.
        """
        result = self._decide(finding, snapshot, autonomy_level, now, dry_run=False)
        if (
            result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
            and result.execution_id is not None
        ):
            self.commit_auto_execute(result.execution_id, now)
        return result

    def evaluate_canary(
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        autonomy_level: int,
        now: datetime,
    ) -> ActionPolicyResult:
        """
        Evaluate execution policy **without mutating any instance state**.

        The same decision as ``evaluate()``, except that canary mode reports
        what would run whether or not live execution is on, so a dry-run
        review needs no switch that could act. Nothing is counted or
        remembered: the caller commits a live action once it has run.
        """
        return self._decide(finding, snapshot, autonomy_level, now, dry_run=True)

    def _decide(  # noqa: PLR0911
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        autonomy_level: int,
        now: datetime,
        *,
        dry_run: bool,
    ) -> ActionPolicyResult:
        """Return the policy decision for *finding*, changing no state."""

        def _left_to_user(reason: str) -> ActionPolicyResult:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=DATA_QUALITY_NOT_EVALUATED,
                data_quality_details={},
                execution_id=None,
                block_reason=reason,
            )

        # 1. Autonomy level gate — auto-execute requires level 2+.
        if autonomy_level < _MIN_AUTO_EXECUTE_LEVEL:
            return _left_to_user("autonomy_level_below_2")

        # 2. Auto-execution gate. Off: nothing to decide, the user is notified
        # as at level 1. Checked before data quality so that levels 2 and 3
        # drop no finding while no action can run (#783).
        enabled = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
                RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
            )
        ) or (
            dry_run
            and bool(
                self._options.get(
                    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
                    RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
                )
            )
        )
        if not enabled:
            return _left_to_user("auto_execution_disabled")

        # 3. Nothing to run. Checked before data quality for the same reason:
        # a finding that could never act is an alert like any other, and a
        # stale or missing sensor is often what it is about. This also covers
        # an owed alert sent later, which is restored without its actions.
        if not executable_actions(finding):
            return _left_to_user("no_executable_action")

        # 4. Data quality gate.
        dq, dq_details = self._compute_data_quality(finding, snapshot, now)

        def _result(
            path: str, reason: str | None, execution_id: str | None = None
        ) -> ActionPolicyResult:
            return ActionPolicyResult(
                action_policy_path=path,
                data_quality=dq,
                data_quality_details=dq_details,
                execution_id=execution_id,
                block_reason=reason,
            )

        if dq == DATA_QUALITY_UNAVAILABLE:
            high_sev = finding.severity == "high"
            path = ACTION_POLICY_PROMPT_USER if high_sev else ACTION_POLICY_BLOCKED
            return _result(path, "data_quality_unavailable")
        if dq == DATA_QUALITY_STALE:
            return _result(ACTION_POLICY_PROMPT_USER, "data_quality_stale")

        # 5. Action gate: well-formed, on the allowlist, never critical.
        if (block_reason := self._actions_block_reason(finding)) is not None:
            return _result(ACTION_POLICY_PROMPT_USER, block_reason)

        # 6. Rate limit gate.
        window_start = now - timedelta(hours=1)
        recent = [t for t in self._recent_action_times if t > window_start]
        max_per_hour = int(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
            )
        )
        if len(recent) >= max_per_hour:
            return _result(ACTION_POLICY_PROMPT_USER, "rate_limit_exceeded")

        # 7. Idempotency gate.
        execution_id = self._compute_execution_id(
            finding.anomaly_id,
            ACTION_POLICY_AUTO_EXECUTE,
            self._compute_policy_version(autonomy_level),
        )
        window_seconds = (
            int(
                self._options.get(
                    CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
                    RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
                )
            )
            * 60
        )
        elapsed = time.monotonic() - self._idempotency_window_reset_at
        if elapsed < window_seconds and execution_id in self._idempotency_seen:
            LOGGER.debug(
                "Auto-execute blocked for %s: idempotency duplicate (execution_id=%s).",
                finding.anomaly_id,
                execution_id,
            )
            return _result(
                ACTION_POLICY_PROMPT_USER, "idempotency_duplicate", execution_id
            )

        return _result(ACTION_POLICY_AUTO_EXECUTE, None, execution_id)

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
            # When it last reported: an always-on power sensor's last_changed
            # is when its current run started (#982).
            last_heard = dt_util.parse_datetime(entity_last_heard(entity) or "")
            if last_heard is None:
                stale_entities.append(entity_id)
                continue
            age = now - dt_util.as_utc(last_heard)
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
    # Action gate
    # ---------------------------------------------------------------------- #

    def _actions_block_reason(self, finding: AnomalyFinding) -> str | None:
        """
        Return why *finding*'s actions cannot run on their own, or None.

        Only the finding's ``actions`` are ever called (its suggested actions
        are advice), so only those must be on the allowlist; with none, there
        is nothing to run and the user decides. An action that is not a
        targeted call (``is_callable_action``) stops them all, as does a
        target Home Assistant has no state for or holds unavailable. The
        reasons returned are ``no_executable_action``, ``malformed_action``,
        ``target_unavailable``, ``service_not_allowlisted`` and
        ``critical_action``. An empty allowlist permits nothing. A critical
        action never runs without a person, whatever the allowlist says
        (see ``_is_critical``).
        """
        actions = executable_actions(finding)
        if not actions:
            return "no_executable_action"
        if not all(is_callable_action(action) for action in actions):
            LOGGER.debug("Auto-execute blocked: malformed action in %s.", actions)
            return "malformed_action"
        if (missing := self._unavailable_targets(actions)) is not None:
            LOGGER.debug("Auto-execute blocked: %s cannot be acted on.", missing)
            return "target_unavailable"
        allowed: list[str] = list(
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
            )
            or []
        )
        for action in actions:
            service = action.service
            if service not in allowed and service.split(".", 1)[0] not in allowed:
                LOGGER.debug(
                    "Auto-execute blocked: service %s not in allowlist %s.",
                    service,
                    allowed,
                )
                return "service_not_allowlisted"
        if self._is_critical(actions):
            return "critical_action"
        return None

    def _unavailable_targets(self, actions: list[FindingAction]) -> str | None:
        """
        Return a target Home Assistant would skip, or None.

        Home Assistant drops a missing or unavailable entity from a service
        call and returns normally, so the call would be recorded as done with
        nothing acted on. The data-quality gate reads the finding's triggering
        entities, which are often not the targets.
        """
        states = getattr(self._hass, "states", None)
        if states is None:
            return None
        for action in actions:
            for entity_id in action.entity_ids:
                state = states.get(entity_id)
                if state is None or state.state == STATE_UNAVAILABLE:
                    return entity_id
        return None

    def _is_critical(self, actions: list[FindingAction]) -> bool:
        """
        Return True if any action needs a person.

        The floor, which no rule or allowlist lifts: any ``homeassistant.*``
        call (it restarts or stops Home Assistant, or runs whatever service
        its target has), any alarm panel, cover, valve, siren or button
        action, any lock action but ``lock.lock``, and a call on the switch
        under a switch_as_x wrapper of one of those domains or of a lock,
        which is the relay that moves it. Then what the user's Critical
        Action PIN rules cover, and what the call would start (a script,
        scene or automation, a template entity's own actions). A check that
        cannot be made counts as critical.
        """
        try:
            return self._floor_bars(actions) or self._rules_bar(actions)
        except Exception:
            LOGGER.exception(
                "Auto-execute blocked: could not check whether %s needs a person.",
                ", ".join(a.service for a in actions),
            )
            return True

    def _floor_bars(self, actions: list[FindingAction]) -> bool:
        """Return True if Sentinel's own floor bars any of *actions*."""
        from custom_components.home_generative_agent.agent.rule_entities import (  # noqa: PLC0415
            read_switch_wrappers,
        )

        guarded = {*_NEVER_AUTO_DOMAINS, "lock"}
        for action in actions:
            service = action.service
            domain = service.split(".", 1)[0]
            if (
                domain in _NEVER_AUTO_DOMAINS
                or domain == _GENERIC_DOMAIN
                or (domain == "lock" and service != _SAFE_LOCK_SERVICE)
            ):
                LOGGER.debug("Auto-execute blocked: %s needs a person.", service)
                return True
        targets = {e for action in actions for e in action.entity_ids}
        if self._hass is None or not any(e.startswith("switch.") for e in targets):
            return False
        for wrapper, (switch_id, _inverted) in read_switch_wrappers(self._hass).items():
            if switch_id in targets and wrapper.split(".", 1)[0] in guarded:
                LOGGER.debug(
                    "Auto-execute blocked: %s is the relay under %s.",
                    switch_id,
                    wrapper,
                )
                return True
        return False

    def _rules_bar(self, actions: list[FindingAction]) -> bool:
        """Return True if a PIN rule covers *actions* or what they start."""
        from custom_components.home_generative_agent.agent.automation_pin import (  # noqa: PLC0415
            find_critical_automation_calls,
        )
        from custom_components.home_generative_agent.agent.helpers import (  # noqa: PLC0415
            resolve_critical_action_policy,
        )
        from custom_components.home_generative_agent.agent.indirect_pin import (  # noqa: PLC0415
            stored_action_calls,
        )
        from custom_components.home_generative_agent.agent.rule_entities import (  # noqa: PLC0415
            expand_critical_rules,
        )

        # Extended as the PIN gate extends them (#773): a rule's word also
        # reaches entities by device class, name and group.
        rules = expand_critical_rules(
            self._hass, resolve_critical_action_policy(self._options).critical_actions
        )
        steps = [
            {"action": a.service, "target": {"entity_id": list(a.entity_ids)}}
            for a in actions
        ]
        # What the call starts runs too: a template switch's actions, a
        # button's or an automation it triggers, which can open the garage.
        # The PIN gate follows them, and so does Sentinel (#831).
        found = find_critical_automation_calls(
            {"actions": steps}, rules
        ) or stored_action_calls(self._hass, steps, [*rules, *_FLOOR_RULES])
        if found:
            LOGGER.debug(
                "Auto-execute blocked: critical action %s.",
                "; ".join(call.describe() for call in found),
            )
        return bool(found)

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

    def _compute_policy_version(self, autonomy_level: int) -> str:
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
            "max_actions_per_hour": int(
                self._options.get(
                    CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                    RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                )
            ),
            "require_pin_for_increase": level_increase_pin_required(self._options),
        }
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
