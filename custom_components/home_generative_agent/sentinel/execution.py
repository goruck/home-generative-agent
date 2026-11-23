"""
Sentinel execution service — Issue #259.

Provides ``SentinelExecutionService``, which evaluates the action-policy gate
for each finding.  All auto-execution guardrails live here so that both the
mobile-action flow and the autonomous-execution flow share a single policy
evaluation path.

Guardrail evaluation order
--------------------------
1. Autonomy level < 2                 → ``prompt_user``
2. Auto-execution disabled            → ``prompt_user`` (a dry run still
                                        evaluates in canary mode)
3. The finding carries no action
   (its suggested actions are advice,
   never called)                      → ``prompt_user``
4. Data quality != "fresh"            → ``prompt_user``
5. Actions: one is not a targeted
   ``domain.service`` call on entities
   of that domain, a target is missing
   or unavailable, one is not
   allowlisted, one is critical (see
   ``_critical_reason``), or the relays
   of guarded devices cannot be read  → ``prompt_user``
6. Rate limit exceeded                → ``prompt_user``
7. Idempotency duplicate              → ``prompt_user``
8. All checks passed                  → ``auto_execute``

No path here ends in silence: a finding whose action is refused is still
sent to the user. No model takes part: neither ``is_sensitive`` (which
governs what reaches a model or a notification) nor ``confidence`` (which a
model chooses for a discovered rule) is read, and the engine does not send
a finding whose action this gate approves to LLM triage (#789); its
explanation is written after the action ran. The gate is reached
only by a finding the engine would notify about: one a cooldown, snooze,
quiet hours or a pending prompt holds back does not act either.
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
MIN_AUTO_EXECUTE_LEVEL: int = 2
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
# Domains whose relay (the switch under a switch_as_x wrapper, what a template
# cover or lock drives) the floor bars with them.
_RELAY_GUARDED_DOMAINS = frozenset({*_NEVER_AUTO_DOMAINS, "lock"})
# No sensor reading is fresh for longer than this, whatever the setting says.
_MAX_STALENESS_SECONDS = 7 * 24 * 3600


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
    any domain, and the floor bars it outright.
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
        # What could not be read of the guarded relays, as last logged.
        self._unread_relays_logged: tuple[str, ...] = ()
        # Monotonic timestamp of last idempotency window reset.
        self._idempotency_window_reset_at: float = time.monotonic()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def commit_auto_execute(self, execution_id: str, now: datetime) -> None:
        """
        Record a live auto-execution after its service calls were made.

        Called by the engine, never by the gate, so that the hourly limit and
        the idempotency window count actions that ran, not approvals.
        """
        window_start = now - timedelta(hours=1)
        self._recent_action_times = [
            t for t in self._recent_action_times if t > window_start
        ]
        self._refresh_idempotency_window()
        self._recent_action_times.append(now)
        self._idempotency_seen.add(execution_id)
        LOGGER.info("Recorded auto-execute completion (execution_id=%s).", execution_id)

    def evaluate_canary(  # noqa: PLR0911
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        autonomy_level: int,
        now: datetime,
    ) -> ActionPolicyResult:
        """
        Return the policy decision for *finding*, changing no state.

        The returned ``action_policy_path`` is one of:

        - ``"prompt_user"`` — notify and let the user decide
        - ``"auto_execute"`` — every guardrail passed; the actions would run

        Canary (dry-run) mode gets the same answer live execution would, so a
        dry-run review needs no switch that could act; the engine runs
        nothing in that mode. Nothing is counted or remembered here: the
        engine calls ``commit_auto_execute`` once a live action has run. Not
        a coroutine; it performs no I/O.
        """

        def _left_to_user(reason: str) -> ActionPolicyResult:
            return ActionPolicyResult(
                action_policy_path=ACTION_POLICY_PROMPT_USER,
                data_quality=DATA_QUALITY_NOT_EVALUATED,
                data_quality_details={},
                execution_id=None,
                block_reason=reason,
            )

        # 1. Autonomy level gate — auto-execute requires level 2+.
        if autonomy_level < MIN_AUTO_EXECUTE_LEVEL:
            return _left_to_user("autonomy_level_below_2")

        # 2. Auto-execution gate. Off: nothing to decide, the user is notified
        # as at level 1.
        if not (
            self._options.get(
                CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
                RECOMMENDED_SENTINEL_AUTO_EXECUTION_ENABLED,
            )
            or self._options.get(
                CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
                RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
            )
        ):
            return _left_to_user("auto_execution_disabled")

        # 3. Nothing to run: an alert like any other. This also covers an
        # owed alert sent later, which is restored without its actions.
        if not executable_actions(finding):
            return _left_to_user("no_executable_action")

        # 4. Data quality gate. The action does not run on data that is
        # missing or old; the user is still told, at every severity.
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

        if dq != DATA_QUALITY_FRESH:
            return _result(ACTION_POLICY_PROMPT_USER, f"data_quality_{dq}")

        # 5. Action gate: well-formed, on the allowlist, never critical.
        if (block_reason := self._actions_block_reason(finding)) is not None:
            return _result(ACTION_POLICY_PROMPT_USER, block_reason)

        # 6. Rate limit gate.
        window_start = now - timedelta(hours=1)
        recent = [t for t in self._recent_action_times if t > window_start]
        max_per_hour = self._int_option(
            CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
            RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
            minimum=0,
        )
        if len(recent) >= max_per_hour:
            return _result(ACTION_POLICY_PROMPT_USER, "rate_limit_exceeded")

        # 7. Idempotency gate.
        execution_id = self._compute_execution_id(
            finding.anomaly_id,
            ACTION_POLICY_AUTO_EXECUTE,
            self._compute_policy_version(autonomy_level),
        )
        elapsed = time.monotonic() - self._idempotency_window_reset_at
        if (
            elapsed < self._idempotency_window_seconds()
            and execution_id in self._idempotency_seen
        ):
            LOGGER.debug(
                "Auto-execute blocked for %s: idempotency duplicate (execution_id=%s).",
                finding.anomaly_id,
                execution_id,
            )
            return _result(
                ACTION_POLICY_PROMPT_USER, "idempotency_duplicate", execution_id
            )

        return _result(ACTION_POLICY_AUTO_EXECUTE, None, execution_id)

    def _int_option(
        self, key: str, default: int, *, minimum: int, maximum: int | None = None
    ) -> int:
        """
        Return a whole-number option kept within its bounds.

        One that is not a number is read as the default: these settings are
        in no form yet, so a bad one was edited by hand, and raising here
        would stop the finding from being sent at all.
        """
        try:
            value = int(float(self._options.get(key, default)))
        except (TypeError, ValueError, OverflowError):
            value = default
        value = max(minimum, value)
        return value if maximum is None else min(maximum, value)

    def _allowed_services(self) -> list[str]:
        """Return the allowlist, sorted; anything in it but text allows nothing."""
        allowed = self._options.get(
            CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
            RECOMMENDED_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
        )
        if not isinstance(allowed, (list, tuple, set, frozenset)):
            return []
        return sorted(item for item in allowed if isinstance(item, str))

    def _idempotency_window_seconds(self) -> int:
        """Return the idempotency window, never less than a minute."""
        return 60 * self._int_option(
            CONF_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
            RECOMMENDED_SENTINEL_EXECUTION_IDEMPOTENCY_WINDOW_MINUTES,
            minimum=1,
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
        threshold_seconds = self._int_option(
            CONF_SENTINEL_STALENESS_THRESHOLD_SECONDS,
            RECOMMENDED_SENTINEL_STALENESS_THRESHOLD_SECONDS,
            minimum=0,
            maximum=_MAX_STALENESS_SECONDS,
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
        are advice), so only those must be on the allowlist; the caller has
        already left a finding with none to the user. An action that is not a
        targeted call (``is_callable_action``) stops them all, as does a
        target Home Assistant has no state for or holds unavailable. The
        reasons returned are ``malformed_action``, ``target_unavailable``,
        ``service_not_allowlisted``, ``critical_action`` and
        ``relay_map_unreadable``. An empty allowlist permits nothing. A
        critical action never runs without a person, whatever the allowlist
        says (see ``_critical_reason``).
        """
        actions = executable_actions(finding)
        if not all(is_callable_action(action) for action in actions):
            LOGGER.debug("Auto-execute blocked: malformed action in %s.", actions)
            return "malformed_action"
        if (missing := self._unavailable_targets(actions)) is not None:
            LOGGER.debug("Auto-execute blocked: %s cannot be acted on.", missing)
            return "target_unavailable"
        allowed = self._allowed_services()
        for action in actions:
            service = action.service
            if service not in allowed and service.split(".", 1)[0] not in allowed:
                LOGGER.debug(
                    "Auto-execute blocked: service %s not in allowlist %s.",
                    service,
                    allowed,
                )
                return "service_not_allowlisted"
        return self._critical_reason(actions)

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

    def _critical_reason(self, actions: list[FindingAction]) -> str | None:
        """
        Return why *actions* need a person, or None.

        ``critical_action`` for the floor, which no rule or allowlist lifts:
        any ``homeassistant.*`` call (it restarts or stops Home Assistant, or
        runs whatever service its target has), any alarm panel, cover, valve,
        siren or button action, and any lock action but ``lock.lock``. Then
        for a call that reaches the relay moving a cover, valve, siren or
        lock, as far as that can be read (``guarded_relay_reach``): the
        switch under a switch_as_x wrapper, what a template cover or lock
        drives, a group holding either, another wrapper of it. Then for what
        the user's Critical Action PIN rules cover, and for what the call
        would start (a script, scene or automation, a template entity's own
        actions), judged by those rules and the floor. A check that raises
        counts as critical.

        ``relay_map_unreadable`` when a switch_as_x entry or a template
        cover's or lock's actions cannot be read in full: a relay could be
        behind it, so nothing runs until it can be read.

        The relay check is best effort, not a guarantee. It does not see a
        template whose service name is itself a template, a template entity
        that is not loaded, a relay a template reaches through a wrapper of
        another domain, or what a template alarm panel or button drives
        (#789). With no Home Assistant instance only the services and entity
        ids are judged (the engine always supplies one).
        """
        try:
            if self._floor_bars_service(actions):
                return "critical_action"
            if (reason := self._relay_reason(actions)) is not None:
                return reason
            return "critical_action" if self._rules_bar(actions) else None
        except Exception:
            LOGGER.exception(
                "Auto-execute blocked: could not check whether %s needs a person.",
                ", ".join(a.service for a in actions),
            )
            return "critical_action"

    def _floor_bars_service(self, actions: list[FindingAction]) -> bool:
        """Return True if the floor bars the service of any of *actions*."""
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
        return False

    def _relay_reason(self, actions: list[FindingAction]) -> str | None:
        """Return why *actions* cannot run for the relays they may reach."""
        from custom_components.home_generative_agent.agent.rule_entities import (  # noqa: PLC0415
            guarded_relay_reach,
        )

        if self._hass is None:
            return None
        reach, unread = guarded_relay_reach(self._hass, _RELAY_GUARDED_DOMAINS)
        if unread:
            # Once per set of problems, not on every run.
            if (key := tuple(sorted(unread))) != self._unread_relays_logged:
                self._unread_relays_logged = key
                LOGGER.warning(
                    "Sentinel runs no automatic action while it cannot read "
                    "what these move: %s.",
                    "; ".join(key),
                )
            return "relay_map_unreadable"
        self._unread_relays_logged = ()
        relays = reach & {e for action in actions for e in action.entity_ids}
        if relays:
            LOGGER.debug(
                "Auto-execute blocked: %s reaches the relay under a guarded device.",
                ", ".join(sorted(relays)),
            )
            return "critical_action"
        return None

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
        # reaches entities by device class, name and group. The floor's rules
        # go with them, so that a relay a template switch drives is held like
        # the cover it moves.
        rules = expand_critical_rules(
            self._hass,
            [
                *resolve_critical_action_policy(self._options).critical_actions,
                *_FLOOR_RULES,
            ],
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
        ) or stored_action_calls(self._hass, steps, rules)
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
        now_monotonic = time.monotonic()
        elapsed = now_monotonic - self._idempotency_window_reset_at
        if elapsed >= self._idempotency_window_seconds():
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
        window_bucket = int(time.time() / self._idempotency_window_seconds())
        raw = f"{anomaly_id}:{action_policy_path}:{policy_version}:{window_bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _compute_policy_version(self, autonomy_level: int) -> str:
        """
        Compute a hash that captures the current execution policy.

        Any change in execution-relevant config automatically produces a
        different hash, invalidating existing idempotency keys.
        """
        allowed = self._allowed_services()
        policy: dict[str, Any] = {
            "allowed_services": allowed,
            "autonomy_level": autonomy_level,
            "max_actions_per_hour": self._int_option(
                CONF_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                RECOMMENDED_SENTINEL_AUTO_EXECUTE_MAX_ACTIONS_PER_HOUR,
                minimum=0,
            ),
            "require_pin_for_increase": level_increase_pin_required(self._options),
        }
        return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
