"""Sentinel engine for proactive anomaly detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_BLOCKED,
    CONF_EXPLAIN_ENABLED,
    CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
    CONF_SENTINEL_AUTONOMY_LEVEL,
    CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
    CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES,
    CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
    CONF_SENTINEL_CAMERA_ENTRY_LINKS,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    CONF_SENTINEL_PRESENCE_GRACE_MINUTES,
    CONF_SENTINEL_QUIET_HOURS_END,
    CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
    CONF_SENTINEL_QUIET_HOURS_START,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
    RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
    RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
    RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES,
    RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
    RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    RECOMMENDED_SENTINEL_PRESENCE_GRACE_MINUTES,
    RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES,
    RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
    SIGNAL_SENTINEL_RUN_COMPLETE,
)
from custom_components.home_generative_agent.core.utils import verify_pin
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)

from .baseline import CYCLICAL_LOAD_HINTS
from .correlator import SentinelCorrelator
from .dynamic_rules import evaluate_dynamic_rules
from .execution import (
    SentinelExecutionService,
)
from .models import AnomalyFinding, CompoundFinding
from .rules.alarm_disarmed_external_threat import AlarmDisarmedDuringExternalThreatRule
from .rules.appliance_power_duration import AppliancePowerDurationRule
from .rules.camera_entry_unsecured import CameraEntryUnsecuredRule
from .rules.camera_missing_snapshot import CameraMissingSnapshotRule
from .rules.open_entry_while_away import OpenEntryWhileAwayRule
from .rules.phone_battery_low_at_night import PhoneBatteryLowAtNightRule
from .rules.unknown_person_camera_night_home import UnknownPersonAtNightWhileHomeRule
from .rules.unknown_person_camera_no_home import UnknownPersonCameraNoHomeRule
from .rules.unlocked_lock_at_night import UnlockedLockAtNightRule
from .rules.vehicle_detected_near_camera import VehicleDetectedNearCameraRule
from .suppression import (
    SuppressionManager,
    purge_expired_prompts,
    register_finding,
    register_presence_grace,
    register_prompt,
    should_suppress,
)
from .triage import TRIAGE_SUPPRESS, SentinelTriageService
from .trigger_scheduler import SentinelTriggerScheduler, TriggerRecord

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

    from .baseline import SentinelBaselineUpdater
    from .notifier import SentinelNotifier
    from .rule_registry import RuleRegistry

LOGGER = logging.getLogger(__name__)

# Module-level in-memory store for runtime autonomy-level overrides.
# Maps entry_id -> (level: int, expires_at: datetime)
_AUTONOMY_OVERRIDES: dict[str, tuple[int, datetime]] = {}

# ---------------------------------------------------------------------------
# Entity → anomaly-type mapping for event-driven triggering
# ---------------------------------------------------------------------------

# Domains always mapped to a specific anomaly type.
_DOMAIN_TO_ANOMALY_TYPE: dict[str, str] = {
    "lock": "unlocked_lock_at_night",
    "camera": "camera_entry_unsecured",
    "person": "open_entry_while_away",
    "alarm_control_panel": "alarm_disarmed_during_external_threat",
}

# Binary-sensor device classes mapped to anomaly types.
_DEVICE_CLASS_TO_ANOMALY_TYPE: dict[str, str] = {
    "door": "open_entry_while_away",
    "window": "open_entry_while_away",
    "gate": "open_entry_while_away",
    "motion": "camera_entry_unsecured",
    "occupancy": "unknown_person_camera_no_home",
}


def _anomaly_type_for_state(entity_id: str, new_state: State) -> str | None:
    """
    Return the sentinel anomaly type for an entity state change.

    Returns None if the entity is not relevant for event-driven triggering.
    """
    domain = entity_id.split(".", maxsplit=1)[0] if "." in entity_id else ""

    if domain in _DOMAIN_TO_ANOMALY_TYPE:
        return _DOMAIN_TO_ANOMALY_TYPE[domain]

    if domain == "binary_sensor":
        device_class = new_state.attributes.get("device_class", "")
        return _DEVICE_CLASS_TO_ANOMALY_TYPE.get(device_class)

    return None


def _is_cyclical(entity_id: str, friendly_name: str) -> bool:
    """
    Return True if this entity is a known normal high-power cycler.

    Splits on dots, underscores, and spaces so that entity IDs like
    ``sensor.refrigerator_power`` correctly tokenise to ``{"sensor",
    "refrigerator", "power"}``.  Matching is token-exact (no substring),
    which prevents short hints from matching unrelated tokens.

    Entities matching CYCLICAL_LOAD_HINTS undergo the sustained deviation gate
    before firing baseline_deviation or time_of_day_anomaly findings.
    """
    tokens = set(
        re.split(r"[._\s]+", entity_id.lower())
        + re.split(r"[._\s]+", friendly_name.lower())
    )
    return bool(tokens & CYCLICAL_LOAD_HINTS)


class SentinelEngine:
    """Periodic sentinel evaluation loop."""

    def __init__(  # noqa: PLR0913
        self,
        hass: HomeAssistant,
        options: dict[str, object],
        suppression: SuppressionManager,
        notifier: SentinelNotifier,
        audit_store: AuditStore,
        explainer: LLMExplainer | None = None,
        *,
        rule_registry: RuleRegistry | None = None,
        entry_id: str | None = None,
        triage_service: SentinelTriageService | None = None,
        baseline_updater: SentinelBaselineUpdater | None = None,
    ) -> None:
        """Initialize sentinel dependencies and runtime state."""
        self._hass = hass
        self._options = options
        self._suppression = suppression
        self._notifier = notifier
        self._audit_store = audit_store
        self._explainer = explainer
        self._rule_registry = rule_registry
        self._entry_id = entry_id
        self._triage_service = triage_service
        self._baseline_updater = baseline_updater
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._correlator = SentinelCorrelator()
        self._trigger_scheduler = SentinelTriggerScheduler()
        self._execution_service = SentinelExecutionService(dict(options))
        self._rules = [
            UnlockedLockAtNightRule(),
            OpenEntryWhileAwayRule(),
            AppliancePowerDurationRule(),
            CameraEntryUnsecuredRule(
                camera_entry_links=cast(
                    "dict[str, list[str]]",
                    options.get(CONF_SENTINEL_CAMERA_ENTRY_LINKS) or {},
                ),
            ),
            UnknownPersonCameraNoHomeRule(),
            UnknownPersonAtNightWhileHomeRule(),
            VehicleDetectedNearCameraRule(),
            CameraMissingSnapshotRule(),
            AlarmDisarmedDuringExternalThreatRule(),
            PhoneBatteryLowAtNightRule(),
        ]
        # Event-driven triggering — unsubscribe callbacks.
        self._event_unsubscribers: list[Callable[[], None]] = []
        # Presence tracking for grace-window registration.
        # Stores the set of person entity IDs known to be home from the last run.
        self._last_people_home: set[str] = set()
        # Cyclical load sustained deviation gate (Sprint 4).
        # Maps entity_id -> datetime when entity first exceeded the deviation
        # threshold in the current sustained episode.  Reset on engine stop.
        # Note: HA restart clears this dict; the first compressor-on cycle after
        # restart waits the full sustained_minutes before notifying (acceptable).
        self._cyclical_deviation_above_since: dict[str, datetime] = {}
        # Run telemetry exposed to SentinelHealthSensor.
        self.run_stats: dict[str, Any] = {}

    @property
    def learned_suppressions_count(self) -> int:
        """
        Count of distinct ``{rule_type}:{entity_id}`` pairs with learned multipliers.

        Each entry represents a finding type+entity that was snoozed or dismissed,
        extending its cooldown.  Exposed on the health sensor as
        ``learned_suppressions_active``.
        """
        return len(self._suppression.state.learned_cooldown_multipliers)

    @property
    def cyclical_entities_gated_count(self) -> int:
        """
        Count of cyclical entities currently held by the sustained deviation gate.

        Each entry is an entity_id whose deviation finding is being suppressed
        while waiting for the sustained_minutes threshold to elapse.  Exposed on
        the health sensor as ``cyclical_entities_gated``.
        """
        return len(self._cyclical_deviation_above_since)

    # ---------------------------------------------------------------------- #
    # Autonomy level management
    # ---------------------------------------------------------------------- #

    def set_autonomy_level(
        self, entry_id: str, level: int, pin: str | None = None
    ) -> None:
        """
        Set a runtime autonomy-level override (TTL-bounded).

        Raises HomeAssistantError if a PIN is required for a level increase and
        no PIN is supplied.  (PIN validation is a TODO; the stub just checks
        presence when the option is enabled.)
        """
        require_pin = bool(
            self._options.get(
                CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
                RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
            )
        )
        current_level = self.get_autonomy_level(entry_id)
        if require_pin and level > current_level and not pin:
            msg = (
                "A PIN is required to increase the autonomy level. "
                "Provide 'pin' in the service call data."
            )
            raise HomeAssistantError(msg)
        if require_pin and level > current_level:
            pin_hash = str(self._options.get(CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH, ""))
            pin_salt = str(self._options.get(CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT, ""))
            if not pin_hash or not pin_salt:
                msg = (
                    "Sentinel level-increase PIN validation is enabled, but no PIN "
                    "hash is configured."
                )
                raise HomeAssistantError(msg)
            if not verify_pin(str(pin), hashed=pin_hash, salt=pin_salt):
                msg = "Invalid PIN for autonomy level increase."
                raise HomeAssistantError(msg)
        ttl_minutes = _coerce_int(
            self._options.get(CONF_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES),
            default=RECOMMENDED_SENTINEL_RUNTIME_OVERRIDE_TTL_MINUTES,
        )
        expires_at = dt_util.utcnow() + timedelta(minutes=ttl_minutes)
        _AUTONOMY_OVERRIDES[entry_id] = (level, expires_at)
        LOGGER.info(
            "Sentinel autonomy level set to %s for entry %s (expires %s).",
            level,
            entry_id,
            expires_at.isoformat(),
        )

    def get_autonomy_level(self, entry_id: str) -> int:
        """
        Return the current effective autonomy level.

        Checks the in-memory override (respecting TTL); falls back to the
        config value ``CONF_SENTINEL_AUTONOMY_LEVEL``, then the recommended
        default.
        """
        override = _AUTONOMY_OVERRIDES.get(entry_id)
        if override is not None:
            level, expires_at = override
            if dt_util.utcnow() < expires_at:
                return level
            # TTL expired - clean up and fall through to config default
            del _AUTONOMY_OVERRIDES[entry_id]
        return _coerce_int(
            self._options.get(CONF_SENTINEL_AUTONOMY_LEVEL),
            default=RECOMMENDED_SENTINEL_AUTONOMY_LEVEL,
        )

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def start(self) -> None:
        """Start the sentinel loop and subscribe to HA state-change events."""
        if self._task is not None:
            return
        self._stop_event.clear()

        # Subscribe to state-change events for relevant entity domains.
        unsub = self._hass.bus.async_listen(EVENT_STATE_CHANGED, self._on_state_changed)
        self._event_unsubscribers.append(unsub)

        self._task = self._hass.async_create_task(self._run_loop())

        # Start the baseline updater independently if configured.
        if self._baseline_updater is not None:
            self._baseline_updater.start()

        LOGGER.debug("Sentinel engine started (event-driven triggering active).")

    async def stop(self) -> None:
        """Stop the sentinel loop and unsubscribe from HA events."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

        # Stop baseline updater.
        if self._baseline_updater is not None:
            await self._baseline_updater.stop()

        # Unsubscribe event listeners.
        for unsub in self._event_unsubscribers:
            unsub()
        self._event_unsubscribers.clear()

    # ---------------------------------------------------------------------- #
    # Event-driven triggering
    # ---------------------------------------------------------------------- #

    @callback
    def _on_state_changed(self, event: Event[EventStateChangedData]) -> None:
        """
        Handle a HA state-change event.

        Maps the changed entity to a sentinel anomaly type and enqueues a
        trigger in the scheduler.  Irrelevant entities are silently ignored.
        """
        entity_id: str = event.data.get("entity_id", "")
        new_state: State | None = event.data.get("new_state")
        if new_state is None:
            # Entity was removed — ignore.
            return

        anomaly_type = _anomaly_type_for_state(entity_id, new_state)
        if anomaly_type is None:
            return

        LOGGER.debug(
            "State change on %s → anomaly type %s; enqueueing trigger.",
            entity_id,
            anomaly_type,
        )
        self._trigger_scheduler.enqueue(TriggerRecord(anomaly_type=anomaly_type))

    # ---------------------------------------------------------------------- #
    # Run loop
    # ---------------------------------------------------------------------- #

    async def _run_loop(self) -> None:
        interval = _coerce_int(
            self._options.get(CONF_SENTINEL_INTERVAL_SECONDS), default=300
        )
        LOGGER.info("Sentinel loop started (interval=%ss).", interval)
        while not self._stop_event.is_set():
            # Check the trigger scheduler first.  If a queued trigger is ready
            # the single-flight lock inside the scheduler wraps _run_once().
            # When no trigger is ready (or the lock is already held) fall
            # through to the normal polling path, which also runs _run_once()
            # under the same single-flight guard so concurrent runs are
            # prevented regardless of the execution path taken.
            triggered = await self._trigger_scheduler.run_once_if_triggered(
                lambda: self._timed_run("event")
            )
            if not triggered:
                await self._trigger_scheduler.run_polling(
                    lambda: self._timed_run("poll")
                )
            try:
                # Wait for a new trigger or the polling interval — whichever
                # comes first.  Using wait_for_trigger() means the loop wakes
                # up immediately when _on_state_changed enqueues a record
                # instead of sleeping for the full interval.
                await asyncio.wait_for(
                    self._trigger_scheduler.wait_for_trigger(), timeout=interval
                )
            except TimeoutError:
                continue

    async def async_run_now(self) -> bool:
        """Run one sentinel evaluation cycle immediately if idle."""
        return await self._trigger_scheduler.run_now(
            lambda: self._timed_run("on_demand")
        )

    async def _timed_run(self, trigger_source: str = "poll") -> None:
        """Record run timing, call _run_once, then fire the run-complete signal."""
        _start = dt_util.utcnow()
        self.run_stats["last_run_start"] = _start.isoformat()
        try:
            await self._run_once(trigger_source)
        finally:
            _end = dt_util.utcnow()
            self.run_stats["last_run_end"] = _end.isoformat()
            self.run_stats["run_duration_ms"] = int(
                (_end - _start).total_seconds() * 1000
            )
            self.run_stats["active_rule_count"] = len(self._rules) + (
                len(self._rule_registry.list_rules())
                if self._rule_registry is not None
                else 0
            )
            self.run_stats["scheduler"] = self._trigger_scheduler.stats
            async_dispatcher_send(self._hass, SIGNAL_SENTINEL_RUN_COMPLETE)

    async def _run_once(self, trigger_source: str = "poll") -> None:  # noqa: PLR0912, PLR0915
        try:
            snapshot = await async_build_full_state_snapshot(self._hass)
        except (ValueError, TypeError, KeyError):
            LOGGER.warning("Failed to build snapshot for sentinel.")
            return

        # Force Level 0 when suppression state is in read-only mode (version
        # mismatch after a downgrade).
        if self._suppression.is_read_only:
            LOGGER.warning(
                "Suppression state is read-only (schema version mismatch); "
                "sentinel forced to Level 0 (notify-only)."
            )
            # Continue but will not dispatch (autonomy level effectively 0).

        now = dt_util.utcnow()
        cooldown_type = timedelta(
            minutes=_coerce_int(
                self._options.get(CONF_SENTINEL_COOLDOWN_MINUTES), default=30
            )
        )
        cooldown_entity = timedelta(
            minutes=_coerce_int(
                self._options.get(CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES), default=15
            )
        )
        explain_enabled = bool(self._options.get(CONF_EXPLAIN_ENABLED, False))
        pending_prompt_ttl = timedelta(
            minutes=_coerce_int(
                self._options.get(CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES),
                default=RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
            )
        )

        purged = purge_expired_prompts(
            self._suppression.state,
            now,
            pending_prompt_ttl=pending_prompt_ttl,
        )
        if purged:
            LOGGER.debug(
                "Purged %d expired pending prompt(s) (TTL=%s).",
                purged,
                pending_prompt_ttl,
            )
            await self._suppression.async_save()

        # Update presence grace windows by comparing current people_home to
        # the last known set.  Register grace for any person whose state changed.
        self._update_presence_grace(snapshot, now)

        all_findings: list[AnomalyFinding] = []
        for rule in self._rules:
            try:
                findings = rule.evaluate(snapshot)
            except (KeyError, ValueError, TypeError):
                LOGGER.warning("Sentinel rule %s failed to evaluate.", rule.rule_id)
                continue
            if findings:
                LOGGER.info(
                    "Sentinel rule %s produced %s finding(s).",
                    rule.rule_id,
                    len(findings),
                )
            all_findings.extend(findings)

        if self._rule_registry is not None:
            dynamic_rules = self._rule_registry.list_rules()
            LOGGER.debug(
                "Sentinel dynamic registry has %s rule(s).",
                len(dynamic_rules),
            )
            if dynamic_rules:
                baselines: dict[str, dict[str, float]] = {}
                dow_data: dict[str, dict[str, tuple[float, int]]] | None = None
                if self._baseline_updater is not None:
                    baselines = await self._baseline_updater.async_fetch_baselines()
                    # Inject baseline_ready_entities into derived so the discovery
                    # reducer passes them through to the LLM prompt context.
                    ready_ids = (
                        await self._baseline_updater.async_fetch_ready_entity_ids()
                    )
                    snapshot["derived"]["baseline_ready_entities"] = ready_ids
                    # Fetch DOW data when weekly_patterns is enabled.
                    weekly_patterns = bool(
                        self._options.get(
                            CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                            RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                        )
                    )
                    if weekly_patterns:
                        dow_data = await self._baseline_updater.async_fetch_dow_data()
                dow_min_samples = _coerce_int(
                    self._options.get(CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES),
                    RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
                )
                global_drift_pct = _coerce_float(
                    self._options.get(CONF_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT),
                    RECOMMENDED_SENTINEL_BASELINE_DRIFT_THRESHOLD_PCT,
                )
                dynamic_findings = evaluate_dynamic_rules(
                    snapshot,
                    dynamic_rules,
                    baselines=baselines,
                    dow_data=dow_data,
                    dow_min_samples=dow_min_samples,
                    global_drift_pct=global_drift_pct,
                )
                LOGGER.debug(
                    "Sentinel evaluated %s dynamic rule(s), produced %s finding(s).",
                    len(dynamic_rules),
                    len(dynamic_findings),
                )
                sustained_minutes = _coerce_int(
                    self._options.get(CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES),
                    RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES,
                )
                if sustained_minutes > 0:
                    dynamic_findings = self._apply_sustained_gate(
                        dynamic_findings, now, sustained_minutes
                    )
                if dynamic_findings:
                    LOGGER.info(
                        "Sentinel dynamic rules produced %s finding(s).",
                        len(dynamic_findings),
                    )
                all_findings.extend(dynamic_findings)

        if not all_findings:
            LOGGER.debug("Sentinel cycle completed with no findings.")
            return

        # Correlation pass: group related findings from this single cycle.
        # Each call to correlate() is stateless — no cross-run merging occurs.
        correlated = self._correlator.correlate(all_findings)

        for item in correlated:
            await self._dispatch_item(
                item,
                snapshot,
                now,
                cooldown_type,
                cooldown_entity,
                explain_enabled,
                trigger_source=trigger_source,
            )
        LOGGER.debug("Sentinel cycle completed with %s finding(s).", len(all_findings))

    # ---------------------------------------------------------------------- #
    # Cyclical load sustained deviation gate
    # ---------------------------------------------------------------------- #

    # Template IDs subject to the cyclical load gate.  Any new dynamic rule
    # template that represents a deviation-style alert (not a point-in-time
    # anomaly) should be added here so cyclical entities are gated correctly.
    # Template IDs are defined in sentinel/dynamic_rules.py (TEMPLATE_ID
    # constants) and sentinel/baseline.py (evaluate_baseline_deviation /
    # evaluate_time_of_day_anomaly).
    _GATED_TEMPLATE_IDS: frozenset[str] = frozenset(
        {"baseline_deviation", "time_of_day_anomaly"}
    )

    def _apply_sustained_gate(
        self,
        findings: list[AnomalyFinding],
        now: datetime,
        sustained_minutes: int,
    ) -> list[AnomalyFinding]:
        """
        Apply a sustained-presence gate to cyclical-entity deviation findings.

        For each finding whose ``template_id`` is ``baseline_deviation`` or
        ``time_of_day_anomaly`` and whose entity matches
        :data:`CYCLICAL_LOAD_HINTS`:

        * First appearance: record ``now`` in
          ``_cyclical_deviation_above_since`` and **suppress** the finding
          (start the clock).
        * Subsequent appearances while elapsed < ``sustained_minutes``:
          **suppress** (keep suppressing).
        * After ``sustained_minutes`` have elapsed: **fire** the finding,
          reset the clock to ``now`` (so re-fire requires another full
          duration).

        All other findings pass through unchanged.

        Clock cleanup: after iterating findings, any entity_id that was in
        ``_cyclical_deviation_above_since`` but did **not** appear in any
        finding this run is removed (the entity dropped back below threshold).
        Cleanup runs after gate processing so an entity that rises and falls
        within a single run is tracked from first rise and cleared at end of
        that same run.

        Note on semantics: "sustained" means the entity was present in
        findings on consecutive engine runs, not that it was continuously
        above threshold for N wall-clock minutes.  At the default 5-minute
        run cadence the approximation is close enough; document this rather
        than adding wall-clock gap tracking.

        Note on static rules: this gate is applied only to ``dynamic_findings``
        from ``evaluate_dynamic_rules()``.  If static baseline rules are ever
        added, apply the gate to their output as well.

        Note on HA restart: ``_cyclical_deviation_above_since`` is in-memory
        only and is cleared on restart.  The first compressor-on cycle after
        restart waits the full ``sustained_minutes`` before notifying.
        """
        if sustained_minutes <= 0:
            return findings

        passed: list[AnomalyFinding] = []
        # Track which entity_ids appeared in this run's cyclical findings so we
        # can clear stale entries at the end.
        seen_cyclical: set[str] = set()

        for finding in findings:
            template_id = finding.evidence.get("template_id", "")
            if template_id not in self._GATED_TEMPLATE_IDS:
                passed.append(finding)
                continue

            # Determine the entity_id for gate tracking.  Use the first entry
            # in triggering_entities (baseline rules always produce a single-
            # entity finding).  If the list is empty, pass the finding through
            # unchanged — tokenising the anomaly_id would be unreliable and
            # could produce false positive or false negative gate matches.
            if not finding.triggering_entities:
                LOGGER.warning(
                    "Cyclical gate: finding %s has no triggering_entities; "
                    "skipping gate for this finding.",
                    finding.anomaly_id,
                )
                passed.append(finding)
                continue
            entity_id = finding.triggering_entities[0]
            friendly_name = finding.evidence.get("friendly_name", "")

            if not _is_cyclical(entity_id, str(friendly_name)):
                passed.append(finding)
                continue

            seen_cyclical.add(entity_id)

            first_seen = self._cyclical_deviation_above_since.get(entity_id)
            if first_seen is None:
                # First time above threshold — start the clock, suppress.
                self._cyclical_deviation_above_since[entity_id] = now
                LOGGER.debug(
                    "Cyclical gate started for %s (sustained_minutes=%d).",
                    entity_id,
                    sustained_minutes,
                )
                continue

            elapsed = (now - first_seen).total_seconds() / 60.0
            if elapsed < sustained_minutes:
                # Still within the gate window — suppress.
                LOGGER.debug(
                    "Cyclical gate suppressing %s (%.1f / %d min elapsed).",
                    entity_id,
                    elapsed,
                    sustained_minutes,
                )
                continue

            # Gate exceeded — fire and reset clock.
            LOGGER.info(
                "Cyclical gate exceeded for %s (%.1f min); firing finding.",
                entity_id,
                elapsed,
            )
            self._cyclical_deviation_above_since[entity_id] = now
            passed.append(finding)

        # Remove entries for entities that dropped below threshold this run.
        # (Not in seen_cyclical means no cyclical finding appeared for them.)
        stale = set(self._cyclical_deviation_above_since) - seen_cyclical
        for entity_id in stale:
            LOGGER.debug(
                "Cyclical gate cleared for %s (no finding this run).", entity_id
            )
            del self._cyclical_deviation_above_since[entity_id]

        return passed

    # ---------------------------------------------------------------------- #
    # Presence grace window maintenance
    # ---------------------------------------------------------------------- #

    def _update_presence_grace(
        self,
        snapshot: FullStateSnapshot,
        now: datetime,
    ) -> None:
        """
        Compare current ``people_home`` to the last known set.

        For every person whose state has changed (departed or arrived), open a
        presence-grace window in the suppression state.
        """
        current_people_home: set[str] = set(
            snapshot.get("derived", {}).get("people_home", [])
        )

        grace_minutes = _coerce_int(
            self._options.get(CONF_SENTINEL_PRESENCE_GRACE_MINUTES),
            default=RECOMMENDED_SENTINEL_PRESENCE_GRACE_MINUTES,
        )

        changed_persons = self._last_people_home.symmetric_difference(
            current_people_home
        )
        for person_id in changed_persons:
            register_presence_grace(
                self._suppression.state,
                person_id,
                now,
                grace_minutes=grace_minutes,
            )

        self._last_people_home = current_people_home

    # ---------------------------------------------------------------------- #
    # Dispatch
    # ---------------------------------------------------------------------- #

    async def _dispatch_item(  # noqa: PLR0913
        self,
        item: AnomalyFinding | CompoundFinding,
        snapshot: FullStateSnapshot,
        now: datetime,
        cooldown_type: timedelta,
        cooldown_entity: timedelta,
        explain_enabled: bool,  # noqa: FBT001
        *,
        trigger_source: str = "poll",
    ) -> None:
        """Route a finding or compound finding through suppression and dispatch."""
        if isinstance(item, CompoundFinding):
            await self._dispatch_compound(
                item,
                snapshot,
                now,
                cooldown_type,
                cooldown_entity,
                explain_enabled,
                trigger_source=trigger_source,
            )
            return

        # Plain AnomalyFinding
        finding: AnomalyFinding = item

        # Build suppression kwargs from options.
        suppress_kwargs = _build_suppress_kwargs(self._options, snapshot)

        suppression_decision = should_suppress(
            self._suppression.state,
            finding,
            now,
            cooldown_type,
            cooldown_entity,
            **suppress_kwargs,
        )
        if suppression_decision.suppress:
            LOGGER.debug(
                "Suppressed finding %s for %s (%s).",
                finding.anomaly_id,
                finding.type,
                suppression_decision.reason_code,
            )
            await _append_finding_audit(
                self._audit_store,
                snapshot,
                finding,
                None,
                suppression_decision.reason_code,
                trigger_source=trigger_source,
            )
            return

        # Suppression state is read-only → downgrade to Level 0.
        effective_autonomy = (
            0
            if self._suppression.is_read_only
            else (
                self.get_autonomy_level(self._entry_id or "") if self._entry_id else 1
            )
        )

        # LLM triage pass (Level 1+): may suppress the finding.
        triage_decision_value: str | None = None
        triage_reason_code_value: str | None = None
        triage_confidence_value: float | None = None
        if effective_autonomy >= 1 and self._triage_service is not None:
            triage_result = await self._triage_service.triage(finding, snapshot)
            triage_decision_value = triage_result.decision
            triage_reason_code_value = triage_result.reason_code
            triage_confidence_value = triage_result.triage_confidence
            if triage_result.decision == TRIAGE_SUPPRESS:
                LOGGER.info(
                    "Triage suppressed finding %s (%s): reason=%s.",
                    finding.anomaly_id,
                    finding.type,
                    triage_result.reason_code,
                )
                await _append_finding_audit(
                    self._audit_store,
                    snapshot,
                    finding,
                    None,
                    suppression_decision.reason_code,
                    triage_decision=triage_decision_value,
                    triage_reason_code=triage_reason_code_value,
                    triage_confidence=triage_confidence_value,
                    autonomy_level_at_decision=effective_autonomy,
                    trigger_source=trigger_source,
                )
                return

        # Execution policy evaluation.
        canary_mode = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
                RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
            )
        )
        # Use the side-effect-free evaluator here; live execution state is
        # committed only after the HA service call actually succeeds.
        exec_result = self._execution_service.evaluate_canary(
            finding, snapshot, effective_autonomy, now
        )

        # Canary: record would_auto_execute without acting.
        canary_would_execute: bool | None = None
        if canary_mode:
            canary_would_execute = (
                exec_result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
            )
            if canary_would_execute:
                LOGGER.info(
                    "Canary: would auto-execute finding %s (execution_id=%s).",
                    finding.anomaly_id,
                    exec_result.execution_id,
                )

        # Live auto-execute: call HA services when policy approves and canary is off.
        action_outcome: dict[str, Any] | None = None
        if (
            exec_result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
            and not canary_mode
        ):
            action_outcome = await _auto_execute_finding(
                self._hass, finding, exec_result.execution_id
            )
            if exec_result.execution_id is not None and action_outcome["status"] in {
                "success",
                "partial",
            }:
                self._execution_service.commit_auto_execute(
                    exec_result.execution_id, now
                )

        # Always register the finding for cooldown tracking, regardless of policy.
        register_finding(self._suppression.state, finding, now)

        # BLOCKED findings: audit and skip notification — no pending user action.
        if exec_result.action_policy_path == ACTION_POLICY_BLOCKED:
            await self._suppression.async_save()
            LOGGER.debug(
                "Finding %s blocked by execution policy; no notification dispatched.",
                finding.anomaly_id,
            )
            await _append_finding_audit(
                self._audit_store,
                snapshot,
                finding,
                None,
                suppression_decision.reason_code,
                triage_decision=triage_decision_value,
                triage_reason_code=triage_reason_code_value,
                triage_confidence=triage_confidence_value,
                data_quality=exec_result.data_quality,
                action_policy_path=exec_result.action_policy_path,
                autonomy_level_at_decision=effective_autonomy,
                trigger_source=trigger_source,
            )
            return

        register_prompt(self._suppression.state, finding, now)
        await self._suppression.async_save()
        LOGGER.info(
            "Dispatching finding %s (%s) → policy=%s.",
            finding.anomaly_id,
            finding.type,
            exec_result.action_policy_path,
        )

        explanation = None
        if explain_enabled and self._explainer is not None:
            explanation = await self._explainer.async_explain(finding)

        await self._notifier.async_notify(finding, snapshot, explanation)
        await _append_finding_audit(
            self._audit_store,
            snapshot,
            finding,
            explanation,
            suppression_decision.reason_code,
            triage_decision=triage_decision_value,
            triage_reason_code=triage_reason_code_value,
            triage_confidence=triage_confidence_value,
            data_quality=exec_result.data_quality,
            action_policy_path=exec_result.action_policy_path,
            execution_id=exec_result.execution_id,
            canary_would_execute=canary_would_execute,
            action_outcome=action_outcome,
            autonomy_level_at_decision=effective_autonomy,
            trigger_source=trigger_source,
        )

    async def _dispatch_compound(  # noqa: PLR0913
        self,
        compound: CompoundFinding,
        snapshot: FullStateSnapshot,
        now: datetime,
        cooldown_type: timedelta,
        cooldown_entity: timedelta,
        explain_enabled: bool,  # noqa: FBT001
        *,
        trigger_source: str = "poll",
    ) -> None:
        """
        Apply suppression to a CompoundFinding and dispatch it when appropriate.

        A compound finding is suppressed only when **all** of its constituents
        would individually be suppressed.  When at least one constituent passes
        the suppression check, the compound is dispatched and all passing
        constituents are registered for cooldown tracking.
        """
        suppress_kwargs = _build_suppress_kwargs(self._options, snapshot)
        effective_autonomy = (
            0
            if self._suppression.is_read_only
            else (
                self.get_autonomy_level(self._entry_id or "") if self._entry_id else 1
            )
        )

        passing: list[tuple[AnomalyFinding, str]] = []
        for constituent in compound.constituent_findings:
            decision = should_suppress(
                self._suppression.state,
                constituent,
                now,
                cooldown_type,
                cooldown_entity,
                **suppress_kwargs,
            )
            if not decision.suppress:
                passing.append((constituent, decision.reason_code))

        if not passing:
            LOGGER.debug(
                "Suppressed compound finding %s (all %d constituents suppressed).",
                compound.compound_id,
                len(compound.constituent_findings),
            )
            await _append_finding_audit(
                self._audit_store,
                snapshot,
                compound,
                None,
                "suppressed",
                trigger_source=trigger_source,
            )
            return

        best = max(compound.constituent_findings, key=lambda f: f.confidence)

        canary_mode = bool(
            self._options.get(
                CONF_SENTINEL_AUTO_EXEC_CANARY_MODE,
                RECOMMENDED_SENTINEL_AUTO_EXEC_CANARY_MODE,
            )
        )
        # Use the side-effect-free evaluator here; live execution state is
        # committed only after the HA service call actually succeeds.
        exec_result = self._execution_service.evaluate_canary(
            best, snapshot, effective_autonomy, now
        )

        # Always register cooldown for passing constituents.
        for constituent, _reason in passing:
            register_finding(self._suppression.state, constituent, now)

        # BLOCKED findings: audit and skip notification — no pending user action.
        if exec_result.action_policy_path == ACTION_POLICY_BLOCKED:
            await self._suppression.async_save()
            LOGGER.debug(
                "Compound finding %s blocked by execution policy; "
                "no notification dispatched.",
                compound.compound_id,
            )
            await _append_finding_audit(
                self._audit_store,
                snapshot,
                compound,
                None,
                "not_suppressed",
                data_quality=exec_result.data_quality,
                action_policy_path=exec_result.action_policy_path,
                autonomy_level_at_decision=effective_autonomy,
                trigger_source=trigger_source,
            )
            return

        for constituent, _reason in passing:
            register_prompt(self._suppression.state, constituent, now)
        await self._suppression.async_save()

        LOGGER.info(
            "Dispatching compound finding %s (%d constituents, %d passed suppression).",
            compound.compound_id,
            len(compound.constituent_findings),
            len(passing),
        )

        canary_would_execute: bool | None = None
        if canary_mode:
            canary_would_execute = (
                exec_result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
            )

        action_outcome: dict[str, Any] | None = None
        if (
            exec_result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE
            and not canary_mode
        ):
            action_outcome = await _auto_execute_finding(
                self._hass, best, exec_result.execution_id
            )
            if exec_result.execution_id is not None and action_outcome["status"] in {
                "success",
                "partial",
            }:
                self._execution_service.commit_auto_execute(
                    exec_result.execution_id, now
                )

        explanation = None
        if explain_enabled and self._explainer is not None:
            explanation = await self._explainer.async_explain(best)

        await self._notifier.async_notify(best, snapshot, explanation)
        await _append_finding_audit(
            self._audit_store,
            snapshot,
            compound,
            explanation,
            "not_suppressed",
            triage_decision=None,
            triage_reason_code=None,
            triage_confidence=None,
            data_quality=exec_result.data_quality,
            action_policy_path=exec_result.action_policy_path,
            execution_id=exec_result.execution_id,
            canary_would_execute=canary_would_execute,
            action_outcome=action_outcome,
            autonomy_level_at_decision=effective_autonomy,
            trigger_source=trigger_source,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _auto_execute_finding(
    hass: Any,
    finding: AnomalyFinding,
    execution_id: str | None,
) -> dict[str, Any]:
    """
    Execute suggested actions for *finding* by calling HA services.

    Only actions in ``domain.service`` format are dispatched.  Actions that
    are advisory strings (no ``.``) are silently skipped.

    Returns an ``action_outcome`` dict with:
    - ``status``: ``"success"`` | ``"partial"`` | ``"error"`` | ``"no_actions"``
    - ``actions``: list of per-action result dicts
    - ``execution_id``: echoed from the caller
    """
    service_actions = [a for a in finding.suggested_actions if "." in a]

    if not service_actions:
        return {
            "status": "no_actions",
            "actions": [],
            "execution_id": execution_id,
        }

    results: list[dict[str, Any]] = []
    for action in service_actions:
        domain, _, service = action.partition(".")
        service_data: dict[str, Any] = {}
        if finding.triggering_entities:
            service_data["entity_id"] = (
                finding.triggering_entities[0]
                if len(finding.triggering_entities) == 1
                else list(finding.triggering_entities)
            )
        try:
            await hass.services.async_call(
                domain,
                service,
                service_data,
                blocking=True,
            )
            results.append({"service": action, "status": "ok", "error": None})
            LOGGER.info(
                "Auto-executed service %s for finding %s (execution_id=%s).",
                action,
                finding.anomaly_id,
                execution_id,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"service": action, "status": "error", "error": str(exc)})
            LOGGER.warning(
                "Auto-execute service %s failed for finding %s: %s.",
                action,
                finding.anomaly_id,
                exc,
            )

    error_count = sum(1 for r in results if r["status"] == "error")
    if error_count == 0:
        status = "success"
    elif error_count < len(results):
        status = "partial"
    else:
        status = "error"

    return {
        "status": status,
        "actions": results,
        "execution_id": execution_id,
    }


def _coerce_int(value: object | None, default: int) -> int:
    """Coerce option values to int with a deterministic fallback."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object | None, default: float) -> float:
    """Coerce option values to float with a deterministic fallback."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _build_suppress_kwargs(
    options: dict[str, Any],
    snapshot: FullStateSnapshot,
) -> dict[str, Any]:
    """Build keyword args for ``should_suppress()`` from options + snapshot."""
    timezone: str | None = snapshot.get("derived", {}).get("timezone")

    quiet_start_raw = options.get(CONF_SENTINEL_QUIET_HOURS_START)
    quiet_end_raw = options.get(CONF_SENTINEL_QUIET_HOURS_END)
    quiet_start: int | None = (
        int(quiet_start_raw) if quiet_start_raw is not None else None
    )
    quiet_end: int | None = int(quiet_end_raw) if quiet_end_raw is not None else None
    quiet_severities: list[str] = list(
        options.get(
            CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
            RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES,
        )
    )

    return {
        "pending_prompt_ttl": timedelta(
            minutes=_coerce_int(
                options.get(CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES),
                default=RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
            )
        ),
        "snapshot_timezone": timezone,
        "quiet_hours_start": quiet_start,
        "quiet_hours_end": quiet_end,
        "quiet_hours_severities": quiet_severities,
    }


async def _append_finding_audit(  # noqa: PLR0913
    audit_store: AuditStore,
    snapshot: FullStateSnapshot,
    finding: AnomalyFinding | CompoundFinding,
    explanation: str | None,
    suppression_reason_code: str,
    *,
    triage_decision: str | None = None,
    triage_reason_code: str | None = None,
    triage_confidence: float | None = None,
    data_quality: str | None = None,
    action_policy_path: str | None = None,
    execution_id: str | None = None,
    canary_would_execute: bool | None = None,
    action_outcome: dict | None = None,
    autonomy_level_at_decision: int | None = None,
    trigger_source: str | None = None,
) -> None:
    """Append a finding to audit with suppression reason and execution metadata."""
    await audit_store.async_append_finding(
        snapshot,
        finding,
        explanation,
        suppression_reason_code=suppression_reason_code,
        triage_decision=triage_decision,
        triage_reason_code=triage_reason_code,
        triage_confidence=triage_confidence,
        data_quality={"quality": data_quality} if data_quality else None,
        action_policy_path=action_policy_path,
        execution_id=execution_id,
        canary_would_execute=canary_would_execute,
        action_outcome=action_outcome,
        autonomy_level_at_decision=(
            str(autonomy_level_at_decision)
            if autonomy_level_at_decision is not None
            else None
        ),
        trigger_source=trigger_source,
    )
