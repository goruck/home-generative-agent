"""Sentinel engine for proactive anomaly detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    CONF_EXPLAIN_ENABLED,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)

from .dynamic_rules import evaluate_dynamic_rules
from .rules.appliance_power_duration import AppliancePowerDurationRule
from .rules.camera_entry_unsecured import CameraEntryUnsecuredRule
from .rules.open_entry_while_away import OpenEntryWhileAwayRule
from .rules.unlocked_lock_at_night import UnlockedLockAtNightRule
from .suppression import (
    SuppressionManager,
    register_finding,
    register_prompt,
    should_suppress,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.explain.llm_explain import LLMExplainer
    from custom_components.home_generative_agent.notify.dispatcher import (
        NotificationDispatcher,
    )

    from .models import AnomalyFinding
    from .rule_registry import RuleRegistry

LOGGER = logging.getLogger(__name__)


class SentinelEngine:
    """Periodic sentinel evaluation loop."""

    def __init__(  # noqa: PLR0913
        self,
        hass: HomeAssistant,
        options: dict[str, object],
        suppression: SuppressionManager,
        notifier: NotificationDispatcher,
        audit_store: AuditStore,
        explainer: LLMExplainer | None = None,
        *,
        rule_registry: RuleRegistry | None = None,
    ) -> None:
        """Initialize sentinel dependencies and runtime state."""
        self._hass = hass
        self._options = options
        self._suppression = suppression
        self._notifier = notifier
        self._audit_store = audit_store
        self._explainer = explainer
        self._rule_registry = rule_registry
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._rules = [
            UnlockedLockAtNightRule(),
            OpenEntryWhileAwayRule(),
            AppliancePowerDurationRule(),
            CameraEntryUnsecuredRule(),
        ]

    def start(self) -> None:
        """Start the sentinel loop."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = self._hass.async_create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the sentinel loop."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run_loop(self) -> None:
        interval = _coerce_int(
            self._options.get(CONF_SENTINEL_INTERVAL_SECONDS), default=300
        )
        LOGGER.info("Sentinel loop started (interval=%ss).", interval)
        while not self._stop_event.is_set():
            await self._run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def _run_once(self) -> None:
        try:
            snapshot = await async_build_full_state_snapshot(self._hass)
        except (ValueError, TypeError, KeyError):
            LOGGER.warning("Failed to build snapshot for sentinel.")
            return

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
                dynamic_findings = evaluate_dynamic_rules(snapshot, dynamic_rules)
                LOGGER.debug(
                    "Sentinel evaluated %s dynamic rule(s), produced %s finding(s).",
                    len(dynamic_rules),
                    len(dynamic_findings),
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

        for finding in all_findings:
            if should_suppress(
                self._suppression.state, finding, now, cooldown_type, cooldown_entity
            ):
                LOGGER.debug(
                    "Suppressed finding %s for %s.",
                    finding.anomaly_id,
                    finding.type,
                )
                continue
            register_finding(self._suppression.state, finding, now)
            register_prompt(self._suppression.state, finding, now)
            await self._suppression.async_save()
            LOGGER.info(
                "Dispatching finding %s (%s).",
                finding.anomaly_id,
                finding.type,
            )

            explanation = None
            if explain_enabled and self._explainer is not None:
                explanation = await self._explainer.async_explain(finding)

            await self._notifier.async_notify(finding, snapshot, explanation)
            await self._audit_store.async_append_finding(snapshot, finding, explanation)
        LOGGER.debug("Sentinel cycle completed with %s finding(s).", len(all_findings))


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
