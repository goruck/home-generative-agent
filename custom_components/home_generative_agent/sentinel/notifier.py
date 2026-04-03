"""
Sentinel notification orchestrator — Issue #261.

Provides ``SentinelNotifier``, which wraps the notification dispatch layer
with:

* Action buttons: primary action (Ask Agent / Execute), False Alarm, Snooze
  24 h, Snooze Always.
* ``always`` confirmation guard: a permanent snooze fires a confirmation
  notification before writing to ``SuppressionState``; no HA service is
  called until the user explicitly confirms.
* Per-area routing: when ``CONF_SENTINEL_AREA_NOTIFY_MAP`` maps an area name
  to a notify service, findings whose triggering entities belong to that area
  are routed to that service instead of the global one.
* ``is_sensitive`` redaction: recognised-person names in the explanation text
  are replaced with ``"a recognised person"`` before the message is sent.

Non-snooze action callbacks (``execute``, ``handoff``, ``dismiss``) are
delegated to ``ActionHandler``.  The ``dismiss`` action sets
``user_response.false_positive = True`` in the audit record.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import callback
from homeassistant.helpers.event import async_call_later, async_track_time_change
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    ACT_SNOOZE_24H,
    ACT_SNOOZE_ALWAYS,
    ACT_SNOOZE_CANCEL,
    ACT_SNOOZE_CONFIRM,
    ACTION_PREFIX,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_AREA_NOTIFY_MAP,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
    SNOOZE_24H,
    SNOOZE_PERMANENT,
)
from custom_components.home_generative_agent.core.utils import extract_final
from custom_components.home_generative_agent.sentinel.suppression import (
    SUPPRESSION_REASON_NOT_SUPPRESSED,
    register_snooze,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from homeassistant.core import Event, HomeAssistant

    from custom_components.home_generative_agent.audit.store import AuditStore
    from custom_components.home_generative_agent.notify.actions import ActionHandler
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

    from .suppression import SuppressionManager

LOGGER = logging.getLogger(__name__)

MAX_MOBILE_MESSAGE_CHARS = 220
_AUDIT_FETCH_LIMIT = 1000

_SEVERITY_INTERRUPT_LEVEL: dict[str, str] = {
    "high": "time-sensitive",
    "medium": "active",
    "low": "passive",
}
_SEVERITY_TITLE: dict[str, str] = {
    "high": "Security Alert",
    "medium": "Home Alert",
    "low": "Home Update",
}

# Notification batching / rate-limiting.
_BATCH_RATE_LIMIT = 3
_BATCH_RATE_WINDOW_SECS = 60
_BATCH_FLUSH_DELAY_SECS = 30

# Per-finding cooldown: suppress repeated notifications for the same anomaly.
_FINDING_COOLDOWN_SECS = 1800  # 30 minutes

_SNOOZE_VERBS = frozenset(
    {
        ACT_SNOOZE_24H,
        ACT_SNOOZE_ALWAYS,
        ACT_SNOOZE_CONFIRM,
        ACT_SNOOZE_CANCEL,
    }
)


class SentinelNotifier:
    """
    Notification orchestrator for sentinel findings.

    Notification orchestrator injected into ``SentinelEngine``.  Exposes
    ``async_notify(finding, snapshot, explanation)`` and ``start()`` /
    ``stop()`` lifecycle methods.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        options: dict[str, Any],
        suppression: SuppressionManager,
        action_handler: ActionHandler,
        audit_store: AuditStore | None = None,
    ) -> None:
        """Initialise the sentinel notifier."""
        self._hass = hass
        self._options = options
        self._suppression = suppression
        self._action_handler = action_handler
        self._audit_store = audit_store
        self._unsub: Callable[[], None] | None = None
        # Pending permanent-snooze intents: anomaly_id -> finding_type.
        # Written when the user taps "Snooze Always"; cleared on confirm/cancel.
        self._pending_always_snooze: dict[str, str] = {}
        # Notification batching state.
        self._notification_times: list[datetime] = []
        self._held_batch: list[tuple[AnomalyFinding, str | None, str | None]] = []
        self._batch_cancel: Callable[[], None] | None = None
        # Per-finding cooldown: anomaly_id -> time of last dispatch.
        self._cooldown_times: dict[str, datetime] = {}
        # Daily digest time-change unsubscribe handle.
        self._digest_unsub: Callable[[], None] | None = None
        # Task spawned by the daily digest callback; kept so stop() can cancel it.
        self._digest_task: asyncio.Task[None] | None = None

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def start(self) -> None:
        """Subscribe to mobile app action events and start the daily digest timer."""
        if self._unsub is not None:
            return
        self._unsub = self._hass.bus.async_listen(
            "mobile_app_notification_action",
            self._handle_action_event,
        )
        enabled = bool(
            self._options.get(
                CONF_SENTINEL_DAILY_DIGEST_ENABLED,
                RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
            )
        )
        if enabled and self._audit_store is not None:
            if self._digest_unsub is not None:
                self._digest_unsub()
                self._digest_unsub = None
            time_str = str(
                self._options.get(
                    CONF_SENTINEL_DAILY_DIGEST_TIME,
                    RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
                )
            )
            try:
                hour, minute = (int(p) for p in time_str.split(":", 1))
            except (ValueError, TypeError):
                LOGGER.warning(
                    "Invalid daily digest time %r; defaulting to 08:00.", time_str
                )
                hour, minute = 8, 0
            self._digest_unsub = async_track_time_change(
                self._hass,
                self._async_send_daily_digest,
                hour=hour,
                minute=minute,
                second=0,
            )

    def stop(self) -> None:
        """Unsubscribe from action events and cancel the daily digest timer."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        if self._batch_cancel is not None:
            self._batch_cancel()
            self._batch_cancel = None
        if self._digest_unsub is not None:
            self._digest_unsub()
            self._digest_unsub = None
        if self._digest_task is not None:
            self._digest_task.cancel()
            self._digest_task = None

    # ---------------------------------------------------------------------- #
    # Notification dispatch
    # ---------------------------------------------------------------------- #

    async def async_notify(
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
        explanation: str | None,
    ) -> None:
        """
        Send a proactive notification for *finding*.

        * Adds snooze action buttons.
        * Redacts person names when ``finding.is_sensitive`` is True.
        * Routes to a per-area notify service when configured.
        """
        # Register finding with the action handler so execute/handoff work.
        self._action_handler.register_finding(finding)

        # Sensitive redaction before building the message.
        clean_explanation = _redact_if_sensitive(explanation, finding)

        severity = finding.severity
        title = _SEVERITY_TITLE.get(severity, "Home Alert")
        interrupt_level = _SEVERITY_INTERRUPT_LEVEL.get(severity, "active")
        if finding.evidence.get("is_completion"):
            # Use the appliance's display name so the subtitle reads
            # "Dishwasher finished" rather than a raw rule-type slug.
            # Strip trailing power-sensor suffixes ("Power", "Wattage", etc.)
            # so "Dishwasher Power" → "Dishwasher" → "Dishwasher finished".
            raw_name = str(finding.evidence.get("friendly_name") or "").strip()
            if not raw_name and finding.triggering_entities:
                raw_name = _friendly_entity(finding.triggering_entities[0])
            appliance = _strip_power_suffix(raw_name).title()
            subtitle = (
                f"{appliance} finished" if appliance else "Appliance cycle complete"
            )
        else:
            subtitle = _friendly_type(finding.type)
        mobile_msg = _mobile_message(clean_explanation, finding)
        persistent_msg = _persistent_message(clean_explanation, finding)
        actions = _build_actions(finding)

        # Per-area routing.
        target_service = _resolve_notify_service(finding, snapshot, self._options)

        # Shared timestamp for both cooldown and burst-batching guards.
        now_dt = datetime.now()  # noqa: DTZ005

        # Per-finding cooldown: suppress if the same anomaly fired recently.
        # High-severity always bypasses so security events are never silenced.
        if severity != "high":
            last_fired = self._cooldown_times.get(finding.anomaly_id)
            if (
                last_fired is not None
                and (now_dt - last_fired).total_seconds() < _FINDING_COOLDOWN_SECS
            ):
                LOGGER.debug(
                    "Sentinel suppressed duplicate finding %s (cooldown %ds).",
                    finding.anomaly_id,
                    _FINDING_COOLDOWN_SECS,
                )
                return
            # Record cooldown before the batch check so that batched findings
            # also consume their cooldown slot (prevents re-fire after flush).
            self._cooldown_times[finding.anomaly_id] = now_dt
            # Prune entries older than 2x the cooldown window to bound memory.
            cutoff_cd = now_dt - timedelta(seconds=_FINDING_COOLDOWN_SECS * 2)
            self._cooldown_times = {
                aid: ts for aid, ts in self._cooldown_times.items() if ts >= cutoff_cd
            }

        # Batching / rate-limiting for non-high severity.
        if severity != "high":
            cutoff = now_dt - timedelta(seconds=_BATCH_RATE_WINDOW_SECS)
            self._notification_times = [
                t for t in self._notification_times if t >= cutoff
            ]
            if len(self._notification_times) >= _BATCH_RATE_LIMIT:
                # Rate limit exceeded — buffer this finding.
                self._held_batch.append((finding, explanation, target_service))
                if self._batch_cancel is None:
                    self._batch_cancel = async_call_later(
                        self._hass,
                        _BATCH_FLUSH_DELAY_SECS,
                        self._async_flush_batch,
                    )
                return
            self._notification_times.append(now_dt)

        if target_service:
            domain, _, service = target_service.partition(".")
            if not service:
                service = target_service
                domain = "notify"
            tag = f"hga_sentinel_{finding.anomaly_id[:32]}"
            data: dict[str, Any] = {
                "title": title,
                "message": mobile_msg,
                "data": {
                    "actions": actions,
                    "tag": tag,
                    "subtitle": subtitle,
                    "push": {"interruption-level": interrupt_level},
                },
            }
            LOGGER.info("Sending sentinel notification via %s.", target_service)
            await self._hass.services.async_call(domain, service, data, blocking=False)
        else:
            LOGGER.info("Sending sentinel notification via persistent_notification.")
            await self._hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": persistent_msg,
                    "notification_id": f"hga_sentinel_{finding.anomaly_id}",
                },
                blocking=False,
            )

    # ---------------------------------------------------------------------- #
    # Action event handling
    # ---------------------------------------------------------------------- #

    @callback
    def _handle_action_event(self, event: Event) -> None:
        """Dispatch incoming mobile-action events."""
        action = event.data.get("action")
        if not isinstance(action, str):
            return
        if not action.startswith(ACTION_PREFIX):
            return

        stripped = action.removeprefix(ACTION_PREFIX)

        # Try to match a snooze verb prefix.
        for verb in _SNOOZE_VERBS:
            prefix = f"{verb}_"
            if stripped.startswith(prefix):
                anomaly_id = stripped[len(prefix) :]
                self._hass.async_create_task(self._handle_snooze(verb, anomaly_id))
                return

        # Non-snooze action — delegate to ActionHandler.
        self._hass.async_create_task(
            self._action_handler.handle_action(action, dict(event.data))
        )

    async def _handle_snooze(self, verb: str, anomaly_id: str) -> None:
        """Process a snooze action for *anomaly_id*."""
        now = dt_util.utcnow()
        finding = self._action_handler._pending_findings.get(  # noqa: SLF001
            anomaly_id
        )

        if verb == ACT_SNOOZE_24H:
            if finding:
                register_snooze(self._suppression.state, finding.type, SNOOZE_24H, now)
                await self._suppression.async_save()
                LOGGER.info("Snooze 24 h registered for finding type %s.", finding.type)

        elif verb == ACT_SNOOZE_ALWAYS:
            # Guard: send confirmation notification; do NOT write snooze yet.
            if finding:
                self._pending_always_snooze[anomaly_id] = finding.type
                await self._send_always_confirmation(finding)

        elif verb == ACT_SNOOZE_CONFIRM:
            # User confirmed — write permanent snooze now.
            finding_type = self._pending_always_snooze.pop(anomaly_id, None)
            if finding_type:
                register_snooze(
                    self._suppression.state, finding_type, SNOOZE_PERMANENT, now
                )
                await self._suppression.async_save()
                LOGGER.info(
                    "Permanent snooze confirmed for finding type %s.", finding_type
                )
            else:
                LOGGER.debug(
                    "Snooze confirm for %s but no pending intent; ignoring.",
                    anomaly_id,
                )

        elif verb == ACT_SNOOZE_CANCEL:
            self._pending_always_snooze.pop(anomaly_id, None)
            LOGGER.debug("Permanent snooze cancelled for %s.", anomaly_id)

    @callback
    def _async_flush_batch(self, _now: Any = None) -> None:
        """Flush held batch of non-high-severity notifications as a single summary."""
        self._batch_cancel = None
        held = self._held_batch[:]
        self._held_batch.clear()
        if not held:
            return

        count = len(held)
        types = list({_friendly_type(f.type) for f, _, _svc in held})
        type_summary = ", ".join(types)
        message = f"{count} home update{'s' if count > 1 else ''}: {type_summary}."

        # Use the first non-None resolved service from the held batch (which
        # already incorporated the area map), then fall back to the global service.
        target_service = next(
            (svc for _f, _e, svc in held if svc is not None), None
        ) or self._options.get(CONF_NOTIFY_SERVICE)
        if target_service and isinstance(target_service, str):
            domain, _, service = target_service.partition(".")
            if not service:
                service = target_service
                domain = "notify"
            data: dict[str, Any] = {
                "title": "Home Update",
                "message": message,
                "data": {
                    "tag": "hga_sentinel_batch_summary",
                    "push": {"interruption-level": "passive"},
                },
            }
            self._hass.async_create_task(
                self._hass.services.async_call(domain, service, data, blocking=False)
            )
        else:
            self._hass.async_create_task(
                self._hass.services.async_call(
                    "persistent_notification",
                    "create",
                    {
                        "title": "Home Update",
                        "message": message,
                        "notification_id": "hga_sentinel_batch_summary",
                    },
                    blocking=False,
                )
            )

    @callback
    def _async_send_daily_digest(self, _now: Any = None) -> None:
        """Schedule the async daily digest coroutine from the time-change callback."""
        if self._digest_task is not None and not self._digest_task.done():
            LOGGER.debug(
                "Daily digest already in progress; skipping duplicate trigger."
            )
            return
        self._digest_task = self._hass.async_create_task(self._async_run_daily_digest())

    async def _async_run_daily_digest(self) -> None:
        """Fetch the last 24 hours of notified findings and push a summary."""
        if self._audit_store is None:
            return

        cutoff = dt_util.utcnow() - timedelta(hours=24)
        try:
            records = await self._audit_store.async_get_latest(_AUDIT_FETCH_LIMIT)
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Daily digest: failed to fetch audit records.", exc_info=True
            )
            return

        notified = [
            r
            for r in records
            if r.get("suppression_reason_code") == SUPPRESSION_REASON_NOT_SUPPRESSED
            and _record_notified_after(r, cutoff)
        ]
        count = len(notified)
        if count == 0:
            LOGGER.debug("Daily digest: no notified findings in the last 24 h.")
            return

        severity_counts: dict[str, int] = {}
        for r in notified:
            sev = r.get("finding", {}).get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        sev_summary = ", ".join(f"{v} {k}" for k, v in sorted(severity_counts.items()))
        message = (
            f"Sentinel: {count} alert{'s' if count > 1 else ''} in the last 24 h"
            f" ({sev_summary})."
        )

        notify_service = self._options.get(CONF_NOTIFY_SERVICE)
        if notify_service and isinstance(notify_service, str):
            domain, _, service = notify_service.partition(".")
            if not service:
                service = notify_service
                domain = "notify"
            data: dict[str, Any] = {
                "title": "Sentinel Daily Digest",
                "message": message,
                "data": {
                    "tag": "hga_sentinel_daily_digest",
                    "push": {"interruption-level": "passive"},
                },
            }
            await self._hass.services.async_call(domain, service, data, blocking=False)
        else:
            await self._hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "Sentinel Daily Digest",
                    "message": message,
                    "notification_id": "hga_sentinel_daily_digest",
                },
                blocking=False,
            )
        LOGGER.info("Daily digest sent: %s.", message)

    async def _send_always_confirmation(self, finding: AnomalyFinding) -> None:
        """
        Send a mobile confirmation notification for permanent snooze.

        No HA action is taken until the user taps Confirm.
        """
        notify_service = self._options.get(CONF_NOTIFY_SERVICE)
        if not notify_service or not isinstance(notify_service, str):
            LOGGER.debug(
                "No notify_service configured; cannot send snooze confirmation."
            )
            return

        friendly = _friendly_type(finding.type)
        confirm_action = f"{ACTION_PREFIX}{ACT_SNOOZE_CONFIRM}_{finding.anomaly_id}"
        cancel_action = f"{ACTION_PREFIX}{ACT_SNOOZE_CANCEL}_{finding.anomaly_id}"
        domain, _, service = notify_service.partition(".")
        if not service:
            service = notify_service
            domain = "notify"

        data: dict[str, Any] = {
            "title": "Confirm permanent snooze",
            "message": (
                f"Permanently suppress '{friendly}' alerts? "
                "This can only be undone from settings."
            ),
            "data": {
                "actions": [
                    {"action": confirm_action, "title": "Confirm"},
                    {"action": cancel_action, "title": "Cancel"},
                ],
                "tag": f"hga_sentinel_snooze_{finding.anomaly_id[:32]}",
            },
        }
        await self._hass.services.async_call(domain, service, data, blocking=False)
        LOGGER.debug(
            "Permanent snooze confirmation sent for finding %s.", finding.anomaly_id
        )


# ---------------------------------------------------------------------------
# Action list builder
# ---------------------------------------------------------------------------


def _build_actions(finding: AnomalyFinding) -> list[dict[str, Any]]:
    """
    Build mobile action buttons for *finding*.

    Primary action (execute or ask) is first, then False Alarm, then snooze.
    """
    actions: list[dict[str, Any]] = []

    if finding.suggested_actions:
        if finding.is_sensitive:
            actions.append(
                {
                    "action": f"{ACTION_PREFIX}handoff_{finding.anomaly_id}",
                    "title": "Ask Agent",
                }
            )
        else:
            actions.append(
                {
                    "action": f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
                    "title": "Execute",
                }
            )

    actions.extend(
        [
            {
                "action": f"{ACTION_PREFIX}dismiss_{finding.anomaly_id}",
                "title": "False Alarm",
            },
            {
                "action": f"{ACTION_PREFIX}{ACT_SNOOZE_24H}_{finding.anomaly_id}",
                "title": "Snooze 24 h",
            },
            {
                "action": f"{ACTION_PREFIX}{ACT_SNOOZE_ALWAYS}_{finding.anomaly_id}",
                "title": "Snooze Always",
            },
        ]
    )
    return actions


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _redact_if_sensitive(
    explanation: str | None, finding: AnomalyFinding
) -> str | None:
    """
    Return *explanation* with recognised person names replaced.

    When ``finding.is_sensitive`` is True, any string values in
    ``finding.evidence["recognized_people"]`` are replaced with the generic
    phrase ``"a recognised person"``.  Returns the original string unchanged
    when the finding is not sensitive or there are no names to redact.
    """
    if not explanation or not finding.is_sensitive:
        return explanation

    recognized: list[Any] = finding.evidence.get("recognized_people", [])
    if not recognized:
        return explanation

    redacted = explanation
    for person in recognized:
        if isinstance(person, str) and person:
            redacted = re.sub(
                re.escape(person), "a recognised person", redacted, flags=re.IGNORECASE
            )
    return redacted


def _resolve_notify_service(
    finding: AnomalyFinding,
    snapshot: FullStateSnapshot,
    options: dict[str, Any],
) -> str | None:
    """
    Return the notify service to use for *finding*.

    Checks the per-area map first; falls back to the global notify service.
    Returns ``None`` when neither is configured (→ persistent notification).
    """
    area_map: dict[str, str] = options.get(CONF_SENTINEL_AREA_NOTIFY_MAP) or {}
    if area_map:
        area = _get_finding_area(finding, snapshot)
        if area and area in area_map:
            return area_map[area]

    global_service = options.get(CONF_NOTIFY_SERVICE)
    return global_service if isinstance(global_service, str) else None


def _get_finding_area(
    finding: AnomalyFinding, snapshot: FullStateSnapshot
) -> str | None:
    """Return the area of the first triggering entity found in the snapshot."""
    entity_map = {e["entity_id"]: e for e in snapshot.get("entities", [])}
    for entity_id in finding.triggering_entities:
        entity = entity_map.get(entity_id)
        if entity:
            area = entity.get("area")
            if area:
                return str(area)
    return None


def _normalize_text(text: str) -> str:
    text = extract_final(text)  # strips <think> blocks and collapses whitespace
    return text.replace("**", "").replace("`", "")


def _friendly_type(anomaly_type: str) -> str:
    known = {
        "open_entry_while_away": "Open entry while away",
        "open_entry_at_night_when_home": "Open entry at night",
        "open_entry_at_night_when_home_window": "Open entry at night",
        "open_entry_at_night_while_away": "Open entry at night",
        "open_any_window_at_night_while_away": "Window open at night",
        "unlocked_lock_at_night": "Door lock left unlocked",
        "camera_entry_unsecured": "Activity near unsecured entry",
    }
    if anomaly_type in known:
        return known[anomaly_type]
    # Strip internal prefixes so they never appear in user-visible text:
    # • "candidate_"          — LLM-proposed dynamic rules awaiting approval
    # • "rule_NN_"            — LLM-generated rules with sequential numbering
    #                           e.g. "rule_02_high_energy_consumption_away"
    display = anomaly_type.removeprefix("candidate_")
    # Strip "rule_<digits>_" prefix (e.g. "rule_02_")
    parts = display.split("_")
    if len(parts) >= 3 and parts[0] == "rule" and parts[1].isdigit():  # noqa: PLR2004
        display = "_".join(parts[2:])
    return display.replace("_", " ").strip().capitalize()


def _record_notified_after(record: dict[str, Any], cutoff: datetime) -> bool:
    """Return True if *record* has a notification timestamp on or after *cutoff*."""
    notified_at_str = record.get("notification", {}).get("notified_at")
    if not notified_at_str:
        return False
    notified_dt = dt_util.parse_datetime(str(notified_at_str))
    if notified_dt is None:
        return False
    return notified_dt >= cutoff


def _friendly_entity(entity_id: str) -> str:
    if "." in entity_id:
        _, _, name = entity_id.partition(".")
    else:
        name = entity_id
    return name.replace("_", " ").strip().title()


# Suffixes appended to HA power-sensor entity names that don't describe the
# appliance itself (e.g. "Dishwasher Power" → strip " Power" → "Dishwasher").
_POWER_SUFFIXES: tuple[str, ...] = (
    " Power",
    " Wattage",
    " Energy",
    " Consumption",
    " Usage",
    " Draw",
    " Load",
)


def _strip_power_suffix(name: str) -> str:
    """Remove trailing power-sensor label words from an appliance display name."""
    for suffix in _POWER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _fallback_message(finding: AnomalyFinding) -> str:
    summary = _friendly_type(finding.type)
    entity = (
        _friendly_entity(finding.triggering_entities[0])
        if finding.triggering_entities
        else "Unknown entity"
    )
    return f"{summary}: {entity}. {_severity_action_hint(finding.severity)}"


def _mobile_message(explanation: str | None, finding: AnomalyFinding) -> str:
    if explanation:
        text = _normalize_text(explanation)
        if text and len(text) <= MAX_MOBILE_MESSAGE_CHARS:
            return text
    return _fallback_message(finding)[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _persistent_message(explanation: str | None, finding: AnomalyFinding) -> str:
    if explanation:
        text = _normalize_text(explanation)
        if text:
            return text

    entities = ", ".join(
        _friendly_entity(entity) for entity in finding.triggering_entities
    )
    entities = entities or "Unknown entity"
    return (
        f"{_friendly_type(finding.type)} "
        f"(severity {finding.severity}) for {entities}. "
        + _severity_action_hint(finding.severity)
    )


def _severity_action_hint(severity: str) -> str:
    if severity == "high":
        return "Urgent: check and secure it now."
    if severity == "medium":
        return "Check soon and secure it if unexpected."
    return "Review when convenient."
