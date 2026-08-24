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
import math
import re
import unicodedata
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
    CONF_SENTINEL_RESPONSE_LANGUAGE,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
    SNOOZE_24H,
    SNOOZE_PERMANENT,
)
from custom_components.home_generative_agent.core.utils import extract_final
from custom_components.home_generative_agent.sentinel.models import enrolled_people
from custom_components.home_generative_agent.sentinel.notifier_messages import (
    notif_msg,
)
from custom_components.home_generative_agent.sentinel.power_units import (
    is_energy_unit,
    is_power_unit,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SUPPRESSION_REASON_NOT_SUPPRESSED,
    record_cooldown_feedback,
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
# Severity title strings now live in notifier_messages.py (localized).
# Localized severity words (digest summary, persistent fallback). Unknown
# severities must NOT be routed through notif_msg as keys: stored audit data
# could collide with an unrelated message id and surface its copy.
_SEVERITY_WORD_KEYS: dict[str, str] = {
    "high": "severity_word_high",
    "medium": "severity_word_medium",
    "low": "severity_word_low",
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
                parts = time_str.split(":")
                hour, minute = int(parts[0]), int(parts[1])
            except (ValueError, TypeError, IndexError):
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
        title_key = {
            "high": "severity_title_high",
            "low": "severity_title_low",
        }.get(severity, "severity_title_medium")
        title = notif_msg(self._hass, title_key)
        interrupt_level = _SEVERITY_INTERRUPT_LEVEL.get(severity, "active")
        subtitle = _build_subtitle(finding, self._hass)
        mobile_msg = _mobile_message(
            clean_explanation,
            finding,
            str(self._options.get(CONF_SENTINEL_RESPONSE_LANGUAGE, "") or ""),
            self._hass,
        )
        persistent_msg = _persistent_message(clean_explanation, finding, self._hass)
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
                # Rate limit exceeded — buffer this finding.  Hold the redacted
                # explanation, never the raw one: the flush discards it today,
                # but storing the unredacted text would silently bypass
                # _redact_if_sensitive the moment anyone renders it.
                self._held_batch.append((finding, clean_explanation, target_service))
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
            LOGGER.debug("Sending sentinel notification via %s.", target_service)
            await self._hass.services.async_call(domain, service, data, blocking=False)
        else:
            LOGGER.debug("Sending sentinel notification via persistent_notification.")
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
                for _entity_id in finding.triggering_entities:
                    record_cooldown_feedback(
                        self._suppression.state, _entity_id, finding.type
                    )
                await self._suppression.async_save()
                LOGGER.debug(
                    "Snooze 24 h registered for finding type %s.", finding.type
                )

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
                LOGGER.debug(
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
        types = list({_display_type(f, self._hass) for f, _, _svc in held})
        type_summary = ", ".join(types)
        message = notif_msg(
            self._hass,
            "batch_message",
            count=count,
            plural="s" if count > 1 else "",
            type_summary=type_summary,
        )
        batch_title = notif_msg(self._hass, "batch_title")

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
                "title": batch_title,
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
                        "title": batch_title,
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
        except Exception:
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
            # Stored records are untrusted shapes: "finding" may be null or
            # a non-dict, and severity may be None/non-string. None of that
            # may break sorted() below or kill the digest task.
            finding_rec = r.get("finding")
            sev_raw = (
                finding_rec.get("severity") if isinstance(finding_rec, dict) else None
            )
            sev = str(sev_raw or "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        sev_summary = ", ".join(
            f"{v} {notif_msg(self._hass, _SEVERITY_WORD_KEYS[k]) if k in _SEVERITY_WORD_KEYS else k}"  # noqa: E501
            for k, v in sorted(severity_counts.items())
        )
        digest_title = notif_msg(self._hass, "digest_title")
        message = notif_msg(
            self._hass,
            "digest_message",
            count=count,
            plural="s" if count > 1 else "",
            sev_summary=sev_summary,
        )

        notify_service = self._options.get(CONF_NOTIFY_SERVICE)
        if notify_service and isinstance(notify_service, str):
            domain, _, service = notify_service.partition(".")
            if not service:
                service = notify_service
                domain = "notify"
            data: dict[str, Any] = {
                "title": digest_title,
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
                    "title": digest_title,
                    "message": message,
                    "notification_id": "hga_sentinel_daily_digest",
                },
                blocking=False,
            )
        LOGGER.debug("Daily digest sent: %s.", message)

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

        friendly = _display_type(finding, self._hass)
        confirm_action = f"{ACTION_PREFIX}{ACT_SNOOZE_CONFIRM}_{finding.anomaly_id}"
        cancel_action = f"{ACTION_PREFIX}{ACT_SNOOZE_CANCEL}_{finding.anomaly_id}"
        domain, _, service = notify_service.partition(".")
        if not service:
            service = notify_service
            domain = "notify"

        data: dict[str, Any] = {
            "title": notif_msg(self._hass, "snooze_confirm_title"),
            "message": notif_msg(
                self._hass, "snooze_confirm_message", friendly=friendly
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
        elif "arm_alarm" in finding.suggested_actions:
            actions.append(
                {
                    "action": f"{ACTION_PREFIX}execute_{finding.anomaly_id}",
                    "title": "Arm Alarm",
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

    When ``finding.is_sensitive`` is True, enrolled names in
    ``finding.evidence["recognized_people"]`` are replaced with the generic
    phrase ``"a recognised person"``.  Reserved pipeline labels ("Unknown
    Person", "Indeterminate") are never redacted — they are not private
    identities, and substituting "Unknown Person" would rewrite "an unknown
    person was seen" into "a recognised person was seen", inverting the
    security meaning of the very findings that carry the label.  Returns the
    original string unchanged when the finding is not sensitive or there are
    no names to redact.
    """
    if not explanation or not finding.is_sensitive:
        return explanation

    recognized: list[Any] = finding.evidence.get("recognized_people", []) or []
    names = enrolled_people(
        person for person in recognized if isinstance(person, str) and person
    )
    if not names:
        return explanation

    redacted = explanation
    for person in names:
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


# Anomaly-type / template-id keys mapped to notif_msg() message ids. Values
# are message keys in notifier_messages.py, not display text -- the actual
# strings are localized there.
_KNOWN_TYPE_LABEL_KEYS = {
    "open_entry_while_away": "type_open_entry_while_away",
    "open_entry_at_night_when_home": "type_open_entry_at_night",
    "open_entry_at_night_when_home_window": "type_open_entry_at_night",
    "open_entry_at_night_while_away": "type_open_entry_at_night",
    "open_entry_at_night": "type_open_entry_at_night",
    "open_entry_at_night_window": "type_open_entry_at_night",
    "open_entry_at_night_door": "type_open_entry_at_night",
    "open_entry_at_night_entry": "type_open_entry_at_night",
    "open_any_window_at_night_while_away": "type_open_any_window_at_night_while_away",
    "motion_detected_at_night_while_away": "type_motion_detected_at_night_while_away",
    "motion_detected_while_away": "type_motion_detected_while_away",
    "unlocked_lock_at_night": "type_unlocked_lock_at_night",
    "camera_entry_unsecured": "type_camera_entry_unsecured",
    "alarm_disarmed_during_external_threat": (
        "type_alarm_disarmed_during_external_threat"
    ),
    "appliance_power_duration": "type_appliance_power_duration",
}


def _display_type(finding: AnomalyFinding, hass: HomeAssistant | None = None) -> str:
    """
    Friendly label for a finding.

    Dynamic rules carry slugified candidate IDs as ``finding.type``; when the
    type has no curated label but the rule's template does, prefer the
    template label so notifications never show raw candidate slugs
    (issue #516 review).
    """
    if finding.type in _KNOWN_TYPE_LABEL_KEYS:
        return notif_msg(hass, _KNOWN_TYPE_LABEL_KEYS[finding.type])
    template_id = str(finding.evidence.get("template_id") or "")
    if template_id in _KNOWN_TYPE_LABEL_KEYS:
        return notif_msg(hass, _KNOWN_TYPE_LABEL_KEYS[template_id])
    return _friendly_type(finding.type, hass)


def _friendly_type(anomaly_type: str, hass: HomeAssistant | None = None) -> str:
    if anomaly_type in _KNOWN_TYPE_LABEL_KEYS:
        return notif_msg(hass, _KNOWN_TYPE_LABEL_KEYS[anomaly_type])
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
    name_lower = name.lower()
    for suffix in _POWER_SUFFIXES:
        if name_lower.endswith(suffix.lower()):
            return name[: -len(suffix)].strip()
    return name


def _is_power_class_evidence(ev: dict[str, Any]) -> bool:
    """
    Return True when a baseline finding's entity is a power/energy sensor.

    Baseline rules can target any numeric sensor (humidity, temperature, …),
    but the power/appliance copy only fits power-class entities.  Findings
    persisted before unit/device_class were captured in evidence fall back to
    the entity_id substring heuristic the copy previously relied on.
    """
    device_class = str(ev.get("device_class") or "")
    if device_class in {"power", "energy"}:
        return True
    unit = str(ev.get("unit_of_measurement") or "")
    if unit and (is_power_unit(unit) or is_energy_unit(unit)):
        # A power-dimension unit wins even under an exotic device_class
        # (e.g. energy_storage sensors reporting kWh).
        return True
    if "device_class" in ev or "unit_of_measurement" in ev:
        # New finding whose captured metadata says non-power; never fall back
        # to the entity_id heuristic here or unitless sensors named like
        # sensor.energy_score would get appliance copy.
        return False
    # Legacy findings persisted before the metadata was captured.
    entity_id = str(ev.get("entity_id") or "")
    return "power" in entity_id or "energy" in entity_id


def _appliance_power_duration_mobile_message(finding: AnomalyFinding) -> str:
    """Deterministic mobile copy for appliance_power_duration."""
    ev = finding.evidence
    raw_name = (ev.get("friendly_name") or "").strip()
    if raw_name:
        appliance = _strip_power_suffix(raw_name)
    else:
        entity_id = str(
            ev.get("entity_id")
            or (finding.triggering_entities[0] if finding.triggering_entities else "")
        )
        appliance = _strip_power_suffix(_friendly_entity(entity_id))

    power_w = ev.get("power_w")
    duration_min = ev.get("duration_min")
    threshold_min = ev.get("threshold_min")

    power_str = (
        f"about {round(float(power_w))} W" if power_w is not None else "high power"
    )
    dur_str = (
        f"{int(duration_min)} min" if duration_min is not None else "an extended period"
    )
    thr_str = (
        f"{int(threshold_min)} min" if threshold_min is not None else "the configured"
    )

    msg = (
        f"{appliance} drew {power_str} for {dur_str},"
        f" above the {thr_str} threshold. Check it."
    )
    return msg[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _build_subtitle(finding: AnomalyFinding, hass: HomeAssistant | None = None) -> str:
    """Return the notification subtitle line for *finding*."""
    if finding.evidence.get("is_completion"):
        raw_name = str(finding.evidence.get("friendly_name") or "").strip()
        if not raw_name and finding.triggering_entities:
            raw_name = _friendly_entity(finding.triggering_entities[0])
        appliance = _strip_power_suffix(raw_name).title()
        if appliance:
            return notif_msg(hass, "subtitle_appliance_finished", appliance=appliance)
        return notif_msg(hass, "subtitle_appliance_cycle_complete")
    if finding.evidence.get("template_id") == "alarm_disarmed_open_entry":
        entry_id = str(finding.evidence.get("entry_entity_id") or "")
        entry_name = (
            _friendly_entity(entry_id)
            if entry_id
            else notif_msg(hass, "fallback_entry")
        )
        return notif_msg(
            hass, "subtitle_entry_open_alarm_disarmed", entry_name=entry_name
        )
    if finding.evidence.get("template_id") in {
        "baseline_deviation",
        "time_of_day_anomaly",
    }:
        raw_name = str(finding.evidence.get("friendly_name") or "").strip()
        if not raw_name and finding.triggering_entities:
            raw_name = _friendly_entity(finding.triggering_entities[0])
        appliance = _strip_power_suffix(raw_name).title() or notif_msg(
            hass, "fallback_sensor"
        )
        deviation = str(finding.evidence.get("deviation_direction") or "")
        direction_key = (
            "direction_lower" if deviation == "below" else "direction_higher"
        )
        direction = notif_msg(hass, direction_key)
        subtitle_key = (
            "subtitle_power_deviation"
            if _is_power_class_evidence(finding.evidence)
            else "subtitle_reading_deviation"
        )
        return notif_msg(hass, subtitle_key, appliance=appliance, direction=direction)
    return _display_type(finding, hass)


def _fallback_message(
    finding: AnomalyFinding, hass: HomeAssistant | None = None
) -> str:
    summary = _display_type(finding, hass)
    entity = (
        _friendly_entity(finding.triggering_entities[0])
        if finding.triggering_entities
        else notif_msg(hass, "fallback_unknown_entity")
    )
    return notif_msg(
        hass,
        "fallback_message",
        summary=summary,
        entity=entity,
        action_hint=_severity_action_hint(finding.severity, hass),
    )


def _format_disarm_since(parsed: datetime) -> str:
    """
    Return a human-readable 'since …' string for a disarm timestamp.

    Shows time-only when the disarm was today; prepends the date otherwise so
    a disarm from days ago is not mistaken for earlier today.
    """
    local = dt_util.as_local(parsed)
    local_now = dt_util.now()
    time_str = local.strftime("%-I:%M %p")
    if local.date() == local_now.date():
        return time_str
    return local.strftime(f"%-d %b at {time_str}")


def _alarm_disarmed_mobile_message(finding: AnomalyFinding) -> str:
    """Deterministic mobile copy for alarm_disarmed_during_external_threat."""
    ev = finding.evidence

    cam_name: str = (ev.get("camera_friendly_name") or "").strip()[:30]
    if not cam_name:
        cam_id = str(ev.get("camera_entity_id", ""))
        if cam_id:
            cam_name = cam_id.partition(".")[2].replace("_", " ").title()[:30]
        else:
            cam_name = ""
    if not cam_name:
        cam_name = "A camera"

    age = ev.get("camera_activity_age_minutes")
    if age is not None:
        age_mins = max(1, round(float(age)))
        activity_phrase = f"{cam_name} saw an unrecognized person {age_mins} min ago."
    else:
        # Unreachable for newly generated findings — the rule's freshness
        # gate guarantees a numeric age — but findings persisted before that
        # gate existed can still re-render through this branch.
        activity_phrase = f"{cam_name} saw an unrecognized person."

    last_changed = ev.get("alarm_last_changed")
    alarm_phrase = ""
    if last_changed:
        parsed = dt_util.parse_datetime(str(last_changed))
        if parsed is not None:
            alarm_phrase = (
                f" The alarm has been disarmed since {_format_disarm_since(parsed)}."
            )
    if not alarm_phrase:
        alarm_phrase = " The alarm is currently disarmed."

    cta = " Arm the alarm or view the camera."
    return (activity_phrase + alarm_phrase + cta)[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _alarm_disarmed_open_entry_mobile_message(finding: AnomalyFinding) -> str:
    """Deterministic mobile copy for alarm_disarmed_open_entry dynamic rule."""
    ev = finding.evidence
    entry_id = str(ev.get("entry_entity_id") or "")
    entry_name = _friendly_entity(entry_id) if entry_id else "An entry"

    alarm_last_changed = ev.get("alarm_last_changed")
    if alarm_last_changed:
        parsed = dt_util.parse_datetime(str(alarm_last_changed))
        if parsed is not None:
            alarm_phrase = f"Alarm disarmed since {_format_disarm_since(parsed)}."
        else:
            alarm_phrase = "Alarm is disarmed."
    else:
        alarm_phrase = "Alarm is disarmed."

    msg = f"{entry_name} is open. {alarm_phrase} Close it or snooze if expected."
    return msg[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _baseline_deviation_mobile_message(finding: AnomalyFinding) -> str:
    """Deterministic mobile copy for baseline_deviation and time_of_day_anomaly."""
    ev = finding.evidence
    raw_name = (ev.get("friendly_name") or "").strip()
    if not raw_name and finding.triggering_entities:
        raw_name = _friendly_entity(finding.triggering_entities[0])
    appliance = _strip_power_suffix(raw_name).title() or "Sensor"

    current_value = ev.get("current_value")
    # DOW-blended time_of_day_anomaly findings carry the comparison value as
    # expected_value, not baseline_value (see _evaluate_dow_anomaly).
    baseline_value = ev.get("baseline_value")
    if baseline_value is None:
        baseline_value = ev.get("expected_value")
    deviation_pct = ev.get("deviation_pct")
    direction = str(ev.get("deviation_direction") or "")

    is_power = _is_power_class_evidence(ev)
    # Untrusted entity attribute: drop control/format characters (bidi
    # overrides can visually reorder push text), collapse whitespace, and cap
    # before it is embedded in notification copy.
    raw_unit = str(ev.get("unit_of_measurement") or "")
    unit = " ".join(
        "".join(ch for ch in raw_unit if unicodedata.category(ch)[0] != "C").split()
    )[:12]
    if not unit and is_power and "unit_of_measurement" not in ev:
        # Findings persisted before the unit was captured in evidence: power
        # circuits report watts by convention, energy circuits kWh.  Key
        # ABSENCE is the legacy signal — a present-but-empty unit means a
        # genuinely unitless sensor, where fabricating a unit would be wrong.
        unit = "kWh" if "energy" in str(ev.get("entity_id") or "") else "W"
    cta = "Check appliance." if is_power else "Worth checking."

    direction_word = "below" if direction == "below" else "above"
    # Persisted evidence re-renders through here: values may be missing,
    # non-numeric, or non-finite (poisoned baselines).  Rendering must never
    # raise — a crash here kills the dispatch path.
    pct_val: int | None = None
    if deviation_pct is not None:
        try:
            pct_f = float(deviation_pct)
            pct_val = round(pct_f) if math.isfinite(pct_f) else None
        except (TypeError, ValueError):
            pct_val = None
    have_values = False
    cur_f = base_f = 0.0
    if current_value is not None and baseline_value is not None:
        try:
            cur_f = float(current_value)
            base_f = float(baseline_value)
            have_values = math.isfinite(cur_f) and math.isfinite(base_f)
        except (TypeError, ValueError):
            have_values = False
    if have_values:
        cur = round(cur_f, 1)
        base = round(base_f, 1)
        pct = f" ({pct_val}% {direction_word} normal)" if pct_val is not None else ""
        msg = f"{appliance}: {cur}{unit} vs usual {base}{unit}{pct}. {cta}"
    else:
        noun = "power" if is_power else "reading"
        dev = f" {pct_val}%" if pct_val is not None else ""
        msg = f"{appliance} {noun}{dev} {direction_word} normal. {cta}"
    return msg[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _entity_staleness_mobile_message(finding: AnomalyFinding) -> str:
    """Deterministic mobile copy for entity_staleness findings."""
    ev = finding.evidence
    entity_id = str(ev.get("entity_id") or "").strip()
    raw_name = str(ev.get("friendly_name") or "").strip()
    if not raw_name and entity_id:
        raw_name = _friendly_entity(entity_id)

    age_hours = ev.get("age_hours")
    age_str = "an extended period"
    if age_hours is not None:
        try:
            age_h = float(age_hours)
            if age_h >= 48:  # noqa: PLR2004
                age_str = f"about {int(age_h // 24)} days"
            elif age_h >= 24:  # noqa: PLR2004
                age_str = "about 1 day"
            elif age_h >= 2:  # noqa: PLR2004
                age_str = f"about {int(age_h)} hours"
            else:
                age_str = "about 1 hour"
        except (TypeError, ValueError):
            pass

    if entity_id.startswith("person."):
        msg = (
            f"{raw_name}'s location tracking has been outdated for {age_str}. "
            f"Check if their phone is on and reachable."
        )
    else:
        msg = (
            f"{raw_name or 'Sensor'} data has been outdated for {age_str}. "
            f"Please check the sensor."
        )
    return msg[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


_TEMPLATE_MOBILE_FORMATTERS: dict[
    str,
    Any,
] = {
    "alarm_disarmed_open_entry": _alarm_disarmed_open_entry_mobile_message,
    "baseline_deviation": _baseline_deviation_mobile_message,
    "time_of_day_anomaly": _baseline_deviation_mobile_message,
    "entity_staleness": _entity_staleness_mobile_message,
}


# Deterministic copy that carries security actuation detail — which camera,
# which entry, when the alarm was disarmed, and what to do about it. An LLM
# paraphrase blurs those or asserts things the evidence does not support
# ("someone is still inside"), so these never defer to a translated
# explanation, even when a response language is configured. Translating them
# needs real string templates, not model prose.
_SECURITY_MESSAGE_TYPES = frozenset({"alarm_disarmed_during_external_threat"})
_SECURITY_MESSAGE_TEMPLATE_IDS = frozenset({"alarm_disarmed_open_entry"})


def _is_security_copy(finding: AnomalyFinding) -> bool:
    """Return True when *finding*'s deterministic copy must never be paraphrased."""
    return (
        finding.type in _SECURITY_MESSAGE_TYPES
        or str(finding.evidence.get("template_id") or "")
        in _SECURITY_MESSAGE_TEMPLATE_IDS
    )


def _deterministic_mobile_message(finding: AnomalyFinding) -> str | None:
    """
    Return the hardcoded-English template message for *finding*, or ``None``.

    These formatters render exact figures, units, and entity names that an LLM
    paraphrase can blur, so they are preferred whenever the notification is
    meant to be English.
    """
    if finding.type == "alarm_disarmed_during_external_threat":
        return _alarm_disarmed_mobile_message(finding)
    if finding.type == "appliance_power_duration":
        return _appliance_power_duration_mobile_message(finding)
    formatter = _TEMPLATE_MOBILE_FORMATTERS.get(
        str(finding.evidence.get("template_id") or "")
    )
    if formatter:
        return formatter(finding)
    return None


def _mobile_message(
    explanation: str | None,
    finding: AnomalyFinding,
    response_language: str = "",
    hass: HomeAssistant | None = None,
) -> str:
    """
    Return the mobile push body for *finding*.

    Deterministic per-template formatters win by default.  When a response
    language is configured the translated *explanation* wins instead for
    informational findings: the deterministic strings are English-only, so
    they would otherwise be the one part of the notification that ignores the
    setting — and because ``_persistent_message`` already prefers the
    explanation, the mobile push and the persistent notification would
    disagree for the same finding.

    Security copy (see ``_is_security_copy``) is exempt and stays
    deterministic in every case: losing the camera, the entry, the disarm
    time, or the call to action matters more than the language it is in.

    Falls back to the deterministic string when no translation is usable
    (explainer disabled or empty explanation): accurate English beats a
    generic English fallback.
    """
    deterministic = _deterministic_mobile_message(finding)
    if deterministic is not None and (
        not response_language or _is_security_copy(finding)
    ):
        return deterministic
    if explanation:
        text = _normalize_text(explanation)
        if text and len(text) <= MAX_MOBILE_MESSAGE_CHARS:
            return text
    if deterministic is not None:
        return deterministic
    return _fallback_message(finding, hass)[:MAX_MOBILE_MESSAGE_CHARS].rstrip()


def _persistent_message(
    explanation: str | None,
    finding: AnomalyFinding,
    hass: HomeAssistant | None = None,
) -> str:
    if explanation:
        text = _normalize_text(explanation)
        if text:
            return text

    if finding.type == "appliance_power_duration":
        return _appliance_power_duration_mobile_message(finding)

    entities = ", ".join(
        _friendly_entity(entity) for entity in finding.triggering_entities
    )
    entities = entities or notif_msg(hass, "fallback_unknown_entity")
    severity_key = _SEVERITY_WORD_KEYS.get(finding.severity)
    severity_word = notif_msg(hass, severity_key) if severity_key else finding.severity
    return notif_msg(
        hass,
        "persistent_fallback",
        summary=_display_type(finding, hass),
        severity=severity_word,
        entities=entities,
        hint=_severity_action_hint(finding.severity, hass),
    )


def _severity_action_hint(severity: str, hass: HomeAssistant | None = None) -> str:
    key = {"high": "action_hint_high", "medium": "action_hint_medium"}.get(
        severity, "action_hint_low"
    )
    return notif_msg(hass, key)
