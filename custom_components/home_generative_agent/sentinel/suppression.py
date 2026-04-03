"""Suppression and cooldown handling for sentinel findings — Issue #260."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    SNOOZE_7D,
    SNOOZE_24H,
    SNOOZE_PERMANENT,
)

if TYPE_CHECKING:
    from datetime import datetime

    from homeassistant.core import HomeAssistant

    from .models import AnomalyFinding

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_suppression"

# Current in-memory schema version.  Increment when new fields are added.
SUPPRESSION_STATE_VERSION = 4

# Maximum factor by which learned feedback can extend the base entity cooldown.
MAX_COOLDOWN_MULTIPLIER = 8

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suppression reason codes
# ---------------------------------------------------------------------------

SUPPRESSION_REASON_NOT_SUPPRESSED = "not_suppressed"
SUPPRESSION_REASON_PENDING_PROMPT = "pending_prompt"
SUPPRESSION_REASON_TYPE_COOLDOWN = "type_cooldown"
SUPPRESSION_REASON_ENTITY_COOLDOWN = "entity_cooldown"
SUPPRESSION_REASON_QUIET_HOURS = "quiet_hours"
SUPPRESSION_REASON_PRESENCE_GRACE = "presence_grace"
SUPPRESSION_REASON_USER_SNOOZE_24H = "user_snooze_24h"
SUPPRESSION_REASON_USER_SNOOZE_7D = "user_snooze_7d"
SUPPRESSION_REASON_USER_SNOOZE_PERMANENT = "user_snooze_permanent"
PENDING_PROMPT_DEFAULT_TTL = timedelta(hours=4)

# Finding types considered presence-sensitive.  These are suppressed when a
# person is within their departure/arrival grace window.
_PRESENCE_SENSITIVE_TYPES: frozenset[str] = frozenset(
    {
        "open_entry_while_away",
        "unknown_person_camera_no_home",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuppressionDecision:
    """Structured suppression decision."""

    suppress: bool
    reason_code: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuppressionState:
    """Persistent suppression state (v4 schema)."""

    last_by_type: dict[str, str] = field(default_factory=dict)
    last_by_entity: dict[str, dict[str, str]] = field(default_factory=dict)
    pending_prompts: dict[str, str] = field(default_factory=dict)

    # v2 additions
    version: int = field(default=SUPPRESSION_STATE_VERSION)
    # Finding type → {"until": ISO-str or "permanent", "code": reason_code}
    snoozed_until: dict[str, dict[str, str]] = field(default_factory=dict)
    # Person entity ID → ISO expiry string
    presence_grace_until: dict[str, str] = field(default_factory=dict)

    # v4 additions
    # Entity ID → learned cooldown multiplier (1 = base, up to MAX_COOLDOWN_MULTIPLIER).
    # Bumped each time feedback is recorded via record_cooldown_feedback().
    learned_cooldown_multipliers: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuppressionState:
        """Create suppression state from persisted data."""
        return cls(
            last_by_type=dict(data.get("last_by_type", {})),
            last_by_entity={
                key: dict(value)
                for key, value in data.get("last_by_entity", {}).items()
            },
            pending_prompts=dict(data.get("pending_prompts", {})),
            version=int(data.get("version", SUPPRESSION_STATE_VERSION)),
            snoozed_until={
                k: dict(v) for k, v in data.get("snoozed_until", {}).items()
            },
            presence_grace_until=dict(data.get("presence_grace_until", {})),
            learned_cooldown_multipliers={
                k: int(v)
                for k, v in data.get("learned_cooldown_multipliers", {}).items()
            },
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert suppression state to persisted data."""
        return {
            "last_by_type": dict(self.last_by_type),
            "last_by_entity": {
                key: dict(value) for key, value in self.last_by_entity.items()
            },
            "pending_prompts": dict(self.pending_prompts),
            "version": self.version,
            "snoozed_until": {k: dict(v) for k, v in self.snoozed_until.items()},
            "presence_grace_until": dict(self.presence_grace_until),
            "learned_cooldown_multipliers": dict(self.learned_cooldown_multipliers),
        }


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _migrate_suppression_state(data: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate persisted suppression state to the current version in-place.

    v1 → v2: add ``snoozed_until``, ``presence_grace_until``, ``version``.
    v2 → v3: drop ``presence_grace_until`` keys that are not ``person.*``
             entity IDs (previously keyed by friendly name).
    v3 → v4: add ``learned_cooldown_multipliers`` (empty dict).

    Returns the same dict object (mutations are in-place).
    """
    stored_version = int(data.get("version", 1))
    if stored_version >= SUPPRESSION_STATE_VERSION:
        return data

    # v1 → v2
    if stored_version < 2:  # noqa: PLR2004
        data.setdefault("snoozed_until", {})
        data.setdefault("presence_grace_until", {})
        data["version"] = 2
        stored_version = 2
        LOGGER.info("Suppression state migrated from v1 to v2.")

    # v2 → v3: presence_grace_until keys must be person entity IDs
    if stored_version < 3:  # noqa: PLR2004
        stale_keys = [
            k
            for k in data.get("presence_grace_until", {})
            if not k.startswith("person.")
        ]
        for key in stale_keys:
            LOGGER.warning(
                "Dropping stale presence_grace_until key %r "
                "(not a person entity ID — was written using friendly name).",
                key,
            )
            del data["presence_grace_until"][key]
        data["version"] = 3
        stored_version = 3
        LOGGER.info("Suppression state migrated from v2 to v3.")

    # v3 → v4: learned per-entity cooldown multipliers
    if stored_version < 4:  # noqa: PLR2004
        data.setdefault("learned_cooldown_multipliers", {})
        data["version"] = 4
        LOGGER.info("Suppression state migrated from v3 to v4.")

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return dt_util.parse_datetime(value)


def _is_pending_prompt_expired(
    last_seen: datetime | None,
    now: datetime,
    *,
    pending_prompt_ttl: timedelta,
) -> bool:
    """Return True when a pending prompt timestamp is missing/invalid/expired."""
    if last_seen is None or last_seen.tzinfo is None:
        return True
    try:
        return (now - last_seen) >= pending_prompt_ttl
    except TypeError:
        return True


def _check_pending_prompt(
    state: SuppressionState,
    finding: AnomalyFinding,
    now: datetime,
    *,
    pending_prompt_ttl: timedelta,
) -> SuppressionDecision | None:
    """
    Return a suppression decision for a pending prompt, if still active.

    Entries with missing/invalid timestamps are treated as expired and removed.
    """
    pending_prompt_last_seen = state.pending_prompts.get(finding.anomaly_id)
    if pending_prompt_last_seen is None:
        return None

    parsed_last_seen = _parse_dt(pending_prompt_last_seen)
    if _is_pending_prompt_expired(
        parsed_last_seen,
        now,
        pending_prompt_ttl=pending_prompt_ttl,
    ):
        del state.pending_prompts[finding.anomaly_id]
        return None

    LOGGER.debug(
        "Suppressing %s (%s): pending prompt exists.",
        finding.anomaly_id,
        finding.type,
    )
    return SuppressionDecision(
        suppress=True,
        reason_code=SUPPRESSION_REASON_PENDING_PROMPT,
        context={
            "anomaly_id": finding.anomaly_id,
            "last_seen": pending_prompt_last_seen,
        },
    )


def purge_expired_prompts(
    state: SuppressionState,
    now: datetime,
    *,
    pending_prompt_ttl: timedelta,
) -> int:
    """
    Remove expired pending prompts in-place.

    Returns the number of entries removed (0 when nothing was purged).
    """
    expired: list[str] = []
    for anomaly_id, last_seen_iso in state.pending_prompts.items():
        last_seen = _parse_dt(last_seen_iso)
        if _is_pending_prompt_expired(
            last_seen,
            now,
            pending_prompt_ttl=pending_prompt_ttl,
        ):
            expired.append(anomaly_id)

    for anomaly_id in expired:
        del state.pending_prompts[anomaly_id]
    return len(expired)


def _check_snooze(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> str | None:
    """
    Return the snooze reason code if *finding* is actively snoozed, else None.

    Expired snooze entries are cleaned up in-place.
    """
    entry = state.snoozed_until.get(finding.type)
    if entry is None:
        return None

    until = entry.get("until", "")
    code = entry.get("code", SUPPRESSION_REASON_USER_SNOOZE_PERMANENT)

    if until == SNOOZE_PERMANENT:
        return code

    expiry = _parse_dt(until)
    if expiry is not None and now < expiry:
        return code

    # Expired — clean up
    del state.snoozed_until[finding.type]
    return None


def _check_presence_grace(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> bool:
    """
    Return True if any person is in their grace window.

    The finding must be presence-sensitive. Expired entries are cleaned up in-place.
    """
    if finding.type not in _PRESENCE_SENSITIVE_TYPES:
        return False

    expired_keys: list[str] = []
    in_grace = False

    for person_id, expiry_iso in state.presence_grace_until.items():
        expiry = _parse_dt(expiry_iso)
        if expiry is None or now >= expiry:
            expired_keys.append(person_id)
            continue
        in_grace = True

    for key in expired_keys:
        del state.presence_grace_until[key]

    return in_grace


def _is_quiet_hours(  # noqa: PLR0913
    now: datetime,
    *,
    start_hour: int | None,
    end_hour: int | None,
    timezone: str | None,
    severity: str,
    quiet_severities: list[str],
) -> bool:
    """
    Return True if *now* falls within the configured quiet-hours window for *severity*.

    ``start_hour`` and ``end_hour`` are local hours (0-23).  When
    ``start_hour > end_hour`` the window wraps midnight
    (e.g. start=22, end=7 means 22:00-07:00).

    Returns False when quiet-hours config is absent or incomplete.
    """
    if start_hour is None or end_hour is None:
        return False
    if severity not in quiet_severities:
        return False

    import zoneinfo  # noqa: PLC0415

    try:
        tz = zoneinfo.ZoneInfo(timezone or "UTC")
    except (zoneinfo.ZoneInfoNotFoundError, KeyError):
        tz = zoneinfo.ZoneInfo("UTC")

    local_now = now.astimezone(tz)
    local_hour = local_now.hour

    if start_hour <= end_hour:
        return start_hour <= local_hour < end_hour
    # Wraps midnight
    return local_hour >= start_hour or local_hour < end_hour


# ---------------------------------------------------------------------------
# Core suppression logic
# ---------------------------------------------------------------------------


def should_suppress(  # noqa: PLR0911, PLR0913
    state: SuppressionState,
    finding: AnomalyFinding,
    now: datetime,
    cooldown_type: timedelta,
    cooldown_entity: timedelta,
    *,
    pending_prompt_ttl: timedelta = PENDING_PROMPT_DEFAULT_TTL,
    snapshot_timezone: str | None = None,
    quiet_hours_start: int | None = None,
    quiet_hours_end: int | None = None,
    quiet_hours_severities: list[str] | None = None,
) -> SuppressionDecision:
    """
    Determine whether a finding should be suppressed.

    Evaluation order
    ----------------
    1. Pending prompt
    2. Active snooze
    3. Presence grace window
    4. Quiet hours
    5. Type cooldown
    6. Entity cooldown
    """
    # 1. Pending prompt
    pending_prompt_decision = _check_pending_prompt(
        state,
        finding,
        now,
        pending_prompt_ttl=pending_prompt_ttl,
    )
    if pending_prompt_decision is not None:
        return pending_prompt_decision

    # 2. Snooze
    snooze_code = _check_snooze(state, finding, now)
    if snooze_code is not None:
        LOGGER.debug(
            "Suppressing %s (%s): snooze active (%s).",
            finding.anomaly_id,
            finding.type,
            snooze_code,
        )
        return SuppressionDecision(
            suppress=True,
            reason_code=snooze_code,
            context={"type": finding.type},
        )

    # 3. Presence grace
    if _check_presence_grace(state, finding, now):
        LOGGER.debug(
            "Suppressing %s (%s): presence grace window active.",
            finding.anomaly_id,
            finding.type,
        )
        return SuppressionDecision(
            suppress=True,
            reason_code=SUPPRESSION_REASON_PRESENCE_GRACE,
            context={"type": finding.type},
        )

    # 4. Quiet hours
    if _is_quiet_hours(
        now,
        start_hour=quiet_hours_start,
        end_hour=quiet_hours_end,
        timezone=snapshot_timezone,
        severity=finding.severity,
        quiet_severities=quiet_hours_severities or [],
    ):
        LOGGER.debug(
            "Suppressing %s (%s): quiet hours active for severity=%s.",
            finding.anomaly_id,
            finding.type,
            finding.severity,
        )
        return SuppressionDecision(
            suppress=True,
            reason_code=SUPPRESSION_REASON_QUIET_HOURS,
            context={"severity": finding.severity},
        )

    # 5. Type cooldown
    last_type = _parse_dt(state.last_by_type.get(finding.type))
    if last_type is not None and now - last_type < cooldown_type:
        LOGGER.debug(
            "Suppressing %s (%s): type cooldown active (last=%s).",
            finding.anomaly_id,
            finding.type,
            state.last_by_type.get(finding.type),
        )
        return SuppressionDecision(
            suppress=True,
            reason_code=SUPPRESSION_REASON_TYPE_COOLDOWN,
            context={
                "type": finding.type,
                "last_seen": state.last_by_type.get(finding.type),
            },
        )

    # 6. Entity cooldown (scaled by any learned multiplier for the entity)
    for entity_id in finding.triggering_entities:
        multiplier = min(
            state.learned_cooldown_multipliers.get(entity_id, 1),
            MAX_COOLDOWN_MULTIPLIER,
        )
        effective_cooldown = cooldown_entity * multiplier
        last_entity = _parse_dt(
            state.last_by_entity.get(entity_id, {}).get(finding.type)
        )
        if last_entity is not None and now - last_entity < effective_cooldown:
            LOGGER.debug(
                "Suppressing %s (%s): entity cooldown active for %s "
                "(last=%s, multiplier=%d).",
                finding.anomaly_id,
                finding.type,
                entity_id,
                state.last_by_entity.get(entity_id, {}).get(finding.type),
                multiplier,
            )
            return SuppressionDecision(
                suppress=True,
                reason_code=SUPPRESSION_REASON_ENTITY_COOLDOWN,
                context={
                    "entity_id": entity_id,
                    "type": finding.type,
                    "last_seen": state.last_by_entity.get(entity_id, {}).get(
                        finding.type
                    ),
                    "multiplier": multiplier,
                },
            )

    return SuppressionDecision(
        suppress=False,
        reason_code=SUPPRESSION_REASON_NOT_SUPPRESSED,
        context={},
    )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_finding(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> None:
    """Register a finding occurrence for cooldown tracking."""
    iso = dt_util.as_utc(now).isoformat()
    state.last_by_type[finding.type] = iso
    for entity_id in finding.triggering_entities:
        per_entity = state.last_by_entity.setdefault(entity_id, {})
        per_entity[finding.type] = iso


def register_prompt(
    state: SuppressionState, finding: AnomalyFinding, now: datetime
) -> None:
    """Register an active prompt to avoid duplicates."""
    state.pending_prompts[finding.anomaly_id] = dt_util.as_utc(now).isoformat()


def resolve_prompt(state: SuppressionState, anomaly_id: str) -> None:
    """Resolve an active prompt once user responds."""
    state.pending_prompts.pop(anomaly_id, None)


def register_snooze(
    state: SuppressionState,
    finding_type: str,
    duration: str,
    now: datetime,
) -> None:
    """
    Register a snooze for *finding_type*.

    *duration* must be one of ``SNOOZE_24H``, ``SNOOZE_7D``, or
    ``SNOOZE_PERMANENT``.  Callers are responsible for obtaining explicit user
    confirmation before writing a permanent snooze.
    """
    if duration == SNOOZE_24H:
        expiry = dt_util.as_utc(now + timedelta(hours=24)).isoformat()
        code = SUPPRESSION_REASON_USER_SNOOZE_24H
    elif duration == SNOOZE_7D:
        expiry = dt_util.as_utc(now + timedelta(days=7)).isoformat()
        code = SUPPRESSION_REASON_USER_SNOOZE_7D
    elif duration == SNOOZE_PERMANENT:
        expiry = SNOOZE_PERMANENT
        code = SUPPRESSION_REASON_USER_SNOOZE_PERMANENT
    else:
        LOGGER.warning("Unknown snooze duration %r; ignoring.", duration)
        return

    state.snoozed_until[finding_type] = {"until": expiry, "code": code}
    LOGGER.info(
        "Snooze registered for finding type %s (duration=%s, code=%s).",
        finding_type,
        duration,
        code,
    )


def record_cooldown_feedback(state: SuppressionState, entity_id: str) -> int:
    """
    Increment the learned cooldown multiplier for *entity_id* by one.

    Call this when a user signals that an entity is generating too many
    alerts (e.g. via a snooze or explicit feedback action).  The multiplier
    is capped at ``MAX_COOLDOWN_MULTIPLIER`` and applied to the base entity
    cooldown in :func:`should_suppress`.

    Returns the new multiplier value.
    """
    current = state.learned_cooldown_multipliers.get(entity_id, 1)
    new_value = min(current + 1, MAX_COOLDOWN_MULTIPLIER)
    state.learned_cooldown_multipliers[entity_id] = new_value
    LOGGER.debug(
        "Learned cooldown multiplier for %s updated: %d → %d.",
        entity_id,
        current,
        new_value,
    )
    return new_value


def register_presence_grace(
    state: SuppressionState,
    person_entity_id: str,
    now: datetime,
    grace_minutes: int = 10,
) -> None:
    """
    Open a presence-grace window for *person_entity_id*.

    Called when a person's state changes (departure or arrival) so that
    transient presence-sensitive findings are suppressed during the
    transition period.
    """
    expiry = dt_util.as_utc(now + timedelta(minutes=grace_minutes)).isoformat()
    state.presence_grace_until[person_entity_id] = expiry
    LOGGER.debug(
        "Presence grace opened for %s (expires %s).",
        person_entity_id,
        expiry,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class SuppressionManager:
    """Manage suppression persistence for sentinel findings."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize persistent suppression state management."""
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._state = SuppressionState()
        self._read_only = False

    @property
    def state(self) -> SuppressionState:
        """Return current suppression state."""
        return self._state

    @property
    def is_read_only(self) -> bool:
        """
        Return True when the store was loaded in read-only compatibility mode.

        This occurs when the persisted schema version is newer than the
        current code version (e.g. after a downgrade).  No new writes are
        accepted in this mode.
        """
        return self._read_only

    async def async_load(self) -> None:
        """Load suppression state from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if not isinstance(data, dict):
            return

        stored_version = int(data.get("version", 1))
        if stored_version > SUPPRESSION_STATE_VERSION:
            # Newer schema than we understand — load read-only, force Level 0.
            LOGGER.warning(
                "Suppression state version %d is newer than supported version %d. "
                "Loading in read-only compatibility mode; sentinel forced to Level 0.",
                stored_version,
                SUPPRESSION_STATE_VERSION,
            )
            self._read_only = True
            # Still load what we can with safe defaults for unknown fields.
            self._state = SuppressionState.from_dict(data)
            return

        migrated = _migrate_suppression_state(data)
        self._state = SuppressionState.from_dict(migrated)

    async def async_save(self) -> None:
        """Persist suppression state to storage."""
        if self._read_only:
            LOGGER.debug("Suppression state is read-only; skipping save.")
            return
        try:
            await self._store.async_save(self._state.as_dict())
        except (HomeAssistantError, OSError, ValueError):
            return
