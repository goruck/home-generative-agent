"""Template-based proposal normalization for discovery candidates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from custom_components.home_generative_agent.const import (
    SENTINEL_OCCUPANCY_ARMED_STATES,
)

from .evidence_paths import (
    ANYONE_HOME_FALSE_PATTERN,
    ANYONE_HOME_PATH,
    ANYONE_HOME_TRUE_PATTERN,
    AWAY_TERMS_PATTERN,
    NOT_ANYONE_HOME_PATH,
    NOT_ANYONE_HOME_TEXT_PATTERN,
    PresenceSignal,
    has_derived_path,
    night_signal,
    presence_signal,
)

SUPPORTED_TEMPLATES = {
    "alarm_disarmed_open_entry",
    "low_battery_sensors",
    "motion_detected_at_night_while_alarm_disarmed",
    "motion_detected_at_night_while_away",
    "motion_detected_while_away",
    "motion_while_alarm_disarmed_and_home_present",
    "unavailable_sensors",
    "unavailable_sensors_while_home",
    "open_any_window_at_night_while_away",
    "open_entry_when_home",
    "open_entry_while_away",
    "open_entry_at_night",
    "open_entry_at_night_when_home",
    "open_entry_at_night_while_away",
    "unlocked_lock_when_home",
    "unlocked_lock_while_away",
    "motion_without_camera_activity",
    "unknown_person_camera_no_home",
    "unknown_person_camera_when_home",
    # Issue #265 — baseline-driven detectors
    "baseline_deviation",
    "time_of_day_anomaly",
    # Issue #266 — lambda/expression rules
    "lambda",
    # Flexible templates for common patterns
    "alarm_state_mismatch",
    "entity_state_duration",
    "sensor_threshold_condition",
    "entity_staleness",
    "multiple_entries_open_count",
}

# The anyone_home boolean-expression and away/home term patterns live in
# evidence_paths.py (issue #524) — one copy shared with the semantic keys.
# Comma decimals cover locale prose (issue #522 red-team review: Czech
# "klesla pod 20,5 %" would otherwise match the bare "5 %" and silently
# register a 4x-lower threshold).
_PERCENT_THRESHOLD_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_DOT_NOTATION_ENTITY_PATTERN = re.compile(r"^([a-z_]+\.[a-z0-9_]+)(?:[.\[]|$)")
_HA_ENTITY_DOMAINS = frozenset(
    {
        "alarm_control_panel",
        "binary_sensor",
        "camera",
        "cover",
        "input_boolean",
        "input_number",
        "light",
        "lock",
        "media_player",
        "person",
        "sensor",
        "switch",
        "vacuum",
    }
)
_RELATIVE_THRESHOLD_PATTERN = re.compile(
    r"(?:at\s+or\s+below|below|under|<=|less than)\s*(\d+(?:[.,]\d+)?)"
)
_HOURS_THRESHOLD_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?",
    re.IGNORECASE,
)
_NUMERIC_THRESHOLD_PATTERN = re.compile(
    r"(?:>|above|exceeds?|over|more than|greater than)\s*(\d+(?:\.\d+)?)"
)
_MAX_PERCENT = 100.0
_DEFAULT_DURATION_HOURS = 2.0
_DEFAULT_STALE_HOURS = 24.0
_MIN_MULTI_ENTRY_COUNT = 2
# Word-bounded so "before"/"forecast" don't match "for", "along" doesn't
# match "long" (issue #504 adversarial review).
_DURATION_TERMS_PATTERN = re.compile(
    r"\b(?:duration|extended|prolonged|for|since|long)\b"
)
# Qualitative hours phrases ("many hours", "several hours") are durations even
# without a digit; bare "night hours"/"daytime hours" phrasing is not.
_QUALITATIVE_HOURS_PATTERN = re.compile(
    r"\b(?:many|several|(?:a\s+)?few|(?:a\s+)?couple(?:\s+of)?)\s+(?:hour|hr)s?\b",
    re.IGNORECASE,
)
_STALENESS_TERMS = (
    "stale",
    "not updated",
    "tracking",
    "staleness",
    "last seen",
    "last updated",
    "gps",
)
_MULTIPLE_TERMS = (
    "multiple",
    "simultaneous",
    "several",
    "more than",
    "at once",
    "at the same time",
)
_POWER_ENERGY_TERMS = (
    "power",
    "energy",
    "watt",
    "consumption",
    "usage",
    "kilowatt",
)
# Entity-name tokens that identify loads whose compressor/motor cycles continuously.
# These sensors have a bimodal power distribution (on vs. standby) that makes a
# rolling average a poor baseline — time_of_day_anomaly is used instead, because
# its variance-aware threshold (max(2*stddev, drift%)) tolerates the cycling noise.
# Mirrors CYCLICAL_LOAD_HINTS in baseline.py; kept inline to avoid HA-stack imports.
_CYCLICAL_LOAD_TOKENS: frozenset[str] = frozenset(
    {"fridge", "refrigerator", "freezer", "compressor"}
)
_ALARM_STATES = ("armed_home", "armed_away", "armed_night", "disarmed", "triggered")
_UNKNOWN_TERMS = (
    "unknown",
    "unrecognized",
    "stranger",
    "unidentified",
    "indeterminate",
)
_PERSON_TERMS = (
    "person",
    "people",
    "face",
    "occupant",
    "occupants",
    "resident",
    "residents",
)
_CAMERA_TERMS = ("camera", "cam")
_MIN_CAMERA_TOKEN_LEN = 3
# Explicit someone-is-home phrasing. Availability prose routinely contains
# incidental home wording ("sensors around the home"), which must not scope
# an outage rule to occupied hours — the while_home evaluator is silent
# exactly when nobody is home, the highest-value moment for an availability
# alert (issue #514 adversarial review).
_EXPLICIT_HOME_OCCUPANCY_PATTERN = re.compile(
    r"\b(?:someone|anyone|somebody|occupants?|residents?)\s+(?:is\s+|are\s+)?"
    r"(?:at\s+)?(?:home|present)\b"
    r"|\b(?:while|when)\s+(?:the\s+home\s+is\s+|the\s+house\s+is\s+)?"
    r"(?:at\s+)?(?:home|occupied)\b"
)
_ENTRY_ID_TOKENS = ("door", "window", "entry")
# Word-bounded so "indoor", "outdoor", and "doorbell" don't read as entries.
_ENTRY_TEXT_PATTERN = re.compile(r"\b(?:doors?|windows?|entry|entries)\b")
_WINDOW_TEXT_PATTERN = re.compile(r"\bwindows?\b")
_DOOR_TEXT_PATTERN = re.compile(r"\bdoors?\b")
# Entity-ID tokens that disqualify a binary_sensor/cover from the text-driven
# entry fallback — these are sensor kinds that commonly appear alongside entry
# candidates but are never door/window contacts themselves. ("co"/"carbon
# monoxide" tokens are deliberately absent: "co" is a substring of "contact".)
_NON_ENTRY_ID_TOKENS = (
    "motion",
    "vmd",
    "battery",
    "occupancy",
    "presence",
    "smoke",
    "gas",
    "leak",
    "moisture",
    "flood",
    "tamper",
    "vibration",
    "carbon",
    "safety",
)
# Entity-ID tokens that disqualify a sensor.* from the text-driven battery
# fallback — sensor kinds that commonly appear alongside battery candidates
# but are never battery-level readings themselves. Promoting one would
# poison the all-of low_battery_sensors evaluator: a non-numeric state
# deadlocks the rule, a numeric non-percentage reading (temperature,
# wattage) false-fires it against the percent threshold (issue #522).
_NON_BATTERY_ID_TOKENS = (
    "power",
    "energy",
    "watt",
    "voltage",
    "current",
    "temperature",
    "humidity",
    "illuminance",
    "lux",
    "pressure",
    "co2",
    "motion",
    "door",
    "window",
    "occupancy",
    "presence",
    "smoke",
    "gas",
    "leak",
    "moisture",
    "flood",
    "signal",
    "rssi",
    "linkquality",
)
# Quote characters the discovery LLM sometimes wraps around entity IDs inside
# evidence paths, e.g. entities[entity_ids contains 'binary_sensor.x'].state.
_EVIDENCE_QUOTE_CHARS = "'\"`"
# Dot-notation entity IDs embedded in candidate prose, e.g. "(binary_sensor.
# xiao_esp32_c5_espectre_motion)". Lookarounds keep a domain-qualified ID from
# matching inside a longer one (sensor.x inside binary_sensor.x); callers
# filter on _HA_ENTITY_DOMAINS so snapshot paths (derived.anyone_home) and
# ordinary prose ("e.g." never has a domain) don't read as entities.
_TEXT_ENTITY_ID_PATTERN = re.compile(r"(?<![a-z0-9_.])([a-z_]+\.[a-z0-9_]+)")
# Word-bounded so incidental prose ("this is alarming") doesn't read as an
# alarm-system condition and divert a motion candidate into the
# missing-alarm-entity failure path; "armed" is included so armed-system
# candidates keep their alarm context (issue #516 review).
_ALARM_TEXT_PATTERN = re.compile(r"\b(?:alarms?|(?:dis)?armed)\b")
# Word-bounded lock wording ("blocked"/"locker" must not match): a motion
# candidate whose prose carries a lock condition that resolved no lock
# entity must stay unsupported rather than register a motion-only rule that
# silently drops the lock predicate (issue #518 Codex adversarial P1).
_LOCK_TEXT_PATTERN = re.compile(r"\b(?:un)?lock(?:s|ed)?\b")
# Contrastive any-hour phrasing ("day or night", "not only at night"): the
# candidate explicitly proposes all-hours coverage, so the bare "night"
# substring must not narrow it to the night-gated template now that a
# day-agnostic sibling exists (issue #518 red-team review).
_ANY_HOUR_TEXT_PATTERN = re.compile(
    r"\b(?:day (?:or|and) night|night (?:or|and) day|any ?time|any hour"
    r"|24/7|around the clock|regardless of (?:the )?time"
    r"|not (?:just|only) (?:at )?night|including night(?:time)?)\b"
)


@dataclass(frozen=True)
class NormalizedRule:
    """Normalized rule derived from a discovery candidate."""

    rule_id: str
    template_id: str
    params: dict[str, Any]
    severity: str
    confidence: float
    is_sensitive: bool
    suggested_actions: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Convert the normalized rule to a persisted mapping."""
        return {
            "rule_id": self.rule_id,
            "template_id": self.template_id,
            "params": self.params,
            "severity": self.severity,
            "confidence": self.confidence,
            "is_sensitive": self.is_sensitive,
            "suggested_actions": list(self.suggested_actions),
        }


@dataclass(frozen=True)
class NormalizationResult:
    """Structured normalization result for a discovery candidate."""

    normalized: NormalizedRule | None
    reason_code: str | None = None
    details: dict[str, Any] | None = None


def normalize_candidate(candidate: dict[str, Any]) -> NormalizedRule | None:
    """Map a discovery candidate to a supported template."""
    return explain_normalize_candidate(candidate).normalized


def explain_normalize_candidate(  # noqa: C901, PLR0911, PLR0912, PLR0915
    candidate: dict[str, Any],
) -> NormalizationResult:
    """Map a discovery candidate to a supported template with failure reasons."""
    evidence_paths = candidate.get("evidence_paths", [])
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("pattern", "")),
            str(candidate.get("suggested_type", "")),
        ]
    ).lower()
    # candidate_id is often the only English surface when the discovery LLM
    # writes title/summary in the home's locale (issue #522: Czech prose
    # whose low-battery signal lives solely in
    # "zamek_vrata_baterie_low_battery"). Scoped to the battery checks only:
    # feeding the slug into every text signal would let a key-echoing
    # candidate_id (e.g. "...night=1...") silently narrow a day-agnostic
    # proposal to the night template — the #518 under-coverage class.
    slug_text = str(candidate.get("candidate_id", "")).lower()
    # Threshold prose for the battery branches: slug tokens spaced out so
    # "…_battery_below_10" contributes its explicit 10% threshold instead of
    # silently broadening to the 40% default (issue #522 Codex adversarial).
    # Prose patterns run first, so an explicit percent in the summary wins.
    battery_threshold_text = f"{text} {_SLUG_TOKEN_SPLIT_RE.sub(' ', slug_text)}"
    lock_ids = _find_entity_ids(evidence_paths, "lock")
    alarm_id = _find_entity_id(evidence_paths, "alarm_control_panel")
    entry_ids = _find_entry_entity_ids(evidence_paths)
    entry_ids_from_text = False
    if not entry_ids and not lock_ids and alarm_id is None:
        # Entity IDs follow the user's locale (e.g. Czech "okno" for window),
        # so keyword matching on the ID can miss real entry sensors. The
        # candidate text is usually English — fall back to it (issue #504;
        # prose can also be locale-written, issue #522, in which case this
        # fallback misses and the candidate stays unsupported).
        # Skipped when lock/alarm entities resolved: lock candidates routinely
        # say "door" ("front door lock"), and a promoted non-entry sensor
        # would defeat the `not entry_ids` guards on those branches.
        entry_ids = _find_text_entry_entity_ids(evidence_paths, text)
        entry_ids_from_text = bool(entry_ids)
    motion_ids = _find_motion_entity_ids(evidence_paths)
    if not motion_ids:
        # Discovery sometimes emits index-based evidence paths
        # (entities[31].state, issue #518) that cannot resolve to an entity
        # ID — the index is only meaningful against the snapshot the
        # candidate was drafted from. The prose names the sensor directly,
        # so fall back to motion-named dot-notation IDs found in the text.
        motion_ids = _find_text_motion_entity_ids(text)
    person_ids = _find_entity_ids(evidence_paths, "person")
    sensor_ids = _find_sensor_entity_ids(evidence_paths)
    availability_ids = _find_availability_entity_ids(evidence_paths)
    battery_sensor_ids = _find_battery_sensor_entity_ids(evidence_paths)
    if not battery_sensor_ids and _has_low_battery_signal(text, slug_text):
        # Entity IDs follow the user's locale (issue #522: Czech "baterie"
        # in sensor.zamek_vrata_baterie), so the "battery" ID-token match
        # can miss real battery sensors. When the candidate carries an
        # explicit low-battery signal, promote sensor.* evidence IDs that
        # are not some other recognizable sensor kind (same tolerant
        # pattern as the issue #504 entry fallback).
        battery_sensor_ids = _find_text_battery_sensor_entity_ids(evidence_paths)
    camera_id = _find_camera_id(evidence_paths, candidate)
    has_night = _has_night_signal(evidence_paths, text)
    presence = _presence_signal(evidence_paths, text)
    # Text-derived kind only for text-derived entry IDs — keyword-derived IDs
    # keep their historical rule_id suffixes for registry stability.
    entry_kind = _entry_kind(entry_ids, text if entry_ids_from_text else "")
    has_unknown_person_signal = _contains_any(text, _UNKNOWN_TERMS) and _contains_any(
        text, _PERSON_TERMS
    )
    has_camera_signal = camera_id is not None or _contains_any(text, _CAMERA_TERMS)
    summary = {
        "alarm_id": alarm_id,
        "lock_ids": lock_ids,
        "entry_ids": entry_ids,
        "motion_ids": motion_ids,
        "person_ids": person_ids,
        "sensor_ids": sensor_ids,
        "availability_ids": availability_ids,
        "battery_sensor_ids": battery_sensor_ids,
        "camera_id": camera_id,
        "presence": presence,
        "has_night": has_night,
    }

    if (
        alarm_id
        and motion_ids
        and has_night
        and _contains_any(text, ("motion", "vmd"))
        and _contains_any(text, ("alarm", "disarmed"))
    ):
        default_rule_id = "motion_detected_at_night_while_alarm_disarmed"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="motion_detected_at_night_while_alarm_disarmed",
                params={
                    "alarm_entity_id": alarm_id,
                    "motion_entity_ids": motion_ids,
                    "required_entity_ids": person_ids,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.8)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    if (
        alarm_id
        and motion_ids
        and person_ids
        and presence == "home"
        and _contains_any(text, ("motion", "vmd"))
        and _contains_any(text, ("alarm", "disarmed"))
    ):
        default_rule_id = (
            f"motion_while_alarm_disarmed_and_home_present_{alarm_id.replace('.', '_')}"
        )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(
                    candidate,
                    default=default_rule_id,
                ),
                template_id="motion_while_alarm_disarmed_and_home_present",
                params={
                    "alarm_entity_id": alarm_id,
                    "motion_entity_ids": motion_ids,
                    "home_entity_ids": person_ids,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.75)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    # motion_detected_at_night_while_away (issue #516): binary_sensor motion
    # at night with nobody home and no alarm involved. Alarm-motion branches
    # above take priority; alarm/armed-worded candidates without a resolved
    # alarm entity keep failing with missing_required_entities rather than
    # losing their alarm gate; entry/lock candidates that also cite a motion
    # sensor keep routing to their entry/lock branches below. Availability
    # and battery candidates keep routing to their own templates further
    # down — a motion-named sensor that is unavailable or battery-low would
    # otherwise register a rule whose state=="on" evaluator can never match
    # (issue #514 invariant). The binary_sensor restriction keeps sensor.*
    # battery/numeric entities and light.* motion-named entities out of the
    # rule params for the same reason.
    away_motion_ids = [m for m in motion_ids if m.startswith("binary_sensor.")]
    # Motion-named IDs that merely contain an entry substring
    # (binary_sensor.outdoor_motion, *_doorbell_motion) are motion sensors,
    # not entries — only a genuine non-motion entry entity blocks this
    # branch (issue #516 review).
    non_motion_entry_ids = [e for e in entry_ids if e not in motion_ids]
    is_away_motion = _is_away_motion_candidate(
        away_motion_ids=away_motion_ids,
        alarm_id=alarm_id,
        non_motion_entry_ids=non_motion_entry_ids,
        lock_ids=lock_ids,
        battery_sensor_ids=battery_sensor_ids,
        presence=presence,
        text=text,
        slug_text=slug_text,
    )
    # Contrastive any-hour phrasing suppresses the night gate so "day or
    # night" candidates keep the all-hours coverage they proposed instead of
    # silently narrowing to the night template (issue #518 red-team review).
    if is_away_motion and has_night and not _ANY_HOUR_TEXT_PATTERN.search(text):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(
                    candidate, default="motion_detected_at_night_while_away"
                ),
                template_id="motion_detected_at_night_while_away",
                params={"motion_entity_ids": away_motion_ids},
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.8)),
                is_sensitive=False,
                # Advisory action only: the finding carries motion entities,
                # so close_entry would have no deterministic target and a
                # non-sensitive execute tap could actuate an unrelated entry
                # (issue #516 cross-model review).
                suggested_actions=["check_camera"],
            )
        )

    # motion_detected_while_away (issue #518): same shape as the night
    # variant above but with no time-of-day gate — motion while nobody is
    # home, any hour. The night branch precedes with the identical shared
    # guard set plus has_night, so night-worded candidates always route
    # there (unless contrastively any-hour worded); the shared guards keep
    # alarm/entry/lock/battery/availability candidates on their existing
    # routing. Two extra guards this branch only (the night branch keeps its
    # shipped v3.22 capture to avoid rule-key churn — see TODOS.md):
    # unknown-person candidates keep routing to the camera templates below
    # (is_sensitive high-confidence must not silently downgrade to a low
    # advisory rule), and camera-evidence candidates keep their
    # motion_without_camera_activity correlation semantics (issue #518
    # adversarial review). Severity and confidence follow the proposal
    # (daytime motion while away has more benign explanations than night
    # motion, hence low/0.6 vs medium/0.8).
    if is_away_motion and not has_unknown_person_signal and camera_id is None:
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(
                    candidate, default="motion_detected_while_away"
                ),
                template_id="motion_detected_while_away",
                params={"motion_entity_ids": away_motion_ids},
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=False,
                # Advisory action only — same rationale as the night variant.
                suggested_actions=["check_camera"],
            )
        )

    if alarm_id and entry_ids and _contains_any(text, ("alarm", "disarmed", "armed")):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"alarm_disarmed_open_entry_{alarm_id.replace('.', '_')}",
                template_id="alarm_disarmed_open_entry",
                params={"alarm_entity_id": alarm_id, "entry_entity_ids": entry_ids},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=True,
                suggested_actions=_entry_suggested_actions(entry_ids),
            )
        )

    # alarm_state_mismatch: alarm in a specific state that contradicts occupancy.
    # Must follow alarm+motion and alarm+entry branches above.
    # Also matches "disarmed" as a detected alarm state (e.g. "alarm disarmed during
    # external threat"). When presence is unknown ("any"), default to "home" so the
    # dynamic rule fires when someone is present with the alarm in this state.
    if (
        alarm_id
        and not motion_ids
        and not entry_ids
        and _contains_any(text, ("armed_home", "armed_away", "armed_night", "disarmed"))
    ):
        detected_state = _extract_alarm_state(text) or "armed_home"
        effective_presence = presence if presence != "any" else "home"
        # armed_home / armed_night are designed for use while occupants are present —
        # they are never a mismatch when presence is "home".
        if (
            detected_state in SENTINEL_OCCUPANCY_ARMED_STATES
            and effective_presence == "home"
        ):
            return NormalizationResult(
                normalized=None,
                reason_code="unsupported_pattern",
                details={
                    "reason": f"{detected_state} with home presence is not a mismatch",
                    **summary,
                },
            )
        alarm_slug = alarm_id.replace(".", "_")
        default_rule_id = (
            f"alarm_state_mismatch_{detected_state}_{effective_presence}_{alarm_slug}"
        )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="alarm_state_mismatch",
                params={
                    "alarm_entity_id": alarm_id,
                    "alarm_state": detected_state,
                    "expected_presence": effective_presence,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=False,
                suggested_actions=["alarm_control_panel.alarm_disarm"],
            )
        )

    # unavailable_sensors: availability candidates route before the entry/
    # lock/battery branches — an entry- or battery-named sensor that is
    # *unavailable* would otherwise be captured by a state-based template
    # whose evaluator can never match "unavailable" (issue #514 adversarial
    # review). Person-tracker staleness candidates ("offline" + "last seen"/
    # "not updated") keep routing to entity_staleness below.
    availability_target_ids = (
        _availability_target_ids(candidate, availability_ids)
        if availability_ids
        else []
    )
    if (
        availability_target_ids
        and _contains_any(text, ("unavailable", "offline", "unreachable"))
        and not (person_ids and _has_staleness_signal(text))
    ):
        if _has_explicit_home_occupancy_signal(evidence_paths, text):
            default_rule_id = "unavailable_sensors_while_home"
            return NormalizationResult(
                normalized=NormalizedRule(
                    rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                    template_id="unavailable_sensors_while_home",
                    params={"sensor_entity_ids": availability_target_ids},
                    severity="low",
                    confidence=float(candidate.get("confidence_hint", 0.8)),
                    is_sensitive=False,
                    suggested_actions=["check_sensor"],
                )
            )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default="unavailable_sensors"),
                template_id="unavailable_sensors",
                params={"sensor_entity_ids": availability_target_ids},
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    # entity_state_duration: lock unlocked for too long.
    if (
        lock_ids
        and not entry_ids
        and _has_duration_signal(text)
        and _contains_any(text, ("lock", "unlocked"))
    ):
        lock_id = lock_ids[0]
        threshold_hours = _extract_threshold_hours(text)
        default_rule_id = f"lock_unlocked_duration_{lock_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_state_duration",
                params={
                    "entity_id": lock_id,
                    "target_state": "unlocked",
                    "threshold_hours": threshold_hours,
                },
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.75)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    # entity_state_duration: entry sensor open for too long.
    if (
        entry_ids
        and _has_duration_signal(text)
        and _contains_any(text, ("open", "window", "door", "entry"))
    ):
        entry_id = entry_ids[0]
        threshold_hours = _extract_threshold_hours(text)
        default_rule_id = f"entry_open_duration_{entry_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_state_duration",
                params={
                    "entity_id": entry_id,
                    "target_state": "on",
                    "threshold_hours": threshold_hours,
                },
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    # low_battery on a lock entity: battery signal takes priority over unlocked routing.
    # Requires a sensor.* battery entity — lock entities cannot be sensor_entity_ids.
    if lock_ids and _has_low_battery_signal(text, slug_text):
        if not battery_sensor_ids:
            return NormalizationResult(
                normalized=None,
                reason_code="unsupported_pattern",
                details={
                    "reason": "lock battery candidate lacks sensor.* battery IDs",
                    **summary,
                },
            )
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default="low_battery_sensors"),
                template_id="low_battery_sensors",
                params={
                    "sensor_entity_ids": battery_sensor_ids,
                    "threshold": _extract_threshold_percent(
                        battery_threshold_text, default=40.0
                    ),
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.62)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    # unlocked_lock_while_away: lock unlocked when nobody is home.
    if (
        lock_ids
        and not entry_ids
        and presence == "away"
        and _contains_any(text, ("lock", "unlocked"))
    ):
        lock_id = lock_ids[0]
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"unlocked_lock_while_away_{lock_id.replace('.', '_')}",
                template_id="unlocked_lock_while_away",
                params={"lock_entity_id": lock_id},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    if lock_ids and not entry_ids and _contains_any(text, ("lock", "unlocked")):
        lock_id = lock_ids[0]
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"unlocked_lock_when_home_{lock_id.replace('.', '_')}",
                template_id="unlocked_lock_when_home",
                params={"lock_entity_id": lock_id},
                severity="medium",
                confidence=float(candidate.get("confidence_hint", 0.5)),
                is_sensitive=True,
                suggested_actions=["lock.lock", "lock_entity"],
            )
        )

    # multiple_entries_open_count: several entries open simultaneously.
    # Must precede the per-entry open branches below.
    if (
        len(entry_ids) >= _MIN_MULTI_ENTRY_COUNT
        and _has_multiple_signal(text)
        and _contains_any(text, ("open", "window", "door", "entry"))
    ):
        require_home = presence == "home"
        require_away = presence in {"away", "any"}
        default_rule_id = "multiple_entries_open_count"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="multiple_entries_open_count",
                params={
                    "entry_entity_ids": entry_ids,
                    "min_count": 2,
                    "require_home": require_home,
                    "require_away": require_away,
                },
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.75)),
                is_sensitive=True,
                suggested_actions=_entry_suggested_actions(entry_ids),
            )
        )

    if entry_ids and _contains_any(text, ("open", "window", "door", "entry")):
        if has_night and presence == "away":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_at_night_while_away",
                    f"open_entry_at_night_while_away_{entry_kind}",
                    entry_ids,
                )
            )
        if has_night and presence == "home":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_at_night_when_home",
                    f"open_entry_at_night_when_home_{entry_kind}",
                    entry_ids,
                )
            )
        if has_night:
            # presence == "any": night is explicit but occupancy direction is
            # unknown — use the presence-agnostic night template so the rule
            # fires whether or not anyone is home (issue #504).
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_at_night",
                    f"open_entry_at_night_{entry_kind}",
                    entry_ids,
                )
            )
        if presence == "away":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_while_away",
                    f"open_entry_while_away_{entry_kind}",
                    entry_ids,
                )
            )
        if presence == "home":
            return NormalizationResult(
                normalized=_build_entry_rule(
                    candidate,
                    "open_entry_when_home",
                    f"open_entry_when_home_{entry_kind}",
                    entry_ids,
                )
            )
        # presence == "any": occupancy unknown — default to "away" template (safer,
        # fires whenever an entry is open regardless of confirmed home/away state).
        return NormalizationResult(
            normalized=_build_entry_rule(
                candidate,
                "open_entry_while_away",
                f"open_entry_while_away_{entry_kind}",
                entry_ids,
            )
        )
    if (
        not entry_ids
        and has_night
        and presence == "away"
        and _contains_any(text, ("open", "window"))
    ):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id="open_any_window_at_night_while_away",
                template_id="open_any_window_at_night_while_away",
                params={"entry_selector": "window"},
                severity="high",
                confidence=float(candidate.get("confidence_hint", 0.6)),
                is_sensitive=True,
                suggested_actions=["close_entry"],
            )
        )

    if has_camera_signal and presence == "away" and has_unknown_person_signal:
        if camera_id:
            rule_id = f"unknown_person_camera_no_home_{camera_id.replace('.', '_')}"
            params = {"camera_entity_id": camera_id}
        else:
            rule_id = "unknown_person_camera_no_home_any_camera"
            params = {"camera_selector": "any"}
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=rule_id,
                template_id="unknown_person_camera_no_home",
                params=params,
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.85)),
                is_sensitive=True,
                suggested_actions=["close_entry"],
            )
        )
    if has_camera_signal and presence == "home" and has_unknown_person_signal:
        if camera_id:
            default_rule_id = (
                f"unknown_person_camera_when_home_{camera_id.replace('.', '_')}"
            )
            params = {"camera_entity_id": camera_id}
        else:
            default_rule_id = "unknown_person_camera_when_home_any_camera"
            params = {"camera_selector": "any"}
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=default_rule_id,
                template_id="unknown_person_camera_when_home",
                params=params,
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["close_entry"],
            )
        )

    if (
        motion_ids
        and camera_id
        and _contains_any(text, ("motion", "camera", "activity"))
    ):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=f"motion_without_camera_{camera_id.replace('.', '_')}",
                template_id="motion_without_camera_activity",
                params={
                    "motion_entity_ids": motion_ids,
                    "camera_entity_id": camera_id,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.5)),
                is_sensitive=False,
                suggested_actions=["check_camera"],
            )
        )

    # Prose keeps its legacy any-of predicate; the candidate_id slug counts
    # only via the conjunctive signal so an incidental "low" substring in a
    # slug ("slow", "flow") cannot route a non-battery candidate here
    # (issue #522 security review).
    if battery_sensor_ids and (
        _contains_any(text, ("battery", "low", "below"))
        or _has_low_battery_signal(text, slug_text)
    ):
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default="low_battery_sensors"),
                template_id="low_battery_sensors",
                params={
                    "sensor_entity_ids": battery_sensor_ids,
                    "threshold": _extract_threshold_percent(
                        battery_threshold_text, default=40.0
                    ),
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.62)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    # sensor_threshold_condition: numeric sensor exceeds a threshold, with optional
    # night/away/home condition. Excludes battery sensors (handled above).
    # Also check entity IDs for power/energy keywords — the LLM may describe an
    # appliance as "active" or "running" rather than mentioning "power" or "watt",
    # but the entity ID (e.g. sensor.washing_machine_switch_0_power) is unambiguous.
    non_battery_sensor_ids = [s for s in sensor_ids if s not in battery_sensor_ids]
    if non_battery_sensor_ids and (
        _has_power_energy_signal(text)
        or any(_has_power_energy_signal(eid) for eid in non_battery_sensor_ids)
    ):
        threshold = _extract_threshold_numeric(text)
        sensor_id = non_battery_sensor_ids[0]
        if threshold is not None:
            default_rule_id = f"sensor_threshold_{sensor_id.replace('.', '_')}"
            return NormalizationResult(
                normalized=NormalizedRule(
                    rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                    template_id="sensor_threshold_condition",
                    params={
                        "sensor_entity_id": sensor_id,
                        "threshold": threshold,
                        "require_night": has_night,
                        "require_away": presence == "away",
                        "require_home": presence == "home",
                    },
                    severity="low",
                    confidence=float(candidate.get("confidence_hint", 0.7)),
                    is_sensitive=False,
                    suggested_actions=["check_appliance"],
                )
            )
        # No numeric threshold — fall back to a statistical detector.
        # When the candidate bundles multiple sensors (e.g. LLM lists both
        # *_power and *_energy variants), pick the first non-cumulative one so
        # that energy counters don't silently discard the whole candidate.
        instantaneous_id = next(
            (s for s in non_battery_sensor_ids if not _is_cumulative_energy_sensor(s)),
            None,
        )
        if instantaneous_id is None:
            return NormalizationResult(
                normalized=None,
                reason_code="cumulative_energy_sensor",
                details={"sensor_id": sensor_id},
            )
        sensor_id = instantaneous_id
        # Cyclical loads (fridge, freezer, compressor) have a bimodal power
        # distribution — rolling-average baselines mix on/off states and produce
        # systematic false positives on every normal off-cycle.  time_of_day_anomaly
        # uses a variance-aware threshold (max(2*stddev, drift%)) that tolerates it.
        if _is_cyclical_load(sensor_id):
            default_rule_id = f"sensor_tod_{sensor_id.replace('.', '_')}"
            return NormalizationResult(
                normalized=NormalizedRule(
                    rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                    template_id="time_of_day_anomaly",
                    params={"entity_id": sensor_id},
                    severity="low",
                    confidence=float(candidate.get("confidence_hint", 0.65)),
                    is_sensitive=False,
                    suggested_actions=["check_appliance"],
                )
            )
        default_rule_id = f"sensor_baseline_{sensor_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="baseline_deviation",
                params={"entity_id": sensor_id},
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.65)),
                is_sensitive=False,
                suggested_actions=["check_appliance"],
            )
        )

    # entity_staleness: entity last_changed has not advanced past a threshold.
    # Matches person tracking (person_ids) or explicit sensor staleness signals.
    if (person_ids or sensor_ids) and _has_staleness_signal(text):
        entity_id = (person_ids or sensor_ids)[0]
        max_stale_hours = _extract_threshold_hours(text, default=_DEFAULT_STALE_HOURS)
        default_rule_id = f"entity_staleness_{entity_id.replace('.', '_')}"
        return NormalizationResult(
            normalized=NormalizedRule(
                rule_id=_candidate_rule_id(candidate, default=default_rule_id),
                template_id="entity_staleness",
                params={
                    "entity_id": entity_id,
                    "max_stale_hours": max_stale_hours,
                },
                severity="low",
                confidence=float(candidate.get("confidence_hint", 0.7)),
                is_sensitive=False,
                suggested_actions=["check_sensor"],
            )
        )

    return _normalization_failure(
        text=text,
        slug_text=slug_text,
        summary=summary,
        candidate=candidate,
        alarm_id=alarm_id,
        lock_ids=lock_ids,
        entry_ids=entry_ids,
        motion_ids=motion_ids,
        person_ids=person_ids,
        sensor_ids=sensor_ids,
        availability_ids=availability_ids,
        battery_sensor_ids=battery_sensor_ids,
        camera_id=camera_id,
        presence=presence,
        has_night=has_night,
    )


def _is_away_motion_candidate(  # noqa: PLR0913
    *,
    away_motion_ids: list[str],
    alarm_id: str | None,
    non_motion_entry_ids: list[str],
    lock_ids: list[str],
    battery_sensor_ids: list[str],
    presence: str,
    text: str,
    slug_text: str,
) -> bool:
    """
    Shared guard set for the two away-motion templates (issues #516/#518).

    Entity guards keep alarm/entry/lock/battery candidates on their existing
    routing. The text guards handle predicates that resolved no matching
    entity (index-based evidence paths, issue #518 Codex adversarial P1): a
    candidate whose prose carries a battery/lock/staleness/open-entry
    condition must stay unsupported rather than register a motion rule that
    silently drops that predicate — the registered rule would alert on plain
    motion while the user believes the stated condition is enforced.
    """
    return bool(
        away_motion_ids
        and not alarm_id
        and not non_motion_entry_ids
        and not lock_ids
        and not battery_sensor_ids
        and presence == "away"
        and _contains_any(text, ("motion", "vmd"))
        and not _ALARM_TEXT_PATTERN.search(text)
        and not _contains_any(text, ("unavailable", "offline", "unreachable"))
        and not _has_low_battery_signal(text, slug_text)
        and not _LOCK_TEXT_PATTERN.search(text)
        # A stale/not-updated motion sensor candidate describes a dead
        # sensor; routing it here would invert the semantics into alerting
        # on normal motion (issue #518 adversarial review). Explicit stale
        # wording only — _has_staleness_signal's broader "tracking"/"gps"
        # terms would reject legitimate "motion tracking" candidates
        # (verification round 5).
        and not _contains_any(
            text,
            ("stale", "staleness", "not updated", "last seen", "last updated"),
        )
        and not (_ENTRY_TEXT_PATTERN.search(text) and "open" in text)
    )


def _build_entry_rule(
    candidate: dict[str, Any],
    template_id: str,
    rule_id: str,
    entry_ids: list[str],
) -> NormalizedRule:
    return NormalizedRule(
        rule_id=rule_id,
        template_id=template_id,
        params={"entry_entity_ids": entry_ids},
        severity="high" if "away" in template_id else "medium",
        confidence=float(candidate.get("confidence_hint", 0.6)),
        is_sensitive=True,
        suggested_actions=_entry_suggested_actions(entry_ids),
    )


def _entry_suggested_actions(entry_ids: list[str]) -> list[str]:
    if entry_ids and all(entity_id.startswith("cover.") for entity_id in entry_ids):
        return ["cover.close_cover", "close_entry"]
    return ["close_entry"]


def _normalization_failure(  # noqa: PLR0911, PLR0913
    *,
    text: str,
    slug_text: str,
    summary: dict[str, Any],
    candidate: dict[str, Any],
    alarm_id: str | None,
    lock_ids: list[str],
    entry_ids: list[str],
    motion_ids: list[str],
    person_ids: list[str],
    sensor_ids: list[str],
    availability_ids: list[str],
    battery_sensor_ids: list[str],
    camera_id: str | None,
    presence: str,
    has_night: bool,
) -> NormalizationResult:
    if _contains_any(text, ("alarm", "disarmed", "armed")) and not alarm_id:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["alarm_control_panel"], **summary},
        )
    if _contains_any(text, ("lock", "unlocked")) and not lock_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["lock"], **summary},
        )
    if _contains_any(text, ("open", "window", "door", "entry")) and not entry_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["entry"], **summary},
        )
    if _contains_any(text, ("motion", "vmd")) and not motion_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["motion"], **summary},
        )
    if (
        _contains_any(text, ("battery", "low", "below"))
        or _has_low_battery_signal(text, slug_text)
    ) and not battery_sensor_ids:
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["battery_sensor"], **summary},
        )
    if (
        _contains_any(text, ("unavailable", "offline", "unreachable"))
        and not availability_ids
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["sensor", "binary_sensor"], **summary},
        )
    if (
        _contains_any(text, _CAMERA_TERMS)
        and camera_id is None
        and "camera" in text
        and _contains_any(text, _UNKNOWN_TERMS)
        and _contains_any(text, _PERSON_TERMS)
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="missing_required_entities",
            details={"required": ["camera"], **summary},
        )
    if (
        any(
            (
                alarm_id,
                lock_ids,
                entry_ids,
                motion_ids,
                person_ids,
                sensor_ids,
                availability_ids,
                battery_sensor_ids,
                camera_id,
            )
        )
        or has_night
        or presence != "any"
    ):
        return NormalizationResult(
            normalized=None,
            reason_code="unsupported_pattern",
            details={"candidate_id": candidate.get("candidate_id"), **summary},
        )
    return NormalizationResult(
        normalized=None,
        reason_code="no_matching_entity_types",
        details={"candidate_id": candidate.get("candidate_id"), **summary},
    )


def _extract_entity_id_from_evidence_path(path: str) -> str | None:
    """
    Extract entity_id from an evidence path in any known format.

    Handles:
    - ``entities[entity_id=domain.object_id]`` (snapshot query format)
    - ``entities[entity_ids contains domain.object_id].attr`` (discovery format)
    - ``entities[domain.object_id].attr`` (bare-bracket format, issue #522)
    - ``domain.object_id`` or ``domain.object_id.attribute`` (dot-notation)

    Entity IDs may be wrapped in quotes (LLM output variance); quotes are
    stripped before the ID is returned.
    """
    if path.startswith("entities[entity_id="):
        token = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
        return token.strip(_EVIDENCE_QUOTE_CHARS) or None
    if path.startswith("entities[entity_ids contains "):
        token = path.split("entities[entity_ids contains ", 1)[1].split("]", 1)[0]
        return token.strip(_EVIDENCE_QUOTE_CHARS) or None
    if path.startswith("entities["):
        # Bare-bracket form: the bracket token must itself be a
        # domain-qualified entity ID. Index-based brackets
        # (entities[31].state, issue #518) stay unresolvable — the index is
        # only meaningful against the snapshot the candidate was drafted
        # from.
        token = path.split("entities[", 1)[1].split("]", 1)[0]
        token = token.strip(_EVIDENCE_QUOTE_CHARS)
        m = _DOT_NOTATION_ENTITY_PATTERN.match(token)
        if m:
            entity_id = m.group(1)
            if entity_id.split(".", 1)[0] in _HA_ENTITY_DOMAINS:
                return entity_id
        return None
    m = _DOT_NOTATION_ENTITY_PATTERN.match(path.strip(_EVIDENCE_QUOTE_CHARS))
    if m:
        entity_id = m.group(1)
        domain = entity_id.split(".", 1)[0]
        if domain in _HA_ENTITY_DOMAINS:
            return entity_id
    return None


def _find_entity_id(evidence_paths: list[str], domain: str) -> str | None:
    ids = _find_entity_ids(evidence_paths, domain)
    if not ids:
        return None
    return ids[0]


def _find_entity_ids(evidence_paths: list[str], domain: str) -> list[str]:
    ids = [
        eid
        for path in evidence_paths
        if (eid := _extract_entity_id_from_evidence_path(path)) is not None
        and f"{domain}." in eid
    ]
    return sorted(set(ids))


def _find_entry_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None:
            continue
        if "." not in entity_id:
            # Object ID without domain — check for entry keywords and accept
            # as-is (same tolerant pattern used by _find_sensor_entity_ids).
            if any(key in entity_id for key in _ENTRY_ID_TOKENS):
                ids.append(entity_id)
            continue
        domain = entity_id.split(".", 1)[0]
        if domain not in {"binary_sensor", "cover"}:
            continue
        if any(key in entity_id for key in _ENTRY_ID_TOKENS):
            ids.append(entity_id)
    return sorted(set(ids))


def _find_text_entry_entity_ids(evidence_paths: list[str], text: str) -> list[str]:
    """
    Fallback entry detection for entity IDs without English entry keywords.

    Entity IDs are named in the user's locale (e.g. Czech ``okno`` for
    window), so ``_find_entry_entity_ids`` keyword matching can miss real
    entry sensors. The discovery LLM's candidate text is usually English
    (though it can be locale prose too — issue #522 — in which case this
    fallback misses and the candidate stays unsupported), so when the text
    names a door/window/entry, promote binary_sensor/cover evidence IDs
    that are not some other recognizable sensor kind.
    """
    if not _ENTRY_TEXT_PATTERN.search(text):
        return []
    ids: list[str] = []
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None or "." not in entity_id:
            continue
        domain = entity_id.split(".", 1)[0]
        if domain not in {"binary_sensor", "cover"}:
            continue
        if any(token in entity_id for token in _NON_ENTRY_ID_TOKENS):
            continue
        ids.append(entity_id)
    return sorted(set(ids))


def _find_motion_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None:
            continue
        if "motion" in entity_id or "vmd" in entity_id:
            ids.append(entity_id)
    return sorted(set(ids))


def _find_text_motion_entity_ids(text: str) -> list[str]:
    """
    Fallback motion detection for candidates without resolvable evidence IDs.

    Discovery sometimes emits index-based evidence paths
    (``entities[31].state``, issue #518) whose index is only meaningful
    against the snapshot the candidate was drafted from and so never
    resolves to an entity ID. The candidate prose names the sensor
    directly; promote motion-named ``binary_sensor.`` dot-notation entity
    IDs found in the text. Restricted to ``binary_sensor`` because these
    IDs also feed the alarm-motion branches' rule params, whose state=="on"
    evaluator can never match numeric ``sensor.*`` or ``light.*`` entities
    (issue #514 invariant; #518 multi-reviewer finding).
    """
    ids: list[str] = []
    for match in _TEXT_ENTITY_ID_PATTERN.finditer(text):
        entity_id = match.group(1)
        if not entity_id.startswith("binary_sensor."):
            continue
        if "motion" in entity_id or "vmd" in entity_id:
            ids.append(entity_id)
    return sorted(set(ids))


def _find_domain_entity_ids(
    evidence_paths: list[str], domains: frozenset[str]
) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None:
            continue
        if "." not in entity_id:
            # Legacy discovery drafts may store object IDs without domain.
            ids.append(entity_id)
            continue
        domain = entity_id.split(".", 1)[0]
        if domain in domains:
            ids.append(entity_id)
    return sorted(set(ids))


def _find_sensor_entity_ids(evidence_paths: list[str]) -> list[str]:
    return _find_domain_entity_ids(evidence_paths, frozenset({"sensor"}))


def _find_availability_entity_ids(evidence_paths: list[str]) -> list[str]:
    """
    Entity IDs eligible for the unavailable-sensors templates.

    Availability candidates routinely cite ``binary_sensor.*`` entities
    (occupancy/presence sensors, issue #514), which the ``sensor.``-only
    collector misses. The dynamic-rule evaluator resolves any entity ID
    present in the snapshot, so accept both measurement domains here along
    with legacy domainless object IDs.
    """
    return _find_domain_entity_ids(
        evidence_paths, frozenset({"sensor", "binary_sensor"})
    )


def _find_text_battery_sensor_entity_ids(evidence_paths: list[str]) -> list[str]:
    """
    Fallback battery detection for entity IDs without an English battery token.

    Entity IDs are named in the user's locale (issue #522: Czech ``baterie``
    in ``sensor.zamek_vrata_baterie``), so the ``battery`` ID-token match in
    ``_find_battery_sensor_entity_ids`` can miss real battery sensors. The
    caller invokes this only when the candidate carries an explicit
    low-battery signal and no battery-named ID resolved; promote ``sensor.*``
    evidence IDs that are not some other recognizable sensor kind. The
    domain restriction matters: the low_battery_sensors evaluator requires
    a numeric state from every listed sensor, so binary_sensor/lock/etc.
    entities would deadlock the rule (issue #514 invariant). And because the
    kind-token filter is English-only, two or more surviving locale-named
    IDs are ambiguous — which one is the battery? — so only a single
    unambiguous target is promoted; multi-ID candidates stay honestly
    unsupported rather than register an all-of rule that deadlocks or
    false-fires on the contextual sensor's reading.

    The single-ID heuristic is advisory only — an English token filter
    cannot classify locale IDs, so a lone locale-named contextual sensor
    still survives it. The authoritative gate is approval-time
    (``_is_battery_like_state``): the live state must carry a battery
    device_class / percent unit and a numeric reading before the rule
    registers.
    """
    ids: set[str] = set()
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None or not entity_id.startswith("sensor."):
            continue
        if any(token in entity_id for token in _NON_BATTERY_ID_TOKENS):
            continue
        ids.add(entity_id)
    if len(ids) != 1:
        return []
    return sorted(ids)


# One term list across the normalizer, discovery_semantic's battery
# predicate leg, and the card's _lowBatteryContext — an asymmetric list
# ("weak" here, "under" there) breaks dedup in both directions: a
# weak-worded candidate registers a rule its own key never covers, and an
# under-worded one stays unsupported while minting a low_battery history
# key that suppresses later approvable proposals (issue #522 adversarial
# review, reproduced).
_LOW_BATTERY_QUALIFIERS = ("low", "below", "under", "weak")
_SLUG_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _has_low_battery_signal(text: str, slug_text: str) -> bool:
    """
    Low-battery signal from prose OR the candidate_id slug (issue #522).

    Prose keeps substring semantics ("batteries", "lower"), but the slug is
    matched on whole tokens and each surface must carry the full conjunctive
    signal on its own — substring matching over a concatenated corpus lets
    "backup_battery_water_flow" qualify because "flow" contains "low"
    (issue #522 Codex adversarial, reproduced).
    """
    if "battery" in text and _contains_any(text, _LOW_BATTERY_QUALIFIERS):
        return True
    tokens = set(_SLUG_TOKEN_SPLIT_RE.split(slug_text))
    return "battery" in tokens and any(
        qualifier in tokens for qualifier in _LOW_BATTERY_QUALIFIERS
    )


def _find_battery_sensor_entity_ids(evidence_paths: list[str]) -> list[str]:
    ids: list[str] = []
    for path in evidence_paths:
        entity_id = _extract_entity_id_from_evidence_path(path)
        if entity_id is None:
            continue
        if "battery" not in entity_id.lower():
            continue
        if "." not in entity_id:
            ids.append(entity_id)
            continue
        domain = entity_id.split(".", 1)[0]
        if domain == "sensor":
            ids.append(entity_id)
    return sorted(set(ids))


def _extract_threshold_percent(text: str, *, default: float) -> float:
    for pattern in (_PERCENT_THRESHOLD_PATTERN, _RELATIVE_THRESHOLD_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        if 0 <= value <= _MAX_PERCENT:
            return value
    return default


def _has_night_signal(evidence_paths: list[str], text: str) -> bool:
    return night_signal(evidence_paths, text)


def _presence_signal(evidence_paths: list[str], text: str) -> PresenceSignal:
    # Priority order documented in evidence_paths.presence_signal: structured
    # negation, anyone_home boolean expressions, English terms, then the bare
    # positive path resolving "home". The path alone used to return "any" —
    # which the entry branch defaults to the away template — silently
    # inverting home candidates whose prose carries no English direction
    # words (issue #524).
    return presence_signal(evidence_paths, text)


# The evaluator fires only on the literal HA "unavailable" state
# (_eval_unavailable_sensors), so only that predicate marks a target —
# accepting e.g. == 'unknown' would register a rule that can never fire.
_AVAILABILITY_TARGET_STATE = "unavailable"
# State literal following an equality comparison in a candidate pattern
# clause, e.g. ``.state == 'off'`` or ``state = unavailable``. Negative
# lookbehind keeps ``!=``/``<=``/``>=`` comparisons from reading as equality.
_PREDICATE_STATE_PATTERN = re.compile(r"(?<![!<>=])==?\s*['\"]?([a-z_]+)['\"]?")
# Bare state words in free-form clauses without an equality operator, e.g.
# ``sensor.temperature unavailable AND binary_sensor.occupancy off``. Word
# boundaries keep ``state_unavailable`` (underscore-joined) from matching.
_BARE_STATE_PATTERN = re.compile(
    r"\b(?:unavailable|unknown|on|off|open|closed|locked|unlocked"
    r"|home|not_home|detected|clear|idle)\b"
)
# Any comparison operator: a clause carrying one but failing the equality
# regex is a negated/inequality condition (!=, <, >=) — contextual, never a
# target, even when the bare word "unavailable" appears in it.
_COMPARISON_OPERATOR_PATTERN = re.compile(r"!=|<=?|>=?|==?")
_BOOLEAN_CONNECTOR_PATTERN = re.compile(r"\band\b|\bor\b")
# Sentinel predicate for explicitly non-availability conditions; any value
# other than "unavailable" excludes the entity from the target list.
_CONTEXTUAL_PREDICATE = "__contextual__"


def _availability_target_ids(
    candidate: dict[str, Any], availability_ids: list[str]
) -> list[str]:
    """
    Restrict availability targets to entities the candidate says are unavailable.

    Compound candidate patterns can mix the unavailable target with a
    contextual condition (``sensor.x == 'unavailable' AND binary_sensor.y ==
    'off'``). Collecting the contextual entity as a target deadlocks the
    all-of evaluator — ``off`` never equals ``unavailable`` — so the rule
    silently never fires (issue #514 adversarial review).

    When the pattern carries no per-entity predicates at all (issue #514
    lists bare evidence paths with a candidate-wide ``state_unavailable``
    pattern), every collected entity is a target. Once any per-entity
    predicate is present, only entities explicitly compared to
    ``unavailable`` qualify — entities omitted from the pattern, or compared
    to any other state, are contextual. May return an empty list: the caller
    must then skip the availability templates entirely rather than register
    a rule with different semantics.
    """
    pattern = str(candidate.get("pattern", "")).lower()
    predicate_by_id: dict[str, str | None] = {}
    has_any_predicate = False
    for entity_id in availability_ids:
        # Token-bounded on both sides so a prefix ID (sensor.hall) doesn't
        # read as an occurrence of sensor.hall_temperature, and an ID sharing
        # an object-id suffix (sensor.temperature) doesn't match inside
        # binary_sensor.temperature.
        id_match = re.search(
            r"(?<![a-z0-9_.])" + re.escape(entity_id.lower()) + r"(?![a-z0-9_])",
            pattern,
        )
        if id_match is None:
            predicate_by_id[entity_id] = None
            continue
        clause = pattern[id_match.end() :]
        clause = _BOOLEAN_CONNECTOR_PATTERN.split(clause, 1)[0]
        equality = _PREDICATE_STATE_PATTERN.search(clause)
        if equality is not None:
            state: str | None = equality.group(1)
        elif _COMPARISON_OPERATOR_PATTERN.search(clause):
            # A comparison the equality regex rejected (!=, <, >=) is a
            # negated/inequality condition — contextual, never a target.
            state = _CONTEXTUAL_PREDICATE
        else:
            # Free-form patterns state per-entity semantics without an
            # equality operator ("sensor.x unavailable AND sensor.y off").
            bare = _BARE_STATE_PATTERN.search(clause)
            if bare is not None and re.search(r"\bnot\s*$", clause[: bare.start()]):
                state = _CONTEXTUAL_PREDICATE
            else:
                state = None if bare is None else bare.group(0)
        predicate_by_id[entity_id] = state
        if state is not None:
            has_any_predicate = True
    if not has_any_predicate:
        return availability_ids
    return [
        entity_id
        for entity_id, state in predicate_by_id.items()
        if state == _AVAILABILITY_TARGET_STATE
    ]


def _has_explicit_home_occupancy_signal(evidence_paths: list[str], text: str) -> bool:
    """
    Return True only for an explicit someone-is-home condition.

    Availability prose routinely contains incidental home wording ("sensors
    around the home", "presence sensors"), which must not scope an outage
    rule to occupied hours — the while_home evaluator is silent exactly when
    nobody is home, the highest-value moment for an availability alert
    (issue #514 adversarial review).
    """
    if has_derived_path(evidence_paths, NOT_ANYONE_HOME_PATH):
        return False
    # Explicit absence signals override the bare evidence path — the
    # candidate cites occupancy to require absence, not presence: an
    # anyone_home == false expression or a negated path spelled in the
    # pattern text (any variant spelling the canonicalizer accepts).
    if ANYONE_HOME_FALSE_PATTERN.search(text) or NOT_ANYONE_HOME_TEXT_PATTERN.search(
        text
    ):
        return False
    # Machine syntax outranks prose terms — same tier order as
    # presence_signal. Checking away terms first keyed home=1 in the
    # semantic key while routing skipped the while_home split, so the
    # activated rule could never dedup the candidate (issue #524 red-team).
    if ANYONE_HOME_TRUE_PATTERN.search(text):
        return True
    if AWAY_TERMS_PATTERN.search(text):
        return False
    if has_derived_path(evidence_paths, ANYONE_HOME_PATH):
        return True
    return _EXPLICIT_HOME_OCCUPANCY_PATTERN.search(text) is not None


def _entry_kind(entry_ids: list[str], text: str = "") -> str:
    if any("window" in entity_id for entity_id in entry_ids):
        return "window"
    if any("door" in entity_id for entity_id in entry_ids):
        return "door"
    # Locale-named entity IDs carry no English kind token — fall back to the
    # candidate text (callers pass it only for text-derived entry IDs).
    if _WINDOW_TEXT_PATTERN.search(text):
        return "window"
    if _DOOR_TEXT_PATTERN.search(text):
        return "door"
    return "entry"


def _find_camera_id(  # noqa: PLR0911
    evidence_paths: list[str], candidate: dict[str, Any]
) -> str | None:
    for path in evidence_paths:
        if path.startswith("camera_activity[entity_id="):
            token = path.split("camera_activity[entity_id=", 1)[1].split("]", 1)[0]
            return token.strip(_EVIDENCE_QUOTE_CHARS) or None
        if path.startswith("camera_activity[camera_entity_id="):
            token = path.split("camera_activity[camera_entity_id=", 1)[1].split("]", 1)[
                0
            ]
            return token.strip(_EVIDENCE_QUOTE_CHARS) or None
        if path.startswith("entities[entity_id="):
            token = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
            token = token.strip(_EVIDENCE_QUOTE_CHARS)
            if token.startswith("camera."):
                return token
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    normalized = re.sub(r"[^a-z0-9_]+", "_", candidate_id.lower())
    tokens = [token for token in normalized.split("_") if token]
    try:
        camera_idx = tokens.index("camera")
    except ValueError:
        return None
    suffix = tokens[camera_idx + 1 :]
    stopwords = {
        "home",
        "away",
        "while",
        "when",
        "day",
        "night",
        "during",
        "outside",
        "inside",
        "unknown",
        "person",
        "people",
        "and",
        "motion",
    }
    object_candidates = [
        token
        for token in suffix
        if token and token not in stopwords and len(token) >= _MIN_CAMERA_TOKEN_LEN
    ]
    if not object_candidates:
        return None
    return f"camera.{object_candidates[-1]}"


def _extract_threshold_hours(
    text: str, *, default: float = _DEFAULT_DURATION_HOURS
) -> float:
    match = _HOURS_THRESHOLD_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default


def _extract_threshold_numeric(text: str) -> float | None:
    match = _NUMERIC_THRESHOLD_PATTERN.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def _extract_alarm_state(text: str) -> str | None:
    for state in _ALARM_STATES:
        if state in text:
            return state
    return None


def _has_duration_signal(text: str) -> bool:
    if _DURATION_TERMS_PATTERN.search(text):
        return True
    # Bare "hours" counts only with a numeric ("2 hours") or qualitative
    # ("many hours") quantifier — phrases like "during night hours" are
    # time-of-day context, not a duration.
    if _QUALITATIVE_HOURS_PATTERN.search(text):
        return True
    return _HOURS_THRESHOLD_PATTERN.search(text) is not None


def _has_staleness_signal(text: str) -> bool:
    return _contains_any(text, _STALENESS_TERMS)


def _has_multiple_signal(text: str) -> bool:
    return _contains_any(text, _MULTIPLE_TERMS)


def _has_power_energy_signal(text: str) -> bool:
    return _contains_any(text, _POWER_ENERGY_TERMS)


def _is_cumulative_energy_sensor(entity_id: str) -> bool:
    local = entity_id.split(".", 1)[-1] if "." in entity_id else entity_id
    return local.endswith("_energy") or local == "energy"


def _is_cyclical_load(entity_id: str) -> bool:
    import re  # noqa: PLC0415

    tokens = set(re.split(r"[._\s]+", entity_id.lower()))
    return bool(tokens & _CYCLICAL_LOAD_TOKENS)


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _candidate_rule_id(candidate: dict[str, Any], *, default: str) -> str:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str):
        return default
    # ASCII-only, run-collapsing — converges with the card's
    # _sanitizeRuleName so a locale candidate_id ("baterie_nízká…") cannot
    # register a non-ASCII rule_id that diverges from the card's inferred
    # rule id, dedup key, and issue prefill (issue #522 red-team review).
    # str.isalnum() alone accepts Unicode letters.
    slug = re.sub(r"[^a-z0-9]+", "_", candidate_id.lower()).strip("_")
    if not candidate_id.isascii():
        # Slugging drops non-ASCII characters, so distinct locale IDs
        # (玄関_low_battery / 寝室_low_battery) would collapse to the same
        # slug and the second proposal would be treated as a duplicate rule
        # (issue #522 Codex verification round). A stable digest suffix
        # preserves uniqueness while staying ASCII and deterministic.
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug}_{digest}".strip("_")
    return slug or default
