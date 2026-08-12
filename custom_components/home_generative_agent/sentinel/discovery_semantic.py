"""Deterministic semantic key utilities for discovery candidates and rules."""

from __future__ import annotations

import re
from typing import Any

# Away/home/night context resolves through evidence_paths.presence_signal /
# night_signal — the same functions the normalizer delegates to — so the
# candidate key provably matches the normalizer's routing. The former local
# regex mirrors are gone: this module's decoupling convention was about not
# importing the normalization module, and a shared leaf module is the fix
# for the drift hazard the mirrors created (issue #518 review noted an
# asymmetric pair keys a candidate home=1 while its activated rule keys
# home=0, breaking dedup in both directions; issue #524).
from .evidence_paths import night_signal, presence_signal

_SEMANTIC_KEY_CONTEXT_RE = re.compile(r"\|(?:template|night|home|scope)=[^|]+")
# Quote characters the discovery LLM sometimes wraps around entity IDs in
# evidence paths. Mirrors proposal_templates._EVIDENCE_QUOTE_CHARS; kept local
# to avoid coupling the semantic-key module to the normalization module.
_EVIDENCE_QUOTE_CHARS = "'\"`"
# Dot-notation entity IDs embedded in candidate prose. Mirrors
# proposal_templates._TEXT_ENTITY_ID_PATTERN / _find_text_motion_entity_ids
# (issue #518): index-based evidence paths (entities[31].state) never resolve
# to an entity ID, so without a prose fallback such a candidate keys
# subject=unknown|entities= and can never dedup against its activated rule.
# Motion-only, matching what the normalizer can actually normalize from
# prose: a broader fallback would mint coverage keys for candidate classes
# (lock/entry) that stay unsupported, letting an unresolvable candidate's
# history key suppress a later fully-evidenced, approvable proposal
# (issue #518 Codex structured review). Also reused by _extract_entity_ids
# to shape-check bare-bracket evidence tokens (issue #522) — the domain
# gate for that use lives in _BARE_BRACKET_DOMAINS below.
_TEXT_ENTITY_ID_RE = re.compile(r"(?<![a-z0-9_.])([a-z_]+\.[a-z0-9_]+)")
# Bare-bracket token shape: a dot-notation entity ID optionally followed by
# an attribute suffix (entities[sensor.x.state], LLM output variance).
# Mirrors proposal_templates._DOT_NOTATION_ENTITY_PATTERN — a fullmatch-only
# check would reject the attribute-suffixed form the normalizer accepts,
# keying entities= empty while the activated rule keys the sensor (issue
# #522 testing review, empirically reproduced dedup break).
_BARE_BRACKET_TOKEN_RE = re.compile(r"^([a-z_]+\.[a-z0-9_]+)(?:[.\[]|$)")
# Domains accepted from bare-bracket evidence tokens (issue #522). Mirrors
# proposal_templates._HA_ENTITY_DOMAINS; kept local per this module's
# decoupling convention. Without the gate, snapshot-path tokens like
# entities[derived.is_night] or entities[attributes.window_state] would key
# pseudo-entities the normalizer resolves to nothing ("window" substring →
# subject=entry_window), diverging candidate keys from their rule keys.
_BARE_BRACKET_DOMAINS = frozenset(
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
# Mirrors proposal_templates._LOW_BATTERY_QUALIFIERS — an asymmetric list
# breaks dedup in both directions (issue #522 adversarial review).
_LOW_BATTERY_QUALIFIERS = ("low", "below", "under", "weak")
_SLUG_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
# Mirrors proposal_templates._ENVIRONMENTAL_SIGNAL_TERMS (issue #541); kept
# local per this module's decoupling convention. An asymmetric vocabulary
# breaks dedup in both directions: a term only the normalizer knows registers
# a statistical rule whose key the candidate never matches, and a term only
# this module knows collapses context on a candidate that stays unsupported.
_ENVIRONMENTAL_SIGNAL_TERMS = (
    "temperature",
    "temp",
    "humidity",
    "pressure",
    "co2",
    "co",
    "carbon dioxide",
    "carbon monoxide",
    "pm25",
    "pm2 5",
    "pm10",
    "aqi",
    "air quality",
    "moisture",
    "illuminance",
    "lux",
)
# Mirrors proposal_templates._NUMERIC_THRESHOLD_PATTERN: the normalizer
# routes an environmental candidate with positive-numeric-threshold prose to
# sensor_threshold_condition (context-carrying params) and everything else to
# the context-free statistical templates, so the key's context collapse must
# flip on the same detector.
_NUMERIC_THRESHOLD_RE = re.compile(
    r"(?:>|above|exceeds?|over|more than|greater than)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+,\d{1,2}|\d+(?:\.\d+)?)"
    r"(?!\s*(?:hour|hr|minute|min)s?\b)(?![\d,.])"
)


def _has_environmental_signal(text_or_entity_id: str) -> bool:
    """
    Mirror of proposal_templates._has_environmental_signal (issue #541).

    Whole-token matching over a non-alphanumeric split; never fed the
    candidate_id slug (issue #522 posture — no slug-only routing).
    """
    normalized = f" {_SLUG_TOKEN_SPLIT_RE.sub(' ', text_or_entity_id.lower())} "
    return any(f" {term} " in normalized for term in _ENVIRONMENTAL_SIGNAL_TERMS)


def _has_numeric_threshold(text: str) -> bool:
    """Mirror of proposal_templates._extract_threshold_numeric's success path."""
    match = _NUMERIC_THRESHOLD_RE.search(text)
    if not match:
        return False
    raw = match.group(1)
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
        raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return False
    return value > 0


# Mirrors proposal_templates._STALENESS_TERMS for the environmental leg's
# staleness gate ONLY (issue #541 ship red-team review); kept local per this
# module's decoupling convention. The pre-existing staleness predicate leg
# below keeps its own narrower tuple — this mirror exists so the env verdict
# flips on exactly the signal that makes the normalizer's env arm defer to
# its entity_staleness branch.
_ENV_STALENESS_GATE_TERMS = (
    "stale",
    "not updated",
    "tracking",
    "staleness",
    "last seen",
    "last updated",
    "gps",
)


def _has_environmental_candidate_signal(
    text: str, entity_ids: list[str], named_battery_ids: list[str], subject: str
) -> bool:
    """
    Environmental leg verdict (issue #541), mirroring the normalizer's guard.

    Non-battery ``sensor.*`` evidence must be present (the statistical branch
    cannot register a rule without it), and the signal reads from prose or an
    evidence entity ID — never the candidate_id slug (issue #522 posture).
    Battery-NAMED ids are excluded so home-battery telemetry like
    ``sensor.battery_temperature`` keeps its base treatment.

    The ``subject == "sensor"`` gate is the #540 battery-arm lesson applied
    to this leg: the key chain's earlier subject classes (entry, lock,
    motion, alarm) mirror normalizer branches that run BEFORE its statistical
    branch, so a motion candidate that incidentally cites a temperature
    sensor ("motion in the attic near the temperature sensor while away")
    must keep its subject=motion|predicate=active keying — the normalizer
    registers a motion rule for it, and an environmental predicate steal
    would leave that rule unable to dedup its own candidate. Sensor-only
    evidence is exactly ``subject == "sensor"`` here, because the subject
    chain assigns "sensor" only when no earlier class matched.
    """
    if subject != "sensor":
        return False
    # Staleness gate (ship red-team review): the normalizer's env arm defers
    # staleness-worded candidates to its entity_staleness branch, so the key
    # must fall through to the staleness predicate leg for the same prose or
    # the mirror breaks in both directions.
    if _contains_any(text, _ENV_STALENESS_GATE_TERMS):
        return False
    non_battery_sensor_ids = [
        entity_id
        for entity_id in entity_ids
        if entity_id.startswith("sensor.") and entity_id not in named_battery_ids
    ]
    if not non_battery_sensor_ids:
        return False
    return _has_environmental_signal(text) or any(
        _has_environmental_signal(entity_id) for entity_id in non_battery_sensor_ids
    )


# Prose wording that keys a candidate predicate=power_anomaly. Baseline /
# deviation wording is included because statistical candidates describe their
# pattern that way regardless of the measured quantity. Near-mirror of
# proposal_templates._POWER_ENERGY_TERMS with two pre-existing, intentional
# deltas: "baseline"/"deviation" exist only here (keying-side statistical
# wording), and "usage" exists only on the normalizer side (a usage-only-prose
# candidate keys via other legs; closing that residual would rekey existing
# pending proposals, so it stays documented rather than fixed).
_POWER_ANOMALY_TERMS = (
    "power",
    "energy",
    "watt",
    "consumption",
    "kilowatt",
    "baseline",
    "deviation",
)


def _has_power_anomaly_signal(text: str, *, environmental: bool) -> bool:
    """Power/baseline prose wording, or the environmental leg (issue #541)."""
    return environmental or _contains_any(text, _POWER_ANOMALY_TERMS)


def _override_env_prose_steal(
    text: str,
    predicate: str,
    entities: list[str],
    sensor_ids: list[str],
    *,
    environmental: bool,
) -> tuple[str, list[str]]:
    """
    Re-key environmental candidates whose prose grazed an entry/lock/alarm leg.

    "Temperature drops sharply, which may mean a window was left open" keys
    predicate=open via the bare-substring leg, but the normalizer's entry,
    lock, and alarm branches all require matching entity evidence while its
    statistical branch does not — so this sensor-only candidate registers
    baseline_deviation and an open/unlocked/disarmed key would never dedup
    against it (ship red-team review, reproduced). The environmental verdict
    is subject-gated to sensor-only evidence, so candidates with real entry,
    lock, or alarm evidence never reach this override; unavailable, camera,
    and battery predicates are deliberately not listed — their normalizer
    branches run before the statistical branch and fire on prose alone.

    Availability wording blocks the override entirely (Codex adversarial
    review, reproduced): "unavailable; unable to open its connection" keys
    predicate=open (the open leg precedes the unavailable leg), and
    rewriting it to power_anomaly would let an active baseline rule on the
    same sensor falsely cover the OUTAGE proposal the normalizer's
    availability branch actually registers. Keeping the base key restores
    the pre-#541 no-cover/no-suppress behavior.
    """
    if (
        environmental
        and predicate in ("unlocked", "open", "disarmed")
        and not _contains_any(text, ("unavailable", "offline", "unreachable"))
    ):
        predicate = "power_anomaly"
        if not entities and sensor_ids:
            entities = sensor_ids
    return predicate, entities


def _resolve_night_home(
    evidence_paths: list[str],
    text: str,
    predicate: str,
    *,
    environmental: bool,
) -> tuple[str, str]:
    """
    Resolve the key's night/home fields, applying context-free overrides.

    Occupancy resolves through the shared evidence-first signal (issue
    #524): without the structured paths, an evidence-only away candidate
    keys home=any and never dedups against its activated home=0 rule
    (issue #516 review).

    low_battery override: the normalizer's battery templates ignore
    night/occupancy context entirely, and rule_semantic_key hardcodes
    night=any|home=any for low_battery_sensors — a battery candidate
    carrying night or occupancy wording would otherwise never dedup against
    its own activated rule and be re-proposed indefinitely (issue #522
    red-team review).

    Environmental override: see _environmental_context_collapses —
    statistical environmental candidates collapse to the context-free key
    their registered rule carries; threshold-prose candidates keep context
    (issue #541).
    """
    night = "1" if night_signal(evidence_paths, text) else "any"
    home = {"away": "0", "home": "1", "any": "any"}[
        presence_signal(evidence_paths, text)
    ]
    if predicate == "low_battery" or _environmental_context_collapses(
        text, predicate=predicate, environmental=environmental
    ):
        night = "any"
        home = "any"
    return night, home


def _environmental_context_collapses(
    text: str,
    *,
    predicate: str,
    environmental: bool,
) -> bool:
    """
    Return True when an env candidate's night/home context must collapse.

    Statistical environmental candidates key context-free (issue #541): the
    normalizer routes them to baseline_deviation / time_of_day_anomaly, whose
    params and rule keys carry no occupancy/night context, so context
    variants must collapse to one key — the #540 battery pile-up lesson.
    Threshold-prose candidates keep context, mirroring
    sensor_threshold_condition's require_night/require_away/require_home
    params. ``environmental`` is the subject-gated
    _has_environmental_candidate_signal verdict, and the predicate gate
    keeps the collapse off candidates an earlier keying leg claimed.
    """
    return (
        predicate == "power_anomaly"
        and environmental
        and not _has_numeric_threshold(text)
    )


def _has_low_battery_signal(text: str, slug_text: str) -> bool:
    """
    Mirror of proposal_templates._has_low_battery_signal (issue #522).

    Prose keeps substring semantics; the candidate_id slug is matched on
    whole tokens, and each surface must carry the full conjunctive signal
    on its own ("backup_battery_water_flow" must not qualify via "flow").
    """
    if "battery" in text and _contains_any(text, _LOW_BATTERY_QUALIFIERS):
        return True
    tokens = set(_SLUG_TOKEN_SPLIT_RE.split(slug_text))
    return "battery" in tokens and any(
        qualifier in tokens for qualifier in _LOW_BATTERY_QUALIFIERS
    )


# Mirrors proposal_templates._NON_BATTERY_ID_TOKENS for the battery-leg
# entity normalization below; kept local per this module's decoupling
# convention.
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


# Measurement-kind tokens that mark a battery-NAMED sensor as a real
# measurement stream rather than a charge level, but only when they appear
# AFTER the "battery" token: sensor.battery_power / sensor.battery_temperature
# are home-battery telemetry, while sensor.garage_temperature_sensor_battery /
# sensor.front_door_battery are charge levels of the named device.
# Deliberately narrower than _NON_BATTERY_ID_TOKENS: that list also carries
# device-type tokens (door, motion, window, smoke, leak) used to disqualify
# UNNAMED locale-fallback candidates — applying it to battery-named IDs would
# misclassify the canonical HA device-battery names and keep them gap-hinted,
# which is exactly the confusing-candidate behavior being suppressed
# (pre-landing review, adversarial round).
_BATTERY_MEASUREMENT_STREAM_TOKENS = (
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
    "signal",
    "rssi",
    "linkquality",
)


def is_battery_level_entity_id(entity_id: str) -> bool:
    """
    Classify battery-level ``sensor.*`` IDs for baseline-eligibility gating.

    Battery percentage declines monotonically, so a rolling-average deviation
    on it is only a laggy low-battery detector — threshold rules are the right
    tool, and the discovery gap hint must not push battery sensors at the LLM
    as statistical-anomaly material. A measurement-kind token AFTER the
    battery token (``sensor.battery_power`` on a home battery) marks a real
    measurement stream that stays baseline-eligible; device-named battery
    levels (``sensor.front_door_battery``,
    ``sensor.garage_temperature_sensor_battery``) are charge levels and are
    excluded. Lowercased defensively — registry IDs are canonically
    lowercase, but this also takes DB-sourced baseline IDs.
    """
    entity_id = entity_id.lower()
    if not entity_id.startswith("sensor."):
        return False
    object_id = entity_id.split(".", 1)[1]
    battery_pos = object_id.rfind("battery")
    if battery_pos == -1:
        return False
    tail = object_id[battery_pos + len("battery") :]
    return not any(token in tail for token in _BATTERY_MEASUREMENT_STREAM_TOKENS)


def _named_battery_sensor_entity_ids(entity_ids: list[str]) -> list[str]:
    """Battery-named ``sensor.*`` IDs — the normalizer's primary collection."""
    return sorted(
        {
            entity_id
            for entity_id in entity_ids
            if entity_id.startswith("sensor.") and "battery" in entity_id
        }
    )


def _battery_sensor_entity_ids(entity_ids: list[str]) -> list[str]:
    """
    Mirror of the normalizer's battery-sensor collection for key entities.

    Battery-named ``sensor.*`` IDs win; otherwise the single unambiguous
    non-excluded ``sensor.*`` survivor (the issue #522 locale fallback).
    Returns [] when the normalizer would also resolve nothing.
    """
    named = _named_battery_sensor_entity_ids(entity_ids)
    if named:
        return named
    fallback = sorted(
        {
            entity_id
            for entity_id in entity_ids
            if entity_id.startswith("sensor.")
            and not any(token in entity_id for token in _NON_BATTERY_ID_TOKENS)
        }
    )
    if len(fallback) == 1:
        return fallback
    return []


def candidate_semantic_key(  # noqa: PLR0912, PLR0915
    candidate: dict[str, Any],
) -> str | None:
    """Build a stable semantic key for a discovery candidate."""
    evidence_paths = _string_list(candidate.get("evidence_paths"))
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("pattern", "")),
            str(candidate.get("suggested_type", "")),
        ]
    ).lower()
    # Mirrors the normalizer's slug_text (issue #522): candidate_id is
    # often the only English surface when the discovery LLM writes prose in
    # the home's locale, and the normalizer routes such candidates to
    # low_battery_sensors from the slug alone. Scoped to the battery
    # predicate leg only — the normalizer keeps every other signal
    # prose-only, and keying night/home/subject from slug tokens the
    # normalizer ignores would break the key mirror.
    slug_text = str(candidate.get("candidate_id", "")).lower()
    entity_ids = _extract_entity_ids(evidence_paths)
    camera_ids = sorted(_extract_camera_ids(evidence_paths))
    lock_ids = sorted(
        entity_id for entity_id in entity_ids if entity_id.startswith("lock.")
    )
    # Motion-named IDs with an entry substring (binary_sensor.front_door_motion,
    # *_doorbell_motion) are motion sensors, not entries — without this the
    # candidate keys subject=entry_door while its activated motion rule keys
    # subject=motion and dedup never fires (#516 mirror, issue #518 review).
    # sensor.-domain IDs are excluded outright, mirroring the normalizer's
    # _find_entry_entity_ids domain gate ({binary_sensor, cover}): a
    # sensor.* entity can never register an entry rule there, and #541 makes
    # the divergence bite — sensor.outdoor_temperature would key
    # subject=entry_door via the "door" substring while its promotion
    # registers a subject=sensor baseline rule that then never dedups its
    # own re-proposals (ship testing-specialist review, reproduced).
    # Domainless legacy IDs keep the tolerant substring behavior, matching
    # the normalizer's domainless leg.
    window_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if "window" in entity_id
        and not _is_motion_named(entity_id)
        and not entity_id.startswith("sensor.")
    )
    door_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if ("door" in entity_id or "entry" in entity_id)
        and not _is_motion_named(entity_id)
        and not entity_id.startswith("sensor.")
    )
    motion_ids = sorted(
        entity_id for entity_id in entity_ids if _is_motion_named(entity_id)
    )
    if not motion_ids:
        # Index-based evidence paths (entities[31].state, issue #518) carry
        # no entity ID — fall back to motion-named binary_sensor IDs in the
        # prose so the candidate keys the same subject/entities as its
        # activated rule. Gated per-class (no *motion* evidence resolved,
        # not "no evidence at all") to mirror the normalizer: a candidate
        # citing a person tracker in evidence plus an index-based motion
        # path still normalizes from prose (issue #518 adversarial review).
        motion_ids = _extract_text_motion_ids(text)
    sensor_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if entity_id.startswith(("sensor.", "binary_sensor."))
    )
    alarm_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if entity_id.startswith("alarm_control_panel.")
    )

    subject = "unknown"
    entities: list[str] = []
    if window_ids:
        subject = "entry_window"
        entities = window_ids
    elif door_ids:
        subject = "entry_door"
        entities = door_ids
    elif lock_ids:
        subject = "lock"
        entities = lock_ids
    elif motion_ids:
        subject = "motion"
        entities = motion_ids
    elif alarm_ids:
        subject = "alarm"
        entities = alarm_ids
    elif sensor_ids:
        subject = "sensor"
        entities = sensor_ids

    predicate = "unknown"
    named_battery_ids = _named_battery_sensor_entity_ids(entity_ids)
    # Environmental leg (issue #541): mirrors the normalizer's widened
    # statistical branch, subject-gated per the #540 battery-arm lesson.
    # Computed once so the power_anomaly leg below and the context collapse
    # in _resolve_night_home share one verdict.
    has_environmental_signal = _has_environmental_candidate_signal(
        text, entity_ids, named_battery_ids, subject
    )
    lock_battery_targets = (
        _battery_sensor_entity_ids(entity_ids)
        if lock_ids
        and _has_low_battery_signal(text, slug_text)
        # The normalizer's availability branch precedes its lock-battery
        # branch — an unavailable lock-battery sensor routes to
        # unavailable_sensors, so the hoist must not steal the predicate
        # from the unavailable leg below (issue #522 verification round 2).
        and not _contains_any(text, ("unavailable", "offline", "unreachable"))
        else []
    )
    if lock_battery_targets:
        # Mirrors the normalizer's lock-battery precedence (issue #522
        # verification round): a lock candidate with a low-battery signal
        # routes to low_battery_sensors BEFORE the unlocked-lock and
        # open-entry branches, so "unlocked"/"open" prose on a compound
        # lock-battery candidate must not steal the predicate — the key
        # would never match the activated battery rule.
        predicate = "low_battery"
        subject = "sensor"
        entities = lock_battery_targets
    elif "unlocked" in text:
        predicate = "unlocked"
    elif "open" in text:
        predicate = "open"
    elif _contains_any(text, ("unavailable", "offline", "unreachable")):
        predicate = "unavailable"
    elif "disarmed" in text:
        predicate = "disarmed"
    elif (
        (camera_ids or _contains_any(text, ("camera", "cam")))
        and _contains_any(
            text,
            ("unknown", "unrecognized", "stranger", "unidentified", "indeterminate"),
        )
        and _contains_any(
            text,
            ("person", "people", "face", "occupant", "resident"),
        )
        # The normalizer's lock-battery branch precedes its camera branches:
        # a compound candidate citing a lock plus low-battery wording
        # normalizes to low_battery_sensors, so it must not key as an
        # unknown-person camera candidate (verification round 4).
        # Full conjunctive signal so a slug-only battery candidate (issue
        # #522) keeps the same lock-battery precedence the normalizer
        # applies — and prose with incidental "battery" but no qualifier
        # keeps its camera keying, matching the normalizer branch order.
        and not (lock_ids and _has_low_battery_signal(text, slug_text))
    ):
        # Checked before the battery/power/motion legs, mirroring the
        # normalizer's branch order (camera branches precede the battery
        # and power branches): an unknown-person camera candidate that also
        # cites a motion sensor or a power spike normalizes to the
        # sensitive camera template, so keying it subject=motion or
        # predicate=power_anomaly would let a plain motion or baseline
        # rule's coverage check silently swallow the camera proposal
        # (issue #518 verification reviews). Term lists mirror
        # proposal_templates._UNKNOWN_TERMS/_PERSON_TERMS ("occupant"/
        # "resident" cover their plurals as substrings). Text-only camera
        # signals key entities= to match any-camera rules.
        subject = "camera"
        predicate = "unknown_person"
        entities = camera_ids
    elif _has_low_battery_signal(text, slug_text) or (
        # Second arm mirrors the normalizer's disjunctive battery branch:
        # battery-named sensor evidence plus battery/low/below prose routes
        # to low_battery_sensors even WITHOUT a low-battery qualifier — a
        # "battery baseline deviation" candidate normalizes to the threshold
        # template, so keying it predicate=power_anomaly (with night/home
        # context preserved) would let context variants pile up as distinct
        # novel candidates and the activated rule never dedup re-proposals.
        # Named IDs only: the normalizer invokes its locale fallback solely
        # on the conjunctive signal, so the fallback must not widen this arm.
        named_battery_ids
        and _contains_any(text, ("battery", "low", "below"))
        # Sensor-only-evidence gate (pre-landing adversarial review,
        # empirically reproduced regression): the normalizer's battery
        # branch sits AFTER its alarm/motion/camera/entry branches, and the
        # key chain does not mirror all of them — a motion+camera candidate
        # that incidentally cites a battery sensor with bare "low" prose
        # ("camera activity stays low") must keep its motion/camera keying,
        # or an activated low_battery rule on that sensor silently swallows
        # the unrelated proposal and the activated motion rule never dedups
        # it. The arm therefore fires only when battery sensors are the
        # ONLY subject evidence. Window/door/motion-named ids that are
        # themselves battery-named (sensor.front_door_battery,
        # sensor.hallway_motion_battery) do not block: the normalizer's
        # _NON_ENTRY_ID_TOKENS contains "battery" and its away-motion
        # branches guard on battery_sensor_ids, so such ids are never
        # entries or motion subjects there and its battery branch handles
        # them (Codex structured review). Mixed motion+battery shapes keep
        # their (pre-existing, base-parity) keying rather than gaining a
        # new mismatch.
        and not camera_ids
        and not lock_ids
        and not alarm_ids
        and all(
            eid in named_battery_ids for eid in (*window_ids, *door_ids, *motion_ids)
        )
    ):
        predicate = "low_battery"
        # Mirror the normalizer's battery routing (issue #522 adversarial
        # review): it always registers subject=sensor rules on the
        # battery-named sensor.* IDs (or the single fallback survivor),
        # regardless of lock evidence, night wording, or occupancy — so the
        # key must not carry a lock subject, contextual evidence entities,
        # or night/home context the rule key (night=any|home=any) never has,
        # or the activated rule can never dedup re-proposals.
        battery_entity_ids = _battery_sensor_entity_ids(entity_ids)
        if battery_entity_ids:
            subject = "sensor"
            entities = battery_entity_ids
    elif _has_power_anomaly_signal(text, environmental=has_environmental_signal):
        predicate = "power_anomaly"
        if not entities and sensor_ids:
            entities = sensor_ids
    elif subject != "unknown" and _contains_any(
        text,
        ("stale", "staleness", "not updated", "last seen", "last updated"),
    ):
        # Mirrors the normalizer's staleness guard: a stale/dead-sensor
        # candidate must not key predicate=active, or an active motion rule
        # on the same sensor makes discovery drop it as already-covered
        # before the approval gate can return the honest "unsupported"
        # (verification round 4). Subject-less staleness candidates keep
        # keying None so identity-hash dedup applies — a shared
        # subject=unknown|predicate=staleness|entities= key would collide
        # across unrelated stale-tracker candidates.
        predicate = "staleness"
    elif "motion" in text or "activity" in text:
        predicate = "active"
    if predicate == "unavailable" and sensor_ids:
        subject = "sensor"
        entities = sensor_ids
    predicate, entities = _override_env_prose_steal(
        text, predicate, entities, sensor_ids, environmental=has_environmental_signal
    )

    night, home = _resolve_night_home(
        evidence_paths, text, predicate, environmental=has_environmental_signal
    )
    if subject == "unknown" and _contains_any(text, ("window", "windows")):
        subject = "entry_window"
    if predicate == "unknown" and "open" in text:
        predicate = "open"

    if subject == "unknown" and predicate == "unknown":
        return None
    entities_csv = ",".join(sorted(set(entities)))
    return (
        f"v1|subject={subject}|predicate={predicate}|night={night}|"
        f"home={home}|scope=any|entities={entities_csv}"
    )


def rule_semantic_key(  # noqa: C901, PLR0911, PLR0912, PLR0915
    rule: dict[str, Any],
) -> str | None:
    """Build a stable semantic key for an active/generated rule."""
    template_id = str(rule.get("template_id", ""))
    params = rule.get("params", {}) or {}
    if template_id == "unlocked_lock_when_home":
        lock_id = str(params.get("lock_entity_id", ""))
        if not lock_id:
            return None
        return (
            "v1|subject=lock|predicate=unlocked|night=any|home=1|scope=any|"
            f"entities={lock_id}"
        )
    if template_id == "alarm_disarmed_open_entry":
        entry_ids = sorted(set(_string_list(params.get("entry_entity_ids"))))
        if not entry_ids:
            return None
        entry_subject = (
            "entry_window"
            if any("window" in item for item in entry_ids)
            else "entry_door"
        )
        return (
            f"v1|subject={entry_subject}|predicate=open|night=any|home=any|scope=any|"
            f"entities={','.join(entry_ids)}"
        )
    if template_id == "open_any_window_at_night_while_away":
        return (
            "v1|subject=entry_window|predicate=open|night=1|home=0|scope=any|entities="
        )
    if template_id == "unknown_person_camera_no_home":
        camera_id = str(params.get("camera_entity_id", ""))
        if not camera_id and params.get("camera_selector") != "any":
            return None
        return (
            "v1|subject=camera|predicate=unknown_person|night=any|home=0|scope=any|"
            f"entities={camera_id}"
        )
    if template_id == "unknown_person_camera_when_home":
        camera_id = str(params.get("camera_entity_id", ""))
        if not camera_id and params.get("camera_selector") != "any":
            return None
        return (
            "v1|subject=camera|predicate=unknown_person|night=any|home=1|scope=any|"
            f"entities={camera_id}"
        )
    if template_id == "motion_without_camera_activity":
        motion_ids = sorted(set(_string_list(params.get("motion_entity_ids"))))
        if not motion_ids:
            return None
        return (
            "v1|subject=motion|predicate=active|night=any|home=any|scope=any|"
            f"entities={','.join(motion_ids)}"
        )
    if template_id == "motion_detected_at_night_while_alarm_disarmed":
        motion_ids = sorted(set(_string_list(params.get("motion_entity_ids"))))
        if not motion_ids:
            return None
        return (
            "v1|subject=motion|predicate=active|night=1|home=any|scope=any|"
            f"entities={','.join(motion_ids)}"
        )
    if template_id == "motion_detected_at_night_while_away":
        motion_ids = sorted(set(_string_list(params.get("motion_entity_ids"))))
        if not motion_ids:
            return None
        return (
            "v1|subject=motion|predicate=active|night=1|home=0|scope=any|"
            f"entities={','.join(motion_ids)}"
        )
    if template_id == "motion_detected_while_away":
        motion_ids = sorted(set(_string_list(params.get("motion_entity_ids"))))
        if not motion_ids:
            return None
        return (
            "v1|subject=motion|predicate=active|night=any|home=0|scope=any|"
            f"entities={','.join(motion_ids)}"
        )
    if template_id == "unavailable_sensors_while_home":
        sensor_ids = sorted(set(_string_list(params.get("sensor_entity_ids"))))
        if not sensor_ids:
            return None
        return (
            "v1|subject=sensor|predicate=unavailable|night=any|home=1|scope=any|"
            f"entities={','.join(sensor_ids)}"
        )
    if template_id == "unavailable_sensors":
        sensor_ids = sorted(set(_string_list(params.get("sensor_entity_ids"))))
        if not sensor_ids:
            return None
        return (
            "v1|subject=sensor|predicate=unavailable|night=any|home=any|scope=any|"
            f"entities={','.join(sensor_ids)}"
        )
    if template_id == "low_battery_sensors":
        sensor_ids = sorted(set(_string_list(params.get("sensor_entity_ids"))))
        if not sensor_ids:
            return None
        return (
            "v1|subject=sensor|predicate=low_battery|night=any|home=any|scope=any|"
            f"entities={','.join(sensor_ids)}"
        )
    if template_id in {"baseline_deviation", "time_of_day_anomaly"}:
        entity_id = str(params.get("entity_id", ""))
        if not entity_id:
            return None
        return (
            f"v1|subject=sensor|predicate=power_anomaly"
            f"|template={template_id}|entities={entity_id}"
        )
    if template_id == "sensor_threshold_condition":
        sensor_id = str(params.get("sensor_entity_id", ""))
        if not sensor_id:
            return None
        return f"v1|subject=sensor|predicate=power_threshold|entities={sensor_id}"
    if template_id == "entity_state_duration":
        entity_id = str(params.get("entity_id", ""))
        target_state = str(params.get("target_state", ""))
        if not entity_id:
            return None
        return (
            f"v1|subject=entity|predicate=state_duration"
            f"|entities={entity_id}|state={target_state}"
        )
    if template_id == "entity_staleness":
        entity_id = str(params.get("entity_id", ""))
        if not entity_id:
            return None
        return f"v1|subject=entity|predicate=staleness|entities={entity_id}"
    return None


def rule_key_covers_candidate_key(rule_key: str, candidate_key: str) -> bool:
    """
    Return True if rule_key semantically covers candidate_key.

    For most templates the keys are structurally identical and simple equality
    suffices. For baseline_deviation / time_of_day_anomaly, rule_semantic_key
    embeds |template=<name>| and omits |night=|home=|scope=|, while
    candidate_semantic_key always emits those context fields. When |template=|
    is present in the rule key, normalize both to subject+predicate+entities
    before comparing.

    For superset-safe predicates, a rule keyed night=any / home=any also
    covers a candidate keying the specific value: the rule fires in a
    superset of the candidate's conditions. Without this, a pre-#524
    approved unavailable_sensors rule (home=any) stops covering the same
    idea once structured evidence keys the candidate home=1 — the candidate
    is re-proposed as a new while_home rule and approving it double-alerts
    on every occupied-hours outage (issue #524 red-team). The converse is
    NOT coverage: a home=1 rule is silent while away and cannot cover a
    home=any candidate.
    """
    if rule_key == candidate_key:
        return True
    if "|template=" in rule_key:
        return _SEMANTIC_KEY_CONTEXT_RE.sub(
            "", rule_key
        ) == _SEMANTIC_KEY_CONTEXT_RE.sub("", candidate_key)
    return _key_fields_covered(rule_key, candidate_key)


# Predicates whose rule templates carry no firing qualifier OUTSIDE the
# semantic key, so night=any|home=any truly means "fires unconditionally".
# NOT safe (issue #524 adversarial review, empirically reproduced):
# predicate=active — motion_without_camera_activity keys any/any but fires
# only while cameras are idle; predicate=open — alarm_disarmed_open_entry
# keys any/any but fires only while the alarm is disarmed. Generalized
# any-covers-specific for those falsely reports distinct night/away
# candidates as already covered (promote flow returns "already_active").
_SUPERSET_SAFE_PREDICATES = frozenset({"unavailable"})


def _key_fields_covered(rule_key: str, candidate_key: str) -> bool:
    """Field-wise coverage: rule night/home 'any' covers a specific value."""
    rule_fields = rule_key.split("|")
    candidate_fields = candidate_key.split("|")
    if len(rule_fields) != len(candidate_fields):
        return False
    predicate = next(
        (
            field.removeprefix("predicate=")
            for field in rule_fields
            if field.startswith("predicate=")
        ),
        "",
    )
    if predicate not in _SUPERSET_SAFE_PREDICATES:
        return False
    for rule_field, candidate_field in zip(rule_fields, candidate_fields, strict=True):
        if rule_field == candidate_field:
            continue
        prefix, _, rule_value = rule_field.partition("=")
        candidate_prefix, _, _candidate_value = candidate_field.partition("=")
        if (
            prefix != candidate_prefix
            or prefix not in ("night", "home")
            or rule_value != "any"
        ):
            return False
    return True


def _extract_camera_ids(evidence_paths: list[str]) -> list[str]:
    camera_ids: list[str] = []
    for path in evidence_paths:
        if path.startswith("camera_activity[entity_id="):
            camera_ids.append(
                path.split("camera_activity[entity_id=", 1)[1].split("]", 1)[0]
            )
        elif path.startswith("camera_activity[camera_entity_id="):
            camera_ids.append(
                path.split("camera_activity[camera_entity_id=", 1)[1].split("]", 1)[0]
            )
    return camera_ids


def _extract_entity_ids(evidence_paths: list[str]) -> list[str]:
    entity_ids: list[str] = []
    for path in evidence_paths:
        if path.startswith("entities[entity_id="):
            entity_id = path.split("entities[entity_id=", 1)[1].split("]", 1)[0]
            entity_id = entity_id.strip(_EVIDENCE_QUOTE_CHARS)
            if entity_id:
                entity_ids.append(entity_id)
        elif "entity_ids contains " in path:
            # LLM-generated format: entities[entity_ids contains sensor.foo].state
            part = path.split("entity_ids contains ", 1)[1]
            entity_id = part.split("]")[0].strip().strip(_EVIDENCE_QUOTE_CHARS)
            if entity_id:
                entity_ids.append(entity_id)
        elif path.startswith("entities["):
            # Bare-bracket format: entities[sensor.foo].state (issue #522).
            # Mirrors proposal_templates._extract_entity_id_from_evidence_path:
            # the bracket token must be a dot-notation entity ID with a known
            # HA domain, so index-based brackets (entities[31].state, issue
            # #518) and snapshot paths (entities[derived.is_night]) still
            # resolve nothing.
            token = path.split("entities[", 1)[1].split("]", 1)[0]
            token = token.strip(_EVIDENCE_QUOTE_CHARS)
            token_match = _BARE_BRACKET_TOKEN_RE.match(token)
            if token_match is not None:
                entity_id = token_match.group(1)
                if entity_id.split(".", 1)[0] in _BARE_BRACKET_DOMAINS:
                    entity_ids.append(entity_id)
    return entity_ids


def _is_motion_named(entity_id: str) -> bool:
    return "motion" in entity_id or "vmd" in entity_id


def _extract_text_motion_ids(text: str) -> list[str]:
    """Motion-named binary_sensor IDs written in candidate prose (#518)."""
    entity_ids: list[str] = []
    for match in _TEXT_ENTITY_ID_RE.finditer(text):
        entity_id = match.group(1)
        if entity_id.startswith("binary_sensor.") and _is_motion_named(entity_id):
            entity_ids.append(entity_id)
    return sorted(set(entity_ids))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
