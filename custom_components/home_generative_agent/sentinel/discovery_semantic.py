"""Deterministic semantic key utilities for discovery candidates and rules."""

from __future__ import annotations

import re
from typing import Any

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
# (issue #518 Codex structured review).
_TEXT_ENTITY_ID_RE = re.compile(r"(?<![a-z0-9_.])([a-z_]+\.[a-z0-9_]+)")
# Away/home occupancy wording. Word-bounded mirrors of
# proposal_templates._AWAY_TERMS_PATTERN / _HOME_TERMS_PATTERN — the
# normalizer accepts "no one at home"/"without occupants" as away, and a
# substring "home" match here would key those candidates home=1 while their
# activated rule keys home=0, breaking dedup (issue #518 Codex structured
# review, empirically reproduced).
_AWAY_TERMS_RE = re.compile(
    r"\b(?:away|no(?:body|\s+one)\s+(?:is\s+)?(?:at\s+)?home|empty|unoccupied"
    r"|no occupants|without occupants)\b"
)
_HOME_TERMS_RE = re.compile(
    r"\b(?:someone home|occupied|home|present|occupants|residents)\b"
)


def candidate_semantic_key(  # noqa: C901, PLR0912, PLR0915
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
    entity_ids = _extract_entity_ids(evidence_paths)
    camera_ids = sorted(_extract_camera_ids(evidence_paths))
    lock_ids = sorted(
        entity_id for entity_id in entity_ids if entity_id.startswith("lock.")
    )
    # Motion-named IDs with an entry substring (binary_sensor.front_door_motion,
    # *_doorbell_motion) are motion sensors, not entries — without this the
    # candidate keys subject=entry_door while its activated motion rule keys
    # subject=motion and dedup never fires (#516 mirror, issue #518 review).
    window_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if "window" in entity_id and not _is_motion_named(entity_id)
    )
    door_ids = sorted(
        entity_id
        for entity_id in entity_ids
        if ("door" in entity_id or "entry" in entity_id)
        and not _is_motion_named(entity_id)
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
    if "unlocked" in text:
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
        and not (lock_ids and "battery" in text)
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
    elif "battery" in text and _contains_any(text, ("low", "below", "under")):
        predicate = "low_battery"
    elif _contains_any(
        text,
        ("power", "energy", "watt", "consumption", "kilowatt", "baseline", "deviation"),
    ):
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

    night = "any"
    if "night" in text or "derived.is_night" in evidence_paths:
        night = "1"

    home = "any"
    # "not derived.anyone_home" is the LLM's canonical absence-of-occupancy
    # path; without this, an evidence-only away candidate keys home=any and
    # never dedups against its activated home=0 rule (issue #516 review).
    if "not derived.anyone_home" in evidence_paths or _AWAY_TERMS_RE.search(text):
        home = "0"
    elif _HOME_TERMS_RE.search(text):
        home = "1"
    if "derived.anyone_home" in evidence_paths and home == "any":
        home = "1"
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
    """
    if rule_key == candidate_key:
        return True
    if "|template=" not in rule_key:
        return False
    return _SEMANTIC_KEY_CONTEXT_RE.sub("", rule_key) == _SEMANTIC_KEY_CONTEXT_RE.sub(
        "", candidate_key
    )


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
