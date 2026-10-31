"""Rule: pet detected by camera at night while no one is home."""

from __future__ import annotations

import bisect
import logging
import re
from typing import TYPE_CHECKING

from custom_components.home_generative_agent.const import (
    SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    build_anomaly_id,
    enrolled_people,
    minutes_between,
    sighting_timestamp,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)

# The animal words a caption can name, each mapped to the singular the push
# names. Regular plurals are the pattern's optional "s"; irregular ones are
# listed. Order does not matter: no word is another plus an optional "s".
_PET_WORDS: dict[str, str] = {
    "bird": "bird",
    "bunnies": "bunny",
    "bunny": "bunny",
    "cat": "cat",
    "dog": "dog",
    "hamster": "hamster",
    "kitten": "kitten",
    "kitties": "kitty",
    "kitty": "kitty",
    "parrot": "parrot",
    "pet": "pet",
    "puppies": "puppy",
    "puppy": "puppy",
    "rabbit": "rabbit",
}

# Compounds in which the animal word names an object, not an animal: "a cat
# door", "the dog-bed", "a bird feeder", "a hot dog". Singular; the pattern
# adds the optional "s" (so "beds down" and "toys with" are excluded too: a
# missed low-severity push over a push asserting an animal that is a bed).
_OBJECT_AFTER = (
    "bed",
    "bowl",
    "carrier",
    "collar",
    "door",
    "feeder",
    "flap",
    "food",
    "house",
    "leash",
    "litter",
    "toy",
    "tree",
)

# A space, or any hyphen or dash a caption model may write (U+2010..U+2015).
_SEP = r"[\s\-\u2010-\u2015]"

# A third-party camera's caption attribute is not capped like HGA's own
# captions; the scan below is linear, and this keeps it short regardless.
_MAX_CAPTION_CHARS = 2000

# Whole words only, singular or plural, and an object compound is not an
# animal. Case folding is ASCII-only inside the word group, so a dotless i
# (U+0131) or a long s (U+017F) never matches and the table lookup cannot
# raise; the word boundaries around it stay Unicode-aware, so "catégorie" is
# one word and not "cat" + "égorie".
_PET_PATTERN = re.compile(
    rf"(?<!\bhot{_SEP})\b(?a:(?i:("
    + "|".join(_PET_WORDS)
    + rf"))s?)\b(?!{_SEP}+(?a:(?i:"
    + "|".join(_OBJECT_AFTER)
    + r")s?)\b)"
)

# A caption can say what is absent. The animal word is negated by a negator
# in its clause (a clause ends at sentence punctuation or at "but") that is
# at most three words before it with no comma between ("not a cat", "no
# people or pets"), or that opens a list of single words the animal
# continues ("no dogs, cats or birds"); and by "is not", "not visible",
# "absent" or "no longer" right after it. A comma otherwise ends a
# negator's reach, so "No people are visible, a cat sits on the step" is a
# cat: the shape a presence-minded caption takes when the home is empty.
_NEGATORS = r"(?:no|not|without|neither|nor|free of)"
_NEG_SHORT = re.compile(rf"\b{_NEGATORS}\b(?:\s+[^\s,]+){{0,3}}\s*$", re.IGNORECASE)
_NEG_LIST = re.compile(
    rf"\b{_NEGATORS}\s+[^\s,]+(?:\s*,\s*[^\s,]+){{0,2}}\s*(?:,|\s(?:or|and|nor))\s*$",
    re.IGNORECASE,
)
_NEG_AFTER = re.compile(
    r"^\s*(?:(?:is|are|was|were)\s+not\b|not\s+(?:visible|present|seen)\b"
    r"|is\s+absent\b|(?:is|are)\s+no\s+longer\b)",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[.;:!?]|\bbut\b", re.IGNORECASE)


def pet_in_summary(summary: str) -> str | None:
    """
    Return the animal the snapshot summary names (singular), or None.

    One pass over the caption for clause ends, one for animal words; each
    word's clause is found by bisection, so a long caption costs its length,
    not its length times its matches. A named animal wins over the bare
    "pet", which is also a verb ("a child pets the dog").
    """
    text = summary[:_MAX_CAPTION_CHARS]
    clause_starts = [0, *(m.end() for m in _CLAUSE_END.finditer(text))]
    generic: str | None = None
    for match in _PET_PATTERN.finditer(text):
        i = bisect.bisect_right(clause_starts, match.start()) - 1
        start = clause_starts[i]
        end = clause_starts[i + 1] if i + 1 < len(clause_starts) else len(text)
        before = text[start : match.start()]
        if _NEG_SHORT.search(before) or _NEG_LIST.search(before):
            continue
        if _NEG_AFTER.match(text[match.end() : end]):
            continue
        animal = _PET_WORDS.get(match.group(1).lower(), "pet")
        if animal != "pet":
            return animal
        generic = animal
    return generic


class PetDetectedAtNightNoOccupancyRule:
    """Detect a pet on camera at night while no residents are home."""

    rule_id = "pet_detected_at_night_no_occupancy"
    requires: frozenset[str] = frozenset()
    cooldown_minutes = 0

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return findings when a pet is seen at night with no one home."""
        if not snapshot["derived"]["is_night"]:
            return []
        if snapshot["derived"]["anyone_home"]:
            return []

        generated_at = snapshot["generated_at"]
        entity_map = {e["entity_id"]: e for e in snapshot["entities"]}
        findings: list[AnomalyFinding] = []
        for activity in snapshot["camera_activity"]:
            summary = activity.get("snapshot_summary")
            if not summary:
                continue
            if (
                not activity.get("motion_entities")
                and not activity.get("vmd_entities")
                and not activity.get("last_activity")
            ):
                continue
            animal = pet_in_summary(summary)
            if animal is None:
                continue
            camera_id = activity["camera_entity_id"]
            # Fresh sightings only, dated like the other camera rules: the
            # recognition event that produced the caption when there is one,
            # else the camera's last activity. A caption from the afternoon
            # must not fire at nightfall once the home empties, and an
            # unprovable timestamp (missing, unparseable, in the future) is
            # not fresh.
            sighting = sighting_timestamp(activity)
            age = minutes_between(sighting, generated_at)
            if age is None or age > SENTINEL_CAMERA_ACTIVITY_STALENESS_MINUTES:
                LOGGER.debug(
                    "Pet on %s skipped: sighting not fresh (age %s min).",
                    camera_id,
                    age,
                )
                continue
            people = [str(p) for p in (activity.get("recognized_people") or []) if p]
            if known := enrolled_people(people):
                # A known person on the camera means someone is home whatever
                # presence says (a resident whose phone still reads away):
                # the rule's premise fails, and a push would claim the home
                # is empty (#730). The sighting is fresh (gate above), so the
                # recognition is current when it dates the sighting.
                LOGGER.debug(
                    "Pet on %s at night skipped: %d recognized person(s) there.",
                    camera_id,
                    len(known),
                )
                continue

            # Stable identity: the camera, the animal, the caption and the
            # sighting it came from. Not the age (it grows every evaluation),
            # the recognition labels (a batch-level union that flips between
            # "Indeterminate" and "Unknown Person" for one scene) or who is
            # away: a new id per evaluation would re-notify the same sighting
            # through every cooldown and miss its pending prompt.
            id_evidence = {
                "camera_entity_id": camera_id,
                "animal": animal,
                "snapshot_summary": summary,
                "sighting_last_event": sighting,
            }
            anomaly_id = build_anomaly_id(self.rule_id, [camera_id], id_evidence)
            cam_entity = entity_map.get(camera_id)
            evidence = {
                **id_evidence,
                "area": activity.get("area"),
                "camera_friendly_name": (
                    cam_entity.get("friendly_name") if cam_entity else None
                ),
                "camera_activity_age_minutes": round(age, 1),
                "last_activity": activity.get("last_activity"),
                "recognized_people": people,
                "is_night": snapshot["derived"]["is_night"],
                "anyone_home": snapshot["derived"]["anyone_home"],
                "people_away": snapshot["derived"]["people_away"],
            }
            findings.append(
                AnomalyFinding(
                    anomaly_id=anomaly_id,
                    type=self.rule_id,
                    severity="low",
                    confidence=0.85,
                    triggering_entities=[camera_id],
                    evidence=evidence,
                    suggested_actions=[],
                    # Not sensitive: no model prose is ever written for this
                    # type (its copy is fixed, see notifier.is_security_copy),
                    # so there is nothing to redact, and an enrolled person on
                    # the camera skips it above. The flag would only bias
                    # triage and mark a compound finding sensitive.
                    is_sensitive=False,
                )
            )
        return findings
