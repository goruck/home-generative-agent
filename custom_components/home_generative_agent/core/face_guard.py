"""
Tell the user when the unknown-person Sentinel rules cannot fire.

Five evaluators (under four rule ids) key on a *positive* stranger signal:
``unknown_person_camera_no_home``, ``unknown_person_camera_night_home``,
``alarm_disarmed_during_external_threat``, and the dynamic templates
``unknown_person_camera_no_home`` and ``unknown_person_camera_when_home``
(the first id is shared with the static rule). All of them go through
``unknown_person_sighting_is_actionable`` (``sentinel/models.py``), which
requires the literal ``"Unknown Person"`` label in a camera activity's
``recognized_people`` list.

Only the face-recognition pipeline ever writes that label. With no face
service, ``recognized_people`` is always empty, so the predicate is
permanently ``False`` and every one of them is inert -- silently. Nothing
errors, nothing is logged, and the rules still appear active on the health
sensor, because from the engine's point of view they simply found nothing.

The honest response is the same one ``pipeline_guard`` gives for a PIN that
cannot fire: say so, name the rules, and point at the replacement. This
module raises a repair issue when Sentinel is on and face recognition is not
operative, and clears it once that changes.

**Operative, not configured.** The predicate cares whether the label is
actually produced, which needs the face service *and* the person gallery --
and the gallery needs the database. ``async_setup_entry`` already downgrades
``face_recognition`` to ``False`` when the gallery is unavailable, so the
caller passes that effective value rather than the raw option. A user who
enabled face recognition but has no Database subentry is in exactly the
silent-inert state this issue exists to surface, and would be missed by a
check that only read the option.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

ISSUE_UNKNOWN_PERSON_RULES_INERT = "unknown_person_rules_inert"

# The rules whose only trigger is a positive "Unknown Person" label, named here
# rather than derived from the engine's rule list: the issue text has to name
# them for a user who cannot read the code, and the dynamic templates are not
# static rule objects at all. Every id was read from its source of truth rather
# than transcribed -- the static ones from each rule class's ``rule_id``, the
# dynamic one from the evaluator dispatch map in ``sentinel/dynamic_rules.py``.
# Four ids cover five evaluators: ``unknown_person_camera_no_home`` is both a
# static rule and a dynamic template, so it is listed once.
INERT_WITHOUT_FACE_RECOGNITION: tuple[str, ...] = (
    "unknown_person_camera_no_home",
    "unknown_person_camera_night_home",
    "alarm_disarmed_during_external_threat",
    "unknown_person_camera_when_home (approved discovery rule)",
)


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_{entry_id}"


@callback
def async_check_unknown_person_rules(
    hass: HomeAssistant,
    entry_id: str,
    *,
    sentinel_enabled: bool,
    face_recognition_operative: bool,
) -> None:
    """
    Raise or clear the repair issue for inert unknown-person rules.

    Idempotent: safe on every setup and reload. Deleting an issue that does not
    exist is a no-op in the issue registry, so the clear paths need no
    existence check.

    Nothing is raised when Sentinel is off -- then *no* rule fires and saying
    so about five of them would be noise rather than news.
    """
    issue_id = _issue_id(entry_id)

    if not sentinel_enabled or face_recognition_operative:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_UNKNOWN_PERSON_RULES_INERT,
        translation_placeholders={
            "rules": "\n".join(f"- {name}" for name in INERT_WITHOUT_FACE_RECOGNITION),
        },
    )


@callback
def async_clear_unknown_person_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the issue when the entry is removed; nothing re-evaluates it after."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
