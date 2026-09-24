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

# One issue id, three translation keys. A repair's description renders in the
# *viewing* user's frontend language, so a cause sentence composed in Python --
# or fetched with async_common_translation, which follows the *server* locale --
# could land in a different language than the text around it. A distinct
# translation_key per cause keeps the whole notice in one language and lets each
# one lead with the remedy that actually applies.
ISSUE_UNKNOWN_PERSON_RULES_INERT = "unknown_person_rules_inert"
TRANSLATION_KEY_NO_FACE = "unknown_person_rules_inert_no_face"
TRANSLATION_KEY_NO_GALLERY = "unknown_person_rules_inert_no_gallery"
TRANSLATION_KEY_NO_ANALYZER = "unknown_person_rules_inert_no_analyzer"

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
    # Approved discovery rules are minted per camera -- proposal_templates.py
    # builds `unknown_person_camera_when_home_<camera>` or
    # `..._any_camera` -- so the template id alone matches nothing in the
    # user's rule list. Say "starting with" rather than send them hunting.
    "any rule whose id starts with unknown_person_camera_when_home",
)


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_{entry_id}"


@callback
def async_check_unknown_person_rules(  # noqa: PLR0913
    hass: HomeAssistant,
    entry_id: str,
    *,
    sentinel_enabled: bool,
    face_recognition_configured: bool,
    person_gallery_available: bool,
    video_analysis_enabled: bool,
) -> None:
    """
    Raise or clear the repair issue for inert unknown-person rules.

    Idempotent: safe on every setup and reload. Deleting an issue that does not
    exist is a no-op in the issue registry, so the clear paths need no
    existence check.

    Takes the three facts separately rather than one "operative" flag so the
    notice can name the state the install is actually in. ``async_setup_entry``
    collapses the first two (it downgrades ``face_recognition`` to ``False``
    when the gallery is unavailable), and a user who switched face recognition
    on needs to be told about the missing database, not told to switch it on.

    Precedence when several apply: no face recognition, then no gallery, then
    no analyzer -- most fundamental first, since fixing an earlier one is a
    precondition for the later ones mattering.

    Nothing is raised when Sentinel is off -- then *no* rule fires and singling
    out four rule ids would be noise rather than news.

    **Known gap:** an unreachable or misconfigured face service leaves the rules
    inert too, and is not detected here -- ``recognize_faces`` logs a warning
    and returns an empty list, so unlike the states above that one is at least
    visible in the log. Tracked in TODOS.md.
    """
    issue_id = _issue_id(entry_id)

    if not sentinel_enabled:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    if not face_recognition_configured:
        translation_key = TRANSLATION_KEY_NO_FACE
    elif not person_gallery_available:
        translation_key = TRANSLATION_KEY_NO_GALLERY
    elif not video_analysis_enabled:
        translation_key = TRANSLATION_KEY_NO_ANALYZER
    else:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    # The issue id is stable across causes, so a change of cause replaces the
    # notice in place instead of stacking a second one; create_issue overwrites
    # an existing id.
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=translation_key,
        translation_placeholders={
            "rules": "\n".join(f"- {name}" for name in INERT_WITHOUT_FACE_RECOGNITION),
        },
    )


@callback
def async_clear_unknown_person_issue(hass: HomeAssistant, entry_id: str) -> None:
    """Drop the issue when the entry is removed; nothing re-evaluates it after."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
