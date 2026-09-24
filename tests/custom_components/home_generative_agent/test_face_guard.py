# ruff: noqa: S101
"""
Tests for the inert-unknown-person-rules repair issue (core/face_guard.py).

Five Sentinel rules trigger only on the literal ``"Unknown Person"`` label that
``unknown_person_sighting_is_actionable`` requires, and only the
face-recognition pipeline writes it. Without that pipeline the predicate is
permanently ``False``, so all five are inert with nothing logged and nothing on
the health sensor to say so. The guard turns that silence into a repair issue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from homeassistant.helpers import issue_registry as ir

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core.face_guard import (
    INERT_WITHOUT_FACE_RECOGNITION,
    ISSUE_UNKNOWN_PERSON_RULES_INERT,
    async_check_unknown_person_rules,
    async_clear_unknown_person_issue,
)
from custom_components.home_generative_agent.sentinel.rules.alarm_disarmed_external_threat import (
    AlarmDisarmedDuringExternalThreatRule,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_night_home import (
    UnknownPersonAtNightWhileHomeRule,
)
from custom_components.home_generative_agent.sentinel.rules.unknown_person_camera_no_home import (
    UnknownPersonCameraNoHomeRule,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_ENTRY = "entry_abc"
_ISSUE_ID = f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_{_ENTRY}"


def _issue(hass: HomeAssistant) -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _ISSUE_ID)


@pytest.mark.asyncio
async def test_issue_raised_when_sentinel_on_and_face_recognition_off(
    hass: HomeAssistant,
) -> None:
    """The silent-inert state is exactly Sentinel on + face recognition off."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )

    issue = _issue(hass)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.is_fixable is False
    assert issue.translation_key == ISSUE_UNKNOWN_PERSON_RULES_INERT


@pytest.mark.asyncio
async def test_issue_names_every_inert_rule(hass: HomeAssistant) -> None:
    """A user who cannot read the code needs the rule names in the text."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )

    issue = _issue(hass)
    assert issue is not None
    placeholders = issue.translation_placeholders
    assert placeholders is not None
    rules = placeholders["rules"]
    for name in INERT_WITHOUT_FACE_RECOGNITION:
        assert name in rules, f"issue text omits {name}"
    # Guard against transcribed-from-memory ids: every static id listed must be
    # a real rule_id the engine registers, and every dynamic one a real key of
    # the evaluator dispatch map.
    static_ids = {
        UnknownPersonCameraNoHomeRule.rule_id,
        UnknownPersonAtNightWhileHomeRule.rule_id,
        AlarmDisarmedDuringExternalThreatRule.rule_id,
    }
    for rule_id in static_ids:
        assert rule_id in rules, f"issue text omits real rule_id {rule_id}"
    assert "unknown_person_camera_when_home" in rules


@pytest.mark.asyncio
async def test_no_issue_when_face_recognition_is_operative(
    hass: HomeAssistant,
) -> None:
    """With the pipeline running the rules can fire; nothing to report."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=True,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_no_issue_when_sentinel_is_off(hass: HomeAssistant) -> None:
    """With Sentinel off no rule fires, so singling out five would be noise."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=False,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_issue_cleared_when_face_recognition_is_turned_on(
    hass: HomeAssistant,
) -> None:
    """Enabling the pipeline must retract the notice on the next reload."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is not None

    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=True,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_issue_cleared_when_sentinel_is_turned_off(
    hass: HomeAssistant,
) -> None:
    """Turning Sentinel off also retracts it."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is not None

    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=False,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_check_is_idempotent(hass: HomeAssistant) -> None:
    """Called on every setup and reload, so repeats must not accumulate."""
    for _ in range(3):
        async_check_unknown_person_rules(
            hass,
            _ENTRY,
            sentinel_enabled=True,
            face_recognition_operative=False,
            video_analysis_enabled=True,
        )

    issues = [
        i
        for i in ir.async_get(hass).issues.values()
        if i.domain == DOMAIN and i.issue_id == _ISSUE_ID
    ]
    assert len(issues) == 1


@pytest.mark.asyncio
async def test_clear_on_entry_removal(hass: HomeAssistant) -> None:
    """Nothing re-evaluates the issue after the entry is gone."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is not None

    async_clear_unknown_person_issue(hass, _ENTRY)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_clear_is_safe_when_no_issue_exists(hass: HomeAssistant) -> None:
    """Deleting a missing issue is a registry no-op; the clear path relies on it."""
    async_clear_unknown_person_issue(hass, "never_seen")


@pytest.mark.asyncio
async def test_issue_id_is_per_entry(hass: HomeAssistant) -> None:
    """Two config entries must not clobber each other's issue."""
    async_check_unknown_person_rules(
        hass,
        "entry_one",
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    async_check_unknown_person_rules(
        hass,
        "entry_two",
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )

    registry = ir.async_get(hass)
    assert (
        registry.async_get_issue(
            DOMAIN, f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_entry_one"
        )
        is not None
    )
    assert (
        registry.async_get_issue(
            DOMAIN, f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_entry_two"
        )
        is not None
    )

    # Clearing one leaves the other standing.
    async_clear_unknown_person_issue(hass, "entry_one")
    assert (
        registry.async_get_issue(
            DOMAIN, f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_entry_one"
        )
        is None
    )
    assert (
        registry.async_get_issue(
            DOMAIN, f"{ISSUE_UNKNOWN_PERSON_RULES_INERT}_entry_two"
        )
        is not None
    )


@pytest.mark.asyncio
async def test_issue_raised_when_video_analyzer_is_disabled(
    hass: HomeAssistant,
) -> None:
    """
    Face recognition on + healthy gallery + analyzer off is still inert.

    ``recognize_faces`` only ever runs from the video analyzer, whose
    recommended mode is ``disable``, so this combination reaches the same
    silent-inert state from a different direction -- and the option and the
    gallery both look healthy, which is what made it easy to miss.
    """
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=True,
        video_analysis_enabled=False,
    )
    assert _issue(hass) is not None


@pytest.mark.asyncio
async def test_issue_cleared_when_the_analyzer_is_turned_back_on(
    hass: HomeAssistant,
) -> None:
    """Both halves of the predicate have to hold before the notice retracts."""
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=True,
        video_analysis_enabled=False,
    )
    assert _issue(hass) is not None

    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=True,
        video_analysis_enabled=True,
    )
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_discovery_rule_ids_are_named_as_a_prefix(hass: HomeAssistant) -> None:
    """
    Discovery rule ids are named as a prefix, not a bare template id.

    Approved discovery rules are minted per camera, so a bare template id
    matches nothing the user can search for -- proposal_templates.py builds
    ``unknown_person_camera_when_home_<camera>`` / ``..._any_camera``.
    """
    async_check_unknown_person_rules(
        hass,
        _ENTRY,
        sentinel_enabled=True,
        face_recognition_operative=False,
        video_analysis_enabled=True,
    )
    issue = _issue(hass)
    assert issue is not None
    placeholders = issue.translation_placeholders
    assert placeholders is not None
    assert "starts with unknown_person_camera_when_home" in placeholders["rules"]
