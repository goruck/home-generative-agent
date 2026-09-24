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

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from homeassistant.helpers import issue_registry as ir

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core.face_guard import (
    INERT_WITHOUT_FACE_RECOGNITION,
    ISSUE_UNKNOWN_PERSON_RULES_INERT,
    TRANSLATION_KEY_NO_ANALYZER,
    TRANSLATION_KEY_NO_FACE,
    TRANSLATION_KEY_NO_GALLERY,
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


def _check(  # noqa: PLR0913
    hass: HomeAssistant,
    entry_id: str = _ENTRY,
    *,
    sentinel: bool = True,
    face: bool = True,
    gallery: bool = True,
    analyzer: bool = True,
) -> None:
    """Run the guard with the healthy state as the default, overriding one fact."""
    async_check_unknown_person_rules(
        hass,
        entry_id,
        sentinel_enabled=sentinel,
        face_recognition_configured=face,
        person_gallery_available=gallery,
        video_analysis_enabled=analyzer,
    )


@pytest.mark.asyncio
async def test_issue_raised_when_sentinel_on_and_face_recognition_off(
    hass: HomeAssistant,
) -> None:
    """The silent-inert state is exactly Sentinel on + face recognition off."""
    _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)

    issue = _issue(hass)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.is_fixable is False
    assert issue.translation_key == TRANSLATION_KEY_NO_FACE


@pytest.mark.asyncio
async def test_issue_names_every_inert_rule(hass: HomeAssistant) -> None:
    """A user who cannot read the code needs the rule names in the text."""
    _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)

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
    _check(hass, sentinel=True, face=True, gallery=True, analyzer=True)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_no_issue_when_sentinel_is_off(hass: HomeAssistant) -> None:
    """With Sentinel off no rule fires, so singling out five would be noise."""
    _check(hass, sentinel=False, face=False, gallery=True, analyzer=True)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_issue_cleared_when_face_recognition_is_turned_on(
    hass: HomeAssistant,
) -> None:
    """Enabling the pipeline must retract the notice on the next reload."""
    _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)
    assert _issue(hass) is not None

    _check(hass, sentinel=True, face=True, gallery=True, analyzer=True)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_issue_cleared_when_sentinel_is_turned_off(
    hass: HomeAssistant,
) -> None:
    """Turning Sentinel off also retracts it."""
    _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)
    assert _issue(hass) is not None

    _check(hass, sentinel=False, face=False, gallery=True, analyzer=True)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_check_is_idempotent(hass: HomeAssistant) -> None:
    """Called on every setup and reload, so repeats must not accumulate."""
    for _ in range(3):
        _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)

    issues = [
        i
        for i in ir.async_get(hass).issues.values()
        if i.domain == DOMAIN and i.issue_id == _ISSUE_ID
    ]
    assert len(issues) == 1


@pytest.mark.asyncio
async def test_clear_on_entry_removal(hass: HomeAssistant) -> None:
    """Nothing re-evaluates the issue after the entry is gone."""
    _check(hass, sentinel=True, face=False, gallery=True, analyzer=True)
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
    _check(hass, "entry_one", sentinel=True, face=False, gallery=True, analyzer=True)
    _check(hass, "entry_two", sentinel=True, face=False, gallery=True, analyzer=True)

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


@pytest.mark.parametrize(
    ("face", "gallery", "analyzer", "expected_key"),
    [
        (False, True, True, TRANSLATION_KEY_NO_FACE),
        (True, False, True, TRANSLATION_KEY_NO_GALLERY),
        (True, True, False, TRANSLATION_KEY_NO_ANALYZER),
    ],
)
@pytest.mark.asyncio
async def test_each_cause_selects_its_own_message(
    hass: HomeAssistant,
    face: bool,  # noqa: FBT001
    gallery: bool,  # noqa: FBT001
    analyzer: bool,  # noqa: FBT001
    expected_key: str,
) -> None:
    """
    The notice names the state the install is actually in.

    One generic description covering three causes made the reader diagnose
    themselves, and led with a remedy that did not apply -- seen on a live box
    where the analyzer was the cause but the text opened on "turn face
    recognition on".
    """
    _check(hass, face=face, gallery=gallery, analyzer=analyzer)

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_key == expected_key


@pytest.mark.asyncio
async def test_no_gallery_beats_no_analyzer(hass: HomeAssistant) -> None:
    """
    Precedence is most-fundamental-first, so the remedy is actionable.

    Telling someone to enable the video analyzer when their real problem is a
    missing database would have them fix the wrong thing and see the notice
    again.
    """
    _check(hass, face=True, gallery=False, analyzer=False)

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_key == TRANSLATION_KEY_NO_GALLERY


@pytest.mark.asyncio
async def test_no_face_beats_everything(hass: HomeAssistant) -> None:
    """Face recognition off is the most fundamental cause."""
    _check(hass, face=False, gallery=False, analyzer=False)

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_key == TRANSLATION_KEY_NO_FACE


@pytest.mark.asyncio
async def test_changing_cause_replaces_the_notice_in_place(
    hass: HomeAssistant,
) -> None:
    """One stable issue id, so a new cause must not stack a second warning."""
    _check(hass, face=False)
    _check(hass, face=True, gallery=True, analyzer=False)

    issues = [
        i
        for i in ir.async_get(hass).issues.values()
        if i.domain == DOMAIN and i.issue_id == _ISSUE_ID
    ]
    assert len(issues) == 1
    assert issues[0].translation_key == TRANSLATION_KEY_NO_ANALYZER


@pytest.mark.asyncio
async def test_issue_cleared_when_the_analyzer_is_turned_back_on(
    hass: HomeAssistant,
) -> None:
    """Every fact has to hold before the notice retracts."""
    _check(hass, analyzer=False)
    assert _issue(hass) is not None

    _check(hass)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_discovery_rule_ids_are_named_as_a_prefix(hass: HomeAssistant) -> None:
    """
    Discovery rule ids are named as a prefix, not a bare template id.

    Approved discovery rules are minted per camera, so a bare template id
    matches nothing the user can search for -- proposal_templates.py builds
    ``unknown_person_camera_when_home_<camera>`` / ``..._any_camera``.
    """
    _check(hass, face=False)

    issue = _issue(hass)
    assert issue is not None
    placeholders = issue.translation_placeholders
    assert placeholders is not None
    assert "starts with unknown_person_camera_when_home" in placeholders["rules"]


def test_no_variant_claims_baseline_is_both_off_and_unaffected() -> None:
    """
    No notice may promise detection that the state it describes has disabled.

    This exact claim has now been wrong three times. First the single
    description said baseline anomalies were "unaffected" while the notice also
    fired when there was no database -- and baseline needs the pool, so on the
    very install being warned about it promised detection that was off (Codex
    caught it). Adding a caveat fixed that. Splitting one description into three
    then moved the caveat into the no-gallery *cause* and left the unqualified
    claim in the shared tail, so that one notice said both. The claim is gone
    from the shared text; this keeps it gone.
    """
    for name in ("strings.json", "translations/en.json", "translations/cs.json"):
        path = (
            Path(__file__).parents[3]
            / "custom_components"
            / "home_generative_agent"
            / name
        )
        issues = json.loads(path.read_text(encoding="utf-8"))["issues"]
        for key, payload in issues.items():
            if not key.startswith("unknown_person_rules_inert"):
                continue
            description = payload["description"]
            says_off = (
                "Baseline anomaly detection is off" in description
                or "je vypnutá i detekce anomálií" in description
            )
            says_fine = (
                "baseline anomalies and the" in description
                or "anomálie vůči základní úrovni i audit" in description
            )
            assert not (says_off and says_fine), (
                f"{name}:{key} says baseline is both off and unaffected"
            )
            # The no-gallery notice fires precisely when the pool is absent, so
            # it must never be the one making the unqualified claim.
            if key.endswith("no_gallery"):
                assert not says_fine, (
                    f"{name}:{key} promises baseline detection, but this notice "
                    "fires when there is no database and baseline needs the pool"
                )
