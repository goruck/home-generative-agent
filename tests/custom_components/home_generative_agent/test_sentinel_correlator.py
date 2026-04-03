"""Tests for the SentinelCorrelator and CompoundFinding."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from custom_components.home_generative_agent.sentinel.correlator import (
    SentinelCorrelator,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    CompoundFinding,
    Severity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(
    anomaly_id: str = "id1",
    rule_type: str = "unlocked_lock_at_night",
    severity: Severity = "high",
    confidence: float = 0.8,
    triggering_entities: list[str] | None = None,
    area: str | None = "Front",
    is_sensitive: bool = True,
) -> AnomalyFinding:
    """Build a minimal AnomalyFinding for testing."""
    entities = (
        triggering_entities if triggering_entities is not None else ["lock.front"]
    )
    evidence: dict[str, Any] = {"entity_id": entities[0] if entities else "unknown"}
    if area is not None:
        evidence["area"] = area
    return AnomalyFinding(
        anomaly_id=anomaly_id,
        type=rule_type,
        severity=severity,
        confidence=confidence,
        triggering_entities=entities,
        evidence=evidence,
        suggested_actions=["check"],
        is_sensitive=is_sensitive,
    )


# ---------------------------------------------------------------------------
# Same-run grouping tests
# ---------------------------------------------------------------------------


def test_same_area_groups_findings() -> None:
    """Two findings in the same area are grouped into a CompoundFinding."""
    f1 = _finding(anomaly_id="a1", rule_type="unlocked_lock_at_night", area="Front")
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    assert len(compound.constituent_findings) == 2
    assert f1 in compound.constituent_findings
    assert f2 in compound.constituent_findings


def test_shared_entity_groups_findings() -> None:
    """Two findings sharing a triggering entity are grouped."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="unlocked_lock_at_night",
        triggering_entities=["lock.front"],
        area=None,
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="open_entry_while_away",
        triggering_entities=["lock.front"],
        area=None,
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    assert len(result) == 1
    assert isinstance(result[0], CompoundFinding)


def test_complementary_rule_types_group() -> None:
    """Complementary rule types are grouped even without area/entity overlap."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="open_entry_while_away",
        area=None,
        triggering_entities=["binary_sensor.back_door"],
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="unknown_person_camera_no_home",
        area=None,
        triggering_entities=["camera.backyard"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    assert len(result) == 1
    assert isinstance(result[0], CompoundFinding)


def test_unrelated_findings_pass_through() -> None:
    """Findings with no relation are returned as individual AnomalyFindings."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="appliance_power_duration",
        area="Laundry",
        triggering_entities=["sensor.washer_power"],
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="unlocked_lock_at_night",
        area="Garage",
        triggering_entities=["lock.garage_door"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    assert len(result) == 2
    assert all(isinstance(r, AnomalyFinding) for r in result)


def test_empty_list_returns_empty() -> None:
    """An empty findings list returns an empty result."""
    correlator = SentinelCorrelator()
    assert correlator.correlate([]) == []


def test_single_finding_passes_through() -> None:
    """A single finding is returned unchanged as an AnomalyFinding."""
    f = _finding()
    correlator = SentinelCorrelator()
    result = correlator.correlate([f])

    assert len(result) == 1
    assert result[0] is f


def test_three_way_group_via_area() -> None:
    """Three findings in the same area are grouped into one CompoundFinding."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        triggering_entities=["lock.front"],
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )
    f3 = _finding(
        anomaly_id="a3",
        rule_type="open_entry_while_away",
        area="Front",
        triggering_entities=["binary_sensor.front_window"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2, f3])

    assert len(result) == 1
    assert isinstance(result[0], CompoundFinding)
    assert len(result[0].constituent_findings) == 3  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Cross-run isolation
# ---------------------------------------------------------------------------


def test_cross_run_isolation() -> None:
    """Findings from separate correlate() calls are never merged."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        triggering_entities=["lock.front"],
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    # First run: both findings
    result_run1 = correlator.correlate([f1, f2])
    # Second run: only f1
    result_run2 = correlator.correlate([f1])

    # First run: grouped
    assert len(result_run1) == 1
    assert isinstance(result_run1[0], CompoundFinding)

    # Second run: singleton — f2 must NOT appear here
    assert len(result_run2) == 1
    assert isinstance(result_run2[0], AnomalyFinding)
    assert result_run2[0] is f1


def test_correlator_is_stateless_across_calls() -> None:
    """SentinelCorrelator carries no state between calls."""
    f1 = _finding(anomaly_id="a1", area="Front", triggering_entities=["lock.front"])

    correlator = SentinelCorrelator()
    # Call twice with the same single-finding input.
    result1 = correlator.correlate([f1])
    result2 = correlator.correlate([f1])

    assert len(result1) == 1
    assert len(result2) == 1
    assert isinstance(result1[0], AnomalyFinding)
    assert isinstance(result2[0], AnomalyFinding)


# ---------------------------------------------------------------------------
# CompoundFinding shape tests
# ---------------------------------------------------------------------------


def test_compound_finding_has_expected_fields() -> None:
    """CompoundFinding contains all required fields with correct types."""
    f1 = _finding(
        anomaly_id="a1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        confidence=0.9,
        triggering_entities=["lock.front"],
    )
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        confidence=0.6,
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)

    # compound_id is a non-empty string
    assert isinstance(compound.compound_id, str)
    assert compound.compound_id  # non-empty

    # constituent_findings is a tuple of the originals
    assert isinstance(compound.constituent_findings, tuple)
    assert len(compound.constituent_findings) == 2

    # merged_evidence is a dict
    assert isinstance(compound.merged_evidence, dict)
    assert compound.merged_evidence  # non-empty

    # severity: max of constituents (both "high" here)
    assert compound.severity == "high"

    # confidence: mean of 0.9 and 0.6 = 0.75
    assert abs(compound.confidence - 0.75) < 1e-9

    # triggering_entities: union of both
    assert "lock.front" in compound.triggering_entities
    assert "camera.front" in compound.triggering_entities

    # is_sensitive: True because both constituents are sensitive
    assert compound.is_sensitive is True


def test_compound_severity_is_max() -> None:
    """CompoundFinding severity equals the highest among constituents."""
    f_low = _finding(
        anomaly_id="a1",
        rule_type="appliance_power_duration",
        area="Kitchen",
        severity="low",
        triggering_entities=["sensor.oven_power"],
    )
    f_medium = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Kitchen",
        severity="medium",
        triggering_entities=["camera.kitchen"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_low, f_medium])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    assert compound.severity == "medium"


def test_compound_is_sensitive_if_any_constituent_is() -> None:
    """CompoundFinding is sensitive when any constituent is sensitive."""
    f_sensitive = _finding(
        anomaly_id="a1",
        area="Front",
        triggering_entities=["lock.front"],
        is_sensitive=True,
    )
    f_not_sensitive = _finding(
        anomaly_id="a2",
        rule_type="appliance_power_duration",
        area="Front",
        triggering_entities=["sensor.washer"],
        is_sensitive=False,
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_sensitive, f_not_sensitive])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    assert compound.is_sensitive is True


def test_compound_as_dict_serializes_correctly() -> None:
    """CompoundFinding.as_dict() returns all expected keys."""
    f1 = _finding(anomaly_id="a1", area="Front", triggering_entities=["lock.front"])
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    d = compound.as_dict()

    assert "compound_id" in d
    assert "constituent_findings" in d
    assert isinstance(d["constituent_findings"], list)
    assert len(d["constituent_findings"]) == 2
    assert "merged_evidence" in d
    assert "severity" in d
    assert "confidence" in d
    assert "triggering_entities" in d
    assert "is_sensitive" in d


def test_compound_finding_from_findings_raises_on_empty() -> None:
    """CompoundFinding.from_findings() raises ValueError for an empty list."""
    with pytest.raises(ValueError, match="at least one"):
        CompoundFinding.from_findings([])


# ---------------------------------------------------------------------------
# Immutability tests
# ---------------------------------------------------------------------------


def test_compound_finding_is_immutable() -> None:
    """Attempting to set an attribute on CompoundFinding raises FrozenInstanceError."""
    f1 = _finding(anomaly_id="a1", area="Front", triggering_entities=["lock.front"])
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f1, f2])

    compound = result[0]
    assert isinstance(compound, CompoundFinding)

    with pytest.raises(dataclasses.FrozenInstanceError):
        compound.compound_id = "tampered"  # type: ignore[misc]


def test_compound_constituent_tuple_is_immutable() -> None:
    """constituent_findings is a tuple (immutable sequence)."""
    f1 = _finding(anomaly_id="a1", area="Front", triggering_entities=["lock.front"])
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    compound = correlator.correlate([f1, f2])[0]
    assert isinstance(compound, CompoundFinding)

    assert isinstance(compound.constituent_findings, tuple)
    with pytest.raises(TypeError):
        compound.constituent_findings[0] = f2  # type: ignore[index]


def test_compound_triggering_entities_is_immutable() -> None:
    """triggering_entities is a tuple (immutable sequence)."""
    f1 = _finding(anomaly_id="a1", area="Front", triggering_entities=["lock.front"])
    f2 = _finding(
        anomaly_id="a2",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )

    correlator = SentinelCorrelator()
    compound = correlator.correlate([f1, f2])[0]
    assert isinstance(compound, CompoundFinding)

    assert isinstance(compound.triggering_entities, tuple)
    with pytest.raises(TypeError):
        compound.triggering_entities[0] = "hacked"  # type: ignore[index]


def test_anomaly_finding_is_also_immutable() -> None:
    """AnomalyFinding (frozen dataclass) is immutable — regression guard."""
    f = _finding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.type = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Area-aware complementary pair tests (regression guards)
# ---------------------------------------------------------------------------


def test_camera_entry_unsecured_cross_area_not_correlated() -> None:
    """
    camera_entry_unsecured and unlocked_lock_at_night in different areas are NOT correlated.

    Regression guard: ensures the area-aware Rule 3 check is never removed.
    Without it, a camera in 'Front' and a lock in 'Garage' would be merged
    into a compound notification implying a false spatial relationship.
    """
    f_camera = _finding(
        anomaly_id="cam1",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )
    f_lock = _finding(
        anomaly_id="lock1",
        rule_type="unlocked_lock_at_night",
        area="Garage",
        triggering_entities=["lock.garage_door"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_camera, f_lock])

    assert len(result) == 2  # not correlated — different areas
    assert all(isinstance(r, AnomalyFinding) for r in result)


def test_camera_entry_unsecured_not_contaminated_by_transitive_bridge() -> None:
    """
    Three findings must NOT all merge when the camera bridges an off-area entry.

    Scenario: camera_entry_unsecured (Front) + unlocked_lock_at_night (Front)
    + open_entry_while_away (Garage).

    Without the post-grouping eject pass:
      - camera + lock share area 'Front'  → unioned (Rule 1)
      - lock + open_entry are a complementary pair (no area guard) → unioned
      - Result: one compound implying the Front camera saw a Garage entry event

    With the fix:
      - camera + lock are still grouped (shared area Front)
      - open_entry_while_away (Garage) is ejected as a singleton
    """
    f_camera = _finding(
        anomaly_id="cam1",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )
    f_lock = _finding(
        anomaly_id="lock1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        triggering_entities=["lock.front_door"],
    )
    f_away = _finding(
        anomaly_id="away1",
        rule_type="open_entry_while_away",
        area="Garage",
        triggering_entities=["binary_sensor.garage_door"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_camera, f_lock, f_away])

    assert len(result) == 2  # camera+lock compound + Garage singleton

    compound = next((r for r in result if isinstance(r, CompoundFinding)), None)
    singleton = next((r for r in result if isinstance(r, AnomalyFinding)), None)

    assert compound is not None, "Expected a CompoundFinding for Front area"
    assert singleton is not None, "Expected a singleton for Garage area"

    compound_ids = {f.anomaly_id for f in compound.constituent_findings}
    assert "cam1" in compound_ids
    assert "lock1" in compound_ids
    assert "away1" not in compound_ids

    assert singleton.anomaly_id == "away1"


def test_camera_entry_unsecured_same_area_still_correlated() -> None:
    """
    camera_entry_unsecured and unlocked_lock_at_night in the SAME area still correlate.

    Verifies Rule 1 (same area) continues to group findings when spatial
    relationship is valid.  The area-aware Rule 3 change must not break this.
    """
    f_camera = _finding(
        anomaly_id="cam1",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )
    f_lock = _finding(
        anomaly_id="lock1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        triggering_entities=["lock.front_door"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_camera, f_lock])

    assert len(result) == 1
    assert isinstance(result[0], CompoundFinding)


def test_camera_entry_unsecured_three_way_same_area_not_split() -> None:
    """
    Three findings all in the same area must still group into one CompoundFinding.

    Regression guard: the transitive-contamination eject pass must not split
    groups where every finding shares the camera's area.
    """
    f_camera = _finding(
        anomaly_id="cam1",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front"],
    )
    f_lock = _finding(
        anomaly_id="lock1",
        rule_type="unlocked_lock_at_night",
        area="Front",
        triggering_entities=["lock.front_door"],
    )
    f_away = _finding(
        anomaly_id="away1",
        rule_type="open_entry_while_away",
        area="Front",
        triggering_entities=["binary_sensor.front_door"],
    )

    correlator = SentinelCorrelator()
    result = correlator.correlate([f_camera, f_lock, f_away])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    assert len(compound.constituent_findings) == 3


def test_camera_entry_unsecured_multi_camera_different_areas_skips_ejection() -> None:
    """
    Multi-camera group with different areas skips spatial ejection entirely.

    When two camera_entry_unsecured findings have *different* areas in the same
    group, _eject_camera_area_violations cannot determine a single reference area
    and must skip ejection to avoid incorrect splits.

    The group must be returned intact.

    Scenario: two cameras with differing areas (Front, Back) are bridged by a
    shared triggering entity (e.g. a dual-zone sensor) so they land in the same
    union-find group.  A third finding with area=Front would normally be ejected
    if only the Back camera determined the reference area — but since both cameras
    are present with different areas, ejection is skipped entirely.
    """
    shared_sensor = "sensor.dual_zone"

    f_camera_front = _finding(
        anomaly_id="cam_front",
        rule_type="camera_entry_unsecured",
        area="Front",
        triggering_entities=["camera.front", shared_sensor],
    )
    f_camera_back = _finding(
        anomaly_id="cam_back",
        rule_type="camera_entry_unsecured",
        area="Back",
        triggering_entities=["camera.back", shared_sensor],
    )
    # This finding shares an area with the front camera.
    # It would be incorrectly ejected if the back camera's area were chosen as
    # the reference, so the multi-camera guard must prevent any ejection.
    f_front_entry = _finding(
        anomaly_id="entry_front",
        rule_type="open_entry_while_away",
        area="Front",
        triggering_entities=["binary_sensor.front_door"],
    )

    correlator = SentinelCorrelator()
    # f_camera_front and f_camera_back share shared_sensor → same group.
    # f_front_entry shares area "Front" with f_camera_front → same group.
    # All three are in one group; ejection must be skipped (len(camera_areas)==2).
    result = correlator.correlate([f_camera_front, f_camera_back, f_front_entry])

    assert len(result) == 1
    compound = result[0]
    assert isinstance(compound, CompoundFinding)
    assert len(compound.constituent_findings) == 3
