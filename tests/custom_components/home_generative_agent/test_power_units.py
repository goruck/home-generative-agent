# ruff: noqa: S101
"""Tests for sentinel power unit normalization helpers."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.power_units import (
    POWER_UNITS,
    is_power_unit,
    watts_per_unit,
)


def test_watts_per_unit_missing_unit_is_watts() -> None:
    """Bare power sensors report watts by convention."""
    assert watts_per_unit(None) == 1.0
    assert watts_per_unit("") == 1.0


def test_watts_per_unit_canonical_units() -> None:
    """Every canonical HA power unit converts."""
    assert watts_per_unit("W") == 1.0
    assert watts_per_unit("kW") == 1000.0
    assert watts_per_unit("MW") == pytest.approx(1e6)
    assert watts_per_unit("GW") == pytest.approx(1e9)
    assert watts_per_unit("TW") == pytest.approx(1e12)
    assert watts_per_unit("mW") == pytest.approx(0.001)
    assert watts_per_unit("BTU/h") == pytest.approx(0.29307107)


def test_watts_per_unit_case_and_whitespace_variants() -> None:
    """
    Hand-typed unit spellings normalize when unambiguous.

    ESPHome/MQTT/template configs carry free-text units; pre-normalization
    these sensors were compared raw, so skipping them would be a silent
    monitoring regression (issue #461 follow-up).
    """
    assert watts_per_unit("w") == 1.0
    assert watts_per_unit(" W ") == 1.0
    assert watts_per_unit("KW") == 1000.0
    assert watts_per_unit("kw") == 1000.0
    assert watts_per_unit("btu/h") == pytest.approx(0.29307107)


def test_watts_per_unit_milli_mega_must_match_exact_case() -> None:
    """Lowercase "mw" is ambiguous between mW (0.001) and MW (1e6) — never folded."""
    assert watts_per_unit("mw") is None
    assert watts_per_unit("Mw") is None
    assert watts_per_unit("mW") == pytest.approx(0.001)
    assert watts_per_unit("MW") == pytest.approx(1e6)
    # Exact-case with surrounding whitespace must still match — stripping
    # happens before the exact lookup, not only in the ambiguous fold.
    assert watts_per_unit(" MW") == pytest.approx(1e6)
    assert watts_per_unit("mW ") == pytest.approx(0.001)


def test_watts_per_unit_non_power_units_rejected() -> None:
    """Apparent power and other units cannot be compared to watts."""
    assert watts_per_unit("VA") is None
    assert watts_per_unit("kWh") is None
    assert watts_per_unit("°C") is None


def test_watts_per_unit_non_string_values() -> None:
    """Malformed attribute values are coerced, never raise."""
    assert watts_per_unit(0) is None  # falsy but not missing — not watts
    assert watts_per_unit(["W"]) is None  # unhashable — coerced then rejected
    assert watts_per_unit({"u": "W"}) is None


def test_is_power_unit_admission_semantics() -> None:
    """A missing unit alone must not make a sensor a power sensor."""
    assert is_power_unit(None) is False
    assert is_power_unit("") is False
    assert is_power_unit("W") is True
    assert is_power_unit("w") is True
    assert is_power_unit("VA") is False
    # Never raises on non-hashable attribute values (snapshot admission test).
    assert is_power_unit(["W"]) is False
    assert is_power_unit({"unit": "kW"}) is False


def test_power_units_set_matches_ha() -> None:
    """POWER_UNITS mirrors HA's PowerConverter units."""
    assert {"W", "kW", "MW", "GW", "TW", "mW", "BTU/h"} == POWER_UNITS
