"""Normalization of power-sensor readings to watts."""

from __future__ import annotations

import logging
from functools import lru_cache

from homeassistant.const import UnitOfPower
from homeassistant.util.unit_conversion import EnergyConverter, PowerConverter

_LOGGER = logging.getLogger(__name__)

# Unit strings a power sensor may legitimately report in (W, kW, MW, GW, TW,
# mW, BTU/h).  Sensors carrying any of these are treated as power sensors even
# without a power device_class.
POWER_UNITS: frozenset[str] = frozenset(str(u) for u in PowerConverter.VALID_UNITS)

# Unit strings an energy sensor may legitimately report in (Wh, kWh, MWh, J,
# kJ, GJ, cal, ...).  The energy-dimension sibling of POWER_UNITS.
ENERGY_UNITS: frozenset[str] = frozenset(str(u) for u in EnergyConverter.VALID_UNITS)


def _build_folded_units(units: frozenset[str]) -> dict[str, str]:
    """
    Map case-folded unit spellings to their canonical unit string.

    Fallback for hand-typed unit strings ("w", "KW", " W" from ESPHome/MQTT/
    template configs).  Only unambiguous folds are included: "mw" could mean
    either mW (0.001 W) or MW (1e6 W), so milli/mega must match exact-case.
    """
    folded: dict[str, str] = {}
    ambiguous: set[str] = set()
    for u in units:
        key = u.strip().lower()
        if key in folded:
            ambiguous.add(key)
        folded[key] = u
    for key in ambiguous:
        del folded[key]
    return folded


_FOLDED_UNITS: dict[str, str] = _build_folded_units(POWER_UNITS)
_FOLDED_ENERGY_UNITS: dict[str, str] = _build_folded_units(ENERGY_UNITS)


def watts_per_unit(unit: object | None) -> float | None:
    """
    Return the factor converting one native unit of ``unit`` to watts.

    Accepts the raw ``unit_of_measurement`` attribute value: a missing/empty
    unit is treated as watts (bare power sensors report W by convention),
    non-string values are coerced, and unit strings are matched
    case-insensitively where unambiguous ("w", "KW"; but mW/MW must match
    exact-case).  Returns None for units that are not power units — callers
    must skip such readings rather than compare them against a watts
    threshold, which would be silently wrong for e.g. VA (apparent power).
    """
    if unit is None or unit == "":
        return 1.0
    return _watts_per_unit_cached(str(unit))


def is_power_unit(unit: object | None) -> bool:
    """
    Return True when ``unit`` identifies a power sensor on its own.

    Unlike ``watts_per_unit``, a missing/empty unit is False here — a bare
    unit does not make a sensor a power sensor.  Never raises, even for
    non-hashable attribute values (lists/dicts from misbehaving integrations),
    so it is safe as a snapshot admission test.
    """
    if unit is None or unit == "":
        return False
    return _watts_per_unit_cached(str(unit)) is not None


def is_energy_unit(unit: object | None) -> bool:
    """
    Return True when ``unit`` identifies an energy sensor on its own.

    Mirrors ``is_power_unit`` for the energy dimension, including the
    case-folded fallback for hand-typed spellings ("kwh", "WH").  Never
    raises, even for non-hashable attribute values.
    """
    if unit is None:
        return False
    stripped = str(unit).strip()
    if not stripped:
        return False
    return stripped in ENERGY_UNITS or stripped.lower() in _FOLDED_ENERGY_UNITS


@lru_cache
def _watts_per_unit_cached(unit: str) -> float | None:
    # Strip before the exact-case match too: " MW" must not fall through to
    # the folded lookup, where milli/mega is deliberately absent as ambiguous.
    stripped = unit.strip()
    canonical = (
        stripped if stripped in POWER_UNITS else _FOLDED_UNITS.get(stripped.lower())
    )
    if canonical is None:
        # Cached, so this logs once per distinct unit string: the skip is
        # otherwise invisible (rule, enrichment, and baseline all silently
        # ignore sensors whose unit cannot be normalized to watts).
        _LOGGER.debug(
            "Unit %r is not a recognized power unit; power sensors reporting "
            "it are skipped by watts-denominated Sentinel checks",
            unit,
        )
        return None
    return PowerConverter.convert(1.0, canonical, UnitOfPower.WATT)
