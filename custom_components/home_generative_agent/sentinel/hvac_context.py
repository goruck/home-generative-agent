"""
Heating and cooling context for Sentinel's power findings (#922).

A heat pump drawing a lot of power on a cold day, with the thermostat calling
for heat, is expected. A power finding on a heating or cooling load is
explained while a climate entity it belongs to is actively heating or
cooling: its evidence carries ``hvac_context`` for the alert text, triage and
the explanation. Nothing else about the finding changes: its severity,
cooldowns and pending prompt stay as they are, and nothing here drops it, so
a unit faulting while it heats still alerts.

A power sensor is a heating or cooling load when it is on the same device as
a climate entity (then only that device's climate entities count), or when
its name says so (heat pump, HVAC, furnace, boiler, air conditioner, aircon,
mini split, air handler; then any climate entity counts, for the direction
the equipment can run: an air conditioner is explained by cooling only, a
furnace or boiler by heating only). On either path a load named for water
heating (water heater, hot water, immersion, DHW), refrigeration (fridge,
freezer, wine cooler) or a pool, spa, hot tub or dryer is never explained,
and neither does a climate entity named for those explain anything. A
demand counts only for a room-like target in the home's unit (heating 5 to 30
°C, cooling 10 to 30 °C, or the °F equivalents): a fridge compartment, tank,
sauna or car cabin exposed as a climate entity is not a room. Area is
deliberately not used: a freezer in the room with a heater must not be
explained by the heater.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .power_units import is_power_unit

if TYPE_CHECKING:
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

    from .models import AnomalyFinding

# Findings that report power drawn above normal or above a threshold.
_APPLIANCE_RULE = "appliance_power_duration"
_DEVIATION_TEMPLATES = frozenset({"baseline_deviation", "time_of_day_anomaly"})

# Name hints for a heating or cooling load, matched as whole words (or
# word pairs) in the entity id and friendly name. "ac" alone is left out:
# it matches too much ("vac", "tac", "ac adapter").
_HVAC_NAME_RE = re.compile(
    r"\b(?:heat ?pumps?|hvacs?|furnaces?|boilers?|air ?condition(?:ers?|ing)?"
    r"|aircons?|mini ?splits?|minisplits?|air ?handlers?)\b"
)
# Loads that run on their own schedule, whatever the room thermostats do: a
# stuck tank heater, a failing freezer compressor or a pool heat pump must
# never read as "likely from heating". Climate entities named for them
# explain nothing either.
_NOT_SPACE_HVAC_RE = re.compile(
    r"\b(?:water ?heaters?|hot ?water|immersion|dhw|fridges?|refrigerators?"
    r"|freezers?|wine ?coolers?|pools?|spas?|hot ?tubs?|jacuzzis?|dryers?"
    r"|tumble|saunas?|terrariums?|tanks?|cylinders?)\b"
)
# Name hints that only run one way.
_COOLING_ONLY_RE = re.compile(r"\b(?:air ?condition(?:ers?|ing)?|aircons?)\b")
_HEATING_ONLY_RE = re.compile(r"\b(?:furnaces?|boilers?)\b")
# Room-like targets, in the home's temperature unit: a fridge compartment,
# a hot-water tank, a sauna or a car cabin exposed as a climate entity sits
# outside these.
_ROOM_COOL_TARGET = {"°C": (10.0, 30.0), "°F": (50.0, 86.0)}
_ROOM_HEAT_TARGET = {"°C": (5.0, 30.0), "°F": (40.0, 86.0)}

HEATING = "heating"
COOLING = "cooling"
HEATING_AND_COOLING = "heating_and_cooling"
_HEATING_ACTIONS = frozenset({"heating", "preheating", "defrosting"})
_COOLING_ACTIONS = frozenset({"cooling"})
# A climate entity in one of these states explains nothing, whatever its
# hvac_action attribute still says.
_INACTIVE_STATES = frozenset({"off", "unavailable", "unknown", ""})
# Climate entities whose hvac_action is not reported: their mode and the
# gap between the room and the target say whether they are calling.
_HEAT_MODES = frozenset({"heat", "heat_cool", "auto"})
_COOL_MODES = frozenset({"cool", "heat_cool", "auto"})
# How far (in the climate's own unit) the room must be from its target for
# a climate with no hvac_action to count as calling.
_DEADBAND = 0.5
# How many climate entities the context lists.
_MAX_CLIMATES = 3
# Outdoor readings outside this range (°C) are taken as bad data.
_OUTDOOR_MIN_C = -90.0
_OUTDOOR_MAX_C = 60.0


def _name_texts(entity_id: str, friendly_name: str | None) -> list[str]:
    return [
        re.sub(r"[_\-]+", " ", text.lower())
        for text in (entity_id.split(".", 1)[-1], friendly_name or "")
    ]


def is_not_space_hvac(entity_id: str, friendly_name: str | None) -> bool:
    """Return True for water heating, refrigeration, pool, spa or dryer names."""
    return any(
        _NOT_SPACE_HVAC_RE.search(text)
        for text in _name_texts(entity_id, friendly_name)
    )


def is_hvac_name(entity_id: str, friendly_name: str | None) -> bool:
    """Return True when an entity's id or name names a heating or cooling load."""
    if is_not_space_hvac(entity_id, friendly_name):
        return False
    return any(
        _HVAC_NAME_RE.search(text) for text in _name_texts(entity_id, friendly_name)
    )


def name_demands(entity_id: str, friendly_name: str | None) -> frozenset[str]:
    """Return the demands an HVAC-named load can be explained by."""
    texts = _name_texts(entity_id, friendly_name)
    names_both = any(
        _HVAC_NAME_RE.search(_COOLING_ONLY_RE.sub(" ", _HEATING_ONLY_RE.sub(" ", text)))
        for text in texts
    )
    if names_both:
        return frozenset({HEATING, COOLING})
    demands: set[str] = set()
    if any(_COOLING_ONLY_RE.search(text) for text in texts):
        demands.add(COOLING)
    if any(_HEATING_ONLY_RE.search(text) for text in texts):
        demands.add(HEATING)
    return frozenset(demands)


def _room_like(target: float | None, demand: str, unit: str) -> bool:
    """Return True when a target is a room's, not a fridge's, tank's or sauna's."""
    if target is None:
        return True
    bands = _ROOM_COOL_TARGET if demand == COOLING else _ROOM_HEAT_TARGET
    low, high = bands.get(unit, bands["°C"])
    return low <= target <= high


def _finite(value: Any) -> float | None:
    """Return *value* as a finite float, or None for anything else."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def climate_demand(entity: SnapshotEntity, unit: str = "°C") -> str | None:
    """
    Return ``heating`` or ``cooling`` while a climate entity is calling.

    An off or unavailable climate entity never calls. One whose room is
    already at or past its target is not calling either, whatever its
    hvac_action says: a generic thermostat reports "heating" because its
    switch draws power, so a heater stuck on in a hot room must not be
    explained by itself. A target outside a room's range (in *unit*, the
    home's temperature unit) is not a room: a fridge compartment, a tank or
    a sauna explains nothing.
    """
    if str(entity.get("state") or "") in _INACTIVE_STATES:
        return None
    if is_not_space_hvac(entity.get("entity_id", ""), entity.get("friendly_name")):
        return None
    attrs = entity.get("attributes") or {}
    current = _finite(attrs.get("current_temperature"))
    target = _finite(attrs.get("temperature"))
    heat_to = target if target is not None else _finite(attrs.get("target_temp_low"))
    cool_to = target if target is not None else _finite(attrs.get("target_temp_high"))
    action = attrs.get("hvac_action")
    if isinstance(action, str) and action:
        demand = (
            HEATING
            if action in _HEATING_ACTIONS
            else COOLING
            if action in _COOLING_ACTIONS
            else None
        )
        if current is not None and (
            (demand == HEATING and heat_to is not None and current >= heat_to)
            or (demand == COOLING and cool_to is not None and current <= cool_to)
        ):
            demand = None
        if demand is not None and not _room_like(
            cool_to if demand == COOLING else heat_to, demand, unit
        ):
            demand = None
        return demand
    # No hvac_action reported: infer from the mode and the room temperature.
    # A thermostat that lists its modes must offer the direction: "auto" on
    # a heat-only valve is a schedule, not cooling.
    mode = str(entity.get("state") or "")
    modes = attrs.get("hvac_modes")
    offered = {str(m) for m in modes} if isinstance(modes, list) else None
    can_heat = offered is None or bool(offered & {"heat", "heat_cool"})
    can_cool = offered is None or bool(offered & {"cool", "heat_cool"})
    demand = None
    if current is not None:
        if (
            mode in _HEAT_MODES
            and can_heat
            and heat_to is not None
            and current <= heat_to - _DEADBAND
            and _room_like(heat_to, HEATING, unit)
        ):
            demand = HEATING
        elif (
            mode in _COOL_MODES
            and can_cool
            and cool_to is not None
            and current >= cool_to + _DEADBAND
            and _room_like(cool_to, COOLING, unit)
        ):
            demand = COOLING
    return demand


def _celsius(value: float, unit: str) -> float | None:
    """Return *value* in °C from a °C, °F or K reading, or None."""
    normalized = unit.strip().lower().replace("°", "").replace("º", "")
    if normalized in {"c", "celsius"}:
        return value
    if normalized in {"f", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    if normalized in {"k", "kelvin"}:
        return value - 273.15
    return None


def outdoor_temperature(
    snapshot: FullStateSnapshot, entity_id: str | None
) -> tuple[float, str] | None:
    """
    Return ``(temperature, unit)`` outdoors, or None when not known.

    The unit is ``°C`` or ``°F`` (a Kelvin reading is given in °C); a reading
    in any other unit, or outside the range a weather reading can have, is
    not used. *entity_id* is the configured weather entity or temperature
    sensor (one with no reading gives none); when it is empty, the first
    weather entity with a reading is used.
    """
    entities = {e["entity_id"]: e for e in snapshot.get("entities", [])}

    def _reading(entity: SnapshotEntity) -> tuple[float, str] | None:
        # An unavailable entity keeps its last attributes: not a reading.
        if str(entity.get("state") or "") in {"unavailable", "unknown", ""}:
            return None
        attrs = entity.get("attributes") or {}
        if entity.get("domain") == "weather":
            value = _finite(attrs.get("temperature"))
            unit = str(attrs.get("temperature_unit") or "")
        else:
            value = _finite(entity.get("state"))
            unit = str(attrs.get("unit_of_measurement") or "")
        if value is None:
            return None
        celsius = _celsius(value, unit)
        if celsius is None or not _OUTDOOR_MIN_C <= celsius <= _OUTDOOR_MAX_C:
            return None
        if "f" in unit.lower():
            return value, "°F"
        return celsius, "°C"

    if entity_id:
        # A configured entity with no reading gives none: falling back could
        # show another place's temperature.
        configured = entities.get(entity_id)
        return _reading(configured) if configured is not None else None
    for weather_id in sorted(entities):
        if entities[weather_id].get("domain") == "weather" and (
            reading := _reading(entities[weather_id])
        ):
            return reading
    return None


def _power_finding_entity(finding: AnomalyFinding) -> str | None:
    """Return the power entity of a power-above finding, or None."""
    evidence = finding.evidence or {}
    if finding.type == _APPLIANCE_RULE:
        return str(evidence.get("entity_id") or "") or None
    if evidence.get("template_id") not in _DEVIATION_TEMPLATES:
        return None
    if evidence.get("deviation_direction") != "above":
        return None
    unit = str(evidence.get("unit_of_measurement") or "")
    if evidence.get("device_class") != "power" and not is_power_unit(unit):
        return None
    return str(evidence.get("entity_id") or "") or None


def _climate_summary(entity: SnapshotEntity, demand: str) -> dict[str, Any]:
    attrs = entity.get("attributes") or {}
    summary: dict[str, Any] = {
        "name": entity.get("friendly_name") or entity["entity_id"],
        "demand": demand,
    }
    for key in ("current_temperature", "temperature"):
        if (value := _finite(attrs.get(key))) is not None:
            summary[key] = value
    return summary


def apply_hvac_context(
    findings: Iterable[AnomalyFinding],
    snapshot: FullStateSnapshot,
    *,
    device_climates: Callable[[str], Iterable[str]],
    outdoor_entity_id: str | None = None,
    temperature_unit: str = "°C",
) -> list[AnomalyFinding]:
    """
    Return *findings* with heating or cooling context added where it applies.

    *device_climates* returns the climate entity ids on the same device as
    an entity. A load with any is explained only by those (one missing from
    the snapshot explains nothing); a load with none, matched by name, by
    any climate entity. An explained finding gets ``hvac_context`` in its
    evidence and is otherwise unchanged; other findings come back as they
    were.
    """
    result = list(findings)
    if not result:
        return result
    entities = {e["entity_id"]: e for e in snapshot.get("entities", [])}
    climates = {
        eid: entity for eid, entity in entities.items() if entity["domain"] == "climate"
    }
    if not climates:
        return result
    outdoor: tuple[float, str] | None = None
    outdoor_read = False

    for index, finding in enumerate(result):
        power_id = _power_finding_entity(finding)
        if power_id is None:
            continue
        power = entities.get(power_id)
        name = (power.get("friendly_name") if power else None) or str(
            finding.evidence.get("friendly_name") or ""
        )
        if is_not_space_hvac(power_id, name):
            continue
        on_device = list(device_climates(power_id))
        allowed: frozenset[str] = frozenset({HEATING, COOLING})
        if on_device:
            linked = [eid for eid in on_device if eid in climates]
        else:
            if not is_hvac_name(power_id, name):
                continue
            linked = sorted(climates)
            allowed = name_demands(power_id, name)
        calling = [
            (eid, demand)
            for eid in linked
            if (demand := climate_demand(climates[eid], temperature_unit)) is not None
            and demand in allowed
        ]
        if not calling:
            continue
        if not outdoor_read:
            outdoor = outdoor_temperature(snapshot, outdoor_entity_id)
            outdoor_read = True
        demands = sorted({demand for _eid, demand in calling})
        context: dict[str, Any] = {
            "explained_by": demands[0] if len(demands) == 1 else HEATING_AND_COOLING,
            "climate": [
                _climate_summary(climates[eid], demand)
                for eid, demand in calling[:_MAX_CLIMATES]
            ],
        }
        if outdoor is not None:
            context["outdoor_temperature"] = round(outdoor[0], 1)
            context["outdoor_temperature_unit"] = outdoor[1]
        result[index] = replace(
            finding, evidence={**finding.evidence, "hvac_context": context}
        )
    return result


def hvac_explained_by(evidence: Mapping[str, Any]) -> str | None:
    """Return what explains a finding's power draw, from its evidence."""
    context = evidence.get("hvac_context")
    if not isinstance(context, Mapping):
        return None
    explained = str(context.get("explained_by") or "")
    return explained if explained in {HEATING, COOLING, HEATING_AND_COOLING} else None


def outdoor_reading(evidence: Mapping[str, Any]) -> tuple[int, str] | None:
    """Return the rounded outdoor temperature and its unit (°C or °F), or None."""
    context = evidence.get("hvac_context")
    if not isinstance(context, Mapping):
        return None
    value = _finite(context.get("outdoor_temperature"))
    unit = context.get("outdoor_temperature_unit")
    if value is None or unit not in {"°C", "°F"}:
        return None
    return round(value), str(unit)
