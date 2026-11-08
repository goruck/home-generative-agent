# ruff: noqa: S101
"""Heating and cooling explain power findings, never hide them (#922)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant.common import MockConfigEntry

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_OUTDOOR_TEMPERATURE_ENTITY,
)
from custom_components.home_generative_agent.sentinel.hvac_context import (
    apply_hvac_context,
    climate_demand,
    hvac_explained_by,
    is_hvac_name,
    name_demands,
    outdoor_temperature,
)
from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
from custom_components.home_generative_agent.sentinel.notifier import (
    _appliance_power_duration_mobile_message,
    _baseline_deviation_mobile_message,
    _mobile_message,
    _persistent_message,
)
from custom_components.home_generative_agent.sentinel.triage import _build_prompt

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _entity(
    entity_id: str,
    state: str,
    attributes: dict[str, Any] | None = None,
    friendly_name: str | None = None,
) -> Any:
    return {
        "entity_id": entity_id,
        "domain": entity_id.split(".", 1)[0],
        "state": state,
        "friendly_name": friendly_name,
        "area": None,
        "attributes": attributes or {},
        "last_changed": "2026-11-07T00:00:00+00:00",
        "last_updated": "2026-11-07T00:00:00+00:00",
    }


def _snapshot(*entities: dict[str, Any]) -> Any:
    return {"entities": list(entities), "derived": {}}


_HEATING = _entity(
    "climate.living_room",
    "heat",
    {"hvac_action": "heating", "current_temperature": 19.0, "temperature": 21.0},
    "Living room",
)
_IDLE = _entity(
    "climate.living_room",
    "heat",
    {"hvac_action": "idle", "current_temperature": 21.5, "temperature": 21.0},
    "Living room",
)
_WEATHER = _entity(
    "weather.home", "cloudy", {"temperature": 2.4, "temperature_unit": "°C"}
)


def _appliance(entity_id: str = "sensor.heat_pump_power", **extra: Any) -> Any:
    return AnomalyFinding(
        anomaly_id="id-1",
        type="appliance_power_duration",
        severity="medium",
        confidence=0.6,
        triggering_entities=[entity_id],
        evidence={
            "entity_id": entity_id,
            "power_w": 3100.0,
            "duration_min": 95,
            "threshold_min": 60,
            "friendly_name": extra.pop("friendly_name", None),
            **extra,
        },
        suggested_actions=["check_appliance"],
        is_sensitive=False,
    )


def _deviation(
    direction: str = "above", device_class: str = "power", unit: str = "W"
) -> Any:
    return AnomalyFinding(
        anomaly_id="id-2",
        type="baseline_deviation",
        severity="medium",
        confidence=0.7,
        triggering_entities=["sensor.heat_pump_power"],
        evidence={
            "template_id": "baseline_deviation",
            "entity_id": "sensor.heat_pump_power",
            "friendly_name": "Heat pump power",
            "device_class": device_class,
            "unit_of_measurement": unit,
            "current_value": 3000.0,
            "baseline_value": 900.0,
            "deviation_pct": 233.3,
            "deviation_direction": direction,
        },
        suggested_actions=[],
        is_sensitive=False,
    )


def _no_device(_entity_id: str) -> list[str]:
    return []


# ---------------------------------------------------------------------------
# What counts as a heating or cooling load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entity_id", "name", "expected"),
    [
        ("sensor.heat_pump_power", None, True),
        ("sensor.heatpump_power", None, True),
        ("sensor.plug_3", "Heat Pump", True),
        ("sensor.hvac_power", None, True),
        ("sensor.furnace_watts", None, True),
        ("sensor.boiler", None, True),
        ("sensor.air_conditioner_power", None, True),
        ("sensor.plug", "Air conditioning", True),
        ("sensor.mini_split_power", None, True),
        ("sensor.fridge_power", None, False),
        ("sensor.freezer_power", None, False),
        ("sensor.server_rack_power", None, False),
        ("sensor.vacuum_power", None, False),
        ("sensor.washer_power", None, False),
        ("sensor.ac_adapter", None, False),
        ("sensor.heat_lamp", None, False),
        ("sensor.aircon", None, True),
        ("sensor.air_handler_power", None, True),
        # Water heating is never a heating or cooling load.
        ("sensor.heat_pump_water_heater_power", None, False),
        ("sensor.plug", "Boiler hot water", False),
        ("sensor.immersion_boiler", None, False),
        ("sensor.dhw_heat_pump", None, False),
        ("sensor.heat_pump_dryer", None, False),
        ("sensor.pool_heat_pump", None, False),
        ("sensor.spa_heat_pump", None, False),
        ("sensor.space_heat_pump", None, True),
        ("sensor.freezer_hvac", None, False),
        ("sensor.freezers_heat_pump", None, False),
        ("sensor.hotwater_heatpump_power", None, False),
        ("sensor.buffer_tank_heat_pump", None, False),
        ("sensor.sauna_boiler", None, False),
        ("sensor.furnaces", None, True),
        ("sensor.air_conditioners", None, True),
    ],
)
def test_is_hvac_name(entity_id: str, name: str | None, *, expected: bool) -> None:
    assert is_hvac_name(entity_id, name) is expected


@pytest.mark.parametrize(
    ("state", "attributes", "expected"),
    [
        ("heat", {"hvac_action": "heating"}, "heating"),
        # Off or unavailable: a stale hvac_action explains nothing.
        ("off", {"hvac_action": "heating"}, None),
        ("unavailable", {"hvac_action": "heating"}, None),
        ("unknown", {"hvac_action": "cooling"}, None),
        # A thermostat "heating" because its own switch draws power, in a
        # room already past its target, is not calling.
        (
            "heat",
            {"hvac_action": "heating", "current_temperature": 30, "temperature": 21},
            None,
        ),
        (
            "heat",
            {"hvac_action": "heating", "current_temperature": 19, "temperature": 21},
            "heating",
        ),
        (
            "cool",
            {"hvac_action": "cooling", "current_temperature": 20, "temperature": 24},
            None,
        ),
        ("heat", {"hvac_action": "preheating"}, "heating"),
        ("heat", {"hvac_action": "defrosting"}, "heating"),
        ("cool", {"hvac_action": "cooling"}, "cooling"),
        ("heat", {"hvac_action": "idle"}, None),
        ("off", {"hvac_action": "off"}, None),
        ("cool", {"hvac_action": "fan"}, None),
        # No hvac_action: the mode and the gap to the target decide.
        ("heat", {"current_temperature": 18, "temperature": 21}, "heating"),
        # Within the 0.5° deadband: not calling.
        ("heat", {"current_temperature": 20.8, "temperature": 21}, None),
        ("heat", {"current_temperature": 10**400, "temperature": 21}, None),
        ("heat", {"current_temperature": 22, "temperature": 21}, None),
        ("cool", {"current_temperature": 27, "temperature": 24}, "cooling"),
        (
            "heat_cool",
            {"current_temperature": 17, "target_temp_low": 19, "target_temp_high": 25},
            "heating",
        ),
        (
            "heat_cool",
            {"current_temperature": 21, "target_temp_low": 19, "target_temp_high": 25},
            None,
        ),
        ("off", {"current_temperature": 15, "temperature": 21}, None),
        ("heat", {"temperature": 21}, None),
        ("heat", {"current_temperature": "nan", "temperature": 21}, None),
    ],
)
def test_climate_demand(
    state: str, attributes: dict[str, Any], expected: str | None
) -> None:
    assert climate_demand(_entity("climate.x", state, attributes)) == expected


# ---------------------------------------------------------------------------
# Outdoor temperature
# ---------------------------------------------------------------------------


def test_outdoor_temperature_defaults_to_the_first_weather_entity() -> None:
    later = _entity(
        "weather.zz", "sunny", {"temperature": 30, "temperature_unit": "°C"}
    )
    assert outdoor_temperature(_snapshot(later, _WEATHER), None) == (2.4, "°C")


def test_a_configured_sensor_wins() -> None:
    sensor = _entity("sensor.outside", "-3.5", {"unit_of_measurement": "°C"})
    snapshot = _snapshot(_WEATHER, sensor)
    assert outdoor_temperature(snapshot, "sensor.outside") == (-3.5, "°C")


def test_a_configured_entity_without_a_reading_gives_none() -> None:
    """No fallback: another weather entity may be another place."""
    sensor = _entity("sensor.outside", "unavailable", {"unit_of_measurement": "°C"})
    snapshot = _snapshot(_WEATHER, sensor)
    assert outdoor_temperature(snapshot, "sensor.outside") is None
    assert outdoor_temperature(snapshot, "sensor.missing") is None


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (35.0, "°F", (35.0, "°F")),
        (35.0, "F", (35.0, "°F")),
        (275.15, "K", (2.0, "°C")),
        (5.0, "C", (5.0, "°C")),
        (5.0, "", None),  # a unit that cannot be read
        (5.0, "%", None),
        (1e300, "°C", None),  # out of range
        (10**400, "°C", None),  # too large for a float
        (-120.0, "°C", None),
    ],
)
def test_outdoor_readings_are_checked(
    value: Any, unit: str, expected: tuple[float, str] | None
) -> None:
    sensor = _entity("sensor.outside", str(value), {"unit_of_measurement": unit})
    reading = outdoor_temperature(_snapshot(sensor), "sensor.outside")
    if expected is None:
        assert reading is None
    else:
        assert reading is not None
        assert reading[1] == expected[1]
        assert reading[0] == pytest.approx(expected[0])


def test_no_weather_entity_means_no_outdoor_temperature() -> None:
    assert outdoor_temperature(_snapshot(_HEATING), None) is None


# ---------------------------------------------------------------------------
# apply_hvac_context
# ---------------------------------------------------------------------------


def test_a_heat_pump_drawing_while_heating_is_explained() -> None:
    finding = _appliance()
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING, _WEATHER), device_climates=_no_device
    )

    # Severity, and with it cooldowns and quiet hours, is left alone.
    assert result.severity == "medium"
    assert result.anomaly_id == finding.anomaly_id
    context = result.evidence["hvac_context"]
    assert context["explained_by"] == "heating"
    assert context["outdoor_temperature"] == 2.4
    assert context["climate"] == [
        {
            "name": "Living room",
            "demand": "heating",
            "current_temperature": 19.0,
            "temperature": 21.0,
        }
    ]
    # The original is not modified.
    assert "hvac_context" not in finding.evidence
    assert finding.severity == "medium"


def test_an_idle_thermostat_explains_nothing() -> None:
    finding = _appliance()
    (result,) = apply_hvac_context(
        [finding], _snapshot(_IDLE, _WEATHER), device_climates=_no_device
    )
    assert result is finding


def test_a_fridge_is_never_explained_by_the_heating() -> None:
    finding = _appliance("sensor.fridge_power")
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING), device_climates=_no_device
    )
    assert result is finding


def test_a_load_on_the_climate_device_is_explained_whatever_its_name() -> None:
    """A generically named plug on the thermostat's device counts."""
    finding = _appliance("sensor.plug_3_power")
    office = _entity("climate.office", "cool", {"hvac_action": "cooling"}, "Office")

    (result,) = apply_hvac_context(
        [finding],
        _snapshot(_IDLE, office),
        device_climates=lambda eid: (
            ["climate.office"] if eid.startswith("sensor.plug_3") else []
        ),
    )

    assert result.evidence["hvac_context"]["explained_by"] == "cooling"
    assert [c["name"] for c in result.evidence["hvac_context"]["climate"]] == ["Office"]


def test_a_device_linked_load_follows_its_own_thermostat_only() -> None:
    """The load's own thermostat is idle: another room heating explains nothing."""
    finding = _appliance("sensor.heat_pump_power")
    bedroom = _entity("climate.bedroom", "heat", {"hvac_action": "heating"}, "Bedroom")

    (result,) = apply_hvac_context(
        [finding],
        _snapshot(_IDLE, bedroom),
        device_climates=lambda _eid: ["climate.living_room"],
    )

    assert result is finding


def test_a_missing_device_thermostat_does_not_fall_back_to_others() -> None:
    """The load's own thermostat is gone from the snapshot: nothing explains it."""
    bedroom = _entity("climate.bedroom", "heat", {"hvac_action": "heating"}, "Bedroom")
    finding = _appliance()

    (result,) = apply_hvac_context(
        [finding],
        _snapshot(bedroom),
        device_climates=lambda _eid: ["climate.heat_pump"],
    )

    assert result is finding


def test_a_deviation_above_normal_is_explained() -> None:
    (result,) = apply_hvac_context(
        [_deviation()], _snapshot(_HEATING), device_climates=_no_device
    )
    assert result.severity == "medium"
    assert hvac_explained_by(result.evidence) == "heating"


@pytest.mark.parametrize(
    ("direction", "device_class", "unit"),
    [("below", "power", "W"), ("above", "temperature", "°C")],
)
def test_other_deviations_are_left_alone(
    direction: str, device_class: str, unit: str
) -> None:
    finding = _deviation(direction, device_class, unit)
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING), device_climates=_no_device
    )
    assert result is finding


def test_other_findings_are_left_alone() -> None:
    finding = AnomalyFinding(
        anomaly_id="x",
        type="open_entry_while_away",
        severity="high",
        confidence=0.9,
        triggering_entities=["binary_sensor.heat_pump_door"],
        evidence={"entity_id": "binary_sensor.heat_pump_door"},
        suggested_actions=[],
        is_sensitive=False,
    )
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING), device_climates=_no_device
    )
    assert result is finding


def test_heating_and_cooling_together() -> None:
    cooling = _entity("climate.server", "cool", {"hvac_action": "cooling"}, "Server")
    (result,) = apply_hvac_context(
        [_appliance()], _snapshot(_HEATING, cooling), device_climates=_no_device
    )
    assert result.evidence["hvac_context"]["explained_by"] == "heating_and_cooling"


# ---------------------------------------------------------------------------
# Alert text and triage
# ---------------------------------------------------------------------------


def _explained(finding: Any, *entities: dict[str, Any]) -> Any:
    (result,) = apply_hvac_context(
        [finding], _snapshot(*entities), device_climates=_no_device
    )
    return result


def test_the_appliance_alert_says_why() -> None:
    message = _appliance_power_duration_mobile_message(
        _explained(_appliance(friendly_name="Heat pump power"), _HEATING, _WEATHER)
    )
    assert message.endswith("Likely from heating, 2 °C outside.")
    assert "Check it" not in message


def test_the_deviation_alert_replaces_its_call_to_action() -> None:
    message = _baseline_deviation_mobile_message(_explained(_deviation(), _HEATING))
    assert message.endswith("Likely from heating.")
    assert "Check appliance" not in message


def test_an_unexplained_alert_is_unchanged() -> None:
    message = _appliance_power_duration_mobile_message(_appliance())
    assert message.endswith("Check it.")


def test_an_unreadable_outdoor_unit_leaves_the_temperature_out() -> None:
    weather = _entity(
        "weather.home",
        "cloudy",
        {"temperature": 2, "temperature_unit": "°C\u202e and then some"},
    )
    message = _appliance_power_duration_mobile_message(
        _explained(_appliance(), _HEATING, weather)
    )
    assert "\u202e" not in message
    assert message.endswith("Likely from heating.")


def test_triage_sees_the_demand_and_temperature_but_no_names() -> None:
    finding = _explained(_appliance(), _HEATING, _WEATHER)
    prompt = _build_prompt(finding, _snapshot(_HEATING, _WEATHER))
    assert "hvac_explained_by: heating" in prompt
    assert "outdoor_temperature_C: 2" in prompt
    # The size of the draw goes with it, so a fault can still be told apart.
    assert "duration_min: 95" in prompt
    assert "threshold_min: 60" in prompt
    assert "Living room" not in prompt


def test_triage_gets_fahrenheit_as_fahrenheit() -> None:
    weather = _entity(
        "weather.home", "cloudy", {"temperature": 35, "temperature_unit": "°F"}
    )
    finding = _explained(_appliance(), _HEATING, weather)
    prompt = _build_prompt(finding, _snapshot(_HEATING, weather))
    assert "outdoor_temperature_F: 35" in prompt
    assert "outdoor_temperature_C" not in prompt


def test_triage_gets_the_deviation_size() -> None:
    finding = _explained(_deviation(), _HEATING)
    prompt = _build_prompt(finding, _snapshot(_HEATING))
    assert "deviation_pct: 233" in prompt


def test_an_unexplained_finding_gets_no_hvac_lines() -> None:
    prompt = _build_prompt(_appliance(), _snapshot(_IDLE))
    assert "hvac_explained_by" not in prompt
    assert "duration_min" not in prompt


def test_the_record_of_an_explained_deviation_says_why() -> None:
    message = _persistent_message(None, _explained(_deviation(), _HEATING))
    assert message.endswith("Likely from heating.")


@pytest.mark.parametrize(
    ("language", "with_outdoor", "expected_end"),
    [
        ("cs", True, "Nejspíš kvůli vytápění, venku 2 °C."),
        ("cs", False, "Nejspíš kvůli vytápění."),
        ("zh-Hans", True, "可能是供暖所致，室外 2 °C。"),
        ("zh-Hans", False, "可能是供暖所致。"),
    ],
)
def test_the_note_reads_right_in_other_languages(
    language: str, *, with_outdoor: bool, expected_end: str
) -> None:
    hass = SimpleNamespace(config=SimpleNamespace(language=language))
    entities = (_HEATING, _WEATHER) if with_outdoor else (_HEATING,)
    message = _baseline_deviation_mobile_message(
        _explained(_deviation(), *entities), cast("Any", hass)
    )
    assert message.endswith(expected_end)
    assert ".." not in message
    assert "。。" not in message


# ---------------------------------------------------------------------------
# Engine wiring: real registries, and a failure never loses findings
# ---------------------------------------------------------------------------


def _engine(hass: Any, options: dict[str, Any] | None = None) -> Any:
    from custom_components.home_generative_agent.sentinel.engine import (  # noqa: PLC0415
        SentinelEngine,
    )
    from custom_components.home_generative_agent.sentinel.logging_utils import (  # noqa: PLC0415
        RepeatingLogLimiter,
    )

    engine = SentinelEngine.__new__(SentinelEngine)
    engine._hass = hass
    engine._options = options or {}
    engine._log_limiter = RepeatingLogLimiter(logging.getLogger(__name__))
    return engine


@pytest.mark.asyncio
async def test_the_engine_links_a_plug_to_the_thermostat_on_its_device(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(domain="test")
    entry.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={("test", "heat-pump")}
    )
    registry = er.async_get(hass)
    plug = registry.async_get_or_create(
        "sensor", "test", "plug-power", device_id=device.id
    )
    thermostat = registry.async_get_or_create(
        "climate", "test", "thermostat", device_id=device.id
    )
    climate = _entity(
        thermostat.entity_id, "heat", {"hvac_action": "heating"}, "Thermostat"
    )

    (result,) = _engine(hass)._with_hvac_context(
        [_appliance(plug.entity_id)], _snapshot(climate, _WEATHER)
    )

    assert result.severity == "medium"
    assert result.evidence["hvac_context"]["climate"][0]["name"] == "Thermostat"


@pytest.mark.asyncio
async def test_the_engine_uses_the_configured_outdoor_entity(
    hass: HomeAssistant,
) -> None:
    sensor = _entity("sensor.outside", "-7", {"unit_of_measurement": "°C"})
    engine = _engine(hass, {CONF_SENTINEL_OUTDOOR_TEMPERATURE_ENTITY: "sensor.outside"})

    (result,) = engine._with_hvac_context(
        [_appliance()], _snapshot(_HEATING, _WEATHER, sensor)
    )

    assert result.evidence["hvac_context"]["outdoor_temperature"] == -7.0


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TypeError, RuntimeError, OverflowError])
async def test_a_failure_keeps_the_findings_as_they_were(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: type[Exception],
) -> None:
    """Any failure, not only the expected kinds, keeps the run's findings."""
    import custom_components.home_generative_agent.sentinel.engine as engine_mod  # noqa: PLC0415

    def _broken(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(engine_mod, "apply_hvac_context", _broken)
    finding = _appliance()

    assert _engine(hass)._with_hvac_context([finding], _snapshot(_HEATING)) == [finding]
    assert "Heating and cooling context failed" in caplog.text


# ---------------------------------------------------------------------------
# Cycle 2: direction, water heating on the device path, fridges, weather
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entity_id", "expected"),
    [
        ("sensor.heat_pump_power", {"heating", "cooling"}),
        ("sensor.hvac_power", {"heating", "cooling"}),
        ("sensor.air_conditioner_power", {"cooling"}),
        ("sensor.aircon", {"cooling"}),
        ("sensor.furnace_watts", {"heating"}),
        ("sensor.boiler", {"heating"}),
        ("sensor.mini_split_power", {"heating", "cooling"}),
    ],
)
def test_name_demands(entity_id: str, expected: set[str]) -> None:
    assert name_demands(entity_id, None) == frozenset(expected)


def test_an_air_conditioner_is_not_explained_by_heating() -> None:
    finding = _appliance("sensor.air_conditioner_power")
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING), device_climates=_no_device
    )
    assert result is finding


def test_a_furnace_is_not_explained_by_cooling() -> None:
    cooling = _entity(
        "climate.office",
        "cool",
        {"hvac_action": "cooling", "current_temperature": 26, "temperature": 23},
    )
    finding = _appliance("sensor.furnace_power")
    (result,) = apply_hvac_context(
        [finding], _snapshot(cooling), device_climates=_no_device
    )
    assert result is finding


def test_a_water_heater_on_the_heat_pump_device_is_never_explained() -> None:
    """A DHW element on the heat pump's own device still alerts unexplained."""
    finding = _appliance("sensor.dhw_booster_heater_power")
    (result,) = apply_hvac_context(
        [finding],
        _snapshot(_HEATING),
        device_climates=lambda _eid: ["climate.living_room"],
    )
    assert result is finding


@pytest.mark.parametrize("target", [4.0, -18.0, 38.0])
def test_a_fridge_compartment_thermostat_explains_nothing(target: float) -> None:
    """A cooling 'climate' with a fridge-like target is not a room."""
    compartment = _entity(
        "climate.compartment",
        "cool",
        {
            "hvac_action": "cooling",
            "current_temperature": target + 6,
            "temperature": target,
        },
    )
    assert climate_demand(compartment) is None


@pytest.mark.parametrize(
    ("target", "unit", "expected"),
    [
        (24.0, "°C", "cooling"),
        (75.0, "°F", "cooling"),
        (75.0, "°C", None),  # no room is cooled to 75 °C
        (25.0, "°F", None),  # a soft-freeze drawer in a °F home
    ],
)
def test_room_cooling_targets_follow_the_home_unit(
    target: float, unit: str, expected: str | None
) -> None:
    room = _entity(
        "climate.room",
        "cool",
        {
            "hvac_action": "cooling",
            "current_temperature": target + 3,
            "temperature": target,
        },
    )
    assert climate_demand(room, unit) == expected


@pytest.mark.parametrize(
    ("target", "unit", "expected"),
    [
        (21.0, "°C", "heating"),
        (68.0, "°F", "heating"),
        (55.0, "°C", None),  # a hot-water tank
        (80.0, "°C", None),  # a sauna
        (150.0, "°F", None),
    ],
)
def test_room_heating_targets_follow_the_home_unit(
    target: float, unit: str, expected: str | None
) -> None:
    heater = _entity(
        "climate.x",
        "heat",
        {
            "hvac_action": "heating",
            "current_temperature": target - 3,
            "temperature": target,
        },
    )
    assert climate_demand(heater, unit) == expected


def test_auto_on_a_heat_only_valve_is_not_cooling() -> None:
    valve = _entity(
        "climate.trv",
        "auto",
        {
            "hvac_modes": ["off", "heat", "auto"],
            "current_temperature": 22.0,
            "temperature": 21.0,
        },
    )
    assert climate_demand(valve) is None


def test_a_spa_thermostat_explains_nothing() -> None:
    spa = _entity("climate.spa", "heat", {"hvac_action": "heating"}, "Hot tub")
    assert climate_demand(spa) is None


def test_an_unavailable_weather_entity_is_skipped() -> None:
    stale = _entity(
        "weather.aaa", "unavailable", {"temperature": -10, "temperature_unit": "°C"}
    )
    assert outdoor_temperature(_snapshot(stale, _WEATHER), None) == (2.4, "°C")
    assert outdoor_temperature(_snapshot(stale, _WEATHER), "weather.aaa") is None


@pytest.mark.parametrize(
    ("template_id", "device_class", "unit", "explained"),
    [
        ("time_of_day_anomaly", "power", "W", True),
        ("baseline_deviation", "", "kW", True),
        ("baseline_deviation", "energy", "kWh", False),
    ],
)
def test_deviation_kinds(
    template_id: str, device_class: str, unit: str, *, explained: bool
) -> None:
    finding = _deviation(device_class=device_class, unit=unit)
    finding.evidence["template_id"] = template_id
    (result,) = apply_hvac_context(
        [finding], _snapshot(_HEATING), device_climates=_no_device
    )
    assert (hvac_explained_by(result.evidence) is not None) is explained


def test_triage_magnitude_is_bounded_and_typed() -> None:
    finding = _explained(_appliance(), _HEATING)
    finding.evidence.update(
        {"deviation_pct": 1e12, "duration_min": float("nan"), "threshold_min": True}
    )
    prompt = _build_prompt(finding, _snapshot(_HEATING))
    assert "deviation_pct: 100000" in prompt
    assert "power_w: 3100" in prompt
    assert "duration_min" not in prompt
    assert "threshold_min" not in prompt


def test_a_long_name_keeps_the_note_in_the_push() -> None:
    long_name = "Heat pump " + "x" * 300
    message = _appliance_power_duration_mobile_message(
        _explained(_appliance(friendly_name=long_name), _HEATING, _WEATHER)
    )
    assert len(message) <= 220
    assert message.endswith("Likely from heating, 2 °C outside.")


def test_the_record_keeps_the_full_text_and_the_note() -> None:
    long_name = "Heat pump " + "x" * 300
    finding = _explained(_appliance(friendly_name=long_name), _HEATING, _WEATHER)
    record = _persistent_message(None, finding)
    assert "x" * 300 in record
    assert record.endswith("Likely from heating, 2 °C outside.")


def test_the_record_adds_the_note_after_an_explanation() -> None:
    finding = _explained(_deviation(), _HEATING)
    record = _persistent_message("Baseline deviation: Heat pump power.", finding)
    assert record.startswith("Baseline deviation: Heat pump power.")
    assert record.endswith("Likely from heating.")


def test_a_translated_push_keeps_its_language() -> None:
    """No fixed note in another language after a translated explanation."""
    finding = _explained(_deviation(), _HEATING)
    message = _mobile_message("Tepelné čerpadlo odebírá víc.", finding, "Czech")
    assert message == "Tepelné čerpadlo odebírá víc."


def test_the_evidence_name_counts_when_the_entity_is_missing() -> None:
    """A plug named 'Freezer' that left the snapshot is still excluded."""
    finding = _appliance("sensor.plug_3_power", friendly_name="Freezer")
    (result,) = apply_hvac_context(
        [finding],
        _snapshot(_HEATING),
        device_climates=lambda _eid: ["climate.living_room"],
    )
    assert result is finding


@pytest.mark.parametrize(
    ("unit", "current", "expected"),
    [("W", 3000.0, "power_w: 3000"), ("kW", 20.0, "power_w: 20000")],
)
def test_triage_gets_the_deviation_in_watts(
    unit: str, current: float, expected: str
) -> None:
    finding = _deviation(unit=unit)
    finding.evidence["current_value"] = current
    prompt = _build_prompt(_explained(finding, _HEATING), _snapshot(_HEATING))
    assert expected in prompt


@pytest.mark.asyncio
async def test_the_engine_reads_the_home_temperature_unit(hass: HomeAssistant) -> None:
    from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM  # noqa: PLC0415

    hass.config.units = US_CUSTOMARY_SYSTEM
    engine = _engine(hass)
    assert engine._home_temperature_unit() == "°F"
    room = _entity(
        "climate.room",
        "heat",
        {"hvac_action": "heating", "current_temperature": 64, "temperature": 68},
    )

    (result,) = engine._with_hvac_context([_appliance()], _snapshot(room))

    assert hvac_explained_by(result.evidence) == "heating"
