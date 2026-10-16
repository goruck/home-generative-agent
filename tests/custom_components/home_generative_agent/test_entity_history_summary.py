# ruff: noqa: S101
"""Long histories keep every state the device was in (#715)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from custom_components.home_generative_agent.agent import tools as agent_tools
from custom_components.home_generative_agent.const import HISTORY_TOOL_CONTEXT_LIMIT

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LIMIT = HISTORY_TOOL_CONTEXT_LIMIT
MIDNIGHT = datetime.fromisoformat("2026-10-16T00:00:00-07:00")


def _at(minute: int) -> str:
    return (MIDNIGHT + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _history(states: list[str]) -> list[dict[str, str]]:
    """One entry per minute from midnight."""
    return [{"state": s, "last_changed": _at(i)} for i, s in enumerate(states)]


def _toggles(count: int) -> list[dict[str, str]]:
    return _history(["off" if i % 2 == 0 else "on" for i in range(count)])


def test_toggling_light_keeps_both_states(hass: HomeAssistant) -> None:
    """106 toggles used to sample down to 53 entries, every one "off"."""
    hass.states.async_set("light.mudroom", "on")
    data = _toggles(106)
    period = (MIDNIGHT, MIDNIGHT + timedelta(minutes=110))

    result = agent_tools._filter_data("light.mudroom", data, hass, period)

    assert {v["state"] for v in result["values"]} == {"on", "off"}
    assert result["values"] == data[-LIMIT:]
    assert result["state_at_start"] == data[0]
    on, off = result["states"]["on"], result["states"]["off"]
    assert on["times_entered"] == 53
    assert off["times_entered"] == 52
    # The first "on" is at minute 1, long before the listed values start.
    assert on["first_entered"] == _at(1)
    assert on["last_entered"] == _at(105)
    # 53 one-minute "on" runs, the last one lasting until the period ends.
    assert on["minutes_in_state"] == 52 + 5
    assert off["minutes_in_state"] == 53
    assert "106 state changes" in result["note"]
    assert "No history before" not in result["note"]


def test_long_history_lists_consecutive_latest_entries(hass: HomeAssistant) -> None:
    """An odd sampling step kept both states but skipped changes in between."""
    hass.states.async_set("binary_sensor.front_door", "off")
    data = _toggles(150)

    result = agent_tools._filter_data("binary_sensor.front_door", data, hass)

    assert result["values"] == data[-LIMIT:]
    assert result["states"]["on"]["times_entered"] == 75


def test_repeated_rows_do_not_use_up_the_limit(hass: HomeAssistant) -> None:
    """A thermostat logs a row per attribute update; only changes are kept."""
    hass.states.async_set("climate.hall", "heat")
    states = ["off", "heat"] + ["heat"] * 200 + ["off"]

    result = agent_tools._filter_data("climate.hall", _history(states), hass)

    assert [v["state"] for v in result["values"]] == ["off", "heat", "off"]
    assert result["values"][-1]["last_changed"] == _at(202)


def test_unavailable_blip_is_not_a_new_change(hass: HomeAssistant) -> None:
    """on, unavailable, on is one time on, not two."""
    hass.states.async_set("light.porch", "on")
    states = ["off"] + ["on", "unavailable", "on", "off"] * 15

    result = agent_tools._filter_data("light.porch", _history(states), hass)

    assert result["states"]["on"]["times_entered"] == 15
    assert result["states"]["unavailable"]["times_entered"] == 15
    assert result["states"]["off"]["times_entered"] == 15


def test_history_that_starts_late_says_so(hass: HomeAssistant) -> None:
    """A period reaching past the recorder's history notes where data begins."""
    hass.states.async_set("light.mudroom", "on")
    data = _toggles(60)
    period = (MIDNIGHT - timedelta(days=12), MIDNIGHT + timedelta(minutes=60))

    result = agent_tools._filter_data("light.mudroom", data, hass, period)

    assert f"No history before {_at(0)}" in result["note"]


def test_short_history_is_returned_whole(hass: HomeAssistant) -> None:
    """At or under the limit nothing is dropped and no summary is added."""
    hass.states.async_set("light.mudroom", "on")
    data = _toggles(LIMIT)

    assert agent_tools._filter_data("light.mudroom", data, hass) == {"values": data}


def test_measurement_thinning_keeps_latest_reading(hass: HomeAssistant) -> None:
    """A numeric series is still thinned, within the limit, ending on now."""
    hass.states.async_set(
        "sensor.temp",
        "20.5",
        {"state_class": "measurement", "unit_of_measurement": "°C"},
    )
    data = _history([str(20 + i / 100) for i in range(106)])

    result = agent_tools._filter_data("sensor.temp", data, hass)

    assert len(result["values"]) <= LIMIT
    assert result["values"][0] == data[0]
    assert result["values"][-1] == data[-1]
    assert result["units"] == "°C"


def test_number_without_state_class_is_thinned(hass: HomeAssistant) -> None:
    """A template sensor's readings are not summarized state by state."""
    hass.states.async_set("sensor.template_power", "12.5")
    data = _history([str(i) for i in range(300)] + ["unavailable"])

    result = agent_tools._filter_data("sensor.template_power", data, hass)

    assert "states" not in result
    assert len(result["values"]) <= LIMIT
    assert result["values"][-1] == data[-1]
