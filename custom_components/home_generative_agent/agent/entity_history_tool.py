"""Entity history tool for Home Generative Agent."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

import homeassistant.util.dt as dt_util
from homeassistant.components.recorder import history as recorder_history
from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.const import ATTR_FRIENDLY_NAME
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance as get_recorder_instance
from homeassistant.helpers.recorder import session_scope as recorder_session_scope
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool

from ..const import (  # noqa: TID252
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# Allow domains like "sensor", "binary_sensor", "camera", etc.
_DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Valid HA entity_id = <domain>.<object_id>
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def _get_state_and_decimate(
    data: list[dict[str, str]],
    keys: list[str] | None = None,
    limit: int = HISTORY_TOOL_CONTEXT_LIMIT,
) -> list[dict[str, str]]:
    if keys is None:
        keys = ["state", "last_changed"]
    # Filter entity data to only state values with datetimes.
    state_values = [d for d in data if all(key in d for key in keys)]
    state_values = [{k: sv[k] for k in keys} for sv in state_values]
    # Decimate to avoid adding unnecessary fine grained data to context.
    length = len(state_values)
    if length > limit:
        LOGGER.debug("Decimating sensor data set.")
        factor = max(1, length // limit)
        state_values = state_values[::factor]
    return state_values


def _gen_dict_extract(key: str, var: dict) -> Generator[str]:
    """Find a key in nested dict."""
    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                yield from _gen_dict_extract(key, v)
            elif isinstance(v, list):
                for d in v:
                    yield from _gen_dict_extract(key, d)


def _filter_data(
    entity_id: str, data: list[dict[str, str]], hass: HomeAssistant
) -> dict[str, Any]:
    state_obj = hass.states.get(entity_id)
    if not state_obj:
        return {}

    state_class = state_obj.attributes.get("state_class")

    if state_class in ("measurement", "total"):
        state_values = _get_state_and_decimate(data)
        units = state_obj.attributes.get("unit_of_measurement")
        return {"values": state_values, "units": units}

    if state_class == "total_increasing":
        # For sensors with state class 'total_increasing', the data contains the
        # accumulated growth of the sensor's value since it was first added.
        # Therefore, return the net change.
        state_values = []
        for x in list(_gen_dict_extract("state", {entity_id: data})):
            try:
                state_values.append(float(x))
            except ValueError:
                LOGGER.warning("Found string that could not be converted to float.")
                continue
        # Check if sensor was reset during the time of interest.
        zero_indices = [i for i, x in enumerate(state_values) if math.isclose(x, 0.0)]
        if zero_indices:
            LOGGER.warning("Sensor was reset during time of interest.")
            state_values = state_values[zero_indices[-1] :]
        state_value_change = max(state_values) - min(state_values)
        units = state_obj.attributes.get("unit_of_measurement")
        return {"value": state_value_change, "units": units}

    return {"values": _get_state_and_decimate(data)}


async def _fetch_data_from_history(
    hass: HomeAssistant, start_time: datetime, end_time: datetime, entity_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    filters = None
    include_start_time_state = True
    significant_changes_only = True
    minimal_response = True  # If True filter out duplicate states
    no_attributes = False
    compressed_state_format = False

    with recorder_session_scope(hass=hass, read_only=True) as session:
        result = await get_recorder_instance(hass).async_add_executor_job(
            recorder_history.get_significant_states_with_session,
            hass,
            session,
            start_time,
            end_time,
            entity_ids,
            filters,
            include_start_time_state,
            significant_changes_only,
            minimal_response,
            no_attributes,
            compressed_state_format,
        )

    if not result:
        return {}

    # Convert any State objects to dict.
    return {
        e: [s.as_dict() if isinstance(s, State) else s for s in v]
        for e, v in result.items()
    }


async def _fetch_data_from_long_term_stats(
    hass: HomeAssistant, start_time: datetime, end_time: datetime, entity_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    period = "hour"
    units = None

    # Only concerned with two statistic types. The "state" type is associated with
    # sensor entities that have State Class of total or total_increasing.
    # The "mean" type is associated with entities with State Class measurement.
    types = {"state", "mean"}

    result = await get_recorder_instance(hass).async_add_executor_job(
        recorder_statistics.statistics_during_period,
        hass,
        start_time,
        end_time,
        set(entity_ids),
        period,
        units,
        types,
    )

    # Make data format consistent with the History format.
    parsed_result: dict[str, list[dict[str, Any]]] = {}
    for k, v in result.items():
        data: list[dict[str, Any]] = [
            {
                "state": d["state"] if "state" in d else d.get("mean"),
                "last_changed": dt_util.as_local(dt_util.utc_from_timestamp(d["end"]))
                if "end" in d
                else None,
            }
            for d in v
        ]
        parsed_result[k] = data

    return parsed_result


def _as_utc(dattim: str, default: datetime, error_message: str) -> datetime:
    """
    Convert a string representing a datetime into a datetime.datetime.

    Args:
        dattim: String representing a datetime.
        default: datatime.datetime to use as default.
        error_message: Message to raise in case of error.

    Raises:
        Homeassistant error if datetime cannot be parsed.

    Returns:
        A datetime.datetime of the string in UTC.

    """
    if dattim is None:
        return default

    parsed_datetime = dt_util.parse_datetime(dattim)
    if parsed_datetime is None:
        raise HomeAssistantError(error_message)

    return dt_util.as_utc(parsed_datetime)


async def _get_existing_entity_id(
    name: str | None, hass: HomeAssistant, domain: str | None = "sensor"
) -> str:
    """
    Lookup an existing entity by its friendly name.

    Raises ValueError if not found, ambiguous, or invalid domain/entity_id.
    """
    if not isinstance(name, str) or not name.strip():
        msg = "Name must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(domain, str) or not _DOMAIN_PATTERN.match(domain):
        msg = "Domain invalid; must be a valid Home Assistant domain"
        raise ValueError(msg)

    target = name.strip().lower()
    prefix = f"{domain}."
    candidates: list[str] = []

    for state in hass.states.async_all():
        eid = state.entity_id
        if not eid.startswith(prefix):
            continue
        fn = state.attributes.get(ATTR_FRIENDLY_NAME, "")
        if isinstance(fn, str) and fn.strip().lower() == target:
            candidates.append(eid)

    if not candidates:
        msg = f"No '{domain}' entity found with friendly name '{name}'"
        raise ValueError(msg)
    if len(candidates) > 1:
        msg = f"Multiple '{domain}' entities found for '{name}': {candidates}"
        raise ValueError(msg)

    eid = candidates[0]
    if not _ENTITY_ID_PATTERN.match(eid):
        msg = f"Found entity_id '{eid}' is not valid"
        raise ValueError(msg)

    return eid


@tool(parse_docstring=True)
async def get_entity_history(  # noqa: D417
    friendly_names: list[str],
    domains: list[str],
    local_start_time: str,
    local_end_time: str,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """
    Get entity state histories from Home Assistant.

    Args:
        friendly_names: List of Home Assistant friendly names to get history for,
            for example, ["Front Door", "Living Room Light"].
        domains: List of Home Assistant domains associated with the friendly names,
            for example, ["binary_sensor", "light"]. These must be in the same order as
            friendly_names.
        local_start_time: Start of local time history period in "%Y-%m-%dT%H:%M:%S%z".
        local_end_time: End of local time history period in "%Y-%m-%dT%H:%M:%S%z".

    Returns:
        Entity histories in local time format, for example:
            {
                "binary_sensor.front_door": {"values": [
                    {"state": "off", "last_changed": "2025-07-24T00:00:00-0700"},
                    {"state": "on", "last_changed": "2025-07-24T04:47:28-0700"},
                    ...]}
            }.

    """
    if "configurable" not in config:
        LOGGER.warning("Configuration not found. Please check your setup.")
        return {}

    hass: HomeAssistant = config["configurable"]["hass"]

    try:
        entity_ids = [
            await _get_existing_entity_id(n, hass, d)
            for n in friendly_names
            for d in domains
        ]
    except ValueError:
        LOGGER.exception("Invalid name %s or domain: %s", friendly_names, domains)
        return {}

    now = dt_util.utcnow()
    one_day = timedelta(days=1)
    try:
        start_time = _as_utc(
            dattim=local_start_time,
            default=now - one_day,
            error_message="start_time not valid",
        )
        end_time = _as_utc(
            dattim=local_end_time,
            default=start_time + one_day,
            error_message="end_time not valid",
        )
    except HomeAssistantError:
        LOGGER.exception("Error parsing start or end time.")
        return {}

    threshold = dt_util.now() - timedelta(days=HISTORY_TOOL_PURGE_KEEP_DAYS)

    data: dict[str, list[dict[str, Any]]]
    if start_time < threshold and end_time >= threshold:
        data = await _fetch_data_from_long_term_stats(
            hass=hass, start_time=start_time, end_time=threshold, entity_ids=entity_ids
        )
        data.update(
            await _fetch_data_from_history(
                hass=hass,
                start_time=threshold,
                end_time=end_time,
                entity_ids=entity_ids,
            )
        )
    elif end_time < threshold:
        data = await _fetch_data_from_long_term_stats(
            hass=hass, start_time=start_time, end_time=end_time, entity_ids=entity_ids
        )
    else:
        data = await _fetch_data_from_history(
            hass=hass, start_time=start_time, end_time=end_time, entity_ids=entity_ids
        )

    if not data:
        return {}

    for lst in data.values():
        for d in lst:
            for k, v in d.items():
                try:
                    dattim = dt_util.parse_datetime(v, raise_on_error=True)
                    dattim_local = dt_util.as_local(dattim)
                    d[k] = dattim_local.strftime("%Y-%m-%dT%H:%M:%S%z")
                except (ValueError, TypeError):
                    pass

    return {k: _filter_data(k, v, hass) for k, v in data.items()}
