"""Langgraph tools for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import aiofiles
import homeassistant.util.dt as dt_util
import yaml
from homeassistant.components import camera
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.automation.const import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.components.recorder import history as recorder_history
from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import ATTR_FRIENDLY_NAME, SERVICE_RELOAD
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance as get_recorder_instance
from homeassistant.helpers.recorder import session_scope as recorder_session_scope
from homeassistant.util import ulid
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore  # noqa: TC002
from voluptuous import MultipleInvalid

from ..const import (  # noqa: TID252
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_NOTIFY_SERVICE,
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)


async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VLM_IMAGE_WIDTH,
            height=VLM_IMAGE_HEIGHT,
        )
    except HomeAssistantError:
        LOGGER.exception("Error getting image from camera %s", camera_entity_id)
        return None

    return image.content


def _prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
    system = data["system"]
    text = data["text"]
    image = data["image"]
    prev_text = data.get("prev_text")

    # Build the user content (text first, then optional previous frame text, then image)
    content_parts: list[str | dict[str, Any]] = []

    # Main instruction text
    content_parts.append({"type": "text", "text": text})

    # OPTIONAL: previous frame's one-line description to aid motion/direction grounding
    if prev_text:
        # Keep it short and explicit that it is text-only context, not metadata
        content_parts.append(
            {"type": "text", "text": f'Previous frame (text only): "{prev_text}"'}
        )

    # Image payload last
    content_parts.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
    )

    return [SystemMessage(content=system), HumanMessage(content=content_parts)]


async def analyze_image(
    vlm_model: RunnableSerializable[LanguageModelInput, BaseMessage],
    image: bytes,
    detection_keywords: list[str] | None = None,
    prev_text: str | None = None,
) -> str:
    """Analyze an image with the preconfigured VLM model."""
    await asyncio.sleep(0)  # keep the event loop snappy

    image_data = base64.b64encode(image).decode("utf-8")
    chain = _prompt_func | vlm_model

    if detection_keywords is not None:
        prompt = VLM_USER_KW_TEMPLATE.format(key_words=" or ".join(detection_keywords))
    else:
        prompt = VLM_USER_PROMPT

    try:
        resp = await chain.ainvoke(
            {
                "system": VLM_SYSTEM_PROMPT,
                "text": prompt,
                "image": image_data,
                "prev_text": prev_text,
            }
        )
    except HomeAssistantError:
        msg = "Error analyzing image with VLM model."
        LOGGER.exception(msg)
        return msg

    # Normalize return text across adapters
    if hasattr(resp, "content"):
        return str(resp.content)
    if hasattr(resp, "text"):
        try:
            return str(resp.text())
        except Exception:
            LOGGER.exception("Error converting response to string.")
            return "Error converting response to string."
    return str(resp)


@tool(parse_docstring=True)
async def get_and_analyze_camera_image(  # noqa: D417
    camera_name: str,
    detection_keywords: list[str] | None = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Get a camera image and perform scene analysis on it.

    Args:
        camera_name: Name of the camera for scene analysis.
        detection_keywords: Specific objects to look for in image, if any.
            For example, If user says "check the front porch camera for
            boxes and dogs", detection_keywords would be ["boxes", "dogs"].

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    image = await _get_camera_image(hass, camera_name)
    if image is None:
        return "Error getting image from camera."
    return await analyze_image(vlm_model, image, detection_keywords)


@tool(parse_docstring=True)
async def upsert_memory(  # noqa: D417
    content: str,
    context: str = "",
    *,
    memory_id: str = "",
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    INSERT or UPDATE a memory about users in the database.

    You MUST use this tool to INSERT or UPDATE memories about users.
    Examples of memories are specific facts or concepts learned from interactions
    with users. If a memory conflicts with an existing one then just UPDATE the
    existing one by passing in "memory_id" and DO NOT create two memories that are
    the same. If the user corrects a memory then UPDATE it.

    Args:
        content: The main content of the memory.
            e.g., "I would like to learn french."
        context: Additional relevant context for the memory, if any.
            e.g., "This was mentioned while discussing career options in Europe."
        memory_id: The memory to overwrite.
            ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    mem_id = memory_id or ulid.ulid_now()

    user_id = config["configurable"]["user_id"]
    await store.aput(
        namespace=(user_id, "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


@tool(parse_docstring=True)
async def add_automation(  # noqa: D417
    automation_yaml: str = "",
    time_pattern: str = "",
    message: str = "",
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Add an automation to Home Assistant.

    You are provided a Home Assistant blueprint as part of this tool if you need it.
    You MUST ONLY use the blueprint to create automations that involve camera image
    analysis. You MUST generate Home Assistant automation YAML for everything else.
    If using the blueprint you MUST provide the arguments "time_pattern" and "message"
    and DO NOT provide the argument "automation_yaml".

    Args:
        automation_yaml: A Home Assistant automation in valid YAML format.
            ONLY provide if NOT using the camera image analysis blueprint.
        time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
            ONLY provide if using the camera image analysis blueprint.
        message: Image analysis prompt (e.g.,"check the front porch camera for boxes")
            ONLY provide if using the camera image analysis blueprint.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    mobile_push_service = config["configurable"]["options"].get(CONF_NOTIFY_SERVICE)

    if time_pattern and message:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {AUTOMATION_TOOL_BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": AUTOMATION_TOOL_BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                    "mobile_push_service": mobile_push_service or "",
                },
            },
        }
        automation_yaml = yaml.dump(automation_data)

    automation_parsed = yaml.safe_load(automation_yaml)
    ha_automation_config: dict[str, Any] = {"id": ulid.ulid_now()}
    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    if isinstance(automation_parsed, dict):
        ha_automation_config.update(automation_parsed)

    try:
        await _async_validate_config_item(
            hass=hass,
            config=ha_automation_config,
            raise_on_errors=True,
            warn_on_errors=False,
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        return f"Invalid automation configuration {err}"

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH, encoding="utf-8"
    ) as f:
        ha_exsiting_automation_configs = await f.read()
        ha_exsiting_automations_yaml = yaml.safe_load(ha_exsiting_automation_configs)

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        "a" if ha_exsiting_automations_yaml else "w",
        encoding="utf-8",
    ) as f:
        ha_automation_config_raw = yaml.dump(
            [ha_automation_config], allow_unicode=True, sort_keys=False
        )
        await f.write("\n" + ha_automation_config_raw)

    await hass.services.async_call(AUTOMATION_DOMAIN, SERVICE_RELOAD)
    hass.bus.async_fire(
        AUTOMATION_TOOL_EVENT_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"


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


# Allow domains like "sensor", "binary_sensor", "camera", etc.
_DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Valid HA entity_id = <domain>.<object_id>
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


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


###
# This tool has been replaced by the HA native tool GetLiveContext.
# It is no longer used. Keeping it here for reference only.
###
@tool(parse_docstring=True)
async def get_current_device_state(  # noqa: D417
    names: list[str],
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, str]:
    """
    Get the current state of one or more Home Assistant devices.

    Args:
        names: List of Home Assistant device names.

    """

    def _parse_input_to_yaml(input_text: str) -> dict[str, Any]:
        split_marker = "An overview of the areas and the devices in this smart home:"
        if split_marker not in input_text:
            msg = "Input text format is invalid. Marker not found."
            raise ValueError(msg)

        instructions_part, devices_part = input_text.split(split_marker, 1)
        instructions = instructions_part.strip()
        devices_yaml = devices_part.strip()
        devices = yaml.safe_load(devices_yaml)
        return {"instructions": instructions, "devices": devices}

    if "configurable" not in config:
        LOGGER.warning("Configuration not found. Please check your setup.")
        return {}
    llm_api = config["configurable"]["ha_llm_api"]
    try:
        overview = _parse_input_to_yaml(llm_api.api_prompt)
    except ValueError:
        LOGGER.exception("There was a problem getting device state.")
        return {}

    devices = overview.get("devices", [])
    state_dict: dict[str, str] = {}
    for device in devices:
        name = device.get("names", "Unnamed Device")
        if name not in names:
            continue
        state = device.get("state", None)
        state_dict[name] = state

    return state_dict
