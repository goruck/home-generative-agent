"""Langgraph tools for Home Generative Agent."""
from __future__ import annotations

import base64
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import aiofiles
import homeassistant.util.dt as dt_util
import yaml
from homeassistant.components import automation, camera, recorder
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.core import State
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.util import ulid
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langchain_core.tools import InjectedToolArg, tool
from langchain_ollama import ChatOllama  # noqa: TCH002
from langgraph.prebuilt import InjectedStore  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from voluptuous import MultipleInvalid

from .const import (
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_VLM,
    CONF_VLM_TEMPERATURE,
    CONF_VLM_TOP_P,
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
    RECOMMENDED_VLM,
    RECOMMENDED_VLM_TEMPERATURE,
    RECOMMENDED_VLM_TOP_P,
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_NUM_CTX,
    VLM_NUM_PREDICT,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VLM_IMAGE_WIDTH,
            height=VLM_IMAGE_HEIGHT
        )
    except HomeAssistantError as err:
        LOGGER.error(
            "Error getting image from camera '%s' with error: %s",
            camera_entity_id, err
        )
        return None

    return image.content

def _prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
    system = data["system"]
    text = data["text"]
    image = data["image"]

    text_part = {"type": "text", "text": text}
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    }

    content_parts = []
    content_parts.append(text_part)
    content_parts.append(image_part)

    return [SystemMessage(content=system), HumanMessage(content=content_parts)]

async def analyze_image(
        vlm_model: ChatOllama,
        options: dict[str, Any] | MappingProxyType[str, Any],
        image: bytes,
        detection_keywords: list[str] | None = None
    ) -> str:
    """Analyze an image."""
    image_data = base64.b64encode(image).decode("utf-8")

    model = vlm_model
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_VLM,
                RECOMMENDED_VLM,
            ),
            "temperature": options.get(
                CONF_VLM_TEMPERATURE,
                RECOMMENDED_VLM_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_VLM_TOP_P,
                RECOMMENDED_VLM_TOP_P,
            ),
            "num_predict": VLM_NUM_PREDICT,
            "num_ctx": VLM_NUM_CTX,
        }
    )

    chain = _prompt_func | model_with_config

    if detection_keywords is not None:
        prompt = VLM_USER_KW_TEMPLATE.format(
            key_words=f"{' or '.join(detection_keywords)}"
        )
    else:
        prompt = VLM_USER_PROMPT

    try:
        response =  await chain.ainvoke(
            {
                "system": VLM_SYSTEM_PROMPT,
                "text": prompt,
                "image": image_data
            }
        )
    except HomeAssistantError as err: #TODO: add validation error handling and retry prompt
        LOGGER.error("Error analyzing image %s", err)

    return response.content

@tool(parse_docstring=True)
async def get_and_analyze_camera_image( # noqa: D417
        camera_name: str,
        detection_keywords: list[str],
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
    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    image = await _get_camera_image(hass, camera_name)
    if image is None:
        return "Error getting image from camera."
    return await analyze_image(vlm_model, options, image, detection_keywords)

@tool(parse_docstring=True)
async def upsert_memory( # noqa: D417
    content: str,
    context: str = "",
    *,
    memory_id: str = "",
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    INSERT or UPDATE a memory in the database.
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
    mem_id = memory_id or ulid.ulid_now()
    await store.aput(
        namespace=(config["configurable"]["user_id"], "memories"),
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
    config: Annotated[RunnableConfig, InjectedToolArg()]
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
    hass = config["configurable"]["hass"]

    if time_pattern and message:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {AUTOMATION_TOOL_BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": AUTOMATION_TOOL_BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                }
            }
        }
        automation_yaml = yaml.dump(automation_data)

    automation_parsed = yaml.safe_load(automation_yaml)
    ha_automation_config = {"id": ulid.ulid_now()}
    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    if isinstance(automation_parsed, dict):
        ha_automation_config.update(automation_parsed)

    try:
        await _async_validate_config_item(
            hass = hass,
            config = ha_automation_config,
            raise_on_errors = True,
            warn_on_errors = False
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        return f"Invalid automation configuration {err}"

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        encoding="utf-8"
    ) as f:
        ha_exsiting_automation_configs = await f.read()
        ha_exsiting_automations_yaml = yaml.safe_load(ha_exsiting_automation_configs)

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        "a" if ha_exsiting_automations_yaml else "w",
        encoding="utf-8"
    ) as f:
        ha_automation_config_raw = yaml.dump(
            [ha_automation_config], allow_unicode=True, sort_keys=False
        )
        await f.write("\n" + ha_automation_config_raw)

    await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)

    hass.bus.async_fire(
        AUTOMATION_TOOL_EVENT_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"

def _get_state_and_decimate(
        data:list[dict[str, str]],
        keys:list[str] | None = None,
        limit:int = HISTORY_TOOL_CONTEXT_LIMIT
    ) -> list[dict[str, str]]:
    if keys is None:
        keys = ["state", "last_changed"]
    # Filter entity data to only state values with datetimes.
    state_values = [d for d in data if all(key in d for key in keys)]
    state_values = [{k: sv[k] for k in keys} for sv in state_values]
    # Decimate to avoid adding unnecessary fine grained date to context.
    length = len(state_values)
    if length > limit:
        LOGGER.debug("Decimating sensor data set.")
        factor = length // limit
        state_values = state_values[::factor]
    return state_values

def _gen_dict_extract(key: str, var: dict) -> Generator[str, None, None]:
    """Find a key in nested dict."""
    if hasattr(var,"items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in _gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in _gen_dict_extract(key, d):
                        yield result

def _filter_data(
        entity_id:str,
        data:list[dict[str, str]],
        hass:HomeAssistant
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

        # Filter history to just state values (no datetimes).
        state_values = []
        for x in list(_gen_dict_extract("state", {entity_id:data})):
            try:
                state_values.append(float(x))
            except ValueError:
                LOGGER.debug("Found string that could not be converted to float.")
                continue
        # Check if sensor was reset during the time of interest.
        zero_indices = [i for i, x in enumerate(state_values) if math.isclose(x, 0)]
        if zero_indices:
            # Start data set from last time the sensor was reset.
            LOGGER.debug("Sensor was reset during time of interest.")
            state_values = state_values[zero_indices[-1]:]
        state_value_change = max(state_values) - min(state_values)
        units = state_obj.attributes.get("unit_of_measurement")
        return {"value": state_value_change, "units": units}

    return {"values": _get_state_and_decimate(data)}

async def _fetch_data_from_history(
        hass:HomeAssistant,
        start_time:datetime,
        end_time:datetime,
        entity_ids:list[str]
    ) -> dict[str, list[str, str]]:
    filters = None
    include_start_time_state = True
    significant_changes_only = True
    minimal_response = True # If True filter out duplicate states
    no_attributes = False
    compressed_state_format = False

    with recorder.util.session_scope(hass=hass, read_only=True) as session:
        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.history.get_significant_states_with_session,
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
            compressed_state_format
        )

async def _fetch_data_from_long_term_stats(
        hass:HomeAssistant,
        start_time:datetime,
        end_time:datetime,
        entity_ids:list[str]
    ) -> dict[str, list[str, str]]:
    period = "hour"
    units = None

    # Only concerned with two statistic types. The "state" type is associated with
    # sensor entities that have State Class of total or total_increasing.
    # The "mean" type is associated with entities with State Class measurement.
    types = {"state", "mean"}

    result = await recorder.get_instance(hass).async_add_executor_job(
        recorder.statistics.statistics_during_period,
        hass,
        start_time,
        end_time,
        set(entity_ids),
        period,
        units,
        types
    )

    # Make data format consistent with the History format.
    parsed_result:dict[str, list[str, str]] = {}
    for k, v in result.items():
        data = [
            {
                "state": d["state"] if "state" in d else d.get("mean"),
                "last_changed": dt_util.as_local(
                    dt_util.utc_from_timestamp(d.get("end"))
                )
            } for d in v
        ]
        parsed_result.update({k: data})

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

@tool(parse_docstring=True)
async def get_entity_history(  # noqa: D417
    entity_ids: list[str],
    local_start_time: str,
    local_end_time: str,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> dict[str, list[dict[str, Any]]]:
    """
    Get entity state history from Home Assistant.

    Args:
        entity_ids: List of Home Assistant entity ids to retrieve the history for.
            For example if the user says "how much energy did the washing machine
            consume last week", entity_id is "sensor.washing_machine_switch_0_energy"
            DO NOT use use the name "washing machine Switch 0 energy" for entity_id.
            You MUST use an underscore symbol (e.g., "_") as a word deliminator.
        local_start_time: Start of local time history period in "%Y-%m-%dT%H:%M:%S%z".
        local_end_time: End of local time history period in "%Y-%m-%dT%H:%M:%S%z".

    Returns:
        Entity history in local time.

    """
    hass = config["configurable"]["hass"]

    entity_ids = [i.lower() for i in entity_ids]

    now = dt_util.utcnow()
    one_day = timedelta(days=1)
    try:
        start_time = _as_utc(
            dattim = local_start_time,
            default = now - one_day,
            error_message = "start_time not valid"
        )
        end_time = _as_utc(
            dattim = local_end_time,
            default = start_time + one_day,
            error_message = "end_time not valid"
        )
    except HomeAssistantError as err:
        return f"Invalid time {err}"

     # Calculate the threshold to fetch data from history or long-term statistics.
    threshold = dt_util.now() - timedelta(days=HISTORY_TOOL_PURGE_KEEP_DAYS)

    # Check if the range spans both "old" and "recent" dates.
    data:dict[str, list[str, str]] = {}
    if start_time < threshold and end_time >= threshold:
        # The range includes datetimes both older than threshold and more recent.
        data = await _fetch_data_from_long_term_stats(
            hass=hass,
            start_time=start_time,
            end_time=threshold,
            entity_ids=entity_ids
        )
        data.update(await _fetch_data_from_history(
            hass=hass,
            start_time=threshold,
            end_time=end_time,
            entity_ids=entity_ids
        ))
    elif end_time < threshold:
        # Entire range is older than threshold.
        data = await _fetch_data_from_long_term_stats(
            hass=hass,
            start_time=start_time,
            end_time=threshold,
            entity_ids=entity_ids
        )
    else:
        # Entire range is more recent than threshold.
        data = await _fetch_data_from_history(
            hass=hass,
            start_time=threshold,
            end_time=end_time,
            entity_ids=entity_ids
        )

    if not data:
        return {}

    # Convert any State objects to dict.
    data = {
        e: [
            s.as_dict() if isinstance(s, State) else s for s in v
        ] for e, v in data.items()
    }

    # Convert datetimes in UTC to local timezone.
    for lst in data.values():
        for d in lst:
            for k, v in d.items():
                try:
                    dattim = dt_util.parse_datetime(v, raise_on_error=True)
                    dattim_local = dt_util.as_local(dattim)
                    d.update({k: dattim_local.strftime("%Y-%m-%dT%H:%M:%S%z")})
                except (ValueError, TypeError):
                    pass
                except HomeAssistantError as err:
                    return f"Unexpected datetime conversion error {err}"

    # Return filtered entity data set to avoid filling context with too much data.
    return {k: _filter_data(k,v,hass) for k,v in data.items()}

###
# This tool has been replaced by the HA native tool GetLiveContext.
# It is no longer used. Keeping it here for reference only.
###
@tool(parse_docstring=True)
async def get_current_device_state( # noqa: D417
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
        # Define the marker that separates instructions from the device list.
        split_marker = "An overview of the areas and the devices in this smart home:"

        # Check if the marker exists in the input text.
        if split_marker not in input_text:
            msg = "Input text format is invalid. Marker not found."
            raise ValueError(msg)

        # Split the input text into instructions and devices part
        instructions_part, devices_part = input_text.split(split_marker, 1)

        # Clean up whitespace
        instructions = instructions_part.strip()
        devices_yaml = devices_part.strip()

        # Parse the devices list using PyYAML
        devices = yaml.safe_load(devices_yaml)

        # Combine into a single dictionary
        return {
            "instructions": instructions,
            "devices": devices
        }

    # Use the HA LLM API to get overview of all devices.
    llm_api = config["configurable"]["ha_llm_api"]
    try:
        overview = _parse_input_to_yaml(llm_api.api_prompt)
    except ValueError as e:
        LOGGER.error("There was a problem getting device state: %s", e)
        return {}

    # Get the list of devices.
    devices = overview.get("devices", [])

    # Create a dictionary mapping desired device names to their state.
    state_dict = {}
    for device in devices:
        name = device.get("names", "Unnamed Device")
        if name not in names:
            continue
        state = device.get("state", None)
        state_dict[name] = state

    return state_dict
