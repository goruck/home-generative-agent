"""Langgraph tools for Home Generative Agent."""
from __future__ import annotations

import base64
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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
from ulid import ULID  # noqa: TCH002

from .const import (
    BLUEPRINT_NAME,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TOP_P,
    CONF_VLM,
    EVENT_AUTOMATION_REGISTERED,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    VISION_MODEL_IMAGE_HEIGHT,
    VISION_MODEL_IMAGE_WIDTH,
    VISION_MODEL_SYSTEM_PROMPT,
    VISION_MODEL_USER_KW_PROMPT,
    VISION_MODEL_USER_PROMPT,
    VLM_NUM_PREDICT,
)

if TYPE_CHECKING:
    from datetime import datetime
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

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

async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    try:
        image = await camera.async_get_image(
            hass=hass,
            entity_id=camera_entity_id,
            width=VISION_MODEL_IMAGE_WIDTH,
            height=VISION_MODEL_IMAGE_HEIGHT
        )
    except HomeAssistantError as err:
        LOGGER.error(
            "Error getting image from camera '%s' with error: %s",
            camera_entity_id, err
        )

    return image.content

async def _analyze_image(
        vlm_model: ChatOllama,
        options: dict[str, Any] | MappingProxyType[str, Any],
        image: bytes,
        detection_keywords: list[str] | None = None
    ) -> str:
    """Analyze an image."""
    encoded_image = base64.b64encode(image).decode("utf-8")

    def prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
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

    model = vlm_model
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_VLM,
                RECOMMENDED_VLM,
            ),
            "temperature": options.get(
                CONF_VISION_MODEL_TEMPERATURE,
                RECOMMENDED_VISION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_VISION_MODEL_TOP_P,
                RECOMMENDED_VISION_MODEL_TOP_P,
            ),
            "num_predict": VLM_NUM_PREDICT,
        }
    )

    chain = prompt_func | model_with_config

    if detection_keywords is not None:
        prompt = f"{VISION_MODEL_USER_KW_PROMPT} {' or '.join(detection_keywords):}"
    else:
        prompt = VISION_MODEL_USER_PROMPT

    try:
        response =  await chain.ainvoke(
            {
                "system": VISION_MODEL_SYSTEM_PROMPT,
                "text": prompt,
                "image": encoded_image
            }
        )
    except HomeAssistantError as err: #TODO: add validation error handling and retry prompt
        LOGGER.error("Error analyzing image %s", err)

    return response

@tool(parse_docstring=False)
async def get_and_analyze_camera_image( # noqa: D417
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
    hass = config["configurable"]["hass"]
    vlm_model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    image = await _get_camera_image(hass, camera_name)
    return await _analyze_image(vlm_model, options, image, detection_keywords)

@tool(parse_docstring=False)
async def upsert_memory( # noqa: D417
    content: str,
    context: str,
    *,
    memory_id: ULID | None = None,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
            The memory to overwrite

    Returns:
        A string containing the stored memory id.

    """
    mem_id = memory_id or ulid.ulid_now()
    await store.aput(
        namespace=(config["configurable"]["user_id"], "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"

@tool(parse_docstring=False)
async def add_automation(  # noqa: D417
    automation_yaml: str | None = None,
    time_pattern: str | None = None,
    message: str | None = None,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """
    Add an automation to Home Assistant.

    You are provided a Home Assistant blueprint as part of this tool if you need it.
    You MUST ONLY use the blueprint to create automations that involve camera image
    analysis. You MUST generate Home Assistant automation yaml for everything else.
    If using the blueprint you MUST provide the arguments "time_pattern" and "message"
    and DO NOT provide the argument "automation_yaml".

    Args:
        automation_yaml: A Home Assistant automation in valid yaml format.
        time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
        message: Image analysis prompt (e.g.,"check the front porch camera for boxes")

    """
    hass = config["configurable"]["hass"]

    if time_pattern is not None and message is not None:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": BLUEPRINT_NAME,
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
    except HomeAssistantError as err:
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
        EVENT_AUTOMATION_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"

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
        local_start_time: Start of local time history period in "%Y-%m-%dT%H:%M:%S%z".
        local_end_time: End of local time history period in "%Y-%m-%dT%H:%M:%S%z".

    Returns:
        Entity history in local time.

    """
    hass = config["configurable"]["hass"]

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

    filters = None
    include_start_time_state = True
    significant_changes_only = True
    minimal_response = True # If True filter out duplicate states
    no_attributes = False
    compressed_state_format = False

    with recorder.util.session_scope(hass=hass, read_only=True) as session:
        history = await recorder.get_instance(hass).async_add_executor_job(
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

    if not history:
        return {}

    # Convert any State objects in history to dict.
    history = {
        e: [
            s.as_dict() if isinstance(s, State) else s for s in v
        ] for e, v in history.items()
    }

    # Convert history datetimes in UTC to local timezone.
    for lst in history.values():
        for d in lst:
            for k, v in d.items():
                LOGGER.debug("HISTORY TOOL: K: %s\n V: %s\n", k, v)
                try:
                    dattim = dt_util.parse_datetime(v, raise_on_error=True)
                    dattim_local = dt_util.as_local(dattim)
                    d.update({k: dattim_local.strftime("%Y-%m-%dT%H:%M:%S%z")})
                except (ValueError, TypeError):
                    pass
                except HomeAssistantError as err:
                    return f"Unexpected datetime conversion error {err}"

    return history
