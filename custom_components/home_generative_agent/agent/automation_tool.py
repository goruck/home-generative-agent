"""Automation creation tool for Home Generative Agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import aiofiles
import yaml
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.automation.const import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import ulid
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from voluptuous import MultipleInvalid

from ..const import (  # noqa: TID252
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_NOTIFY_SERVICE,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


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

    hass: HomeAssistant = config["configurable"]["hass"]
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
