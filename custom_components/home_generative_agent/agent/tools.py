"""Langgraph tools for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import aiofiles
import async_timeout
import homeassistant.util.dt as dt_util
import voluptuous as vol
import yaml
from homeassistant.components import camera
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.automation.const import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.components.recorder import history as recorder_history
from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import (
    ATTR_FRIENDLY_NAME,
    SERVICE_RELOAD,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
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
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_NOTIFY_SERVICE,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)
from ..core.utils import extract_final, verify_pin  # noqa: TID252
from .helpers import (
    ConfigurableData,
    maybe_fill_lock_entity,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    sanitize_tool_args,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)


def _map_alarm_service(tool_name: str, requested_state: str) -> str:
    """Map requested alarm state or intent to the HA service."""
    service_map = {
        "arm_home": "alarm_arm_home",
        "home": "alarm_arm_home",
        "armed_home": "alarm_arm_home",
        "arm_away": "alarm_arm_away",
        "away": "alarm_arm_away",
        "armed_away": "alarm_arm_away",
        "arm_night": "alarm_arm_night",
        "night": "alarm_arm_night",
        "armed_night": "alarm_arm_night",
        "vacation": "alarm_arm_vacation",
        "arm_vacation": "alarm_arm_vacation",
        "armed_vacation": "alarm_arm_vacation",
        "custom_bypass": "alarm_arm_custom_bypass",
        "arm_custom_bypass": "alarm_arm_custom_bypass",
        "armed_custom_bypass": "alarm_arm_custom_bypass",
        "disarm": "alarm_disarm",
        "off": "alarm_disarm",
        "disarmed": "alarm_disarm",
    }
    default_service = "alarm_arm_home" if tool_name == "HassTurnOn" else "alarm_disarm"
    return service_map.get(requested_state, default_service)


def _extract_alarm_code(tool_args: dict[str, Any]) -> str:
    """Pull the alarm code from the best available slot."""
    code = str(tool_args.get("code", "")).strip()
    if code:
        return code

    dc_val = tool_args.get("device_class")
    if (
        isinstance(dc_val, list)
        and len(dc_val) == 1
        and str(dc_val[0]).strip().isdigit()
    ):
        code = str(dc_val[0]).strip()
        tool_args["device_class"] = []
    elif isinstance(dc_val, str) and dc_val.strip().isdigit():
        code = dc_val.strip()
        tool_args["device_class"] = []
    elif (floor := tool_args.get("floor")) and str(floor).strip().isdigit():
        code = str(floor).strip()
    elif name := tool_args.get("name"):
        tokens = str(name).strip().split()
        if tokens and tokens[-1].isdigit():
            code = tokens[-1]

    if not code:
        msg = "Alarm code required to arm/disarm."
        raise HomeAssistantError(msg)
    tool_args["code"] = code
    return code


def _resolve_alarm_entity(hass: HomeAssistant, tool_args: dict[str, Any]) -> str:
    """Resolve alarm entity_id from args or the environment."""
    entity_id = tool_args.get("entity_id")
    alarm_entities = hass.states.async_entity_ids("alarm_control_panel")
    if entity_id not in alarm_entities:
        entity_id = None

    if not entity_id and (name := tool_args.get("name")):
        slug = str(name).strip().lower().replace(" ", "_")
        parts = [p for p in slug.split("_") if p]
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        if parts:
            candidate = f"alarm_control_panel.{'_'.join(parts)}"
            if candidate in alarm_entities:
                entity_id = candidate

    if not entity_id and len(alarm_entities) == 1:
        entity_id = alarm_entities[0]

    if not entity_id:
        msg = "Missing alarm entity_id; cannot arm/disarm."
        raise HomeAssistantError(msg)
    return entity_id


def _infer_alarm_state(state_obj: State | None) -> str:
    """Infer a simplified alarm state from the HA state object."""
    if not state_obj:
        return "unknown"
    current_state = str(state_obj.state).lower()
    if current_state in {
        "armed_home",
        "armed_away",
        "armed_night",
        "armed_vacation",
        "armed_custom_bypass",
    }:
        return "armed"
    if current_state in {
        "disarmed",
        "pending",
        "arming",
        "triggered",
        "disarming",
    }:
        return current_state
    return "unknown"


def _alarm_warning(*, is_arm_request: bool, inferred_status: str) -> str | None:
    """Build a warning message if the alarm state is unexpected."""
    if inferred_status == "unknown":
        return "Alarm panel state is unknown after the request."
    if is_arm_request and inferred_status == "disarmed":
        return """
        Arming requested but the panel still shows disarmed; it may still be updating.
        """
    if not is_arm_request and inferred_status != "disarmed":
        return (
            f"Disarm requested but the panel reports {inferred_status}; "
            "please verify at the panel."
        )
    return None


async def _perform_alarm_control(
    hass: HomeAssistant, tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """Arm or disarm an alarm_control_panel entity and report its state."""
    requested_state = str(tool_args.get("state") or "").lower()
    resolved_service = _map_alarm_service(tool_name, requested_state)
    code = _extract_alarm_code(tool_args)
    entity_id = _resolve_alarm_entity(hass, tool_args)

    data: dict[str, Any] = {"entity_id": entity_id, "code": code}
    await hass.services.async_call(
        "alarm_control_panel",
        resolved_service,
        data,
        blocking=True,
    )

    # Give HA a moment to update state before reading it back.
    await asyncio.sleep(2.0)

    state_obj = hass.states.get(entity_id)
    inferred_status = _infer_alarm_state(state_obj)
    is_arm_request = resolved_service.startswith("alarm_arm")
    warning = _alarm_warning(
        is_arm_request=is_arm_request, inferred_status=inferred_status
    )
    result_text = warning or (
        f"Alarm service {resolved_service} completed; panel state: {inferred_status}."
    )
    expected_states = (
        {"armed", "arming", "pending"} if is_arm_request else {"disarmed", "disarming"}
    )

    return {
        "success": warning is None and inferred_status in expected_states,
        "entity_id": entity_id,
        "service": resolved_service,
        "inferred_state": inferred_status,
        "warning": warning,
        "result_text": result_text,
    }


async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes | None:
    """Get an image from a given camera."""
    camera_entity_id: str = f"camera.{camera_name.lower()}"
    state = hass.states.get(camera_entity_id)
    if state and state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        LOGGER.warning(
            "Camera %s is %s; skipping capture", camera_entity_id, state.state
        )
        return None

    max_attempts = 3
    timeout_sec = 5
    backoff_base = 0.35

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            await asyncio.sleep(backoff_base * (attempt - 1))

        try:
            async with async_timeout.timeout(timeout_sec):
                image = await camera.async_get_image(
                    hass=hass,
                    entity_id=camera_entity_id,
                    width=VLM_IMAGE_WIDTH,
                    height=VLM_IMAGE_HEIGHT,
                )
        except TimeoutError:
            LOGGER.warning(
                "Timed out (%ss) getting image from %s (attempt %s/%s)",
                timeout_sec,
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue
        except HomeAssistantError:
            LOGGER.exception(
                "Error getting image from camera %s (attempt %s/%s)",
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue

        if image is None or image.content is None:
            LOGGER.warning(
                "Camera %s returned empty image (attempt %s/%s)",
                camera_entity_id,
                attempt,
                max_attempts,
            )
            continue

        return image.content

    LOGGER.error(
        "Failed to capture image from camera %s after %s attempts",
        camera_entity_id,
        max_attempts,
    )
    return None


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

    LOGGER.debug("Raw VLM model response: %s", resp)

    return extract_final(getattr(resp, "content", "") or "")


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


@tool(parse_docstring=True)
async def confirm_sensitive_action(  # noqa: D417, PLR0911
    action_id: str,
    pin: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],  # noqa: ARG001
) -> str:
    """
    Confirm and execute a pending sensitive action that requires a PIN.

    Args:
        action_id: The action to confirm (provided by agent when it asked for a PIN).
        pin: The user-provided PIN.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    cfg = cast("ConfigurableData", config.get("configurable", {}))
    opts = cfg.get("options", {})
    pin_hash = opts.get(CONF_CRITICAL_ACTION_PIN_HASH, "")
    salt = opts.get(CONF_CRITICAL_ACTION_PIN_SALT, "")
    pending_actions = cfg.get("pending_actions", {})
    provided_pin = str(pin or "").strip()
    requested_action_id = str(action_id or "").strip()

    resolved_action_id = _resolve_action_id(pending_actions, requested_action_id)
    if resolved_action_id is None:
        return "Pending action not found or expired."

    action, action_err = _load_pending_action(pending_actions, resolved_action_id)
    if action_err or not action:
        return action_err or "Pending action not found."

    if _is_wrong_user(cfg, action):
        return "Pending action belongs to a different user; please re-run the request."

    pin_err = _validate_pin_for_action(
        provided_pin=provided_pin,
        pin_hash=pin_hash,
        salt=salt,
        action=action,
    )
    if pin_err:
        return pin_err

    ha_llm_api, api_err = _ensure_api(cfg)
    if api_err or ha_llm_api is None:
        return api_err or "Home Assistant LLM API unavailable."

    result, exec_err = await _execute_pending_action(
        resolved_action_id, action, ha_llm_api, cfg
    )
    return result or exec_err or "Unable to process the confirmation."


def _resolve_action_id(
    pending_actions: dict[str, dict[str, Any]], requested_action_id: str
) -> str | None:
    """Return a valid action_id or None if it cannot be resolved safely."""
    if requested_action_id and requested_action_id in pending_actions:
        return requested_action_id
    if not requested_action_id and len(pending_actions) == 1:
        return next(iter(pending_actions))
    return None


def _load_pending_action(
    pending_actions: dict[str, dict[str, Any]], resolved_action_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and return a pending action."""
    action = pending_actions.get(resolved_action_id)
    if not action:
        return None, "Pending action not found or expired."

    created_at = action.get("created_at")
    if created_at:
        try:
            ts = datetime.fromisoformat(created_at)
        except ValueError:
            pending_actions.pop(resolved_action_id, None)
            return None, "Pending action is invalid; please try again."
        if dt_util.utcnow() - ts > timedelta(minutes=10):
            pending_actions.pop(resolved_action_id, None)
            return None, "Pending action expired; please re-run the request."

    action.setdefault("attempts", 0)
    return action, None


def _validate_pin_for_action(
    *, provided_pin: str, pin_hash: str, salt: str, action: dict[str, Any]
) -> str | None:
    """Validate PIN format and value against stored hash/salt."""
    max_pin_attempts = 5
    if not pin_hash or not salt:
        return "No PIN configured; cannot confirm the action."
    if not provided_pin.isdigit() or not (
        CRITICAL_PIN_MIN_LEN <= len(provided_pin) <= CRITICAL_PIN_MAX_LEN
    ):
        return f"Invalid PIN. Use {CRITICAL_PIN_MIN_LEN}-{CRITICAL_PIN_MAX_LEN} digits."
    attempts = int(action.get("attempts", 0) or 0)
    if attempts >= max_pin_attempts:
        return "Too many incorrect attempts; please re-run the request."
    if not verify_pin(provided_pin, hashed=pin_hash, salt=salt):
        action["attempts"] = attempts + 1
        return "Incorrect PIN. Action not executed."
    return None


def _ensure_api(cfg: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    """Return the HA LLM API or an error message."""
    ha_llm_api = cfg.get("ha_llm_api")
    if ha_llm_api is None:
        return None, "Home Assistant LLM API unavailable."
    return ha_llm_api, None


def _is_wrong_user(cfg: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    requester_id = cfg.get("user_id")
    action_owner = action.get("user")
    return bool(requester_id and action_owner and requester_id != action_owner)


async def _execute_pending_action(
    resolved_action_id: str,
    action: dict[str, Any],
    ha_llm_api: Any,
    cfg: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Normalize args, execute the pending action, and clear it."""
    raw_tool_name = action.get("tool_name")
    if not isinstance(raw_tool_name, str) or not raw_tool_name:
        return None, "Pending action is invalid; missing tool name."
    tool_name = raw_tool_name
    tool_args = action.get("tool_args") or {}
    tool_args = normalize_intent_for_alarm(tool_name, tool_args)
    tool_args = normalize_intent_for_lock(tool_name, tool_args)
    tool_args = maybe_fill_lock_entity(tool_args, cfg.get("hass"))
    tool_args = sanitize_tool_args(tool_args)
    try:
        tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
        response = await ha_llm_api.async_call_tool(tool_input)
    except (HomeAssistantError, vol.Invalid) as err:
        return None, f"Failed to execute action: {err!r}"

    pending_actions = cfg.get("pending_actions", {})
    pending_actions.pop(resolved_action_id, None)
    return json.dumps(
        {"status": "completed", "action_id": resolved_action_id, "result": response}
    ), None


@tool(parse_docstring=True)
async def alarm_control(  # noqa: D417, PLR0913
    name: str | None = None,
    entity_id: str | None = None,
    state: str | None = None,
    code: str | None = None,
    *,
    bypass_open_sensors: bool | None = None,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Arm or disarm an alarm control panel using the alarm system code (not the PIN).

    Args:
        name: Friendly name of the alarm panel (for example, "Home Alarm").
        entity_id: Specific alarm entity_id (if known). If not provided, the tool will
            try to resolve it from name or fall back to the only alarm entity.
        state: Desired target state. Examples: "arm_home", "armed_home", "arm_away",
            "armed_away", "disarm", "disarmed". If omitted, defaults to "arm_home".
        code: The alarm panel code (required by most alarm integrations).
        bypass_open_sensors: If True, request bypass of open sensors (if supported).

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass: HomeAssistant = config["configurable"]["hass"]
    tool_args: dict[str, Any] = {
        "name": name,
        "entity_id": entity_id,
        "state": state,
        "code": code,
        "bypass_open_sensors": bypass_open_sensors,
    }
    # Drop None/empty values so the alarm helper can apply its own defaults.
    tool_args = {k: v for k, v in tool_args.items() if v not in (None, "")}

    try:
        result = await _perform_alarm_control(hass, "alarm_control", tool_args)
    except HomeAssistantError as err:
        return f"Error controlling alarm: {err}"
    return json.dumps(result)


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
