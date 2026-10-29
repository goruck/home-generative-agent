"""Langgraph tools for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import aiofiles
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
from langgraph.prebuilt.tool_node import InjectedStore
from langgraph.store.base import BaseStore  # noqa: TC002
from ollama import ResponseError as OllamaResponseError
from voluptuous import MultipleInvalid

from ..const import (  # noqa: TID252
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_NOTIFY_SERVICE,
    CONF_VLM_PROMPT_EXTRA,
    CONF_VLM_RESPONSE_LANGUAGE,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    HISTORY_TOOL_CONTEXT_LIMIT,
    HISTORY_TOOL_PURGE_KEEP_DAYS,
    NETWORK_AUDIT_REPORT_MAX_CHARS,
    NETWORK_AUDIT_REPORT_NOTIFICATION_ID,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE_NOT_ADMIN,
    NETWORK_AUDIT_TOOL_DIGEST_NOTE_UNDELIVERED,
    NETWORK_AUDIT_TOOL_LABEL_NOTE,
    NETWORK_AUDIT_TOOL_MAX_ENTITIES,
    NETWORK_AUDIT_TOOL_MAX_NOTES,
    NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS,
    RECOMMENDED_VLM_PROMPT_EXTRA,
    RECOMMENDED_VLM_RESPONSE_LANGUAGE,
    VLM_IMAGE_HEIGHT,
    VLM_IMAGE_WIDTH,
    VLM_SYSTEM_PROMPT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)
from ..core.conversation_helpers import _resolve_entity_id  # noqa: TID252
from ..core.fallback import (  # noqa: TID252
    ProviderStalledError,
    ainvoke_dropping_unsupported_params,
)
from ..core.utils import extract_final, verify_pin  # noqa: TID252
from ..sentinel.network_audit import (  # noqa: TID252
    NetworkAuditReport,
)
from ..sentinel.network_audit import (  # noqa: TID252
    digest as network_audit_digest,
)
from ..sentinel.network_audit import (  # noqa: TID252
    render_report_markdown as render_network_audit_markdown,
)
from ..sentinel.network_audit import (  # noqa: TID252
    summarize as summarize_network_audit,
)
from ..sentinel.notifier import escape_markdown  # noqa: TID252
from ..sentinel.notifier_messages import notif_msg  # noqa: TID252
from ..sentinel.redaction import redact_network_identifiers  # noqa: TID252
from .automation_pin import find_critical_automation_calls
from .automation_targets import (
    AUTOMATION_REFUSAL_PREFIX,
    describe_missing_targets,
    exposed_to_assist,
    exposure_available,
    find_missing_automation_targets,
    normalize_notify_service,
    service_exists,
)
from .camera_activity import get_camera_last_events_from_states
from .critical_push import CriticalPushResult, apply_critical_push
from .helpers import (
    PENDING_ACTION_TTL,
    PIN_AWAITING_USER_REPLY,
    PIN_OUTCOME_DONE,
    PIN_OUTCOME_FAILED,
    PIN_OUTCOME_UNKNOWN,
    PIN_REFUSAL_ENDS_TURN,
    ConfigurableData,
    base_tool_name,
    drop_empty_args,
    keep_pin_prompt,
    minimal_payload_for_domain,
    narrow_intent_args,
    normalize_intent_for_alarm,
    normalize_intent_for_lock,
    parse_tool_response,
    pin_digits,
    pin_in_text,
    register_pending_action,
    resolve_critical_action_policy,
    sanitize_tool_args,
    sanitize_tool_text,
    tool_call_data,
)
from .indirect_pin import (
    BUILTIN_INTENT,
    OTHER_INTENT,
    stored_action_calls,
    tool_args_kind,
    tool_origin,
)
from .on_off_direction import with_action_performed
from .pin_messages import describe_pending_action, pin_msg
from .rule_entities import expand_critical_rules

if TYPE_CHECKING:
    from collections.abc import Generator

    from homeassistant.core import HomeAssistant
    from langchain_core.language_models import BaseMessage, LanguageModelInput
    from langchain_core.runnables.base import RunnableSerializable

LOGGER = logging.getLogger(__name__)

# Limit on a PIN confirmation, counted from the start of the tool call. Below
# the graph's per-tool timeout (graph.py, 30 s), so an action that is slow
# after its claim is answered as "may have run" here.
POST_CLAIM_TIMEOUT_S = 25

# Wrong PINs allowed for one pending action, carried over to a request that
# replaces it.
MAX_PIN_ATTEMPTS = 5


@tool(parse_docstring=True)
async def resolve_entity_ids(  # noqa: D417
    entity_ids: list[str],
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> dict[str, str]:
    """
    Resolve suggested entity_ids to existing Home Assistant entity_ids.

    Args:
        entity_ids: List of entity_id strings.

    """
    if "configurable" not in config:
        return {}

    hass: HomeAssistant = config["configurable"]["hass"]
    # A fuzzy match may only name an entity exposed to Assist (issue #716),
    # like add_automation's suggestions: otherwise a near-miss id would
    # confirm that a hidden device exists. Without the exposure registry no
    # entity can be shown to be exposed, so no fuzzy match is offered.
    can_check_exposure = exposure_available(hass)

    def _allow(entity_id: str) -> bool:
        return can_check_exposure and exposed_to_assist(hass, entity_id)

    resolved: dict[str, str] = {}
    for entity_id in entity_ids:
        if isinstance(entity_id, str):
            resolved[entity_id] = _resolve_entity_id(entity_id, hass, _allow)
    return resolved


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
    is_arm_intent = base_tool_name(tool_name) == "HassTurnOn"
    default_service = "alarm_arm_home" if is_arm_intent else "alarm_disarm"
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


def _normalize_camera_name(name: str) -> str:
    """
    Normalize a camera name for entity matching.

    Strips diacritics, lowercases, and removes everything that isn't
    alphanumeric, so "Kamera obývák 2", "kamera_obyvak_2" and "kamera obyvak2"
    all normalize to the same value.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    normalized = re.sub(r"[^a-z0-9]", "", ascii_name.lower())
    unicode_form = "".join(
        c for c in unicodedata.normalize("NFKC", name).casefold() if c.isalnum()
    )
    # Non-Latin characters (Cyrillic, CJK, Greek, ...) vanish under the ASCII
    # fold, so "Камера улица 1" would collapse to just "1" and cross-match any
    # other digit-bearing foreign name. Whenever the fold discarded characters
    # the unicode form kept, use the lossless casefolded unicode form instead.
    # Latin diacritics fold 1:1 (obývák -> obyvak), so equal lengths keep the
    # diacritic-insensitive ASCII behavior.
    if len(normalized) == len(unicode_form):
        return normalized
    return unicode_form


def _resolve_camera_entity_id(hass: HomeAssistant, camera_name: str) -> str | None:
    """
    Resolve a free-form camera name from the LLM to a real camera entity_id.

    The LLM produces friendly names as spoken by the user (e.g. "kamera
    obývák 2"), which frequently differ from the entity_id slug (e.g.
    "camera.kamera_obyvak_2"). Tries, in order: the literal entity_id, the
    legacy lowercase-prefixed form, then a diacritics-stripped alphanumeric
    match against every camera's friendly name (preferred) or entity_id slug.
    """
    lowered = camera_name.lower()
    candidate = lowered if lowered.startswith("camera.") else f"camera.{lowered}"
    if hass.states.get(candidate) is not None:
        return candidate

    normalized_target = _normalize_camera_name(lowered.removeprefix("camera."))
    if not normalized_target:
        return None
    friendly_matches: list[str] = []
    slug_matches: list[str] = []
    for cam_state in hass.states.async_all("camera"):
        # Coerce: custom integrations can set friendly_name to None.
        friendly = str(cam_state.attributes.get(ATTR_FRIENDLY_NAME) or "")
        if _normalize_camera_name(friendly) == normalized_target:
            friendly_matches.append(cam_state.entity_id)
            continue
        slug = cam_state.entity_id.removeprefix("camera.")
        if _normalize_camera_name(slug) == normalized_target:
            slug_matches.append(cam_state.entity_id)

    matches = friendly_matches or slug_matches
    if len(matches) > 1:
        # Two cameras normalize to the same name: picking one by state-machine
        # iteration order could silently analyze the wrong room. Refuse and
        # let the tool return the camera list so the LLM can disambiguate.
        LOGGER.warning(
            "Camera name %r is ambiguous; matches %s. Not resolving.",
            camera_name,
            matches,
        )
        return None
    return matches[0] if matches else None


# Cap the not-found camera hint: an unbounded list could exceed the model
# context on large installations and turn a harmless miss into a failed turn.
_AVAILABLE_CAMERA_NAMES_MAX = 25


def _available_camera_names(hass: HomeAssistant) -> str:
    """Return a comma-separated list of camera friendly names for the LLM."""
    names = sorted(
        str(s.attributes.get(ATTR_FRIENDLY_NAME) or s.entity_id)
        for s in hass.states.async_all("camera")
    )
    if len(names) > _AVAILABLE_CAMERA_NAMES_MAX:
        hidden = len(names) - _AVAILABLE_CAMERA_NAMES_MAX
        names = [*names[:_AVAILABLE_CAMERA_NAMES_MAX], f"and {hidden} more"]
    return ", ".join(names)


async def _get_camera_image(hass: HomeAssistant, camera_entity_id: str) -> bytes | None:
    """Get an image from a given camera entity."""
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
            async with asyncio.timeout(timeout_sec):
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


# Returned by analyze_image when the VLM call fails with HomeAssistantError.
# Callers that chain frame context (video_analyzer._process_batch) must never
# treat this caption as a real scene description.
VLM_ERROR_CAPTION = "Error analyzing image with VLM model."


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


@dataclass(frozen=True)
class VLMPromptOverrides:
    """
    User-configurable additions layered onto VLM_SYSTEM_PROMPT.

    Both fields are optional and appended after VLM_SYSTEM_PROMPT rather
    than replacing it, so the Repeated-scene / Brevity / Motion-description
    rules and their sentinel-detection contract stay intact regardless of
    what a user adds here. See CONF_VLM_RESPONSE_LANGUAGE and
    CONF_VLM_PROMPT_EXTRA in const.py.
    """

    response_language: str | None = None
    prompt_extra: str | None = None


async def analyze_image(
    vlm_model: RunnableSerializable[LanguageModelInput, BaseMessage],
    image: bytes,
    detection_keywords: list[str] | None = None,
    prev_text: str | None = None,
    overrides: VLMPromptOverrides | None = None,
) -> str:
    """Analyze an image with the preconfigured VLM model."""
    await asyncio.sleep(0)  # keep the event loop snappy

    image_data = base64.b64encode(image).decode("utf-8")

    if detection_keywords is not None:
        prompt = VLM_USER_KW_TEMPLATE.format(key_words=" or ".join(detection_keywords))
    else:
        prompt = VLM_USER_PROMPT

    system_prompt = VLM_SYSTEM_PROMPT
    if overrides and overrides.response_language:
        sentinel_note = 'the exact sentinel reply "Scene unchanged."'
        system_prompt += (
            f"\n\nRespond in {overrides.response_language} for all descriptive "
            f"text. Exception: {sentinel_note} defined above under the "
            "Repeated-scene rule must stay in English, word for word, "
            "whenever that rule applies — it is matched by code, not read "
            "by a person, so do not translate or paraphrase it."
        )
    if overrides and overrides.prompt_extra:
        system_prompt += (
            "\n\nUser instruction (highest priority — follow this even "
            "where it conflicts with or narrows the rules above, except "
            'the "Scene unchanged." sentinel contract, which always '
            f"applies as written):\n{overrides.prompt_extra}"
        )
        # Chat-tuned VLMs weight the user turn's description request over
        # anything stated only in the system prompt, so the override must be
        # restated on the request side to reliably take effect. Skipped on
        # the keyword path: detection keywords come from the live chat
        # request, which outranks stored config.
        if detection_keywords is None:
            prompt += (
                "\n\nApply this instruction to the request above; where "
                "they conflict, this instruction takes precedence: "
                f"{overrides.prompt_extra}"
            )

    messages = _prompt_func(
        {
            "system": system_prompt,
            "text": prompt,
            "image": image_data,
            "prev_text": prev_text,
        }
    )

    # ollama.ResponseError (e.g. a configured model that was never pulled)
    # intentionally propagates: callers decide whether to skip the frame,
    # return a message to the LLM, or surface a HomeAssistantError. Returning
    # an error string here would let it masquerade as a real caption.
    try:
        resp = await ainvoke_dropping_unsupported_params(vlm_model, messages)
    except (TimeoutError, ProviderStalledError):
        # A chain timeout or a stalled provider is the caller's to count and
        # report (the video analyzer's timeout metric and failure log), not a
        # caption (#828). Both are HomeAssistantErrors, so this comes first.
        raise
    except HomeAssistantError:
        LOGGER.exception(VLM_ERROR_CAPTION)
        return VLM_ERROR_CAPTION

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
    options = config["configurable"].get("options", {})
    camera_entity_id = _resolve_camera_entity_id(hass, camera_name)
    if camera_entity_id is None:
        LOGGER.error(
            "Could not resolve camera name %r to a known camera entity", camera_name
        )
        available = _available_camera_names(hass)
        if available:
            return (
                f"Camera '{camera_name}' was not found. Available cameras: {available}."
            )
        return f"Camera '{camera_name}' was not found. No cameras are available."
    image = await _get_camera_image(hass, camera_entity_id)
    if image is None:
        return "Error getting image from camera."
    try:
        overrides = VLMPromptOverrides(
            response_language=options.get(
                CONF_VLM_RESPONSE_LANGUAGE, RECOMMENDED_VLM_RESPONSE_LANGUAGE
            ),
            prompt_extra=options.get(
                CONF_VLM_PROMPT_EXTRA, RECOMMENDED_VLM_PROMPT_EXTRA
            ),
        )
        return await analyze_image(
            vlm_model, image, detection_keywords, overrides=overrides
        )
    except (OllamaResponseError, TimeoutError, ProviderStalledError) as err:
        if isinstance(err, OllamaResponseError):
            LOGGER.exception(VLM_ERROR_CAPTION)
        else:
            # A timeout or a stalled provider: the reason, without a traceback.
            LOGGER.warning("%s %s", VLM_ERROR_CAPTION, err)
        return VLM_ERROR_CAPTION


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

    Entity IDs and service names must be real. Entity IDs are not the names
    shown in the home overview: use ones you have seen in a tool result, or
    the corrections this tool returns. State triggers compare raw states:
    binary sensors, switches and lights are "on" or "off" (never labels like
    Wet, Open or Detected) and people are "home" or "not_home". Mobile push
    goes through the notify.mobile_app_* service configured for this
    integration; if you do not know it, the tool tells you when you get it
    wrong.

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
    raw_push_service = config["configurable"]["options"].get(CONF_NOTIFY_SERVICE)
    mobile_push_service = (
        normalize_notify_service(str(raw_push_service)) if raw_push_service else None
    )

    if time_pattern and message:
        # The blueprint calls the push service through a template, which the
        # existence check below cannot read, so check the configured value
        # here: a phone that was renamed would otherwise install a camera
        # automation whose push silently fails.
        if mobile_push_service and not service_exists(hass, mobile_push_service):
            return (
                f"{AUTOMATION_REFUSAL_PREFIX}: the mobile push service configured for "
                f"this integration, '{sanitize_tool_text(mobile_push_service)}', "
                "is not a Home Assistant service, so the camera automation's "
                "notification could never be delivered. This is a configuration "
                "problem, not something to retry: tell the user to pick a "
                "current notify service in this integration's options (the "
                "camera analysis or Sentinel notification setting) and to ask "
                "again afterwards."
            )
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
        validated_config = await _async_validate_config_item(
            hass=hass,
            config=ha_automation_config,
            raise_on_errors=True,
            warn_on_errors=False,
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        return f"Invalid automation configuration {err}"

    # Schema validation passes an automation whose trigger names an entity
    # that does not exist or whose action calls a service that does not, and
    # such an automation installs and then silently never runs. Refuse it and
    # hand the model the closest real names so its retry is deterministic.
    missing = find_missing_automation_targets(
        hass, validated_config, notify_service=mobile_push_service
    )
    if missing:
        LOGGER.info(
            "add_automation refused '%s': %s",
            sanitize_tool_text(str(ha_automation_config.get("alias", "<unnamed>"))),
            ", ".join(item.name or item.kind for item in missing),
        )
        return describe_missing_targets(missing)

    # A leak, smoke, gas or CO alert must not sit unseen in a notification
    # group or behind a Focus mode (#869). Rewrite the config that is written
    # (and held for the PIN), not the validated copy.
    critical_push = apply_critical_push(hass, ha_automation_config, validated_config)
    if (
        critical_push.rewritten
        or critical_push.left_ordinary
        or critical_push.unchecked
    ):
        LOGGER.info(
            "add_automation made %d push(es) in '%s' critical alerts; %d left "
            "ordinary (not a mobile_app service, or a templated payload)%s",
            critical_push.rewritten,
            sanitize_tool_text(str(ha_automation_config.get("alias", "<unnamed>"))),
            critical_push.left_ordinary,
            "; too deeply nested to check" if critical_push.unchecked else "",
        )
    install_note = _critical_push_note(critical_push)

    pin_challenge = _maybe_require_pin_for_automation(
        cfg=cast("ConfigurableData", config["configurable"]),
        validated_config=validated_config,
        ha_automation_config=ha_automation_config,
        install_note=install_note,
    )
    if pin_challenge:
        return pin_challenge

    return await _async_write_automation(hass, ha_automation_config) + install_note


def _critical_push_note(critical_push: CriticalPushResult) -> str:
    """Return what the install reply adds about a safety alarm's notifications."""
    note = ""
    if critical_push.rewritten:
        note += (
            ". Its phone notifications are sent as critical alerts, which sound "
            "even in Do Not Disturb or a Focus mode; on iPhone the Home Assistant "
            "app needs Critical Alerts allowed in its notification settings"
        )
    if critical_push.left_ordinary or critical_push.unchecked:
        reason = (
            "nests its steps too deeply to check its notifications"
            if critical_push.unchecked
            else f"{critical_push.left_ordinary} of its notifications are not sent "
            "through a phone's notify.mobile_app_* service (or use a templated "
            "payload)"
        )
        # The automation is already installed: another add_automation call
        # would install a second one next to it.
        note += (
            ". This automation fires on a leak, smoke, gas, carbon monoxide or "
            f"safety alarm, but it {reason}, so those are not critical alerts. "
            "It is installed as written; do not add it again. Tell the user, "
            "who can switch those notifications to a phone's notify.mobile_app_* "
            "service in Settings > Automations"
        )
    return f"{note}." if note else ""


def _maybe_require_pin_for_automation(
    *,
    cfg: ConfigurableData,
    validated_config: Mapping[str, Any],
    ha_automation_config: dict[str, Any],
    install_note: str = "",
) -> str | None:
    """
    Register a pending PIN confirmation when an automation is critical.

    Returns the ``requires_pin`` payload the model must resolve with
    ``confirm_sensitive_action``, or None when the automation may be written
    immediately.
    """
    policy = resolve_critical_action_policy(cfg.get("options", {}))
    if not policy.enabled:
        return None

    rules = expand_critical_rules(cfg.get("hass"), policy.critical_actions)
    critical_calls = find_critical_automation_calls(validated_config, rules)
    if not critical_calls:
        # A step on a template switch or an input button runs its stored
        # actions, which the config itself does not show (#780).
        actions = validated_config.get("actions", validated_config.get("action"))
        critical_calls = stored_action_calls(cfg.get("hass"), actions, rules)
    if not critical_calls:
        return None

    if not policy.enforceable:
        # Unlike a direct tool call, which the guard lets through with a
        # warning, an automation is durable — it keeps firing. Refuse instead.
        LOGGER.warning(
            "Critical action PIN is enabled but no PIN is configured. "
            "Refusing to install an automation containing %s. "
            "Set a PIN in the integration options to enforce confirmation.",
            ", ".join(call.describe() for call in critical_calls),
        )
        return pin_msg(cfg.get("hass"), "automation_pin_unset")

    pending_actions = cfg.get("pending_actions")
    if pending_actions is None:
        LOGGER.error("Cannot gate critical automation: no pending action store.")
        return pin_msg(cfg.get("hass"), "automation_pending_store_unavailable")

    action = {
        "tool_name": "add_automation",
        "tool_args": {},
        "automation_config": ha_automation_config,
        # What made it critical, so the fixed line names the unlock inside
        # it, not only the alias the model chose.
        "critical_calls": [call.describe() for call in critical_calls],
        # So the confirmed reply can say which pushes are critical alerts.
        "install_note": install_note,
        "user": cfg.get("user_id"),
        "thread_id": cfg.get("thread_id"),
        "turn_id": _turn_id(cfg),
    }
    action_id = register_pending_action(pending_actions, action)
    if action_id is None:
        return pin_msg(cfg.get("hass"), "pin_one_action_at_a_time")
    return json.dumps(
        {
            "status": "requires_pin",
            "action_id": action_id,
            "action": describe_pending_action(cfg.get("hass"), action),
            "reason": pin_msg(
                cfg.get("hass"),
                "automation_requires_pin_reason",
                reason=critical_calls[0].reason,
                calls=", ".join(call.describe() for call in critical_calls),
            ),
        }
    )


async def _async_write_automation(
    hass: HomeAssistant, ha_automation_config: dict[str, Any]
) -> str:
    """Append a validated automation to automations.yaml and reload."""
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
async def write_yaml_file(  # noqa: D417
    yaml_text: Any,
    filename_prefix: str = "hga",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Write YAML content to a unique file under /config/www.

    Args:
        yaml_text: The YAML content to write.
        filename_prefix: Optional filename prefix (letters, numbers, '-' or '_').

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    config_dir = Path(hass.config.config_dir)
    www_dir = config_dir / "www"
    await asyncio.to_thread(www_dir.mkdir, parents=True, exist_ok=True)

    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_prefix).strip("_")
    if not safe_prefix:
        safe_prefix = "hga"
    filename = f"{safe_prefix}_{ulid.ulid_now()}.yaml"
    path = www_dir / filename

    code_fence_min_lines = 2
    if isinstance(yaml_text, (dict, list)):
        text = yaml.dump(
            yaml_text, allow_unicode=True, sort_keys=False, default_flow_style=False
        ).strip()
    else:
        text = str(yaml_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) >= code_fence_min_lines
            and lines[0].startswith("```")
            and lines[-1].startswith("```")
        ):
            text = "\n".join(lines[1:-1]).strip()

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(text.rstrip() + "\n")

    return f"Saved YAML to {path}. Access it at /local/{filename}"


@tool(parse_docstring=True)
async def confirm_sensitive_action(  # noqa: D417
    action_id: str,
    pin: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],  # noqa: ARG001
) -> str:
    """
    Confirm and execute a pending sensitive action that requires a PIN.

    Only call this tool after receiving a tool response with status "requires_pin".
    That response contains the action_id to pass here. Do not call this tool
    speculatively or before the action tool has returned a "requires_pin" status.
    Only the user's reply to the PIN prompt can confirm, with the PIN written in
    that reply, and it confirms one action.

    Args:
        action_id: The action ID from the "requires_pin" tool response.
        pin: The PIN provided by the user.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    # The action must finish inside the graph's own tool timeout, counted from
    # here, so a slow device gets the "may have run" reply rather than a
    # timeout that reads as still pending.
    deadline = asyncio.get_running_loop().time() + POST_CLAIM_TIMEOUT_S
    cfg = cast("ConfigurableData", config.get("configurable", {}))
    hass = cfg.get("hass")
    opts = cfg.get("options", {})
    pin_hash = opts.get(CONF_CRITICAL_ACTION_PIN_HASH, "")
    salt = opts.get(CONF_CRITICAL_ACTION_PIN_SALT, "")
    pending_actions = cfg.get("pending_actions", {})
    raw_pin = str(pin or "").strip()
    # The model may pass the PIN as the user said it ("1 2 3 4", "one two
    # three four", "PIN 1234", full-width digits). The message check and the
    # hash check must see the same ASCII digits, which is also how setup
    # stores the PIN.
    provided_pin = pin_digits(raw_pin) or raw_pin
    requested_action_id = str(action_id or "").strip()

    resolved_action_id = _resolve_action_id(
        pending_actions, requested_action_id, cfg.get("thread_id")
    )
    if resolved_action_id is None:
        return pin_msg(hass, "pending_action_not_found_or_expired")

    action, action_err = _load_pending_action(pending_actions, resolved_action_id, hass)
    if action_err or not action:
        return action_err or pin_msg(hass, "pending_action_not_found")

    refusal, pin_not_in_message = _confirmation_refusal(
        cfg, action, resolved_action_id, provided_pin
    )
    if refusal:
        if pin_not_in_message:
            await _warn_if_correct_pin_refused(cfg, provided_pin)
        return refusal

    attempts_before = action.get("attempts", 0)
    pin_err = _validate_pin_for_action(
        provided_pin=provided_pin,
        pin_hash=pin_hash,
        salt=salt,
        action=action,
        hass=hass,
    )
    if pin_err:
        attempts = action.get("attempts", 0)
        if attempts != attempts_before:
            # A wrong PIN: one message is one attempt, so it cannot be
            # retried from the same message. A value that never reached the
            # hash (wrong format) leaves the message usable.
            _spend_pin_reply(cfg)
        if attempts < MAX_PIN_ATTEMPTS and pin_hash and salt:
            # A typo or a misheard PIN: the user can answer again in their
            # next message, unless the prompt has taken all its replies.
            kept = keep_pin_prompt(action, cfg.get("pin_reply"))
            hint = "pin_retry_next_message" if kept else "pin_request_again"
        else:
            # The request's last attempt: the prompt is gone with it.
            hint = "pin_request_again"
        return f"{pin_err} {pin_msg(hass, hint)}"

    result, exec_err = await _execute_pending_action(
        resolved_action_id, action, cfg, deadline
    )
    return result or exec_err or pin_msg(hass, "confirmation_unable_to_process")


def _resolve_action_id(
    pending_actions: dict[str, dict[str, Any]],
    requested_action_id: str,
    thread_id: str | None,
) -> str | None:
    """Return a valid action_id or None if it cannot be resolved safely."""
    if not thread_id:
        return None
    if requested_action_id:
        action = pending_actions.get(requested_action_id)
        if action and action.get("thread_id") == thread_id:
            return requested_action_id
    # Without an id, or with one this conversation no longer holds (a
    # re-issued request replaces its action under a new id, and the model may
    # still hold the old one), take the conversation's only pending action.
    # The store is shared by every conversation, and satellites and
    # automations share one user, so another conversation's action must
    # never be picked up here: its id reads as missing like any other. This
    # only picks the action; `_confirmation_refusal` still requires the
    # user's reply to that action's prompt, with the PIN in it, unspent.
    own = [
        key
        for key, action in pending_actions.items()
        if action.get("thread_id") == thread_id
    ]
    return own[0] if len(own) == 1 else None


def _load_pending_action(
    pending_actions: dict[str, dict[str, Any]],
    resolved_action_id: str,
    hass: HomeAssistant | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and return a pending action."""
    action = pending_actions.get(resolved_action_id)
    if not action:
        return None, pin_msg(hass, "pending_action_not_found_or_expired")

    created_at = action.get("created_at")
    if created_at:
        try:
            ts = datetime.fromisoformat(created_at)
            expired = dt_util.utcnow() - ts > PENDING_ACTION_TTL
        except (ValueError, TypeError):
            # TypeError covers a timezone-naive stored timestamp.
            pending_actions.pop(resolved_action_id, None)
            return None, pin_msg(hass, "pending_action_invalid")
        if expired:
            pending_actions.pop(resolved_action_id, None)
            return None, pin_msg(hass, "pending_action_expired")

    action.setdefault("attempts", 0)
    return action, None


def _validate_pin_for_action(
    *,
    provided_pin: str,
    pin_hash: str,
    salt: str,
    action: dict[str, Any],
    hass: HomeAssistant | None = None,
) -> str | None:
    """Validate PIN format and value against stored hash/salt."""
    if not pin_hash or not salt:
        return pin_msg(hass, "pin_not_configured")
    if not provided_pin.isdigit() or not (
        CRITICAL_PIN_MIN_LEN <= len(provided_pin) <= CRITICAL_PIN_MAX_LEN
    ):
        return pin_msg(
            hass,
            "pin_invalid_format",
            min_len=CRITICAL_PIN_MIN_LEN,
            max_len=CRITICAL_PIN_MAX_LEN,
        )
    attempts = int(action.get("attempts", 0) or 0)
    if attempts >= MAX_PIN_ATTEMPTS:
        return pin_msg(hass, "pin_too_many_attempts")
    if not verify_pin(provided_pin, hashed=pin_hash, salt=salt):
        action["attempts"] = attempts + 1
        return pin_msg(hass, "pin_incorrect")
    return None


def _ensure_api(cfg: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    """Return the HA LLM API or an error message."""
    ha_llm_api = cfg.get("ha_llm_api")
    if ha_llm_api is None:
        return None, "Home Assistant LLM API unavailable."
    return ha_llm_api, None


def _confirmation_refusal(
    cfg: Mapping[str, Any],
    action: dict[str, Any],
    action_id: str,
    provided_pin: str,
) -> tuple[str | None, bool]:
    """
    Return why this caller may not confirm the action, before any PIN check.

    The flag is True when the refusal is a PIN that is not in the user's
    message, which the caller checks for the diagnostic warning. That refusal
    also keeps the prompt for the user's next message: they answered, but
    their PIN could not be read.
    """
    hass = cfg.get("hass")
    if _is_wrong_user(cfg, action):
        return pin_msg(hass, "pending_action_wrong_user"), False
    thread_id = cfg.get("thread_id")
    if not thread_id or action.get("thread_id") != thread_id:
        # _resolve_action_id already refuses another conversation's action;
        # kept so this check never depends on how the id was resolved.
        return pin_msg(hass, "pending_action_not_found_or_expired"), False
    # Refused before the PIN is checked, so no attempt is counted.
    pin_reply = cfg.get("pin_reply")
    not_in_message = False
    ends_turn = False
    hint = None
    if pin_reply is None or action.get("reply_turn") != pin_reply.turn_id:
        # Only the user's reply to this action's prompt can confirm it: not
        # the message that started it (a request made, or injected, during
        # this turn), and not a later one (the user moved on).
        # The user has not been asked yet. The reply asks in fixed words, so
        # the turn ends here (graph.py, `_after_tools`) instead of going back
        # to a model that may ask again, or try another PIN of its own. Not
        # when this message's PIN was already used (an action ran, or a PIN
        # failed) and the model re-issued a request: the model's reply is
        # where the user learns what happened first. The text the model reads
        # next turn, right before the user's PIN, says so: "ask the user now"
        # would be stale there, and a model that obeyed it would re-ask
        # instead of confirming, which drops the prompt.
        ends_turn = pin_reply is not None and not pin_reply.spent
        reason = "pin_reply_next_message" if ends_turn else "pin_reply_to_prompt_only"
    elif not pin_in_text(provided_pin, pin_reply.text):
        # The PIN came from the model (guessed, or reused from earlier in the
        # conversation), not from the user's message for this turn.
        not_in_message = True
        reason = "pin_awaiting_user_reply"
        kept = keep_pin_prompt(action, pin_reply)
        hint = "pin_say_digits" if kept else "pin_request_again"
    elif pin_reply.spent:
        reason = "pin_reply_used"
    else:
        return None, False
    text = pin_msg(hass, reason)
    if hint:
        text = f"{text} {pin_msg(hass, hint)}"
    payload: dict[str, Any] = {
        "status": PIN_AWAITING_USER_REPLY,
        "action_id": action_id,
        "reason": text,
    }
    if ends_turn:
        payload[PIN_REFUSAL_ENDS_TURN] = True
    return json.dumps(payload), not_in_message


async def _warn_if_correct_pin_refused(
    cfg: Mapping[str, Any], provided_pin: str
) -> None:
    """
    Log a correct PIN refused because it is not in the user's message.

    Either the model reused the PIN from earlier in the conversation, or the
    user said it in a form the matcher does not read, which would otherwise
    leave them asked for the PIN again with nothing to report.
    """
    pin_reply = cfg.get("pin_reply")
    opts = cfg.get("options", {})
    pin_hash = opts.get(CONF_CRITICAL_ACTION_PIN_HASH, "")
    salt = opts.get(CONF_CRITICAL_ACTION_PIN_SALT, "")
    if (
        pin_reply is None
        or pin_reply.refusal_checked
        or not (pin_hash and salt and provided_pin.isdigit())
        or not CRITICAL_PIN_MIN_LEN <= len(provided_pin) <= CRITICAL_PIN_MAX_LEN
    ):
        return
    # Only a value that could be the PIN uses up the one hash per message, so
    # a junk first call does not hide a correct PIN refused later.
    pin_reply.refusal_checked = True
    check = partial(verify_pin, provided_pin, hashed=pin_hash, salt=salt)
    # The hash is slow, and this only feeds a log line: keep it off the loop.
    hass = cfg.get("hass")
    correct = await hass.async_add_executor_job(check) if hass else check()
    if correct:
        LOGGER.warning(
            "Refused a correct critical-action PIN that is not in the user's "
            "current message: the model took it from earlier in the conversation, "
            "or the user's PIN was transcribed in a form that is not read as "
            "digits. Ask the user to say or type the PIN digits."
        )


def _is_wrong_user(cfg: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    requester_id = cfg.get("user_id")
    action_owner = action.get("user")
    return bool(requester_id and action_owner and requester_id != action_owner)


def _turn_id(cfg: Mapping[str, Any]) -> str | None:
    """Return this turn's id, which a pending action records as its own."""
    pin_reply = cfg.get("pin_reply")
    return pin_reply.turn_id if pin_reply is not None else None


def _claim_pending_action(cfg: Mapping[str, Any], action_id: str) -> bool:
    """
    Take the action out of the store and spend the user's PIN message on it.

    Called right before the action runs, with nothing awaited in between, so
    one confirmation runs the action at most once (also across confirmations
    in one asyncio.gather batch) and one PIN message runs one action: not a
    second one, and not this one again after it fails.
    """
    if cfg.get("pending_actions", {}).pop(action_id, None) is None:
        return False
    _spend_pin_reply(cfg)
    return True


def _record_confirmed(
    cfg: Mapping[str, Any], action: Mapping[str, Any], outcome: str
) -> None:
    """Record on the user's message the action its PIN was accepted for and its end."""
    pin_reply = cfg.get("pin_reply")
    if pin_reply is not None:
        pin_reply.confirmed = describe_pending_action(cfg.get("hass"), action)
        pin_reply.outcome = outcome


def _spend_pin_reply(cfg: Mapping[str, Any]) -> None:
    """Mark the user's PIN message as used for this turn."""
    pin_reply = cfg.get("pin_reply")
    if pin_reply is not None:  # Always set here: the refusal requires it.
        pin_reply.spent = True


def _held_api(ha_llm_api: Any, held_api: object, tool_name: str) -> Any | None:
    """
    Return the API a held call goes to, or None if it is no longer there.

    The call goes to the API it was held for, not the one the reply's turn
    routes it to (or a fallback through every API): a local and a remote Home
    Assistant offer the same tool names (#840). A call held with no API on
    record goes to the one API offering its tool; with several, it is not
    run, since the wrapper's fallback could run it on two (#843 review).
    """
    apis = getattr(ha_llm_api, "apis", None)
    if not isinstance(apis, Mapping):
        return ha_llm_api
    if isinstance(held_api, str):
        return apis.get(held_api)
    owners = [
        instance
        for instance in apis.values()
        if any(
            getattr(tool, "name", None) == tool_name
            for tool in getattr(instance, "tools", None) or []
        )
    ]
    return owners[0] if len(owners) == 1 else None


async def _execute_pending_action(  # noqa: PLR0911, PLR0912
    resolved_action_id: str,
    action: dict[str, Any],
    cfg: Mapping[str, Any],
    deadline: float,
) -> tuple[str | None, str | None]:
    """Normalize args, claim the pending action, and execute it once."""
    raw_tool_name = action.get("tool_name")
    if not isinstance(raw_tool_name, str) or not raw_tool_name:
        # Nothing can ever run it, so do not keep it until the TTL sweep.
        cfg.get("pending_actions", {}).pop(resolved_action_id, None)
        return None, pin_msg(cfg.get("hass"), "pending_action_missing_tool_name")
    tool_name = raw_tool_name

    if tool_name == "add_automation":
        return await _execute_pending_automation(
            resolved_action_id, action, cfg, deadline
        )

    ha_llm_api, api_err = _ensure_api(cfg)
    if api_err or ha_llm_api is None:
        return None, pin_msg(cfg.get("hass"), "ha_llm_api_unavailable")

    tool_args = action.get("tool_args") or {}
    # The call runs as it was held: it was normalized and its lock filled in
    # with hass before the PIN was asked for. Looking the names up again
    # here could bind a different device than the one the PIN line named,
    # e.g. after the held one was removed, so only the hass-free
    # normalization runs.
    # The same payload the normal path sends after its guard: HA's intents
    # read name/area/floor and ignore an entity id, so a held lock call with
    # only an entity id would otherwise act on every exposed lock. Any other
    # tool runs with its arguments as held (#836).
    held_api = action.get("api_id")
    # The tool's own name, stored when it was held: the name the agent saw
    # for a remote one may differ by now (#843 review).
    held_name = action.get("api_tool_name")
    stored = isinstance(held_name, str) and bool(held_name)
    origin = tool_origin(
        ha_llm_api,
        str(held_name) if stored else tool_name,
        held_api if isinstance(held_api, str) else None,
        own_name=stored,
    )
    kind = tool_args_kind(cfg.get("hass"), tool_name, remote=origin.remote)
    if action.get("args_kind") not in (None, kind):
        # An intent script took over the tool, or gave it back, since the
        # PIN was asked for: the line no longer names what would run, so the
        # request is dropped and has to be asked again (#836 review).
        cfg.get("pending_actions", {}).pop(resolved_action_id, None)
        return None, pin_msg(cfg.get("hass"), "pending_action_not_found_or_expired")
    if kind == BUILTIN_INTENT:
        tool_args = normalize_intent_for_alarm(tool_name, tool_args, None)
        tool_args = normalize_intent_for_lock(tool_name, tool_args, None)
        tool_args = sanitize_tool_args(
            minimal_payload_for_domain(sanitize_tool_args(tool_args), tool_name)
        )
    elif kind == OTHER_INTENT:
        tool_args = narrow_intent_args(drop_empty_args(tool_args))
    else:
        tool_args = drop_empty_args(tool_args)

    target_api = _held_api(ha_llm_api, held_api, origin.tool_name or tool_name)
    if target_api is None:
        # The API that would have run it is gone.
        cfg.get("pending_actions", {}).pop(resolved_action_id, None)
        return None, pin_msg(cfg.get("hass"), "pending_action_not_found_or_expired")

    if not _claim_pending_action(cfg, resolved_action_id):
        return None, pin_msg(cfg.get("hass"), "pending_action_not_found_or_expired")

    reply_key, outcome = "action_execute_failed", PIN_OUTCOME_FAILED
    try:
        # Sent to the held API itself, under the name it knows the tool by: a
        # remote Home Assistant's tools reach the agent under their server's
        # name (#843).
        own_name = (
            origin.tool_name
            if target_api is not ha_llm_api and origin.tool_name
            else tool_name
        )
        tool_input = llm.ToolInput(tool_name=own_name, tool_args=tool_args)
        async with asyncio.timeout_at(deadline):
            response = tool_call_data(await target_api.async_call_tool(tool_input))
        # A tool that reports failure in its payload (success: false, a
        # warning) did not act, the same as the normal path reads it. Inside
        # the try: after the claim, even reading the reply must not raise.
        # What was done, so the reply cannot say the opposite (#797).
        data = with_action_performed(tool_name, response.data)
        content, status = parse_tool_response(data, tool_args)
    except (HomeAssistantError, vol.Invalid) as err:
        failure = repr(err)
    except Exception as err:
        # Claimed already, so the device may have acted: say so rather than
        # let the error end the turn with the action looking still pending.
        LOGGER.exception("PIN-confirmed %s ended with an unexpected error", tool_name)
        # The type only: the message can hold URLs or tokens.
        reply_key, outcome = "action_outcome_unknown", PIN_OUTCOME_UNKNOWN
        failure = type(err).__name__
    else:
        if not response.error and status != "error":
            _record_confirmed(cfg, action, PIN_OUTCOME_DONE)
            # default=str: a result the JSON encoder cannot take must not
            # raise after the action ran.
            return json.dumps(
                {
                    "status": "completed",
                    "action_id": resolved_action_id,
                    "result": data,
                },
                default=str,
            ), None
        # HA 2026.10's error flag gets the same reply as a raised error.
        failure = (
            content if status == "error" else json.dumps(response.data, default=str)
        )
    _record_confirmed(cfg, action, outcome)
    return None, pin_msg(cfg.get("hass"), reply_key, err=failure)


async def _execute_pending_automation(
    resolved_action_id: str,
    action: dict[str, Any],
    cfg: Mapping[str, Any],
    deadline: float,
) -> tuple[str | None, str | None]:
    """Write a PIN-confirmed automation and clear the pending action."""
    hass = cfg.get("hass")
    automation_config = action.get("automation_config")
    if hass is None or not isinstance(automation_config, dict):
        return None, pin_msg(hass, "pending_automation_invalid")

    if not _claim_pending_action(cfg, resolved_action_id):
        return None, pin_msg(hass, "pending_action_not_found_or_expired")

    try:
        async with asyncio.timeout_at(deadline):
            result = await _async_write_automation(hass, automation_config)
    except Exception as err:
        # Claimed already, and any step can fail after the file was appended
        # (the reload, or a timeout, which is also an OSError): never tell
        # the user to request it again, which could install it twice.
        LOGGER.exception("PIN-confirmed automation ended with an error")
        _record_confirmed(cfg, action, PIN_OUTCOME_UNKNOWN)
        # The type only: the message can hold paths or tokens.
        return None, pin_msg(hass, "automation_outcome_unknown", err=type(err).__name__)

    _record_confirmed(cfg, action, PIN_OUTCOME_DONE)
    install_note = action.get("install_note")
    if isinstance(install_note, str):
        result += install_note
    return json.dumps(
        {"status": "completed", "action_id": resolved_action_id, "result": result}
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


def _state_values(data: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the entries that have a state and a time, reduced to those two."""
    keys = ("state", "last_changed")
    return [{k: d[k] for k in keys} for d in data if all(k in d for k in keys)]


def _get_state_and_decimate(
    data: list[dict[str, str]], limit: int = HISTORY_TOOL_CONTEXT_LIMIT
) -> list[dict[str, str]]:
    """
    Thin a numeric series to at most `limit` entries, keeping the latest.

    Only for numbers: every Nth reading still traces the curve. A series of
    discrete states must not be sampled this way, see
    `_summarize_state_changes`.
    """
    state_values = _state_values(data)
    length = len(state_values)
    if length > limit:
        LOGGER.debug("Decimating sensor data set.")
        last = state_values[-1]
        state_values = state_values[:: math.ceil(length / limit)]
        state_values[-1] = last
    return state_values


def _is_numeric_series(state_values: list[dict[str, str]]) -> bool:
    """Return True if every known state is a number (and there is one)."""
    known = [
        v["state"]
        for v in state_values
        if v["state"] not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
    ]
    try:
        for state in known:
            float(state)
    except ValueError:
        return False
    return bool(known)


_SUMMARY_MAX_STATES = 10


def _summarize_state_changes(
    data: list[dict[str, str]],
    period: tuple[datetime, datetime] | None = None,
    limit: int = HISTORY_TOOL_CONTEXT_LIMIT,
) -> dict[str, Any]:
    """
    Return a discrete history, or its latest changes and a summary of all.

    Sampling every Nth entry of a history that alternates (off, on, off, ...)
    with an even N kept one state only, so a light that toggled 100 times
    read as never on (#715). Rows that repeat the previous state (a
    thermostat records one per attribute update) are dropped first, so the
    limit counts changes. Over `limit` changes, the latest `limit` are kept
    whole, and each state gets, over the whole period: how many times it was
    entered, when first and last, and the minutes spent in it. A blip to
    unavailable or unknown and back does not count as entering the state
    again.
    """
    changes: list[dict[str, str]] = []
    for value in _state_values(data):
        if not changes or value["state"] != changes[-1]["state"]:
            changes.append(value)
    if len(changes) <= limit:
        return {"values": changes}

    times = [dt_util.parse_datetime(c["last_changed"]) for c in changes]
    start, end = period or (times[0], times[-1])
    stats: dict[str, dict[str, Any]] = {}
    previous_known: str | None = None
    for i, change in enumerate(changes):
        state = change["state"]
        entry = stats.setdefault(state, {"times_entered": 0, "minutes_in_state": 0.0})
        began, ended = times[i], times[i + 1] if i + 1 < len(times) else end
        if began is not None and ended is not None and start is not None:
            seconds = (ended - max(began, start)).total_seconds()
            entry["minutes_in_state"] += max(seconds, 0.0) / 60
        unavailable = state in (STATE_UNAVAILABLE, STATE_UNKNOWN)
        if i and (unavailable or state != previous_known):
            entry["times_entered"] += 1
            entry.setdefault("first_entered", change["last_changed"])
            entry["last_entered"] = change["last_changed"]
        if not unavailable:
            previous_known = state
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["minutes_in_state"])
    states = {
        state: {**entry, "minutes_in_state": round(entry["minutes_in_state"], 1)}
        for state, entry in ranked[:_SUMMARY_MAX_STATES]
    }
    note = (
        f"{len(changes)} state changes in this period; values lists only the "
        f"latest {limit}. states covers the whole period: times_entered, "
        "first_entered, last_entered and minutes_in_state per state."
    )
    if start is not None and times[0] is not None and times[0] > start:
        note += f" No history before {changes[0]['last_changed']}."
    return {
        "state_at_start": changes[0],
        "states": states,
        "values": changes[-limit:],
        "note": note,
    }


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
    entity_id: str,
    data: list[dict[str, str]],
    hass: HomeAssistant,
    period: tuple[datetime, datetime] | None = None,
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
        units = state_obj.attributes.get("unit_of_measurement")
        if not state_values:
            return {"value": 0.0, "units": units}
        # Check if sensor was reset during the time of interest.
        zero_indices = [i for i, x in enumerate(state_values) if math.isclose(x, 0.0)]
        if zero_indices:
            LOGGER.warning("Sensor was reset during time of interest.")
            state_values = state_values[zero_indices[-1] :]
        if not state_values:
            return {"value": 0.0, "units": units}
        state_value_change = max(state_values) - min(state_values)
        return {"value": state_value_change, "units": units}

    # A number with no state class (a template sensor, say) is thinned too.
    if _is_numeric_series(_state_values(data)):
        summary: dict[str, Any] = {"values": _get_state_and_decimate(data)}
    else:
        summary = _summarize_state_changes(data, period)
    return summary


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
    Convert a local wall-clock datetime string into a UTC datetime.datetime.

    The string is read as Home Assistant's local time and any UTC offset in it
    is ignored. The model sees only the local wall-clock time and a timezone
    name, never an offset, so an offset it writes is a guess: on a
    America/Los_Angeles install qwen3.8 labelled local times "+00:00" in 4 of
    14 calls, which moved the window seven hours and missed a morning
    departure. Reading the wall clock locally also gets DST right.

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

    local = parsed_datetime.replace(tzinfo=dt_util.get_default_time_zone())
    return dt_util.as_utc(local)


# Allow domains like "sensor", "binary_sensor", "camera", etc.
_DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Valid HA entity_id = <domain>.<object_id>
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


async def _get_existing_entity_id(
    name: str | None, hass: HomeAssistant, domain: str | None = "sensor"
) -> str:
    """
    Lookup an existing entity by its friendly name.

    Only entities exposed to Assist resolve (issue #716): exposure is the
    user's control over what the assistant may read, and a caller at a voice
    satellite can name an entity the model was never shown. A hidden entity
    raises the same "not found" error as a nonexistent one, so the reply does
    not confirm it exists. Without Home Assistant's exposure registry nothing
    can be shown to be exposed, so nothing resolves.

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
    can_check_exposure = exposure_available(hass)
    if not can_check_exposure:
        LOGGER.warning(
            "Assist exposure registry unavailable; entity history lookups "
            "resolve nothing until it is"
        )

    for state in hass.states.async_all():
        eid = state.entity_id
        if not eid.startswith(prefix):
            continue
        fn = state.attributes.get(ATTR_FRIENDLY_NAME, "")
        if (
            isinstance(fn, str)
            and fn.strip().lower() == target
            and can_check_exposure
            and exposed_to_assist(hass, eid)
        ):
            candidates.append(eid)

    if not candidates:
        msg = f"No '{domain}' entity found with friendly name '{name}'"
        raise ValueError(msg)
    if len(candidates) > 1:
        # Prefer the canonical entity (no _N suffix) over duplicates from re-adds.
        canonical = [e for e in candidates if not re.search(r"_\d+$", e)]
        if len(canonical) == 1:
            return canonical[0]
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
) -> dict[str, Any]:
    """
    Retrieve historical state changes for Home Assistant entities over a time range.

    Use this to answer questions like: how many times was a door opened or closed,
    how long was a light on, when did a motion sensor last trigger, how often did an
    appliance run, or any other question about past device states or activity.

    Args:
        friendly_names: List of Home Assistant friendly names to get history for,
            for example, ["Front Door", "Living Room Light"].
        domains: List of Home Assistant domains associated with the friendly names,
            for example, ["binary_sensor", "light"]. These must be in the same order as
            friendly_names.
        local_start_time: Start of the period in "%Y-%m-%dT%H:%M:%S" local time.
            Use the home's local wall-clock time and no UTC offset.
        local_end_time: End of the period in "%Y-%m-%dT%H:%M:%S" local time.
            Use the home's local wall-clock time and no UTC offset.

    Returns:
        Entity histories in local time format, for example:
            {
                "binary_sensor.front_door": {"values": [
                    {"state": "off", "last_changed": "2025-07-24T00:00:00-0700"},
                    {"state": "on", "last_changed": "2025-07-24T04:47:28-0700"},
                    ...]}
            }.
        When a device changed state more times than can be listed, its entry
        lists only the latest changes in "values" and adds "state_at_start",
        "states" (for each state over the whole period: times_entered,
        first_entered, last_entered, minutes_in_state) and a "note". Answer
        counts, first times and durations from "states".
        On error, returns {"error": "<description>"} so the model can report the
        problem to the user (e.g. ambiguous entity name) and request clarification.

    """
    if "configurable" not in config:
        LOGGER.warning("Configuration not found. Please check your setup.")
        return {}

    hass: HomeAssistant = config["configurable"]["hass"]

    if len(friendly_names) != len(domains):
        LOGGER.warning(
            "get_entity_history: friendly_names/domains lengths differ (%d vs %d); "
            "pairing best-effort",
            len(friendly_names),
            len(domains),
        )

    try:
        entity_ids = [
            await _get_existing_entity_id(n, hass, d)
            for n, d in zip(friendly_names, domains, strict=False)
        ]
    except ValueError as e:
        LOGGER.exception("Invalid name %s or domain: %s", friendly_names, domains)
        return {"error": str(e)}

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

    period = (start_time, min(end_time, now))
    return {k: _filter_data(k, v, hass, period) for k, v in data.items()}


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


@tool(parse_docstring=True)
async def get_camera_last_events(  # noqa: D417
    camera_entity_id: str | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Get the latest video analysis events from home cameras.

    Returns face recognition results, AI-generated scene descriptions, and
    event timestamps from the most recent video analysis for each camera.
    Use this when handling camera-related security alerts, or when you need
    to know who or what the cameras have recently observed.

    Args:
        camera_entity_id: Optional camera entity ID to limit results to one
            camera (e.g. 'camera.front_door'). If omitted, returns all cameras.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass = config["configurable"]["hass"]
    results = get_camera_last_events_from_states(hass, camera_entity_id)

    if not results:
        return "No camera last event data available."

    return yaml.dump(
        {"camera_last_events": results},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def _clip_text(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters with a visible marker."""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "…"


def _clip_list(items: list[Any], limit: int) -> list[Any]:
    """Keep the first *limit* items and say how many were left out."""
    if len(items) <= limit:
        return items
    return [*items[:limit], f"… and {len(items) - limit} more"]


async def _post_network_audit_report(hass: Any, report: NetworkAuditReport) -> bool:
    """
    Post the full audit report as a persistent notification; True on success.

    One fixed notification id, so a repeated audit replaces the previous
    report instead of stacking. The text is for the owner, so it is neither
    redacted nor clipped per finding, only Markdown-escaped and capped.
    Persistent notifications are visible to every signed-in Home Assistant
    user, exactly like the alerts Sentinel already posts there with the same
    names; the caller posts only when an administrator asked.
    """
    if hass is None:
        return False
    try:
        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "notification_id": NETWORK_AUDIT_REPORT_NOTIFICATION_ID,
                "title": notif_msg(hass, "audit_report_title"),
                "message": render_network_audit_markdown(
                    report, hass, escape_markdown, NETWORK_AUDIT_REPORT_MAX_CHARS
                ),
            },
            blocking=True,
        )
    except Exception:
        LOGGER.warning("Could not post the security audit report.", exc_info=True)
        return False
    return True


@tool(parse_docstring=True)
async def audit_home_security(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Audit the Home Assistant and network security posture of this home.

    Runs Sentinel's security checks right now and returns the findings by
    severity plus the checks that could not run and why. Checks cover locks
    and alarms exposed to voice assistants without PIN protection, new
    administrators or access tokens, stale tokens, failed logins, cloud
    remote access, add-ons listening on host ports or running unprotected,
    public webhook automations, login bypass and IP-ban settings, security
    devices that are offline, devices Home Assistant discovered but nobody
    configured, and pending router updates. Use it when the user asks whether
    the home is secure or safe, or about network or Home Assistant security,
    privacy, access tokens, or unknown devices. Never scans the network or
    changes anything.

    Args:
        config: Injected runtime configuration; not a model argument.

    """
    configurable = config.get("configurable") or {}
    runtime_data = configurable.get("hga_runtime_data")
    sentinel = getattr(runtime_data, "sentinel", None)
    if sentinel is None:
        return (
            "Sentinel is not enabled, so no security checks can run. Enable "
            "Sentinel in the Home Generative Agent integration's Sentinel "
            "settings (Advanced setup) to audit this home."
        )
    report = await sentinel.async_audit_network()
    if report["status"] != "ok":
        # The engine owns the wording for both "disabled" and "unavailable".
        return "The security audit could not run: " + "; ".join(report["notes"])
    # Fails closed: a sentinel object that cannot say details may be shared
    # (a wrapper, a renamed property) gets the digest, never the details.
    if getattr(sentinel, "network_audit_share_details", False) is not True:
        # The owner keeps audit details out of the conversation: the model
        # gets a deterministic digest, an administrator gets the report.
        if configurable.get("requester_is_admin") is not True:
            note = NETWORK_AUDIT_TOOL_DIGEST_NOTE_NOT_ADMIN
        elif await _post_network_audit_report(configurable.get("hass"), report):
            note = NETWORK_AUDIT_TOOL_DIGEST_NOTE
        else:
            note = NETWORK_AUDIT_TOOL_DIGEST_NOTE_UNDELIVERED
        return yaml.dump(
            network_audit_digest(report, note),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    # Summaries and notes carry text from the LAN (gateway names, discovery
    # titles) and, once a router adapter lands, client addresses: strip
    # every MAC, IP, and hostname before the conversation model sees them.
    # Redaction runs before clipping so a cut cannot leave an address
    # fragment the patterns no longer recognise.
    report = redact_network_identifiers(report)
    payload: dict[str, Any] = {
        "generated_at": report["generated_at"],
        "summary": summarize_network_audit(report),
        "findings": [
            {
                "severity": finding["severity"],
                "type": finding["type"],
                "summary": _clip_text(
                    str(finding["summary"]), NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS
                ),
                "suggested_actions": finding["suggested_actions"],
                "triggering_entities": _clip_list(
                    list(finding["triggering_entities"]),
                    NETWORK_AUDIT_TOOL_MAX_ENTITIES,
                ),
            }
            for finding in report["findings"]
        ],
        "checks_run": report["checks_run"],
        "checks_not_run": report["checks_not_run"],
        "notes": [
            NETWORK_AUDIT_TOOL_LABEL_NOTE,
            *_clip_list(
                [
                    _clip_text(str(n), NETWORK_AUDIT_TOOL_MAX_SUMMARY_CHARS)
                    for n in report["notes"]
                ],
                NETWORK_AUDIT_TOOL_MAX_NOTES,
            ),
        ],
        "privacy_notes": report["privacy_notes"],
    }
    inventory = report.get("inventory")
    if inventory is not None:
        # Counts only: device names stay out of the model's context.
        payload["device_inventory"] = {
            "trusted": inventory["trusted"],
            "untrusted": inventory["untrusted"],
        }
    return yaml.dump(
        payload, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
