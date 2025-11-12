"""Langgraph tools for Home Generative Agent.

This module exports all available tools and provides shared utilities
for tool validation and response sanitization.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .automation_tool import add_automation
from .camera_image_tool import analyze_image, get_and_analyze_camera_image
from .entity_history_tool import get_entity_history
from .memory_tool import upsert_memory
from .travel_times import get_travel_time
from .web_search_tool import web_search

# Keep deprecated tool for backward compatibility
# Deprecated: This tool has been replaced by the HA native tool GetLiveContext.
# It is kept here for reference only and should not be used in new code.
# from .device_state_tool import get_current_device_state

from ..const import TOOL_RESPONSE_MAX_LENGTH  # noqa: TID252

LOGGER = logging.getLogger(__name__)

# Export all tools
__all__ = [
    "add_automation",
    "analyze_image",
    "get_and_analyze_camera_image",
    "get_entity_history",
    "get_travel_time",
    "upsert_memory",
    "web_search",
    # Shared utilities
    "_validate_tool_params",
    "_sanitize_tool_response",
]


# ----- Shared Helper Functions for Tool Validation & Sanitization -----


def _validate_tool_params(
    tool_schema: dict[str, Any],
    provided_args: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate tool parameters against schema.

    Args:
        tool_schema: Schema defining expected parameters
        provided_args: Arguments provided to the tool

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required = tool_schema.get("required", [])
    for field in required:
        if field not in provided_args:
            return False, f"Missing required parameter: {field}"

    # Check field types
    properties = tool_schema.get("properties", {})
    for field, value in provided_args.items():
        if field not in properties:
            return False, f"Unknown parameter: {field}"

        field_schema = properties[field]
        expected_type = field_schema.get("type")

        if expected_type == "string" and not isinstance(value, str):
            return (
                False,
                f"Parameter '{field}' must be a string, got {type(value).__name__}",
            )
        elif expected_type == "array" and not isinstance(value, list):
            return (
                False,
                f"Parameter '{field}' must be a list, got {type(value).__name__}",
            )
        elif expected_type == "object" and not isinstance(value, dict):
            return (
                False,
                f"Parameter '{field}' must be an object, got {type(value).__name__}",
            )
        elif expected_type == "number" and not isinstance(value, (int, float)):
            return (
                False,
                f"Parameter '{field}' must be a number, got {type(value).__name__}",
            )
        elif expected_type == "boolean" and not isinstance(value, bool):
            return (
                False,
                f"Parameter '{field}' must be a boolean, got {type(value).__name__}",
            )

        # Check string length constraints
        if isinstance(value, str):
            max_length = field_schema.get("maxLength")
            if max_length and len(value) > max_length:
                return (
                    False,
                    f"Parameter '{field}' exceeds max length of {max_length}",
                )
            min_length = field_schema.get("minLength")
            if min_length and len(value) < min_length:
                return (
                    False,
                    f"Parameter '{field}' below min length of {min_length}",
                )

        # Check enum values
        enum_values = field_schema.get("enum")
        if enum_values and value not in enum_values:
            return (
                False,
                f"Parameter '{field}' must be one of {enum_values}, got {value}",
            )

    return True, None


def _sanitize_tool_response(
    response: Any,
    max_length: int = TOOL_RESPONSE_MAX_LENGTH,
) -> str:
    """Sanitize and limit tool response size.

    Args:
        response: The tool response (any type)
        max_length: Maximum length in characters

    Returns:
        Sanitized, truncated response as string
    """
    if response is None:
        return ""

    # Convert to string
    if isinstance(response, str):
        response_str = response
    elif isinstance(response, dict) or isinstance(response, list):
        try:
            response_str = json.dumps(response)
        except (TypeError, ValueError):
            response_str = str(response)
    else:
        response_str = str(response)

    # Sanitize - remove control characters except newlines/tabs
    sanitized = "".join(
        char if ord(char) >= 32 or char in ("\n", "\t", "\r") else ""
        for char in response_str
    )

    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3] + "..."  # Reserve space for ellipsis
        LOGGER.debug(
            "Tool response truncated from %d to %d characters",
            len(response_str),
            max_length,
        )

    return sanitized
