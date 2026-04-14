# ruff: noqa: S101
"""Tests for handling unhashable selectors and extracting enums."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.helpers import llm
from homeassistant.helpers.selector import DeviceSelector, SelectSelector

from custom_components.home_generative_agent.agent.helpers import (
    format_tool,
    safe_convert,
)


class UnhashableCallable:
    """A callable object that is unhashable."""

    def __call__(self, v: Any) -> Any:
        return v

    def __repr__(self) -> str:
        """Return representation of the object."""
        return "<UnhashableCallable>"

    __hash__ = None  # type: ignore[assignment]


def test_safe_convert_select_selector() -> None:
    """Test that SelectSelector options are correctly extracted into an OpenAPI enum."""
    selector = SelectSelector({"options": ["red", "green", "blue"]})
    schema = vol.Schema({"color": selector})

    result = safe_convert(schema)

    assert result["type"] == "object"
    assert "color" in result["properties"]
    assert result["properties"]["color"]["type"] == "string"
    assert result["properties"]["color"]["enum"] == ["red", "green", "blue"]


def test_safe_convert_select_selector_dict_options() -> None:
    """Test SelectSelector with dict options (value/label)."""
    selector = SelectSelector(
        {"options": [{"value": "r", "label": "Red"}, {"value": "g", "label": "Green"}]}
    )
    schema = vol.Schema({"color": selector})

    result = safe_convert(schema)

    assert result["properties"]["color"]["enum"] == ["r", "g"]


def test_safe_convert_device_selector() -> None:
    """Test that other selectors (like DeviceSelector) fall back to string without crashing."""
    selector = DeviceSelector({"multiple": False})
    schema = vol.Schema({"device": selector})

    result = safe_convert(schema)

    assert result["properties"]["device"]["type"] == "string"
    assert "enum" not in result["properties"]["device"]


def test_safe_convert_unhashable_object() -> None:
    """Test that a general unhashable object falls back to string without crashing."""
    schema = vol.Schema({"test": UnhashableCallable()})

    result = safe_convert(schema)

    assert result["properties"]["test"]["type"] == "string"


def test_format_tool_with_selector() -> None:
    """End-to-end test of format_tool with a selector."""

    class FakeTool(llm.Tool):
        def __init__(self) -> None:
            self.name = "test_tool"
            self.description = "Test Description"
            self.parameters = vol.Schema(
                {"option": SelectSelector({"options": ["a", "b"]})}
            )

        async def async_call(
            self,
            hass: Any,
            tool_input: llm.ToolInput,
            llm_context: llm.LLMContext,
        ) -> Any:
            return None

    tool = FakeTool()
    formatted = format_tool(tool, None)

    assert formatted["type"] == "function"
    assert formatted["function"]["name"] == "test_tool"
    assert formatted["function"]["parameters"]["properties"]["option"]["enum"] == [
        "a",
        "b",
    ]
