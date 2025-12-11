# ruff: noqa: S101
"""Unit tests for confirm_sensitive_action."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from custom_components.home_generative_agent.agent.tools import confirm_sensitive_action
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
)
from custom_components.home_generative_agent.core.utils import hash_pin

ConfirmTool = Callable[..., Awaitable[str]]
# Tools from langchain expose a coroutine attribute:
# cast through Any so type checkers are happy.
confirm_tool = cast("ConfirmTool", cast("Any", confirm_sensitive_action).coroutine)


class FakeAPI:
    """Simple stand-in for the HA LLM API."""

    def __init__(self) -> None:
        """Initialize fake API."""
        self.calls: list[dict[str, object]] = []

    @property
    def called_with(self) -> list[dict[str, object]]:
        """Compatibility alias for stored calls."""
        return self.calls

    async def async_call_tool(self, tool_input: Any) -> dict[str, object]:  # type: ignore[override]
        """Record tool calls."""
        self.calls.append({"name": tool_input.tool_name, "args": tool_input.tool_args})
        return {"ok": True, "result": "done"}


def _base_config(
    pin: str, action_id: str, user_id: str = "user1"
) -> dict[str, dict[str, Any]]:
    """Generate a base configuration for testing."""
    hashed, salt = hash_pin(pin)
    return {
        "configurable": {
            "options": {
                CONF_CRITICAL_ACTION_PIN_HASH: hashed,
                CONF_CRITICAL_ACTION_PIN_SALT: salt,
            },
            "pending_actions": {
                action_id: {
                    "tool_name": "HassTurnOn",
                    "tool_args": {
                        "domain": ["lock"],
                        "name": "Test Lock",
                        "service": "lock",
                    },
                    "user": user_id,
                }
            },
            "user_id": user_id,
            "ha_llm_api": FakeAPI(),
            "hass": None,
        }
    }


@pytest.mark.asyncio
async def test_missing_configurable() -> None:
    """Test missing configurable data."""
    result = await confirm_tool("abc", "1234", config={}, store=None)  # type: ignore[arg-type]
    assert result == "Configuration not found. Please check your setup."


@pytest.mark.asyncio
async def test_pending_action_missing() -> None:
    """Test missing pending action."""
    config = _base_config("1234", "aid")
    config["configurable"]["pending_actions"] = {}
    result = await confirm_tool("aid", "1234", config=config, store=None)
    assert result == "Pending action not found or expired."


@pytest.mark.asyncio
async def test_wrong_user() -> None:
    """Test action requested by different user."""
    config = _base_config("1234", "aid", user_id="owner")
    config["configurable"]["user_id"] = "other"
    result = await confirm_tool("aid", "1234", config=config, store=None)
    assert "different user" in result


@pytest.mark.asyncio
async def test_invalid_pin_format() -> None:
    """Test invalid PIN format."""
    config = _base_config("1234", "aid")
    result = await confirm_tool("aid", "abcd", config=config, store=None)
    assert "Invalid PIN" in result


@pytest.mark.asyncio
async def test_incorrect_pin_increments_attempts() -> None:
    """Test incorrect PIN increments attempts."""
    config = _base_config("1234", "aid")
    pending = config["configurable"]["pending_actions"]["aid"]
    result = await confirm_tool("aid", "0000", config=config, store=None)
    assert result.startswith("Incorrect PIN")
    assert pending["attempts"] == 1


@pytest.mark.asyncio
async def test_success_executes_and_clears_pending() -> None:
    """Test successful confirmation executes action and clears pending."""
    config = _base_config("1234", "aid")
    api: FakeAPI = config["configurable"]["ha_llm_api"]  # type: ignore[assignment]

    result = await confirm_tool("aid", "1234", config=config, store=None)
    result_payload = json.loads(result)
    assert result_payload.get("status") == "completed"
    assert result_payload.get("action_id") == "aid"
    assert "aid" not in config["configurable"]["pending_actions"]
    assert api.calls  # tool was invoked
