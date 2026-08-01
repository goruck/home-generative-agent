# ruff: noqa: S101
"""Tests for the global options schema layout."""

from __future__ import annotations

from typing import Any, cast

import pytest
from homeassistant.helpers.selector import ConstantSelector, TextSelector

from custom_components.home_generative_agent.config_flow import _schema_for_options
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_SCHEMA_FIRST_YAML,
    CONF_STT_HALLUCINATION_EXACT_PATTERNS,
    CONF_STT_HALLUCINATION_PATTERNS,
)


def _schema_keys(schema: dict[Any, Any]) -> list[str]:
    """Return schema option keys in render order."""
    return [str(cast("Any", key).schema) for key in schema]


def _schema_key(schema: dict[Any, Any], key_name: str) -> Any:
    """Return a schema marker by name."""
    return next(key for key in schema if cast("Any", key).schema == key_name)


@pytest.mark.asyncio
async def test_options_schema_pin_is_directly_under_pin_switch(hass: Any) -> None:
    """PIN entry should render immediately after the critical-action switch."""
    schema = await _schema_for_options(hass, {CONF_CRITICAL_ACTION_PIN_ENABLED: True})
    keys = _schema_keys(schema)

    pin_switch_idx = keys.index(CONF_CRITICAL_ACTION_PIN_ENABLED)
    assert keys[pin_switch_idx + 1] == CONF_CRITICAL_ACTION_PIN
    assert keys[pin_switch_idx + 2] == CONF_SCHEMA_FIRST_YAML


@pytest.mark.asyncio
async def test_options_schema_stt_filters_are_bottom_multiline_section(
    hass: Any,
) -> None:
    """STT filters should render as multiline fields in a bottom section."""
    schema = await _schema_for_options(
        hass,
        {
            CONF_STT_HALLUCINATION_PATTERNS: ["back to our show"],
            CONF_STT_HALLUCINATION_EXACT_PATTERNS: ["the end"],
        },
    )
    keys = _schema_keys(schema)

    assert keys[-3:] == [
        "stt_filters_section",
        CONF_STT_HALLUCINATION_PATTERNS,
        CONF_STT_HALLUCINATION_EXACT_PATTERNS,
    ]
    assert isinstance(
        schema[_schema_key(schema, "stt_filters_section")],
        ConstantSelector,
    )
    assert isinstance(
        schema[_schema_key(schema, CONF_STT_HALLUCINATION_PATTERNS)],
        TextSelector,
    )
    assert isinstance(
        schema[_schema_key(schema, CONF_STT_HALLUCINATION_EXACT_PATTERNS)],
        TextSelector,
    )
    assert (
        cast("Any", _schema_key(schema, CONF_STT_HALLUCINATION_PATTERNS)).default()
        == "back to our show"
    )
