# ruff: noqa: S101
"""Tests for the global options schema layout."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from homeassistant.const import CONF_LLM_HASS_API
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


def _fake_apis(*ids: str) -> list[Any]:
    """Build fake registered LLM APIs with the given ids."""
    return [SimpleNamespace(id=api_id, name=api_id.title()) for api_id in ids]


def _suggested_llm_apis(schema: dict[Any, Any]) -> list[str]:
    """Return the pre-filled LLM API selection from the schema."""
    marker = cast("Any", _schema_key(schema, CONF_LLM_HASS_API))
    return marker.description["suggested_value"]


@pytest.mark.asyncio
async def test_options_schema_drops_stale_llm_api_ids(hass: Any) -> None:
    """
    Removed API ids (e.g. deleted MCP servers) must not pre-fill the form.

    A stale id pre-filled into the selector fails SelectSelector validation on
    submit, leaving the options form permanently unsaveable (issue #568).
    """
    with patch(
        "custom_components.home_generative_agent.config_flow.llm.async_get_apis",
        return_value=_fake_apis("assist", "mcp-new"),
    ):
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["assist", "mcp-deleted"]}
        )

    assert _suggested_llm_apis(schema) == ["assist"]


@pytest.mark.asyncio
async def test_options_schema_keeps_valid_llm_api_ids(hass: Any) -> None:
    """Stored ids that are still registered pre-fill the form unchanged."""
    with patch(
        "custom_components.home_generative_agent.config_flow.llm.async_get_apis",
        return_value=_fake_apis("assist", "mcp-new"),
    ):
        schema = await _schema_for_options(
            hass, {CONF_LLM_HASS_API: ["assist", "mcp-new"]}
        )

    assert _suggested_llm_apis(schema) == ["assist", "mcp-new"]


@pytest.mark.asyncio
async def test_options_schema_handles_legacy_string_llm_api(hass: Any) -> None:
    """A legacy single-string stored value is normalized and filtered."""
    with patch(
        "custom_components.home_generative_agent.config_flow.llm.async_get_apis",
        return_value=_fake_apis("assist"),
    ):
        valid = await _schema_for_options(hass, {CONF_LLM_HASS_API: "assist"})
        stale = await _schema_for_options(hass, {CONF_LLM_HASS_API: "mcp-deleted"})

    assert _suggested_llm_apis(valid) == ["assist"]
    assert _suggested_llm_apis(stale) == []
