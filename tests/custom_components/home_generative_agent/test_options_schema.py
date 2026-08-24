# ruff: noqa: S101
"""Tests for the global options schema layout."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import ConstantSelector, TextSelector
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentOptionsFlow,
    _schema_for_options,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_SCHEMA_FIRST_YAML,
    CONF_STT_HALLUCINATION_EXACT_PATTERNS,
    CONF_STT_HALLUCINATION_PATTERNS,
    DOMAIN,
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


def _patch_registered_apis(*api_ids: str) -> Any:
    """Patch the config-flow LLM API registry to the given ids."""
    return patch(
        "custom_components.home_generative_agent.config_flow.llm.async_get_apis",
        return_value=_fake_apis(*api_ids),
    )


async def _schema_with_apis(
    hass: Any, opts: dict[str, Any], *api_ids: str
) -> dict[Any, Any]:
    """Build the options schema with the given registered LLM APIs."""
    with _patch_registered_apis(*api_ids):
        return await _schema_for_options(hass, opts)


def _suggested_llm_apis(schema: dict[Any, Any]) -> list[str]:
    """Return the pre-filled LLM API selection from the schema."""
    marker = cast("Any", _schema_key(schema, CONF_LLM_HASS_API))
    return marker.description["suggested_value"]


def _llm_api_options(schema: dict[Any, Any]) -> dict[str, str]:
    """Return the LLM API selector options as a value -> label mapping."""
    selector = cast("Any", schema[_schema_key(schema, CONF_LLM_HASS_API)])
    return {opt["value"]: opt["label"] for opt in selector.config["options"]}


@pytest.mark.asyncio
async def test_options_schema_preserves_stale_llm_api_ids(hass: Any) -> None:
    """
    Removed API ids (e.g. deleted MCP servers) stay selected and selectable.

    A stale id pre-filled into a selector that no longer lists it fails
    SelectSelector validation on submit, leaving the options form permanently
    unsaveable (issue #568). The stale id is therefore re-added as a labeled
    option: the form saves again, but nothing is dropped without an explicit
    user deselection — a transient provider outage can never erase the
    selection, nor silently swap in the Assist default.
    """
    schema = await _schema_with_apis(
        hass, {CONF_LLM_HASS_API: ["assist", "mcp-deleted"]}, "assist", "mcp-new"
    )

    assert _suggested_llm_apis(schema) == ["assist", "mcp-deleted"]
    options = _llm_api_options(schema)
    assert options["mcp-deleted"] == "mcp-deleted (no longer available)"
    assert "mcp-new" in options


@pytest.mark.asyncio
async def test_options_schema_keeps_valid_llm_api_ids(hass: Any) -> None:
    """Stored ids that are still registered pre-fill the form unchanged."""
    schema = await _schema_with_apis(
        hass, {CONF_LLM_HASS_API: ["assist", "mcp-new"]}, "assist", "mcp-new"
    )

    assert _suggested_llm_apis(schema) == ["assist", "mcp-new"]
    assert not [
        label
        for label in _llm_api_options(schema).values()
        if "(no longer available)" in label
    ]


@pytest.mark.asyncio
async def test_options_schema_defaults_to_empty_llm_api_selection(hass: Any) -> None:
    """A config entry with no stored LLM API pre-fills an empty selection."""
    schema = await _schema_with_apis(hass, {}, "assist")

    assert _suggested_llm_apis(schema) == []


@pytest.mark.asyncio
async def test_options_schema_warns_only_for_stale_llm_api_ids(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The stale-id warning names every unavailable id and stays silent otherwise.

    Also covers a stored list whose ids are ALL stale: every id stays selected
    and selectable, so the form remains saveable (issue #568) while the
    selection survives verbatim.
    """
    schema = await _schema_with_apis(
        hass, {CONF_LLM_HASS_API: ["mcp-deleted", "mcp-gone"]}, "assist"
    )

    assert _suggested_llm_apis(schema) == ["mcp-deleted", "mcp-gone"]
    assert set(_llm_api_options(schema)) == {"assist", "mcp-deleted", "mcp-gone"}
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "mcp-deleted" in r.getMessage() and "mcp-gone" in r.getMessage()
        for r in warnings
    )

    caplog.clear()
    await _schema_with_apis(hass, {CONF_LLM_HASS_API: ["assist"]}, "assist")

    assert not [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "no longer registered" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_options_schema_handles_legacy_string_llm_api(hass: Any) -> None:
    """A legacy single-string stored value is normalized, valid or not."""
    valid = await _schema_with_apis(hass, {CONF_LLM_HASS_API: "assist"}, "assist")
    stale = await _schema_with_apis(hass, {CONF_LLM_HASS_API: "mcp-deleted"}, "assist")

    assert _suggested_llm_apis(valid) == ["assist"]
    assert _suggested_llm_apis(stale) == ["mcp-deleted"]
    assert "mcp-deleted" in _llm_api_options(stale)


@pytest.mark.asyncio
async def test_options_schema_tolerates_degenerate_llm_api_values(
    hass: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """
    None, "" and non-string list elements render cleanly with no warning.

    An explicit None previously crashed the schema build (TypeError on
    iteration), re-creating the form-blocking failure class of issue #568;
    "" would wrap to [""] and a dict element would raise TypeError on the
    set-membership test.
    """
    none_schema = await _schema_with_apis(hass, {CONF_LLM_HASS_API: None}, "assist")
    empty_schema = await _schema_with_apis(hass, {CONF_LLM_HASS_API: ""}, "assist")
    mixed_schema = await _schema_with_apis(
        hass, {CONF_LLM_HASS_API: ["assist", {"id": "mcp-old"}]}, "assist"
    )

    assert _suggested_llm_apis(none_schema) == []
    assert _suggested_llm_apis(empty_schema) == []
    assert _suggested_llm_apis(mixed_schema) == ["assist"]
    assert not [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "no longer registered" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_options_flow_init_form_preserves_stale_llm_api(hass: Any) -> None:
    """
    The rendered options-flow form keeps a stale id selected and selectable.

    Exercises the real #568 repro surface (async_step_init merging
    DEFAULT_OPTIONS with the entry's stored options), not just the schema
    helper the other tests call directly.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        options={CONF_LLM_HASS_API: ["assist", "mcp-deleted"]},
    )
    entry.add_to_hass(hass)
    flow = HomeGenerativeAgentOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id

    with _patch_registered_apis("assist", "mcp-new"):
        result = cast("dict[str, Any]", await flow.async_step_init(None))

    assert result["type"] == FlowResultType.FORM
    schema = cast("Any", result["data_schema"]).schema
    assert _suggested_llm_apis(schema) == ["assist", "mcp-deleted"]
    assert "mcp-deleted" in _llm_api_options(schema)
