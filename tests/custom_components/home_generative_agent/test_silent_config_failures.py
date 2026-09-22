# ruff: noqa: S101
"""
Regression tests for three settings that failed with nothing in the logs.

* ``TargetSelectorData`` is deprecated in Home Assistant core with removal in
  2026.12.0. Every startup logged the deprecation against this integration, and
  the ``save_and_analyze_snapshot`` service would stop resolving targets
  outright once core drops the name.
* Re-running Sentinel **Basic setup** rebuilt the payload from
  ``_default_payload()``, wiping ``sentinel_rule_entity_exclusions`` and
  ``sentinel_camera_entry_links`` — two hand-curated maps no default can
  reconstruct. The symptom was phantom alerts returning with nothing in the UI
  to explain why.
* ``sentinel_triage_enabled`` had no config-flow field and was missing from
  ``_apply_sentinel_options``' defaults dict, so the whole triage service was
  unreachable from the UI and would have been ignored even if injected into
  Sentinel subentry data (the #480 allowlist-omission shape).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest
from homeassistant.helpers import target as ha_target

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_CAMERA_ENTRY_LINKS,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    CONF_SENTINEL_TRIAGE_ENABLED,
    CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
    RECOMMENDED_SENTINEL_TRIAGE_ENABLED,
    RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.flows.sentinel_subentry_flow import (
    SentinelSubentryFlow,
    _default_payload,
)

from .test_subentries import (
    DummyEntry,
    DummySubentry,
    _make_sentinel_flow_for_mode,
)

if TYPE_CHECKING:
    import voluptuous as vol
    from homeassistant.core import HomeAssistant


# ---------------------------------------------------------------------------
# TargetSelectorData -> TargetSelection (breaks in HA 2026.12.0)
# ---------------------------------------------------------------------------


def test_component_does_not_use_the_deprecated_target_selector() -> None:
    """The integration binds TargetSelection, not the deprecated alias."""
    selection = hga_component.TargetSelection
    assert selection is not getattr(ha_target, "TargetSelectorData", object()), (
        "TargetSelectorData is deprecated with removal in HA 2026.12.0"
    )
    assert selection is ha_target.TargetSelection

    source = inspect.getsource(hga_component)
    assert "TargetSelectorData(" not in source, (
        "no call site may construct the deprecated class"
    )


def test_target_selection_resolves_a_service_target() -> None:
    """The bound class still parses a raw service ``target:`` block."""
    selection = hga_component.TargetSelection(
        {"entity_id": "camera.front", "area_id": ["garage"]}
    )
    assert selection.entity_ids == {"camera.front"}
    assert selection.area_ids == {"garage"}
    assert selection.has_any_target is True
    assert hga_component.TargetSelection({}).has_any_target is False


# ---------------------------------------------------------------------------
# Basic setup must not wipe the two hand-curated maps
# ---------------------------------------------------------------------------

_EXCLUSIONS = {"unlocked_lock_at_night": ["lock.template_mirror"]}
_LINKS = {"camera.porch": "entry_abc"}


def _sentinel_flow_with_existing(
    hass: HomeAssistant, data: dict[str, Any]
) -> tuple[SentinelSubentryFlow, list[Any]]:
    """Return a Basic-setup flow over an existing Sentinel subentry."""
    existing = DummySubentry("sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", data)
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    update_calls: list[Any] = []
    flow.async_update_and_abort = lambda *_args, **kwargs: (  # type: ignore[assignment]
        update_calls.append(kwargs)
        or {"type": "abort", "reason": "reconfigure_successful"}
    )
    return flow, update_calls


@pytest.mark.asyncio
async def test_basic_setup_carries_exclusions_and_camera_links(
    hass: HomeAssistant,
) -> None:
    """Re-running Basic setup keeps exclusions and camera entry links."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: _EXCLUSIONS,
            CONF_SENTINEL_CAMERA_ENTRY_LINKS: _LINKS,
        },
    )

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )

    assert len(update_calls) == 1
    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS] == _EXCLUSIONS
    assert data[CONF_SENTINEL_CAMERA_ENTRY_LINKS] == _LINKS


@pytest.mark.asyncio
async def test_basic_setup_carry_over_is_a_copy(hass: HomeAssistant) -> None:
    """The carried exclusions are copied, not aliased to the subentry's lists."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: _EXCLUSIONS,
        },
    )

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )

    carried = update_calls[0]["data"][CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS]
    carried["unlocked_lock_at_night"].append("lock.other")
    assert _EXCLUSIONS["unlocked_lock_at_night"] == ["lock.template_mirror"]


@pytest.mark.asyncio
async def test_basic_setup_still_resets_everything_else(hass: HomeAssistant) -> None:
    """Only the two curated maps survive; other settings go back to defaults."""
    flow, update_calls = _sentinel_flow_with_existing(
        hass,
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS: 99,
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: _EXCLUSIONS,
        },
    )

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )

    data = update_calls[0]["data"]
    assert data[CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS] == (
        RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_basic_setup_on_a_fresh_install_writes_the_defaults(
    hass: HomeAssistant,
) -> None:
    """With no existing subentry the carry-over is a no-op."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )

    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS] == {}
    assert data[CONF_SENTINEL_CAMERA_ENTRY_LINKS] == {}


# ---------------------------------------------------------------------------
# Triage is reachable: schema field + resolver allowlist (#262, #480 shape)
# ---------------------------------------------------------------------------


def _schema_keys(schema: vol.Schema) -> set[str]:
    return {str(key) for key in schema.schema}


def test_triage_fields_are_in_the_advanced_schema(hass: HomeAssistant) -> None:
    """Both triage options have a field on the Sentinel settings form."""
    flow = SentinelSubentryFlow()
    flow.hass = hass
    keys = _schema_keys(flow._schema(_default_payload()))
    assert CONF_SENTINEL_TRIAGE_ENABLED in keys
    assert CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS in keys


def test_triage_defaults_are_in_the_default_payload() -> None:
    """A new Sentinel subentry persists the triage defaults (off)."""
    payload = _default_payload()
    assert payload[CONF_SENTINEL_TRIAGE_ENABLED] is RECOMMENDED_SENTINEL_TRIAGE_ENABLED
    assert payload[CONF_SENTINEL_TRIAGE_ENABLED] is False, (
        "triage costs an LLM call per finding; it must stay opt-in"
    )
    assert payload[CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS] == (
        RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS
    )


def test_resolver_propagates_triage_from_the_sentinel_subentry() -> None:
    """Subentry triage values reach runtime options (the #480 allowlist shape)."""
    entry = DummyEntry(options={})
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_TRIAGE_ENABLED: True,
            CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS: 25,
        },
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_TRIAGE_ENABLED] is True
    assert options[CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS] == 25


def test_resolver_defaults_triage_off_when_unset() -> None:
    """A subentry without the keys resolves to the recommended defaults."""
    entry = DummyEntry(options={})
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True},
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_TRIAGE_ENABLED] is False
    assert options[CONF_SENTINEL_TRIAGE_TIMEOUT_SECONDS] == (
        RECOMMENDED_SENTINEL_TRIAGE_TIMEOUT_SECONDS
    )
