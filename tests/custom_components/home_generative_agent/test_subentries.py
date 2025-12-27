# ruff: noqa: S101
"""Tests for configuration subentries and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.const import CONF_API_KEY, CONF_HOST, CONF_PASSWORD, CONF_USERNAME
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent import async_migrate_entry
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentConfigFlow,
)
from custom_components.home_generative_agent.const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_NAME,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM_URL,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VLM_PROVIDER,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    legacy_model_provider_configs,
    resolve_runtime_options,
)
from custom_components.home_generative_agent.flows.feature_subentry_flow import (
    FeatureSubentryFlow,
)
from custom_components.home_generative_agent.flows.model_provider_subentry_flow import (
    ModelProviderSubentryFlow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


class DummySubentry:
    """Simple stand-in for ConfigSubentry."""

    def __init__(
        self,
        subentry_id: str,
        subentry_type: str,
        title: str,
        data: dict[str, Any],
    ) -> None:
        """Initialize dummy subentry."""
        self.subentry_id = subentry_id
        self.subentry_type = subentry_type
        self.title = title
        self.data: dict[str, Any] = data


class DummyEntry:
    """Simple stand-in for ConfigEntry used in unit tests."""

    def __init__(
        self, data: dict[str, Any] | None = None, options: dict[str, Any] | None = None
    ) -> None:
        """Initialize dummy entry."""
        self.entry_id = "entry1"
        self.domain = DOMAIN
        self.data: dict[str, Any] = data or {}
        self.options: dict[str, Any] = options or {}
        self.subentries: dict[str, DummySubentry] = {}
        self.version = CONFIG_ENTRY_VERSION


def _patch_entry(flow: Any, entry: DummyEntry) -> None:
    """Attach hass/entry to a subentry flow."""
    flow._get_entry = lambda: entry  # type: ignore[attr-defined]  # noqa: SLF001
    flow._source = "user"  # noqa: SLF001
    flow.context = {"source": "user"}


def test_supported_subentry_types() -> None:
    """Ensure all subentry flow types are registered."""
    entry = MockConfigEntry(domain=DOMAIN, title="Home Generative Agent")
    supported = HomeGenerativeAgentConfigFlow.async_get_supported_subentry_types(entry)
    assert set(supported) == {SUBENTRY_TYPE_MODEL_PROVIDER, SUBENTRY_TYPE_FEATURE}


@pytest.mark.asyncio
async def test_model_provider_flow_creates_ollama(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model provider flow creates an Ollama subentry payload."""
    entry = DummyEntry()
    flow = ModelProviderSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
    }
    flow.async_create_entry = lambda **kwargs: {  # type: ignore[assignment]
        "type": "create_entry",
        "title": kwargs.get("title"),
        "data": kwargs.get("data"),
    }
    flow.async_abort = lambda **kwargs: {  # type: ignore[assignment]
        "type": "abort",
        "reason": kwargs.get("reason"),
    }
    flow._schedule_reload = lambda: None  # type: ignore[assignment]  # noqa: SLF001
    _patch_entry(flow, entry)

    async def _noop_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_ollama_url",
        _noop_validate,
    )

    first = await flow.async_step_user()
    assert first.get("type") == "form"

    second = await flow.async_step_deployment({"deployment": "edge"})
    assert second.get("type") == "form"

    third = await flow.async_step_provider(
        {"provider_type": "ollama", "name": "Primary Ollama"}
    )
    assert third.get("type") == "form"

    result = await flow.async_step_settings({"base_url": "http://localhost:11434"})
    assert result.get("type") == "create_entry"
    result_data = result.get("data")
    assert result_data is not None
    assert result_data["provider_type"] == "ollama"
    assert "settings" in result_data


@pytest.mark.asyncio
async def test_feature_flow_links_provider(hass: HomeAssistant) -> None:
    """Feature flow saves provider reference."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Primary Ollama",
        {"provider_type": "ollama", "capabilities": ["chat"]},
    )
    feature = DummySubentry(
        "feature1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "prov1"},
    )
    entry = DummyEntry()
    entry.subentries[provider.subentry_id] = provider
    entry.subentries[feature.subentry_id] = feature

    flow = FeatureSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
    }
    flow.async_abort = lambda **kwargs: {  # type: ignore[assignment]
        "type": "abort",
        "reason": kwargs.get("reason"),
    }

    def _update_subentry(
        _entry: DummyEntry,
        subentry: DummySubentry,
        data: Mapping[str, Any],
        title: str | None,
    ) -> None:
        _ = title
        subentry.data = {**subentry.data, **data}

    flow.hass.config_entries.async_update_subentry = (  # type: ignore[assignment]
        _update_subentry
    )
    flow._schedule_reload = lambda: None  # type: ignore[assignment]  # noqa: SLF001
    _patch_entry(flow, entry)
    flow.context["subentry_id"] = feature.subentry_id

    first = await flow.async_step_user()
    assert first.get("type") == "form"
    provider_step = await flow.async_step_conversation({"model_provider_id": "prov1"})
    assert provider_step.get("type") == "form"
    result = await flow.async_step_feature_model({"model_name": "gpt-oss"})
    assert result.get("type") == "abort"
    assert feature.data["model_provider_id"] == "prov1"
    assert feature.data[CONF_FEATURE_MODEL][CONF_FEATURE_MODEL_NAME] == "gpt-oss"


def test_resolve_runtime_options_prefers_subentries() -> None:
    """Subentries should override legacy options when present."""
    provider = DummySubentry(
        "provider1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Primary Ollama",
        {
            "provider_type": "ollama",
            "capabilities": ["chat", "vlm", "summarization", "embedding"],
            "settings": {"base_url": "http://ollama:11434"},
        },
    )
    feature = DummySubentry(
        "feature1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "provider1",
            "name": "Conversation",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "chat-ollama"},
        },
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "ollama"
    assert options[CONF_OLLAMA_CHAT_MODEL] == "chat-ollama"


def test_legacy_ollama_urls_split_providers() -> None:
    """Legacy Ollama URLs should map to separate providers when they differ."""
    entry = DummyEntry(
        options={
            CONF_OLLAMA_URL: "http://ollama-base:11434",
            CONF_OLLAMA_CHAT_URL: "http://ollama-chat:11434",
            CONF_OLLAMA_VLM_URL: "http://ollama-vlm:11434",
            CONF_OLLAMA_SUMMARIZATION_URL: "http://ollama-sum:11434",
        }
    )

    providers = legacy_model_provider_configs(
        cast("ConfigEntry[Any]", entry), entry.options
    )
    ollama_providers = [
        provider
        for provider in providers.values()
        if provider.provider_type == "ollama"
    ]
    expected_provider_count = 3
    assert len(ollama_providers) == expected_provider_count
    assert any(
        provider.data["settings"]["base_url"] == "http://ollama-chat:11434"
        and "chat" in provider.capabilities
        and "embedding" in provider.capabilities
        for provider in ollama_providers
    )
    assert any(
        provider.data["settings"]["base_url"] == "http://ollama-vlm:11434"
        and provider.capabilities == {"vlm"}
        for provider in ollama_providers
    )
    assert any(
        provider.data["settings"]["base_url"] == "http://ollama-sum:11434"
        and provider.capabilities == {"summarization"}
        for provider in ollama_providers
    )

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "ollama"
    assert options[CONF_VLM_PROVIDER] == "ollama"
    assert options[CONF_SUMMARIZATION_MODEL_PROVIDER] == "ollama"
    assert options[CONF_OLLAMA_CHAT_URL] == "http://ollama-chat:11434"
    assert options[CONF_OLLAMA_VLM_URL] == "http://ollama-vlm:11434"
    assert options[CONF_OLLAMA_SUMMARIZATION_URL] == "http://ollama-sum:11434"


@pytest.mark.asyncio
async def test_migration_creates_provider_and_feature_subentries(
    hass: HomeAssistant,
) -> None:
    """Migration should add provider and feature subentries from legacy options."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=2,
        data={},
        options={
            CONF_API_KEY: "sk-key",
            CONF_CHAT_MODEL_PROVIDER: "openai",
            CONF_VLM_PROVIDER: "openai",
            CONF_SUMMARIZATION_MODEL_PROVIDER: "openai",
            CONF_DB_NAME: "db",
            CONF_DB_PARAMS: [],
            CONF_USERNAME: "user",
            CONF_PASSWORD: "pass",
            CONF_HOST: "localhost",
        },
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    providers = [
        s
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
    ]
    features = [
        s for s in entry.subentries.values() if s.subentry_type == SUBENTRY_TYPE_FEATURE
    ]
    assert providers
    assert features
    assert entry.version == CONFIG_ENTRY_VERSION
