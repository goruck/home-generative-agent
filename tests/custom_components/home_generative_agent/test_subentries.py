# ruff: noqa: S101
"""Tests for configuration subentries and resolution."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent import async_migrate_entry
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentConfigFlow,
)
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_USERNAME

from homeassistant.const import CONF_API_KEY

from custom_components.home_generative_agent.const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VLM_PROVIDER,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.flows.feature_subentry_flow import (
    FeatureSubentryFlow,
)
from custom_components.home_generative_agent.flows.model_provider_subentry_flow import (
    ModelProviderSubentryFlow,
)
from custom_components.home_generative_agent.flows.pgvector_db_subentry_flow import (
    PgVectorDbSubentryFlow,
)


class DummySubentry:
    """Simple stand-in for ConfigSubentry."""

    def __init__(self, subentry_id: str, subentry_type: str, title: str, data: dict) -> None:
        """Initialize dummy subentry."""
        self.subentry_id = subentry_id
        self.subentry_type = subentry_type
        self.title = title
        self.data = data


class DummyEntry:
    """Simple stand-in for ConfigEntry used in unit tests."""

    def __init__(self, data: dict | None = None, options: dict | None = None) -> None:
        """Initialize dummy entry."""
        self.entry_id = "entry1"
        self.domain = DOMAIN
        self.data = data or {}
        self.options = options or {}
        self.subentries: dict[str, DummySubentry] = {}
        self.version = CONFIG_ENTRY_VERSION


def _patch_entry(flow, entry) -> None:
    """Attach hass/entry to a subentry flow."""
    flow._get_entry = lambda: entry  # type: ignore[attr-defined]  # noqa: SLF001
    flow._source = "user"  # noqa: SLF001
    flow.context = {"source": "user"}


def test_supported_subentry_types() -> None:
    """Ensure all subentry flow types are registered."""
    supported = HomeGenerativeAgentConfigFlow.async_get_supported_subentry_types(None)
    assert set(supported) == {
        "database",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        SUBENTRY_TYPE_FEATURE,
    }


@pytest.mark.asyncio
async def test_pgvector_flow_defaults(hass) -> None:
    """Database flow shows recommended defaults when no subentry exists."""
    entry = DummyEntry()
    flow = PgVectorDbSubentryFlow()
    flow.hass = hass
    _patch_entry(flow, entry)

    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
    }
    result = await flow.async_step_user()
    assert result["type"] == "form"
    schema_dict = result["data_schema"].schema
    assert CONF_HOST in schema_dict


@pytest.mark.asyncio
async def test_model_provider_flow_creates_ollama(hass, monkeypatch) -> None:
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
    async def _noop_validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_ollama_url",
        _noop_validate,
    )

    first = await flow.async_step_user({"provider_type": "ollama", "name": "Primary Ollama"})
    assert first["type"] == "form"

    result = await flow.async_step_config(
        {
            "base_url": "http://localhost:11434",
            "chat_model": "gpt-oss",
            "vlm_model": "qwen",
            "summarization_model": "gpt-oss",
        }
    )
    assert result["type"] == "create_entry"
    assert result["data"]["provider_type"] == "ollama"
    assert "settings" in result["data"]


@pytest.mark.asyncio
async def test_feature_flow_links_provider(hass) -> None:
    """Feature flow saves provider reference."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Primary Ollama",
        {"provider_type": "ollama"},
    )
    entry = DummyEntry()
    entry.subentries[provider.subentry_id] = provider

    flow = FeatureSubentryFlow()
    flow.hass = hass
    flow.async_create_entry = lambda **kwargs: {  # type: ignore[assignment]
        "type": "create_entry",
        "title": kwargs.get("title"),
        "data": kwargs.get("data"),
    }
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
    }
    flow.async_abort = lambda **kwargs: {  # type: ignore[assignment]
        "type": "abort",
        "reason": kwargs.get("reason"),
    }
    flow._schedule_reload = lambda: None  # type: ignore[assignment]  # noqa: SLF001
    _patch_entry(flow, entry)

    result = await flow.async_step_user(
        {
            "feature_type": "conversation",
            "model_provider_id": "prov1",
            "name": "Conversation Agent",
        }
    )
    assert result["type"] == "create_entry"
    assert result["data"]["model_provider_id"] == "prov1"


def test_resolve_runtime_options_prefers_subentries() -> None:
    """Subentries should override legacy options when present."""
    provider = DummySubentry(
        "provider1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Primary Ollama",
        {
            "provider_type": "ollama",
            "capabilities": ["chat", "vlm", "summarization", "embedding"],
            "settings": {
                "base_url": "http://ollama:11434",
                "chat_model": "chat-ollama",
                "vlm_model": "vlm-ollama",
                "summarization_model": "sum-ollama",
                "embedding_model": "embed-ollama",
            },
        },
    )
    feature = DummySubentry(
        "feature1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "provider1", "name": "Conversation"},
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "ollama"
    assert options[CONF_OLLAMA_CHAT_MODEL] == "chat-ollama"


@pytest.mark.asyncio
async def test_migration_creates_provider_and_feature_subentries(hass) -> None:
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
    entry.subentries = {}

    assert await async_migrate_entry(hass, entry)
    providers = [
        s for s in entry.subentries.values() if s.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
    ]
    features = [
        s for s in entry.subentries.values() if s.subentry_type == SUBENTRY_TYPE_FEATURE
    ]
    assert providers
    assert features
    assert entry.version == CONFIG_ENTRY_VERSION
