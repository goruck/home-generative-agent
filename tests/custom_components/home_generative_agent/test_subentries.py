# ruff: noqa: S101
"""Tests for configuration subentries and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_LLM_HASS_API,
    CONF_PASSWORD,
    CONF_USERNAME,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentConfigFlow,
)
from custom_components.home_generative_agent.const import (
    CONF_ANTHROPIC_API_KEY,
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CRITICAL_ACTION_PIN,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_EXPLAIN_ENABLED,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_NAME,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_EMBEDDING_URL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM_URL,
    CONF_OPENAI_COMPATIBLE_API_KEY,
    CONF_OPENAI_COMPATIBLE_BASE_URL,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
    CONF_OPENAI_COMPATIBLE_EMBEDDING_URL,
    CONF_SENTINEL_CAMERA_ENTRY_LINKS,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_QUIET_HOURS_END,
    CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
    CONF_SENTINEL_QUIET_HOURS_START,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VLM_PROVIDER,
    CONFIG_ENTRY_VERSION,
    DEFAULT_FEATURE_TYPES,
    DOMAIN,
    FEATURE_CATEGORY_MAP,
    FEATURE_DEFS,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS,
    RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES,
    SUBENTRY_TYPE_DATABASE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_SENTINEL,
    SUBENTRY_TYPE_STT_PROVIDER,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    build_model_deployments,
    legacy_model_provider_configs,
    resolve_runtime_options,
)
from custom_components.home_generative_agent.core.subentry_types import (
    ModelProviderConfig,
)
from custom_components.home_generative_agent.core.utils import (
    CannotConnectError,
    InvalidAuthError,
)
from custom_components.home_generative_agent.flows.feature_subentry_flow import (
    FeatureSubentryFlow,
)
from custom_components.home_generative_agent.flows.model_provider_subentry_flow import (
    ModelProviderSubentryFlow,
)
from custom_components.home_generative_agent.flows.sentinel_subentry_flow import (
    SentinelSubentryFlow,
    _default_payload,
    _quiet_hour_str,
)
from custom_components.home_generative_agent.flows.stt_provider_subentry_flow import (
    SttProviderSubentryFlow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


_ASYNC_MIGRATE_ATTR = "async_migrate_entry"
async_migrate_entry = cast(
    "Any",
    getattr(hga_component, _ASYNC_MIGRATE_ATTR),
)


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
    flow._get_entry = lambda: entry  # type: ignore[attr-defined]
    flow._source = "user"
    flow.context = {"source": "user"}


def test_supported_subentry_types() -> None:
    """Ensure all subentry flow types are registered."""
    entry = MockConfigEntry(domain=DOMAIN, title="Home Generative Agent")
    supported = HomeGenerativeAgentConfigFlow.async_get_supported_subentry_types(entry)
    assert set(supported) == {
        SUBENTRY_TYPE_MODEL_PROVIDER,
        SUBENTRY_TYPE_FEATURE,
        SUBENTRY_TYPE_STT_PROVIDER,
        SUBENTRY_TYPE_SENTINEL,
    }


@pytest.mark.asyncio
async def test_stt_provider_flow_reuses_openai_provider(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STT flow can reuse an existing OpenAI model provider."""
    entry = DummyEntry()
    openai_provider = DummySubentry(
        "openai1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Cloud-LLM OpenAI",
        {"provider_type": "openai", "settings": {"api_key": "sk-reused"}},
    )
    entry.subentries[openai_provider.subentry_id] = openai_provider

    flow = SttProviderSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    async def _noop_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.stt_provider_subentry_flow.validate_openai_key",
        _noop_validate,
    )

    first = await flow.async_step_user()
    assert first.get("type") == "form"

    second = await flow.async_step_provider(
        {"provider_type": "openai", "name": "STT - OpenAI"}
    )
    assert second.get("type") == "form"

    third = await flow.async_step_credentials(
        {"openai_provider_subentry_id": "openai1"}
    )
    assert third.get("type") == "form"

    result = await flow.async_step_model({"model_name": "whisper-1"})
    assert result.get("type") == "create_entry"
    result_data = result.get("data")
    assert result_data is not None
    assert result_data["provider_type"] == "openai"
    assert result_data["settings"]["openai_provider_subentry_id"] == "openai1"
    assert result_data["settings"]["api_key"] is None


@pytest.mark.asyncio
async def test_stt_provider_flow_uses_separate_key(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STT flow stores a separate API key when no providers exist."""
    entry = DummyEntry()
    flow = SttProviderSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    async def _noop_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.stt_provider_subentry_flow.validate_openai_key",
        _noop_validate,
    )

    first = await flow.async_step_user()
    assert first.get("type") == "form"

    second = await flow.async_step_provider(
        {"provider_type": "openai", "name": "STT - OpenAI"}
    )
    assert second.get("type") == "form"

    third = await flow.async_step_credentials({"api_key": "sk-separate"})
    assert third.get("type") == "form"

    result = await flow.async_step_model({"model_name": "gpt-4o-mini-transcribe"})
    assert result.get("type") == "create_entry"
    result_data = result.get("data")
    assert result_data is not None
    assert result_data["settings"]["api_key"] == "sk-separate"
    assert result_data["settings"]["openai_provider_subentry_id"] is None


@pytest.mark.asyncio
async def test_sentinel_subentry_flow_hashes_level_increase_pin(
    hass: HomeAssistant,
) -> None:
    """Sentinel flow stores only the hashed level-increase PIN."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    first = await flow.async_step_user()
    assert first.get("type") == "form"

    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            "sentinel_cooldown_minutes": 30,
            "sentinel_entity_cooldown_minutes": 15,
            "sentinel_pending_prompt_ttl_minutes": 240,
            "sentinel_discovery_enabled": False,
            "sentinel_discovery_interval_seconds": 3600,
            "sentinel_discovery_max_records": 200,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_CRITICAL_ACTION_PIN: "1234",
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert CONF_CRITICAL_ACTION_PIN not in data
    assert data[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH]
    assert data[CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT]


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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
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


def test_resolve_runtime_options_no_sentinel_subentry_disables_sentinel() -> None:
    """When no Sentinel subentry exists, CONF_SENTINEL_ENABLED must be False."""
    entry = DummyEntry(options={})
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_ENABLED] is False


def test_resolve_runtime_options_prefers_sentinel_subentry() -> None:
    """Sentinel subentry should override legacy sentinel option keys."""
    sentinel_interval = 120
    entry = DummyEntry(
        options={
            CONF_SENTINEL_ENABLED: False,
            CONF_SENTINEL_INTERVAL_SECONDS: 999,
            CONF_EXPLAIN_ENABLED: False,
        }
    )
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: sentinel_interval,
            CONF_EXPLAIN_ENABLED: True,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: True,
            CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH: "hash",
            CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT: "salt",
        },
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_ENABLED] is True
    assert options[CONF_SENTINEL_INTERVAL_SECONDS] == sentinel_interval
    assert options[CONF_EXPLAIN_ENABLED] is True
    assert options[CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE] is True
    assert options[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] == "hash"
    assert options[CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT] == "salt"


def test_resolve_runtime_options_sentinel_notify_fallback() -> None:
    """Sentinel notify service should override global only when explicitly set."""
    entry = DummyEntry(options={CONF_NOTIFY_SERVICE: "notify.mobile_app_global"})
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_NOTIFY_SERVICE: "notify.mobile_app_sentinel",
        },
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_NOTIFY_SERVICE] == "notify.mobile_app_sentinel"

    sentinel.data.pop(CONF_NOTIFY_SERVICE)
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_NOTIFY_SERVICE] == "notify.mobile_app_global"


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
    sentinel = [
        s
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_SENTINEL
    ]
    assert providers
    assert features
    assert sentinel
    assert entry.version == CONFIG_ENTRY_VERSION


# ---------------------------------------------------------------------------
# openai_compatible provider tests
# ---------------------------------------------------------------------------


def _make_compat_flow(hass: Any, entry: DummyEntry) -> ModelProviderSubentryFlow:
    """Return a ModelProviderSubentryFlow wired up for testing."""
    flow = ModelProviderSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)
    return flow


@pytest.mark.asyncio
async def test_model_provider_flow_creates_openai_compatible(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model provider flow creates an openai_compatible subentry with base_url and api_key."""
    entry = DummyEntry()
    flow = _make_compat_flow(hass, entry)

    async def _noop_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_openai_compatible_url",
        _noop_validate,
    )

    first = await flow.async_step_user()
    assert first.get("type") == "form"

    second = await flow.async_step_deployment({"deployment": "edge"})
    assert second.get("type") == "form"

    third = await flow.async_step_provider(
        {"provider_type": "openai_compatible", "name": "Edge-LLM OpenAI Compatible"}
    )
    assert third.get("type") == "form"

    result = await flow.async_step_settings(
        {"base_url": "http://localhost:8000", CONF_API_KEY: "sk-local"}
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data["provider_type"] == "openai_compatible"
    assert data["deployment"] == "edge"
    settings = data["settings"]
    assert settings["base_url"] == "http://localhost:8000"
    assert settings["api_key"] == "sk-local"


@pytest.mark.asyncio
async def test_model_provider_flow_openai_compatible_defaults_api_key(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai_compatible flow stores 'none' when no api_key is supplied."""
    entry = DummyEntry()
    flow = _make_compat_flow(hass, entry)

    async def _noop_validate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_openai_compatible_url",
        _noop_validate,
    )

    await flow.async_step_deployment({"deployment": "edge"})
    await flow.async_step_provider({"provider_type": "openai_compatible"})

    # Omit api_key entirely — should default to "none"
    result = await flow.async_step_settings({"base_url": "http://vllm:8000"})
    assert result.get("type") == "create_entry"
    result_data = result.get("data")
    assert result_data is not None
    assert result_data["settings"]["api_key"] == "none"


@pytest.mark.asyncio
async def test_model_provider_flow_openai_compatible_cannot_connect(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation failure shows cannot_connect error and re-displays the form."""
    entry = DummyEntry()
    flow = _make_compat_flow(hass, entry)

    async def _raise_connect(*_args: Any, **_kwargs: Any) -> None:
        raise CannotConnectError

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_openai_compatible_url",
        _raise_connect,
    )

    await flow.async_step_deployment({"deployment": "edge"})
    await flow.async_step_provider({"provider_type": "openai_compatible"})

    result = await flow.async_step_settings({"base_url": "http://bad-host:8000"})
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "cannot_connect"


@pytest.mark.asyncio
async def test_model_provider_flow_openai_compatible_invalid_auth(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation 401 shows invalid_auth error and re-displays the form."""
    entry = DummyEntry()
    flow = _make_compat_flow(hass, entry)

    async def _raise_auth(*_args: Any, **_kwargs: Any) -> None:
        raise InvalidAuthError

    monkeypatch.setattr(
        "custom_components.home_generative_agent.flows.model_provider_subentry_flow.validate_openai_compatible_url",
        _raise_auth,
    )

    await flow.async_step_deployment({"deployment": "edge"})
    await flow.async_step_provider({"provider_type": "openai_compatible"})

    result = await flow.async_step_settings(
        {"base_url": "http://vllm:8000", CONF_API_KEY: "bad-key"}
    )
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_auth"


def test_model_provider_flow_openai_compatible_only_in_edge() -> None:
    """openai_compatible must appear in edge options but NOT in cloud options."""
    flow = ModelProviderSubentryFlow()

    flow._deployment = "edge"
    edge_values = [opt["value"] for opt in flow._provider_options()]
    assert "openai_compatible" in edge_values
    assert "ollama" in edge_values
    assert "openai" not in edge_values  # openai is cloud-only

    flow._deployment = "cloud"
    cloud_values = [opt["value"] for opt in flow._provider_options()]
    assert "openai_compatible" not in cloud_values
    assert "openai" in cloud_values
    assert "gemini" in cloud_values


def test_resolve_runtime_options_openai_compatible() -> None:
    """openai_compatible subentry propagates base_url and api_key into runtime options."""
    provider = DummySubentry(
        "compat1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Edge-LLM OpenAI Compatible",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["chat", "vlm", "summarization", "embedding"],
            "settings": {
                "base_url": "http://localhost:8000",
                "api_key": "sk-local",
            },
        },
    )
    feature = DummySubentry(
        "feature1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "compat1",
            "name": "Conversation",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "gpt-4o"},
        },
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_OPENAI_COMPATIBLE_BASE_URL] == "http://localhost:8000"
    assert options[CONF_OPENAI_COMPATIBLE_API_KEY] == "sk-local"


def test_provider_capabilities_includes_openai_compatible() -> None:
    """MODEL_CATEGORY_SPECS includes openai_compatible in all four model categories."""
    for category in ("chat", "vlm", "summarization", "embedding"):
        spec = MODEL_CATEGORY_SPECS[category]
        assert "openai_compatible" in spec["providers"], (
            f"openai_compatible missing from {category} providers"
        )
        assert "openai_compatible" in spec["recommended_models"], (
            f"openai_compatible missing from {category} recommended_models"
        )
        assert "openai_compatible" in spec["model_keys"], (
            f"openai_compatible missing from {category} model_keys"
        )


def test_vlm_ollama_supported_models() -> None:
    """Curated Ollama VLM options include the tested models (issue #469)."""
    vlm_ollama = MODEL_CATEGORY_SPECS["vlm"]["providers"]["ollama"]
    assert "qwen2.5vl:7b" in vlm_ollama
    assert "qwen3-vl:8b" in vlm_ollama
    assert "gemma3:4b" in vlm_ollama
    recommended = MODEL_CATEGORY_SPECS["vlm"]["recommended_models"]["ollama"]
    assert recommended in vlm_ollama


# ---------------------------------------------------------------------------
# Sentinel subentry — daily digest fields (Fix 4)
# ---------------------------------------------------------------------------


def test_sentinel_default_payload_contains_digest_keys() -> None:
    """_default_payload() includes both daily digest keys with non-empty defaults."""
    payload = _default_payload()
    assert CONF_SENTINEL_DAILY_DIGEST_ENABLED in payload
    assert CONF_SENTINEL_DAILY_DIGEST_TIME in payload
    # Default time must be a non-empty HH:MM:SS string.
    assert isinstance(payload[CONF_SENTINEL_DAILY_DIGEST_TIME], str)
    assert len(payload[CONF_SENTINEL_DAILY_DIGEST_TIME]) > 0


def test_sentinel_schema_contains_digest_fields(hass: Any) -> None:
    """_schema() includes both daily digest selectors."""
    flow = SentinelSubentryFlow()
    flow.hass = hass
    schema = flow._schema(_default_payload())
    schema_keys = {str(k) for k in schema.schema}
    assert CONF_SENTINEL_DAILY_DIGEST_ENABLED in schema_keys
    assert CONF_SENTINEL_DAILY_DIGEST_TIME in schema_keys


@pytest.mark.asyncio
async def test_sentinel_subentry_flow_accepts_digest_fields(hass: Any) -> None:
    """Sentinel flow stores digest fields when provided in user_input."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            "sentinel_cooldown_minutes": 30,
            "sentinel_entity_cooldown_minutes": 15,
            "sentinel_pending_prompt_ttl_minutes": 240,
            "sentinel_discovery_enabled": False,
            "sentinel_discovery_interval_seconds": 3600,
            "sentinel_discovery_max_records": 200,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "07:30:00",
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_DAILY_DIGEST_ENABLED] is True
    assert data[CONF_SENTINEL_DAILY_DIGEST_TIME] == "07:30:00"


def test_sentinel_default_payload_contains_camera_entry_links() -> None:
    """_default_payload() includes sentinel_camera_entry_links as an empty dict."""
    payload = _default_payload()
    assert CONF_SENTINEL_CAMERA_ENTRY_LINKS in payload
    assert payload[CONF_SENTINEL_CAMERA_ENTRY_LINKS] == {}


def test_sentinel_schema_contains_camera_entry_links(hass: Any) -> None:
    """_schema() includes the sentinel_camera_entry_links text selector."""
    flow = SentinelSubentryFlow()
    flow.hass = hass
    schema = flow._schema(_default_payload())
    schema_keys = {str(k) for k in schema.schema}
    assert CONF_SENTINEL_CAMERA_ENTRY_LINKS in schema_keys


@pytest.mark.asyncio
async def test_sentinel_flow_camera_entry_links_valid_json(hass: Any) -> None:
    """Flow parses a valid JSON string into a dict for sentinel_camera_entry_links."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_CAMERA_ENTRY_LINKS: '{"camera.driveway": ["lock.front_door"]}',
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_CAMERA_ENTRY_LINKS] == {
        "camera.driveway": ["lock.front_door"]
    }


@pytest.mark.asyncio
async def test_sentinel_flow_camera_entry_links_invalid_json(hass: Any) -> None:
    """Flow returns an error for malformed sentinel_camera_entry_links JSON."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_CAMERA_ENTRY_LINKS: "not valid json {{",
        }
    )
    assert result is not None
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_camera_entry_links"


@pytest.mark.asyncio
async def test_sentinel_flow_camera_entry_links_wrong_structure(hass: Any) -> None:
    """Flow returns an error when sentinel_camera_entry_links is not dict[str, list[str]]."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            # Value is a string, not a list — wrong structure.
            CONF_SENTINEL_CAMERA_ENTRY_LINKS: '{"camera.driveway": "lock.front_door"}',
        }
    )
    assert result is not None
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_camera_entry_links"


def _quiet_hours_flow(hass: Any) -> tuple[SentinelSubentryFlow, DummyEntry]:
    """Return a Sentinel flow wired with stub form/entry callbacks."""
    entry = DummyEntry()
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)
    return flow, entry


def test_sentinel_default_payload_contains_quiet_hours_severities() -> None:
    """_default_payload() includes quiet-hours severities but no start/end (off)."""
    payload = _default_payload()
    assert payload[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == (
        RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES
    )
    # Start/end absent means quiet hours are disabled by default.
    assert CONF_SENTINEL_QUIET_HOURS_START not in payload
    assert CONF_SENTINEL_QUIET_HOURS_END not in payload


def test_sentinel_schema_contains_quiet_hours_fields(hass: Any) -> None:
    """_schema() includes the three quiet-hours selectors."""
    flow = SentinelSubentryFlow()
    flow.hass = hass
    schema = flow._schema(_default_payload())
    schema_keys = {str(k) for k in schema.schema}
    assert CONF_SENTINEL_QUIET_HOURS_START in schema_keys
    assert CONF_SENTINEL_QUIET_HOURS_END in schema_keys
    assert CONF_SENTINEL_QUIET_HOURS_SEVERITIES in schema_keys


@pytest.mark.asyncio
async def test_sentinel_flow_stores_quiet_hours(hass: Any) -> None:
    """Flow converts quiet-hours select values to ints and stores severities."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "22",
            CONF_SENTINEL_QUIET_HOURS_END: "7",
            CONF_SENTINEL_QUIET_HOURS_SEVERITIES: ["low", "medium"],
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_QUIET_HOURS_START] == 22
    assert data[CONF_SENTINEL_QUIET_HOURS_END] == 7
    assert data[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == ["low", "medium"]


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_disabled_by_default(hass: Any) -> None:
    """Empty quiet-hours selects leave the keys absent (feature off)."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "",
            CONF_SENTINEL_QUIET_HOURS_END: "",
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert CONF_SENTINEL_QUIET_HOURS_START not in data
    assert CONF_SENTINEL_QUIET_HOURS_END not in data
    assert data[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == (
        RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES
    )


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_incomplete_pair_errors(hass: Any) -> None:
    """Setting only one of start/end returns a form error."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "22",
            CONF_SENTINEL_QUIET_HOURS_END: "",
        }
    )
    assert result is not None
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "quiet_hours_incomplete"


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_invalid_hour_errors(hass: Any) -> None:
    """Out-of-range or non-numeric quiet-hours values return a form error."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "25",
            CONF_SENTINEL_QUIET_HOURS_END: "7",
        }
    )
    assert result is not None
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_quiet_hours"


def _suggested_value(schema: Any, key: str) -> Any:
    """Return the suggested_value attached to a schema key, or None."""
    for marker in schema.schema:
        if str(marker) == key:
            description = getattr(marker, "description", None)
            if isinstance(description, dict):
                return description.get("suggested_value")
    return None


def test_quiet_hour_str_normalizes_values() -> None:
    """_quiet_hour_str maps stored ints to select strings and garbage to ''."""
    key = CONF_SENTINEL_QUIET_HOURS_START
    assert _quiet_hour_str({}, key) == ""
    assert _quiet_hour_str({key: None}, key) == ""
    assert _quiet_hour_str({key: ""}, key) == ""
    assert _quiet_hour_str({key: 22}, key) == "22"
    assert _quiet_hour_str({key: "7"}, key) == "7"
    # Corrupted stored values degrade to disabled rather than crashing the form.
    assert _quiet_hour_str({key: "garbage"}, key) == ""


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_non_numeric_errors(hass: Any) -> None:
    """Non-numeric quiet-hours values return the invalid_quiet_hours error."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "abc",
            CONF_SENTINEL_QUIET_HOURS_END: "7",
        }
    )
    assert result is not None
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_quiet_hours"


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_reconfigure_prefill(hass: Any) -> None:
    """Stored int quiet hours prefill the form as select string values."""
    flow, entry = _quiet_hours_flow(hass)
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_QUIET_HOURS_START: 22,
            CONF_SENTINEL_QUIET_HOURS_END: 7,
            CONF_SENTINEL_QUIET_HOURS_SEVERITIES: ["low", "medium"],
        },
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    result = await flow.async_step_settings(None)
    assert result.get("type") == "form"
    schema = result.get("data_schema")
    assert schema is not None
    # Ints stored in the subentry must surface as select option strings.
    assert _suggested_value(schema, CONF_SENTINEL_QUIET_HOURS_START) == "22"
    assert _suggested_value(schema, CONF_SENTINEL_QUIET_HOURS_END) == "7"
    assert _suggested_value(schema, CONF_SENTINEL_QUIET_HOURS_SEVERITIES) == [
        "low",
        "medium",
    ]


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_error_redisplay_preserves_input(
    hass: Any,
) -> None:
    """After a quiet-hours error the redisplayed form keeps the entered values."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "22",
            CONF_SENTINEL_QUIET_HOURS_END: "",
        }
    )
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "quiet_hours_incomplete"
    schema = result.get("data_schema")
    assert schema is not None
    assert _suggested_value(schema, CONF_SENTINEL_QUIET_HOURS_START) == "22"
    assert _suggested_value(schema, CONF_SENTINEL_QUIET_HOURS_END) == ""


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_midnight_boundary(hass: Any) -> None:
    """Hour 0 (midnight) is a valid boundary value, not treated as Disabled."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "0",
            CONF_SENTINEL_QUIET_HOURS_END: "23",
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_QUIET_HOURS_START] == 0
    assert data[CONF_SENTINEL_QUIET_HOURS_END] == 23
    # A stored int 0 must round-trip to the "0" select value, not "Disabled".
    assert (
        _quiet_hour_str(
            {CONF_SENTINEL_QUIET_HOURS_START: 0}, CONF_SENTINEL_QUIET_HOURS_START
        )
        == "0"
    )


@pytest.mark.asyncio
async def test_sentinel_flow_quiet_hours_severities_filtered(hass: Any) -> None:
    """Unknown severity values are dropped on save; known values are kept."""
    flow, _entry = _quiet_hours_flow(hass)

    await flow.async_step_user()
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_INTERVAL_SECONDS: 300,
            CONF_EXPLAIN_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_QUIET_HOURS_START: "22",
            CONF_SENTINEL_QUIET_HOURS_END: "7",
            CONF_SENTINEL_QUIET_HOURS_SEVERITIES: ["low", "critical", "Medium"],
        }
    )
    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == ["low"]


def test_resolve_runtime_options_passes_quiet_hours_through() -> None:
    """Quiet-hours keys stored in the Sentinel subentry reach runtime options."""
    entry = DummyEntry(options={})
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_QUIET_HOURS_START: 22,
            CONF_SENTINEL_QUIET_HOURS_END: 7,
            CONF_SENTINEL_QUIET_HOURS_SEVERITIES: ["low", "medium"],
        },
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_QUIET_HOURS_START] == 22
    assert options[CONF_SENTINEL_QUIET_HOURS_END] == 7
    assert options[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == ["low", "medium"]


def test_resolve_runtime_options_quiet_hours_default_disabled() -> None:
    """Without stored quiet-hours keys, resolved options disable the feature."""
    entry = DummyEntry(options={})
    sentinel = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True},
    )
    entry.subentries[sentinel.subentry_id] = sentinel

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_QUIET_HOURS_START] is None
    assert options[CONF_SENTINEL_QUIET_HOURS_END] is None
    assert options[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] == (
        RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES
    )


# ---------------------------------------------------------------------------
# v5 -> v6 migration: CONF_LLM_HASS_API normalisation
# ---------------------------------------------------------------------------


def _make_v5_entry(hass: Any, options: dict[str, Any]) -> Any:
    """Return a v5 MockConfigEntry with the given options, added to hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=5,
        data={},
        options=options,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_migration_v5_to_v6_string_api_id(hass: HomeAssistant) -> None:
    """v5 string API id is wrapped in a list."""
    entry = _make_v5_entry(hass, {CONF_LLM_HASS_API: "assist"})
    assert await async_migrate_entry(hass, entry)
    assert entry.options[CONF_LLM_HASS_API] == ["assist"]
    assert entry.version == CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migration_v5_to_v6_none_sentinel(hass: HomeAssistant) -> None:
    """v5 'none' sentinel is converted to an empty list (no APIs)."""
    entry = _make_v5_entry(hass, {CONF_LLM_HASS_API: "none"})
    assert await async_migrate_entry(hass, entry)
    assert entry.options.get(CONF_LLM_HASS_API) == []
    assert entry.version == CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migration_v5_to_v6_absent_key(hass: HomeAssistant) -> None:
    """v5 absent CONF_LLM_HASS_API key is converted to empty list."""
    entry = _make_v5_entry(hass, {})
    assert await async_migrate_entry(hass, entry)
    assert entry.options.get(CONF_LLM_HASS_API) == []
    assert entry.version == CONFIG_ENTRY_VERSION


# ---------------------------------------------------------------------------
# LM Studio / openai_compatible embedding dims (fix #375)
# ---------------------------------------------------------------------------


def test_resolve_runtime_options_openai_compatible_embedding_dims() -> None:
    """Configured embedding dims in provider settings propagate to runtime options."""
    provider = DummySubentry(
        "compat1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Edge-LLM",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["embedding"],
            "settings": {
                "base_url": "http://lmstudio:1234",
                "api_key": "none",
                CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS: 768,
            },
        },
    )
    entry = DummyEntry()
    entry.subentries = {provider.subentry_id: provider}

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_EMBEDDING_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS] == 768


def test_resolve_runtime_options_openai_compatible_embedding_dims_none() -> None:
    """When dims is absent from provider settings, key is not written to options."""
    provider = DummySubentry(
        "compat1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Edge-LLM",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["embedding"],
            "settings": {
                "base_url": "http://lmstudio:1234",
                "api_key": "none",
            },
        },
    )
    entry = DummyEntry()
    entry.subentries = {provider.subentry_id: provider}

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS not in options


def test_recommended_openai_compatible_embedding_dims_value() -> None:
    """Default dims constant is 768 — matches nomic-embed-text native output size."""
    assert RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_DIMS == 768


# ---------------------------------------------------------------------------
# Embedding feature (issue #457)
# ---------------------------------------------------------------------------


def test_embedding_feature_registered() -> None:
    """The embedding feature type is defined and mapped to its model category."""
    assert "embedding" in FEATURE_DEFS
    assert FEATURE_DEFS["embedding"]["required"] is False
    assert FEATURE_CATEGORY_MAP["embedding"] == "embedding"
    assert "embedding" in DEFAULT_FEATURE_TYPES


def _make_entry_with_features(
    providers: list[DummySubentry], features: list[DummySubentry]
) -> DummyEntry:
    entry = DummyEntry()
    entry.subentries = {s.subentry_id: s for s in [*providers, *features]}
    return entry


def test_embedding_feature_selects_separate_openai_compatible_server() -> None:
    """
    Chat and embedding features on different OpenAI-compatible servers.

    The embedding provider's base URL must land in the embedding-specific
    option key without clobbering the chat provider's base URL.
    """
    chat_provider = DummySubentry(
        "chat_prov",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Chat llama-server",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["chat", "embedding"],
            "settings": {"base_url": "http://chat-host:8080", "api_key": "sk-chat"},
        },
    )
    emb_provider = DummySubentry(
        "emb_prov",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Embedding llama-server",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["embedding"],
            "settings": {
                "base_url": "http://emb-host:8081",
                "api_key": "sk-emb",
                CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS: 768,
            },
        },
    )
    chat_feature = DummySubentry(
        "chat_feat",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "chat_prov",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "qwen3.5-vl"},
        },
    )
    emb_feature = DummySubentry(
        "emb_feat",
        SUBENTRY_TYPE_FEATURE,
        "Embeddings",
        {
            "feature_type": "embedding",
            "model_provider_id": "emb_prov",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "nomic-embed-text-v1.5"},
        },
    )
    entry = _make_entry_with_features(
        [chat_provider, emb_provider], [chat_feature, emb_feature]
    )

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_EMBEDDING_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_OPENAI_COMPATIBLE_BASE_URL] == "http://chat-host:8080"
    assert options[CONF_OPENAI_COMPATIBLE_API_KEY] == "sk-chat"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_URL] == "http://emb-host:8081"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY] == "sk-emb"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_DIMS] == 768
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL] == "nomic-embed-text-v1.5"


def test_embedding_feature_selects_ollama_on_custom_url() -> None:
    """An Ollama embedding provider propagates its own URL for embeddings."""
    chat_provider = DummySubentry(
        "chat_prov",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Chat llama-server",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["chat", "embedding"],
            "settings": {"base_url": "http://chat-host:8080", "api_key": "none"},
        },
    )
    emb_provider = DummySubentry(
        "emb_prov",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Embedding Ollama",
        {
            "provider_type": "ollama",
            "capabilities": ["embedding"],
            "settings": {"base_url": "http://ollama-host:11434"},
        },
    )
    chat_feature = DummySubentry(
        "chat_feat",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "chat_prov",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "qwen3.5-vl"},
        },
    )
    emb_feature = DummySubentry(
        "emb_feat",
        SUBENTRY_TYPE_FEATURE,
        "Embeddings",
        {
            "feature_type": "embedding",
            "model_provider_id": "emb_prov",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "mxbai-embed-large"},
        },
    )
    entry = _make_entry_with_features(
        [chat_provider, emb_provider], [chat_feature, emb_feature]
    )

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_EMBEDDING_MODEL_PROVIDER] == "ollama"
    assert options[CONF_OLLAMA_EMBEDDING_URL] == "http://ollama-host:11434"
    # Chat provider's URL is untouched.
    assert options[CONF_OPENAI_COMPATIBLE_BASE_URL] == "http://chat-host:8080"


def test_embedding_without_feature_inherits_chat_provider() -> None:
    """Without an embedding feature, the chat provider is inherited (legacy behavior)."""
    chat_provider = DummySubentry(
        "chat_prov",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Chat llama-server",
        {
            "provider_type": "openai_compatible",
            "capabilities": ["chat", "embedding"],
            "settings": {"base_url": "http://chat-host:8080", "api_key": "sk-x"},
        },
    )
    chat_feature = DummySubentry(
        "chat_feat",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "chat_prov",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "qwen3.5-vl"},
        },
    )
    entry = _make_entry_with_features([chat_provider], [chat_feature])

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_EMBEDDING_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_URL] == "http://chat-host:8080"
    assert options[CONF_OPENAI_COMPATIBLE_EMBEDDING_API_KEY] == "sk-x"


# ---------------------------------------------------------------------------
# build_model_deployments
# ---------------------------------------------------------------------------


def test_build_model_deployments_ollama_returns_edge() -> None:
    """Ollama provider must map its capabilities to 'edge' deployment."""
    provider = ModelProviderConfig(
        entry_id="p_ollama",
        name="Ollama Local",
        provider_type="ollama",
        capabilities={"chat", "vlm", "summarization"},
        data={},
        deployment="edge",
    )
    entry = DummyEntry()
    result = build_model_deployments(entry, {"p_ollama": provider}, {})  # type: ignore[arg-type]
    assert result.get("chat") == "edge"


def test_build_model_deployments_openai_returns_cloud() -> None:
    """OpenAI provider must map its capabilities to 'cloud' deployment."""
    provider = ModelProviderConfig(
        entry_id="p_openai",
        name="OpenAI Cloud",
        provider_type="openai",
        capabilities={"chat"},
        data={"settings": {}},
        deployment="cloud",
    )
    entry = DummyEntry()
    result = build_model_deployments(entry, {"p_openai": provider}, {})  # type: ignore[arg-type]
    assert result.get("chat") == "cloud"


def test_build_model_deployments_empty_providers_returns_empty() -> None:
    """No providers must yield an empty deployment map."""
    entry = DummyEntry()
    result = build_model_deployments(entry, {}, {})  # type: ignore[arg-type]
    assert result == {}


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


def test_model_provider_flow_anthropic_only_in_cloud() -> None:
    """Anthropic must appear in cloud options but NOT in edge options."""
    flow = ModelProviderSubentryFlow()

    flow._deployment = "cloud"
    cloud_values = [opt["value"] for opt in flow._provider_options()]
    assert "anthropic" in cloud_values

    flow._deployment = "edge"
    edge_values = [opt["value"] for opt in flow._provider_options()]
    assert "anthropic" not in edge_values


def test_provider_capabilities_includes_anthropic_for_chat_vlm_summarization() -> None:
    """MODEL_CATEGORY_SPECS includes anthropic in chat, vlm, and summarization."""
    for category in ("chat", "vlm", "summarization"):
        spec = MODEL_CATEGORY_SPECS[category]
        assert "anthropic" in spec["providers"], (
            f"anthropic missing from {category} providers"
        )
        assert "anthropic" in spec["recommended_models"], (
            f"anthropic missing from {category} recommended_models"
        )
        assert "anthropic" in spec["model_keys"], (
            f"anthropic missing from {category} model_keys"
        )


def test_provider_capabilities_anthropic_absent_from_embedding() -> None:
    """Anthropic must NOT appear in the embedding category."""
    spec = MODEL_CATEGORY_SPECS["embedding"]
    assert "anthropic" not in spec.get("providers", {})


def test_resolve_runtime_options_anthropic_propagates_api_key() -> None:
    """Anthropic subentry propagates api_key into CONF_ANTHROPIC_API_KEY."""
    provider = DummySubentry(
        "anthro1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Cloud-LLM Anthropic",
        {
            "provider_type": "anthropic",
            "deployment": "cloud",
            "capabilities": ["chat", "vlm", "summarization"],
            "settings": {"api_key": "sk-ant-test"},
        },
    )
    feature = DummySubentry(
        "feature1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {
            "feature_type": "conversation",
            "model_provider_id": "anthro1",
            "name": "Conversation",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "claude-sonnet-4-6"},
        },
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_CHAT_MODEL_PROVIDER] == "anthropic"
    assert options[CONF_ANTHROPIC_API_KEY] == "sk-ant-test"


def test_build_model_deployments_anthropic_returns_cloud() -> None:
    """Anthropic provider must map its capabilities to 'cloud' deployment."""
    provider = ModelProviderConfig(
        entry_id="p_anthropic",
        name="Cloud-LLM Anthropic",
        provider_type="anthropic",
        capabilities={"chat", "vlm", "summarization"},
        data={"settings": {}},
        deployment="cloud",
    )
    entry = DummyEntry()
    result = build_model_deployments(entry, {"p_anthropic": provider}, {})  # type: ignore[arg-type]
    assert result.get("chat") == "cloud"
    assert result.get("vlm") == "cloud"
    assert result.get("summarization") == "cloud"


# ---------------------------------------------------------------------------
# Basic and Advanced setup mode — feature flow (Issue #433)
# ---------------------------------------------------------------------------


def _make_feature_flow_for_basic(hass: Any, entry: DummyEntry) -> FeatureSubentryFlow:
    """Return a FeatureSubentryFlow patched for basic-setup tests."""
    flow = FeatureSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "step_id": kwargs.get("step_id"),
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
        "description_placeholders": kwargs.get("description_placeholders"),
    }
    flow.async_abort = lambda **kwargs: {  # type: ignore[assignment]
        "type": "abort",
        "reason": kwargs.get("reason"),
    }
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)

    def _add_subentry(_entry: Any, subentry: Any) -> None:
        entry.subentries[subentry.subentry_id] = subentry

    def _update_subentry(
        _entry: Any, subentry: Any, data: Any = None, **_kw: Any
    ) -> None:
        if data is not None:
            subentry.data = dict(data)

    def _update_entry(_entry: Any, **kwargs: Any) -> None:
        if "options" in kwargs:
            _entry.options = kwargs["options"]

    flow.hass.config_entries.async_add_subentry = _add_subentry  # type: ignore[assignment]
    flow.hass.config_entries.async_update_subentry = _update_subentry  # type: ignore[assignment]
    flow.hass.config_entries.async_update_entry = _update_entry  # type: ignore[assignment]
    return flow


@pytest.mark.asyncio
async def test_feature_new_setup_shows_mode_selector(hass: HomeAssistant) -> None:
    """New setup shows the Basic/Advanced mode selector."""
    entry = DummyEntry()
    flow = _make_feature_flow_for_basic(hass, entry)

    result = await flow.async_step_user()
    assert result.get("type") == "form"
    assert result.get("step_id") == "setup_mode"


@pytest.mark.asyncio
async def test_feature_mode_selector_shows_overwrite_warning_when_features_exist(
    hass: HomeAssistant,
) -> None:
    """Mode selector includes overwrite warning when feature subentries already exist."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider",
        {"provider_type": "openai", "capabilities": ["chat"]},
    )
    existing_feature = DummySubentry(
        "feat1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "prov1"},
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        existing_feature.subentry_id: existing_feature,
    }
    flow = _make_feature_flow_for_basic(hass, entry)

    result = await flow.async_step_user()
    placeholders = result.get("description_placeholders") or {}
    assert placeholders.get("overwrite_warning") != ""


@pytest.mark.asyncio
async def test_feature_mode_selector_no_warning_when_no_features_exist(
    hass: HomeAssistant,
) -> None:
    """Mode selector has an empty overwrite_warning when no feature subentries exist."""
    entry = DummyEntry()
    flow = _make_feature_flow_for_basic(hass, entry)

    result = await flow.async_step_user()
    placeholders = result.get("description_placeholders") or {}
    assert placeholders.get("overwrite_warning") == ""


@pytest.mark.asyncio
async def test_feature_basic_setup_creates_all_features(hass: HomeAssistant) -> None:
    """Basic setup creates feature subentries for all three features when a compatible provider exists."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "All-in-one Provider",
        {"provider_type": "openai", "capabilities": ["chat", "vlm", "summarization"]},
    )
    entry = DummyEntry()
    entry.subentries[provider.subentry_id] = provider

    flow = _make_feature_flow_for_basic(hass, entry)

    await flow.async_step_user()
    result = await flow.async_step_setup_mode({"setup_mode": "basic"})

    # Basic setup skips the database form and shows the status screen directly.
    assert result.get("type") == "form"
    assert result.get("step_id") == "setup_status"

    feature_types = {
        s.data.get("feature_type")
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_FEATURE
    }
    assert feature_types == {
        "conversation",
        "camera_image_analysis",
        "conversation_summary",
    }

    # A database subentry with default credentials was created.
    db_subentry = next(
        (
            s
            for s in entry.subentries.values()
            if s.subentry_type == SUBENTRY_TYPE_DATABASE
        ),
        None,
    )
    assert db_subentry is not None


@pytest.mark.asyncio
async def test_feature_basic_setup_assigns_first_compatible_provider(
    hass: HomeAssistant,
) -> None:
    """Basic setup assigns the first provider in entry.subentries.values() order."""
    prov_a = DummySubentry(
        "prov_a",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider A",
        {"provider_type": "openai", "capabilities": ["chat", "vlm", "summarization"]},
    )
    prov_b = DummySubentry(
        "prov_b",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider B",
        {"provider_type": "openai", "capabilities": ["chat", "vlm", "summarization"]},
    )
    entry = DummyEntry()
    entry.subentries["prov_a"] = prov_a
    entry.subentries["prov_b"] = prov_b

    flow = _make_feature_flow_for_basic(hass, entry)

    await flow.async_step_setup_mode({"setup_mode": "basic"})

    conversation_subentry = next(
        (
            s
            for s in entry.subentries.values()
            if s.subentry_type == SUBENTRY_TYPE_FEATURE
            and s.data.get("feature_type") == "conversation"
        ),
        None,
    )
    assert conversation_subentry is not None
    assert conversation_subentry.data.get("model_provider_id") == "prov_a"


@pytest.mark.asyncio
async def test_feature_basic_setup_skips_camera_without_vlm_provider(
    hass: HomeAssistant,
) -> None:
    """Basic setup skips Camera Image Analysis when no VLM-capable provider exists."""
    chat_only = DummySubentry(
        "chat1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Chat Only",
        {"provider_type": "openai", "capabilities": ["chat", "summarization"]},
    )
    entry = DummyEntry()
    entry.subentries[chat_only.subentry_id] = chat_only

    flow = _make_feature_flow_for_basic(hass, entry)

    await flow.async_step_setup_mode({"setup_mode": "basic"})

    feature_types = {
        s.data.get("feature_type")
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_FEATURE
    }
    assert "camera_image_analysis" not in feature_types
    assert "conversation" in feature_types
    assert "camera_image_analysis" in flow._basic_skipped_features


@pytest.mark.asyncio
async def test_feature_basic_setup_aborts_when_no_providers(
    hass: HomeAssistant,
) -> None:
    """Basic setup aborts with no_provider_for_basic_setup when no providers exist."""
    entry = DummyEntry()
    flow = _make_feature_flow_for_basic(hass, entry)

    result = await flow.async_step_setup_mode({"setup_mode": "basic"})

    assert result.get("type") == "abort"
    assert result.get("reason") == "no_provider_for_basic_setup"


@pytest.mark.asyncio
async def test_feature_basic_setup_status_screen(hass: HomeAssistant) -> None:
    """Basic setup status screen shows feature and database status after writes."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider",
        {"provider_type": "openai", "capabilities": ["chat", "vlm", "summarization"]},
    )
    db_subentry = DummySubentry(
        "db1",
        SUBENTRY_TYPE_DATABASE,
        "Database",
        {"username": "user"},
    )
    feature_subentry = DummySubentry(
        "feat1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation"},
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        db_subentry.subentry_id: db_subentry,
        feature_subentry.subentry_id: feature_subentry,
    }

    flow = _make_feature_flow_for_basic(hass, entry)
    flow._basic_mode = True
    flow._basic_skipped_features = ["camera_image_analysis"]

    result = await flow.async_step_setup_status(None)

    assert result.get("type") == "form"
    assert result.get("step_id") == "setup_status"
    placeholders = result.get("description_placeholders") or {}
    assert "Conversation" in placeholders.get("features_enabled", "")
    assert "Configured" in placeholders.get("database_status", "")
    assert "Camera Image Analysis" in placeholders.get("skipped_section", "")

    # Submitting the form aborts with setup_complete.
    done = await flow.async_step_setup_status({})
    assert done.get("type") == "abort"
    assert done.get("reason") == "setup_complete"


@pytest.mark.asyncio
async def test_feature_reconfigure_bypasses_mode_selector(
    hass: HomeAssistant,
) -> None:
    """async_step_reconfigure goes directly to the Advanced path, not the mode selector."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider",
        {"provider_type": "openai", "capabilities": ["chat"]},
    )
    feature = DummySubentry(
        "feat1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "prov1"},
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    flow = _make_feature_flow_for_basic(hass, entry)
    flow.context["subentry_id"] = feature.subentry_id

    result = await flow.async_step_reconfigure()

    assert result.get("type") == "form"
    assert result.get("step_id") != "setup_mode"


@pytest.mark.asyncio
async def test_feature_advanced_mode_reaches_feature_enable(
    hass: HomeAssistant,
) -> None:
    """Selecting Advanced setup routes to the feature_enable form."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Provider",
        {"provider_type": "openai", "capabilities": ["chat"]},
    )
    feature = DummySubentry(
        "feat1",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "prov1"},
    )
    entry = DummyEntry()
    entry.subentries = {
        provider.subentry_id: provider,
        feature.subentry_id: feature,
    }

    flow = _make_feature_flow_for_basic(hass, entry)

    await flow.async_step_user()
    result = await flow.async_step_setup_mode({"setup_mode": "advanced"})

    assert result.get("type") == "form"
    assert result.get("step_id") == "feature_enable"


# ---------------------------------------------------------------------------
# Basic and Advanced setup mode — Sentinel flow (Issue #433)
# ---------------------------------------------------------------------------


def _make_sentinel_flow_for_mode(hass: Any, entry: DummyEntry) -> SentinelSubentryFlow:
    """Return a SentinelSubentryFlow patched for mode-selector tests."""
    flow = SentinelSubentryFlow()
    flow.hass = hass
    flow.async_show_form = lambda **kwargs: {  # type: ignore[assignment]
        "type": "form",
        "step_id": kwargs.get("step_id"),
        "data_schema": kwargs["data_schema"],
        "errors": kwargs.get("errors"),
        "description_placeholders": kwargs.get("description_placeholders"),
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
    flow._schedule_reload = lambda: None  # type: ignore[assignment]
    _patch_entry(flow, entry)
    return flow


@pytest.mark.asyncio
async def test_sentinel_new_setup_shows_mode_selector(hass: HomeAssistant) -> None:
    """New Sentinel setup shows the Basic/Advanced mode selector."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    result = await flow.async_step_user()
    assert result.get("type") == "form"
    assert result.get("step_id") == "setup_mode"


@pytest.mark.asyncio
async def test_sentinel_mode_selector_shows_overwrite_warning_when_configured(
    hass: HomeAssistant,
) -> None:
    """Mode selector includes overwrite warning when a Sentinel subentry already exists."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)

    result = await flow.async_step_user()
    placeholders = result.get("description_placeholders") or {}
    assert placeholders.get("overwrite_warning") != ""


@pytest.mark.asyncio
async def test_sentinel_mode_selector_no_warning_when_not_configured(
    hass: HomeAssistant,
) -> None:
    """Mode selector has an empty overwrite_warning when no Sentinel subentry exists."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    result = await flow.async_step_user()
    placeholders = result.get("description_placeholders") or {}
    assert placeholders.get("overwrite_warning") == ""


@pytest.mark.asyncio
async def test_sentinel_basic_setup_stores_defaults_plus_user_fields(
    hass: HomeAssistant,
) -> None:
    """Sentinel Basic setup stores _default_payload() values plus the user-provided fields."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "07:00:00",
        }
    )

    assert result.get("type") == "create_entry"
    data = result.get("data")
    assert data is not None
    assert data[CONF_SENTINEL_ENABLED] is False
    assert data[CONF_SENTINEL_DAILY_DIGEST_ENABLED] is True
    assert data[CONF_SENTINEL_DAILY_DIGEST_TIME] == "07:00:00"
    # All default-payload keys must also be present.
    defaults = _default_payload()
    for key in defaults:
        assert key in data, f"missing default key in basic settings result: {key}"


@pytest.mark.asyncio
async def test_sentinel_basic_setup_rejects_invalid_pin(hass: HomeAssistant) -> None:
    """Sentinel Basic setup returns an error for a non-digit PIN."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
            CONF_CRITICAL_ACTION_PIN: "abc",
        }
    )
    assert result.get("type") == "form"
    assert (result.get("errors") or {}).get("base") == "invalid_pin"


@pytest.mark.asyncio
async def test_sentinel_advanced_setup_still_works(hass: HomeAssistant) -> None:
    """Selecting Advanced setup routes to the full settings form."""
    entry = DummyEntry()
    flow = _make_sentinel_flow_for_mode(hass, entry)

    await flow.async_step_user()
    result = await flow.async_step_setup_mode({"setup_mode": "advanced"})

    assert result.get("type") == "form"
    assert result.get("step_id") == "settings"


@pytest.mark.asyncio
async def test_sentinel_reconfigure_bypasses_mode_selector(
    hass: HomeAssistant,
) -> None:
    """async_step_reconfigure goes directly to the full settings form."""
    sentinel_sub = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True},
    )
    entry = DummyEntry()
    entry.subentries[sentinel_sub.subentry_id] = sentinel_sub

    flow = _make_sentinel_flow_for_mode(hass, entry)

    result = await flow.async_step_reconfigure()

    assert result.get("type") == "form"
    assert result.get("step_id") != "setup_mode"


@pytest.mark.asyncio
async def test_sentinel_user_flow_updates_existing_not_duplicate(
    hass: HomeAssistant,
) -> None:
    """User flow with an existing Sentinel subentry updates it, not create a duplicate."""
    existing = DummySubentry(
        "sentinel1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing

    flow = _make_sentinel_flow_for_mode(hass, entry)
    update_calls: list[Any] = []
    flow.async_update_and_abort = lambda *args, **_kwargs: (  # type: ignore[assignment]
        update_calls.append(args)
        or {"type": "abort", "reason": "reconfigure_successful"}
    )

    await flow.async_step_setup_mode({"setup_mode": "advanced"})
    # Minimal user_input — schema validation is bypassed in unit tests
    result = await flow.async_step_settings(
        {
            CONF_SENTINEL_ENABLED: False,
            CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: False,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
            CONF_SENTINEL_CAMERA_ENTRY_LINKS: "{}",
        }
    )

    assert result.get("type") == "abort", "expected update-and-abort, not create_entry"
    assert len(update_calls) == 1, "async_update_and_abort should be called once"


@pytest.mark.asyncio
async def test_sentinel_user_flow_with_duplicate_sentinels_updates_first(
    hass: HomeAssistant,
) -> None:
    """User flow with multiple Sentinel subentries updates the first, adds no more."""
    sub1 = DummySubentry("sent1", SUBENTRY_TYPE_SENTINEL, "Sentinel", {})
    sub2 = DummySubentry("sent2", SUBENTRY_TYPE_SENTINEL, "Sentinel", {})
    entry = DummyEntry()
    entry.subentries[sub1.subentry_id] = sub1
    entry.subentries[sub2.subentry_id] = sub2

    flow = _make_sentinel_flow_for_mode(hass, entry)
    update_calls: list[Any] = []
    flow.async_update_and_abort = lambda *args, **_kwargs: (  # type: ignore[assignment]
        update_calls.append(args)
        or {"type": "abort", "reason": "reconfigure_successful"}
    )

    await flow.async_step_setup_mode({"setup_mode": "basic"})
    result = await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )

    assert result.get("type") == "abort", (
        "must update, not create, when duplicates exist"
    )
    assert len(update_calls) == 1
