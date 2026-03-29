# ruff: noqa: S101
"""Tests for configuration subentries and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from homeassistant.const import CONF_API_KEY, CONF_HOST, CONF_PASSWORD, CONF_USERNAME
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentConfigFlow,
)
from custom_components.home_generative_agent.const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CRITICAL_ACTION_PIN,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_EXPLAIN_ENABLED,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_NAME,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM_URL,
    CONF_OPENAI_COMPATIBLE_API_KEY,
    CONF_OPENAI_COMPATIBLE_BASE_URL,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VLM_PROVIDER,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    MODEL_CATEGORY_SPECS,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_SENTINEL,
    SUBENTRY_TYPE_STT_PROVIDER,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    legacy_model_provider_configs,
    resolve_runtime_options,
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

@pytest.mark.asyncio
async def test_migration_v5_to_v6_converts_string_to_list(
    hass: HomeAssistant,
) -> None:
    """Migration should convert a single string LLM API to a list."""
    from homeassistant.const import CONF_LLM_HASS_API
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=5,
        data={},
        options={
            CONF_LLM_HASS_API: "assist",
        },
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.version == 6
    assert entry.options.get(CONF_LLM_HASS_API) == ["assist"]


@pytest.mark.asyncio
async def test_migration_v5_to_v6_none_sentinel_removed(
    hass: HomeAssistant,
) -> None:
    """Migration should remove the LLM_HASS_API key if its value is 'none'."""
    from homeassistant.const import CONF_LLM_HASS_API
    from custom_components.home_generative_agent.const import LLM_HASS_API_NONE
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=5,
        data={},
        options={
            CONF_LLM_HASS_API: LLM_HASS_API_NONE,
        },
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.version == 6
    assert CONF_LLM_HASS_API not in entry.options


@pytest.mark.asyncio
async def test_migration_v5_to_v6_missing_key_unchanged(
    hass: HomeAssistant,
) -> None:
    """Migration should leave an absent LLM_HASS_API key absent."""
    from homeassistant.const import CONF_LLM_HASS_API
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=5,
        data={},
        options={},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.version == 6
    assert CONF_LLM_HASS_API not in entry.options


@pytest.mark.asyncio
async def test_migration_v5_to_v6_already_list_unchanged(
    hass: HomeAssistant,
) -> None:
    """Migration should not modify a value that is already a list."""
    from homeassistant.const import CONF_LLM_HASS_API
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        version=5,
        data={},
        options={
            CONF_LLM_HASS_API: ["search_services", "weather_forecast"],
        },
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.version == 6
    assert entry.options.get(CONF_LLM_HASS_API) == ["search_services", "weather_forecast"]
