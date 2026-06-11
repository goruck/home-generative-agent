# ruff: noqa: S101
"""Setup-time tests for fallback model wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_EXPLAIN_ENABLED,
    CONF_FACE_RECOGNITION,
    CONF_FEATURE_MODEL_TEMPERATURE,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_URL,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_TRIAGE_ENABLED,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROVIDER,
    DOMAIN,
    VIDEO_ANALYZER_MODE_DISABLE,
)
from custom_components.home_generative_agent.core.fallback import (
    FallbackChatModel,
    FallbackEmbeddings,
    FallbackVLM,
)
from custom_components.home_generative_agent.core.subentry_types import (
    ModelProviderConfig,
)


class FakeConfiguredModel:
    """Model returned from with_config."""

    def __init__(self, base: FakeRunnable, config: dict[str, Any]) -> None:
        """Initialize fake configured model."""
        self.base = base
        self.config = config

    def bind_tools(self, *_args: Any, **_kwargs: Any) -> FakeConfiguredModel:
        """Return self for bind_tools compatibility."""
        return self


class FakeRunnable:
    """Minimal LangChain runnable stand-in."""

    def __init__(self, name: str) -> None:
        """Initialize fake runnable."""
        self.name = name
        self.configured: list[FakeConfiguredModel] = []

    def configurable_fields(self, **_kwargs: Any) -> FakeRunnable:
        """Return self for configurable_fields compatibility."""
        return self

    def with_config(
        self, config: dict[str, Any] | None = None, **_kwargs: Any
    ) -> FakeConfiguredModel:
        """Record config and return a configured wrapper."""
        configured = FakeConfiguredModel(self, config or {})
        self.configured.append(configured)
        return configured


class FakeEmbeddings:
    """Minimal embeddings stand-in."""

    def __init__(self, name: str) -> None:
        """Initialize fake embeddings."""
        self.name = name

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic fake embeddings."""
        return [[float(len(text))] for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Return deterministic fake query embedding."""
        return [float(len(text))]


class NoopAsyncStore:
    """Async store stand-in with optional async_load."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize no-op store."""

    async def async_load(self) -> None:
        """Load no state."""

    def start(self) -> None:
        """Start no background work."""

    def stop(self) -> None:
        """Stop no background work."""

    async def async_initialize(self) -> None:
        """Initialize no state."""


class NoopAsyncEngine:
    """Engine stand-in with async stop."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize no-op engine."""

    def start(self) -> None:
        """Start no background work."""

    async def stop(self) -> None:
        """Stop no background work."""


class NoopNotifier(NoopAsyncStore):
    """Notifier stand-in."""


class NoopRuleRegistry(NoopAsyncStore):
    """Rule registry stand-in."""

    def list_rules(self) -> list[dict[str, Any]]:
        """Return no rules."""
        return []


class NoopHttpClient:
    """httpx.Client stand-in."""

    def close(self) -> None:
        """Close no resources."""


def _register_capture_service(
    hass: Any,
    domain: str,
    service: str,
    calls: list[dict[str, Any]],
) -> None:
    """Register a fake service and capture all calls."""

    async def _handler(call: Any) -> None:
        calls.append(
            {
                "domain": domain,
                "service": service,
                "data": dict(call.data),
            }
        )

    hass.services.async_register(domain, service, _handler)


@dataclass
class FallbackSetupData:
    """Test data for setup fallback wiring."""

    providers: dict[str, ModelProviderConfig]
    options: dict[str, Any]
    fallback_chains: dict[str, list[ModelProviderConfig]]
    deployments: dict[str, str]
    captured_embedding_chain: dict[str, list[tuple[Any, str, str]]]
    openai_available: bool = True
    openai_embeddings_available: bool = True


def _fallback_setup_data() -> FallbackSetupData:
    """Return providers/options/chains for fallback setup tests."""
    openai_provider = ModelProviderConfig(
        entry_id="openai1",
        name="OpenAI",
        provider_type="openai",
        capabilities={"chat", "vlm", "summarization", "embedding"},
        data={"settings": {"api_key": "sk-test"}},
        deployment="cloud",
    )
    ollama_provider = ModelProviderConfig(
        entry_id="ollama1",
        name="Ollama",
        provider_type="ollama",
        capabilities={"chat", "vlm", "summarization", "embedding"},
        data={
            "settings": {
                "base_url": "http://ollama",
                "chat_model": "qwen-chat-fallback",
                "vlm_model": "llava-fallback",
                "summarization_model": "qwen-summary-fallback",
                "embedding_model": "nomic-fallback",
            }
        },
        deployment="edge",
    )
    providers = {
        openai_provider.entry_id: openai_provider,
        ollama_provider.entry_id: ollama_provider,
    }
    options: dict[str, Any] = {
        CONF_API_KEY: "sk-test",
        CONF_CHAT_MODEL_PROVIDER: "openai",
        CONF_VLM_PROVIDER: "openai",
        CONF_SUMMARIZATION_MODEL_PROVIDER: "openai",
        CONF_EMBEDDING_MODEL_PROVIDER: "openai",
        CONF_OPENAI_CHAT_MODEL: "gpt-primary-chat",
        CONF_OPENAI_VLM: "gpt-primary-vlm",
        CONF_OPENAI_SUMMARIZATION_MODEL: "gpt-primary-summary",
        CONF_OLLAMA_URL: "http://ollama",
        CONF_OLLAMA_CHAT_MODEL: "qwen-primary-unused",
        CONF_FEATURE_MODEL_TEMPERATURE: 0.2,
        CONF_EXPLAIN_ENABLED: False,
        CONF_SENTINEL_ENABLED: False,
        CONF_SENTINEL_DISCOVERY_ENABLED: False,
        CONF_SENTINEL_TRIAGE_ENABLED: False,
        CONF_FACE_RECOGNITION: False,
        CONF_VIDEO_ANALYZER_MODE: VIDEO_ANALYZER_MODE_DISABLE,
        CONF_NOTIFY_SERVICE: "",
    }
    fallback_chains = {
        "chat": [openai_provider, ollama_provider],
        "vlm": [openai_provider, ollama_provider],
        "summarization": [openai_provider, ollama_provider],
        "embedding": [openai_provider, ollama_provider],
    }
    deployments = {
        "chat": "cloud",
        "vlm": "cloud",
        "summarization": "cloud",
        "embedding": "cloud",
    }
    return FallbackSetupData(
        providers=providers,
        options=options,
        fallback_chains=fallback_chains,
        deployments=deployments,
        captured_embedding_chain={},
    )


def _patch_setup_dependencies(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    data: FallbackSetupData,
) -> None:
    """Patch setup dependencies so the test only verifies model assembly."""

    async def _healthy(*_args: Any, **_kwargs: Any) -> bool:
        return True

    async def _openai_healthy(*_args: Any, **_kwargs: Any) -> bool:
        return data.openai_available

    async def _unhealthy(*_args: Any, **_kwargs: Any) -> bool:
        return False

    real_fallback_embeddings = FallbackEmbeddings

    def _capture_fallback_embeddings(
        chain: list[tuple[Any, str, str]], *args: Any, **kwargs: Any
    ) -> FallbackEmbeddings:
        data.captured_embedding_chain["chain"] = chain
        return real_fallback_embeddings(chain, *args, **kwargs)

    monkeypatch.setattr(hga_component, "_register_services", lambda *_args: None)
    monkeypatch.setattr(
        hga_component, "_ensure_default_feature_subentries", lambda *_args: None
    )
    monkeypatch.setattr(
        hga_component, "_assign_first_provider_if_needed", lambda *_args: None
    )
    monkeypatch.setattr(
        hga_component, "resolve_runtime_options", lambda _entry: data.options
    )
    monkeypatch.setattr(
        hga_component, "resolve_model_provider_configs", lambda *_args: data.providers
    )
    monkeypatch.setattr(
        hga_component, "resolve_fallback_chains", lambda *_args: data.fallback_chains
    )
    monkeypatch.setattr(
        hga_component, "build_model_deployments", lambda *_args: data.deployments
    )
    monkeypatch.setattr(hga_component, "build_database_uri_from_entry", lambda _e: None)
    monkeypatch.setattr(
        hga_component,
        "configured_ollama_urls",
        lambda *_args, **_kwargs: ["http://ollama"],
    )
    monkeypatch.setattr(hga_component, "openai_healthy", _openai_healthy)
    monkeypatch.setattr(hga_component, "ollama_healthy", _healthy)
    monkeypatch.setattr(hga_component, "gemini_healthy", _unhealthy)
    monkeypatch.setattr(hga_component, "openai_compatible_healthy", _unhealthy)
    monkeypatch.setattr(hga_component, "anthropic_healthy", _unhealthy)
    monkeypatch.setattr(
        hga_component, "ChatOpenAI", lambda *_args, **_kwargs: FakeRunnable("openai")
    )
    monkeypatch.setattr(
        hga_component, "ChatOllama", lambda *_args, **_kwargs: FakeRunnable("ollama")
    )
    monkeypatch.setattr(
        hga_component,
        "OpenAIEmbeddings",
        lambda *_args, **_kwargs: (
            FakeEmbeddings("openai") if data.openai_embeddings_available else None
        ),
    )
    monkeypatch.setattr(
        hga_component,
        "OllamaEmbeddings",
        lambda *_args, **_kwargs: FakeEmbeddings("ollama"),
    )
    monkeypatch.setattr(
        cast("Any", hga_component).httpx,
        "Client",
        lambda *_args, **_kwargs: NoopHttpClient(),
    )
    monkeypatch.setattr(hga_component, "VideoAnalyzer", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "SuppressionManager", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "AuditStore", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "DiscoveryStore", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "ProposalStore", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "RuleRegistry", NoopRuleRegistry)
    monkeypatch.setattr(hga_component, "ActionHandler", NoopAsyncStore)
    monkeypatch.setattr(hga_component, "SentinelNotifier", NoopNotifier)
    monkeypatch.setattr(hga_component, "SentinelEngine", NoopAsyncEngine)
    monkeypatch.setattr(hga_component, "SentinelDiscoveryEngine", NoopAsyncEngine)
    monkeypatch.setattr(
        hga_component, "FallbackEmbeddings", _capture_fallback_embeddings
    )
    monkeypatch.setattr(hga_component, "_log_ollama_server_info", AsyncMock())
    monkeypatch.setattr(
        hass.config_entries,
        "async_forward_entry_setups",
        AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_setup_configures_fallback_models_and_embedding_chain(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback setup should configure category models and use embedding clients."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    chat_model = entry.runtime_data.chat_model
    assert isinstance(chat_model, FallbackChatModel)
    chat_fallback = cast("FakeConfiguredModel", chat_model.chain[1][0])
    assert chat_fallback.base.name == "ollama"
    assert chat_fallback.config["configurable"]["model"] == "qwen-chat-fallback"
    assert chat_fallback.config["configurable"]["num_predict"] is not None

    vision_model = entry.runtime_data.vision_model
    assert isinstance(vision_model, FallbackVLM)
    vlm_fallback = cast("FakeConfiguredModel", vision_model.chain[1][0])
    assert vlm_fallback.config["configurable"]["model"] == "llava-fallback"
    assert vlm_fallback.config["configurable"]["mirostat"] is not None

    summarization_model = entry.runtime_data.summarization_model
    assert isinstance(summarization_model, FallbackChatModel)
    summary_fallback = cast("FakeConfiguredModel", summarization_model.chain[1][0])
    assert summary_fallback.config["configurable"]["model"] == "qwen-summary-fallback"
    assert summary_fallback.config["configurable"]["num_predict"] is not None

    embedding_chain = data.captured_embedding_chain["chain"]
    assert [model.name for model, _deployment, _provider_id in embedding_chain] == [
        "openai",
        "ollama",
    ]


@pytest.mark.asyncio
async def test_setup_embedding_chain_skips_unavailable_primary(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable primary embeddings must not enter the fallback chain as None."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.openai_embeddings_available = False
    ollama_provider = data.providers["ollama1"]
    secondary_ollama = ModelProviderConfig(
        entry_id="ollama2",
        name="Ollama Secondary",
        provider_type="ollama",
        capabilities={"embedding"},
        data=ollama_provider.data,
        deployment="edge",
    )
    data.providers[secondary_ollama.entry_id] = secondary_ollama
    data.fallback_chains["embedding"] = [
        data.providers["openai1"],
        ollama_provider,
        secondary_ollama,
    ]
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    embedding_chain = data.captured_embedding_chain["chain"]
    assert len(embedding_chain) == 2
    assert all(
        model is not None for model, _deployment, _provider_id in embedding_chain
    )
    assert [provider_id for _model, _deployment, provider_id in embedding_chain] == [
        "ollama1",
        "ollama2",
    ]


@pytest.mark.asyncio
async def test_setup_model_fallback_skips_unavailable_primary(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unavailable primary chat-like models must not short-circuit fallback."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.openai_available = False
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    chat_model = cast("FakeConfiguredModel", entry.runtime_data.chat_model)
    assert chat_model.base.name == "ollama"
    assert chat_model.config["configurable"]["model"] == "qwen-chat-fallback"

    vision_model = cast("FakeConfiguredModel", entry.runtime_data.vision_model)
    assert vision_model.base.name == "ollama"
    assert vision_model.config["configurable"]["model"] == "llava-fallback"

    summarization_model = cast(
        "FakeConfiguredModel", entry.runtime_data.summarization_model
    )
    assert summarization_model.base.name == "ollama"
    assert (
        summarization_model.config["configurable"]["model"] == "qwen-summary-fallback"
    )
    assert entry.runtime_data.model_deployments["chat"] == "edge"
    assert entry.runtime_data.model_deployments["vlm"] == "edge"
    assert entry.runtime_data.model_deployments["summarization"] == "edge"
    assert "Fallback selected at setup for chat" in caplog.text
    assert "using provider ollama1 (deployment=edge)" in caplog.text


@pytest.mark.asyncio
async def test_setup_model_fallback_sends_persistent_notification(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup-selected fallbacks notify even when no mobile notify service exists."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    calls: list[dict[str, Any]] = []
    _register_capture_service(hass, "persistent_notification", "create", calls)
    data = _fallback_setup_data()
    data.openai_available = False
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)

    assert result is True
    assert calls
    chat_call = next(call for call in calls if "for chat" in call["data"]["message"])
    payload = chat_call["data"]
    assert chat_call["domain"] == "persistent_notification"
    assert chat_call["service"] == "create"
    assert payload["title"] == "HGA model fallback active"
    assert payload["message"] == (
        "Using Ollama for chat; OpenAI was unavailable at startup."
    )


@pytest.mark.asyncio
async def test_runtime_cloud_model_fallback_sends_mobile_notification_once(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime cloud fallbacks notify once and include the cost warning."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    calls: list[dict[str, Any]] = []
    _register_capture_service(hass, "notify", "mobile_app_phone", calls)
    data = _fallback_setup_data()
    data.options[CONF_CHAT_MODEL_PROVIDER] = "ollama"
    data.options[CONF_NOTIFY_SERVICE] = "notify.mobile_app_phone"
    data.fallback_chains["chat"] = [
        data.providers["ollama1"],
        data.providers["openai1"],
    ]
    data.deployments["chat"] = "edge"
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    calls.clear()

    chat_model = entry.runtime_data.chat_model
    assert isinstance(chat_model, FallbackChatModel)
    primary = cast("Any", chat_model.chain[0][0])
    fallback = cast("Any", chat_model.chain[1][0])
    primary.ainvoke = AsyncMock(side_effect=HomeAssistantError("edge down"))
    fallback.ainvoke = AsyncMock(return_value=type("Msg", (), {"content": "ok"})())

    assert (await chat_model.ainvoke(["hello"])).content == "ok"
    assert (await chat_model.ainvoke(["hello"])).content == "ok"
    await hass.async_block_till_done()

    assert len(calls) == 1
    payload = calls[0]["data"]
    assert calls[0]["domain"] == "notify"
    assert calls[0]["service"] == "mobile_app_phone"
    assert payload["title"] == "HGA model fallback active"
    assert "Using OpenAI for chat; Ollama failed." in payload["message"]
    assert "Cloud model usage may incur provider costs." in payload["message"]
    assert payload["data"]["tag"].startswith("hga_model_fallback_chat_")


@pytest.mark.asyncio
async def test_model_fallback_notification_strips_role_prefixes(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider role prefixes should not be repeated in fallback notifications."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    calls: list[dict[str, Any]] = []
    _register_capture_service(hass, "notify", "mobile_app_phone", calls)
    data = _fallback_setup_data()
    data.providers["ollama1"] = ModelProviderConfig(
        entry_id="ollama1",
        name="Primary Ollama",
        provider_type="ollama",
        capabilities={"chat", "embedding"},
        data=data.providers["ollama1"].data,
        deployment="edge",
    )
    data.providers["openai1"] = ModelProviderConfig(
        entry_id="openai1",
        name="Primary OpenAI",
        provider_type="openai",
        capabilities={"chat", "embedding"},
        data=data.providers["openai1"].data,
        deployment="cloud",
    )
    data.options[CONF_CHAT_MODEL_PROVIDER] = "ollama"
    data.options[CONF_NOTIFY_SERVICE] = "notify.mobile_app_phone"
    data.fallback_chains["chat"] = [
        data.providers["ollama1"],
        data.providers["openai1"],
    ]
    data.deployments["chat"] = "edge"
    _patch_setup_dependencies(hass, monkeypatch, data)

    result = await cast("Any", hga_component).async_setup_entry(hass, entry)
    assert result is True
    calls.clear()

    chat_model = entry.runtime_data.chat_model
    assert isinstance(chat_model, FallbackChatModel)
    primary = cast("Any", chat_model.chain[0][0])
    fallback = cast("Any", chat_model.chain[1][0])
    primary.ainvoke = AsyncMock(side_effect=HomeAssistantError("edge down"))
    fallback.ainvoke = AsyncMock(return_value=type("Msg", (), {"content": "ok"})())

    assert (await chat_model.ainvoke(["hello"])).content == "ok"
    await hass.async_block_till_done()

    message = calls[0]["data"]["message"]
    assert "Primary OpenAI" not in message
    assert "Primary provider Primary Ollama" not in message
    assert message.startswith("Using OpenAI for chat; Ollama failed.")
