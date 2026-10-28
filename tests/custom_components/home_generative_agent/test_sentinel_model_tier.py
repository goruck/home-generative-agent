# ruff: noqa: S101
"""
The Sentinel Model feature: a model tier of its own for Sentinel (#866).

Without the feature, Sentinel's triage, discovery and explanations run on the
chat model, as they always have; with it, on their own provider, model and
fallback chain. Any provider that serves chat serves the category.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_generative_agent.const import (
    CONF_EXPLAIN_ENABLED,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_NAME,
    CONF_OLLAMA_SENTINEL_CONTEXT_SIZE,
    CONF_OLLAMA_SENTINEL_KEEPALIVE,
    CONF_OLLAMA_SENTINEL_MODEL,
    CONF_OLLAMA_SENTINEL_URL,
    CONF_OPENAI_COMPATIBLE_BASE_URL,
    CONF_OPENAI_SENTINEL_MODEL,
    CONF_SENTINEL_MODEL_PROVIDER,
    CONF_SENTINEL_MODEL_TEMPERATURE,
    CONF_SENTINEL_TRIAGE_ENABLED,
    DEFAULT_FEATURE_TYPES,
    DOMAIN,
    FEATURE_CATEGORY_MAP,
    FEATURE_DEFS,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_OLLAMA_SENTINEL_MODEL,
    RECOMMENDED_SENTINEL_MODEL_TEMPERATURE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from custom_components.home_generative_agent.core.fallback import FallbackChatModel
from custom_components.home_generative_agent.core.subentry_resolver import (
    OPT_IN_CATEGORIES,
    build_model_deployments,
    resolve_fallback_chains,
    resolve_model_provider_configs,
    resolve_runtime_options,
    with_implied_capabilities,
)
from custom_components.home_generative_agent.flows.feature_subentry_flow import (
    _fallback_provider_options,
    _provider_options,
    _shared_slot_clash,
)

from .test_fallback_setup import (
    FakeConfiguredModel,
    NoopAsyncEngine,
    _fallback_setup_data,
    _patch_setup_dependencies,
    hga_component,
)
from .test_subentries import DummyEntry, DummySubentry, _make_feature_flow_for_basic

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def _ollama_provider(stored_caps: list[str] | None) -> DummySubentry:
    """Return a provider subentry stored before the category existed."""
    data: dict[str, Any] = {
        "provider_type": "ollama",
        "settings": {"base_url": "http://ollama-2:11434"},
    }
    if stored_caps is not None:
        data["capabilities"] = stored_caps
    return DummySubentry("prov1", SUBENTRY_TYPE_MODEL_PROVIDER, "Ollama", data)


def _entry_with(*subentries: DummySubentry) -> DummyEntry:
    entry = DummyEntry()
    for subentry in subentries:
        entry.subentries[subentry.subentry_id] = subentry
    return entry


# ---------------------------------------------------------------------------
# Registry and capabilities
# ---------------------------------------------------------------------------


def test_the_sentinel_tier_is_registered_but_not_a_default_feature() -> None:
    """A fifth category, mapped one-to-one from its own opt-in feature type."""
    assert FEATURE_CATEGORY_MAP["sentinel_model"] == "sentinel"
    assert "sentinel" in MODEL_CATEGORY_SPECS
    assert FEATURE_DEFS["sentinel_model"]["required"] is False
    assert FEATURE_DEFS["sentinel_model"]["basic"] is False
    assert "sentinel_model" not in DEFAULT_FEATURE_TYPES
    assert "sentinel" in OPT_IN_CATEGORIES
    # Every provider type that serves chat is listed for the tier.
    expected = {"openai", "openai_compatible", "ollama", "gemini", "anthropic"}
    spec = MODEL_CATEGORY_SPECS["sentinel"]
    assert set(spec["providers"]) == expected
    assert set(spec["recommended_models"]) == expected
    assert set(spec["model_keys"]) == expected
    assert set(MODEL_CATEGORY_SPECS["chat"]["providers"]) == expected


def test_chat_capability_implies_the_sentinel_capability() -> None:
    """Providers stored before the tier existed qualify without a migration."""
    assert "sentinel" in with_implied_capabilities({"chat"})
    assert "sentinel" not in with_implied_capabilities({"vlm", "embedding"})
    assert with_implied_capabilities(set()) == set()


def test_a_provider_stored_without_the_capability_serves_the_tier() -> None:
    """The resolver and the picker both see the implied capability."""
    entry = _entry_with(_ollama_provider(["chat", "vlm"]))
    providers = resolve_model_provider_configs(entry, {})  # type: ignore[arg-type]
    assert "sentinel" in providers["prov1"].capabilities
    assert [o["value"] for o in _provider_options(entry, "sentinel")] == [  # type: ignore[arg-type]
        "prov1"
    ]
    # A second chat-only server is offered as a Sentinel fallback too.
    second = DummySubentry(
        "prov2",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Ollama 3",
        {
            "provider_type": "ollama",
            "capabilities": ["chat"],
            "settings": {"base_url": "http://ollama-3:11434"},
        },
    )
    entry.subentries["prov2"] = second
    fallbacks = _fallback_provider_options(entry, "sentinel", "prov1")  # type: ignore[arg-type]
    assert [o["value"] for o in fallbacks] == ["prov2"]


# ---------------------------------------------------------------------------
# Resolution: off by default, on when the feature is set up
# ---------------------------------------------------------------------------


def test_without_the_feature_nothing_is_resolved_for_sentinel() -> None:
    """No automatic pick: Sentinel then runs on the chat model."""
    conversation = DummySubentry(
        "conv",
        SUBENTRY_TYPE_FEATURE,
        "Conversation",
        {"feature_type": "conversation", "model_provider_id": "prov1"},
    )
    entry = _entry_with(_ollama_provider(["chat", "vlm"]), conversation)
    providers = resolve_model_provider_configs(entry, {})  # type: ignore[arg-type]

    assert "sentinel" not in build_model_deployments(entry, providers, {})  # type: ignore[arg-type]
    assert "sentinel" not in resolve_fallback_chains(entry, providers, {})  # type: ignore[arg-type]
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert CONF_SENTINEL_MODEL_PROVIDER not in options
    assert CONF_OLLAMA_SENTINEL_MODEL not in options


def test_with_the_feature_the_tier_resolves_like_any_other() -> None:
    """Provider, model, temperature, URL and chain all come from the feature."""
    feature = DummySubentry(
        "sent",
        SUBENTRY_TYPE_FEATURE,
        "Sentinel Model",
        {
            "feature_type": "sentinel_model",
            "model_provider_id": "prov1",
            CONF_FEATURE_MODEL: {CONF_FEATURE_MODEL_NAME: "qwen3:1.7b"},
        },
    )
    entry = _entry_with(_ollama_provider(["chat"]), feature)
    providers = resolve_model_provider_configs(entry, {})  # type: ignore[arg-type]

    assert build_model_deployments(entry, providers, {})["sentinel"] == "edge"  # type: ignore[arg-type]
    chains = resolve_fallback_chains(entry, providers, {})  # type: ignore[arg-type]
    assert [p.entry_id for p in chains["sentinel"]] == ["prov1"]
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_MODEL_PROVIDER] == "ollama"
    assert options[CONF_OLLAMA_SENTINEL_MODEL] == "qwen3:1.7b"
    assert options[CONF_SENTINEL_MODEL_TEMPERATURE] == (
        RECOMMENDED_SENTINEL_MODEL_TEMPERATURE
    )
    assert options[CONF_OLLAMA_SENTINEL_URL] == "http://ollama-2:11434"


def test_the_feature_defaults_to_the_recommended_small_model() -> None:
    feature = DummySubentry(
        "sent",
        SUBENTRY_TYPE_FEATURE,
        "Sentinel Model",
        {"feature_type": "sentinel_model", "model_provider_id": "prov1"},
    )
    entry = _entry_with(_ollama_provider(["chat"]), feature)
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert options[CONF_OLLAMA_SENTINEL_MODEL] == RECOMMENDED_OLLAMA_SENTINEL_MODEL


# ---------------------------------------------------------------------------
# The setup flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_setup_leaves_the_sentinel_model_off(hass: HomeAssistant) -> None:
    """Basic setup creates every other feature, never this one."""
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "All-in-one Provider",
        {"provider_type": "openai", "capabilities": ["chat", "vlm", "summarization"]},
    )
    entry = _entry_with(provider)
    flow = _make_feature_flow_for_basic(hass, entry)

    await flow.async_step_user()
    await flow.async_step_setup_mode({"setup_mode": "basic"})

    feature_types = {
        s.data.get("feature_type")
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_FEATURE
    }
    assert "sentinel_model" not in feature_types
    assert "conversation" in feature_types


@pytest.mark.asyncio
async def test_advanced_setup_offers_the_sentinel_model_off_by_default(
    hass: HomeAssistant,
) -> None:
    provider = DummySubentry(
        "prov1",
        SUBENTRY_TYPE_MODEL_PROVIDER,
        "Ollama",
        {"provider_type": "ollama", "capabilities": ["chat", "vlm"]},
    )
    entry = _entry_with(provider)
    flow = _make_feature_flow_for_basic(hass, entry)

    result = cast("dict[str, Any]", await flow.async_step_feature_enable())

    fields = {str(key): key for key in result["data_schema"].schema}
    assert "sentinel_model" in fields
    assert fields["sentinel_model"].default() is False


@pytest.mark.asyncio
async def test_the_feature_step_exists(hass: HomeAssistant) -> None:
    """The flow dispatches to async_step_<feature_type>."""
    flow = _make_feature_flow_for_basic(hass, _entry_with(_ollama_provider(["chat"])))
    assert callable(flow.async_step_sentinel_model)


# ---------------------------------------------------------------------------
# Setup wiring
# ---------------------------------------------------------------------------


class _Recorder:
    """Stand in for a Sentinel LLM consumer and record what setup hands it."""

    built: ClassVar[list[_Recorder]] = []

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        self.model = kwargs.get("model", model)
        self.deployment = kwargs.get("deployment")
        type(self).built.append(self)

    def begin_run(self) -> None:
        """No-op."""


class _RecordingTriage(_Recorder):
    built: ClassVar[list[_Recorder]] = []


class _RecordingExplainer(_Recorder):
    built: ClassVar[list[_Recorder]] = []


class _RecordingDiscovery(NoopAsyncEngine):
    built: ClassVar[list[_RecordingDiscovery]] = []

    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        self.model = kwargs.get("model")
        self.deployment = kwargs.get("deployment")
        _RecordingDiscovery.built.append(self)


@pytest.fixture(autouse=True)
def _record_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingTriage.built = []
    _RecordingExplainer.built = []
    _RecordingDiscovery.built = []
    monkeypatch.setattr(hga_component, "SentinelTriageService", _RecordingTriage)
    monkeypatch.setattr(hga_component, "LLMExplainer", _RecordingExplainer)


async def _set_up(hass: Any, monkeypatch: pytest.MonkeyPatch, data: Any) -> Any:
    """Set the entry up and return its runtime data."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data.options[CONF_SENTINEL_TRIAGE_ENABLED] = True
    data.options[CONF_EXPLAIN_ENABLED] = True
    _patch_setup_dependencies(hass, monkeypatch, data)
    monkeypatch.setattr(hga_component, "SentinelDiscoveryEngine", _RecordingDiscovery)
    assert await cast("Any", hga_component).async_setup_entry(hass, entry) is True
    return entry.runtime_data


def _sentinel_consumers() -> list[Any]:
    """Return what triage, discovery and explanations were each handed."""
    (triage,) = _RecordingTriage.built
    (discovery,) = _RecordingDiscovery.built
    (explainer,) = _RecordingExplainer.built
    return [triage, discovery, explainer]


@pytest.mark.asyncio
async def test_without_the_feature_sentinel_runs_on_the_chat_model(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = await _set_up(hass, monkeypatch, _fallback_setup_data())

    assert "sentinel" not in runtime.model_deployments
    for consumer in _sentinel_consumers():
        assert consumer.model is runtime.chat_model
        assert consumer.deployment == runtime.model_deployments["chat"]


@pytest.mark.asyncio
async def test_with_the_feature_sentinel_gets_its_own_chain(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its own primary, its own fallback model, its own deployment."""
    data = _fallback_setup_data()
    data.options[CONF_SENTINEL_MODEL_PROVIDER] = "openai"
    data.options[CONF_OPENAI_SENTINEL_MODEL] = "gpt-sentinel"
    data.providers["ollama1"].data["settings"]["sentinel_model"] = (
        "qwen-sentinel-fallback"
    )
    data.fallback_chains["sentinel"] = [
        data.providers["openai1"],
        data.providers["ollama1"],
    ]
    data.deployments["sentinel"] = "cloud"

    runtime = await _set_up(hass, monkeypatch, data)

    triage, discovery, explainer = _sentinel_consumers()
    sentinel_model = triage.model
    assert isinstance(sentinel_model, FallbackChatModel)
    assert sentinel_model is not runtime.chat_model
    primary = cast("FakeConfiguredModel", sentinel_model.chain[0][0])
    assert primary.config["configurable"]["model_name"] == "gpt-sentinel"
    fallback = cast("FakeConfiguredModel", sentinel_model.chain[1][0])
    assert fallback.config["configurable"]["model"] == "qwen-sentinel-fallback"
    assert fallback.config["configurable"]["num_predict"] is None
    assert runtime.model_deployments["sentinel"] == "cloud"
    for consumer in (triage, discovery, explainer):
        assert consumer.model is sentinel_model
        assert consumer.deployment == "cloud"


@pytest.mark.asyncio
async def test_an_ollama_sentinel_model_uses_its_own_options(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recommended provider: model, context and keep-alive of its own."""
    data = _fallback_setup_data()
    data.options[CONF_SENTINEL_MODEL_PROVIDER] = "ollama"
    data.options[CONF_OLLAMA_SENTINEL_MODEL] = "qwen3:1.7b"
    data.options[CONF_OLLAMA_SENTINEL_CONTEXT_SIZE] = 4096
    data.options[CONF_OLLAMA_SENTINEL_KEEPALIVE] = 60
    data.fallback_chains["sentinel"] = [data.providers["ollama1"]]
    data.deployments["sentinel"] = "edge"

    runtime = await _set_up(hass, monkeypatch, data)

    triage, discovery, explainer = _sentinel_consumers()
    sentinel_model = triage.model
    assert isinstance(sentinel_model, FallbackChatModel)
    (head,) = [member[0] for member in sentinel_model.chain]
    assert isinstance(head, FakeConfiguredModel)
    configurable = head.config["configurable"]
    assert configurable["model"] == "qwen3:1.7b"
    assert configurable["num_ctx"] == 4096
    assert configurable["keep_alive"] == 60
    assert runtime.model_deployments["sentinel"] == "edge"
    assert discovery.model is explainer.model is sentinel_model


@pytest.mark.asyncio
async def test_an_unreachable_sentinel_provider_falls_back_to_chat(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Never a placeholder beside a healthy chat model, and never silently."""
    data = _fallback_setup_data()
    data.openai_available = False
    data.options[CONF_SENTINEL_MODEL_PROVIDER] = "openai"
    data.fallback_chains["sentinel"] = [data.providers["openai1"]]
    data.deployments["sentinel"] = "cloud"

    runtime = await _set_up(hass, monkeypatch, data)

    assert "sentinel" not in runtime.model_deployments
    for consumer in _sentinel_consumers():
        assert consumer.model is runtime.chat_model
        assert consumer.deployment == runtime.model_deployments["chat"]
    assert "Sentinel Model provider OpenAI is unavailable" in caplog.text


# ---------------------------------------------------------------------------
# One runtime slot per provider type
# ---------------------------------------------------------------------------


def _compatible_provider(subentry_id: str, title: str, url: str) -> DummySubentry:
    return DummySubentry(
        subentry_id,
        SUBENTRY_TYPE_MODEL_PROVIDER,
        title,
        {
            "provider_type": "openai_compatible",
            "capabilities": ["chat"],
            "settings": {"base_url": url, "api_key": f"key-{subentry_id}"},
        },
    )


def _two_server_entry(sentinel_provider_id: str) -> DummyEntry:
    """Conversation on a LAN server, the Sentinel Model on another provider."""
    return _entry_with(
        _compatible_provider("lan", "LAN", "http://lan:8000/v1"),
        _compatible_provider("cloud", "Cloud", "https://cloud.example/v1"),
        DummySubentry(
            "conv",
            SUBENTRY_TYPE_FEATURE,
            "Conversation",
            {"feature_type": "conversation", "model_provider_id": "lan"},
        ),
        DummySubentry(
            "sent",
            SUBENTRY_TYPE_FEATURE,
            "Sentinel Model",
            {
                "feature_type": "sentinel_model",
                "model_provider_id": sentinel_provider_id,
            },
        ),
    )


def test_a_second_server_of_one_type_never_moves_conversation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sentinel runs on chat rather than take over Conversation's slot."""
    entry = _two_server_entry("cloud")
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]

    assert options[CONF_OPENAI_COMPATIBLE_BASE_URL] == "http://lan:8000/v1"
    assert CONF_SENTINEL_MODEL_PROVIDER not in options
    providers = resolve_model_provider_configs(entry, {})  # type: ignore[arg-type]
    assert "sentinel" not in build_model_deployments(entry, providers, {})  # type: ignore[arg-type]
    assert "sentinel" not in resolve_fallback_chains(entry, providers, {})  # type: ignore[arg-type]
    assert "runs on the Conversation model instead" in caplog.text


def test_the_conversation_provider_itself_serves_the_tier() -> None:
    options = resolve_runtime_options(_two_server_entry("lan"))  # type: ignore[arg-type]
    assert options[CONF_SENTINEL_MODEL_PROVIDER] == "openai_compatible"
    assert options[CONF_OPENAI_COMPATIBLE_BASE_URL] == "http://lan:8000/v1"


def test_setup_refuses_a_second_server_of_one_type() -> None:
    entry = _two_server_entry("lan")
    assert _shared_slot_clash(entry, "sentinel_model", "cloud") == "LAN"  # type: ignore[arg-type]
    assert _shared_slot_clash(entry, "sentinel_model", "lan") is None  # type: ignore[arg-type]
    # Only the opt-in tier is refused; the other features keep their behaviour.
    assert _shared_slot_clash(entry, "conversation", "cloud") is None  # type: ignore[arg-type]


def test_a_dangling_pin_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A deleted provider leaves Sentinel on chat, and says so."""
    entry = _two_server_entry("deleted")
    options = resolve_runtime_options(entry)  # type: ignore[arg-type]
    assert CONF_SENTINEL_MODEL_PROVIDER not in options
    assert "Feature 'sentinel_model' has no model provider" in caplog.text
