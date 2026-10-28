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
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_NAME,
    CONF_OLLAMA_SENTINEL_MODEL,
    CONF_OLLAMA_SENTINEL_URL,
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
    _provider_options,
)

from .test_fallback_setup import (
    FakeConfiguredModel,
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
    assert set(MODEL_CATEGORY_SPECS["sentinel"]["providers"]) == set(
        MODEL_CATEGORY_SPECS["chat"]["providers"]
    )


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


class _RecordingTriage:
    """Stand in for SentinelTriageService and record what setup hands it."""

    built: ClassVar[list[_RecordingTriage]] = []

    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.deployment = kwargs.get("deployment")
        _RecordingTriage.built.append(self)

    def begin_run(self) -> None:
        """No-op."""


@pytest.fixture(autouse=True)
def _record_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingTriage.built = []
    monkeypatch.setattr(hga_component, "SentinelTriageService", _RecordingTriage)


@pytest.mark.asyncio
async def test_without_the_feature_sentinel_runs_on_the_chat_model(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_SENTINEL_TRIAGE_ENABLED] = True
    _patch_setup_dependencies(hass, monkeypatch, data)

    assert await cast("Any", hga_component).async_setup_entry(hass, entry) is True

    runtime = entry.runtime_data
    assert runtime.sentinel_model is runtime.chat_model
    assert "sentinel" not in runtime.model_deployments
    (triage,) = _RecordingTriage.built
    assert triage.model is runtime.chat_model
    assert triage.deployment == runtime.model_deployments["chat"]


@pytest.mark.asyncio
async def test_with_the_feature_sentinel_gets_its_own_chain(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its own primary, its own fallback model, its own deployment."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_SENTINEL_TRIAGE_ENABLED] = True
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
    _patch_setup_dependencies(hass, monkeypatch, data)

    assert await cast("Any", hga_component).async_setup_entry(hass, entry) is True

    runtime = entry.runtime_data
    sentinel_model = runtime.sentinel_model
    assert isinstance(sentinel_model, FallbackChatModel)
    assert sentinel_model is not runtime.chat_model
    primary = cast("FakeConfiguredModel", sentinel_model.chain[0][0])
    assert primary.config["configurable"]["model_name"] == "gpt-sentinel"
    fallback = cast("FakeConfiguredModel", sentinel_model.chain[1][0])
    assert fallback.config["configurable"]["model"] == "qwen-sentinel-fallback"
    assert fallback.config["configurable"]["num_predict"] is None
    assert runtime.model_deployments["sentinel"] == "cloud"
    (triage,) = _RecordingTriage.built
    assert triage.model is sentinel_model
    assert triage.deployment == "cloud"
