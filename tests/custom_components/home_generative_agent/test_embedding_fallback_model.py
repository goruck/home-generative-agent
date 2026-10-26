# ruff: noqa: S101
"""
An embedding fallback never mixes models in the memory store (#751).

Each provider type embeds with the one model configured for it, so only a
fallback of the primary's type embeds with the primary's model, and of those
only another Ollama server is kept: its build is checked (#853).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from custom_components.home_generative_agent.const import (
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    chosen_fallback_ids,
    split_embedding_chain,
)
from custom_components.home_generative_agent.core.subentry_types import (
    FeatureConfig,
    ModelProviderConfig,
)
from custom_components.home_generative_agent.flows.feature_subentry_flow import (
    _fallback_provider_options,
)


def _provider(entry_id: str, provider_type: str) -> ModelProviderConfig:
    return ModelProviderConfig(
        entry_id=entry_id,
        name=entry_id,
        provider_type=cast("Any", provider_type),
        capabilities={"embedding"},
        data={},
    )


def test_only_ollama_fallbacks_are_kept_behind_an_ollama_primary() -> None:
    """Order is kept; every other provider is left out."""
    chain = [
        _provider("ollama-a", "ollama"),
        _provider("openai", "openai"),
        _provider("ollama-b", "ollama"),
        _provider("gemini", "gemini"),
    ]

    kept, left_out = split_embedding_chain(chain)

    assert [p.entry_id for p in kept] == ["ollama-a", "ollama-b"]
    assert [p.entry_id for p in left_out] == ["openai", "gemini"]


@pytest.mark.parametrize("provider_type", ["openai_compatible", "openai", "gemini"])
def test_no_fallback_is_kept_behind_another_primary(provider_type: str) -> None:
    """
    Behind a non-Ollama primary every fallback is left out (#853).

    OpenAI and Gemini have one client, so a second provider of the type is the
    primary again; a second OpenAI-compatible server can serve other weights
    under the model's name, and no build check can tell.
    """
    chain = [
        _provider("primary", provider_type),
        _provider("same-type", provider_type),
        _provider("ollama", "ollama"),
    ]

    kept, left_out = split_embedding_chain(chain)

    assert [p.entry_id for p in kept] == ["primary"]
    assert [p.entry_id for p in left_out] == ["same-type", "ollama"]


def test_an_empty_or_single_chain_is_unchanged() -> None:
    """Nothing to leave out."""
    assert split_embedding_chain([]) == ([], [])
    only = _provider("openai", "openai")
    assert split_embedding_chain([only]) == ([only], [])


def _entry(*providers: tuple[str, str, str | None]) -> Any:
    return SimpleNamespace(
        subentries={
            subentry_id: SimpleNamespace(
                subentry_type=SUBENTRY_TYPE_MODEL_PROVIDER,
                subentry_id=subentry_id,
                title=subentry_id,
                data={
                    "provider_type": provider_type,
                    "capabilities": ["embedding", "chat"],
                    **({"settings": {"base_url": url}} if url else {}),
                },
            )
            for subentry_id, provider_type, url in providers
        }
    )


_PROVIDERS = (
    ("ollama-a", "ollama", "http://ollama-a:11434"),
    ("ollama-b", "ollama", "http://ollama-b:11434"),
    ("ollama-a2", "ollama", "http://ollama-a:11434/"),
    ("openai", "openai", None),
    ("openai-2", "openai", None),
)


def test_the_embeddings_form_offers_only_other_ollama_servers() -> None:
    """Another model, or the primary's own server again, is no failover."""
    entry = _entry(*_PROVIDERS)

    options = _fallback_provider_options(entry, "embedding", "ollama-a")

    assert [o["value"] for o in options] == ["ollama-b"]


@pytest.mark.parametrize("primary_id", ["openai", None, "missing"])
def test_the_embeddings_form_offers_no_fallback_behind_other_primaries(
    primary_id: str | None,
) -> None:
    """One client per type: a second OpenAI provider is the primary again."""
    entry = _entry(*_PROVIDERS)

    assert _fallback_provider_options(entry, "embedding", primary_id) == []


def test_other_features_still_offer_every_capable_provider() -> None:
    """Chat may fall back across types: it keeps no vectors."""
    entry = _entry(*_PROVIDERS)

    options = _fallback_provider_options(entry, "chat", "ollama-a")

    assert [o["value"] for o in options] == [
        "ollama-b",
        "ollama-a2",
        "openai",
        "openai-2",
    ]


def test_chosen_fallbacks_are_read_from_the_embedding_feature_only() -> None:
    """A chat feature's fallbacks are not embedding choices."""
    features = {
        "f1": FeatureConfig(
            entry_id="f1",
            name="Embeddings",
            feature_type="embedding",
            model_provider_id="ollama-a",
            model={},
            config={},
            fallback_provider_ids=["openai"],
        ),
        "f2": FeatureConfig(
            entry_id="f2",
            name="Conversation",
            feature_type="conversation",
            model_provider_id="ollama-a",
            model={},
            config={},
            fallback_provider_ids=["openai-2"],
        ),
    }
    with patch(
        "custom_components.home_generative_agent.core.subentry_resolver."
        "resolve_feature_configs",
        return_value=features,
    ):
        assert chosen_fallback_ids(cast("Any", None), {}, {}, "embedding") == {"openai"}
        assert chosen_fallback_ids(cast("Any", None), {}, {}, "chat") == {"openai-2"}
