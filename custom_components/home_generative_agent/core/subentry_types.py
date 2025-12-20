"""Shared types for configuration subentries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderType = Literal[
    "ollama",
    "openai",
    "gemini",
    "anthropic",
    "triton",
    "docker_runner",
]


@dataclass
class ModelProviderConfig:
    """Subentry metadata for a model provider."""

    entry_id: str
    name: str
    provider_type: ProviderType
    capabilities: set[str]
    data: dict


@dataclass
class FeatureConfig:
    """Subentry metadata for a feature/tool."""

    entry_id: str
    name: str
    feature_type: str
    model_provider_id: str | None
    data: dict
