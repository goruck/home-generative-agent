"""Helpers for resolving configuration subentries and legacy options."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    CONF_API_KEY,
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
)

from ..const import (  # noqa: TID252
    CONF_CHAT_MODEL_PROVIDER,
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_CONTEXT_SIZE,
    CONF_FEATURE_MODEL_KEEPALIVE,
    CONF_FEATURE_MODEL_NAME,
    CONF_FEATURE_MODEL_REASONING,
    CONF_FEATURE_MODEL_TEMPERATURE,
    CONF_GEMINI_API_KEY,
    CONF_GEMINI_CHAT_MODEL,
    CONF_GEMINI_EMBEDDING_MODEL,
    CONF_GEMINI_SUMMARIZATION_MODEL,
    CONF_GEMINI_VLM,
    CONF_OLLAMA_CHAT_CONTEXT_SIZE,
    CONF_OLLAMA_CHAT_KEEPALIVE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_REASONING,
    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM,
    CONF_OLLAMA_VLM_CONTEXT_SIZE,
    CONF_OLLAMA_VLM_KEEPALIVE,
    CONF_OLLAMA_VLM_URL,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_EMBEDDING_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_VLM_PROVIDER,
    FEATURE_CATEGORY_MAP,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_DB_HOST,
    RECOMMENDED_DB_NAME,
    RECOMMENDED_DB_PARAMS,
    RECOMMENDED_DB_PASSWORD,
    RECOMMENDED_DB_PORT,
    RECOMMENDED_DB_USERNAME,
    RECOMMENDED_GEMINI_CHAT_MODEL,
    RECOMMENDED_GEMINI_EMBEDDING_MODEL,
    RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
    RECOMMENDED_GEMINI_VLM,
    RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_URL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    RECOMMENDED_OLLAMA_VLM_URL,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    SUBENTRY_TYPE_DATABASE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from .db_utils import build_postgres_uri
from .subentry_types import FeatureConfig, ModelProviderConfig, ProviderType

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry, ConfigSubentry

LOGGER = logging.getLogger(__name__)


def _options_for(entry: ConfigEntry) -> dict[str, Any]:
    """Merge entry data + options for convenience."""
    return {**entry.data, **entry.options}


def _model_key_for(cat: str, provider: str) -> str | None:
    """Return the model option key for a category/provider pair."""
    spec = MODEL_CATEGORY_SPECS.get(cat)
    if not spec:
        return None
    return spec.get("model_keys", {}).get(provider)


def _coerce_capabilities(raw: Any) -> set[str]:
    """Safely coerce stored capabilities to a set of strings."""
    if not raw:
        return set()
    try:
        return {str(c) for c in raw}
    except (TypeError, ValueError):
        return set()


def get_database_subentry(
    _hass: Any, config_entry: ConfigEntry
) -> ConfigSubentry | None:
    """Return the database subentry if present."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_TYPE_DATABASE:
            return subentry
    return None


def build_database_uri_from_entry(entry: ConfigEntry) -> str | None:
    """
    Return the configured database URI.

    Prefers the database subentry, then falls back to legacy CONF_DB_URI or
    discrete database fields.
    """
    opts = _options_for(entry)
    db_subentry = get_database_subentry(None, entry)
    if db_subentry:
        return build_postgres_uri(dict(db_subentry.data))

    if uri := opts.get(CONF_DB_URI):
        return str(uri)

    required = (CONF_DB_NAME, CONF_USERNAME, CONF_PASSWORD, CONF_HOST, CONF_PORT)
    if any(k not in opts for k in required):
        return None

    return build_postgres_uri(
        {
            CONF_USERNAME: opts.get(CONF_USERNAME, RECOMMENDED_DB_USERNAME),
            CONF_PASSWORD: opts.get(CONF_PASSWORD, RECOMMENDED_DB_PASSWORD),
            CONF_HOST: opts.get(CONF_HOST, RECOMMENDED_DB_HOST),
            CONF_PORT: opts.get(CONF_PORT, RECOMMENDED_DB_PORT),
            CONF_DB_NAME: opts.get(CONF_DB_NAME, RECOMMENDED_DB_NAME),
            CONF_DB_PARAMS: opts.get(CONF_DB_PARAMS, RECOMMENDED_DB_PARAMS),
        }
    )


def _provider_capabilities_from_settings(settings: Mapping[str, Any]) -> set[str]:
    """Derive capability categories from settings keys."""
    caps: set[str] = set()
    for cat in ("chat", "vlm", "summarization", "embedding"):
        if settings.get(f"{cat}_model") or settings.get(cat):
            caps.add(cat)
    return caps or {"chat"}


def _model_provider_from_subentry(subentry: ConfigSubentry) -> ModelProviderConfig:
    """Convert a stored model provider subentry into a dataclass."""
    data = dict(subentry.data)
    settings = data.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}
    provider_type: ProviderType = data.get("provider_type", "ollama")
    name = data.get("name") or subentry.title or provider_type
    caps = _coerce_capabilities(data.get("capabilities"))
    if not caps:
        caps = _provider_capabilities_from_settings(settings)
    return ModelProviderConfig(
        entry_id=subentry.subentry_id,
        name=name,
        provider_type=provider_type,
        capabilities=caps,
        data={"settings": dict(settings), "name": name},
    )


def get_model_provider_subentries(
    _hass: Any, config_entry: ConfigEntry
) -> dict[str, ModelProviderConfig]:
    """Return all configured model provider subentries."""
    providers: dict[str, ModelProviderConfig] = {}
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
            continue
        provider = _model_provider_from_subentry(subentry)
        providers[provider.entry_id] = provider
    return providers


def _legacy_provider_id(entry: ConfigEntry, provider: str) -> str:
    """Generate a stable pseudo-ID for legacy providers."""
    return f"{entry.entry_id}_{provider}_legacy"


def _legacy_ollama_urls(options: Mapping[str, Any]) -> dict[str, str]:
    """Return resolved legacy Ollama URLs by category."""
    base_url = str(options.get(CONF_OLLAMA_URL) or RECOMMENDED_OLLAMA_URL)
    return {
        "base": base_url,
        "chat": str(options.get(CONF_OLLAMA_CHAT_URL) or base_url),
        "vlm": str(
            options.get(CONF_OLLAMA_VLM_URL, RECOMMENDED_OLLAMA_VLM_URL) or base_url
        ),
        "summarization": str(options.get(CONF_OLLAMA_SUMMARIZATION_URL) or base_url),
    }


def _ollama_legacy_provider_settings(
    options: Mapping[str, Any],
    name: str,
    base_url: str,
    categories: set[str],
) -> dict[str, Any]:
    """Build settings for a legacy Ollama provider."""
    settings: dict[str, Any] = {
        "name": name,
        "provider_type": "ollama",
        "base_url": base_url,
    }

    if "chat" in categories:
        settings.update(
            {
                "chat_url": base_url,
                "chat_model": options.get(
                    CONF_OLLAMA_CHAT_MODEL, RECOMMENDED_OLLAMA_CHAT_MODEL
                ),
                "chat_keepalive": options.get(
                    CONF_OLLAMA_CHAT_KEEPALIVE, RECOMMENDED_OLLAMA_CHAT_KEEPALIVE
                ),
                "chat_context": options.get(
                    CONF_OLLAMA_CHAT_CONTEXT_SIZE, RECOMMENDED_OLLAMA_CONTEXT_SIZE
                ),
                "reasoning": options.get(CONF_OLLAMA_REASONING),
            }
        )

    if "vlm" in categories:
        settings.update(
            {
                "vlm_url": base_url,
                "vlm_model": options.get(CONF_OLLAMA_VLM, RECOMMENDED_OLLAMA_VLM),
                "vlm_keepalive": options.get(
                    CONF_OLLAMA_VLM_KEEPALIVE, RECOMMENDED_OLLAMA_VLM_KEEPALIVE
                ),
                "vlm_context": options.get(
                    CONF_OLLAMA_VLM_CONTEXT_SIZE, RECOMMENDED_OLLAMA_CONTEXT_SIZE
                ),
            }
        )

    if "summarization" in categories:
        settings.update(
            {
                "summarization_url": base_url,
                "summarization_model": options.get(
                    CONF_OLLAMA_SUMMARIZATION_MODEL,
                    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
                ),
                "summarization_keepalive": options.get(
                    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
                    RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
                ),
                "summarization_context": options.get(
                    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
                    RECOMMENDED_OLLAMA_CONTEXT_SIZE,
                ),
            }
        )

    if "embedding" in categories:
        settings["embedding_model"] = options.get(
            CONF_OLLAMA_EMBEDDING_MODEL, RECOMMENDED_OLLAMA_EMBEDDING_MODEL
        )

    return settings


def _build_ollama_legacy_providers(
    entry: ConfigEntry, options: Mapping[str, Any]
) -> dict[str, ModelProviderConfig]:
    """Create legacy Ollama provider configs from options."""
    urls = _legacy_ollama_urls(options)
    category_urls = {
        "chat": urls["chat"],
        "vlm": urls["vlm"],
        "summarization": urls["summarization"],
        "embedding": urls["chat"],
    }

    url_groups: dict[str, set[str]] = {}
    for category, url in category_urls.items():
        url_groups.setdefault(url, set()).add(category)

    providers: dict[str, ModelProviderConfig] = {}
    multi_provider = len(url_groups) > 1
    label_map = {
        "chat": "Chat",
        "vlm": "VLM",
        "summarization": "Summarization",
        "embedding": "Embedding",
    }
    for url, categories in url_groups.items():
        label_categories = sorted(
            cat for cat in categories if cat != "embedding"
        ) or sorted(categories)
        label = ", ".join(label_map.get(cat, cat.title()) for cat in label_categories)
        name = "Primary Ollama" if not multi_provider else f"Ollama ({label})"
        settings = _ollama_legacy_provider_settings(options, name, url, categories)
        provider_id = _legacy_provider_id(
            entry, f"ollama_{'_'.join(sorted(categories))}"
        )
        providers[provider_id] = ModelProviderConfig(
            entry_id=provider_id,
            name=settings["name"],
            provider_type="ollama",
            capabilities=categories,
            data={"settings": settings},
        )

    return providers


def _build_openai_legacy_provider(
    entry: ConfigEntry, options: Mapping[str, Any]
) -> ModelProviderConfig | None:
    """Create a legacy OpenAI provider if configured."""
    api_key = options.get(CONF_API_KEY)
    if not api_key:
        return None
    settings = {
        "name": "Cloud LLM - OpenAI",
        "provider_type": "openai",
        "api_key": api_key,
        "chat_model": options.get(
            CONF_OPENAI_CHAT_MODEL, RECOMMENDED_OPENAI_CHAT_MODEL
        ),
        "vlm_model": options.get(CONF_OPENAI_VLM, RECOMMENDED_OPENAI_VLM),
        "summarization_model": options.get(
            CONF_OPENAI_SUMMARIZATION_MODEL, RECOMMENDED_OPENAI_SUMMARIZATION_MODEL
        ),
        "embedding_model": options.get(
            CONF_OPENAI_EMBEDDING_MODEL, RECOMMENDED_OPENAI_EMBEDDING_MODEL
        ),
    }
    return ModelProviderConfig(
        entry_id=_legacy_provider_id(entry, "openai"),
        name=settings["name"],
        provider_type="openai",
        capabilities={"chat", "vlm", "summarization", "embedding"},
        data={"settings": settings},
    )


def _build_gemini_legacy_provider(
    entry: ConfigEntry, options: Mapping[str, Any]
) -> ModelProviderConfig | None:
    """Create a legacy Gemini provider if configured."""
    api_key = options.get(CONF_GEMINI_API_KEY)
    if not api_key:
        return None
    settings = {
        "name": "Cloud LLM - Gemini",
        "provider_type": "gemini",
        "api_key": api_key,
        "chat_model": options.get(
            CONF_GEMINI_CHAT_MODEL, RECOMMENDED_GEMINI_CHAT_MODEL
        ),
        "vlm_model": options.get(CONF_GEMINI_VLM, RECOMMENDED_GEMINI_VLM),
        "summarization_model": options.get(
            CONF_GEMINI_SUMMARIZATION_MODEL, RECOMMENDED_GEMINI_SUMMARIZATION_MODEL
        ),
        "embedding_model": options.get(
            CONF_GEMINI_EMBEDDING_MODEL, RECOMMENDED_GEMINI_EMBEDDING_MODEL
        ),
    }
    return ModelProviderConfig(
        entry_id=_legacy_provider_id(entry, "gemini"),
        name=settings["name"],
        provider_type="gemini",
        capabilities={"chat", "vlm", "summarization", "embedding"},
        data={"settings": settings},
    )


def legacy_model_provider_configs(
    entry: ConfigEntry, options: Mapping[str, Any]
) -> dict[str, ModelProviderConfig]:
    """Build pseudo provider configs from legacy options."""
    providers: dict[str, ModelProviderConfig] = {}
    providers.update(_build_ollama_legacy_providers(entry, options))
    for builder in (_build_openai_legacy_provider, _build_gemini_legacy_provider):
        provider = builder(entry, options)
        if provider:
            providers[provider.entry_id] = provider
    return providers


def resolve_model_provider_configs(
    entry: ConfigEntry, options: Mapping[str, Any]
) -> dict[str, ModelProviderConfig]:
    """Return explicit provider subentries or legacy fallbacks."""
    providers = get_model_provider_subentries(None, entry)
    if providers:
        return providers
    return legacy_model_provider_configs(entry, options)


def _feature_from_subentry(subentry: ConfigSubentry) -> FeatureConfig:
    """Convert a stored feature subentry to a dataclass."""
    data = dict(subentry.data)
    return FeatureConfig(
        entry_id=subentry.subentry_id,
        name=data.get("name") or subentry.title,
        feature_type=data.get("feature_type", ""),
        model_provider_id=data.get("model_provider_id"),
        model=dict(data.get(CONF_FEATURE_MODEL, {})),
        config=dict(data.get("config", {})),
    )


def get_feature_subentries(
    _hass: Any, config_entry: ConfigEntry
) -> dict[str, FeatureConfig]:
    """Return explicit feature subentries."""
    features: dict[str, FeatureConfig] = {}
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_FEATURE:
            continue
        feature = _feature_from_subentry(subentry)
        features[feature.entry_id] = feature
    return features


def legacy_feature_configs(
    entry: ConfigEntry,
    providers: Mapping[str, ModelProviderConfig],
    options: Mapping[str, Any],
) -> dict[str, FeatureConfig]:
    """Infer feature configs from legacy options."""
    providers_by_type: dict[str, list[ModelProviderConfig]] = {}
    providers_by_type_category: dict[str, dict[str, ModelProviderConfig]] = {}
    for provider in providers.values():
        providers_by_type.setdefault(provider.provider_type, []).append(provider)
        for capability in provider.capabilities:
            providers_by_type_category.setdefault(provider.provider_type, {})[
                capability
            ] = provider

    def _legacy_model_data(
        category: str | None, provider_type: str, opts: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not category:
            return {}
        spec = MODEL_CATEGORY_SPECS.get(category, {})
        model_key = _model_key_for(category, provider_type)
        model_name = (
            opts.get(model_key)
            if model_key
            else spec.get("recommended_models", {}).get(provider_type)
        )
        model_data: dict[str, Any] = {}
        if model_name:
            model_data[CONF_FEATURE_MODEL_NAME] = model_name
        temp_key = spec.get("temperature_key")
        if temp_key and opts.get(temp_key) is not None:
            model_data[CONF_FEATURE_MODEL_TEMPERATURE] = opts.get(temp_key)

        if provider_type == "ollama":
            keepalive_map = {
                "chat": CONF_OLLAMA_CHAT_KEEPALIVE,
                "vlm": CONF_OLLAMA_VLM_KEEPALIVE,
                "summarization": CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
            }
            context_map = {
                "chat": CONF_OLLAMA_CHAT_CONTEXT_SIZE,
                "vlm": CONF_OLLAMA_VLM_CONTEXT_SIZE,
                "summarization": CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
            }
            keepalive_key = keepalive_map.get(category)
            if keepalive_key and opts.get(keepalive_key) is not None:
                model_data[CONF_FEATURE_MODEL_KEEPALIVE] = opts.get(keepalive_key)
            context_key = context_map.get(category)
            if context_key and opts.get(context_key) is not None:
                model_data[CONF_FEATURE_MODEL_CONTEXT_SIZE] = opts.get(context_key)
            if category == "chat" and opts.get(CONF_OLLAMA_REASONING) is not None:
                model_data[CONF_FEATURE_MODEL_REASONING] = opts.get(
                    CONF_OLLAMA_REASONING
                )

        return model_data

    def _feature(
        feature_type: str, provider_type_opt: str, default_provider: str
    ) -> FeatureConfig | None:
        provider_type = str(options.get(provider_type_opt, default_provider))
        category = FEATURE_CATEGORY_MAP.get(feature_type)
        provider = providers_by_type_category.get(provider_type, {}).get(category or "")
        if not provider:
            provider_list = providers_by_type.get(provider_type, [])
            provider = provider_list[0] if provider_list else None
        if not provider:
            return None
        model_data = _legacy_model_data(category, provider.provider_type, options)
        return FeatureConfig(
            entry_id=f"{entry.entry_id}_{feature_type}_legacy",
            name=feature_type.replace("_", " ").title(),
            feature_type=feature_type,
            model_provider_id=provider.entry_id,
            model=model_data,
            config={},
        )

    features: dict[str, FeatureConfig] = {}
    mapping = [
        ("conversation", CONF_CHAT_MODEL_PROVIDER, "ollama"),
        ("camera_image_analysis", CONF_VLM_PROVIDER, "ollama"),
        ("conversation_summary", CONF_SUMMARIZATION_MODEL_PROVIDER, "ollama"),
    ]
    for feat_type, opt_key, default in mapping:
        feature = _feature(feat_type, opt_key, default)
        if feature:
            features[feature.entry_id] = feature
    return features


def resolve_feature_configs(
    entry: ConfigEntry,
    providers: Mapping[str, ModelProviderConfig],
    options: Mapping[str, Any],
) -> dict[str, FeatureConfig]:
    """Return explicit feature configs or inferred legacy defaults."""
    features = get_feature_subentries(None, entry)
    if features:
        return features
    return legacy_feature_configs(entry, providers, options)


def _apply_provider_to_category(
    options: dict[str, Any], category: str, provider: ModelProviderConfig
) -> None:
    """Overlay provider settings for a model category onto options."""
    settings = provider.data.get("settings", {})
    spec = MODEL_CATEGORY_SPECS.get(category, {})
    provider_key = spec.get("provider_key")
    if provider_key:
        options[provider_key] = provider.provider_type

    if provider.provider_type == "ollama":
        base_url = settings.get("base_url", RECOMMENDED_OLLAMA_URL)
        options.setdefault(CONF_OLLAMA_URL, base_url)
        if category == "chat":
            options[CONF_OLLAMA_CHAT_URL] = base_url
        if category == "vlm":
            options[CONF_OLLAMA_VLM_URL] = base_url
        if category == "summarization":
            options[CONF_OLLAMA_SUMMARIZATION_URL] = base_url

    if provider.provider_type == "openai" and (api_key := settings.get("api_key")):
        options[CONF_API_KEY] = api_key

    if provider.provider_type == "gemini" and (api_key := settings.get("api_key")):
        options[CONF_GEMINI_API_KEY] = api_key

    if category == "embedding":
        options[CONF_EMBEDDING_MODEL_PROVIDER] = provider.provider_type


def _apply_feature_model_to_options(
    options: dict[str, Any],
    category: str,
    provider_type: str,
    model_data: Mapping[str, Any],
) -> None:
    spec = MODEL_CATEGORY_SPECS.get(category, {})
    model_key = _model_key_for(category, provider_type)
    model_name = model_data.get(CONF_FEATURE_MODEL_NAME) or spec.get(
        "recommended_models", {}
    ).get(provider_type)
    if model_key and model_name:
        options[model_key] = model_name

    temp_key = spec.get("temperature_key")
    temp_value = model_data.get(CONF_FEATURE_MODEL_TEMPERATURE)
    if temp_value is None:
        temp_value = spec.get("recommended_temperature")
    if temp_key and temp_value is not None:
        options[temp_key] = temp_value

    if provider_type == "ollama":
        keepalive_map = {
            "chat": CONF_OLLAMA_CHAT_KEEPALIVE,
            "vlm": CONF_OLLAMA_VLM_KEEPALIVE,
            "summarization": CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
        }
        context_map = {
            "chat": CONF_OLLAMA_CHAT_CONTEXT_SIZE,
            "vlm": CONF_OLLAMA_VLM_CONTEXT_SIZE,
            "summarization": CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
        }
        keepalive_key = keepalive_map.get(category)
        keepalive_val = model_data.get(CONF_FEATURE_MODEL_KEEPALIVE)
        if keepalive_key and keepalive_val is not None:
            options[keepalive_key] = keepalive_val
        context_key = context_map.get(category)
        context_val = model_data.get(CONF_FEATURE_MODEL_CONTEXT_SIZE)
        if context_key and context_val is not None:
            options[context_key] = context_val
        if (
            category == "chat"
            and model_data.get(CONF_FEATURE_MODEL_REASONING) is not None
        ):
            options[CONF_OLLAMA_REASONING] = model_data.get(
                CONF_FEATURE_MODEL_REASONING
            )


def resolve_runtime_options(entry: ConfigEntry) -> dict[str, Any]:
    """
    Return effective options merging subentries with legacy values.

    This keeps global flags intact while allowing model/feature selection to
    flow through subentries.
    """
    base_options = _options_for(entry)
    providers = resolve_model_provider_configs(entry, base_options)
    features = resolve_feature_configs(entry, providers, base_options)

    options = dict(base_options)
    providers_by_id = dict(providers)

    category_provider: dict[str, ModelProviderConfig] = {}
    for feature in features.values():
        cat = FEATURE_CATEGORY_MAP.get(feature.feature_type)
        if not cat:
            continue
        provider = providers_by_id.get(feature.model_provider_id or "")
        if provider:
            category_provider.setdefault(cat, provider)

    for cat in ("chat", "vlm", "summarization", "embedding"):
        if cat in category_provider:
            continue
        if cat == "embedding" and "chat" in category_provider:
            chat_provider = category_provider["chat"]
            if "embedding" in chat_provider.capabilities:
                category_provider[cat] = chat_provider
                continue
        for provider in providers_by_id.values():
            if cat in provider.capabilities:
                category_provider[cat] = provider
                break

    for cat, provider in category_provider.items():
        _apply_provider_to_category(options, cat, provider)
        for feature in features.values():
            if FEATURE_CATEGORY_MAP.get(feature.feature_type) != cat:
                continue
            _apply_feature_model_to_options(
                options, cat, provider.provider_type, feature.model
            )

    return options
