"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    selector,
)
from langchain_openai import ChatOpenAI

from .const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_EMBEDDING_MODEL,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_VLM,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_EMBEDDING_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
)

# ---- Dynamic, data-driven provider+model registry ----
# If you add a new provider (e.g., "anthropic"), just extend the dicts below.
MODEL_CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "chat": {
        "provider_key": CONF_CHAT_MODEL_PROVIDER,
        "temperature_key": CONF_CHAT_MODEL_TEMPERATURE,
        "recommended_provider": RECOMMENDED_CHAT_MODEL_PROVIDER,
        "recommended_temperature": RECOMMENDED_CHAT_MODEL_TEMPERATURE,
        # Per-provider available models for UI
        "providers": {
            "openai": ["gpt-4.1", "gpt-4o", "o4-mini"],
            "ollama": ["qwen2.5:32b", "qwen3:32b", "qwen3:8b"],
            # "anthropic": ["claude-3-5-sonnet"]  # example: add here
        },
        # Default model per provider
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_CHAT_MODEL,
            "ollama": RECOMMENDED_OLLAMA_CHAT_MODEL,
        },
        # Stable option-key per (category, provider). If missing, we fall back to model__{cat}__{provider}
        "model_keys": {
            "openai": CONF_OPENAI_CHAT_MODEL,
            "ollama": CONF_OLLAMA_CHAT_MODEL,
        },
    },
    "vlm": {
        "provider_key": CONF_VLM_PROVIDER,
        "temperature_key": CONF_VLM_TEMPERATURE,
        "recommended_provider": RECOMMENDED_VLM_PROVIDER,
        "recommended_temperature": RECOMMENDED_VLM_TEMPERATURE,
        "providers": {
            "openai": ["gpt-4.1", "gpt-4.1-nano"],
            "ollama": ["qwen2.5vl:7b"],
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_VLM,
            "ollama": RECOMMENDED_OLLAMA_VLM,
        },
        "model_keys": {
            "openai": CONF_OPENAI_VLM,
            "ollama": CONF_OLLAMA_VLM,
        },
    },
    "summarization": {
        "provider_key": CONF_SUMMARIZATION_MODEL_PROVIDER,
        "temperature_key": CONF_SUMMARIZATION_MODEL_TEMPERATURE,
        "recommended_provider": RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
        "recommended_temperature": RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
        "providers": {
            "openai": ["gpt-4.1", "gpt-4.1-nano"],
            "ollama": ["qwen3:1.7b", "qwen3:8b"],
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "ollama": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_SUMMARIZATION_MODEL,
            "ollama": CONF_OLLAMA_SUMMARIZATION_MODEL,
        },
    },
    "embedding": {
        "provider_key": CONF_EMBEDDING_MODEL_PROVIDER,
        "temperature_key": None,  # embeddings don’t use temperature
        "recommended_provider": RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
        "recommended_temperature": None,
        "providers": {
            "openai": ["text-embedding-3-large", "text-embedding-3-small"],
            "ollama": ["mxbai-embed-large"],
        },
        "recommended_models": {
            "openai": RECOMMENDED_OPENAI_EMBEDDING_MODEL,
            "ollama": RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
        },
        "model_keys": {
            "openai": CONF_OPENAI_EMBEDDING_MODEL,
            "ollama": CONF_OLLAMA_EMBEDDING_MODEL,
        },
    },
}

def _all_category_provider_keys() -> tuple[str, ...]:
    return tuple(spec["provider_key"] for spec in MODEL_CATEGORY_SPECS.values())

def _get_selected_provider(options: dict[str, Any], cat: str) -> str | None:
    spec = MODEL_CATEGORY_SPECS[cat]
    return options.get(spec["provider_key"], spec["recommended_provider"])

def _model_option_key(cat: str, provider: str) -> str:
    """Return the stable option key for the model field of (category, provider)."""
    spec = MODEL_CATEGORY_SPECS[cat]
    key = spec.get("model_keys", {}).get(provider)
    if key:
        return key
    # Fallback stable naming for new providers without explicit constants
    return f"model__{cat}__{provider}"

def _provider_selector_config(cat: str) -> SelectSelectorConfig:
    providers = list(MODEL_CATEGORY_SPECS[cat]["providers"].keys())
    return SelectSelectorConfig(
        options=[SelectOptionDict(label=p.title(), value=p) for p in providers],
        mode="dropdown",
        sort=False,
        custom_value=False,
    )

def _model_selector_config(cat: str, provider: str) -> SelectSelectorConfig:
    models = MODEL_CATEGORY_SPECS[cat]["providers"].get(provider, [])
    # Allow custom_value=True so users can type an arbitrary model name
    return SelectSelectorConfig(
        options=[SelectOptionDict(label=m, value=m) for m in models],
        mode="dropdown",
        sort=False,
        custom_value=True,
    )
    
def _apply_recommended_defaults(opts: dict[str, Any]) -> dict[str, Any]:
    """Force providers (and their model keys) to recommended values."""
    updated = dict(opts)
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        rec_provider = spec["recommended_provider"]
        provider_key = spec["provider_key"]
        updated[provider_key] = rec_provider

        # Clear all model keys for this category
        for prov in spec["providers"].keys():
            updated.pop(_model_option_key(cat, prov), None)

        # Set recommended model for the recommended provider, if known
        rec_model = spec.get("recommended_models", {}).get(rec_provider)
        if rec_model:
            updated[_model_option_key(cat, rec_provider)] = rec_model

        # Temperatures are hidden in recommended mode; do not set them here.
        # (They'll be pruned by _prune_irrelevant_model_fields if present.)
    return updated

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_VIDEO_ANALYZER_MODE: "disable",
    CONF_CHAT_MODEL_PROVIDER: RECOMMENDED_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE: RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    CONF_VLM_PROVIDER: RECOMMENDED_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE: RECOMMENDED_VLM_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_PROVIDER: RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE: RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL_PROVIDER: RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_CHAT_MODEL: RECOMMENDED_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL: RECOMMENDED_OPENAI_CHAT_MODEL,
    CONF_OLLAMA_VLM: RECOMMENDED_OLLAMA_VLM,
    CONF_OPENAI_VLM: RECOMMENDED_OPENAI_VLM,
    CONF_OLLAMA_SUMMARIZATION_MODEL: RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL: RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
}


async def _validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """
    Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = ChatOpenAI(api_key=data[CONF_API_KEY], async_client=get_async_client(hass))
    await hass.async_add_executor_job(client.bind(timeout=10).get_name)


class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await _validate_input(self.hass, user_input)
        except CannotConnectError:
            errors["base"] = "cannot_connect"
        except InvalidAuthError:
            errors["base"] = "invalid_auth"
        except Exception:
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="HGA",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)


class HomeGenerativeAgentOptionsFlow(OptionsFlow):
    def __init__(self, config_entry: ConfigEntry) -> None:
        self.config_entry = config_entry
        self.last_rendered_recommended = config_entry.options.get(CONF_RECOMMENDED, False)
        self._last_providers = _extract_provider_state(config_entry.options)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        LOGGER.debug("Options flow init called with user_input: %s", user_input)

        options = dict(self.config_entry.options)

        if user_input is None:
            schema = _config_option_schema(self.hass, options)
            return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))

        # Merge new input
        options.update(user_input)

        recommended_now = options.get(CONF_RECOMMENDED, False)
        recommended_changed = (recommended_now != self.last_rendered_recommended)
        provider_changed = _providers_changed(self._last_providers, options)

        # If Recommended just turned ON → apply defaults and SAVE NOW (no second submit)
        if recommended_changed and recommended_now:
            options = _apply_recommended_defaults(options)
            options = _prune_irrelevant_model_fields(options)
            if options.get(CONF_LLM_HASS_API) == "none":
                options.pop(CONF_LLM_HASS_API, None)
            self.last_rendered_recommended = True
            self._last_providers = _extract_provider_state(options)
            return self.async_create_entry(title="", data=options)

        # If Recommended toggled OFF or any provider changed → re-render once
        if recommended_changed or provider_changed:
            options = _prune_irrelevant_model_fields(options)
            self.last_rendered_recommended = recommended_now
            self._last_providers = _extract_provider_state(options)
            schema = _config_option_schema(self.hass, options)
            return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))

        # No schema-affecting changes → finalize and save
        final_options = _prune_irrelevant_model_fields(options)
        if final_options.get(CONF_RECOMMENDED, False):
            final_options = _apply_recommended_defaults(final_options)
            final_options = _prune_irrelevant_model_fields(final_options)
        if final_options.get(CONF_LLM_HASS_API) == "none":
            final_options.pop(CONF_LLM_HASS_API, None)

        return self.async_create_entry(title="", data=final_options)

# ---- generic helpers for provider state + pruning ----

def _extract_provider_state(opts: dict[str, Any] | MappingProxyType[str, Any]) -> dict[str, Any]:
    keys = _all_category_provider_keys()
    return {k: opts.get(k) for k in keys}

def _providers_changed(prev: dict[str, Any], current: dict[str, Any]) -> bool:
    for k in _all_category_provider_keys():
        if prev.get(k) != current.get(k):
            return True
    return False

def _prune_irrelevant_model_fields(opts: dict[str, Any]) -> dict[str, Any]:
    """Drop model fields that don't match the chosen provider for each category.
    Also drop temperatures when recommended is enabled.
    """
    pruned = dict(opts)

    # Strip temps if recommended
    if pruned.get(CONF_RECOMMENDED):
        for cat, spec in MODEL_CATEGORY_SPECS.items():
            temp_key = spec.get("temperature_key")
            if temp_key:
                pruned.pop(temp_key, None)

    # Remove model keys for providers not selected
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        selected = pruned.get(spec["provider_key"])
        for provider in spec["providers"].keys():
            key = _model_option_key(cat, provider)
            if provider != selected:
                pruned.pop(key, None)
        if not selected:
            for provider in spec["providers"].keys():
                pruned.pop(_model_option_key(cat, provider), None)

    return pruned

class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuthError(HomeAssistantError):
    """Error to indicate there is invalid auth."""


def _config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    *,
    show_submenus: bool = False,  # unused
) -> VolDictType:
    """Providers, models, temps only when Recommended is OFF."""
    hass_apis = [SelectOptionDict(label="No control", value="none")] + [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]
    video_analyzer_mode: list[SelectOptionDict] = [
        SelectOptionDict(label="Disable", value="disable"),
        SelectOptionDict(label="Notify on anomaly", value="notify_on_anomaly"),
        SelectOptionDict(label="Always notify", value="always_notify"),
    ]

    # Base fields (always visible)
    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": options.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE,
        ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode)),
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": options.get(CONF_RECOMMENDED)},
            default=options.get(CONF_RECOMMENDED, False),
        ): bool,
    }

    recommended_on = options.get(CONF_RECOMMENDED, False)
    if recommended_on:
        # In recommended mode: no providers, no models, no temperatures.
        return schema

    # Recommended is OFF → show providers, with their model + temperature directly under each.
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        provider_key = spec["provider_key"]

        # Provider select
        schema[vol.Optional(
            provider_key,
            description={"suggested_value": options.get(provider_key)},
            default=options.get(provider_key, spec["recommended_provider"]),
        )] = SelectSelector(_provider_selector_config(cat))

        # Model select for the chosen provider
        selected_provider = _get_selected_provider(options, cat)
        if selected_provider:
            model_key = _model_option_key(cat, selected_provider)
            default_model = spec.get("recommended_models", {}).get(selected_provider)
            schema[vol.Optional(
                model_key,
                description={"suggested_value": options.get(model_key)},
                default=options.get(model_key, default_model),
            )] = SelectSelector(_model_selector_config(cat, selected_provider))

        # Temperature directly under model (if this category uses temperature)
        temp_key = spec.get("temperature_key")
        if temp_key:
            schema[vol.Optional(
                temp_key,
                description={"suggested_value": options.get(temp_key)},
                default=spec.get("recommended_temperature", 1.0),
            )] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))

    return schema

def _append_temperature_controls(schema: VolDictType, options: dict[str, Any]) -> VolDictType:
    """Append temperature controls for categories that use them."""
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        temp_key = spec.get("temperature_key")
        if not temp_key:
            continue
        default_temp = spec.get("recommended_temperature", 1.0)
        schema[vol.Optional(
            temp_key,
            description={"suggested_value": options.get(temp_key)},
            default=default_temp,
        )] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))
    return schema
