"""Config flow for Home Generative Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_VLM,
    CONF_OPENAI_CHAT_MODEL,
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
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_UNAUTHORIZED,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
)

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
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

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType


async def _validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """
    Check that the user has entered a valid OpenAI API key.

    Data has the key from STEP_USER_DATA_SCHEMA provided by the user.
    """
    api_key = data[CONF_API_KEY]
    client = get_async_client(hass)

    try:
        # Fast, non-fatal reachability check like __init__.py
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except Exception as err:
        LOGGER.debug("OpenAI connectivity exception during validation: %s", err)
        raise CannotConnectError from err
    if resp.status_code == HTTP_STATUS_UNAUTHORIZED:
        raise InvalidAuthError
    if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
        raise CannotConnectError


def _all_category_provider_keys() -> tuple[str, ...]:
    return tuple(spec["provider_key"] for spec in MODEL_CATEGORY_SPECS.values())


def _get_selected_provider(
    options: dict[str, Any] | MappingProxyType[str, Any], cat: str
) -> str | None:
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
        mode=SelectSelectorMode.DROPDOWN,
        sort=False,
        custom_value=False,
    )


def _model_selector_config(cat: str, provider: str) -> SelectSelectorConfig:
    models = MODEL_CATEGORY_SPECS[cat]["providers"].get(provider, [])
    return SelectSelectorConfig(
        options=[SelectOptionDict(label=m, value=m) for m in models],
        mode=SelectSelectorMode.DROPDOWN,
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
        for prov in spec["providers"]:
            updated.pop(_model_option_key(cat, prov), None)

        # Set recommended model for the recommended provider, if known
        rec_model = spec.get("recommended_models", {}).get(rec_provider)
        if rec_model:
            updated[_model_option_key(cat, rec_provider)] = rec_model
    return updated


def _extract_provider_state(
    opts: dict[str, Any] | MappingProxyType[str, Any],
) -> dict[str, Any]:
    keys = _all_category_provider_keys()
    return {k: opts.get(k) for k in keys}


def _providers_changed(prev: dict[str, Any], current: dict[str, Any]) -> bool:
    return any(prev.get(k) != current.get(k) for k in _all_category_provider_keys())


def _prune_irrelevant_model_fields(opts: dict[str, Any]) -> dict[str, Any]:
    """
    Drop model fields that don't match the chosen provider for each category.

    Also drop temperatures when recommended is enabled.
    """
    pruned = dict(opts)

    # Strip temps if recommended
    if pruned.get(CONF_RECOMMENDED):
        for spec in MODEL_CATEGORY_SPECS.values():
            temp_key = spec.get("temperature_key")
            if temp_key:
                pruned.pop(temp_key, None)

    # Remove model keys for providers not selected
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        selected = pruned.get(spec["provider_key"])
        for provider in spec["providers"]:
            key = _model_option_key(cat, provider)
            if provider != selected:
                pruned.pop(key, None)

    return pruned


def _schema_for(hass: HomeAssistant, opts: dict[str, Any]) -> VolDictType:
    """Generate the options schema with API key field first."""
    hass_apis = [SelectOptionDict(label="No control", value="none")] + [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]
    video_analyzer_mode: list[SelectOptionDict] = [
        SelectOptionDict(label="Disable", value="disable"),
        SelectOptionDict(label="Notify on anomaly", value="notify_on_anomaly"),
        SelectOptionDict(label="Always notify", value="always_notify"),
    ]

    schema: VolDictType = {
        # --- OpenAI API key (optional, masked) ---
        vol.Optional(
            CONF_API_KEY,
            description={"suggested_value": opts.get(CONF_API_KEY)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        # --- existing base fields ---
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": opts.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": opts.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": opts.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE,
        ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode)),
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": opts.get(CONF_RECOMMENDED)},
            default=opts.get(CONF_RECOMMENDED, False),
        ): bool,
    }

    # Recommended ON → no providers/models/temps
    recommended_on = opts.get(CONF_RECOMMENDED, False)
    if recommended_on:
        return schema

    # Recommended OFF → show providers + their model + temp
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        provider_key = spec["provider_key"]

        # Provider select
        schema[
            vol.Optional(
                provider_key,
                description={"suggested_value": opts.get(provider_key)},
                default=opts.get(provider_key, spec["recommended_provider"]),
            )
        ] = SelectSelector(_provider_selector_config(cat))

        # Model select for chosen provider
        selected_provider = _get_selected_provider(opts, cat)
        if selected_provider:
            model_key = _model_option_key(cat, selected_provider)
            default_model = spec.get("recommended_models", {}).get(selected_provider)
            schema[
                vol.Optional(
                    model_key,
                    description={"suggested_value": opts.get(model_key)},
                    default=opts.get(model_key, default_model),
                )
            ] = SelectSelector(_model_selector_config(cat, selected_provider))

        # Temperature (if used by this category)
        temp_key = spec.get("temperature_key")
        if temp_key:
            schema[
                vol.Optional(
                    temp_key,
                    description={"suggested_value": opts.get(temp_key)},
                    default=spec.get("recommended_temperature", 1.0),
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05))

    return schema


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

        api_key = (user_input.get(CONF_API_KEY) or "").strip()

        if api_key:
            try:
                await _validate_input(self.hass, user_input)
            except CannotConnectError:
                errors["base"] = "cannot_connect"
            except InvalidAuthError:
                errors["base"] = "invalid_auth"
            except Exception:
                LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
            )

        # Allow empty api_key, proceed
        return self.async_create_entry(
            title="Home Generative Agent",
            data=user_input,  # may omit CONF_API_KEY
            options=RECOMMENDED_OPTIONS,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)


class HomeGenerativeAgentOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Home Generative Agent."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )
        self._last_providers = _extract_provider_state(config_entry.options)

    async def async_step_init(  # noqa: PLR0912
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the options flow init step."""
        # Start from current options + DATA
        # (for backward-compat if API key was set at setup)
        options = dict(self.config_entry.options)
        if CONF_API_KEY not in options and self.config_entry.data.get(CONF_API_KEY):
            # Surface existing key from data into the form as suggested_value,
            # we won't mutate data here.
            options[CONF_API_KEY] = self.config_entry.data[CONF_API_KEY]

        # First render
        if user_input is None:
            return self.async_show_form(
                step_id="init", data_schema=vol.Schema(_schema_for(self.hass, options))
            )

        # Merge new input
        options.update(user_input or {})

        errors: dict[str, str] = {}

        # --- Handle API key edits explicitly ---
        api_key = CONF_API_KEY in (user_input or {})
        if api_key:
            raw = (user_input.get(CONF_API_KEY) or "").strip()
            if raw:
                # validate only when provided
                try:
                    await _validate_input(self.hass, {CONF_API_KEY: raw})
                except CannotConnectError:
                    errors["base"] = "cannot_connect"
                except InvalidAuthError:
                    errors["base"] = "invalid_auth"
                except Exception:
                    LOGGER.exception("Unexpected exception in Options validation")
                    errors["base"] = "unknown"
                else:
                    options[CONF_API_KEY] = raw
            else:
                # explicit delete when user clears the field
                options.pop(CONF_API_KEY, None)
        # If not provided in user_input: leave existing options[CONF_API_KEY] as-is

        if errors:
            # Re-render with same options and show errors
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(_schema_for(self.hass, options)),
                errors=errors,
            )

        # Handle schema-affecting toggles (recommended/providers)
        recommended_now = options.get(CONF_RECOMMENDED, False)
        recommended_changed = recommended_now != self.last_rendered_recommended
        provider_changed = _providers_changed(self._last_providers, options)

        # If Recommended turned ON → apply defaults and SAVE immediately
        if recommended_changed and recommended_now:
            options = _apply_recommended_defaults(options)
            options = _prune_irrelevant_model_fields(options)
            if options.get(CONF_LLM_HASS_API) == "none":
                options.pop(CONF_LLM_HASS_API, None)
            # Normalize API key: drop if blank
            if not api_key:
                options.pop(CONF_API_KEY, None)
            self.last_rendered_recommended = True
            self._last_providers = _extract_provider_state(options)
            return self.async_create_entry(title="", data=options)

        # If Recommended toggled OFF or providers changed → re-render once
        if recommended_changed or provider_changed:
            options = _prune_irrelevant_model_fields(options)
            self.last_rendered_recommended = recommended_now
            self._last_providers = _extract_provider_state(options)
            return self.async_show_form(
                step_id="init", data_schema=vol.Schema(_schema_for(self.hass, options))
            )

        # Finalize and save (no schema changes)
        final_options = _prune_irrelevant_model_fields(options)
        if final_options.get(CONF_RECOMMENDED, False):
            final_options = _apply_recommended_defaults(final_options)
            final_options = _prune_irrelevant_model_fields(final_options)
        if final_options.get(CONF_LLM_HASS_API) == "none":
            final_options.pop(CONF_LLM_HASS_API, None)

        # Normalize API key: drop if blank so you don't store empty strings
        if not api_key:
            final_options.pop(CONF_API_KEY, None)

        return self.async_create_entry(title="", data=final_options)


class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuthError(HomeAssistantError):
    """Error to indicate there is invalid auth."""
