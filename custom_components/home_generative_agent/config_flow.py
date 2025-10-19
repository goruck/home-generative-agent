"""Config flow for Home Generative Agent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
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
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
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
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION_MODE,
    CONF_GEMINI_API_KEY,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_REASONING,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_URL,
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
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_DB_URI,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_FACE_API_URL,
    RECOMMENDED_FACE_RECOGNITION_MODE,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_REASONING,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_URL,
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
from .core.utils import (
    CannotConnectError,
    InvalidAuthError,
    ensure_http_url,
    list_mobile_notify_services,
    validate_db_uri,
    validate_face_api_url,
    validate_gemini_key,
    validate_ollama_url,
    validate_openai_key,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_GEMINI_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(
            CONF_OLLAMA_URL,
            description={"suggested_value": RECOMMENDED_OLLAMA_URL},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_FACE_API_URL,
            description={"suggested_value": RECOMMENDED_FACE_API_URL},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Required(
            CONF_DB_URI,
            description={"suggested_value": RECOMMENDED_DB_URI},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
    },
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
    CONF_FACE_RECOGNITION_MODE: RECOMMENDED_FACE_RECOGNITION_MODE,
    CONF_FACE_API_URL: RECOMMENDED_FACE_API_URL,
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
    CONF_OLLAMA_REASONING: RECOMMENDED_OLLAMA_REASONING,
}


# ---------------------------
# Helpers
# ---------------------------


def _get_str(src: Mapping[str, Any], key: str) -> str:
    """Get a trimmed string from a mapping (missing -> '')."""
    return str(src.get(key, "") or "").strip()


def _model_option_key(cat: str, provider: str) -> str:
    """Return the stable option key for the model field of (category, provider)."""
    spec = MODEL_CATEGORY_SPECS[cat]
    key = spec.get("model_keys", {}).get(provider)
    if key:
        return key
    return f"model__{cat}__{provider}"


def _prune_irrelevant_model_fields(opts: Mapping[str, Any]) -> dict[str, Any]:
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


def _get_selected_provider(options: Mapping[str, Any], cat: str) -> str | None:
    spec = MODEL_CATEGORY_SPECS[cat]
    return options.get(spec["provider_key"], spec["recommended_provider"])


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


def _schema_for(hass: HomeAssistant, opts: Mapping[str, Any]) -> VolDictType:
    """Generate the options schema."""
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
        vol.Optional(
            CONF_API_KEY,
            description={"suggested_value": opts.get(CONF_API_KEY)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(
            CONF_GEMINI_API_KEY,
            description={"suggested_value": opts.get(CONF_GEMINI_API_KEY)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(
            CONF_OLLAMA_URL,
            description={"suggested_value": (opts.get(CONF_OLLAMA_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_FACE_API_URL,
            description={"suggested_value": (opts.get(CONF_FACE_API_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Required(
            CONF_DB_URI,
            description={"suggested_value": (opts.get(CONF_DB_URI))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
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
        vol.Optional(
            CONF_FACE_RECOGNITION_MODE,
            description={"suggested_value": opts.get(CONF_FACE_RECOGNITION_MODE)},
            default=RECOMMENDED_FACE_RECOGNITION_MODE,
        ): vol.In(["enable", "disable"]),
        vol.Optional(
            CONF_OLLAMA_REASONING,
            description={"suggested_value": opts.get(CONF_OLLAMA_REASONING)},
            default=RECOMMENDED_OLLAMA_REASONING,
        ): BooleanSelector(),
    }

    selected_mode = opts.get(CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE)
    if selected_mode != "disable":
        mobile_opts = list_mobile_notify_services(hass)
        if mobile_opts:
            schema[
                vol.Optional(
                    CONF_NOTIFY_SERVICE,
                    description={"suggested_value": opts.get(CONF_NOTIFY_SERVICE)},
                    default=opts.get(CONF_NOTIFY_SERVICE, mobile_opts[0]),
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=s.replace("notify.", ""), value=s)
                        for s in mobile_opts
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            )

    schema[
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": opts.get(CONF_RECOMMENDED)},
            default=opts.get(CONF_RECOMMENDED, False),
        )
    ] = bool

    # Recommended ON → no providers/models/temps
    if opts.get(CONF_RECOMMENDED, False):
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


@dataclass(frozen=True)
class _SecretSpec:
    field: str
    validator: Callable[[Any, str], Awaitable[None]]
    label: str


# ---------------------------
# Config Flow
# ---------------------------


class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""

    VERSION = 1

    async def _validate_present(
        self,
        hass: HomeAssistant,
        value: str,
        validator: Callable[[HomeAssistant, str], Any],
        log_label: str,
    ) -> str | None:
        """Run validator only when value is non-empty."""
        if not value:
            return None
        try:
            await validator(hass, value)
        except InvalidAuthError:
            return "invalid_auth"
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception during %s validation", log_label)
            return "unknown"
        else:
            return None

    async def _run_validations_user(
        self, data: Mapping[str, Any]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Validate inputs for the user step and normalize results."""
        errors: dict[str, str] = {}
        normalized: dict[str, Any] = dict(data)

        vals = {
            CONF_API_KEY: _get_str(data, CONF_API_KEY),
            CONF_OLLAMA_URL: _get_str(data, CONF_OLLAMA_URL),
            CONF_GEMINI_API_KEY: _get_str(data, CONF_GEMINI_API_KEY),
            CONF_DB_URI: _get_str(data, CONF_DB_URI),
            CONF_FACE_API_URL: _get_str(data, CONF_FACE_API_URL),
        }

        # Ordered, table-driven validation; short-circuits on first error.
        for key, validator, label in (
            (CONF_API_KEY, validate_openai_key, "OpenAI"),
            (CONF_OLLAMA_URL, validate_ollama_url, "Ollama"),
            (CONF_GEMINI_API_KEY, validate_gemini_key, "Gemini"),
            (CONF_DB_URI, validate_db_uri, "Database URI"),
            (CONF_FACE_API_URL, validate_face_api_url, "Face Recognition API"),
        ):
            code = await self._validate_present(self.hass, vals[key], validator, label)
            if code:
                errors["base"] = code
                break

        # Normalize URLs only on success.
        if not errors and vals[CONF_OLLAMA_URL]:
            normalized[CONF_OLLAMA_URL] = ensure_http_url(vals[CONF_OLLAMA_URL])
        if not errors and vals[CONF_FACE_API_URL]:
            normalized[CONF_FACE_API_URL] = ensure_http_url(vals[CONF_FACE_API_URL])

        # Drop empties so defaults can apply later.
        for key in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if not vals[key]:
                normalized.pop(key, None)

        return errors, normalized

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors, normalized = await self._run_validations_user(user_input)
        if errors:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
            )

        return self.async_create_entry(
            title="Home Generative Agent",
            data=normalized,
            options=RECOMMENDED_OPTIONS,
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)


# ---------------------------
# Options Flow
# ---------------------------


class HomeGenerativeAgentOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Home Generative Agent."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )
        self._last_providers = self._extract_provider_state(config_entry.options)
        self._last_analyzer_mode = config_entry.options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )

    # ---- helpers ----

    def _all_category_provider_keys(self) -> tuple[str, ...]:
        return tuple(spec["provider_key"] for spec in MODEL_CATEGORY_SPECS.values())

    def _providers_changed(
        self, prev: Mapping[str, Any], current: Mapping[str, Any]
    ) -> bool:
        return any(
            prev.get(k) != current.get(k) for k in self._all_category_provider_keys()
        )

    def _extract_provider_state(self, opts: Mapping[str, Any]) -> dict[str, Any]:
        keys = self._all_category_provider_keys()
        return {k: opts.get(k) for k in keys}

    def _apply_recommended_defaults(self, opts: Mapping[str, Any]) -> dict[str, Any]:
        """Force providers (and their model keys) to recommended values."""
        updated: dict[str, Any] = dict(opts)
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

    def _base_options_with_entry_data(self) -> dict[str, Any]:
        """Start from current options, overlaying any setup-time data for visibility."""
        options = dict(self.config_entry.options)
        for k in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if k not in options and self.config_entry.data.get(k):
                options[k] = self.config_entry.data[k]
        return options

    async def _maybe_edit_face_recognition_url(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply face recog URL when present; return error code or None."""
        if user_input is None or CONF_FACE_API_URL not in user_input:
            return None

        raw = _get_str(user_input, CONF_FACE_API_URL)
        if not raw:
            options.pop(CONF_FACE_API_URL, None)
            return None

        try:
            await validate_face_api_url(self.hass, raw)
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating face recognition api URL")
            return "unknown"

        options[CONF_FACE_API_URL] = ensure_http_url(raw)
        return None

    async def _maybe_edit_ollama(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply Ollama URL when present; return error code or None."""
        if user_input is None or CONF_OLLAMA_URL not in user_input:
            return None

        raw = _get_str(user_input, CONF_OLLAMA_URL)
        if not raw:
            options.pop(CONF_OLLAMA_URL, None)
            return None

        try:
            await validate_ollama_url(self.hass, raw)
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating Ollama URL")
            return "unknown"

        options[CONF_OLLAMA_URL] = ensure_http_url(raw)
        return None

    async def _maybe_edit_db_uri(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply DB URI when present; return error code or None."""
        if user_input is None or CONF_DB_URI not in user_input:
            return None

        raw = _get_str(user_input, CONF_DB_URI)
        if not raw:
            options.pop(CONF_DB_URI, None)
            return None

        try:
            await validate_db_uri(self.hass, raw)
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating DB URI")
            return "unknown"

        options[CONF_DB_URI] = raw
        return None

    async def _maybe_edit_secret(
        self,
        spec: _SecretSpec,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply a secret field when present; return error code or None."""
        if user_input is None or spec.field not in user_input:
            return None

        raw = _get_str(user_input, spec.field)
        if not raw:
            options.pop(spec.field, None)
            return None

        try:
            await spec.validator(self.hass, raw)
        except InvalidAuthError:
            return "invalid_auth"
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception in %s validation", spec.label)
            return "unknown"

        options[spec.field] = raw
        return None

    def _drop_empty_fields(self, final_options: dict[str, Any]) -> None:
        """Remove empty strings for fields to avoid storing empties."""
        for k in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if not _get_str(final_options, k):
                final_options.pop(k, None)

    def _cleanup_none_llm_api(self, options: dict[str, Any]) -> None:
        """Remove the 'none' sentinel so options omit the key when unset."""
        if options.get(CONF_LLM_HASS_API) == "none":
            options.pop(CONF_LLM_HASS_API, None)

    def _schema_changes_since_last(
        self, options: Mapping[str, Any]
    ) -> tuple[bool, bool, bool]:
        """Detect changes that require re-render."""
        recommended_now = options.get(CONF_RECOMMENDED, False)
        recommended_changed = recommended_now != self.last_rendered_recommended
        provider_changed = self._providers_changed(self._last_providers, options)
        analyzer_now = options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )
        analyzer_changed = analyzer_now != self._last_analyzer_mode
        return recommended_changed, provider_changed, analyzer_changed

    def _remember_schema_baseline(self, options: Mapping[str, Any]) -> None:
        """Record the state we used for last-rendered schema."""
        self.last_rendered_recommended = bool(options.get(CONF_RECOMMENDED, False))
        self._last_providers = self._extract_provider_state(options)
        self._last_analyzer_mode = options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )

    # ---- main step ----

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the options flow init step."""
        options = self._base_options_with_entry_data()

        # First render
        if user_input is None:
            return self.async_show_form(
                step_id="init", data_schema=vol.Schema(_schema_for(self.hass, options))
            )

        # Merge new input for non-validated fields
        options.update(user_input or {})
        errors: dict[str, str] = {}

        # Field-specific edits with validation/normalization
        err = await self._maybe_edit_ollama(options, user_input)
        if not err:
            err = await self._maybe_edit_db_uri(options, user_input)
        if not err:
            err = await self._maybe_edit_face_recognition_url(options, user_input)
        if not err:
            err = await self._maybe_edit_secret(
                _SecretSpec(CONF_GEMINI_API_KEY, validate_gemini_key, "Gemini Options"),
                options,
                user_input,
            )
        if not err:
            err = await self._maybe_edit_secret(
                _SecretSpec(CONF_API_KEY, validate_openai_key, "OpenAI Options"),
                options,
                user_input,
            )
        if err:
            errors["base"] = err

        if errors:
            # Re-render with the same options and show errors
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(_schema_for(self.hass, options)),
                errors=errors,
            )

        # Handle schema-affecting toggles
        recommended_changed, provider_changed, analyzer_changed = (
            self._schema_changes_since_last(options)
        )

        # If Recommended turned ON → apply defaults and SAVE immediately
        if recommended_changed and options.get(CONF_RECOMMENDED, False):
            final_options = self._apply_recommended_defaults(options)
            final_options = _prune_irrelevant_model_fields(final_options)
            self._cleanup_none_llm_api(final_options)
            self._drop_empty_fields(final_options)
            self._remember_schema_baseline(final_options)
            return self.async_create_entry(title="", data=final_options)

        # If Recommended toggled OFF or provider/analyzer changes, re-render once
        if recommended_changed or provider_changed or analyzer_changed:
            pruned = _prune_irrelevant_model_fields(options)
            self._remember_schema_baseline(pruned)
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(_schema_for(self.hass, pruned)),
            )

        # Finalize and save (no schema changes)
        final_options = _prune_irrelevant_model_fields(options)
        if final_options.get(CONF_RECOMMENDED, False):
            final_options = self._apply_recommended_defaults(final_options)
            final_options = _prune_irrelevant_model_fields(final_options)
        self._cleanup_none_llm_api(final_options)
        self._drop_empty_fields(final_options)

        return self.async_create_entry(title="", data=final_options)
