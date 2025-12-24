"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    OptionsFlowWithReload,
)
from homeassistant.const import (
    CONF_LLM_HASS_API,
)
from homeassistant.core import callback
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
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION,
    CONF_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_MAX_TOKENS_IN_CONTEXT,
    CONF_NOTIFY_SERVICE,
    CONF_PROMPT,
    CONF_VIDEO_ANALYZER_MODE,
    CONFIG_ENTRY_VERSION,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    DOMAIN,
    LLM_HASS_API_NONE,
    RECOMMENDED_FACE_RECOGNITION,
    RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
    RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY,
    VIDEO_ANALYZER_MODE_DISABLE,
    VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY,
)
from .core.utils import (
    CannotConnectError,
    ensure_http_url,
    hash_pin,
    list_mobile_notify_services,
    validate_face_api_url,
)
from .flows.feature_subentry_flow import FeatureSubentryFlow
from .flows.model_provider_subentry_flow import ModelProviderSubentryFlow

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CRITICAL_ACTION_PIN_ENABLED: True,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
    CONF_FACE_RECOGNITION: RECOMMENDED_FACE_RECOGNITION,
    CONF_MANAGE_CONTEXT_WITH_TOKENS: RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_TOKENS_IN_CONTEXT: RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    CONF_MAX_MESSAGES_IN_CONTEXT: RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
}

# ---------------------------
# Helpers
# ---------------------------


def _get_str(src: Mapping[str, Any], key: str) -> str:
    """Get a trimmed string from a mapping (missing -> '')."""
    return str(src.get(key, "") or "").strip()


async def _schema_for_options(
    hass: HomeAssistant, opts: Mapping[str, Any]
) -> VolDictType:
    """Generate the options schema for non-provider settings."""
    hass_apis = [SelectOptionDict(label="No control", value=LLM_HASS_API_NONE)] + [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    video_analyzer_mode_opts: list[SelectOptionDict] = [
        SelectOptionDict(label="Disable", value=VIDEO_ANALYZER_MODE_DISABLE),
        SelectOptionDict(
            label="Notify on anomaly", value=VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY
        ),
        SelectOptionDict(
            label="Always notify", value=VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY
        ),
    ]

    context_mgmt_modes = [
        SelectOptionDict(label="Use tokens", value="true"),
        SelectOptionDict(label="Use messages", value="false"),
    ]

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": opts.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": opts.get(CONF_LLM_HASS_API)},
            default=LLM_HASS_API_NONE,
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": opts.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE,
        ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode_opts)),
        vol.Optional(
            CONF_FACE_API_URL,
            description={"suggested_value": opts.get(CONF_FACE_API_URL)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_MANAGE_CONTEXT_WITH_TOKENS,
            description={
                "suggested_value": opts.get(CONF_MANAGE_CONTEXT_WITH_TOKENS, "true")
            },
            default=RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
        ): SelectSelector(
            SelectSelectorConfig(
                options=context_mgmt_modes,
                mode=SelectSelectorMode.DROPDOWN,
                sort=False,
                custom_value=False,
            )
        ),
        vol.Optional(
            CONF_MAX_TOKENS_IN_CONTEXT,
            description={"suggested_value": opts.get(CONF_MAX_TOKENS_IN_CONTEXT)},
            default=RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
        ): NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1)),
        vol.Optional(
            CONF_MAX_MESSAGES_IN_CONTEXT,
            description={"suggested_value": opts.get(CONF_MAX_MESSAGES_IN_CONTEXT)},
            default=RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
        ): NumberSelector(NumberSelectorConfig(min=15, max=240, step=1)),
        vol.Optional(
            CONF_CRITICAL_ACTION_PIN_ENABLED,
            description={
                "suggested_value": opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True)
            },
            default=opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True),
        ): BooleanSelector(),
    }

    if opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True):
        schema[
            vol.Optional(
                CONF_CRITICAL_ACTION_PIN,
                description={
                    "suggested_value": "",
                    "placeholder": "Set/replace PIN for critical actions",
                },
            )
        ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

    video_analyzer_mode = opts.get(
        CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
    )
    if video_analyzer_mode != VIDEO_ANALYZER_MODE_DISABLE:
        schema[
            vol.Optional(
                CONF_FACE_RECOGNITION,
                description={"suggested_value": opts.get(CONF_FACE_RECOGNITION)},
                default=RECOMMENDED_FACE_RECOGNITION,
            )
        ] = BooleanSelector()

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

    return schema


# ---------------------------
# Config Flow
# ---------------------------


class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""

    VERSION = CONFIG_ENTRY_VERSION

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({}))

        return self.async_create_entry(
            title="Home Generative Agent",
            data={},
            options=dict(DEFAULT_OPTIONS),
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        _ = config_entry
        return HomeGenerativeAgentOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return supported subentry flow handlers."""
        _ = config_entry
        return {
            SUBENTRY_TYPE_MODEL_PROVIDER: ModelProviderSubentryFlow,
            SUBENTRY_TYPE_FEATURE: FeatureSubentryFlow,
        }


# ---------------------------
# Options Flow
# ---------------------------


class HomeGenerativeAgentOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Home Generative Agent."""

    # ---- helpers ----

    def _base_options(self) -> dict[str, Any]:
        options = dict(DEFAULT_OPTIONS)
        options.update(self.config_entry.options)
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

    def _maybe_edit_pin(
        self, options: dict[str, Any], user_input: Mapping[str, Any] | None
    ) -> str | None:
        """Hash and store the critical-action PIN if provided."""
        if user_input is None:
            return None

        pin_enabled = user_input.get(
            CONF_CRITICAL_ACTION_PIN_ENABLED,
            options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True),
        )
        options[CONF_CRITICAL_ACTION_PIN_ENABLED] = pin_enabled

        if not pin_enabled:
            options.pop(CONF_CRITICAL_ACTION_PIN, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_HASH, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_SALT, None)
            return None

        if CONF_CRITICAL_ACTION_PIN not in user_input:
            return None

        raw = _get_str(user_input, CONF_CRITICAL_ACTION_PIN)
        options.pop(CONF_CRITICAL_ACTION_PIN, None)
        if not raw:
            options.pop(CONF_CRITICAL_ACTION_PIN_HASH, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_SALT, None)
            return None

        if (
            not raw.isdigit()
            or not CRITICAL_PIN_MIN_LEN <= len(raw) <= CRITICAL_PIN_MAX_LEN
        ):
            return "invalid_pin"

        hashed, salt = hash_pin(raw)
        options[CONF_CRITICAL_ACTION_PIN_HASH] = hashed
        options[CONF_CRITICAL_ACTION_PIN_SALT] = salt
        return None

    def _drop_empty_fields(self, final_options: dict[str, Any]) -> None:
        """Remove empty strings for fields to avoid storing empties."""
        for k in (
            CONF_FACE_API_URL,
            CONF_NOTIFY_SERVICE,
        ):
            if not _get_str(final_options, k):
                final_options.pop(k, None)

    def _cleanup_none_llm_api(self, options: dict[str, Any]) -> None:
        """Remove the 'none' sentinel so options omit the key when unset."""
        if options.get(CONF_LLM_HASS_API) == LLM_HASS_API_NONE:
            options.pop(CONF_LLM_HASS_API, None)

    # ---- main step ----

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the options flow init step."""
        options = self._base_options()

        # First render
        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(await _schema_for_options(self.hass, options)),
            )

        # Merge new input for non-validated fields
        options.update(user_input or {})
        errors: dict[str, str] = {}

        # Field-specific edits with validation/normalization
        err = await self._maybe_edit_face_recognition_url(options, user_input)
        if not err:
            err = self._maybe_edit_pin(options, user_input)
        if err:
            errors["base"] = err

        if errors:
            # Re-render with the same options and show errors
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(await _schema_for_options(self.hass, options)),
                errors=errors,
            )

        self._cleanup_none_llm_api(options)
        self._drop_empty_fields(options)
        return self.async_create_entry(title="", data=options)
