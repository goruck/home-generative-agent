"""Text-to-speech provider config subentry flow."""

from __future__ import annotations

import logging
from typing import Any, get_args

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import (  # noqa: TID252
    CONF_TTS_INSTRUCTIONS,
    CONF_TTS_MODEL_NAME,
    CONF_TTS_OPENAI_PROVIDER_ID,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_LOCAL_TTS_MODEL,
    RECOMMENDED_LOCAL_TTS_VOICE,
    RECOMMENDED_OPENAI_TTS_MODEL,
    RECOMMENDED_OPENAI_TTS_VOICE,
    SUBENTRY_TYPE_TTS_PROVIDER,
    TTS_MODEL_OPENAI_SUPPORTED,
    TTS_SPEED_DEFAULT,
    TTS_SPEED_MAX,
    TTS_SPEED_MIN,
)
from .openai_compatible_endpoint import (
    build_local_endpoint_settings,
    build_openai_key_settings,
    current_subentry,
    local_endpoint_schema,
    openai_key_schema,
    openai_provider_options,
    resolve_provider_name,
)

LOGGER = logging.getLogger(__name__)

ProviderNames = {
    "openai": "TTS - OpenAI",
    "local": "TTS - Local",
}
_FALLBACK_NAME = "TTS Provider"


def _coerce_speed(value: Any) -> float:
    """Clamp a submitted speed into the API's range; blank means the default."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return TTS_SPEED_DEFAULT
    return min(max(speed, TTS_SPEED_MIN), TTS_SPEED_MAX)


class TtsProviderSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for TTS provider subentries."""

    def __init__(self) -> None:
        """Initialize the TTS provider flow."""
        self._provider_type: str | None = None
        self._name: str | None = None
        self._settings: dict[str, Any] = {}
        self._model: dict[str, Any] = {}

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for TTS provider setup or reconfigure."""
        current = current_subentry(self, SUBENTRY_TYPE_TTS_PROVIDER)
        if current:
            self._provider_type = current.data.get("provider_type")
            self._name = current.data.get("name") or ProviderNames.get(
                self._provider_type or "openai", _FALLBACK_NAME
            )
            self._settings = dict(current.data.get("settings") or {})
            self._model = dict(current.data.get("model") or {})
        return await self.async_step_provider(user_input)

    async_step_reconfigure = async_step_user

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select provider type and name."""
        errors: dict[str, str] = {}
        if user_input is not None:
            provider_type = user_input.get("provider_type") or "openai"
            type_changed = (
                self._provider_type is not None and provider_type != self._provider_type
            )
            if type_changed:
                # Switching provider types must not carry state across: the
                # stored OpenAI key would otherwise prefill the local server's
                # optional key field, and the other type's model and voice ids
                # are invalid for the new one.
                self._settings = {}
                self._model = {}
            self._name = resolve_provider_name(
                user_input.get("name"),
                provider_type,
                previous_type=self._provider_type,
                current_name=self._name,
                provider_names=ProviderNames,
            ) or ProviderNames.get(provider_type, _FALLBACK_NAME)
            self._provider_type = provider_type
            return await self.async_step_credentials()

        provider_type = self._provider_type or "openai"
        default_name = self._name or ProviderNames.get(provider_type, _FALLBACK_NAME)
        schema = vol.Schema(
            {
                vol.Required(
                    "provider_type",
                    default=provider_type,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="OpenAI", value="openai"),
                            SelectOptionDict(
                                label="Local (OpenAI-compatible)", value="local"
                            ),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
                vol.Optional(
                    "name",
                    description={"suggested_value": default_name},
                    default=default_name,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            }
        )
        return self.async_show_form(
            step_id="provider", data_schema=schema, errors=errors
        )

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure provider credentials."""
        errors: dict[str, str] = {}
        provider_type = self._provider_type or "openai"
        openai_opts = openai_provider_options(self)

        if user_input is not None:
            if provider_type == "local":
                settings, error = await build_local_endpoint_settings(
                    self.hass, user_input, provider_id_key=CONF_TTS_OPENAI_PROVIDER_ID
                )
            elif provider_type == "openai":
                settings, error = await build_openai_key_settings(
                    self.hass,
                    openai_opts,
                    user_input,
                    provider_id_key=CONF_TTS_OPENAI_PROVIDER_ID,
                )
            else:
                # Unreachable today (the provider select admits only the two
                # types above, custom_value=False) — kept so a future type
                # added to the select but not wired here fails visibly.
                settings, error = {}, "not_supported"
            if error:
                errors["base"] = error
            else:
                self._settings = settings
                return await self.async_step_model()

        if provider_type == "local":
            # On a validation error, redisplay what was just typed — not the
            # stored settings — so a retry against a booting server does not
            # demand retyping the URL.
            prefill = (
                user_input if user_input is not None and errors else self._settings
            )
            return self.async_show_form(
                step_id="credentials",
                data_schema=local_endpoint_schema(prefill),
                errors=errors,
            )

        return self.async_show_form(
            step_id="credentials",
            data_schema=openai_key_schema(
                openai_opts,
                self._settings,
                provider_id_key=CONF_TTS_OPENAI_PROVIDER_ID,
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure model, voice, and advanced TTS options."""
        errors: dict[str, str] = {}
        current = current_subentry(self, SUBENTRY_TYPE_TTS_PROVIDER)
        model_data = dict(self._model)
        provider_type = self._provider_type or "openai"
        if provider_type == "local":
            recommended_model = RECOMMENDED_LOCAL_TTS_MODEL
            recommended_voice = RECOMMENDED_LOCAL_TTS_VOICE
        else:
            recommended_model = RECOMMENDED_OPENAI_TTS_MODEL
            recommended_voice = RECOMMENDED_OPENAI_TTS_VOICE

        if user_input is not None:
            model_name = user_input.get(CONF_TTS_MODEL_NAME)
            if not model_name:
                errors["base"] = "invalid_model"
            else:
                model_data[CONF_TTS_MODEL_NAME] = model_name
                voice = str(user_input.get(CONF_TTS_VOICE) or "").strip()
                if provider_type == "openai":
                    # The API's voice enum is lowercase while the advertised
                    # labels are display-cased ("Nova"); accept either.
                    voice = voice.lower()
                model_data[CONF_TTS_VOICE] = voice or recommended_voice
                model_data[CONF_TTS_SPEED] = _coerce_speed(
                    user_input.get(CONF_TTS_SPEED)
                )
                model_data[CONF_TTS_INSTRUCTIONS] = (
                    user_input.get(CONF_TTS_INSTRUCTIONS) or None
                )

            if not errors:
                payload = {
                    "provider_type": provider_type,
                    "name": self._name
                    or ProviderNames.get(provider_type, _FALLBACK_NAME),
                    "settings": self._settings,
                    "model": model_data,
                }
                if current is None:
                    if self.source not in (SOURCE_USER, SOURCE_RECONFIGURE):
                        return self.async_abort(reason="no_existing_subentry")
                    if self.source == SOURCE_RECONFIGURE:
                        self._source = SOURCE_USER
                        self.context["source"] = SOURCE_USER
                    self._schedule_reload()
                    return self.async_create_entry(title=payload["name"], data=payload)
                self._schedule_reload()
                return self.async_update_and_abort(
                    self._get_entry(),
                    current,
                    data=payload,
                    title=payload["name"],
                )

        if provider_type == "local":
            model_options = [
                SelectOptionDict(label=recommended_model, value=recommended_model)
            ]
            allow_custom_model = True
        else:
            model_options = [
                SelectOptionDict(label=model, value=model)
                for model in get_args(TTS_MODEL_OPENAI_SUPPORTED)
            ]
            allow_custom_model = False

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_TTS_MODEL_NAME,
                    default=model_data.get(CONF_TTS_MODEL_NAME, recommended_model),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=allow_custom_model,
                    )
                ),
                vol.Optional(
                    CONF_TTS_VOICE,
                    description={
                        "suggested_value": model_data.get(CONF_TTS_VOICE)
                        or recommended_voice
                    },
                    default=model_data.get(CONF_TTS_VOICE) or recommended_voice,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    CONF_TTS_SPEED,
                    description={
                        "suggested_value": model_data.get(
                            CONF_TTS_SPEED, TTS_SPEED_DEFAULT
                        )
                    },
                    default=model_data.get(CONF_TTS_SPEED, TTS_SPEED_DEFAULT),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=TTS_SPEED_MIN, max=TTS_SPEED_MAX, step=0.05
                    )
                ),
                vol.Optional(
                    CONF_TTS_INSTRUCTIONS,
                    description={
                        "suggested_value": model_data.get(CONF_TTS_INSTRUCTIONS)
                    },
                    default=model_data.get(CONF_TTS_INSTRUCTIONS) or "",
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            }
        )

        return self.async_show_form(step_id="model", data_schema=schema, errors=errors)
