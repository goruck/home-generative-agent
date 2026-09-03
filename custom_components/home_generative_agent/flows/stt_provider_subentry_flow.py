"""Speech-to-text provider config subentry flow."""

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
    BooleanSelector,
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
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL_NAME,
    CONF_STT_OPENAI_PROVIDER_ID,
    CONF_STT_PROMPT,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TEMPERATURE,
    CONF_STT_TRANSLATE,
    RECOMMENDED_LOCAL_STT_MODEL,
    RECOMMENDED_OPENAI_STT_MODEL,
    STT_MODEL_OPENAI_SUPPORTED,
    STT_RESPONSE_FORMATS,
    SUBENTRY_TYPE_STT_PROVIDER,
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
    "openai": "STT - OpenAI",
    "local": "STT - Local",
}
_FALLBACK_NAME = "STT Provider"


class SttProviderSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for STT provider subentries."""

    def __init__(self) -> None:
        """Initialize the STT provider flow."""
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
        """Entry point for STT provider setup or reconfigure."""
        current = current_subentry(self, SUBENTRY_TYPE_STT_PROVIDER)
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
                # Switching provider types must not carry state across.
                # Settings: the stored OpenAI key would otherwise prefill the
                # local server's optional key field and be sent to a plaintext
                # LAN endpoint (the model-provider flow guards the same leak).
                # Model: the other type's model ID is invalid or a silent 404
                # for the new one.
                self._settings = {}
                self._model = {}
            self._name = resolve_provider_name(
                user_input.get("name"),
                provider_type,
                provider_names=ProviderNames,
                stale_name=self._name if type_changed else None,
                fallback=_FALLBACK_NAME,
            )
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
                    self.hass, user_input, provider_id_key=CONF_STT_OPENAI_PROVIDER_ID
                )
            elif provider_type == "openai":
                settings, error = await build_openai_key_settings(
                    self.hass,
                    openai_opts,
                    user_input,
                    provider_id_key=CONF_STT_OPENAI_PROVIDER_ID,
                )
            else:
                # Unreachable today (the provider select admits only the two
                # types above, custom_value=False) — kept so a future type
                # added to the select but not wired here fails visibly
                # instead of storing an empty credentials payload.
                settings, error = {}, "not_supported"
            if error:
                errors["base"] = error
            else:
                self._settings = settings
                return await self.async_step_model()

        if provider_type == "local":
            # On a validation error, redisplay what was just typed — not the
            # stored settings — so a retry against a booting server does not
            # demand retyping the URL, and a corrected typo does not silently
            # revert to the saved value.
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
                provider_id_key=CONF_STT_OPENAI_PROVIDER_ID,
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure model and advanced STT options."""
        errors: dict[str, str] = {}
        current = current_subentry(self, SUBENTRY_TYPE_STT_PROVIDER)
        model_data = dict(self._model)

        if user_input is not None:
            model_name = user_input.get(CONF_STT_MODEL_NAME)
            if not model_name:
                errors["base"] = "invalid_model"
            else:
                model_data[CONF_STT_MODEL_NAME] = model_name
                model_data[CONF_STT_LANGUAGE] = (
                    user_input.get(CONF_STT_LANGUAGE) or None
                )
                model_data[CONF_STT_PROMPT] = user_input.get(CONF_STT_PROMPT) or None
                model_data[CONF_STT_TEMPERATURE] = user_input.get(CONF_STT_TEMPERATURE)
                model_data[CONF_STT_TRANSLATE] = bool(
                    user_input.get(CONF_STT_TRANSLATE)
                )
                model_data[CONF_STT_RESPONSE_FORMAT] = user_input.get(
                    CONF_STT_RESPONSE_FORMAT
                )

            if not errors:
                payload = {
                    "provider_type": self._provider_type or "openai",
                    "name": self._name
                    or ProviderNames.get(
                        self._provider_type or "openai", _FALLBACK_NAME
                    ),
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

        provider_type = self._provider_type or "openai"
        if provider_type == "local":
            model_options = [
                SelectOptionDict(
                    label=RECOMMENDED_LOCAL_STT_MODEL,
                    value=RECOMMENDED_LOCAL_STT_MODEL,
                )
            ]
            recommended_model = RECOMMENDED_LOCAL_STT_MODEL
            allow_custom_model = True
        else:
            model_options = [
                SelectOptionDict(label=model, value=model)
                for model in get_args(STT_MODEL_OPENAI_SUPPORTED)
            ]
            recommended_model = RECOMMENDED_OPENAI_STT_MODEL
            allow_custom_model = False

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_STT_MODEL_NAME,
                    default=model_data.get(CONF_STT_MODEL_NAME, recommended_model),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=allow_custom_model,
                    )
                ),
                vol.Optional(
                    CONF_STT_LANGUAGE,
                    description={"suggested_value": model_data.get(CONF_STT_LANGUAGE)},
                    default=model_data.get(CONF_STT_LANGUAGE) or "",
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    CONF_STT_PROMPT,
                    description={"suggested_value": model_data.get(CONF_STT_PROMPT)},
                    default=model_data.get(CONF_STT_PROMPT) or "",
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    CONF_STT_TEMPERATURE,
                    description={
                        "suggested_value": model_data.get(CONF_STT_TEMPERATURE)
                    },
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.1)),
                vol.Optional(
                    CONF_STT_TRANSLATE,
                    description={"suggested_value": model_data.get(CONF_STT_TRANSLATE)},
                    default=bool(model_data.get(CONF_STT_TRANSLATE)),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_STT_RESPONSE_FORMAT,
                    description={
                        "suggested_value": model_data.get(CONF_STT_RESPONSE_FORMAT)
                    },
                    default=model_data.get(CONF_STT_RESPONSE_FORMAT) or "text",
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=fmt, value=fmt)
                            for fmt in STT_RESPONSE_FORMATS
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="model", data_schema=schema, errors=errors)
