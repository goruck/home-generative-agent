"""Speech-to-text provider config subentry flow."""

from __future__ import annotations

import logging
from typing import Any, get_args

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY
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
    RECOMMENDED_OPENAI_STT_MODEL,
    STT_MODEL_OPENAI_SUPPORTED,
    STT_RESPONSE_FORMATS,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_STT_PROVIDER,
)
from ..core.utils import (  # noqa: TID252
    CannotConnectError,
    InvalidAuthError,
    validate_openai_key,
)

LOGGER = logging.getLogger(__name__)

ProviderNames = {
    "openai": "STT - OpenAI",
    "local": "STT - Local",
}


def _get_entry_from_flow(flow: ConfigSubentryFlow) -> Any:
    """Return the config entry for a subentry flow."""
    return flow._get_entry()  # type: ignore[attr-defined]  # noqa: SLF001


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the subentry currently being edited, if any."""
    entry = _get_entry_from_flow(flow)
    subentry_id = getattr(flow, "_subentry_id", None)
    if not subentry_id:
        subentry_id = flow.context.get("subentry_id")
    if subentry_id:
        return entry.subentries.get(subentry_id)
    if flow.source == SOURCE_RECONFIGURE:
        matches = [
            subentry
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_STT_PROVIDER
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _openai_provider_options(flow: ConfigSubentryFlow) -> list[SelectOptionDict]:
    """Return OpenAI model provider subentries for reuse."""
    entry = _get_entry_from_flow(flow)
    options: list[SelectOptionDict] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
            continue
        if subentry.data.get("provider_type") != "openai":
            continue
        options.append(
            SelectOptionDict(
                label=subentry.title or subentry.subentry_id,
                value=subentry.subentry_id,
            )
        )
    return options


async def _build_openai_settings(
    hass: Any, openai_opts: list[SelectOptionDict], user_input: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Build OpenAI settings and return error key if any."""
    settings: dict[str, Any] = {}
    api_key = user_input.get(CONF_API_KEY)
    provider_id = user_input.get(CONF_STT_OPENAI_PROVIDER_ID)

    if openai_opts and provider_id and provider_id != "none":
        settings[CONF_STT_OPENAI_PROVIDER_ID] = provider_id
        settings[CONF_API_KEY] = None
        return settings, None

    if not api_key:
        return {}, "invalid_auth"

    try:
        await validate_openai_key(hass, api_key)
    except InvalidAuthError:
        return {}, "invalid_auth"
    except CannotConnectError:
        return {}, "cannot_connect"
    except Exception:
        LOGGER.exception("Unexpected exception validating OpenAI key")
        return {}, "unknown"

    settings[CONF_API_KEY] = api_key
    settings[CONF_STT_OPENAI_PROVIDER_ID] = None
    return settings, None


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
        current = _current_subentry(self)
        if current:
            self._provider_type = current.data.get("provider_type")
            self._name = current.data.get("name") or ProviderNames.get(
                self._provider_type or "openai", "STT Provider"
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
            if provider_type == "local":
                errors["base"] = "not_supported"
            else:
                self._provider_type = provider_type
                self._name = user_input.get("name") or ProviderNames.get(
                    provider_type, "STT Provider"
                )
                return await self.async_step_credentials()

        provider_type = self._provider_type or "openai"
        default_name = self._name or ProviderNames.get(provider_type, "STT Provider")
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
                                label="Local (coming soon)", value="local"
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
        openai_opts = _openai_provider_options(self)

        if user_input is not None:
            if provider_type != "openai":
                errors["base"] = "not_supported"
            else:
                settings, error = await _build_openai_settings(
                    self.hass, openai_opts, user_input
                )
                if error:
                    errors["base"] = error
                else:
                    self._settings = settings
                    return await self.async_step_model()

        schema_dict: dict[Any, Any] = {}
        if openai_opts:
            reuse_opts = [
                SelectOptionDict(label="Use a separate key", value="none"),
                *openai_opts,
            ]
            stored_provider_id = self._settings.get(CONF_STT_OPENAI_PROVIDER_ID)
            stored_api_key = self._settings.get(CONF_API_KEY)
            if stored_provider_id is None and stored_api_key:
                default_id = "none"
            else:
                default_id = stored_provider_id or reuse_opts[1]["value"]
            schema_dict[
                vol.Required(
                    CONF_STT_OPENAI_PROVIDER_ID,
                    default=default_id,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=reuse_opts,
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            )

        schema_dict[
            vol.Optional(
                CONF_API_KEY,
                description={"suggested_value": self._settings.get(CONF_API_KEY)},
                default=self._settings.get(CONF_API_KEY) or "",
            )
        ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

        return self.async_show_form(
            step_id="credentials", data_schema=vol.Schema(schema_dict), errors=errors
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure model and advanced STT options."""
        errors: dict[str, str] = {}
        current = _current_subentry(self)
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
                    "name": self._name or ProviderNames.get("openai", "STT Provider"),
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

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_STT_MODEL_NAME,
                    default=model_data.get(
                        CONF_STT_MODEL_NAME, RECOMMENDED_OPENAI_STT_MODEL
                    ),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=model, value=model)
                            for model in get_args(STT_MODEL_OPENAI_SUPPORTED)
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
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
