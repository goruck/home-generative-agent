"""Model provider config subentry flow."""

from __future__ import annotations

import logging
from typing import Any

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
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import (  # noqa: TID252
    CONF_GEMINI_API_KEY,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from ..core.utils import (  # noqa: TID252
    CannotConnectError,
    InvalidAuthError,
    ensure_http_url,
    validate_gemini_key,
    validate_ollama_url,
    validate_openai_key,
)

LOGGER = logging.getLogger(__name__)

ProviderNames = {
    "ollama": "Primary Ollama",
    "openai": "Cloud-LLM OpenAI",
    "gemini": "Cloud-LLM Gemini",
}

ProviderCapabilities = {
    "ollama": {"chat", "vlm", "summarization", "embedding"},
    "openai": {"chat", "vlm", "summarization", "embedding"},
    "gemini": {"chat", "vlm", "summarization", "embedding"},
}


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the subentry currently being edited, if any."""
    entry = flow._get_entry()  # noqa: SLF001
    subentry_id = getattr(flow, "_subentry_id", None)
    if not subentry_id:
        subentry_id = flow.context.get("subentry_id")
    if subentry_id:
        return entry.subentries.get(subentry_id)
    if flow.source == SOURCE_RECONFIGURE:
        matches = [
            subentry
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
        ]
        if len(matches) == 1:
            return matches[0]
    return None


class ModelProviderSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for model provider subentries."""

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._deployment: str | None = None
        self._provider_type: str | None = None
        self._name: str | None = None

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _provider_options(self) -> list[SelectOptionDict]:
        opts: list[SelectOptionDict] = []
        if self._deployment == "edge":
            opts.append(SelectOptionDict(label="Ollama", value="ollama"))
        if self._deployment == "cloud":
            opts.extend(
                [
                    SelectOptionDict(label="OpenAI", value="openai"),
                    SelectOptionDict(label="Gemini", value="gemini"),
                ]
            )
        if not opts:
            opts = [
                SelectOptionDict(label="Ollama", value="ollama"),
                SelectOptionDict(label="OpenAI", value="openai"),
                SelectOptionDict(label="Gemini", value="gemini"),
            ]
        return opts

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for model provider setup or reconfigure."""
        current = _current_subentry(self)
        if current:
            self._provider_type = current.data.get("provider_type")
            provider_type = self._provider_type or "ollama"
            self._name = current.title or ProviderNames.get(provider_type, "Provider")
            self._deployment = current.data.get("deployment")
            return await self.async_step_provider(user_input)
        return await self.async_step_deployment(user_input)

    async_step_reconfigure = async_step_user

    async def async_step_deployment(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select deployment type (edge or cloud)."""
        if user_input is not None:
            self._deployment = user_input.get("deployment")
            return await self.async_step_provider()

        schema = vol.Schema(
            {
                vol.Required(
                    "deployment",
                    default=self._deployment or "edge",
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="Edge", value="edge"),
                            SelectOptionDict(label="Cloud", value="cloud"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                )
            }
        )
        return self.async_show_form(step_id="deployment", data_schema=schema)

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select provider type and name."""
        if user_input is not None:
            self._provider_type = user_input["provider_type"]
            provider_type = self._provider_type or "ollama"
            self._name = user_input.get("name") or ProviderNames.get(
                provider_type, "Model Provider"
            )
            return await self.async_step_settings()

        provider_type = self._provider_type or "ollama"
        default_name = self._name or ProviderNames.get(provider_type, "Model Provider")
        schema = vol.Schema(
            {
                vol.Required(
                    "provider_type",
                    default=provider_type,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=self._provider_options(),
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
        return self.async_show_form(step_id="provider", data_schema=schema)

    async def async_step_settings(  # noqa: PLR0912, PLR0915
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure provider-specific settings."""
        provider_type = self._provider_type or "ollama"
        errors: dict[str, str] = {}

        if user_input is not None:
            settings: dict[str, Any] = {}
            if provider_type == "ollama":
                base_url = user_input.get("base_url")
                if not base_url:
                    errors["base"] = "cannot_connect"
                else:
                    settings["base_url"] = ensure_http_url(str(base_url))
                    try:
                        await validate_ollama_url(self.hass, settings["base_url"])
                    except CannotConnectError:
                        errors["base"] = "cannot_connect"
                    except Exception:
                        LOGGER.exception("Unexpected exception validating Ollama URL")
                        errors["base"] = "unknown"
            elif provider_type == "openai":
                api_key = user_input.get(CONF_API_KEY)
                if not api_key:
                    errors["base"] = "invalid_auth"
                else:
                    try:
                        await validate_openai_key(self.hass, api_key)
                    except InvalidAuthError:
                        errors["base"] = "invalid_auth"
                    except CannotConnectError:
                        errors["base"] = "cannot_connect"
                    except Exception:
                        LOGGER.exception("Unexpected exception validating OpenAI key")
                        errors["base"] = "unknown"
                    settings["api_key"] = api_key
            elif provider_type == "gemini":
                api_key = user_input.get(CONF_GEMINI_API_KEY)
                if not api_key:
                    errors["base"] = "invalid_auth"
                else:
                    try:
                        await validate_gemini_key(self.hass, api_key)
                    except InvalidAuthError:
                        errors["base"] = "invalid_auth"
                    except CannotConnectError:
                        errors["base"] = "cannot_connect"
                    except Exception:
                        LOGGER.exception("Unexpected exception validating Gemini key")
                        errors["base"] = "unknown"
                    settings["api_key"] = api_key

            if not errors:
                caps = ProviderCapabilities.get(provider_type, {"chat"})
                payload = {
                    "provider_type": provider_type,
                    "name": self._name or ProviderNames.get(provider_type, "Provider"),
                    "deployment": self._deployment,
                    "capabilities": sorted(caps),
                    "settings": settings,
                }
                current = _current_subentry(self)
                if current is None:
                    if self.source not in (SOURCE_USER, SOURCE_RECONFIGURE):
                        return self.async_abort(reason="no_existing_subentry")
                    if self.source == SOURCE_RECONFIGURE:
                        self._source = SOURCE_USER
                        self.context["source"] = SOURCE_USER
                    self._schedule_reload()
                    result = self.async_create_entry(
                        title=payload["name"], data=payload
                    )
                else:
                    self._schedule_reload()
                    result = self.async_update_and_abort(
                        self._get_entry(),
                        current,
                        data=payload,
                        title=payload["name"],
                    )

                return result

        if provider_type == "ollama":
            schema = vol.Schema(
                {
                    vol.Required("base_url"): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.TEXT)
                    )
                }
            )
        elif provider_type == "openai":
            schema = vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            )
        else:
            schema = vol.Schema(
                {
                    vol.Required(CONF_GEMINI_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            )

        return self.async_show_form(
            step_id="settings", data_schema=schema, errors=errors
        )
