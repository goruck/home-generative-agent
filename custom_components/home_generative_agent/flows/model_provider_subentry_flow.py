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
    CONF_GEMINI_API_KEY,
    CONF_OLLAMA_URL,
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
    RECOMMENDED_OLLAMA_SUMMARIZATION_URL,
    RECOMMENDED_OLLAMA_URL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    RECOMMENDED_OLLAMA_VLM_URL,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_EMBEDDING_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
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
            if subentry.subentry_type == "model_provider"
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _capabilities_from_settings(data: dict[str, Any]) -> list[str]:
    """Infer capability categories from model fields."""
    caps = [
        cat
        for cat in ("chat", "vlm", "summarization", "embedding")
        if data.get(f"{cat}_model")
    ]
    return caps or ["chat"]


class ModelProviderSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for model provider subentries."""

    def __init__(self) -> None:
        """Initialize state."""
        self._provider_type: str | None = None
        self._name: str | None = None

    def _schedule_reload(self) -> None:
        """Reload the parent entry to apply subentry changes."""
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _default_provider_from_entry(self) -> str:
        """Choose a reasonable default based on configured options."""
        entry = self._get_entry()
        options = {**entry.data, **entry.options}
        if options.get(CONF_API_KEY):
            return "openai"
        if options.get(CONF_GEMINI_API_KEY):
            return "gemini"
        if options.get(CONF_OLLAMA_URL):
            return "ollama"
        return "ollama"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select provider type and friendly name."""
        current = _current_subentry(self)
        default_provider = self._provider_type or (
            current.data.get("provider_type")
            if current
            else self._default_provider_from_entry()
        )
        default_name = self._name or (current.title if current else None)
        provider_key = default_provider or "ollama"
        default_name = default_name or ProviderNames.get(provider_key, "Model Provider")

        if user_input is not None:
            self._provider_type = user_input["provider_type"]
            provider_type = self._provider_type or "ollama"
            self._name = user_input.get("name") or ProviderNames.get(
                provider_type, "Model Provider"
            )
            return await self.async_step_config()

        schema = vol.Schema(
            {
                vol.Required(
                    "provider_type",
                    default=default_provider,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="Ollama", value="ollama"),
                            SelectOptionDict(label="OpenAI", value="openai"),
                            SelectOptionDict(label="Gemini", value="gemini"),
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
        return self.async_show_form(step_id="user", data_schema=schema)

    async_step_reconfigure = async_step_user

    def _ollama_defaults(self) -> dict[str, Any]:
        return {
            "name": ProviderNames["ollama"],
            "base_url": RECOMMENDED_OLLAMA_URL,
            "chat_url": RECOMMENDED_OLLAMA_URL,
            "vlm_url": RECOMMENDED_OLLAMA_VLM_URL,
            "summarization_url": RECOMMENDED_OLLAMA_SUMMARIZATION_URL,
            "chat_model": RECOMMENDED_OLLAMA_CHAT_MODEL,
            "vlm_model": RECOMMENDED_OLLAMA_VLM,
            "summarization_model": RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
            "embedding_model": RECOMMENDED_OLLAMA_EMBEDDING_MODEL,
            "chat_keepalive": RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
            "vlm_keepalive": RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
            "summarization_keepalive": RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
            "chat_context": RECOMMENDED_OLLAMA_CONTEXT_SIZE,
            "vlm_context": RECOMMENDED_OLLAMA_CONTEXT_SIZE,
            "summarization_context": RECOMMENDED_OLLAMA_CONTEXT_SIZE,
            "reasoning": None,
        }

    def _openai_defaults(self) -> dict[str, Any]:
        return {
            "name": ProviderNames["openai"],
            "api_key": "",
            "chat_model": RECOMMENDED_OPENAI_CHAT_MODEL,
            "vlm_model": RECOMMENDED_OPENAI_VLM,
            "summarization_model": RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
            "embedding_model": RECOMMENDED_OPENAI_EMBEDDING_MODEL,
        }

    def _gemini_defaults(self) -> dict[str, Any]:
        return {
            "name": ProviderNames["gemini"],
            "api_key": "",
            "chat_model": RECOMMENDED_GEMINI_CHAT_MODEL,
            "vlm_model": RECOMMENDED_GEMINI_VLM,
            "summarization_model": RECOMMENDED_GEMINI_SUMMARIZATION_MODEL,
            "embedding_model": RECOMMENDED_GEMINI_EMBEDDING_MODEL,
        }

    def _defaults_for_provider(self, provider_type: str) -> dict[str, Any]:
        current = _current_subentry(self)
        if current:
            settings = current.data.get("settings") or {}
            if settings:
                return dict(settings)

        if provider_type == "openai":
            return self._openai_defaults()
        if provider_type == "gemini":
            return self._gemini_defaults()
        return self._ollama_defaults()

    def _ollama_schema(self, defaults: dict[str, Any]) -> vol.Schema:
        reasoning_value = defaults.get("reasoning")
        if reasoning_value is True:
            reasoning_default = "true"
        elif reasoning_value is False or reasoning_value is None:
            reasoning_default = "false"
        else:
            reasoning_default = str(reasoning_value)
        return vol.Schema(
            {
                vol.Required(
                    "base_url",
                    description={"suggested_value": defaults.get("base_url")},
                    default=defaults.get("base_url"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "chat_url",
                    description={"suggested_value": defaults.get("chat_url")},
                    default=defaults.get("chat_url"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "vlm_url",
                    description={"suggested_value": defaults.get("vlm_url")},
                    default=defaults.get("vlm_url"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "summarization_url",
                    description={"suggested_value": defaults.get("summarization_url")},
                    default=defaults.get("summarization_url"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Required(
                    "chat_model",
                    description={"suggested_value": defaults.get("chat_model")},
                    default=defaults.get("chat_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Required(
                    "vlm_model",
                    description={"suggested_value": defaults.get("vlm_model")},
                    default=defaults.get("vlm_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Required(
                    "summarization_model",
                    description={
                        "suggested_value": defaults.get("summarization_model")
                    },
                    default=defaults.get("summarization_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "embedding_model",
                    description={"suggested_value": defaults.get("embedding_model")},
                    default=defaults.get("embedding_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "chat_keepalive",
                    description={"suggested_value": defaults.get("chat_keepalive")},
                    default=defaults.get("chat_keepalive"),
                ): NumberSelector(NumberSelectorConfig(step=1)),
                vol.Optional(
                    "vlm_keepalive",
                    description={"suggested_value": defaults.get("vlm_keepalive")},
                    default=defaults.get("vlm_keepalive"),
                ): NumberSelector(NumberSelectorConfig(step=1)),
                vol.Optional(
                    "summarization_keepalive",
                    description={
                        "suggested_value": defaults.get("summarization_keepalive")
                    },
                    default=defaults.get("summarization_keepalive"),
                ): NumberSelector(NumberSelectorConfig(step=1)),
                vol.Optional(
                    "chat_context",
                    description={"suggested_value": defaults.get("chat_context")},
                    default=defaults.get("chat_context"),
                ): NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1)),
                vol.Optional(
                    "vlm_context",
                    description={"suggested_value": defaults.get("vlm_context")},
                    default=defaults.get("vlm_context"),
                ): NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1)),
                vol.Optional(
                    "summarization_context",
                    description={
                        "suggested_value": defaults.get("summarization_context")
                    },
                    default=defaults.get("summarization_context"),
                ): NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1)),
                vol.Optional(
                    "reasoning",
                    description={"suggested_value": reasoning_default},
                    default=reasoning_default,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="Off", value="false"),
                            SelectOptionDict(label="On", value="true"),
                            SelectOptionDict(label="GPT-OSS effort", value="low"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=True,
                    )
                ),
            }
        )

    def _openai_schema(self, defaults: dict[str, Any]) -> vol.Schema:
        return vol.Schema(
            {
                vol.Required(
                    CONF_API_KEY,
                    description={"suggested_value": defaults.get("api_key")},
                    default=defaults.get("api_key"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Required(
                    "chat_model",
                    description={"suggested_value": defaults.get("chat_model")},
                    default=defaults.get("chat_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "vlm_model",
                    description={"suggested_value": defaults.get("vlm_model")},
                    default=defaults.get("vlm_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "summarization_model",
                    description={
                        "suggested_value": defaults.get("summarization_model")
                    },
                    default=defaults.get("summarization_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "embedding_model",
                    description={"suggested_value": defaults.get("embedding_model")},
                    default=defaults.get("embedding_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            }
        )

    def _gemini_schema(self, defaults: dict[str, Any]) -> vol.Schema:
        return vol.Schema(
            {
                vol.Required(
                    CONF_GEMINI_API_KEY,
                    description={"suggested_value": defaults.get("api_key")},
                    default=defaults.get("api_key"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Required(
                    "chat_model",
                    description={"suggested_value": defaults.get("chat_model")},
                    default=defaults.get("chat_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "vlm_model",
                    description={"suggested_value": defaults.get("vlm_model")},
                    default=defaults.get("vlm_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "summarization_model",
                    description={
                        "suggested_value": defaults.get("summarization_model")
                    },
                    default=defaults.get("summarization_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                vol.Optional(
                    "embedding_model",
                    description={"suggested_value": defaults.get("embedding_model")},
                    default=defaults.get("embedding_model"),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            }
        )

    def _schema_for_provider(
        self, provider_type: str, defaults: dict[str, Any]
    ) -> vol.Schema:
        if provider_type == "openai":
            return self._openai_schema(defaults)
        if provider_type == "gemini":
            return self._gemini_schema(defaults)
        return self._ollama_schema(defaults)

    async def _validate_ollama(
        self, data: dict[str, Any], errors: dict[str, str]
    ) -> dict[str, Any]:
        """Normalize Ollama URLs and validate connectivity."""
        normalized = dict(data)
        for field in ("base_url", "chat_url", "vlm_url", "summarization_url"):
            if normalized.get(field):
                normalized[field] = ensure_http_url(str(normalized[field]))
        try:
            await validate_ollama_url(self.hass, normalized["base_url"])
        except CannotConnectError:
            errors["base"] = "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating Ollama URL")
            errors["base"] = "unknown"
        return normalized

    async def _validate_openai(
        self, data: dict[str, Any], errors: dict[str, str]
    ) -> dict[str, Any]:
        """Validate OpenAI API key when present."""
        if not data.get(CONF_API_KEY):
            errors["base"] = "invalid_auth"
            return data
        try:
            await validate_openai_key(self.hass, data[CONF_API_KEY])
        except InvalidAuthError:
            LOGGER.warning("OpenAI key validation failed; saving without verification.")
        except CannotConnectError:
            errors["base"] = "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating OpenAI key")
            errors["base"] = "unknown"
        return data

    async def _validate_gemini(
        self, data: dict[str, Any], errors: dict[str, str]
    ) -> dict[str, Any]:
        """Validate Gemini API key when present."""
        if not data.get(CONF_GEMINI_API_KEY):
            errors["base"] = "invalid_auth"
            return data
        try:
            await validate_gemini_key(self.hass, data[CONF_GEMINI_API_KEY])
        except InvalidAuthError:
            errors["base"] = "invalid_auth"
        except CannotConnectError:
            errors["base"] = "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating Gemini key")
            errors["base"] = "unknown"
        return data

    async def async_step_config(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure provider-specific settings."""
        errors: dict[str, str] = {}
        provider_type = self._provider_type or "ollama"
        defaults = self._defaults_for_provider(provider_type)
        name = (
            self._name
            or defaults.get("name")
            or ProviderNames.get(provider_type, "Model Provider")
        )

        if user_input is not None:
            settings = dict(defaults)
            settings.update(user_input)
            if "reasoning" in settings:
                raw_reasoning = settings.get("reasoning")
                if raw_reasoning in ("false", "", None):
                    settings["reasoning"] = None
                elif raw_reasoning == "true":
                    settings["reasoning"] = True
                else:
                    settings["reasoning"] = raw_reasoning
            if provider_type == "ollama":
                settings = await self._validate_ollama(settings, errors)
            elif provider_type == "openai":
                settings = await self._validate_openai(settings, errors)
            elif provider_type == "gemini":
                settings = await self._validate_gemini(settings, errors)

            if not errors:
                caps = _capabilities_from_settings(settings)
                payload = {
                    "provider_type": provider_type,
                    "name": name,
                    "capabilities": caps,
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
                    return self.async_create_entry(title=name, data=payload)
                self._schedule_reload()
                return self.async_update_and_abort(
                    self._get_entry(),
                    current,
                    data=payload,
                    title=name,
                )

        schema = self._schema_for_provider(provider_type, defaults)
        return self.async_show_form(
            step_id="config",
            data_schema=schema,
            errors=errors,
            description_placeholders={"name": name},
        )
