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
)
from langchain_openai import ChatOpenAI

from .const import (
    CONF_CHAT_MODEL,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TOP_P,
    CONF_VLM,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TOP_P,
    RECOMMENDED_VLM,
)

if TYPE_CHECKING:
    from types import MappingProxyType

    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)
RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = ChatOpenAI(
        api_key=data[CONF_API_KEY], async_client=get_async_client(hass)
    )
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

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        # TODO: improve error handling
        except Exception:
            _LOGGER.exception("Unexpected exception")
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
    """Config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            }

        schema = config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )

class CannotConnectError(HomeAssistantError):
    """Error to indicate we cannot connect."""

class InvalidAuthError(HomeAssistantError):
    """Error to indicate there is invalid auth."""

def config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> dict:
    """Return a schema for completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_VLM,
                description={"suggested_value": options.get(RECOMMENDED_VLM)},
                default=RECOMMENDED_VLM,
            ): str,
            vol.Optional(
                CONF_VISION_MODEL_TEMPERATURE,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_VISION_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_VISION_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_VISION_MODEL_TOP_P,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_VISION_MODEL_TOP_P)
                },
                default=RECOMMENDED_VISION_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_SUMMARIZATION_MODEL_TOP_P)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_EMBEDDING_MODEL,
                description={
                    "suggested_value":
                    options.get(RECOMMENDED_EMBEDDING_MODEL)
                },
                default=RECOMMENDED_EMBEDDING_MODEL,
            ): str,
        }
    )

    return schema
