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
    CONF_CHAT_MODEL_LOCATION,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL,
    CONF_EDGE_CHAT_MODEL_TEMPERATURE,
    CONF_EDGE_CHAT_MODEL_TOP_P,
    CONF_EMBEDDING_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TOP_P,
    CONF_VLM,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL_LOCATION,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL,
    RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
    RECOMMENDED_EMBEDDING_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TOP_P,
    RECOMMENDED_VLM,
)

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

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
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
    """Config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
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
                CONF_VIDEO_ANALYZER_MODE: user_input[CONF_VIDEO_ANALYZER_MODE]
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
) -> VolDictType:
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

    video_analyzer_mode: list[SelectOptionDict] = [
        SelectOptionDict(
            label="Disable",
            value="disable",
        ),
        SelectOptionDict(
            label="Notify on anomaly",
            value="notify_on_anomaly",
        ),
        SelectOptionDict(
            label="Always notify",
            value="always_notify",
        )
    ]

    schema : VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": options.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE
            ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode)),
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": options.get(CONF_RECOMMENDED)},
            default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    chat_model_location: list[SelectOptionDict] = [
        SelectOptionDict(
            label="cloud",
            value="cloud",
        ),
        SelectOptionDict(
            label="edge",
            value="edge",
        )
    ]

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL_LOCATION,
                description={"suggested_value": options.get(CONF_CHAT_MODEL_LOCATION)},
                default=RECOMMENDED_CHAT_MODEL_LOCATION
                ): SelectSelector(SelectSelectorConfig(options=chat_model_location)),
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_EDGE_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_EDGE_CHAT_MODEL)},
                default=RECOMMENDED_EDGE_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_EDGE_CHAT_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_EDGE_CHAT_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_EDGE_CHAT_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_EDGE_CHAT_MODEL_TOP_P,
                description={
                    "suggested_value": options.get(CONF_EDGE_CHAT_MODEL_TOP_P)
                },
                default=RECOMMENDED_EDGE_CHAT_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_VLM,
                description={"suggested_value": options.get(CONF_VLM)},
                default=RECOMMENDED_VLM,
            ): str,
            vol.Optional(
                CONF_VISION_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_VISION_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_VISION_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_VISION_MODEL_TOP_P,
                description={"suggested_value": options.get(CONF_VISION_MODEL_TOP_P)},
                default=RECOMMENDED_VISION_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL,
                description={"suggested_value": options.get(CONF_SUMMARIZATION_MODEL)},
                default=RECOMMENDED_SUMMARIZATION_MODEL,
            ): str,
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                description={
                    "suggested_value": options.get(CONF_SUMMARIZATION_MODEL_TEMPERATURE)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                description={
                    "suggested_value": options.get(CONF_SUMMARIZATION_MODEL_TOP_P)
                },
                default=RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_EMBEDDING_MODEL,
                description={"suggested_value": options.get(CONF_EMBEDDING_MODEL)},
                default=RECOMMENDED_EMBEDDING_MODEL,
            ): str,
        }
    )

    return schema
