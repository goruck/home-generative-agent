# ---------------------------
# Database Config Flow
# ---------------------------

from typing import TYPE_CHECKING, Any
from ..core.db_utils import build_postgres_uri
from ..core.utils import (
    CannotConnectError,
    InvalidAuthError,
    validate_db_uri,
)
import voluptuous as vol
import logging
from ..const import (
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    RECOMMENDED_DB_USERNAME,
    RECOMMENDED_DB_PASSWORD,
    RECOMMENDED_DB_HOST,
    RECOMMENDED_DB_PORT,
    RECOMMENDED_DB_NAME,
    RECOMMENDED_DB_PARAMS,
)
from homeassistant.const import (
    CONF_USERNAME,
    CONF_PASSWORD,
    CONF_HOST,
    CONF_PORT,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    ObjectSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.config_entries import ConfigSubentryFlow, SubentryFlowResult

if TYPE_CHECKING:
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

class PgVectorDbSubentryFlow(ConfigSubentryFlow):
    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:

        errors: dict[str, str] = {}

        entry = self._get_entry()
        current_subentry = next(
            (v for v in entry.subentries.values() if v.subentry_type == "database"),
            None,
        )

        if user_input is None:
            if current_subentry is None:
                options: dict[str, Any] = {
                    CONF_USERNAME: RECOMMENDED_DB_USERNAME,
                    CONF_PASSWORD: RECOMMENDED_DB_PASSWORD,
                    CONF_HOST: RECOMMENDED_DB_HOST,
                    CONF_PORT: RECOMMENDED_DB_PORT,
                    CONF_DB_NAME: RECOMMENDED_DB_NAME,
                    CONF_DB_PARAMS: RECOMMENDED_DB_PARAMS,
                }
            else:
                options = current_subentry.data.copy()
        else:
            try:
                db_uri = build_postgres_uri(user_input)
            except Exception:
                errors["base"] = "invalid_uri"
            else:
                try:
                    await validate_db_uri(self.hass, db_uri)
                except InvalidAuthError:
                    errors["base"] = "invalid_auth"
                except CannotConnectError:
                    errors["base"] = "cannot_connect"
                except Exception:
                    LOGGER.exception("Unexpected exception during database validation")
                    errors["base"] = "unknown"

            if errors:
                options = user_input
            elif current_subentry is None:
                return self.async_create_entry(title="Database", data=user_input)
            else:
                return self.async_update_and_abort(
                    self._get_entry(), current_subentry, data=user_input
                )

        schema: VolDictType = {
            vol.Required(
                CONF_USERNAME,
                description={"suggested_value": options.get(CONF_USERNAME)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Required(
                CONF_PASSWORD,
                description={"suggested_value": options.get(CONF_PASSWORD)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Required(
                CONF_HOST,
                description={"suggested_value": options.get(CONF_HOST)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_PORT,
                description={"suggested_value": options.get(CONF_PORT)},
            ): NumberSelector(NumberSelectorConfig()),
            vol.Required(
                CONF_DB_NAME,
                description={"suggested_value": options.get(CONF_DB_NAME)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_DB_PARAMS,
                description={"suggested_value": options.get(CONF_DB_PARAMS)},
            ): ObjectSelector(
                {
                    "fields": {
                        "key": {"required": True, "selector": {"text": {}}},
                        "value": {"required": True, "selector": {"text": {}}},
                    },
                    "multiple": True,
                    "label_field": "key",
                    "translation_key": "db_params",
                }
            ),
        }

        return self.async_show_form(
            step_id="set_options", data_schema=vol.Schema(schema), errors=errors
        )

    async_step_reconfigure = async_step_set_options
    async_step_user = async_step_set_options
