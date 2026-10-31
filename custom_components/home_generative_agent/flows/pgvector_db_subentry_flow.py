"""Database Config Flow."""

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
)
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    ObjectSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import (  # noqa: TID252
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    RECOMMENDED_DB_HOST,
    RECOMMENDED_DB_NAME,
    RECOMMENDED_DB_PARAMS,
    RECOMMENDED_DB_PASSWORD,
    RECOMMENDED_DB_PORT,
    RECOMMENDED_DB_USERNAME,
    SUBENTRY_TYPE_DATABASE,
)
from ..core.db_utils import build_postgres_uri  # noqa: TID252
from ..core.lifecycle import DATABASE_DOWN_STATES  # noqa: TID252
from ..core.utils import (  # noqa: TID252
    CannotConnectError,
    InvalidAuthError,
    validate_db_uri,
)

if TYPE_CHECKING:
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)


class PgVectorDbSubentryFlow(ConfigSubentryFlow):
    """
    Config flow handler for pgvector (Postgres) database subentries.

    This flow presents a form for users to input database connection details,
    validates the URI, and either creates a new subentry or updates an existing
    one.

    """

    def _async_update_and_reload_if_down(
        self, subentry: ConfigSubentry, data: dict[str, Any]
    ) -> SubentryFlowResult:
        """Save the new settings; reload an entry that is not running."""
        entry = self._get_entry()
        result = self.async_update_and_abort(entry, subentry, data=data)
        # An entry waiting to retry, or failed, on these settings has no
        # update listener to reload it; reload now so the fix applies at once
        # instead of at the next retry, up to 10 minutes away (#748).
        if entry.state in DATABASE_DOWN_STATES:
            self.hass.config_entries.async_schedule_reload(entry.entry_id)
        return result

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """
        Show and handle the 'set_options' step.

        When called without `user_input`, this method builds the form schema and
        suggested defaults (either recommended values or existing subentry
        values). When `user_input` is provided, it builds a Postgres URI from the
        input and validates the connection. On success, it creates or updates
        the subentry; on failure, it returns the form with errors.

        Args:
            user_input: Mapping of form keys to values supplied by the user, or
                None when the form is first shown.

        Returns:
            A SubentryFlowResult which is either a form to display or an entry
            creation/update result.

        """
        errors: dict[str, str] = {}

        entry = self._get_entry()
        current_subentry = next(
            (
                v
                for v in entry.subentries.values()
                if v.subentry_type == SUBENTRY_TYPE_DATABASE
            ),
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
            except (KeyError, ValueError):
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
                return self._async_update_and_reload_if_down(
                    current_subentry, user_input
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
