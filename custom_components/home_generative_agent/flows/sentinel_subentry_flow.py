"""Sentinel configuration subentry flow."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentry,
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
    CONF_EXPLAIN_ENABLED,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
    RECOMMENDED_EXPLAIN_ENABLED,
    RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
    RECOMMENDED_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
    SUBENTRY_TYPE_SENTINEL,
)
from ..core.utils import list_mobile_notify_services  # noqa: TID252


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the Sentinel subentry currently being edited, if any."""
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
            if subentry.subentry_type == SUBENTRY_TYPE_SENTINEL
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _default_payload() -> dict[str, Any]:
    """Return default Sentinel configuration payload."""
    return {
        CONF_SENTINEL_ENABLED: RECOMMENDED_SENTINEL_ENABLED,
        CONF_SENTINEL_INTERVAL_SECONDS: RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
        CONF_SENTINEL_COOLDOWN_MINUTES: RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
        CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES: (
            RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES
        ),
        CONF_SENTINEL_DISCOVERY_ENABLED: RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
        CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS: (
            RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS
        ),
        CONF_SENTINEL_DISCOVERY_MAX_RECORDS: RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
        CONF_EXPLAIN_ENABLED: RECOMMENDED_EXPLAIN_ENABLED,
    }


class SentinelSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for Sentinel subentries."""

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _schema(self, payload: dict[str, Any]) -> vol.Schema:
        """Build Sentinel form schema from payload."""
        schema: dict[Any, Any] = {
            vol.Required(
                CONF_SENTINEL_ENABLED,
                default=bool(
                    payload.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_INTERVAL_SECONDS,
                default=int(
                    payload.get(
                        CONF_SENTINEL_INTERVAL_SECONDS,
                        RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=60, max=3600, step=30)),
            vol.Required(
                CONF_SENTINEL_COOLDOWN_MINUTES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_COOLDOWN_MINUTES,
                        RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=5, max=240, step=5)),
            vol.Required(
                CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
                        RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=5, max=240, step=5)),
            vol.Required(
                CONF_SENTINEL_DISCOVERY_ENABLED,
                default=bool(
                    payload.get(
                        CONF_SENTINEL_DISCOVERY_ENABLED,
                        RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
                    )
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
                default=int(
                    payload.get(
                        CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
                        RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=300, max=86400, step=300)),
            vol.Required(
                CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
                default=int(
                    payload.get(
                        CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
                        RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=10, max=1000, step=10)),
            vol.Required(
                CONF_EXPLAIN_ENABLED,
                default=bool(
                    payload.get(CONF_EXPLAIN_ENABLED, RECOMMENDED_EXPLAIN_ENABLED)
                ),
            ): BooleanSelector(),
        }

        mobile_opts = list_mobile_notify_services(self.hass)
        notify_value = str(payload.get(CONF_NOTIFY_SERVICE, "") or "")
        if mobile_opts:
            default_notify = notify_value if notify_value in mobile_opts else ""
            schema[
                vol.Optional(
                    CONF_NOTIFY_SERVICE,
                    description={"suggested_value": notify_value},
                    default=default_notify,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=s.replace("notify.", ""), value=s)
                        for s in ["", *mobile_opts]
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            )
        else:
            schema[
                vol.Optional(
                    CONF_NOTIFY_SERVICE,
                    description={"suggested_value": notify_value},
                    default=notify_value,
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))

        return vol.Schema(schema)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Entry point for Sentinel setup/reconfigure."""
        return await self.async_step_settings(user_input)

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create or edit Sentinel configuration."""
        current = _current_subentry(self)
        payload = _default_payload()
        if current is not None:
            payload.update(dict(current.data))

        if user_input is None:
            return self.async_show_form(
                step_id="settings",
                data_schema=self._schema(payload),
            )

        data = dict(user_input)
        notify_service = str(data.get(CONF_NOTIFY_SERVICE, "") or "").strip()
        if notify_service:
            data[CONF_NOTIFY_SERVICE] = notify_service
        else:
            data.pop(CONF_NOTIFY_SERVICE, None)

        if current is None:
            if self.source not in (SOURCE_USER, SOURCE_RECONFIGURE):
                return self.async_abort(reason="no_existing_subentry")
            if self.source == SOURCE_RECONFIGURE:
                self._source = SOURCE_USER
                self.context["source"] = SOURCE_USER
            self._schedule_reload()
            return self.async_create_entry(title="Sentinel", data=data)

        self._schedule_reload()
        return self.async_update_and_abort(
            self._get_entry(),
            current,
            data=data,
            title="Sentinel",
        )

    async_step_reconfigure = async_step_user
