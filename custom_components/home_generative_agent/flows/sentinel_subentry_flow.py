"""Sentinel configuration subentry flow."""

from __future__ import annotations

import json
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
    TimeSelector,
)

from ..const import (  # noqa: TID252
    CONF_CRITICAL_ACTION_PIN,
    CONF_EXPLAIN_ENABLED,
    CONF_NOTIFY_SERVICE,
    CONF_SENTINEL_BASELINE_ENABLED,
    CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    CONF_SENTINEL_CAMERA_ENTRY_LINKS,
    CONF_SENTINEL_COOLDOWN_MINUTES,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_DISCOVERY_ENABLED,
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    CONF_SENTINEL_DISCOVERY_MAX_RECORDS,
    CONF_SENTINEL_ENABLED,
    CONF_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    CONF_SENTINEL_INTERVAL_SECONDS,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH,
    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT,
    CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    RECOMMENDED_EXPLAIN_ENABLED,
    RECOMMENDED_SENTINEL_BASELINE_ENABLED,
    RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS,
    RECOMMENDED_SENTINEL_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
    RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
    RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
    RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
    RECOMMENDED_SENTINEL_ENABLED,
    RECOMMENDED_SENTINEL_ENTITY_COOLDOWN_MINUTES,
    RECOMMENDED_SENTINEL_INTERVAL_SECONDS,
    RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
    RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    SUBENTRY_TYPE_SENTINEL,
)
from ..core.utils import hash_pin, list_mobile_notify_services  # noqa: TID252


def _camera_entry_links_json(payload: dict[str, Any]) -> str:
    """Serialize sentinel_camera_entry_links from payload to a JSON string."""
    value = payload.get(
        CONF_SENTINEL_CAMERA_ENTRY_LINKS, RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS
    )
    if not isinstance(value, dict):
        value = {}
    return json.dumps(value)


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
        CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES: (
            RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES
        ),
        CONF_SENTINEL_DISCOVERY_ENABLED: RECOMMENDED_SENTINEL_DISCOVERY_ENABLED,
        CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS: (
            RECOMMENDED_SENTINEL_DISCOVERY_INTERVAL_SECONDS
        ),
        CONF_SENTINEL_DISCOVERY_MAX_RECORDS: RECOMMENDED_SENTINEL_DISCOVERY_MAX_RECORDS,
        CONF_SENTINEL_BASELINE_ENABLED: RECOMMENDED_SENTINEL_BASELINE_ENABLED,
        CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES: (
            RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES
        ),
        CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS: (
            RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS
        ),
        CONF_EXPLAIN_ENABLED: RECOMMENDED_EXPLAIN_ENABLED,
        CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: (
            RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE
        ),
        CONF_SENTINEL_DAILY_DIGEST_ENABLED: RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME: RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
        CONF_SENTINEL_CAMERA_ENTRY_LINKS: RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS,
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
                CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
                        RECOMMENDED_SENTINEL_PENDING_PROMPT_TTL_MINUTES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=5, max=1440, step=5)),
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
                CONF_SENTINEL_BASELINE_ENABLED,
                default=bool(
                    payload.get(
                        CONF_SENTINEL_BASELINE_ENABLED,
                        RECOMMENDED_SENTINEL_BASELINE_ENABLED,
                    )
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
                        RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=1, max=1440, step=1)),
            vol.Required(
                CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
                default=int(
                    payload.get(
                        CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
                        RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=60, max=86400, step=60)),
            vol.Required(
                CONF_EXPLAIN_ENABLED,
                default=bool(
                    payload.get(CONF_EXPLAIN_ENABLED, RECOMMENDED_EXPLAIN_ENABLED)
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
                default=bool(
                    payload.get(
                        CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
                        RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
                    )
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_DAILY_DIGEST_ENABLED,
                default=bool(
                    payload.get(
                        CONF_SENTINEL_DAILY_DIGEST_ENABLED,
                        RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
                    )
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_DAILY_DIGEST_TIME,
                default=str(
                    payload.get(
                        CONF_SENTINEL_DAILY_DIGEST_TIME,
                        RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
                    )
                ),
            ): TimeSelector(),
            vol.Optional(
                CONF_SENTINEL_CAMERA_ENTRY_LINKS,
                description={
                    "suggested_value": _camera_entry_links_json(payload),
                },
                default=_camera_entry_links_json(payload),
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_CRITICAL_ACTION_PIN,
                description={
                    "suggested_value": "",
                },
                default="",
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
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

    async def async_step_settings(  # noqa: PLR0912, PLR0915
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
        errors: dict[str, str] = {}
        notify_service = str(data.get(CONF_NOTIFY_SERVICE, "") or "").strip()
        if notify_service:
            data[CONF_NOTIFY_SERVICE] = notify_service
        else:
            data.pop(CONF_NOTIFY_SERVICE, None)

        raw_pin = str(data.get(CONF_CRITICAL_ACTION_PIN, "") or "").strip()
        data.pop(CONF_CRITICAL_ACTION_PIN, None)
        if raw_pin:
            if (
                not raw_pin.isdigit()
                or not CRITICAL_PIN_MIN_LEN <= len(raw_pin) <= CRITICAL_PIN_MAX_LEN
            ):
                errors["base"] = "invalid_pin"
            else:
                hashed, salt = hash_pin(raw_pin)
                data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] = hashed
                data[CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT] = salt
        elif not data.get(CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE, False):
            data.pop(CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH, None)
            data.pop(CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT, None)
        elif current is not None:
            existing_data = dict(current.data)
            if existing_data.get(CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH):
                data[CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH] = existing_data[
                    CONF_SENTINEL_LEVEL_INCREASE_PIN_HASH
                ]
            if existing_data.get(CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT):
                data[CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT] = existing_data[
                    CONF_SENTINEL_LEVEL_INCREASE_PIN_SALT
                ]

        raw_links = str(data.get(CONF_SENTINEL_CAMERA_ENTRY_LINKS, "") or "").strip()
        if raw_links and raw_links != "{}":
            try:
                parsed_links = json.loads(raw_links)
                if not isinstance(parsed_links, dict) or not all(
                    isinstance(k, str)
                    and isinstance(v, list)
                    and all(isinstance(i, str) for i in v)
                    for k, v in parsed_links.items()
                ):
                    errors["base"] = "invalid_camera_entry_links"
                else:
                    data[CONF_SENTINEL_CAMERA_ENTRY_LINKS] = parsed_links
            except (json.JSONDecodeError, ValueError):
                errors["base"] = "invalid_camera_entry_links"
        else:
            data[CONF_SENTINEL_CAMERA_ENTRY_LINKS] = {}

        if errors:
            # Strip any raw (non-dict) value for camera_entry_links so the schema
            # helper receives a dict and json.dumps() produces valid JSON for the
            # form pre-fill rather than a Python repr string.
            error_payload = {**payload, **data}
            if not isinstance(
                error_payload.get(CONF_SENTINEL_CAMERA_ENTRY_LINKS), dict
            ):
                error_payload.pop(CONF_SENTINEL_CAMERA_ENTRY_LINKS, None)
            return self.async_show_form(
                step_id="settings",
                data_schema=self._schema(error_payload),
                errors=errors,
            )

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
