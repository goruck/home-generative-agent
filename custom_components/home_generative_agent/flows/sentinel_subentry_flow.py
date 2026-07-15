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
    CONF_SENTINEL_APPLIANCE_DURATION_MIN,
    CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
    CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    CONF_SENTINEL_BASELINE_ENABLED,
    CONF_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    CONF_SENTINEL_BASELINE_MIN_SAMPLES,
    CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES,
    CONF_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
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
    CONF_SENTINEL_QUIET_HOURS_END,
    CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
    CONF_SENTINEL_QUIET_HOURS_START,
    CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    RECOMMENDED_EXPLAIN_ENABLED,
    RECOMMENDED_SENTINEL_APPLIANCE_DURATION_MIN,
    RECOMMENDED_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
    RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_ENABLED,
    RECOMMENDED_SENTINEL_BASELINE_FRESHNESS_THRESHOLD_SECONDS,
    RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
    RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES,
    RECOMMENDED_SENTINEL_BASELINE_UPDATE_INTERVAL_MINUTES,
    RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
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
    RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES,
    RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE,
    RECOMMENDED_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    SENTINEL_SEVERITIES,
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


def _parse_json_entity_map(
    data: dict[str, Any],
    key: str,
    errors: dict[str, str],
    error_key: str,
) -> None:
    """
    Parse a JSON ``dict[str, list[str]]`` text field into ``data[key]`` in place.

    Records ``error_key`` in ``errors`` (leaving the raw string in ``data``)
    when the value is not valid JSON of that shape.  Empty input stores ``{}``.
    """
    raw = str(data.get(key, "") or "").strip()
    if not raw or raw == "{}":
        data[key] = {}
        return
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        errors.setdefault("base", error_key)
        return
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str)
        and isinstance(v, list)
        and all(isinstance(i, str) for i in v)
        for k, v in parsed.items()
    ):
        errors.setdefault("base", error_key)
        return
    data[key] = parsed


def _rule_entity_exclusions_json(payload: dict[str, Any]) -> str:
    """Serialize sentinel_rule_entity_exclusions from payload to a JSON string."""
    value = payload.get(
        CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
        RECOMMENDED_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    )
    if not isinstance(value, dict):
        value = {}
    return json.dumps(value)


_QUIET_HOUR_MAX = 23

_QUIET_HOUR_OPTIONS = [
    SelectOptionDict(label="Disabled", value=""),
    *(
        SelectOptionDict(label=f"{hour:02d}:00", value=str(hour))
        for hour in range(_QUIET_HOUR_MAX + 1)
    ),
]

_QUIET_SEVERITY_OPTIONS = [
    SelectOptionDict(label=severity.capitalize(), value=severity)
    for severity in SENTINEL_SEVERITIES
]


def _quiet_hour_selector() -> SelectSelector:
    """Return a fresh hour-of-day dropdown selector for quiet hours."""
    return SelectSelector(
        SelectSelectorConfig(
            options=_QUIET_HOUR_OPTIONS,
            mode=SelectSelectorMode.DROPDOWN,
            sort=False,
            custom_value=False,
        )
    )


def _parse_quiet_hour(raw: str) -> int | None:
    """Parse a quiet-hours select value into a local hour, or None if invalid."""
    try:
        value = int(raw)
    except ValueError:
        return None
    if not 0 <= value <= _QUIET_HOUR_MAX:
        return None
    return value


def _quiet_hour_str(payload: dict[str, Any], key: str) -> str:
    """Return the quiet-hours value from payload as a select value ("" = disabled)."""
    value = payload.get(key)
    if value is None or value == "":
        return ""
    parsed = _parse_quiet_hour(str(value).strip())
    return "" if parsed is None else str(parsed)


def _raw_quiet_hour(data: dict[str, Any], key: str) -> str:
    """Return the submitted quiet-hours value as a stripped string ("" = unset)."""
    value = data.get(key)
    if value is None:
        return ""
    # str() (not truthiness) so hour 0 is treated as set, not as "Disabled".
    return str(value).strip()


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the Sentinel subentry currently being edited, if any."""
    entry = flow._get_entry()  # noqa: SLF001
    subentry_id = getattr(flow, "_subentry_id", None)
    if not subentry_id:
        subentry_id = flow.context.get("subentry_id")
    if subentry_id:
        # For reconfigure flows this will find the subentry being edited.
        # For user flows HA pre-generates a UUID that doesn't exist yet, so
        # the get() returns None and we fall through to the type scan below.
        found = entry.subentries.get(subentry_id)
        if found is not None:
            return found
    matches = [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_SENTINEL
    ]
    if matches:
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
        CONF_SENTINEL_BASELINE_MIN_SAMPLES: RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
        CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES: (
            RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES
        ),
        CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W: (
            RECOMMENDED_SENTINEL_APPLIANCE_POWER_THRESHOLD_W
        ),
        CONF_SENTINEL_APPLIANCE_DURATION_MIN: (
            RECOMMENDED_SENTINEL_APPLIANCE_DURATION_MIN
        ),
        CONF_EXPLAIN_ENABLED: RECOMMENDED_EXPLAIN_ENABLED,
        CONF_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE: (
            RECOMMENDED_SENTINEL_REQUIRE_PIN_FOR_LEVEL_INCREASE
        ),
        CONF_SENTINEL_DAILY_DIGEST_ENABLED: RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
        CONF_SENTINEL_DAILY_DIGEST_TIME: RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
        # Quiet-hours start/end default to absent (feature disabled).
        CONF_SENTINEL_QUIET_HOURS_SEVERITIES: list(
            RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES
        ),
        CONF_SENTINEL_CAMERA_ENTRY_LINKS: RECOMMENDED_SENTINEL_CAMERA_ENTRY_LINKS,
        CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: (
            RECOMMENDED_SENTINEL_RULE_ENTITY_EXCLUSIONS
        ),
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
                CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                default=bool(
                    payload.get(
                        CONF_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                        RECOMMENDED_SENTINEL_BASELINE_WEEKLY_PATTERNS,
                    )
                ),
            ): BooleanSelector(),
            vol.Required(
                CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
                        RECOMMENDED_SENTINEL_BASELINE_DOW_MIN_SAMPLES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=1, max=52, step=1)),
            vol.Required(
                CONF_SENTINEL_BASELINE_MIN_SAMPLES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_BASELINE_MIN_SAMPLES,
                        RECOMMENDED_SENTINEL_BASELINE_MIN_SAMPLES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=1, max=500, step=1)),
            vol.Required(
                CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES,
                default=int(
                    payload.get(
                        CONF_SENTINEL_BASELINE_SUSTAINED_MINUTES,
                        RECOMMENDED_SENTINEL_BASELINE_SUSTAINED_MINUTES,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=0, max=120, step=5)),
            vol.Required(
                CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
                default=float(
                    payload.get(
                        CONF_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
                        RECOMMENDED_SENTINEL_APPLIANCE_POWER_THRESHOLD_W,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=10, max=10000, step=10)),
            vol.Required(
                CONF_SENTINEL_APPLIANCE_DURATION_MIN,
                default=int(
                    payload.get(
                        CONF_SENTINEL_APPLIANCE_DURATION_MIN,
                        RECOMMENDED_SENTINEL_APPLIANCE_DURATION_MIN,
                    )
                ),
            ): NumberSelector(NumberSelectorConfig(min=5, max=1440, step=5)),
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
                CONF_SENTINEL_QUIET_HOURS_START,
                default=_quiet_hour_str(payload, CONF_SENTINEL_QUIET_HOURS_START),
            ): _quiet_hour_selector(),
            vol.Optional(
                CONF_SENTINEL_QUIET_HOURS_END,
                default=_quiet_hour_str(payload, CONF_SENTINEL_QUIET_HOURS_END),
            ): _quiet_hour_selector(),
            vol.Required(
                CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
                default=list(
                    payload.get(
                        CONF_SENTINEL_QUIET_HOURS_SEVERITIES,
                        RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES,
                    )
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=_QUIET_SEVERITY_OPTIONS,
                    multiple=True,
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            ),
            vol.Optional(
                CONF_SENTINEL_CAMERA_ENTRY_LINKS,
                description={
                    "suggested_value": _camera_entry_links_json(payload),
                },
                default=_camera_entry_links_json(payload),
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
                description={
                    "suggested_value": _rule_entity_exclusions_json(payload),
                },
                default=_rule_entity_exclusions_json(payload),
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
        """Entry point for new Sentinel setup — show mode selector."""
        return await self.async_step_setup_mode(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure — bypass mode selector, go directly to full settings."""
        return await self.async_step_settings(user_input)

    async def async_step_setup_mode(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show Basic/Advanced mode selector for new Sentinel setup."""
        if user_input is None:
            entry = self._get_entry()
            has_existing = any(
                s.subentry_type == SUBENTRY_TYPE_SENTINEL
                for s in entry.subentries.values()
            )
            overwrite_warning = (
                "\n\n⚠️ Sentinel is already configured. "
                "Choosing **Basic setup** will overwrite your current "
                "settings with recommended defaults."
                if has_existing
                else ""
            )
            return self.async_show_form(
                step_id="setup_mode",
                data_schema=vol.Schema(
                    {
                        vol.Required("setup_mode", default="basic"): SelectSelector(
                            SelectSelectorConfig(
                                options=[
                                    SelectOptionDict(
                                        label="Basic setup", value="basic"
                                    ),
                                    SelectOptionDict(
                                        label="Advanced setup", value="advanced"
                                    ),
                                ],
                                mode=SelectSelectorMode.LIST,
                                sort=False,
                                custom_value=False,
                            )
                        )
                    }
                ),
                description_placeholders={"overwrite_warning": overwrite_warning},
            )
        if user_input.get("setup_mode", "basic") == "basic":
            return await self.async_step_basic_settings(None)
        return await self.async_step_settings(None)

    async def async_step_basic_settings(  # noqa: PLR0912
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Expose only the four essential Sentinel fields; fill rest from defaults."""
        current = _current_subentry(self)
        payload = _default_payload()
        if current is not None:
            payload.update(dict(current.data))

        mobile_opts = list_mobile_notify_services(self.hass)
        # Basic setup always starts from defaults — no pre-existing notify service.
        notify_value = ""

        schema: dict[Any, Any] = {
            vol.Required(
                CONF_SENTINEL_ENABLED,
                default=bool(
                    payload.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
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
                CONF_CRITICAL_ACTION_PIN,
                description={"suggested_value": ""},
                default="",
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        }

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

        # Basic setup always shows recommended defaults in the form, regardless of
        # any existing configuration (the overwrite warning makes this intent explicit).
        defaults = _default_payload()
        suggested: dict[str, Any] = {
            k: v for k, v in defaults.items() if k != CONF_CRITICAL_ACTION_PIN
        }

        if user_input is None:
            return self.async_show_form(
                step_id="basic_settings",
                data_schema=self.add_suggested_values_to_schema(
                    vol.Schema(schema), suggested
                ),
            )

        data = _default_payload()
        errors: dict[str, str] = {}

        notify_service = str(user_input.get(CONF_NOTIFY_SERVICE, "") or "").strip()
        if notify_service:
            data[CONF_NOTIFY_SERVICE] = notify_service
        else:
            data.pop(CONF_NOTIFY_SERVICE, None)

        raw_pin = str(user_input.get(CONF_CRITICAL_ACTION_PIN, "") or "").strip()
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

        data[CONF_SENTINEL_ENABLED] = bool(
            user_input.get(CONF_SENTINEL_ENABLED, RECOMMENDED_SENTINEL_ENABLED)
        )
        data[CONF_SENTINEL_DAILY_DIGEST_ENABLED] = bool(
            user_input.get(
                CONF_SENTINEL_DAILY_DIGEST_ENABLED,
                RECOMMENDED_SENTINEL_DAILY_DIGEST_ENABLED,
            )
        )
        data[CONF_SENTINEL_DAILY_DIGEST_TIME] = str(
            user_input.get(
                CONF_SENTINEL_DAILY_DIGEST_TIME,
                RECOMMENDED_SENTINEL_DAILY_DIGEST_TIME,
            )
        )

        if errors:
            return self.async_show_form(
                step_id="basic_settings",
                data_schema=self.add_suggested_values_to_schema(
                    vol.Schema(schema), suggested
                ),
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

    async def async_step_settings(  # noqa: PLR0912, PLR0915
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create or edit Sentinel configuration."""
        current = _current_subentry(self)
        payload = _default_payload()
        if current is not None:
            payload.update(dict(current.data))

        if user_input is None:
            suggested: dict[str, Any] = {
                k: v for k, v in payload.items() if k != CONF_CRITICAL_ACTION_PIN
            }
            suggested[CONF_SENTINEL_CAMERA_ENTRY_LINKS] = _camera_entry_links_json(
                payload
            )
            suggested[CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS] = (
                _rule_entity_exclusions_json(payload)
            )
            for key in (CONF_SENTINEL_QUIET_HOURS_START, CONF_SENTINEL_QUIET_HOURS_END):
                suggested[key] = _quiet_hour_str(payload, key)
            return self.async_show_form(
                step_id="settings",
                data_schema=self.add_suggested_values_to_schema(
                    self._schema(payload), suggested
                ),
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

        raw_quiet_start = _raw_quiet_hour(data, CONF_SENTINEL_QUIET_HOURS_START)
        raw_quiet_end = _raw_quiet_hour(data, CONF_SENTINEL_QUIET_HOURS_END)
        if (raw_quiet_start == "") != (raw_quiet_end == ""):
            errors.setdefault("base", "quiet_hours_incomplete")
        elif raw_quiet_start and raw_quiet_end:
            quiet_start = _parse_quiet_hour(raw_quiet_start)
            quiet_end = _parse_quiet_hour(raw_quiet_end)
            if quiet_start is None or quiet_end is None:
                errors.setdefault("base", "invalid_quiet_hours")
            else:
                data[CONF_SENTINEL_QUIET_HOURS_START] = quiet_start
                data[CONF_SENTINEL_QUIET_HOURS_END] = quiet_end
        else:
            data.pop(CONF_SENTINEL_QUIET_HOURS_START, None)
            data.pop(CONF_SENTINEL_QUIET_HOURS_END, None)

        quiet_severities = data.get(CONF_SENTINEL_QUIET_HOURS_SEVERITIES)
        data[CONF_SENTINEL_QUIET_HOURS_SEVERITIES] = (
            [str(s) for s in quiet_severities if str(s) in SENTINEL_SEVERITIES]
            if isinstance(quiet_severities, list)
            else list(RECOMMENDED_SENTINEL_QUIET_HOURS_SEVERITIES)
        )

        _parse_json_entity_map(
            data,
            CONF_SENTINEL_CAMERA_ENTRY_LINKS,
            errors,
            "invalid_camera_entry_links",
        )
        _parse_json_entity_map(
            data,
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
            errors,
            "invalid_rule_entity_exclusions",
        )

        if errors:
            # Strip any raw (non-dict) value for camera_entry_links so the schema
            # helper receives a dict and json.dumps() produces valid JSON for the
            # form pre-fill rather than a Python repr string.
            error_payload = {**payload, **data}
            if not isinstance(
                error_payload.get(CONF_SENTINEL_CAMERA_ENTRY_LINKS), dict
            ):
                error_payload.pop(CONF_SENTINEL_CAMERA_ENTRY_LINKS, None)
            if not isinstance(
                error_payload.get(CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS), dict
            ):
                error_payload.pop(CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS, None)
            error_suggested: dict[str, Any] = {
                k: v for k, v in error_payload.items() if k != CONF_CRITICAL_ACTION_PIN
            }
            error_suggested[CONF_SENTINEL_CAMERA_ENTRY_LINKS] = (
                _camera_entry_links_json(error_payload)
            )
            error_suggested[CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS] = (
                _rule_entity_exclusions_json(error_payload)
            )
            for key in (CONF_SENTINEL_QUIET_HOURS_START, CONF_SENTINEL_QUIET_HOURS_END):
                error_suggested[key] = _quiet_hour_str(error_payload, key)
            return self.async_show_form(
                step_id="settings",
                data_schema=self.add_suggested_values_to_schema(
                    self._schema(error_payload), error_suggested
                ),
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
