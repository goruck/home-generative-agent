"""Feature/tool config subentry flow."""

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
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import SUBENTRY_TYPE_MODEL_PROVIDER  # noqa: TID252

FeatureNames = {
    "conversation": "Conversation Agent",
    "camera_image_analysis": "Camera Image Analysis",
    "home_state_summary": "Home State Summary",
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
            if subentry.subentry_type == "feature"
        ]
        if len(matches) == 1:
            return matches[0]
    return None


class FeatureSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for feature/tool subentries."""

    def _provider_options(self) -> list[SelectOptionDict]:
        """Return available model provider options for the parent entry."""
        entry = self._get_entry()
        options: list[SelectOptionDict] = []
        for subentry in entry.subentries.values():
            if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
                continue
            options.append(
                SelectOptionDict(
                    label=subentry.title or subentry.subentry_id,
                    value=subentry.subentry_id,
                )
            )
        return options

    def _schedule_reload(self) -> None:
        """Reload the parent entry to apply subentry changes."""
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure a feature and associate it with a model provider."""
        errors: dict[str, str] = {}
        current = _current_subentry(self)
        default_type = "conversation"
        default_provider = None
        default_name = FeatureNames[default_type]

        if current:
            data = current.data or {}
            default_type = data.get("feature_type", default_type)
            default_provider = data.get("model_provider_id")
            default_name = data.get(
                "name", current.title or FeatureNames.get(default_type, default_type)
            )

        provider_opts = self._provider_options()
        if not provider_opts:
            errors["base"] = "no_providers"

        if user_input is not None and not errors:
            feature_type = user_input["feature_type"]
            provider_id = user_input["model_provider_id"]
            if provider_id not in {opt["value"] for opt in provider_opts}:
                errors["base"] = "no_providers"
            else:
                name = (
                    user_input.get("name")
                    or FeatureNames.get(feature_type)
                    or feature_type
                )
                payload = {
                    "feature_type": feature_type,
                    "name": name,
                    "model_provider_id": provider_id,
                    "config": {},
                }
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

        schema = vol.Schema(
            {
                vol.Required(
                    "feature_type",
                    default=default_type,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=v, value=k)
                            for k, v in FeatureNames.items()
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                ),
                vol.Required(
                    "model_provider_id",
                    default=default_provider,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=provider_opts,
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

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async_step_reconfigure = async_step_user
