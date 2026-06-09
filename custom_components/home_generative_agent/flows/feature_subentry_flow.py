"""Feature setup and configuration subentry flow."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
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
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    ObjectSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import (  # noqa: TID252
    CONF_DB_NAME,
    CONF_DB_PARAMS,
    CONF_DISABLED_FEATURES,
    CONF_FEATURE_FALLBACK_PROVIDER_IDS,
    CONF_FEATURE_MODEL,
    CONF_FEATURE_MODEL_CONTEXT_SIZE,
    CONF_FEATURE_MODEL_KEEPALIVE,
    CONF_FEATURE_MODEL_NAME,
    CONF_FEATURE_MODEL_REASONING,
    CONF_FEATURE_MODEL_TEMPERATURE,
    FEATURE_CATEGORY_MAP,
    FEATURE_DEFS,
    KEEPALIVE_MAX_SECONDS,
    KEEPALIVE_SENTINEL,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_DB_HOST,
    RECOMMENDED_DB_NAME,
    RECOMMENDED_DB_PARAMS,
    RECOMMENDED_DB_PASSWORD,
    RECOMMENDED_DB_PORT,
    RECOMMENDED_DB_USERNAME,
    RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    RECOMMENDED_OLLAMA_REASONING,
    RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
    RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    SUBENTRY_TYPE_DATABASE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from ..core.db_utils import build_postgres_uri  # noqa: TID252
from ..core.utils import (  # noqa: TID252
    CannotConnectError,
    InvalidAuthError,
    list_ollama_models,
    validate_db_uri,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

KEEPALIVE_DEFAULTS = {
    "chat": RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    "vlm": RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    "summarization": RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
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
            if subentry.subentry_type == SUBENTRY_TYPE_FEATURE
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _feature_subentry(entry: ConfigEntry, feature_type: str) -> ConfigSubentry | None:
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_FEATURE:
            continue
        if subentry.data.get("feature_type") == feature_type:
            return subentry
    return None


def _provider_options(entry: ConfigEntry, category: str) -> list[SelectOptionDict]:
    options: list[SelectOptionDict] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
            continue
        caps = set(subentry.data.get("capabilities") or [])
        if caps and category not in caps:
            continue
        options.append(
            SelectOptionDict(
                label=subentry.title or subentry.subentry_id,
                value=subentry.subentry_id,
            )
        )
    return options


def _provider_type_for_id(entry: ConfigEntry, provider_id: str | None) -> str | None:
    if not provider_id:
        return None
    subentry = entry.subentries.get(provider_id)
    if not subentry:
        return None
    return subentry.data.get("provider_type")


def _fallback_provider_options(
    entry: ConfigEntry, category: str, primary_id: str | None
) -> list[SelectOptionDict]:
    """Return provider options excluding the primary."""
    options: list[SelectOptionDict] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
            continue
        if primary_id and subentry.subentry_id == primary_id:
            continue
        caps = set(subentry.data.get("capabilities") or [])
        if caps and category not in caps:
            continue
        options.append(
            SelectOptionDict(
                label=subentry.title or subentry.subentry_id,
                value=subentry.subentry_id,
            )
        )
    return options


def _default_model_data(category: str, provider_type: str | None) -> dict[str, Any]:
    spec = MODEL_CATEGORY_SPECS.get(category, {})
    defaults: dict[str, Any] = {}
    if provider_type:
        defaults[CONF_FEATURE_MODEL_NAME] = spec.get("recommended_models", {}).get(
            provider_type
        )
        temp = spec.get("recommended_temperature")
        if temp is not None:
            defaults[CONF_FEATURE_MODEL_TEMPERATURE] = temp

    if provider_type == "ollama":
        keepalive = KEEPALIVE_DEFAULTS.get(category)
        if keepalive is not None:
            defaults[CONF_FEATURE_MODEL_KEEPALIVE] = keepalive
        defaults[CONF_FEATURE_MODEL_CONTEXT_SIZE] = RECOMMENDED_OLLAMA_CONTEXT_SIZE
        if category == "chat":
            defaults[CONF_FEATURE_MODEL_REASONING] = RECOMMENDED_OLLAMA_REASONING

    return defaults


def _normalize_reasoning(value: Any) -> Any:
    if value in ("false", "", None):
        return None
    if value == "true":
        return True
    return value


class FeatureSubentryFlow(ConfigSubentryFlow):
    """Config flow handler for feature/tool subentries."""

    def __init__(self) -> None:
        """Initialize the feature subentry flow."""
        self._setup_mode = False
        self._basic_mode = False
        self._basic_skipped_features: list[str] = []
        self._feature_queue: list[str] = []
        self._active_feature: str | None = None
        self._pending_provider_id: str | None = None
        self._ollama_model_cache: dict[str, list[str]] = {}

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _disabled_feature_cache(self) -> dict[str, Any]:
        entry = self._get_entry()
        return dict(entry.options.get(CONF_DISABLED_FEATURES, {}))

    def _persist_disabled_cache(self, cache: dict[str, Any]) -> None:
        entry = self._get_entry()
        options = dict(entry.options)
        if cache:
            options[CONF_DISABLED_FEATURES] = cache
        else:
            options.pop(CONF_DISABLED_FEATURES, None)
        self.hass.config_entries.async_update_entry(entry, options=options)

    def _feature_defaults(self, feature_type: str) -> dict[str, Any]:
        category = FEATURE_CATEGORY_MAP.get(feature_type)
        return {
            "feature_type": feature_type,
            "name": FEATURE_DEFS[feature_type]["name"],
            "model_provider_id": None,
            CONF_FEATURE_MODEL: _default_model_data(category or "", None),
            "config": {},
        }

    def _feature_payload(
        self,
        feature_type: str,
        provider_id: str | None,
        model_data: dict[str, Any],
        config_data: dict[str, Any],
        fallback_provider_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_type": feature_type,
            "name": FEATURE_DEFS[feature_type]["name"],
            "model_provider_id": provider_id,
            CONF_FEATURE_MODEL: model_data,
            "config": config_data,
        }
        if fallback_provider_ids:
            payload[CONF_FEATURE_FALLBACK_PROVIDER_IDS] = fallback_provider_ids
        return payload

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the initial step."""
        current = _current_subentry(self)
        if current is not None:
            self._setup_mode = False
            self._active_feature = current.data.get("feature_type")
            return await self._async_step_feature(self._active_feature, user_input)
        return await self.async_step_setup_mode(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure — bypass mode selector, go directly to the Advanced path."""
        current = _current_subentry(self)
        if current is not None:
            self._setup_mode = False
            self._active_feature = current.data.get("feature_type")
            return await self._async_step_feature(self._active_feature, user_input)
        self._setup_mode = True
        return await self.async_step_feature_enable(user_input)

    async def async_step_setup_mode(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show Basic/Advanced mode selector for new setup."""
        if user_input is None:
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
            )
        self._setup_mode = True
        if user_input.get("setup_mode", "basic") == "basic":
            self._basic_mode = True
            return await self._async_step_basic_setup()
        self._basic_mode = False
        return await self.async_step_feature_enable(None)

    async def _async_step_basic_setup(self) -> SubentryFlowResult:
        """Auto-create feature subentries with the first compatible provider."""
        entry = self._get_entry()
        any_providers = any(
            s.subentry_type == SUBENTRY_TYPE_MODEL_PROVIDER
            for s in entry.subentries.values()
        )
        if not any_providers:
            return self.async_abort(reason="no_provider_for_basic_setup")

        self._basic_skipped_features = []
        disabled_cache = self._disabled_feature_cache()

        for feature_type, info in FEATURE_DEFS.items():
            category = FEATURE_CATEGORY_MAP.get(feature_type, "")
            provider_opts = _provider_options(entry, category)
            if not provider_opts:
                self._basic_skipped_features.append(feature_type)
                continue
            provider_id = provider_opts[0]["value"]
            provider_type = _provider_type_for_id(entry, provider_id)
            model_data = _default_model_data(category, provider_type)
            payload = self._feature_payload(feature_type, provider_id, model_data, {})
            existing = _feature_subentry(entry, feature_type)
            if existing is None:
                disabled_cache.pop(feature_type, None)
                subentry = ConfigSubentry(
                    subentry_type=SUBENTRY_TYPE_FEATURE,
                    title=info["name"],
                    unique_id=f"{entry.entry_id}_{feature_type}",
                    data=MappingProxyType(payload),
                )
                self.hass.config_entries.async_add_subentry(entry, subentry)
            else:
                self.hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
                    entry, existing, data=payload, title=info["name"]
                )

        self._persist_disabled_cache(disabled_cache)
        self._schedule_reload()
        return await self.async_step_database(None)

    async def async_step_setup_status(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show final status screen after basic setup writes."""
        if user_input is not None:
            return self.async_abort(reason="setup_complete")

        entry = self._get_entry()
        enabled_names = [
            FEATURE_DEFS[ft]["name"]
            for ft in FEATURE_DEFS
            if _feature_subentry(entry, ft) is not None
        ]
        db_ok = any(
            s.subentry_type == SUBENTRY_TYPE_DATABASE for s in entry.subentries.values()
        )
        skipped_names = [
            FEATURE_DEFS[ft]["name"] for ft in self._basic_skipped_features
        ]
        skipped_section = (
            f"\n\n**Skipped (no compatible provider):** {', '.join(skipped_names)}"
            if skipped_names
            else ""
        )
        return self.async_show_form(
            step_id="setup_status",
            data_schema=vol.Schema({}),
            description_placeholders={
                "features_enabled": (
                    ", ".join(enabled_names) if enabled_names else "None"
                ),
                "database_status": "Configured" if db_ok else "Not configured",
                "skipped_section": skipped_section,
            },
        )

    async def async_step_feature_enable(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the feature enable step."""
        entry = self._get_entry()
        disabled_cache = self._disabled_feature_cache()

        if user_input is None:
            schema_dict: dict[Any, Any] = {}
            for feature_type, info in FEATURE_DEFS.items():
                if info["required"]:
                    continue
                enabled = _feature_subentry(entry, feature_type) is not None
                if feature_type in disabled_cache:
                    enabled = False
                schema_dict[
                    vol.Optional(
                        feature_type,
                        default=enabled,
                        description={"suggested_value": enabled},
                    )
                ] = BooleanSelector()
            return self.async_show_form(
                step_id="feature_enable", data_schema=vol.Schema(schema_dict)
            )

        cache = self._disabled_feature_cache()
        enabled_features = {"conversation"}
        for feature_type, info in FEATURE_DEFS.items():
            if info["required"]:
                continue
            enabled = bool(user_input.get(feature_type, False))
            if enabled:
                enabled_features.add(feature_type)
            existing = _feature_subentry(entry, feature_type)
            if not enabled and existing:
                cache[feature_type] = dict(existing.data)
                self.hass.config_entries.async_remove_subentry(  # type: ignore[attr-defined]
                    entry, existing.subentry_id
                )
            if enabled and not existing:
                payload = cache.pop(feature_type, None) or self._feature_defaults(
                    feature_type
                )
                subentry = ConfigSubentry(
                    subentry_type=SUBENTRY_TYPE_FEATURE,
                    title=info["name"],
                    unique_id=f"{entry.entry_id}_{feature_type}",
                    data=MappingProxyType(payload),
                )
                self.hass.config_entries.async_add_subentry(entry, subentry)

        self._persist_disabled_cache(cache)
        self._feature_queue = [
            feature_type
            for feature_type in FEATURE_DEFS
            if feature_type in enabled_features
        ]
        return await self._advance_setup()

    async def _advance_setup(self) -> SubentryFlowResult:
        next_feature = self._feature_queue.pop(0) if self._feature_queue else None
        if next_feature is None:
            return await self.async_step_database()
        self._active_feature = next_feature
        return await self._async_step_feature(next_feature, None)

    async def _async_step_feature(
        self, feature_type: str | None, user_input: dict[str, Any] | None
    ) -> SubentryFlowResult:
        """Start the feature flow by selecting a provider."""
        if feature_type not in FEATURE_DEFS:
            return self.async_abort(reason="no_existing_subentry")

        entry = self._get_entry()
        subentry = _feature_subentry(entry, feature_type)
        if subentry is None:
            return self.async_abort(reason="no_existing_subentry")

        self._active_feature = feature_type
        self._pending_provider_id = None

        category = FEATURE_CATEGORY_MAP.get(feature_type)
        provider_opts = _provider_options(entry, category or "")

        provider_notice = ""
        if not provider_opts:
            provider_notice = """
            No model provider is configured.
            A default model will be assigned once available.
            """
            return self.async_show_form(
                step_id="feature_model",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "provider_notice": provider_notice,
                    "feature_name": FEATURE_DEFS[feature_type]["name"],
                },
            )

        provider_ids = {opt["value"] for opt in provider_opts}
        existing_provider_id = subentry.data.get("model_provider_id")
        provider_id = (
            existing_provider_id
            if existing_provider_id in provider_ids
            else provider_opts[0]["value"]
        )

        schema = vol.Schema(
            {
                vol.Required(
                    "model_provider_id",
                    default=provider_id,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=provider_opts,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                )
            }
        )
        if user_input is not None:
            return await self.async_step_feature_provider(user_input)

        return self.async_show_form(
            step_id="feature_provider",
            data_schema=schema,
            description_placeholders={
                "feature_name": FEATURE_DEFS[feature_type]["name"],
            },
        )

    async def async_step_feature_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the provider selection step."""
        feature_type = self._active_feature
        if feature_type not in FEATURE_DEFS:
            return self.async_abort(reason="no_existing_subentry")

        entry = self._get_entry()
        subentry = _feature_subentry(entry, feature_type)
        if subentry is None:
            return self.async_abort(reason="no_existing_subentry")

        category = FEATURE_CATEGORY_MAP.get(feature_type)
        provider_opts = _provider_options(entry, category or "")
        if not provider_opts:
            return await self.async_step_feature_model()

        provider_ids = {opt["value"] for opt in provider_opts}
        existing_provider_id = subentry.data.get("model_provider_id")
        provider_id = (
            existing_provider_id
            if existing_provider_id in provider_ids
            else provider_opts[0]["value"]
        )

        schema = vol.Schema(
            {
                vol.Required(
                    "model_provider_id",
                    default=provider_id,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=provider_opts,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                        custom_value=False,
                    )
                )
            }
        )

        if user_input is None:
            return self.async_show_form(
                step_id="feature_provider",
                data_schema=schema,
                description_placeholders={
                    "feature_name": FEATURE_DEFS[feature_type]["name"],
                },
            )

        provider_id = user_input.get("model_provider_id")
        if provider_id not in provider_ids:
            return self.async_show_form(
                step_id="feature_provider",
                data_schema=schema,
                errors={"base": "no_providers"},
                description_placeholders={
                    "feature_name": FEATURE_DEFS[feature_type]["name"],
                },
            )

        self._pending_provider_id = provider_id
        return await self.async_step_feature_model()

    async def async_step_feature_model(  # noqa: PLR0911, PLR0912, PLR0915
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the model configuration step."""
        feature_type = self._active_feature
        if feature_type not in FEATURE_DEFS:
            return self.async_abort(reason="no_existing_subentry")

        entry = self._get_entry()
        subentry = _feature_subentry(entry, feature_type)
        if subentry is None:
            return self.async_abort(reason="no_existing_subentry")

        category = FEATURE_CATEGORY_MAP.get(feature_type)
        provider_opts = _provider_options(entry, category or "")
        existing_model = dict(subentry.data.get(CONF_FEATURE_MODEL, {}))
        existing_config = dict(subentry.data.get("config", {}))
        existing_provider_id = subentry.data.get("model_provider_id")
        existing_fallback_ids = list(
            subentry.data.get(CONF_FEATURE_FALLBACK_PROVIDER_IDS) or []
        )

        provider_notice = ""
        if not provider_opts:
            provider_notice = """
            No model provider is configured.
            A default model will be assigned once available.
            """
            if user_input is not None:
                if self._setup_mode:
                    return await self._advance_setup()
                return self.async_update_and_abort(
                    entry,
                    subentry,
                    data=subentry.data,
                    title=subentry.title,
                )
            return self.async_show_form(
                step_id="feature_model",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "provider_notice": provider_notice,
                    "feature_name": FEATURE_DEFS[feature_type]["name"],
                },
            )

        provider_ids = {opt["value"] for opt in provider_opts}
        provider_id = (
            self._pending_provider_id
            if self._pending_provider_id in provider_ids
            else (
                existing_provider_id
                if existing_provider_id in provider_ids
                else provider_opts[0]["value"]
            )
        )
        if (
            self._pending_provider_id
            and self._pending_provider_id != existing_provider_id
        ):
            existing_model = {}
            existing_fallback_ids = []
        provider_type = _provider_type_for_id(entry, provider_id)
        defaults = _default_model_data(category or "", provider_type)

        model_options = (
            MODEL_CATEGORY_SPECS.get(category or "", {})
            .get("providers", {})
            .get(provider_type or "", [])
        )
        if provider_type == "ollama":
            provider_subentry = (
                entry.subentries.get(provider_id) if provider_id else None
            )
            settings = (
                provider_subentry.data.get("settings", {}) if provider_subentry else {}
            )
            ollama_url = settings.get("base_url")
            if isinstance(ollama_url, str) and ollama_url:
                cached = self._ollama_model_cache.get(ollama_url)
                if cached is None:
                    cached = await list_ollama_models(self.hass, ollama_url)
                    self._ollama_model_cache[ollama_url] = cached
                if cached:
                    model_options = cached
        fallback_opts = _fallback_provider_options(entry, category or "", provider_id)
        fallback_defaults = [
            fid
            for fid in existing_fallback_ids
            if any(opt["value"] == fid for opt in fallback_opts)
        ]
        schema: dict[Any, Any] = {
            vol.Required(
                CONF_FEATURE_MODEL_NAME,
                default=existing_model.get(CONF_FEATURE_MODEL_NAME)
                or defaults.get(CONF_FEATURE_MODEL_NAME),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[SelectOptionDict(label=m, value=m) for m in model_options],
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=True,
                )
            ),
        }
        if fallback_opts:
            schema[
                vol.Optional(
                    CONF_FEATURE_FALLBACK_PROVIDER_IDS,
                    default=fallback_defaults,
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=fallback_opts,
                    mode=SelectSelectorMode.LIST,
                    sort=False,
                    custom_value=False,
                    multiple=True,
                )
            )

        temp_default = existing_model.get(
            CONF_FEATURE_MODEL_TEMPERATURE
        ) or defaults.get(CONF_FEATURE_MODEL_TEMPERATURE)
        if temp_default is not None:
            schema[
                vol.Optional(
                    CONF_FEATURE_MODEL_TEMPERATURE,
                    default=temp_default,
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))

        if provider_type == "ollama":
            keepalive_default = existing_model.get(
                CONF_FEATURE_MODEL_KEEPALIVE
            ) or defaults.get(CONF_FEATURE_MODEL_KEEPALIVE)
            if keepalive_default is not None:
                schema[
                    vol.Optional(
                        CONF_FEATURE_MODEL_KEEPALIVE,
                        default=keepalive_default,
                    )
                ] = NumberSelector(
                    NumberSelectorConfig(
                        min=KEEPALIVE_SENTINEL,
                        max=KEEPALIVE_MAX_SECONDS,
                        step=1,
                    )
                )

            context_default = existing_model.get(
                CONF_FEATURE_MODEL_CONTEXT_SIZE
            ) or defaults.get(CONF_FEATURE_MODEL_CONTEXT_SIZE)
            if context_default is not None:
                schema[
                    vol.Optional(
                        CONF_FEATURE_MODEL_CONTEXT_SIZE,
                        default=context_default,
                    )
                ] = NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1))

            if category == "chat":
                reasoning_val = existing_model.get(CONF_FEATURE_MODEL_REASONING)
                if reasoning_val is True:
                    reasoning_default = "true"
                elif reasoning_val is None or reasoning_val is False:
                    reasoning_default = "false"
                else:
                    reasoning_default = str(reasoning_val)
                schema[
                    vol.Optional(
                        CONF_FEATURE_MODEL_REASONING,
                        default=reasoning_default,
                    )
                ] = SelectSelector(
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
                )

        if user_input is not None:
            provider_type = _provider_type_for_id(entry, provider_id)
            model_data = dict(defaults)
            model_data.update(existing_model)
            model_data.update(
                {
                    CONF_FEATURE_MODEL_NAME: user_input.get(CONF_FEATURE_MODEL_NAME),
                    CONF_FEATURE_MODEL_TEMPERATURE: user_input.get(
                        CONF_FEATURE_MODEL_TEMPERATURE
                    ),
                    CONF_FEATURE_MODEL_KEEPALIVE: user_input.get(
                        CONF_FEATURE_MODEL_KEEPALIVE
                    ),
                    CONF_FEATURE_MODEL_CONTEXT_SIZE: user_input.get(
                        CONF_FEATURE_MODEL_CONTEXT_SIZE
                    ),
                    CONF_FEATURE_MODEL_REASONING: _normalize_reasoning(
                        user_input.get(CONF_FEATURE_MODEL_REASONING)
                    ),
                }
            )
            model_data = {k: v for k, v in model_data.items() if v is not None}
            fallback_ids = user_input.get(CONF_FEATURE_FALLBACK_PROVIDER_IDS) or []
            if fallback_ids and not isinstance(fallback_ids, list):
                fallback_ids = [str(fallback_ids)]
            fallback_ids = [
                fid
                for fid in fallback_ids
                if any(opt["value"] == fid for opt in fallback_opts)
            ]

            payload = self._feature_payload(
                feature_type,
                provider_id,
                model_data,
                existing_config,
                fallback_provider_ids=fallback_ids,
            )
            self.hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
                entry, subentry, data=payload, title=FEATURE_DEFS[feature_type]["name"]
            )
            self._pending_provider_id = None
            self._schedule_reload()
            if self._setup_mode:
                return await self._advance_setup()
            return self.async_abort(reason="reconfigure_successful")

        return self.async_show_form(
            step_id="feature_model",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "provider_notice": provider_notice,
                "feature_name": FEATURE_DEFS[feature_type]["name"],
            },
        )

    async def async_step_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the conversation feature step."""
        return await self._async_step_feature("conversation", user_input)

    async def async_step_camera_image_analysis(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the camera image analysis feature step."""
        return await self._async_step_feature("camera_image_analysis", user_input)

    async def async_step_conversation_summary(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the conversation summary feature step."""
        return await self._async_step_feature("conversation_summary", user_input)

    async def async_step_database(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the database configuration step."""
        entry = self._get_entry()
        current_subentry = next(
            (
                v
                for v in entry.subentries.values()
                if v.subentry_type == SUBENTRY_TYPE_DATABASE
            ),
            None,
        )

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
            options = dict(current_subentry.data)

        schema = vol.Schema(
            {
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
        )
        if user_input is None:
            return self.async_show_form(step_id="database", data_schema=schema)

        errors: dict[str, str] = {}
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

        if errors:
            return self.async_show_form(
                step_id="database",
                data_schema=schema,
                errors=errors,
            )

        if current_subentry is None:
            subentry = ConfigSubentry(
                subentry_type=SUBENTRY_TYPE_DATABASE,
                title="Database",
                unique_id=f"{entry.entry_id}_database",
                data=MappingProxyType(user_input),
            )
            self.hass.config_entries.async_add_subentry(entry, subentry)
        else:
            self.hass.config_entries.async_update_subentry(  # type: ignore[attr-defined]
                entry, current_subentry, data=user_input, title="Database"
            )
        self._schedule_reload()

        if self._basic_mode:
            return await self.async_step_setup_status()
        if not _provider_options(entry, "chat"):
            return await self.async_step_instructions()
        return self.async_abort(reason="setup_complete")

    async def async_step_instructions(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the instructions step."""
        if user_input is None:
            return self.async_show_form(
                step_id="instructions", data_schema=vol.Schema({})
            )
        return self.async_abort(reason="setup_complete")
