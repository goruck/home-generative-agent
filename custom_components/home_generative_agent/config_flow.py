"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import async_timeout
import httpx
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE,
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_DB_URI,
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION,
    CONF_GEMINI_API_KEY,
    CONF_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_MAX_TOKENS_IN_CONTEXT,
    CONF_NOTIFY_SERVICE,
    CONF_OLLAMA_CHAT_CONTEXT_SIZE,
    CONF_OLLAMA_CHAT_KEEPALIVE,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OLLAMA_CHAT_URL,
    CONF_OLLAMA_REASONING,
    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE,
    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE,
    CONF_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OLLAMA_SUMMARIZATION_URL,
    CONF_OLLAMA_URL,
    CONF_OLLAMA_VLM,
    CONF_OLLAMA_VLM_CONTEXT_SIZE,
    CONF_OLLAMA_VLM_KEEPALIVE,
    CONF_OLLAMA_VLM_URL,
    CONF_OPENAI_CHAT_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL,
    CONF_OPENAI_VLM,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    DOMAIN,
    HTTP_STATUS_BAD_REQUEST,
    KEEPALIVE_MAX_SECONDS,
    KEEPALIVE_MIN_SECONDS,
    KEEPALIVE_SENTINEL,
    LLM_HASS_API_NONE,
    MODEL_CATEGORY_SPECS,
    RECOMMENDED_CHAT_MODEL_PROVIDER,
    RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    RECOMMENDED_DB_URI,
    RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    RECOMMENDED_FACE_RECOGNITION,
    RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
    RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    RECOMMENDED_OLLAMA_CHAT_MODEL,
    RECOMMENDED_OLLAMA_CHAT_URL,
    RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    RECOMMENDED_OLLAMA_REASONING,
    RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
    RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    RECOMMENDED_OLLAMA_SUMMARIZATION_URL,
    RECOMMENDED_OLLAMA_VLM,
    RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    RECOMMENDED_OLLAMA_VLM_URL,
    RECOMMENDED_OPENAI_CHAT_MODEL,
    RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    RECOMMENDED_OPENAI_VLM,
    RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VLM_PROVIDER,
    RECOMMENDED_VLM_TEMPERATURE,
    VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY,
    VIDEO_ANALYZER_MODE_DISABLE,
    VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY,
)
from .core.utils import (
    CannotConnectError,
    InvalidAuthError,
    ensure_http_url,
    hash_pin,
    list_mobile_notify_services,
    ollama_url_for_category,
    validate_db_uri,
    validate_face_api_url,
    validate_gemini_key,
    validate_ollama_url,
    validate_openai_key,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(CONF_GEMINI_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Optional(
            CONF_OLLAMA_URL,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_CHAT_URL,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_VLM_URL,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_SUMMARIZATION_URL,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_FACE_API_URL,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Required(
            CONF_DB_URI,
            description={"suggested_value": RECOMMENDED_DB_URI},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
    },
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CRITICAL_ACTION_PIN_ENABLED: True,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
    CONF_FACE_RECOGNITION: RECOMMENDED_FACE_RECOGNITION,
    CONF_CHAT_MODEL_PROVIDER: RECOMMENDED_CHAT_MODEL_PROVIDER,
    CONF_CHAT_MODEL_TEMPERATURE: RECOMMENDED_CHAT_MODEL_TEMPERATURE,
    CONF_VLM_PROVIDER: RECOMMENDED_VLM_PROVIDER,
    CONF_VLM_TEMPERATURE: RECOMMENDED_VLM_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_PROVIDER: RECOMMENDED_SUMMARIZATION_MODEL_PROVIDER,
    CONF_SUMMARIZATION_MODEL_TEMPERATURE: RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_EMBEDDING_MODEL_PROVIDER: RECOMMENDED_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_CHAT_MODEL: RECOMMENDED_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL: RECOMMENDED_OPENAI_CHAT_MODEL,
    CONF_OLLAMA_VLM: RECOMMENDED_OLLAMA_VLM,
    CONF_OPENAI_VLM: RECOMMENDED_OPENAI_VLM,
    CONF_OLLAMA_SUMMARIZATION_MODEL: RECOMMENDED_OLLAMA_SUMMARIZATION_MODEL,
    CONF_OPENAI_SUMMARIZATION_MODEL: RECOMMENDED_OPENAI_SUMMARIZATION_MODEL,
    CONF_OLLAMA_REASONING: RECOMMENDED_OLLAMA_REASONING,
    CONF_MANAGE_CONTEXT_WITH_TOKENS: RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_TOKENS_IN_CONTEXT: RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    CONF_MAX_MESSAGES_IN_CONTEXT: RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
    CONF_OLLAMA_CHAT_KEEPALIVE: RECOMMENDED_OLLAMA_CHAT_KEEPALIVE,
    CONF_OLLAMA_VLM_KEEPALIVE: RECOMMENDED_OLLAMA_VLM_KEEPALIVE,
    CONF_OLLAMA_SUMMARIZATION_KEEPALIVE: RECOMMENDED_OLLAMA_SUMMARIZATION_KEEPALIVE,
    CONF_OLLAMA_CHAT_CONTEXT_SIZE: RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    CONF_OLLAMA_VLM_CONTEXT_SIZE: RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE: RECOMMENDED_OLLAMA_CONTEXT_SIZE,
    CONF_OLLAMA_CHAT_URL: RECOMMENDED_OLLAMA_CHAT_URL,
    CONF_OLLAMA_VLM_URL: RECOMMENDED_OLLAMA_VLM_URL,
    CONF_OLLAMA_SUMMARIZATION_URL: RECOMMENDED_OLLAMA_SUMMARIZATION_URL,
}


_PROVIDER_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "openai": (CONF_API_KEY,),
    "gemini": (CONF_GEMINI_API_KEY,),
}

# ---------------------------
# Helpers
# ---------------------------


def _provider_is_configured(
    opts: Mapping[str, Any], provider: str, cat: str | None = None
) -> bool:
    if provider == "ollama":
        return bool(
            ollama_url_for_category(opts, cat or "", fallback=opts.get(CONF_OLLAMA_URL))
        )
    required = _PROVIDER_REQUIRED_KEYS.get(provider, ())
    for k in required:
        v = str(opts.get(k, "") or "").strip()
        if not v:
            return False
    return True


def _pick_configured_provider_for_cat(
    opts: Mapping[str, Any], cat: str, preferred: str | None = None
) -> str:
    """
    Choose a provider for the category that actually has credentials/URL.

    Priority:
      1) preferred if configured
      2) category's recommended_provider if configured
      3) first configured in providers list order
      4) fallback to preferred or recommended_provider (even if not configured)
    """
    spec = MODEL_CATEGORY_SPECS[cat]
    providers = list(spec["providers"].keys())
    rec = spec["recommended_provider"]
    # 1) preferred
    if preferred and _provider_is_configured(opts, preferred, cat):
        return preferred
    # 2) recommended for this category
    if _provider_is_configured(opts, rec, cat):
        return rec
    # 3) first configured by order
    for p in providers:
        if _provider_is_configured(opts, p, cat):
            return p
    # 4) nothing configured; keep stable
    return preferred or rec


def _auto_select_configured_providers(opts: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of opts with configured providers."""
    new_opts = dict(opts)
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        key = spec["provider_key"]
        chosen = new_opts.get(key, spec["recommended_provider"])
        fixed = _pick_configured_provider_for_cat(new_opts, cat, chosen)
        new_opts[key] = fixed
    return new_opts


def _get_str(src: Mapping[str, Any], key: str) -> str:
    """Get a trimmed string from a mapping (missing -> '')."""
    return str(src.get(key, "") or "").strip()


def _model_option_key(cat: str, provider: str) -> str:
    """Return the stable option key for the model field of (category, provider)."""
    spec = MODEL_CATEGORY_SPECS[cat]
    key = spec.get("model_keys", {}).get(provider)
    if key:
        return key
    return f"model__{cat}__{provider}"


def _prune_irrelevant_model_fields(opts: Mapping[str, Any]) -> dict[str, Any]:
    """
    Drop model fields that don't match the chosen provider for each category.

    Also drop temperatures when recommended is enabled.
    """
    pruned = dict(opts)

    # Strip temps and keepalive if recommended
    if pruned.get(CONF_RECOMMENDED):
        for cat, spec in MODEL_CATEGORY_SPECS.items():
            temp_key = spec.get("temperature_key")
            if temp_key:
                pruned.pop(temp_key, None)
            keep_key = _ollama_keepalive_key_for_cat(cat)
            if keep_key:
                pruned.pop(keep_key, None)
            context_key = _ollama_context_size_key_for_cat(cat)
            if context_key:
                pruned.pop(context_key, None)

    # Remove model keys for providers not selected
    for cat, spec in MODEL_CATEGORY_SPECS.items():
        selected = pruned.get(spec["provider_key"])
        for provider in spec["providers"]:
            key = _model_option_key(cat, provider)
            if provider != selected:
                pruned.pop(key, None)

        # Remove keepalive if provider is not ollama
        keep_key = _ollama_keepalive_key_for_cat(cat)
        if keep_key and selected != "ollama":
            pruned.pop(keep_key, None)

        # Remove context size if provider is not ollama
        context_key = _ollama_context_size_key_for_cat(cat)
        if context_key and selected != "ollama":
            pruned.pop(context_key, None)

    return pruned


def _get_selected_provider(options: Mapping[str, Any], cat: str) -> str | None:
    spec = MODEL_CATEGORY_SPECS[cat]
    return options.get(spec["provider_key"], spec["recommended_provider"])


def _provider_selector_config(cat: str) -> SelectSelectorConfig:
    providers = list(MODEL_CATEGORY_SPECS[cat]["providers"].keys())
    return SelectSelectorConfig(
        options=[SelectOptionDict(label=p.title(), value=p) for p in providers],
        mode=SelectSelectorMode.DROPDOWN,
        sort=False,
        custom_value=False,
    )


def _coerce_keepalive_value(value: Any) -> int:
    """Accept -1, 0, or any integer in range."""
    try:
        seconds = int(value)
    except (TypeError, ValueError) as err:
        msg = f"invalid keepalive: {value!r}"
        raise ValueError(msg) from err

    if seconds == KEEPALIVE_SENTINEL:
        return KEEPALIVE_SENTINEL
    if not (KEEPALIVE_MIN_SECONDS <= seconds <= KEEPALIVE_MAX_SECONDS):
        msg = f"keepalive out of range: {seconds}"
        raise ValueError(msg)
    return seconds


def _ollama_keepalive_key_for_cat(cat: str) -> str | None:
    """Return the keepalive option key for a given model category."""
    if cat == "chat":
        return CONF_OLLAMA_CHAT_KEEPALIVE
    if cat == "vlm":
        return CONF_OLLAMA_VLM_KEEPALIVE
    if cat == "summarization":
        return CONF_OLLAMA_SUMMARIZATION_KEEPALIVE
    return None


def _ollama_context_size_key_for_cat(cat: str) -> str | None:
    """Return the context size option key for a given model category."""
    if cat == "chat":
        return CONF_OLLAMA_CHAT_CONTEXT_SIZE
    if cat == "vlm":
        return CONF_OLLAMA_VLM_CONTEXT_SIZE
    if cat == "summarization":
        return CONF_OLLAMA_SUMMARIZATION_CONTEXT_SIZE
    return None


async def _fetch_ollama_models(
    hass: HomeAssistant, base_url: str, timeout_s: float = 5.0
) -> list[str]:
    """Return a sorted list of models installed on the given Ollama host."""
    client = get_async_client(hass)
    base_url = ensure_http_url(base_url).rstrip("/") + "/"
    try:
        async with async_timeout.timeout(timeout_s):
            resp = await client.get(urljoin(base_url, "api/tags"))
    except (TimeoutError, httpx.HTTPError) as err:
        LOGGER.debug("Unable to reach Ollama at %s: %s", base_url, err)
        return []

    if resp.status_code >= HTTP_STATUS_BAD_REQUEST:
        LOGGER.debug(
            "Ollama responded with HTTP %s when fetching tags from %s",
            resp.status_code,
            base_url,
        )
        return []

    try:
        payload = resp.json()
    except ValueError:
        LOGGER.debug("Invalid JSON when fetching Ollama tags from %s", base_url)
        return []

    names: list[str] = []
    for model in payload.get("models", []):
        name = model.get("model") or model.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    # Preserve order but drop duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    return sorted(deduped)


async def _ollama_models_by_category(
    hass: HomeAssistant, opts: Mapping[str, Any]
) -> dict[str, list[str]]:
    """
    Fetch installed Ollama models for each configured category URL.

    Categories that share a URL reuse the same response.
    """
    url_for_cat: dict[str, str] = {}
    for cat in MODEL_CATEGORY_SPECS:
        url = ollama_url_for_category(opts, cat, fallback=opts.get(CONF_OLLAMA_URL))
        if url:
            url_for_cat[cat] = url

    if not url_for_cat:
        return {}

    unique_urls = set(url_for_cat.values())
    tasks = {url: _fetch_ollama_models(hass, url) for url in unique_urls}
    results = await asyncio.gather(*tasks.values())
    models_by_url = dict(zip(tasks.keys(), results, strict=False))
    return {cat: models_by_url.get(url, []) for cat, url in url_for_cat.items()}


def _model_selector_config(
    cat: str, provider: str, ollama_models: Mapping[str, list[str]] | None
) -> SelectSelectorConfig:
    models = MODEL_CATEGORY_SPECS[cat]["providers"].get(provider, [])
    if provider == "ollama" and ollama_models is not None:
        models = ollama_models.get(cat) or models
    return SelectSelectorConfig(
        options=[SelectOptionDict(label=m, value=m) for m in models],
        mode=SelectSelectorMode.DROPDOWN,
        sort=False,
        custom_value=True,
    )


async def _schema_for(hass: HomeAssistant, opts: Mapping[str, Any]) -> VolDictType:
    """Generate the options schema."""
    # Coerce provider selections to ones that are actually configured
    opts = _auto_select_configured_providers(opts)

    hass_apis = [SelectOptionDict(label="No control", value=LLM_HASS_API_NONE)] + [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    video_analyzer_mode_opts: list[SelectOptionDict] = [
        SelectOptionDict(label="Disable", value=VIDEO_ANALYZER_MODE_DISABLE),
        SelectOptionDict(
            label="Notify on anomaly", value=VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY
        ),
        SelectOptionDict(
            label="Always notify", value=VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY
        ),
    ]

    schema: VolDictType = {
        vol.Optional(
            CONF_API_KEY,
            description={"suggested_value": opts.get(CONF_API_KEY)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(
            CONF_GEMINI_API_KEY,
            description={"suggested_value": opts.get(CONF_GEMINI_API_KEY)},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(
            CONF_OLLAMA_URL,
            description={"suggested_value": (opts.get(CONF_OLLAMA_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_CHAT_URL,
            description={"suggested_value": (opts.get(CONF_OLLAMA_CHAT_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_VLM_URL,
            description={"suggested_value": (opts.get(CONF_OLLAMA_VLM_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_OLLAMA_SUMMARIZATION_URL,
            description={"suggested_value": (opts.get(CONF_OLLAMA_SUMMARIZATION_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_FACE_API_URL,
            description={"suggested_value": (opts.get(CONF_FACE_API_URL))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Required(
            CONF_DB_URI,
            description={"suggested_value": (opts.get(CONF_DB_URI))},
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": opts.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": opts.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
    }

    schema[
        vol.Optional(
            CONF_CRITICAL_ACTION_PIN_ENABLED,
            description={
                "suggested_value": opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True)
            },
            default=opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True),
        )
    ] = BooleanSelector()

    pin_enabled = opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True)
    if pin_enabled:
        schema[
            vol.Optional(
                CONF_CRITICAL_ACTION_PIN,
                description={
                    "suggested_value": "",
                    "placeholder": "Set/replace PIN for critical actions",
                },
            )
        ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

    schema[
        vol.Optional(
            CONF_VIDEO_ANALYZER_MODE,
            description={"suggested_value": opts.get(CONF_VIDEO_ANALYZER_MODE)},
            default=RECOMMENDED_VIDEO_ANALYZER_MODE,
        )
    ] = SelectSelector(SelectSelectorConfig(options=video_analyzer_mode_opts))

    video_analyzer_mode = opts.get(
        CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
    )
    if video_analyzer_mode != VIDEO_ANALYZER_MODE_DISABLE:
        # Show Face Recognition toggle only when analyzer is enabled.
        schema[
            vol.Optional(
                CONF_FACE_RECOGNITION,
                description={"suggested_value": opts.get(CONF_FACE_RECOGNITION)},
                default=RECOMMENDED_FACE_RECOGNITION,
            )
        ] = BooleanSelector()

        # Conditional notify service select.
        mobile_opts = list_mobile_notify_services(hass)
        if mobile_opts:
            schema[
                vol.Optional(
                    CONF_NOTIFY_SERVICE,
                    description={"suggested_value": opts.get(CONF_NOTIFY_SERVICE)},
                    default=opts.get(CONF_NOTIFY_SERVICE, mobile_opts[0]),
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=s.replace("notify.", ""), value=s)
                        for s in mobile_opts
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            )

    schema[
        vol.Required(
            CONF_RECOMMENDED,
            description={"suggested_value": opts.get(CONF_RECOMMENDED)},
            default=opts.get(CONF_RECOMMENDED, False),
        )
    ] = bool

    # Recommended ON → no providers/models/temps
    if opts.get(CONF_RECOMMENDED, False):
        return schema

    selected_is_ollama = any(
        _get_selected_provider(opts, cat) == "ollama" for cat in MODEL_CATEGORY_SPECS
    )
    ollama_models = (
        await _ollama_models_by_category(hass, opts) if selected_is_ollama else {}
    )

    schema[
        vol.Optional(
            CONF_OLLAMA_REASONING,
            description={"suggested_value": opts.get(CONF_OLLAMA_REASONING)},
            default=RECOMMENDED_OLLAMA_REASONING,
        )
    ] = BooleanSelector()

    context_mgmt_modes = [
        SelectOptionDict(label="Use tokens", value="true"),
        SelectOptionDict(label="Use messages", value="false"),
    ]

    schema[
        vol.Optional(
            CONF_MANAGE_CONTEXT_WITH_TOKENS,
            description={
                "suggested_value": opts.get(CONF_MANAGE_CONTEXT_WITH_TOKENS, "true")
            },
            default=RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
        )
    ] = SelectSelector(
        SelectSelectorConfig(
            options=context_mgmt_modes,
            mode=SelectSelectorMode.DROPDOWN,
            sort=False,
            custom_value=False,
        )
    )

    schema[
        vol.Optional(
            CONF_MAX_TOKENS_IN_CONTEXT,
            description={"suggested_value": opts.get(CONF_MAX_TOKENS_IN_CONTEXT)},
            default=RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
        )
    ] = NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1))

    schema[
        vol.Optional(
            CONF_MAX_MESSAGES_IN_CONTEXT,
            description={"suggested_value": opts.get(CONF_MAX_MESSAGES_IN_CONTEXT)},
            default=RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
        )
    ] = NumberSelector(NumberSelectorConfig(min=15, max=240, step=1))

    for cat, spec in MODEL_CATEGORY_SPECS.items():
        provider_key = spec["provider_key"]

        # Provider select
        schema[
            vol.Optional(
                provider_key,
                description={"suggested_value": opts.get(provider_key)},
                default=opts.get(provider_key, spec["recommended_provider"]),
            )
        ] = SelectSelector(_provider_selector_config(cat))

        # Model select for chosen provider
        selected_provider = _get_selected_provider(opts, cat)
        if selected_provider:
            model_key = _model_option_key(cat, selected_provider)
            default_model = spec.get("recommended_models", {}).get(selected_provider)
            schema[
                vol.Optional(
                    model_key,
                    description={"suggested_value": opts.get(model_key)},
                    default=opts.get(model_key, default_model),
                )
            ] = SelectSelector(
                _model_selector_config(cat, selected_provider, ollama_models)
            )

        # Ollama keepalive (only when provider is ollama)
        if selected_provider == "ollama":
            keep_key = _ollama_keepalive_key_for_cat(cat)
            if keep_key is not None:
                schema[
                    vol.Optional(
                        keep_key,
                        description={"suggested_value": opts.get(keep_key)},
                        default=opts.get(keep_key, 5 * 60),
                    )
                ] = NumberSelector(
                    NumberSelectorConfig(
                        min=KEEPALIVE_SENTINEL,
                        max=KEEPALIVE_MAX_SECONDS,
                        step=1,
                    )
                )

        # Ollama context size (only when provider is ollama)
        if selected_provider == "ollama":
            context_key = _ollama_context_size_key_for_cat(cat)
            if context_key:
                schema[
                    vol.Optional(
                        context_key,
                        description={"suggested_value": opts.get(context_key)},
                        default=opts.get(context_key, RECOMMENDED_OLLAMA_CONTEXT_SIZE),
                    )
                ] = NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1))

        # Temperature (if used by this category)
        temp_key = spec.get("temperature_key")
        if temp_key:
            schema[
                vol.Optional(
                    temp_key,
                    description={"suggested_value": opts.get(temp_key)},
                    default=spec.get("recommended_temperature", 1.0),
                )
            ] = NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05))

    return schema


@dataclass(frozen=True)
class _SecretSpec:
    field: str
    validator: Callable[[Any, str], Awaitable[None]]
    label: str


# ---------------------------
# Config Flow
# ---------------------------


class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""

    VERSION = 1

    async def _validate_present(
        self,
        hass: HomeAssistant,
        value: str,
        validator: Callable[[HomeAssistant, str], Any],
        log_label: str,
    ) -> str | None:
        """Run validator only when value is non-empty."""
        if not value:
            return None
        try:
            await validator(hass, value)
        except InvalidAuthError:
            return "invalid_auth"
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception during %s validation", log_label)
            return "unknown"
        else:
            return None

    async def _run_validations_user(
        self, data: Mapping[str, Any]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Validate inputs for the user step and normalize results."""
        errors: dict[str, str] = {}
        normalized: dict[str, Any] = dict(data)

        vals = {
            CONF_API_KEY: _get_str(data, CONF_API_KEY),
            CONF_OLLAMA_URL: _get_str(data, CONF_OLLAMA_URL),
            CONF_OLLAMA_CHAT_URL: _get_str(data, CONF_OLLAMA_CHAT_URL),
            CONF_OLLAMA_VLM_URL: _get_str(data, CONF_OLLAMA_VLM_URL),
            CONF_OLLAMA_SUMMARIZATION_URL: _get_str(
                data, CONF_OLLAMA_SUMMARIZATION_URL
            ),
            CONF_GEMINI_API_KEY: _get_str(data, CONF_GEMINI_API_KEY),
            CONF_DB_URI: _get_str(data, CONF_DB_URI),
            CONF_FACE_API_URL: _get_str(data, CONF_FACE_API_URL),
        }

        # Ordered, table-driven validation; short-circuits on first error.
        for key, validator, label in (
            (CONF_API_KEY, validate_openai_key, "OpenAI"),
            (CONF_OLLAMA_URL, validate_ollama_url, "Ollama"),
            (CONF_OLLAMA_CHAT_URL, validate_ollama_url, "Ollama (chat)"),
            (CONF_OLLAMA_VLM_URL, validate_ollama_url, "Ollama (vlm)"),
            (
                CONF_OLLAMA_SUMMARIZATION_URL,
                validate_ollama_url,
                "Ollama (summarization)",
            ),
            (CONF_GEMINI_API_KEY, validate_gemini_key, "Gemini"),
            (CONF_DB_URI, validate_db_uri, "Database URI"),
            (CONF_FACE_API_URL, validate_face_api_url, "Face Recognition API"),
        ):
            code = await self._validate_present(self.hass, vals[key], validator, label)
            if code:
                errors["base"] = code
                break

        # Normalize URLs only on success.
        if not errors and vals[CONF_OLLAMA_URL]:
            normalized[CONF_OLLAMA_URL] = ensure_http_url(vals[CONF_OLLAMA_URL])
        if not errors and vals[CONF_OLLAMA_CHAT_URL]:
            normalized[CONF_OLLAMA_CHAT_URL] = ensure_http_url(
                vals[CONF_OLLAMA_CHAT_URL]
            )
        if not errors and vals[CONF_OLLAMA_VLM_URL]:
            normalized[CONF_OLLAMA_VLM_URL] = ensure_http_url(vals[CONF_OLLAMA_VLM_URL])
        if not errors and vals[CONF_OLLAMA_SUMMARIZATION_URL]:
            normalized[CONF_OLLAMA_SUMMARIZATION_URL] = ensure_http_url(
                vals[CONF_OLLAMA_SUMMARIZATION_URL]
            )
        if not errors and vals[CONF_FACE_API_URL]:
            normalized[CONF_FACE_API_URL] = ensure_http_url(vals[CONF_FACE_API_URL])

        # Drop empties so defaults can apply later.
        for key in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_OLLAMA_CHAT_URL,
            CONF_OLLAMA_VLM_URL,
            CONF_OLLAMA_SUMMARIZATION_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if not vals[key]:
                normalized.pop(key, None)

        return errors, normalized

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors, normalized = await self._run_validations_user(user_input)
        if errors:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
            )

        opts = _auto_select_configured_providers({**RECOMMENDED_OPTIONS, **normalized})
        opts = _prune_irrelevant_model_fields(opts)

        return self.async_create_entry(
            title="Home Generative Agent",
            data=normalized,
            options=opts,
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return HomeGenerativeAgentOptionsFlow(config_entry)


# ---------------------------
# Options Flow
# ---------------------------


class HomeGenerativeAgentOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Home Generative Agent."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )
        self._last_providers = self._extract_provider_state(config_entry.options)
        self._last_analyzer_mode = config_entry.options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )

    # ---- helpers ----

    def _all_category_provider_keys(self) -> tuple[str, ...]:
        return tuple(spec["provider_key"] for spec in MODEL_CATEGORY_SPECS.values())

    def _providers_changed(
        self, prev: Mapping[str, Any], current: Mapping[str, Any]
    ) -> bool:
        return any(
            prev.get(k) != current.get(k) for k in self._all_category_provider_keys()
        )

    def _extract_provider_state(self, opts: Mapping[str, Any]) -> dict[str, Any]:
        keys = self._all_category_provider_keys()
        return {k: opts.get(k) for k in keys}

    def _apply_recommended_defaults(self, opts: Mapping[str, Any]) -> dict[str, Any]:
        """Reset only model-related settings to recommended values."""
        updated: dict[str, Any] = dict(opts)

        # Per-category provider/model/temp + prune provider-specific extras
        for cat, spec in MODEL_CATEGORY_SPECS.items():
            # Pick a configured provider
            rec_provider = _pick_configured_provider_for_cat(
                updated, cat, spec["recommended_provider"]
            )
            provider_key = spec["provider_key"]
            updated[provider_key] = rec_provider

            # Clear all model keys for this category, then set recommended
            for prov in spec["providers"]:
                updated.pop(_model_option_key(cat, prov), None)

            rec_model = spec.get("recommended_models", {}).get(rec_provider)
            if rec_model:
                updated[_model_option_key(cat, rec_provider)] = rec_model

            temp_key = spec.get("temperature_key")
            if temp_key is not None:
                updated[temp_key] = spec.get("recommended_temperature", 1.0)

            # Provider-specific extras should not persist when switching to recommended
            keep_key = _ollama_keepalive_key_for_cat(cat)
            if keep_key:
                updated.pop(keep_key, None)
            ctx_key = _ollama_context_size_key_for_cat(cat)
            if ctx_key:
                updated.pop(ctx_key, None)

        # Global model-policy defaults (cross-category, model-related)
        updated[CONF_OLLAMA_REASONING] = RECOMMENDED_OLLAMA_REASONING
        updated[CONF_MANAGE_CONTEXT_WITH_TOKENS] = (
            RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS
        )
        updated[CONF_MAX_TOKENS_IN_CONTEXT] = RECOMMENDED_MAX_TOKENS_IN_CONTEXT
        updated[CONF_MAX_MESSAGES_IN_CONTEXT] = RECOMMENDED_MAX_MESSAGES_IN_CONTEXT

        return updated

    def _base_options_with_entry_data(self) -> dict[str, Any]:
        """Start from current options, overlaying any setup-time data for visibility."""
        options = dict(self.config_entry.options)
        for k in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_OLLAMA_CHAT_URL,
            CONF_OLLAMA_VLM_URL,
            CONF_OLLAMA_SUMMARIZATION_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if k not in options and self.config_entry.data.get(k):
                options[k] = self.config_entry.data[k]
        return options

    async def _maybe_edit_face_recognition_url(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply face recog URL when present; return error code or None."""
        if user_input is None or CONF_FACE_API_URL not in user_input:
            return None

        raw = _get_str(user_input, CONF_FACE_API_URL)
        if not raw:
            options.pop(CONF_FACE_API_URL, None)
            return None

        try:
            await validate_face_api_url(self.hass, raw)
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating face recognition api URL")
            return "unknown"

        options[CONF_FACE_API_URL] = ensure_http_url(raw)
        return None

    async def _maybe_edit_ollama_urls(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply Ollama URLs when present; return error code or None."""
        if not user_input:
            return None

        for field in (
            CONF_OLLAMA_URL,
            CONF_OLLAMA_CHAT_URL,
            CONF_OLLAMA_VLM_URL,
            CONF_OLLAMA_SUMMARIZATION_URL,
        ):
            if field not in user_input:
                continue

            raw = _get_str(user_input, field)
            if not raw:
                options.pop(field, None)
                continue

            try:
                await validate_ollama_url(self.hass, raw)
            except CannotConnectError:
                return "cannot_connect"
            except Exception:
                LOGGER.exception(
                    "Unexpected exception validating Ollama URL field %s", field
                )
                return "unknown"

            options[field] = ensure_http_url(raw)

        return None

    async def _maybe_edit_db_uri(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply DB URI when present; return error code or None."""
        if user_input is None or CONF_DB_URI not in user_input:
            return None

        raw = _get_str(user_input, CONF_DB_URI)
        if not raw:
            options.pop(CONF_DB_URI, None)
            return None

        try:
            await validate_db_uri(self.hass, raw)
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception validating DB URI")
            return "unknown"

        options[CONF_DB_URI] = raw
        return None

    async def _maybe_edit_secret(
        self,
        spec: _SecretSpec,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """Validate/apply a secret field when present; return error code or None."""
        if user_input is None or spec.field not in user_input:
            return None

        raw = _get_str(user_input, spec.field)
        if not raw:
            options.pop(spec.field, None)
            return None

        try:
            await spec.validator(self.hass, raw)
        except InvalidAuthError:
            return "invalid_auth"
        except CannotConnectError:
            return "cannot_connect"
        except Exception:
            LOGGER.exception("Unexpected exception in %s validation", spec.label)
            return "unknown"

        options[spec.field] = raw
        return None

    def _maybe_edit_pin(
        self, options: dict[str, Any], user_input: Mapping[str, Any] | None
    ) -> str | None:
        """Hash and store the critical-action PIN if provided."""
        if user_input is None:
            return None

        pin_enabled = user_input.get(
            CONF_CRITICAL_ACTION_PIN_ENABLED,
            options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, True),
        )
        options[CONF_CRITICAL_ACTION_PIN_ENABLED] = pin_enabled

        if not pin_enabled:
            options.pop(CONF_CRITICAL_ACTION_PIN, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_HASH, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_SALT, None)
            return None

        if CONF_CRITICAL_ACTION_PIN not in user_input:
            return None

        raw = _get_str(user_input, CONF_CRITICAL_ACTION_PIN)
        options.pop(CONF_CRITICAL_ACTION_PIN, None)
        if not raw:
            options.pop(CONF_CRITICAL_ACTION_PIN_HASH, None)
            options.pop(CONF_CRITICAL_ACTION_PIN_SALT, None)
            return None

        if (
            not raw.isdigit()
            or not CRITICAL_PIN_MIN_LEN <= len(raw) <= CRITICAL_PIN_MAX_LEN
        ):
            return "invalid_pin"

        hashed, salt = hash_pin(raw)
        options[CONF_CRITICAL_ACTION_PIN_HASH] = hashed
        options[CONF_CRITICAL_ACTION_PIN_SALT] = salt
        return None

    def _maybe_edit_keepalives(
        self,
        options: dict[str, Any],
        user_input: Mapping[str, Any] | None,
    ) -> str | None:
        """
        Validate/normalize Ollama keepalive fields if present.

        Return error code or None.
        """
        if not user_input:
            return None

        # Only these categories have keepalive
        for cat in ("chat", "vlm", "summarization"):
            keep_key = _ollama_keepalive_key_for_cat(cat)
            if keep_key and keep_key in user_input:
                try:
                    options[keep_key] = _coerce_keepalive_value(user_input[keep_key])
                except (ValueError, TypeError):
                    # Keep message key stable; HA will show a generic error at top.
                    return "invalid_keepalive"

        return None

    def _drop_empty_fields(self, final_options: dict[str, Any]) -> None:
        """Remove empty strings for fields to avoid storing empties."""
        for k in (
            CONF_API_KEY,
            CONF_OLLAMA_URL,
            CONF_OLLAMA_CHAT_URL,
            CONF_OLLAMA_VLM_URL,
            CONF_OLLAMA_SUMMARIZATION_URL,
            CONF_GEMINI_API_KEY,
            CONF_DB_URI,
            CONF_FACE_API_URL,
        ):
            if not _get_str(final_options, k):
                final_options.pop(k, None)

    def _cleanup_none_llm_api(self, options: dict[str, Any]) -> None:
        """Remove the 'none' sentinel so options omit the key when unset."""
        if options.get(CONF_LLM_HASS_API) == LLM_HASS_API_NONE:
            options.pop(CONF_LLM_HASS_API, None)

    def _schema_changes_since_last(
        self, options: Mapping[str, Any]
    ) -> tuple[bool, bool, bool]:
        """Detect changes that require re-render."""
        recommended_now = options.get(CONF_RECOMMENDED, False)
        recommended_changed = recommended_now != self.last_rendered_recommended
        provider_changed = self._providers_changed(self._last_providers, options)
        analyzer_now = options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )
        analyzer_changed = analyzer_now != self._last_analyzer_mode
        return recommended_changed, provider_changed, analyzer_changed

    def _remember_schema_baseline(self, options: Mapping[str, Any]) -> None:
        """Record the state we used for last-rendered schema."""
        self.last_rendered_recommended = bool(options.get(CONF_RECOMMENDED, False))
        self._last_providers = self._extract_provider_state(options)
        self._last_analyzer_mode = options.get(
            CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
        )

    # ---- main step ----

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the options flow init step."""
        options = self._base_options_with_entry_data()

        # First render
        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(await _schema_for(self.hass, options)),
            )

        # Merge new input for non-validated fields
        options.update(user_input or {})
        errors: dict[str, str] = {}

        # Field-specific edits with validation/normalization
        err = await self._maybe_edit_ollama_urls(options, user_input)
        if not err:
            err = await self._maybe_edit_db_uri(options, user_input)
        if not err:
            err = await self._maybe_edit_face_recognition_url(options, user_input)
        if not err:
            err = await self._maybe_edit_secret(
                _SecretSpec(CONF_GEMINI_API_KEY, validate_gemini_key, "Gemini Options"),
                options,
                user_input,
            )
        if not err:
            err = await self._maybe_edit_secret(
                _SecretSpec(CONF_API_KEY, validate_openai_key, "OpenAI Options"),
                options,
                user_input,
            )
        if not err:
            err = self._maybe_edit_pin(options, user_input)
        if not err:
            err = self._maybe_edit_keepalives(options, user_input)
        if err:
            errors["base"] = err

        if errors:
            # Re-render with the same options and show errors
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(await _schema_for(self.hass, options)),
                errors=errors,
            )

        # Handle schema-affecting toggles
        recommended_changed, provider_changed, analyzer_changed = (
            self._schema_changes_since_last(options)
        )

        # If Recommended turned ON → apply defaults and SAVE immediately
        if recommended_changed and options.get(CONF_RECOMMENDED, False):
            final_options = self._apply_recommended_defaults(options)
            final_options = _auto_select_configured_providers(final_options)
            final_options = _prune_irrelevant_model_fields(final_options)
            self._cleanup_none_llm_api(final_options)
            self._drop_empty_fields(final_options)
            self._remember_schema_baseline(final_options)
            return self.async_create_entry(title="", data=final_options)

        # If Recommended toggled OFF or provider/analyzer changes, re-render once
        if recommended_changed or provider_changed or analyzer_changed:
            pruned = _prune_irrelevant_model_fields(options)
            self._remember_schema_baseline(pruned)
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(await _schema_for(self.hass, pruned)),
            )

        # Finalize and save (no schema changes)
        final_options = _prune_irrelevant_model_fields(options)
        if final_options.get(CONF_RECOMMENDED, False):
            final_options = self._apply_recommended_defaults(final_options)
            final_options = _prune_irrelevant_model_fields(final_options)
        final_options = _auto_select_configured_providers(final_options)
        self._cleanup_none_llm_api(final_options)
        self._drop_empty_fields(final_options)

        return self.async_create_entry(title="", data=final_options)
