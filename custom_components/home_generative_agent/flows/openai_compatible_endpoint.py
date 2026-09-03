"""
Shared credential helpers for the OpenAI-backed provider subentry flows.

The STT and TTS provider flows configure the same two kinds of backend: the
OpenAI API (a key, or a link to an OpenAI model-provider subentry) and a local
OpenAI-compatible server (a URL plus an optional key). This module holds the
one copy of how those credentials are collected, validated, and stored so the
two flows cannot drift apart. The stored shape is::

    openai -> {"api_key": str | None, <provider_id_key>: str | None}
    local  -> {"base_url": "<...>/v1", "api_key": str | None,
               <provider_id_key>: None}

Local URLs are normalized (scheme, no trailing slash, ``/v1`` suffix) at write
time, so runtime readers use the stored value verbatim. A keyless local server
stores ``api_key: None``; the runtime substitutes a placeholder the SDK
accepts and strips the Authorization header per request.

The model-provider flow's ``openai_compatible`` type stores the same concept in
an older shape (raw URL, literal ``"none"`` key sentinel); migrating it onto
this helper is tracked in TODOS.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import SOURCE_RECONFIGURE, ConfigSubentryFlow
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from ..const import (  # noqa: TID252
    CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL,
    SUBENTRY_TYPE_MODEL_PROVIDER,
)
from ..core.utils import (  # noqa: TID252
    CannotConnectError,
    InvalidAuthError,
    normalize_openai_compatible_base_url,
    validate_openai_compatible_url,
    validate_openai_key,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.config_entries import ConfigSubentry

LOGGER = logging.getLogger(__name__)

# Value of the "reuse an OpenAI model provider" selector meaning "no, use a
# separate key typed below".
SEPARATE_KEY_OPTION = "none"


def get_entry_from_flow(flow: ConfigSubentryFlow) -> Any:
    """Return the parent config entry of a subentry flow."""
    return flow._get_entry()  # noqa: SLF001


def current_subentry(
    flow: ConfigSubentryFlow, subentry_type: str
) -> ConfigSubentry | None:
    """
    Return the subentry a flow is reconfiguring, if any.

    Prefers the flow's own subentry id, then the context, and finally — on a
    reconfigure with neither — the single subentry of ``subentry_type`` when
    exactly one exists.
    """
    entry = get_entry_from_flow(flow)
    subentry_id = getattr(flow, "_subentry_id", None) or flow.context.get("subentry_id")
    if subentry_id:
        return entry.subentries.get(subentry_id)
    if flow.source == SOURCE_RECONFIGURE:
        matching = [
            sub
            for sub in entry.subentries.values()
            if sub.subentry_type == subentry_type
        ]
        if len(matching) == 1:
            return matching[0]
    return None


def openai_provider_options(flow: ConfigSubentryFlow) -> list[SelectOptionDict]:
    """Return selector options for every OpenAI model-provider subentry."""
    entry = get_entry_from_flow(flow)
    options: list[SelectOptionDict] = []
    for sub in entry.subentries.values():
        if sub.subentry_type != SUBENTRY_TYPE_MODEL_PROVIDER:
            continue
        if sub.data.get("provider_type") != "openai":
            continue
        options.append(
            SelectOptionDict(label=sub.title or sub.subentry_id, value=sub.subentry_id)
        )
    return options


def resolve_provider_name(
    submitted_name: str | None,
    provider_type: str,
    *,
    provider_names: Mapping[str, str],
    stale_name: str | None,
    fallback: str,
) -> str:
    """
    Return the provider name to store after the provider step is submitted.

    Two kinds of untouched pre-fill are replaced with the selected type's own
    default. ``stale_name`` is the previously stored name when the provider
    type just changed (``None`` otherwise): a submission equal to it is the
    form's pre-fill and would mislabel the new provider in the Assist pipeline
    dropdown. On an add, the form renders before the dropdown is touched, so it
    pre-fills the default type's name; a submitted name that is another type's
    default is that stale pre-fill, not a choice. Any other name survives.
    """
    name = submitted_name
    if stale_name is not None and name == stale_name:
        name = None
    own_default = provider_names.get(provider_type)
    if name and name in provider_names.values() and name != own_default:
        name = None
    return name or own_default or fallback


async def build_openai_key_settings(
    hass: Any,
    openai_opts: list[SelectOptionDict],
    user_input: dict[str, Any],
    *,
    provider_id_key: str,
) -> tuple[dict[str, Any], str | None]:
    """
    Build the settings for an OpenAI provider and return an error key if any.

    Linking to an OpenAI model-provider subentry stores its id and blanks the
    key so the linked provider stays authoritative; a separate key is validated
    against the OpenAI API before it is stored.
    """
    api_key = user_input.get(CONF_API_KEY)
    provider_id = user_input.get(provider_id_key)

    if openai_opts and provider_id and provider_id != SEPARATE_KEY_OPTION:
        return {provider_id_key: provider_id, CONF_API_KEY: None}, None

    if not api_key:
        return {}, "invalid_auth"

    try:
        await validate_openai_key(hass, api_key)
    except InvalidAuthError:
        return {}, "invalid_auth"
    except CannotConnectError:
        return {}, "cannot_connect"
    except Exception:
        LOGGER.exception("Unexpected exception validating OpenAI key")
        return {}, "unknown"

    return {CONF_API_KEY: api_key, provider_id_key: None}, None


async def build_local_endpoint_settings(
    hass: Any,
    user_input: dict[str, Any],
    *,
    provider_id_key: str,
) -> tuple[dict[str, Any], str | None]:
    """
    Build the settings for a local OpenAI-compatible server.

    The URL is normalized before validation so what is stored is exactly what
    was proven reachable. A blank key is stored as ``None`` (keyless server).
    """
    base_url = user_input.get(CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL)
    if not isinstance(base_url, str) or not base_url.strip():
        return {}, "cannot_connect"
    base_url = normalize_openai_compatible_base_url(base_url)
    api_key = user_input.get(CONF_API_KEY) or None

    try:
        await validate_openai_compatible_url(hass, base_url, api_key)
    except InvalidAuthError:
        return {}, "invalid_auth"
    except CannotConnectError:
        return {}, "cannot_connect"
    except Exception:
        LOGGER.exception("Unexpected exception validating local endpoint")
        return {}, "unknown"

    return {
        CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL: base_url,
        CONF_API_KEY: api_key,
        provider_id_key: None,
    }, None


def local_endpoint_schema(prefill: Mapping[str, Any]) -> vol.Schema:
    """
    Return the credentials form for a local OpenAI-compatible server.

    ``prefill`` is what the form redisplays: the stored settings normally, or
    the just-typed input after a validation error, so a retry against a booting
    server does not demand retyping the URL and a corrected typo does not
    silently revert to the saved value.
    """
    base_url = prefill.get(CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL)
    api_key = prefill.get(CONF_API_KEY)
    return vol.Schema(
        {
            vol.Required(
                CONF_OPENAI_COMPATIBLE_ENDPOINT_BASE_URL,
                description={"suggested_value": base_url},
                default=base_url or "",
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_API_KEY,
                description={"suggested_value": api_key},
                default=api_key or "",
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        }
    )


def openai_key_schema(
    openai_opts: list[SelectOptionDict],
    stored: Mapping[str, Any],
    *,
    provider_id_key: str,
) -> vol.Schema:
    """
    Return the credentials form for the OpenAI provider.

    When OpenAI model-provider subentries exist, a selector offers to reuse one
    (defaulting to the stored link, or to "separate key" when a standalone key
    is stored). The key field is always present.
    """
    schema_dict: dict[Any, Any] = {}
    if openai_opts:
        reuse_opts = [
            SelectOptionDict(label="Use a separate key", value=SEPARATE_KEY_OPTION),
            *openai_opts,
        ]
        stored_provider_id = stored.get(provider_id_key)
        stored_api_key = stored.get(CONF_API_KEY)
        if stored_provider_id is None and stored_api_key:
            default_id = SEPARATE_KEY_OPTION
        else:
            default_id = stored_provider_id or reuse_opts[1]["value"]
        schema_dict[vol.Required(provider_id_key, default=default_id)] = SelectSelector(
            SelectSelectorConfig(
                options=reuse_opts,
                mode=SelectSelectorMode.DROPDOWN,
                sort=False,
                custom_value=False,
            )
        )

    schema_dict[
        vol.Optional(
            CONF_API_KEY,
            description={"suggested_value": stored.get(CONF_API_KEY)},
            default=stored.get(CONF_API_KEY) or "",
        )
    ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
    return vol.Schema(schema_dict)
