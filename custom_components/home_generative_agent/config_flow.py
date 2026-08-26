"""Config flow for Home Generative Agent integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    OptionsFlowWithReload,
)
from homeassistant.const import (
    CONF_LLM_HASS_API,
)
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    ConstantSelector,
    ConstantSelectorConfig,
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

from .agent.helpers import (
    TOOL_KEY_SEP,
    api_id_is_form_representable,
    normalize_llm_api_value,
    normalize_tool_exclusions,
    sanitize_tool_text,
    split_tool_index_key,
    tool_index_key,
)
from .const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_FACE_API_URL,
    CONF_FACE_RECOGNITION,
    CONF_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_MESSAGES_IN_CONTEXT,
    CONF_MAX_TOKENS_IN_CONTEXT,
    CONF_NOTIFY_SERVICE,
    CONF_PROMPT,
    CONF_SCHEMA_FIRST_YAML,
    CONF_STT_HALLUCINATION_EXACT_PATTERNS,
    CONF_STT_HALLUCINATION_PATTERNS,
    CONF_TOOL_EXCLUSIONS,
    CONF_TOOL_RELEVANCE_THRESHOLD,
    CONF_TOOL_RETRIEVAL_LIMIT,
    CONF_VIDEO_ANALYZER_MODE,
    CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP,
    CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
    CONF_VLM_PROMPT_EXTRA,
    CONF_VLM_RESPONSE_LANGUAGE,
    CONFIG_ENTRY_VERSION,
    CRITICAL_PIN_MAX_LEN,
    CRITICAL_PIN_MIN_LEN,
    DOMAIN,
    RECOMMENDED_FACE_RECOGNITION,
    RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
    RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
    RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
    RECOMMENDED_VIDEO_ANALYZER_MODE,
    RECOMMENDED_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
    RECOMMENDED_VLM_PROMPT_EXTRA,
    RECOMMENDED_VLM_RESPONSE_LANGUAGE,
    SUBENTRY_TYPE_FEATURE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_SENTINEL,
    SUBENTRY_TYPE_STT_PROVIDER,
    VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY,
    VIDEO_ANALYZER_MODE_DISABLE,
    VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY,
)
from .core.utils import (
    CannotConnectError,
    ensure_http_url,
    hash_pin,
    list_mobile_notify_services,
    validate_face_api_url,
)
from .flows.feature_subentry_flow import FeatureSubentryFlow
from .flows.model_provider_subentry_flow import ModelProviderSubentryFlow
from .flows.sentinel_subentry_flow import SentinelSubentryFlow
from .flows.stt_provider_subentry_flow import SttProviderSubentryFlow

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import VolDictType

LOGGER = logging.getLogger(__name__)

_CONF_STT_FILTERS_SECTION = "stt_filters_section"

# `conversation.DOMAIN`, spelled out rather than imported: pulling
# `homeassistant.components.conversation` into the config flow drags in hassil
# and the whole intent stack for a single string.  It must keep matching the
# assistant the conversation entity indexes tools under (`conversation.py`), or
# the exclusion picker would list a different tool set than the agent loads.
_CONVERSATION_DOMAIN = "conversation"

DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_SCHEMA_FIRST_YAML: False,
    CONF_CRITICAL_ACTION_PIN_ENABLED: False,
    CONF_VIDEO_ANALYZER_MODE: RECOMMENDED_VIDEO_ANALYZER_MODE,
    CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED: (
        RECOMMENDED_VIDEO_ANALYZER_UNIQUENESS_ENABLED
    ),
    CONF_FACE_RECOGNITION: RECOMMENDED_FACE_RECOGNITION,
    CONF_MANAGE_CONTEXT_WITH_TOKENS: RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
    CONF_MAX_TOKENS_IN_CONTEXT: RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
    CONF_MAX_MESSAGES_IN_CONTEXT: RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
    CONF_TOOL_RETRIEVAL_LIMIT: RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
    CONF_TOOL_RELEVANCE_THRESHOLD: RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
}

# ---------------------------
# Helpers
# ---------------------------


def _get_str(src: Mapping[str, Any], key: str) -> str:
    """Get a trimmed string from a mapping (missing -> '')."""
    return str(src.get(key, "") or "").strip()


def _patterns_as_text(raw: Any) -> str:
    """Render list/string pattern options as one pattern per line."""
    if isinstance(raw, list):
        return "\n".join(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, str):
        return raw
    return ""


def _map_as_text(raw: Any) -> str:
    """Render a dict as 'key: value' lines for display in a text area."""
    if isinstance(raw, dict):
        return "\n".join(f"{k}: {v}" for k, v in raw.items() if k and v)
    if isinstance(raw, str):
        return raw
    return ""


def _text_as_map(text: str) -> dict[str, str]:
    """Parse 'key: value' lines into a dict; silently skips malformed lines."""
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def _tool_exclusions_as_list(raw: Any) -> list[str]:
    """
    Render the stored ``{api_id: [name]}`` exclusions as flat picker values.

    Also accepts the flat list itself, because a form re-render after a
    validation error sees the raw ``user_input`` value, not the parsed one --
    the same dual-shape contract ``_map_as_text`` has.
    """
    if isinstance(raw, list):
        return [value for value in raw if isinstance(value, str) and value]
    return sorted(
        tool_index_key(api_id, name)
        for api_id, names in normalize_tool_exclusions(raw).items()
        for name in names
    )


def _list_as_tool_exclusions(values: list[Any]) -> dict[str, list[str]]:
    """Group flat picker values back into the stored ``{api_id: [name]}`` map."""
    grouped: dict[str, list[str]] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        if (parts := split_tool_index_key(value)) is None:
            continue
        api_id, name = parts
        grouped.setdefault(api_id, []).append(name)
    return normalize_tool_exclusions(grouped)


# Suffixes marking a stored exclusion whose tool is not in the live list. They
# are distinct on purpose -- see the three-state comment in
# `_tool_exclusion_choices`. Named so the wording lives in one place: the tests
# import these rather than re-spelling them.
_SUFFIX_NOT_SELECTED = " (API not selected)"
_SUFFIX_NOT_AVAILABLE = " (not currently available)"


def _label_text(text: str) -> str:
    """
    Sanitize remote text for a picker label and disarm the marker vocabulary.

    ``sanitize_tool_text`` stops text that lies structurally (controls, bidi
    overrides). It cannot stop text that lies *lexically*: every character of
    ``list_files (not currently available)`` is printable, so a server may name
    a tool that and produce a label byte-identical to a genuinely-missing
    tool's. That is a deception aimed squarely at this control -- an operator
    scanning the picker to switch off a dangerous tool sees it already marked
    inactive and moves on, while the tool is live and bound.

    The markers are parenthesised, and the remote half of the label has no
    legitimate need for parentheses (MCP tool names are identifiers), so
    rewriting them to square brackets makes the real suffix unreproducible
    while leaving such a name readable: ``list_files (not currently
    available)`` renders as ``list_files [not currently available]``, which no
    longer imitates the trusted marker. Deliberately ASCII rather than the
    fullwidth lookalikes -- answering a spoofing attack with visually
    confusable characters just moves the deception somewhere else.
    """
    return sanitize_tool_text(text).replace("(", "[").replace(")", "]")


def _tool_choice(
    api_name: str, tool_name: str, value: str, *, suffix: str
) -> SelectOptionDict:
    """
    Build one excluded-tools picker option.

    Single definition so the sanitize-the-label / never-touch-the-value
    invariant cannot drift between the live and stored branches: the label is
    remote-controlled text and is defanged before display, while the value must
    stay byte-identical to the real tool name or the exclusion it encodes would
    silently stop matching at runtime.  The suffix is trusted text appended
    *after* both remote halves have been through ``_label_text``, so it is the
    only parenthesised run the label can contain.
    """
    return SelectOptionDict(
        label=f"{_label_text(api_name)}: {_label_text(tool_name)}{suffix}",
        value=value,
    )


async def _tool_exclusion_choices(
    hass: HomeAssistant,
    selected_apis: list[str],
    selected_exclusions: list[str],
    indexed_keys: set[str] | None = None,
) -> list[SelectOptionDict]:
    """
    Build the excluded-tools picker options for the currently selected APIs.

    Enumeration is per API and failures are isolated: one unreachable MCP
    server -- or one malformed tool descriptor from an otherwise healthy one --
    must not empty the whole picker, let alone stop the options form opening.
    Whatever cannot be enumerated is re-added from the *stored* selection
    instead, labelled with why it is missing, so a server that is merely
    offline at render time keeps its exclusions -- dropping them would both
    fail SelectSelector validation on submit (issue #568's unsaveable-form
    class) and silently re-expose tools the user had switched off.
    """
    llm_context = llm.LLMContext(
        platform=DOMAIN,
        context=None,
        language=hass.config.language,
        assistant=_CONVERSATION_DOMAIN,
        device_id=None,
    )
    api_names = {api.id: api.name for api in llm.async_get_apis(hass)}
    selected_set = set(selected_apis)

    choices: list[SelectOptionDict] = []
    seen: set[str] = set()
    # dict.fromkeys, not set(): a repeated api id would otherwise re-run the
    # whole enumeration, and for the Assist API that means a second sort of
    # every state plus a second registry sweep. Order is preserved so the
    # picker still groups by the user's own API order.
    for api_id in dict.fromkeys(selected_apis):
        if not api_id_is_form_representable(api_id):
            # Offering this would mint a picker value that regroups onto a
            # DIFFERENT api id on save ("a::b" + "t" -> "a" + "b::t"), storing
            # an exclusion that matches nothing while the form reports success.
            # Refuse at the boundary rather than store a lie.
            LOGGER.warning(
                "Not offering tool exclusions for LLM API id %r: ids containing "
                "%r cannot round-trip through the options form",
                api_id,
                TOOL_KEY_SEP,
            )
            continue
        try:
            instance = await llm.async_get_api(hass, api_id, llm_context)
            tools = list(instance.tools or [])
        except Exception as err:  # noqa: BLE001
            # `err` carries remote text for an MCP API (HTTP body, JSON-RPC
            # error message), so it is a log-line forgery sink like the tool
            # names beside it -- a message containing a newline writes
            # attacker-chosen WARNING records into the HA log.
            LOGGER.warning(
                "Could not list tools of LLM API %s for the exclusion picker: %s",
                api_id,
                sanitize_tool_text(str(err), limit=200),
            )
            continue
        # A tool descriptor is remote data, so its `.name` is not guaranteed to
        # be a string. Sorting a mixed list raises TypeError, and this loop sits
        # outside the try above -- unguarded, one malformed descriptor from one
        # server takes down the whole options form, not just the picker (issue
        # #568's unopenable-form class). Drop the bad entries and keep that
        # server's good tools rather than losing the server to one bad name.
        named = [
            tool
            for tool in tools
            if isinstance(getattr(tool, "name", None), str) and tool.name
        ]
        if len(named) != len(tools):
            LOGGER.warning(
                "LLM API %s advertised %d tool(s) with an unusable name; "
                "they are omitted from the exclusion picker",
                api_id,
                len(tools) - len(named),
            )
        api_name = api_names.get(api_id, api_id)
        for tool in sorted(named, key=lambda tool: tool.name):
            value = tool_index_key(api_id, tool.name)
            if value in seen:
                continue
            seen.add(value)
            choices.append(_tool_choice(api_name, tool.name, value, suffix=""))

    # Device-gated tools. Home Assistant exposes some Assist tools only when
    # the requesting device supports them -- the timer intents are gated on
    # `llm_context.device_id is not None` (components/intent/llm.py) -- and an
    # options form has no device, so enumerating above can never see them. They
    # are still bound on every voice-satellite turn, so without this the user
    # cannot switch them off at all and the picker quietly under-reports what
    # the agent can call. The tool index is the compensating source: its keys
    # already carry whatever the per-turn top-up discovered from real devices,
    # in exactly this composite-key form.
    #
    # Caveat, accepted: index rows are never evicted (see the tool-index
    # hygiene entry in TODOS.md), so a tool a server has since dropped can
    # linger here and render as though it were live. Excluding one is a
    # harmless no-op, which is why this is worth less than the timer intents it
    # buys -- but it is the reason these are unioned in rather than trusted as
    # the authoritative list.
    for key in sorted(indexed_keys or ()):
        if key in seen:
            continue
        parts = split_tool_index_key(key)
        if parts is None:
            continue
        api_id, tool_name = parts
        if api_id not in selected_set or not api_id_is_form_representable(api_id):
            continue
        seen.add(key)
        choices.append(
            _tool_choice(api_names.get(api_id, api_id), tool_name, key, suffix="")
        )

    for value in selected_exclusions:
        if value in seen:
            continue
        seen.add(value)
        parts = split_tool_index_key(value)
        api_id, tool_name = parts or (value, value)
        # Three states reach this loop, and only two labels are honest:
        #   registered + deselected -> the user turned this API off here
        #   registered + selected   -> enumeration failed, i.e. really unreachable
        #   not registered          -> the API is gone from Home Assistant
        # "Not selected" and "not available" are different facts the user acts
        # on differently: one is undone in this very form, the other means go
        # look at the server. Testing both halves of the condition matters --
        # keying on registration alone would label a selected-but-unreachable
        # server "not selected" and send the user to the wrong place.
        suffix = (
            _SUFFIX_NOT_SELECTED
            if api_id in api_names and api_id not in selected_set
            else _SUFFIX_NOT_AVAILABLE
        )
        choices.append(
            _tool_choice(api_names.get(api_id, api_id), tool_name, value, suffix=suffix)
        )

    return choices


async def _schema_for_options(
    hass: HomeAssistant,
    opts: Mapping[str, Any],
    indexed_keys: set[str] | None = None,
) -> VolDictType:
    """Generate the options schema for non-provider settings."""
    # `api.name` is `entry.title` for an MCP API, which HA sets from the
    # server's own `serverInfo.name` -- remote text, in a label, in the same
    # form whose tool picker sanitizes the identical value.
    hass_apis = [
        SelectOptionDict(label=sanitize_tool_text(api.name), value=api.id)
        for api in llm.async_get_apis(hass)
    ]
    valid_api_ids = {api["value"] for api in hass_apis}
    selected_apis = normalize_llm_api_value(opts.get(CONF_LLM_HASS_API, []))
    # Stored ids that are no longer registered (e.g. a removed MCP server)
    # stay selected but are re-added as labeled selector options: without
    # them the pre-filled value fails SelectSelector validation on submit,
    # leaving the form permanently unsaveable (issue #568). Keeping them
    # selectable — rather than silently dropping them — means a transient
    # provider outage at render time can never erase the user's choice, and
    # deselecting a dead server is always an explicit user action.
    if stale := [api_id for api_id in selected_apis if api_id not in valid_api_ids]:
        LOGGER.warning(
            "LLM API ids no longer registered in Home Assistant "
            "(shown as 'no longer available' in the options form): %s",
            stale,
        )
        hass_apis.extend(
            SelectOptionDict(label=f"{api_id} (no longer available)", value=api_id)
            for api_id in stale
        )

    selected_exclusions = _tool_exclusions_as_list(opts.get(CONF_TOOL_EXCLUSIONS))
    tool_exclusion_choices = await _tool_exclusion_choices(
        hass, selected_apis, selected_exclusions, indexed_keys
    )

    video_analyzer_mode_opts: list[SelectOptionDict] = [
        SelectOptionDict(label="Disable", value=VIDEO_ANALYZER_MODE_DISABLE),
        SelectOptionDict(
            label="Notify on anomaly", value=VIDEO_ANALYZER_MODE_NOTIFY_ON_ANOMALY
        ),
        SelectOptionDict(
            label="Always notify", value=VIDEO_ANALYZER_MODE_ALWAYS_NOTIFY
        ),
    ]

    context_mgmt_modes = [
        SelectOptionDict(label="Use tokens", value="true"),
        SelectOptionDict(label="Use messages", value="false"),
    ]

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": opts.get(CONF_PROMPT)},
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_VLM_RESPONSE_LANGUAGE,
            description={
                "suggested_value": opts.get(
                    CONF_VLM_RESPONSE_LANGUAGE, RECOMMENDED_VLM_RESPONSE_LANGUAGE
                )
            },
            default=RECOMMENDED_VLM_RESPONSE_LANGUAGE,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        vol.Optional(
            CONF_VLM_PROMPT_EXTRA,
            description={
                "suggested_value": opts.get(
                    CONF_VLM_PROMPT_EXTRA, RECOMMENDED_VLM_PROMPT_EXTRA
                )
            },
            default=RECOMMENDED_VLM_PROMPT_EXTRA,
        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": selected_apis},
            default=[],
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Required(
            CONF_TOOL_RETRIEVAL_LIMIT,
            default=opts.get(
                CONF_TOOL_RETRIEVAL_LIMIT, RECOMMENDED_TOOL_RETRIEVAL_LIMIT
            ),
        ): NumberSelector(NumberSelectorConfig(min=1, max=20, step=1)),
        vol.Required(
            CONF_TOOL_RELEVANCE_THRESHOLD,
            default=opts.get(
                CONF_TOOL_RELEVANCE_THRESHOLD, RECOMMENDED_TOOL_RELEVANCE_THRESHOLD
            ),
        ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
    }

    # Rendered next to the other tool-selection settings, and only when there is
    # something to list: no API selected, or every provider unreachable *and*
    # nothing stored, means the field is omitted entirely.  An absent field is
    # never present in `user_input`, so a render-time provider outage can never
    # write an empty selection over a stored one.
    if tool_exclusion_choices:
        schema[
            vol.Optional(
                CONF_TOOL_EXCLUSIONS,
                description={"suggested_value": selected_exclusions},
                default=[],
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=tool_exclusion_choices,
                multiple=True,
                mode=SelectSelectorMode.DROPDOWN,
                sort=False,
                custom_value=False,
            )
        )

    schema.update(
        {
            vol.Optional(
                CONF_VIDEO_ANALYZER_MODE,
                description={"suggested_value": opts.get(CONF_VIDEO_ANALYZER_MODE)},
                default=RECOMMENDED_VIDEO_ANALYZER_MODE,
            ): SelectSelector(SelectSelectorConfig(options=video_analyzer_mode_opts)),
            vol.Optional(
                CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
                description={
                    "suggested_value": opts.get(
                        CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
                        RECOMMENDED_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
                    )
                },
                default=opts.get(
                    CONF_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
                    RECOMMENDED_VIDEO_ANALYZER_UNIQUENESS_ENABLED,
                ),
            ): BooleanSelector(),
            vol.Optional(
                CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP,
                default=_map_as_text(
                    opts.get(CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP, {})
                ),
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            ),
            vol.Optional(
                CONF_FACE_API_URL,
                description={"suggested_value": opts.get(CONF_FACE_API_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_MANAGE_CONTEXT_WITH_TOKENS,
                description={
                    "suggested_value": opts.get(CONF_MANAGE_CONTEXT_WITH_TOKENS, "true")
                },
                default=RECOMMENDED_MANAGE_CONTEXT_WITH_TOKENS,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=context_mgmt_modes,
                    mode=SelectSelectorMode.DROPDOWN,
                    sort=False,
                    custom_value=False,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS_IN_CONTEXT,
                description={"suggested_value": opts.get(CONF_MAX_TOKENS_IN_CONTEXT)},
                default=RECOMMENDED_MAX_TOKENS_IN_CONTEXT,
            ): NumberSelector(NumberSelectorConfig(min=64, max=65536, step=1)),
            vol.Optional(
                CONF_MAX_MESSAGES_IN_CONTEXT,
                description={"suggested_value": opts.get(CONF_MAX_MESSAGES_IN_CONTEXT)},
                default=RECOMMENDED_MAX_MESSAGES_IN_CONTEXT,
            ): NumberSelector(NumberSelectorConfig(min=15, max=240, step=1)),
            vol.Optional(
                CONF_CRITICAL_ACTION_PIN_ENABLED,
                description={
                    "suggested_value": opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False)
                },
                default=opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False),
            ): BooleanSelector(),
        }
    )

    if opts.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False):
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
            CONF_SCHEMA_FIRST_YAML,
            description={"suggested_value": opts.get(CONF_SCHEMA_FIRST_YAML, False)},
            default=opts.get(CONF_SCHEMA_FIRST_YAML, False),
        )
    ] = BooleanSelector()

    video_analyzer_mode = opts.get(
        CONF_VIDEO_ANALYZER_MODE, RECOMMENDED_VIDEO_ANALYZER_MODE
    )
    if video_analyzer_mode != VIDEO_ANALYZER_MODE_DISABLE:
        schema[
            vol.Optional(
                CONF_FACE_RECOGNITION,
                description={"suggested_value": opts.get(CONF_FACE_RECOGNITION)},
                default=RECOMMENDED_FACE_RECOGNITION,
            )
        ] = BooleanSelector()

    if video_analyzer_mode != VIDEO_ANALYZER_MODE_DISABLE:
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

    schema.update(
        {
            vol.Optional(
                _CONF_STT_FILTERS_SECTION,
                default="speech_input_filters",
            ): ConstantSelector(
                ConstantSelectorConfig(
                    label="Speech input filters",
                    value="speech_input_filters",
                )
            ),
            vol.Optional(
                CONF_STT_HALLUCINATION_PATTERNS,
                default=_patterns_as_text(
                    opts.get(CONF_STT_HALLUCINATION_PATTERNS, [])
                ),
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            ),
            vol.Optional(
                CONF_STT_HALLUCINATION_EXACT_PATTERNS,
                default=_patterns_as_text(
                    opts.get(CONF_STT_HALLUCINATION_EXACT_PATTERNS, [])
                ),
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
            ),
        }
    )

    return schema


# ---------------------------
# Config Flow
# ---------------------------


class HomeGenerativeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Generative Agent."""

    VERSION = CONFIG_ENTRY_VERSION

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({}))

        return self.async_create_entry(
            title="Home Generative Agent",
            data={},
            options=dict(DEFAULT_OPTIONS),
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        _ = config_entry
        return HomeGenerativeAgentOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return supported subentry flow handlers."""
        _ = config_entry
        return {
            SUBENTRY_TYPE_MODEL_PROVIDER: ModelProviderSubentryFlow,
            SUBENTRY_TYPE_FEATURE: FeatureSubentryFlow,
            SUBENTRY_TYPE_STT_PROVIDER: SttProviderSubentryFlow,
            SUBENTRY_TYPE_SENTINEL: SentinelSubentryFlow,
        }


# ---------------------------
# Options Flow
# ---------------------------


class HomeGenerativeAgentOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Home Generative Agent."""

    # ---- helpers ----

    def _base_options(self) -> dict[str, Any]:
        options = dict(DEFAULT_OPTIONS)
        options.update(self.config_entry.options)
        return options

    def _indexed_tool_keys(self) -> set[str]:
        """
        Return composite keys of every tool the running entry has indexed.

        These carry the device-gated tools an options form can never enumerate
        for itself (timer intents and friends, exposed only when the requesting
        device supports them), so the picker unions them in — see the
        device-gated block in ``_tool_exclusion_choices``.

        Best effort by design: the entry may not be loaded when the form opens,
        in which case there is no ``runtime_data`` and the picker simply falls
        back to what it can enumerate live. Failing to offer a device-gated
        tool is a smaller harm than failing to open the form.
        """
        runtime_data = getattr(self.config_entry, "runtime_data", None)
        hashes = getattr(runtime_data, "tool_content_hashes", None)
        if not isinstance(hashes, dict):
            return set()
        return {key for key in hashes if isinstance(key, str)}

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

    def _maybe_edit_pin(
        self, options: dict[str, Any], user_input: Mapping[str, Any] | None
    ) -> str | None:
        """Hash and store the critical-action PIN if provided."""
        if user_input is None:
            return None

        pin_enabled = user_input.get(
            CONF_CRITICAL_ACTION_PIN_ENABLED,
            options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False),
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

    def _drop_empty_fields(self, final_options: dict[str, Any]) -> None:
        """Remove empty strings for fields to avoid storing empties."""
        for k in (
            CONF_FACE_API_URL,
            CONF_NOTIFY_SERVICE,
            CONF_VLM_RESPONSE_LANGUAGE,
            CONF_VLM_PROMPT_EXTRA,
        ):
            if not _get_str(final_options, k):
                final_options.pop(k, None)

    def _cleanup_none_llm_api(self, options: dict[str, Any]) -> None:
        """Remove the key when no APIs are selected so options stay clean."""
        if not options.get(CONF_LLM_HASS_API):
            options.pop(CONF_LLM_HASS_API, None)

    def _cleanup_ui_only_options(self, options: dict[str, Any]) -> None:
        """Remove schema-only fields before storing options."""
        options.pop(_CONF_STT_FILTERS_SECTION, None)

    def _parse_tool_exclusions(self, options: dict[str, Any]) -> None:
        """
        Convert the flat excluded-tools picker value back to ``{api_id: [name]}``.

        A dict is what storage already holds, and it reaches here unchanged when
        the picker was not rendered this pass (nothing enumerable) -- normalize
        and keep it, because the user did not touch it.  A list is a submitted
        selection: an empty one is an explicit "exclude nothing", so the key is
        dropped rather than stored as ``{}``, keeping "absent" the single
        spelling of the default.
        """
        raw = options.get(CONF_TOOL_EXCLUSIONS)
        if isinstance(raw, list):
            normalized = _list_as_tool_exclusions(raw)
        else:
            normalized = normalize_tool_exclusions(raw)
        if normalized:
            options[CONF_TOOL_EXCLUSIONS] = normalized
        else:
            options.pop(CONF_TOOL_EXCLUSIONS, None)

    def _parse_motion_camera_map(self, options: dict[str, Any]) -> None:
        """Convert the motion camera map text area to a dict before storing."""
        raw = options.get(CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP, "")
        if isinstance(raw, str):
            parsed = _text_as_map(raw)
            if parsed:
                options[CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP] = parsed
            else:
                options.pop(CONF_VIDEO_ANALYZER_MOTION_CAMERA_MAP, None)

    # ---- main step ----

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the options flow init step."""
        options = self._base_options()

        # First render
        if user_input is None:
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(
                    await _schema_for_options(
                        self.hass, options, self._indexed_tool_keys()
                    )
                ),
            )

        # Merge new input for non-validated fields
        options.update(user_input or {})
        errors: dict[str, str] = {}

        # Field-specific edits with validation/normalization
        err = await self._maybe_edit_face_recognition_url(options, user_input)
        if not err:
            err = self._maybe_edit_pin(options, user_input)
        if err:
            errors["base"] = err

        if errors:
            # Re-render with the same options and show errors
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(
                    await _schema_for_options(
                        self.hass, options, self._indexed_tool_keys()
                    )
                ),
                errors=errors,
            )

        self._cleanup_none_llm_api(options)
        self._cleanup_ui_only_options(options)
        self._drop_empty_fields(options)
        self._parse_motion_camera_map(options)
        self._parse_tool_exclusions(options)
        return self.async_create_entry(title="", data=options)
