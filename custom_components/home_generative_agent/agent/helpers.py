"""Helper utilities for Home Generative Agent tool conversion and normalization."""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import re
import unicodedata
import weakref
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import intent, llm
from homeassistant.util import ulid
from pydantic import PydanticInvalidForJsonSchema

from custom_components.home_generative_agent.const import (
    ACTUATION_LANGCHAIN_TOOLS,
    ACTUATION_TOOL_PREFIXES,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_CRITICAL_ACTIONS,
    CONF_TOOL_EXCLUSIONS,
    CONF_TOOL_INCLUSIONS,
    RECOMMENDED_CRITICAL_ACTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from homeassistant.core import HomeAssistant, State

LOGGER = logging.getLogger(__name__)


def _resolve_converter() -> tuple[Any, Any]:
    """
    Return the (converter, deferral sentinel) pair matching the running core.

    Home Assistant 2026.9 swapped voluptuous for probatio and
    voluptuous-openapi for probatio's ``to_openapi``, aliasing probatio under
    the ``voluptuous`` name (``install_as_voluptuous()`` in
    ``homeassistant/__init__.py``) for everything that still imports
    voluptuous — ``vol`` here included. The converter has to match the
    library the schemas actually come from, since neither one renders the
    other's ``Schema``. Core's ``llm`` helper imports exactly the pair the
    running core was built against (``convert`` + ``UNSUPPORTED`` from
    voluptuous-openapi on <= 2026.8; ``to_openapi`` + ``UNSUPPORTED`` from
    probatio on 2026.9; a bare ``import probatio`` from 2026.10), so prefer
    those — they cannot diverge from core by construction, where sniffing
    ``vol.Schema.__module__`` breaks the day probatio stamps a
    voluptuous-compatible ``__module__`` on its shim. The sniff survives only
    as a fallback for a core that exposes neither. Neither library is
    declared in our manifest on purpose: core installs exactly the one it
    uses.
    """
    core_convert = getattr(llm, "to_openapi", None) or getattr(llm, "convert", None)
    core_unsupported = getattr(llm, "UNSUPPORTED", None)
    if core_convert is not None and core_unsupported is not None:
        return core_convert, core_unsupported
    core_probatio = getattr(llm, "probatio", None)
    if isinstance(core_probatio, ModuleType):
        return core_probatio.to_openapi, core_probatio.UNSUPPORTED
    if vol.Schema.__module__.startswith("probatio"):
        module = importlib.import_module("probatio")
        return module.to_openapi, module.UNSUPPORTED
    module = importlib.import_module("voluptuous_openapi")
    return module.convert, module.UNSUPPORTED


convert, UNSUPPORTED = _resolve_converter()


def _collect_unsupported_sentinels() -> tuple[Any, ...]:
    """
    Return every deferral sentinel a serializer might hand back.

    ``UNSUPPORTED`` above is core's own, but ``APIInstance.custom_serializer``
    is a public field a third-party ``llm.API`` can set with a serializer
    built against the sibling library — pip does not uninstall
    voluptuous-openapi when a core upgrade brings probatio — and sentinels
    compare by identity, so a foreign one would read "I cannot render this"
    as a rendered schema. Accept the sentinel of every importable converter
    library; what we hand back stays ours, so the converter in use always
    recognises it.
    """
    sentinels: list[Any] = [UNSUPPORTED]
    for module_name in ("voluptuous_openapi", "probatio"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        sentinel = getattr(module, "UNSUPPORTED", None)
        if sentinel is not None and all(sentinel is not seen for seen in sentinels):
            sentinels.append(sentinel)
    return tuple(sentinels)


_UNSUPPORTED_SENTINELS: tuple[Any, ...] = _collect_unsupported_sentinels()


def _is_unsupported(value: Any) -> bool:
    """Return True if a serializer deferred rather than returning a schema."""
    return any(value is sentinel for sentinel in _UNSUPPORTED_SENTINELS)


class ToolCallData(NamedTuple):
    """The JSON data of an HA tool call and whether the tool reported an error."""

    data: Any
    error: bool


def tool_call_data(result: Any) -> ToolCallData:
    """
    Return the JSON data of an ``APIInstance.async_call_tool`` result.

    Home Assistant 2026.10 wraps the data in ``llm.ToolResult(data, error)``;
    earlier versions return the data itself. ``llm.ToolResult`` does not exist
    before 2026.10, so look it up rather than import it.
    """
    tool_result_cls = getattr(llm, "ToolResult", None)
    if tool_result_cls is not None and isinstance(result, tool_result_cls):
        return ToolCallData(result.data, bool(getattr(result, "error", False)))
    return ToolCallData(result, error=False)


def chat_log_tool_result(data: Any, *, error: bool = False) -> dict[str, Any]:
    """
    Return the tool result keyword for HA's chat log on this HA version.

    Home Assistant 2026.10 replaced ``ToolResultContent.tool_result`` with
    ``result: llm.ToolResult``: the old keyword is a TypeError on the content
    class and a deprecated delta key. Earlier versions have no
    ``llm.ToolResult`` and take ``tool_result``.
    """
    tool_result_cls = getattr(llm, "ToolResult", None)
    if tool_result_cls is None:
        return {"tool_result": data}
    return {"result": tool_result_cls(data=data, error=error)}


def active_llm_api_ids(options: Mapping[str, Any]) -> list[str]:
    """
    Return the LLM API ids to expose, defaulting to Assist when unset.

    Single definition on purpose.  The options flow *deletes* the key when no
    API is selected (``_cleanup_none_llm_api`` in ``config_flow.py``), so an
    absent key is a normal state meaning "the recommended default", not "no
    APIs at all".  Every reader must apply that same default, because they are
    not independent: ``conversation.py`` derives
    ``ConversationEntityFeature.CONTROL`` from this, and Home Assistant reads
    that feature flag to decide whether its built-in intent handler may take
    control commands *before* the agent ever sees them
    (``assist_pipeline/pipeline.py`` installs a local-intent allowlist only for
    agents that advertise CONTROL).  A reader that treats the absent key as
    "no APIs" therefore drops the flag and silently routes lock/unlock around
    the critical-action PIN while the rest of the integration still runs the
    Assist API.
    """
    if CONF_LLM_HASS_API not in options:
        return [llm.LLM_API_ASSIST]
    return normalize_llm_api_value(options[CONF_LLM_HASS_API])


def normalize_llm_api_value(raw: Any) -> list[str]:
    """
    Normalize a stored ``CONF_LLM_HASS_API`` value to a list of id strings.

    Storage has carried several shapes over the integration's life: a list of
    ids, a pre-v6 bare string, and — via programmatic options updates that
    bypass the form schema — ``None``, ``""``, or lists holding non-string
    elements.  Iterating or set-testing those degenerate shapes crashes both
    the options-form build and ``active_llm_api_ids`` at runtime (issue #568's
    failure class), so every reader funnels through this one normalizer:
    strings wrap to a single-element list, everything non-list collapses to
    ``[]``, and non-string or empty elements are dropped.
    """
    if isinstance(raw, str):
        return [raw] if raw else []
    if not isinstance(raw, list):
        return []
    return [api_id for api_id in raw if isinstance(api_id, str) and api_id]


def normalize_tool_exclusions(raw: Any) -> dict[str, list[str]]:
    """
    Normalize a stored ``CONF_TOOL_EXCLUSIONS`` value to ``{api_id: [name]}``.

    Mirrors ``normalize_llm_api_value``'s contract: every reader funnels through
    one normalizer so a degenerate shape written by a programmatic options
    update (``None``, a bare string, non-string elements) can never crash the
    options-form build or the per-turn tool filter.  API ids mapping to no
    surviving names are dropped, because "present with an empty list" and
    "absent" mean the same thing — expose every tool of that API — and keeping
    both spellings would invite readers to disagree about which is which.
    """
    return normalize_api_tool_map(raw)


def normalize_tool_inclusions(raw: Any) -> dict[str, list[str]]:
    """
    Normalize a stored ``CONF_TOOL_INCLUSIONS`` value to ``{api_id: [name]}``.

    Same contract and same shape as ``normalize_tool_exclusions`` — the two
    options are mirror images (subtractive vs. additive), and here too an api
    id mapping to no surviving names means the same thing as an absent one:
    nothing extra is force-bound for that API.
    """
    return normalize_api_tool_map(raw)


def normalize_api_tool_map(raw: Any) -> dict[str, list[str]]:
    """Normalize an ``{api_id: [tool_name]}`` option value; see the wrappers."""
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for api_id, raw_names in raw.items():
        if not isinstance(api_id, str) or not api_id:
            continue
        names = [raw_names] if isinstance(raw_names, str) else raw_names
        if not isinstance(names, list):
            continue
        cleaned = sorted({n for n in names if isinstance(n, str) and n})
        if cleaned:
            normalized[api_id] = cleaned
    return normalized


def api_id_is_form_representable(api_id: str) -> bool:
    """
    Return whether an API id can round-trip through the options form.

    The picker flattens exclusions to ``api_id::tool_name`` values and regroups
    them by splitting on the FIRST separator, so an id containing the separator
    does not survive: ``{"a::b": ["t"]}`` comes back as ``{"a": ["b::t"]}``, an
    exclusion matching nothing.

    This is deliberately **not** enforced inside ``normalize_tool_exclusions``.
    That normalizer is shared by the options form and the per-turn runtime
    filter, and the two want opposite things: the form must refuse to *offer*
    an exclusion it cannot store faithfully, while the runtime must keep
    *honouring* one that is already stored. Refusing in the shared path means a
    stored exclusion for such an id stops being enforced — the tools silently
    go live again — and is then erased from storage by the next unrelated save.
    Fail-open on a security control, so: refuse at the form boundary only.
    """
    return TOOL_KEY_SEP not in api_id


def tool_exclusions(options: Mapping[str, Any]) -> dict[str, set[str]]:
    """
    Return the configured per-API excluded tool names as lookup sets.

    Sets, not the sorted lists ``normalize_tool_exclusions`` returns, because
    this is the per-turn runtime path: ``filter_excluded_tools`` membership-
    tests every tool of every loaded API against them.  Storage and the options
    form want the list form instead — it is what round-trips through JSON and
    what gives the picker a stable order — so the two shapes are deliberate,
    not redundant.  Take this one when you are filtering, that one when you are
    reading or writing the stored value.
    """
    raw = options.get(CONF_TOOL_EXCLUSIONS)
    return {
        api_id: set(names) for api_id, names in normalize_tool_exclusions(raw).items()
    }


def tool_inclusions(options: Mapping[str, Any]) -> list[tuple[str, str]]:
    """
    Return the configured always-included tools as sorted (api_id, name) pairs.

    Pairs, not the ``{api_id: [name]}`` map storage holds, because the runtime
    consumer (the always-included step in ``_retrieve_tools``) iterates them to
    fetch each tool individually — and sorted so the append order, and with it
    the bound tool order, is stable across turns rather than dict-insertion
    dependent.
    """
    raw = options.get(CONF_TOOL_INCLUSIONS)
    return sorted(
        (api_id, name)
        for api_id, names in normalize_tool_inclusions(raw).items()
        for name in names
    )


def is_tool_excluded(tool_name: str, excluded_names: Collection[str]) -> bool:
    """
    Return True when a live tool matches the user's stored exclusion list.

    Exact match first. A stored name that carries no ``__`` is a *legacy* entry
    -- saved before HA 2026.9 moved the built-in tools to ``<domain>__<Name>``
    -- and is additionally matched against the live tool's base name, so an
    exclusion of ``HassTurnOff`` keeps excluding the tool now published as
    ``intent__HassTurnOff``. Without this the deny-list silently stopped
    matching on upgrade: the tool the user switched off came back, while the
    picker still showed it as excluded.

    Deliberately asymmetric. A stored name that ALREADY carries a namespace is
    matched exactly and never stripped, because stripping it would quietly
    widen one specific exclusion (``intent__HassTurnOff``) to every same-suffix
    tool. Widening only bare legacy entries can over-match in the other
    direction -- one bare ``HassTurnOn`` would cover several namespaced tools
    if HA ever published more than one -- and for a deny-list that is the
    correct way to be wrong. (On HA 2026.9 it cannot happen: only
    ``components/intent/llm.py`` registers HassTurnOn/HassTurnOff, there is no
    ``lock/llm.py``, and ``light/llm.py`` exposes only HassLightSet.)
    """
    if tool_name in excluded_names:
        return True
    base = base_tool_name(tool_name)
    if base == tool_name:
        # Nothing was stripped, so the exact test above was the whole question.
        return False
    return any(
        base == name for name in excluded_names if TOOL_NAMESPACE_SEP not in name
    )


def filter_excluded_tools(
    api_id: str,
    api: llm.APIInstance,
    excluded: Mapping[str, set[str]],
) -> tuple[llm.APIInstance, list[str]]:
    """
    Return an APIInstance exposing only the tools the user has not excluded.

    Returns the (possibly unchanged) instance and the names actually dropped.

    A *copy* is returned rather than an in-place edit of ``api.tools``: Home
    Assistant's MCP integration hands every ``APIInstance`` the very same
    coordinator-owned list object (``tools=self.coordinator.data``), so mutating
    it would strip the excluded tools from every other consumer of that server
    in the whole instance — and keep them stripped until the coordinator's next
    30-minute refresh.

    Filtering here, on the instance the conversation entity loads, is the single
    enforcement point: tool binding, the RAG fallback set, the live-tool filter
    and ``APIInstance.async_call_tool`` all read ``.tools``, so an excluded tool
    is not merely hidden from the model — a hallucinated call to it is rejected
    deterministically with ``Tool "x" not found``.
    """
    names = excluded.get(api_id)
    if not names:
        return api, []
    # `.tools` is None for an MCP API whose coordinator has not completed its
    # first refresh (HA builds the instance with `tools=self.coordinator.data`),
    # and a third-party API may pass None freely. The options form already
    # guards this with `instance.tools or []`; unguarded here it raises
    # TypeError on the per-turn path, outside the caller's
    # `except HomeAssistantError`, and kills every conversation turn until the
    # coordinator recovers.
    tools = list(api.tools or [])
    # `tool.name` is remote data and is not guaranteed to be a string, let alone
    # a hashable one. An unguarded `tool.name not in names` raises TypeError on
    # a list/dict name, and this runs on the per-turn path *outside* the
    # `except HomeAssistantError` in `_async_init_llm_apis` -- so an unguarded
    # comprehension turns one malformed descriptor into every conversation turn
    # failing, for as long as that server advertises it. A non-str name can
    # never equal an excluded name, so treating it as "keep" is also correct,
    # not merely safe. Guarding here rather than wrapping the call in the caller
    # is deliberate: a try/except around a security control fails *open*.
    live_names = [
        tool.name for tool in tools if isinstance(getattr(tool, "name", None), str)
    ]
    kept = [
        tool
        for tool in tools
        if not (
            isinstance(getattr(tool, "name", None), str)
            and is_tool_excluded(tool.name, names)
        )
    ]
    if len(kept) == len(tools):
        return api, []
    # Report the LIVE names that were dropped, not the stored spellings: after
    # HA's rename the two differ, and the live name is what the operator sees
    # everywhere else in the log.
    dropped = sorted(n for n in live_names if is_tool_excluded(n, names))
    # replace() rather than the constructor so a field added to APIInstance by a
    # future Home Assistant release is carried over instead of silently reset.
    return replace(api, tools=kept), dropped


def safe_convert(
    schema: Any, custom_serializer: Callable[[Any], Any] | None = None
) -> Any:
    """Safely convert a voluptuous schema to OpenAPI, handling unhashable Selectors."""

    def robust_serializer(obj: Any) -> Any:
        """Robustly handle HA types that might be unhashable or need mapping."""
        # 0. Skip basic types that the schema converter handles natively.
        # vol.Schema, dict, and list are unhashable but handled by the library.
        if isinstance(obj, (vol.Schema, dict, list)):
            return UNSUPPORTED

        # 1. First, call the original custom serializer if it exists
        if custom_serializer:
            try:
                res = custom_serializer(obj)
                # If external serializer returns non-None/UNSUPPORTED, use it.
                # Some HA serializers might return None for unhandled types.
                if res is not None and not _is_unsupported(res):
                    return res
            except Exception:  # noqa: BLE001, S110
                # Fallback to the robust serializer if custom_serializer fails.
                pass

        # 2. Check for Home Assistant Selectors by looking for 'config'
        # These are often unhashable and require specific extraction.
        config = getattr(obj, "config", obj if isinstance(obj, dict) else {})
        if isinstance(config, dict) and "options" in config:
            # Extract options into an Enum for SelectSelector
            raw_options = config.get("options")
            if isinstance(raw_options, list):
                options = [
                    opt.get("value", opt) if isinstance(opt, dict) else opt
                    for opt in raw_options
                ]
                return {"type": "string", "enum": options}

        # 3. Handle other selectors that have a config but aren't SelectSelector
        if hasattr(obj, "config") or (isinstance(obj, dict) and "selector" in obj):
            return {"type": "string"}

        # 4. General unhashable safety net to prevent a converter crash
        try:
            hash(obj)
        except TypeError:
            return {"type": "string"}

        return UNSUPPORTED

    return convert(schema, custom_serializer=robust_serializer)


class ConfigurableData(TypedDict, total=False):
    """Typed view of the configurable payload passed through tools."""

    options: Mapping[str, Any]
    pending_actions: dict[str, dict[str, Any]]
    hass: HomeAssistant
    user_id: str
    ha_llm_api: Any
    thread_id: str
    # The user's message for this turn. A PIN confirmation is accepted only
    # when the PIN is in it, and runs at most one action.
    pin_reply: PinReply


@dataclass(slots=True)
class PinReply:
    """The user's message for one conversation turn, as a PIN source."""

    # The message can hold the PIN, so keep it out of reprs and logs.
    text: str = field(repr=False)
    # Set when a PIN from this message fails the check (one message is one
    # attempt) or its action is claimed to run (one message runs one action).
    # A correct PIN whose action cannot start, such as with no HA LLM API,
    # leaves it unset, so the user's reply can still be used once it can.
    spent: bool = False
    # Set once a refused PIN has been checked for the diagnostic warning, so
    # the hash runs at most once per message.
    refusal_checked: bool = False
    # Identifies this turn: a pending action records the turn that created it
    # and the one turn whose message may answer its PIN prompt.
    turn_id: str = field(default_factory=ulid.ulid_now)
    # The action this message's PIN was accepted for, as
    # `describe_pending_action` names it, so the reply can say what happened
    # in words the model did not choose, and how it ended (`outcome`).
    confirmed: str = ""
    outcome: str = ""


# How a PIN-confirmed action ended, for `PinReply.outcome`.
PIN_OUTCOME_DONE = "done"
PIN_OUTCOME_FAILED = "failed"
# An error or timeout after the claim: the device may have acted.
PIN_OUTCOME_UNKNOWN = "unknown"


# How long a pending PIN confirmation stays valid, used by the sweep here and
# by `_load_pending_action` (tools.py) for the one action being confirmed. A
# prompt is normally dropped sooner, by the turn after its last reply turn.
PENDING_ACTION_TTL = timedelta(minutes=10)

# How many of the user's messages may answer one PIN prompt: the first reply,
# plus a retry after a wrong or unreadable PIN or a failed turn.
MAX_PIN_REPLIES = 3

# Hard cap on concurrently pending confirmations, so an unconfirmed challenge
# loop cannot grow the store without bound.
MAX_PENDING_ACTIONS = 16

# Status of a confirmation refused before the PIN is checked: the message is
# not the reply to the action's PIN prompt, the PIN is not in it, or it was
# already used. The action stays pending for the user's reply.
PIN_AWAITING_USER_REPLY = "awaiting_user_pin"
# Key of such a refusal when the user has not been asked for the PIN yet (the
# message is not the prompt's reply): the turn ends with the fixed prompt, so
# the model is not called again to ask, or to try a PIN of its own.
PIN_REFUSAL_ENDS_TURN = "ends_turn"


def pin_refusal(content: str, tool_name: str | None = None) -> dict[str, Any] | None:
    """
    Return the payload of a confirmation refused before the PIN was checked.

    The message was not the reply to the action's PIN prompt, the PIN was not
    in it, or it was already used (a PIN from it failed, or it confirmed an
    action). None for any other tool reply, and, when the tool's name is
    given, for any tool but `confirm_sensitive_action`: a third-party tool's
    result shaped like the refusal must not steer the turn (review finding,
    2026-10-16). The name is matched exactly, and that holds because of how
    `_call_tools` (graph.py) dispatches: a call named `confirm_sensitive_action`
    always runs our LangChain tool, before any HA LLM API tool is considered,
    so a tool message with exactly that name was written by tools.py. An
    API's tool of the same name is shadowed (one API selected: its names
    arrive bare) or namespaced (several selected).
    """
    if tool_name is not None and tool_name != CONFIRM_TOOL_NAME:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(data, dict) and data.get("status") == PIN_AWAITING_USER_REPLY:
        return data
    return None


# The LangChain tool that checks the PIN and runs the held action (tools.py).
CONFIRM_TOOL_NAME = "confirm_sensitive_action"


# Spoken digits as speech-to-text may write them, in the languages the PIN
# messages ship in (pin_messages.py); Czech also without diacritics, except
# "pet" and "tri", which are English words.
_SPOKEN_DIGITS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "nula": "0",
    "jedna": "1",
    "jeden": "1",
    "jedno": "1",
    "dva": "2",
    "dvě": "2",
    "dve": "2",
    "tři": "3",
    "čtyři": "4",
    "ctyri": "4",
    "pět": "5",
    "šest": "6",
    "sest": "6",
    "sedm": "7",
    "osm": "8",
    "devět": "9",
    "devet": "9",
}
# "Oh" for zero, read as a digit only next to another digit ("one oh four",
# "one two three oh").
_SPOKEN_ZERO = "oh"
# Words that may join digits read out one by one ("1, 2, 3 and 4"; Czech "a").
_DIGIT_JOINERS = frozenset({"and", "a"})
# A token of the message: a word or number, a run of whitespace, or one other
# character.
_TOKEN_RE = re.compile(r"\w+|\s+|.", re.DOTALL)
# What may stand between digits read out one by one ("1, 2, 3 and 4",
# "1-2-3-4", "1. 2. 3. 4."). Only whitespace joins longer groups ("12 34").
_DIGIT_SEPARATORS = frozenset(",.-\u2013\u2014")
_DASHES = frozenset("-\u2013\u2014\u2212")
# Between digit groups with spaces around them, these make a range, date or
# time ("555 - 1234", "2026 / 10 / 15", "12 : 34"): never a PIN or part of one.
_RANGE_SEPARATORS = _DASHES | {"/", ":"}
# Units glued to a number make it a measurement ("21°", "1234%"), not a PIN.
_UNIT_MARKS = frozenset("\u00b0%\u2030")
# Joiners inside a compound written with no spaces, such as a time, date,
# decimal, phone number or IP address ("12:34", "2024-10-14", "12,5",
# "555-1234", "1.1.1.1", or an Arabic decimal or thousands mark), which
# is never read as a PIN or split into one.
_COMPOUND_JOINERS = frozenset(".:/,\u066b\u066c") | _DASHES
# Invisible characters speech-to-text or a paste can leave inside a number.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


def _ascii_digits(text: str) -> str:
    """
    Return the text in compatibility form (NFKC), with Unicode digits as ASCII.

    NFKC composes Czech digit words typed or transcribed as a letter plus a
    combining mark ("tři") into one word, and folds full-width, circled and
    superscript characters to plain ones, punctuation included (full-width
    "1234.56" is the decimal 1234.56). Arabic-Indic and other decimal digits
    then read as ASCII; invisible joiners are removed, so "1\u200b234" is
    1234.
    """
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char
        for char in unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    )


def _is_number(token: str) -> bool:
    """Return True for a token of ASCII decimal digits only."""
    return token.isascii() and token.isdecimal()


def _compound_end(tokens: list[str], index: int) -> int:
    """Return the index of the last token of the number starting at index."""
    end = index
    while (
        end + 2 < len(tokens)
        and tokens[end + 1] in _COMPOUND_JOINERS
        # A digit first is enough: "1234.56kWh" is one compound.
        and tokens[end + 2][:1].isascii()
        and tokens[end + 2][:1].isdecimal()
    ):
        end += 2
    return end


def _is_digit_list(parts: list[str], joiners: set[str]) -> bool:
    """Return True for digits read out one by one with no spaces ("1,2,3,4")."""
    return joiners <= {",", *_DASHES} and all(
        len(part) == 1 and _is_number(part) for part in parts
    )


def _thousands(parts: list[str], joiners: set[str]) -> str | None:
    """Return the digits of a number with thousands commas ("1,234"), else None."""
    if (
        joiners == {","}
        and all(_is_number(part) for part in parts)
        and len(parts[0]) <= 3  # noqa: PLR2004
        and all(len(part) == 3 for part in parts[1:])  # noqa: PLR2004
    ):
        return "".join(parts)
    return None


def _compound_items(tokens: list[str]) -> list[tuple[str, str, bool]]:
    """Return the items of a number written with no spaces ("1234", "1,2,3,4")."""
    parts, joiners = tokens[::2], tokens[1::2]
    if not joiners:
        return [("digits", parts[0], True)]
    if _is_digit_list(parts, set(joiners)):
        # One digit at a time, so it joins digits around it like "1, 2, 3, 4"
        # does: "1,2,3,4, 5" is one longer number.
        items: list[tuple[str, str, bool]] = [("digits", parts[0], True)]
        for joiner, part in zip(joiners, parts[1:], strict=True):
            items.extend((("sep", joiner, False), ("digits", part, True)))
        return items
    if digits := _thousands(parts, set(joiners)):
        return [("digits", digits, True)]
    return [("compound", "", True)]


def _number_items(text: str) -> list[tuple[str, str, bool]]:
    """
    Split the text into digit items, separators and breaks.

    Each item is (kind, value, numeral): kind "digits" with the digits and
    whether they were written as numerals, "compound" for a time, date,
    decimal or address (no digits of its own), "sep" with the separator, or
    "break" for anything that ends a number.
    """
    tokens = _TOKEN_RE.findall(_ascii_digits(text))
    items: list[tuple[str, str, bool]] = []

    # Whether a digit comes next, past separators, joining words and other
    # "oh"s, for each token: one backward pass, so a long run of "oh"s stays
    # linear.
    digit_next = [False] * (len(tokens) + 1)
    for position in range(len(tokens) - 1, -1, -1):
        following = tokens[position + 1] if position + 1 < len(tokens) else ""
        lowered = following.lower()
        if (
            following.isspace()
            or following in _DIGIT_SEPARATORS
            or lowered in _DIGIT_JOINERS
            or lowered == _SPOKEN_ZERO
        ):
            digit_next[position] = digit_next[position + 1]
        else:
            digit_next[position] = _is_number(following) or lowered in _SPOKEN_DIGITS

    def after_digit() -> bool:
        # Whitespace and separators only between the last digit and here.
        for kind, _, _ in reversed(items):
            if kind != "sep":
                return kind == "digits"
        return False

    index = 0
    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()
        if _is_number(token):
            end = _compound_end(tokens, index)
            if end + 1 < len(tokens) and tokens[end + 1] in _UNIT_MARKS:
                items.append(("compound", "", True))
                index = end + 2
                continue
            items.extend(_compound_items(tokens[index : end + 1]))
            index = end + 1
            continue
        if lowered in _SPOKEN_DIGITS:
            items.append(("digits", _SPOKEN_DIGITS[lowered], False))
        elif lowered == _SPOKEN_ZERO and (digit_next[index] or after_digit()):
            items.append(("digits", "0", False))
        elif token.isspace():
            items.append(("sep", " ", False))
        elif (
            token in _DIGIT_SEPARATORS
            or token in _RANGE_SEPARATORS
            or lowered in _DIGIT_JOINERS
        ):
            items.append(("sep", lowered, False))
        else:
            # A word, a word joined to digits ("one234", "abc1234"), a digit
            # that is not a decimal digit ("①"), or any other character ends
            # the number.
            items.append(("break", "", False))
        index += 1
    return items


def _joins(
    previous: tuple[str, str, bool], current: tuple[str, str, bool], seps: set[str]
) -> bool:
    """Return True when two digit items separated by seps are one number."""
    if not seps:
        # Whitespace only: "12 34", "one two three four", "555 one two three
        # four" (one longer number, never its last four digits).
        return True
    if seps & _RANGE_SEPARATORS - _DASHES:
        return False
    if len(previous[1]) != 1 or len(current[1]) != 1:
        return False
    # "1. 2. 3. 4." but not "...three four. One door only."
    return "." not in seps or (previous[2] and current[2])


def _number_runs(text: str) -> list[str]:
    """
    Return the digits of each number said in the text, in order.

    A number is written ("1234", "1,234"), said in groups separated by spaces
    ("12 34"), or read out digit by digit with spaces, commas, dashes, ". " or
    "and" between the digits ("1, 2, 3 and 4", "one two three four"). Anything
    else ends it. A compound ("12:34", "12,5", "1.1.1.1", "1234.56kWh", "21°",
    "50%") or a range ("555 - 1234", "2026 / 10 / 15") is not a number at all,
    and neither is a number next to one with only spaces between ("1234
    56.78").
    """
    runs: list[str] = []
    digits = ""
    # Inside a compound or range, whose parts are not numbers of their own.
    in_compound = False
    previous: tuple[str, str, bool] | None = None
    seps: set[str] = set()
    for item in _number_items(text):
        kind, value, _ = item
        if kind == "sep":
            if value != " ":
                seps.add(value)
            continue
        numeric = kind in {"digits", "compound"}
        if numeric and previous is not None:
            if (
                kind == "digits"
                and previous[0] == "digits"
                and _joins(previous, item, seps)
            ):
                if not in_compound:
                    digits += value
                previous, seps = item, set()
                continue
            if seps & _RANGE_SEPARATORS or (
                not seps and "compound" in {kind, previous[0]}
            ):
                digits, in_compound = "", True
                previous, seps = item, set()
                continue
        if digits:
            runs.append(digits)
        in_compound = kind == "compound"
        digits = value if kind == "digits" else ""
        previous = item if numeric else None
        seps = set()
    if digits:
        runs.append(digits)
    return runs


def pin_digits(text: str) -> str | None:
    """
    Return the digits of the one number in the text, else None.

    For the PIN as the model passes it, which may be said in words or
    labelled ("one two three four", "PIN 1234"); two numbers give None.
    """
    runs = _number_runs(text)
    return runs[0] if len(runs) == 1 else None


def normalize_pin(pin: str) -> str:
    """
    Return a PIN as entered, folded the way the user's message is read.

    Full-width, circled, superscript and Arabic-Indic digits become ASCII, so
    setup stores the PIN that confirmation hashes.
    """
    return _ascii_digits(pin.strip())


def pin_in_text(pin: str, text: str) -> bool:
    """
    Return True when one number said in the user's message is exactly the PIN.

    The number can be written ("1234", "1,234"), said in groups ("12 34") or
    read out digit by digit, as digits or spoken digit words ("1, 2, 3, 4",
    "one two three four"). The whole number must equal the PIN: a longer
    number that contains it ("12345", "555 1234", "nine one two three four")
    does not count, and neither does a time, date, decimal or address
    ("12:34", "12.34", "1.2.3.4"). A number standing apart from other numbers
    and equal to the PIN always counts, whatever it means to the user ("set
    it to 1234", "$1234").
    """
    if not pin or not text:
        return False
    return _ascii_digits(pin) in _number_runs(text)


class CriticalActionPolicy(NamedTuple):
    """Resolved critical-action PIN policy for a config entry."""

    enabled: bool
    pin_hash: str
    pin_salt: str
    critical_actions: list[dict[str, str]]

    @property
    def enforceable(self) -> bool:
        """Return True when the PIN is enabled *and* actually configured."""
        return self.enabled and bool(self.pin_hash and self.pin_salt)


def resolve_critical_action_policy(options: Mapping[str, Any]) -> CriticalActionPolicy:
    """Resolve the critical-action PIN policy from integration options."""
    pin_hash = options.get(CONF_CRITICAL_ACTION_PIN_HASH) or ""
    pin_salt = options.get(CONF_CRITICAL_ACTION_PIN_SALT) or ""
    return CriticalActionPolicy(
        # Always respect a configured PIN, even if the toggle somehow reads False.
        enabled=bool(
            options.get(CONF_CRITICAL_ACTION_PIN_ENABLED, False)
            or (pin_hash and pin_salt)
        ),
        pin_hash=pin_hash,
        pin_salt=pin_salt,
        # Copy: the default is a module-level list handed to every config
        # entry, so an in-place edit anywhere would corrupt it process-wide.
        critical_actions=list(
            options.get(CONF_CRITICAL_ACTIONS) or RECOMMENDED_CRITICAL_ACTIONS
        ),
    )


def pending_action_expired(action: Mapping[str, Any], now: datetime) -> bool:
    """
    Return True when a pending action is past its TTL or has no usable time.

    A record with no usable timestamp could never expire, so it counts as
    expired; a timezone-naive one cannot be compared with the aware `now`.
    """
    created_at = action.get("created_at")
    if not isinstance(created_at, str):
        return True
    try:
        return now - datetime.fromisoformat(created_at) > PENDING_ACTION_TTL
    except (ValueError, TypeError):
        return True


def _request_key(action: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """
    Return what makes two pending actions the same request.

    The arguments as they would run (intent normalized, sanitized, names
    casefolded), and the automation without the id add_automation generates
    on every call, so a model re-issuing the request unchanged is recognized.
    """
    tool_name = action.get("tool_name")
    tool_args = action.get("tool_args") or {}
    if isinstance(tool_name, str) and isinstance(tool_args, dict):
        # Without hass: both actions compared were held after normalizing
        # with hass, so their stored arguments already carry the domain.
        tool_args = normalize_intent_for_alarm(tool_name, tool_args, None)
        tool_args = sanitize_tool_args(
            normalize_intent_for_lock(tool_name, tool_args, None)
        )
        tool_args = {
            key: value.casefold() if key == "name" and isinstance(value, str) else value
            for key, value in tool_args.items()
        }
    automation = action.get("automation_config")
    if isinstance(automation, dict):
        automation = {key: value for key, value in automation.items() if key != "id"}
    return tool_name, tool_args, automation


def _same_request(existing: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    """Return True when the action re-issues the existing one unchanged."""
    return _request_key(existing) == _request_key(action)


def register_pending_action(
    pending_actions: dict[str, dict[str, Any]],
    action: dict[str, Any],
) -> str | None:
    """
    Store a pending PIN-confirmation action and return its action ID.

    Sweeps expired entries and caps the store on every registration, so
    challenges the model abandons are not kept for the life of the entry.

    A conversation has at most one pending action per user, so the user's PIN
    only ever confirms their latest request: with two, the model would choose
    which one the PIN runs. A new request replaces one from an earlier turn,
    keeping its failed attempts so a new request does not reset the limit. A
    second, different request in the same turn is refused (None): the first
    one's prompt is already on its way to the user.

    Re-issuing the same request keeps its prompt, so the model calling the
    tool again in the reply turn is not locked out. Re-issued in the reply
    turn, the request was most likely repeated by the user, who now gets the
    prompt again, so the prompt is also kept for their next message. An
    expired action is dropped first and carries nothing.
    """
    now = dt_util.utcnow()
    thread_id = action.get("thread_id")
    carried: dict[str, Any] = {}
    for key, existing in list(pending_actions.items()):
        if pending_action_expired(existing, now):
            pending_actions.pop(key, None)
        elif (
            thread_id
            and existing.get("thread_id") == thread_id
            and existing.get("user") == action.get("user")
        ):
            if _same_request(existing, action) and existing.get("turn_id"):
                reissued_in_reply = bool(
                    existing.get("reply_turn")
                    and existing.get("reply_turn") == action.get("turn_id")
                )
                if reissued_in_reply and not prompt_renewable(existing):
                    # The prompt has taken all its replies, so the request
                    # said again starts a fresh one (answered by the next
                    # message) rather than one that promises a reply it
                    # cannot take. The attempts still carry over.
                    carried = {}
                else:
                    carried = {
                        name: existing[name]
                        for name in (
                            "turn_id",
                            "reply_turn",
                            "replies",
                            "renew",
                            "disclosed",
                        )
                        if name in existing
                    }
                    if reissued_in_reply:
                        carried["renew"] = True
            elif existing.get("turn_id") == action.get("turn_id"):
                return None
            carried["attempts"] = int(existing.get("attempts", 0) or 0)
            pending_actions.pop(key, None)

    while len(pending_actions) >= MAX_PENDING_ACTIONS:
        pending_actions.pop(next(iter(pending_actions)), None)

    action_id = ulid.ulid_now()
    pending_actions[action_id] = {
        **action,
        "attempts": 0,
        **carried,
        "created_at": now.isoformat(),
    }
    return action_id


def start_pin_turn(
    pending_actions: dict[str, dict[str, Any]], thread_id: str, text: str
) -> PinReply:
    """
    Open a user turn in a conversation and return its message as a PinReply.

    A pending action is answered by the user's next message or not at all:
    the first turn after the one that created it is its reply turn, and the
    turn after that drops it. A prompt the user moved on from ("never mind",
    or just another request) cannot be confirmed later by a number that
    happens to equal the PIN.

    While the user is still answering, the prompt is kept for one more
    message: after a wrong or unreadable PIN, the request repeated, or a
    turn that failed (see `keep_pin_prompts`), up to MAX_PIN_REPLIES replies.
    Expired actions are dropped here too, and so is a prompt whose fixed line
    (`disclose_pin_prompts`) was never delivered: the turn that made it was
    cancelled or failed before its reply, so the user was never told, in
    words the model did not choose, what their PIN would run.
    """
    reply = PinReply(text)
    turn_id = reply.turn_id
    now = dt_util.utcnow()
    for key, action in list(pending_actions.items()):
        if pending_action_expired(action, now):
            pending_actions.pop(key, None)
            continue
        if action.get("thread_id") != thread_id or action.get("turn_id") == turn_id:
            continue
        if not action.get("disclosed"):
            pending_actions.pop(key, None)
            continue
        replies = int(action.get("replies", 0) or 0)
        renew = action.pop("renew", False) and prompt_renewable(action)
        if action.get("reply_turn") is None or renew:
            action["reply_turn"] = turn_id
            action["replies"] = replies + 1
        else:
            pending_actions.pop(key, None)
    return reply


def keep_pin_prompt(action: dict[str, Any], reply: PinReply | None) -> bool:
    """
    Keep the action's prompt for the user's next message, if this turn answers it.

    Returns False when it is not kept: this is not the prompt's reply turn, or
    the prompt has taken its MAX_PIN_REPLIES replies.
    """
    if reply is None or action.get("reply_turn") != reply.turn_id:
        return False
    if not prompt_renewable(action):
        return False
    action["renew"] = True
    return True


def prompt_renewable(action: Mapping[str, Any]) -> bool:
    """Return True while the prompt has not taken all its MAX_PIN_REPLIES replies."""
    return int(action.get("replies", 0) or 0) < MAX_PIN_REPLIES


def keep_pin_prompts(
    pending_actions: dict[str, dict[str, Any]], reply: PinReply
) -> None:
    """
    Keep every prompt this turn answers for the user's next message.

    For a turn that failed (a model or graph error): the user's reply never
    got an answer, so they can send the PIN again.
    """
    for action in pending_actions.values():
        keep_pin_prompt(action, reply)


def prompts_awaiting_reply(
    pending_actions: Mapping[str, dict[str, Any]], reply: PinReply
) -> list[dict[str, Any]]:
    """
    Return the actions whose PIN prompt the user's next message may answer.

    A prompt made this turn, and one this turn answered that was kept for
    another message (`keep_pin_prompt`), while it can still take one.
    """
    return [
        action
        for action in pending_actions.values()
        if action.get("turn_id") == reply.turn_id
        or (
            action.get("reply_turn") == reply.turn_id
            and action.get("renew")
            and prompt_renewable(action)
        )
    ]


def pin_lines_follow(
    pending_actions: Mapping[str, dict[str, Any]], reply: PinReply
) -> bool:
    """
    Return True when the turn's reply carries a fixed PIN line.

    The lines themselves are built in conversation.py (`_pin_lines`) from the
    same three facts: a PIN from this message ran an action (its outcome), a
    PIN from it failed the check, or a prompt awaits the user's next message.
    The graph's round-limit guard uses this to add nothing when a line will
    be the reply.
    """
    return bool(
        reply.confirmed or reply.spent or prompts_awaiting_reply(pending_actions, reply)
    )


def disclose_pin_prompts(
    pending_actions: Mapping[str, dict[str, Any]], reply: PinReply
) -> list[dict[str, Any]]:
    """
    Mark the prompts this turn's reply names as delivered, and return them.

    Called once the fixed line naming each action is in the reply. Only a
    disclosed prompt can be answered (`start_pin_turn`): the user has been
    told, in words the model did not choose, what their PIN would run.
    """
    actions = prompts_awaiting_reply(pending_actions, reply)
    for action in actions:
        action["disclosed"] = True
    return actions


def thread_has_pending_action(
    pending_actions: Mapping[str, Mapping[str, Any]], thread_id: str | None
) -> bool:
    """Return True when this conversation has an unexpired pending action."""
    if not thread_id:
        return False
    now = dt_util.utcnow()
    return any(
        action.get("thread_id") == thread_id and not pending_action_expired(action, now)
        for action in pending_actions.values()
    )


def matches_critical_rule(
    *,
    domain: str,
    service: str,
    entity_ids: Sequence[str],
    critical_actions: Iterable[Mapping[str, str]],
    unresolved_target: bool = False,
) -> bool:
    """
    Return True when a domain/service call matches a critical-action rule.

    Shared by the conversation-tool guard (``_is_critical_action``) and the
    automation-content guard (``automation_pin``) so both enforce identical
    rule semantics.

    ``unresolved_target`` fails closed for ``entity_match`` rules: when a call
    targets an area, device, label, or template, the entity IDs it will hit are
    unknown at check time, so an otherwise-matching rule is treated as a match.
    """
    domain = domain.lower()
    service = service.lower()
    entities = [e.lower() for e in entity_ids]

    for rule in critical_actions:
        rule_domain = (rule.get("domain") or "").lower()
        rule_service = (rule.get("service") or "").lower()
        entity_match = (rule.get("entity_match") or "").lower()

        if rule_domain and rule_domain != domain:
            continue
        if rule_service and rule_service != service:
            continue
        if (
            entity_match
            and not any(entity_match in e for e in entities)
            and not unresolved_target
        ):
            continue
        return True
    return False


def _payload_target(tool_args: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """
    Return the (name, entity_id) a payload may carry, as HA will read them.

    HA reads a name of "all" as no name, so a single entity id the model gave
    is the better target; a list of ids is read like one id.
    """
    name = tool_args.get("name")
    if name == INTENT_NAME_ALL or not isinstance(name, str) or not name:
        name = None
    entity_id = tool_args.get("entity_id") or tool_args.get("entity_ids")
    if isinstance(entity_id, list):
        entity_id = entity_id[0] if entity_id else None
    if not isinstance(entity_id, str) or not entity_id:
        entity_id = None
    return name, entity_id


def _lock_payload(
    tool_args: Mapping[str, Any], name: str | None, entity_id: str | None
) -> dict[str, Any]:
    """Return the payload for a lock intent, narrowed to one lock or one place."""
    if not entity_id and name:
        slug = str(name).strip().lower().replace(" ", "_")
        entity_id = f"lock.{slug}"
    payload: dict[str, Any] = {
        "domain": ["lock"],
        "service": tool_args.get("service"),
    }
    # The payload must carry a target NAME. HA's turn-on/turn-off intents
    # resolve targets from name/area/floor and never read an `entity_id`
    # slot (`helpers/intent.py` contains no such lookup), while `domains`
    # alone already satisfies `MatchTargetsConstraints.has_constraints` --
    # so domain+entity_id sails past the "cannot target all devices" guard
    # and matches EVERY exposed lock. `_filter_by_name` accepts an entity
    # id as the name value, so one key narrows both shapes to one entity.
    if target := (name or entity_id):
        payload["name"] = target
    if entity_id:
        payload["entity_id"] = entity_id
    # The area or floor the user named narrows a call with no name to
    # the locks there, rather than every lock in the home.
    # HA reads a number as text (floor 0 is the floor "0"), so only a
    # missing or blank value is dropped; dropping 0 would widen the call
    # to every lock in the home.
    for key in ("area", "floor"):
        value = tool_args.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            payload[key] = value
    if code := tool_args.get("code"):
        payload["code"] = code
    return payload


def _alarm_payload(tool_args: Mapping[str, Any]) -> dict[str, Any]:
    """Return the payload for an alarm panel intent."""
    payload: dict[str, Any] = {
        "domain": ["alarm_control_panel"],
        "service": tool_args.get("service"),
    }
    for key in ("entity_id", "code", "state"):
        if value := tool_args.get(key):
            payload[key] = value
    return payload


def minimal_payload_for_domain(tool_args: dict[str, Any]) -> dict[str, Any]:
    """
    Return a reduced payload for HA services to avoid extra keys.

    The normal tool path applies this after the critical-action guard, and
    the PIN confirmation applies it to the held call before running it, so
    the two send the same payload (the fixed line names what this sends).
    """
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    domain = str(domains[0]).lower() if domains else ""
    name, entity_id = _payload_target(tool_args)

    if domain == "lock":
        return _lock_payload(tool_args, name, entity_id)
    if domain == "alarm_control_panel":
        return _alarm_payload(tool_args)
    # Any other domain: HA's intents ignore an entity id, so without a name
    # the whole domain acts. An entity id the model gave is sent as the name
    # (`_filter_by_name` accepts one), which narrows the call to that device.
    if domain and not name and entity_id:
        return {**tool_args, "name": entity_id}
    return tool_args


# The target name Home Assistant's intents read as "no name".
INTENT_NAME_ALL = "all"


def parse_tool_response(
    response: Any, tool_args: dict[str, Any]
) -> tuple[str, str | None]:
    """
    Return (content, status) from a tool response.

    Shared by the normal tool path and the PIN confirmation, so a tool that
    reports failure in its payload (`success: false`, a `warning`, an alarm
    that did not change state) reads as a failure on both.
    """
    parsed: dict[str, Any] | None = None
    if isinstance(response, dict):
        parsed = response
    elif isinstance(response, str):
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            decoded = None
        # A JSON list, number or bool carries no status fields.
        parsed = decoded if isinstance(decoded, dict) else None

    status: str | None = None
    content_str = json.dumps(response, default=str)
    if parsed is None:
        return content_str, status

    warning = parsed.get("warning")
    success_val = parsed.get("success", True)
    result_text = (
        parsed.get("result_text") or parsed.get("response") or parsed.get("result")
    )
    inferred_state = parsed.get("inferred_state")

    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    is_alarm = any(str(d).lower() == "alarm_control_panel" for d in domains)
    service = str(tool_args.get("service") or "").lower()

    if warning:
        status = "error"
        content_str = str(result_text or warning)
    elif success_val is False:
        status = "error"
        content_str = str(result_text or "Action failed.")
    elif is_alarm:
        if "arm" in service and inferred_state not in {
            "armed",
            "armed_home",
            "armed_away",
            "armed_night",
            "arming",
        }:
            status = "error"
            content_str = str(
                result_text
                or f"Alarm did not change state (panel_state={inferred_state})."
            )
        if "disarm" in service and inferred_state not in {
            "disarmed",
            "disarming",
        }:
            status = "error"
            content_str = str(
                result_text
                or f"Alarm did not change state (panel_state={inferred_state})."
            )

    return content_str, status


def sanitize_tool_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Remove empty/None slot values for HA intent tools to avoid validation errors."""
    cleaned: dict[str, Any] = {}
    for key, val in tool_args.items():
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if isinstance(val, list) and all((not v and v != 0) for v in val):
            continue
        cleaned[key] = val
    if "domain" in cleaned and isinstance(cleaned["domain"], str):
        cleaned["domain"] = [cleaned["domain"]]
    return cleaned


def maybe_fill_lock_entity(
    tool_args: dict[str, Any], hass: HomeAssistant | None
) -> dict[str, Any]:
    """
    Name the one lock a lock call means when HA cannot match it as given.

    Matched with Home Assistant's own matcher, among locks exposed to Assist,
    so a name in any language or an alias reads as HA reads it. A name HA
    already matches is left as is. Otherwise the name with a trailing "lock"
    added or removed ("Front Door" for "Front Door Lock", and the reverse)
    is tried, and filled in only when exactly one lock answers. Never a near
    match: comparing names reduced to ASCII sent a call meant for one lock,
    or another device, to a different door (#782), so a name that matches no
    lock is left for Home Assistant to refuse.
    """
    if not hass or tool_args.get("entity_id"):
        return tool_args
    if _LOCK_DOMAIN not in _declared_domains(tool_args):
        return tool_args
    name = tool_args.get("name")
    if not isinstance(name, str) or not name.strip():
        return tool_args
    as_given = _match_exposed(hass, name, _LOCK_DOMAIN)
    if as_given is None or as_given:
        return tool_args

    stripped = name.strip()
    variants = [f"{stripped} {_LOCK_WORD}"]
    if stripped.casefold().endswith(f" {_LOCK_WORD}"):
        variants.append(stripped[: -len(_LOCK_WORD)].strip())
    found: dict[str, str] = {}
    for variant in variants:
        for state in _match_exposed(hass, variant, _LOCK_DOMAIN) or []:
            found[state.entity_id] = state.name
    if len(found) == 1:
        ((entity_id, friendly),) = found.items()
        tool_args["entity_id"] = entity_id
        tool_args["name"] = friendly
    return tool_args


_LOCK_DOMAIN = "lock"
_ALARM_DOMAIN = "alarm_control_panel"
# The word a lock name may carry or leave off ("Front Door" / "Front Door Lock").
_LOCK_WORD = "lock"
# The assistant this agent answers as; Assist exposure is kept per assistant.
_ASSIST_ASSISTANT = "conversation"


def _declared_domains(tool_args: Mapping[str, Any]) -> set[str]:
    """Return the domains a call declares, read the way the normalizers read them."""
    domains = tool_args.get("domain") or []
    domains = domains if isinstance(domains, list) else [domains]
    return {str(d).strip().lower() for d in domains if str(d).strip()}


def _match_exposed(
    hass: HomeAssistant, name: str, domain: str | None = None
) -> list[State] | None:
    """
    Return the entities HA's matcher finds for `name`, as Assist sees them.

    Limited to entities exposed to Assist, as HA's own intent handlers are,
    so a hidden device never becomes a call's target or shows in a PIN line
    (#716). None when the matcher raised, which is logged.
    """
    from homeassistant.components.homeassistant.const import (  # noqa: PLC0415
        DATA_EXPOSED_ENTITIES,
    )

    if name.strip() == INTENT_NAME_ALL:
        return []
    try:
        result = intent.async_match_targets(
            hass,
            intent.MatchTargetsConstraints(
                name=name,
                domains=[domain] if domain else None,
                assistant=_ASSIST_ASSISTANT
                if DATA_EXPOSED_ENTITIES in hass.data
                else None,
                allow_duplicate_names=True,
            ),
        )
    except Exception:
        LOGGER.warning("Could not match the name %r", name, exc_info=True)
        return None
    return list(result.states) if result.is_match else []


def _name_matches_only(hass: HomeAssistant | None, name: Any, domain: str) -> bool:
    """Return True when HA's matcher resolves `name` to `domain` and nothing else."""
    if hass is None or not isinstance(name, str) or not name.strip():
        return False
    matches = _match_exposed(hass, name)
    return bool(matches) and all(state.domain == domain for state in matches)


# Separator Home Assistant puts between a tool's namespace and its own name.
# HA 2026.9 moved the built-in intent tools into per-integration `llm.py`
# modules that name them `f"{DOMAIN}__{intent_type}"` (`intent__HassTurnOff`,
# `light__HassLightSet`, `homeassistant__GetLiveContext`), and `MergedAPI`
# wraps tools from multiple APIs in a `NamespacedTool` that prepends a second
# prefix. Every comparison against a known tool name must therefore run on the
# base name: a bare-name check silently stops matching after an HA upgrade,
# which is how the critical-action PIN gate and the actuation classifier both
# went dead without a single error in the log.
TOOL_NAMESPACE_SEP = "__"


def base_tool_name(name: str) -> str:
    """Return a tool name with any Home Assistant namespace prefixes stripped."""
    if not name:
        return name
    return name.rsplit(TOOL_NAMESPACE_SEP, 1)[-1]


def is_on_off_intent(tool_name: str) -> bool:
    """Return True for the HA turn-on/turn-off intent tools, namespaced or not."""
    return base_tool_name(tool_name) in {"HassTurnOn", "HassTurnOff"}


def normalize_intent_for_alarm(
    tool_name: str, tool_args: dict[str, Any], hass: HomeAssistant | None
) -> dict[str, Any]:
    """
    Route alarm control panel intents to the proper service.

    A call is for an alarm panel when the model declares that domain, or
    declares none and Home Assistant's matcher resolves the name to alarm
    panels only. Never from the letters of the name: "Security Light" was
    sent to the alarm tool, and "Security Door Lock" lost its lock (#782).
    """
    if not is_on_off_intent(tool_name):
        return tool_args

    declared = _declared_domains(tool_args)
    if declared:
        is_alarm = _ALARM_DOMAIN in declared
    else:
        is_alarm = _name_matches_only(hass, tool_args.get("name"), _ALARM_DOMAIN)
    if not is_alarm:
        return tool_args

    is_arm = base_tool_name(tool_name) == "HassTurnOn"
    desired_service = "alarm_arm_home" if is_arm else "alarm_disarm"
    tool_args = {
        **tool_args,
        "domain": ["alarm_control_panel"],
        "service": desired_service,
    }
    if "entity_id" not in tool_args and (name := tool_args.get("name")):
        slug = str(name).strip().lower().replace(" ", "_")
        tool_args["entity_id"] = f"alarm_control_panel.{slug}"
    return tool_args


def normalize_intent_for_lock(
    tool_name: str, tool_args: dict[str, Any], hass: HomeAssistant | None
) -> dict[str, Any]:
    """
    Normalize lock intents: set domain and service for lock/unlock.

    A call is a lock call when the model declares the `lock` domain, or
    declares none and the name is one Home Assistant's matcher resolves to
    locks only. Never from the letters of the name: "lock" inside "Kitchen
    Clock" or a scene named "Unlock" turned a switch or a scene into a call
    on a door lock (#782).
    """
    if not is_on_off_intent(tool_name):
        return tool_args

    declared = _declared_domains(tool_args)
    if declared:
        is_lock = _LOCK_DOMAIN in declared
    else:
        is_lock = _name_matches_only(hass, tool_args.get("name"), _LOCK_DOMAIN)
    if not is_lock:
        return tool_args

    normalized = {**tool_args, "domain": ["lock"]}
    if base_tool_name(tool_name) == "HassTurnOff":
        normalized.setdefault("service", "unlock")
    else:
        normalized.setdefault("service", "lock")
    return normalized


# The services Home Assistant's own intent handlers call for a matched
# entity, by its domain (`components/intent/__init__.py`): the on/off intents
# open and close covers and valves, lock and unlock locks, press buttons (and
# refuse to turn one off), and call `turn_on`/`turn_off` on everything else;
# HassSetPosition sets a cover's or a valve's position. The `domain` slot only
# narrows the match, it never picks the service, so the gate must read the
# service from each matched entity.
_ON_OFF_INTENT_SERVICES: dict[str, tuple[str | None, str | None]] = {
    "cover": ("open_cover", "close_cover"),
    "valve": ("open_valve", "close_valve"),
    "lock": ("lock", "unlock"),
    "button": ("press", None),
    "input_button": ("press", None),
}
_SET_POSITION_SERVICES: dict[str, str] = {
    "cover": "set_cover_position",
    "valve": "set_valve_position",
}
_STOP_MOVING_SERVICES: dict[str, str] = {
    "cover": "stop_cover",
    "valve": "stop_valve",
}
SET_POSITION_INTENT = "HassSetPosition"
STOP_MOVING_INTENT = "HassStopMoving"
# The intents that act on covers and valves only; HA refuses any other match.
_COVER_VALVE_INTENTS: dict[str, dict[str, str]] = {
    SET_POSITION_INTENT: _SET_POSITION_SERVICES,
    STOP_MOVING_INTENT: _STOP_MOVING_SERVICES,
}


class IntentServiceCall(NamedTuple):
    """One service call an intent makes: its domain, service and entity."""

    domain: str
    service: str
    entity_id: str | None


class _UnclassifiableSlotError(ValueError):
    """A slot value whose meaning to Home Assistant the gate cannot read."""


def is_actuating_intent(tool_name: str) -> bool:
    """Return True for the intents whose service depends on the target."""
    return is_on_off_intent(tool_name) or (
        base_tool_name(tool_name) in _COVER_VALVE_INTENTS
    )


def _intent_services(tool_name: str, domain: str, entity_id: str | None) -> list[str]:
    """
    Return the services an actuating intent calls on an entity of `domain`.

    A script turned on runs, so a rule written for the script's own service
    (`script.open_garage`) applies to it as well as one for `script.turn_on`.
    """
    if (services := _COVER_VALVE_INTENTS.get(base_tool_name(tool_name))) is not None:
        service = services.get(domain)
        return [service] if service else []
    turns_on = base_tool_name(tool_name) == "HassTurnOn"
    on, off = _ON_OFF_INTENT_SERVICES.get(domain, ("turn_on", "turn_off"))
    service = on if turns_on else off
    if service is None:
        return []
    services = [service]
    if domain == "script" and turns_on and entity_id and "." in entity_id:
        services.append(entity_id.split(".", 1)[1])
    return services


def _intent_calls(
    tool_name: str, domain: str, entity_id: str | None
) -> list[IntentServiceCall]:
    """Return the service calls an actuating intent makes on one entity."""
    return [
        IntentServiceCall(domain, service, entity_id)
        for service in _intent_services(tool_name, domain, entity_id)
    ]


def _slot_text(value: Any) -> str | None:
    """
    Return a slot value as Home Assistant's slot schema reads it.

    HA omits a blank slot and coerces anything else with `cv.string`, which
    turns a number into its text (`floor: 1` is the floor "1") and rejects a
    list or a dict. Reading a number as absent would drop a target HA uses.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (list, dict)):
        raise _UnclassifiableSlotError(value)
    return str(value)


def _text_list(value: Any) -> list[str]:
    """Return a list slot (`domain`, `device_class`) as lowercase text."""
    values = value if isinstance(value, list) else [value]
    texts = [_slot_text(v) for v in values]
    return [t.strip().lower() for t in texts if t]


def intent_service_calls(
    hass: HomeAssistant | None, tool_name: str, tool_args: Mapping[str, Any]
) -> list[IntentServiceCall] | None:
    """
    Return the service calls an actuating intent will make, one per entity.

    Matched with Home Assistant's own `async_match_targets` on the payload the
    call is sent as (`minimal_payload_for_domain`), with the slot values read
    the way HA's slot schema reads them, so the gate checks what HA will act
    on rather than what the model wrote: a cover opened by name alone, a
    garage door named only by its entity id, every device in an area. The
    match ignores Assist exposure and allows duplicate names, so it finds at
    least the entities HA will act on. Returns None when it cannot tell: no
    `hass`, a slot shape it cannot read, or a matcher that raised. The gate
    then treats the call as critical, or with no `hass` checks the domain and
    entity ids the model gave.
    """
    if hass is None or not is_actuating_intent(tool_name):
        return None
    payload = minimal_payload_for_domain(dict(tool_args))
    try:
        name = _slot_text(payload.get("name"))
        area = _slot_text(payload.get("area"))
        floor = _slot_text(payload.get("floor"))
        domains = _text_list(payload.get("domain"))
        device_classes = _text_list(payload.get("device_class"))
    except _UnclassifiableSlotError:
        return None
    if name == INTENT_NAME_ALL:
        # HA's own comparison: exact, so " all " stays a name it looks up.
        name = None
    # No domains are added for the cover/valve intents: HA adds none, so a
    # call with no target is refused there, and a light it matches has no
    # service in `_intent_services`.
    constraints = intent.MatchTargetsConstraints(
        name=name,
        area_name=area,
        floor_name=floor,
        domains=domains or None,
        device_classes=device_classes or None,
        allow_duplicate_names=True,
    )
    if not constraints.has_constraints:
        # HA refuses an intent that would target every device.
        return []
    try:
        result = intent.async_match_targets(hass, constraints)
    except Exception:
        LOGGER.warning("Could not match the targets of %s", tool_name, exc_info=True)
        return None
    if not result.is_match:
        return []
    return [
        call
        for state in result.states
        for call in _intent_calls(tool_name, state.domain, state.entity_id)
    ]


def declared_intent_service_calls(
    tool_name: str, tool_args: Mapping[str, Any]
) -> list[IntentServiceCall] | None:
    """
    Return the service calls an actuating intent declares, without `hass`.

    From the domain slot, or the domain of the entity ids the model gave.
    With no entity id the target is unknown, so the call carries none and
    an `entity_match` rule treats it as matching. None for a slot shape it
    cannot read, which the gate holds.
    """
    if not is_actuating_intent(tool_name):
        return []
    try:
        entity_ids = [
            e
            for e in _text_list(tool_args.get("entity_id"))
            + _text_list(tool_args.get("entity_ids"))
            if "." in e
        ]
        domains = _text_list(tool_args.get("domain"))
    except _UnclassifiableSlotError:
        return None
    calls = [
        call
        for entity_id in entity_ids
        if not domains or entity_id.split(".", 1)[0] in domains
        for call in _intent_calls(tool_name, entity_id.split(".", 1)[0], entity_id)
    ]
    if calls:
        return calls
    return [
        call for domain in domains for call in _intent_calls(tool_name, domain, None)
    ]


# Separator between the API id and the tool name in a composite tool key.
# Shared by `tool_index_key` and `split_tool_index_key` so the join and the
# split can never drift — the options flow round-trips these keys through a
# flat picker value, and a one-sided change would void every stored exclusion.
TOOL_KEY_SEP = "::"

# Cap for any single rendered tool string. Long enough for real MCP tool names,
# short enough that one hostile name cannot flood a log line or a form label.
TOOL_TEXT_MAX_LEN = 120


def sanitize_tool_text(text: str, limit: int = TOOL_TEXT_MAX_LEN) -> str:
    """
    Make a remote-controlled tool string safe to render, and bound its length.

    Tool names and API titles arrive from remote MCP servers, so they are
    attacker-controlled text that this integration renders in two places: log
    lines and the options-form exclusion picker.  ``str.isprintable()`` is
    False for exactly the classes that make rendered text lie — C0/C1 controls,
    newlines, zero-width characters, and the bidi overrides (U+202E and
    friends) that reverse the text after them — so replacing every
    non-printable with ``?`` closes log-line forgery and the reordering
    attacks.  ASCII space stays printable, so ordinary names are untouched.

    What this deliberately does **not** do is stop a name whose characters are
    all perfectly printable from *reading* like something else.
    ``list_files (not currently available)`` passes through here unchanged, so
    this function alone cannot keep remote text from imitating a trust marker
    it is concatenated with; the picker's own ``_label_text`` in
    ``config_flow.py`` is what does that.  Treat this as making remote text
    safe to *render*, not safe to *concatenate with trusted text*.

    Single definition on purpose: a second copy would drift from this one, and
    a renderer that forgot to sanitize is exactly the hole this closes.

    The cap is applied *before* the per-character pass, not after. The
    transform is one character in, one character out, so the two orders return
    byte-identical strings — but capping afterwards would run the Python-level
    generator across the whole input first, which is unbounded in the length of
    text a remote server chose. A multi-megabyte tool name costs ~132ms of
    event-loop block that way against ~0.05ms this way, which is the difference
    between a cap that bounds rendering and a cap that only bounds the result.
    """
    # Fast path: one C-level scan, and it is what virtually every real tool
    # name hits, so the common case never enters the generator at all.
    if len(text) <= limit and text.isprintable():
        return text
    return "".join(ch if ch.isprintable() else "?" for ch in text[:limit])


def tool_index_key(api_id: str, name: str) -> str:
    """
    Composite tool-index store key.

    Single definition on purpose: the per-turn top-up compares live keys
    against cached hashes, so any drift between the key spelling used for
    discovery writes and the one used for the live-set comparison would make
    a key permanently "missing" and re-fire the top-up every turn.
    """
    return f"{api_id}{TOOL_KEY_SEP}{name}"


def split_tool_index_key(key: str) -> tuple[str, str] | None:
    """
    Split a composite tool key back into ``(api_id, tool_name)``.

    Returns ``None`` for anything that is not a well-formed key.  Splits on the
    *first* separator: registered API ids never contain ``::`` (Home Assistant
    builds them from a domain or a config-entry id), while a remote MCP server
    is free to name a tool anything at all.
    """
    api_id, sep, name = key.partition(TOOL_KEY_SEP)
    if not sep or not api_id or not name:
        return None
    return api_id, name


_SCHEMA_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    PydanticInvalidForJsonSchema,
)

_EMPTY_SCHEMA = "{}"

# Memo of the extracted schema string, keyed by tool identity.  ``BaseTool`` is
# a Pydantic model, so it is neither hashable nor usable as a WeakKeyDictionary
# key -- hence id() plus a weakref that proves the id still names the object it
# was recorded for.  Without that proof a freed tool's recycled id would serve
# a stale schema to whatever object landed on the address next.
_schema_memo: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


def _drop_schema_memo(key: int, dead: weakref.ReferenceType[Any]) -> None:
    """Evict a memo entry once its tool is collected."""
    # The local tools are singletons, so in practice nothing is ever evicted.
    # This exists so the memo cannot become a leak if per-turn tool objects
    # (an MCP set rebuilt each turn) are ever routed through here: the weakref
    # would die while its dict entry lived on forever.  Guarded on identity so
    # a late callback cannot delete the entry of a live tool that has already
    # claimed the recycled id.
    if _schema_memo.get(key, (None, ""))[0] is dead:
        del _schema_memo[key]


def _dump_json_schema(schema_obj: Any) -> str:
    """Serialize a dict or Pydantic model to a JSON-schema string."""
    if isinstance(schema_obj, dict):
        try:
            return json.dumps(schema_obj, sort_keys=True)
        except (TypeError, ValueError):
            return _EMPTY_SCHEMA
    try:
        model_json_schema = getattr(schema_obj, "model_json_schema", None)
        if callable(model_json_schema):
            dumped = model_json_schema()
            if isinstance(dumped, dict):
                return json.dumps(dumped, sort_keys=True)
        schema_func = getattr(schema_obj, "schema", None)
        if callable(schema_func):
            dumped = schema_func()
            if isinstance(dumped, dict):
                return json.dumps(dumped, sort_keys=True)
    except _SCHEMA_ERRORS:
        return _EMPTY_SCHEMA
    return _EMPTY_SCHEMA


def _extract_tool_parameters_json(lc_tool: Any) -> str:
    """Extract the model-facing schema string for one tool (uncached)."""
    # getattr() evaluates tool_call_schema, which is a *computed* property --
    # it runs get_input_schema() + _create_subset_model() on every access -- so
    # the access itself has to sit inside the guard, not just the serialization.
    # Its default only swallows AttributeError; anything else the property
    # raises would otherwise escape into _get_fallback_tools(), which has no
    # guard of its own and would lose every tool in the turn over one bad
    # schema instead of just that tool's parameters.
    try:
        tool_call_schema = getattr(lc_tool, "tool_call_schema", None)
        if tool_call_schema is not None:
            return _dump_json_schema(tool_call_schema)
        args_schema = getattr(lc_tool, "args_schema", None)
        if args_schema is not None:
            return _dump_json_schema(args_schema)
    except _SCHEMA_ERRORS:
        return _EMPTY_SCHEMA
    return _EMPTY_SCHEMA


def langchain_tool_parameters_json(lc_tool: Any) -> str:
    """
    Return the model-facing JSON schema for a LangChain tool.

    Prefers ``tool_call_schema``, which excludes ``InjectedToolArg`` /
    ``InjectedStore`` fields (``config``, ``store``, ``BaseStore``) that must
    not be shown to the model and cannot be serialized. ``args_schema`` is
    used only when ``tool_call_schema`` is absent, so a failed injected-arg
    schema never falls through to ``args_schema.schema()``.

    Memoized per tool object because ``tool_call_schema`` is an *uncached*
    property that rebuilds a Pydantic subset model on every read -- ~2.9x the
    cost of the ``args_schema.schema()`` call it replaces (16.4ms against
    5.7ms for the ten local tools).  ``_get_fallback_tools()`` runs once per
    always-included tool on ``_get_tool_by_name()``'s miss path, so that delta
    is multiplied by up to ``TOOL_INCLUSIONS_MAX_PER_TURN`` on the very path
    that already exists because rebuilding local schemas per lookup is
    wasteful.  The local tools are module-level singletons with static
    schemas, so one extraction per object is all that is ever needed.

    An empty ``{}`` is returned -- and logged -- when no schema can be
    extracted.  The log line matters: ``{}`` is indistinguishable downstream
    from a tool that genuinely takes no arguments (``_format_and_dedupe_tools``
    normalizes it to ``{"type": "object", "properties": {}}``), which is
    exactly how ``upsert_memory`` and ``confirm_sensitive_action`` were
    silently advertised to the model as argument-free on the fallback path.
    """
    key = id(lc_tool)
    cached = _schema_memo.get(key)
    if cached is not None and cached[0]() is lc_tool:
        return cached[1]

    params = _extract_tool_parameters_json(lc_tool)
    if params == _EMPTY_SCHEMA:
        LOGGER.warning(
            "No JSON schema could be extracted for tool %s; it will be offered "
            "to the model with no parameters",
            sanitize_tool_text(str(getattr(lc_tool, "name", "?")), limit=200),
        )

    # Not weak-referenceable: correctness does not depend on the memo, so a
    # tool that cannot be tracked simply pays the extraction cost every time.
    with contextlib.suppress(TypeError):
        _schema_memo[key] = (
            weakref.ref(lc_tool, partial(_drop_schema_memo, key)),
            params,
        )
    return params


def format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format Home Assistant LLM tools to be compatible with OpenAI format."""
    tool_spec = {
        "name": tool.name,
        "parameters": safe_convert(
            tool.parameters, custom_serializer=custom_serializer
        ),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def is_actuation_tool(name: str) -> bool:
    """Check if a tool name indicates an actuation tool."""
    # The prefixes below name HA *intents* ("HassTurn...", "HassLight..."), so
    # they must be matched against the base name -- HA's namespaced tool names
    # ("light__HassLightSet") start with the domain, not the intent.
    name_lower = base_tool_name(name).lower()
    # Check exact matches first (greedy check for specific tools)
    if name_lower in {t.lower() for t in ACTUATION_LANGCHAIN_TOOLS}:
        return True
    # Check prefix matches for provider tools
    return any(name_lower.startswith(p.lower()) for p in ACTUATION_TOOL_PREFIXES)
