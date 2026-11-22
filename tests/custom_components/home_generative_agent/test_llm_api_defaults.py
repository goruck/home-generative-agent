# ruff: noqa: S101
"""
Regression tests for the CONF_LLM_HASS_API default.

Before issue #949 the options flow *deleted* ``CONF_LLM_HASS_API`` when no API
was selected, so an absent key was a normal, reachable state that means "the
recommended default", not "no APIs". The v6.2 migration now writes that default
out and the form refuses an empty selection, but an absent key keeps the same
meaning for every reader.

Before the fix, three readers disagreed about that: two defaulted to
``[LLM_API_ASSIST]`` while ``HGAConversationEntity.__init__`` treated the
absent key as falsy.  That single disagreement dropped
``ConversationEntityFeature.CONTROL`` from the entity's state attributes, so
Home Assistant never installed its local-intent allowlist
(``_async_local_fallback_intent_filter`` in assist_pipeline/pipeline.py) and
handled ``HassTurnOn``/``HassTurnOff`` on a lock in its own default agent.  The
agent never saw the turn, so the critical-action PIN never ran.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from homeassistant.helpers.selector import SelectOptionDict
from pytest_homeassistant.common import MockConfigEntry

from custom_components.home_generative_agent.agent.graph import _get_allowed_api_ids
from custom_components.home_generative_agent.agent.helpers import (
    active_llm_api_ids,
    normalize_llm_api_value,
)
from custom_components.home_generative_agent.config_flow import (
    HomeGenerativeAgentOptionsFlow,
)
from custom_components.home_generative_agent.const import (
    CONF_CRITICAL_ACTION_PIN,
    CONF_CRITICAL_ACTION_PIN_ENABLED,
    CONF_CRITICAL_ACTION_PIN_HASH,
    CONF_CRITICAL_ACTION_PIN_SALT,
    CONF_FACE_API_URL,
    CONF_TOOL_EXCLUSIONS,
    CONF_TOOL_INCLUSIONS,
    DOMAIN,
)
from custom_components.home_generative_agent.core.utils import (
    CannotConnectError,
    hash_pin,
    verify_pin,
)


def test_absent_key_defaults_to_assist() -> None:
    """An absent key means the recommended default, not 'no APIs'."""
    assert active_llm_api_ids({}) == [llm.LLM_API_ASSIST]


def test_absent_key_is_truthy() -> None:
    """
    The absent-key result must be truthy.

    conversation.py gates ConversationEntityFeature.CONTROL on this value. A
    falsy result there is the exact bug: HA then routes lock/unlock through its
    own intent handler and the critical-action PIN is bypassed.
    """
    assert active_llm_api_ids({})


def test_stored_empty_list_stays_empty() -> None:
    """
    A stored empty list is preserved rather than re-defaulted.

    The options flow never stores this: it refuses an empty selection, and the
    v6.2 migration rewrites a stored [] to the Assist default (issue #949),
    because no API at all fails every conversation. Do not read this test as
    proof that a user can turn every API off.
    """
    assert active_llm_api_ids({CONF_LLM_HASS_API: []}) == []
    assert not active_llm_api_ids({CONF_LLM_HASS_API: []})


def test_explicit_ids_are_returned() -> None:
    """Configured ids pass through unchanged."""
    assert active_llm_api_ids({CONF_LLM_HASS_API: ["assist", "other"]}) == [
        "assist",
        "other",
    ]


def test_legacy_string_value_is_wrapped() -> None:
    """Pre-v6 configs stored a bare string; it must normalize to a list."""
    assert active_llm_api_ids({CONF_LLM_HASS_API: "assist"}) == ["assist"]


def test_degenerate_stored_shapes_normalize_instead_of_crashing() -> None:
    """
    None, "" and non-string list elements normalize to clean lists.

    Before the shared normalizer, an explicit None (reachable via programmatic
    options updates that bypass the form schema) crashed ``list(None)`` at
    runtime — blocking conversation-entity setup — and "" wrapped to [""],
    a truthy id that is guaranteed to fail loading. The options form and the
    runtime readers must agree on these shapes (issue #568 review follow-up).
    """
    assert active_llm_api_ids({CONF_LLM_HASS_API: None}) == []
    assert active_llm_api_ids({CONF_LLM_HASS_API: ""}) == []
    assert active_llm_api_ids({CONF_LLM_HASS_API: ["assist", {"id": "x"}]}) == [
        "assist"
    ]
    assert normalize_llm_api_value(["", "assist", 3]) == ["assist"]


def test_returned_list_is_a_copy() -> None:
    """Mutating the result must not write back into stored options."""
    options = {CONF_LLM_HASS_API: ["assist"]}
    active_llm_api_ids(options).append("injected")
    assert options[CONF_LLM_HASS_API] == ["assist"]


def test_graph_allowed_api_ids_matches_the_same_default() -> None:
    """
    The graph's tool gate applies the identical absent-key default.

    If these two readers drift apart again, the agent binds Assist tools while
    telling HA it has no control capability -- which is the shipped bug.
    """
    allowed = _get_allowed_api_ids({"configurable": {"options": {}}})
    assert llm.LLM_API_ASSIST in allowed
    assert "hga_local" in allowed


def test_graph_allowed_api_ids_respects_explicit_empty() -> None:
    """An explicit 'no APIs' choice still leaves only the local tools."""
    allowed = _get_allowed_api_ids(
        {"configurable": {"options": {CONF_LLM_HASS_API: []}}}
    )
    assert allowed == {"hga_local"}


def _options_flow(hass: Any, options: dict[str, Any]) -> Any:
    """Build an options flow bound to a stored config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Home Generative Agent", options=options
    )
    entry.add_to_hass(hass)
    flow = HomeGenerativeAgentOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id
    return flow


@pytest.mark.asyncio
async def test_options_flow_refuses_an_empty_selection(hass: Any) -> None:
    """
    Deselecting every API is refused, not saved as the Assist default.

    Before issue #949 the flow deleted the key, so the save quietly kept
    running Assist; storing [] instead would fail every conversation.
    """
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    result = cast("dict[str, Any]", await flow.async_step_init({CONF_LLM_HASS_API: []}))

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "llm_api_required"}


def _admin_api_registry() -> Any:
    """Register Assist plus HA 2026.11's admin-only "homeassistant" API."""
    return patch(
        "custom_components.home_generative_agent.config_flow.llm.async_get_apis",
        return_value=[
            SimpleNamespace(id="assist", name="Assist"),
            SimpleNamespace(
                id="homeassistant", name="Home Assistant", requires_admin=True
            ),
        ],
    )


@pytest.mark.asyncio
async def test_options_flow_refuses_a_selection_of_only_admin_apis(
    hass: Any,
) -> None:
    """
    A selection holding only admin-only APIs is refused.

    It is not empty, so the empty-selection check passed it, and a voice
    satellite's conversation, or one by a user who is not an admin, then
    failed with "No LLM APIs could be loaded" (#998, #918).
    """
    flow = _options_flow(
        hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST, "homeassistant"]}
    )

    with _admin_api_registry():
        result = cast(
            "dict[str, Any]",
            await flow.async_step_init({CONF_LLM_HASS_API: ["homeassistant"]}),
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {"base": "llm_api_admin_only"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        pytest.param([llm.LLM_API_ASSIST, "homeassistant"], id="with-one-for-everyone"),
        # Unregistered is often a transient MCP outage, kept on purpose (#568).
        pytest.param(["homeassistant", "mcp-gone"], id="with-an-unregistered"),
    ],
)
async def test_options_flow_saves_an_admin_only_api_beside_another(
    hass: Any, selection: list[str]
) -> None:
    """An admin-only API is refused only when nothing else is selected."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    with _admin_api_registry():
        result = cast(
            "dict[str, Any]",
            await flow.async_step_init({CONF_LLM_HASS_API: selection}),
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_LLM_HASS_API] == selection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refused",
    [pytest.param([], id="empty"), pytest.param(["homeassistant"], id="unsupported")],
)
async def test_a_refused_selection_keeps_the_stored_apis_tools_listed(
    hass: Any, refused: list[str]
) -> None:
    """
    The re-render after a refused selection lists the stored APIs' tools.

    Built from the refused selection, the tool pickers had no choices, so
    re-selecting Assist in the same form showed no tools to pick (#998).
    """
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})
    tools = AsyncMock(
        return_value=(
            [SelectOptionDict(label="Assist: HassTurnOn", value="assist|HassTurnOn")],
            {"assist|HassTurnOn"},
            {"assist": "Assist"},
            set(),
        )
    )

    with (
        _admin_api_registry(),
        patch(
            "custom_components.home_generative_agent.config_flow._live_tool_choices",
            tools,
        ),
    ):
        result = cast(
            "dict[str, Any]",
            await flow.async_step_init({CONF_LLM_HASS_API: refused}),
        )

    assert result["type"] == FlowResultType.FORM
    assert tools.await_args is not None
    assert tools.await_args.args[1] == [llm.LLM_API_ASSIST]
    keys = {str(key) for key in result["data_schema"].schema}
    assert CONF_TOOL_EXCLUSIONS in keys
    assert CONF_TOOL_INCLUSIONS in keys


def _data_held_flow(hass: Any) -> Any:
    """Build an options flow whose entry keeps its selection in data only."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Generative Agent",
        data={CONF_LLM_HASS_API: ["mcp-abc"], "db_bootstrapped": True},
        options={},
    )
    entry.add_to_hass(hass)
    flow = HomeGenerativeAgentOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id
    return flow


def test_options_form_shows_a_selection_held_in_data(hass: Any) -> None:
    """
    The form starts from the selection that runs, even one held in data.

    Runtime lays options over data; a form that read options alone showed
    the Assist default, and its next save wrote Assist over the selection
    (#998).
    """
    flow = _data_held_flow(hass)

    options = flow._base_options()

    assert options[CONF_LLM_HASS_API] == ["mcp-abc"]
    assert "db_bootstrapped" not in options


@pytest.mark.asyncio
async def test_a_refused_selection_lists_a_data_held_selections_tools(
    hass: Any,
) -> None:
    """The refused-save pickers fall back to a selection held in data too."""
    flow = _data_held_flow(hass)
    tools = AsyncMock(return_value=([], set(), {}, set()))

    with patch(
        "custom_components.home_generative_agent.config_flow._live_tool_choices",
        tools,
    ):
        await flow.async_step_init({CONF_LLM_HASS_API: []})

    assert tools.await_args is not None
    assert tools.await_args.args[1] == ["mcp-abc"]


@pytest.mark.asyncio
async def test_options_flow_stores_the_selection(hass: Any) -> None:
    """A non-empty selection is stored as submitted."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init({CONF_LLM_HASS_API: ["mcp-abc"]}),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_LLM_HASS_API] == ["mcp-abc"]


@pytest.mark.asyncio
async def test_pin_set_with_a_refused_save_survives_the_retry(hass: Any) -> None:
    """
    A new PIN typed into a refused save is kept for the corrected save.

    The PIN field re-renders blank, so the user saves again without retyping
    it. Before the fix the PIN was hashed, the save refused, and the hash
    dropped: the corrected save kept the old PIN with no error.
    """
    old_hash, old_salt = hash_pin("1111")
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: old_hash,
            CONF_CRITICAL_ACTION_PIN_SALT: old_salt,
        },
    )

    refused = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN: "4321",
            }
        ),
    )
    assert refused["errors"] == {"base": "llm_api_required"}

    saved = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    assert saved["type"] == FlowResultType.CREATE_ENTRY
    data = saved["data"]
    assert verify_pin(
        "4321",
        hashed=data[CONF_CRITICAL_ACTION_PIN_HASH],
        salt=data[CONF_CRITICAL_ACTION_PIN_SALT],
    )
    assert CONF_CRITICAL_ACTION_PIN not in data


@pytest.mark.asyncio
async def test_pin_set_with_an_unreachable_face_url_survives_the_retry(
    hass: Any,
) -> None:
    """A face-service error refuses the save too; the PIN still carries over."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    with patch(
        "custom_components.home_generative_agent.config_flow.validate_face_api_url",
        side_effect=CannotConnectError,
    ):
        refused = cast(
            "dict[str, Any]",
            await flow.async_step_init(
                {
                    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                    CONF_FACE_API_URL: "http://face.invalid",
                    CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                    CONF_CRITICAL_ACTION_PIN: "2468",
                }
            ),
        )
    assert refused["errors"] == {"base": "cannot_connect"}

    saved = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    data = saved["data"]
    assert verify_pin(
        "2468",
        hashed=data[CONF_CRITICAL_ACTION_PIN_HASH],
        salt=data[CONF_CRITICAL_ACTION_PIN_SALT],
    )


@pytest.mark.asyncio
async def test_a_carried_pin_is_dropped_when_the_pin_is_turned_off(
    hass: Any,
) -> None:
    """Turning the PIN off on the retry clears the carried PIN as well."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})
    await flow.async_step_init(
        {
            CONF_LLM_HASS_API: [],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "4321",
        }
    )

    saved = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: False,
            }
        ),
    )

    assert CONF_CRITICAL_ACTION_PIN_HASH not in saved["data"]
    assert CONF_CRITICAL_ACTION_PIN_SALT not in saved["data"]


@pytest.mark.asyncio
async def test_a_pin_cleared_with_a_refused_save_stays_cleared(hass: Any) -> None:
    """Clearing the PIN in a refused save is not undone by the corrected save."""
    old_hash, old_salt = hash_pin("1111")
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: old_hash,
            CONF_CRITICAL_ACTION_PIN_SALT: old_salt,
        },
    )
    refused = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN: "",
            }
        ),
    )
    assert refused["errors"] == {"base": "pin_required"}

    retry = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    # The clear carried over, so the old PIN does not come back; with the
    # PIN still on and none set, the save asks for one.
    assert retry["errors"] == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_an_invalid_retype_drops_the_carried_pin(hass: Any) -> None:
    """
    An invalid retype drops the carried PIN.

    After an accepted PIN, the invalid retype is the latest intent, so a blank
    save keeps the stored PIN.
    """
    old_hash, old_salt = hash_pin("1111")
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: old_hash,
            CONF_CRITICAL_ACTION_PIN_SALT: old_salt,
        },
    )
    await flow.async_step_init(
        {
            CONF_LLM_HASS_API: [],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "4321",
        }
    )
    invalid = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN: "12",
            }
        ),
    )
    assert invalid["errors"] == {"base": "invalid_pin"}

    saved = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    data = saved["data"]
    assert verify_pin(
        "1111",
        hashed=data[CONF_CRITICAL_ACTION_PIN_HASH],
        salt=data[CONF_CRITICAL_ACTION_PIN_SALT],
    )


@pytest.mark.asyncio
async def test_pin_set_with_a_tool_overlap_survives_the_retry(hass: Any) -> None:
    """The tool-overlap refusal carries the accepted PIN like the others."""
    flow = _options_flow(hass, {CONF_LLM_HASS_API: ["mcp-abc"]})
    refused = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_TOOL_EXCLUSIONS: ["mcp-abc::web_search"],
                CONF_TOOL_INCLUSIONS: ["mcp-abc::web_search"],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
                CONF_CRITICAL_ACTION_PIN: "2468",
            }
        ),
    )
    assert refused["errors"] == {"base": "tool_excluded_and_included"}

    saved = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: ["mcp-abc"],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    data = saved["data"]
    assert verify_pin(
        "2468",
        hashed=data[CONF_CRITICAL_ACTION_PIN_HASH],
        salt=data[CONF_CRITICAL_ACTION_PIN_SALT],
    )


@pytest.mark.asyncio
async def test_the_pin_cannot_be_on_with_no_pin_set(hass: Any) -> None:
    """
    Turning the PIN on without setting one is refused.

    With no hash stored, the agent lets critical actions through without
    asking (agent/graph.py), so the switch alone would look on and do nothing.
    """
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    assert result["errors"] == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_an_invalid_retype_with_no_stored_pin_is_not_saved_off(
    hass: Any,
) -> None:
    """
    A carried PIN dropped by an invalid retype cannot leave the gate open.

    Accepted PIN, refused save; invalid retype; blank save. With no PIN
    stored, that last save used to store the switch on and no PIN.
    """
    flow = _options_flow(hass, {CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]})
    await flow.async_step_init(
        {
            CONF_LLM_HASS_API: [],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "4321",
        }
    )
    await flow.async_step_init(
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "12",
        }
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    assert result["errors"] == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_a_refused_disable_does_not_bring_a_cleared_pin_back(
    hass: Any,
) -> None:
    """
    Clear (refused), turn off (refused), turn back on: the old PIN stays gone.

    Turning the PIN off removes it, so turning it back on in the same form
    needs a new PIN rather than reviving the stored one.
    """
    old_hash, old_salt = hash_pin("1111")
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: old_hash,
            CONF_CRITICAL_ACTION_PIN_SALT: old_salt,
        },
    )
    await flow.async_step_init(
        {
            CONF_LLM_HASS_API: [],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN: "",
        }
    )
    await flow.async_step_init(
        {CONF_LLM_HASS_API: [], CONF_CRITICAL_ACTION_PIN_ENABLED: False}
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    assert result["errors"] == {"base": "pin_required"}


@pytest.mark.asyncio
async def test_a_stored_pin_still_saves_without_retyping(hass: Any) -> None:
    """An unrelated save with the PIN on and already set is not refused."""
    old_hash, old_salt = hash_pin("1111")
    flow = _options_flow(
        hass,
        {
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
            CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            CONF_CRITICAL_ACTION_PIN_HASH: old_hash,
            CONF_CRITICAL_ACTION_PIN_SALT: old_salt,
        },
    )

    result = cast(
        "dict[str, Any]",
        await flow.async_step_init(
            {
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_CRITICAL_ACTION_PIN_ENABLED: True,
            }
        ),
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_CRITICAL_ACTION_PIN_HASH] == old_hash
