# ruff: noqa: S101
"""
Regression tests for the CONF_LLM_HASS_API default.

The options flow *deletes* ``CONF_LLM_HASS_API`` when no API is selected
(``_cleanup_none_llm_api`` in config_flow.py), so an absent key is a normal,
reachable state that means "the recommended default", not "no APIs".

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

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

from custom_components.home_generative_agent.agent.graph import _get_allowed_api_ids
from custom_components.home_generative_agent.agent.helpers import (
    active_llm_api_ids,
    normalize_llm_api_value,
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

    Note this is NOT how the options flow records "no APIs" — config_flow.py's
    _cleanup_none_llm_api pops the key whenever it is falsy, so a user who
    deselects everything ends up with an ABSENT key and therefore the Assist
    default. An empty list only reaches here from a YAML import or a future
    flow that stores the choice instead of erasing it. The distinction matters:
    do not read this test as proof that a user can turn every API off.
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
