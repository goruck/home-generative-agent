"""
Detect voice pipelines whose settings silently disarm the critical-action PIN.

The PIN can only gate commands the conversation agent actually receives.  When
a pipeline has ``prefer_local_intents`` enabled ("Prefer handling commands
locally"), Home Assistant matches the sentence against its own built-in intents
first and, on a match, executes it *without* ever calling the agent -- see
``assist_pipeline/pipeline.py`` calling ``default_agent.async_handle_intents``.
"Unlock the front door" matches ``HassTurnOff`` on a lock, so it runs locally
and the PIN never fires.

The ``ConversationEntityFeature.CONTROL`` flag does not help.  Its only effect
is to install ``_async_local_fallback_intent_filter``, and that filter is a
*reject* list -- ``async_handle_intents`` returns None when the filter matches.
It matches only ``HassGetState`` and media search, so those two go to the agent
and every other intent, control commands included, stays local.

No integration can intercept that path, so the honest response is to tell the
user their PIN is not doing what they think.  This module raises a repair issue
naming the offending pipelines and clears it once the conflict is resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..const import DOMAIN  # noqa: TID252

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

ISSUE_PIN_BYPASSED = "pin_bypassed_by_local_intents"


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_PIN_BYPASSED}_{entry_id}"


def find_conflicting_pipelines(hass: HomeAssistant, entity_id: str) -> list[str]:
    """
    Return the names of pipelines that route to *entity_id* with local intents on.

    Returns names rather than Pipeline objects because the only consumer is the
    repair issue's placeholder text, and names are what the user recognizes in
    the voice-assistant UI.
    """
    # Imported lazily: assist_pipeline is an after_dependency, so it may be
    # absent entirely on installs that never set up a voice pipeline. A missing
    # optional component must degrade to "nothing to warn about", never to a
    # setup failure.
    try:
        from homeassistant.components.assist_pipeline import (  # noqa: PLC0415
            async_get_pipelines,
        )
    except ImportError:
        return []

    try:
        pipelines = async_get_pipelines(hass)
    except (KeyError, AttributeError):
        # assist_pipeline is installed but not set up (its storage never
        # loaded). Same disposition as absent.
        return []

    return [
        pipeline.name
        for pipeline in pipelines
        if pipeline.prefer_local_intents and pipeline.conversation_engine == entity_id
    ]


@callback
def async_check_pin_pipeline_conflict(
    hass: HomeAssistant,
    entry_id: str,
    entity_id: str,
    *,
    pin_enabled: bool,
) -> None:
    """
    Raise or clear the repair issue for PIN-disarming pipeline settings.

    Idempotent: safe to call on every setup and reload. Deleting an issue that
    does not exist is a no-op in the issue registry, so the clear path needs no
    existence check.
    """
    issue_id = _issue_id(entry_id)

    if not pin_enabled:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    conflicting = find_conflicting_pipelines(hass, entity_id)
    if not conflicting:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_PIN_BYPASSED,
        translation_placeholders={
            "pipelines": ", ".join(sorted(conflicting)),
        },
    )
