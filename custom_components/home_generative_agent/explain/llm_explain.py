"""Optional LLM explanation layer for anomaly findings."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

LOGGER = logging.getLogger(__name__)
MAX_EXPLANATION_CHARS = 220


class LLMExplainer:
    """Generate non-authoritative explanation text for findings."""

    def __init__(self, model: Any) -> None:
        """Initialize the explainer with an optional LLM model."""
        self._model = model

    async def async_explain(self, finding: AnomalyFinding) -> str | None:
        """Return explanation text or None on failure."""
        if self._model is None:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            anomaly_type=finding.type,
            severity=finding.severity,
            evidence=finding.evidence,
            suggested_actions=finding.suggested_actions,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            result = await self._model.ainvoke(messages)
        except (ValueError, TypeError, RuntimeError) as err:
            LOGGER.warning("LLM explanation failed: %s", err)
            return None

        content = getattr(result, "content", None)
        if not content:
            return None
        text = _sanitize_text(str(content))
        if not text:
            return _compact_fallback(finding)
        if len(text) > MAX_EXPLANATION_CHARS:
            return _compact_fallback(finding)
        return text


def _sanitize_text(text: str) -> str:
    """Normalize model text for mobile notifications."""
    normalized = text.replace("**", "").replace("`", "")
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _friendly_type(anomaly_type: str) -> str:
    known = {
        "open_entry_while_away": "Open entry while away",
        "open_entry_at_night_when_home": "Open entry at night",
        "open_entry_at_night_when_home_window": "Open entry at night",
        "open_entry_at_night_while_away": "Open entry at night",
        "open_any_window_at_night_while_away": "Window open at night",
        "unlocked_lock_at_night": "Door lock left unlocked",
        "camera_entry_unsecured": "Activity near unsecured entry",
    }
    if anomaly_type in known:
        return known[anomaly_type]
    return anomaly_type.replace("_", " ").strip().capitalize()


def _friendly_entity(entity_id: str) -> str:
    if "." in entity_id:
        _, _, name = entity_id.partition(".")
    else:
        name = entity_id
    return name.replace("_", " ").strip().title()


def _compact_fallback(finding: AnomalyFinding) -> str:
    summary = _friendly_type(finding.type)
    entity = "an entry"
    if finding.triggering_entities:
        entity = _friendly_entity(finding.triggering_entities[0])
    message = f"{summary}: {entity}. {_severity_action_hint(finding.severity)}"
    return message[:MAX_EXPLANATION_CHARS].rstrip()


def _severity_action_hint(severity: str) -> str:
    if severity == "high":
        return "Urgent: check and secure it now."
    if severity == "medium":
        return "Check soon and secure it if unexpected."
    return "Review when convenient."
