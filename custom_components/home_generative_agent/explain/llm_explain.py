"""Optional LLM explanation layer for anomaly findings."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.core.utils import extract_final

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
            evidence=_relativize_timestamps(finding.evidence),
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
        text = extract_final(str(content)).replace("**", "").replace("`", "")
        if not text:
            return _compact_fallback(finding)
        if len(text) > MAX_EXPLANATION_CHARS:
            return _compact_fallback(finding)
        return text


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


_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def _iso_to_relative(value: str) -> str:
    """Convert an ISO-8601 timestamp to a relative duration string."""
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(tz=UTC) - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < _SECONDS_PER_MINUTE:
            return "just now"
        minutes = total_seconds // _SECONDS_PER_MINUTE
        if minutes < _MINUTES_PER_HOUR:
            return f"about {minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // _MINUTES_PER_HOUR
    except (ValueError, TypeError):
        return value
    else:
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"


# Regex matching full ISO-8601 timestamps (e.g. 2025-01-15T20:09:00+00:00)
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s,}]*")


def _relativize_timestamps(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of evidence with ISO timestamps as relative durations."""
    out: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, str) and _ISO_RE.fullmatch(value):
            out[key] = _iso_to_relative(value)
        else:
            out[key] = value
    return out
