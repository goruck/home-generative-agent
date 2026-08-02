"""Optional LLM explanation layer for anomaly findings."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.core.utils import (
    SENTINEL_ADMISSION_TIMEOUT_S,
    SentinelLLMDeferredError,
    extract_final,
    run_sentinel_model_call,
)

from .prompts import LANGUAGE_INSTRUCTION_TEMPLATE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding

LOGGER = logging.getLogger(__name__)
MAX_EXPLANATION_CHARS = 220
_EXPLAIN_LLM_TIMEOUT_S: Final[float] = 30.0


class LLMExplainer:
    """Generate non-authoritative explanation text for findings."""

    def __init__(
        self,
        model: Any,
        *,
        deployment: str = "edge",
        health_stats: dict[str, Any] | None = None,
        response_language: str = "",
    ) -> None:
        """Initialize the explainer with an optional LLM model."""
        self._model = model
        self._deployment = deployment
        self._health_stats = health_stats
        self._response_language = response_language

    async def async_explain(self, finding: AnomalyFinding) -> str | None:  # noqa: PLR0911
        """Return explanation text or None on failure."""
        if self._model is None:
            return None

        evidence = finding.evidence
        if self._response_language and finding.is_sensitive:
            # Deterministic privacy boundary for translated explanations:
            # notifier._redact_if_sensitive matches recognized_people names
            # with an exact (case-insensitive) string match, but inflected
            # languages decline names ("Petra" -> "Petru"), so a translated
            # explanation could carry a form that redaction never matches.
            # Redacting the evidence structure BEFORE rendering (rather than
            # the rendered prompt string) matters twice over: repr-escaping
            # would hide names containing quotes from a post-render match,
            # and a post-render match on a short name like "Max" would
            # corrupt the template's own instruction text ("Max 2 short
            # sentences."). A name the model never sees cannot be emitted in
            # any inflection; the prompt's nominative instruction remains
            # only as defense in depth.
            evidence = _redact_person_names(evidence)
        prompt = USER_PROMPT_TEMPLATE.format(
            anomaly_type=_display_type(finding),
            severity=finding.severity,
            evidence=_relativize_timestamps(evidence),
            suggested_actions=finding.suggested_actions,
        )
        system_prompt = SYSTEM_PROMPT
        if self._response_language:
            system_prompt += LANGUAGE_INSTRUCTION_TEMPLATE.format(
                language=self._response_language
            )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            result = await run_sentinel_model_call(
                self._model,
                messages,
                deployment=self._deployment,
                category="explain",
                admission_timeout_s=SENTINEL_ADMISSION_TIMEOUT_S,
                call_timeout_s=_EXPLAIN_LLM_TIMEOUT_S,
                health_stats=self._health_stats,
            )
        except SentinelLLMDeferredError as err:
            LOGGER.debug("LLM explanation deferred: %s", err)
            return None
        except TimeoutError:
            LOGGER.warning(
                "LLM explanation timed out after %.0fs; skipping.",
                _EXPLAIN_LLM_TIMEOUT_S,
            )
            return None
        except Exception as err:  # noqa: BLE001
            # Provider errors (e.g. ollama.ResponseError, httpx transport
            # failures) do not subclass the families above; explanations are
            # best-effort, so any failure degrades to None (issue #465).
            LOGGER.warning("LLM explanation failed: %s", err)
            return None

        content = getattr(result, "content", None)
        if not content:
            return None
        text = extract_final(content).replace("**", "").replace("`", "")
        if not text:
            return _compact_fallback(finding)
        if len(text) > MAX_EXPLANATION_CHARS:
            return _compact_fallback(finding)
        return text


_KNOWN_TYPE_LABELS = {
    "open_entry_while_away": "Open entry while away",
    "open_entry_at_night_when_home": "Open entry at night",
    "open_entry_at_night_when_home_window": "Open entry at night",
    "open_entry_at_night_while_away": "Open entry at night",
    "open_entry_at_night": "Open entry at night",
    "open_entry_at_night_window": "Open entry at night",
    "open_entry_at_night_door": "Open entry at night",
    "open_entry_at_night_entry": "Open entry at night",
    "open_any_window_at_night_while_away": "Window open at night",
    "motion_detected_at_night_while_away": "Motion at night while away",
    "motion_detected_while_away": "Motion while away",
    "unlocked_lock_at_night": "Door lock left unlocked",
    "camera_entry_unsecured": "Activity near unsecured entry",
    "alarm_disarmed_during_external_threat": "Outdoor activity while alarm disarmed",
}


def _display_type(finding: AnomalyFinding) -> str:
    """
    Friendly label for a finding.

    Dynamic rules carry slugified candidate IDs as ``finding.type``; when the
    type has no curated label but the rule's template does, prefer the
    template label so explanations never show raw candidate slugs
    (issue #516 review).
    """
    if finding.type in _KNOWN_TYPE_LABELS:
        return _KNOWN_TYPE_LABELS[finding.type]
    template_id = str(finding.evidence.get("template_id") or "")
    if template_id in _KNOWN_TYPE_LABELS:
        return _KNOWN_TYPE_LABELS[template_id]
    return _friendly_type(finding.type)


def _friendly_type(anomaly_type: str) -> str:
    if anomaly_type in _KNOWN_TYPE_LABELS:
        return _KNOWN_TYPE_LABELS[anomaly_type]
    display = anomaly_type.removeprefix("candidate_")
    # Strip "rule_<digits>_" prefix (e.g. "rule_02_high_energy_consumption_away")
    parts = display.split("_")
    if len(parts) >= 3 and parts[0] == "rule" and parts[1].isdigit():  # noqa: PLR2004
        display = "_".join(parts[2:])
    return display.replace("_", " ").strip().capitalize()


def _friendly_entity(entity_id: str) -> str:
    if "." in entity_id:
        _, _, name = entity_id.partition(".")
    else:
        name = entity_id
    return name.replace("_", " ").strip().title()


def _compact_fallback(finding: AnomalyFinding) -> str:
    summary = _display_type(finding)
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


_REDACTED_NAME = "a recognised person"


def _redact_person_names(evidence: dict[str, Any]) -> dict[str, Any]:
    """
    Return a deep copy of evidence with recognized person names replaced.

    Walks every string in the structure -- values, dict keys, and members of
    lists, tuples, sets, and frozensets -- so a name is caught no matter
    where the evidence carried it (e.g. embedded in a camera caption).
    Redacting before rendering matters: Python's repr escapes quotes, so a
    name like ``D'Angelo`` could fail to match its rendered form after the
    prompt is built. Uses the same generic phrase and case-insensitive
    matching as notifier._redact_if_sensitive. Names are matched
    longest-first so an overlapping pair like "Alex"/"Alexander" cannot
    leave a partial name behind. Builds new containers throughout -- the
    original evidence dict is never mutated (its key order feeds anomaly-id
    derivation elsewhere).
    """
    names = [
        person
        for person in evidence.get("recognized_people", [])
        if isinstance(person, str) and person
    ]
    if not names:
        return evidence
    pattern = re.compile(
        "|".join(re.escape(name) for name in sorted(set(names), key=len, reverse=True)),
        flags=re.IGNORECASE,
    )

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(_REDACTED_NAME, value)
        if isinstance(value, dict):
            return {_walk(key): _walk(item) for key, item in value.items()}
        if isinstance(value, list | tuple | set | frozenset):
            return type(value)(_walk(item) for item in value)
        return value

    return _walk(evidence)


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
