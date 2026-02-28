"""
Sentinel LLM triage service — Issue #262.

Provides ``SentinelTriageService``, which evaluates a single finding through
an optional LLM triage pass at autonomy level >= 1.

Design constraints (from sentinel_plan.md §6, §7)
--------------------------------------------------
* Input allowlist — the prompt contains **only** these fields:
  - Always included: ``type``, ``severity``, ``confidence``,
    ``is_sensitive``, ``entity_count``, ``suggested_actions_count``.
  - Optional sanitised evidence (when present in the finding):
    ``is_night``, ``anyone_home``, ``recognized_people_count``,
    ``last_changed_age_seconds``.
  - **Never** included: raw entity state values, attribute strings, area
    names, or free-form evidence text.
* Output schema: ``decision`` (``notify``|``suppress``), ``reason_code``,
  ``triage_confidence`` (float, audit-only), ``summary``.
* Triage cannot alter any finding field.
* ``triage_confidence`` is stored in the audit record but never gates
  execution — only the ``decision`` field gates ``notify`` vs ``suppress``.
* On timeout or any error the service fails-open: decision becomes
  ``notify`` with reason_code ``triage_error``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)

# Triage decision tokens.
TRIAGE_NOTIFY = "notify"
TRIAGE_SUPPRESS = "suppress"
TRIAGE_ERROR = "error"

# Reason codes for audit.
TRIAGE_REASON_LLM_NOTIFY = "llm_notify"
TRIAGE_REASON_LLM_SUPPRESS = "llm_suppress"
TRIAGE_REASON_TIMEOUT = "triage_timeout"
TRIAGE_REASON_ERROR = "triage_error"
TRIAGE_REASON_DISABLED = "triage_disabled"

# Input allowlist for optional sanitised evidence fields.
_ALLOWED_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "is_night",
        "anyone_home",
        "recognized_people_count",
        "last_changed_age_seconds",
    }
)

_SYSTEM_PROMPT = """\
You are a security alert triage assistant for a home automation system.
Your task is to evaluate a security finding and decide whether the homeowner
should be notified or whether the alert can be safely suppressed.

Respond with a JSON object only — no markdown, no preamble:
{
  "decision": "notify" or "suppress",
  "reason_code": "<short_snake_case_code>",
  "triage_confidence": <float 0.0-1.0>,
  "summary": "<one-sentence explanation>"
}

Rules:
- "notify" means the homeowner SHOULD receive this alert.
- "suppress" means the alert is routine or low-value and can be skipped.
- "triage_confidence" reflects your confidence in this specific decision
  (0=uncertain, 1=certain).
- Keep "summary" under 120 characters.
- If you are uncertain, default to "notify"."""

_USER_PROMPT_TEMPLATE = """\
Security alert to triage:
  type: {type}
  severity: {severity}
  confidence: {confidence:.2f}
  is_sensitive: {is_sensitive}
  entity_count: {entity_count}
  suggested_actions_count: {suggested_actions_count}{evidence_block}

Decision (JSON only):"""


@dataclass(frozen=True)
class TriageDecision:
    """Result of a triage pass."""

    decision: str  # "notify" | "suppress" | "error"
    reason_code: str
    triage_confidence: float | None  # audit-only; never gates execution
    summary: str


class SentinelTriageService:
    """
    LLM-based triage service for sentinel findings.

    Accepts any LangChain-compatible chat model (``model.ainvoke(messages)``).
    Pass ``None`` to disable triage (all findings pass through as notify).
    """

    def __init__(self, model: Any, *, timeout_seconds: int = 10) -> None:
        """Initialise with a LangChain-compatible model."""
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def triage(
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
    ) -> TriageDecision:
        """
        Run LLM triage on *finding*.

        Returns a ``TriageDecision``.  Fails-open on timeout or any
        exception: the caller treats an ``error`` decision as ``notify``.
        """
        if self._model is None:
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_DISABLED,
                triage_confidence=None,
                summary="Triage disabled.",
            )

        prompt = _build_prompt(finding, snapshot)
        start = time.monotonic()

        try:
            from langchain_core.messages import (  # noqa: PLC0415
                HumanMessage,
                SystemMessage,
            )

            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            result = await asyncio.wait_for(
                self._model.ainvoke(messages),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            LOGGER.warning(
                "Triage LLM call timed out after %d ms; failing open to notify.",
                elapsed_ms,
            )
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_TIMEOUT,
                triage_confidence=None,
                summary="Triage timed out; defaulting to notify.",
            )
        except (ValueError, TypeError, RuntimeError, OSError) as err:
            LOGGER.warning("Triage LLM call failed: %s; failing open to notify.", err)
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_ERROR,
                triage_confidence=None,
                summary=f"Triage error: {err!s}",
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return _parse_response(result, elapsed_ms)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(
    finding: AnomalyFinding,
    snapshot: FullStateSnapshot,
) -> str:
    """Build the triage prompt respecting the strict input allowlist."""
    # Always-allowed fields.
    entity_count = len(finding.triggering_entities)
    suggested_actions_count = len(finding.suggested_actions)

    # Optional sanitised evidence — only from the allowlist.
    evidence_items: list[str] = []
    raw_evidence = finding.evidence or {}
    derived = snapshot.get("derived", {})

    # is_night and anyone_home come from derived context (safer source).
    is_night = derived.get("is_night")
    anyone_home = derived.get("anyone_home")
    if is_night is not None:
        evidence_items.append(f"is_night: {is_night}")
    if anyone_home is not None:
        evidence_items.append(f"anyone_home: {anyone_home}")

    # recognized_people_count — count only, never names.
    people = raw_evidence.get("recognized_people")
    if isinstance(people, list):
        evidence_items.append(f"recognized_people_count: {len(people)}")

    # last_changed_age_seconds — derived elapsed time, not raw timestamp.
    if "last_changed_age_seconds" in raw_evidence:
        val = raw_evidence["last_changed_age_seconds"]
        if isinstance(val, (int, float)):
            evidence_items.append(f"last_changed_age_seconds: {val:.0f}")

    evidence_block = ""
    if evidence_items:
        evidence_block = "\n  evidence:\n    " + "\n    ".join(evidence_items)

    return _USER_PROMPT_TEMPLATE.format(
        type=finding.type,
        severity=finding.severity,
        confidence=finding.confidence,
        is_sensitive=finding.is_sensitive,
        entity_count=entity_count,
        suggested_actions_count=suggested_actions_count,
        evidence_block=evidence_block,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(result: Any, elapsed_ms: int) -> TriageDecision:  # noqa: ARG001
    """
    Parse LLM response into a ``TriageDecision``.

    Fails-open to ``notify`` on any parse error.
    """
    content = getattr(result, "content", None) or ""
    if not isinstance(content, str):
        content = str(content)

    # Strip markdown fences if the model wrapped the JSON.
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()

    try:
        parsed: dict[str, Any] = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        LOGGER.warning(
            "Triage LLM returned non-JSON response (%d chars); failing open.",
            len(content),
        )
        return TriageDecision(
            decision=TRIAGE_NOTIFY,
            reason_code=TRIAGE_REASON_ERROR,
            triage_confidence=None,
            summary="Could not parse triage response.",
        )

    decision_raw = str(parsed.get("decision", TRIAGE_NOTIFY)).lower().strip()
    decision = TRIAGE_SUPPRESS if decision_raw == TRIAGE_SUPPRESS else TRIAGE_NOTIFY

    reason_code = str(
        parsed.get("reason_code", TRIAGE_REASON_LLM_NOTIFY)
    ).strip() or (
        TRIAGE_REASON_LLM_SUPPRESS
        if decision == TRIAGE_SUPPRESS
        else TRIAGE_REASON_LLM_NOTIFY
    )

    raw_conf = parsed.get("triage_confidence")
    triage_confidence: float | None
    try:
        triage_confidence = float(raw_conf)  # type: ignore[arg-type]
        triage_confidence = max(0.0, min(1.0, triage_confidence))
    except (TypeError, ValueError):
        triage_confidence = None

    summary = str(parsed.get("summary", "")).strip()[:240]

    LOGGER.debug(
        "Triage result: decision=%s reason=%s confidence=%s",
        decision,
        reason_code,
        triage_confidence,
    )
    return TriageDecision(
        decision=decision,
        reason_code=reason_code,
        triage_confidence=triage_confidence,
        summary=summary,
    )
