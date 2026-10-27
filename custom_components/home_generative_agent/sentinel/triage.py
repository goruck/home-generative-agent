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
    ``is_night``, ``anyone_home``, ``recognized_people_count`` (enrolled
    identities only), ``unknown_person_present``,
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

Load on a shared local model (#866)
-----------------------------------
Triage is one model call per finding, in series. A run with many findings
used to send them back to back, and on a single-GPU host that shares the
chat model a voice turn arriving mid-burst cancelled the call in flight,
which on Ollama can wedge the runner and stall the voice turn itself. So,
per run (``begin_run``):

* at most ``max_per_run`` findings are triaged; the rest pass through with
  reason_code ``triage_cap``;
* consecutive calls are ``call_gap_s`` apart, so a chat turn arriving in
  the gap starts without cancelling anything;
* once a call is deferred or interrupted by chat, no further call is made
  that run (``triage_deferred``): the admission gate would only defer each
  one in turn, and a call started between two voice turns is cancelled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_TRIAGE_MAX_PER_RUN,
    SENTINEL_TRIAGE_CALL_GAP_S,
)
from custom_components.home_generative_agent.core.utils import (
    SENTINEL_ADMISSION_TIMEOUT_S,
    SentinelLLMDeferredError,
    extract_final,
    run_sentinel_model_call,
)
from custom_components.home_generative_agent.sentinel.logging_utils import (
    RepeatingLogLimiter,
)
from custom_components.home_generative_agent.sentinel.models import (
    enrolled_people,
    has_unknown_person,
)
from custom_components.home_generative_agent.sentinel.redaction import (
    redact_network_identifiers,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import AnomalyFinding
    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

LOGGER = logging.getLogger(__name__)
_PARSE_LOG_LIMITER = RepeatingLogLimiter(LOGGER)

_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)

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
TRIAGE_REASON_DEFERRED = "triage_deferred"
TRIAGE_REASON_CAP = "triage_cap"

# Reason codes of a finding the model never saw. The health sensor leaves
# these out of ``triage_suppress_rate``.
TRIAGE_REASONS_NOT_RUN: frozenset[str] = frozenset(
    {TRIAGE_REASON_DISABLED, TRIAGE_REASON_DEFERRED, TRIAGE_REASON_CAP}
)

# Input allowlist for optional sanitised evidence fields.
_ALLOWED_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "is_night",
        "anyone_home",
        "recognized_people_count",
        "unknown_person_present",
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
- HARD LIMIT: "summary" must be under 120 characters. Stop writing when you reach it.
- If you are uncertain, default to "notify".
- Return only the JSON object. Do not explain your reasoning."""

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

    def __init__(  # noqa: PLR0913
        self,
        model: Any,
        *,
        timeout_seconds: int = 10,
        deployment: str = "edge",
        health_stats: dict[str, Any] | None = None,
        max_per_run: int = RECOMMENDED_SENTINEL_TRIAGE_MAX_PER_RUN,
        call_gap_s: float = SENTINEL_TRIAGE_CALL_GAP_S,
    ) -> None:
        """Initialise with a LangChain-compatible model."""
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._deployment = deployment
        self._health_stats = health_stats
        self._log_limiter = RepeatingLogLimiter(LOGGER)
        self._max_per_run = max(1, int(max_per_run))
        self._call_gap_s = max(0.0, float(call_gap_s))
        # Per-run state (#866); see begin_run.
        self._run_calls = 0
        self._run_skipped = 0
        self._run_deferred = False
        self._last_call_end = 0.0

    @property
    def max_per_run(self) -> int:
        """Most findings one run sends to the model."""
        return self._max_per_run

    def begin_run(self) -> None:
        """
        Start a Sentinel run: reset the per-run call budget (#866).

        The engine calls this once per run, before dispatching findings.
        """
        self._run_calls = 0
        self._run_skipped = 0
        self._run_deferred = False
        self._last_call_end = 0.0
        if self._health_stats is not None:
            self._health_stats["triage_calls_last_run"] = 0
            self._health_stats["triage_passed_through_last_run"] = 0

    def run_stats(self) -> dict[str, int]:
        """Return how many findings this run triaged and passed through."""
        return {"calls": self._run_calls, "skipped": self._run_skipped}

    def _pass_through(self, reason_code: str, summary: str) -> TriageDecision:
        """Return the decision of a finding this run did not send to the model."""
        self._run_skipped += 1
        if self._health_stats is not None:
            self._health_stats["triage_passed_through_last_run"] = self._run_skipped
        return TriageDecision(
            decision=TRIAGE_NOTIFY,
            reason_code=reason_code,
            triage_confidence=None,
            summary=summary,
        )

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
        if self._run_deferred:
            # Chat took the model earlier this run; do not start a call its
            # next turn would cancel (#866).
            return self._pass_through(
                TRIAGE_REASON_DEFERRED, "Triage deferred; chat was active this run."
            )
        if self._run_calls >= self._max_per_run:
            if self._run_skipped == 0:
                LOGGER.info(
                    "Sentinel triage reached its limit of %d call(s) this run; "
                    "the run's remaining findings pass through untriaged.",
                    self._max_per_run,
                )
            return self._pass_through(
                TRIAGE_REASON_CAP, "Triage limit reached this run; passing through."
            )
        if self._run_calls > 0 and self._call_gap_s > 0.0:
            # Space the calls so a chat turn arriving between two of them
            # starts without cancelling anything (#866).
            gap_left = self._call_gap_s - (time.monotonic() - self._last_call_end)
            if gap_left > 0.0:
                await asyncio.sleep(gap_left)
        self._run_calls += 1
        if self._health_stats is not None:
            self._health_stats["triage_calls_last_run"] = self._run_calls
        try:
            return await self._triage_call(finding, snapshot)
        finally:
            self._last_call_end = time.monotonic()

    async def _triage_call(
        self,
        finding: AnomalyFinding,
        snapshot: FullStateSnapshot,
    ) -> TriageDecision:
        """Send one finding to the model; fail open on anything."""
        prompt = _build_prompt(finding, snapshot)
        start = time.monotonic()

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            result = await run_sentinel_model_call(
                self._model,
                messages,
                deployment=self._deployment,
                category="triage",
                admission_timeout_s=SENTINEL_ADMISSION_TIMEOUT_S,
                call_timeout_s=float(self._timeout_seconds),
                health_stats=self._health_stats,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._log_limiter.recovered(
                "llm_call",
                "Triage LLM call recovered after %d failed attempt(s).",
            )
            # Parsing runs inside the try on purpose. It used to sit after the
            # handlers, so anything it raised bypassed the fail-open contract
            # and propagated through _dispatch_item, _run_once and _timed_run
            # into _run_loop -- none of which catch -- ending the engine task.
            # SentinelEngine.start() then refuses to replace a non-None task,
            # so Sentinel stayed dead until the config entry was reloaded.
            return _parse_response(result, elapsed_ms)
        except SentinelLLMDeferredError as err:
            LOGGER.debug("Triage deferred: %s", err)
            # The rest of this run passes through (#866).
            self._run_deferred = True
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_DEFERRED,
                triage_confidence=None,
                summary="Triage deferred; chat was active.",
            )
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._log_limiter.warning(
                "llm_call",
                "Triage LLM call timed out after %d ms; failing open to notify.",
                elapsed_ms,
            )
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_TIMEOUT,
                triage_confidence=None,
                summary="Triage timed out; defaulting to notify.",
            )
        except Exception as err:  # noqa: BLE001
            # Fail-open must hold for *any* exception: provider errors such
            # as ollama.ResponseError (un-pulled model) or httpx transport
            # failures do not subclass the families above, and an escaped
            # error here would abort finding processing (issue #465).
            self._log_limiter.warning(
                "llm_call",
                "Triage LLM call failed: %s; failing open to notify.",
                err,
            )
            return TriageDecision(
                decision=TRIAGE_NOTIFY,
                reason_code=TRIAGE_REASON_ERROR,
                triage_confidence=None,
                summary=f"Triage error: {err!s}",
            )


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

    # recognized_people_count — count only, never names. Count ENROLLED
    # identities only: the raw list carries reserved pipeline labels, and an
    # unknown-person finding always contains "Unknown Person" — counting it
    # would tell the triage LLM a known person was recognized and invite it
    # to suppress the exact stranger alert the finding exists for. The
    # stranger signal is surfaced explicitly instead.
    people = raw_evidence.get("recognized_people")
    if isinstance(people, list):
        evidence_items.append(
            f"recognized_people_count: {len(enrolled_people(people))}"
        )
        if has_unknown_person(people):
            evidence_items.append("unknown_person_present: true")

    # last_changed_age_seconds — derived elapsed time, not raw timestamp.
    if "last_changed_age_seconds" in raw_evidence:
        val = raw_evidence["last_changed_age_seconds"]
        if isinstance(val, (int, float)):
            evidence_items.append(f"last_changed_age_seconds: {val:.0f}")

    evidence_block = ""
    if evidence_items:
        evidence_block = "\n  evidence:\n    " + "\n    ".join(evidence_items)

    prompt = _USER_PROMPT_TEMPLATE.format(
        type=finding.type,
        severity=finding.severity,
        confidence=finding.confidence,
        is_sensitive=finding.is_sensitive,
        entity_count=entity_count,
        suggested_actions_count=suggested_actions_count,
        evidence_block=evidence_block,
    )
    # The allowlist admits no identifier today; the gate keeps that true if
    # it ever grows (docs/network-security-plan.md, Privacy of the Audit).
    return redact_network_identifiers(prompt)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(result: Any, elapsed_ms: int) -> TriageDecision:  # noqa: ARG001
    """
    Parse LLM response into a ``TriageDecision``.

    Fails-open to ``notify`` on any parse error.
    """
    content = extract_final(getattr(result, "content", None) or "")

    # Strip think blocks, then markdown fences if the model wrapped the JSON.
    content = _THINK_BLOCK.sub("", content).strip()
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        _PARSE_LOG_LIMITER.warning(
            "non_json",
            "Triage LLM returned non-JSON response (%d chars); failing open.",
            len(content),
        )
        return TriageDecision(
            decision=TRIAGE_NOTIFY,
            reason_code=TRIAGE_REASON_ERROR,
            triage_confidence=None,
            summary="Could not parse triage response.",
        )

    # `[]`, `null` and a bare `"notify"` are all valid JSON that json.loads
    # accepts, so the decode above does not guarantee a mapping. Calling .get()
    # on the result raised AttributeError, which is neither JSONDecodeError nor
    # ValueError and so escaped both this handler and triage()'s fail-open
    # block -- killing the Sentinel run loop for the rest of the process (see
    # the module docstring's fail-open contract).
    if not isinstance(parsed, dict):
        _PARSE_LOG_LIMITER.warning(
            "non_json",
            "Triage LLM returned JSON %s rather than an object; failing open.",
            type(parsed).__name__,
        )
        return TriageDecision(
            decision=TRIAGE_NOTIFY,
            reason_code=TRIAGE_REASON_ERROR,
            triage_confidence=None,
            summary="Could not parse triage response.",
        )

    decision_raw = str(parsed.get("decision", TRIAGE_NOTIFY)).lower().strip()
    decision = TRIAGE_SUPPRESS if decision_raw == TRIAGE_SUPPRESS else TRIAGE_NOTIFY

    reason_code = str(parsed.get("reason_code", TRIAGE_REASON_LLM_NOTIFY)).strip() or (
        TRIAGE_REASON_LLM_SUPPRESS
        if decision == TRIAGE_SUPPRESS
        else TRIAGE_REASON_LLM_NOTIFY
    )
    _PARSE_LOG_LIMITER.recovered(
        "non_json",
        "Triage JSON parsing recovered after %d invalid response(s).",
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
