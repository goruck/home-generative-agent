"""LLM discovery engine for advisory anomaly ideas."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.util import dt as dt_util
from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
)
from custom_components.home_generative_agent.core.utils import (
    SENTINEL_ADMISSION_TIMEOUT_S,
    SentinelLLMDeferredError,
    extract_final,
    run_sentinel_model_call,
)
from custom_components.home_generative_agent.explain.discovery_prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from custom_components.home_generative_agent.snapshot.builder import (
    async_build_full_state_snapshot,
)
from custom_components.home_generative_agent.snapshot.discovery_reducer import (
    reduce_snapshot_for_discovery,
)

from .discovery_schema import DISCOVERY_OUTPUT_SCHEMA, DISCOVERY_SCHEMA_VERSION
from .discovery_semantic import candidate_semantic_key, rule_semantic_key
from .logging_utils import RepeatingLogLimiter

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .baseline import SentinelBaselineUpdater
    from .discovery_store import DiscoveryStore
    from .proposal_store import ProposalStore
    from .rule_registry import RuleRegistry

LOGGER = logging.getLogger(__name__)

# Hard cap on discovery LLM calls: prevents the chat model from being
# monopolised when the snapshot is large (observed 180s+ without a cap).
_DISCOVERY_LLM_TIMEOUT_S: float = 60.0
# Max semantic keys sent in the discovery prompt. The post-hoc filter catches
# any duplicates that slip through, so a generous cap is safe.
_MAX_SEMANTIC_KEYS_IN_PROMPT: int = 60

# Markers that identify rule-derived baseline monitoring hint keys.
# rule_semantic_key() for baseline_deviation and time_of_day_anomaly embeds
# |template=<name>| in the key; candidate_semantic_key() never does.
# Using template markers (not predicate markers) is critical: candidate keys
# with predicate=power_anomaly may be multi-entity bundles from rejected or
# pending proposals, and a single bundle key must not suppress individual
# entity proposals for all entities it lists.
_BASELINE_TEMPLATE_MARKERS: frozenset[str] = frozenset(
    {
        "|template=baseline_deviation",
        "|template=time_of_day_anomaly",
    }
)

_ENTITIES_FIELD_RE = re.compile(r"\|entities=([^|]*)")
_EVIDENCE_ENTITY_RE = re.compile(
    r"\bentity_id=([a-z0-9_]+\.[a-z0-9_]+)|"
    r"\bentity_ids\s+contains\s+([a-z0-9_]+\.[a-z0-9_]+)"
)
_TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENTITY_DESCRIPTOR_STOPWORDS: frozenset[str] = frozenset(
    {
        "binary",
        "class",
        "device",
        "entity",
        "high",
        "low",
        "percentage",
        "sensor",
        "state",
        "switch",
    }
)
_MIN_PLURAL_TOKEN_LENGTH = 3


def _is_cumulative_energy_entity(entity_id: str) -> bool:
    """
    Return True for monotonically increasing kWh counters.

    These sensors can never produce meaningful rolling-average baseline
    proposals — the ever-growing value drifts away from any fixed baseline.
    Mirrors proposal_templates._is_cumulative_energy_sensor; kept local to
    avoid coupling the discovery pipeline to the normalization module.
    """
    local = entity_id.split(".", 1)[-1] if "." in entity_id else entity_id
    return local.endswith("_energy") or local == "energy"


def _entity_ids_from_key(key: str) -> set[str]:
    """Parse the entities= CSV field from a semantic key into exact entity IDs."""
    m = _ENTITIES_FIELD_RE.search(key)
    if not m:
        return set()
    return {e for e in m.group(1).split(",") if e}


_STATIC_RULE_IDS: frozenset[str] = frozenset(
    {
        "unlocked_lock_at_night",
        "open_entry_while_away",
        "appliance_power_duration",
        "camera_entry_unsecured",
        "unknown_person_camera_no_home",
        "unknown_person_camera_night_home",
        "vehicle_detected_near_camera_home",
        "camera_missing_snapshot_night_home",
        "alarm_disarmed_during_external_threat",
        "phone_battery_low_at_night_home",
    }
)


class SentinelDiscoveryEngine:
    """Periodic LLM discovery loop (advisory only)."""

    def __init__(  # noqa: PLR0913
        self,
        hass: HomeAssistant,
        options: dict[str, object],
        model: Any,
        store: DiscoveryStore,
        rule_registry: RuleRegistry | None = None,
        proposal_store: ProposalStore | None = None,
        baseline_updater: SentinelBaselineUpdater | None = None,
        deployment: str = "edge",
        health_stats: dict[str, Any] | None = None,
    ) -> None:
        """Initialize advisory discovery dependencies and runtime state."""
        self._hass = hass
        self._options = options
        self._model = model
        self._store = store
        self._rule_registry = rule_registry
        self._proposal_store = proposal_store
        self._baseline_updater = baseline_updater
        self._deployment = deployment
        self._health_stats = health_stats
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._discovery_cycle_stats: dict[str, int] = {}
        self._log_limiter = RepeatingLogLimiter(LOGGER)

    def start(self) -> None:
        """Start the discovery loop."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = self._hass.async_create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the discovery loop."""
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run_loop(self) -> None:
        interval = _coerce_int(
            self._options.get(CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS), default=3600
        )
        while not self._stop_event.is_set():
            try:
                await self._run_once_guarded()
            except Exception:  # noqa: BLE001
                # A dead loop silently disables discovery until the
                # integration reloads (issue #465); drop the cycle and
                # keep the loop alive.
                self._log_limiter.warning(
                    "cycle_error",
                    "Discovery cycle failed unexpectedly; will retry next interval.",
                    exc_info=True,
                )
            else:
                self._log_limiter.recovered(
                    "cycle_error",
                    "Discovery cycle recovered after %d failed cycle(s).",
                )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def async_run_now(self) -> bool:
        """Run one discovery cycle immediately if idle."""
        if self._run_lock.locked():
            LOGGER.debug("Discovery run already in progress; skipping manual trigger.")
            return False
        await self._run_once_guarded()
        return True

    async def _run_once_guarded(self) -> None:
        async with self._run_lock:
            await self._run_once()

    @property
    def discovery_cycle_stats(self) -> dict[str, int]:
        """Return stats from the most recent discovery cycle."""
        return dict(self._discovery_cycle_stats)

    async def _run_once(self) -> None:  # noqa: PLR0911, PLR0912, PLR0915
        self._discovery_cycle_stats = {
            "candidates_generated": 0,
            "candidates_novel": 0,
            "candidates_deduplicated": 0,
            "unsupported_ttl_expired": 0,
        }
        if self._model is None:
            LOGGER.warning("Discovery skipped: no model configured.")
            return

        try:
            snapshot = await async_build_full_state_snapshot(self._hass)
        except (ValueError, TypeError, KeyError):
            self._log_limiter.warning(
                "snapshot_build",
                "Failed to build snapshot for discovery.",
            )
            return
        self._log_limiter.recovered(
            "snapshot_build",
            "Discovery snapshot build recovered after %d failed cycle(s).",
        )

        if self._baseline_updater is not None:
            ready_ids = await self._baseline_updater.async_fetch_ready_entity_ids()
            snapshot["derived"]["baseline_ready_entities"] = ready_ids

        reduced_snapshot = reduce_snapshot_for_discovery(snapshot)
        compact_snapshot = json.dumps(
            reduced_snapshot, default=str, separators=(",", ":")
        )
        # Bug 3 fix: prune stale unsupported proposals before building exclusion
        # context so they stop blocking the card indefinitely.  Wrapped in
        # try/except so a cleanup failure never aborts the discovery cycle.
        if self._proposal_store is not None:
            try:
                expired = await self._proposal_store.cleanup_unsupported_ttl()
                if expired:
                    LOGGER.debug(
                        "Discovery TTL cleanup: expired %d unsupported proposal(s).",
                        expired,
                    )
                    self._discovery_cycle_stats["unsupported_ttl_expired"] += expired
            except Exception:  # noqa: BLE001
                self._log_limiter.warning(
                    "ttl_cleanup",
                    "Discovery TTL cleanup failed; continuing.",
                    exc_info=True,
                )
        (
            active_rule_ids,
            hint_keys,
            filter_keys,
        ) = await self._existing_semantic_context()
        # hint_keys: active rules + pending proposals — what is truly active or
        # awaiting approval.  Sent to the LLM so it avoids re-proposing covered
        # topics.  History record keys are intentionally excluded: a past
        # multi-entity bundle key listing "sensor.fridge_switch_0_power" among
        # several others would mislead the LLM into thinking each individual
        # entity is already monitored, suppressing standalone proposals.
        # filter_keys: full set including history — used only by the post-hoc
        # dedup filter to prevent identical candidates from re-appearing.
        capped_keys = sorted(hint_keys)[:_MAX_SEMANTIC_KEYS_IN_PROMPT]

        # Compute monitoring gaps: baseline-ready entities with no active
        # statistical anomaly rule.  Only rule_semantic_key()-derived keys
        # contain |template=<name>|; candidate_semantic_key()-derived keys never
        # do.  Using template markers prevents rejected or pending multi-entity
        # bundle proposals (which list many entity_ids in one key) from
        # suppressing individual entity proposals for every entity they mention.
        baseline_hint_keys = {
            k for k in hint_keys if any(m in k for m in _BASELINE_TEMPLATE_MARKERS)
        }
        try:
            baseline_ready: list[str] = (
                reduced_snapshot.get("derived", {}).get("baseline_ready_entities") or []
            )
        except (AttributeError, TypeError):
            baseline_ready = []
        covered_entity_ids: set[str] = set()
        for key in baseline_hint_keys:
            covered_entity_ids.update(_entity_ids_from_key(key))
        unmonitored = [
            eid
            for eid in baseline_ready
            if eid not in covered_entity_ids and not _is_cumulative_energy_entity(eid)
        ]
        if LOGGER.isEnabledFor(logging.DEBUG):
            for eid in baseline_ready:
                if eid in covered_entity_ids:
                    covering = [
                        k for k in baseline_hint_keys if eid in _entity_ids_from_key(k)
                    ]
                    LOGGER.debug(
                        "Discovery gap: %s covered by %d key(s): %s",
                        eid,
                        len(covering),
                        covering,
                    )
        LOGGER.debug(
            "Discovery gap analysis: %d baseline-ready, %d baseline hint keys, "
            "%d unmonitored: %s",
            len(baseline_ready),
            len(baseline_hint_keys),
            len(unmonitored),
            unmonitored,
        )
        unmonitored_json = json.dumps(unmonitored, separators=(",", ":"))

        now = dt_util.utcnow().isoformat()
        prompt = USER_PROMPT_TEMPLATE.format(
            snapshot=compact_snapshot,
            active_rule_ids=json.dumps(sorted(active_rule_ids), separators=(",", ":")),
            existing_semantic_keys=json.dumps(capped_keys, separators=(",", ":")),
            unmonitored_baseline_entities=unmonitored_json,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            result = await run_sentinel_model_call(
                self._model,
                messages,
                deployment=self._deployment,
                category="discovery",
                admission_timeout_s=SENTINEL_ADMISSION_TIMEOUT_S,
                call_timeout_s=_DISCOVERY_LLM_TIMEOUT_S,
                health_stats=self._health_stats,
            )
        except SentinelLLMDeferredError as err:
            LOGGER.info("Discovery LLM call deferred: %s", err)
            return
        except TimeoutError:
            self._log_limiter.warning(
                "llm_call",
                "Discovery LLM call timed out after %.0fs; skipping cycle.",
                _DISCOVERY_LLM_TIMEOUT_S,
            )
            return
        except Exception as err:  # noqa: BLE001
            # Provider errors (e.g. ollama.ResponseError for an un-pulled
            # model, httpx transport failures) do not subclass the usual
            # ValueError/RuntimeError families; a narrow catch here let them
            # escape and kill the discovery loop (issue #465).
            self._log_limiter.warning(
                "llm_call",
                "Discovery LLM call failed: %s",
                err,
            )
            return
        self._log_limiter.recovered(
            "llm_call",
            "Discovery LLM call recovered after %d failed cycle(s).",
        )

        content = extract_final(getattr(result, "content", None) or "")
        if not content:
            LOGGER.debug("Discovery LLM returned empty content.")
            return

        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            self._log_limiter.warning(
                "invalid_output",
                "Discovery output was not valid JSON.",
            )
            return

        payload.setdefault("schema_version", DISCOVERY_SCHEMA_VERSION)
        payload.setdefault("generated_at", now)
        payload.setdefault("model", _resolved_model_name(self._model))

        try:
            validated = cast("dict[str, Any]", DISCOVERY_OUTPUT_SCHEMA(payload))
        except vol.Invalid:
            self._log_limiter.warning(
                "invalid_output",
                "Discovery output failed schema validation.",
            )
            return
        self._log_limiter.recovered(
            "invalid_output",
            "Discovery output recovered after %d invalid response(s).",
        )

        raw_candidates = validated.get("candidates", [])
        if not isinstance(raw_candidates, list):
            self._log_limiter.warning(
                "invalid_candidates",
                "Discovery output candidates were not a list.",
            )
            return
        self._log_limiter.recovered(
            "invalid_candidates",
            "Discovery candidates recovered after %d invalid response(s).",
        )
        candidates = [item for item in raw_candidates if isinstance(item, dict)]
        self._discovery_cycle_stats["candidates_generated"] = len(candidates)
        filtered, filtered_candidates = self._filter_novel_candidates(
            candidates,
            filter_keys,
            baseline_ready,
        )
        self._discovery_cycle_stats["candidates_novel"] = len(filtered)
        self._discovery_cycle_stats["candidates_deduplicated"] = len(
            filtered_candidates
        )
        validated["candidates"] = filtered
        validated["filtered_candidates"] = filtered_candidates
        await self._store.async_append(validated)
        LOGGER.info(
            "Discovery cycle complete: %d generated, %d novel, %d deduplicated.",
            len(candidates),
            len(filtered),
            len(filtered_candidates),
        )

    async def _existing_semantic_context(  # noqa: PLR0912
        self,
    ) -> tuple[set[str], set[str], set[str]]:
        """
        Return (active_rule_ids, hint_keys, filter_keys).

        hint_keys — active rules + pending proposals.  Sent to the LLM so it
        avoids re-proposing what is genuinely monitored or already in the
        approval queue.  History record keys are excluded because past
        multi-entity bundles would mislead the LLM into suppressing individual
        entity proposals that were never actually activated.

        filter_keys — full superset including discovery history.  Used only by
        the post-hoc dedup filter so identical candidates do not re-appear.
        """
        # Bug 4 fix: seed with static built-in rule IDs so the LLM is told
        # those topics are already covered and stops re-suggesting them.
        active_rule_ids: set[str] = set(_STATIC_RULE_IDS)
        hint_keys: set[str] = set()
        filter_keys: set[str] = set()

        if self._rule_registry is not None:
            for rule in self._rule_registry.list_rules():
                rule_id = str(rule.get("rule_id", ""))
                if rule_id:
                    active_rule_ids.add(rule_id)
                key = rule_semantic_key(rule)
                if key:
                    hint_keys.add(key)

        if self._proposal_store is not None:
            proposals = await self._proposal_store.async_get_latest(200)
            for proposal in proposals:
                # Bug 1 fix: rejected proposals must still block re-suggestion.
                # The old `continue` meant rejected candidates' keys were never
                # added to the exclusion set, letting identical proposals recur.
                #
                # Accepted proposals are excluded from hint_keys: their coverage
                # is tracked by the live rule via rule_semantic_key (if enabled).
                # If the user later disables the rule, the topic becomes
                # re-proposable — the accepted proposal must not silently suppress
                # it forever.
                status = str(proposal.get("status", "pending"))
                if status == "approved":
                    continue
                candidate = proposal.get("candidate")
                if isinstance(candidate, dict):
                    key = candidate_semantic_key(candidate)
                    if key:
                        hint_keys.add(key)
                    else:
                        # Bug 2 fix: null-key candidates (e.g. "stale tracking"
                        # patterns that don't resolve to a known subject/predicate)
                        # use a title+summary hash so they are still excluded.
                        hint_keys.add(_candidate_identity_hash(candidate))

        # filter_keys starts as a copy of hint_keys, then adds history records.
        filter_keys = set(hint_keys)
        discovery_records = await self._store.async_get_latest(200)
        for payload in discovery_records:
            for candidate in payload.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                key = str(candidate.get("semantic_key", "")) or candidate_semantic_key(
                    candidate
                )
                if key:
                    filter_keys.add(key)
                else:
                    # Bug 2 fix: same null-key hash fallback for past records.
                    filter_keys.add(_candidate_identity_hash(candidate))

        return active_rule_ids, hint_keys, filter_keys

    def _filter_novel_candidates(
        self,
        candidates: list[dict[str, Any]],
        existing_keys: set[str],
        known_entity_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        seen_batch: set[str] = set()
        filtered: list[dict[str, Any]] = []
        dropped: list[dict[str, str]] = []
        entity_descriptor_index = _build_entity_descriptor_index(known_entity_ids or [])
        for candidate in candidates:
            key = candidate_semantic_key(candidate)
            # Bug 2 fix: null-key candidates (unknown subject/predicate) fall
            # back to a title+summary hash so they are still deduplicated.
            identity_key = key or _candidate_identity_hash(candidate)

            # Hard filter: candidates whose every evidence_path starts with
            # 'derived.' have no concrete entity evidence and can never be
            # promoted to a rule.  Gate BEFORE the dedup check so these do not
            # pollute the dedup exclusion set.
            evidence_paths = candidate.get("evidence_paths")
            if (
                evidence_paths is not None
                and len(evidence_paths) > 0
                and all(p.startswith("derived.") for p in evidence_paths)
            ):
                dropped.append(
                    {
                        "candidate_id": str(candidate.get("candidate_id", "")),
                        "dedupe_reason": "derived_only_paths",
                    }
                )
                continue

            evidence_entity_ids = _entity_ids_from_evidence_paths(evidence_paths)
            mismatch_entities = _candidate_text_entity_mismatches(
                candidate,
                evidence_entity_ids,
                entity_descriptor_index,
            )
            if mismatch_entities:
                dropped.append(
                    {
                        "candidate_id": str(candidate.get("candidate_id", "")),
                        "dedupe_reason": "entity_text_mismatch",
                        "mismatch_entities": ",".join(sorted(mismatch_entities)),
                    }
                )
                continue

            dedupe_reason: str | None = None
            if identity_key in existing_keys:
                dedupe_reason = (
                    "existing_semantic_key" if key else "existing_identity_hash"
                )
            elif identity_key in seen_batch:
                dedupe_reason = "batch_duplicate"
            if dedupe_reason is not None:
                dropped_entry: dict[str, str] = {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "dedupe_reason": dedupe_reason,
                }
                if key:
                    dropped_entry["semantic_key"] = key
                else:
                    dropped_entry["identity_hash"] = identity_key
                dropped.append(dropped_entry)
                continue
            enriched = dict(candidate)
            if key:
                enriched["semantic_key"] = key
            seen_batch.add(identity_key)
            enriched["dedupe_reason"] = "novel"
            filtered.append(enriched)
        return filtered, dropped


def _candidate_identity_hash(candidate: dict[str, Any]) -> str:
    """
    Return a stable identity key for candidates that have no semantic key.

    Uses the first 16 hex digits of SHA-256(title + NUL + summary) so that
    two candidates with identical wording collide even if their other fields
    differ (e.g. different confidence_hint or candidate_id).
    """
    title = str(candidate.get("title", "")).strip().lower()
    summary = str(candidate.get("summary", "")).strip().lower()
    blob = f"{title}\x00{summary}".encode()
    return "ident|sha256=" + hashlib.sha256(blob).hexdigest()[:16]


def _entity_ids_from_evidence_paths(evidence_paths: object) -> set[str]:
    """Extract concrete entity IDs from LLM evidence path strings."""
    if not isinstance(evidence_paths, list):
        return set()
    entity_ids: set[str] = set()
    for path in evidence_paths:
        if not isinstance(path, str):
            continue
        for match in _EVIDENCE_ENTITY_RE.finditer(path.lower()):
            entity_id = match.group(1) or match.group(2)
            if entity_id:
                entity_ids.add(entity_id)
    return entity_ids


def _build_entity_descriptor_index(
    entity_ids: list[str],
) -> dict[str, frozenset[str]]:
    """Map entity IDs to distinctive text tokens usable in LLM output checks."""
    index: dict[str, frozenset[str]] = {}
    for entity_id in entity_ids:
        if not isinstance(entity_id, str) or "." not in entity_id:
            continue
        normalized = entity_id.lower()
        _domain, _dot, object_id = normalized.partition(".")
        tokens = _normalized_descriptor_tokens(object_id)
        if tokens:
            index[normalized] = frozenset(tokens)
    return index


def _candidate_text_entity_mismatches(
    candidate: dict[str, Any],
    evidence_entity_ids: set[str],
    entity_descriptor_index: dict[str, frozenset[str]],
) -> set[str]:
    """Return known non-evidence entities named by candidate text."""
    if not evidence_entity_ids or not entity_descriptor_index:
        return set()
    text = " ".join(
        str(candidate.get(field, ""))
        for field in ("candidate_id", "title", "summary", "pattern")
    ).lower()
    text_tokens = _normalized_descriptor_tokens(text)
    if not text_tokens:
        return set()

    evidence_entity_ids = {entity_id.lower() for entity_id in evidence_entity_ids}
    mismatches: set[str] = set()
    for entity_id, descriptor_tokens in entity_descriptor_index.items():
        if entity_id in evidence_entity_ids:
            continue
        if descriptor_tokens and descriptor_tokens.issubset(text_tokens):
            mismatches.add(entity_id)
    return mismatches


def _normalized_descriptor_tokens(value: str) -> frozenset[str]:
    """Return normalized entity-description tokens for text comparison."""
    return frozenset(
        _singularize_token(token)
        for token in _TEXT_TOKEN_RE.findall(value)
        if len(token) > 1
        and not token.isdigit()
        and token not in _ENTITY_DESCRIPTOR_STOPWORDS
    )


def _singularize_token(token: str) -> str:
    """Normalize simple English plurals that appear in LLM summaries."""
    if len(token) > _MIN_PLURAL_TOKEN_LENGTH and token.endswith("ies"):
        return token[: -len("ies")] + "y"
    if len(token) > _MIN_PLURAL_TOKEN_LENGTH and token.endswith("s"):
        return token[:-1]
    return token


def _resolved_model_name(model: Any) -> str:
    """
    Best-effort name of the model a runnable will actually invoke.

    Models arrive here as bound runnables (``.with_config(...)``) whose
    effective model lives in the binding's configurable section; fallback
    wrappers (``FallbackChatModel``) carry an empty wrapper config and expose
    the bound per-provider models via ``chain``, whose first member is the
    primary selected at setup; raw chat models expose a ``model`` attribute.
    Recording the resolved name in discovery payloads makes model
    misconfiguration diagnosable from the audit trail (issue #465).
    """
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            for key in ("model", "model_name"):
                name = configurable.get(key)
                if name:
                    return str(name)
    chain = getattr(model, "chain", None)
    if isinstance(chain, list) and chain:
        first = chain[0]
        if isinstance(first, tuple) and first:
            return _resolved_model_name(first[0])
    name = getattr(model, "model", None)
    return str(name) if name else "unknown"


def _coerce_int(value: object | None, default: int) -> int:
    """Coerce option values to int with a deterministic fallback."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
