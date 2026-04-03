"""LLM discovery engine for advisory anomaly ideas."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.util import dt as dt_util
from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
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

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .baseline import SentinelBaselineUpdater
    from .discovery_store import DiscoveryStore
    from .proposal_store import ProposalStore
    from .rule_registry import RuleRegistry

LOGGER = logging.getLogger(__name__)

# Rule IDs for static built-in rules (deterministic, always active).
# Included in active_rule_ids so the LLM knows these topics are already covered
# and does not re-suggest candidates that map to an existing static rule.
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
    ) -> None:
        """Initialize advisory discovery dependencies and runtime state."""
        self._hass = hass
        self._options = options
        self._model = model
        self._store = store
        self._rule_registry = rule_registry
        self._proposal_store = proposal_store
        self._baseline_updater = baseline_updater
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._discovery_cycle_stats: dict[str, int] = {}

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
        LOGGER.info("Sentinel discovery loop started (interval=%ss).", interval)
        while not self._stop_event.is_set():
            await self._run_once_guarded()
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
            "proposals_promoted": 0,
            "unsupported_ttl_expired": 0,
        }
        if self._model is None:
            LOGGER.debug("Discovery skipped: no model available.")
            return

        try:
            snapshot = await async_build_full_state_snapshot(self._hass)
        except (ValueError, TypeError, KeyError):
            LOGGER.warning("Failed to build snapshot for discovery.")
            return

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
                    LOGGER.info(
                        "Discovery TTL cleanup: expired %d unsupported proposal(s).",
                        expired,
                    )
                    self._discovery_cycle_stats["unsupported_ttl_expired"] += expired
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "Discovery TTL cleanup failed; continuing.", exc_info=True
                )
        active_rule_ids, existing_keys = await self._existing_semantic_context()
        now = dt_util.utcnow().isoformat()
        prompt = USER_PROMPT_TEMPLATE.format(
            snapshot=compact_snapshot,
            active_rule_ids=json.dumps(sorted(active_rule_ids), separators=(",", ":")),
            existing_semantic_keys=json.dumps(
                sorted(existing_keys), separators=(",", ":")
            ),
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            result = await self._model.ainvoke(messages)
        except (ValueError, TypeError, RuntimeError) as err:
            LOGGER.warning("Discovery LLM call failed: %s", err)
            return

        content = getattr(result, "content", None)
        if not content:
            LOGGER.debug("Discovery LLM returned empty content.")
            return

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            LOGGER.warning("Discovery output was not valid JSON.")
            return

        payload.setdefault("schema_version", DISCOVERY_SCHEMA_VERSION)
        payload.setdefault("generated_at", now)
        payload.setdefault("model", str(getattr(self._model, "model", "unknown")))

        try:
            validated = cast("dict[str, Any]", DISCOVERY_OUTPUT_SCHEMA(payload))
        except vol.Invalid:
            LOGGER.warning("Discovery output failed schema validation.")
            return

        raw_candidates = validated.get("candidates", [])
        if not isinstance(raw_candidates, list):
            LOGGER.warning("Discovery output candidates were not a list.")
            return
        candidates = [item for item in raw_candidates if isinstance(item, dict)]
        self._discovery_cycle_stats["candidates_generated"] = len(candidates)
        filtered, filtered_candidates = self._filter_novel_candidates(
            candidates,
            existing_keys,
        )
        self._discovery_cycle_stats["candidates_novel"] = len(filtered)
        self._discovery_cycle_stats["candidates_deduplicated"] = len(
            filtered_candidates
        )
        validated["candidates"] = filtered
        validated["filtered_candidates"] = filtered_candidates
        if not filtered:
            LOGGER.info(
                "Discovery produced no novel candidates after dedupe (filtered=%s).",
                len(filtered_candidates),
            )
        else:
            LOGGER.debug(
                "Discovery kept %s novel candidate(s) and filtered %s candidate(s).",
                len(filtered),
                len(filtered_candidates),
            )

        await self._store.async_append(validated)
        LOGGER.info(
            "Discovery stored %s candidate(s).", len(validated.get("candidates", []))
        )

    async def _existing_semantic_context(  # noqa: PLR0912
        self,
    ) -> tuple[set[str], set[str]]:
        # Bug 4 fix: seed with static built-in rule IDs so the LLM is told
        # those topics are already covered and stops re-suggesting them.
        active_rule_ids: set[str] = set(_STATIC_RULE_IDS)
        semantic_keys: set[str] = set()

        if self._rule_registry is not None:
            for rule in self._rule_registry.list_rules():
                rule_id = str(rule.get("rule_id", ""))
                if rule_id:
                    active_rule_ids.add(rule_id)
                key = rule_semantic_key(rule)
                if key:
                    semantic_keys.add(key)

        if self._proposal_store is not None:
            proposals = await self._proposal_store.async_get_latest(200)
            for proposal in proposals:
                # Bug 1 fix: rejected proposals must still block re-suggestion.
                # The old `continue` meant rejected candidates' keys were never
                # added to the exclusion set, letting identical proposals recur.
                candidate = proposal.get("candidate")
                if isinstance(candidate, dict):
                    key = candidate_semantic_key(candidate)
                    if key:
                        semantic_keys.add(key)
                    else:
                        # Bug 2 fix: null-key candidates (e.g. "stale tracking"
                        # patterns that don't resolve to a known subject/predicate)
                        # use a title+summary hash so they are still excluded.
                        semantic_keys.add(_candidate_identity_hash(candidate))

        discovery_records = await self._store.async_get_latest(200)
        for payload in discovery_records:
            for candidate in payload.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                key = str(candidate.get("semantic_key", "")) or candidate_semantic_key(
                    candidate
                )
                if key:
                    semantic_keys.add(key)
                else:
                    # Bug 2 fix: same null-key hash fallback for past records.
                    semantic_keys.add(_candidate_identity_hash(candidate))

        return active_rule_ids, semantic_keys

    def _filter_novel_candidates(
        self,
        candidates: list[dict[str, Any]],
        existing_keys: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        seen_batch: set[str] = set()
        filtered: list[dict[str, Any]] = []
        dropped: list[dict[str, str]] = []
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
                LOGGER.debug(
                    "Discovery dropped candidate %s (derived_only_paths).",
                    candidate.get("candidate_id"),
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
                LOGGER.debug(
                    "Discovery dropped candidate %s with key %s (%s).",
                    candidate.get("candidate_id"),
                    identity_key,
                    dedupe_reason,
                )
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
