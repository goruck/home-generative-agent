"""LLM discovery engine for advisory anomaly ideas."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from langchain_core.messages import HumanMessage, SystemMessage

from ..const import (
    CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS,
)
from ..explain.discovery_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from ..snapshot.builder import async_build_full_state_snapshot
from ..snapshot.discovery_reducer import reduce_snapshot_for_discovery
from .discovery_semantic import candidate_semantic_key, rule_semantic_key
from .discovery_schema import DISCOVERY_OUTPUT_SCHEMA, DISCOVERY_SCHEMA_VERSION
from .discovery_store import DiscoveryStore

if TYPE_CHECKING:
    from .proposal_store import ProposalStore
    from .rule_registry import RuleRegistry

LOGGER = logging.getLogger(__name__)


class SentinelDiscoveryEngine:
    """Periodic LLM discovery loop (advisory only)."""

    def __init__(
        self,
        hass: HomeAssistant,
        options: dict[str, object],
        model: Any,
        store: DiscoveryStore,
        rule_registry: "RuleRegistry | None" = None,
        proposal_store: "ProposalStore | None" = None,
    ) -> None:
        self._hass = hass
        self._options = options
        self._model = model
        self._store = store
        self._rule_registry = rule_registry
        self._proposal_store = proposal_store
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

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
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        interval = int(
            self._options.get(CONF_SENTINEL_DISCOVERY_INTERVAL_SECONDS, 3600)
        )
        LOGGER.info("Sentinel discovery loop started (interval=%ss).", interval)
        while not self._stop_event.is_set():
            await self._run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _run_once(self) -> None:
        if self._model is None:
            LOGGER.debug("Discovery skipped: no model available.")
            return

        try:
            snapshot = await async_build_full_state_snapshot(self._hass)
        except (ValueError, TypeError, KeyError):
            LOGGER.warning("Failed to build snapshot for discovery.")
            return

        reduced_snapshot = reduce_snapshot_for_discovery(snapshot)
        safe_snapshot = json.loads(json.dumps(reduced_snapshot, default=str))
        active_rule_ids, existing_keys = await self._existing_semantic_context()
        now = dt_util.utcnow().isoformat()
        prompt = USER_PROMPT_TEMPLATE.format(
            snapshot=safe_snapshot,
            active_rule_ids=sorted(active_rule_ids),
            existing_semantic_keys=sorted(existing_keys),
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
            validated = DISCOVERY_OUTPUT_SCHEMA(payload)
        except vol.Invalid:
            LOGGER.warning("Discovery output failed schema validation.")
            return

        filtered, filtered_candidates = self._filter_novel_candidates(
            validated.get("candidates", []),
            existing_keys,
        )
        validated["candidates"] = filtered
        validated["filtered_candidates"] = filtered_candidates
        if not filtered:
            LOGGER.info(
                "Discovery produced no novel candidates after dedupe "
                "(filtered=%s).",
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

    async def _existing_semantic_context(self) -> tuple[set[str], set[str]]:
        active_rule_ids: set[str] = set()
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
                status = str(proposal.get("status", "draft"))
                if status == "rejected":
                    continue
                candidate = proposal.get("candidate")
                if isinstance(candidate, dict):
                    key = candidate_semantic_key(candidate)
                    if key:
                        semantic_keys.add(key)

        discovery_records = await self._store.async_get_latest(200)
        for payload in discovery_records:
            for candidate in payload.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                key = str(candidate.get("semantic_key", "")) or candidate_semantic_key(candidate)
                if key:
                    semantic_keys.add(key)

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
            dedupe_reason: str | None = None
            if key and key in existing_keys:
                dedupe_reason = "existing_semantic_key"
            elif key and key in seen_batch:
                dedupe_reason = "batch_duplicate"
            if dedupe_reason is not None:
                LOGGER.debug(
                    "Discovery dropped candidate %s with semantic key %s (%s).",
                    candidate.get("candidate_id"),
                    key,
                    dedupe_reason,
                )
                dropped_entry: dict[str, str] = {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "dedupe_reason": dedupe_reason,
                }
                if key:
                    dropped_entry["semantic_key"] = key
                dropped.append(dropped_entry)
                continue
            enriched = dict(candidate)
            if key:
                enriched["semantic_key"] = key
                seen_batch.add(key)
            enriched["dedupe_reason"] = "novel"
            filtered.append(enriched)
        return filtered, dropped
