"""Jinja stripping for RAG embeddings and instruction dual-index score fusion."""

from __future__ import annotations

import re
from typing import Any

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_BLOCK = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_JINJA_VAR = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def strip_for_embedding(text: str) -> str:
    """Remove Jinja for stable embeddings; preserve word boundaries."""
    if not text:
        return ""
    out = _JINJA_COMMENT.sub(" ", text)
    out = _JINJA_BLOCK.sub(" ", out)
    out = _JINJA_VAR.sub("[value]", out)
    return re.sub(r"\s+", " ", out).strip()


def fused_similarity(
    sim_intent: float | None,
    sim_body: float | None,
    alpha: float,
    *,
    noise_floor: float,
) -> float:
    """Normalize dual-channel scores; drop garbage similarities before fusion."""
    if sim_intent is not None and sim_intent < noise_floor:
        sim_intent = None
    if sim_body is not None and sim_body < noise_floor:
        sim_body = None
    wi = alpha if sim_intent is not None else 0.0
    wb = (1.0 - alpha) if sim_body is not None else 0.0
    denom = wi + wb
    if denom <= 0.0:
        return 0.0
    return (wi * (sim_intent or 0.0) + wb * (sim_body or 0.0)) / denom


def instruction_keys_fused_from_search_results(  # noqa: PLR0913
    intent_items: list[Any],
    body_items: list[Any],
    *,
    instruction_limit: int,
    instruction_threshold: float,
    alpha: float,
    noise_floor: float,
) -> list[str]:
    """Merge instruction + instructions_body vector hits; optionally empty lists."""

    def scores_by_key(items: list[Any]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for item in items:
            key = str(item.key)
            sc = getattr(item, "score", None)
            out[key] = float(sc) if sc is not None else None
        return out

    intent_scores = scores_by_key(intent_items)
    body_scores = scores_by_key(body_items)
    all_keys = set(intent_scores) | set(body_scores)

    ranked: list[tuple[str, float]] = []
    for key in all_keys:
        fused = fused_similarity(
            intent_scores.get(key),
            body_scores.get(key),
            alpha,
            noise_floor=noise_floor,
        )
        if fused >= instruction_threshold:
            ranked.append((key, fused))

    ranked.sort(key=lambda kv: -kv[1])
    return [k for k, _ in ranked[:instruction_limit]]
