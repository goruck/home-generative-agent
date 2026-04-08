"""Jinja stripping for RAG embeddings."""

from __future__ import annotations

import re

# Max characters embedded per vector-store content row
EMBEDDING_INDEX_TEXT_MAX_CHARS = 1200

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


def truncate_for_embedding_index(
    text: str,
    *,
    max_chars: int = EMBEDDING_INDEX_TEXT_MAX_CHARS,
) -> str:
    """Trim text for pgvector content so embeddings stay under per-input limits."""
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return f"{t[: max_chars - 3]}..."
