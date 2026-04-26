# ruff: noqa: S101
"""Tests for the chat-priority GPU gate primitives in core/utils.py.

Covers:
- _chat_idle initial state (set = idle by default)
- chat_priority_context clears _chat_idle on entry, restores it on exit
- Concurrent chat contexts: _chat_idle only re-sets when the last chat exits
- Background workers cannot acquire _bg_vlm_lock or _bg_llm_lock while
  chat_priority_context holds them
- generate_embeddings waits on _chat_idle before proceeding
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import custom_components.home_generative_agent.core.utils as utils_mod
from custom_components.home_generative_agent.core.utils import (
    _bg_llm_lock,
    _bg_vlm_lock,
    _chat_idle,
    chat_priority_context,
    generate_embeddings,
)


# ---------------------------------------------------------------------------
# Fixture: reset module-level gate state between tests so tests are isolated.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_gate_state() -> None:  # type: ignore[return]
    """Restore gate primitives to initial state before and after each test."""
    _chat_idle.set()
    utils_mod._chat_active_count = 0
    # Release locks if somehow held (shouldn't happen, but defensive)
    if _bg_vlm_lock.locked():
        _bg_vlm_lock.release()
    if _bg_llm_lock.locked():
        _bg_llm_lock.release()
    yield
    _chat_idle.set()
    utils_mod._chat_active_count = 0


# ---------------------------------------------------------------------------
# _chat_idle initial state
# ---------------------------------------------------------------------------


def test_chat_idle_is_set_initially() -> None:
    """_chat_idle must be set (idle) at module load time."""
    assert _chat_idle.is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: entry / exit semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_priority_context_clears_idle_on_entry() -> None:
    """_chat_idle must be cleared while inside chat_priority_context."""
    assert _chat_idle.is_set()
    async with chat_priority_context():
        assert not _chat_idle.is_set()


@pytest.mark.asyncio
async def test_chat_priority_context_restores_idle_on_exit() -> None:
    """_chat_idle must be set again after chat_priority_context exits."""
    async with chat_priority_context():
        pass
    assert _chat_idle.is_set()


@pytest.mark.asyncio
async def test_chat_priority_context_restores_idle_on_exception() -> None:
    """_chat_idle must be restored even when the body raises."""
    with pytest.raises(RuntimeError):
        async with chat_priority_context():
            raise RuntimeError("boom")
    assert _chat_idle.is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: concurrent-chat reference counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_chat_contexts_keep_idle_cleared() -> None:
    """Two overlapping chat_priority_context calls: idle stays cleared until both exit."""
    results: list[bool] = []

    async def _enter_and_sample() -> None:
        async with chat_priority_context():
            results.append(_chat_idle.is_set())
            # Yield so the other task can also enter before we exit.
            await asyncio.sleep(0)
            results.append(_chat_idle.is_set())

    await asyncio.gather(_enter_and_sample(), _enter_and_sample())
    # Both tasks saw the event cleared while they were inside.
    assert all(r is False for r in results)
    # After both tasks exit, idle should be set again.
    assert _chat_idle.is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: lock exclusion of background workers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bg_vlm_lock_not_acquirable_inside_chat_context() -> None:
    """Background VLM worker cannot acquire _bg_vlm_lock while chat holds it."""
    async with chat_priority_context():
        # Attempt a non-blocking acquire — must fail because chat holds the lock.
        acquired = _bg_vlm_lock.locked()
        assert acquired, "_bg_vlm_lock should be locked inside chat_priority_context"


@pytest.mark.asyncio
async def test_bg_llm_lock_not_acquirable_inside_chat_context() -> None:
    """Background LLM worker cannot acquire _bg_llm_lock while chat holds it."""
    async with chat_priority_context():
        acquired = _bg_llm_lock.locked()
        assert acquired, "_bg_llm_lock should be locked inside chat_priority_context"


@pytest.mark.asyncio
async def test_bg_locks_released_after_chat_context() -> None:
    """Both background locks must be released after chat_priority_context exits."""
    async with chat_priority_context():
        pass
    assert not _bg_vlm_lock.locked()
    assert not _bg_llm_lock.locked()


# ---------------------------------------------------------------------------
# generate_embeddings: honours _chat_idle gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embeddings_waits_while_chat_active() -> None:
    """generate_embeddings must not proceed until _chat_idle is set."""
    _chat_idle.clear()  # simulate active chat

    embed_started = asyncio.Event()
    embed_result: list[object] = []

    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    # Not a GoogleGenerativeAIEmbeddings instance.
    mock_emb.__class__ = MagicMock  # type: ignore[assignment]

    async def _run_embed() -> None:
        embed_started.set()
        result = await generate_embeddings(mock_emb, ["hello"])
        embed_result.append(result)

    task = asyncio.create_task(_run_embed())
    await embed_started.wait()
    # Give the task a chance to (wrongly) proceed before we release the gate.
    await asyncio.sleep(0.01)
    assert not embed_result, "generate_embeddings should block while chat is active"

    _chat_idle.set()  # release the gate
    await asyncio.wait_for(task, timeout=2.0)
    assert embed_result, "generate_embeddings should complete after chat gate released"


@pytest.mark.asyncio
async def test_generate_embeddings_proceeds_when_idle() -> None:
    """generate_embeddings completes immediately when no chat is active."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    mock_emb.__class__ = MagicMock  # type: ignore[assignment]

    result = await generate_embeddings(mock_emb, ["hello"])
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_generate_embeddings_empty_texts_returns_empty() -> None:
    """generate_embeddings with empty input returns [] without hitting the model."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[])

    result = await generate_embeddings(mock_emb, [])
    mock_emb.aembed_documents.assert_not_called()
    assert result == []
