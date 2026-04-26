# ruff: noqa: S101
"""
Tests for the chat-priority GPU gate primitives in core/utils.py.

Covers:
- _chat_idle initial state (set = idle by default)
- chat_priority_context clears _chat_idle on entry, restores it on exit
- Concurrent chat contexts: _chat_idle only re-sets when the last chat exits
- Background workers cannot acquire _bg_vlm_lock or _bg_llm_lock while
  chat_priority_context holds them
- generate_embeddings (Ollama) waits on _chat_idle AND holds _bg_llm_lock
  during the embed call, so chat_priority_context blocks it
- generate_embeddings (non-Ollama) bypasses the gate entirely
- chat_priority_context cleans up _chat_idle on cancellation before lock
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_ollama import OllamaEmbeddings

import custom_components.home_generative_agent.core.utils as utils_mod

# ---------------------------------------------------------------------------
# Fixture: create fresh asyncio primitives per test.
#
# Module-level asyncio.Event / asyncio.Lock objects bind to the first event
# loop that touches them.  pytest-asyncio gives each async test its own loop,
# so re-using the module globals across tests causes
# "bound to a different event loop" errors.  We replace the module attributes
# with brand-new primitives created inside the test's own loop, then restore
# them afterwards via monkeypatch.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _fresh_gate_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace module-level gate primitives with fresh instances for each test."""
    new_idle = asyncio.Event()
    new_idle.set()  # "no chat active" by default
    new_llm_lock = asyncio.Lock()
    new_vlm_lock = asyncio.Lock()

    monkeypatch.setattr(utils_mod, "_chat_idle", new_idle)
    monkeypatch.setattr(utils_mod, "_bg_llm_lock", new_llm_lock)
    monkeypatch.setattr(utils_mod, "_bg_vlm_lock", new_vlm_lock)
    monkeypatch.setattr(utils_mod, "_chat_active_count", 0)


# ---------------------------------------------------------------------------
# Helpers: read current primitives through the module to pick up monkeypatched
# values (direct imports would cache the original objects).
# ---------------------------------------------------------------------------


def _idle() -> asyncio.Event:
    return utils_mod._chat_idle  # type: ignore[attr-defined]


def _llm_lock() -> asyncio.Lock:
    return utils_mod._bg_llm_lock  # type: ignore[attr-defined]


def _vlm_lock() -> asyncio.Lock:
    return utils_mod._bg_vlm_lock  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _chat_idle initial state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_idle_is_set_initially() -> None:
    """_chat_idle must be set (idle) after fixture reset."""
    assert _idle().is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: entry / exit semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_priority_context_clears_idle_on_entry() -> None:
    """_chat_idle must be cleared while inside chat_priority_context."""
    assert _idle().is_set()
    async with utils_mod.chat_priority_context():
        assert not _idle().is_set()


@pytest.mark.asyncio
async def test_chat_priority_context_restores_idle_on_exit() -> None:
    """_chat_idle must be set again after chat_priority_context exits."""
    async with utils_mod.chat_priority_context():
        pass
    assert _idle().is_set()


@pytest.mark.asyncio
async def test_chat_priority_context_restores_idle_on_exception() -> None:
    """_chat_idle must be restored even when the body raises."""
    err_msg = "boom"
    with pytest.raises(RuntimeError):
        async with utils_mod.chat_priority_context():
            raise RuntimeError(err_msg)
    assert _idle().is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: concurrent-chat reference counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_chat_contexts_keep_idle_cleared() -> None:
    """Two overlapping chat_priority_context calls: idle stays cleared until both exit."""
    results: list[bool] = []

    async def _enter_and_sample() -> None:
        async with utils_mod.chat_priority_context():
            results.append(_idle().is_set())
            await asyncio.sleep(0)
            results.append(_idle().is_set())

    await asyncio.gather(_enter_and_sample(), _enter_and_sample())
    assert all(r is False for r in results)
    assert _idle().is_set()


# ---------------------------------------------------------------------------
# chat_priority_context: lock exclusion of background workers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bg_vlm_lock_held_inside_chat_context() -> None:
    """_bg_vlm_lock must be locked while inside chat_priority_context."""
    async with utils_mod.chat_priority_context():
        assert _vlm_lock().locked()


@pytest.mark.asyncio
async def test_bg_llm_lock_held_inside_chat_context() -> None:
    """_bg_llm_lock must be locked while inside chat_priority_context."""
    async with utils_mod.chat_priority_context():
        assert _llm_lock().locked()


@pytest.mark.asyncio
async def test_bg_locks_released_after_chat_context() -> None:
    """Both background locks must be released after chat_priority_context exits."""
    async with utils_mod.chat_priority_context():
        pass
    assert not _vlm_lock().locked()
    assert not _llm_lock().locked()


# ---------------------------------------------------------------------------
# generate_embeddings: TOCTOU-safe gate (waits + holds _bg_llm_lock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embeddings_waits_while_chat_active() -> None:
    """Ollama generate_embeddings must not proceed until _chat_idle is set."""
    _idle().clear()  # simulate active chat

    embed_started = asyncio.Event()
    embed_result: list[object] = []

    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    async def _run_embed() -> None:
        embed_started.set()
        result = await utils_mod.generate_embeddings(mock_emb, ["hello"])
        embed_result.append(result)

    task = asyncio.create_task(_run_embed())
    await embed_started.wait()
    await asyncio.sleep(0.01)
    assert not embed_result, "generate_embeddings should block while chat is active"

    _idle().set()
    await asyncio.wait_for(task, timeout=2.0)
    assert embed_result, "generate_embeddings should complete after chat gate released"


@pytest.mark.asyncio
async def test_generate_embeddings_proceeds_when_idle() -> None:
    """Ollama generate_embeddings completes immediately when no chat is active."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    result = await utils_mod.generate_embeddings(mock_emb, ["hello"])
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_generate_embeddings_holds_bg_llm_lock_during_call() -> None:
    """Ollama generate_embeddings must hold _bg_llm_lock while calling aembed_documents."""
    lock_held_during_call = False

    async def _check_lock(_texts: list[str], **_: object) -> list[list[float]]:
        nonlocal lock_held_during_call
        lock_held_during_call = _llm_lock().locked()
        return [[0.1]]

    mock_emb = MagicMock()
    mock_emb.aembed_documents = _check_lock
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    await utils_mod.generate_embeddings(mock_emb, ["hello"])
    assert lock_held_during_call, "_bg_llm_lock must be held during aembed_documents"


@pytest.mark.asyncio
async def test_chat_blocks_embeddings_via_bg_llm_lock() -> None:
    """chat_priority_context must block Ollama generate_embeddings via _bg_llm_lock."""
    embed_acquired_lock: list[bool] = []

    async def _check_lock(_texts: list[str], **_: object) -> list[list[float]]:
        embed_acquired_lock.append(True)
        return [[0.1]]

    mock_emb = MagicMock()
    mock_emb.aembed_documents = _check_lock
    mock_emb.__class__ = OllamaEmbeddings  # type: ignore[assignment]

    async with utils_mod.chat_priority_context():
        embed_task = asyncio.create_task(
            utils_mod.generate_embeddings(mock_emb, ["hello"])
        )
        await asyncio.sleep(0.01)
        assert not embed_acquired_lock, (
            "generate_embeddings should not start while chat holds _bg_llm_lock"
        )

    await asyncio.wait_for(embed_task, timeout=2.0)
    assert embed_acquired_lock


@pytest.mark.asyncio
async def test_generate_embeddings_empty_texts_returns_empty() -> None:
    """generate_embeddings with empty input returns [] without hitting the model."""
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[])

    result = await utils_mod.generate_embeddings(mock_emb, [])
    mock_emb.aembed_documents.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# generate_embeddings: non-Ollama providers bypass the GPU gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embeddings_non_ollama_bypasses_gate() -> None:
    """Non-Ollama providers must not wait on _chat_idle or acquire _bg_llm_lock."""
    _idle().clear()  # simulate active chat

    embed_result: list[object] = []

    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.5, 0.6]])
    # Leave __class__ as MagicMock — not OllamaEmbeddings, so gate is bypassed.

    async def _run_embed() -> None:
        result = await utils_mod.generate_embeddings(mock_emb, ["hello"])
        embed_result.append(result)

    task = asyncio.create_task(_run_embed())
    await asyncio.sleep(0.01)
    # Should have completed immediately despite chat being "active".
    assert embed_result, "non-Ollama embeddings must not block on _chat_idle"
    assert not _llm_lock().locked(), (
        "_bg_llm_lock must not be held after non-Ollama call"
    )
    await task


# ---------------------------------------------------------------------------
# chat_priority_context: cancellation safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_priority_context_cleanup_on_cancellation_before_lock() -> None:
    """_chat_idle and _chat_active_count must be reset if cancelled waiting for a lock."""
    # Hold _bg_vlm_lock so chat_priority_context blocks waiting to acquire it.
    async with _vlm_lock():
        ctx_task = asyncio.create_task(_enter_chat_context())
        # Give the task time to increment the count and block on the lock.
        await asyncio.sleep(0.01)
        assert not _idle().is_set(), "_chat_idle should be cleared"
        assert utils_mod._chat_active_count == 1  # type: ignore[attr-defined]

        ctx_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ctx_task

    # After cancellation _chat_idle must be restored and the count reset.
    assert _idle().is_set(), "_chat_idle must be set after cancellation"
    assert utils_mod._chat_active_count == 0  # type: ignore[attr-defined]


async def _enter_chat_context() -> None:
    """Enter chat_priority_context and hold until cancelled."""
    async with utils_mod.chat_priority_context():
        await asyncio.sleep(9999)
