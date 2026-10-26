# ruff: noqa: S101
"""
Tests for the memory store's tables and vector size (#760).

The store's vector column keeps the size it was created with. A non-Ollama
provider of another size fails every embedding write with only a pgvector
error in the log, so a repair issue must name it; a size that cannot be told
must never raise or clear that issue.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import psycopg
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.agent.graph import _search_memories
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import embedding_store
from custom_components.home_generative_agent.core.embedding_store import (
    ISSUE_EMBEDDING_DIMS_MISMATCH,
    EmbeddingProvider,
    UnavailableEmbeddings,
    _issue_id,
    async_clear_embedding_size_issue,
    async_forget_migrations,
    async_store_tables,
    async_sync_embedding_size_issue,
    embedding_size_check_provider,
)

from .test_database_guard import _setup_without_database

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

POOL: Any = object()


def _issue(hass: Any, entry_id: str = "entry1") -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _issue_id(entry_id))


def _loaded_entry(hass: Any, state: ConfigEntryState = ConfigEntryState.LOADED) -> None:
    MockConfigEntry(domain=DOMAIN, entry_id="entry1", state=state).add_to_hass(hass)


def _store_size(monkeypatch: pytest.MonkeyPatch, size: int | None) -> None:
    async def _dims(_pool: Any, *, default: int | None) -> int | None:
        assert default is None
        return size

    monkeypatch.setattr(embedding_store, "async_store_vector_dims", _dims)


class _Embeddings:
    """An OpenAI-compatible client returning `size`-long vectors, or raising."""

    def __init__(self, size: int | None) -> None:
        self.size = size
        self.calls = 0

    async def aembed_query(self, _text: str) -> list[float]:
        self.calls += 1
        if self.size is None:
            msg = "server down"
            raise ConnectionError(msg)
        return [0.0] * self.size


def _compat(size: int | None, name: str = "Local vLLM") -> EmbeddingProvider:
    return EmbeddingProvider(name, "openai_compatible", _Embeddings(size))


def _openai(name: str = "OpenAI") -> EmbeddingProvider:
    return EmbeddingProvider(name, "openai", object())


def _raise(hass: Any) -> None:
    ir.async_create_issue(
        hass,
        DOMAIN,
        _issue_id("entry1"),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_EMBEDDING_DIMS_MISMATCH,
        translation_placeholders={"providers": "x", "dims": "1", "expected": "2"},
    )


async def _sync(
    hass: Any, provider: EmbeddingProvider | None, pool: Any = POOL
) -> None:
    await async_sync_embedding_size_issue(hass, "entry1", provider=provider, pool=pool)


@pytest.mark.asyncio
async def test_a_provider_of_another_size_raises_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A store created at 768 by an OpenAI-compatible primary, now OpenAI's 1024."""
    _loaded_entry(hass)
    _store_size(monkeypatch, 768)

    await _sync(hass, _openai())

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "OpenAI",
        "dims": "1024",
        "expected": "768",
        "default_dims": "1024",
    }
    assert "produces 1024-dimension vectors" in caplog.text


@pytest.mark.asyncio
async def test_a_measured_right_size_clears_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI-compatible server is measured with one embedding."""
    _loaded_entry(hass)
    _store_size(monkeypatch, 1024)
    _raise(hass)
    compat = _compat(1024)

    await _sync(hass, compat)

    assert _issue(hass) is None
    assert cast("_Embeddings", compat.embeddings).calls == 1


@pytest.mark.asyncio
async def test_an_unmeasured_provider_leaves_the_issue_alone(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that failed is neither right nor wrong."""
    _loaded_entry(hass)
    _store_size(monkeypatch, 1024)
    _raise(hass)

    await _sync(hass, _compat(None))

    assert _issue(hass) is not None


@pytest.mark.asyncio
async def test_an_unknown_store_size_decides_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No vector table, or a database that cannot be asked."""
    _loaded_entry(hass)
    _store_size(monkeypatch, None)

    await _sync(hass, _openai())
    assert _issue(hass) is None

    _raise(hass)
    await _sync(hass, _compat(1024))
    assert _issue(hass) is not None


@pytest.mark.asyncio
async def test_no_providers_or_no_database_clears_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embeddings on Ollama only, or no database to use them."""
    _loaded_entry(hass)
    _store_size(monkeypatch, 768)
    _raise(hass)
    await _sync(hass, None)
    assert _issue(hass) is None

    _raise(hass)
    await _sync(hass, _openai(), pool=None)
    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_no_issue_for_an_entry_that_is_not_running(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check whose reply lands after the unload must not raise the issue."""
    _loaded_entry(hass, ConfigEntryState.NOT_LOADED)
    _store_size(monkeypatch, 768)

    await _sync(hass, _openai())

    assert _issue(hass) is None


@pytest.mark.asyncio
async def test_unload_drops_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry's unload hook clears the issue."""
    entry = await _setup_without_database(hass, monkeypatch)
    ir.async_create_issue(
        hass,
        DOMAIN,
        _issue_id(entry.entry_id),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_EMBEDDING_DIMS_MISMATCH,
    )

    assert await cast("Any", hga_component).async_unload_entry(hass, entry)
    assert _issue(hass, entry.entry_id) is None


@pytest.mark.asyncio
async def test_removal_drops_the_issue(hass: Any) -> None:
    """async_remove_entry clears what unload would."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    entry.add_to_hass(hass)
    _raise(hass)

    await cast("Any", hga_component).async_remove_entry(hass, entry)

    assert _issue(hass) is None


def test_clear_drops_the_issue(hass: Any) -> None:
    """The callback the unload and removal hooks use."""
    _raise(hass)

    async_clear_embedding_size_issue(hass, "entry1")

    assert _issue(hass) is None


# ----- table reads -----


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]], error: Exception | None) -> None:
        self.rows = rows
        self.error = error
        self.sql: list[tuple[str, Any]] = []

    async def execute(self, sql: str, params: Any = None) -> None:
        if self.error is not None:
            raise self.error
        self.sql.append((sql, params))

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _Pool:
    def __init__(
        self, rows: list[dict[str, Any]] | None = None, error: Exception | None = None
    ) -> None:
        self.cursor = _Cursor(rows or [], error)

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncGenerator[Any]:
        cursor = self.cursor

        class _Conn:
            @contextlib.asynccontextmanager
            async def cursor(self) -> AsyncGenerator[_Cursor]:
                yield cursor

        yield _Conn()


@pytest.mark.asyncio
async def test_store_tables_reads_the_tables_present() -> None:
    """Every table the bootstrap depends on is asked for in one query."""
    pool = _Pool([{"name": "store"}, {"name": "checkpoints"}])

    assert await async_store_tables(cast("Any", pool)) == {"store", "checkpoints"}
    (_, params), *_ = pool.cursor.sql
    assert set(params["tables"]) == {
        "store",
        "store_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
        "store_vectors",
        "vector_migrations",
    }


@pytest.mark.asyncio
async def test_store_tables_is_unknown_on_a_database_error() -> None:
    """A failed read is not an empty database."""
    pool = _Pool(error=psycopg.OperationalError("gone"))

    assert await async_store_tables(cast("Any", pool)) is None


@pytest.mark.parametrize(
    ("tables", "deletes"),
    [
        # Setup then runs them all again: extension, table, index.
        (
            {"store", "store_migrations", "vector_migrations"},
            ['DELETE FROM "vector_migrations"'],
        ),
        # A record table that is gone too is left for setup to create.
        ({"store", "store_migrations"}, []),
        # Nothing it created is gone: its record stays.
        ({"store_vectors", "vector_migrations"}, []),
    ],
)
@pytest.mark.asyncio
async def test_only_records_of_missing_tables_are_forgotten(
    tables: set[str], deletes: list[str]
) -> None:
    """A record is cleared only when a table its migrations create is gone."""
    pool = _Pool()

    await async_forget_migrations(
        cast("Any", pool), frozenset(tables), ("vector_migrations",)
    )

    assert [cast("Any", q).as_string() for q, _ in pool.cursor.sql] == deletes


# ----- which providers are measured -----


def _config(entry_id: str, provider_type: str) -> Any:
    return SimpleNamespace(
        entry_id=entry_id, name=entry_id, provider_type=provider_type
    )


def test_only_a_non_ollama_primary_is_measured() -> None:
    """Ollama has its own check; a primary whose client is missing stays."""
    compat = _config("compat", "openai_compatible")

    provider = embedding_size_check_provider(
        compat, primary_type="openai_compatible", primary_embeddings=None
    )
    assert provider == EmbeddingProvider("compat", "openai_compatible", None)
    assert (
        embedding_size_check_provider(
            _config("ollama", "ollama"),
            primary_type="ollama",
            primary_embeddings=object(),
        )
        is None
    )


def test_without_a_chain_the_primary_is_named_by_its_type() -> None:
    """The raw option key never reaches the issue text; no client, no check."""
    provider = embedding_size_check_provider(
        None, primary_type="openai_compatible", primary_embeddings=object()
    )

    assert provider is not None
    assert provider.name == "OpenAI Compatible"
    for primary_type, client in (("ollama", object()), ("openai", None)):
        assert (
            embedding_size_check_provider(
                None, primary_type=primary_type, primary_embeddings=client
            )
            is None
        )


@pytest.mark.asyncio
async def test_a_known_size_needs_no_client(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini down at boot still raises: its size is fixed, not probed."""
    _loaded_entry(hass)
    _store_size(monkeypatch, 768)

    await _sync(hass, EmbeddingProvider("Gemini", "gemini", None))

    issue = cast("ir.IssueEntry", _issue(hass))
    assert (issue.translation_placeholders or {})["providers"] == "Gemini"


@pytest.mark.asyncio
async def test_memory_search_without_embeddings_returns_recent_memories(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stand-in fails at once; search falls back to recency, quietly."""
    calls: list[Any] = []

    class _Store:
        async def asearch(self, _namespace: Any, **kwargs: Any) -> list[str]:
            calls.append(kwargs.get("query"))
            if kwargs.get("query") is not None:
                await UnavailableEmbeddings("Ollama").aembed_query("q")
            return ["recent"]

    result = await _search_memories(cast("Any", _Store()), "user", "where are my keys")

    assert result == ["recent"]
    assert calls == ["where are my keys", None]
    assert "Unexpected memory store search failure" not in caplog.text
