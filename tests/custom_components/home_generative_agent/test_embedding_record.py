# ruff: noqa: S101
"""
Tests for the memory store's embedding model record (#855).

The size check (#760) misses a model change that keeps the size, so the store
records the model that filled it. A store with no record is taken to be the
current model's; another model raises a fixable issue whose fix re-embeds the
store and resumes after a restart; an Ollama fallback holding another build of
the model is left out of the chain. An unknown digest decides nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import psycopg
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent import repairs
from custom_components.home_generative_agent.const import (
    CONF_EMBEDDING_MODEL_PROVIDER,
    CONF_OLLAMA_EMBEDDING_MODEL,
    DOMAIN,
)
from custom_components.home_generative_agent.core import embedding_record as er
from custom_components.home_generative_agent.core import utils
from custom_components.home_generative_agent.core.embedding_record import (
    ISSUE_EMBEDDING_FALLBACK_HELD,
    ISSUE_EMBEDDING_FALLBACK_HELD_NO_REFERENCE,
    ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD,
    ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD,
    ISSUE_EMBEDDING_MODEL_CHANGED,
    EmbeddingFallbackGate,
    EmbeddingFingerprint,
    FallbackBuild,
    embedding_fingerprint,
    text_at_field,
)
from custom_components.home_generative_agent.core.embedding_store import (
    EmbeddingsUnavailableError,
    UnavailableEmbeddings,
)
from custom_components.home_generative_agent.core.fallback import FallbackEmbeddings
from custom_components.home_generative_agent.core.utils import configured_ollama_urls

from .test_fallback_setup import (
    FakeEmbeddings,
    _capture_recheck,
    _fallback_setup_data,
    _patch_setup_dependencies,
    _register_capture_service,
    _second_ollama,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

OLLAMA_A = embedding_fingerprint(
    "ollama", "mxbai-embed-large", dims=1024, digest="w:a1"
)
OPENAI = embedding_fingerprint(
    "openai", "text-embedding-3-small", dims=1024, digest=None
)


# --- The fingerprint -------------------------------------------------------


def test_an_unknown_digest_never_stands_in_for_another_build() -> None:
    """Same tag: equal unless both digests are known and differ."""
    no_digest = embedding_fingerprint(
        "ollama", "mxbai-embed-large:latest", dims=1024, digest=None
    )
    other_build = embedding_fingerprint(
        "ollama", "mxbai-embed-large", dims=1024, digest="w:b2"
    )

    assert OLLAMA_A.same_model(no_digest)
    assert no_digest.same_model(OLLAMA_A)
    assert not OLLAMA_A.same_model(other_build)
    assert not OLLAMA_A.same_model(OPENAI)


def test_an_ollama_name_is_spelled_as_ollama_resolves_it() -> None:
    """An untagged name and its :latest tag are the same model."""
    assert OLLAMA_A.model == "mxbai-embed-large:latest"
    assert embedding_fingerprint("openai", " m ", dims=None, digest="").model == "m"


def test_a_stored_fingerprint_round_trips_and_junk_reads_as_none() -> None:
    """Anything that is not a fingerprint reads as no record."""
    assert EmbeddingFingerprint.from_json(OLLAMA_A.as_json().obj) == OLLAMA_A
    assert EmbeddingFingerprint.from_json(None) is None
    assert EmbeddingFingerprint.from_json({"model": "m"}) is None
    assert EmbeddingFingerprint.from_json(
        {"provider_type": "openai", "model": "m", "dims": "x", "digest": ""}
    ) == EmbeddingFingerprint("openai", "m")


def test_the_label_names_the_provider_model_and_short_digest() -> None:
    """A user reads the label in the issue."""
    digest = "0123456789abcdef"
    fingerprint = embedding_fingerprint("ollama", "m", dims=None, digest=digest)

    assert fingerprint.label == "Ollama m:latest (0123456789ab)"
    assert OPENAI.label == "OpenAI text-embedding-3-small"


def test_the_text_a_vector_came_from_is_read_back() -> None:
    """The field path langgraph embedded, with .<n> for one of several texts."""
    assert text_at_field({"content": "hello"}, "content") == "hello"
    assert text_at_field({"a": {"b": "deep"}}, "a.b") == "deep"
    assert text_at_field({"content": ["one", "two"]}, "content[*].1") == "two"
    assert text_at_field({"x": {"a": "A", "b": "B"}}, "x.{a,b}.0") == "A"
    assert text_at_field({"content": ["one"]}, "content[*].5") is None
    assert text_at_field({"content": ""}, "content") is None
    assert text_at_field({}, "content") is None


# --- The record's decision -------------------------------------------------

OPENAI_B = embedding_fingerprint(
    "openai", "text-embedding-3-large", dims=1024, digest=None
)
OLLAMA_UNREAD = embedding_fingerprint(
    "ollama", "mxbai-embed-large", dims=1024, digest=None
)
OLLAMA_B = embedding_fingerprint(
    "ollama", "mxbai-embed-large", dims=1024, digest="w:b2"
)
CURSOR = ["memories.u1", "k001", "content"]


@pytest.mark.parametrize(
    ("record", "current", "expected"),
    [
        # Same model: nothing to do.
        (er._Record(OLLAMA_A, None, None), OLLAMA_A, er._Decision.MATCH),
        # Another model, or another build of the same tag.
        (er._Record(OPENAI, None, None), OLLAMA_A, er._Decision.MISMATCH),
        (er._Record(OLLAMA_A, None, None), OLLAMA_B, er._Decision.MISMATCH),
        # A re-embed to the current model was started: carry on.
        (er._Record(OPENAI, OLLAMA_A, CURSOR), OLLAMA_A, er._Decision.RESUME),
        (er._Record(OPENAI, OLLAMA_A, None), OLLAMA_A, er._Decision.RESUME),
        # Back on the recorded model before the re-embed did anything.
        (er._Record(OLLAMA_A, OPENAI, None), OLLAMA_A, er._Decision.MATCH),
        # Back on it after part of the store was re-embedded: still mixed.
        (er._Record(OLLAMA_A, OPENAI, CURSOR), OLLAMA_A, er._Decision.MISMATCH),
        # An unread build never decides while the record names one.
        (er._Record(OLLAMA_A, None, None), OLLAMA_UNREAD, er._Decision.WAIT),
        (er._Record(OPENAI, OLLAMA_B, CURSOR), OLLAMA_UNREAD, er._Decision.WAIT),
        (er._Record(OPENAI, None, None, (OLLAMA_B,)), OLLAMA_UNREAD, er._Decision.WAIT),
        # ...but a record that never had a build matches by name.
        (er._Record(OLLAMA_UNREAD, None, None), OLLAMA_UNREAD, er._Decision.MATCH),
        # ...and another model altogether is still another model.
        (er._Record(OPENAI, None, None), OLLAMA_UNREAD, er._Decision.MISMATCH),
        (er._Record(OPENAI, None, None), OPENAI_B, er._Decision.MISMATCH),
    ],
)
def test_the_record_decides_what_the_current_model_means(
    record: er._Record, current: EmbeddingFingerprint, expected: er._Decision
) -> None:
    """Every state of the record against the current model, by explicit row."""
    assert er._decide(record, current) is expected


def test_a_partly_reembedded_store_is_named_as_holding_both() -> None:
    """The issue names both models while a re-embed finished part of the store."""
    assert er._Record(OLLAMA_A, OPENAI, CURSOR).held_label == (
        f"{OLLAMA_A.label} + {OPENAI.label}"
    )
    assert er._Record(OLLAMA_A, OPENAI, None).held_label == OLLAMA_A.label


def test_a_gemini_model_is_named_with_or_without_its_prefix() -> None:
    """Gemini takes "models/x" and "x" for the same model."""
    assert embedding_fingerprint(
        "gemini", "models/gemini-embedding-001", dims=None, digest=None
    ) == embedding_fingerprint("gemini", "gemini-embedding-001", dims=None, digest=None)


# --- The record at setup ---------------------------------------------------


class _Cursor:
    """Answer the record's queries from a script; record what ran."""

    def __init__(
        self,
        *,
        record: dict[str, Any] | None,
        has_table: bool = True,
        has_vectors: bool = True,
    ) -> None:
        self.record = record
        self.has_table = has_table
        self.has_vectors = has_vectors
        self.executed: list[tuple[str, Any]] = []
        self._last = ""

    async def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((query, params))
        self._last = query

    async def fetchone(self) -> dict[str, Any] | None:
        if self._last in (er._READ_RECORD_SQL, er._READ_RECORD_FOR_UPDATE_SQL):
            return self.record
        if self._last == er._HAS_VECTOR_TABLE_SQL:
            return {"has_table": self.has_table}
        if self._last == er._HAS_VECTORS_SQL:
            return {"has_vectors": self.has_vectors}
        return None

    def ran(self, query: str) -> list[Any]:
        return [params for q, params in self.executed if q == query]


class _Pool:
    def __init__(self, cursor: _Cursor) -> None:
        self.cur = cursor

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncGenerator[Any]:
        @contextlib.asynccontextmanager
        async def _cursor() -> AsyncGenerator[_Cursor]:
            yield self.cur

        @contextlib.asynccontextmanager
        async def _transaction() -> AsyncGenerator[None]:
            self.cur.executed.append(("BEGIN", None))
            yield
            self.cur.executed.append(("COMMIT", None))

        yield SimpleNamespace(cursor=_cursor, transaction=_transaction)


def _row(
    model: EmbeddingFingerprint | None,
    target: EmbeddingFingerprint | None = None,
    cursor: Any = None,
    mixed: tuple[EmbeddingFingerprint, ...] = (),
) -> dict[str, Any]:
    return {
        "model": model.as_json().obj if model else None,
        "reembed_target": target.as_json().obj if target else None,
        "reembed_cursor": cursor,
        "mixed_with": [m.as_json().obj for m in mixed] or None,
    }


def _written(cur: _Cursor) -> list[EmbeddingFingerprint | None]:
    return [
        EmbeddingFingerprint.from_json(p["model"].obj)
        for p in cur.ran(er._WRITE_RECORD_SQL)
    ]


async def _check(cur: _Cursor, current: EmbeddingFingerprint) -> Any:
    return await er._async_check_record(cast("Any", _Pool(cur)), current)


@pytest.mark.asyncio
async def test_a_store_without_a_record_records_the_current_model() -> None:
    """Upgrading installs adopt the current model: no issue for an unmixed store."""
    cur = _Cursor(record=None)

    assert await _check(cur, OLLAMA_A) == (er._Decision.MATCH, None)
    assert _written(cur) == [OLLAMA_A]


@pytest.mark.asyncio
async def test_a_store_with_no_vectors_records_the_current_model() -> None:
    """A store emptied (the documented drop and reload) follows the primary."""
    cur = _Cursor(record=_row(OPENAI), has_vectors=False)

    assert await _check(cur, OLLAMA_A) == (er._Decision.MATCH, None)
    assert _written(cur) == [OLLAMA_A]


@pytest.mark.asyncio
async def test_an_unreadable_record_is_recorded_again_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A record row whose model cannot be read is replaced, and says so."""
    cur = _Cursor(record={"model": {"junk": 1}, "reembed_target": None})
    cur.record = {**cast("dict[str, Any]", cur.record), "reembed_cursor": None}

    assert await _check(cur, OLLAMA_A) == (er._Decision.MATCH, None)
    assert _written(cur) == [OLLAMA_A]
    assert "could not be read" in caplog.text


@pytest.mark.asyncio
async def test_no_vector_table_records_nothing() -> None:
    """Without a vector table there is no store to record a model for."""
    cur = _Cursor(record=None, has_table=False)

    assert await _check(cur, OLLAMA_A) == (er._Decision.MATCH, None)
    assert _written(cur) == []
    assert cur.ran(er._HAS_VECTORS_SQL) == []


@pytest.mark.asyncio
async def test_the_same_model_writes_nothing() -> None:
    """A matching record is left as it is."""
    cur = _Cursor(record=_row(OLLAMA_A))

    decision, _record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MATCH
    assert _written(cur) == []


@pytest.mark.asyncio
async def test_a_digest_the_record_lacks_is_filled_in() -> None:
    """A record made while the server could not say gets the digest later."""
    cur = _Cursor(record=_row(OLLAMA_UNREAD))

    decision, _record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MATCH
    assert _written(cur) == [OLLAMA_A]


@pytest.mark.asyncio
async def test_back_on_the_recorded_model_drops_an_untouched_reembed() -> None:
    """A re-embed to another model that never finished a batch is dropped."""
    cur = _Cursor(record=_row(OLLAMA_A, target=OPENAI, cursor=None))

    decision, _record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MATCH
    assert _written(cur) == [OLLAMA_A]


@pytest.mark.asyncio
async def test_back_on_the_recorded_model_keeps_a_partial_reembed() -> None:
    """Part of the store holds the other model: it is noted, never recorded clean."""
    cur = _Cursor(record=_row(OLLAMA_A, target=OPENAI, cursor=CURSOR))

    decision, record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MISMATCH
    assert _written(cur) == []
    # The re-embed's target wrote into the store; the place it kept no longer
    # covers what the recorded model writes from now on. The recorded model is
    # noted as a writer too: rows the run already did now hold it, so the run
    # must not finish on the writers it started with (#860).
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    assert [EmbeddingFingerprint.from_json(m) for m in marked["mixed"].obj] == [
        OPENAI,
        OLLAMA_A,
    ]
    assert marked["cursor"] is None
    assert record.held_label == f"{OLLAMA_A.label} + {OPENAI.label}"


@pytest.mark.asyncio
async def test_a_mismatch_notes_that_the_current_model_writes_now() -> None:
    """Switching back later must not certify what the other model wrote."""
    cur = _Cursor(record=_row(OPENAI))

    decision, _record = await _check(cur, OLLAMA_A)
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    switched_back = _Cursor(record=_row(OPENAI, mixed=(OLLAMA_A,)))
    back, back_record = await _check(switched_back, OPENAI)

    assert decision is er._Decision.MISMATCH
    assert [EmbeddingFingerprint.from_json(m) for m in marked["mixed"].obj] == [
        OLLAMA_A
    ]
    assert back is er._Decision.MISMATCH
    assert back_record.held_label == f"{OPENAI.label} + {OLLAMA_A.label}"
    assert _written(switched_back) == []


@pytest.mark.asyncio
async def test_a_writer_already_noted_is_not_noted_again() -> None:
    """A second boot on the same model leaves the row as it is."""
    cur = _Cursor(record=_row(OPENAI, mixed=(OLLAMA_A,)))

    await _check(cur, OLLAMA_A)

    assert cur.ran(er._MARK_MIXED_SQL) == []


@pytest.mark.asyncio
async def test_the_record_is_read_and_written_under_a_row_lock() -> None:
    """A run's batch commit and a setup check's write never interleave."""
    cur = _Cursor(record=_row(OPENAI))

    await _check(cur, OLLAMA_A)

    names = [q for q, _ in cur.executed]
    assert names.index("BEGIN") < names.index(er._READ_RECORD_FOR_UPDATE_SQL)
    assert names.index(er._READ_RECORD_FOR_UPDATE_SQL) < names.index(er._MARK_MIXED_SQL)
    assert names.index(er._MARK_MIXED_SQL) < names.index("COMMIT")


@pytest.mark.asyncio
async def test_a_digest_fill_in_is_written_under_the_row_lock() -> None:
    """The record write a MATCH makes lands before the lock is released."""
    cur = _Cursor(record=_row(OLLAMA_UNREAD))

    await _check(cur, OLLAMA_A)

    names = [q for q, _ in cur.executed]
    assert names.index(er._WRITE_RECORD_SQL) < names.index("COMMIT")


@pytest.mark.asyncio
async def test_an_unread_build_records_nothing_for_an_unrecorded_store() -> None:
    """A fallback may be writing: no build is certified until one is read."""
    cur = _Cursor(record=None)

    assert await _check(cur, OLLAMA_UNREAD) == (er._Decision.WAIT, None)
    assert _written(cur) == []


@pytest.mark.asyncio
async def test_a_mismatch_mid_reembed_to_a_third_model_resets_its_place() -> None:
    """The third model's run loses its place; every model that wrote is named."""
    cur = _Cursor(record=_row(OPENAI, target=OPENAI_B, cursor=CURSOR))

    decision, record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MISMATCH
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    assert [EmbeddingFingerprint.from_json(m) for m in marked["mixed"].obj] == [
        OPENAI_B,
        OLLAMA_A,
    ]
    assert marked["cursor"] is None
    assert record.held_label == f"{OPENAI.label} + {OPENAI_B.label} + {OLLAMA_A.label}"


@pytest.mark.asyncio
async def test_a_create_table_race_is_harmless() -> None:
    """Two entries starting together both create the table; the loser goes on."""
    cur = _Cursor(record=None)
    real = cur.execute

    async def _execute(query: str, params: Any = None) -> None:
        if query == er._CREATE_RECORD_SQL:
            msg = "pg_type_typname_nsp_index"
            raise psycopg.errors.UniqueViolation(msg)
        await real(query, params)

    cur.execute = _execute  # type: ignore[method-assign]

    assert await _check(cur, OLLAMA_A) == (er._Decision.MATCH, None)
    assert _written(cur) == [OLLAMA_A]


@pytest.mark.asyncio
async def test_an_unread_build_writes_nothing() -> None:
    """The recorded build and a re-embed in flight survive a server that is down."""
    cur = _Cursor(record=_row(OPENAI, target=OLLAMA_B, cursor=CURSOR))

    decision, _record = await _check(cur, OLLAMA_UNREAD)

    assert decision is er._Decision.WAIT
    assert _written(cur) == []


@pytest.mark.asyncio
async def test_a_malformed_cursor_reads_as_no_cursor() -> None:
    """A cursor that is not three strings starts the run over."""
    cur = _Cursor(record=_row(OPENAI, target=OLLAMA_A, cursor=["p", "k"]))

    _decision, record = await _check(cur, OLLAMA_A)

    assert record.cursor is None


@pytest.mark.asyncio
async def test_another_model_returns_the_record() -> None:
    """A different model is reported, not overwritten."""
    cur = _Cursor(record=_row(OPENAI))

    decision, record = await _check(cur, OLLAMA_A)

    assert decision is er._Decision.MISMATCH
    assert record.model == OPENAI
    assert _written(cur) == []


_REAL_CHECK = er._async_check_record
_REAL_SLEEP = asyncio.sleep
OLLAMA_SOURCE = ("http://ollama", "mxbai-embed-large")


def _issue(hass: Any, key: str, entry_id: str = "entry1") -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, f"{key}_{entry_id}")


def _entry(  # noqa: PLR0913
    hass: Any,
    *,
    record: er._Record | Exception | None,
    fingerprint: EmbeddingFingerprint | None = OLLAMA_A,
    embeddings: Any = None,
    state: ConfigEntryState = ConfigEntryState.LOADED,
    monkeypatch: pytest.MonkeyPatch,
) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1", state=state)
    entry.add_to_hass(hass)
    client = embeddings if embeddings is not None else FakeEmbeddings("e")
    entry.runtime_data = SimpleNamespace(
        pool=object(),
        embedding_fingerprint=fingerprint,
        embeddings=client,
        primary_embeddings=client,
        embedding_digest_source=None,
        embedding_fallback_gate=None,
        reembed_task=None,
    )

    async def _checked(_pool: Any, current: Any) -> Any:
        if isinstance(record, Exception):
            raise record
        if record is None or record.model is None:
            return er._Decision.MATCH, None
        return er._decide(record, current), record

    monkeypatch.setattr(er, "_async_check_record", _checked)
    return entry


def _no_task(monkeypatch: pytest.MonkeyPatch, entry: Any) -> list[str]:
    """Record the tasks setup would start, closing their coroutines unrun."""
    names: list[str] = []

    def _create(_hass: Any, coro: Any, **kwargs: Any) -> Any:
        coro.close()
        names.append(kwargs.get("name", ""))
        return asyncio.get_running_loop().create_future()

    monkeypatch.setattr(entry, "async_create_background_task", _create)
    return names


def _capture_starts(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    started: list[bool] = []

    def _start(_h: Any, _e: Any, *, start: bool) -> bool:
        started.append(start)
        return True

    monkeypatch.setattr(er, "_start_reembed_task", _start)
    return started


@pytest.mark.asyncio
async def test_another_model_raises_a_fixable_issue_naming_both(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issue names the recorded and current model and carries the entry."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)

    await er.async_sync_embedding_record(hass, entry)

    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.is_fixable
    assert issue.translation_placeholders == {
        "recorded": OPENAI.label,
        "current": OLLAMA_A.label,
    }
    assert issue.data == {"entry_id": "entry1"}


@pytest.mark.asyncio
async def test_a_partial_reembed_left_behind_names_both_models(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching back mid re-embed asks to re-embed the rest back."""
    entry = _entry(
        hass, record=er._Record(OLLAMA_A, OPENAI, CURSOR), monkeypatch=monkeypatch
    )

    await er.async_sync_embedding_record(hass, entry)

    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.translation_placeholders == {
        "recorded": f"{OLLAMA_A.label} + {OPENAI.label}",
        "current": OLLAMA_A.label,
    }


@pytest.mark.asyncio
async def test_a_match_clears_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back on the recorded model, or a fresh record: the issue goes."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)
    await er.async_sync_embedding_record(hass, entry)
    monkeypatch.setattr(
        er, "_async_check_record", AsyncMock(return_value=(er._Decision.MATCH, None))
    )

    await er.async_sync_embedding_record(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


@pytest.mark.asyncio
async def test_a_started_reembed_to_the_current_model_resumes(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart mid re-embed picks it up where it stopped, with no issue."""
    entry = _entry(
        hass, record=er._Record(OPENAI, OLLAMA_A, CURSOR), monkeypatch=monkeypatch
    )
    started = _capture_starts(monkeypatch)

    await er.async_sync_embedding_record(hass, entry)

    assert started == [False]
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


@pytest.mark.asyncio
async def test_an_unread_build_leaves_the_issue_and_starts_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A primary whose build was not read neither clears nor resumes anything."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)
    await er.async_sync_embedding_record(hass, entry)
    started = _capture_starts(monkeypatch)
    entry.runtime_data.embedding_fingerprint = OLLAMA_UNREAD
    monkeypatch.setattr(
        er,
        "_async_check_record",
        AsyncMock(return_value=(er._Decision.WAIT, er._Record(OLLAMA_A, None, None))),
    )

    await er.async_sync_embedding_record(hass, entry)

    assert started == []
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None


@pytest.mark.asyncio
async def test_a_reembed_to_yet_another_model_is_not_resumed(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model changed again mid re-embed asks again."""
    entry = _entry(
        hass, record=er._Record(OPENAI, OPENAI_B, CURSOR), monkeypatch=monkeypatch
    )
    started = _capture_starts(monkeypatch)

    await er.async_sync_embedding_record(hass, entry)

    assert started == []
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None


@pytest.mark.asyncio
async def test_a_database_error_decides_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable record leaves the issue as it is, through every retry."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)
    await er.async_sync_embedding_record(hass, entry)
    check = AsyncMock(side_effect=psycopg.OperationalError("x"))
    monkeypatch.setattr(er, "_async_check_record", check)
    waits = _fake_sleep(monkeypatch)

    await er.async_sync_embedding_record(hass, entry)

    assert waits == list(er._RECORD_RETRY_DELAYS_S)
    assert check.await_count == 1 + len(er._RECORD_RETRY_DELAYS_S)
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None
    assert "Could not read which embedding model" in caplog.text
    assert "Reload Home Generative Agent once the database answers" in caplog.text


def _fake_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    return waits


@pytest.mark.asyncio
async def test_a_record_check_that_fails_once_lands_on_the_retry(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database timed out at setup: the retry still notes the new model (#857)."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)
    real = er._async_check_record
    check = AsyncMock(
        side_effect=[TimeoutError(), await real(cast("Any", None), OLLAMA_A)]
    )
    monkeypatch.setattr(er, "_async_check_record", check)
    waits = _fake_sleep(monkeypatch)

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [er._RECORD_RETRY_DELAYS_S[0]]
    assert check.await_count == 2
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None


@pytest.mark.asyncio
async def test_an_unload_during_the_record_retry_wait_stops_it(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry unloaded while the retry waits checks nothing more."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    check = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(er, "_async_check_record", check)

    async def _sleep(_delay: float) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)

    await er.async_sync_embedding_record(hass, entry)

    assert check.await_count == 1
    assert "Reload Home Generative Agent once the database answers" not in caplog.text


@pytest.mark.asyncio
async def test_a_reply_after_the_unload_raises_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check's reply can land after the unload: it must not raise the issue."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)

    async def _unloaded_meanwhile(_pool: Any, _current: Any) -> Any:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
        return er._Decision.MISMATCH, er._Record(OPENAI, None, None)

    monkeypatch.setattr(er, "_async_check_record", _unloaded_meanwhile)

    await er.async_sync_embedding_record(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


@pytest.mark.asyncio
async def test_no_issue_for_an_entry_that_is_not_running(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that is not loaded checks nothing."""
    entry = _entry(
        hass,
        record=er._Record(OPENAI, None, None),
        state=ConfigEntryState.NOT_LOADED,
        monkeypatch=monkeypatch,
    )

    await er.async_sync_embedding_record(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


@pytest.mark.asyncio
async def test_no_current_model_checks_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without embeddings there is nothing to compare."""
    check = AsyncMock()
    entry = _entry(hass, record=None, fingerprint=None, monkeypatch=monkeypatch)
    monkeypatch.setattr(er, "_async_check_record", check)

    await er.async_sync_embedding_record(hass, entry)

    check.assert_not_called()


def _unread_at_setup(
    hass: Any, monkeypatch: pytest.MonkeyPatch, builds: list[str | None]
) -> tuple[MockConfigEntry, AsyncMock, list[float]]:
    """Set up an Ollama primary whose build setup missed, then reads as `builds`."""
    entry = _entry(
        hass, record=None, fingerprint=OLLAMA_UNREAD, monkeypatch=monkeypatch
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(side_effect=builds))
    check = AsyncMock(return_value=(er._Decision.MATCH, None))
    monkeypatch.setattr(er, "_async_check_record", check)
    waits: list[float] = []

    async def _sleep(delay: float) -> None:
        # asyncio.sleep is patched for the whole loop: still yield to it.
        waits.append(delay)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    return entry, check, waits


@pytest.mark.asyncio
async def test_a_build_setup_missed_is_read_again_before_the_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A startup too busy to read the build no longer skips the check."""
    entry, check, waits = _unread_at_setup(hass, monkeypatch, [None, "w:a1"])

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0]
    check.assert_awaited_once_with(entry.runtime_data.pool, OLLAMA_A)
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A


@pytest.mark.asyncio
async def test_a_build_never_read_is_checked_as_unread_and_logged(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After the last try the check runs as before, and the log says so."""
    entry, check, waits = _unread_at_setup(hass, monkeypatch, [None, None, None])

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0, 30.0]
    check.assert_awaited_once_with(entry.runtime_data.pool, OLLAMA_UNREAD)
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_UNREAD
    assert "Could not read which build of mxbai-embed-large" in caplog.text


@pytest.mark.asyncio
async def test_an_unload_while_reading_the_build_checks_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry unloaded during the wait neither reads nor checks again."""
    entry, check, _waits = _unread_at_setup(hass, monkeypatch, [None, "w:a1"])

    async def _unloaded(_delay: float) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _unloaded)

    await er.async_sync_embedding_record(hass, entry)

    assert er.fetch_ollama_model_digest.await_count == 1  # type: ignore[attr-defined]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_UNREAD
    check.assert_not_called()


@pytest.mark.asyncio
async def test_a_primary_down_at_setup_is_not_read_again(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback may be writing: the check waits for the reload, as before."""
    entry, check, waits = _unread_at_setup(hass, monkeypatch, ["w:a1"])
    entry.runtime_data.primary_embeddings = None

    await er.async_sync_embedding_record(hass, entry)

    assert waits == []
    er.fetch_ollama_model_digest.assert_not_called()  # type: ignore[attr-defined]
    check.assert_awaited_once_with(entry.runtime_data.pool, OLLAMA_UNREAD)


@pytest.mark.asyncio
async def test_a_repair_confirmed_during_the_read_keeps_its_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that lands after the repair read the build never replaces it."""
    entry, check, _waits = _unread_at_setup(hass, monkeypatch, [])

    async def _read_while_repaired(*_args: Any, **_kwargs: Any) -> str:
        entry.runtime_data.embedding_fingerprint = OLLAMA_A
        return "w:b2"

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _read_while_repaired)

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    check.assert_awaited_once_with(entry.runtime_data.pool, OLLAMA_A)


@pytest.mark.asyncio
async def test_a_late_build_unlike_the_record_raises_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real record check: the build pulled while down is flagged."""
    entry, _check, _waits = _unread_at_setup(hass, monkeypatch, [None, "w:a1"])
    cur = _Cursor(record=_row(OLLAMA_B))
    entry.runtime_data.pool = _Pool(cur)
    monkeypatch.setattr(er, "_async_check_record", _REAL_CHECK)

    await er.async_sync_embedding_record(hass, entry)

    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    # The record also notes that the current build now writes into the store.
    assert issue.translation_placeholders == {
        "recorded": f"{OLLAMA_B.label} + {OLLAMA_A.label}",
        "current": OLLAMA_A.label,
    }


@pytest.mark.asyncio
async def test_a_repair_confirmed_during_the_wait_keeps_its_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair read the build meanwhile: the check uses that, read once."""
    entry, check, _waits = _unread_at_setup(hass, monkeypatch, [None, "w:a1"])
    starts = _capture_starts(monkeypatch)

    repaired: list[bool] = []

    async def _repair_meanwhile(_delay: float) -> None:
        # asyncio.sleep is patched for everyone: repair on the first wait only.
        if not repaired:
            repaired.append(True)
            await er.async_start_reembed(hass, entry)

    monkeypatch.setattr(er.asyncio, "sleep", _repair_meanwhile)

    await er.async_sync_embedding_record(hass, entry)

    assert starts == [True]
    assert er.fetch_ollama_model_digest.await_count == 2  # type: ignore[attr-defined]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    check.assert_awaited_once_with(entry.runtime_data.pool, OLLAMA_A)


# --- Kept fallbacks checked again after setup (#860) -----------------------

FALLBACK_ID = "p2"
FALLBACK_URL = "http://ollama-fallback:11434"
SECONDARY = "Ollama Secondary"


def _kept_unchecked(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    builds: list[str | None],
    *,
    fingerprint: EmbeddingFingerprint = OLLAMA_A,
    read_at_setup: str | None = None,
    left_out: tuple[str, ...] = (),
    answers: dict[str, list[str | None]] | None = None,
) -> tuple[MockConfigEntry, FallbackEmbeddings, MagicMock, list[float]]:
    """
    Set up a two-server chain whose fallback setup kept without a full check.

    `builds` answers each build read in turn, for one server's reads; when
    both servers are read (concurrently), `answers` gives each URL its own
    sequence instead. `read_at_setup` is the fallback's build as setup read
    it. The returned mock is the chain's removed callback.
    """
    removed = MagicMock()
    chain = FallbackEmbeddings(
        [
            (FakeEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ],
        switch_callback=MagicMock(),
        removed_callback=removed,
    )
    entry = _entry(
        hass,
        record=None,
        fingerprint=fingerprint,
        embeddings=chain,
        monkeypatch=monkeypatch,
    )
    entry.runtime_data.embedding_fallback_gate = EmbeddingFallbackGate(
        primary="Ollama",
        model="mxbai-embed-large",
        left_out=left_out,
        unchecked=(FallbackBuild(FALLBACK_ID, SECONDARY, FALLBACK_URL, read_at_setup),),
    )
    if answers is None:
        fetch = AsyncMock(side_effect=builds)
    else:
        fetch = AsyncMock(side_effect=lambda _h, url, *_a, **_k: answers[url].pop(0))
    monkeypatch.setattr(er, "fetch_ollama_model_digest", fetch)
    waits: list[float] = []

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    return entry, chain, removed, waits


def _chain_ids(chain: FallbackEmbeddings) -> list[str]:
    return [provider_id for _m, _d, provider_id in chain.chain]


@pytest.mark.asyncio
async def test_a_kept_fallback_read_late_as_another_build_is_removed(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Setup missed its build: read now, it differs, so it leaves the chain."""
    entry, chain, removed, waits = _kept_unchecked(
        hass, monkeypatch, ["w:b2"], left_out=("Ollama Third",)
    )

    await er.async_sync_embedding_record(hass, entry)

    assert waits == []
    er.fetch_ollama_model_digest.assert_awaited_once_with(  # type: ignore[attr-defined]
        hass, FALLBACK_URL, "mxbai-embed-large", timeout_s=er.DIGEST_FETCH_TIMEOUT_S
    )
    assert _chain_ids(chain) == ["p1"]
    removed.assert_not_called()
    # It never answered a call, so nothing of its build is in the store.
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": f"Ollama Third, {SECONDARY}",
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }
    gate = entry.runtime_data.embedding_fallback_gate
    assert gate.left_out == ("Ollama Third", SECONDARY)
    assert gate.unchecked == ()
    assert f"Embedding fallback provider(s) {SECONDARY} removed" in caplog.text


@pytest.mark.asyncio
async def test_a_kept_fallback_read_late_as_the_same_build_stays(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read on the second try, the same build: the chain is left alone."""
    entry, chain, _removed, waits = _kept_unchecked(hass, monkeypatch, [None, "w:a1"])
    # An earlier boot left it out; the user has since pulled the same build.
    _standing_issue(hass, entry, [SECONDARY])

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_a_fallback_judged_by_the_record_is_checked_against_the_late_primary(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primary's build setup missed is read, and the fallback's again (#857)."""
    entry, chain, _switched, _waits = _kept_unchecked(
        hass,
        monkeypatch,
        [],
        fingerprint=OLLAMA_UNREAD,
        read_at_setup="w:a1",
        answers={OLLAMA_SOURCE[0]: ["w:a1"], FALLBACK_URL: ["w:b2"]},
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE

    await er.async_sync_embedding_record(hass, entry)

    assert er.fetch_ollama_model_digest.await_count == 2  # type: ignore[attr-defined]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_fallback_build_never_read_is_kept_and_logged(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After the last try an unknown build still decides nothing."""
    entry, chain, _switched, waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None]
    )

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0, 30.0]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()
    assert (
        f"Could not read which build of mxbai-embed-large the embedding "
        f"fallback(s) {SECONDARY} serve" in caplog.text
    )


@pytest.mark.asyncio
async def test_fallbacks_are_not_checked_while_the_primarys_build_is_unknown(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to compare with: the primary's own warning covers it."""
    entry, chain, _switched, _waits = _kept_unchecked(
        hass, monkeypatch, ["w:b2"], fingerprint=OLLAMA_UNREAD, read_at_setup="w:b2"
    )

    await er.async_sync_embedding_record(hass, entry)

    # Read, since the record check may learn the primary's build meanwhile,
    # but nothing to compare it with.
    assert er.fetch_ollama_model_digest.await_count == 1  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()


@pytest.mark.asyncio
async def test_fallbacks_are_not_checked_with_the_primary_down_at_setup(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup compared them with the record; the reload once it is back re-checks."""
    entry, chain, _switched, _waits = _kept_unchecked(hass, monkeypatch, [])
    entry.runtime_data.primary_embeddings = None

    await er.async_sync_embedding_record(hass, entry)

    er.fetch_ollama_model_digest.assert_not_called()  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_an_unload_during_the_fallback_read_removes_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry unloaded during the wait is not changed or given an issue."""
    entry, chain, _switched, _waits = _kept_unchecked(hass, monkeypatch, [None, "w:b2"])

    async def _unloaded(_delay: float) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _unloaded)

    await er.async_sync_embedding_record(hass, entry)

    assert er.fetch_ollama_model_digest.await_count == 1  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_removing_the_active_fallback_hands_the_chain_back(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback already embedding for a failed primary: back to the primary."""
    entry, chain, removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    chain._active_provider_id = FALLBACK_ID

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1"]
    removed.assert_called_once_with(FALLBACK_ID, "p1")
    chain.switch_callback.assert_not_called()  # type: ignore[union-attr]


def _served_by_the_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any] | None
) -> tuple[MockConfigEntry, _Cursor]:
    """Set up a removed fallback that answered calls, over a store with this record."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    chain.served.add(FALLBACK_ID)
    cur = _Cursor(record=record)
    entry.runtime_data.pool = _Pool(cur)
    return entry, cur


@pytest.mark.asyncio
async def test_a_removed_fallback_that_answered_is_noted_as_a_writer(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Its vectors may be in the store: the record notes it, the repair is raised."""
    entry, cur = _served_by_the_fallback(hass, monkeypatch, _row(OLLAMA_A))

    await er.async_sync_embedding_record(hass, entry)

    marked = cur.ran(er._MARK_MIXED_SQL)
    assert len(marked) == 1
    assert marked[0]["mixed"].obj == [asdict(OLLAMA_B)]
    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.translation_placeholders == {
        "recorded": f"{OLLAMA_A.label} + {OLLAMA_B.label}",
        "current": OLLAMA_A.label,
    }
    assert "may hold vectors of Ollama mxbai-embed-large:latest (w:b2)" in caplog.text
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.parametrize("row", [None, _row(None)], ids=["no_row", "junk_model"])
@pytest.mark.asyncio
async def test_a_writer_on_a_store_without_a_record_gets_one_to_be_noted_in(
    hass: Any, monkeypatch: pytest.MonkeyPatch, row: dict[str, Any] | None
) -> None:
    """No readable model yet: the current one is recorded, the writer noted."""
    entry, cur = _served_by_the_fallback(hass, monkeypatch, row)

    await er.async_sync_embedding_record(hass, entry)

    assert _written(cur) == [OLLAMA_A]
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    assert marked["mixed"].obj == [asdict(OLLAMA_B)]
    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.translation_placeholders == {
        "recorded": f"{OLLAMA_A.label} + {OLLAMA_B.label}",
        "current": OLLAMA_A.label,
    }


@pytest.mark.parametrize(
    "failure", [psycopg.OperationalError("gone"), TimeoutError()], ids=["db", "timeout"]
)
@pytest.mark.asyncio
async def test_a_writer_the_database_cannot_note_is_retried_then_logged(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    """The note is the only trace of the vectors: tried again, then an error."""
    entry, chain, _removed, waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    chain.served.add(FALLBACK_ID)
    cur = _Cursor(record=_row(OLLAMA_A))
    cur.execute = AsyncMock(side_effect=failure)  # type: ignore[method-assign]
    entry.runtime_data.pool = _Pool(cur)

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0, 30.0, 60.0]
    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None
    assert "Could not note Ollama mxbai-embed-large:latest (w:b2) as a writer" in (
        caplog.text
    )
    assert "this could not be recorded; to clear them" in caplog.text


@pytest.mark.asyncio
async def test_a_writer_note_that_fails_once_is_noted_on_the_retry(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-embed holding the row for a batch releases it before the retry."""
    entry, chain, _removed, waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    chain.served.add(FALLBACK_ID)
    cur = _Cursor(record=_row(OLLAMA_A))
    real_execute = cur.execute
    attempts = 0

    async def _busy_once(query: str, params: Any = None) -> None:
        nonlocal attempts
        if query == er._READ_RECORD_FOR_UPDATE_SQL:
            attempts += 1
            if attempts == 1:
                raise TimeoutError
        await real_execute(query, params)

    cur.execute = _busy_once  # type: ignore[method-assign]
    entry.runtime_data.pool = _Pool(cur)

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0]
    assert len(cur.ran(er._MARK_MIXED_SQL)) == 1
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None


@pytest.mark.asyncio
async def test_an_unload_during_the_writer_note_raises_no_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write already ran, so the record is marked; the unloaded entry gets no issue."""
    entry, cur = _served_by_the_fallback(hass, monkeypatch, _row(OLLAMA_A))
    real_execute = cur.execute

    async def _unloads(query: str, params: Any = None) -> None:
        await real_execute(query, params)
        if query == er._MARK_MIXED_SQL:
            entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    cur.execute = _unloads  # type: ignore[method-assign]

    await er.async_sync_embedding_record(hass, entry)

    assert len(cur.ran(er._MARK_MIXED_SQL)) == 1
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_a_writer_of_the_recorded_build_mid_reembed_is_noted_too(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store recorded as the fallback's build B, re-embedding to A: B is a writer."""
    entry, cur = _served_by_the_fallback(
        hass,
        monkeypatch,
        _row(OLLAMA_B, target=OLLAMA_A, cursor=CURSOR, mixed=(OLLAMA_A,)),
    )

    await er.async_sync_embedding_record(hass, entry)

    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    # The writers list changes, so the run's finish guard refuses to certify.
    assert marked["mixed"].obj == [asdict(OLLAMA_A), asdict(OLLAMA_B)]
    assert marked["cursor"] is None


@pytest.mark.asyncio
async def test_a_failing_fallback_read_is_logged_and_leaves_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected error in the read half: nothing changes, and the log says so."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, [])
    _standing_issue(hass, entry, [SECONDARY])
    monkeypatch.setattr(
        er, "_async_read_fallback_builds", AsyncMock(side_effect=RuntimeError("boom"))
    )

    await er.async_sync_embedding_record(hass, entry)

    assert "fallbacks' build check failed: RuntimeError('boom')" in caplog.text
    assert SECONDARY in caplog.text
    assert _chain_ids(chain) == ["p1", "p2"]
    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


def _standing_issue(hass: Any, entry: Any, names: list[str]) -> None:
    """Raise the other-build issue as an earlier run of this entry would have."""
    er.async_sync_other_build_fallback_issue(
        hass,
        entry.entry_id,
        primary="Ollama",
        model="mxbai-embed-large",
        left_out=names,
    )
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)


@pytest.mark.asyncio
async def test_a_fallback_never_read_keeps_a_standing_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known mismatch from the last boot is not forgotten on a bad read."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None]
    )
    _standing_issue(hass, entry, [SECONDARY])

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_fallback_never_read_beside_setups_left_out_keeps_their_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup's positive evidence stays synced; the unread one adds nothing."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None], left_out=("Ollama Third",)
    )
    _standing_issue(hass, entry, ["Ollama Third"])

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p2"]
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "Ollama Third",
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }


@pytest.mark.asyncio
async def test_a_chain_that_is_not_a_fallback_chain_is_left_alone(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to remove from: the gate is left as setup wrote it."""
    entry, _chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    entry.runtime_data.embeddings = FakeEmbeddings("alone")

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_an_unload_during_the_note_retry_wait_logs_no_error(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A reload closed the pool under the try: no 'drop your vectors' error."""
    entry, chain, _removed, waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    chain.served.add(FALLBACK_ID)
    cur = _Cursor(record=_row(OLLAMA_A))
    cur.execute = AsyncMock(side_effect=TimeoutError)  # type: ignore[method-assign]
    entry.runtime_data.pool = _Pool(cur)

    async def _unloads(delay: float) -> None:
        waits.append(delay)
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _unloads)

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0]
    assert "could not be recorded" not in caplog.text
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


THIRD_URL = "http://ollama-third:11434"
FOURTH_URL = "http://ollama-fourth:11434"


def _kept_three(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, list[str | None]],
    *,
    left_out: tuple[str, ...] = (),
) -> tuple[MockConfigEntry, FallbackEmbeddings]:
    """Set up three kept fallbacks setup could not check, each with its own answers."""
    chain = FallbackEmbeddings(
        [
            (FakeEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f2"), "edge", "p2"),
            (FakeEmbeddings("f3"), "edge", "p3"),
            (FakeEmbeddings("f4"), "edge", "p4"),
        ],
        removed_callback=MagicMock(),
    )
    entry = _entry(hass, record=None, embeddings=chain, monkeypatch=monkeypatch)
    entry.runtime_data.embedding_fallback_gate = EmbeddingFallbackGate(
        primary="Ollama",
        model="mxbai-embed-large",
        left_out=left_out,
        unchecked=(
            FallbackBuild("p2", SECONDARY, FALLBACK_URL),
            FallbackBuild("p3", "Ollama Third", THIRD_URL),
            FallbackBuild("p4", "Ollama Fourth", FOURTH_URL),
        ),
    )
    monkeypatch.setattr(
        er,
        "fetch_ollama_model_digest",
        AsyncMock(side_effect=lambda _h, url, *_a, **_k: answers[url].pop(0)),
    )

    async def _sleep(_delay: float) -> None:
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    return entry, chain


@pytest.mark.asyncio
async def test_three_fallbacks_are_judged_each_on_its_own_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One matches, one differs, one never answers: only the second leaves."""
    entry, chain = _kept_three(
        hass,
        monkeypatch,
        {
            FALLBACK_URL: ["w:a1"],
            THIRD_URL: [None, "w:b2"],
            FOURTH_URL: [None, None, None],
        },
        left_out=("Ollama Prior",),
    )
    chain.served.update({"p3", "p4"})
    cur = _Cursor(record=_row(OLLAMA_A))
    entry.runtime_data.pool = _Pool(cur)

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p2", "p4"]
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "Ollama Prior, Ollama Third",
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }
    assert entry.runtime_data.embedding_fallback_gate.left_out == (
        "Ollama Prior",
        "Ollama Third",
    )
    assert "fallback(s) Ollama Fourth serve" in caplog.text
    # Only the removed one that answered is a writer; the unread one stays.
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    assert marked["mixed"].obj == [asdict(OLLAMA_B)]


@pytest.mark.asyncio
async def test_a_removal_beside_an_unread_sibling_still_updates_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive evidence is synced even while another build is still unknown."""
    entry, chain = _kept_three(
        hass,
        monkeypatch,
        {FALLBACK_URL: ["w:b2"], THIRD_URL: [None, None, None], FOURTH_URL: ["w:a1"]},
    )

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p3", "p4"]
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": SECONDARY,
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }


@pytest.mark.asyncio
async def test_a_failing_record_check_does_not_skip_the_fallback_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected error in the record half is logged; the settle still runs."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:b2"])
    monkeypatch.setattr(
        er, "_async_check_record", AsyncMock(side_effect=RuntimeError("boom"))
    )

    await er.async_sync_embedding_record(hass, entry)

    assert "embedding model check failed: RuntimeError('boom')" in caplog.text
    assert _chain_ids(chain) == ["p1"]


# --- The state matrix: both builds unread, the primary's unread --------------


def _both_unread(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    primary: list[str | None],
    fallback: list[str | None],
) -> tuple[MockConfigEntry, FallbackEmbeddings, MagicMock, list[float]]:
    """Neither build was read at setup; each server answers its own sequence."""
    entry, chain, removed, waits = _kept_unchecked(
        hass,
        monkeypatch,
        [],
        fingerprint=OLLAMA_UNREAD,
        answers={OLLAMA_SOURCE[0]: primary, FALLBACK_URL: fallback},
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE
    return entry, chain, removed, waits


@pytest.mark.asyncio
async def test_both_builds_read_late_and_different_remove_the_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #860 case: setup read neither build; both are learned in one pass."""
    entry, chain, _removed, waits = _both_unread(
        hass, monkeypatch, primary=[None, "w:a1"], fallback=[None, "w:b2"]
    )

    await er.async_sync_embedding_record(hass, entry)

    # Both re-reads waited once, alongside each other.
    assert waits == [10.0, 10.0]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert _chain_ids(chain) == ["p1"]
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": SECONDARY,
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }
    assert entry.runtime_data.embedding_fallback_gate.left_out == (SECONDARY,)


@pytest.mark.asyncio
async def test_both_builds_read_late_and_the_same_keep_the_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, chain, removed, _waits = _both_unread(
        hass, monkeypatch, primary=["w:a1"], fallback=["w:a1"]
    )

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert _chain_ids(chain) == ["p1", "p2"]
    removed.assert_not_called()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_a_late_primary_with_a_fallback_never_read_keeps_it_and_logs(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    entry, chain, _removed, waits = _both_unread(
        hass, monkeypatch, primary=["w:a1"], fallback=[None, None, None]
    )

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0, 30.0]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert _chain_ids(chain) == ["p1", "p2"]
    assert f"fallback(s) {SECONDARY} serve" in caplog.text
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_a_primary_never_read_leaves_the_fallbacks_named_in_the_log(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Three failed primary reads: the fallback's known build has no reference."""
    entry, chain, _removed, waits = _both_unread(
        hass, monkeypatch, primary=[None, None, None], fallback=["w:b2"]
    )

    await er.async_sync_embedding_record(hass, entry)

    assert waits == [10.0, 30.0]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_UNREAD
    assert _chain_ids(chain) == ["p1", "p2"]
    assert f"fallback(s) {SECONDARY} serve could not be compared" in caplog.text
    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_an_unload_during_both_rereads_changes_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, chain, _removed, _waits = _both_unread(
        hass, monkeypatch, primary=[None, "w:a1"], fallback=[None, "w:b2"]
    )

    async def _unloaded(_delay: float) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _unloaded)

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p2"]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_UNREAD
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()


@pytest.mark.asyncio
async def test_a_fallback_judged_by_the_record_stays_when_the_late_primary_matches(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, chain, removed, _waits = _kept_unchecked(
        hass,
        monkeypatch,
        [],
        fingerprint=OLLAMA_UNREAD,
        read_at_setup="w:a1",
        answers={OLLAMA_SOURCE[0]: ["w:a1"], FALLBACK_URL: ["w:a1"]},
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE

    await er.async_sync_embedding_record(hass, entry)

    assert er.fetch_ollama_model_digest.await_count == 2  # type: ignore[attr-defined]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert _chain_ids(chain) == ["p1", "p2"]
    removed.assert_not_called()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_an_unload_during_the_primary_reread_removes_no_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback's build was known; only the primary's re-read waited."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass,
        monkeypatch,
        [None, "w:a1"],
        fingerprint=OLLAMA_UNREAD,
        read_at_setup="w:b2",
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE

    async def _unloaded(_delay: float) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er.asyncio, "sleep", _unloaded)

    await er.async_sync_embedding_record(hass, entry)

    assert er.fetch_ollama_model_digest.await_count == 1  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked != ()


@pytest.mark.asyncio
async def test_an_unload_during_the_fallback_fetch_removes_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unloaded while the read itself ran, not during a wait."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, [])

    async def _fetch(*_args: Any, **_kwargs: Any) -> str:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
        return "w:b2"

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _fetch)

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


# --- Starting the re-embed -------------------------------------------------


@pytest.mark.asyncio
async def test_the_repair_starts_a_run_that_records_its_target(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair hands the target to the run, which sets it under the lock."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    started = _capture_starts(monkeypatch)

    assert await er.async_start_reembed(hass, entry)

    assert started == [True]


@pytest.mark.parametrize(
    ("embeddings", "fingerprint", "state"),
    [
        # The stand-in for a provider down at boot.
        (UnavailableEmbeddings("Ollama"), OLLAMA_A, ConfigEntryState.LOADED),
        # An Ollama primary whose build was not read: a fallback may serve.
        (None, OLLAMA_UNREAD, ConfigEntryState.LOADED),
        # Not loaded: no pool or client to run on.
        (None, OLLAMA_A, ConfigEntryState.NOT_LOADED),
    ],
)
@pytest.mark.asyncio
async def test_the_repair_does_not_start_when_it_cannot_run(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    embeddings: Any,
    fingerprint: EmbeddingFingerprint,
    state: ConfigEntryState,
) -> None:
    """Each reason a run cannot start makes the fix flow abort."""
    entry = _entry(
        hass,
        record=None,
        embeddings=embeddings,
        fingerprint=fingerprint,
        state=state,
        monkeypatch=monkeypatch,
    )
    created = _no_task(monkeypatch, entry)

    assert not await er.async_start_reembed(hass, entry)
    assert created == []


@pytest.mark.asyncio
async def test_the_run_embeds_with_the_primary_not_the_fallback_chain(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback whose build nobody read never supplies the re-embed."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    entry.runtime_data.embeddings = FakeEmbeddings("chain")
    entry.runtime_data.primary_embeddings = FakeEmbeddings("primary")
    runs: list[Any] = []

    async def _reembed(_pool: Any, embeddings: Any, *_a: Any, **_k: Any) -> None:
        runs.append(embeddings.name)

    monkeypatch.setattr(er, "_async_reembed", _reembed)
    monkeypatch.setattr(
        entry,
        "async_create_background_task",
        lambda _hass, coro, **_kw: asyncio.get_running_loop().create_task(coro),
    )

    assert await er.async_start_reembed(hass, entry)
    await entry.runtime_data.reembed_task

    assert runs == ["primary"]


@pytest.mark.parametrize(
    ("setup_digest", "now", "starts", "recorded_digest"),
    [
        # Same build as at setup.
        ("w:a1", "w:a1", True, "w:a1"),
        # Unread at setup, read now: learned, and the run can go.
        (None, "w:a1", True, "w:a1"),
        # Another build since setup: the target is stale.
        ("w:a1", "w:b2", False, "w:a1"),
        # Unread now but known at setup: go with what setup read.
        ("w:a1", None, True, "w:a1"),
        # Never read: nobody can say which build the store would hold.
        (None, None, False, None),
    ],
)
@pytest.mark.asyncio
async def test_the_repair_reads_the_primary_build_again(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    setup_digest: str | None,
    now: str | None,
    starts: bool,  # noqa: FBT001
    recorded_digest: str | None,
) -> None:
    """The build the run records is the one the primary serves when confirmed."""
    entry = _entry(
        hass,
        record=None,
        fingerprint=embedding_fingerprint(
            "ollama", "mxbai-embed-large", dims=1024, digest=setup_digest
        ),
        monkeypatch=monkeypatch,
    )
    entry.runtime_data.embedding_digest_source = ("http://ollama", "mxbai-embed-large")
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value=now))
    created = _no_task(monkeypatch, entry)

    assert await er.async_start_reembed(hass, entry) is starts
    assert created == (["hga_embedding_reembed"] if starts else [])
    assert entry.runtime_data.embedding_fingerprint.digest == recorded_digest


@pytest.mark.asyncio
async def test_a_running_reembed_is_not_started_twice(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second confirm, or a resume, while a run is going starts nothing."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    running = asyncio.get_running_loop().create_future()
    entry.runtime_data.reembed_task = running
    created = _no_task(monkeypatch, entry)

    assert er._start_reembed_task(hass, entry, start=True)

    assert entry.runtime_data.reembed_task is running
    assert created == []
    running.cancel()


@pytest.mark.asyncio
async def test_a_run_that_cannot_start_waits_with_a_warning(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A resume while the primary is down starts no task and says so."""
    entry = _entry(
        hass,
        record=None,
        embeddings=UnavailableEmbeddings("Ollama"),
        monkeypatch=monkeypatch,
    )
    created = _no_task(monkeypatch, entry)

    assert not er._start_reembed_task(hass, entry, start=False)

    assert created == []
    assert "waits for its embedding primary" in caplog.text


@pytest.mark.asyncio
async def test_a_stopped_run_brings_the_issue_back(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix flow deleted the issue; a run that stops raises it again."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    runs: list[dict[str, Any]] = []

    async def _run(*_args: Any, **kwargs: Any) -> None:
        runs.append(kwargs)

    monkeypatch.setattr(er, "_async_reembed", _run)
    monkeypatch.setattr(
        entry,
        "async_create_background_task",
        lambda _hass, coro, **_kw: asyncio.get_running_loop().create_task(coro),
    )

    er._start_reembed_task(hass, entry, start=True)
    await entry.runtime_data.reembed_task
    (run,) = runs
    assert run["start"] is True
    run["stopped"](OPENAI.label)

    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.translation_placeholders == {
        "recorded": OPENAI.label,
        "current": OLLAMA_A.label,
    }


@pytest.mark.asyncio
async def test_the_fix_flow_starts_the_reembed_or_aborts(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flow is the re-embed's; it aborts when the run cannot start."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    entry.add_to_hass(hass)
    er._raise_changed_issue(hass, entry, OPENAI.label, OLLAMA_A.label)
    issue_id = f"{ISSUE_EMBEDDING_MODEL_CHANGED}_entry1"
    started = AsyncMock(return_value=False)
    monkeypatch.setattr(repairs, "async_start_reembed", started)

    flow = await repairs.async_create_fix_flow(hass, issue_id, {"entry_id": "entry1"})
    assert isinstance(flow, repairs.ReembedStoreFlow)
    flow.hass = hass
    flow.handler = DOMAIN
    flow.issue_id = issue_id

    # Home Assistant opens a fix flow with the issue id as input: that is not
    # the user's confirmation.
    form = await flow.async_step_init({"issue_id": issue_id})
    started.assert_not_called()
    assert form.get("type") == "form"
    assert form.get("step_id") == "confirm"
    assert form.get("description_placeholders") == {
        "recorded": OPENAI.label,
        "current": OLLAMA_A.label,
    }
    aborted = await flow.async_step_confirm({})
    assert aborted.get("type") == "abort"
    assert aborted.get("reason") == "not_ready"

    started.return_value = True
    done = await flow.async_step_confirm({})
    assert done.get("type") == "create_entry"


@pytest.mark.asyncio
async def test_the_fix_flow_aborts_when_its_issue_is_gone(hass: Any) -> None:
    """A dialog left open across a reload that cleared the issue starts nothing."""
    MockConfigEntry(domain=DOMAIN, entry_id="entry1").add_to_hass(hass)
    issue_id = f"{ISSUE_EMBEDDING_MODEL_CHANGED}_entry1"
    flow = await repairs.async_create_fix_flow(hass, issue_id, {"entry_id": "entry1"})
    assert isinstance(flow, repairs.ReembedStoreFlow)
    flow.hass = hass
    flow.handler = DOMAIN
    flow.issue_id = issue_id

    result = await flow.async_step_init()

    assert result.get("type") == "abort"
    assert result.get("reason") == "not_ready"


# --- The re-embed run ------------------------------------------------------

STAMP = "2026-10-24T00:00:00+00:00"


class _RunConn:
    """A connection serving `rows` in batches and holding the record row."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        record: dict[str, Any] | None = None,
        locked: bool = True,
    ) -> None:
        self.rows = rows
        self.record = record if record is not None else _row(OPENAI)
        self.locked = locked
        self.queries: list[tuple[str, Any]] = []
        self.updates: list[tuple[Any, ...]] = []
        self.deletes: list[tuple[Any, ...]] = []
        self.fail_update: Exception | None = None
        self.fail_unlock = False
        self.closed = False
        self.transactions = 0
        self.rollbacks = 0

    def _keys(self) -> list[tuple[str, str, str]]:
        return [(r["prefix"], r["key"], r["field_name"]) for r in self.rows]

    def _apply(  # noqa: PLR0911, PLR0912 - one branch per scripted query
        self, query: str, params: Any
    ) -> tuple[list[Any], int]:
        target = params.get("target") if isinstance(params, dict) else None
        target_obj = target.obj if target is not None else None
        held = self.record.get("reembed_target")
        if query == er._REEMBED_LOCK_SQL:
            return [{"locked": self.locked}], 1
        if query == er._REEMBED_UNLOCK_SQL:
            if self.fail_unlock:
                msg = "connection lost"
                raise psycopg.OperationalError(msg)
            return [{"unlocked": True}], 1
        if query == er._COUNT_VECTORS_SQL:
            return [{"total": len(self.rows)}], 1
        if query == er._FIRST_BATCH_SQL:
            return self.rows[: params["limit"]], 0
        if query == er._NEXT_BATCH_SQL:
            after = (params["prefix"], params["key"], params["field"])
            start = self._keys().index(after) + 1
            return self.rows[start : start + params["limit"]], 0
        if query in (er._READ_RECORD_SQL, er._READ_RECORD_FOR_UPDATE_SQL):
            return [dict(self.record)], 1
        if query == er._CLEAR_UNSTARTED_TARGET_SQL:
            if held != target_obj or self.record["reembed_cursor"] is not None:
                return [], 0
            self.record["reembed_target"] = None
            return [], 1
        if query == er._START_REEMBED_SQL:
            if held != target_obj:
                self.record["reembed_cursor"] = None
            self.record["reembed_target"] = target_obj
            return [], 1
        if query == er._SAVE_CURSOR_SQL:
            previous = params["previous"].obj if params["previous"] else None
            mixed = params["mixed"].obj if params["mixed"] else None
            if (
                held != target_obj
                or self.record["reembed_cursor"] != previous
                or self.record.get("mixed_with") != mixed
            ):
                return [], 0
            self.record["reembed_cursor"] = params["cursor"].obj
            return [], 1
        if query == er._BUILD_CHANGED_SQL:
            if held != target_obj:
                return [], 0
            self.record.update(reembed_cursor=None, mixed_with=params["mixed"].obj)
            return [], 1
        if query == getattr(er, "_END_ON_REJECTIONS_SQL", None):
            if held != target_obj:
                return [], 0
            self.record.update(
                reembed_target=None,
                reembed_cursor=None,
                mixed_with=params["mixed"].obj,
            )
            return [], 1
        if query == er._FINISH_REEMBED_SQL:
            mixed = params["mixed"].obj if params["mixed"] else None
            cursor = params["cursor"].obj if params["cursor"] else None
            if (
                held != target_obj
                or self.record.get("mixed_with") != mixed
                or self.record["reembed_cursor"] != cursor
            ):
                return [], 0
            self.record.update(
                model=held, reembed_target=None, reembed_cursor=None, mixed_with=None
            )
            return [], 1
        return [], 0

    async def execute(self, query: str, params: Any = None) -> Any:
        self.queries.append((query, params))
        rows, rowcount = self._apply(query, params)
        return SimpleNamespace(
            rowcount=rowcount,
            fetchone=AsyncMock(return_value=rows[0] if rows else None),
            fetchall=AsyncMock(return_value=rows),
        )

    async def close(self) -> None:
        self.closed = True

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        """Roll back the record and the vector writes when the block raises."""
        record = {
            k: (list(v) if isinstance(v, list) else v) for k, v in self.record.items()
        }
        updates, deletes = len(self.updates), len(self.deletes)
        self.transactions += 1
        try:
            yield
        except BaseException:
            self.record = record
            del self.updates[updates:]
            del self.deletes[deletes:]
            self.rollbacks += 1
            raise

    @contextlib.asynccontextmanager
    async def cursor(self, **_kwargs: Any) -> AsyncGenerator[Any]:
        conn = self

        class _Cur:
            rowcount = 0
            last: Any = None

            async def execute(self, query: str, params: Any = None) -> None:
                self.last = await conn.execute(query, params)
                self.rowcount = self.last.rowcount

            async def fetchone(self) -> Any:
                return await self.last.fetchone() if self.last else None

            async def executemany(
                self, query: str, params: list[tuple[Any, ...]]
            ) -> None:
                if query == er._UPDATE_VECTOR_SQL:
                    if conn.fail_update is not None:
                        raise conn.fail_update
                    conn.updates.extend(params)
                elif query == er._DELETE_VECTOR_SQL:
                    conn.deletes.extend(params)

        yield _Cur()

    def ran(self, query: str) -> list[Any]:
        return [p for q, p in self.queries if q == query]


class _RunPool:
    def __init__(self, conn: _RunConn) -> None:
        self.conn = conn

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncGenerator[_RunConn]:
        yield self.conn


def _rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "prefix": "memories.u1",
            "key": f"k{i:03}",
            "field_name": "content",
            "updated_at": f"{STAMP}#{i}",
            "value": {"content": f"memory {i}" if i != 1 else ""},
        }
        for i in range(count)
    ]


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the run's pauses instead of sleeping them."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    return slept


async def _run(
    conn: _RunConn,
    embeddings: Any = None,
    *,
    start: bool = True,
    target: EmbeddingFingerprint = OLLAMA_A,
    build: str | None = "",
) -> list[str]:
    """Run the re-embed; `build` is what the primary serves at the finish."""
    stopped: list[str] = []

    async def _current_build() -> str | None:
        return target.digest if build == "" else build

    await er._async_reembed(
        cast("Any", _RunPool(conn)),
        embeddings if embeddings is not None else FakeEmbeddings("new"),
        target,
        start=start,
        stopped=stopped.append,
        current_build=_current_build,
    )
    return stopped


@pytest.mark.asyncio
async def test_the_run_reembeds_every_row_and_records_the_model(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """Batches under the lock, guards each row, drops a textless one, records."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    conn = _RunConn(_rows(5))

    assert await _run(conn) == []

    assert [u[2] for u in conn.updates] == ["k000", "k002", "k003", "k004"]
    assert conn.updates[0] == (
        [float(len("memory 0"))],
        "memories.u1",
        "k000",
        "content",
        f"{STAMP}#0",
    )
    assert conn.deletes == [("memories.u1", "k001", "content", f"{STAMP}#1")]
    assert [p["cursor"].obj for p in conn.ran(er._SAVE_CURSOR_SQL)] == [
        # The last row done, then the texts rejected so far (#857).
        ["memories.u1", "k001", "content", 0],
        ["memories.u1", "k003", "content", 0],
        ["memories.u1", "k004", "content", 0],
    ]
    assert conn.record["model"] == OLLAMA_A.as_json().obj
    assert conn.record["reembed_target"] is None
    assert conn.ran(er._REEMBED_UNLOCK_SQL)
    # The lock comes first; the target is set only under it, on a record
    # read under the row lock.
    names = [q for q, _ in conn.queries]
    assert names.index(er._REEMBED_LOCK_SQL) < names.index(er._START_REEMBED_SQL)
    assert names.index(er._READ_RECORD_FOR_UPDATE_SQL) < names.index(
        er._START_REEMBED_SQL
    )
    assert len(pauses) == 3


@pytest.mark.asyncio
async def test_a_confirmed_again_run_resumes_after_its_cursor(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A repair confirmed again for the same target keeps the run's place."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4), record=_row(OPENAI, target=OLLAMA_A, cursor=CURSOR))

    await _run(conn, start=True)

    assert [u[2] for u in conn.updates] == ["k002", "k003"]
    assert pauses


@pytest.mark.asyncio
async def test_a_resumed_run_starts_after_its_cursor(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A run resumed at setup reads where it stopped from the record."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4), record=_row(OPENAI, target=OLLAMA_A, cursor=CURSOR))

    await _run(conn, start=False)

    assert [u[2] for u in conn.updates] == ["k002", "k003"]
    assert conn.ran(er._START_REEMBED_SQL) == []
    assert pauses


@pytest.mark.asyncio
async def test_a_store_already_holding_the_target_needs_no_run(
    pauses: list[float],
) -> None:
    """A repair left open after a switch back re-embeds nothing."""
    conn = _RunConn(_rows(3), record=_row(OLLAMA_A))

    assert await _run(conn) == []

    assert conn.updates == []
    assert conn.ran(er._START_REEMBED_SQL) == []
    assert pauses == []


@pytest.mark.asyncio
async def test_a_resume_whose_target_changed_does_not_run(
    pauses: list[float],
) -> None:
    """The record now names another target: this run leaves it alone."""
    conn = _RunConn(_rows(3), record=_row(OPENAI, target=OPENAI_B, cursor=None))

    await _run(conn, start=False)

    assert conn.updates == []
    assert pauses == []


@pytest.mark.asyncio
async def test_a_run_another_entry_holds_does_nothing(pauses: list[float]) -> None:
    """Two entries sharing the database do not run it twice; the issue returns."""
    conn = _RunConn(_rows(3), locked=False)

    # The issue comes back naming what the store holds, not the target twice.
    assert await _run(conn) == [OPENAI.label]

    assert conn.updates == []
    assert conn.ran(er._START_REEMBED_SQL) == []
    assert pauses == []


@pytest.mark.asyncio
async def test_a_failed_run_keeps_its_place_unlocks_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """A provider that stays down stops the run without recording the model."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    conn = _RunConn(_rows(4))
    embeddings = MagicMock()
    embeddings.aembed_documents = AsyncMock(
        # Row 1 has no text, so the first batch embeds one.
        side_effect=[[[1.0]], EmbeddingsUnavailableError("down")]
    )

    assert await _run(conn, embeddings) == [OPENAI.label]

    assert len(conn.ran(er._SAVE_CURSOR_SQL)) == 1
    assert conn.record["model"] == OPENAI.as_json().obj
    assert conn.record["reembed_cursor"] == ["memories.u1", "k001", "content", 0]
    assert conn.ran(er._REEMBED_UNLOCK_SQL)
    assert "resumes where it stopped" in caplog.text
    assert len(pauses) == 1


@pytest.mark.asyncio
async def test_a_wrong_size_run_stops_unlocked_and_names_the_size_issue(
    caplog: pytest.LogCaptureFixture, pauses: list[float]
) -> None:
    """Vectors of another size than the store's stop the run cleanly."""
    conn = _RunConn(_rows(2))
    conn.fail_update = psycopg.DataError("expected 1024 dimensions, not 768")

    assert await _run(conn) == [OPENAI.label]

    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert conn.ran(er._REEMBED_UNLOCK_SQL)
    assert "another size than the store's" in caplog.text
    assert pauses == []


@pytest.mark.asyncio
async def test_a_superseded_run_stops_at_its_next_save(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """Another run took the target: this one stops instead of rewriting on."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 1)
    conn = _RunConn(_rows(3))
    real_apply = conn._apply

    taken = False

    def _taken_over(query: str, params: Any) -> tuple[list[Any], int]:
        # The other run's write is committed: it survives this run's rollback.
        nonlocal taken
        if query == er._SAVE_CURSOR_SQL:
            taken = True
        if taken:
            conn.record["reembed_target"] = OPENAI_B.as_json().obj
        return real_apply(query, params)

    monkeypatch.setattr(conn, "_apply", _taken_over)

    assert await _run(conn) == []

    # The place is saved first, so a batch whose target was taken writes
    # nothing at all.
    assert conn.updates == []
    assert conn.rollbacks == 1
    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert "another run changed its target" in caplog.text
    assert pauses == []


@pytest.mark.asyncio
async def test_a_cancelled_run_keeps_its_place_and_unlocks() -> None:
    """The unload cancels the run: no finish, no issue, the lock released."""
    conn = _RunConn(_rows(2))
    embeddings = MagicMock()

    async def _hang(_texts: list[str]) -> list[list[float]]:
        await asyncio.Event().wait()
        return []

    embeddings.aembed_documents = _hang
    stopped: list[str] = []
    task = asyncio.get_running_loop().create_task(
        er._async_reembed(
            cast("Any", _RunPool(conn)),
            embeddings,
            OLLAMA_A,
            start=True,
            stopped=stopped.append,
            current_build=AsyncMock(return_value=OLLAMA_A.digest),
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert conn.ran(er._REEMBED_UNLOCK_SQL)
    assert stopped == []


@pytest.mark.asyncio
async def test_a_failed_unlock_closes_the_connection(pauses: list[float]) -> None:
    """A pooled connection still holding the lock would refuse every later run."""
    conn = _RunConn(_rows(1))
    conn.fail_unlock = True

    await _run(conn)

    assert conn.closed
    assert pauses


@pytest.mark.asyncio
async def test_a_failed_embedding_is_retried_before_the_run_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure waits and tries again; a lasting one raises."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)
    embeddings = MagicMock()
    embeddings.aembed_documents = AsyncMock(side_effect=[TimeoutError(), [[1.0]]])

    vectors, _took = await er._embed_with_retry(embeddings, ["a"])
    assert vectors == [[1.0]]
    assert sleeps == [er._RETRY_DELAYS_S[0]]

    embeddings.aembed_documents = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(TimeoutError):
        await er._embed_with_retry(embeddings, ["a"])

    embeddings.aembed_documents = AsyncMock(return_value=[])
    with pytest.raises(ValueError, match="0 vectors for 1 texts"):
        await er._embed_with_retry(embeddings, ["a"])


@pytest.mark.asyncio
async def test_a_hung_embedding_call_times_out_and_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that accepts the call and never answers does not stall the run."""
    monkeypatch.setattr(er, "_EMBED_TIMEOUT_S", 0.01)
    monkeypatch.setattr(er, "_RETRY_DELAYS_S", (0.0,))
    calls = 0

    async def _first_hangs(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return [[1.0] for _ in texts]

    embeddings = MagicMock()
    embeddings.aembed_documents = _first_hangs

    vectors, _took = await er._embed_with_retry(embeddings, ["a"])
    assert vectors == [[1.0]]
    assert calls == 2


# --- The Ollama digest -----------------------------------------------------


@pytest.mark.asyncio
async def test_each_pause_lasts_as_long_as_its_batch(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """The run gives the server back as much time as each batch took."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    durations = iter([0.5, 1.5, 0.25])

    async def _embed(_e: Any, texts: Any) -> tuple[list[list[float]], float]:
        return [[1.0] for _ in texts], next(durations)

    monkeypatch.setattr(er, "_embed_with_retry", _embed)

    await _run(_RunConn(_rows(6)))

    assert pauses == [0.5, 1.5, 0.25]


@pytest.mark.asyncio
async def test_a_failed_batch_write_keeps_the_place_it_had(
    pauses: list[float],
) -> None:
    """Vectors and their place commit together: a failed write saves neither."""
    conn = _RunConn(_rows(2))
    conn.fail_update = psycopg.OperationalError("connection lost")

    assert await _run(conn) == [OPENAI.label]

    assert conn.updates == []
    assert conn.record["reembed_cursor"] is None
    assert conn.rollbacks == 1
    assert pauses == []


@pytest.mark.asyncio
async def test_a_wrong_size_target_is_dropped_so_setup_does_not_loop(
    pauses: list[float],
) -> None:
    """A first batch of another size changed nothing: its target goes."""
    conn = _RunConn(_rows(2))
    conn.fail_update = psycopg.DataError("expected 1024 dimensions, not 768")

    await _run(conn)

    assert conn.record["reembed_target"] is None
    assert conn.record["model"] == OPENAI.as_json().obj
    assert pauses == []


@pytest.mark.asyncio
async def test_a_build_pulled_during_the_run_is_not_certified(
    caplog: pytest.LogCaptureFixture, pauses: list[float]
) -> None:
    """The primary's build is read again before the store is recorded."""
    conn = _RunConn(_rows(2))

    assert await _run(conn, build="w:pulled") == [OPENAI.label]

    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert conn.record["model"] == OPENAI.as_json().obj
    assert "now serves another build" in caplog.text
    # The rows done may hold the pulled build: the next run starts over, and
    # that build counts as a writer.
    assert conn.record["reembed_cursor"] is None
    assert [EmbeddingFingerprint.from_json(m) for m in conn.record["mixed_with"]] == [
        replace(OLLAMA_A, digest="w:pulled")
    ]
    assert pauses


@pytest.mark.asyncio
async def test_an_unread_build_at_the_finish_is_not_certified(
    caplog: pytest.LogCaptureFixture, pauses: list[float]
) -> None:
    """A server that no longer names its build may serve the one pulled."""
    conn = _RunConn(_rows(2))

    assert await _run(conn, build=None) == [OPENAI.label]

    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert conn.record["reembed_cursor"] == ["memories.u1", "k001", "content", 0]
    assert "did not say which build" in caplog.text
    assert pauses


@pytest.mark.asyncio
async def test_a_writer_noted_during_the_run_stops_its_next_save(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """A setup check resetting the place mid-run is not written over."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 1)
    conn = _RunConn(_rows(3))
    real_embed = er._embed_with_retry
    batches = 0

    async def _check_commits_meanwhile(embeddings: Any, texts: Any) -> Any:
        # Another entry's setup check commits its reset while batch 2 embeds.
        nonlocal batches
        batches += 1
        if batches == 2:
            conn.record["reembed_cursor"] = None
        return await real_embed(embeddings, texts)

    monkeypatch.setattr(er, "_embed_with_retry", _check_commits_meanwhile)

    assert await _run(conn) == [OPENAI.label]

    assert [u[2] for u in conn.updates] == ["k000"]
    assert conn.record["reembed_cursor"] is None
    # The record still names this run's target: a writer was noted, not a
    # takeover, so the repair comes back and the next run starts over.
    assert "another model wrote into the store during the run" in caplog.text
    assert pauses


@pytest.mark.asyncio
async def test_a_writer_noted_during_the_first_batch_stops_the_first_save(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """No place to reset yet: the writers list alone must stop the run."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 1)
    conn = _RunConn(_rows(3))
    real_embed = er._embed_with_retry

    async def _note_lands_meanwhile(embeddings: Any, texts: Any) -> Any:
        if conn.record.get("mixed_with") is None:
            conn.record["mixed_with"] = [OPENAI_B.as_json().obj]
        return await real_embed(embeddings, texts)

    monkeypatch.setattr(er, "_embed_with_retry", _note_lands_meanwhile)

    assert await _run(conn) == [f"{OPENAI.label} + {OPENAI_B.label}"]

    assert conn.updates == []
    assert conn.record["reembed_cursor"] is None
    assert conn.ran(er._FINISH_REEMBED_SQL) == []
    assert "another model wrote into the store during the run" in caplog.text
    assert pauses == []


@pytest.mark.asyncio
async def test_a_writer_already_listed_still_keeps_the_run_from_finishing(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """Its note resets only the place; the finish guard sees the reset."""
    conn = _RunConn(_rows(1), record=_row(OPENAI, mixed=(OPENAI_B,)))
    real_apply = conn._apply

    def _place_reset_at_finish(query: str, params: Any) -> tuple[list[Any], int]:
        if query == er._FINISH_REEMBED_SQL:
            conn.record["reembed_cursor"] = None
        return real_apply(query, params)

    monkeypatch.setattr(conn, "_apply", _place_reset_at_finish)

    assert await _run(conn) == [f"{OPENAI.label} + {OPENAI_B.label}"]

    assert conn.record["model"] == OPENAI.as_json().obj
    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj
    assert pauses


@pytest.mark.asyncio
async def test_a_record_unreadable_after_a_stop_brings_the_repair_back(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """Not knowing whether a run took over, the conservative stop re-raises."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 1)
    conn = _RunConn(_rows(2))
    real_apply = conn._apply
    stopped = False

    def _read_fails_after_the_stop(query: str, params: Any) -> tuple[list[Any], int]:
        nonlocal stopped
        if query == er._SAVE_CURSOR_SQL:
            conn.record["mixed_with"] = [OPENAI_B.as_json().obj]
            stopped = True
        if stopped and query == er._READ_RECORD_SQL:
            msg = "connection lost"
            raise psycopg.OperationalError(msg)
        return real_apply(query, params)

    monkeypatch.setattr(conn, "_apply", _read_fails_after_the_stop)

    assert await _run(conn) == [OPENAI.label]

    assert "the record could not be read again" in caplog.text
    assert pauses == []


@pytest.mark.asyncio
async def test_a_writer_noted_during_the_run_keeps_it_from_finishing(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """The store is not certified without a writer noted after the run began."""
    conn = _RunConn(_rows(1))
    real_apply = conn._apply

    def _noted_at_finish(query: str, params: Any) -> tuple[list[Any], int]:
        if query == er._FINISH_REEMBED_SQL:
            conn.record["mixed_with"] = [OPENAI_B.as_json().obj]
        return real_apply(query, params)

    monkeypatch.setattr(conn, "_apply", _noted_at_finish)

    await _run(conn)

    assert conn.record["model"] == OPENAI.as_json().obj
    assert conn.record["mixed_with"] == [OPENAI_B.as_json().obj]
    assert pauses


@pytest.mark.asyncio
async def test_a_run_whose_target_was_taken_at_finish_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pauses: list[float],
) -> None:
    """Another run took the target after the last batch: no false success."""
    conn = _RunConn(_rows(1))
    real_apply = conn._apply

    def _taken_at_finish(query: str, params: Any) -> tuple[list[Any], int]:
        if query == er._FINISH_REEMBED_SQL:
            conn.record["reembed_target"] = OPENAI_B.as_json().obj
        return real_apply(query, params)

    monkeypatch.setattr(conn, "_apply", _taken_at_finish)

    assert await _run(conn) == []

    assert conn.record["model"] == OPENAI.as_json().obj
    assert "another run changed its target" in caplog.text
    assert "Re-embedded the memory store" not in caplog.text
    assert pauses


@pytest.mark.asyncio
async def test_a_resume_continues_the_stored_target_and_records_it(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """Resuming a target stored with another size still finishes and records."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    stored = embedding_fingerprint(
        "ollama", "mxbai-embed-large", dims=768, digest="w:a1"
    )
    conn = _RunConn(_rows(4), record=_row(OPENAI, target=stored, cursor=CURSOR))

    await _run(conn, start=False, target=OLLAMA_A)

    assert [u[2] for u in conn.updates] == ["k002", "k003"]
    assert conn.record["model"] == stored.as_json().obj
    assert conn.record["reembed_target"] is None
    assert pauses


@pytest.mark.asyncio
async def test_a_confirm_after_a_partial_switch_back_starts_over(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """Re-embedding back to the recorded model covers the whole store."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(
        _rows(3), record=_row(OLLAMA_A, target=OPENAI, cursor=None, mixed=(OPENAI,))
    )

    await _run(conn, start=True, target=OLLAMA_A)

    assert [u[2] for u in conn.updates] == ["k000", "k002"]
    assert conn.record["model"] == OLLAMA_A.as_json().obj
    assert conn.record["mixed_with"] is None
    assert pauses


@pytest.mark.asyncio
async def test_a_store_without_a_record_is_not_reembedded(
    caplog: pytest.LogCaptureFixture, pauses: list[float]
) -> None:
    """The record row was dropped: nothing to re-embed until setup records one."""
    conn = _RunConn(_rows(2), record=_row(None))

    assert await _run(conn) == []

    assert conn.updates == []
    assert "no embedding model record" in caplog.text
    assert pauses == []


@pytest.mark.asyncio
async def test_a_lock_that_cannot_be_taken_brings_the_issue_back(
    pauses: list[float],
) -> None:
    """A database that fails at the start leaves the repair visible."""
    conn = _RunConn(_rows(2))
    real_apply = conn._apply

    def _fails(query: str, params: Any) -> tuple[list[Any], int]:
        if query == er._REEMBED_LOCK_SQL:
            msg = "terminating connection"
            raise psycopg.OperationalError(msg)
        return real_apply(query, params)

    conn._apply = _fails  # type: ignore[method-assign]

    assert await _run(conn) == [OLLAMA_A.label]
    assert pauses == []


@pytest.mark.asyncio
async def test_the_run_bounds_its_statements_and_resets_them(
    pauses: list[float],
) -> None:
    """A stalled statement ends the run; the pooled connection is reset."""
    conn = _RunConn(_rows(1))

    await _run(conn)

    names = [q for q, _ in conn.queries]
    for statement in er._RUN_TIMEOUTS_SQL:
        assert statement in names
    for statement in er._RESET_TIMEOUTS_SQL:
        assert names.index(statement) > names.index(er._RUN_TIMEOUTS_SQL[0])
    assert pauses


@pytest.mark.asyncio
async def test_a_stopped_run_after_the_unload_raises_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run failing as the pool closes must not raise an issue for the entry."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    runs: list[dict[str, Any]] = []

    async def _reembed(*_args: Any, **kwargs: Any) -> None:
        runs.append(kwargs)

    monkeypatch.setattr(er, "_async_reembed", _reembed)
    monkeypatch.setattr(
        entry,
        "async_create_background_task",
        lambda _hass, coro, **_kw: asyncio.get_running_loop().create_task(coro),
    )
    er._start_reembed_task(hass, entry, start=True)
    await entry.runtime_data.reembed_task
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    runs[0]["stopped"](OPENAI.label)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None


@pytest.mark.asyncio
async def test_a_resume_that_cannot_start_keeps_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half re-embedded store with its primary down stays visible."""
    entry = _entry(
        hass,
        record=er._Record(OPENAI, OLLAMA_A, CURSOR),
        embeddings=UnavailableEmbeddings("Ollama"),
        monkeypatch=monkeypatch,
    )

    await er.async_sync_embedding_record(hass, entry)

    issue = _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED)
    assert issue is not None
    assert issue.translation_placeholders == {
        "recorded": f"{OPENAI.label} + {OLLAMA_A.label}",
        "current": OLLAMA_A.label,
    }


@pytest.mark.parametrize("served", ["w:a1", "w:b2", None])
@pytest.mark.asyncio
async def test_the_run_reads_the_primary_build_it_finishes_on(
    hass: Any, monkeypatch: pytest.MonkeyPatch, served: str | None
) -> None:
    """The run's build check asks the primary's own server, every time."""
    entry = _entry(hass, record=None, monkeypatch=monkeypatch)
    entry.runtime_data.embedding_digest_source = ("http://ollama", "mxbai-embed-large")
    runs: list[dict[str, Any]] = []

    async def _reembed(*_args: Any, **kwargs: Any) -> None:
        runs.append(kwargs)

    monkeypatch.setattr(er, "_async_reembed", _reembed)
    monkeypatch.setattr(
        entry,
        "async_create_background_task",
        lambda _hass, coro, **_kw: asyncio.get_running_loop().create_task(coro),
    )
    fetch = AsyncMock(return_value=served)
    monkeypatch.setattr(er, "fetch_ollama_model_digest", fetch)
    er._start_reembed_task(hass, entry, start=True)
    await entry.runtime_data.reembed_task

    assert await runs[0]["current_build"]() == served
    assert fetch.await_args is not None
    assert fetch.await_args.args[1:3] == ("http://ollama", "mxbai-embed-large")


@pytest.mark.asyncio
async def test_a_resume_after_a_reset_place_covers_the_whole_store(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A place reset by another writer means every row is re-embedded."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(
        _rows(3),
        record=_row(OPENAI, target=OLLAMA_A, cursor=None, mixed=(OLLAMA_A, OPENAI_B)),
    )

    await _run(conn, start=False)

    assert [u[2] for u in conn.updates] == ["k000", "k002"]
    assert conn.record["model"] == OLLAMA_A.as_json().obj
    assert conn.record["mixed_with"] is None
    assert pauses


@pytest.mark.asyncio
async def test_a_wrong_size_after_committed_batches_keeps_the_target(
    pauses: list[float],
) -> None:
    """Rows already re-embedded keep their run: its target is not dropped."""
    conn = _RunConn(_rows(4), record=_row(OPENAI, target=OLLAMA_A, cursor=CURSOR))
    conn.fail_update = psycopg.DataError("expected 1024 dimensions, not 768")

    assert await _run(conn, start=False) == [f"{OPENAI.label} + {OLLAMA_A.label}"]

    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj
    assert conn.record["reembed_cursor"] == CURSOR
    assert pauses == []


def test_a_whole_document_vector_is_read_back() -> None:
    """A "$" field is the whole item, never mistaken for missing text."""
    assert text_at_field({"content": "x"}, "$") == '{"content": "x"}'


WEIGHTS = "a" * 64


@pytest.mark.parametrize(
    ("modelfile", "expected"),
    [
        # The weights blob /api/show's modelfile names.
        (f"# Modelfile\nFROM /root/.ollama/models/blobs/sha256-{WEIGHTS}\n", WEIGHTS),
        # A Windows server writes the path with backslashes.
        (f"FROM C:\\Users\\me\\.ollama\\models\\blobs\\sha256-{WEIGHTS}", WEIGHTS),
        # No weights line: unknown, never another build.
        ("FROM mxbai-embed-large:latest", None),
        (None, None),
    ],
)
@pytest.mark.asyncio
async def test_the_build_digest_is_the_weights_blob(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    modelfile: str | None,
    expected: str | None,
) -> None:
    """Only the weights decide the vectors; the manifest digest is not used."""
    shown = {"modelfile": modelfile} if modelfile is not None else None
    monkeypatch.setattr(utils, "_ollama_json", AsyncMock(return_value=shown))

    assert await utils.fetch_ollama_model_digest(hass, "u", "m") == expected


@pytest.mark.parametrize(
    ("a", "b", "differ"),
    [
        ("1", "2", True),
        ("1", "1", False),
        (None, "2", False),
        ("1", None, False),
    ],
)
def test_only_two_known_digests_can_differ(
    a: str | None,
    b: str | None,
    differ: bool,  # noqa: FBT001
) -> None:
    """An unknown or other-kind digest never stands in for another build."""
    assert utils.builds_differ(a, b) is differ


@pytest.mark.asyncio
async def test_servers_without_a_digest_are_left_out_of_the_map(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each URL keeps its own digest; one with none is not in the map."""

    async def _digest(_hass: Any, url: str, _model: str, _timeout: float) -> Any:
        return {"u1": "w:a"}.get(url)

    monkeypatch.setattr(utils, "fetch_ollama_model_digest", _digest)

    assert await utils.ollama_model_digests(hass, ["u1", "u2"], "m") == {"u1": "w:a"}


# --- Setup -----------------------------------------------------------------


class _SetupPool:
    """A pool every query succeeds on, answering nothing."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.closed = False

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncGenerator[Any]:
        @contextlib.asynccontextmanager
        async def _cursor(**_kwargs: Any) -> AsyncGenerator[Any]:
            yield SimpleNamespace(
                execute=AsyncMock(), fetchone=AsyncMock(return_value=None)
            )

        @contextlib.asynccontextmanager
        async def _transaction() -> AsyncGenerator[None]:
            yield

        yield SimpleNamespace(cursor=_cursor, transaction=_transaction)


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _with_database(
    monkeypatch: pytest.MonkeyPatch,
    digests: dict[str, str],
    *,
    recorded: str | None = None,
    record_check: bool = False,
) -> list[Any]:
    """
    Give setup a database that accepts everything, and these digests.

    `recorded` is the build the store's record names. The returned list gets
    each digest request and, unless `record_check` runs the real one, each
    record check setup schedules as ("record", fingerprint).
    """
    asked: list[Any] = []

    async def _digests(
        _hass: Any, urls: list[str], model: str, **kwargs: Any
    ) -> dict[str, str]:
        asked.append((urls, model))
        asked.append(("timeout", kwargs.get("timeout_s")))
        return {url: digests[url] for url in urls if url in digests}

    async def _recorded(_db_uri: str, model: str) -> str | None:
        asked.append(("recorded", model))
        return recorded

    async def _record_check(_hass: Any, entry: Any) -> None:
        asked.append(("record", entry.runtime_data.embedding_fingerprint))

    monkeypatch.setattr(
        hga_component, "build_database_uri_from_entry", lambda _e: "postgresql://x/y"
    )
    monkeypatch.setattr(hga_component, "AsyncConnectionPool", _SetupPool)
    monkeypatch.setattr(hga_component, "AsyncPostgresStore", MagicMock())
    monkeypatch.setattr(hga_component, "AsyncPostgresSaver", MagicMock())
    monkeypatch.setattr(hga_component, "PersonGalleryDAO", MagicMock())
    for name in (
        "async_ensure_pgvector",
        "_bootstrap_db_once",
        "_bootstrap_vectors_once",
        "migrate_person_gallery",
        "async_sync_embedding_size_issue",
        "async_sync_ollama_embedding_model_issue",
    ):
        monkeypatch.setattr(hga_component, name, _noop)
    monkeypatch.setattr(hga_component, "ollama_model_digests", _digests)
    monkeypatch.setattr(hga_component, "async_recorded_ollama_digest", _recorded)
    if not record_check:
        monkeypatch.setattr(hga_component, "async_sync_embedding_record", _record_check)
    return asked


async def _setup_two_ollama_servers(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    digests: dict[str, str],
    *,
    recorded: str | None = None,
    primary_up: bool = True,
    record_check: bool = False,
    entry: MockConfigEntry | None = None,
    database: bool = True,
    primary_client: bool = True,
    openai_fallback: bool = False,
) -> tuple[MockConfigEntry, Any, list[Any]]:
    """
    Set up an Ollama primary with a second Ollama server as embedding fallback.

    With `record_check` the real background check runs, which needs a loaded
    entry (setup is called directly here, so the state is set by hand).
    Without `database` no build is read and nothing is handed to the check.
    """
    if entry is None:
        entry = MockConfigEntry(domain=DOMAIN, data={})
        entry.add_to_hass(hass)
    if record_check:
        entry.mock_state(hass, ConfigEntryState.LOADED)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_EMBEDDING_MODEL_PROVIDER] = "ollama"
    data.options[CONF_OLLAMA_EMBEDDING_MODEL] = "configured-embed"
    second = _second_ollama(FALLBACK_URL)
    data.providers[second.entry_id] = second
    data.fallback_chains["embedding"] = [data.providers["ollama1"], second]
    if openai_fallback:
        data.fallback_chains["embedding"] = [
            data.providers["ollama1"],
            data.providers["openai1"],
        ]
    _patch_setup_dependencies(hass, monkeypatch, data)
    monkeypatch.setattr(hga_component, "configured_ollama_urls", configured_ollama_urls)

    def _client(*_args: Any, model: str, base_url: str, **_kwargs: Any) -> Any:
        if not primary_client and base_url == "http://ollama":
            msg = "bad client"
            raise ValueError(msg)
        return FakeEmbeddings(f"{model}@{base_url}")

    monkeypatch.setattr(hga_component, "OllamaEmbeddings", _client)
    _capture_recheck(monkeypatch)
    if not primary_up:

        async def _primary_down(_hass: Any, url: str, **_kwargs: Any) -> bool:
            return url != "http://ollama"

        monkeypatch.setattr(hga_component, "ollama_healthy", _primary_down)
    asked: list[Any] = []
    if database:
        asked = _with_database(
            monkeypatch, digests, recorded=recorded, record_check=record_check
        )
    else:
        monkeypatch.setattr(
            hga_component, "ollama_model_digests", AsyncMock(return_value={})
        )

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)
    # The real check is an entry background task, which the plain wait skips.
    await hass.async_block_till_done(wait_background_tasks=record_check)
    return entry, data, asked


@pytest.mark.asyncio
async def test_a_fallback_holding_another_build_is_left_out(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another digest of the model on the fallback server: out, with an issue."""
    entry, data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1", FALLBACK_URL: "w:b2"}
    )

    assert asked[0] == (["http://ollama", FALLBACK_URL], "configured-embed")
    # The primary's build is known: the record is not read for it.
    assert ("recorded", "configured-embed") not in asked
    assert "chain" not in data.captured_embedding_chain
    assert entry.runtime_data.embeddings.name == "configured-embed@http://ollama"
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "Ollama Secondary",
        "primary": "Ollama",
        "model": "configured-embed",
    }
    assert entry.runtime_data.embedding_fingerprint == embedding_fingerprint(
        "ollama", "configured-embed", dims=1024, digest="w:a1"
    )


@pytest.mark.asyncio
async def test_a_fallback_with_the_same_or_an_unknown_build_is_kept(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown digest on either side never leaves a fallback out."""
    entry, data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1"}
    )

    chain = data.captured_embedding_chain["chain"]
    assert len(chain) == 2
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None


@pytest.mark.asyncio
async def test_setup_schedules_the_record_check_with_the_current_model(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the check, nothing records the store's model or raises the issue."""
    _entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1"}
    )

    assert (
        "record",
        embedding_fingerprint("ollama", "configured-embed", dims=1024, digest="w:a1"),
    ) in asked


@pytest.mark.asyncio
async def test_with_the_primary_down_a_fallback_is_judged_by_the_record(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback writes while the primary is down: its build must match."""
    entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:b2"}, recorded="w:a1", primary_up=False
    )

    assert ("recorded", "configured-embed") in asked
    assert isinstance(entry.runtime_data.embeddings, UnavailableEmbeddings)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    # The primary's build is unread: the record check must wait, not match.
    assert entry.runtime_data.embedding_fingerprint.undetermined


def _assert_held_alone(entry: Any) -> None:
    """Assert the secondary server is the whole chain, held until checked."""
    chain = entry.runtime_data.embeddings
    assert isinstance(chain, FallbackEmbeddings)
    [(client, _deployment, provider_id)] = chain.chain
    assert client.name == f"configured-embed@{FALLBACK_URL}"
    assert chain.unverified == {provider_id}


@pytest.mark.asyncio
async def test_with_the_primary_down_a_matching_fallback_takes_over(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A fallback holding the recorded build keeps embedding failover.

    It is the only member, but wrapped so it is held until its build is
    checked again on its first call (#857).
    """
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:a1"}, recorded="w:a1", primary_up=False
    )

    _assert_held_alone(entry)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None


@pytest.mark.asyncio
async def test_setup_reads_builds_with_the_digest_timeout_not_the_health_checks(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2 s health timeout ran past on a busy startup and left the gate blind."""
    _entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1"}
    )

    assert ("timeout", er.DIGEST_FETCH_TIMEOUT_S) in asked
    # Quoted in docs/configuration.md; the health check waits 2 s.
    assert er.DIGEST_FETCH_TIMEOUT_S == 5.0


@pytest.mark.parametrize(
    ("digests", "unchecked", "left_out"),
    [
        # The fallback's build unread: kept, to be read again.
        (
            {"http://ollama": "w:a1"},
            (FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, None),),
            (),
        ),
        # The primary's build unread: the fallback's was judged by the record
        # (none here), so it is compared with the primary's once that is read.
        (
            {FALLBACK_URL: "w:b2"},
            (FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, "w:b2"),),
            (),
        ),
        # Both read and the same: fully checked.
        ({"http://ollama": "w:a1", FALLBACK_URL: "w:a1"}, (), ()),
        # Both read and different: left out, nothing to check again.
        ({"http://ollama": "w:a1", FALLBACK_URL: "w:b2"}, (), ("Ollama Secondary",)),
        # Neither read (the busy boot #860 is about): kept, both re-read later.
        ({}, (FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, None),), ()),
        # The primary's unread, the fallback's read: still handed to the check.
        (
            {FALLBACK_URL: "w:a1"},
            (FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, "w:a1"),),
            (),
        ),
    ],
)
@pytest.mark.asyncio
async def test_setup_hands_the_fallbacks_it_could_not_fully_check_to_the_record_check(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    digests: dict[str, str],
    unchecked: tuple[FallbackBuild, ...],
    left_out: tuple[str, ...],
) -> None:
    """
    A kept fallback not compared with the primary's live build is checked later.

    Until then the chain holds it: it answers nothing before its build is
    checked (#857).
    """
    entry, _data, _asked = await _setup_two_ollama_servers(hass, monkeypatch, digests)

    assert entry.runtime_data.embedding_fallback_gate == EmbeddingFallbackGate(
        primary="Ollama",
        model="configured-embed",
        left_out=left_out,
        unchecked=unchecked,
    )
    chain = entry.runtime_data.embeddings
    if left_out:
        # The fallback is out: the primary alone, unwrapped.
        assert not isinstance(chain, FallbackEmbeddings)
        return
    assert isinstance(chain, FallbackEmbeddings)
    assert chain.unverified == {fb.provider_id for fb in unchecked}


@pytest.mark.asyncio
async def test_with_the_primarys_build_unread_a_fallback_unlike_the_record_waits(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Primary up but unread: the fallback waits for the primary's build.

    The record may be stale after a pull on both servers, so the fallback is
    compared with the primary's build once read, not left out with an issue
    blaming the primary.
    """
    entry, data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:b2"}, recorded="w:a1"
    )

    assert ("recorded", "configured-embed") not in asked
    assert len(data.captured_embedding_chain["chain"]) == 2
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None
    assert entry.runtime_data.embedding_fallback_gate == EmbeddingFallbackGate(
        primary="Ollama",
        model="configured-embed",
        left_out=(),
        unchecked=(FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, "w:b2"),),
    )


@pytest.mark.asyncio
async def test_with_the_primary_down_an_unread_fallback_takes_over_unchecked(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Unknown decides nothing at setup: the fallback is the chain, held (#857).

    It answers only once its build is read and matches the record's, so an
    unread build no longer embeds unchecked.
    """
    entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, recorded="w:a1", primary_up=False
    )

    assert ("recorded", "configured-embed") not in asked
    _assert_held_alone(entry)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == (
        FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, None),
    )
    assert entry.runtime_data.primary_embeddings is None


@pytest.mark.asyncio
async def test_without_a_database_a_kept_fallback_is_not_handed_to_the_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No store to protect: both servers embed, nothing is re-read later."""
    entry, data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, database=False
    )

    assert len(data.captured_embedding_chain["chain"]) == 2
    assert entry.runtime_data.embedding_fallback_gate == EmbeddingFallbackGate(
        primary="Ollama", model="configured-embed", left_out=(), unchecked=()
    )


@pytest.mark.asyncio
async def test_setup_with_two_secondaries_syncs_the_one_it_left_out(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read as another build, one unread: the issue names the first now."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_EMBEDDING_MODEL_PROVIDER] = "ollama"
    data.options[CONF_OLLAMA_EMBEDDING_MODEL] = "configured-embed"
    second = _second_ollama(FALLBACK_URL)
    third = replace(
        second,
        entry_id="ollama3",
        name="Ollama Third",
        data={"settings": {"base_url": THIRD_URL}},
    )
    data.providers[second.entry_id] = second
    data.providers[third.entry_id] = third
    data.fallback_chains["embedding"] = [data.providers["ollama1"], second, third]
    _patch_setup_dependencies(hass, monkeypatch, data)
    monkeypatch.setattr(hga_component, "configured_ollama_urls", configured_ollama_urls)
    monkeypatch.setattr(
        hga_component,
        "OllamaEmbeddings",
        lambda *_args, model, base_url, **_kwargs: FakeEmbeddings(
            f"{model}@{base_url}"
        ),
    )
    _capture_recheck(monkeypatch)
    _with_database(monkeypatch, {"http://ollama": "w:a1", FALLBACK_URL: "w:b2"})

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)
    await hass.async_block_till_done()

    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "Ollama Secondary",
        "primary": "Ollama",
        "model": "configured-embed",
    }
    assert entry.runtime_data.embedding_fallback_gate == EmbeddingFallbackGate(
        primary="Ollama",
        model="configured-embed",
        left_out=("Ollama Secondary",),
        unchecked=(FallbackBuild("ollama3", "Ollama Third", THIRD_URL, None),),
    )


@pytest.mark.asyncio
async def test_with_the_primary_down_an_unread_fallback_keeps_a_standing_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No check will run, and nothing is known: the last boot's warning stands."""
    stale = MockConfigEntry(domain=DOMAIN, data={})
    stale.add_to_hass(hass)
    _standing_issue(hass, stale, ["Ollama Secondary"])

    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, recorded="w:a1", primary_up=False, entry=stale
    )

    _assert_held_alone(entry)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)


@pytest.mark.asyncio
async def test_with_the_primary_down_and_no_record_a_read_fallback_takes_over(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Nothing recorded to compare with: setup keeps it, held (#857).

    With no build to compare with, it does not stand in until one is known:
    its vectors could not be told apart from the primary's later.
    """
    entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:b2"}, recorded=None, primary_up=False
    )

    assert ("recorded", "configured-embed") in asked
    _assert_held_alone(entry)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None
    assert entry.runtime_data.embedding_fallback_gate.unchecked == (
        FallbackBuild("ollama2", "Ollama Secondary", FALLBACK_URL, "w:b2"),
    )


@pytest.mark.asyncio
async def test_a_primary_whose_client_failed_is_stood_in_for_by_the_record(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Healthy server, no client: the fallback writes alone, so the record judges."""
    entry, _data, asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:b2"}, recorded="w:a1", primary_client=False
    )

    assert ("recorded", "configured-embed") in asked
    assert isinstance(entry.runtime_data.embeddings, UnavailableEmbeddings)
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_a_non_ollama_fallback_is_never_handed_to_the_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only another Ollama server has a build to read."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, openai_fallback=True
    )

    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_a_dismissed_issue_survives_the_check_that_names_it_again(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settle updates the standing issue in place, so a dismissal holds."""
    registry = ir.async_get(hass)
    stale = MockConfigEntry(domain=DOMAIN, data={})
    stale.add_to_hass(hass)
    _standing_issue(hass, stale, ["Ollama Secondary"])
    issue_id = er._other_build_issue_id(stale.entry_id)
    registry.async_ignore(DOMAIN, issue_id, ignore=True)
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value="w:b2"))

    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1"}, record_check=True, entry=stale
    )

    assert _chain_ids(entry.runtime_data.embeddings) == ["ollama1"]
    standing = registry.async_get_issue(DOMAIN, issue_id)
    assert standing is not None
    assert standing.translation_placeholders == {
        "providers": "Ollama Secondary",
        "primary": "Ollama",
        "model": "configured-embed",
    }
    assert standing.dismissed_version is not None


@pytest.mark.asyncio
async def test_setup_and_the_record_check_remove_an_unread_other_build_end_to_end(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup's provider ids are the chain's: the real check finds and drops it."""
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value="w:b2"))
    notices: list[dict[str, Any]] = []
    _register_capture_service(hass, "persistent_notification", "create", notices)
    entry, data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1"}, record_check=True
    )

    chain = entry.runtime_data.embeddings
    assert isinstance(chain, FallbackEmbeddings)
    assert _chain_ids(chain) == ["ollama1"]
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    assert issue is not None
    assert issue.translation_placeholders == {
        "providers": "Ollama Secondary",
        "primary": "Ollama",
        "model": "configured-embed",
    }
    # Removing an active fallback marks the tool index stale without the
    # "fell back" notice a real switch sends.
    entry.runtime_data.tool_index_ready = True
    data.captured_embedding_kwargs["removed_callback"]("ollama2", "ollama1")
    await hass.async_block_till_done()
    assert entry.runtime_data.tool_index_ready is False
    assert notices == []


@pytest.mark.asyncio
async def test_setup_leaves_a_standing_issue_to_the_record_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dismissed issue is not deleted at setup only to come back fresh (#860)."""
    registry = ir.async_get(hass)
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    er.async_sync_other_build_fallback_issue(
        hass,
        entry.entry_id,
        primary="Ollama",
        model="configured-embed",
        left_out=["Ollama Secondary"],
    )
    issue_id = er._other_build_issue_id(entry.entry_id)
    registry.async_ignore(DOMAIN, issue_id, ignore=True)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_EMBEDDING_MODEL_PROVIDER] = "ollama"
    data.options[CONF_OLLAMA_EMBEDDING_MODEL] = "configured-embed"
    second = _second_ollama(FALLBACK_URL)
    data.providers[second.entry_id] = second
    data.fallback_chains["embedding"] = [data.providers["ollama1"], second]
    _patch_setup_dependencies(hass, monkeypatch, data)
    monkeypatch.setattr(hga_component, "configured_ollama_urls", configured_ollama_urls)
    monkeypatch.setattr(
        hga_component,
        "OllamaEmbeddings",
        lambda *_args, model, base_url, **_kwargs: FakeEmbeddings(
            f"{model}@{base_url}"
        ),
    )
    _capture_recheck(monkeypatch)
    # The fallback's build is unread at setup: the check will judge it.
    _with_database(monkeypatch, {"http://ollama": "w:a1"})

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)
    await hass.async_block_till_done()

    standing = registry.async_get_issue(DOMAIN, issue_id)
    assert standing is not None
    assert standing.dismissed_version is not None


@pytest.mark.asyncio
async def test_setup_syncs_the_issue_itself_when_nothing_is_checked_later(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fully checked at setup, or primary down: setup's sync is the only one."""
    stale = MockConfigEntry(domain=DOMAIN, data={})
    stale.add_to_hass(hass)
    _standing_issue(hass, stale, ["Ollama Secondary"])
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass,
        monkeypatch,
        {"http://ollama": "w:a1", FALLBACK_URL: "w:a1"},
        entry=stale,
    )
    assert entry is stale
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None

    down, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:b2"}, recorded="w:a1", primary_up=False
    )
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, down.entry_id)


@pytest.mark.parametrize(
    ("provider_type", "key"),
    [
        ("openai", hga_component.CONF_OPENAI_EMBEDDING_MODEL),
        ("gemini", hga_component.CONF_GEMINI_EMBEDDING_MODEL),
        ("openai_compatible", hga_component.CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL),
        ("ollama", hga_component.CONF_OLLAMA_EMBEDDING_MODEL),
    ],
)
def test_each_provider_type_reads_its_own_embedding_model(
    provider_type: str, key: str
) -> None:
    """A swapped key would record the wrong model and misfire the issue."""
    others = {
        hga_component.CONF_OPENAI_EMBEDDING_MODEL: "openai-model",
        hga_component.CONF_GEMINI_EMBEDDING_MODEL: "gemini-model",
        hga_component.CONF_OPENAI_COMPATIBLE_EMBEDDING_MODEL: "compatible-model",
        hga_component.CONF_OLLAMA_EMBEDDING_MODEL: "ollama-model",
    }

    assert (
        hga_component._configured_embedding_model(others, provider_type)
        == (others[key])
    )


@pytest.mark.parametrize(
    ("provider_type", "default"),
    [
        ("openai", hga_component.RECOMMENDED_OPENAI_EMBEDDING_MODEL),
        ("gemini", hga_component.RECOMMENDED_GEMINI_EMBEDDING_MODEL),
        (
            "openai_compatible",
            hga_component.RECOMMENDED_OPENAI_COMPATIBLE_EMBEDDING_MODEL,
        ),
        ("ollama", hga_component.RECOMMENDED_OLLAMA_EMBEDDING_MODEL),
    ],
)
def test_each_provider_type_defaults_to_its_recommended_model(
    provider_type: str, default: str
) -> None:
    """With nothing configured, each type records its own default model."""
    assert hga_component._configured_embedding_model({}, provider_type) == default


class _DigestConn:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    @contextlib.asynccontextmanager
    async def cursor(self, **_kwargs: Any) -> AsyncGenerator[Any]:
        yield SimpleNamespace(
            execute=AsyncMock(), fetchone=AsyncMock(return_value=self.row)
        )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"model": OLLAMA_A.as_json().obj}, "w:a1"),
        # Another model's build says nothing about this one.
        (
            {
                "model": embedding_fingerprint(
                    "ollama", "nomic-embed-text", dims=None, digest="w:n1"
                )
                .as_json()
                .obj
            },
            None,
        ),
        ({"model": OPENAI.as_json().obj}, None),
        (None, None),
    ],
)
@pytest.mark.asyncio
async def test_the_recorded_build_is_read_for_the_configured_model(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any] | None, expected: str | None
) -> None:
    """Only the configured Ollama model's recorded build counts."""

    async def _connect(*_args: Any, **_kwargs: Any) -> _DigestConn:
        return _DigestConn(row)

    monkeypatch.setattr(er.psycopg.AsyncConnection, "connect", _connect)

    assert (
        await er.async_recorded_ollama_digest("postgresql://x/y", "mxbai-embed-large")
        == expected
    )


@pytest.mark.asyncio
async def test_an_unreachable_database_has_no_recorded_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database that does not answer decides nothing."""
    monkeypatch.setattr(
        er.psycopg.AsyncConnection,
        "connect",
        AsyncMock(side_effect=psycopg.OperationalError("down")),
    )

    assert (
        await er.async_recorded_ollama_digest("postgresql://x/y", "mxbai-embed-large")
        is None
    )


@pytest.mark.asyncio
async def test_setup_without_a_database_asks_no_digest(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing embeds without a database, so no server is asked."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    data.options[CONF_EMBEDDING_MODEL_PROVIDER] = "ollama"
    _patch_setup_dependencies(hass, monkeypatch, data)
    asked = AsyncMock(return_value={})
    monkeypatch.setattr(hga_component, "ollama_model_digests", asked)

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)

    asked.assert_not_called()
    fingerprint = entry.runtime_data.embedding_fingerprint
    assert fingerprint is not None
    assert fingerprint.digest is None
    # No store to protect: nothing is handed to the record check either.
    assert entry.runtime_data.embedding_fallback_gate.unchecked == ()


@pytest.mark.asyncio
async def test_a_non_ollama_primary_is_fingerprinted_by_its_model(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenAI's configured embedding model goes into the fingerprint."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})["http_registered"] = True
    data = _fallback_setup_data()
    _patch_setup_dependencies(hass, monkeypatch, data)

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)

    fingerprint = entry.runtime_data.embedding_fingerprint
    assert fingerprint is not None
    assert (fingerprint.provider_type, fingerprint.digest) == ("openai", None)
    assert fingerprint.model == hga_component.RECOMMENDED_OPENAI_EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_unload_cancels_the_reembed_and_a_disable_drops_the_issues(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run stops before the pool closes; a reload keeps the issues."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1", FALLBACK_URL: "w:b2"}
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        f"{ISSUE_EMBEDDING_MODEL_CHANGED}_{entry.entry_id}",
        is_fixable=True,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_EMBEDDING_MODEL_CHANGED,
    )
    _raise_held_issue(hass, entry.entry_id)
    run = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
    entry.runtime_data.reembed_task = run
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    assert await hga_component.async_unload_entry(hass, entry)

    assert run.cancelled()
    assert entry.runtime_data.reembed_task is None
    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED, entry.entry_id) is not None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id)
    # A reload keeps a dismissal of the held-fallback issue too (#873, #810).
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_HELD, entry.entry_id) is not None

    monkeypatch.setattr(entry, "disabled_by", "user")
    assert await hga_component.async_unload_entry(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED, entry.entry_id) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_HELD, entry.entry_id) is None


@pytest.mark.asyncio
async def test_a_setup_with_nothing_to_hold_drops_a_held_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both builds read and the same at setup: no fallback can be held (#873)."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    _raise_held_issue(hass, entry.entry_id)

    await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1", FALLBACK_URL: "w:a1"}, entry=entry
    )

    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_HELD, entry.entry_id) is None


@pytest.mark.asyncio
async def test_setup_starts_the_caption_backfill_and_unload_cancels_it(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It runs on the pool, so it stops before the pool closes (#706)."""
    started: list[Any] = []

    async def _backfill(pool: Any) -> None:
        started.append(pool)
        await asyncio.sleep(3600)

    monkeypatch.setattr(hga_component, "async_backfill_caption_cameras", _backfill)
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {"http://ollama": "w:a1", FALLBACK_URL: "w:a1"}
    )
    await asyncio.sleep(0)
    run = entry.runtime_data.caption_backfill_task
    assert run is not None
    assert started == [entry.runtime_data.pool]
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    assert await hga_component.async_unload_entry(hass, entry)

    assert run.cancelled()
    assert entry.runtime_data.caption_backfill_task is None


@pytest.mark.asyncio
async def test_no_database_starts_no_caption_backfill(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a pool there are no stored captions to backfill."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, database=False
    )

    assert entry.runtime_data.caption_backfill_task is None


@pytest.mark.asyncio
async def test_removal_drops_the_issues(hass: Any) -> None:
    """A removed entry leaves neither issue behind."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    for key in (
        ISSUE_EMBEDDING_MODEL_CHANGED,
        ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD,
        ISSUE_EMBEDDING_FALLBACK_HELD,
    ):
        ir.async_create_issue(
            hass,
            DOMAIN,
            f"{key}_entry1",
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key=key,
        )

    await hga_component.async_remove_entry(hass, cast("Any", entry))

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_HELD) is None


# --- A fallback held until its build is checked (#857) ----------------------


class _DownEmbeddings(FakeEmbeddings):
    """An embedding server that times out on every call."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise TimeoutError

    async def aembed_query(self, text: str) -> list[float]:
        raise TimeoutError


def _held_chain(verdicts: list[bool]) -> tuple[FallbackEmbeddings, AsyncMock]:
    """Build a chain with its primary down and its fallback held on `verdicts`."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    verifier = AsyncMock(side_effect=verdicts)
    chain.hold_until_verified(["p2", "not-a-member"], verifier)
    return chain, verifier


@pytest.mark.asyncio
async def test_a_held_fallback_answers_once_its_build_is_admitted() -> None:
    """Checked on its first call, then admitted for good."""
    chain, verifier = _held_chain([True])
    assert chain.unverified == {"p2"}

    assert await chain.aembed_documents(["abc"]) == [[3.0]]
    assert await chain.aembed_query("ab") == [2.0]

    verifier.assert_awaited_once_with("p2")
    assert chain.unverified == frozenset()
    assert "p2" in chain.served


@pytest.mark.asyncio
async def test_a_held_fallback_not_admitted_answers_nothing_for_a_while(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused: the call fails naming it, and it is not read again at once."""
    chain, verifier = _held_chain([False, True])
    clock = [100.0]
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.fallback.monotonic",
        lambda: clock[0],
    )

    with pytest.raises(Exception, match="All embedding providers failed") as err:
        await chain.aembed_documents(["abc"])
    # A member held: unavailable, so memory search uses recent memories.
    assert isinstance(err.value, EmbeddingsUnavailableError)
    with pytest.raises(Exception, match="All embedding providers failed"):
        await chain.aembed_query("abc")
    assert verifier.await_count == 1
    assert "p2" not in chain.served

    clock[0] += 61.0
    assert await chain.aembed_query("abc") == [3.0]
    assert verifier.await_count == 2


@pytest.mark.asyncio
async def test_a_chain_of_held_members_alone_says_why_nothing_was_tried() -> None:
    """
    With no other member, the error names the held one.

    It is the unavailable error, so memory search falls back to recent
    memories and a re-embed stops with every vector kept.
    """
    chain = FallbackEmbeddings([(FakeEmbeddings("f"), "edge", "p2")])
    chain.hold_until_verified(["p2"], AsyncMock(return_value=False))

    with pytest.raises(EmbeddingsUnavailableError, match="not yet known") as err:
        await chain.aembed_documents(["abc"])
    assert "p2" in str(err.value)
    with pytest.raises(EmbeddingsUnavailableError, match="not yet known"):
        await chain.aembed_query("abc")


@pytest.mark.asyncio
async def test_concurrent_calls_read_a_held_build_once() -> None:
    """Calls arriving together wait on one check."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    gate = asyncio.Event()
    calls = 0

    async def _verify(_provider_id: str) -> bool:
        nonlocal calls
        calls += 1
        await gate.wait()
        return True

    chain.hold_until_verified(["p2"], _verify)
    pending = [asyncio.create_task(chain.aembed_query("ab")) for _ in range(3)]
    await _REAL_SLEEP(0)
    gate.set()

    assert await asyncio.gather(*pending) == [[2.0]] * 3
    assert calls == 1


@pytest.mark.asyncio
async def test_a_held_fallback_removed_during_its_check_is_not_admitted() -> None:
    """The verifier removed it for another build: nothing answers."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )

    async def _other_build(provider_id: str) -> bool:
        chain.remove_provider(provider_id)
        return False

    chain.hold_until_verified(["p2"], _other_build)

    with pytest.raises(Exception, match="All embedding providers failed"):
        await chain.aembed_documents(["abc"])
    assert _chain_ids(chain) == ["p1"]
    assert chain.unverified == frozenset()


def _held_in_entry(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fallback_build: str | None,
    fingerprint: EmbeddingFingerprint = OLLAMA_A,
    recorded: str | None = None,
    read_at_setup: str | None = None,
) -> tuple[MockConfigEntry, FallbackEmbeddings, dict[str, FallbackBuild]]:
    """Set up an entry whose chain holds the secondary server, for the verifier."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass,
        monkeypatch,
        [fallback_build],
        fingerprint=fingerprint,
        read_at_setup=read_at_setup,
    )
    monkeypatch.setattr(er, "_async_recorded_build", AsyncMock(return_value=recorded))
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))
    fallbacks = {
        fb.provider_id: fb
        for fb in entry.runtime_data.embedding_fallback_gate.unchecked
    }
    return entry, chain, fallbacks


@pytest.mark.asyncio
async def test_the_verifier_admits_the_primarys_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read now and found the same as the primary's: admitted, nothing raised."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:a1")

    assert await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)

    er._async_recorded_build.assert_not_awaited()  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_the_verifier_removes_another_build_and_names_it(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Another build than the store's: out of the chain, in the issue."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:b2")

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )

    assert _chain_ids(chain) == ["p1"]
    assert entry.runtime_data.embedding_fallback_gate.left_out == (SECONDARY,)
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert (issue.translation_placeholders or {})["providers"] == SECONDARY
    assert f"Embedding fallback provider {SECONDARY} removed" in caplog.text


@pytest.mark.asyncio
async def test_the_verifier_compares_with_the_record_while_the_primarys_is_unknown(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primary is down and unread: the store's recorded build decides."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build="w:a1",
        fingerprint=OLLAMA_UNREAD,
        recorded="w:a1",
    )

    assert await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)
    er._async_recorded_build.assert_awaited_once()  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]


@pytest.mark.parametrize(
    ("fallback_build", "recorded", "read_at_setup", "admitted"),
    [
        (None, "w:a1", None, False),  # its build unread now and at setup
        ("w:a1", None, None, False),  # nothing to compare with
        # Setup's read never stands in for a failed one: the build may have
        # changed since (#857).
        (None, "w:a1", "w:a1", False),
    ],
    ids=["unread", "no_reference", "setup_read_is_not_evidence"],
)
@pytest.mark.asyncio
async def test_the_verifier_needs_both_builds(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fallback_build: str | None,
    recorded: str | None,
    read_at_setup: str | None,
    admitted: bool,  # noqa: FBT001
) -> None:
    """Unknown decides nothing, so it keeps the fallback held, not serving."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build=fallback_build,
        fingerprint=OLLAMA_UNREAD,
        recorded=recorded,
        read_at_setup=read_at_setup,
    )

    assert (
        await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)
        is admitted
    )
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None
    if not admitted:
        assert f"{SECONDARY} does not stand in for the primary" in caplog.text


@pytest.mark.asyncio
async def test_the_verifier_admits_nothing_for_an_unloaded_entry(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call after the unload reads nothing and changes nothing."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:b2")
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    er.fetch_ollama_model_digest.assert_not_awaited()  # type: ignore[attr-defined]
    assert _chain_ids(chain) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_the_record_check_admits_a_held_fallback_of_the_same_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read as the primary's build after setup: no longer held."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:a1"])
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == frozenset()


@pytest.mark.asyncio
async def test_the_record_check_keeps_an_unread_fallback_held(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Still unread after every try: held, checked again when the chain needs it."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None]
    )
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == {FALLBACK_ID}
    assert "only once their build is read and matches" in caplog.text


@pytest.mark.asyncio
async def test_the_record_check_keeps_a_left_out_the_chain_found_meanwhile(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removal the chain made while the check ran is not written over."""
    entry, _chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, ["w:a1"])
    real_read = er._async_read_fallback_builds

    async def _read_while_the_chain_removes(*args: Any) -> Any:
        gate = entry.runtime_data.embedding_fallback_gate
        entry.runtime_data.embedding_fallback_gate = replace(
            gate, left_out=("Ollama Third",)
        )
        return await real_read(*args)

    monkeypatch.setattr(
        er, "_async_read_fallback_builds", _read_while_the_chain_removes
    )

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fallback_gate.left_out == ("Ollama Third",)
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert (issue.translation_placeholders or {})["providers"] == "Ollama Third"


# --- A text the provider always rejects (#857) -------------------------------


class _RejectsOne(FakeEmbeddings):
    """Rejects any call that holds `bad`; with `down`, every call fails."""

    def __init__(self, bad: str) -> None:
        super().__init__("picky")
        self.bad = bad
        self.down = False

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.down:
            msg = "connection refused"
            raise ConnectionError(msg)
        if self.bad in texts:
            raise _StatusError(400)
        return await super().aembed_documents(texts)


@pytest.mark.asyncio
async def test_a_rejected_text_no_longer_blocks_the_run(
    monkeypatch: pytest.MonkeyPatch,
    pauses: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its old vector is dropped, the others are re-embedded, the run finishes."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))

    assert await _run(conn, _RejectsOne("memory 2")) == []

    assert [u[2] for u in conn.updates] == ["k000", "k003"]
    assert ("memories.u1", "k002", "content", f"{STAMP}#2") in conn.deletes
    assert conn.record["model"] == OLLAMA_A.as_json().obj
    assert "rejected 1 stored text(s)" in caplog.text
    assert "memories.u1/k002" in caplog.text
    # An input error: no batch retries, the one-at-a-time pass ran at once.
    assert er._REJECT_RECHECK_S in pauses


@pytest.mark.asyncio
async def test_a_provider_down_for_every_text_still_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """The probe fails too: nothing is dropped and the repair comes back."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _RejectsOne("memory 2")
    embeddings.down = True

    stopped = await _run(conn, embeddings)

    assert stopped
    assert conn.updates == []
    assert conn.deletes == []
    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj
    assert pauses == list(er._RETRY_DELAYS_S)


@pytest.mark.asyncio
async def test_many_rejected_texts_are_named_up_to_a_limit(
    monkeypatch: pytest.MonkeyPatch,
    pauses: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The log stays readable when a whole batch is rejected; all are counted."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    monkeypatch.setattr(er, "_REJECTED_LOGGED", 2)
    embeddings = MagicMock()

    async def _only_the_probe(texts: list[str]) -> list[list[float]]:
        if texts == [er._PROBE_TEXT]:
            return [[1.0]]
        raise _StatusError(400)

    embeddings.aembed_documents = _only_the_probe
    conn = _RunConn(_rows(5))

    assert await _run(conn, embeddings) == []

    assert conn.updates == []
    # k001 has no text; the other four are rejected.
    assert len(conn.deletes) == 5
    assert "rejected 4 stored text(s)" in caplog.text
    assert "memories.u1/k000, memories.u1/k002 ..." in caplog.text
    assert pauses


class _FailsOnce(FakeEmbeddings):
    """Fails the first call holding `flaky` alone with `error`, then embeds it."""

    def __init__(self, flaky: str, error: Exception) -> None:
        super().__init__("flaky")
        self.flaky = flaky
        self.error = error
        self.failed = False

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            msg = "the batch keeps failing"
            raise ValueError(msg)
        if texts == [self.flaky] and not self.failed:
            self.failed = True
            raise self.error
        return await super().aembed_documents(texts)


@pytest.mark.asyncio
async def test_a_text_that_fails_once_alone_keeps_its_vector(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """One input error, then a vector: not proof of a rejection."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _FailsOnce("memory 2", _StatusError(400))

    assert await _run(conn, embeddings) == []

    assert [u[2] for u in conn.updates] == ["k000", "k002", "k003"]
    assert ("memories.u1", "k002", "content", f"{STAMP}#2") not in conn.deletes
    assert er._REJECT_RECHECK_S in pauses


class RateLimitError(Exception):
    """Named like the SDKs' rate-limit errors."""


class _StatusError(Exception):
    """An HTTP error carrying its status, as httpx and the SDKs do."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _CodeError(Exception):
    """Google api_core's errors (Gemini's): the status is `code`, an int."""

    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


def _caused_by(cause: Exception) -> Exception:
    err = ValueError("embedding call failed")
    err.__cause__ = cause
    return err


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        RateLimitError("slow down"),
        _caused_by(_StatusError(429)),
        _CodeError(429),
        _caused_by(_CodeError(429)),
        _StatusError(503),
        _StatusError(401),
        _StatusError(408),
        ConnectionResetError(),
        ValueError("input is too long"),
    ],
    ids=[
        "timeout",
        "rate_limit_type",
        "rate_limit_status_cause",
        "gemini_rate_limit_code",
        "gemini_rate_limit_code_cause",
        "overloaded",
        "auth",
        "request_timeout",
        "connection_reset",
        "no_status",
    ],
)
@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_slow_or_rate_limited_text_stops_the_run_and_drops_nothing(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Only a status saying the input is bad is a rejection (#857)."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _FailsOnce("memory 2", error)

    stopped = await _run(conn, embeddings)

    assert stopped
    assert conn.deletes == []
    assert conn.updates == []
    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_an_unavailable_provider_mid_single_pass_drops_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider becoming unavailable is never read as a rejection."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = MagicMock()

    async def _docs(texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            msg = "too long"
            raise ValueError(msg)
        msg = "down"
        raise EmbeddingsUnavailableError(msg)

    embeddings.aembed_documents = _docs

    stopped = await _run(conn, embeddings)

    assert stopped
    assert conn.updates == []
    assert conn.deletes == []
    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj


@pytest.mark.asyncio
async def test_the_verifier_compares_with_a_primary_build_learned_during_its_wait(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex: the record check learns the primary's build while the verifier waits."""
    entry, chain, fallbacks = _held_in_entry(
        hass, monkeypatch, fallback_build="w:a1", fingerprint=OLLAMA_UNREAD
    )

    async def _record_read_while_the_primary_is_learned(*_args: Any) -> str:
        entry.runtime_data.embedding_fingerprint = OLLAMA_B
        return "w:a1"

    monkeypatch.setattr(
        er, "_async_recorded_build", _record_read_while_the_primary_is_learned
    )

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_record_unlike_the_fallback_only_holds_it_while_the_primary_is_up(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record may be stale after a pull on both servers: hold, do not remove."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build="w:b2",
        fingerprint=OLLAMA_UNREAD,
        recorded="w:a1",
    )

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_a_record_unlike_the_fallback_removes_it_with_the_primary_down(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the primary down at setup the record is the only reference: it decides."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build="w:b2",
        fingerprint=OLLAMA_UNREAD,
        recorded="w:a1",
    )
    entry.runtime_data.primary_embeddings = UnavailableEmbeddings("down")

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_admitting_a_held_fallback_tells_the_chain_once() -> None:
    """Its first admission runs the callback; later calls and siblings do not."""
    admitted = MagicMock()
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    chain.hold_until_verified(
        ["p2"], AsyncMock(return_value=True), admitted_callback=admitted
    )

    await chain.aembed_query("ab")
    await chain.aembed_query("ab")
    chain.verify_provider("p2")

    admitted.assert_called_once_with("p2")


@pytest.mark.asyncio
async def test_the_record_check_admitting_a_held_fallback_tells_the_chain() -> None:
    """Admitted by the record check rather than its own first call: same callback."""
    admitted = MagicMock()
    chain = FallbackEmbeddings(
        [(FakeEmbeddings("primary"), "edge", "p1"), (FakeEmbeddings("f"), "edge", "p2")]
    )
    chain.hold_until_verified(
        ["p2"], AsyncMock(return_value=False), admitted_callback=admitted
    )

    chain.verify_provider("p2")
    chain.verify_provider("p1")

    admitted.assert_called_once_with("p2")


@pytest.mark.asyncio
async def test_a_held_fallback_admitted_after_a_failed_index_rearms_it(
    hass: Any,
) -> None:
    """Red Team: a refusal at startup latched the tool index as failed for good."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        tool_index_failed=True,
        tool_index_ready=False,
        tool_indexing_in_progress=False,
        tool_content_hashes={"a": "b"},
    )

    hga_component._retry_failed_tool_index(hass, entry, "p2")

    assert entry.runtime_data.tool_index_failed is False
    assert entry.runtime_data.tool_content_hashes == {"a": "b"}


@pytest.mark.asyncio
async def test_a_held_fallback_admitted_beside_a_good_index_leaves_it(
    hass: Any,
) -> None:
    """Same build as the store's: an index that built needs no rebuild."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        tool_index_failed=False,
        tool_index_ready=True,
        tool_indexing_in_progress=False,
        tool_content_hashes={"a": "b"},
    )

    hga_component._retry_failed_tool_index(hass, entry, "p2")

    assert entry.runtime_data.tool_index_ready is True


@pytest.mark.asyncio
async def test_a_held_fallback_admitted_while_the_index_builds_marks_it(
    hass: Any,
) -> None:
    """Red Team: an admission before the index task fails left it latched."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    entry.add_to_hass(hass)
    entry.runtime_data = SimpleNamespace(
        tool_index_failed=False,
        tool_index_ready=False,
        tool_indexing_in_progress=True,
        tool_index_rearmed=False,
        tool_content_hashes={},
    )

    hga_component._retry_failed_tool_index(hass, entry, "p2")

    # The task's failure, if it ends in one, then does not latch.
    assert entry.runtime_data.tool_index_rearmed is True
    assert entry.runtime_data.tool_index_failed is False


@pytest.mark.asyncio
async def test_a_held_fallback_at_setup_is_logged_as_held_not_announced(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Red Team: no 'fell back' notice for a fallback that may answer nothing."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, recorded="w:a1", primary_up=False
    )

    _assert_held_alone(entry)
    assert "Ollama Secondary stands in once its build" in caplog.text
    assert "Could not send model fallback notification" not in caplog.text


@pytest.mark.asyncio
async def test_without_a_database_a_fallback_at_setup_is_announced(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No store to protect, nothing held: the notice still goes out."""
    await _setup_two_ollama_servers(
        hass, monkeypatch, {}, primary_up=False, database=False
    )

    assert "stands in once its build" not in caplog.text
    assert "Could not send model fallback notification" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_record_check_beside_a_fallback_check_retries_after_it(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback is judged first; the record check's retry waits do not hold it."""
    entry, chain, _removed, waits = _kept_unchecked(hass, monkeypatch, ["w:a1"])
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))
    held_at_each_check: list[frozenset[str]] = []

    async def _check(_pool: Any, _current: Any) -> Any:
        held_at_each_check.append(chain.unverified)
        if len(held_at_each_check) == 1:
            raise TimeoutError
        return er._Decision.MATCH, None

    monkeypatch.setattr(er, "_async_check_record", _check)

    await er.async_sync_embedding_record(hass, entry)

    assert len(held_at_each_check) == 2
    # Admitted by the fallback check before the retry's first wait ended.
    assert held_at_each_check[1] == frozenset()
    assert waits == [er._RECORD_RETRY_DELAYS_S[0]]


@pytest.mark.asyncio
async def test_a_verifier_that_raises_keeps_the_member_held() -> None:
    """An unexpected error never lets a member in, and backs off like a refusal."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    verifier = AsyncMock(side_effect=RuntimeError("boom"))
    chain.hold_until_verified(["p2"], verifier)

    with pytest.raises(Exception, match="not yet known to match"):
        await chain.aembed_query("ab")
    with pytest.raises(Exception, match="not yet known to match"):
        await chain.aembed_query("ab")

    assert chain.unverified == {"p2"}
    verifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_wires_the_real_verifier_into_the_chain(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain built at setup asks the real check, keyed by provider id."""
    entry, _chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:b2")
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    entry.runtime_data.embeddings = chain
    hga_component._hold_unchecked_embedding_fallbacks(
        hass, entry, chain, list(fallbacks.values())
    )

    with pytest.raises(Exception, match="All embedding providers failed"):
        await chain.aembed_query("ab")

    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_verifier_of_a_replaced_chain_decides_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call still in flight across a reload leaves the new chain alone."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:b2")
    old_chain = FallbackEmbeddings([(FakeEmbeddings("f"), "edge", FALLBACK_ID)])

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID, chain=old_chain
    )
    assert _chain_ids(chain) == ["p1", "p2"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_the_verifier_keeps_what_the_record_check_settled_meanwhile(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its gate write starts from the live gate, not the one read before its waits."""
    entry, _chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:b2")

    async def _read_while_the_check_settles(*_args: Any, **_kwargs: Any) -> str:
        gate = entry.runtime_data.embedding_fallback_gate
        entry.runtime_data.embedding_fallback_gate = replace(
            gate, left_out=("Ollama Third",), unchecked=()
        )
        return "w:b2"

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _read_while_the_check_settles)

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )

    gate = entry.runtime_data.embedding_fallback_gate
    assert gate.left_out == ("Ollama Third", SECONDARY)
    assert gate.unchecked == ()
    issue = _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD)
    assert issue is not None
    assert (issue.translation_placeholders or {})["providers"] == (
        f"Ollama Third, {SECONDARY}"
    )


@pytest.mark.asyncio
async def test_a_held_fallback_logs_its_hold_once(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A long hold logs one warning, then debug lines."""
    entry, _chain, fallbacks = _held_in_entry(
        hass, monkeypatch, fallback_build=None, fingerprint=OLLAMA_UNREAD
    )
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value=None))
    warned: set[str] = set()

    for _ in range(3):
        await er.async_verify_embedding_fallback(
            hass, entry, fallbacks, FALLBACK_ID, warned=warned
        )

    warnings = [
        r
        for r in caplog.records
        if "does not stand in" in r.getMessage() and r.levelname == "WARNING"
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_an_input_error_is_a_rejection_whatever_its_message(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A 400 quoting a token count like 14290 is not read as a rate limit."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _RejectsOne("memory 2")

    async def _over_length(texts: list[str]) -> list[list[float]]:
        if embeddings.bad in texts:
            err = _StatusError(400)
            err.args = (
                (
                    "This model's maximum context length is 8192 tokens, "
                    "however you requested 14290 tokens"
                ),
            )
            raise err
        return [[float(len(t))] for t in texts]

    embeddings.aembed_documents = _over_length  # type: ignore[method-assign]

    assert await _run(conn, embeddings) == []
    assert ("memories.u1", "k002", "content", f"{STAMP}#2") in conn.deletes
    assert pauses


@pytest.mark.parametrize(
    "error",
    [_CodeError(400), _caused_by(_StatusError(413)), _StatusError(422)],
    ids=["gemini_invalid_argument", "too_large_cause", "unprocessable"],
)
@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_each_input_error_status_is_a_rejection(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Told by status however the SDK carries it: Gemini's is `code`."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _RejectsOne("memory 2")

    async def _rejects(texts: list[str]) -> list[list[float]]:
        if embeddings.bad in texts:
            raise error
        return [[float(len(t))] for t in texts]

    embeddings.aembed_documents = _rejects  # type: ignore[method-assign]

    assert await _run(conn, embeddings) == []
    assert ("memories.u1", "k002", "content", f"{STAMP}#2") in conn.deletes


@pytest.mark.asyncio
async def test_a_held_fallback_at_setup_is_announced_once_admitted(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The 'fell back' notice waits for the admission, then goes out once."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {}, recorded="w:a1", primary_up=False
    )
    chain = entry.runtime_data.embeddings
    assert "Could not send model fallback notification" not in caplog.text
    [(_client, _deployment, provider_id)] = chain.chain

    chain.verify_provider(provider_id)
    await hass.async_block_till_done()

    assert caplog.text.count("Could not send model fallback notification") == 1
    chain.verify_provider(provider_id)
    await hass.async_block_till_done()
    assert caplog.text.count("Could not send model fallback notification") == 1


# --- Review round 4 (#857) ---------------------------------------------------


class _NullFor(FakeEmbeddings):
    """Answers a null vector for `bad`, in a batch or alone, as a buggy server."""

    def __init__(self, bad: str) -> None:
        super().__init__("null")
        self.bad = bad

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await super().aembed_documents(texts)
        return [
            cast("Any", None) if t == self.bad else v
            for t, v in zip(texts, vectors, strict=True)
        ]


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_null_vector_is_never_read_as_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex: a server answering null deleted the vector with no rejection."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))

    stopped = await _run(conn, _NullFor("memory 2"))

    assert stopped
    assert conn.deletes == []
    assert conn.updates == []
    assert conn.record["reembed_target"] == OLLAMA_A.as_json().obj


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_too_many_rejections_stop_the_run_with_every_vector_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A proxy size limit rejects every long text: no bulk drop (#857)."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 3)
    embeddings = MagicMock()

    async def _only_the_probe(texts: list[str]) -> list[list[float]]:
        if texts == [er._PROBE_TEXT]:
            return [[1.0]]
        raise _StatusError(413)

    embeddings.aembed_documents = _only_the_probe
    conn = _RunConn(_rows(5))

    stopped = await _run(conn, embeddings)

    assert stopped
    assert conn.deletes == []
    assert conn.updates == []
    assert "server or proxy limit" in caplog.text


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_rejection_on_another_build_drops_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex: a build pulled mid-run rejected what the target takes."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))

    stopped = await _run(conn, _RejectsOne("memory 2"), build="w:b2")

    assert stopped
    assert conn.deletes == []
    assert conn.updates == []


@pytest.mark.asyncio
async def test_an_input_error_batch_skips_the_retry_waits(
    monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A batch rejected for its input goes straight to one text at a time."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))

    assert await _run(conn, _RejectsOne("memory 2")) == []

    assert not set(er._RETRY_DELAYS_S) & set(pauses)
    assert ("memories.u1", "k002", "content", f"{STAMP}#2") in conn.deletes


def test_the_status_is_read_from_an_httpx_response_and_never_a_bool() -> None:
    """Httpx carries it on the response; a bool `code` is no status."""
    request = httpx.Request("POST", "http://ollama/api/embed")
    err = httpx.HTTPStatusError(
        "too large", request=request, response=httpx.Response(413, request=request)
    )

    class _BoolCodeError(Exception):
        code = True

    assert er._http_status(err) == 413
    assert er._rejects_the_input(err)
    assert er._http_status(_BoolCodeError()) is None


@pytest.mark.asyncio
async def test_a_failing_primary_beside_a_held_fallback_is_unavailable() -> None:
    """All reviewers: memory search fell to no memories with a traceback."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    chain.hold_until_verified(["p2"], AsyncMock(return_value=False))

    with pytest.raises(EmbeddingsUnavailableError, match="not yet known") as err:
        await chain.aembed_query("abc")
    assert err.value.__cause__ is not None


@pytest.mark.asyncio
async def test_a_chain_emptied_by_its_check_is_unavailable() -> None:
    """The last member removed: unavailable, so search uses recent memories."""
    chain = FallbackEmbeddings(
        [(FakeEmbeddings("f"), "edge", "p2")], circuit_breaker=None
    )
    chain.remove_provider("p2")

    with pytest.raises(EmbeddingsUnavailableError):
        await chain.aembed_query("abc")
    with pytest.raises(EmbeddingsUnavailableError):
        await chain.aembed_documents(["abc"])


@pytest.mark.asyncio
async def test_the_record_check_reads_a_fallback_build_now_not_at_setup(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex + Claude: setup's read admitted a fallback pulled since."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass, monkeypatch, ["w:b2"], read_at_setup="w:a1"
    )
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1"]
    assert chain.unverified == frozenset()
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_record_admission_is_revoked_once_the_primary_build_is_read(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex: admitted against the record, then the primary's build was known."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass,
        monkeypatch,
        [],
        fingerprint=OLLAMA_UNREAD,
        answers={OLLAMA_SOURCE[0]: ["w:a1"], FALLBACK_URL: [None, None, None]},
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))
    chain.verify_provider(FALLBACK_ID, by_record=True)
    assert chain.unverified == frozenset()

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert chain.unverified == {FALLBACK_ID}


# --- Review round 5 (#857) ---------------------------------------------------


def _admitting_chain(admitted: MagicMock | None = None) -> FallbackEmbeddings:
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    chain.hold_until_verified(
        ["p2"], AsyncMock(return_value=True), admitted_callback=admitted
    )
    return chain


def test_learning_the_primary_build_revokes_only_record_admissions(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every writer of the primary's build goes through one helper."""
    chain = FallbackEmbeddings(
        [
            (FakeEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
            (FakeEmbeddings("g"), "edge", "p3"),
        ]
    )
    chain.hold_until_verified(["p2", "p3"], AsyncMock(return_value=False))
    chain.verify_provider("p2", by_record=True)
    chain.verify_provider("p3")
    entry = _entry(
        hass,
        record=None,
        fingerprint=OLLAMA_UNREAD,
        embeddings=chain,
        monkeypatch=monkeypatch,
    )

    er._learn_primary_fingerprint(entry, OLLAMA_A)

    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert chain.unverified == {"p2"}


@pytest.mark.asyncio
async def test_the_verifier_marks_a_record_admission(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admitted by the record, the primary's build unknown: revocable."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build="w:a1",
        fingerprint=OLLAMA_UNREAD,
        recorded="w:a1",
    )
    entry.runtime_data.primary_embeddings = None
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    assert await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)
    chain.revoke_record_admissions()

    assert FALLBACK_ID in chain.unverified


class _SlowEmbeddings(FakeEmbeddings):
    """Answers once `go` is set, after `started` says the call is in flight."""

    def __init__(self) -> None:
        super().__init__("slow")
        self.started = asyncio.Event()
        self.go = asyncio.Event()

    async def aembed_query(self, text: str) -> list[float]:
        self.started.set()
        await self.go.wait()
        return await super().aembed_query(text)


@pytest.mark.asyncio
async def test_an_answer_from_a_member_held_again_mid_call_is_discarded() -> None:
    """Codex: a call in flight kept its answer after the hold was renewed."""
    slow = _SlowEmbeddings()
    chain = FallbackEmbeddings(
        [(_DownEmbeddings("primary"), "edge", "p1"), (slow, "edge", "p2")]
    )
    chain.hold_until_verified(["p2"], AsyncMock(return_value=False))
    chain.verify_provider("p2")
    call = asyncio.create_task(chain.aembed_query("abc"))
    await slow.started.wait()

    chain.hold_again("p2")
    slow.go.set()

    with pytest.raises(HomeAssistantError):
        await call
    assert chain.unverified == {"p2"}


def test_hold_again_without_a_verifier_or_member_does_nothing() -> None:
    """Nothing could ever admit it again: it is not held."""
    chain = FallbackEmbeddings([(FakeEmbeddings("f"), "edge", "p2")])
    chain.hold_again("p2")
    assert chain.unverified == frozenset()

    held = _admitting_chain()
    held.hold_again("gone")
    assert held.unverified == {"p2"}


@pytest.mark.asyncio
async def test_a_member_held_again_is_rechecked_at_once_and_admitted_again() -> None:
    """The backoff is dropped; the admitted callback runs on each admission."""
    admitted = MagicMock()
    chain = _admitting_chain(admitted)
    assert await chain.aembed_query("abc") == [3.0]
    chain.hold_again("p2")

    assert await chain.aembed_query("abc") == [3.0]
    assert admitted.call_count == 2


@pytest.mark.asyncio
async def test_a_refused_member_does_not_wait_on_a_check_in_flight() -> None:
    """The backoff is read before the lock: no queue behind a slow check."""
    chain = FallbackEmbeddings(
        [
            (_DownEmbeddings("primary"), "edge", "p1"),
            (FakeEmbeddings("f"), "edge", "p2"),
        ]
    )
    chain.hold_until_verified(["p2"], AsyncMock(return_value=False))
    with pytest.raises(EmbeddingsUnavailableError):
        await chain.aembed_query("a")
    lock = chain._verify_locks["p2"]
    await lock.acquire()
    try:
        with pytest.raises(EmbeddingsUnavailableError):
            await asyncio.wait_for(chain.aembed_query("b"), 0.5)
    finally:
        lock.release()


def _rejects_all(status: int = 413) -> MagicMock:
    embeddings = MagicMock()

    async def _only_the_probe(texts: list[str]) -> list[list[float]]:
        if texts == [er._PROBE_TEXT]:
            return [[1.0]]
        raise _StatusError(status)

    embeddings.aembed_documents = _only_the_probe
    return embeddings


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_the_cap_counts_across_batches_and_ends_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """All reviewers: setup resumed a capped run, 10 more deletions each time."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 3)
    conn = _RunConn(_rows(5))

    stopped = await _run(conn, _rejects_all())

    assert stopped
    # k000, k002 and k003 rejected (k001 has no text); k004 tripped the cap.
    assert len(conn.deletes) == 4
    assert ("memories.u1", "k004", "content", f"{STAMP}#4") not in conn.deletes
    # Not resumed by setup: the target is gone, and named as a writer.
    assert conn.record["reembed_target"] is None
    assert conn.record["reembed_cursor"] is None
    assert OLLAMA_A.as_json().obj in conn.record["mixed_with"]
    assert "3 vector(s) were removed this run" in caplog.text


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_rejections_up_to_the_cap_finish_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly the cap: dropped, and the run finishes."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 4)
    conn = _RunConn(_rows(5))

    assert await _run(conn, _rejects_all()) == []
    assert len(conn.deletes) == 5
    assert conn.record["model"] == OLLAMA_A.as_json().obj


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_whole_rejected_batch_stops_at_the_cap_not_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-text pass stops once over the cap: no calls for the rest."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 1)
    calls: list[list[str]] = []
    embeddings = _rejects_all()
    inner = embeddings.aembed_documents

    async def _counting(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return await inner(texts)

    embeddings.aembed_documents = _counting
    conn = _RunConn(_rows(6))

    assert await _run(conn, embeddings)
    assert conn.deletes == []
    alone = {t[0] for t in calls if len(t) == 1 and t != [er._PROBE_TEXT]}
    assert alone == {"memory 0", "memory 2"}


@pytest.mark.parametrize(
    "bad", [None, [], [float("nan")]], ids=["null", "empty", "nan"]
)
@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_bad_vector_stops_the_run_with_every_vector_kept(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    """Never a rejection, never a size mismatch: the server's fault."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))
    embeddings = _NullFor("memory 2")
    inner = embeddings.aembed_documents

    async def _bad(texts: list[str]) -> list[list[float]]:
        vectors = await inner(texts)
        return [bad if v is None else v for v in vectors]

    embeddings.aembed_documents = _bad  # type: ignore[method-assign]

    assert await _run(conn, embeddings)
    assert conn.deletes == []
    assert conn.updates == []


# --- Review round 6 (#857) ---------------------------------------------------


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_resumed_run_goes_on_with_what_is_left_of_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All reviewers: a reload or restart resumed the run with a fresh 10."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 2)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 3)
    conn = _RunConn(_rows(5))
    # A run stopped after k001 with 3 texts already rejected.
    conn.record["reembed_target"] = OLLAMA_A.as_json().obj
    conn.record["reembed_cursor"] = ["memories.u1", "k001", "content", 3]

    stopped = await _run(conn, _rejects_all(), start=False)

    assert stopped
    assert conn.deletes == []
    assert conn.record["reembed_target"] is None


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_the_rejected_count_is_saved_with_each_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed with the batch's deletes, so a resume reads it back."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    conn = _RunConn(_rows(4))

    assert await _run(conn, _RejectsOne("memory 2")) == []

    [saved] = [p["cursor"].obj for p in conn.ran(er._SAVE_CURSOR_SQL)]
    assert saved == ["memories.u1", "k003", "content", 1]


@pytest.mark.parametrize(
    ("cursor", "valid"),
    [
        (["p", "k", "f"], True),
        (["p", "k", "f", 4], True),
        (["p", "k", "f", -1], False),
        (["p", "k", "f", True], False),
        (["p", "k", "f", "4"], False),
        (["p", "k"], False),
    ],
    ids=["older", "with_count", "negative", "bool", "str", "short"],
)
@pytest.mark.asyncio
async def test_a_stored_cursor_is_read_with_or_without_its_count(
    cursor: list[Any],
    valid: bool,  # noqa: FBT001
) -> None:
    """An older cursor counts no rejections; a malformed one starts over."""
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchone = AsyncMock(
        return_value={
            "model": OLLAMA_A.as_json().obj,
            "reembed_target": OLLAMA_A.as_json().obj,
            "reembed_cursor": cursor,
            "mixed_with": None,
        }
    )

    record = await er._read_record(cur)

    assert record is not None
    assert (record.cursor is not None) is valid
    if valid:
        assert er._rejected_so_far(record.cursor) == (
            cursor[3] if len(cursor) > 3 else 0
        )


@pytest.mark.usefixtures("pauses")
@pytest.mark.asyncio
async def test_a_cap_stop_keeps_a_writer_noted_during_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex + Claude: the stop wrote its starting copy over a new writer."""
    monkeypatch.setattr(er, "_BATCH_SIZE", 10)
    monkeypatch.setattr(er, "_REJECTED_PER_RUN_MAX", 0)
    conn = _RunConn(_rows(4))
    embeddings = _rejects_all()
    inner = embeddings.aembed_documents
    other = OPENAI_B.as_json().obj

    async def _noted_meanwhile(texts: list[str]) -> list[list[float]]:
        conn.record["mixed_with"] = [other]
        return await inner(texts)

    embeddings.aembed_documents = _noted_meanwhile

    assert await _run(conn, embeddings)
    assert other in conn.record["mixed_with"]
    assert OLLAMA_A.as_json().obj in conn.record["mixed_with"]


@pytest.mark.asyncio
async def test_a_discarded_answer_counts_its_member_as_held() -> None:
    """Claude: a discard on the last member raised the wrong error."""
    slow = _SlowEmbeddings()
    chain = FallbackEmbeddings([(slow, "edge", "p2")], circuit_breaker=None)
    chain.hold_until_verified(["p2"], AsyncMock(return_value=False))
    chain.verify_provider("p2")
    call = asyncio.create_task(chain.aembed_query("abc"))
    await slow.started.wait()

    chain.hold_again("p2")
    slow.go.set()

    with pytest.raises(EmbeddingsUnavailableError, match="not yet known"):
        await call


# --- A fallback left held is named, and the primary's build read on (#873) --

HELD_ISSUE = ISSUE_EMBEDDING_FALLBACK_HELD


def _raise_held_issue(hass: Any, entry_id: str) -> None:
    ir.async_create_issue(
        hass,
        DOMAIN,
        f"{ISSUE_EMBEDDING_FALLBACK_HELD}_{entry_id}",
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD,
    )


def _hold_through_the_verifier(
    hass: Any, entry: MockConfigEntry, chain: FallbackEmbeddings
) -> None:
    """Hold the secondary server as setup does, behind the real verifier."""
    fallbacks = {
        fb.provider_id: fb
        for fb in entry.runtime_data.embedding_fallback_gate.unchecked
    }
    chain.hold_until_verified(
        [FALLBACK_ID],
        partial(
            er.async_verify_embedding_fallback,
            hass,
            entry,
            fallbacks,
            chain=chain,
            warned=set(),
        ),
    )


@pytest.mark.parametrize(
    ("fallback_build", "fingerprint", "recorded", "key"),
    [
        (None, OLLAMA_A, None, ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD),
        ("w:a1", OLLAMA_UNREAD, None, ISSUE_EMBEDDING_FALLBACK_HELD_NO_REFERENCE),
        # The primary answers but is unread: the record may be stale.
        ("w:b2", OLLAMA_UNREAD, "w:a1", ISSUE_EMBEDDING_FALLBACK_HELD_NO_REFERENCE),
    ],
    ids=["its_build_unread", "nothing_to_compare_with", "unlike_the_record"],
)
@pytest.mark.asyncio
async def test_a_fallback_the_verifier_leaves_held_is_named_in_an_issue(  # noqa: PLR0913
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    fallback_build: str | None,
    fingerprint: EmbeddingFingerprint,
    recorded: str | None,
    key: str,
) -> None:
    """Failover silently doing nothing: the issue says which one, and why."""
    entry, chain, fallbacks = _held_in_entry(
        hass,
        monkeypatch,
        fallback_build=fallback_build,
        fingerprint=fingerprint,
        recorded=recorded,
    )

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )

    assert _chain_ids(chain) == ["p1", "p2"]
    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == key
    assert not issue.is_fixable
    assert issue.translation_placeholders == {
        "providers": SECONDARY,
        "primary": "Ollama",
        "model": "mxbai-embed-large",
    }
    reason = (
        er.HoldReason.UNREAD
        if key == ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD
        else er.HoldReason.NO_REFERENCE
    )
    assert entry.runtime_data.embedding_fallback_gate.held == (
        er.HeldFallback(FALLBACK_ID, SECONDARY, reason),
    )


@pytest.mark.asyncio
async def test_admitting_a_held_fallback_clears_its_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its build read and matching on a later check: the issue goes."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)
    assert _issue(hass, HELD_ISSUE) is not None
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value="w:a1"))

    assert await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)

    assert _issue(hass, HELD_ISSUE) is None
    assert entry.runtime_data.embedding_fallback_gate.held == ()
    assert _chain_ids(chain) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_removing_a_held_fallback_hands_it_to_the_other_build_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read later as another build: no longer held, but left out."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)
    monkeypatch.setattr(er, "fetch_ollama_model_digest", AsyncMock(return_value="w:b2"))

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )

    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, HELD_ISSUE) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_held_fallback_names_nothing_for_an_unloaded_entry(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unload during the read raises no issue the unload already cleared."""
    entry, _chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)

    async def _unloaded(*_args: Any, **_kwargs: Any) -> None:
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _unloaded)

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_fallback_the_background_check_cannot_read_is_named(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the background check's last try, the issue names it."""
    entry, chain, _removed, _waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None, None]
    )
    _hold_through_the_verifier(hass, entry, chain)

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == frozenset({FALLBACK_ID})
    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is None


@pytest.mark.asyncio
async def test_a_fallback_the_background_check_admits_raises_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read on the second try and matching: admitted, no held issue."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, [None, "w:a1"])
    _hold_through_the_verifier(hass, entry, chain)

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == frozenset()
    assert _issue(hass, HELD_ISSUE) is None


def _held_on_the_primary(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    primary_reads: list[str | None],
    fallback_reads: list[str | None],
) -> tuple[MockConfigEntry, FallbackEmbeddings, list[float]]:
    """Set up a primary that answers but whose build setup and the record check missed."""
    entry, chain, _removed, waits = _kept_unchecked(
        hass,
        monkeypatch,
        [],
        fingerprint=OLLAMA_UNREAD,
        answers={OLLAMA_SOURCE[0]: primary_reads, FALLBACK_URL: fallback_reads},
    )
    entry.runtime_data.embedding_digest_source = OLLAMA_SOURCE
    monkeypatch.setattr(er, "_async_recorded_build", AsyncMock(return_value=None))
    _hold_through_the_verifier(hass, entry, chain)
    return entry, chain, waits


@pytest.mark.asyncio
async def test_a_primary_build_read_late_ends_the_hold_without_a_reload(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record check gave up on the primary's build: it is read on (#873)."""
    entry, chain, waits = _held_on_the_primary(
        hass,
        monkeypatch,
        # The record check's three reads, then two of the watch's.
        [None, None, None, None, "w:a1"],
        # The background read, the chain's check against the record, then
        # its check against the primary's build once read.
        ["w:a1", "w:a1", "w:a1"],
    )

    await er.async_sync_embedding_record(hass, entry)

    assert waits.count(er._PRIMARY_BUILD_WATCH_S) == 2
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_A
    assert chain.unverified == frozenset()
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_primary_build_never_read_keeps_the_hold_named(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to compare with: named as such, while the read goes on."""
    entry, chain, waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None, None], ["w:a1", "w:a1"]
    )

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        # Stop the watch after its first read.
        if delay == er._PRIMARY_BUILD_WATCH_S and waits.count(delay) > 1:
            entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)

    await er.async_sync_embedding_record(hass, entry)

    assert waits.count(er._PRIMARY_BUILD_WATCH_S) == 2
    assert er.fetch_ollama_model_digest.await_count == 6  # type: ignore[attr-defined]
    assert entry.runtime_data.embedding_fingerprint == OLLAMA_UNREAD
    assert chain.unverified == frozenset({FALLBACK_ID})
    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_NO_REFERENCE


@pytest.mark.asyncio
async def test_a_record_admission_keeps_the_primarys_build_read(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admitted against the record, then the primary's build is read unlike it."""
    entry, chain, waits = _held_on_the_primary(
        hass,
        monkeypatch,
        # The record check's three reads, then the watch's: pulled meanwhile.
        [None, None, None, "w:b2"],
        # The background read, the check against the record, then the one
        # against the primary's build once read.
        ["w:a1", "w:a1", "w:a1"],
    )
    monkeypatch.setattr(er, "_async_recorded_build", AsyncMock(return_value="w:a1"))

    await er.async_sync_embedding_record(hass, entry)

    assert waits.count(er._PRIMARY_BUILD_WATCH_S) == 1
    assert entry.runtime_data.embedding_fingerprint.digest == "w:b2"
    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, HELD_ISSUE) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_primary_build_read_late_removes_a_held_other_build(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held with nothing to compare with, then found to be another build."""
    entry, chain, _waits = _held_on_the_primary(
        hass,
        monkeypatch,
        [None, None, None, None, "w:a1"],
        ["w:b2", "w:b2", "w:b2"],
    )

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1"]
    assert entry.runtime_data.embedding_fallback_gate.held == ()
    assert _issue(hass, HELD_ISSUE) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_build_the_repair_reads_revokes_a_record_admission_too(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirming the re-embed reads the build during the watch's wait."""
    entry, chain, waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None], ["w:a1", "w:a1", "w:a1"]
    )
    monkeypatch.setattr(er, "_async_recorded_build", AsyncMock(return_value="w:a1"))

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        if delay == er._PRIMARY_BUILD_WATCH_S:
            er._learn_primary_fingerprint(entry, replace(OLLAMA_UNREAD, digest="w:b2"))
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)

    await er.async_sync_embedding_record(hass, entry)

    assert _chain_ids(chain) == ["p1"]
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD) is not None


@pytest.mark.asyncio
async def test_a_record_check_that_fails_does_not_hold_up_the_watch(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database down at the check: its retries run beside the watch."""
    entry, chain, _waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None, "w:a1"], ["w:a1", "w:a1", "w:a1"]
    )
    order: list[str] = []
    real_recheck = chain.recheck_held

    async def _recheck() -> None:
        order.append("recheck")
        await real_recheck()

    async def _retry(_hass: Any, _entry: Any) -> None:
        order.append("retry starts")
        await _REAL_SLEEP(0.05)
        order.append("retry ends")

    monkeypatch.setattr(chain, "recheck_held", _recheck)
    monkeypatch.setattr(er, "_async_apply_record_check", AsyncMock(return_value=False))
    monkeypatch.setattr(er, "_async_retry_model_record", _retry)

    await er.async_sync_embedding_record(hass, entry)

    assert order.index("recheck", 1) < order.index("retry ends")
    # The watch's own failed check leaves it to the loop already running.
    assert order.count("retry starts") == 1
    assert chain.unverified == frozenset()


@pytest.mark.asyncio
async def test_the_watch_checks_the_record_with_the_build_it_read(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record check runs with the learned build, and retries if it fails."""
    entry, _chain, _waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None, "w:a1"], ["w:a1", "w:a1", "w:a1"]
    )
    # The setup check landed; the watch's own does not.
    applied = AsyncMock(side_effect=[True, False])
    retried = AsyncMock()
    monkeypatch.setattr(er, "_async_apply_record_check", applied)
    monkeypatch.setattr(er, "_async_retry_model_record", retried)

    await er.async_sync_embedding_record(hass, entry)

    assert applied.await_count == 2
    assert applied.await_args_list[1].args[2].digest == "w:a1"
    retried.assert_awaited_once()


@pytest.mark.asyncio
async def test_releasing_one_of_two_held_fallbacks_keeps_the_other_named(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to compare with outranks an unread build; a release renames it."""
    entry, _chain, _fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    er._update_held(
        hass,
        entry,
        held=[
            er.HeldFallback("p2", "B", er.HoldReason.UNREAD),
            er.HeldFallback("p3", "C", er.HoldReason.NO_REFERENCE),
        ],
    )
    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_NO_REFERENCE
    assert (issue.translation_placeholders or {})["providers"] == "B, C"

    er._update_held(hass, entry, released=["p3"])

    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD
    assert (issue.translation_placeholders or {})["providers"] == "B"


@pytest.mark.asyncio
async def test_a_fallback_noted_again_keeps_its_place(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same hold noted twice changes nothing, so the issue is not redone."""
    entry, _chain, _fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    first = er.HeldFallback("p2", "B", er.HoldReason.UNREAD)
    second = er.HeldFallback("p3", "C", er.HoldReason.UNREAD)
    er._update_held(hass, entry, held=[first, second])
    created = MagicMock(wraps=ir.async_create_issue)
    monkeypatch.setattr(er.ir, "async_create_issue", created)

    er._update_held(hass, entry, held=[first])

    assert entry.runtime_data.embedding_fallback_gate.held == (first, second)
    created.assert_not_called()


@pytest.mark.asyncio
async def test_a_standing_issue_from_the_last_run_is_synced_by_its_first_check(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It survives the reload; the first check of the new run decides it."""
    entry, _chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build="w:a1")
    _raise_held_issue(hass, entry.entry_id)

    assert await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)

    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_fallback_admitted_during_the_background_reads_is_not_named(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its call-time check admitted it while the background check could not read."""
    entry, chain, _removed, waits = _kept_unchecked(
        hass, monkeypatch, [None, None, None, None]
    )
    chain.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        chain.verify_provider(FALLBACK_ID)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)

    await er.async_sync_embedding_record(hass, entry)

    assert entry.runtime_data.embedding_fallback_gate.held == ()
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_refusal_for_a_fallback_admitted_meanwhile_names_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The background check admitted it while this read waited."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)

    async def _admitted_meanwhile(*_args: Any, **_kwargs: Any) -> None:
        chain.verify_provider(FALLBACK_ID)

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _admitted_meanwhile)

    assert not await er.async_verify_embedding_fallback(
        hass, entry, fallbacks, FALLBACK_ID
    )
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_check_that_raises_still_names_the_held_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain keeps it held on any error; the issue says so too."""
    entry, _chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    monkeypatch.setattr(
        er, "fetch_ollama_model_digest", AsyncMock(side_effect=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError):
        await er.async_verify_embedding_fallback(hass, entry, fallbacks, FALLBACK_ID)

    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD


@pytest.mark.asyncio
async def test_rechecking_held_members_skips_no_recent_refusal() -> None:
    """A reference that changed is checked at once, not after the retry wait."""
    verifier = AsyncMock(side_effect=[False, True])
    chain = FallbackEmbeddings(
        [(FakeEmbeddings("primary"), "edge", "p1"), (FakeEmbeddings("f"), "edge", "p2")]
    )
    chain.hold_until_verified(["p2"], verifier)
    await chain.recheck_held()
    assert chain.unverified == frozenset({"p2"})

    await chain.recheck_held()

    assert verifier.await_count == 2
    assert chain.unverified == frozenset()


@pytest.mark.asyncio
async def test_a_primary_build_read_meanwhile_ends_the_hold_too(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirming the repair read the build during the wait: checked at once."""
    entry, chain, waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None], ["w:a1", "w:a1", "w:a1"]
    )

    async def _sleep(delay: float) -> None:
        waits.append(delay)
        if delay == er._PRIMARY_BUILD_WATCH_S:
            entry.runtime_data.embedding_fingerprint = OLLAMA_A
        await _REAL_SLEEP(0)

    monkeypatch.setattr(er.asyncio, "sleep", _sleep)

    await er.async_sync_embedding_record(hass, entry)

    assert waits.count(er._PRIMARY_BUILD_WATCH_S) == 1
    assert chain.unverified == frozenset()
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_rechecking_runs_even_after_a_refusal_inside_the_lock() -> None:
    """A call-time refusal landing while the recheck waits does not skip it."""
    chain = FallbackEmbeddings(
        [(FakeEmbeddings("primary"), "edge", "p1"), (FakeEmbeddings("f"), "edge", "p2")]
    )
    release = asyncio.Event()
    answers = [False, True]

    async def _verifier(_provider_id: str) -> bool:
        if len(answers) == 2:
            await release.wait()
        return answers.pop(0)

    chain.hold_until_verified(["p2"], _verifier)
    in_flight = asyncio.get_running_loop().create_task(chain._admitted("p2"))
    await asyncio.sleep(0)
    recheck = asyncio.get_running_loop().create_task(chain.recheck_held())
    await asyncio.sleep(0)
    release.set()

    assert not await in_flight
    await recheck

    assert answers == []
    assert chain.unverified == frozenset()


@pytest.mark.asyncio
async def test_a_primary_build_read_late_names_a_fallback_still_unread(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Something to compare with at last, but the fallback's own build is unread."""
    entry, chain, _waits = _held_on_the_primary(
        hass, monkeypatch, [None, None, None, None, "w:a1"], ["w:a1", "w:a1", None]
    )

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == frozenset({FALLBACK_ID})
    issue = _issue(hass, HELD_ISSUE)
    assert issue is not None
    assert issue.translation_key == ISSUE_EMBEDDING_FALLBACK_HELD_UNREAD


@pytest.mark.asyncio
async def test_the_background_check_admitting_a_fallback_clears_its_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call-time refusal named it; the background check's read admits it."""
    entry, chain, _removed, _waits = _kept_unchecked(hass, monkeypatch, [None, "w:a1"])
    _hold_through_the_verifier(hass, entry, chain)
    er._update_held(
        hass,
        entry,
        held=[er.HeldFallback(FALLBACK_ID, SECONDARY, er.HoldReason.UNREAD)],
    )
    assert _issue(hass, HELD_ISSUE) is not None

    await er.async_sync_embedding_record(hass, entry)

    assert chain.unverified == frozenset()
    assert entry.runtime_data.embedding_fallback_gate.held == ()
    assert _issue(hass, HELD_ISSUE) is None


@pytest.mark.asyncio
async def test_a_release_first_in_a_run_keeps_the_issue_of_one_still_held(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a reload: another fallback admitted first, this one not yet checked."""
    entry, chain, _fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    _raise_held_issue(hass, entry.entry_id)

    er._update_held(hass, entry, released=["p3"])

    assert chain.unverified == frozenset({FALLBACK_ID})
    assert _issue(hass, HELD_ISSUE) is not None


@pytest.mark.asyncio
async def test_a_check_from_a_replaced_chain_names_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload replaced the chain while its check ran: the new run is untouched."""
    entry, chain, fallbacks = _held_in_entry(hass, monkeypatch, fallback_build=None)
    new = FallbackEmbeddings(list(chain.chain))
    new.hold_until_verified([FALLBACK_ID], AsyncMock(return_value=False))

    async def _reloaded_then_failed(*_args: Any, **_kwargs: Any) -> None:
        entry.runtime_data.embeddings = new
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(er, "fetch_ollama_model_digest", _reloaded_then_failed)

    with pytest.raises(RuntimeError):
        await er.async_verify_embedding_fallback(
            hass, entry, fallbacks, FALLBACK_ID, chain=chain
        )

    assert _issue(hass, HELD_ISSUE) is None
