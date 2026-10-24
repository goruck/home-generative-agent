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
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from homeassistant.config_entries import ConfigEntryState
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
    ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD,
    ISSUE_EMBEDDING_MODEL_CHANGED,
    EmbeddingFingerprint,
    embedding_fingerprint,
    text_at_field,
)
from custom_components.home_generative_agent.core.embedding_store import (
    EmbeddingsUnavailableError,
    UnavailableEmbeddings,
)
from custom_components.home_generative_agent.core.utils import configured_ollama_urls

from .test_fallback_setup import (
    FakeEmbeddings,
    _capture_recheck,
    _fallback_setup_data,
    _patch_setup_dependencies,
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
    # covers what the recorded model writes from now on.
    (marked,) = cur.ran(er._MARK_MIXED_SQL)
    assert [EmbeddingFingerprint.from_json(m) for m in marked["mixed"].obj] == [OPENAI]
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
    """An unreadable record leaves the issue as it is."""
    entry = _entry(hass, record=er._Record(OPENAI, None, None), monkeypatch=monkeypatch)
    await er.async_sync_embedding_record(hass, entry)
    monkeypatch.setattr(
        er, "_async_check_record", AsyncMock(side_effect=psycopg.OperationalError("x"))
    )

    await er.async_sync_embedding_record(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED) is not None
    assert "Could not read which embedding model" in caplog.text


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
            if held != target_obj or self.record["reembed_cursor"] != previous:
                return [], 0
            self.record["reembed_cursor"] = params["cursor"].obj
            return [], 1
        if query == er._BUILD_CHANGED_SQL:
            if held != target_obj:
                return [], 0
            self.record.update(reembed_cursor=None, mixed_with=params["mixed"].obj)
            return [], 1
        if query == er._FINISH_REEMBED_SQL:
            mixed = params["mixed"].obj if params["mixed"] else None
            if held != target_obj or self.record.get("mixed_with") != mixed:
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
        ["memories.u1", "k001", "content"],
        ["memories.u1", "k003", "content"],
        ["memories.u1", "k004", "content"],
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
    assert conn.record["reembed_cursor"] == ["memories.u1", "k001", "content"]
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

    def _taken_over(query: str, params: Any) -> tuple[list[Any], int]:
        if query == er._SAVE_CURSOR_SQL:
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
    assert conn.record["reembed_cursor"] == ["memories.u1", "k001", "content"]
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

    assert await _run(conn) == []

    assert [u[2] for u in conn.updates] == ["k000"]
    assert conn.record["reembed_cursor"] is None
    assert "another run changed its target" in caplog.text
    assert pauses


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

        yield SimpleNamespace(cursor=_cursor)


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _with_database(
    monkeypatch: pytest.MonkeyPatch,
    digests: dict[str, str],
    *,
    recorded: str | None = None,
) -> list[Any]:
    """
    Give setup a database that accepts everything, and these digests.

    `recorded` is the build the store's record names. The returned list gets
    each digest request and, as ("record", entry), each record check setup
    schedules.
    """
    asked: list[Any] = []

    async def _digests(
        _hass: Any, urls: list[str], model: str, **_kwargs: Any
    ) -> dict[str, str]:
        asked.append((urls, model))
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
    monkeypatch.setattr(hga_component, "async_sync_embedding_record", _record_check)
    return asked


FALLBACK_URL = "http://ollama-fallback:11434"


async def _setup_two_ollama_servers(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    digests: dict[str, str],
    *,
    recorded: str | None = None,
    primary_up: bool = True,
) -> tuple[MockConfigEntry, Any, list[Any]]:
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
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
    if not primary_up:

        async def _primary_down(_hass: Any, url: str, **_kwargs: Any) -> bool:
            return url != "http://ollama"

        monkeypatch.setattr(hga_component, "ollama_healthy", _primary_down)
    asked = _with_database(monkeypatch, digests, recorded=recorded)

    assert await cast("Any", hga_component).async_setup_entry(hass, entry)
    await hass.async_block_till_done()
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


@pytest.mark.asyncio
async def test_with_the_primary_down_a_matching_fallback_takes_over(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback holding the recorded build keeps embedding failover."""
    entry, _data, _asked = await _setup_two_ollama_servers(
        hass, monkeypatch, {FALLBACK_URL: "w:a1"}, recorded="w:a1", primary_up=False
    )

    assert entry.runtime_data.embeddings.name == f"configured-embed@{FALLBACK_URL}"
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None


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

    monkeypatch.setattr(entry, "disabled_by", "user")
    assert await hga_component.async_unload_entry(hass, entry)

    assert _issue(hass, ISSUE_EMBEDDING_MODEL_CHANGED, entry.entry_id) is None
    assert _issue(hass, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD, entry.entry_id) is None


@pytest.mark.asyncio
async def test_removal_drops_the_issues(hass: Any) -> None:
    """A removed entry leaves neither issue behind."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="entry1")
    for key in (ISSUE_EMBEDDING_MODEL_CHANGED, ISSUE_EMBEDDING_FALLBACK_OTHER_BUILD):
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
