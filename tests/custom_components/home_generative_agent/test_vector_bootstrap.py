# ruff: noqa: S101
"""Tests for vector store bootstrap behavior."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

import psycopg
import pytest
from psycopg_pool import PoolTimeout
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import (
    CONF_DB_BOOTSTRAPPED,
    CONF_VECTORS_BOOTSTRAPPED,
    DOMAIN,
)
from custom_components.home_generative_agent.core.embedding_store import (
    MIGRATION_RECORDS,
    async_forget_migrations,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class DummyStore:
    """Minimal AsyncPostgresStore stand-in."""

    def __init__(self, index_config: object | None) -> None:
        """Initialize dummy store."""
        self.index_config = index_config
        self.setup_calls = 0

    async def setup(self) -> None:
        """Simulate store setup."""
        self.setup_calls += 1


_POOL: Any = object()
_ALL_TABLES = frozenset(
    {
        "store",
        "store_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
        "store_vectors",
        "vector_migrations",
    }
)


@pytest.fixture(autouse=True)
def _tables_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every store table exists unless a test says otherwise."""

    async def _tables(_pool: Any) -> frozenset[str]:
        return _ALL_TABLES

    monkeypatch.setattr(hga_component, "async_store_tables", _tables)


_BOOTSTRAP_VECTORS_ATTR = "_bootstrap_vectors_once"
bootstrap_vectors_once = cast(
    "Any",
    getattr(hga_component, _BOOTSTRAP_VECTORS_ATTR),
)


@pytest.mark.asyncio
async def test_vector_bootstrap_runs_with_index_config(hass: Any) -> None:
    """Vector bootstrap runs when index config is present."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=object())

    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert store.setup_calls == 1
    assert entry.data.get(CONF_VECTORS_BOOTSTRAPPED) is True


@pytest.mark.asyncio
async def test_vector_bootstrap_skips_without_index_config(hass: Any) -> None:
    """Vector bootstrap skips when no index config is present."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=None)

    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert store.setup_calls == 0
    assert entry.data.get(CONF_VECTORS_BOOTSTRAPPED) is None


@pytest.mark.asyncio
async def test_vector_bootstrap_skips_when_already_bootstrapped(hass: Any) -> None:
    """Vector bootstrap does not rerun once flagged."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_VECTORS_BOOTSTRAPPED: True})
    entry.add_to_hass(hass)
    store = DummyStore(index_config=object())

    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert store.setup_calls == 0


# ----- #760: tables gone after the store was bootstrapped -----

bootstrap_db_once = cast("Any", hga_component)._bootstrap_db_once


class _Tables:
    """
    The store's tables, as langgraph's setup leaves them.

    Setup creates the tables of every record it has no entry for: a record
    table that is gone, or one the real `async_forget_migrations` cleared.
    """

    def __init__(self, tables: frozenset[str] | None, *, creates: bool = True) -> None:
        self.tables = tables
        self.creates = creates
        self.forgotten: list[tuple[str, ...]] = []
        self.cleared: set[str] = set()

    async def read(self, _pool: Any) -> frozenset[str] | None:
        return self.tables

    async def forget(
        self, _pool: Any, tables: frozenset[str], records: tuple[str, ...]
    ) -> None:
        deleted: list[str] = []
        await async_forget_migrations(
            cast("Any", _RecordingPool(deleted)), tables, records
        )
        if deleted:
            self.forgotten.append(tuple(deleted))
            self.cleared |= set(deleted)

    def setup_ran(self) -> None:
        if not self.creates or self.tables is None:
            return
        created = set(self.tables)
        for record, names in MIGRATION_RECORDS.items():
            if record not in self.tables or record in self.cleared:
                created |= {record, *names}
        self.tables = frozenset(created)
        self.cleared.clear()


class _RecordingPool:
    """A pool whose cursor records which record tables were cleared."""

    def __init__(self, deleted: list[str]) -> None:
        self.deleted = deleted

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncGenerator[Any]:
        deleted = self.deleted

        class _Cur:
            async def execute(self, query: Any, _params: Any = None) -> None:
                deleted.append(query.as_string().split('"')[1])

        class _Conn:
            @contextlib.asynccontextmanager
            async def cursor(self) -> AsyncGenerator[_Cur]:
                yield _Cur()

        yield _Conn()


class _Store(DummyStore):
    def __init__(self, index_config: object | None, tables: _Tables) -> None:
        super().__init__(index_config)
        self._tables = tables

    async def setup(self) -> None:
        await super().setup()
        self._tables.setup_ran()


def _with_tables(
    monkeypatch: pytest.MonkeyPatch,
    tables: frozenset[str] | None,
    *,
    creates: bool = True,
) -> _Tables:
    state = _Tables(tables, creates=creates)
    monkeypatch.setattr(hga_component, "async_store_tables", state.read)
    monkeypatch.setattr(hga_component, "async_forget_migrations", state.forget)
    return state


def _entry(hass: Any, **data: Any) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    return entry


@pytest.mark.parametrize(
    ("dropped", "forgotten"),
    [
        # Its migrations are forgotten first, or setup would skip them.
        ({"store_vectors"}, [("vector_migrations",)]),
        # A record table that is gone too is not touched: setup creates it.
        ({"store_vectors", "vector_migrations"}, []),
    ],
)
@pytest.mark.asyncio
async def test_a_dropped_vector_table_is_created_again(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    dropped: set[str],
    forgotten: list[tuple[str, ...]],
) -> None:
    """A vector table dropped to change its size comes back at the new size."""
    state = _with_tables(monkeypatch, _ALL_TABLES - dropped)
    store = _Store({"dims": 768}, state)

    await bootstrap_vectors_once(
        hass, _entry(hass, **{CONF_VECTORS_BOOTSTRAPPED: True}), store, _POOL
    )

    assert (state.forgotten, store.setup_calls) == (forgotten, 1)
    assert "creating it again for 768-dimension vectors" in caplog.text
    assert "still missing" not in caplog.text


@pytest.mark.asyncio
async def test_unreadable_tables_keep_the_bootstrap_flags(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database that cannot be asked is not taken for an empty one."""
    state = _with_tables(monkeypatch, None)
    store = _Store({"dims": 1024}, state)
    checkpointer = _Store(None, state)
    entry = _entry(
        hass, **{CONF_VECTORS_BOOTSTRAPPED: True, CONF_DB_BOOTSTRAPPED: True}
    )

    await bootstrap_db_once(hass, entry, store, checkpointer, _POOL)
    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert (state.forgotten, store.setup_calls, checkpointer.setup_calls) == (
        [],
        0,
        0,
    )


@pytest.mark.asyncio
async def test_no_index_config_reads_nothing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without embeddings there is no vector table to bring back."""
    state = _with_tables(monkeypatch, frozenset())
    store = _Store(None, state)

    await bootstrap_vectors_once(
        hass, _entry(hass, **{CONF_VECTORS_BOOTSTRAPPED: True}), store, _POOL
    )

    assert (state.forgotten, store.setup_calls) == ([], 0)


@pytest.mark.parametrize(
    ("dropped", "forgotten"),
    [
        (set(), None),
        # A fresh database: nothing to forget, setup creates everything.
        (set(_ALL_TABLES), []),
        # One dropped table: its record is forgotten so setup creates it.
        ({"checkpoints"}, [("checkpoint_migrations",)]),
        # Its vector table comes back in the vector pass, not this one.
        ({"store", "store_vectors"}, [("store_migrations",)]),
        ({"checkpoint_writes"}, [("checkpoint_migrations",)]),
    ],
)
@pytest.mark.asyncio
async def test_missing_database_tables_are_created_again(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    dropped: set[str],
    forgotten: list[tuple[str, ...]] | None,
) -> None:
    """A Database subentry pointed at a new database gets its tables (#760)."""
    state = _with_tables(monkeypatch, _ALL_TABLES - dropped)
    store = _Store(None, state)
    checkpointer = _Store(None, state)

    await bootstrap_db_once(
        hass,
        _entry(hass, **{CONF_DB_BOOTSTRAPPED: True}),
        store,
        checkpointer,
        _POOL,
    )

    runs = 0 if forgotten is None else 1
    assert (store.setup_calls, checkpointer.setup_calls) == (runs, runs)
    assert state.forgotten == (forgotten or [])


@pytest.mark.asyncio
async def test_a_first_bootstrap_sets_both_up(hass: Any) -> None:
    """Without the flag, the store and checkpointer are set up and it is set."""
    entry = _entry(hass)
    store, checkpointer = DummyStore(None), DummyStore(None)

    await bootstrap_db_once(hass, entry, store, checkpointer, _POOL)

    assert (store.setup_calls, checkpointer.setup_calls) == (1, 1)
    assert entry.data.get(CONF_DB_BOOTSTRAPPED) is True


@pytest.mark.asyncio
async def test_a_fresh_database_is_set_up_once_in_sequence(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store's setup also creates its vector table: no second setup."""
    state = _with_tables(monkeypatch, frozenset())
    store = _Store({"dims": 1024}, state)
    checkpointer = _Store(None, state)
    entry = _entry(
        hass, **{CONF_DB_BOOTSTRAPPED: True, CONF_VECTORS_BOOTSTRAPPED: True}
    )

    await bootstrap_db_once(hass, entry, store, checkpointer, _POOL)
    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert (state.forgotten, store.setup_calls) == ([], 1)


@pytest.mark.asyncio
async def test_a_table_setup_did_not_create_is_logged(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A user that cannot create tables is told, not left to failing writes."""
    state = _with_tables(monkeypatch, _ALL_TABLES - {"store_vectors"}, creates=False)
    store = _Store({"dims": 1024}, state)

    await bootstrap_vectors_once(
        hass, _entry(hass, **{CONF_VECTORS_BOOTSTRAPPED: True}), store, _POOL
    )

    assert "Memory store tables store_vectors are still missing" in caplog.text


@pytest.mark.asyncio
async def test_a_dropped_store_comes_back_with_its_vector_table(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pass forgets its own record; both tables are back after the two."""
    state = _with_tables(monkeypatch, _ALL_TABLES - {"store", "store_vectors"})
    store = _Store({"dims": 1024}, state)
    checkpointer = _Store(None, state)
    entry = _entry(
        hass, **{CONF_DB_BOOTSTRAPPED: True, CONF_VECTORS_BOOTSTRAPPED: True}
    )

    await bootstrap_db_once(hass, entry, store, checkpointer, _POOL)
    await bootstrap_vectors_once(hass, entry, store, _POOL)

    assert state.forgotten == [("store_migrations",), ("vector_migrations",)]
    assert state.tables == _ALL_TABLES


class _FailingStore(DummyStore):
    def __init__(self, error: Exception) -> None:
        super().__init__({"dims": 4096})
        self.error = error

    async def setup(self) -> None:
        raise self.error


@pytest.mark.asyncio
async def test_a_rebuild_that_fails_is_logged_and_setup_goes_on(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing table only ever failed its writes; it must not stop loading."""
    _with_tables(monkeypatch, _ALL_TABLES - {"store_vectors"})
    store = _FailingStore(psycopg.errors.ProgramLimitExceeded("too many dims"))

    await bootstrap_vectors_once(
        hass, _entry(hass, **{CONF_VECTORS_BOOTSTRAPPED: True}), store, _POOL
    )

    assert "Could not create memory store tables again" in caplog.text


@pytest.mark.asyncio
async def test_a_lost_connection_during_a_rebuild_is_raised(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup's retry handles a database that went away."""
    _with_tables(monkeypatch, _ALL_TABLES - {"store_vectors"})
    store = _FailingStore(PoolTimeout("gone"))

    with pytest.raises(PoolTimeout):
        await bootstrap_vectors_once(
            hass, _entry(hass, **{CONF_VECTORS_BOOTSTRAPPED: True}), store, _POOL
        )


@pytest.mark.asyncio
async def test_a_store_that_cannot_be_rebuilt_leaves_the_checkpointer_set_up(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat needs the checkpoint tables on every turn."""
    state = _with_tables(monkeypatch, frozenset())
    store = _FailingStore(psycopg.errors.ProgramLimitExceeded("too many dims"))
    checkpointer = _Store(None, state)

    await bootstrap_db_once(
        hass,
        _entry(hass, **{CONF_DB_BOOTSTRAPPED: True}),
        store,
        checkpointer,
        _POOL,
    )

    assert checkpointer.setup_calls == 1
