# ruff: noqa: S101
"""Tests for long-term memory limits, voice tools and admin services."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import Unauthorized
from langgraph.store.memory import InMemoryStore

from custom_components.home_generative_agent.agent.tools import (
    forget_memory,
    list_memories,
    upsert_memory,
)
from custom_components.home_generative_agent.const import (
    CONF_MEMORY_MAX_PER_PERSON,
    MEMORY_LIST_MAX_SHOWN,
)
from custom_components.home_generative_agent.core.memories import (
    memory_limit,
    memory_user,
    register_memory_services,
)


def _config(user: str = "Anna", limit: int = 0) -> dict[str, Any]:
    return {
        "configurable": {
            "user_id": user,
            "options": {CONF_MEMORY_MAX_PER_PERSON: limit},
        }
    }


async def _save(store: InMemoryStore, content: str, **kwargs: Any) -> str:
    config = _config(limit=kwargs.pop("limit", 0))
    args = {"content": content, "config": config, "store": store, **kwargs}
    return await upsert_memory.coroutine(**args)  # type: ignore[misc]


async def _keys(store: InMemoryStore, user: str = "Anna") -> list[str]:
    return [i.key for i in await store.asearch((user, "memories"), limit=100)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Kitchen tablet", "Kitchen tablet"), ("O'Neil", "ONeil"), (None, "robot")],
)
def test_memory_user_cleans_the_name(name: str | None, expected: str) -> None:
    """Punctuation is not allowed in a namespace, and no name is the shared store."""
    assert memory_user(name) == expected


@pytest.mark.parametrize(("value", "expected"), [(None, 0), (0, 0), (5.0, 5), (-1, 0)])
def test_memory_limit_reads_the_option(value: Any, expected: int) -> None:
    """A missing or zero limit is unlimited."""
    assert memory_limit({CONF_MEMORY_MAX_PER_PERSON: value}) == expected


async def test_zero_limit_saves_without_bound() -> None:
    """The default keeps every memory."""
    store = InMemoryStore()
    for number in range(12):
        assert (await _save(store, f"fact {number}")).startswith("Stored")
    assert len(await _keys(store)) == 12


async def test_full_memory_refuses_a_new_one_but_allows_an_update() -> None:
    """At the limit nothing is dropped, and an existing memory can still change."""
    store = InMemoryStore()
    await _save(store, "first", limit=1, memory_id="a")

    refused = await _save(store, "second", limit=1)
    assert refused.startswith("Not stored")
    assert await _keys(store) == ["a"]

    assert (await _save(store, "changed", limit=1, memory_id="a")).startswith("Stored")
    item = await store.aget(("Anna", "memories"), "a")
    assert item is not None
    assert item.value["content"] == "changed"


async def test_limit_is_per_person() -> None:
    """One person's memories do not use up another's room."""
    store = InMemoryStore()
    await store.aput(("Boris", "memories"), "x", {"content": "hers"})
    assert (await _save(store, "mine", limit=1)).startswith("Stored")


async def test_an_invented_id_counts_as_a_new_memory() -> None:
    """A made-up memory id cannot get around the limit."""
    store = InMemoryStore()
    await _save(store, "first", limit=1, memory_id="a")
    refused = await _save(store, "second", limit=1, memory_id="made-up")
    assert refused.startswith("Not stored")


async def test_list_memories_is_for_the_current_person_only() -> None:
    """A voice hears its own memories, with ids it can pass on."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "mine"})
    await store.aput(("Boris", "memories"), "b", {"content": "hers"})

    args = {"config": _config(), "store": store}
    listed = await list_memories.coroutine(**args)  # type: ignore[misc]

    assert listed == "[a]: mine"


async def test_list_memories_with_none_and_with_too_many() -> None:
    """An empty store says so; a long list is cut to what fits the context."""
    store = InMemoryStore()
    args = {"config": _config(), "store": store}
    assert "no memories" in await list_memories.coroutine(**args)  # type: ignore[misc]

    for number in range(MEMORY_LIST_MAX_SHOWN + 2):
        await store.aput(("Anna", "memories"), str(number), {"content": "x"})
    listed = await list_memories.coroutine(**args)  # type: ignore[misc]
    assert listed.count("]: x") == MEMORY_LIST_MAX_SHOWN
    assert "2 older memories" in listed


async def test_forget_memory_deletes_only_your_own() -> None:
    """Another person's id is not found, so it cannot be deleted by voice."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "mine"})
    await store.aput(("Boris", "memories"), "b", {"content": "hers"})

    args = {"config": _config(), "store": store}
    assert "Nothing was deleted" in await forget_memory.coroutine(  # type: ignore[misc]
        memory_id="b", **args
    )
    assert await _keys(store, "Boris") == ["b"]

    assert (await forget_memory.coroutine(memory_id="a", **args)).startswith("Deleted")  # type: ignore[misc]
    assert await _keys(store) == []


def _services(store: Any, *, admin: bool = True) -> tuple[dict[str, Any], Any]:
    handlers: dict[str, Any] = {}

    def register(_hass: Any, _entry: Any, name: str, handler: Any, **_: Any) -> None:
        handlers[name] = handler

    hass = MagicMock()
    hass.auth.async_get_user = AsyncMock(return_value=SimpleNamespace(is_admin=admin))
    hass.auth.async_get_users = AsyncMock(
        return_value=[
            SimpleNamespace(name="Boris", is_active=True, system_generated=False),
            SimpleNamespace(name="Supervisor", is_active=True, system_generated=True),
        ]
    )
    entry = SimpleNamespace(
        runtime_data=SimpleNamespace(
            store=store, options={CONF_MEMORY_MAX_PER_PERSON: 3}
        )
    )
    register_memory_services(hass, entry, register)
    return handlers, hass


def _call(user_id: str | None = "admin", **data: Any) -> Any:
    return SimpleNamespace(data=data, context=SimpleNamespace(user_id=user_id))


async def test_get_memories_lists_people_and_one_persons_memories() -> None:
    """People with no memories yet are listed, so a memory can be moved to them."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "mine", "context": "c"})
    handlers, _ = _services(store)

    result = await handlers["get_memories"](_call(user="Anna"))

    assert result["limit"] == 3
    assert result["people"] == [
        {"user": "Anna", "count": 1},
        {"user": "Boris", "count": 0},
        {"user": "robot", "count": 0},
    ]
    assert result["memories"][0]["id"] == "a"
    assert result["memories"][0]["content"] == "mine"
    assert result["memories"][0]["context"] == "c"


async def test_update_delete_and_move() -> None:
    """Each action changes only the memory it names."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "old"})
    handlers, _ = _services(store)

    updated = await handlers["update_memory"](
        _call(user="Anna", memory_id="a", content="new", context="")
    )
    assert updated == {"status": "ok"}
    item = await store.aget(("Anna", "memories"), "a")
    assert item is not None
    assert item.value["content"] == "new"

    moved = await handlers["move_memory"](
        _call(user="Anna", memory_id="a", to_user="Boris")
    )
    assert moved == {"status": "ok"}
    assert await _keys(store) == []
    assert await _keys(store, "Boris") == ["a"]

    deleted = await handlers["delete_memory"](_call(user="Boris", memory_id="a"))
    assert deleted == {"status": "ok"}
    assert await _keys(store, "Boris") == []


async def test_moving_to_the_same_person_keeps_the_memory() -> None:
    """A move onto its own owner must not delete what it just wrote."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "old"})
    handlers, _ = _services(store)

    await handlers["move_memory"](_call(user="Anna", memory_id="a", to_user="Anna"))

    assert await _keys(store) == ["a"]


async def test_a_move_cleans_the_new_owners_name() -> None:
    """A name typed with punctuation lands where that person's memories are kept."""
    store = InMemoryStore()
    await store.aput(("Anna", "memories"), "a", {"content": "old"})
    handlers, _ = _services(store)

    await handlers["move_memory"](_call(user="Anna", memory_id="a", to_user="O'Neil"))

    assert await _keys(store, "ONeil") == ["a"]


@pytest.mark.parametrize("service", ["update_memory", "delete_memory", "move_memory"])
async def test_a_missing_memory_is_reported(service: str) -> None:
    """A stale card row gets an answer instead of an error."""
    handlers, _ = _services(InMemoryStore())
    call = _call(user="Anna", memory_id="gone", content="x", context="", to_user="y")
    assert await handlers[service](call) == {"status": "not_found"}


@pytest.mark.parametrize(
    "service", ["get_memories", "update_memory", "delete_memory", "move_memory"]
)
async def test_services_refuse_a_non_admin(service: str) -> None:
    """Memories are private: a regular dashboard user cannot read or change them."""
    handlers, _ = _services(InMemoryStore(), admin=False)
    call = _call(user="Anna", memory_id="a", content="x", context="", to_user="y")
    with pytest.raises(Unauthorized):
        await handlers[service](call)
