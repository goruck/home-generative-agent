"""Long-term memories per person: reading, limits and the admin services."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import config_validation as cv

from ..const import (  # noqa: TID252
    CONF_MEMORY_MAX_PER_PERSON,
    SHARED_MEMORY_USER,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from homeassistant.core import HomeAssistant, ServiceCall
    from langgraph.store.base import BaseStore, SearchItem

_MEMORIES = "memories"
_PAGE = 100


def _without_punctuation(name: str) -> str:
    # Memory namespace labels cannot contain punctuation, and the store's
    # prefix search treats "%" and "_" as wildcards.
    return name.translate(str.maketrans("", "", string.punctuation))


def memory_user(name: str | None) -> str:
    """Return the store's name for a user; no name is the shared satellite store."""
    return _without_punctuation(name or SHARED_MEMORY_USER) or SHARED_MEMORY_USER


def _service_user(value: Any) -> str:
    """
    Clean a service's user field the way the agent names a memory user.

    So a call reaches the namespace the agent writes and never a wildcard
    prefix. A name with nothing left is refused rather than read as the shared
    store; name "robot" to reach that one.
    """
    cleaned = _without_punctuation(cv.string(value)).strip()
    if not cleaned:
        msg = "Name a person"
        raise vol.Invalid(msg)
    return cleaned


GET_MEMORIES_SCHEMA = vol.Schema(
    # Empty lists people only, as the field's description says.
    {
        vol.Optional("user"): vol.Any(
            vol.All(cv.string, vol.Length(max=0)), _service_user
        )
    }
)
DELETE_MEMORY_SCHEMA = vol.Schema(
    {vol.Required("user"): _service_user, vol.Required("memory_id"): cv.string}
)
UPDATE_MEMORY_SCHEMA = DELETE_MEMORY_SCHEMA.extend(
    {
        vol.Required("content"): vol.All(cv.string, vol.Length(min=1)),
        vol.Optional("context", default=""): cv.string,
    }
)
MOVE_MEMORY_SCHEMA = DELETE_MEMORY_SCHEMA.extend(
    {vol.Required("to_user"): _service_user}
)


def memory_namespace(user: str) -> tuple[str, str]:
    """Return the store namespace holding a user's memories."""
    return (user, _MEMORIES)


def memory_limit(options: Mapping[str, Any]) -> int:
    """Return how many memories one person may keep; 0 means unlimited."""
    return max(int(options.get(CONF_MEMORY_MAX_PER_PERSON) or 0), 0)


def store_available(store: Any) -> bool:
    """Return whether memories can be read and changed (a database is set up)."""
    return store is not None and not getattr(store, "is_null_store", False)


async def memories_full(store: BaseStore, user: str, limit: int) -> bool:
    """Return whether a user has no room for another memory."""
    if not limit:
        return False
    # Reads at most one row: is there a memory at position `limit`?
    return bool(await store.asearch(memory_namespace(user), limit=1, offset=limit - 1))


async def all_memories(store: BaseStore, user: str) -> list[SearchItem]:
    """Return a user's memories, the most recently changed first."""
    found: list[SearchItem] = []
    while True:
        page = await store.asearch(
            memory_namespace(user), limit=_PAGE, offset=len(found)
        )
        found.extend(page)
        if len(page) < _PAGE:
            return found


def _as_record(item: Any) -> dict[str, Any]:
    return {
        "id": item.key,
        "content": str(item.value.get("content", "")),
        "context": str(item.value.get("context", "")),
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


async def _memory_owners(store: BaseStore) -> set[str]:
    """Return everyone with a (user, "memories") namespace, page by page."""
    owners: set[str] = set()
    offset = 0
    while True:
        page = await store.alist_namespaces(
            suffix=(_MEMORIES,), limit=_PAGE, offset=offset
        )
        owners.update(ns[0] for ns in page if ns == memory_namespace(ns[0]))
        if len(page) < _PAGE:
            return owners
        offset += _PAGE


async def _people(hass: HomeAssistant, store: BaseStore) -> list[dict[str, Any]]:
    """Return everyone who has memories or could be given some."""
    names = {SHARED_MEMORY_USER, *await _memory_owners(store)}
    names |= {
        memory_user(user.name)
        for user in await hass.auth.async_get_users()
        if user.is_active and not user.system_generated and user.name
    }
    return [
        {"user": name, "count": len(await all_memories(store, name))}
        for name in sorted(names, key=str.casefold)
    ]


def register_memory_services(
    hass: HomeAssistant,
    entry: Any,
    register: Callable[..., None],
) -> None:
    """Register the services behind the memory card; they are for admins only."""

    async def _store(call: ServiceCall) -> BaseStore:
        # Like _async_require_admin, a call with no user is refused too: an
        # automation the agent wrote from anyone's request runs with none.
        user = (
            await hass.auth.async_get_user(call.context.user_id)
            if call.context.user_id
            else None
        )
        if user is None or not user.is_admin:
            raise Unauthorized(context=call.context)
        store = entry.runtime_data.store
        if not store_available(store):
            msg = "The memory store is not available: no database is set up."
            raise HomeAssistantError(msg)
        return store

    async def _get(call: ServiceCall) -> dict[str, Any]:
        store = await _store(call)
        result: dict[str, Any] = {
            "people": await _people(hass, store),
            "limit": memory_limit(entry.runtime_data.options),
        }
        if user := call.data.get("user"):
            result["memories"] = [
                _as_record(item) for item in await all_memories(store, user)
            ]
        return result

    async def _update(call: ServiceCall) -> dict[str, Any]:
        store = await _store(call)
        namespace = memory_namespace(call.data["user"])
        if await store.aget(namespace, call.data["memory_id"]) is None:
            return {"status": "not_found"}
        value = {"content": call.data["content"], "context": call.data["context"]}
        await store.aput(namespace, call.data["memory_id"], value)
        return {"status": "ok"}

    async def _delete(call: ServiceCall) -> dict[str, Any]:
        store = await _store(call)
        namespace = memory_namespace(call.data["user"])
        if await store.aget(namespace, call.data["memory_id"]) is None:
            return {"status": "not_found"}
        await store.adelete(namespace, call.data["memory_id"])
        return {"status": "ok"}

    async def _move(call: ServiceCall) -> dict[str, Any]:
        store = await _store(call)
        user, key = call.data["user"], call.data["memory_id"]
        target = call.data["to_user"]
        item = await store.aget(memory_namespace(user), key)
        if item is None:
            return {"status": "not_found"}
        if target == user:
            return {"status": "ok"}
        if await store.aget(memory_namespace(target), key) is not None:
            msg = f"{target} already has a memory with this id; nothing was moved."
            raise HomeAssistantError(msg)
        limit = memory_limit(entry.runtime_data.options)
        if await memories_full(store, target, limit):
            msg = f"{target}'s memory is full ({limit}); nothing was moved."
            raise HomeAssistantError(msg)
        await store.aput(memory_namespace(target), key, item.value)
        await store.adelete(memory_namespace(user), key)
        return {"status": "ok"}

    for name, handler, schema in (
        ("get_memories", _get, GET_MEMORIES_SCHEMA),
        ("update_memory", _update, UPDATE_MEMORY_SCHEMA),
        ("delete_memory", _delete, DELETE_MEMORY_SCHEMA),
        ("move_memory", _move, MOVE_MEMORY_SCHEMA),
    ):
        register(
            hass,
            entry,
            name,
            handler,
            schema=schema,
            supports_response=SupportsResponse.ONLY,
        )
