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

GET_MEMORIES_SCHEMA = vol.Schema({vol.Optional("user"): cv.string})
DELETE_MEMORY_SCHEMA = vol.Schema(
    {vol.Required("user"): cv.string, vol.Required("memory_id"): cv.string}
)
UPDATE_MEMORY_SCHEMA = DELETE_MEMORY_SCHEMA.extend(
    {
        vol.Required("content"): vol.All(cv.string, vol.Length(min=1)),
        vol.Optional("context", default=""): cv.string,
    }
)
MOVE_MEMORY_SCHEMA = DELETE_MEMORY_SCHEMA.extend({vol.Required("to_user"): cv.string})


def memory_user(name: str | None) -> str:
    """Return the store's name for a user; no name is the shared satellite store."""
    # Memory namespace labels cannot contain punctuation.
    cleaned = (name or SHARED_MEMORY_USER).translate(
        str.maketrans("", "", string.punctuation)
    )
    return cleaned or SHARED_MEMORY_USER


def memory_namespace(user: str) -> tuple[str, str]:
    """Return the store namespace holding a user's memories."""
    return (user, _MEMORIES)


def memory_limit(options: Mapping[str, Any]) -> int:
    """Return how many memories one person may keep; 0 means unlimited."""
    return max(int(options.get(CONF_MEMORY_MAX_PER_PERSON) or 0), 0)


async def memories_full(store: BaseStore, user: str, limit: int) -> bool:
    """Return whether a user has no room for another memory."""
    if not limit:
        return False
    return len(await store.asearch(memory_namespace(user), limit=limit)) >= limit


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


async def _people(hass: HomeAssistant, store: BaseStore) -> list[dict[str, Any]]:
    """Return everyone who has memories or could be given some."""
    namespaces = await store.alist_namespaces(suffix=(_MEMORIES,), limit=1000)
    names = {SHARED_MEMORY_USER, *(ns[0] for ns in namespaces if len(ns) == 2)}  # noqa: PLR2004
    names |= {
        memory_user(user.name)
        for user in await hass.auth.async_get_users()
        if user.is_active and not user.system_generated and user.name
    }
    return [
        {"user": name, "count": len(await all_memories(store, name))}
        for name in sorted(names, key=str.casefold)
    ]


async def _async_move(store: BaseStore, call: ServiceCall) -> dict[str, Any]:
    user, key = call.data["user"], call.data["memory_id"]
    target = memory_user(call.data["to_user"])
    item = await store.aget(memory_namespace(user), key)
    if item is None:
        return {"status": "not_found"}
    await store.aput(memory_namespace(target), key, item.value)
    if target != user:
        await store.adelete(memory_namespace(user), key)
    return {"status": "ok"}


def register_memory_services(
    hass: HomeAssistant,
    entry: Any,
    register: Callable[..., None],
) -> None:
    """Register the services behind the memory card; they are for admins only."""

    async def _store(call: ServiceCall) -> BaseStore:
        # An automation has no user; anyone who could write one is trusted.
        if call.context.user_id:
            user = await hass.auth.async_get_user(call.context.user_id)
            if user is None or not user.is_admin:
                raise Unauthorized(context=call.context)
        store = entry.runtime_data.store
        if store is None:
            msg = "The memory store is not available."
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
        return await _async_move(await _store(call), call)

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
