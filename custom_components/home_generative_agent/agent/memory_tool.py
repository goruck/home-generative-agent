"""Memory management tool for Home Generative Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from homeassistant.util import ulid
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore  # noqa: TC002

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


@tool(parse_docstring=True)
async def upsert_memory(  # noqa: D417
    content: str,
    context: str = "",
    *,
    memory_id: str = "",
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    INSERT or UPDATE a memory about users in the database.

    You MUST use this tool to INSERT or UPDATE memories about users.
    Examples of memories are specific facts or concepts learned from interactions
    with users. If a memory conflicts with an existing one then just UPDATE the
    existing one by passing in "memory_id" and DO NOT create two memories that are
    the same. If the user corrects a memory then UPDATE it.

    Args:
        content: The main content of the memory.
            e.g., "I would like to learn french."
        context: Additional relevant context for the memory, if any.
            e.g., "This was mentioned while discussing career options in Europe."
        memory_id: The memory to overwrite.
            ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    mem_id = memory_id or ulid.ulid_now()

    user_id = config["configurable"]["user_id"]
    await store.aput(
        namespace=(user_id, "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
