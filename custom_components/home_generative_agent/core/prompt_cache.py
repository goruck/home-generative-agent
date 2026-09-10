"""
Provider-aware shaping of the chat system message for prompt caching.

Cloud providers cache a request by exact prefix — Anthropic in the order
tools → system → messages, OpenAI and Gemini implicitly over the same
prefix — so a value that changes every turn must sit *after* the content
that does not. The conversation entity therefore renders the system prompt
as two parts: a stable block (instructions, tool guidance, the exposed-entity
context) and a volatile tail (the wall-clock time, retrieved memories, the
running summary). The graph carries them as two text blocks in one
``SystemMessage`` and this module decides, per concrete model, what each
provider actually receives:

* Anthropic gets both blocks with an explicit ``cache_control`` breakpoint on
  the stable one. That breakpoint is what makes the stable prefix a cache
  entry in its own right; the top-level automatic breakpoint configured on
  the client only ever marks the *last* block, whose prefix (time, memories,
  history) differs on every turn (issue #617).
* Every other provider gets the two blocks collapsed into the plain string
  it always received, so no Anthropic-only key reaches an API that rejects
  unknown content-part fields (OpenAI does).

The decision is made against the *concrete* model that is about to be
called, not the configured provider: a fallback chain can route a turn to a
different provider than the primary, and setup can select a fallback when
the primary is unavailable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

# ``ChatAnthropic._llm_type``; langchain-anthropic itself keys the direct-API
# request shape on this value, so it is the stable discriminator.
ANTHROPIC_LLM_TYPE = "anthropic-chat"

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}

# Runnable wrappers between the graph and the chat model: ``.bind_tools`` and
# ``.with_config`` yield a ``RunnableBinding`` (``.bound``), while
# ``.configurable_fields`` yields a ``RunnableConfigurableFields``
# (``.default``). They nest in either order.
_WRAPPER_ATTRS = ("bound", "default")
_MAX_UNWRAP_DEPTH = 8


def concrete_chat_model(model: Any) -> BaseChatModel | None:
    """
    Return the chat model beneath LangChain's runnable wrappers, if any.

    ``None`` when the object is not a wrapped ``BaseChatModel`` — a fallback
    chain, a placeholder, a test double — so callers can leave the messages
    untouched rather than guess.
    """
    current = model
    for _ in range(_MAX_UNWRAP_DEPTH):
        if isinstance(current, BaseChatModel):
            return current
        for attr in _WRAPPER_ATTRS:
            inner = getattr(current, attr, None)
            if inner is not None and inner is not current:
                current = inner
                break
        else:
            return None
    return None


def is_anthropic_chat_model(model: Any) -> bool:
    """Return True when ``model`` is (a wrapped) direct-API ``ChatAnthropic``."""
    concrete = concrete_chat_model(model)
    if concrete is None:
        return False
    # ``_llm_type`` is the abstract discriminator every BaseChatModel defines.
    return getattr(concrete, "_llm_type", None) == ANTHROPIC_LLM_TYPE


def _text_blocks(content: Any) -> list[str] | None:
    """Return the block texts when ``content`` is a list of text blocks only."""
    if not isinstance(content, list) or not content:
        return None
    texts: list[str] = []
    for block in content:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            return None
        texts.append(block["text"])
    return texts


def build_system_message(stable: str, volatile: str) -> SystemMessage:
    """
    Compose the system message from its stable prefix and volatile tail.

    A single string when there is no volatile tail — nothing to put a
    breakpoint in front of — otherwise two text blocks that
    :func:`adapt_system_message_for_model` shapes per provider at call time.
    """
    if not volatile:
        return SystemMessage(content=stable)
    return SystemMessage(
        content=[
            {"type": "text", "text": stable},
            {"type": "text", "text": volatile},
        ]
    )


def adapt_system_message_for_model(model: Any, messages: Any) -> Any:
    """
    Shape a two-block system message for the concrete model being called.

    Returns the input unchanged unless ``messages`` is a message list whose
    ``SystemMessage`` carries text blocks and ``model`` resolves to a concrete
    chat model. Never mutates the input: the graph reuses the same list for
    its sampling-parameter retry and a fallback chain hands it to each member.
    """
    if not isinstance(messages, list):
        return messages
    concrete = concrete_chat_model(model)
    if concrete is None:
        return messages
    anthropic = is_anthropic_chat_model(concrete)

    adapted: list[Any] = []
    changed = False
    for message in messages:
        if not isinstance(message, SystemMessage):
            adapted.append(message)
            continue
        texts = _text_blocks(message.content)
        if texts is None:
            adapted.append(message)
            continue
        changed = True
        if anthropic:
            blocks: list[dict[str, Any]] = [
                {"type": "text", "text": text} for text in texts
            ]
            # The breakpoint goes on the *first* block: everything up to and
            # including it (tools, then this block) is the reusable prefix.
            blocks[0]["cache_control"] = dict(CACHE_CONTROL_EPHEMERAL)
            adapted.append(message.model_copy(update={"content": blocks}))
        else:
            adapted.append(message.model_copy(update={"content": "\n".join(texts)}))
    return adapted if changed else messages
