"""
Surface an Ollama embedding model that was never pulled onto its server.

The server answers its health check, so setup builds the embeddings client
and loads normally; the missing model only shows up later as a 404 from
``/api/embed``. Long-term memory search then fails open on every turn and the
tool index never builds, which drops tool retrieval to its keyword fallback
(issue #757). Both log errors, but nothing names the cause where a user looks
for it, so this module raises a repair issue that does, and clears it on the
next setup once the model is installed.

Every reachable Ollama server that embeds is checked: the primary and any
Ollama fallback in the embedding chain, which embed with the same configured
model (#749).

Like the other per-entry issues it is not persistent: setup re-syncs it every
time, and unload and removal clear it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from ..const import DOMAIN  # noqa: TID252
from .utils import fetch_ollama_models, ollama_model_installed, redact_url

if TYPE_CHECKING:
    from collections.abc import Sequence

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING = "ollama_embedding_model_missing"

# Setup schedules the check as a background task, and HA runs the entry's
# unload hook (which clears the issue) before it cancels those tasks. A check
# whose reply lands in that gap must not re-raise the issue for an entry that
# is no longer running.
_ACTIVE_STATES = frozenset(
    {ConfigEntryState.SETUP_IN_PROGRESS, ConfigEntryState.LOADED}
)


def _issue_id(entry_id: str) -> str:
    """Per-entry issue id so two config entries cannot clobber each other."""
    return f"{ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING}_{entry_id}"


async def async_sync_ollama_embedding_model_issue(
    hass: HomeAssistant, entry_id: str, *, urls: Sequence[str], model: str | None
) -> None:
    """
    Raise or clear the issue for the entry's Ollama embedding servers.

    ``urls`` are the reachable Ollama servers that embed for this entry; none
    (embeddings are not on Ollama, or there is no database to use them)
    clears the issue. A server that cannot list its models is a "can't tell",
    not a "missing"; one that answers with an empty list has no models at all.
    """
    issue_id = _issue_id(entry_id)
    if not urls or not model:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    listings = await asyncio.gather(*(fetch_ollama_models(hass, url) for url in urls))
    missing = [
        redact_url(url)
        for url, installed in zip(urls, listings, strict=True)
        if installed is not None and not ollama_model_installed(model, installed)
    ]

    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.state not in _ACTIVE_STATES:
        return
    if not missing:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
        return

    servers = ", ".join(missing)
    LOGGER.error(
        "Ollama embedding model %s is not installed on %s, so embeddings from "
        "that server fail. Pull it there (ollama pull %s) or choose an installed "
        "embedding model, then reload the integration.",
        model,
        servers,
        model,
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING,
        translation_placeholders={"model": model, "url": servers},
    )


@callback
def async_clear_ollama_embedding_model_issue(
    hass: HomeAssistant, entry_id: str
) -> None:
    """Drop the issue on unload and on entry removal."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry_id))
