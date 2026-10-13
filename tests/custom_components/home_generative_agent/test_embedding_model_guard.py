# ruff: noqa: S101
"""
Tests for the un-pulled Ollama embedding model guard (#757).

The server passes its health check, so setup loads; the missing model only
shows up as 404s from /api/embed. The guard must name the cause in a repair
issue, must not claim "missing" when it cannot list the server's models, and
must not raise the issue for an entry that is no longer running.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self, cast

import httpx
import psycopg
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.home_generative_agent as hga_component
from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import embedding_model_guard, utils
from custom_components.home_generative_agent.core.embedding_model_guard import (
    ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH,
    ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING,
    ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH,
    _dims_issue_id,
    _fallback_dims_issue_id,
    _issue_id,
    async_clear_ollama_embedding_model_issue,
    async_store_vector_dims,
    async_sync_ollama_embedding_model_issue,
)
from custom_components.home_generative_agent.core.utils import (
    ollama_model_installed,
)

from .test_database_guard import _setup_without_database

URL = "http://ollama:11434"
URL2 = "http://fallback:11434"
MODEL = "qwen3-embedding:0.6b"
DIMS = 1024


def _issue(hass: Any, entry_id: str = "entry1") -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _issue_id(entry_id))


def _loaded_entry(hass: Any) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN, entry_id="entry1", state=ConfigEntryState.LOADED
    )
    entry.add_to_hass(hass)
    return entry


_NO_TABLE = object()


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    listings: dict[str, list[str] | None],
    dims: dict[str, int] | None = None,
    store: object = _NO_TABLE,
) -> list[str]:
    """
    Answer /api/tags per server URL; None is an unreachable server.

    ``dims`` is the vector size /api/show reports per server (none reported
    when absent). ``store`` is the store column's size: by default no table
    yet (so the configured size applies), None for a database that cannot be
    asked. Returns the URLs asked for a size, in order.
    """
    sizes = dims or {}
    asked: list[str] = []

    async def _fetch(_hass: Any, url: str, *_: Any) -> list[str] | None:
        return listings[url]

    async def _dims(_hass: Any, url: str, *_: Any) -> int | None:
        asked.append(url)
        return sizes.get(url)

    async def _store(_pool: Any, *, default: int | None) -> int | None:
        return default if store is _NO_TABLE else cast("int | None", store)

    monkeypatch.setattr(embedding_model_guard, "fetch_ollama_models", _fetch)
    monkeypatch.setattr(utils, "fetch_ollama_embedding_dims", _dims)
    monkeypatch.setattr(embedding_model_guard, "async_store_vector_dims", _store)
    return asked


# ---------------------------------------------------------------------------
# name matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "installed", "expected"),
    [
        ("mxbai-embed-large", ["mxbai-embed-large:latest"], True),
        ("mxbai-embed-large:latest", ["mxbai-embed-large"], True),
        ("qwen3-embedding:0.6b", ["qwen3-embedding:0.6b"], True),
        ("qwen3-embedding:0.6b", ["qwen3-embedding:4b"], False),
        ("qwen3-embedding:0.6b", ["mxbai-embed-large:latest"], False),
        ("qwen3-embedding:0.6b", [], False),
        # Ollama resolves these to the same model.
        ("MXBAI-Embed-Large", ["mxbai-embed-large:latest"], True),
        ("library/mxbai-embed-large", ["mxbai-embed-large:latest"], True),
        (
            "registry.ollama.ai/library/mxbai-embed-large:latest",
            ["mxbai-embed-large:latest"],
            True,
        ),
        # A registry host:port colon is not a tag.
        ("reg:5000/lib/embed", ["reg:5000/lib/embed:latest"], True),
        (" mxbai-embed-large ", ["mxbai-embed-large:latest"], True),
        # /api/tags does not list digests, so a pin cannot be checked.
        ("nomic-embed-text@sha256:abc", ["mxbai-embed-large:latest"], True),
    ],
)
def test_ollama_model_installed(
    model: str, installed: list[str], *, expected: bool
) -> None:
    """Configured names match /api/tags names the way Ollama resolves them."""
    assert ollama_model_installed(model, installed) is expected


# ---------------------------------------------------------------------------
# which servers setup checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Ollama is the selected embedder and healthy.
        ({}, [URL]),
        # Gemini or OpenAI embeds; Ollama is healthy only for chat.
        ({"primary_selected": False}, []),
        # Primary unreachable: its client is never built, the health check
        # already reported it.
        ({"health": {URL: False}}, []),
        # No database: nothing uses embeddings.
        ({"has_database": False}, []),
        # A healthy Ollama fallback embeds with the same model (#749).
        ({"fallback_urls": [URL2], "health": {URL: True, URL2: True}}, [URL, URL2]),
        ({"fallback_urls": [URL2], "health": {URL: True, URL2: False}}, [URL]),
        # Non-Ollama primary with an Ollama fallback.
        (
            {
                "primary_selected": False,
                "fallback_urls": [URL2],
                "health": {URL2: True},
            },
            [URL2],
        ),
        # The same server as primary and fallback is checked once.
        ({"fallback_urls": [URL]}, [URL]),
    ],
)
def test_ollama_embedding_urls_to_check(
    kwargs: dict[str, Any], expected: list[str]
) -> None:
    """Setup passes exactly the reachable Ollama servers that embed."""
    args: dict[str, Any] = {
        "primary_url": URL,
        "primary_selected": True,
        "fallback_urls": [],
        "health": {URL: True},
        "has_database": True,
        **kwargs,
    }
    urls_to_check = cast("Any", hga_component)._ollama_embedding_urls_to_check
    assert urls_to_check(**args) == expected


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"embedding_model_provider": "ollama"}, 1024),
        ({"embedding_model_provider": "openai"}, 1024),
        ({"embedding_model_provider": "gemini"}, 1024),
        ({}, 1024),
        # An OpenAI-compatible primary creates the store at its own size.
        (
            {
                "embedding_model_provider": "openai_compatible",
                "openai_compatible_embedding_dims": 768,
            },
            768,
        ),
        ({"embedding_model_provider": "openai_compatible"}, 768),
    ],
)
def test_configured_embedding_dims(options: dict[str, Any], expected: int) -> None:
    """Setup hands the check the size a new store is created with."""
    configured = cast("Any", hga_component)._configured_embedding_dims
    assert configured(options) == expected


# ---------------------------------------------------------------------------
# raising and clearing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_model_raises_issue_and_logs(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The #755 setup: a reachable server without the configured model."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: ["mxbai-embed-large:latest"]})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_key == ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING
    assert issue.severity == ir.IssueSeverity.ERROR
    assert issue.translation_placeholders == {"model": MODEL, "url": URL}
    assert f"ollama pull {MODEL}" in caplog.text


@pytest.mark.asyncio
async def test_empty_server_counts_as_missing(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that answers with no models at all lacks this one too."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: []})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert _issue(hass) is not None


@pytest.mark.asyncio
async def test_issue_names_only_the_servers_missing_the_model(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback server without the model is named; a healthy primary is not."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL], URL2: ["other:latest"]})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL, URL2],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_placeholders == {"model": MODEL, "url": URL2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("urls", "listings"),
    [
        # Installed: the issue clears once the user pulls the model.
        ([URL], {URL: [MODEL]}),
        # The server could not list its models: can't tell.
        ([URL], {URL: None}),
        # Embeddings are not on a reachable Ollama server at all.
        ([], {}),
    ],
)
async def test_issue_clears_when_not_known_missing(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    urls: list[str],
    listings: dict[str, list[str] | None],
) -> None:
    """Every outcome other than a confirmed-missing model clears the issue."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: ["other:latest"]})
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )
    assert _issue(hass) is not None

    _serve(monkeypatch, listings)
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=urls,
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert _issue(hass) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [ConfigEntryState.NOT_LOADED, ConfigEntryState.UNLOAD_IN_PROGRESS, None],
)
async def test_no_issue_for_an_entry_that_is_not_running(
    hass: Any, monkeypatch: pytest.MonkeyPatch, state: ConfigEntryState | None
) -> None:
    """
    A reply that lands after unload must not re-raise the issue.

    HA runs the unload hook (which clears it) before cancelling the setup's
    background tasks, so the check can finish in between.
    """
    if state is not None:
        MockConfigEntry(domain=DOMAIN, entry_id="entry1", state=state).add_to_hass(hass)
    _serve(monkeypatch, {URL: ["other:latest"], URL2: [MODEL]}, dims={URL2: 768})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL, URL2],
        model=MODEL,
        primary_url=URL2,
        configured_dims=DIMS,
        pool=None,
    )

    assert _issue(hass) is None
    assert _dims_issue(hass) is None


def _dims_issue(hass: Any, entry_id: str = "entry1") -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _dims_issue_id(entry_id))


@pytest.mark.asyncio
async def test_wrong_size_model_raises_its_own_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The #755 follow-up: an installed model of the wrong vector size."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: ["nomic-embed-text:latest"]}, dims={URL: 768})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model="nomic-embed-text",
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert _issue(hass) is None
    issue = _dims_issue(hass)
    assert issue is not None
    assert issue.translation_key == ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH
    assert issue.severity == ir.IssueSeverity.ERROR
    assert issue.translation_placeholders == {
        "model": "nomic-embed-text",
        "dims": "768",
        "expected": "1024",
        "url": URL,
    }
    assert "768-dimension" in caplog.text


@pytest.mark.asyncio
async def test_size_is_checked_only_where_the_model_is_installed(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server missing the model gets the missing issue, not a size check."""
    _loaded_entry(hass)
    asked = _serve(
        monkeypatch, {URL: [MODEL], URL2: ["other:latest"]}, dims={URL: 1024}
    )

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL, URL2],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert asked == [URL]
    assert _issue(hass) is not None
    assert _dims_issue(hass) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("dims", "cleared"), [({URL: 1024}, True), ({}, False)])
async def test_size_issue_clears_only_on_a_measured_right_size(
    hass: Any, monkeypatch: pytest.MonkeyPatch, dims: dict[str, int], *, cleared: bool
) -> None:
    """
    A matching size clears the issue; a size the server won't report keeps it.

    An unmeasured primary is neither right nor wrong, so it decides nothing.
    """
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )
    assert _dims_issue(hass) is not None

    _serve(monkeypatch, {URL: [MODEL]}, dims=dims)
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert (_dims_issue(hass) is None) is cleared


def _fallback_dims_issue(hass: Any, entry_id: str = "entry1") -> ir.IssueEntry | None:
    return ir.async_get(hass).async_get_issue(DOMAIN, _fallback_dims_issue_id(entry_id))


@pytest.mark.asyncio
@pytest.mark.parametrize(("reported", "raised"), [(768, False), (1024, True)])
async def test_size_is_held_to_the_existing_store_column(
    hass: Any, monkeypatch: pytest.MonkeyPatch, reported: int, *, raised: bool
) -> None:
    """
    A store first created at 768 keeps 768, whatever the provider is now.

    The column gets its size once, so a 768 model fits it and the configured
    1024 does not.
    """
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: reported}, store=768)

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        primary_url=URL,
        configured_dims=1024,
        model=MODEL,
        pool=None,
    )

    issue = _dims_issue(hass)
    assert (issue is not None) is raised
    if issue is not None:
        assert (issue.translation_placeholders or {})["expected"] == "768"


@pytest.mark.asyncio
async def test_size_is_not_checked_when_the_store_size_is_unknown(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database that cannot be asked skips the size check, leaving its issue."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )
    assert _dims_issue(hass) is not None

    asked = _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768}, store=None)
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )

    assert asked == []
    assert _dims_issue(hass) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_url", [URL, None])
async def test_wrong_size_fallback_gets_the_fallback_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch, primary_url: str | None
) -> None:
    """
    A wrong-size fallback is reported separately from the primary.

    Its fix differs: behind a non-Ollama primary (primary_url None) the
    fallback's model is not set in the form, so the issue says to remove it.
    """
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL], URL2: [MODEL]}, dims={URL: 1024, URL2: 768})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL, URL2],
        primary_url=primary_url,
        configured_dims=DIMS,
        model=MODEL,
        pool=None,
    )

    issue = _fallback_dims_issue(hass)
    assert issue is not None
    assert issue.translation_key == ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH
    assert issue.translation_placeholders == {
        "model": MODEL,
        "dims": "768",
        "expected": "1024",
        "url": URL2,
    }
    assert _dims_issue(hass) is None


@pytest.mark.asyncio
async def test_wrong_primary_lists_every_wrong_server(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A wrong primary and its same-model fallback make one issue, not two.

    Choosing another model fixes both, so the fallback issue's advice
    (remove the fallback, or match the primary's build) would not apply.
    """
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL], URL2: [MODEL]}, dims={URL: 768, URL2: 768})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL, URL2],
        primary_url=URL,
        configured_dims=DIMS,
        model=MODEL,
        pool=None,
    )

    issue = _dims_issue(hass)
    assert issue is not None
    assert (issue.translation_placeholders or {})["url"] == f"{URL}, {URL2}"
    assert _fallback_dims_issue(hass) is None


async def _sync(hass: Any, urls: list[str], primary_url: str | None) -> None:
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=urls,
        primary_url=primary_url,
        configured_dims=DIMS,
        model=MODEL,
        pool=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("urls", "listings", "dims"),
    [
        # Down at boot: setup leaves the primary out of the servers checked.
        ([URL2], {URL2: [MODEL]}, {URL2: 768}),
        # Could not list its models.
        ([URL, URL2], {URL: None, URL2: [MODEL]}, {URL2: 768}),
        # Has no such model (the missing issue covers it).
        ([URL, URL2], {URL: ["other:latest"], URL2: [MODEL]}, {URL2: 768}),
        # Did not report a size.
        ([URL, URL2], {URL: [MODEL], URL2: [MODEL]}, {URL2: 768}),
    ],
)
async def test_unmeasured_primary_leaves_size_issues_alone(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    urls: list[str],
    listings: dict[str, list[str] | None],
    dims: dict[str, int],
) -> None:
    """
    An Ollama primary that was not measured is neither right nor wrong.

    A wrong fallback running the same model says nothing about which fix
    applies, so no size issue is raised, and an earlier one is kept rather
    than replaced by advice that assumes the primary is fine.
    """
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL], URL2: [MODEL]}, dims={URL: 768, URL2: 768})
    await _sync(hass, [URL, URL2], URL)
    assert _dims_issue(hass) is not None

    _serve(monkeypatch, listings, dims=dims)
    await _sync(hass, urls, URL)

    assert _dims_issue(hass) is not None
    assert _fallback_dims_issue(hass) is None


@pytest.mark.asyncio
async def test_size_issue_moves_between_primary_and_fallback(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixing the primary swaps its issue for the fallback's, and back."""
    _loaded_entry(hass)
    listings: dict[str, list[str] | None] = {URL: [MODEL], URL2: [MODEL]}
    _serve(monkeypatch, listings, dims={URL: 768, URL2: 768})
    await _sync(hass, [URL, URL2], URL)
    assert _dims_issue(hass) is not None
    assert _fallback_dims_issue(hass) is None

    _serve(monkeypatch, listings, dims={URL: 1024, URL2: 768})
    await _sync(hass, [URL, URL2], URL)
    assert _dims_issue(hass) is None
    assert _fallback_dims_issue(hass) is not None

    _serve(monkeypatch, listings, dims={URL: 768, URL2: 768})
    await _sync(hass, [URL, URL2], URL)
    assert _dims_issue(hass) is not None
    assert _fallback_dims_issue(hass) is None


@pytest.mark.asyncio
async def test_unmeasured_fallback_keeps_the_fallback_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback whose size cannot be read now does not clear its issue."""
    _loaded_entry(hass)
    listings: dict[str, list[str] | None] = {URL: [MODEL], URL2: [MODEL]}
    _serve(monkeypatch, listings, dims={URL: 1024, URL2: 768})
    await _sync(hass, [URL, URL2], URL)
    assert _fallback_dims_issue(hass) is not None

    _serve(monkeypatch, {URL: [MODEL], URL2: None}, dims={URL: 1024})
    await _sync(hass, [URL, URL2], URL)

    assert _fallback_dims_issue(hass) is not None


def test_format_embedding_dims_pairs_servers_with_sizes() -> None:
    """Servers that disagree each carry their own size."""
    assert utils.format_embedding_dims({URL: 768}) == (URL, "768")
    assert utils.format_embedding_dims({URL: 768, URL2: 768}) == (
        f"{URL}, {URL2}",
        "768",
    )
    assert utils.format_embedding_dims({URL: 768, URL2: 384}) == (
        f"{URL} (768), {URL2} (384)",
        "384, 768",
    )


@pytest.mark.asyncio
async def test_size_is_not_checked_on_a_store_setup_already_made(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A missing table after setup created one is a broken store, not a new one.

    Its size cannot be told, so it is not held to the configured default.
    """
    MockConfigEntry(
        domain=DOMAIN,
        entry_id="entry1",
        state=ConfigEntryState.LOADED,
        data={"vectors_bootstrapped": True},
    ).add_to_hass(hass)
    asked = _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        primary_url=URL,
        configured_dims=DIMS,
        model=MODEL,
        pool=None,
    )

    assert asked == []
    assert _dims_issue(hass) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("urls", "model"), [([], MODEL), ([URL], None)])
async def test_size_issues_clear_when_embeddings_leave_ollama(
    hass: Any, monkeypatch: pytest.MonkeyPatch, urls: list[str], model: str | None
) -> None:
    """Moving embeddings off Ollama (or dropping the database) clears them."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL], URL2: [MODEL]}, dims={URL: 768, URL2: 768})
    for issue_id in (_dims_issue_id("entry1"), _fallback_dims_issue_id("entry1")):
        ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id,
            is_fixable=False,
            severity=ir.IssueSeverity.ERROR,
            translation_key=ISSUE_OLLAMA_EMBEDDING_DIMS_MISMATCH,
        )

    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=urls,
        primary_url=URL,
        configured_dims=DIMS,
        model=model,
        pool=None,
    )

    assert _dims_issue(hass) is None
    assert _fallback_dims_issue(hass) is None


@pytest.mark.asyncio
async def test_clear_drops_issue(hass: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The clear helper removes all three issues."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})
    await async_sync_ollama_embedding_model_issue(
        hass,
        "entry1",
        urls=[URL],
        model=MODEL,
        primary_url=URL,
        configured_dims=DIMS,
        pool=None,
    )
    _raise_issue(hass, "entry1")
    assert _issue(hass) is not None
    assert _dims_issue(hass) is not None

    ir.async_create_issue(
        hass,
        DOMAIN,
        _fallback_dims_issue_id("entry1"),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_OLLAMA_FALLBACK_EMBEDDING_DIMS_MISMATCH,
    )

    async_clear_ollama_embedding_model_issue(hass, "entry1")

    assert _issue(hass) is None
    assert _dims_issue(hass) is None
    assert _fallback_dims_issue(hass) is None


def _raise_issue(hass: Any, entry_id: str) -> None:
    ir.async_create_issue(
        hass,
        DOMAIN,
        _issue_id(entry_id),
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_OLLAMA_EMBEDDING_MODEL_MISSING,
        translation_placeholders={"model": MODEL, "url": URL},
    )


@pytest.mark.asyncio
async def test_unload_drops_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry's unload hook clears the issue."""
    entry = await _setup_without_database(hass, monkeypatch)
    _raise_issue(hass, entry.entry_id)

    result = await cast("Any", hga_component).async_unload_entry(hass, entry)

    assert result is True
    assert _issue(hass, entry.entry_id) is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("enable_custom_integrations")
async def test_removing_the_entry_through_ha_clears_the_issue(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal through HA's dispatch reaches async_remove_entry and clears it."""
    entry = await _setup_without_database(hass, monkeypatch)
    _raise_issue(hass, entry.entry_id)

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert _issue(hass, entry.entry_id) is None


# ---------------------------------------------------------------------------
# /api/tags parsing
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, {"models": [{"name": "a:latest"}, {"name": "a:latest"}]}, ["a:latest"]),
        # A server with nothing pulled: known empty, so "missing".
        (200, {"models": []}, []),
        # Entries named only under "model" still count.
        (200, {"models": [{"model": "b:latest"}]}, ["b:latest"]),
        # Entries with no usable name say nothing: unknown, not "none".
        (200, {"models": [{"size": 1}, "junk"]}, None),
        (200, {"other": []}, None),
        (200, ["not", "a", "dict"], None),
        (200, ValueError("not json"), None),
        (503, {"models": []}, None),
    ],
)
async def test_fetch_ollama_models_tells_empty_from_unknown(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: Any,
    expected: list[str] | None,
) -> None:
    """Only a readable model listing can prove a model is missing."""

    class _Client:
        async def get(self, *_: Any, **__: Any) -> _Resp:
            return _Resp(status, payload)

    monkeypatch.setattr(utils, "get_async_client", lambda _hass: _Client())

    assert await utils.fetch_ollama_models(hass, URL) == expected
    assert await utils.list_ollama_models(hass, URL) == (expected or [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (
            200,
            {
                "model_info": {
                    "general.architecture": "nomic-bert",
                    "nomic-bert.embedding_length": 768,
                }
            },
            768,
        ),
        (
            200,
            {
                "model_info": {
                    "general.architecture": "bert",
                    "bert.embedding_length": 1024,
                }
            },
            1024,
        ),
        # No architecture, no size field, or junk: unknown, so not checked.
        (200, {"model_info": {"bert.embedding_length": 1024}}, None),
        (200, {"model_info": {"general.architecture": "bert"}}, None),
        (
            200,
            {
                "model_info": {
                    "general.architecture": "bert",
                    "bert.embedding_length": True,
                }
            },
            None,
        ),
        (200, {"model_info": "junk"}, None),
        (200, ValueError("not json"), None),
        # The model is not on the server.
        (404, {"error": "model 'x:latest' not found"}, None),
    ],
)
async def test_fetch_ollama_embedding_dims(
    hass: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: Any,
    expected: int | None,
) -> None:
    """The vector size is read from /api/show, or reported unknown."""
    sent: list[Any] = []

    class _Client:
        async def post(self, url: str, **kwargs: Any) -> _Resp:
            sent.append((url, kwargs.get("json")))
            return _Resp(status, payload)

    monkeypatch.setattr(utils, "get_async_client", lambda _hass: _Client())

    assert await utils.fetch_ollama_embedding_dims(hass, URL, "m") == expected
    assert sent == [(f"{URL}/api/show", {"model": "m"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"model_info": {"general.architecture": "bert", "bert.embedding_length": 0}},
        {
            "model_info": {
                "general.architecture": "bert",
                "bert.embedding_length": "768",
            }
        },
        [],
    ],
)
async def test_fetch_ollama_embedding_dims_rejects_odd_sizes(
    hass: Any, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    """A zero, a string or a non-object reply is no size at all."""

    class _Client:
        async def post(self, *_: Any, **__: Any) -> _Resp:
            return _Resp(200, payload)

    monkeypatch.setattr(utils, "get_async_client", lambda _hass: _Client())

    assert await utils.fetch_ollama_embedding_dims(hass, URL, "m") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), httpx.ConnectError("down")])
async def test_fetch_ollama_embedding_dims_unreachable(
    hass: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """An unreachable server reports no size, and so do blank inputs."""

    class _Client:
        async def post(self, *_: Any, **__: Any) -> _Resp:
            raise error

    monkeypatch.setattr(utils, "get_async_client", lambda _hass: _Client())

    assert await utils.fetch_ollama_embedding_dims(hass, URL, "m") is None
    assert await utils.fetch_ollama_embedding_dims(hass, "", "m") is None
    assert await utils.fetch_ollama_embedding_dims(hass, URL, "") is None


class _Cursor:
    def __init__(self, row: Any, error: Exception | None) -> None:
        self._row = row
        self._error = error

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, sql: str, params: dict[str, str]) -> None:
        if self._error is not None:
            raise self._error
        if self._row == "stall":
            await asyncio.sleep(3600)
        assert "to_regclass(%(table)s)" in sql
        assert params == {"table": "store_vectors", "column": "embedding"}

    async def fetchone(self) -> Any:
        return self._row


class _Conn:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def cursor(self) -> _Cursor:
        return self._cursor


class _Pool:
    def __init__(self, row: Any = None, error: Exception | None = None) -> None:
        self._conn = _Conn(_Cursor(row, error))

    def connection(self, **_: Any) -> _Conn:
        return self._conn


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pool", "expected"),
    [
        # The column keeps the size the store was created with.
        (_Pool({"has_table": True, "dims": 768}), 768),
        # No table yet: the size a new store would be created with.
        (_Pool({"has_table": False, "dims": None}), 1024),
        # A column without a fixed size, or no column: can't tell.
        (_Pool({"has_table": True, "dims": -1}), None),
        (_Pool({"has_table": True, "dims": None}), None),
        # No database, or one that cannot be asked: can't tell.
        (None, None),
        (_Pool(error=psycopg.OperationalError("pool closed")), None),
    ],
)
async def test_async_store_vector_dims(pool: Any, expected: int | None) -> None:
    """The store's size comes from its column, not from the configuration."""
    assert await async_store_vector_dims(pool, default=1024) == expected


@pytest.mark.asyncio
async def test_async_store_vector_dims_without_a_default() -> None:
    """With no default (setup already made the store), a missing table is unknown."""
    pool: Any = _Pool({"has_table": False, "dims": None})
    assert await async_store_vector_dims(pool, default=None) is None


@pytest.mark.asyncio
async def test_async_store_vector_dims_gives_up_on_a_stalled_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query on a connection that stopped replying times out as unknown."""
    monkeypatch.setattr(embedding_model_guard, "_STORE_DIMS_TIMEOUT_S", 0.05)
    stalled: Any = _Pool("stall")
    assert await async_store_vector_dims(stalled, default=1024) is None


def test_store_vector_names_match_langgraph() -> None:
    """The table and column read are the ones langgraph's migration creates."""
    from langgraph.store.postgres.base import VECTOR_MIGRATIONS  # noqa: PLC0415

    create = next(
        m.sql
        for m in VECTOR_MIGRATIONS
        if f"CREATE TABLE IF NOT EXISTS {embedding_model_guard.STORE_VECTORS_TABLE}"
        in m.sql
    )
    assert f"{embedding_model_guard.STORE_VECTORS_COLUMN} %(vector_type)s" in create


@pytest.mark.asyncio
async def test_malformed_url_is_unknown_not_an_error(hass: Any) -> None:
    """A malformed saved URL is skipped like an unreachable server."""
    assert await utils.fetch_ollama_models(hass, "http://[::1") is None
    assert await utils.fetch_ollama_embedding_dims(hass, "http://[::1", "m") is None
