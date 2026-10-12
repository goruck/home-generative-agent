# ruff: noqa: S101
"""
Tests for the un-pulled Ollama embedding model guard (#757).

The server passes its health check, so setup loads; the missing model only
shows up as 404s from /api/embed. The guard must name the cause in a repair
issue, must not claim "missing" when it cannot list the server's models, and
must not raise the issue for an entry that is no longer running.
"""

from __future__ import annotations

from typing import Any, cast

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
    _dims_issue_id,
    _issue_id,
    async_clear_ollama_embedding_model_issue,
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


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    listings: dict[str, list[str] | None],
    dims: dict[str, int] | None = None,
) -> list[str]:
    """
    Answer /api/tags per server URL; None is an unreachable server.

    ``dims`` is the vector size /api/show reports per server (none reported
    when absent). Returns the URLs asked for a size, in order.
    """
    sizes = dims or {}
    asked: list[str] = []

    async def _fetch(_hass: Any, url: str, *_: Any) -> list[str] | None:
        return listings[url]

    async def _dims(_hass: Any, url: str, *_: Any) -> int | None:
        asked.append(url)
        return sizes.get(url)

    monkeypatch.setattr(embedding_model_guard, "fetch_ollama_models", _fetch)
    monkeypatch.setattr(embedding_model_guard, "fetch_ollama_embedding_dims", _dims)
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
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
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
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
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
        hass, "entry1", urls=[URL, URL2], model=MODEL, expected_dims=DIMS
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
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
    )
    assert _issue(hass) is not None

    _serve(monkeypatch, listings)
    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=urls, model=MODEL, expected_dims=DIMS
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
    _serve(monkeypatch, {URL: ["other:latest"]})

    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
    )

    assert _issue(hass) is None


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
        hass, "entry1", urls=[URL], model="nomic-embed-text", expected_dims=DIMS
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
        hass, "entry1", urls=[URL, URL2], model=MODEL, expected_dims=DIMS
    )

    assert asked == [URL]
    assert _issue(hass) is not None
    assert _dims_issue(hass) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("dims", [{URL: 1024}, {}])
async def test_size_issue_clears_on_right_or_unknown_size(
    hass: Any, monkeypatch: pytest.MonkeyPatch, dims: dict[str, int]
) -> None:
    """A matching size clears the issue, and so does one the server won't report."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})
    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
    )
    assert _dims_issue(hass) is not None

    _serve(monkeypatch, {URL: [MODEL]}, dims=dims)
    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
    )

    assert _dims_issue(hass) is None


@pytest.mark.asyncio
async def test_size_is_held_to_the_store_size_passed_in(
    hass: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind a 768-dimension OpenAI-compatible primary, 768 is right."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})

    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=768
    )

    assert _dims_issue(hass) is None


@pytest.mark.asyncio
async def test_clear_drops_issue(hass: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The clear helper removes both issues."""
    _loaded_entry(hass)
    _serve(monkeypatch, {URL: [MODEL]}, dims={URL: 768})
    await async_sync_ollama_embedding_model_issue(
        hass, "entry1", urls=[URL], model=MODEL, expected_dims=DIMS
    )
    _raise_issue(hass, "entry1")
    assert _issue(hass) is not None
    assert _dims_issue(hass) is not None

    async_clear_ollama_embedding_model_issue(hass, "entry1")

    assert _issue(hass) is None
    assert _dims_issue(hass) is None


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
