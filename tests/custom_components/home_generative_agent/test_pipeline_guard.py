# ruff: noqa: S101
"""
Tests for the PIN / local-intents pipeline conflict repair issue.

Background, because the guard exists for a non-obvious reason. The
critical-action PIN can only gate commands the conversation agent receives.
When a pipeline has ``prefer_local_intents`` on, Home Assistant matches the
sentence against its own built-in intents first and, on a match, runs it
without calling the agent at all -- "unlock the front door" included.

``ConversationEntityFeature.CONTROL`` does not prevent that. Its only consumer
installs ``_async_local_fallback_intent_filter``, and ``async_handle_intents``
treats that filter as a REJECT list: it returns None when the filter matches.
The filter matches only HassGetState and media search, so those two go to the
agent and every control command stays local regardless of the flag.

No integration can intercept that path, so the guard's job is to make the
silent gap visible rather than to close it.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from custom_components.home_generative_agent.const import DOMAIN
from custom_components.home_generative_agent.core import pipeline_guard
from custom_components.home_generative_agent.core.pipeline_guard import (
    ISSUE_PIN_BYPASSED,
    async_check_pin_pipeline_conflict,
    find_conflicting_pipelines,
)

ENTITY_ID = "conversation.home_generative_agent"
ENTRY_ID = "entry-1"

# The guard only forwards hass to async_get_pipelines and the issue registry,
# both stubbed here, so it never touches a real HomeAssistant instance.
HASS: Any = None


def _pipeline(name: str, engine: str, *, prefer_local: bool) -> Any:
    return types.SimpleNamespace(
        name=name,
        conversation_engine=engine,
        prefer_local_intents=prefer_local,
    )


@pytest.fixture
def fake_pipelines(monkeypatch: pytest.MonkeyPatch):
    """Install a stub assist_pipeline module the guard imports lazily."""
    installed: list[Any] = []

    mod: Any = types.ModuleType("homeassistant.components.assist_pipeline")
    mod.async_get_pipelines = lambda _hass: list(installed)
    monkeypatch.setitem(sys.modules, "homeassistant.components.assist_pipeline", mod)

    def _set(pipelines: list[Any]) -> None:
        installed[:] = pipelines

    return _set


class _IssueRecorder:
    """Records create/delete calls so tests assert on registry effects."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def create(self, _hass: Any, domain: str, issue_id: str, **kwargs: Any) -> None:
        self.created.append({"domain": domain, "issue_id": issue_id, **kwargs})

    def delete(self, _hass: Any, _domain: str, issue_id: str) -> None:
        self.deleted.append(issue_id)


@pytest.fixture
def issues(monkeypatch: pytest.MonkeyPatch) -> _IssueRecorder:
    rec = _IssueRecorder()
    monkeypatch.setattr(pipeline_guard.ir, "async_create_issue", rec.create)
    monkeypatch.setattr(pipeline_guard.ir, "async_delete_issue", rec.delete)
    return rec


def test_conflict_raises_issue_naming_the_pipeline(fake_pipelines, issues) -> None:
    """PIN on + prefer_local_intents on + routes to us = warn, naming the pipeline."""
    fake_pipelines([_pipeline("Home Voice", ENTITY_ID, prefer_local=True)])

    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)

    assert len(issues.created) == 1
    issue = issues.created[0]
    assert issue["domain"] == DOMAIN
    assert issue["translation_key"] == ISSUE_PIN_BYPASSED
    assert issue["translation_placeholders"]["pipelines"] == "Home Voice"
    assert issue["is_fixable"] is False
    assert not issues.deleted


def test_no_issue_when_pin_disabled(fake_pipelines, issues) -> None:
    """With no PIN there is no guard to disarm, so the setting is not a problem."""
    fake_pipelines([_pipeline("Home Voice", ENTITY_ID, prefer_local=True)])

    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=False)

    assert not issues.created
    assert issues.deleted == [f"{ISSUE_PIN_BYPASSED}_{ENTRY_ID}"]


def test_no_issue_when_local_intents_off(fake_pipelines, issues) -> None:
    """The supported configuration raises nothing."""
    fake_pipelines([_pipeline("Home Voice", ENTITY_ID, prefer_local=False)])

    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)

    assert not issues.created
    assert issues.deleted == [f"{ISSUE_PIN_BYPASSED}_{ENTRY_ID}"]


def test_other_agents_pipelines_are_ignored(fake_pipelines, issues) -> None:
    """
    A risky pipeline pointing at a different agent is not ours to warn about.

    Without the engine check every HGA install would raise this the moment any
    unrelated pipeline enabled local intents.
    """
    fake_pipelines(
        [_pipeline("Other", "conversation.home_assistant", prefer_local=True)]
    )

    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)

    assert not issues.created


def test_multiple_conflicting_pipelines_are_all_named(fake_pipelines, issues) -> None:
    """Every offending pipeline is listed, sorted, so the fix list is complete."""
    fake_pipelines(
        [
            _pipeline("Upstairs", ENTITY_ID, prefer_local=True),
            _pipeline("Garage", ENTITY_ID, prefer_local=True),
            _pipeline("Safe One", ENTITY_ID, prefer_local=False),
        ]
    )

    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)

    assert issues.created[0]["translation_placeholders"]["pipelines"] == (
        "Garage, Upstairs"
    )


def test_issue_id_is_scoped_per_entry(fake_pipelines, issues) -> None:
    """Two config entries must not clobber each other's issue."""
    fake_pipelines([_pipeline("Home Voice", ENTITY_ID, prefer_local=True)])

    async_check_pin_pipeline_conflict(HASS, "entry-a", ENTITY_ID, pin_enabled=True)
    async_check_pin_pipeline_conflict(HASS, "entry-b", ENTITY_ID, pin_enabled=True)

    ids = {issue["issue_id"] for issue in issues.created}
    assert ids == {
        f"{ISSUE_PIN_BYPASSED}_entry-a",
        f"{ISSUE_PIN_BYPASSED}_entry-b",
    }


def test_missing_assist_pipeline_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, issues
) -> None:
    """
    assist_pipeline is an after_dependency and may be absent entirely.

    An optional component that is not installed must degrade to "nothing to
    warn about", never to an exception during entity setup.
    """
    monkeypatch.setitem(sys.modules, "homeassistant.components.assist_pipeline", None)

    assert find_conflicting_pipelines(HASS, ENTITY_ID) == []
    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)
    assert not issues.created


def test_assist_pipeline_present_but_not_set_up(
    monkeypatch: pytest.MonkeyPatch, issues
) -> None:
    """Installed but never set up raises KeyError internally; treat as absent."""
    mod: Any = types.ModuleType("homeassistant.components.assist_pipeline")

    def _boom(_hass: Any) -> list[Any]:
        msg = "assist_pipeline"
        raise KeyError(msg)

    mod.async_get_pipelines = _boom
    monkeypatch.setitem(sys.modules, "homeassistant.components.assist_pipeline", mod)

    assert find_conflicting_pipelines(HASS, ENTITY_ID) == []
    async_check_pin_pipeline_conflict(HASS, ENTRY_ID, ENTITY_ID, pin_enabled=True)
    assert not issues.created
