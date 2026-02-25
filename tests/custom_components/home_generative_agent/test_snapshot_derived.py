# ruff: noqa: S101
"""Tests for snapshot derived context helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from custom_components.home_generative_agent.snapshot.derived import derive_context


@dataclass
class _FakeState:
    """Minimal stand-in for homeassistant.core.State."""

    entity_id: str
    domain: str
    state: str
    attributes: dict[str, Any] = field(default_factory=dict)


_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_TZ = "UTC"


def _derive(states: list[_FakeState]) -> dict[str, Any]:
    return derive_context(
        now=_NOW,
        timezone=_TZ,
        sun_state=None,
        all_states=states,  # type: ignore[arg-type]
        area_lookup={},
    )


# ---------------------------------------------------------------------------
# Presence: all home
# ---------------------------------------------------------------------------


def test_all_home_populates_people_home() -> None:
    """When every person entity is 'home', people_home lists them all."""
    states = [
        _FakeState(
            "person.alice",
            "person",
            "home",
            attributes={"friendly_name": "Alice"},
        ),
        _FakeState(
            "person.bob",
            "person",
            "home",
            attributes={"friendly_name": "Bob"},
        ),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == ["Alice", "Bob"]
    assert ctx["people_away"] == []
    assert ctx["anyone_home"] is True


# ---------------------------------------------------------------------------
# Presence: none home
# ---------------------------------------------------------------------------


def test_none_home_populates_people_away() -> None:
    """When no person entity is 'home', people_away lists them all."""
    states = [
        _FakeState(
            "person.alice",
            "person",
            "not_home",
            attributes={"friendly_name": "Alice"},
        ),
        _FakeState(
            "person.bob",
            "person",
            "work",
            attributes={"friendly_name": "Bob"},
        ),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == []
    assert ctx["people_away"] == ["Alice", "Bob"]
    assert ctx["anyone_home"] is False


# ---------------------------------------------------------------------------
# Presence: partial (some home, some away)
# ---------------------------------------------------------------------------


def test_partial_presence_splits_correctly() -> None:
    """When only some people are home, each list contains only the right names."""
    states = [
        _FakeState(
            "person.alice",
            "person",
            "home",
            attributes={"friendly_name": "Alice"},
        ),
        _FakeState(
            "person.bob",
            "person",
            "not_home",
            attributes={"friendly_name": "Bob"},
        ),
        _FakeState(
            "person.carol",
            "person",
            "work",
            attributes={"friendly_name": "Carol"},
        ),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == ["Alice"]
    assert ctx["people_away"] == ["Bob", "Carol"]
    assert ctx["anyone_home"] is True


# ---------------------------------------------------------------------------
# anyone_home consistency
# ---------------------------------------------------------------------------


def test_anyone_home_true_iff_people_home_nonempty() -> None:
    """anyone_home must be True exactly when people_home is non-empty."""
    # All home case
    all_home = [
        _FakeState(
            "person.alice",
            "person",
            "home",
            attributes={"friendly_name": "Alice"},
        ),
    ]
    ctx = _derive(all_home)
    assert ctx["anyone_home"] is (len(ctx["people_home"]) > 0)

    # None home case
    none_home = [
        _FakeState(
            "person.alice",
            "person",
            "not_home",
            attributes={"friendly_name": "Alice"},
        ),
    ]
    ctx = _derive(none_home)
    assert ctx["anyone_home"] is (len(ctx["people_home"]) > 0)

    # Empty (no person entities)
    ctx = _derive([])
    assert ctx["anyone_home"] is False
    assert ctx["people_home"] == []
    assert ctx["people_away"] == []


# ---------------------------------------------------------------------------
# No person entities → empty lists
# ---------------------------------------------------------------------------


def test_no_person_entities_gives_empty_lists() -> None:
    """When there are no person entities, both lists default to empty."""
    states = [
        _FakeState("light.living_room", "light", "on", attributes={}),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == []
    assert ctx["people_away"] == []
    assert ctx["anyone_home"] is False


# ---------------------------------------------------------------------------
# Falls back to entity_id when friendly_name is missing
# ---------------------------------------------------------------------------


def test_falls_back_to_entity_id_when_no_friendly_name() -> None:
    """If friendly_name attribute is absent, entity_id is used as the name."""
    states = [
        _FakeState("person.alice", "person", "home", attributes={}),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == ["person.alice"]
    assert ctx["anyone_home"] is True


# ---------------------------------------------------------------------------
# Lists are sorted
# ---------------------------------------------------------------------------


def test_people_lists_are_sorted() -> None:
    """Both people_home and people_away must be sorted alphabetically."""
    states = [
        _FakeState(
            "person.zara",
            "person",
            "home",
            attributes={"friendly_name": "Zara"},
        ),
        _FakeState(
            "person.alice",
            "person",
            "home",
            attributes={"friendly_name": "Alice"},
        ),
        _FakeState(
            "person.mike",
            "person",
            "not_home",
            attributes={"friendly_name": "Mike"},
        ),
        _FakeState(
            "person.ben",
            "person",
            "not_home",
            attributes={"friendly_name": "Ben"},
        ),
    ]
    ctx = _derive(states)

    assert ctx["people_home"] == ["Alice", "Zara"]
    assert ctx["people_away"] == ["Ben", "Mike"]
