# ruff: noqa: S101
"""Tests for canonical derived.* evidence-path handling (issue #524)."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.sentinel.evidence_paths import (
    canonical_evidence_paths,
    canonicalize_evidence_path,
    is_derived_path,
    night_signal,
    presence_signal,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Canonical spellings pass through unchanged.
        ("derived.anyone_home", "derived.anyone_home"),
        ("not derived.anyone_home", "not derived.anyone_home"),
        ("derived.is_night", "derived.is_night"),
        (
            "entities[entity_id=lock.front_door].state",
            "entities[entity_id=lock.front_door].state",
        ),
        # Case, whitespace, and prefix variants.
        ("NOT derived.anyone_home", "not derived.anyone_home"),
        (" not  derived.anyone_home ", "not derived.anyone_home"),
        ("!derived.anyone_home", "not derived.anyone_home"),
        ("! derived.anyone_home", "not derived.anyone_home"),
        ("Derived.Anyone_Home", "derived.anyone_home"),
        # Trailing boolean comparisons fold into the negation.
        ("derived.anyone_home == false", "not derived.anyone_home"),
        ("derived.anyone_home=0", "not derived.anyone_home"),
        ("derived.anyone_home == 'false'", "not derived.anyone_home"),
        ("derived.anyone_home == true", "derived.anyone_home"),
        ("derived.anyone_home = 1", "derived.anyone_home"),
        # Double negation resolves positive.
        ("not derived.anyone_home == false", "derived.anyone_home"),
        ("!derived.anyone_home == false", "derived.anyone_home"),
    ],
)
def test_canonicalize_evidence_path(path: str, expected: str) -> None:
    assert canonicalize_evidence_path(path) == expected


def test_canonical_evidence_paths_tolerates_malformed_shapes() -> None:
    assert canonical_evidence_paths(None) == frozenset()
    assert canonical_evidence_paths("derived.is_night") == frozenset()
    assert canonical_evidence_paths({"derived": True}) == frozenset()
    assert canonical_evidence_paths([1, None, "derived.is_night"]) == frozenset(
        {"derived.is_night"}
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("derived.is_night", True),
        ("derived.anyone_home", True),
        ("not derived.anyone_home", True),
        ("NOT derived.anyone_home", True),
        ("derived.anyone_home == false", True),
        ("entities[entity_id=lock.front_door].state", False),
        ("camera_activity[entity_id=camera.porch]", False),
    ],
)
def test_is_derived_path(path: str, expected: bool) -> None:  # noqa: FBT001
    assert is_derived_path(path) is expected


def test_presence_signal_negated_path_beats_home_prose() -> None:
    assert (
        presence_signal(["not derived.anyone_home"], "someone is home occupied")
        == "away"
    )


def test_presence_signal_boolean_expression_beats_positive_path() -> None:
    assert (
        presence_signal(["derived.anyone_home"], "when anyone_home == false") == "away"
    )


def test_presence_signal_away_prose_beats_bare_positive_path() -> None:
    # Load-bearing ordering: citing derived.anyone_home does not assert it
    # is true — the LLM historically cites the positive path while the
    # prose says "while nobody is home". The bare path must stay below the
    # away term tier or such legacy English candidates would invert to home
    # rules (issue #524).
    assert (
        presence_signal(["derived.anyone_home"], "motion while nobody is home")
        == "away"
    )


def test_presence_signal_home_prose_resolves_home() -> None:
    assert presence_signal([], "window open while someone home") == "home"


def test_presence_signal_bare_positive_path_resolves_home() -> None:
    assert presence_signal(["derived.anyone_home"], "") == "home"


def test_presence_signal_positive_path_with_non_english_prose() -> None:
    # The #524 scenario: translated prose carries no English direction
    # words, so the structured path is the only occupancy signal.
    assert (
        presence_signal(["derived.anyone_home"], "okno otevřené, když je někdo doma")
        == "home"
    )


def test_presence_signal_negation_variant_with_non_english_prose() -> None:
    assert (
        presence_signal(["NOT derived.anyone_home"], "pohyb, když nikdo není doma")
        == "away"
    )


def test_presence_signal_no_signals_returns_any() -> None:
    assert presence_signal([], "window open") == "any"
    assert presence_signal(None, "") == "any"


def test_night_signal_structured_and_text() -> None:
    assert night_signal(["derived.is_night"], "") is True
    assert night_signal(["Derived.Is_Night"], "") is True
    assert night_signal([], "open overnight") is True
    assert night_signal([], "v noci") is False
    assert night_signal(["not derived.is_night"], "") is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Quote-wrapped whole paths — the discovery prompt itself renders
        # the negated form inside single quotes, so wrapped spellings are an
        # expected LLM output mode (issue #524 red-team).
        ("'not derived.anyone_home'", "not derived.anyone_home"),
        ('"derived.anyone_home"', "derived.anyone_home"),
        ("`derived.is_night`", "derived.is_night"),
        # HA state idiom booleans alongside JSON booleans.
        ("derived.anyone_home == off", "not derived.anyone_home"),
        ("derived.anyone_home = no", "not derived.anyone_home"),
        ("derived.anyone_home == on", "derived.anyone_home"),
        ("derived.anyone_home = yes", "derived.anyone_home"),
        # U+FEFF is whitespace to JS but not Python — the canonicalizer
        # normalizes the union so the card mirror agrees (issue #524
        # red-team parity vector).
        ("﻿not derived.anyone_home", "not derived.anyone_home"),
    ],
)
def test_canonicalize_evidence_path_hardening(path: str, expected: str) -> None:
    assert canonicalize_evidence_path(path) == expected


def test_presence_signal_quoted_negated_path_resolves_away() -> None:
    assert (
        presence_signal(["'not derived.anyone_home'"], "pohyb, když nikdo není doma")
        == "away"
    )


def test_presence_signal_expression_beats_prose_terms_both_directions() -> None:
    # Machine syntax outranks prose terms in both directions — a reorder of
    # the expression and term tiers in either mirror would pass the rest of
    # the suite (issue #524 testing review).
    assert presence_signal([], "someone home but anyone_home == false") == "away"
    assert presence_signal([], "anyone_home == true even when nobody is home") == "home"


def test_presence_signal_negated_path_in_text_resolves_away() -> None:
    # The negated path spelled inside the pattern text is machine syntax,
    # same tier as the anyone_home expressions — and it beats the bare
    # positive evidence path.
    assert presence_signal([], "is_night and not derived.anyone_home") == "away"
    assert (
        presence_signal(["derived.anyone_home"], "!derived.anyone_home and motion")
        == "away"
    )


def test_presence_signal_requires_pre_lowercased_text() -> None:
    # Callers lowercase the text blob before calling; mixed-case prose
    # intentionally does not match the case-sensitive term tiers (the
    # historical patterns), while the expression tiers are IGNORECASE.
    assert presence_signal([], "While Nobody Is Home") == "any"
    assert presence_signal([], "ANYONE_HOME == FALSE") == "away"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Inner-quoted variants keep a wrapping quote after prefix/suffix
        # removal unless quotes are re-stripped (adversarial review).
        ("not 'derived.anyone_home'", "not derived.anyone_home"),
        ("'derived.anyone_home' == false", "not derived.anyone_home"),
        # Bare derived-key spellings alias to the canonical paths.
        ("anyone_home", "derived.anyone_home"),
        ("anyone_home == false", "not derived.anyone_home"),
        ("is_night", "derived.is_night"),
        # The "is" comparison idiom.
        ("derived.anyone_home is false", "not derived.anyone_home"),
        ("derived.is_night is true", "derived.is_night"),
        # Stacked negation prefixes fold by parity.
        ("!!derived.anyone_home", "derived.anyone_home"),
        ("not not derived.anyone_home", "derived.anyone_home"),
        ("not !derived.anyone_home", "derived.anyone_home"),
    ],
)
def test_canonicalize_evidence_path_adversarial_variants(
    path: str, expected: str
) -> None:
    assert canonicalize_evidence_path(path) == expected


def test_night_signal_negated_path_blocks_text_fallback() -> None:
    # "derived.is_night == false" canonicalizes to the negated path; the
    # "night" substring in the very text that negates it must not flip the
    # candidate to a night rule (adversarial review).
    assert (
        night_signal(
            ["derived.is_night == false"],
            "motion during daytime when is_night is false",
        )
        is False
    )


def test_presence_signal_expression_as_evidence_path() -> None:
    # A bare "anyone_home == false" path (missing the derived. prefix)
    # aliases to the canonical negated path — previously it matched no tier
    # and resolved "any" (adversarial review).
    assert presence_signal(["anyone_home == false"], "") == "away"
    assert presence_signal(["anyone_home"], "") == "home"
