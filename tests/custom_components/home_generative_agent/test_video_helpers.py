# ruff: noqa: S101
"""Tests for core/video_helpers.py text helpers."""

from __future__ import annotations

import pytest

from custom_components.home_generative_agent.const import RESERVED_IDENTITY_LABELS
from custom_components.home_generative_agent.core.video_helpers import (
    _PLACEHOLDER_DROP_ORDER,
    dedupe_desc,
    dedupe_desc_tagged,
    is_no_change_reply,
    limit_sentences_and_chars,
)

# ---------------------------------------------------------------------------
# Override autouse fixtures from pytest-homeassistant-custom-component:
# these are pure synchronous helpers; the HA event-loop/cleanup fixtures
# only add per-case overhead across the parametrized matrix.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    """No-op override: pure sync tests don't need HA's debug-mode hook."""


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    """No-op override: no HA resources to clean up."""


# ---------------------------------------------------------------------------
# limit_sentences_and_chars tests
# ---------------------------------------------------------------------------

_LONG_S1 = (
    "A paved patio area leads up to a closed white picket gate that overlooks "
    "a street where two vehicles are parked in the distance."
)  # 124 chars
_LONG_S2 = (
    "To the right of the gate, a dog is sitting near a planter and looking "
    "toward the camera."
)  # 90 chars
_TWO_SENTENCE_TEXT = f"{_LONG_S1} {_LONG_S2}"


def test_first_sentence_fits_when_combined_exceeds_limit() -> None:
    """When two sentences together exceed max_chars, only the first is returned."""
    result = limit_sentences_and_chars(_TWO_SENTENCE_TEXT, max_chars=150)
    assert result == _LONG_S1
    assert len(result) <= 150


def test_both_sentences_returned_when_both_fit() -> None:
    short = "Dog seen. Gate closed."
    result = limit_sentences_and_chars(short, max_chars=150)
    assert result == short


def test_single_long_sentence_truncated_at_word_boundary() -> None:
    single = "a " * 100  # 200 chars, no sentence ender
    result = limit_sentences_and_chars(single.strip(), max_chars=50)
    assert len(result) <= 50
    # Should end at a word boundary, not mid-word
    assert not result.endswith("a ")
    assert " " not in result[-1:]  # last char isn't a space


def test_short_text_returned_unchanged() -> None:
    text = "Motion detected."
    result = limit_sentences_and_chars(text, max_chars=150)
    assert result == text


def test_no_mid_word_truncation() -> None:
    """Regression: truncation must not cut mid-word (the original bug)."""
    # Build text where hard truncation at 150 would cut mid-word
    result = limit_sentences_and_chars(_TWO_SENTENCE_TEXT, max_chars=150)
    # Result must be a complete word — no trailing partial word
    words = result.split()
    assert all(w == words[-1] or w for w in words)
    assert len(result) <= 150


# ---------------------------------------------------------------------------
# is_no_change_reply tests (issue #493 repeated-scene sentinel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Scene unchanged.",
        "scene unchanged",
        "Scene unchanged",
        "  Scene unchanged.  ",
        "The scene is unchanged.",
        "The scene remains unchanged since the previous frame.",
        "The view appears identical.",
        "No change.",
        "No changes since the last frame.",
        "Nothing has changed.",
        "Nothing changed from the previous image.",
        "Same scene as before.",
        "The setting is the same.",
        "Unchanged.",
        # Models sometimes combine sentinel-ish sentences.
        "Scene unchanged. Still no activity.",
        "The scene is unchanged. No new activity.",
    ],
)
def test_no_change_reply_matches_sentinel_variants(text: str) -> None:
    assert is_no_change_reply(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "An empty driveway with a parked car and a small tree nearby.",
        "A man in a gray shirt stands on a porch with white railing.",
        "A dog sits by the gate of a fenced yard.",
        # Mentions "unchanged" but reports a change — must stay a description.
        "The scene is unchanged except a car has arrived.",
        "The driveway is unchanged.",  # unknown subject noun: be conservative
        "Same porch, but the chair has moved.",
        "No activity, though a package now sits by the door.",
        # "still" is ambiguous (motionless vs. truncated "still <adjective>").
        "The view is still.",
        # Stillness-only phrases say nothing about equality with the previous
        # frame — a delivered package is "no activity" yet a changed scene.
        "Still no activity.",
        "No activity.",
        "No new activity detected.",
        "No motion detected.",
        "The scene is static.",
        # Short mixed reply: exercises the per-sentence branch, not the
        # length gate.
        "Scene unchanged. A person walks by.",
        # Overlong all-sentinel reply: exercises the length gate.
        "Scene unchanged. " * 10,
        # The VLM failure caption must never classify as a sentinel.
        "Error analyzing image with VLM model.",
        # Sentinel-like text buried in a long reply should not match.
        (
            "The scene is unchanged. A person walks down the steps of the house. "
            "The porch light is on and the street beyond is empty of vehicles."
        ),
    ],
)
def test_no_change_reply_rejects_real_descriptions(text: str) -> None:
    assert not is_no_change_reply(text)


# ---------------------------------------------------------------------------
# dedupe_desc tests
# ---------------------------------------------------------------------------


def test_dedupe_desc_merges_identical_texts_across_timestamps() -> None:
    """Regression: the t+<n>s. prefix must not defeat deduplication."""
    descs = [
        {"t+0s. An empty driveway with a parked car.": ["None"]},
        {"t+8s. An empty driveway with a parked car.": ["None"]},
        {"t+16s. An empty driveway with a parked car.": ["None"]},
    ]
    out = dedupe_desc(descs)
    assert len(out) == 1
    assert next(iter(out[0])) == "t+0s. An empty driveway with a parked car."


def test_dedupe_desc_keeps_distinct_texts() -> None:
    descs = [
        {"t+0s. An empty driveway.": ["None"]},
        {"t+8s. A person walks up the driveway.": ["None"]},
        {"t+16s. An empty driveway.": ["None"]},  # not consecutive: kept
    ]
    out = dedupe_desc(descs)
    assert len(out) == 3


def test_dedupe_desc_merges_faces_from_dropped_duplicates() -> None:
    """A face recognized only on a dropped duplicate frame must survive."""
    descs = [
        {"t+0s. A person stands at the door.": ["None"]},
        {"t+8s. A person stands at the door.": ["Alice"]},
        {"t+16s. A person stands at the door.": ["Alice"]},
    ]
    out = dedupe_desc(descs)
    assert len(out) == 1
    # "Alice" seen on two dropped frames must not be double-appended, and the
    # "None" placeholder is dropped rather than inflating the kept entry to a
    # two-face frame (every contributing frame held exactly one identity).
    assert next(iter(out[0].values())) == ["Alice"]


def test_dedupe_desc_empty_input() -> None:
    assert dedupe_desc([]) == []


def test_dedupe_desc_does_not_mutate_input() -> None:
    """Merging identities must not alias into the caller's face lists."""
    faces = ["None"]
    descs = [
        {"t+0s. A person stands at the door.": faces},
        {"t+8s. A person stands at the door.": ["Alice"]},
    ]
    dedupe_desc(descs)
    assert faces == ["None"]


def test_dedupe_desc_tagged_keeps_first_tag_of_duplicate_run() -> None:
    """Tags must stay aligned with kept texts; a run keeps its first tag."""
    descs = [
        {"t+0s. A person stands at the door.": ["None"]},
        {"t+8s. A person stands at the door.": ["Alice"]},
        {"t+16s. An empty driveway.": ["None"]},
    ]
    out, tags = dedupe_desc_tagged(descs, ["p0", "p1", "p2"])
    assert len(out) == len(tags) == 2
    assert tags == ["p0", "p2"]
    # Identity from the dropped duplicate still merged into the kept entry.
    assert next(iter(out[0].values())) == ["Alice"]


def test_dedupe_desc_tagged_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="argument"):
        dedupe_desc_tagged([{"t+0s. A.": ["None"]}], [])


def _fake_person_check(faces: list[str]) -> bool:
    return any(p and p not in ("Indeterminate", "None") for p in faces)


def test_dedupe_desc_tagged_upgrades_tag_to_recognized_face_frame() -> None:
    """A merged run's tag must follow the frame whose own faces saw the person."""
    descs = [
        {"t+0s. A person stands at the door.": ["Indeterminate"]},
        {"t+8s. A person stands at the door.": ["Alice"]},
        {"t+16s. A person stands at the door.": ["Bob"]},
    ]
    out, tags = dedupe_desc_tagged(
        descs, ["p0", "p1", "p2"], tag_person_check=_fake_person_check
    )
    assert len(out) == 1
    # First person-bearing frame in the run wins; later ones don't re-upgrade.
    assert tags == ["p1"]
    # Both real names survive — erasing a present person is worse than naming
    # one imprecisely — but the "Indeterminate" placeholder is trimmed.
    assert next(iter(out[0].values())) == ["Alice", "Bob"]


def test_dedupe_desc_tagged_keeps_first_tag_when_run_starts_with_person() -> None:
    descs = [
        {"t+0s. A person stands at the door.": ["Alice"]},
        {"t+8s. A person stands at the door.": ["Indeterminate"]},
    ]
    _out, tags = dedupe_desc_tagged(
        descs, ["p0", "p1"], tag_person_check=_fake_person_check
    )
    assert tags == ["p0"]


def test_dedupe_desc_does_not_fabricate_a_two_person_frame() -> None:
    """
    A name plus a flapping "Unknown Person" must collapse to the name.

    Field regression (2026-08-16, camera.playroomdoor): one human whose face
    recognized as "Nico" on one frame and "Unknown Person" on the next
    (identical caption) produced a kept entry carrying both. The summarizer
    reads two identities on one frame as two humans, stands its single-actor
    bias down, and narrates a phantom: "A person walks across a paved
    walkway, then Nico stands near a green bush outside the house."
    """
    descs = [
        {"t+0s. A man in a dark shirt stands on a paved walkway.": ["Nico"]},
        {"t+3s. A man in a dark shirt stands on a paved walkway.": ["Unknown Person"]},
    ]
    out = dedupe_desc(descs)
    assert len(out) == 1
    assert next(iter(out[0].values())) == ["Nico"]


def test_dedupe_desc_preserves_a_genuine_two_face_frame() -> None:
    """A frame that really held two faces keeps both through a merge."""
    descs = [
        {"t+0s. Two people stand at the door.": ["Alice", "Unknown Person"]},
        {"t+8s. Two people stand at the door.": ["Alice"]},
    ]
    out = dedupe_desc(descs)
    assert len(out) == 1
    assert next(iter(out[0].values())) == ["Alice", "Unknown Person"]


def test_dedupe_desc_never_drops_a_second_real_name() -> None:
    """Two enrolled names across a run are ambiguous — keep both."""
    descs = [
        {"t+0s. A person stands at the door.": ["Alice"]},
        {"t+8s. A person stands at the door.": ["Bob"]},
    ]
    out = dedupe_desc(descs)
    assert next(iter(out[0].values())) == ["Alice", "Bob"]


def test_dedupe_desc_drops_indeterminate_before_unknown_person() -> None:
    """A seen-but-unrecognized face outranks a no-face-found placeholder."""
    descs = [
        {"t+0s. A person stands at the door.": ["Indeterminate"]},
        {"t+8s. A person stands at the door.": ["Unknown Person"]},
    ]
    out = dedupe_desc(descs)
    assert next(iter(out[0].values())) == ["Unknown Person"]


def test_placeholder_drop_order_matches_the_reserved_labels() -> None:
    """
    The drop order must cover exactly the reserved identity labels.

    A label reserved by the gallery but missing from the drop order would be
    treated as a real name and never trimmed, letting the phantom through.
    A drop-order entry that is NOT reserved could erase a real person.
    """
    assert set(_PLACEHOLDER_DROP_ORDER) == RESERVED_IDENTITY_LABELS
    # Normalized entries only — matching lowercases and strips.
    assert all(p == p.strip().lower() for p in _PLACEHOLDER_DROP_ORDER)


def test_dedupe_desc_drops_legacy_spelled_placeholders() -> None:
    """Legacy gallery casing must not read as a real name."""
    descs = [
        {"t+0s. A man stands at the door.": ["Nico"]},
        {"t+8s. A man stands at the door.": [" Unknown person "]},
    ]
    out = dedupe_desc(descs)
    assert next(iter(out[0].values())) == ["Nico"]


def test_dedupe_desc_drops_the_empty_identity_first() -> None:
    """An empty identity string is the weakest evidence in the drop order."""
    descs = [
        {"t+0s. A person stands at the door.": [""]},
        {"t+8s. A person stands at the door.": ["Indeterminate"]},
    ]
    out = dedupe_desc(descs)
    assert next(iter(out[0].values())) == ["Indeterminate"]


def test_dedupe_desc_face_cap_uses_the_run_maximum() -> None:
    """The cap is the largest per-frame face count anywhere in the run."""
    descs = [
        {"t+0s. People at the door.": ["Indeterminate"]},
        {"t+8s. People at the door.": ["Nico", "Unknown Person"]},
        {"t+16s. People at the door.": ["Indeterminate"]},
    ]
    out = dedupe_desc(descs)
    assert next(iter(out[0].values())) == ["Nico", "Unknown Person"]


def test_dedupe_desc_face_cap_resets_between_runs() -> None:
    """A two-face run must not license a two-face merge in the next run."""
    descs = [
        {"t+0s. Two people at the door.": ["Alice", "Unknown Person"]},
        {"t+8s. A person on the porch.": ["Nico"]},
        {"t+16s. A person on the porch.": ["Unknown Person"]},
    ]
    out = dedupe_desc(descs)
    assert len(out) == 2
    assert next(iter(out[0].values())) == ["Alice", "Unknown Person"]
    assert next(iter(out[1].values())) == ["Nico"]
