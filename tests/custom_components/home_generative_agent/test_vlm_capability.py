"""Tests for VLM capability routing in camera image analysis."""

from __future__ import annotations

from custom_components.home_generative_agent.agent.tools import build_vlm_user_text
from custom_components.home_generative_agent.const import (
    VLM_ADVANCED_OBJECTS_TASK_TEMPLATE,
    VLM_CAPABILITY_ADVANCED,
    VLM_CAPABILITY_BASIC,
    VLM_CAPABILITY_STANDARD,
    VLM_STANDARD_LENGTH_CONSTRAINT,
    VLM_USER_KW_TEMPLATE,
    VLM_USER_PROMPT,
)


def test_basic_ignores_prompt_semantics_by_using_keywords_or_default() -> None:
    """Basic uses keyword template or default; prompt text is not in output."""
    kw = VLM_USER_KW_TEMPLATE.format(key_words="a or b")
    assert (
        build_vlm_user_text(
            vlm_capability=VLM_CAPABILITY_BASIC,
            detection_keywords=["a", "b"],
            analysis_prompt="Read the meter",
        )
        == kw
    )
    assert (
        build_vlm_user_text(
            vlm_capability=VLM_CAPABILITY_BASIC,
            detection_keywords=None,
            analysis_prompt="ignored",
        )
        == VLM_USER_PROMPT
    )


def test_standard_only_keywords_no_constraint() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_STANDARD,
        detection_keywords=["box"],
        analysis_prompt=None,
    )
    assert "CRITICAL" not in text
    assert "box" in text


def test_standard_prompt_gets_length_constraint() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_STANDARD,
        detection_keywords=None,
        analysis_prompt="Read the LCD",
    )
    assert VLM_STANDARD_LENGTH_CONSTRAINT.strip() in text
    assert text.startswith("Read the LCD")


def test_standard_both_combined_with_constraint() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_STANDARD,
        detection_keywords=["multimeter"],
        analysis_prompt="Read the display",
    )
    assert "Focus on these elements: multimeter" in text
    assert "Instruction: Read the display" in text
    assert VLM_STANDARD_LENGTH_CONSTRAINT.strip() in text


def test_advanced_exact_prompt_only() -> None:
    assert (
        build_vlm_user_text(
            vlm_capability=VLM_CAPABILITY_ADVANCED,
            detection_keywords=None,
            analysis_prompt="TASK ONLY",
        )
        == "TASK ONLY"
    )


def test_advanced_objects_task_template() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_ADVANCED,
        detection_keywords=["multimeter", "probe"],
        analysis_prompt="Read digits",
    )
    assert text == VLM_ADVANCED_OBJECTS_TASK_TEMPLATE.format(
        objects="multimeter or probe",
        task="Read digits",
    )


def test_advanced_keywords_only_fallback() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_ADVANCED,
        detection_keywords=["dog"],
        analysis_prompt=None,
    )
    assert text == VLM_USER_KW_TEMPLATE.format(key_words="dog")


def test_unknown_capability_falls_back_to_advanced_behavior() -> None:
    """Invalid capability value behaves like Advanced (recommended default)."""
    text = build_vlm_user_text(
        vlm_capability="not_a_real_level",
        detection_keywords=["x"],
        analysis_prompt="Do it",
    )
    assert text == VLM_ADVANCED_OBJECTS_TASK_TEMPLATE.format(objects="x", task="Do it")


def test_empty_analysis_prompt_treated_as_absent() -> None:
    text = build_vlm_user_text(
        vlm_capability=VLM_CAPABILITY_STANDARD,
        detection_keywords=["k"],
        analysis_prompt="   ",
    )
    assert "CRITICAL" not in text
    assert "k" in text
