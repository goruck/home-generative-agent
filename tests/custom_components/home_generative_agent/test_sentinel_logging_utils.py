# ruff: noqa: S101
"""Tests for Sentinel logging helpers."""

from __future__ import annotations

import logging

from custom_components.home_generative_agent.sentinel.logging_utils import (
    RepeatingLogLimiter,
)


def test_repeating_log_limiter_warns_first_then_every_n(caplog) -> None:
    logger = logging.getLogger("tests.hga.sentinel.logging_limiter")
    limiter = RepeatingLogLimiter(logger, every=3)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        for _ in range(5):
            limiter.warning("snapshot", "Snapshot failed.")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Snapshot failed.",
        "Snapshot failed (occurrence=3).",
    ]


def test_repeating_log_limiter_logs_recovery_once(caplog) -> None:
    logger = logging.getLogger("tests.hga.sentinel.logging_recovery")
    limiter = RepeatingLogLimiter(logger, every=3, recovery_level=logging.INFO)

    with caplog.at_level(logging.INFO, logger=logger.name):
        limiter.warning("llm", "LLM failed.")
        limiter.warning("llm", "LLM failed.")
        limiter.recovered("llm", "LLM recovered after %d failed attempt(s).")
        limiter.recovered("llm", "LLM recovered after %d failed attempt(s).")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "LLM failed.",
        "LLM recovered after 2 failed attempt(s).",
    ]
