# ruff: noqa: S101
"""
Pin the library behaviour the ``None`` num_predict constants rely on.

Issue #614: ``CHAT_MODEL_MAX_TOKENS``, ``VLM_NUM_PREDICT`` and
``SUMMARIZATION_MODEL_PREDICT`` are ``None`` so that no ``num_predict`` reaches
the Ollama server at all (a ``-2`` sentinel is rejected by Ollama Cloud). That
only holds while ``langchain-ollama`` drops ``None`` options before the request
is built; a JSON ``null`` would be rejected by every server, local ones included.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from custom_components.home_generative_agent.const import (
    CHAT_MODEL_MAX_TOKENS,
    SUMMARIZATION_MODEL_PREDICT,
    VLM_NUM_PREDICT,
)


def test_num_predict_constants_are_unset() -> None:
    assert CHAT_MODEL_MAX_TOKENS is None
    assert VLM_NUM_PREDICT is None
    assert SUMMARIZATION_MODEL_PREDICT is None


def test_chat_ollama_omits_none_num_predict_from_request_options() -> None:
    """A ``None`` num_predict must be absent from the wire options, not null."""
    model = ChatOllama(
        model="qwen3:0.6b",
        base_url="http://127.0.0.1:1",
        num_predict=CHAT_MODEL_MAX_TOKENS,
        num_ctx=1024,
    )
    params = model._chat_params([HumanMessage("hi")])
    options = params["options"]
    assert "num_predict" not in options
    assert options["num_ctx"] == 1024
    assert None not in options.values()


def test_chat_ollama_keeps_explicit_num_predict() -> None:
    """The same path must still forward a real cap (the video constants)."""
    model = ChatOllama(
        model="qwen3:0.6b", base_url="http://127.0.0.1:1", num_predict=256
    )
    assert model._chat_params([HumanMessage("hi")])["options"]["num_predict"] == 256
