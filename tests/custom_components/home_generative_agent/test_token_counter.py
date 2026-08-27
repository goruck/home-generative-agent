# ruff: noqa: S101
"""Unit tests for agent/token_counter.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import HumanMessage

from custom_components.home_generative_agent.agent import token_counter
from custom_components.home_generative_agent.agent.token_counter import (
    _count_gemini_tokens,
    _redact_secrets,
    count_tokens_cross_provider,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

FAKE_KEY = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY0000"


@pytest.fixture(autouse=True)
def _clear_remote_count_cooldown() -> Iterator[None]:
    """Keep the fail-open cooldown from leaking between tests."""
    token_counter._remote_count_disabled_until.clear()
    yield
    token_counter._remote_count_disabled_until.clear()


def _make_fake_encoding(token_count: int = 5) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = list(range(token_count))
    return enc


def _make_response(status_code: int, *, text: str = "", json_body: object = None):  # noqa: ANN202
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body
    return resp


def test_count_tokens_cross_provider_anthropic_returns_int() -> None:
    """Anthropic provider uses tiktoken fallback (gpt-4o) and returns a positive int."""
    messages = [HumanMessage(content="Hello, how are you?")]
    with patch(
        "custom_components.home_generative_agent.agent.token_counter._pick_encoding_for_model",
        return_value=_make_fake_encoding(5),
    ):
        result = count_tokens_cross_provider(
            messages,
            model="claude-sonnet-4-5",
            provider="anthropic",
            options={},
            chat_model_options={},
        )
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_cross_provider_anthropic_empty_messages() -> None:
    """Anthropic provider with no messages returns 0."""
    with patch(
        "custom_components.home_generative_agent.agent.token_counter._pick_encoding_for_model",
        return_value=_make_fake_encoding(5),
    ):
        result = count_tokens_cross_provider(
            [],
            model="claude-opus-4-5",
            provider="anthropic",
            options={},
            chat_model_options={},
        )
    assert isinstance(result, int)
    assert result == 0


def test_gemini_count_sends_key_as_header_not_query_param() -> None:
    """The API key must never ride in the URL, where tracebacks would expose it."""
    with patch.object(
        token_counter.requests,
        "post",
        return_value=_make_response(200, json_body={"totalTokens": 42}),
    ) as mock_post:
        result = _count_gemini_tokens(
            [HumanMessage(content="hi")],
            model="gemini-2.5-flash-lite",
            gemini_api_key=FAKE_KEY,
        )

    assert result == 42
    url = mock_post.call_args.args[0]
    assert FAKE_KEY not in url
    assert "key=" not in url
    assert mock_post.call_args.kwargs["headers"] == {"x-goog-api-key": FAKE_KEY}
    assert mock_post.call_args.kwargs.get("params") is None


def test_gemini_count_error_includes_body_and_hides_key() -> None:
    """An HTTP error reports Google's reason without leaking the credential."""
    body = '{"error": {"message": "models/gemini-2.5-flash-lite is not found"}}'
    with (
        patch.object(
            token_counter.requests, "post", return_value=_make_response(404, text=body)
        ),
        pytest.raises(RuntimeError) as excinfo,
    ):
        _count_gemini_tokens(
            [HumanMessage(content="hi")],
            model="gemini-2.5-flash-lite",
            gemini_api_key=FAKE_KEY,
        )

    message = str(excinfo.value)
    assert "is not found" in message
    assert "404" in message
    assert FAKE_KEY not in message


def test_gemini_count_failure_falls_back_to_approximate() -> None:
    """A failing countTokens call degrades the count instead of killing the turn."""
    messages = [HumanMessage(content="Hello, how are you?")]
    with patch.object(
        token_counter.requests, "post", return_value=_make_response(404, text="nope")
    ):
        result = count_tokens_cross_provider(
            messages,
            model="gemini-2.5-flash-lite",
            provider="gemini",
            options={"gemini_api_key": FAKE_KEY},
            chat_model_options={},
        )

    assert isinstance(result, int)
    assert result > 0


def test_gemini_count_failure_suppresses_further_requests() -> None:
    """After a failure the remote counter is skipped for the cooldown window."""
    messages = [HumanMessage(content="Hello, how are you?")]
    with patch.object(
        token_counter.requests,
        "post",
        side_effect=requests.ConnectionError("unreachable"),
    ) as mock_post:
        for _ in range(5):
            count_tokens_cross_provider(
                messages,
                model="gemini-2.5-flash-lite",
                provider="gemini",
                options={"gemini_api_key": FAKE_KEY},
                chat_model_options={},
            )

    assert mock_post.call_count == 1


def test_gemini_missing_key_falls_back_instead_of_raising() -> None:
    """A missing key is reported by the chat call, not by crashing the trimmer."""
    result = count_tokens_cross_provider(
        [HumanMessage(content="Hello, how are you?")],
        model="gemini-2.5-flash-lite",
        provider="gemini",
        options={},
        chat_model_options={},
    )
    assert isinstance(result, int)
    assert result > 0


def test_redact_secrets_scrubs_keys_and_query_params() -> None:
    """Both bare keys and any `?key=`-style parameter are scrubbed."""
    text = (
        f"404 for url: https://example.com/v1beta/models/m:countTokens?key={FAKE_KEY}"
    )
    redacted = _redact_secrets(text)
    assert FAKE_KEY not in redacted
    assert "<redacted>" in redacted
