"""Unit tests for core/utils.py — openai_compatible validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from custom_components.home_generative_agent.core.utils import (
    CannotConnectError,
    InvalidAuthError,
    extract_final,
    extract_redacted_thinking,
    openai_compatible_healthy,
    validate_openai_compatible_url,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# ---------------------------------------------------------------------------
# extract_final tests
# ---------------------------------------------------------------------------


def test_extract_final_strips_think_block() -> None:
    assert extract_final("<think>reasoning</think>answer") == "answer"


def test_extract_redacted_thinking_short_tags() -> None:
    raw = "<think>reasoning</think>answer"
    assert extract_redacted_thinking(raw) == "reasoning"


def test_extract_redacted_thinking_long_tags() -> None:
    raw = "<redacted_thinking>alpha</redacted_thinking>x"
    assert extract_redacted_thinking(raw) == "alpha"


def test_extract_redacted_thinking_unclosed() -> None:
    raw = "<think>tail without close"
    assert extract_redacted_thinking(raw) == "tail without close"


def test_extract_final_no_max_chars_returns_full() -> None:
    text = "word " * 50
    assert extract_final(text.strip()) == text.strip()


def test_extract_final_max_chars_fits_exactly() -> None:
    assert extract_final("hello world", max_chars=11) == "hello world"


def test_extract_final_max_chars_truncates_at_word_boundary() -> None:
    # 20 chars would cut mid-word in "boundary"
    result = extract_final("truncate at word boundary here", max_chars=20)
    assert result == "truncate at word"
    assert len(result) <= 20


def test_extract_final_max_chars_no_space_falls_back_to_hard_cut() -> None:
    result = extract_final("superlongwordwithoutspaces", max_chars=10)
    assert len(result) <= 10


# ---------------------------------------------------------------------------
# Fake HTTP helpers
# ---------------------------------------------------------------------------

HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_SERVER_ERROR = 503


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    """Async HTTP client that records calls and returns a canned response."""

    def __init__(
        self,
        *,
        status_code: int = HTTP_OK,
        exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.exc = exc
        self.last_url: str | None = None
        self.last_headers: dict[str, str] = {}

    async def get(
        self, url: str, headers: dict[str, str] | None = None, **_: Any
    ) -> _FakeResponse:
        self.last_url = url
        self.last_headers = dict(headers or {})
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.status_code)


# ---------------------------------------------------------------------------
# validate_openai_compatible_url tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_empty_raises(hass: HomeAssistant) -> None:
    """Empty base_url immediately raises CannotConnectError without any network call."""
    with pytest.raises(CannotConnectError):
        await validate_openai_compatible_url(hass, "")


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_success_no_key(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 response with no api_key succeeds and omits Authorization header."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    await validate_openai_compatible_url(hass, "http://localhost:8000")

    assert client.last_url == "http://localhost:8000/v1/models"
    assert "Authorization" not in client.last_headers


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_sends_bearer_token(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When api_key is provided the Authorization header is included."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    await validate_openai_compatible_url(hass, "http://localhost:8000/", "sk-test")

    assert client.last_headers.get("Authorization") == "Bearer sk-test"


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_none_key_omits_header(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """api_key='none' (the sentinel default) must not send an Authorization header."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    await validate_openai_compatible_url(hass, "http://localhost:8000", "none")

    assert "Authorization" not in client.last_headers


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_network_error(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """httpx.RequestError is re-raised as CannotConnectError."""
    request = httpx.Request("GET", "http://localhost:8000/v1/models")
    client = _FakeClient(exc=httpx.ConnectError("refused", request=request))
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    with pytest.raises(CannotConnectError):
        await validate_openai_compatible_url(hass, "http://localhost:8000")


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_401_raises_invalid_auth(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 401 is mapped to InvalidAuthError."""
    client = _FakeClient(status_code=HTTP_UNAUTHORIZED)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    with pytest.raises(InvalidAuthError):
        await validate_openai_compatible_url(hass, "http://localhost:8000", "bad-key")


@pytest.mark.asyncio
async def test_validate_openai_compatible_url_5xx_raises_cannot_connect(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 5xx is mapped to CannotConnectError."""
    client = _FakeClient(status_code=HTTP_SERVER_ERROR)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    with pytest.raises(CannotConnectError):
        await validate_openai_compatible_url(hass, "http://localhost:8000")


# ---------------------------------------------------------------------------
# openai_compatible_healthy tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_compatible_healthy_no_url_returns_false(
    hass: HomeAssistant,
) -> None:
    """Missing base_url returns False immediately without any network call."""
    result = await openai_compatible_healthy(hass, None)
    assert result is False


@pytest.mark.asyncio
async def test_openai_compatible_healthy_empty_url_returns_false(
    hass: HomeAssistant,
) -> None:
    """Empty string base_url returns False immediately."""
    result = await openai_compatible_healthy(hass, "")
    assert result is False


@pytest.mark.asyncio
async def test_openai_compatible_healthy_returns_true_on_success(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachable endpoint returns True."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    result = await openai_compatible_healthy(hass, "http://localhost:8000")
    assert result is True


@pytest.mark.asyncio
async def test_openai_compatible_healthy_returns_false_on_connect_error(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network error returns False instead of propagating the exception."""
    request = httpx.Request("GET", "http://localhost:8000/v1/models")
    client = _FakeClient(exc=httpx.ConnectError("refused", request=request))
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    result = await openai_compatible_healthy(hass, "http://localhost:8000")
    assert result is False


@pytest.mark.asyncio
async def test_openai_compatible_healthy_returns_false_on_auth_error(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 401 (InvalidAuthError) returns False instead of propagating."""
    client = _FakeClient(status_code=HTTP_UNAUTHORIZED)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    result = await openai_compatible_healthy(hass, "http://localhost:8000", "bad-key")
    assert result is False
