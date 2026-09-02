# ruff: noqa: S101
"""Unit tests for core/utils.py — openai_compatible validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from custom_components.home_generative_agent.const import (
    ANTHROPIC_THINKING_MAX_BUDGET,
    ANTHROPIC_THINKING_MIN_BUDGET,
    ANTHROPIC_THINKING_RESPONSE_TOKENS,
    GEMINI_3_RECOMMENDED_TEMPERATURE,
)
from custom_components.home_generative_agent.core.utils import (
    CannotConnectError,
    InvalidAuthError,
    anthropic_healthy,
    extract_final,
    gemini_sampling_configurable,
    is_gemini_3_or_later,
    normalize_openai_compatible_base_url,
    openai_compatible_healthy,
    reasoning_field,
    thinking_configurable,
    validate_anthropic_key,
    validate_openai_compatible_url,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# ---------------------------------------------------------------------------
# extract_final tests
# ---------------------------------------------------------------------------


def test_extract_final_strips_think_block() -> None:
    assert extract_final("<think>reasoning</think>answer") == "answer"


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


def test_extract_final_list_of_text_blocks_joins_text() -> None:
    blocks: list[Any] = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    assert extract_final(blocks) == "hello world"


def test_extract_final_list_filters_non_text_entries() -> None:
    blocks: list[Any] = [
        {"type": "tool_use", "id": "abc"},
        {"type": "text", "text": "answer"},
        "bare string",
        42,
    ]
    assert extract_final(blocks) == "answer"


def test_extract_final_empty_list_returns_empty_string() -> None:
    assert extract_final([]) == ""


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
# normalize_openai_compatible_base_url tests
# ---------------------------------------------------------------------------


def test_normalize_base_url_appends_v1() -> None:
    """A bare host:port URL gains the /v1 prefix the OpenAI SDK expects."""
    result = normalize_openai_compatible_base_url("http://localhost:8080")
    assert result == "http://localhost:8080/v1"


def test_normalize_base_url_keeps_existing_v1() -> None:
    """A URL already ending in /v1 is not doubled."""
    result = normalize_openai_compatible_base_url("http://localhost:8080/v1")
    assert result == "http://localhost:8080/v1"


def test_normalize_base_url_strips_trailing_slash() -> None:
    """Trailing slashes are removed before appending /v1."""
    result = normalize_openai_compatible_base_url("http://localhost:8080/")
    assert result == "http://localhost:8080/v1"
    result = normalize_openai_compatible_base_url("http://localhost:8080/v1/")
    assert result == "http://localhost:8080/v1"


def test_normalize_base_url_adds_scheme() -> None:
    """A URL without a scheme gains http://."""
    result = normalize_openai_compatible_base_url("localhost:8080")
    assert result == "http://localhost:8080/v1"


def test_normalize_base_url_preserves_path_prefix() -> None:
    """A reverse-proxy path prefix is preserved with /v1 appended."""
    result = normalize_openai_compatible_base_url("http://myhost/llm")
    assert result == "http://myhost/llm/v1"


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
async def test_validate_openai_compatible_url_accepts_v1_suffix(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base URL entered with /v1 probes /v1/models, not /v1/v1/models."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )

    await validate_openai_compatible_url(hass, "http://localhost:8000/v1")

    assert client.last_url == "http://localhost:8000/v1/models"


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


# ---------------------------------------------------------------------------
# reasoning_field tests
# ---------------------------------------------------------------------------


def test_reasoning_field_unsupported_model_returns_empty() -> None:
    """Non-thinking models always return {} regardless of enabled flag."""
    assert reasoning_field(model="llama3:8b", enabled=True) == {}
    assert reasoning_field(model="llama3:8b", enabled=False) == {}


def test_reasoning_field_qwen3_enabled_returns_true() -> None:
    """Qwen3-family models return {'reasoning': True} when enabled."""
    result = reasoning_field(model="qwen3:32b", enabled=True)
    assert result == {"reasoning": True}


def test_reasoning_field_qwen3_disabled_returns_false() -> None:
    """Qwen3-family models return {'reasoning': False} when disabled."""
    result = reasoning_field(model="qwen3.5:35b", enabled=False)
    assert result == {"reasoning": False}


def test_reasoning_field_gpt_oss_disabled_returns_false() -> None:
    """gpt-oss models return {'reasoning': False} when disabled."""
    result = reasoning_field(model="gpt-oss", enabled=False)
    assert result == {"reasoning": False}


def test_reasoning_field_qwen3_disabled_returns_false_registry_url() -> None:
    """Registry-prefixed model names are stripped before matching."""
    result = reasoning_field(
        model="registry.ollama.ai/library/qwen3.5:35b", enabled=False
    )
    assert result == {"reasoning": False}


# ---------------------------------------------------------------------------
# validate_anthropic_key tests
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"


@pytest.mark.asyncio
async def test_validate_anthropic_key_success(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 200 from Anthropic /v1/models does not raise."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    await validate_anthropic_key(hass, "sk-ant-valid")
    assert client.last_url == ANTHROPIC_MODELS_URL
    assert client.last_headers.get("x-api-key") == "sk-ant-valid"
    assert "anthropic-version" in client.last_headers


@pytest.mark.asyncio
async def test_validate_anthropic_key_401_raises_invalid_auth(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 401 raises InvalidAuthError."""
    client = _FakeClient(status_code=HTTP_UNAUTHORIZED)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    with pytest.raises(InvalidAuthError):
        await validate_anthropic_key(hass, "sk-ant-bad")


@pytest.mark.asyncio
async def test_validate_anthropic_key_network_error_raises_cannot_connect(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network failure raises CannotConnectError."""
    request = httpx.Request("GET", ANTHROPIC_MODELS_URL)
    client = _FakeClient(exc=httpx.ConnectError("refused", request=request))
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    with pytest.raises(CannotConnectError):
        await validate_anthropic_key(hass, "sk-ant-any")


@pytest.mark.asyncio
async def test_validate_anthropic_key_server_error_raises_cannot_connect(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 5xx raises CannotConnectError."""
    client = _FakeClient(status_code=HTTP_SERVER_ERROR)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    with pytest.raises(CannotConnectError):
        await validate_anthropic_key(hass, "sk-ant-any")


@pytest.mark.asyncio
async def test_validate_anthropic_key_empty_key_returns_without_error(
    hass: HomeAssistant,
) -> None:
    """Empty api_key returns immediately without making any network call."""
    await validate_anthropic_key(hass, "")


# ---------------------------------------------------------------------------
# anthropic_healthy tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_healthy_no_key_returns_false(hass: HomeAssistant) -> None:
    """Missing api_key returns False immediately."""
    assert await anthropic_healthy(hass, None) is False


@pytest.mark.asyncio
async def test_anthropic_healthy_returns_true_on_success(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachable Anthropic endpoint returns True."""
    client = _FakeClient(status_code=HTTP_OK)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    assert await anthropic_healthy(hass, "sk-ant-valid") is True


@pytest.mark.asyncio
async def test_anthropic_healthy_returns_false_on_connect_error(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network failure returns False instead of propagating."""
    request = httpx.Request("GET", ANTHROPIC_MODELS_URL)
    client = _FakeClient(exc=httpx.ConnectError("refused", request=request))
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    assert await anthropic_healthy(hass, "sk-ant-any") is False


@pytest.mark.asyncio
async def test_anthropic_healthy_returns_false_on_auth_error(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP 401 returns False instead of propagating."""
    client = _FakeClient(status_code=HTTP_UNAUTHORIZED)
    monkeypatch.setattr(
        "custom_components.home_generative_agent.core.utils.get_async_client",
        lambda _hass: client,
    )
    assert await anthropic_healthy(hass, "sk-ant-bad") is False


# ---------------------------------------------------------------------------
# thinking_configurable tests (issue #580)
# ---------------------------------------------------------------------------


def test_thinking_none_sends_nothing_for_all_providers() -> None:
    """Provider default (None) never sends thinking fields."""
    for provider in ("openai", "openai_compatible", "gemini", "anthropic", "ollama"):
        assert (
            thinking_configurable(provider_type=provider, reasoning=None, budget=512)
            == {}
        )


def test_thinking_unknown_effort_string_sends_nothing_for_cloud_openai() -> None:
    """Cloud OpenAI only forwards known effort levels."""
    assert thinking_configurable(provider_type="openai", reasoning="frobnicate") == {}


def test_thinking_openai_compatible_forwards_custom_effort() -> None:
    """llama.cpp-style servers own the effort vocabulary - pass verbatim."""
    assert thinking_configurable(
        provider_type="openai_compatible", reasoning="xhigh"
    ) == {"reasoning_effort": "xhigh"}
    assert thinking_configurable(
        provider_type="openai_compatible", reasoning="none"
    ) == {"reasoning_effort": "none"}


def test_thinking_ollama_returns_empty() -> None:
    """Ollama thinking flows through reasoning_field, not this helper."""
    assert thinking_configurable(provider_type="ollama", reasoning=True) == {}


def test_thinking_openai_effort_only() -> None:
    """Cloud OpenAI forwards effort levels and ignores On/Off."""
    assert thinking_configurable(provider_type="openai", reasoning="medium") == {
        "reasoning_effort": "medium"
    }
    assert thinking_configurable(provider_type="openai", reasoning=True) == {}
    assert thinking_configurable(provider_type="openai", reasoning=False) == {}


def test_thinking_openai_compatible_off() -> None:
    """Off sends reasoning_effort=none and enable_thinking=False (llama.cpp)."""
    config = thinking_configurable(
        provider_type="openai_compatible", reasoning=False, budget=512
    )
    assert config == {
        "reasoning_effort": "none",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_thinking_openai_compatible_on_with_budget() -> None:
    """On sends enable_thinking=True plus the per-request budget."""
    config = thinking_configurable(
        provider_type="openai_compatible", reasoning=True, budget=512
    )
    assert config == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_budget_tokens": 512,
        }
    }


def test_thinking_openai_compatible_effort() -> None:
    """Effort levels map to reasoning_effort for llama.cpp-style servers."""
    config = thinking_configurable(provider_type="openai_compatible", reasoning="low")
    assert config == {"reasoning_effort": "low"}


def test_thinking_gemini_always_uses_budget_never_level() -> None:
    """
    Gemini sends only thinking_budget on the pinned stack.

    google-ai-generativelanguage 0.10.0's ThinkingConfig protobuf has no
    thinking_level field; setting it crashes langchain-google-genai 3.1.0 at
    proto marshaling ("Unknown field for ThinkingConfig: thinking_level"),
    Gemini 3-family models included. Field-hit during #580 integration
    testing.
    """
    for model in ("gemini-2.5-flash", "gemini-3.5-flash-lite"):
        assert thinking_configurable(
            provider_type="gemini", reasoning=False, model=model
        ) == {"thinking_budget": 0}
        assert thinking_configurable(
            provider_type="gemini", reasoning=True, budget=512, model=model
        ) == {"thinking_budget": 512}
        assert thinking_configurable(
            provider_type="gemini", reasoning=True, model=model
        ) == {"thinking_budget": -1}
        # Effort strings have no budget-API equivalent; known ones mean On.
        assert thinking_configurable(
            provider_type="gemini", reasoning="low", model=model
        ) == {"thinking_budget": -1}
        assert (
            thinking_configurable(
                provider_type="gemini", reasoning="frobnicate", model=model
            )
            == {}
        )
        for config in (
            thinking_configurable(
                provider_type="gemini", reasoning=value, budget=512, model=model
            )
            for value in (False, True, "low", "high")
        ):
            assert "thinking_level" not in config


def test_thinking_anthropic_enforces_api_constraints() -> None:
    """Anthropic: budget floor 1024, max_tokens above budget, temperature 1."""
    config = thinking_configurable(
        provider_type="anthropic", reasoning=True, budget=512
    )
    assert config["thinking"] == {
        "type": "enabled",
        "budget_tokens": ANTHROPIC_THINKING_MIN_BUDGET,
    }
    assert (
        config["max_tokens"]
        == ANTHROPIC_THINKING_MIN_BUDGET + ANTHROPIC_THINKING_RESPONSE_TOKENS
    )
    assert config["temperature"] == 1
    big = thinking_configurable(provider_type="anthropic", reasoning=True, budget=8192)
    assert big["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert thinking_configurable(provider_type="anthropic", reasoning=False) == {}


def test_thinking_anthropic_caps_budget_below_output_limit() -> None:
    """A UI-permitted oversized budget is clamped so max_tokens stays valid."""
    config = thinking_configurable(
        provider_type="anthropic", reasoning=True, budget=131072
    )
    assert config["thinking"] == {
        "type": "enabled",
        "budget_tokens": ANTHROPIC_THINKING_MAX_BUDGET,
    }
    assert (
        config["max_tokens"]
        == ANTHROPIC_THINKING_MAX_BUDGET + ANTHROPIC_THINKING_RESPONSE_TOKENS
    )


def test_thinking_anthropic_ignores_stale_effort_strings() -> None:
    """An effort left over from another provider type must not enable thinking."""
    assert thinking_configurable(provider_type="anthropic", reasoning="medium") == {}


def test_reasoning_field_passes_effort_through_for_gpt_oss() -> None:
    """gpt-oss models get the configured effort verbatim, not the heuristic."""
    assert reasoning_field(model="gpt-oss:20b", enabled="high") == {"reasoning": "high"}
    assert reasoning_field(model="gpt-oss:20b", enabled=True) == {"reasoning": "low"}
    # Boolean-style models treat any truthy value as on.
    assert reasoning_field(model="qwen3:8b", enabled="high") == {"reasoning": True}
    assert reasoning_field(model="qwen3:8b", enabled=False) == {"reasoning": False}


def test_thinking_gemini_output_marshals_to_protobuf() -> None:
    """
    Everything _thinking_gemini emits must survive proto marshaling.

    Guards the field-hit #580 crash: a pydantic field that exists on
    ChatGoogleGenerativeAI but not on the installed ThinkingConfig protobuf
    (thinking_level on google-ai-generativelanguage 0.10.0) only explodes in
    _prepare_params, so this exercises the real marshaling path.
    """
    for reasoning, budget in ((True, 512), (True, None), (False, None)):
        config = thinking_configurable(
            provider_type="gemini",
            reasoning=reasoning,
            budget=budget,
            model="gemini-3.5-flash-lite",
        )
        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=SecretStr("test-key"),
            **config,
        )
        params = model._prepare_params(stop=None)
        assert params.thinking_config is not None


# ---------------------------------------------------------------------------
# gemini_sampling_configurable tests (Gemini 3 default-temperature TODO)
# ---------------------------------------------------------------------------


def test_is_gemini_3_or_later_matches_family() -> None:
    """Gemini 3-family names match; earlier and non-Gemini names do not."""
    assert is_gemini_3_or_later("gemini-3.5-flash-lite")
    assert is_gemini_3_or_later("gemini-3.7-flash")
    assert is_gemini_3_or_later("models/gemini-3.1-flash-lite")
    assert not is_gemini_3_or_later("gemini-2.5-flash")
    assert not is_gemini_3_or_later("gpt-5")
    assert not is_gemini_3_or_later(None)
    assert not is_gemini_3_or_later("")


def test_gemini_sampling_gemini3_default_temp_rebinds_to_recommended() -> None:
    """A still-default 0.2 on a Gemini 3 model becomes 1.0 with top_p unset."""
    assert gemini_sampling_configurable(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        top_p=1.0,
        recommended_temperature=0.2,
    ) == {"temperature": GEMINI_3_RECOMMENDED_TEMPERATURE, "top_p": None}


def test_gemini_sampling_pre_gemini3_keeps_configured_values() -> None:
    """Gemini 2.5 keeps the configured temperature and top_p unchanged."""
    assert gemini_sampling_configurable(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        top_p=1.0,
        recommended_temperature=0.2,
    ) == {"temperature": 0.2, "top_p": 1.0}


def test_gemini_sampling_gemini3_custom_temp_is_honored_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A user-set non-default temperature is kept, with a warning logged."""
    with caplog.at_level("WARNING"):
        config = gemini_sampling_configurable(
            model="gemini-3.5-flash-lite",
            temperature=0.7,
            top_p=1.0,
            recommended_temperature=0.2,
        )
    assert config == {"temperature": 0.7, "top_p": 1.0}
    assert any("Gemini 3" in record.message for record in caplog.records)


def test_gemini_sampling_gemini3_explicit_recommended_temp_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An explicit 1.0 is passed through without any warning."""
    with caplog.at_level("WARNING"):
        config = gemini_sampling_configurable(
            model="gemini-3.7-flash",
            temperature=1.0,
            top_p=1.0,
            recommended_temperature=0.2,
        )
    assert config == {"temperature": 1.0, "top_p": 1.0}
    assert not caplog.records


def test_gemini_sampling_none_inputs_pass_through() -> None:
    """Missing temperature or recommended default never triggers the rebind."""
    assert gemini_sampling_configurable(
        model="gemini-3.5-flash-lite",
        temperature=None,
        top_p=None,
        recommended_temperature=0.2,
    ) == {"temperature": None, "top_p": None}
    assert gemini_sampling_configurable(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        top_p=1.0,
        recommended_temperature=None,
    ) == {"temperature": 0.2, "top_p": 1.0}


def test_gemini_sampling_output_survives_model_reconstruction() -> None:
    """
    The rebound values must survive the configurable-fields path for real.

    RunnableConfigurableFields reconstructs ChatGoogleGenerativeAI via
    __init__ with every configurable entry passed explicitly (temperature
    is a non-Optional float there, so None would raise), and
    _prepare_params drops None values from the request. Exercise both with
    the real class: temperature lands at 1.0 and top_p is absent from the
    generated GenerationConfig.
    """
    config = gemini_sampling_configurable(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        top_p=1.0,
        recommended_temperature=0.2,
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=SecretStr("test-key"),
        **config,
    )
    params = model._prepare_params(stop=None)
    assert params.temperature == GEMINI_3_RECOMMENDED_TEMPERATURE
    assert "top_p" not in params
