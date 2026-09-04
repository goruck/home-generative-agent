# ruff: noqa: S101
"""Tests for the text-to-speech platform (OpenAI and local OpenAI-compatible)."""

from __future__ import annotations

import contextlib
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from homeassistant.components.tts import ATTR_PREFERRED_FORMAT, ATTR_VOICE
from homeassistant.exceptions import HomeAssistantError
from openai import AuthenticationError, Omit, OpenAIError
from openai._models import FinalRequestOptions

from custom_components.home_generative_agent import tts as hga_tts
from custom_components.home_generative_agent.const import (
    CONF_TTS_INSTRUCTIONS,
    CONF_TTS_MODEL_NAME,
    CONF_TTS_OPENAI_PROVIDER_ID,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    LOCAL_KEYLESS_API_KEY,
    OPENAI_TTS_VOICES,
    RECOMMENDED_LOCAL_TTS_MODEL,
    RECOMMENDED_LOCAL_TTS_VOICE,
    RECOMMENDED_OPENAI_TTS_MODEL,
    RECOMMENDED_OPENAI_TTS_VOICE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_TTS_PROVIDER,
)
from custom_components.home_generative_agent.core import openai_endpoint
from custom_components.home_generative_agent.tts import HGATtsEntity

if TYPE_CHECKING:
    from collections.abc import Mapping

AUDIO = b"ID3\x00fake-mp3-bytes"
LOCAL_BASE_URL = "http://speaches-box:8000/v1"


class _FakeSubentry:
    """Minimal stand-in for a Home Assistant config subentry."""

    def __init__(
        self, subentry_type: str, data: dict[str, Any], title: str = "TTS - Test"
    ) -> None:
        self.subentry_type = subentry_type
        self.subentry_id = "tts_1"
        self.data: Mapping[str, Any] = MappingProxyType(dict(data))
        self.title = title

    def reconfigure(self, data: dict[str, Any]) -> None:
        """Replace the whole data mapping, as ``async_update_subentry`` does."""
        self.data = MappingProxyType(dict(data))


class _FakeEntry:
    """Minimal stand-in for a Home Assistant config entry."""

    def __init__(self, subentries: dict[str, _FakeSubentry]) -> None:
        self.entry_id = "entry_1"
        self.subentries = subentries


def _make_entity(
    settings: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    provider_settings: dict[str, Any] | None = None,
    provider_type: str = "openai",
) -> tuple[HGATtsEntity, _FakeEntry]:
    """Build a TTS entity backed by fake subentries."""
    subentries: dict[str, _FakeSubentry] = {
        "tts_1": _FakeSubentry(
            SUBENTRY_TYPE_TTS_PROVIDER,
            {
                "provider_type": provider_type,
                "settings": settings if settings is not None else {"api_key": "key-1"},
                "model": model or {},
            },
        )
    }
    if provider_settings is not None:
        subentries["prov_1"] = _FakeSubentry(
            SUBENTRY_TYPE_MODEL_PROVIDER,
            {"settings": provider_settings},
            title="OpenAI",
        )
    entry = _FakeEntry(subentries)
    entity = HGATtsEntity(cast("Any", entry), "tts_1")
    entity.hass = cast("Any", SimpleNamespace(config=SimpleNamespace(language="en")))
    entity.entity_id = "tts.hga_test"
    return entity, entry


def _make_local_entity(
    settings: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> tuple[HGATtsEntity, _FakeEntry]:
    """Build a local-provider TTS entity with a keyless server by default."""
    return _make_entity(
        settings=settings if settings is not None else {"base_url": LOCAL_BASE_URL},
        model=model,
        provider_type="local",
    )


def _no_network(request: httpx.Request) -> httpx.Response:
    """Fail loudly instead of letting a test reach the real API."""
    msg = f"test attempted a real network request: {request.method} {request.url}"
    raise AssertionError(msg)


@pytest.fixture
async def shared_httpx_client() -> Any:
    """Stand in for Home Assistant's shared httpx client."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(_no_network))
    yield client
    await client.aclose()


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch, shared_httpx_client: Any) -> Any:
    """Patch get_async_client and count AsyncOpenAI constructions."""
    calls: dict[str, Any] = {"http_clients": [], "constructed": [], "kwargs": []}

    def _fake_get_async_client(hass: Any) -> httpx.AsyncClient:
        calls["http_clients"].append(hass)
        return shared_httpx_client

    real_async_openai = openai_endpoint.AsyncOpenAI

    def _counting_async_openai(**kwargs: Any) -> Any:
        client = real_async_openai(**kwargs)
        calls["constructed"].append(client)
        calls["kwargs"].append(kwargs)
        return client

    monkeypatch.setattr(openai_endpoint, "get_async_client", _fake_get_async_client)
    monkeypatch.setattr(openai_endpoint, "AsyncOpenAI", _counting_async_openai)
    return calls


async def _speak(
    entity: HGATtsEntity,
    responses: list[Any] | None = None,
    message: str = "Hello there",
    options: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Synthesize once, stubbing the client's network call once it exists."""
    seen: list[dict[str, Any]] = []
    original = entity._get_client

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        _install_stub(client, responses or [], seen)
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    try:
        merged = {**entity.default_options, **(options or {})}
        result = await entity.async_get_tts_audio(message, "en-US", merged)
    finally:
        # Delete rather than rebind, so the class method is not permanently
        # shadowed by an instance attribute holding a self-reference.
        with contextlib.suppress(AttributeError):
            object.__delattr__(entity, "_get_client")
    return result, seen


def _install_stub(
    client: Any, responses: list[Any], seen: list[dict[str, Any]]
) -> None:
    """Stub ``audio.speech.create`` on a real OpenAI client, recording kwargs."""
    queue = list(responses)

    async def _create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        result = queue.pop(0) if queue else SimpleNamespace(content=AUDIO)
        if isinstance(result, Exception):
            raise result
        return result

    client.audio.speech.create = _create


def _openai_error() -> OpenAIError:
    return OpenAIError("boom")


def _auth_error() -> AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
    response = httpx.Response(401, request=request)
    return AuthenticationError("bad key", response=response, body=None)


# ------------------------------------------------------------------ client


async def test_client_uses_shared_httpx_client(
    patched_client: Any, shared_httpx_client: Any
) -> None:
    """The SDK client is built on HA's shared httpx client, not its own."""
    entity, _ = _make_entity()
    (extension, audio), _ = await _speak(entity)
    assert extension == "mp3"
    assert audio == AUDIO
    assert patched_client["http_clients"] == [entity.hass]
    assert patched_client["kwargs"][0]["http_client"] is shared_httpx_client


async def test_request_timeout_is_pinned(patched_client: Any) -> None:
    """The request timeout is set on the SDK client, not inherited."""
    entity, _ = _make_entity()
    await _speak(entity)
    timeout = patched_client["kwargs"][0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == hga_tts.TTS_REQUEST_TIMEOUT_S
    assert timeout.connect == 5.0
    # One attempt: the SDK's default 2 retries would triple the silence a
    # wedged server causes before the pipeline hears an error.
    assert patched_client["kwargs"][0]["max_retries"] == 0


async def test_client_reused_across_replies(patched_client: Any) -> None:
    """Consecutive replies with unchanged credentials reuse one client."""
    entity, _ = _make_entity()
    await _speak(entity)
    await _speak(entity)
    assert len(patched_client["constructed"]) == 1


async def test_client_rebuilt_when_api_key_changes(patched_client: Any) -> None:
    """A reconfigured key takes effect on the next reply."""
    entity, entry = _make_entity()
    await _speak(entity)
    entry.subentries["tts_1"].reconfigure(
        {"provider_type": "openai", "settings": {"api_key": "key-2"}, "model": {}}
    )
    await _speak(entity)
    assert [k["api_key"] for k in patched_client["kwargs"]] == ["key-1", "key-2"]


async def test_linked_provider_key_is_authoritative(patched_client: Any) -> None:
    """A linked OpenAI model provider's key wins over the TTS-level key."""
    entity, _ = _make_entity(
        settings={"api_key": "stale", CONF_TTS_OPENAI_PROVIDER_ID: "prov_1"},
        provider_settings={"api_key": "provider-key"},
    )
    await _speak(entity)
    assert patched_client["kwargs"][0]["api_key"] == "provider-key"


async def test_linked_provider_without_key_fails(patched_client: Any) -> None:
    """A linked provider with no key fails the reply instead of using a stale key."""
    entity, _ = _make_entity(
        settings={"api_key": "stale", CONF_TTS_OPENAI_PROVIDER_ID: "prov_1"},
        provider_settings={},
    )
    with pytest.raises(HomeAssistantError, match="API key missing"):
        await _speak(entity)
    assert patched_client["constructed"] == []


async def test_missing_api_key_fails(patched_client: Any) -> None:
    """No key anywhere fails the reply before any client is built."""
    entity, _ = _make_entity(settings={})
    with pytest.raises(HomeAssistantError, match="API key missing"):
        await _speak(entity)
    assert patched_client["constructed"] == []


# ------------------------------------------------------------------- local


async def test_local_keyless_sends_no_authorization_header(
    patched_client: Any,
) -> None:
    """A keyless local server gets the placeholder key and no bearer on the wire."""
    entity, _ = _make_local_entity()
    (extension, audio), seen = await _speak(entity)
    assert extension == "mp3"
    assert audio == AUDIO
    kwargs = patched_client["kwargs"][0]
    assert kwargs["base_url"] == LOCAL_BASE_URL
    assert kwargs["api_key"] == LOCAL_KEYLESS_API_KEY
    assert LOCAL_KEYLESS_API_KEY
    extra_headers = seen[0]["extra_headers"]
    assert isinstance(extra_headers["Authorization"], Omit)
    built = patched_client["constructed"][0]._build_request(
        FinalRequestOptions(method="post", url="/audio/speech", headers=extra_headers)
    )
    assert "authorization" not in built.headers


async def test_local_configured_key_is_sent(patched_client: Any) -> None:
    """A configured local key reaches the wire as a bearer token."""
    entity, _ = _make_local_entity(
        settings={"base_url": LOCAL_BASE_URL, "api_key": "local-key"}
    )
    _, seen = await _speak(entity)
    assert patched_client["kwargs"][0]["api_key"] == "local-key"
    assert "extra_headers" not in seen[0]
    assert patched_client["constructed"][0].auth_headers == {
        "Authorization": "Bearer local-key"
    }


async def test_local_missing_base_url_fails(patched_client: Any) -> None:
    """A local provider without a URL fails the reply, not the pipeline."""
    entity, _ = _make_local_entity(settings={})
    with pytest.raises(HomeAssistantError, match="base URL missing"):
        await _speak(entity)
    assert patched_client["constructed"] == []


@pytest.mark.usefixtures("patched_client")
async def test_local_defaults_to_recommended_model_and_voice() -> None:
    """A local provider with no model settings uses the Kokoro defaults."""
    entity, _ = _make_local_entity()
    _, seen = await _speak(entity)
    assert seen[0]["model"] == RECOMMENDED_LOCAL_TTS_MODEL
    assert seen[0]["voice"] == RECOMMENDED_LOCAL_TTS_VOICE
    assert entity.async_get_supported_voices("en-US") == [
        hga_tts.Voice(RECOMMENDED_LOCAL_TTS_VOICE, RECOMMENDED_LOCAL_TTS_VOICE)
    ]


@pytest.mark.usefixtures("patched_client")
async def test_local_opus_request_falls_back_to_mp3() -> None:
    """Speaches cannot produce opus, so an ogg request is served as mp3."""
    entity, _ = _make_local_entity()
    (extension, _), seen = await _speak(entity, options={ATTR_PREFERRED_FORMAT: "ogg"})
    assert extension == "mp3"
    assert seen[0]["response_format"] == "mp3"


# ------------------------------------------------------------------ request


@pytest.mark.usefixtures("patched_client")
async def test_request_uses_configured_model_voice_speed() -> None:
    """Configured model, voice, and a non-default speed reach the request."""
    entity, _ = _make_entity(
        model={
            CONF_TTS_MODEL_NAME: "tts-1-hd",
            CONF_TTS_VOICE: "nova",
            CONF_TTS_SPEED: 1.25,
        }
    )
    _, seen = await _speak(entity, message="Doors locked.")
    request = seen[0]
    assert request["model"] == "tts-1-hd"
    assert request["voice"] == "nova"
    assert request["input"] == "Doors locked."
    assert request["speed"] == 1.25
    assert request["response_format"] == "mp3"


@pytest.mark.usefixtures("patched_client")
async def test_default_speed_is_omitted() -> None:
    """Speed 1.0 is the API default and is not sent."""
    entity, _ = _make_entity(model={CONF_TTS_SPEED: 1.0})
    _, seen = await _speak(entity)
    assert "speed" not in seen[0]


@pytest.mark.usefixtures("patched_client")
async def test_pipeline_voice_option_overrides_configured_voice() -> None:
    """A voice chosen in the Assist pipeline wins over the subentry default."""
    entity, _ = _make_entity(model={CONF_TTS_VOICE: "nova"})
    _, seen = await _speak(entity, options={ATTR_VOICE: "onyx"})
    assert seen[0]["voice"] == "onyx"


@pytest.mark.usefixtures("patched_client")
async def test_instructions_sent_only_for_gpt4o_mini_tts() -> None:
    """Instructions go to gpt-4o-mini-tts models only; tts-1 would reject them."""
    entity, _ = _make_entity(
        model={CONF_TTS_MODEL_NAME: "gpt-4o-mini-tts", CONF_TTS_INSTRUCTIONS: "Calm."}
    )
    _, seen = await _speak(entity)
    assert seen[0]["instructions"] == "Calm."

    entity, _ = _make_entity(
        model={CONF_TTS_MODEL_NAME: "tts-1", CONF_TTS_INSTRUCTIONS: "Calm."}
    )
    _, seen = await _speak(entity)
    assert "instructions" not in seen[0]


@pytest.mark.usefixtures("patched_client")
async def test_instructions_never_sent_to_local() -> None:
    """A local server never receives instructions, whatever its model is named."""
    entity, _ = _make_local_entity(
        model={CONF_TTS_MODEL_NAME: "gpt-4o-mini-tts", CONF_TTS_INSTRUCTIONS: "Calm."}
    )
    _, seen = await _speak(entity)
    assert "instructions" not in seen[0]


@pytest.mark.parametrize(
    ("preferred", "extension", "requested"),
    [
        (None, "mp3", "mp3"),
        ("mp3", "mp3", "mp3"),
        ("wav", "wav", "wav"),
        ("flac", "flac", "flac"),
        ("ogg", "ogg", "opus"),
        ("oga", "oga", "opus"),
        ("raw", "raw", "pcm"),
        ("pcm", "pcm", "pcm"),
        ("m4a", "mp3", "mp3"),
    ],
)
@pytest.mark.usefixtures("patched_client")
async def test_openai_format_negotiation(
    preferred: str | None, extension: str, requested: str
) -> None:
    """The reported extension and requested container follow HA's preference."""
    entity, _ = _make_entity()
    options = {ATTR_PREFERRED_FORMAT: preferred} if preferred else {}
    (got_extension, _), seen = await _speak(entity, options=options)
    assert got_extension == extension
    assert seen[0]["response_format"] == requested


async def test_empty_message_is_rejected(patched_client: Any) -> None:
    """Whitespace-only text is refused before a request is made."""
    entity, _ = _make_entity()
    with pytest.raises(HomeAssistantError, match="No text"):
        await _speak(entity, message="   ")
    assert patched_client["constructed"] == []


@pytest.mark.usefixtures("patched_client")
async def test_empty_audio_is_an_error() -> None:
    """A backend returning no bytes fails the reply instead of playing silence."""
    entity, _ = _make_entity()
    with pytest.raises(HomeAssistantError, match="no audio"):
        await _speak(entity, responses=[SimpleNamespace(content=b"")])


# ------------------------------------------------------------------- errors


@pytest.mark.usefixtures("patched_client")
async def test_authentication_error_raises_home_assistant_error() -> None:
    """A rejected key surfaces as HomeAssistantError, the TTS contract."""
    entity, _ = _make_entity()
    with pytest.raises(HomeAssistantError, match="authentication failed"):
        await _speak(entity, responses=[_auth_error()])


@pytest.mark.usefixtures("patched_client")
async def test_openai_error_raises_home_assistant_error() -> None:
    """Any SDK error surfaces as HomeAssistantError with the cause attached."""
    entity, _ = _make_entity()
    with pytest.raises(HomeAssistantError, match="request failed") as excinfo:
        await _speak(entity, responses=[_openai_error()])
    assert isinstance(excinfo.value.__cause__, OpenAIError)


async def test_unknown_provider_type_fails(patched_client: Any) -> None:
    """An unsupported provider type fails clearly and builds no client."""
    entity, _ = _make_entity(provider_type="mystery")
    with pytest.raises(HomeAssistantError, match="Unsupported"):
        await _speak(entity)
    assert patched_client["constructed"] == []


# ------------------------------------------------------------------- voices


def test_openai_voices_put_configured_voice_first() -> None:
    """The pipeline picks index 0, so the configured voice leads the list."""
    entity, _ = _make_entity(model={CONF_TTS_VOICE: "shimmer"})
    voices = entity.async_get_supported_voices("en-US")
    assert voices is not None
    assert voices[0].voice_id == "shimmer"
    assert {v.voice_id for v in voices} == {v.lower() for v in OPENAI_TTS_VOICES}


def test_openai_custom_voice_is_offered_first() -> None:
    """A voice id outside the built-in set is still offered, and first."""
    entity, _ = _make_entity(model={CONF_TTS_VOICE: "voice_1234"})
    voices = entity.async_get_supported_voices("en-US")
    assert voices is not None
    assert voices[0].voice_id == "voice_1234"
    assert len(voices) == len(OPENAI_TTS_VOICES) + 1


def test_openai_defaults() -> None:
    """Default options and model fall back to the recommended OpenAI values."""
    entity, _ = _make_entity()
    assert entity.default_options == {
        ATTR_VOICE: RECOMMENDED_OPENAI_TTS_VOICE,
        ATTR_PREFERRED_FORMAT: "mp3",
    }
    assert entity._configured_model() == RECOMMENDED_OPENAI_TTS_MODEL
    assert entity.supported_options == [ATTR_VOICE, ATTR_PREFERRED_FORMAT]
    assert entity.default_language == "en-US"
    # tts.speak checks exact membership, so bare subtags must be accepted too.
    assert {"en-US", "en", "cs", "pt-PT"} <= set(entity.supported_languages)


def test_entity_identity_follows_subentry() -> None:
    """Unique id and name come from the entry and subentry."""
    entity, _ = _make_entity()
    assert entity.unique_id == "entry_1_tts_1"
    assert entity.name == "TTS - Test"


# ---------------------------------------------------------------- platform


async def test_setup_entry_adds_one_entity_per_tts_subentry() -> None:
    """Only TTS subentries produce entities, each bound to its subentry."""
    entry = _FakeEntry(
        {
            "tts_1": _FakeSubentry(
                SUBENTRY_TYPE_TTS_PROVIDER,
                {"provider_type": "openai", "settings": {}, "model": {}},
            ),
            "prov_1": _FakeSubentry(SUBENTRY_TYPE_MODEL_PROVIDER, {"settings": {}}),
        }
    )
    added: list[tuple[list[Any], str | None]] = []

    def _add(
        new_entities: Any,
        update_before_add: bool = False,  # noqa: ARG001, FBT001, FBT002
        *,
        config_subentry_id: str | None = None,
    ) -> None:
        added.append((list(new_entities), config_subentry_id))

    await hga_tts.async_setup_entry(cast("Any", None), cast("Any", entry), _add)
    assert len(added) == 1
    entities, subentry_id = added[0]
    assert subentry_id == "tts_1"
    assert isinstance(entities[0], HGATtsEntity)
