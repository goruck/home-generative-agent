# ruff: noqa: S101
"""Tests for the text-to-speech platform (OpenAI and local OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import contextlib
import struct
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from homeassistant.components.tts import ATTR_PREFERRED_FORMAT, ATTR_VOICE
from homeassistant.components.tts.entity import TTSAudioRequest
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
    from collections.abc import AsyncGenerator, Mapping

# A well-formed ID3v2 tag (4-byte body) followed by stand-in mp3 frames; joined
# stream pieces after the first arrive without the tag.


def _wav(samples: bytes, rate: int = 22050, bits: int = 16) -> bytes:
    """Build a minimal mono PCM WAV file around ``samples``."""
    block = bits // 8
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * block, block, bits)
    return (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(samples))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


# What the stub backend returns: a real (tiny) WAV, since the streaming path
# parses it; the buffered path only passes the bytes through.
SAMPLES = b"\x01\x00" * 200
AUDIO = _wav(SAMPLES)
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
    entity.hass = cast(
        "Any",
        SimpleNamespace(
            config=SimpleNamespace(language="en"),
            async_create_background_task=lambda coro, name: asyncio.create_task(
                coro, name=name
            ),
        ),
    )
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


# ---------------------------------------------------------------- streaming


def _stub_client(entity: HGATtsEntity, seen: list[dict[str, Any]]) -> None:
    """Stub every client the entity builds, recording each request."""
    original = entity._get_client

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        _install_stub(client, [], seen)
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]


def _assert_stream_start(chunk: bytes, samples: bytes = SAMPLES) -> None:
    """Check a streamed reply starts with one open-ended WAV header, then audio."""
    assert chunk[:4] == b"RIFF"
    assert chunk[4:8] == b"\xff\xff\xff\xff"
    assert chunk[8:12] == b"WAVE"
    data = chunk.index(b"data")
    assert chunk[data + 4 : data + 8] == b"\xff\xff\xff\xff"
    body = chunk[data + 8 :]
    assert body.startswith(samples)
    # Padded with silence past Home Assistant's converter start threshold.
    assert len(chunk) >= 64 * 1024
    assert set(body[len(samples) :]) <= {0}


async def _text(*chunks: str) -> AsyncGenerator[str]:
    """Yield a whole message at once, as tts.speak does."""
    for chunk in chunks:
        yield chunk


async def _live(*chunks: str) -> AsyncGenerator[str]:
    """Yield a reply that is still being written when synthesis starts."""
    await asyncio.sleep(hga_tts.TTS_ONE_SHOT_GRACE_S * 3)
    for chunk in chunks:
        yield chunk


async def _stream(
    entity: HGATtsEntity, message_gen: Any, options: dict[str, Any] | None = None
) -> tuple[str, list[bytes]]:
    merged = {**entity.default_options, **(options or {})}
    response = await entity.async_stream_tts_audio(
        TTSAudioRequest("en-US", merged, message_gen)
    )
    return response.extension, [chunk async for chunk in response.data_gen]


def test_entity_supports_streaming_input() -> None:
    """The override is what lets the Assist pipeline stream text to us."""
    entity, _ = _make_entity()
    assert entity.async_supports_streaming_input()


@pytest.mark.usefixtures("patched_client")
async def test_stream_speaks_first_sentence_before_text_ends() -> None:
    """Audio for the first sentence is produced while the reply is unfinished."""
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    release = asyncio.Event()

    async def _slow_reply() -> AsyncGenerator[str]:
        yield "Let me check the landing light. "
        yield "It was"
        await release.wait()
        yield " turned off by the evening automation."

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest("en-US", dict(entity.default_options), _slow_reply())
    )
    first = await anext(response.data_gen)
    _assert_stream_start(first)
    assert [req["input"] for req in seen] == ["Let me check the landing light."]

    release.set()
    rest = [chunk async for chunk in response.data_gen]
    assert rest == [SAMPLES]
    assert seen[1]["input"] == "It was turned off by the evening automation."


@pytest.mark.usefixtures("patched_client")
async def test_stream_speaks_held_sentence_when_text_pauses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A finished sentence followed by silence is spoken during the silence.

    The splitter holds "Let me check." until the next word, and on a tool turn
    the next word comes only after the tools return; the idle flush is what
    lets the acknowledgement fill that gap.
    """
    monkeypatch.setattr(hga_tts, "TTS_STREAM_IDLE_FLUSH_S", 0.01)
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    tools_done = asyncio.Event()

    async def _tool_turn() -> AsyncGenerator[str]:
        yield "Let me check."
        await tools_done.wait()
        yield " The landing light is off."

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest("en-US", dict(entity.default_options), _tool_turn())
    )
    _assert_stream_start(await anext(response.data_gen))
    assert [req["input"] for req in seen] == ["Let me check."]

    tools_done.set()
    assert [chunk async for chunk in response.data_gen] == [SAMPLES]
    assert seen[1]["input"] == "The landing light is off."


@pytest.mark.usefixtures("patched_client")
async def test_stream_pause_mid_sentence_keeps_the_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pause inside a sentence does not cut it into two utterances."""
    monkeypatch.setattr(hga_tts, "TTS_STREAM_IDLE_FLUSH_S", 0.01)
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)

    async def _slow() -> AsyncGenerator[str]:
        yield "The landing light"
        await asyncio.sleep(0.05)
        yield " is off."

    _, audio = await _stream(entity, _slow())
    assert len(audio) == 1
    assert [req["input"] for req in seen] == ["The landing light is off."]


@pytest.mark.usefixtures("patched_client")
async def test_stream_batches_sentences_after_the_first() -> None:
    """Sentences that arrive together after the first share one request."""
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    extension, audio = await _stream(
        entity, _live("One is short. Two follows. ", "Three ends it.")
    )
    assert extension == "wav"
    _assert_stream_start(audio[0])
    assert audio[1:] == [SAMPLES]
    assert [req["input"] for req in seen] == [
        "One is short.",
        "Two follows. Three ends it.",
    ]


@pytest.mark.usefixtures("patched_client")
async def test_live_stream_is_wav_whatever_the_preference() -> None:
    """Batches are spliced into one WAV stream, so WAV is requested."""
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    extension, _ = await _stream(
        entity, _live("Hello there."), {ATTR_PREFERRED_FORMAT: "flac"}
    )
    assert extension == "wav"
    assert seen[0]["response_format"] == "wav"


@pytest.mark.usefixtures("patched_client")
async def test_one_shot_message_is_one_request_in_the_preferred_format() -> None:
    """tts.speak text arrives whole: no splitting, no mp3 join, no transcode."""
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    extension, audio = await _stream(
        entity,
        _text("The front door is open. ", "The garage is closed."),
        {ATTR_PREFERRED_FORMAT: "flac"},
    )
    assert extension == "flac"
    assert audio == [AUDIO]
    assert [(req["input"], req["response_format"]) for req in seen] == [
        ("The front door is open. The garage is closed.", "flac")
    ]


@pytest.mark.usefixtures("patched_client")
async def test_stream_stops_waiting_when_text_never_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn the pipeline never finishes does not leave synthesis waiting."""
    monkeypatch.setattr(hga_tts, "TTS_STREAM_TEXT_TIMEOUT_S", 0.2)
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)

    async def _abandoned() -> AsyncGenerator[str]:
        await asyncio.sleep(0.1)
        yield "Let me check."
        await asyncio.Event().wait()  # the agent failed; no end ever comes
        yield "unreachable"

    _, audio = await _stream(entity, _abandoned())
    assert len(audio) == 1
    _assert_stream_start(audio[0])
    assert [req["input"] for req in seen] == ["Let me check."]


@pytest.mark.usefixtures("patched_client")
async def test_stream_pause_after_word_keeps_the_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pause after a space must not glue the next word onto the last one."""
    monkeypatch.setattr(hga_tts, "TTS_STREAM_IDLE_FLUSH_S", 0.01)
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)

    async def _slow() -> AsyncGenerator[str]:
        await asyncio.sleep(0.2)
        yield "Here is the list:\n"
        await asyncio.sleep(0.05)
        yield "first the temperature is "
        await asyncio.sleep(0.05)
        yield "21 degrees."

    await _stream(entity, _slow())
    assert " ".join(req["input"] for req in seen) == (
        "Here is the list:\nfirst the temperature is 21 degrees."
    )


@pytest.mark.usefixtures("patched_client")
async def test_stream_flushes_sentence_ending_in_a_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closing quote after the full stop still ends the sentence."""
    monkeypatch.setattr(hga_tts, "TTS_STREAM_IDLE_FLUSH_S", 0.01)
    entity, _ = _make_entity()
    seen: list[dict[str, Any]] = []
    _stub_client(entity, seen)
    release = asyncio.Event()

    async def _quoted() -> AsyncGenerator[str]:
        await asyncio.sleep(0.2)
        yield 'I will check the "Kitchen" sensor (one moment.)'
        await release.wait()
        yield " Done."

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest("en-US", dict(entity.default_options), _quoted())
    )
    _assert_stream_start(await anext(response.data_gen))
    release.set()
    assert [chunk async for chunk in response.data_gen] == [SAMPLES]


def test_parse_wav_handles_streaming_sizes_and_extra_chunks() -> None:
    """Unknown data sizes run to the end; chunks before fmt/data are skipped."""
    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    data = (
        b"RIFF\xff\xff\xff\xffWAVE"
        b"LIST"
        + struct.pack("<I", 3)
        + b"abc\x00"  # odd size, padded
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data\xff\xff\xff\xff"
        + b"\x02\x00" * 5
    )
    piece = hga_tts._parse_wav(data)
    assert piece.fmt == fmt
    assert piece.samples == b"\x02\x00" * 5
    assert piece.block_align == 2


@pytest.mark.usefixtures("patched_client")
async def test_stream_keeps_the_satellite_fed_while_the_model_thinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A stall between sentences is filled with silence paced to real time.

    A Voice PE buffers ~100 ms; when a streamed reply stalled while the model
    worked, its speaker ran dry and made a brief click -- heard on streamed
    turns, never on one-shot speech of the same text (2026-10-01).
    """
    monkeypatch.setattr(hga_tts, "TTS_STREAM_KEEPALIVE_S", 0.02)
    monkeypatch.setattr(hga_tts, "TTS_STREAM_LEAD_S", 0.05)
    # Pad nothing extra, so the first chunk's audio is short and the pacer
    # has to top up during the stall.
    monkeypatch.setattr(hga_tts, "_CONVERTER_START_BYTES", 0)
    monkeypatch.setattr(hga_tts, "_MIN_TRAILING_SILENCE_S", 0)
    entity, _ = _make_entity()
    _stub_client(entity, [])
    stall_s = 0.4

    async def _thinking() -> AsyncGenerator[str]:
        await asyncio.sleep(hga_tts.TTS_ONE_SHOT_GRACE_S * 3)
        yield "One moment, please.\n\n"
        await asyncio.sleep(stall_s)
        yield "Because I turned it on."

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest("en-US", dict(entity.default_options), _thinking())
    )
    chunks = [chunk async for chunk in response.data_gen]

    assert chunks[0].startswith(b"RIFF")
    assert chunks[-1] == SAMPLES
    fillers = chunks[1:-1]
    assert fillers, "the stall must be filled"
    assert all(set(chunk) <= {0} for chunk in fillers)
    byte_rate = 22050 * 2
    filled_s = sum(len(chunk) for chunk in fillers) / byte_rate
    # Real-time pacing: about the stall, never far beyond it plus the lead.
    assert stall_s * 0.5 < filled_s < stall_s + 0.05 + 0.15


@pytest.mark.usefixtures("patched_client")
async def test_stream_sends_nothing_before_the_first_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence is only fed once audio has started; the header comes first."""
    monkeypatch.setattr(hga_tts, "TTS_STREAM_KEEPALIVE_S", 0.02)
    entity, _ = _make_entity()
    _stub_client(entity, [])

    async def _slow_start() -> AsyncGenerator[str]:
        await asyncio.sleep(0.3)
        yield "Hello there."

    _, audio = await _stream(entity, _slow_start())
    assert audio[0].startswith(b"RIFF")


@pytest.mark.usefixtures("patched_client")
async def test_first_piece_always_ends_in_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A first sentence past the converter threshold still gets trailing silence.

    The converter holds back its last partial frame until more input comes;
    without silence after it, that held tail was speech ("...se") and played
    seconds later, just before the answer (#671).
    """
    monkeypatch.setattr(hga_tts, "_CONVERTER_START_BYTES", 0)
    entity, _ = _make_entity()
    _stub_client(entity, [])
    _, audio = await _stream(entity, _live("Hello there."))
    body = audio[0][audio[0].index(b"data") + 8 :]
    tail = body[len(SAMPLES) :]
    assert len(tail) >= int(hga_tts._MIN_TRAILING_SILENCE_S * 22050 * 2) - 2
    assert set(tail) <= {0}


def test_parse_wav_reads_every_per_sentence_segment() -> None:
    """
    Speaches answers a multi-sentence request with one WAV file per sentence.

    Field report (2026-10-01): reading only the first file cut a reply off
    after its first sentence, a fraction of a second when it was "Yes.".
    """
    first, second = b"\x01\x00" * 50, b"\x02\x00" * 80
    piece = hga_tts._parse_wav(_wav(first) + _wav(second))
    assert piece.samples == first + second


def test_parse_wav_runs_an_unknown_size_to_the_next_segment() -> None:
    """A streaming-sized data chunk stops at the next RIFF header, not the end."""
    first, second = b"\x01\x00" * 50, b"\x02\x00" * 80
    unsized = bytearray(_wav(first))
    unsized[unsized.index(b"data") + 4 : unsized.index(b"data") + 8] = b"\xff" * 4
    piece = hga_tts._parse_wav(bytes(unsized) + _wav(second))
    assert piece.samples == first + second


def test_parse_wav_skips_a_segment_in_another_format() -> None:
    """A segment that cannot be spliced in is dropped, not played as noise."""
    first = b"\x01\x00" * 50
    piece = hga_tts._parse_wav(_wav(first) + _wav(b"\x02\x00" * 80, rate=16000))
    assert piece.samples == first


@pytest.mark.usefixtures("patched_client")
async def test_one_shot_wav_is_merged_into_one_file() -> None:
    """tts.speak to a WAV player must not stop after the first sentence."""
    entity, _ = _make_entity()
    first, second = b"\x01\x00" * 50, b"\x02\x00" * 80
    original = entity._get_client

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        _install_stub(client, [SimpleNamespace(content=_wav(first) + _wav(second))], [])
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    extension, audio = await _stream(
        entity, _text("One. Two."), {ATTR_PREFERRED_FORMAT: "wav"}
    )

    assert extension == "wav"
    assert audio == [_wav(first + second)]


@pytest.mark.parametrize("bad", [b"", b"ID3 not wav", b"RIFF\x00\x00\x00\x00WAVE"])
def test_parse_wav_rejects_what_is_not_wav(bad: bytes) -> None:
    """A backend that ignored the format request fails clearly."""
    with pytest.raises(ValueError, match=r"RIFF|data"):
        hga_tts._parse_wav(bad)


@pytest.mark.usefixtures("patched_client")
async def test_stream_skips_a_batch_in_another_format(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mid-reply format change cannot be spliced in; it is logged and skipped."""
    entity, _ = _make_entity()
    original = entity._get_client
    other = _wav(SAMPLES, rate=16000)
    stubbed: set[int] = set()

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        if id(client) not in stubbed:  # one response queue per client
            stubbed.add(id(client))
            _install_stub(
                client,
                [SimpleNamespace(content=AUDIO), SimpleNamespace(content=other)],
                [],
            )
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    _, audio = await _stream(entity, _live("One is short. ", "Two follows."))

    assert len(audio) == 1
    _assert_stream_start(audio[0])
    assert "changed audio format" in caplog.text


@pytest.mark.usefixtures("patched_client")
async def test_stream_rejects_a_backend_that_returns_no_wav() -> None:
    """Unreadable audio surfaces as a HomeAssistantError, the TTS contract."""
    entity, _ = _make_entity()
    original = entity._get_client

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        _install_stub(client, [SimpleNamespace(content=b"ID3 mp3 bytes")], [])
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    with pytest.raises(HomeAssistantError, match="unreadable WAV"):
        await _stream(entity, _live("Hello there."))


async def test_stream_of_whitespace_is_rejected(patched_client: Any) -> None:
    """A stream with no speakable text fails without a request."""
    entity, _ = _make_entity()
    with pytest.raises(HomeAssistantError, match="No text"):
        await _stream(entity, _text("  ", "\n"))
    assert patched_client["constructed"] == []


@pytest.mark.usefixtures("patched_client")
async def test_stream_backend_error_raises_home_assistant_error() -> None:
    """A failed batch surfaces the same error as the buffered path."""
    entity, _ = _make_entity()
    original = entity._get_client

    def _wrapped(api_key: str, base_url: str | None = None) -> Any:
        client = original(api_key, base_url)
        _install_stub(client, [_openai_error()], [])
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    with pytest.raises(HomeAssistantError, match="request failed"):
        await _stream(entity, _text("Hello there."))


@pytest.mark.usefixtures("patched_client")
async def test_stream_text_failure_is_raised() -> None:
    """A text stream that dies mid-reply is not swallowed as a short reply."""
    entity, _ = _make_entity()
    _stub_client(entity, [])

    async def _broken() -> AsyncGenerator[str]:
        yield "First part. "
        msg = "agent crashed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="agent crashed"):
        await _stream(entity, _broken())


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
