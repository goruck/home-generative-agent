# ruff: noqa: S101
"""Tests for the OpenAI speech-to-text platform."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from homeassistant.components import stt as ha_stt

from custom_components.home_generative_agent import stt as hga_stt
from custom_components.home_generative_agent.const import (
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL_NAME,
    CONF_STT_OPENAI_PROVIDER_ID,
    CONF_STT_PROMPT,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TEMPERATURE,
    CONF_STT_TRANSLATE,
    SUBENTRY_TYPE_MODEL_PROVIDER,
    SUBENTRY_TYPE_STT_PROVIDER,
)
from custom_components.home_generative_agent.stt import HGASttEntity

if TYPE_CHECKING:
    from collections.abc import Mapping

AUDIO = b"\x00\x01" * 32


class _FakeSubentry:
    """
    Minimal stand-in for a Home Assistant config subentry.

    ``data`` is a read-only mapping like the real ``ConfigSubentry.data``, so a
    test cannot invalidate the cache by mutating the mapping in place — a way HA
    itself never updates a subentry.
    """

    def __init__(
        self, subentry_type: str, data: dict[str, Any], title: str = "STT"
    ) -> None:
        self.subentry_type = subentry_type
        self.data: Mapping[str, Any] = MappingProxyType(dict(data))
        self.title = title

    def reconfigure(self, data: dict[str, Any]) -> None:
        """
        Replace the whole data mapping, as ``async_update_subentry`` does.

        HA keeps the same ConfigSubentry object and swaps ``data`` wholesale via
        ``object.__setattr__`` (config_entries.py), so the entity's
        ``__init__``-captured subentry observes the new settings.
        """
        self.data = MappingProxyType(dict(data))


class _FakeEntry:
    """Minimal stand-in for a Home Assistant config entry."""

    def __init__(self, subentries: dict[str, _FakeSubentry]) -> None:
        self.entry_id = "entry_1"
        self.subentries = subentries


async def _stream(payload: bytes = AUDIO) -> Any:
    """Return an async iterator over a short audio payload."""

    async def _gen() -> Any:
        yield payload

    return _gen()


def _metadata(
    fmt: ha_stt.AudioFormats = ha_stt.AudioFormats.WAV,
    codec: ha_stt.AudioCodecs = ha_stt.AudioCodecs.PCM,
) -> ha_stt.SpeechMetadata:
    return ha_stt.SpeechMetadata(
        language="en-US",
        format=fmt,
        codec=codec,
        bit_rate=ha_stt.AudioBitRates.BITRATE_16,
        sample_rate=ha_stt.AudioSampleRates.SAMPLERATE_16000,
        channel=ha_stt.AudioChannels.CHANNEL_MONO,
    )


def _make_entity(
    settings: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    provider_settings: dict[str, Any] | None = None,
    provider_type: str = "openai",
) -> tuple[HGASttEntity, _FakeEntry]:
    """Build an STT entity backed by fake subentries."""
    subentries: dict[str, _FakeSubentry] = {
        "stt_1": _FakeSubentry(
            SUBENTRY_TYPE_STT_PROVIDER,
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
    entity = HGASttEntity(cast("Any", entry), "stt_1")
    entity.hass = cast("Any", SimpleNamespace(config=SimpleNamespace(language="en")))
    entity.entity_id = "stt.hga_test"
    return entity, entry


def _no_network(request: httpx.Request) -> httpx.Response:
    """Fail loudly instead of letting a test reach the real API."""
    msg = f"test attempted a real network request: {request.method} {request.url}"
    raise AssertionError(msg)


@pytest.fixture
async def shared_httpx_client() -> Any:
    """
    Stand in for Home Assistant's shared httpx client.

    Wired to a transport that refuses every request, so a regression that builds
    a client on a path that should not cannot silently call api.openai.com.
    """
    client = httpx.AsyncClient(transport=httpx.MockTransport(_no_network))
    yield client
    await client.aclose()


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch, shared_httpx_client: Any) -> Any:
    """Patch get_async_client and count AsyncOpenAI constructions."""
    calls: dict[str, Any] = {"http_clients": [], "constructed": []}

    calls["kwargs"] = []

    def _fake_get_async_client(hass: Any) -> httpx.AsyncClient:
        calls["http_clients"].append(hass)
        return shared_httpx_client

    real_async_openai = hga_stt.AsyncOpenAI

    def _counting_async_openai(**kwargs: Any) -> Any:
        client = real_async_openai(**kwargs)
        calls["constructed"].append(client)
        calls["kwargs"].append(kwargs)
        return client

    monkeypatch.setattr(hga_stt, "get_async_client", _fake_get_async_client)
    monkeypatch.setattr(hga_stt, "AsyncOpenAI", _counting_async_openai)
    return calls


def _stub_responses(client: Any, responses: list[Any]) -> dict[str, list[Any]]:
    """Stub transcription/translation calls on a real OpenAI client object."""
    seen: dict[str, list[Any]] = {"transcriptions": [], "translations": []}
    queue = list(responses)

    async def _next() -> Any:
        result = queue.pop(0) if queue else SimpleNamespace(text="ok")
        if isinstance(result, Exception):
            raise result
        return result

    async def _transcribe(**kwargs: Any) -> Any:
        seen["transcriptions"].append(kwargs)
        return await _next()

    async def _translate(**kwargs: Any) -> Any:
        seen["translations"].append(kwargs)
        return await _next()

    client.audio.transcriptions.create = _transcribe
    client.audio.translations.create = _translate
    return seen


async def _run(
    entity: HGASttEntity,
    responses: list[Any],
    metadata: ha_stt.SpeechMetadata | None = None,
    payload: bytes = AUDIO,
) -> Any:
    """Run one stream, stubbing the client's network calls once it exists."""
    seen: dict[str, list[Any]] = {}
    original = entity._get_client

    def _wrapped(api_key: str) -> Any:
        client = original(api_key)
        seen.update(_stub_responses(client, responses))
        return client

    entity._get_client = _wrapped  # type: ignore[method-assign]
    try:
        result = await entity.async_process_audio_stream(
            metadata or _metadata(), await _stream(payload)
        )
    finally:
        # Delete rather than rebind, so the class method is not permanently
        # shadowed by an instance attribute holding a self-reference.
        with contextlib.suppress(AttributeError):
            object.__delattr__(entity, "_get_client")
    return result, seen


async def test_client_uses_shared_httpx_client(
    patched_client: Any, shared_httpx_client: Any
) -> None:
    """The SDK is given HA's shared httpx client instead of building its own."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, [SimpleNamespace(text="hello")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert result.text == "hello"
    assert patched_client["http_clients"] == [entity.hass]
    # Our contract: HA's client is handed to the SDK constructor.
    assert patched_client["kwargs"][0]["http_client"] is shared_httpx_client
    client = entity._openai_client
    assert client is not None
    # And the SDK honors it, so it never builds an SSL context of its own.
    assert client._client is shared_httpx_client


async def test_no_ssl_context_is_built_on_the_event_loop(
    patched_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Assert the reported #556 symptom directly, not a proxy for it.

    HA's blocking-call detector flagged ``load_verify_locations`` — the certifi
    bundle being read from disk on the event loop — because the SDK built its own
    SSL context per stream. Spy on the exact call the detector saw.
    """
    calls: list[Any] = []
    real_load = ssl.SSLContext.load_verify_locations

    def _spy(self: ssl.SSLContext, *args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_load(self, *args, **kwargs)

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", _spy)
    entity, _ = _make_entity()
    result, _ = await _run(entity, [SimpleNamespace(text="hello")])
    await _run(entity, [SimpleNamespace(text="again")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert calls == []
    assert len(patched_client["constructed"]) == 1


async def test_request_timeout_is_pinned_not_inherited(patched_client: Any) -> None:
    """
    The timeout must come from us, not from HA's shared client.

    The SDK adopts a supplied client's timeout only when it differs from the
    httpx default, so an unpinned client would silently retime every
    transcription if HA ever set one on the shared client.
    """
    entity, _ = _make_entity()
    await _run(entity, [SimpleNamespace(text="hello")])
    timeout = patched_client["kwargs"][0]["timeout"]
    assert timeout.read == hga_stt.STT_REQUEST_TIMEOUT_S
    assert timeout.connect == 5.0
    client = entity._openai_client
    assert client is not None
    assert client.timeout == timeout


async def test_shared_httpx_client_is_not_mutated(
    patched_client: Any, shared_httpx_client: Any
) -> None:
    """
    The SDK must not write OpenAI auth onto Home Assistant's shared client.

    The client is process-wide and shared with every other integration, so a
    leaked Authorization header would ride along on their requests.
    """
    headers_before = dict(shared_httpx_client.headers)
    entity, _ = _make_entity()
    await _run(entity, [SimpleNamespace(text="hello")])
    assert dict(shared_httpx_client.headers) == headers_before
    assert "authorization" not in {k.lower() for k in shared_httpx_client.headers}
    assert shared_httpx_client.auth is None
    assert patched_client["kwargs"][0]["api_key"] == "key-1"


@pytest.mark.usefixtures("patched_client")
async def test_real_request_path_sends_auth_without_touching_shared_client(
    shared_httpx_client: Any,
) -> None:
    """
    Drive the genuine SDK request path, not a stub.

    Every other test replaces ``audio.transcriptions.create``, so httpx never
    sends and the no-mutation checks only prove the *constructor* is clean. This
    one lets the SDK build and send a real request through MockTransport, which
    is the only place a request-time credential leak could appear.
    """
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "transcribed"})

    shared_httpx_client._transport = httpx.MockTransport(_handler)
    headers_before = dict(shared_httpx_client.headers)

    entity, _ = _make_entity()
    result = await entity.async_process_audio_stream(_metadata(), await _stream())

    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert result.text == "transcribed"
    assert len(seen) == 1
    # The request carried our credential...
    assert seen[0].headers["authorization"] == "Bearer key-1"
    assert seen[0].url.host == "api.openai.com"
    # ...and none of it stuck to the client every other integration shares.
    assert dict(shared_httpx_client.headers) == headers_before
    assert "authorization" not in {k.lower() for k in shared_httpx_client.headers}
    assert shared_httpx_client.auth is None


@pytest.mark.usefixtures("patched_client")
async def test_client_construction_failure_returns_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failure building the client fails the utterance, not the pipeline.

    Building the client now touches ``hass.data`` and the SDK constructor, so it
    sits inside the try. If it escaped, HA's assist pipeline would see an
    unhandled exception instead of a failed transcription.
    """

    def _boom(_hass: Any) -> Any:
        msg = "shared client unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(hga_stt, "get_async_client", _boom)
    entity, _ = _make_entity()
    result = await entity.async_process_audio_stream(_metadata(), await _stream())
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert result.text is None
    assert entity._openai_client is None


async def test_concurrent_streams_build_one_client(patched_client: Any) -> None:
    """
    Overlapping utterances must still share a single client.

    The invariant rests on ``_get_client`` being synchronous with no await
    between the cache check and the assignment. Making it async would silently
    reintroduce per-stream construction, so pin it here.
    """
    entity, _ = _make_entity()
    results = await asyncio.gather(
        _run(entity, [SimpleNamespace(text="one")]),
        _run(entity, [SimpleNamespace(text="two")]),
    )
    assert all(r[0].result == ha_stt.SpeechResultState.SUCCESS for r in results)
    assert len(patched_client["constructed"]) == 1
    assert len(patched_client["http_clients"]) == 1


async def test_client_reused_across_streams(patched_client: Any) -> None:
    """Repeated utterances reuse one client and one connection pool."""
    entity, _ = _make_entity()
    await _run(entity, [SimpleNamespace(text="one")])
    first = entity._openai_client
    assert first is not None
    await _run(entity, [SimpleNamespace(text="two")])
    assert entity._openai_client is first
    assert len(patched_client["constructed"]) == 1
    assert len(patched_client["http_clients"]) == 1


async def test_client_rebuilt_when_api_key_changes(patched_client: Any) -> None:
    """Reconfiguring the STT subentry key takes effect on the next stream."""
    entity, entry = _make_entity(settings={"api_key": "key-1"})
    await _run(entity, [SimpleNamespace(text="one")])
    first = entity._openai_client
    entry.subentries["stt_1"].reconfigure(
        {**entry.subentries["stt_1"].data, "settings": {"api_key": "key-2"}}
    )
    await _run(entity, [SimpleNamespace(text="two")])
    assert entity._openai_client is not first
    assert entity._openai_client is not None
    assert entity._openai_client.api_key == "key-2"
    assert len(patched_client["constructed"]) == 2


@pytest.mark.usefixtures("patched_client")
async def test_client_rebuilt_when_linked_provider_key_changes() -> None:
    """A key change on the linked model provider subentry invalidates the cache."""
    entity, entry = _make_entity(
        settings={CONF_STT_OPENAI_PROVIDER_ID: "prov_1"},
        provider_settings={"api_key": "prov-key-1"},
    )
    await _run(entity, [SimpleNamespace(text="one")])
    assert entity._openai_client is not None
    assert entity._openai_client.api_key == "prov-key-1"
    first = entity._openai_client
    entry.subentries["prov_1"].reconfigure({"settings": {"api_key": "prov-key-2"}})
    await _run(entity, [SimpleNamespace(text="two")])
    assert entity._openai_client is not first
    assert entity._openai_client is not None
    assert entity._openai_client.api_key == "prov-key-2"


@pytest.mark.usefixtures("patched_client")
async def test_request_payload_passes_model_settings() -> None:
    """Model settings are forwarded to the transcription request."""
    entity, _ = _make_entity(
        model={
            CONF_STT_MODEL_NAME: "whisper-1",
            CONF_STT_LANGUAGE: "cs",
            CONF_STT_PROMPT: "smart home",
            CONF_STT_TEMPERATURE: 0.2,
            CONF_STT_RESPONSE_FORMAT: "json",
        }
    )
    _, seen = await _run(entity, [SimpleNamespace(text="ahoj")])
    assert len(seen["transcriptions"]) == 1
    request = seen["transcriptions"][0]
    assert request["model"] == "whisper-1"
    assert request["language"] == "cs"
    assert request["prompt"] == "smart home"
    assert request["temperature"] == 0.2
    assert request["response_format"] == "json"
    assert request["file"].name == "audio.wav"


@pytest.mark.usefixtures("patched_client")
async def test_translate_uses_translations_for_whisper() -> None:
    """whisper-1 with translate enabled uses the translations endpoint."""
    entity, _ = _make_entity(
        model={CONF_STT_MODEL_NAME: "whisper-1", CONF_STT_TRANSLATE: True}
    )
    result, seen = await _run(entity, [SimpleNamespace(text="hello")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert len(seen["translations"]) == 1
    assert not seen["transcriptions"]


@pytest.mark.usefixtures("patched_client")
async def test_translate_falls_back_to_transcription(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-whisper models fall back to transcription with a warning."""
    entity, _ = _make_entity(
        model={
            CONF_STT_MODEL_NAME: "gpt-4o-mini-transcribe",
            CONF_STT_TRANSLATE: True,
        }
    )
    result, seen = await _run(entity, [SimpleNamespace(text="hello")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert len(seen["transcriptions"]) == 1
    assert not seen["translations"]
    assert "does not support translations" in caplog.text


@pytest.mark.usefixtures("patched_client")
async def test_plain_string_response_is_accepted() -> None:
    """A raw text response body still yields a successful result."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, ["plain text"])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert result.text == "plain text"


@pytest.mark.usefixtures("patched_client")
async def test_authentication_error_returns_error() -> None:
    """Authentication failures are logged and reported as an error result."""
    entity, _ = _make_entity()
    error = hga_stt.AuthenticationError(
        "bad key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.openai.com")
        ),
        body=None,
    )
    result, _ = await _run(entity, [error])
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert result.text is None


@pytest.mark.usefixtures("patched_client")
async def test_openai_error_returns_error() -> None:
    """Generic OpenAI errors are logged and reported as an error result."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, [hga_stt.OpenAIError("boom")])
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert result.text is None


@pytest.mark.usefixtures("patched_client")
async def test_missing_text_in_response_is_error() -> None:
    """A response without text is reported as an error result."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, [SimpleNamespace(text="")])
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert result.text is None


async def test_empty_audio_builds_no_client(patched_client: Any) -> None:
    """The empty-audio guard returns before any client is constructed."""
    entity, _ = _make_entity()

    async def _empty() -> Any:
        return
        yield b""  # pragma: no cover

    result = await entity.async_process_audio_stream(_metadata(), _empty())
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert entity._openai_client is None
    assert not patched_client["constructed"]
    assert not patched_client["http_clients"]


async def test_missing_api_key_builds_no_client(patched_client: Any) -> None:
    """A missing API key returns before any client is constructed."""
    entity, _ = _make_entity(settings={})
    result = await entity.async_process_audio_stream(_metadata(), await _stream())
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert entity._openai_client is None
    assert not patched_client["constructed"]
    assert not patched_client["http_clients"]


async def test_non_openai_provider_builds_no_client(patched_client: Any) -> None:
    """Non-OpenAI providers are not handled by this entity."""
    entity, _ = _make_entity(provider_type="whisper")
    result = await entity.async_process_audio_stream(_metadata(), await _stream())
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert entity._openai_client is None
    assert not patched_client["constructed"]
    assert not patched_client["http_clients"]


@pytest.mark.usefixtures("patched_client")
async def test_unexpected_error_returns_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-OpenAI exception is caught, logged and reported as an error."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, [RuntimeError("kaboom")])
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert result.text is None
    assert "Unexpected error during STT processing" in caplog.text


@pytest.mark.usefixtures("patched_client")
async def test_dict_response_text_is_accepted() -> None:
    """A mapping response body still yields a successful result."""
    entity, _ = _make_entity()
    result, _ = await _run(entity, [{"text": "from dict"}])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert result.text == "from dict"


@pytest.mark.usefixtures("patched_client")
async def test_existing_wav_audio_is_not_rewrapped() -> None:
    """Audio that already carries a RIFF/WAVE header is uploaded untouched."""
    payload = b"RIFF\x00\x00\x00\x00WAVE" + AUDIO
    entity, _ = _make_entity()
    _, seen = await _run(entity, [SimpleNamespace(text="ok")], payload=payload)
    request = seen["transcriptions"][0]
    assert request["file"].getvalue() == payload
    assert request["file"].name == "audio.wav"


@pytest.mark.usefixtures("patched_client")
async def test_non_pcm_audio_skips_wav_wrapping() -> None:
    """OGG/OPUS audio keeps its own container and extension."""
    entity, _ = _make_entity()
    _, seen = await _run(
        entity,
        [SimpleNamespace(text="ok")],
        metadata=_metadata(ha_stt.AudioFormats.OGG, ha_stt.AudioCodecs.OPUS),
    )
    request = seen["transcriptions"][0]
    assert request["file"].getvalue() == AUDIO
    assert request["file"].name == "audio.ogg"


@pytest.mark.usefixtures("patched_client")
async def test_missing_provider_subentry_falls_back_to_settings_key() -> None:
    """A dangling provider link falls back to the STT subentry's own key."""
    entity, _ = _make_entity(
        settings={CONF_STT_OPENAI_PROVIDER_ID: "gone", "api_key": "fallback-key"}
    )
    result, _ = await _run(entity, [SimpleNamespace(text="ok")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert entity._openai_client is not None
    assert entity._openai_client.api_key == "fallback-key"


@pytest.mark.usefixtures("patched_client")
async def test_wrong_provider_subentry_type_falls_back_to_settings_key() -> None:
    """A link pointing at a non-model-provider subentry is ignored."""
    entity, entry = _make_entity(
        settings={CONF_STT_OPENAI_PROVIDER_ID: "prov_1", "api_key": "fallback-key"},
        provider_settings={"api_key": "prov-key-1"},
    )
    entry.subentries["prov_1"].subentry_type = SUBENTRY_TYPE_STT_PROVIDER
    result, _ = await _run(entity, [SimpleNamespace(text="ok")])
    assert result.result == ha_stt.SpeechResultState.SUCCESS
    assert entity._openai_client is not None
    assert entity._openai_client.api_key == "fallback-key"


async def test_linked_provider_without_key_builds_no_client(
    patched_client: Any,
) -> None:
    """A linked provider with no key short-circuits before any client is built."""
    entity, _ = _make_entity(
        settings={CONF_STT_OPENAI_PROVIDER_ID: "prov_1", "api_key": "unused-key"},
        provider_settings={},
    )
    result = await entity.async_process_audio_stream(_metadata(), await _stream())
    assert result.result == ha_stt.SpeechResultState.ERROR
    assert entity._openai_client is None
    assert not patched_client["constructed"]
    assert not patched_client["http_clients"]


async def test_client_reused_when_key_value_unchanged(patched_client: Any) -> None:
    """
    The cache is keyed on the key's value, not on string identity.

    A config-entry reload rehydrates subentry data from JSON, so the key arrives
    as a fresh string object. An identity comparison would miss on every
    utterance and rebuild a client per stream — the churn #556 exists to remove.
    Built at runtime because CPython interns equal literals within a module,
    which would make this test pass against an identity comparison.
    """
    entity, entry = _make_entity(settings={"api_key": "key-1"})
    await _run(entity, [SimpleNamespace(text="one")])
    first = entity._openai_client
    assert first is not None
    # A literal here would be interned and defeat the point of this test.
    reloaded = "".join(["key", "-", "1"])  # noqa: FLY002
    assert reloaded == "key-1"
    assert reloaded is not entity._openai_client_api_key
    entry.subentries["stt_1"].reconfigure(
        {**entry.subentries["stt_1"].data, "settings": {"api_key": reloaded}}
    )
    await _run(entity, [SimpleNamespace(text="two")])
    assert entity._openai_client is first
    assert len(patched_client["constructed"]) == 1


async def test_stream_to_bytes_prefers_ha_helper_when_awaitable() -> None:
    """
    The HA helper wins over the fallbacks when it returns an awaitable.

    Installed HA has no ``async_stream_to_bytes``, so this compat branch is dead
    in CI today and would silently rot until the release that adds the helper.
    """

    async def _helper(stream: Any) -> bytes:  # noqa: ARG001
        return b"from-helper"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hga_stt.stt, "async_stream_to_bytes", _helper, raising=False)
        assert await hga_stt._stream_to_bytes(await _stream()) == b"from-helper"


async def test_stream_to_bytes_accepts_sync_helper_result() -> None:
    """The same helper returning bytes directly is used without awaiting."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            hga_stt.stt,
            "async_stream_to_bytes",
            lambda _stream: b"sync-bytes",
            raising=False,
        )
        assert await hga_stt._stream_to_bytes(await _stream()) == b"sync-bytes"


async def test_stream_to_bytes_reads_sync_and_coroutine_read_objects() -> None:
    """Objects exposing ``read()`` are supported, sync or coroutine."""

    class _SyncRead:
        def read(self) -> bytes:
            return b"sync-read"

    class _AsyncRead:
        async def _payload(self) -> bytes:
            return b"async-read"

        def read(self) -> Any:
            return self._payload()

    assert await hga_stt._stream_to_bytes(_SyncRead()) == b"sync-read"
    assert await hga_stt._stream_to_bytes(_AsyncRead()) == b"async-read"


async def test_stream_to_bytes_returns_empty_for_unknown_stream() -> None:
    """An object with none of the three interfaces yields the empty-audio guard."""
    assert await hga_stt._stream_to_bytes(object()) == b""
