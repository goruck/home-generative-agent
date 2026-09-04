"""Speech-to-text platform for Home Generative Agent."""

from __future__ import annotations

import asyncio
import inspect
import io
import logging
import wave
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from homeassistant.components import stt
from homeassistant.components.stt import (
    SpeechMetadata,
    SpeechResult,
    SpeechToTextEntity,
)
from openai import AuthenticationError, OpenAIError

from .const import (
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL_NAME,
    CONF_STT_OPENAI_PROVIDER_ID,
    CONF_STT_PROMPT,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TEMPERATURE,
    CONF_STT_TRANSLATE,
    RECOMMENDED_LOCAL_STT_MODEL,
    RECOMMENDED_OPENAI_STT_MODEL,
    SUBENTRY_TYPE_STT_PROVIDER,
)
from .core.openai_endpoint import (
    OpenAIClientCache,
    OpenAIConnection,
    OpenAIConnectionError,
    load_model_settings,
    resolve_openai_connection,
)

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .core.runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

# Pinned so the effective timeout never depends on what HA's shared httpx
# client happens to carry. Matches the chat provider timeout in __init__.py.
STT_REQUEST_TIMEOUT_S = 120.0

_FORMAT_EXTENSIONS = {
    "wav": "wav",
    "wave": "wav",
    "flac": "flac",
    "mp3": "mp3",
    "ogg": "ogg",
    "m4a": "m4a",
    "webm": "webm",
}


async def _stream_to_bytes(stream: Any) -> bytes:
    """Read audio bytes from a stream across HA versions."""
    stream_to_bytes = getattr(stt, "async_stream_to_bytes", None)
    if callable(stream_to_bytes):
        result = stream_to_bytes(stream)
        if inspect.isawaitable(result):
            return await result
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
    if hasattr(stream, "read"):
        data = stream.read()
        if asyncio.iscoroutine(data):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    if hasattr(stream, "__aiter__"):
        return b"".join(
            [
                bytes(chunk)
                async for chunk in stream
                if isinstance(chunk, (bytes, bytearray))
            ]
        )
    return b""


def _all_enum_values(enum_cls: Any) -> list[Any]:
    """Return all enum values for an STT capability list."""
    try:
        return list(enum_cls)
    except TypeError:
        return []


def _normalize_int(value: Any, default: int) -> int:
    """Return an int from enum/int/str, with a fallback default."""
    if isinstance(value, int):
        return value
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _build_openai_request(  # noqa: PLR0913
    model_name: str,
    audio_file: io.BytesIO,
    language: Any,
    prompt: Any,
    temperature: Any,
    response_format: Any,
) -> dict[str, Any]:
    """Build OpenAI STT request payload."""
    request: dict[str, Any] = {"model": model_name, "file": audio_file}
    if language:
        request["language"] = language
    if prompt:
        request["prompt"] = prompt
    if temperature is not None:
        request["temperature"] = temperature
    if response_format:
        request["response_format"] = response_format
    return request


def _extract_text_response(response: Any) -> str | None:
    """Extract text from OpenAI STT responses."""
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        return response.get("text")
    return text


def _ensure_wav(audio_bytes: bytes, metadata: SpeechMetadata) -> bytes:
    """Ensure audio bytes are WAV with header when metadata indicates PCM."""
    if audio_bytes[:4] == b"RIFF" and b"WAVE" in audio_bytes[:12]:
        return audio_bytes
    channels = _normalize_int(metadata.channel, 1)
    sample_rate = _normalize_int(metadata.sample_rate, 16000)
    bit_rate = _normalize_int(metadata.bit_rate, 16)
    sample_width = max(1, bit_rate // 8)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    entry: HGAConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up STT entities from config entries."""
    entities: list[HGASttEntity] = []
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_STT_PROVIDER:
            continue
        entities.append(HGASttEntity(entry, subentry.subentry_id))
    if entities:
        async_add_entities(entities)


def _format_extension(metadata: SpeechMetadata) -> str:
    fmt: Any = getattr(metadata, "format", None)
    if isinstance(fmt, str):
        return _FORMAT_EXTENSIONS.get(fmt.lower(), "wav")
    if hasattr(fmt, "value") and isinstance(fmt.value, str):
        return _FORMAT_EXTENSIONS.get(fmt.value.lower(), "wav")
    return "wav"


class HGASttEntity(SpeechToTextEntity):
    """Speech-to-text entity for Home Generative Agent."""

    _attr_has_entity_name = True

    def __init__(self, entry: ConfigEntry, subentry_id: str) -> None:
        """Initialize the STT entity."""
        self.entry = entry
        self.subentry_id = subentry_id
        self._subentry = entry.subentries[subentry_id]
        self._attr_unique_id = f"{entry.entry_id}_{subentry_id}"
        self._attr_name = self._subentry.title or "STT"
        self._clients = OpenAIClientCache(STT_REQUEST_TIMEOUT_S)

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        languages: set[str] = set()
        if self.hass and (hass_lang := self.hass.config.language):
            languages.add(hass_lang)
            if "-" in hass_lang:
                languages.add(hass_lang.split("-", 1)[0])
        model_data = self._subentry.data.get("model", {})
        if isinstance(model_data, Mapping):
            language = model_data.get(CONF_STT_LANGUAGE)
            if isinstance(language, str) and language:
                languages.add(language)
                if "-" in language:
                    languages.add(language.split("-", 1)[0])
        if not languages:
            languages.add("en")
        return sorted(languages)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported audio formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported audio codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported audio sample rates."""
        return _all_enum_values(stt.AudioSampleRates)

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported audio bit rates."""
        return _all_enum_values(stt.AudioBitRates)

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported audio channels."""
        return _all_enum_values(stt.AudioChannels)

    def _get_client(self, api_key: str, base_url: str | None) -> Any:
        """Return the cached OpenAI client for these credentials."""
        return self._clients.get(self.hass, api_key, base_url)

    def _resolve_connection(
        self, provider_type: str, data: dict[str, Any]
    ) -> OpenAIConnection | None:
        """Return the connection for this subentry, or None (logged) if unusable."""
        try:
            return resolve_openai_connection(
                self.entry,
                provider_type,
                data,
                provider_id_key=CONF_STT_OPENAI_PROVIDER_ID,
            )
        except OpenAIConnectionError as err:
            LOGGER.warning("STT %s for %s", err, self.entity_id)
            return None

    async def async_process_audio_stream(  # noqa: PLR0912
        self, metadata: SpeechMetadata, stream: Any
    ) -> SpeechResult:
        """Process an audio stream for speech-to-text."""
        result_state = stt.SpeechResultState.ERROR
        text: str | None = None
        data = dict(self._subentry.data)
        provider_type = data.get("provider_type")
        if provider_type not in ("openai", "local"):
            return SpeechResult(result=result_state, text=None)

        connection = self._resolve_connection(provider_type, data)
        if connection is None:
            return SpeechResult(result=result_state, text=None)

        model_data = load_model_settings(data)
        default_model = (
            RECOMMENDED_LOCAL_STT_MODEL
            if provider_type == "local"
            else RECOMMENDED_OPENAI_STT_MODEL
        )
        model_name = model_data.get(CONF_STT_MODEL_NAME, default_model)
        language = model_data.get(CONF_STT_LANGUAGE)
        prompt = model_data.get(CONF_STT_PROMPT)
        temperature = model_data.get(CONF_STT_TEMPERATURE)
        translate = bool(model_data.get(CONF_STT_TRANSLATE))
        response_format = model_data.get(CONF_STT_RESPONSE_FORMAT)

        audio_bytes = await _stream_to_bytes(stream)
        ext = _format_extension(metadata)
        if not audio_bytes:
            LOGGER.warning("STT audio stream is empty for %s", self.entity_id)
            return SpeechResult(result=result_state, text=None)
        if ext == "wav" and metadata.codec == stt.AudioCodecs.PCM:
            audio_bytes = _ensure_wav(audio_bytes, metadata)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{ext}"

        request = _build_openai_request(
            model_name,
            audio_file,
            language,
            prompt,
            temperature,
            response_format,
        )
        connection.apply_to_request(request)

        # Building the client is inside the try: it now touches hass.data and
        # the SDK constructor, and a failure there should fail this utterance,
        # not raise out into the assist pipeline.
        try:
            client = self._get_client(connection.api_key, connection.base_url)
            # Local servers (faster-whisper) serve /audio/translations for any
            # whisper model; OpenAI only does for whisper-1. A local non-whisper
            # model degrades to transcription like the OpenAI path does, rather
            # than 404ing against an endpoint the server never exposes.
            if translate and (
                "whisper" in str(model_name).lower()
                if provider_type == "local"
                else model_name == "whisper-1"
            ):
                # The translations endpoint always outputs English and has no
                # language parameter — passing one is a TypeError in the SDK.
                request.pop("language", None)
                response = await client.audio.translations.create(**request)
            else:
                if translate:
                    LOGGER.warning(
                        "Translate requested but model %s does not support "
                        "translations; using transcription.",
                        model_name,
                    )
                response = await client.audio.transcriptions.create(**request)
        except AuthenticationError:
            LOGGER.warning("OpenAI STT authentication failed for %s", self.entity_id)
        except OpenAIError as err:
            LOGGER.warning("OpenAI STT request failed: %s", err)
        except Exception:
            LOGGER.exception("Unexpected error during STT processing")
        else:
            text = _extract_text_response(response)
            if not text:
                LOGGER.warning(
                    "STT response missing text for %s (format=%s)",
                    self.entity_id,
                    response_format or "text",
                )
            else:
                result_state = stt.SpeechResultState.SUCCESS

        return SpeechResult(
            result=result_state,
            text=text if result_state == stt.SpeechResultState.SUCCESS else None,
        )
