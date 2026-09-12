"""Speech-to-text platform for Home Generative Agent."""

from __future__ import annotations

import asyncio
import base64
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
    CONF_STT_EXTRA_BODY,
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL_NAME,
    CONF_STT_OPENAI_PROVIDER_ID,
    CONF_STT_PROMPT,
    CONF_STT_REQUEST_FORMAT,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TEMPERATURE,
    CONF_STT_TRANSLATE,
    RECOMMENDED_LOCAL_STT_MODEL,
    RECOMMENDED_OPENAI_STT_MODEL,
    STT_REQUEST_FORMAT_JSON,
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
    extra_body: Mapping[str, Any] | None = None,
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
    if extra_body:
        request["extra_body"] = dict(extra_body)
    return request


def _build_json_request(  # noqa: PLR0913
    model_name: str,
    audio_bytes: bytes,
    audio_format: str,
    language: Any,
    temperature: Any,
    response_format: Any,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the JSON transcription body with the audio base64-encoded.

    The configured prompt is deliberately absent: it is not a field of this
    request shape (OpenRouter's parameter table has no ``prompt``, and their
    multipart compatibility layer documents it as accepted-but-ignored), so
    sending it would at best do nothing and at worst be rejected as unknown.
    An endpoint that does want one can be given it through ``extra_body``,
    which is merged last and so can also override anything built here.
    """
    body: dict[str, Any] = {
        "model": model_name,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
        },
    }
    if language:
        body["language"] = language
    if temperature is not None:
        body["temperature"] = temperature
    if response_format:
        body["response_format"] = response_format
    if extra_body:
        body.update(extra_body)
    return body


def _supports_translations(model_name: Any, provider_type: str) -> bool:
    """
    Whether this model can be sent to the ``/audio/translations`` endpoint.

    Local servers (faster-whisper) serve translations for any whisper model;
    OpenAI only does for ``whisper-1``. A local non-whisper model degrades to
    transcription like the OpenAI path does, rather than 404ing against an
    endpoint the server never exposes.
    """
    if provider_type == "local":
        return "whisper" in str(model_name).lower()
    return model_name == "whisper-1"


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

    async def _send_request(  # noqa: PLR0913
        self,
        client: Any,
        connection: OpenAIConnection,
        request: dict[str, Any],
        json_body: dict[str, Any] | None,
        *,
        translate: bool,
        supports_translations: bool,
    ) -> Any:
        """
        Send the built request over the configured transport.

        ``json_body`` set means the JSON request format: a raw POST of the
        JSON body through the same configured client, so the pinned timeout,
        retry policy and keyless-Authorization rule all still apply. That
        shape has no translations endpoint, so translate degrades to
        transcription there the way an unsupported model already does.
        """
        if json_body is not None:
            if translate:
                LOGGER.warning(
                    "Translate requested but the JSON request format has no "
                    "translations endpoint; using transcription."
                )
            return await client.post(
                "/audio/transcriptions",
                body=json_body,
                cast_to=object,
                options=connection.request_options(),
            )
        if translate and supports_translations:
            # The translations endpoint always outputs English and has no
            # language parameter — passing one is a TypeError in the SDK.
            request.pop("language", None)
            return await client.audio.translations.create(**request)
        if translate:
            LOGGER.warning(
                "Translate requested but model %s does not support "
                "translations; using transcription.",
                request.get("model"),
            )
        return await client.audio.transcriptions.create(**request)

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

    async def async_process_audio_stream(
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
        stored_extra_body = model_data.get(CONF_STT_EXTRA_BODY)
        extra_body = (
            dict(stored_extra_body) if isinstance(stored_extra_body, Mapping) else {}
        )
        use_json = model_data.get(CONF_STT_REQUEST_FORMAT) == STT_REQUEST_FORMAT_JSON

        audio_bytes = await _stream_to_bytes(stream)
        ext = _format_extension(metadata)
        if not audio_bytes:
            LOGGER.warning("STT audio stream is empty for %s", self.entity_id)
            return SpeechResult(result=result_state, text=None)
        if ext == "wav" and metadata.codec == stt.AudioCodecs.PCM:
            audio_bytes = _ensure_wav(audio_bytes, metadata)

        json_body: dict[str, Any] | None = None
        request: dict[str, Any] = {}
        if use_json:
            json_body = _build_json_request(
                model_name,
                audio_bytes,
                ext,
                language,
                temperature,
                response_format,
                extra_body,
            )
        else:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{ext}"
            request = _build_openai_request(
                model_name,
                audio_file,
                language,
                prompt,
                temperature,
                response_format,
                extra_body,
            )
            connection.apply_to_request(request)

        # Building the client is inside the try: it now touches hass.data and
        # the SDK constructor, and a failure there should fail this utterance,
        # not raise out into the assist pipeline.
        try:
            client = self._get_client(connection.api_key, connection.base_url)
            response = await self._send_request(
                client,
                connection,
                request,
                json_body,
                translate=translate,
                supports_translations=_supports_translations(model_name, provider_type),
            )
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
