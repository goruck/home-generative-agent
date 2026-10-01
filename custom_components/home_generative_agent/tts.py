"""Text-to-speech platform for Home Generative Agent (OpenAI or a local server)."""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import TYPE_CHECKING, Any, NamedTuple

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from openai import AuthenticationError, OpenAIError
from propcache.api import cached_property
from sentence_stream import SentenceBoundaryDetector

from .const import (
    CONF_TTS_INSTRUCTIONS,
    CONF_TTS_MODEL_NAME,
    CONF_TTS_OPENAI_PROVIDER_ID,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    OPENAI_TTS_VOICES,
    RECOMMENDED_LOCAL_TTS_MODEL,
    RECOMMENDED_LOCAL_TTS_VOICE,
    RECOMMENDED_OPENAI_TTS_MODEL,
    RECOMMENDED_OPENAI_TTS_VOICE,
    SUBENTRY_TYPE_TTS_PROVIDER,
    TTS_DEFAULT_RESPONSE_FORMAT,
    TTS_INSTRUCTIONS_MODEL_PREFIX,
    TTS_LOCAL_RESPONSE_FORMATS,
    TTS_OPENAI_RESPONSE_FORMATS,
    TTS_SPEED_DEFAULT,
    TTS_STREAM_RESPONSE_FORMAT,
)
from .core.openai_endpoint import (
    OpenAIClientCache,
    OpenAIConnectionError,
    load_model_settings,
    resolve_openai_connection,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Mapping

    from homeassistant.components.tts.entity import TTSAudioRequest
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .core.runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

# Pinned so the effective timeout never depends on what HA's shared httpx
# client happens to carry. A spoken reply is a few sentences; a minute covers a
# cold model load on a local server with room to spare.
TTS_REQUEST_TIMEOUT_S = 60.0

# The sentence splitter holds a finished sentence until the next word arrives
# (so "Dr." is not a sentence). A pause this long in the reply means the agent
# stopped writing -- typically to run a tool -- so a held sentence is spoken
# now instead of after the tool returns. Local models stream tokens tens of
# milliseconds apart, well under this.
TTS_STREAM_IDLE_FLUSH_S = 0.4
# ASCII, ellipsis, the CJK full stop/exclamation/question marks, the Arabic
# question mark, and the Devanagari danda and double danda.
_SENTENCE_ENDINGS = (
    ".",
    "!",
    "?",
    "\u2026",
    "\u3002",
    "\uff01",
    "\uff1f",
    "\u061f",
    "\u0964",
    "\u0965",
)
# Closing quotes and brackets that may follow a sentence's final mark.
_SENTENCE_CLOSERS = "\"')]}\u00bb\u201d\u2019\u300d\u300f"
# The Assist pipeline ends its text stream only when the agent returns; a turn
# that fails or is cancelled never does. Give up after this long with no text
# so the synthesis tasks cannot wait forever. Far longer than any tool round.
TTS_STREAM_TEXT_TIMEOUT_S = 300.0
# A one-shot message (tts.speak, an announcement, a reply the pipeline already
# has whole) arrives as a text stream that ends at once. Waiting this long for
# the end tells it from a live reply, which is then synthesized in one request
# in the preferred format instead of split and joined as mp3.
TTS_ONE_SHOT_GRACE_S = 0.05

# Language tags the OpenAI speech models document. The models detect the input
# language themselves, so this list only has to satisfy the Assist pipeline's
# language matching; a local model (e.g. Kokoro) may cover fewer of them.
SUPPORTED_LANGUAGES = [
    "af-ZA",
    "ar-SA",
    "hy-AM",
    "az-AZ",
    "be-BY",
    "bs-BA",
    "bg-BG",
    "ca-ES",
    "zh-CN",
    "hr-HR",
    "cs-CZ",
    "da-DK",
    "nl-NL",
    "en-US",
    "et-EE",
    "fi-FI",
    "fr-FR",
    "gl-ES",
    "de-DE",
    "el-GR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "is-IS",
    "id-ID",
    "it-IT",
    "ja-JP",
    "kn-IN",
    "kk-KZ",
    "ko-KR",
    "lv-LV",
    "lt-LT",
    "mk-MK",
    "ms-MY",
    "mr-IN",
    "mi-NZ",
    "ne-NP",
    "no-NO",
    "fa-IR",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sr-RS",
    "sk-SK",
    "sl-SI",
    "es-ES",
    "sw-KE",
    "sv-SE",
    "fil-PH",
    "ta-IN",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "ur-PK",
    "vi-VN",
    "cy-GB",
]
DEFAULT_LANGUAGE = "en-US"


def _negotiate_format(preferred: Any, supported: frozenset[str]) -> tuple[str, str]:
    """
    Return ``(extension, request_format)`` for a preferred output format.

    ``extension`` is what the entity reports to Home Assistant and
    ``request_format`` is what the backend is asked for. Containers the backend
    cannot produce fall back to mp3 and Home Assistant converts with ffmpeg;
    the preferred_* sample options are never declared as supported, so the
    Voice PE's 16 kHz mono wav request is always converted by HA as well.
    """
    fmt = str(preferred or TTS_DEFAULT_RESPONSE_FORMAT).lower()
    # HA compares the returned extension to the literal preference and only
    # skips ffmpeg on an exact match, so the caller's own spelling is reported
    # even where the backend calls the codec something else (ogg/oga carry
    # opus; raw is pcm). Reporting the backend's name would re-mux every
    # reply, and ``-f pcm`` is not an ffmpeg demuxer at all.
    if fmt in ("ogg", "oga"):
        return (fmt, "opus") if "opus" in supported else ("mp3", "mp3")
    if fmt == "raw":
        return ("raw", "pcm") if "pcm" in supported else ("mp3", "mp3")
    if fmt in supported:
        return fmt, fmt
    return TTS_DEFAULT_RESPONSE_FORMAT, TTS_DEFAULT_RESPONSE_FORMAT


def _wants_instructions(provider_type: str, model_name: str) -> bool:
    """Only OpenAI's gpt-4o-mini-tts family accepts the instructions parameter."""
    return provider_type == "openai" and model_name.startswith(
        TTS_INSTRUCTIONS_MODEL_PREFIX
    )


def _ends_sentence(text: str) -> bool:
    """Return True when ``text`` ends with a sentence mark (quotes allowed)."""
    return text.rstrip().rstrip(_SENTENCE_CLOSERS).endswith(_SENTENCE_ENDINGS)


# Home Assistant converts streamed TTS audio for the satellite with ffmpeg fed
# through a pipe, and nothing comes out until 64 KiB have gone in (measured on
# ffmpeg 7 with HA's own arguments: 63 KiB of WAV stays silent with stdin
# open, 64 KiB is converted in 0.1 s; later pieces pass straight through).
# A short first sentence ("One moment.") would otherwise sit unheard until the
# reply's next sentences arrived -- and a Voice PE gives up on a response
# that sends nothing for 2 s. The first piece is padded with silence past it.
_CONVERTER_START_BYTES = 64 * 1024 + 4096
# The same converter holds back its last partial output frame until more
# input arrives: measured 51 ms of a 1587 ms sentence withheld while stdin
# stayed open. A first sentence long enough to need no padding (Kokoro's
# "One moment, please.") therefore had its tail ("...se") stuck there and
# played seconds later, right before the answer (#671, 15/15 runs). The
# first piece always ends with at least this much silence, so what is held
# back is silence.
_MIN_TRAILING_SILENCE_S = 0.25
# A Voice PE buffers only ~100 ms of speaker audio, and when a stream stalls
# (the model thinking between sentences) it runs dry and its speaker makes a
# brief click/"t" sound -- heard on streamed turns, never on one-shot speech
# of the same text. While a reply waits for text or synthesis, silence is
# fed paced to real time, keeping the stream this far ahead of playback;
# speech is delayed by at most that much.
TTS_STREAM_KEEPALIVE_S = 0.1
TTS_STREAM_LEAD_S = 0.3
_RIFF_HEADER_LEN = 12  # "RIFF", size, "WAVE"
_PCM_FMT_LEN = 16  # the fields every WAV fmt chunk carries
_UNSIGNED_SAMPLE_BITS = 8  # 8-bit WAV samples are unsigned; wider are signed


class _WavPiece(NamedTuple):
    """A synthesized WAV file split into its format and its samples."""

    fmt: bytes
    block_align: int
    bits: int
    samples: bytes


def _parse_wav_segment(data: bytes, pos: int) -> tuple[bytes, bytes, int]:
    """
    Parse one RIFF/WAVE file starting at ``pos``: ``(fmt, samples, end)``.

    A data size of 0 or 0xFFFFFFFF (a server that streams) or one past the
    end runs to the next RIFF header or the end of the response.
    """
    if (
        len(data) - pos < _RIFF_HEADER_LEN
        or data[pos : pos + 4] != b"RIFF"
        or data[pos + 8 : pos + _RIFF_HEADER_LEN] != b"WAVE"
    ):
        msg = "not a RIFF/WAVE file"
        raise ValueError(msg)
    fmt = b""
    chunk = pos + _RIFF_HEADER_LEN
    while chunk + 8 <= len(data):
        chunk_id = data[chunk : chunk + 4]
        size = struct.unpack_from("<I", data, chunk + 4)[0]
        body = chunk + 8
        if chunk_id == b"fmt ":
            fmt = data[body : body + size]
        elif chunk_id == b"data":
            if len(fmt) < _PCM_FMT_LEN:
                msg = "data chunk before a valid fmt chunk"
                raise ValueError(msg)
            if 0 < size <= len(data) - body:
                end = body + size
            else:
                following = data.find(b"RIFF", body)
                end = following if following != -1 else len(data)
            return fmt, data[body:end], end
        chunk = body + size + (size & 1)
    msg = "no data chunk"
    raise ValueError(msg)


def _parse_wav(data: bytes) -> _WavPiece:
    """
    Split a WAV response into its format and all of its samples.

    Some servers answer a multi-sentence request with one complete WAV file
    per sentence, back to back, each header sized for its own sentence
    (Speaches with piper does). Reading only the first file dropped every
    sentence after it, so a reply was cut off after its first sentence --
    a fraction of a second when that sentence was "Yes." or "Sure!". All
    segments are read; one in a different format is skipped.
    """
    fmt, samples, end = _parse_wav_segment(data, 0)
    parts = [samples]
    while (start := data.find(b"RIFF", end)) != -1:
        try:
            seg_fmt, seg_samples, end = _parse_wav_segment(data, start)
        except ValueError:
            break
        if seg_fmt == fmt:
            parts.append(seg_samples)
        else:
            LOGGER.warning("Skipping a WAV segment in a different format")
    # channels, sample rate, byte rate are not needed: the whole fmt chunk is
    # compared between pieces and copied into the header.
    block_align, bits = struct.unpack_from("<HH", fmt, 12)
    return _WavPiece(fmt, max(block_align, 1), bits, b"".join(parts))


def _single_wav(data: bytes) -> bytes:
    """Return a WAV response as one file, merging per-sentence segments."""
    piece = _parse_wav(data)
    header = _streaming_wav_header(piece.fmt)
    riff_size = len(header) - 8 + len(piece.samples)
    return (
        header[:4]
        + struct.pack("<I", riff_size)
        + header[8:-4]
        + struct.pack("<I", len(piece.samples))
        + piece.samples
    )


class _Pacer:
    """Track audio sent against wall time, and top up with silence."""

    def __init__(self, piece: _WavPiece, now: float) -> None:
        self._piece = piece
        self._byte_rate = max(struct.unpack_from("<I", piece.fmt, 8)[0], 1)
        self._start = now
        self._sent_s = 0.0

    @property
    def fmt(self) -> bytes:
        """Return the stream's WAV format chunk."""
        return self._piece.fmt

    def sent(self, audio: bytes) -> bytes:
        """Count ``audio`` as sent and return it."""
        self._sent_s += len(audio) / self._byte_rate
        return audio

    def top_up(self, now: float) -> bytes:
        """Return the silence that keeps the stream TTS_STREAM_LEAD_S ahead."""
        deficit = (now - self._start) + TTS_STREAM_LEAD_S - self._sent_s
        if deficit <= 0:
            return b""
        if deficit > TTS_STREAM_LEAD_S:
            # Everything sent has already played: the satellite ran dry for
            # about this long. Logged so a click can be matched to a stall.
            LOGGER.debug(
                "TTS stream fell behind playback by %.3f s", deficit - TTS_STREAM_LEAD_S
            )
        return self.sent(_silence(int(deficit * self._byte_rate), self._piece))


async def _keep_fed(
    task: asyncio.Future[Any], pacer: _Pacer | None
) -> AsyncGenerator[bytes]:
    """Yield paced silence until ``task`` is done (nothing before audio starts)."""
    loop = asyncio.get_running_loop()
    while not task.done():
        await asyncio.wait({task}, timeout=TTS_STREAM_KEEPALIVE_S)
        if not task.done() and pacer is not None and (gap := pacer.top_up(loop.time())):
            yield gap


def _streaming_wav_header(fmt: bytes) -> bytes:
    """Return a WAV header with unknown lengths, for a stream of samples."""
    unknown = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", unknown)
    )


def _silence(length: int, piece: _WavPiece) -> bytes:
    """Return ``length`` bytes of silence in ``piece``'s sample format."""
    length -= length % piece.block_align
    zero = b"\x80" if piece.bits == _UNSIGNED_SAMPLE_BITS else b"\x00"
    return zero * max(length, 0)


async def _pump_text(
    message_gen: AsyncIterable[str], chunks: asyncio.Queue[str | None]
) -> None:
    """
    Copy the pipeline's text stream into ``chunks``, then ``None``.

    A task of its own, so the splitter's idle timeout never cancels a read of
    the pipeline's generator. Stops after TTS_STREAM_TEXT_TIMEOUT_S without
    text: by then the turn has failed and the generator will never end.
    """
    iterator = aiter(message_gen)
    try:
        while True:
            try:
                async with asyncio.timeout(TTS_STREAM_TEXT_TIMEOUT_S):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            except TimeoutError:
                LOGGER.warning(
                    "TTS text stream sent nothing for %.0f s; ending the reply",
                    TTS_STREAM_TEXT_TIMEOUT_S,
                )
                break
            chunks.put_nowait(chunk)
    finally:
        chunks.put_nowait(None)


async def _split_sentences(
    chunks: asyncio.Queue[str | None], sentences: asyncio.Queue[str | None]
) -> None:
    """Turn text chunks into sentences, then ``None``; see TTS_STREAM_IDLE_FLUSH_S."""

    def _emit(texts: Iterable[str]) -> None:
        for text in texts:
            if text.strip():
                sentences.put_nowait(text.strip())

    # Text chunks are not on word or sentence boundaries.
    detector = SentenceBoundaryDetector()
    holding = False
    try:
        while True:
            try:
                async with asyncio.timeout(
                    TTS_STREAM_IDLE_FLUSH_S if holding else None
                ):
                    chunk = await chunks.get()
            except TimeoutError:
                holding = False
                # Read the held text without finish(), which strips it: a
                # fragment must go back untouched or the next chunk's first
                # word runs into its last one. sentence-stream is pinned.
                held = detector.current_sentence + detector.remaining_text
                if _ends_sentence(held):
                    _emit([detector.finish()])
                continue
            if chunk is None:
                break
            _emit(detector.add_chunk(chunk))
            holding = holding or bool(chunk.strip())
        _emit([detector.finish()])
    finally:
        sentences.put_nowait(None)


async def _finish_tasks(
    tasks: Iterable[asyncio.Task[None]], *, reraise: bool = True
) -> None:
    """
    Cancel unfinished tasks, wait for all, and re-raise the first failure.

    Every exception is retrieved, so none is logged as never retrieved. With
    ``reraise=False`` the caller is already propagating its own failure.
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if reraise:
        for result in results:
            if isinstance(result, Exception):
                raise result


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001
    entry: HGAConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities, one per TTS provider subentry."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_TTS_PROVIDER:
            continue
        # Bound to the subentry so the entity-registry entry is removed with it.
        async_add_entities(
            [HGATtsEntity(entry, subentry.subentry_id)],
            config_subentry_id=subentry.subentry_id,
        )


class HGATtsEntity(TextToSpeechEntity):
    """Text-to-speech entity backed by the OpenAI speech API or a local server."""

    _attr_has_entity_name = True
    _attr_default_language = DEFAULT_LANGUAGE

    def __init__(self, entry: ConfigEntry, subentry_id: str) -> None:
        """Initialize the TTS entity."""
        # ATTR_VOICE must be declared or the pipeline's voice option is
        # rejected; declaring ATTR_PREFERRED_FORMAT means the entity picks the
        # container itself, so HA only runs ffmpeg for sample-rate/channel
        # changes.
        self._attr_supported_options = [ATTR_VOICE, ATTR_PREFERRED_FORMAT]
        self.entry = entry
        self.subentry_id = subentry_id
        self._subentry = entry.subentries[subentry_id]
        self._attr_unique_id = f"{entry.entry_id}_{subentry_id}"
        self._attr_name = self._subentry.title or "TTS"
        self._clients = OpenAIClientCache(TTS_REQUEST_TIMEOUT_S)

    # ------------------------------------------------------------------ config

    @cached_property
    def supported_languages(self) -> list[str]:
        """
        Return the accepted language tags.

        The Assist pipeline matches loosely, but ``tts.speak`` and the media
        source check exact membership, so the bare primary subtags (``en``) and
        Home Assistant's own configured language are accepted alongside the
        region-qualified tags the models document. The models detect the
        language from the text; the tag only has to get past that check.
        Cached like the base class declares; it is first read after the entity
        is added, when ``hass`` is set.
        """
        languages = set(SUPPORTED_LANGUAGES)
        languages.update(tag.split("-", 1)[0] for tag in SUPPORTED_LANGUAGES)
        hass = getattr(self, "hass", None)
        hass_lang = getattr(getattr(hass, "config", None), "language", None)
        if isinstance(hass_lang, str) and hass_lang:
            languages.add(hass_lang)
            languages.add(hass_lang.split("-", 1)[0])
        return sorted(languages)

    @property
    def _provider_type(self) -> str:
        return str(self._subentry.data.get("provider_type") or "openai")

    def _model_settings(self) -> dict[str, Any]:
        return load_model_settings(self._subentry.data)

    def _configured_model(self) -> str:
        model = self._model_settings().get(CONF_TTS_MODEL_NAME)
        if isinstance(model, str) and model:
            return model
        return (
            RECOMMENDED_LOCAL_TTS_MODEL
            if self._provider_type == "local"
            else RECOMMENDED_OPENAI_TTS_MODEL
        )

    def _configured_voice(self) -> str:
        voice = self._model_settings().get(CONF_TTS_VOICE)
        if isinstance(voice, str) and voice:
            return voice
        return (
            RECOMMENDED_LOCAL_TTS_VOICE
            if self._provider_type == "local"
            else RECOMMENDED_OPENAI_TTS_VOICE
        )

    @cached_property
    def default_options(self) -> Mapping[str, Any]:
        """
        Return the options used when the caller does not specify any.

        Cached like the base class declares; a reconfigured subentry reloads the
        config entry and rebuilds the entity, so the cache never goes stale.
        """
        return {
            ATTR_VOICE: self._configured_voice(),
            ATTR_PREFERRED_FORMAT: TTS_DEFAULT_RESPONSE_FORMAT,
        }

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:  # noqa: ARG002
        """
        Return the selectable voices, configured default first.

        The Assist pipeline takes index 0 as its default voice. OpenAI's voices
        are a fixed set; a local server's cannot be listed through the OpenAI
        API, so only the configured voice is offered there (any other voice id
        can still be requested through the pipeline options).
        """
        configured = self._configured_voice()
        if self._provider_type != "openai":
            return [Voice(configured, configured)]
        voices = [Voice(name.lower(), name) for name in OPENAI_TTS_VOICES]
        voices.sort(key=lambda voice: voice.voice_id != configured)
        if voices[0].voice_id != configured:
            voices.insert(0, Voice(configured, configured))
        return voices

    # ------------------------------------------------------------ connection

    def _get_client(self, api_key: str, base_url: str | None) -> Any:
        """Return the cached OpenAI client for these credentials."""
        return self._clients.get(self.hass, api_key, base_url)

    # ------------------------------------------------------------- synthesis

    def _build_request(
        self, message: str, options: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(extension, kwargs)`` for ``client.audio.speech.create``."""
        provider_type = self._provider_type
        model_settings = self._model_settings()
        model_name = self._configured_model()
        supported = (
            TTS_LOCAL_RESPONSE_FORMATS
            if provider_type == "local"
            else TTS_OPENAI_RESPONSE_FORMATS
        )
        extension, response_format = _negotiate_format(
            options.get(ATTR_PREFERRED_FORMAT), supported
        )
        request: dict[str, Any] = {
            "model": model_name,
            "voice": options.get(ATTR_VOICE) or self._configured_voice(),
            "input": message,
            "response_format": response_format,
        }
        speed = model_settings.get(CONF_TTS_SPEED)
        if isinstance(speed, (int, float)) and float(speed) != TTS_SPEED_DEFAULT:
            request["speed"] = float(speed)
        instructions = model_settings.get(CONF_TTS_INSTRUCTIONS)
        if (
            isinstance(instructions, str)
            and instructions
            and _wants_instructions(provider_type, model_name)
        ):
            request["instructions"] = instructions
        return extension, request

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Synthesize ``message`` and return ``(extension, audio bytes)``."""
        _ = language  # the models detect the language from the text itself
        if not message.strip():
            msg = f"No text to synthesize for {self.entity_id}"
            raise HomeAssistantError(msg)
        return await self._synthesize(message, options)

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """
        Synthesize a text stream sentence by sentence.

        Overriding this is what makes the Assist pipeline hand us the reply
        while the conversation agent is still writing it, so a voice satellite
        starts speaking after the first sentence instead of after the whole
        turn. Home Assistant also routes plain ``tts.speak`` messages here once
        the entity streams; those arrive whole and take one request in the
        preferred format (TTS_ONE_SHOT_GRACE_S).

        The speech endpoint takes whole text, so for a live reply each batch
        of sentences is one request. They are sent as one WAV stream (a
        single header, then each batch's samples) whatever the preference;
        Home Assistant converts to the preferred format. See
        _CONVERTER_START_BYTES for why WAV and why the first piece is padded.
        """
        chunks: asyncio.Queue[str | None] = asyncio.Queue()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        tasks = (
            self.hass.async_create_background_task(
                _pump_text(request.message_gen, chunks),
                name=f"{self.entity_id} tts text read",
            ),
            self.hass.async_create_background_task(
                _split_sentences(chunks, sentences),
                name=f"{self.entity_id} tts sentence split",
            ),
        )
        done, _ = await asyncio.wait(tasks, timeout=TTS_ONE_SHOT_GRACE_S)
        if len(done) == len(tasks):
            await _finish_tasks(tasks)
            parts: list[str] = []
            while (sentence := sentences.get_nowait()) is not None:
                parts.append(sentence)
            if not (message := " ".join(parts).strip()):
                msg = f"No text to synthesize for {self.entity_id}"
                raise HomeAssistantError(msg)
            extension, audio = await self._synthesize(message, request.options)
            if extension == "wav":
                # One file per sentence would play only its first sentence.
                try:
                    audio = _single_wav(audio)
                except ValueError as err:
                    msg = f"TTS backend returned unreadable WAV for {self.entity_id}"
                    raise HomeAssistantError(msg) from err

            async def _whole() -> AsyncGenerator[bytes]:
                yield audio

            return TTSAudioResponse(extension, _whole())

        options = {**request.options, ATTR_PREFERRED_FORMAT: TTS_STREAM_RESPONSE_FORMAT}
        return TTSAudioResponse(
            TTS_STREAM_RESPONSE_FORMAT, self._stream_audio(tasks, sentences, options)
        )

    async def _stream_audio(
        self,
        tasks: tuple[asyncio.Task[None], asyncio.Task[None]],
        sentences: asyncio.Queue[str | None],
        options: Mapping[str, Any],
    ) -> AsyncGenerator[bytes]:
        """Yield audio per sentence batch as the text stream completes them."""
        spoke = False
        finished = False
        pacer: _Pacer | None = None
        pending: asyncio.Future[Any] | None = None
        loop = asyncio.get_running_loop()
        try:
            while not finished:
                batch: list[str] = []
                pending = asyncio.ensure_future(sentences.get())
                async for gap in _keep_fed(pending, pacer):
                    yield gap
                item = pending.result()
                # The first sentence goes alone so audio starts as early as
                # possible; after that, one request covers everything that
                # arrived while the previous batch was being synthesized.
                while item is not None:
                    batch.append(item)
                    if not spoke or sentences.empty():
                        break
                    item = sentences.get_nowait()
                finished = item is None
                text = " ".join(batch).strip()
                if not text:
                    continue
                pending = asyncio.ensure_future(self._synthesize(text, options))
                async for gap in _keep_fed(pending, pacer):
                    yield gap
                _extension, audio = pending.result()
                chunk, pacer = self._stream_bytes(audio, text, pacer, loop.time())
                if chunk is None:
                    continue
                yield chunk
                spoke = True
        except BaseException:
            # Keep the failure (or GeneratorExit) that is propagating.
            if pending is not None and not pending.done():
                pending.cancel()
            await _finish_tasks(tasks, reraise=False)
            raise
        # Surface a failure of the text stream itself.
        await _finish_tasks(tasks)
        if not spoke:
            msg = f"No text to synthesize for {self.entity_id}"
            raise HomeAssistantError(msg)

    def _stream_bytes(
        self, audio: bytes, text: str, pacer: _Pacer | None, now: float
    ) -> tuple[bytes | None, _Pacer | None]:
        """
        Turn one synthesized batch into stream bytes; None skips the batch.

        The first batch opens the stream: one open-ended WAV header, then its
        samples padded past _CONVERTER_START_BYTES, and it starts the pacer.
        """
        try:
            piece = _parse_wav(audio)
        except ValueError as err:
            msg = f"TTS backend returned unreadable WAV for {self.entity_id}"
            raise HomeAssistantError(msg) from err
        if pacer is None:
            pacer = _Pacer(piece, now)
            header = _streaming_wav_header(piece.fmt)
            byte_rate = struct.unpack_from("<I", piece.fmt, 8)[0]
            pad = max(
                _CONVERTER_START_BYTES - len(header) - len(piece.samples),
                int(_MIN_TRAILING_SILENCE_S * byte_rate),
            )
            return header + pacer.sent(piece.samples + _silence(pad, piece)), pacer
        if piece.fmt != pacer.fmt:
            # One voice keeps one format; a change mid-reply cannot be spliced
            # into the same stream, so that batch is skipped.
            LOGGER.warning(
                "TTS backend changed audio format mid-reply for %s; "
                "skipping %d characters",
                self.entity_id,
                len(text),
            )
            return None, pacer
        return pacer.sent(piece.samples), pacer

    async def _synthesize(
        self, message: str, options: Mapping[str, Any]
    ) -> tuple[str, bytes]:
        """Run one speech request and return ``(extension, audio bytes)``."""
        provider_type = self._provider_type
        if provider_type not in ("openai", "local"):
            msg = f"Unsupported TTS provider type {provider_type!r}"
            raise HomeAssistantError(msg)

        try:
            connection = resolve_openai_connection(
                self.entry,
                provider_type,
                self._subentry.data,
                provider_id_key=CONF_TTS_OPENAI_PROVIDER_ID,
            )
        except OpenAIConnectionError as err:
            msg = f"TTS {err} for {self.entity_id}"
            raise HomeAssistantError(msg) from err
        extension, request = self._build_request(message, options)
        connection.apply_to_request(request)

        try:
            client = self._get_client(connection.api_key, connection.base_url)
            response = await client.audio.speech.create(**request)
            audio = response.content
        except AuthenticationError as err:
            LOGGER.warning("TTS authentication failed for %s", self.entity_id)
            msg = f"TTS authentication failed for {self.entity_id}"
            raise HomeAssistantError(msg) from err
        except OpenAIError as err:
            LOGGER.warning("TTS request failed for %s: %s", self.entity_id, err)
            msg = f"TTS request failed for {self.entity_id}: {err}"
            raise HomeAssistantError(msg) from err

        if not audio:
            msg = f"TTS backend returned no audio for {self.entity_id}"
            raise HomeAssistantError(msg)
        return extension, audio
