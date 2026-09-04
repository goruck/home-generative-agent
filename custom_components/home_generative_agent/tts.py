"""Text-to-speech platform for Home Generative Agent (OpenAI or a local server)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from openai import AuthenticationError, OpenAIError
from propcache.api import cached_property

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
)
from .core.openai_endpoint import (
    OpenAIClientCache,
    OpenAIConnectionError,
    load_model_settings,
    resolve_openai_connection,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .core.runtime import HGAConfigEntry

LOGGER = logging.getLogger(__name__)

# Pinned so the effective timeout never depends on what HA's shared httpx
# client happens to carry. A spoken reply is a few sentences; a minute covers a
# cold model load on a local server with room to spare.
TTS_REQUEST_TIMEOUT_S = 60.0

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
