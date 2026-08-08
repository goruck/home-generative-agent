"""Translate Sentinel's fixed notification title/subtitle template strings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from custom_components.home_generative_agent.core.utils import (
    SENTINEL_ADMISSION_TIMEOUT_S,
    SentinelLLMDeferredError,
    extract_final,
    run_sentinel_model_call,
)

LOGGER = logging.getLogger(__name__)

_TRANSLATE_TIMEOUT_S = 30.0

# Keys consumed by sentinel/notifier.py. Values are the English originals,
# which also serve as the fallback when translation is unavailable or fails.
TITLE_KEYS: dict[str, str] = {
    "title_high": "Security Alert",
    "title_medium": "Home Alert",
    "title_low": "Home Update",
}

SUBTITLE_KEYS: dict[str, str] = {
    "subtitle_appliance_finished": "{appliance} finished",
    "subtitle_appliance_cycle_complete": "Appliance cycle complete",
    "subtitle_entry_open_disarmed": "{entry_name} open, alarm disarmed",
    "subtitle_power_deviation": "{appliance}: power {direction_word} than expected",
    "direction_lower": "lower",
    "direction_higher": "higher",
}

# Curated display labels for known finding types (notifier.py's
# _display_type() fallback). Fixed and known ahead of time, so -- unlike
# dynamic candidate/rule labels -- they can ride along in the single batch
# translation call rather than needing a per-string runtime translation.
KNOWN_TYPE_LABELS: dict[str, str] = {
    "open_entry_while_away": "Open entry while away",
    "open_entry_at_night_when_home": "Open entry at night",
    "open_entry_at_night_when_home_window": "Open entry at night",
    "open_entry_at_night_while_away": "Open entry at night",
    "open_entry_at_night": "Open entry at night",
    "open_entry_at_night_window": "Open entry at night",
    "open_entry_at_night_door": "Open entry at night",
    "open_entry_at_night_entry": "Open entry at night",
    "open_any_window_at_night_while_away": "Window open at night",
    "motion_detected_at_night_while_away": "Motion at night while away",
    "motion_detected_while_away": "Motion while away",
    "unlocked_lock_at_night": "Door lock left unlocked",
    "camera_entry_unsecured": "Activity near unsecured entry",
    "alarm_disarmed_during_external_threat": "Outdoor activity while alarm disarmed",
}

# type-slug -> ENGLISH_TEMPLATES key, so notifier.py can look up the
# (possibly-translated) label for a known type without re-deriving keys.
TYPE_LABEL_KEYS: dict[str, str] = {slug: f"type_{slug}" for slug in KNOWN_TYPE_LABELS}

TYPE_KEYS: dict[str, str] = {
    TYPE_LABEL_KEYS[slug]: label for slug, label in KNOWN_TYPE_LABELS.items()
}

ENGLISH_TEMPLATES: dict[str, str] = {**TITLE_KEYS, **SUBTITLE_KEYS, **TYPE_KEYS}

_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")

_SYSTEM_PROMPT = (
    "You translate a small, fixed set of UI strings for a home security "
    "app into {language}. You will receive a JSON object mapping string "
    "keys to English template text. Some values contain placeholders "
    "wrapped in curly braces, e.g. {{appliance}} or {{direction_word}} -- "
    "these are substituted with real values at runtime and MUST be copied "
    "through byte-for-byte, unchanged, in whatever position they'd "
    "naturally fall in the translated sentence. Do not translate, "
    "decline, or otherwise alter the placeholder names themselves. If "
    "{language} grammar would normally require a word to agree in "
    "gender/case/number with a placeholder's eventual value, phrase the "
    "sentence in a form that avoids that agreement -- the placeholder's "
    "value is not available to you at translation time. "
    "Reply with ONLY a JSON object using the exact same keys as the "
    "input, no other text, no markdown code fences."
)


class TemplateTranslator:
    """
    Translate Sentinel's fixed notification title/subtitle templates.

    The full set of strings (``ENGLISH_TEMPLATES``) is small and fixed, so
    translation happens once per configured ``response_language`` in a
    single LLM call, and the result is cached in memory for the lifetime of
    the notifier -- no per-notification latency or cost.

    Placeholders such as ``{appliance}`` are preserved by the translation so
    callers can ``.format()`` the returned templates exactly as they do the
    English originals. If translation is unavailable, malformed, or drops a
    placeholder, the English original is used for the affected key(s) --
    this class never raises and never returns a template that would break
    ``.format()``.
    """

    def __init__(
        self,
        model: Any,
        *,
        deployment: str = "edge",
        health_stats: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the translator with an optional LLM model."""
        self._model = model
        self._deployment = deployment
        self._health_stats = health_stats
        self._cache: dict[str, dict[str, str]] = {}
        self._failed: set[str] = set()
        # Per-string cache for dynamic labels that aren't part of the fixed
        # ENGLISH_TEMPLATES set (e.g. candidate/rule display labels that are
        # slugified and generated at runtime by discovery -- unbounded, so
        # they can't ride along in the single async_get() batch call).
        self._label_cache: dict[tuple[str, str], str] = {}

    async def async_get(self, language: str) -> dict[str, str]:
        """Return the (possibly-translated) template map for *language*."""
        if not language or self._model is None:
            return ENGLISH_TEMPLATES
        if language in self._cache:
            return self._cache[language]
        if language in self._failed:
            return ENGLISH_TEMPLATES

        try:
            translated = await self._async_translate(language)
        except SentinelLLMDeferredError as err:
            # Transient: the model call was deferred by admission control
            # (e.g. system busy at HA startup). Don't poison the cache --
            # retry on the next finding instead of staying English forever.
            LOGGER.debug(
                "Sentinel title/subtitle translation to %s deferred: %s",
                language,
                err,
            )
            return ENGLISH_TEMPLATES
        except TimeoutError:
            # Transient: also retry next time rather than caching failure.
            LOGGER.warning(
                "Sentinel title/subtitle translation to %s timed out after %.0fs.",
                language,
                _TRANSLATE_TIMEOUT_S,
            )
            return ENGLISH_TEMPLATES
        except Exception:
            LOGGER.exception(
                "Unexpected error translating Sentinel templates to %s.", language
            )
            self._failed.add(language)
            return ENGLISH_TEMPLATES

        if translated is None:
            self._failed.add(language)
            return ENGLISH_TEMPLATES

        self._cache[language] = translated
        return translated

    async def async_translate_label(self, text: str, language: str) -> str:
        """
        Translate a single short, dynamic label into *language*.

        Unlike ``async_get``, this is for content that isn't known ahead of
        time -- e.g. a discovery-generated candidate/rule's display label --
        so it can't be pre-translated as part of ``ENGLISH_TEMPLATES``.
        Results are cached per ``(language, text)`` so a given label is only
        translated once. Never raises; falls back to *text* unchanged on any
        error, timeout, deferral, or empty/missing model output.
        """
        if not text or not language or self._model is None:
            return text
        cache_key = (language, text)
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        try:
            result = await run_sentinel_model_call(
                self._model,
                [
                    SystemMessage(
                        content=(
                            f"Translate the following short UI label into "
                            f"{language}. Reply with ONLY the translated "
                            "label -- no quotes, no punctuation added, no "
                            "other text."
                        )
                    ),
                    HumanMessage(content=text),
                ],
                deployment=self._deployment,
                category="translate_label",
                admission_timeout_s=SENTINEL_ADMISSION_TIMEOUT_S,
                call_timeout_s=_TRANSLATE_TIMEOUT_S,
                health_stats=self._health_stats,
            )
        except SentinelLLMDeferredError as err:
            LOGGER.debug(
                "Sentinel label translation to %s deferred: %s", language, err
            )
            return text
        except TimeoutError:
            LOGGER.warning(
                "Sentinel label translation to %s timed out after %.0fs.",
                language,
                _TRANSLATE_TIMEOUT_S,
            )
            return text
        except Exception:
            LOGGER.exception(
                "Unexpected error translating label to %s.", language
            )
            return text

        content = getattr(result, "content", None)
        translated = extract_final(content).strip() if content else ""
        if not translated:
            return text
        self._label_cache[cache_key] = translated
        return translated

    async def _async_translate(self, language: str) -> dict[str, str] | None:
        system_prompt = _SYSTEM_PROMPT.format(language=language)
        user_prompt = json.dumps(ENGLISH_TEMPLATES, ensure_ascii=False)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = await run_sentinel_model_call(
            self._model,
            messages,
            deployment=self._deployment,
            category="translate_templates",
            admission_timeout_s=SENTINEL_ADMISSION_TIMEOUT_S,
            call_timeout_s=_TRANSLATE_TIMEOUT_S,
            health_stats=self._health_stats,
        )
        content = getattr(result, "content", None)
        if not content:
            return None
        text = extract_final(content)
        if not text:
            return None
        return _validate(text, language)


def _validate(raw_text: str, language: str) -> dict[str, str] | None:
    """Parse the model's JSON reply, falling back to English per bad key."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        LOGGER.warning(
            "Sentinel template translation to %s returned invalid JSON.", language
        )
        return None
    if not isinstance(parsed, dict):
        LOGGER.warning(
            "Sentinel template translation to %s returned a non-object.", language
        )
        return None

    result: dict[str, str] = {}
    for key, english in ENGLISH_TEMPLATES.items():
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            LOGGER.warning(
                "Sentinel template translation to %s missing key %r; "
                "using English for it.",
                language,
                key,
            )
            result[key] = english
            continue
        if not _placeholders_match(english, value):
            LOGGER.warning(
                "Sentinel template translation to %s for key %r dropped or "
                "altered a placeholder; using English for it.",
                language,
                key,
            )
            result[key] = english
            continue
        result[key] = value
    return result


def _placeholders_match(english: str, translated: str) -> bool:
    """Return True if *translated* contains exactly the same {placeholders}."""
    return set(_PLACEHOLDER_RE.findall(english)) == set(
        _PLACEHOLDER_RE.findall(translated)
    )
