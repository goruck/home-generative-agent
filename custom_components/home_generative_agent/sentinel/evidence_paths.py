"""
Canonical ``derived.*`` evidence-path handling for discovery (issue #524).

Single source of truth for resolving occupancy and night context from a
discovery candidate's ``evidence_paths`` plus its prose. The normalizer
(``proposal_templates``), the semantic dedup keys (``discovery_semantic``),
and the discovery engine all resolve through this module so their answers
cannot drift — the proposals card mirrors ``presence_signal`` /
``night_signal`` in JS (``www/hga-proposals-card.js``).

Structured evidence paths are the primary signal because candidate prose
may not be English forever (issue #524 groundwork for translating discovery
``title``/``summary``); the English regexes remain as a legacy fallback for
candidates generated before the CONTEXT REQUIREMENT prompt clause existed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

ANYONE_HOME_PATH: Final = "derived.anyone_home"
NOT_ANYONE_HOME_PATH: Final = "not derived.anyone_home"
IS_NIGHT_PATH: Final = "derived.is_night"
NOT_IS_NIGHT_PATH: Final = "not derived.is_night"

PresenceSignal = Literal["away", "home", "any"]

# Matches anyone_home == false / anyone_home = 0 / anyone_home=False etc.
# Machine syntax: checked before the prose term patterns so a candidate
# carrying both an anyone_home expression and contrary direction words
# resolves from the unambiguous expression, not the prose.
ANYONE_HOME_FALSE_PATTERN = re.compile(
    r"anyone_home\s*(?:==?\s*|\bis\s+)(?:false|0)\b", re.IGNORECASE
)
ANYONE_HOME_TRUE_PATTERN = re.compile(
    r"anyone_home\s*(?:==?\s*|\bis\s+)(?:true|1)\b", re.IGNORECASE
)
# The negated derived path spelled inside pattern/prose text rather than as
# its own evidence path ("is_night AND not derived.anyone_home") — machine
# syntax like the expressions above, language-independent, same tier.
NOT_ANYONE_HOME_TEXT_PATTERN = re.compile(r"(?:\bnot\s+|!)\s*derived\.anyone_home")
# Away/home occupancy wording. Word-bounded so "present" doesn't match
# "presence" (an availability candidate about presence sensors is not a
# someone-is-home condition) and "home" doesn't match "anyone_home"/
# "armed_home" (issue #514 adversarial review). "nobody/no one (is) (at)
# home" is matched with optional filler words — bare "nobody home" missed
# the flagship phrasing "nobody is home" (issue #516 adversarial review).
# One compiled pattern per direction, shared by the normalizer and the
# semantic keys: an asymmetric pair would key a candidate home=1 while its
# activated rule keys home=0, breaking dedup in both directions (issue #518
# Codex structured review, empirically reproduced).
AWAY_TERMS_PATTERN = re.compile(
    r"\b(?:away|no(?:body|\s+one)\s+(?:is\s+)?(?:at\s+)?home|empty|unoccupied"
    r"|no occupants|without occupants)\b"
)
HOME_TERMS_PATTERN = re.compile(
    r"\b(?:someone home|occupied|home|present|occupants|residents)\b"
)

# Union of Python's and JavaScript's \s classes so the card mirror
# canonicalizes identically: JS \s includes U+FEFF (Python's does not);
# Python \s includes U+0085 and U+001C-U+001F (JS's does not). The JS
# mirror normalizes the same union.
_WHITESPACE_RE = re.compile("[\\s\ufeff]+")
# Quote characters the LLM sometimes wraps around a whole path — the
# discovery prompt itself renders the negated form inside single quotes
# ("the exact string 'not derived.anyone_home'"), so wrapped spellings are
# an expected output mode (cf. _EVIDENCE_QUOTE_CHARS in the entity-ID
# parsers).
_QUOTE_CHARS = "'\"`"
_NOT_PREFIX_RE = re.compile(r"^(?:not\s+|!\s*)")
# Trailing boolean comparisons the LLM appends to a derived path instead of
# using the canonical negated form ("derived.anyone_home == false"). Values
# are optionally quoted (LLM output variance) and include the HA state
# idiom (off/no, on/yes) alongside JSON booleans — the snapshot vocabulary
# the model reads is dominated by on/off entity states.
_FALSE_SUFFIX_RE = re.compile(r"\s*(?:==?\s*|\bis\s+)['\"`]?(?:false|0|off|no)['\"`]?$")
_TRUE_SUFFIX_RE = re.compile(r"\s*(?:==?\s*|\bis\s+)['\"`]?(?:true|1|on|yes)['\"`]?$")
# Bare derived-key spellings the LLM emits without the "derived." prefix —
# unambiguous (no HA domain is named "anyone_home"/"is_night"), so they
# alias to the canonical paths.
_DERIVED_ALIASES: Final = {
    "anyone_home": ANYONE_HOME_PATH,
    "is_night": IS_NIGHT_PATH,
}


def canonicalize_evidence_path(path: str) -> str:
    """
    Normalize an evidence path for membership checks.

    Lowercases, collapses whitespace, rewrites ``!x`` to ``not x``, and
    folds a trailing boolean comparison into the negation: ``x == false`` /
    ``x = 0`` negates, ``x == true`` / ``x = 1`` is the bare path, and a
    double negation (``not x == false``) resolves positive. Consumers were
    previously literal string membership checks, so any spelling variant
    silently fell through to the English-prose fallback (issue #524).
    """
    canonical = _WHITESPACE_RE.sub(" ", path.lower()).strip()
    canonical = canonical.strip(_QUOTE_CHARS).strip()
    negated = False
    # Loop so stacked prefixes fold by parity ("!!x", "not not x").
    while prefix := _NOT_PREFIX_RE.match(canonical):
        negated = not negated
        # Re-strip quotes: inner-quoted variants ("not 'derived.x'") keep a
        # wrapping quote after the prefix is removed and would miss every
        # structured tier (adversarial review).
        canonical = canonical[prefix.end() :].strip(_QUOTE_CHARS).strip()
    if _FALSE_SUFFIX_RE.search(canonical):
        negated = not negated
        canonical = _FALSE_SUFFIX_RE.sub("", canonical)
    else:
        canonical = _TRUE_SUFFIX_RE.sub("", canonical)
    canonical = canonical.strip(_QUOTE_CHARS).strip()
    canonical = _DERIVED_ALIASES.get(canonical, canonical)
    return f"not {canonical}" if negated else canonical


def canonical_evidence_paths(evidence_paths: object) -> frozenset[str]:
    """Canonicalized path set; tolerates malformed LLM output shapes."""
    if not isinstance(evidence_paths, (list, tuple, set, frozenset)):
        return frozenset()
    paths = cast("Iterable[object]", evidence_paths)
    return frozenset(
        canonicalize_evidence_path(path) for path in paths if isinstance(path, str)
    )


def has_derived_path(evidence_paths: object, canonical_path: str) -> bool:
    """Return True when the canonicalized paths contain ``canonical_path``."""
    return canonical_path in canonical_evidence_paths(evidence_paths)


def is_derived_path(path: str) -> bool:
    """Return True for ``derived.*`` paths, including negated spellings."""
    return canonicalize_evidence_path(path).removeprefix("not ").startswith("derived.")


def presence_signal(  # noqa: PLR0911 — one return per priority tier
    evidence_paths: object, text: str
) -> PresenceSignal:
    """
    Resolve occupancy direction from structured evidence, then prose.

    ``text`` must be pre-lowercased by the caller (every caller builds the
    blob with ``.lower()``): the term tiers are case-sensitive by design,
    matching the historical patterns, while the expression tiers are
    IGNORECASE.

    Priority order (mirrored by _presenceSignal in hga-proposals-card.js):
    1. ``not derived.anyone_home`` path → "away" — the LLM's canonical,
       unambiguous absence-of-occupancy assertion.
    2. ``anyone_home ==`` boolean expressions or a negated-path mention in
       the text — machine syntax, language-independent; outranks prose term
       matching so mixed signals resolve from the unambiguous expression.
    3. English away/home term regexes — legacy prose fallback.
    4. Bare ``derived.anyone_home`` path → "home". Citing the path does not
       assert it is true — the LLM historically cites it while the prose
       says "while nobody is home" — so it ranks BELOW the away signals in
       tiers 1-3 to avoid inverting such candidates. When the prose carries
       no direction at all (non-English text, issue #524), it is the only
       signal left and reads as presence-required, matching how
       discovery_semantic keys it and how the discovery prompt's CONTEXT
       REQUIREMENT instructs the model to use it.
    5. Otherwise "any" — occupancy direction unknown.
    """
    paths = canonical_evidence_paths(evidence_paths)
    if NOT_ANYONE_HOME_PATH in paths:
        return "away"
    if ANYONE_HOME_FALSE_PATTERN.search(text) or NOT_ANYONE_HOME_TEXT_PATTERN.search(
        text
    ):
        return "away"
    if ANYONE_HOME_TRUE_PATTERN.search(text):
        return "home"
    if AWAY_TERMS_PATTERN.search(text):
        return "away"
    if HOME_TERMS_PATTERN.search(text):
        return "home"
    if ANYONE_HOME_PATH in paths:
        return "home"
    return "any"


def night_signal(evidence_paths: object, text: str) -> bool:
    """
    Return True when the candidate carries a nighttime signal.

    Structured ``derived.is_night`` evidence first; an explicit negated
    path ("derived.is_night == false") blocks the "night" substring
    fallback, which would otherwise fire on the very text that negates it
    (adversarial review). The substring fallback also covers "nighttime"/
    "overnight". ``text`` must be pre-lowercased by the caller (same
    contract as ``presence_signal``).

    Known limitation (Codex P2, deliberate): a negated path returns the
    same False as no night condition, so a daytime-only candidate
    normalizes to an all-hours template — no daytime-only template exists,
    and broadening (a superset of the stated window, visible at approval)
    beats the pre-#524 behavior of inverting to a night rule or dropping
    the candidate.
    """
    paths = canonical_evidence_paths(evidence_paths)
    if IS_NIGHT_PATH in paths:
        return True
    if NOT_IS_NIGHT_PATH in paths:
        return False
    return "night" in text


# ---------------------------------------------------------------------------
# Network section paths (network security plan, step 10)
# ---------------------------------------------------------------------------
# ``network.clients[key=3fa2c1b0].connected`` cites a router client by its
# pseudonymized key; ``network.posture.upnp_enabled`` cites a router setting.
# The normalizer and the semantic keys resolve both through here so a
# candidate and its activated rule always agree.
NETWORK_CLIENT_PATH_RE: Final = re.compile(
    r"network\.clients\[\s*(?:key\s*=\s*)?['\"`]*([0-9a-f]{8,16})['\"`]*\s*\]"
)
NETWORK_POSTURE_PATH_RE: Final = re.compile(r"network\.posture\.([a-z0-9_]+)")
# The settings a rule may watch, each with the value worth an alert when the
# prose does not say: a setting that opens the network (UPnP, guest Wi-Fi,
# dynamic DNS, IPv6) turning on, a protection (WPA3, threat blocking, ad
# blocking) turning off.
NETWORK_POSTURE_ALERT_VALUE: Final[dict[str, bool]] = {
    "upnp_enabled": True,
    "guest_network_enabled": True,
    "ddns_enabled": True,
    "ipv6_enabled": True,
    "wpa3_enabled": False,
    "malware_blocking_enabled": False,
    "ad_blocking_enabled": False,
}
_NETWORK_ABSENT_TERMS: Final = (
    "absent",
    "missing",
    "not connected",
    "disconnected",
    "offline",
    "gone",
    "not on the network",
    "left the network",
    "drops off",
    "dropped off",
)
_NETWORK_OFF_RE: Final = re.compile(
    r"\b(?:turned off|switched off|disabled|is off|off)\b"
)
_NETWORK_ON_RE: Final = re.compile(
    r"\b(?:turned on|switched on|enabled|is on|active|on)\b"
)


def network_client_keys(evidence_paths: object) -> list[str]:
    """Return the client keys the evidence paths cite, in order, deduplicated."""
    keys: list[str] = []
    if not isinstance(evidence_paths, list):
        return keys
    for path in evidence_paths:
        if not isinstance(path, str):
            continue
        for match in NETWORK_CLIENT_PATH_RE.finditer(path.lower()):
            if match.group(1) not in keys:
                keys.append(match.group(1))
    return keys


def network_posture_keys(evidence_paths: object) -> list[str]:
    """Return the watched router settings the evidence paths cite."""
    keys: list[str] = []
    if not isinstance(evidence_paths, list):
        return keys
    for path in evidence_paths:
        if not isinstance(path, str):
            continue
        for match in NETWORK_POSTURE_PATH_RE.finditer(path.lower()):
            key = match.group(1)
            if key in NETWORK_POSTURE_ALERT_VALUE and key not in keys:
                keys.append(key)
    return keys


def network_absent_signal(text: str) -> bool:
    """Return True when the prose describes a device that is missing."""
    lowered = text.lower()
    return any(term in lowered for term in _NETWORK_ABSENT_TERMS)


def network_posture_expected(key: str, text: str) -> bool:
    """Return the setting value the candidate wants an alert on."""
    lowered = text.lower()
    off = bool(_NETWORK_OFF_RE.search(lowered))
    on = bool(_NETWORK_ON_RE.search(lowered))
    if off and not on:
        return False
    if on and not off:
        return True
    return NETWORK_POSTURE_ALERT_VALUE[key]
