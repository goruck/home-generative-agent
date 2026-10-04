# ruff: noqa: S101
"""
Checks for the in-card translation table of ``www/hga-proposals-card.js``.

The proposals card carries its own ``en``/``cs`` dictionary (PR #697); it is
a translation surface separate from ``translations/*.json`` and has no JS
test harness, so these tests read the module as text and, where ``node`` is
available, execute its helpers with stub browser globals.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_CARD_JS = (
    Path(__file__).resolve().parents[3]
    / "custom_components"
    / "home_generative_agent"
    / "www"
    / "hga-proposals-card.js"
)
_PLACEHOLDER_RE = re.compile(r"\{[a-z]+\}")
# ``key: "value",`` with the value optionally on its own line (long strings).
_ENTRY_RE = re.compile(r'^\s+([a-z_]+):\s*\n?\s*"((?:[^"\\]|\\.)*)",', re.MULTILINE)
_LANG_RE = re.compile(r"^  ([a-z]{2}): \{\n(.*?)^  \},", re.MULTILINE | re.DOTALL)

# Labels the card writes into innerHTML without escaping (static text only).
_INNER_HTML_KEYS = {
    "pipeline_title",
    "refresh",
    "sec_discovery",
    "sec_filtered",
    "sec_pending",
    "sec_history",
    "load_fail_discovery",
    "load_fail_filtered",
    "load_fail_pending",
    "load_fail_history",
    "no_discovery",
    "no_filtered",
    "no_pending",
    "no_history",
    "candidate_id",
    "type",
    "unspecified",
    "status",
    "rule_id",
    "covered_rule",
    "rule_state",
    "reason",
    "semantic_key",
    "unsupported_warn",
    "template_recorded",
    "active",
    "inactive",
}

_NODE_STUB = r"""
const fs = require("node:fs");
const src = fs.readFileSync(process.argv[1], "utf8");
globalThis.window = globalThis;
globalThis.HTMLElement = class {};
globalThis.customElements = { define() {}, get() { return undefined; } };
const { Card, I18N } = new Function(
  src + "\n; return { Card: HgaProposalsCard, I18N: HGA_PROPOSALS_I18N };"
)();
const card = Object.create(Card.prototype);
card._hass = JSON.parse(process.argv[2]);
const calls = JSON.parse(process.argv[3]);
const out = { lang: card._lang(), keys: Object.keys(I18N), results: [] };
for (const [method, args] of calls) {
  out.results.push(card[method](...args));
}
process.stdout.write(JSON.stringify(out));
"""


def _table() -> dict[str, dict[str, str]]:
    src = _CARD_JS.read_text(encoding="utf-8")
    start = src.index("const HGA_PROPOSALS_I18N = {")
    end = src.index("\n};", start)
    table = {
        lang: dict(_ENTRY_RE.findall(body))
        for lang, body in _LANG_RE.findall(src[start:end])
    }
    assert set(table) >= {"en", "cs"}, sorted(table)
    assert len(table["en"]) > 50, "regex extracted too few entries"
    return table


def _run_card(hass: object, calls: list[tuple[str, list[object]]]) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run(  # noqa: S603
        [node, "-e", _NODE_STUB, str(_CARD_JS), json.dumps(hass), json.dumps(calls)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("lang", ["cs"])
def test_card_i18n_keys_and_placeholders_match_en(lang: str) -> None:
    """Every language mirrors ``en`` key for key, with the same placeholders."""
    table = _table()
    en, other = table["en"], table[lang]
    assert set(en) == set(other), sorted(set(en) ^ set(other))
    for key, text in other.items():
        assert sorted(_PLACEHOLDER_RE.findall(text)) == sorted(
            _PLACEHOLDER_RE.findall(en[key])
        ), key
        assert text.strip(), key


def test_card_i18n_inner_html_labels_carry_no_markup() -> None:
    """Static labels are interpolated into innerHTML unescaped."""
    for lang, entries in _table().items():
        for key in _INNER_HTML_KEYS:
            assert key in entries, (lang, key)
            assert not re.search(r"[<>&{}]", entries[key]), (lang, key)


def test_card_t_inserts_parameter_values_literally() -> None:
    """
    ``_t`` must not treat ``$`` in a value as a replacement pattern.

    ``String.prototype.replaceAll`` with a string replacement expands ``$&``,
    ``$$``, ``$'`` and ``$```; candidate ids and error messages are untrusted
    text, and the pre-#697 template literals inserted them verbatim.
    """
    out = _run_card(
        {"language": "en"},
        [
            ("_t", ["promoting", {"id": "sensor_$&"}]),
            ("_t", ["promoting", {"id": "cand$'x"}]),
            ("_t", ["refresh_failed", {"msg": "$& boom"}]),
            ("_t", ["promote_result", {"r": "ok$$"}]),
        ],
    )
    assert out["results"] == [
        "Promoting sensor_$&...",
        "Promoting cand$'x...",
        "Refresh failed: $& boom",
        "Promote result: ok$$",
    ]


def test_card_language_resolution_and_fallbacks() -> None:
    """hass.language wins, then hass.locale.language, then English."""
    assert _run_card(None, [])["lang"] == "en"
    assert _run_card({}, [])["lang"] == "en"
    assert _run_card({"language": "cs-CZ"}, [])["lang"] == "cs"
    assert _run_card({"language": "de"}, [])["lang"] == "en"
    assert _run_card({"locale": {"language": "cs"}}, [])["lang"] == "cs"
    out = _run_card(
        {"language": "cs"},
        [
            ("_statusLabel", ["approved"]),
            ("_statusLabel", ["weird_new"]),
            ("_dedupeReasonLabel", [None]),
            ("_dedupeReasonLabel", ["not_a_real_reason"]),
            ("_dedupeReasonLabel", ["cumulative_energy_sensor"]),
            ("_statusLabel", ["already_active"]),
        ],
    )
    assert out["results"] == [
        "schváleno",
        "weird_new",
        "Neznámé",
        "not_a_real_reason",
        "Kumulativní senzor energie (celkem kWh), nemůže se stát pravidlem",
        "už aktivní",
    ]
