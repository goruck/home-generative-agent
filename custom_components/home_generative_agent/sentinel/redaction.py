"""
Redaction of network identifiers before any model call.

The pseudonymizer (``sentinel/pseudonymizer.py``) tokenizes addresses on the
way *into* the snapshot; this module is the matching gate on the way *out*,
applied to whatever is about to be rendered into a prompt: a finding's
evidence and the rendered explain prompt, the triage prompt, the reduced
discovery snapshot and its rendered prompt, and the ``audit_home_security``
tool payload. Everything those calls show a model passes through
:func:`redact_network_identifiers`, so the invariant "no MAC, IP, or client
hostname reaches a model" holds regardless of which adapter or rule produced
the data (docs/network-security-plan.md, *Privacy of the Audit*).

Two layers, because identifiers arrive in two shapes:

- **Structural.** Dict keys that name an address (``mac``, ``ip``, ``ieee``,
  ...) are dropped wherever they occur. A ``hostname`` is replaced by the
  client's display name (manufacturer plus pseudonymized key) when the dict
  carries a ``key``, and dropped otherwise: an arbitrary hostname cannot be
  recognised by pattern, so it is only ever removed by position.
- **Textual.** Every string, including dict keys and list members, has
  embedded MAC addresses replaced by ``[mac]`` (separated or bare, six or
  eight octets, so an mDNS name such as ``shellyplug-s-3494547A1B2C`` loses
  its address), IP addresses by ``[lan ip]`` or ``[public ip]`` (the subnet
  class is kept because "a port forward to a LAN host" and "a token used from
  a public address" read differently, the address itself does not matter to
  the model), and names under local-network suffixes (``nas.local``,
  ``printer.home.arpa``) by ``[hostname]``. This is what catches an address a
  rule folded into its ``summary`` line or a title advertised on the LAN.

What it deliberately leaves: names advertised on the LAN that carry no
address (discovery titles, gateway names) still reach the model, labelled as
data, per the audit tool's earlier decision (TODOS.md, *Device and add-on
names reach the model verbatim*); and entity ids keep a bare hex run bounded
by underscores, because the model must be able to hand them back to tools.

The walk is total and cannot raise on ordinary data: unknown objects become
their redacted ``str()``, tuple subclasses come back as plain tuples, a
reference cycle or a nesting deeper than :data:`MAX_DEPTH` is cut with a
marker, and a container reached twice is redacted once. New containers are
built throughout and the input is never mutated; a finding's evidence feeds
anomaly-id derivation elsewhere and must stay intact.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any, Final

from custom_components.home_generative_agent.snapshot.network import sanitize_label

MAC_REDACTED: Final = "[mac]"
LAN_IP_REDACTED: Final = "[lan ip]"
PUBLIC_IP_REDACTED: Final = "[public ip]"
HOSTNAME_REDACTED: Final = "[hostname]"
# Emitted in place of a container the walk cannot follow.
CYCLE_MARKER: Final = "[cycle]"
DEPTH_MARKER: Final = "[nested too deep]"
MAX_DEPTH: Final = 64

# Keys whose values are an address by definition. Compared case-insensitively
# against the dict key; the value is dropped whatever its type (``None``
# included, so a missing MAC is not reported as "mac: None").
_DROP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "mac",
        "mac_address",
        "ip",
        "ip_address",
        "ipv4",
        "ipv4_address",
        "ipv6",
        "ipv6_address",
        "ieee",
        "address",
        "bluetooth_address",
        "ble_address",
    }
)
# Keys carrying a DHCP/mDNS hostname: replaced by the display name when the
# dict is a client (has a ``key``), dropped otherwise.
_HOSTNAME_KEYS: Final[frozenset[str]] = frozenset({"hostname", "host_name"})
_CLIENT_KEY: Final = "key"

# A chain of six or more colon- or dash-separated octets (MAC, Bluetooth,
# EUI-64 Zigbee IEEE; a longer chain is nothing legitimate and goes too, so
# ``MAC:aa:bb:cc:dd:ee:ff`` cannot hide its address behind a label), or
# Cisco's three dotted quads. Bounded by hex digits only, so punctuation on
# either side (a sentence's period, a colon after a label) does not shelter
# the address, and a longer hex string is not cut in the middle.
_MAC_RE: Final = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"(?:[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5,}"
    r"|[0-9A-Fa-f]{4}(?:\.[0-9A-Fa-f]{4}){2})"
    r"(?![0-9A-Fa-f])"
)
# A bare run of exactly 12 or 16 hex digits with at least one digit and one
# letter: how mDNS/SSDP names embed the address (``Sonos-7828CA123456``,
# Hue's ``001788FFFE123456``). Bounded by non-word characters, so a run
# inside an entity id (``sensor.shellyplug_s_3494547a1b2c_power``) is left
# for the model to use; an all-digit run could be a serial or a phone number
# and is not claimed.
_BARE_MAC_RE: Final = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[0-9A-Fa-f]*\d)(?=[0-9A-Fa-f]*[A-Fa-f])"
    r"[0-9A-Fa-f]{12}(?:[0-9A-Fa-f]{4})?"
    r"(?![A-Za-z0-9_])"
)
# Four dotted decimal groups; ``ipaddress`` then rejects anything out of
# range (``999.1.1.1``) so it is left as written. The guards reject a fifth
# group on either side but not letters (``host192.168.1.10``) or the period
# that ends a sentence. A four-part version string such as ``1.2.3.4`` is
# indistinguishable from an address and is redacted: a firmware version read
# as an address costs a little readability, an address read as a version
# costs the privacy the plan promises.
_IPV4_RE: Final = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)")
# Candidate IPv6 text: hex groups with at least two colons. ``ipaddress``
# validates each match and :func:`_ip_token` also demands a decimal digit, so
# a clock time (``12:30:45``), a bare ``::``, or a hex-looking word pair
# (``cafe::babe``) stays untouched.
_IPV6_RE: Final = re.compile(
    r"(?<![0-9A-Fa-f:.])(?=[0-9A-Fa-f:]*::|(?:[0-9A-Fa-f]{1,4}:){7})"
    r"[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){1,7}(?![0-9A-Fa-f:])"
)
# A dotted name under a local-network suffix: mDNS (``.local``), the common
# router defaults, and RFC 8375's ``home.arpa``. A name under a public domain
# is not claimed, since it would take a URL in a suggested action with it.
_LOCAL_HOSTNAME_RE: Final = re.compile(
    r"(?<![\w.-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?:local|lan|home|internal|localdomain|intranet|private|arpa)"
    r"(?![\w-])",
    flags=re.IGNORECASE,
)


def client_display_name(client: Any) -> str:
    """
    Return the name a model may see for a network client.

    Manufacturer (from the OUI, not the LAN) plus the pseudonymized key, or
    ``device <key>`` when no manufacturer is known. The notifier can map the
    key back to the inventory's stored name after the model has spoken. The
    result is itself redacted, so a manufacturer string that carries an
    address cannot smuggle it back in.
    """
    key = ""
    manufacturer = ""
    if isinstance(client, Mapping):
        key = sanitize_label(client.get(_CLIENT_KEY))
        manufacturer = sanitize_label(client.get("manufacturer"))
    if manufacturer and key:
        return _redact_text(f"{manufacturer} {key}")
    if key:
        return _redact_text(f"device {key}")
    return "unknown device"


def _ip_token(text: str) -> str | None:
    if not any(ch.isdigit() for ch in text):
        return None
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return None
    # ``is_private`` covers loopback, link-local, and the RFC 1918 / ULA
    # ranges a home network uses; everything else, carrier-grade NAT
    # included, is the address the home is reachable by.
    if address.is_private:
        return LAN_IP_REDACTED
    return PUBLIC_IP_REDACTED


def _redact_text(text: str) -> str:
    """Replace every address and local hostname embedded in *text*."""
    text = _MAC_RE.sub(MAC_REDACTED, text)
    text = _BARE_MAC_RE.sub(MAC_REDACTED, text)
    text = _IPV4_RE.sub(lambda m: _ip_token(m.group(0)) or m.group(0), text)
    text = _IPV6_RE.sub(lambda m: _ip_token(m.group(0)) or m.group(0), text)
    return _LOCAL_HOSTNAME_RE.sub(HOSTNAME_REDACTED, text)


class _Walker:
    """One redaction pass: tracks the active path and containers already done."""

    def __init__(self) -> None:
        self._active: set[int] = set()
        self._done: dict[int, Any] = {}

    def walk(self, value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return _redact_text(value)
        if isinstance(value, bytes | bytearray):
            return _redact_text(bytes(value).decode("utf-8", "replace"))
        if isinstance(value, Mapping | list | tuple | set | frozenset):
            return self._container(value, depth)
        # Anything else renders through str() in the prompt, so redact that.
        return _redact_text(str(value))

    def _container(self, value: Any, depth: int) -> Any:
        ident = id(value)
        if ident in self._done:
            return self._done[ident]
        if ident in self._active:
            return CYCLE_MARKER
        if depth >= MAX_DEPTH:
            return DEPTH_MARKER
        self._active.add(ident)
        try:
            if isinstance(value, Mapping):
                out: Any = self._mapping(value, depth + 1)
            elif isinstance(value, tuple):
                out = tuple(self.walk(item, depth + 1) for item in value)
            elif isinstance(value, frozenset):
                out = frozenset(self.walk(item, depth + 1) for item in value)
            elif isinstance(value, set):
                out = {self.walk(item, depth + 1) for item in value}
            else:
                out = [self.walk(item, depth + 1) for item in value]
        finally:
            self._active.discard(ident)
        self._done[ident] = out
        return out

    def _mapping(self, value: Mapping[Any, Any], depth: int) -> dict[Any, Any]:
        out: dict[Any, Any] = {}
        for key, item in value.items():
            name = key.lower() if isinstance(key, str) else None
            if name in _DROP_KEYS:
                continue
            if name in _HOSTNAME_KEYS:
                if isinstance(value.get(_CLIENT_KEY), str) and value[_CLIENT_KEY]:
                    out[key] = client_display_name(value)
                continue
            new_key = self.walk(key, depth)
            # Two distinct addresses redact to one token; keep both entries
            # so a per-address map still tells the model how many there were.
            if new_key in out and new_key != key:
                suffix = 2
                while f"{new_key} #{suffix}" in out:
                    suffix += 1
                new_key = f"{new_key} #{suffix}"
            out[new_key] = self.walk(item, depth)
        return out


def redact_network_identifiers(value: Any) -> Any:
    """
    Return a deep copy of *value* with network identifiers removed.

    Accepts any structure or a bare string; see the module docstring for
    what is dropped, replaced, and tokenized. ``None``, numbers, and booleans
    pass through; every other non-container object is rendered with
    ``str()`` and redacted. The input is never mutated.
    """
    return _Walker().walk(value)


# A MAC written the way Home Assistant slugs it into an entity id or a
# router names an unresolved client (``7a_66_f6_e1_1c_56``), and a bare
# 12/16-hex run inside a name (``shellyplug_s_3494547a1b2c``). The text
# redactor keeps ``_``-bounded runs on purpose (they are entity ids the
# model must be able to cite), so a LABEL needs its own, stricter check.
_LABEL_MAC_RE: Final = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{2}(?:_[0-9a-f]{2}){5,7}(?![0-9a-f])"
    r"|(?<![0-9a-f])[0-9a-f]{12}(?:[0-9a-f]{4})?(?![0-9a-f])",
    re.IGNORECASE,
)


def label_carries_address(text: str) -> bool:
    """
    Return True when a display label carries a MAC or IP in any spelling.

    Used where a label is about to be shown as a NAME (the snapshot, the
    device inventory, the discovery model's view): such a label is dropped
    in favour of the manufacturer-plus-key display name rather than
    tokenized, so no address rides along under the guise of a name.
    """
    if not text:
        return False
    if str(redact_network_identifiers(text)) != text:
        return True
    return _LABEL_MAC_RE.search(text) is not None
