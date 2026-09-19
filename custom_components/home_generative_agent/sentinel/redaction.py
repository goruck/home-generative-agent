"""
Redaction of network identifiers before any model call.

The pseudonymizer (``sentinel/pseudonymizer.py``) tokenizes addresses on the
way *into* the snapshot; this module is the matching gate on the way *out*,
applied to whatever is about to be rendered into a prompt: a finding's
evidence (explain), the triage prompt, the reduced discovery snapshot, and
the ``audit_home_security`` tool payload. Everything an LLM sees passes
through :func:`redact_network_identifiers`, so the invariant "no MAC, IP, or
hostname reaches a model" holds regardless of which adapter or rule produced
the data (docs/network-security-plan.md, *Privacy of the Audit*).

Two layers, because identifiers arrive in two shapes:

- **Structural.** Dict keys that name an address (``mac``, ``ip``, ``ieee``,
  ...) are dropped wherever they occur. A ``hostname`` is replaced by the
  client's display name (manufacturer plus pseudonymized key) when the dict
  carries a ``key``, and dropped otherwise: a hostname cannot be recognised
  by pattern, so it is only ever removed by position.
- **Textual.** Every string, including dict keys and list members, has
  embedded MAC addresses replaced by ``[mac]`` and IP addresses by
  ``[lan ip]`` or ``[public ip]``: the subnet class is kept because "a port
  forward to a LAN host" and "a token used from a public address" read
  differently, the address itself does not matter to the model. This is what
  catches an address a rule folded into its ``summary`` line or a discovery
  title advertised on the LAN.

The walk builds new containers throughout and never mutates its input; a
finding's evidence feeds anomaly-id derivation elsewhere and must stay intact.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Final

from custom_components.home_generative_agent.snapshot.network import sanitize_label

MAC_REDACTED: Final = "[mac]"
LAN_IP_REDACTED: Final = "[lan ip]"
PUBLIC_IP_REDACTED: Final = "[public ip]"

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

# Eight (EUI-64: Zigbee IEEE addresses) or six (MAC, Bluetooth) colon- or
# dash-separated octets, or Cisco's three dotted quads (``0011.2233.4455``).
# Bounded so a hash or a longer hex string is not cut in the middle; the
# eight-octet form is tried first so an IEEE address is not left as two
# halves.
_MAC_RE: Final = re.compile(
    r"(?<![0-9A-Fa-f:.-])"
    r"(?:[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){7}"
    r"|[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}"
    r"|[0-9A-Fa-f]{4}(?:\.[0-9A-Fa-f]{4}){2})"
    r"(?![0-9A-Fa-f:.-])"
)
# Four dotted decimal groups; ``ipaddress`` then rejects anything out of
# range (``999.1.1.1``) so it is left as written. A four-part version string
# such as ``1.2.3.4`` is indistinguishable from an address and is redacted:
# a firmware version read as an address costs a little readability, an
# address read as a version costs the privacy the plan promises.
# The trailing guard rejects a fifth dotted group but not the period that
# ends a sentence ("changed to 203.0.113.9.").
_IPV4_RE: Final = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)")
# Candidate IPv6 text: hex groups with at least two colons. ``ipaddress``
# validates each match, so a clock time (``12:30:45``, three groups, no
# ``::``) or a MAC is not an address and stays untouched.
_IPV6_RE: Final = re.compile(
    r"(?<![\w:.])(?=[0-9A-Fa-f:]*::|(?:[0-9A-Fa-f]{1,4}:){7})"
    r"[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){1,7}(?![\w:])"
)


def client_display_name(client: Any) -> str:
    """
    Return the name a model may see for a network client.

    Manufacturer (from the OUI, not the LAN) plus the pseudonymized key, or
    ``device <key>`` when no manufacturer is known. The notifier can map the
    key back to the inventory's stored name after the model has spoken.
    """
    key = ""
    manufacturer = ""
    if isinstance(client, dict):
        key = sanitize_label(client.get(_CLIENT_KEY))
        manufacturer = sanitize_label(client.get("manufacturer"))
    if manufacturer and key:
        return f"{manufacturer} {key}"
    if key:
        return f"device {key}"
    return "unknown device"


def _ip_token(text: str) -> str | None:
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
    """Replace every MAC and IP address embedded in *text* with its token."""
    text = _MAC_RE.sub(MAC_REDACTED, text)
    text = _IPV4_RE.sub(lambda m: _ip_token(m.group(0)) or m.group(0), text)
    return _IPV6_RE.sub(lambda m: _ip_token(m.group(0)) or m.group(0), text)


def _redact_mapping(value: dict[Any, Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, item in value.items():
        name = key.lower() if isinstance(key, str) else None
        if name in _DROP_KEYS:
            continue
        if name in _HOSTNAME_KEYS:
            if isinstance(value.get(_CLIENT_KEY), str) and value[_CLIENT_KEY]:
                out[key] = client_display_name(value)
            continue
        out[_walk(key)] = _walk(item)
    return out


def _walk(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return _redact_mapping(value)
    if isinstance(value, list | tuple | set | frozenset):
        return type(value)(_walk(item) for item in value)
    return value


def redact_network_identifiers(value: Any) -> Any:
    """
    Return a deep copy of *value* with network identifiers removed.

    Accepts any JSON-like structure or a bare string; see the module
    docstring for what is dropped, replaced, and tokenized. Scalars other
    than strings pass through unchanged. The input is never mutated.
    """
    return _walk(value)
