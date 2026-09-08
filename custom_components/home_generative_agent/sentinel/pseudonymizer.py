"""
Per-install pseudonymization of network identifiers.

MAC addresses, IPs, and Bluetooth addresses are tokenized with an HMAC keyed by
a per-install salt before any model sees them and before anything is
persisted. The key is stable across runs on one install (so inventories can
compare) and different across installs (so a key from one home says nothing
about another).

The salt lives in its own storage file rather than in the config entry:
writing it into entry options from the engine at runtime would fire the
config-entry change listener and reload the integration mid-cycle. It is
generated once with ``secrets.token_hex`` — the same pattern the
critical-action PIN salt uses.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from typing import TYPE_CHECKING

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_network_salt"
KEY_LENGTH = 8


class Pseudonymizer:
    """Derive short, stable, per-install keys for network identifiers."""

    def __init__(self, salt: str) -> None:
        """Initialize with the per-install salt (hex string)."""
        self._salt = salt.encode("utf-8")

    def key(self, value: str) -> str:
        """Return the pseudonymized key for an arbitrary identifier."""
        normalized = value.strip().lower().encode("utf-8")
        return hmac.new(self._salt, normalized, hashlib.sha256).hexdigest()[:KEY_LENGTH]

    def mac_key(self, mac: str) -> str:
        """Return the key for a MAC address, tolerant of separator style."""
        return self.key(mac.replace("-", ":").replace(".", ":"))

    def ip_key(self, ip: str) -> str:
        """Return the key for an IP address (v4 or v6, textual form)."""
        return self.key(ip)


def new_salt() -> str:
    """Return a fresh random salt."""
    return secrets.token_hex(32)


async def async_load_pseudonymizer(hass: HomeAssistant) -> Pseudonymizer:
    """
    Load the per-install salt, generating and persisting one on first use.

    A storage failure degrades to a process-lifetime salt: keys stay
    consistent within this run and are regenerated on the next start, which
    means inventories re-bootstrap rather than the feature failing.
    """
    store: Store[dict[str, str]] = Store(hass, STORE_VERSION, STORE_KEY)
    try:
        data = await store.async_load()
    except (HomeAssistantError, OSError, ValueError):
        LOGGER.warning(
            "Could not read the network pseudonymization salt; "
            "using a temporary salt for this run."
        )
        return Pseudonymizer(new_salt())
    salt = data.get("salt") if isinstance(data, dict) else None
    if isinstance(salt, str) and salt:
        return Pseudonymizer(salt)
    salt = new_salt()
    try:
        await store.async_save({"salt": salt})
    except (HomeAssistantError, OSError, ValueError):
        LOGGER.warning(
            "Could not persist the network pseudonymization salt; "
            "inventories will re-bootstrap on the next start."
        )
    return Pseudonymizer(salt)


async def async_remove_pseudonymizer_salt(hass: HomeAssistant) -> None:
    """Delete the persisted salt (Sentinel subentry removal)."""
    store: Store[dict[str, str]] = Store(hass, STORE_VERSION, STORE_KEY)
    try:
        await store.async_remove()
    except (HomeAssistantError, OSError):
        LOGGER.debug("Network salt store removal failed; ignoring.")
