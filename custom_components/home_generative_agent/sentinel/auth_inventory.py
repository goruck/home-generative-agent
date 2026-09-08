"""
Persistent, non-secret inventory of Home Assistant users and refresh tokens.

Home Assistant's auth objects describe the present only, and the audit store
keeps just a snapshot hash, so the auth *change* rules (new admin, new
long-lived token, token used from a new address) need something to compare
against. This store holds exactly that and nothing secret:

* per user: ``user_id``, ``is_admin``, ``is_active``, ``name``, ``first_seen``
* per refresh token: ``token_id`` (Home Assistant's own identifier, not the
  token value), ``user_id``, ``token_type``, ``client_name``, ``created_at``,
  ``first_seen``, and ``seen_ips`` — a bounded list of HMAC-pseudonymized
  addresses with first/last-seen timestamps. Raw IPs never enter this file;
  the rule only needs to know whether the current address was seen before.

Lifecycle:

* **Bootstrap.** The first run records every existing user and token as known
  without alerting (``AuthDelta.bootstrap`` is True) so an upgrade never
  produces a flood of "new token" findings for tokens that predate the audit.
* **Diff, then commit.** ``diff()`` is a pure comparison used while the
  snapshot is built; ``async_commit()`` records the observation afterwards.
  Because the store is persistent, a restart neither re-bootstraps nor
  re-alerts on tokens seen before.
* **Retention.** A token row is deleted when its id no longer exists in Home
  Assistant; ``seen_ips`` keeps the most recent ``MAX_SEEN_IPS`` entries and
  drops entries idle for longer than the configured retention.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_auth_inventory"

# Mirrors homeassistant.auth.models.TOKEN_TYPE_* without importing auth
# internals into the pure-data layer.
TOKEN_TYPE_NORMAL = "normal"  # noqa: S105
TOKEN_TYPE_SYSTEM = "system"  # noqa: S105
TOKEN_TYPE_LONG_LIVED = "long_lived_access_token"  # noqa: S105

MAX_SEEN_IPS = 20
DEFAULT_IP_RETENTION_DAYS = 90


@dataclass(frozen=True)
class ObservedToken:
    """One refresh token as observed from ``hass.auth`` (no secrets)."""

    token_id: str
    user_id: str
    token_type: str
    client_name: str | None
    created_at: str | None
    last_used_at: str | None
    # Pseudonymized (HMAC) last-used address, or None when never used.
    last_used_ip_key: str | None


@dataclass(frozen=True)
class ObservedUser:
    """One user as observed from ``hass.auth``."""

    user_id: str
    name: str | None
    is_admin: bool
    is_active: bool
    system_generated: bool
    tokens: tuple[ObservedToken, ...] = ()


@dataclass(frozen=True)
class AuthDelta:
    """What changed since the inventory was last committed."""

    new_admin_users: list[str] = field(default_factory=list)
    new_long_lived_tokens: list[str] = field(default_factory=list)
    tokens_from_new_ip: list[str] = field(default_factory=list)
    # True when the inventory has never been committed: nothing is "new" yet.
    bootstrap: bool = False


def user_label(user: ObservedUser) -> str:
    """Return a display label for a user (name, else a short id prefix)."""
    return user.name or f"user {user.user_id[:8]}"


def token_labels(observation: Sequence[ObservedUser]) -> dict[str, str]:
    """
    Return ``token_id -> label`` for every observed token.

    Labels are the client name Home Assistant shows in the profile page; when
    two tokens share a name the label is disambiguated with a short id prefix
    so counts and evidence never silently merge two tokens.
    """
    tokens = [token for user in observation for token in user.tokens]
    names = Counter(token.client_name or "" for token in tokens)
    labels: dict[str, str] = {}
    for token in tokens:
        base = token.client_name or f"token {token.token_id[:6]}"
        if token.client_name and names[token.client_name] > 1:
            base = f"{token.client_name} ({token.token_id[:6]})"
        labels[token.token_id] = base
    return labels


def _empty_data() -> dict[str, Any]:
    return {"bootstrapped_at": None, "users": {}, "tokens": {}}


def _counts_admin(user: ObservedUser) -> bool:
    return user.is_admin and user.is_active and not user.system_generated


class AuthInventory:
    """Persist the last observed users and tokens for change detection."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the store."""
        self._store: Store[dict[str, Any]] = Store(hass, STORE_VERSION, STORE_KEY)
        self._data: dict[str, Any] = _empty_data()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    async def async_load(self) -> None:
        """Load the inventory from storage (missing or corrupt -> empty)."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Auth inventory unreadable; starting empty.")
            return
        if not isinstance(data, dict):
            return
        users = data.get("users")
        tokens = data.get("tokens")
        self._data = {
            "bootstrapped_at": data.get("bootstrapped_at"),
            "users": dict(users) if isinstance(users, dict) else {},
            "tokens": dict(tokens) if isinstance(tokens, dict) else {},
        }

    async def async_save(self) -> None:
        """Persist the inventory."""
        try:
            await self._store.async_save(self._data)
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Auth inventory could not be saved.")

    async def async_reset(self) -> None:
        """Clear the inventory; the next run bootstraps again without alerts."""
        self._data = _empty_data()
        try:
            await self._store.async_remove()
        except (HomeAssistantError, OSError):
            LOGGER.debug("Auth inventory store removal failed; ignoring.")

    # ------------------------------------------------------------------ #
    # Read access
    # ------------------------------------------------------------------ #

    @property
    def is_bootstrapped(self) -> bool:
        """Return True once an observation has been committed."""
        return self._data.get("bootstrapped_at") is not None

    def summary(self) -> dict[str, Any]:
        """Return non-secret counts for services and notifications."""
        users = self._data["users"]
        tokens = self._data["tokens"]
        return {
            "bootstrapped_at": self._data.get("bootstrapped_at"),
            "user_count": len(users),
            "admin_count": sum(1 for u in users.values() if u.get("is_admin")),
            "token_count": len(tokens),
            "long_lived_token_count": sum(
                1
                for t in tokens.values()
                if t.get("token_type") == TOKEN_TYPE_LONG_LIVED
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a copy of the stored data (for the inspection service)."""
        return {
            "bootstrapped_at": self._data.get("bootstrapped_at"),
            "users": {k: dict(v) for k, v in self._data["users"].items()},
            "tokens": {
                k: {**v, "seen_ips": [dict(ip) for ip in v.get("seen_ips", [])]}
                for k, v in self._data["tokens"].items()
            },
        }

    # ------------------------------------------------------------------ #
    # Diff / commit
    # ------------------------------------------------------------------ #

    def diff(self, observation: Sequence[ObservedUser]) -> AuthDelta:
        """
        Compare *observation* against the stored inventory.

        Pure: reads the in-memory data only. Before bootstrap every row would
        be "new", so the delta is empty and flagged ``bootstrap=True``.
        """
        if not self.is_bootstrapped:
            return AuthDelta(bootstrap=True)
        known_users: dict[str, Any] = self._data["users"]
        known_tokens: dict[str, Any] = self._data["tokens"]
        labels = token_labels(observation)

        new_admins: list[str] = []
        new_ll_tokens: list[str] = []
        new_ip_tokens: list[str] = []
        for user in observation:
            stored = known_users.get(user.user_id)
            if _counts_admin(user) and (stored is None or not stored.get("is_admin")):
                new_admins.append(user_label(user))
            for token in user.tokens:
                if token.token_type != TOKEN_TYPE_LONG_LIVED:
                    continue
                stored_token = known_tokens.get(token.token_id)
                if stored_token is None:
                    new_ll_tokens.append(labels[token.token_id])
                    continue
                ip_key = token.last_used_ip_key
                if ip_key and ip_key not in {
                    entry.get("key") for entry in stored_token.get("seen_ips", [])
                }:
                    new_ip_tokens.append(labels[token.token_id])
        return AuthDelta(
            new_admin_users=sorted(new_admins),
            new_long_lived_tokens=sorted(new_ll_tokens),
            tokens_from_new_ip=sorted(new_ip_tokens),
        )

    async def async_commit(
        self,
        observation: Sequence[ObservedUser],
        now: datetime,
        *,
        ip_retention_days: int = DEFAULT_IP_RETENTION_DAYS,
    ) -> bool:
        """
        Record *observation* as the known state and persist it.

        Returns True when this call bootstrapped the inventory (first commit).
        """
        now_iso = dt_util.as_utc(now).isoformat()
        bootstrap = not self.is_bootstrapped
        users: dict[str, Any] = self._data["users"]
        tokens: dict[str, Any] = self._data["tokens"]

        seen_user_ids: set[str] = set()
        seen_token_ids: set[str] = set()
        for user in observation:
            seen_user_ids.add(user.user_id)
            row: dict[str, Any] = users.get(user.user_id) or {"first_seen": now_iso}
            row.update(
                {
                    "name": user.name,
                    "is_admin": _counts_admin(user),
                    "is_active": user.is_active,
                    "system_generated": user.system_generated,
                }
            )
            users[user.user_id] = row
            for token in user.tokens:
                seen_token_ids.add(token.token_id)
                trow: dict[str, Any] = tokens.get(token.token_id) or {
                    "first_seen": now_iso,
                    "seen_ips": [],
                }
                trow.update(
                    {
                        "user_id": token.user_id,
                        "token_type": token.token_type,
                        "client_name": token.client_name,
                        "created_at": token.created_at,
                    }
                )
                trow["seen_ips"] = _update_seen_ips(
                    trow.get("seen_ips", []),
                    token.last_used_ip_key,
                    now_iso,
                    now,
                    ip_retention_days,
                )
                tokens[token.token_id] = trow

        for user_id in list(users):
            if user_id not in seen_user_ids:
                del users[user_id]
        for token_id in list(tokens):
            if token_id not in seen_token_ids:
                del tokens[token_id]

        if bootstrap:
            self._data["bootstrapped_at"] = now_iso
        await self.async_save()
        return bootstrap


def _update_seen_ips(
    entries: Iterable[dict[str, Any]],
    ip_key: str | None,
    now_iso: str,
    now: datetime,
    retention_days: int,
) -> list[dict[str, Any]]:
    """Refresh/append the current address and apply the bound and retention."""
    rows = [dict(e) for e in entries if isinstance(e, dict) and e.get("key")]
    if ip_key:
        for row in rows:
            if row.get("key") == ip_key:
                row["last_seen"] = now_iso
                break
        else:
            rows.append({"key": ip_key, "first_seen": now_iso, "last_seen": now_iso})

    kept: list[dict[str, Any]] = []
    for row in rows:
        last_seen = dt_util.parse_datetime(str(row.get("last_seen") or ""))
        if last_seen is None:
            continue
        age_days = (dt_util.as_utc(now) - dt_util.as_utc(last_seen)).days
        if retention_days > 0 and age_days > retention_days:
            continue
        kept.append(row)
    kept.sort(key=lambda r: str(r.get("last_seen") or ""), reverse=True)
    return kept[:MAX_SEEN_IPS]
