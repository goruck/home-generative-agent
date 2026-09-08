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
* **Salt binding.** Address keys are HMACs under the per-install salt, so the
  inventory records the salt's fingerprint. A different fingerprint at
  commit time (salt file deleted, temporary salt after a storage failure)
  wipes ``seen_ips`` instead of comparing incomparable keys, which would
  otherwise flag every token as "used from a new address".
* **Held-back changes.** The engine can ask the commit to leave specific
  users and tokens out (their finding was suppressed and never delivered),
  so the change is reported on a later run instead of being baselined.
* **Writes only on change.** Address ``last_seen`` stamps refresh at most
  hourly, so a quiet install does not rewrite the file every cycle.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
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
# An address's last_seen stamp is refreshed at most this often, so a token
# used every cycle does not dirty the store every cycle.
SEEN_IP_REFRESH = timedelta(hours=1)


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
    # Identifiers behind the labels above, for holding a change back from a
    # commit when its finding was not delivered.
    new_admin_user_ids: list[str] = field(default_factory=list)
    new_token_ids: list[str] = field(default_factory=list)
    new_ip_token_ids: list[str] = field(default_factory=list)
    # True when the inventory has never been committed: nothing is "new" yet.
    bootstrap: bool = False

    @property
    def has_changes(self) -> bool:
        """Return True when anything would be reported."""
        return bool(
            self.new_admin_users
            or self.new_long_lived_tokens
            or self.tokens_from_new_ip
        )


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
    return {
        "bootstrapped_at": None,
        "salt_fingerprint": None,
        "users": {},
        "tokens": {},
    }


def _valid_rows(raw: Any) -> dict[str, dict[str, Any]]:
    """Keep only mapping rows; a hand-edited or corrupt row must not raise later."""
    if not isinstance(raw, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        row = dict(value)
        if "seen_ips" in row:
            row["seen_ips"] = (
                [ip for ip in row["seen_ips"] if isinstance(ip, dict) and ip.get("key")]
                if isinstance(row["seen_ips"], list)
                else []
            )
        rows[str(key)] = row
    return rows


def _counts_admin(user: ObservedUser) -> bool:
    return user.is_admin and user.is_active and not user.system_generated


class AuthInventory:
    """Persist the last observed users and tokens for change detection."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the store."""
        self._store: Store[dict[str, Any]] = Store(
            hass, STORE_VERSION, STORE_KEY, private=True, atomic_writes=True
        )
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
        bootstrapped_at = data.get("bootstrapped_at")
        fingerprint = data.get("salt_fingerprint")
        self._data = {
            "bootstrapped_at": (
                bootstrapped_at if isinstance(bootstrapped_at, str) else None
            ),
            "salt_fingerprint": fingerprint if isinstance(fingerprint, str) else None,
            "users": _valid_rows(data.get("users")),
            "tokens": _valid_rows(data.get("tokens")),
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

    # ------------------------------------------------------------------ #
    # Diff / commit
    # ------------------------------------------------------------------ #

    def _addresses_comparable(self, salt_fingerprint: str | None) -> bool:
        """Return True when stored address keys were made with this salt."""
        stored = self._data.get("salt_fingerprint")
        return stored is None or salt_fingerprint is None or stored == salt_fingerprint

    def diff(
        self,
        observation: Sequence[ObservedUser],
        *,
        salt_fingerprint: str | None = None,
    ) -> AuthDelta:
        """
        Compare *observation* against the stored inventory.

        Pure: reads the in-memory data only. Before bootstrap every row would
        be "new", so the delta is empty and flagged ``bootstrap=True``. When
        the salt changed, address keys are incomparable and new-address
        detection is skipped for this run.
        """
        if not self.is_bootstrapped:
            return AuthDelta(bootstrap=True)
        known_users: dict[str, Any] = self._data["users"]
        known_tokens: dict[str, Any] = self._data["tokens"]
        labels = token_labels(observation)
        compare_ips = self._addresses_comparable(salt_fingerprint)

        new_admins: list[tuple[str, str]] = []
        new_ll_tokens: list[tuple[str, str]] = []
        new_ip_tokens: list[tuple[str, str]] = []
        for user in observation:
            stored = known_users.get(user.user_id)
            if _counts_admin(user) and (stored is None or not stored.get("is_admin")):
                new_admins.append((user_label(user), user.user_id))
            for token in user.tokens:
                if token.token_type != TOKEN_TYPE_LONG_LIVED:
                    continue
                stored_token = known_tokens.get(token.token_id)
                if stored_token is None:
                    new_ll_tokens.append((labels[token.token_id], token.token_id))
                    continue
                ip_key = token.last_used_ip_key
                if (
                    compare_ips
                    and ip_key
                    and ip_key
                    not in {
                        entry.get("key") for entry in stored_token.get("seen_ips", [])
                    }
                ):
                    new_ip_tokens.append((labels[token.token_id], token.token_id))
        new_admins.sort()
        new_ll_tokens.sort()
        new_ip_tokens.sort()
        return AuthDelta(
            new_admin_users=[label for label, _ in new_admins],
            new_long_lived_tokens=[label for label, _ in new_ll_tokens],
            tokens_from_new_ip=[label for label, _ in new_ip_tokens],
            new_admin_user_ids=[uid for _, uid in new_admins],
            new_token_ids=[tid for _, tid in new_ll_tokens],
            new_ip_token_ids=[tid for _, tid in new_ip_tokens],
        )

    async def async_commit(  # noqa: PLR0912, PLR0915
        self,
        observation: Sequence[ObservedUser],
        now: datetime,
        *,
        ip_retention_days: int = DEFAULT_IP_RETENTION_DAYS,
        salt_fingerprint: str | None = None,
        hold_back: AuthDelta | None = None,
    ) -> bool:
        """
        Record *observation* as the known state and persist it when changed.

        ``hold_back`` names changes whose finding was not delivered this run;
        those users and tokens are left as they were so the next run reports
        them again. Returns True when this call bootstrapped the inventory.
        """
        now_iso = dt_util.as_utc(now).isoformat()
        bootstrap = not self.is_bootstrapped
        users: dict[str, Any] = self._data["users"]
        tokens: dict[str, Any] = self._data["tokens"]
        changed = bootstrap
        held_users = set(hold_back.new_admin_user_ids) if hold_back else set()
        held_tokens = set(hold_back.new_token_ids) if hold_back else set()
        held_ips = set(hold_back.new_ip_token_ids) if hold_back else set()

        if not self._addresses_comparable(salt_fingerprint):
            # Keys under the old salt cannot be matched against keys under the
            # new one; start the address history over without alerting.
            for stored_row in tokens.values():
                stored_row["seen_ips"] = []
            changed = True
        if salt_fingerprint is not None and (
            self._data.get("salt_fingerprint") != salt_fingerprint
        ):
            self._data["salt_fingerprint"] = salt_fingerprint
            changed = True

        seen_user_ids: set[str] = set()
        seen_token_ids: set[str] = set()
        for user in observation:
            seen_user_ids.add(user.user_id)
            new_user_row = {
                "is_admin": _counts_admin(user),
                "is_active": user.is_active,
                "system_generated": user.system_generated,
            }
            row: dict[str, Any] | None = users.get(user.user_id)
            if user.user_id in held_users:
                # Reported later: keep whatever was stored (possibly nothing).
                pass
            elif row is None:
                users[user.user_id] = {"first_seen": now_iso, **new_user_row}
                changed = True
            elif any(row.get(k) != v for k, v in new_user_row.items()):
                row.update(new_user_row)
                changed = True
            for token in user.tokens:
                seen_token_ids.add(token.token_id)
                trow: dict[str, Any] | None = tokens.get(token.token_id)
                if token.token_id in held_tokens:
                    continue
                new_token_row = {
                    "user_id": token.user_id,
                    "token_type": token.token_type,
                    "client_name": token.client_name,
                    "created_at": token.created_at,
                }
                if trow is None:
                    trow = {"first_seen": now_iso, "seen_ips": [], **new_token_row}
                    tokens[token.token_id] = trow
                    changed = True
                elif any(trow.get(k) != v for k, v in new_token_row.items()):
                    trow.update(new_token_row)
                    changed = True
                ip_key = None if token.token_id in held_ips else token.last_used_ip_key
                seen_ips, ips_changed = _update_seen_ips(
                    trow.get("seen_ips", []), ip_key, now_iso, now, ip_retention_days
                )
                if ips_changed:
                    trow["seen_ips"] = seen_ips
                    changed = True

        for user_id in list(users):
            if user_id not in seen_user_ids and user_id not in held_users:
                del users[user_id]
                changed = True
        for token_id in list(tokens):
            if token_id not in seen_token_ids and token_id not in held_tokens:
                del tokens[token_id]
                changed = True

        if bootstrap:
            self._data["bootstrapped_at"] = now_iso
        if changed:
            await self.async_save()
        return bootstrap


def _update_seen_ips(
    entries: Iterable[dict[str, Any]],
    ip_key: str | None,
    now_iso: str,
    now: datetime,
    retention_days: int,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Refresh/append the current address and apply the bound and retention.

    Returns the new list and whether it differs from *entries*. A known
    address's ``last_seen`` is refreshed only once per ``SEEN_IP_REFRESH`` so
    steady use does not dirty the store every cycle.
    """
    rows = [dict(e) for e in entries if isinstance(e, dict) and e.get("key")]
    changed = False
    if ip_key:
        for row in rows:
            if row.get("key") == ip_key:
                last = dt_util.parse_datetime(str(row.get("last_seen") or ""))
                if (
                    last is None
                    or dt_util.as_utc(now) - dt_util.as_utc(last) >= SEEN_IP_REFRESH
                ):
                    row["last_seen"] = now_iso
                    changed = True
                break
        else:
            rows.append({"key": ip_key, "first_seen": now_iso, "last_seen": now_iso})
            changed = True

    kept: list[dict[str, Any]] = []
    for row in rows:
        last_seen = dt_util.parse_datetime(str(row.get("last_seen") or ""))
        if last_seen is None:
            changed = True
            continue
        age_days = (dt_util.as_utc(now) - dt_util.as_utc(last_seen)).days
        if retention_days > 0 and age_days > retention_days:
            changed = True
            continue
        kept.append(row)
    kept.sort(key=lambda r: str(r.get("last_seen") or ""), reverse=True)
    if len(kept) > MAX_SEEN_IPS:
        changed = True
    return kept[:MAX_SEEN_IPS], changed
