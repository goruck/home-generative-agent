# ruff: noqa: S101
"""Tests for the persistent, non-secret auth inventory."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_generative_agent.sentinel.auth_inventory import (
    MAX_SEEN_IPS,
    TOKEN_TYPE_LONG_LIVED,
    TOKEN_TYPE_NORMAL,
    AuthInventory,
    ObservedToken,
    ObservedUser,
    token_labels,
)
from custom_components.home_generative_agent.sentinel.pseudonymizer import (
    Pseudonymizer,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _token(
    token_id: str,
    *,
    user_id: str = "u1",
    token_type: str = TOKEN_TYPE_LONG_LIVED,
    client_name: str | None = "api",
    ip_key: str | None = "ip-a",
) -> ObservedToken:
    return ObservedToken(
        token_id=token_id,
        user_id=user_id,
        token_type=token_type,
        client_name=client_name,
        created_at=NOW.isoformat(),
        last_used_at=NOW.isoformat(),
        last_used_ip_key=ip_key,
    )


def _user(
    user_id: str = "u1",
    *tokens: ObservedToken,
    admin: bool = True,
    name: str | None = "Lindo",
    system: bool = False,
) -> ObservedUser:
    return ObservedUser(
        user_id=user_id,
        name=name,
        is_admin=admin,
        is_active=True,
        system_generated=system,
        tokens=tuple(tokens),
    )


class _MemoryStore:
    """In-memory stand-in for homeassistant.helpers.storage.Store."""

    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None
        self.removed = False

    async def async_load(self) -> dict[str, Any] | None:
        # Round-trip through JSON so nothing unserializable sneaks in.
        return json.loads(json.dumps(self.data)) if self.data is not None else None

    async def async_save(self, data: dict[str, Any]) -> None:
        self.data = json.loads(json.dumps(data))

    async def async_remove(self) -> None:
        self.data = None
        self.removed = True


def _inventory(store: _MemoryStore | None = None) -> tuple[AuthInventory, _MemoryStore]:
    inventory = AuthInventory(MagicMock())
    backing = store or _MemoryStore()
    inventory._store = backing  # type: ignore[assignment]
    return inventory, backing


@pytest.mark.asyncio
async def test_bootstrap_records_everything_without_alerting() -> None:
    """The first commit flags bootstrap and the diff before it is empty."""
    inventory, store = _inventory()
    await inventory.async_load()
    observation = [_user("u1", _token("t1"))]

    delta = inventory.diff(observation)
    assert delta.bootstrap is True
    assert delta.new_long_lived_tokens == []

    assert await inventory.async_commit(observation, NOW) is True
    assert inventory.is_bootstrapped
    assert inventory.summary()["long_lived_token_count"] == 1
    assert inventory.summary()["admin_count"] == 1
    # A second commit is not a bootstrap.
    assert await inventory.async_commit(observation, NOW) is False
    assert store.data is not None


@pytest.mark.asyncio
async def test_new_admin_and_token_detected_across_restart() -> None:
    """A fresh instance loading the same store sees the change, not a bootstrap."""
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", _token("t1"))], NOW)

    restarted, _ = _inventory(store)
    await restarted.async_load()
    assert restarted.is_bootstrapped
    later = [
        _user("u1", _token("t1"), _token("t2", client_name="new script")),
        _user("u2", name="Guest", admin=True),
        _user("u3", name="Reader", admin=False),
        _user("hassio", name="Supervisor", admin=True, system=True),
    ]
    delta = restarted.diff(later)
    assert delta.bootstrap is False
    assert delta.new_long_lived_tokens == ["new script"]
    assert delta.new_admin_users == ["Guest"]
    assert delta.tokens_from_new_ip == []


@pytest.mark.asyncio
async def test_promotion_to_admin_is_a_new_admin() -> None:
    """A known non-admin user made admin is reported once."""
    inventory, _ = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u2", name="Sam", admin=False)], NOW)
    delta = inventory.diff([_user("u2", name="Sam", admin=True)])
    assert delta.new_admin_users == ["Sam"]
    await inventory.async_commit([_user("u2", name="Sam", admin=True)], NOW)
    assert inventory.diff([_user("u2", name="Sam", admin=True)]).new_admin_users == []


@pytest.mark.asyncio
async def test_token_from_new_address_only_for_long_lived_known_tokens() -> None:
    """A known long-lived token seen from a new address is flagged; browsers are not."""
    inventory, _ = _inventory()
    await inventory.async_load()
    await inventory.async_commit(
        [
            _user(
                "u1",
                _token("t1", ip_key="ip-a"),
                _token(
                    "b1",
                    token_type=TOKEN_TYPE_NORMAL,
                    client_name="Chrome",
                    ip_key="ip-a",
                ),
            )
        ],
        NOW,
    )
    delta = inventory.diff(
        [
            _user(
                "u1",
                _token("t1", ip_key="ip-b"),
                _token(
                    "b1",
                    token_type=TOKEN_TYPE_NORMAL,
                    client_name="Chrome",
                    ip_key="ip-z",
                ),
            )
        ]
    )
    assert delta.tokens_from_new_ip == ["api"]
    # After committing, the address is known and no longer new.
    await inventory.async_commit([_user("u1", _token("t1", ip_key="ip-b"))], NOW)
    assert (
        inventory.diff([_user("u1", _token("t1", ip_key="ip-b"))]).tokens_from_new_ip
        == []
    )


@pytest.mark.asyncio
async def test_token_rows_deleted_when_ha_drops_them() -> None:
    """Rows for tokens and users that vanished from HA are removed."""
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", _token("t1"), _token("t2"))], NOW)
    await inventory.async_commit([_user("u1", _token("t1"))], NOW)
    assert store.data is not None
    assert set(store.data["tokens"]) == {"t1"}
    await inventory.async_commit([], NOW)
    assert store.data["users"] == {}
    assert store.data["tokens"] == {}


@pytest.mark.asyncio
async def test_seen_ips_bounded_and_retained() -> None:
    """seen_ips keeps the most recent MAX_SEEN_IPS and drops idle entries."""
    inventory, store = _inventory()
    await inventory.async_load()
    for i in range(MAX_SEEN_IPS + 5):
        await inventory.async_commit(
            [_user("u1", _token("t1", ip_key=f"ip-{i}"))], NOW + timedelta(minutes=i)
        )
    assert store.data is not None
    ips = store.data["tokens"]["t1"]["seen_ips"]
    assert len(ips) == MAX_SEEN_IPS
    assert ips[0]["key"] == f"ip-{MAX_SEEN_IPS + 4}"  # most recent first

    # Retention: an address idle longer than the window is dropped.
    await inventory.async_commit(
        [_user("u1", _token("t1", ip_key="ip-new"))],
        NOW + timedelta(days=200),
        ip_retention_days=90,
    )
    keys = {row["key"] for row in store.data["tokens"]["t1"]["seen_ips"]}
    assert keys == {"ip-new"}


@pytest.mark.asyncio
async def test_persisted_json_has_no_raw_ip_or_token_value() -> None:
    """What lands on disk is pseudonymized keys and ids only."""
    pseudonymizer = Pseudonymizer("salt")
    ip_key = pseudonymizer.ip_key("10.0.0.7")
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", _token("t1", ip_key=ip_key))], NOW)
    assert store.data is not None
    blob = json.dumps(store.data)
    assert "10.0.0.7" not in blob
    assert ip_key in blob
    assert "jwt" not in blob.lower()
    assert set(store.data["tokens"]["t1"]) == {
        "first_seen",
        "seen_ips",
        "user_id",
        "token_type",
        "client_name",
        "created_at",
    }


@pytest.mark.asyncio
async def test_reset_clears_and_rebootstraps() -> None:
    """Reset removes the store; the next commit is a bootstrap again."""
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", _token("t1"))], NOW)
    await inventory.async_reset()
    assert store.removed
    assert not inventory.is_bootstrapped
    assert inventory.diff([_user("u1", _token("t9"))]).bootstrap is True
    assert await inventory.async_commit([_user("u1", _token("t9"))], NOW) is True


@pytest.mark.asyncio
async def test_load_tolerates_corrupt_store() -> None:
    """A corrupt or failing store starts empty instead of raising."""
    inventory = AuthInventory(MagicMock())
    inventory._store = MagicMock(async_load=AsyncMock(side_effect=OSError("disk")))  # type: ignore[assignment]
    await inventory.async_load()
    assert not inventory.is_bootstrapped
    inventory._store = MagicMock(
        async_load=AsyncMock(return_value=["not", "a", "dict"])
    )  # type: ignore[assignment]
    await inventory.async_load()
    assert not inventory.is_bootstrapped


def test_token_labels_disambiguate_duplicates() -> None:
    """Two tokens named the same get distinct labels; unnamed ones use the id."""
    labels = token_labels(
        [
            _user(
                "u1",
                _token("aaaaaa1", client_name="api"),
                _token("bbbbbb2", client_name="api"),
                _token("cccccc3", client_name=None),
            )
        ]
    )
    assert labels == {
        "aaaaaa1": "api (aaaaaa)",
        "bbbbbb2": "api (bbbbbb)",
        "cccccc3": "token cccccc",
    }


def test_pseudonymizer_is_stable_per_salt_and_differs_across_salts() -> None:
    """Same input + same salt is stable; another install's salt differs."""
    a = Pseudonymizer("salt-a")
    b = Pseudonymizer("salt-b")
    assert a.mac_key("AA:BB:CC:DD:EE:FF") == a.mac_key("aa-bb-cc-dd-ee-ff")
    assert a.mac_key("AA:BB:CC:DD:EE:FF") != b.mac_key("AA:BB:CC:DD:EE:FF")
    assert len(a.ip_key("192.168.1.1")) == 8
    assert a.ip_key("192.168.1.1") != a.ip_key("192.168.1.2")


@pytest.mark.asyncio
async def test_commit_saves_only_when_something_changed() -> None:
    """A quiet install does not rewrite the store every cycle."""
    inventory, store = _inventory()
    await inventory.async_load()
    saves = 0
    original = store.async_save

    async def _counting_save(data: dict[str, Any]) -> None:
        nonlocal saves
        saves += 1
        await original(data)

    store.async_save = _counting_save  # type: ignore[method-assign]
    observation = [_user("u1", _token("t1", ip_key="ip-a"))]
    await inventory.async_commit(observation, NOW)
    assert saves == 1
    # Same observation five minutes later: the address's last_seen refresh is
    # rate-limited to an hour, so nothing is dirty.
    await inventory.async_commit(observation, NOW + timedelta(minutes=5))
    assert saves == 1
    await inventory.async_commit(observation, NOW + timedelta(hours=2))
    assert saves == 2
    await inventory.async_commit([_user("u1", _token("t1", ip_key="ip-b"))], NOW)
    assert saves == 3


@pytest.mark.asyncio
async def test_salt_change_resets_addresses_without_alerting() -> None:
    """Keys under a different salt are never compared; history restarts."""
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit(
        [_user("u1", _token("t1", ip_key="old-a"))], NOW, salt_fingerprint="salt-1"
    )
    delta = inventory.diff(
        [_user("u1", _token("t1", ip_key="new-a"))], salt_fingerprint="salt-2"
    )
    assert delta.tokens_from_new_ip == []
    await inventory.async_commit(
        [_user("u1", _token("t1", ip_key="new-a"))], NOW, salt_fingerprint="salt-2"
    )
    assert store.data is not None
    assert store.data["salt_fingerprint"] == "salt-2"
    assert [ip["key"] for ip in store.data["tokens"]["t1"]["seen_ips"]] == ["new-a"]
    # Under the new salt, a further new address is detected again.
    assert inventory.diff(
        [_user("u1", _token("t1", ip_key="new-b"))], salt_fingerprint="salt-2"
    ).tokens_from_new_ip == ["api"]


@pytest.mark.asyncio
async def test_hold_back_keeps_undelivered_changes_reportable() -> None:
    """Changes whose finding was not delivered are reported again next run."""
    inventory, _ = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", _token("t1", ip_key="ip-a"))], NOW)
    later = [
        _user("u1", _token("t1", ip_key="ip-b"), _token("t2", client_name="new")),
        _user("u2", name="Guest", admin=True),
    ]
    delta = inventory.diff(later)
    assert delta.has_changes
    assert delta.new_token_ids == ["t2"]
    assert delta.new_admin_user_ids == ["u2"]
    assert delta.new_ip_token_ids == ["t1"]
    await inventory.async_commit(later, NOW, hold_back=delta)
    # Still new on the next run, because nothing was recorded for them.
    again = inventory.diff(later)
    assert again.new_long_lived_tokens == ["new"]
    assert again.new_admin_users == ["Guest"]
    assert again.tokens_from_new_ip == ["api"]
    # Delivered this time: committed and quiet afterwards.
    await inventory.async_commit(later, NOW)
    assert not inventory.diff(later).has_changes


@pytest.mark.asyncio
async def test_load_drops_corrupt_rows_instead_of_raising_later() -> None:
    """A hand-edited row that is not a mapping is discarded on load."""
    inventory, store = _inventory()
    store.data = {
        "bootstrapped_at": NOW.isoformat(),
        "users": {"u1": {"is_admin": True}, "bad": "not a row"},
        "tokens": {
            "t1": {"token_type": TOKEN_TYPE_LONG_LIVED, "seen_ips": "nope"},
            "t2": 3,
        },
    }
    await inventory.async_load()
    assert inventory.is_bootstrapped
    assert inventory.summary()["user_count"] == 1
    assert inventory.summary()["token_count"] == 1
    # The surviving token row is usable by diff and commit.
    delta = inventory.diff([_user("u1", _token("t1", ip_key="ip-a"))])
    assert delta.tokens_from_new_ip == ["api"]
    await inventory.async_commit([_user("u1", _token("t1", ip_key="ip-a"))], NOW)


@pytest.mark.asyncio
async def test_persisted_rows_carry_no_user_name() -> None:
    """User rows keep flags and ids only; the display name is not duplicated."""
    inventory, store = _inventory()
    await inventory.async_load()
    await inventory.async_commit([_user("u1", name="Lindo")], NOW)
    assert store.data is not None
    assert "Lindo" not in json.dumps(store.data)


@pytest.mark.asyncio
async def test_failed_save_retries_and_defers_bootstrap_notice() -> None:
    """A failed write is retried next cycle and the bootstrap notice waits for it."""
    from homeassistant.exceptions import HomeAssistantError  # noqa: PLC0415

    inventory, store = _inventory()
    await inventory.async_load()
    original = store.async_save
    failures = 1

    async def _flaky_save(data: dict[str, Any]) -> None:
        nonlocal failures
        if failures:
            failures -= 1
            msg = "disk full"
            raise HomeAssistantError(msg)
        await original(data)

    store.async_save = _flaky_save  # type: ignore[method-assign]
    observation = [_user("u1", _token("t1", ip_key="ip-a"))]
    # Bootstrap in memory, but nothing reached disk: no announcement yet.
    assert await inventory.async_commit(observation, NOW) is False
    assert inventory.is_bootstrapped
    # Same observation, nothing changed: the retry alone triggers the save
    # and the deferred announcement.
    assert await inventory.async_commit(observation, NOW + timedelta(minutes=5)) is True
    assert (
        await inventory.async_commit(observation, NOW + timedelta(minutes=10)) is False
    )
