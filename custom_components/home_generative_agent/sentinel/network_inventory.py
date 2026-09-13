"""
Persistent inventory of the devices on the home's radio networks.

Zigbee, Z-Wave, Bluetooth, and Matter devices land in Home Assistant's device
registry when they are paired, but the registry keeps no notion of "known to
this home before today", so the new-device rule needs something to compare
against (docs/network-security-radio-plan.md). This store holds exactly that:

* per device, keyed ``"<source>:<device registry id>"``: ``source``
  (``zigbee``, ``zwave``, ``bluetooth``, ``matter``), ``platform``,
  ``ha_device_id``, the last observed ``name`` / ``manufacturer`` / ``model``,
  ``first_seen``, ``last_seen`` and ``trusted``. No IEEE addresses, node ids,
  or MAC addresses are stored: the registry id is the device's identity.
* per source, the time it was first recorded (its bootstrap).

Lifecycle:

* **Per-source bootstrap.** The first time a source is present, every device
  it has is recorded as trusted without alerting. A Z-Wave stick added next
  month is recorded silently with whatever it has at that moment, and each
  device paired after that is new.
* **Diff, then commit.** ``diff()`` is a pure comparison used while the
  snapshot is built; ``async_commit()`` records the observation after the
  engine dispatched its findings. A device paired after its source's
  bootstrap is recorded straight away as untrusted with its alert pending
  (``alerted`` False), so it can be trusted and counted even while its
  alert waits; the engine clears the pending flag once the alert was
  delivered or was stopped for good (a snooze, triage, policy), and a
  transient stop (cooldown, quiet hours) leaves it for a later run.
* **Retention.** A row is deleted when its device leaves the device registry;
  Home Assistant restores a removed device's registry id when the same
  hardware returns, so a device removed and paired again is new only if a run
  saw it gone in between.
* **Writes only on change.** ``last_seen`` refreshes at most daily.
"""

from __future__ import annotations

import logging
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

    from custom_components.home_generative_agent.snapshot.schema import RadioDevice

LOGGER = logging.getLogger(__name__)

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_network_inventory"

SOURCE_LABELS: dict[str, str] = {
    "zigbee": "Zigbee",
    "zwave": "Z-Wave",
    "bluetooth": "Bluetooth",
    "matter": "Matter",
}

# A known device's last_seen stamp is refreshed at most this often, so a
# quiet install does not rewrite the file every cycle.
LAST_SEEN_REFRESH = timedelta(days=1)

_ROW_FIELDS = ("platform", "ha_device_id", "name", "manufacturer", "model")


def device_key(device: RadioDevice) -> str:
    """Return the inventory key of a snapshot radio device."""
    return f"{device['protocol']}:{device['device_id']}"


@dataclass(frozen=True)
class InventoryDelta:
    """What changed since the inventory was last committed."""

    # Keys of devices on an already-bootstrapped source whose alert is still
    # owed: not recorded yet, or recorded with the alert pending.
    new_device_keys: list[str] = field(default_factory=list)
    # Sources present now that have never been recorded; their devices are
    # recorded as trusted by the next commit, without alerting.
    bootstrap_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BootstrapSummary:
    """One source recorded for the first time by a commit."""

    source: str
    device_count: int


def _empty_data() -> dict[str, Any]:
    return {"sources": {}, "devices": {}}


def _valid_mapping(raw: Any) -> dict[str, Any]:
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


class NetworkInventory:
    """Persist the radio devices known to this home."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the store."""
        self._store: Store[dict[str, Any]] = Store(
            hass, STORE_VERSION, STORE_KEY, private=True, atomic_writes=True
        )
        self._data: dict[str, Any] = _empty_data()
        # A save that failed leaves memory ahead of disk; retry on the next
        # commit and withhold the bootstrap announcement until it lands.
        self._dirty = False
        self._pending_announcements: list[BootstrapSummary] = []

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    async def async_load(self) -> None:
        """Load the inventory from storage (missing or corrupt -> empty)."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Network inventory unreadable; starting empty.")
            return
        if not isinstance(data, dict):
            return
        sources = {
            key: value
            for key, value in _valid_mapping(data.get("sources")).items()
            if isinstance(value, str)
        }
        devices = {
            key: dict(value)
            for key, value in _valid_mapping(data.get("devices")).items()
            if isinstance(value, dict) and isinstance(value.get("source"), str)
        }
        self._data = {"sources": sources, "devices": devices}

    async def async_save(self) -> bool:
        """Persist the inventory; return False when the write failed."""
        try:
            await self._store.async_save(self._data)
        except (HomeAssistantError, OSError, ValueError):
            LOGGER.warning("Network inventory could not be saved; will retry.")
            return False
        return True

    async def async_reset(self) -> bool:
        """
        Clear the inventory; the next run bootstraps every source again.

        Returns False when the stored file could not be removed, so the caller
        can say the old inventory would come back after a restart.
        """
        self._data = _empty_data()
        self._dirty = False
        self._pending_announcements = []
        try:
            await self._store.async_remove()
        except (HomeAssistantError, OSError):
            LOGGER.warning(
                "Network inventory file could not be removed; it will be "
                "reloaded after a restart."
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    # Read access
    # ------------------------------------------------------------------ #

    def is_source_bootstrapped(self, source: str) -> bool:
        """Return True once *source* has been recorded."""
        return source in self._data["sources"]

    def summary(self) -> dict[str, Any]:
        """Return counts for the audit report, services, and notifications."""
        by_source: dict[str, dict[str, int]] = {}
        trusted = 0
        for row in self._data["devices"].values():
            counts = by_source.setdefault(
                str(row["source"]), {"trusted": 0, "untrusted": 0}
            )
            if row.get("trusted"):
                counts["trusted"] += 1
                trusted += 1
            else:
                counts["untrusted"] += 1
        total = len(self._data["devices"])
        return {
            "device_count": total,
            "trusted": trusted,
            "untrusted": total - trusted,
            "by_source": dict(sorted(by_source.items())),
            "sources": dict(sorted(self._data["sources"].items())),
        }

    def list_devices(self) -> list[dict[str, Any]]:
        """Return every row with its key, ordered by source then name."""
        rows = [{"key": key, **row} for key, row in self._data["devices"].items()]
        rows.sort(key=lambda r: (r["source"], str(r.get("name") or ""), r["key"]))
        return rows

    # ------------------------------------------------------------------ #
    # Diff / commit
    # ------------------------------------------------------------------ #

    def diff(
        self, devices: Sequence[RadioDevice], present_sources: Iterable[str]
    ) -> InventoryDelta:
        """
        Compare *devices* against the stored inventory.

        Pure. ``present_sources`` names sources whose integration exists even
        when it has no devices yet, so a new stick bootstraps on its own.
        """
        known: dict[str, Any] = self._data["devices"]
        sources = {d["protocol"] for d in devices} | set(present_sources)
        bootstrap = sorted(s for s in sources if not self.is_source_bootstrapped(s))
        new_keys = sorted(
            device_key(d)
            for d in devices
            if self.is_source_bootstrapped(d["protocol"])
            and (
                device_key(d) not in known
                or known[device_key(d)].get("alerted", True) is False
            )
        )
        return InventoryDelta(new_device_keys=new_keys, bootstrap_sources=bootstrap)

    async def async_commit(
        self,
        devices: Sequence[RadioDevice],
        now: datetime,
        *,
        present_sources: Iterable[str] = (),
        alerted: Iterable[str] = (),
    ) -> list[BootstrapSummary]:
        """
        Record *devices* as the known state and persist it when changed.

        ``alerted`` names device keys whose pending alert is settled (delivered
        or stopped for good). Returns the sources whose bootstrap reached disk
        with this call (for the one-time announcement), including
        announcements held from a failed save.
        """
        now_iso = dt_util.as_utc(now).isoformat()
        stored_devices: dict[str, Any] = self._data["devices"]
        sources: dict[str, str] = self._data["sources"]
        settled = set(alerted)
        changed = False

        present = {d["protocol"] for d in devices} | set(present_sources)
        bootstrapped_now: list[str] = []
        for source in sorted(present):
            if source not in sources:
                sources[source] = now_iso
                bootstrapped_now.append(source)
                changed = True

        seen: set[str] = set()
        for device in devices:
            key = device_key(device)
            seen.add(key)
            observed = {
                "platform": device["platform"],
                "ha_device_id": device["device_id"],
                "name": device.get("name"),
                "manufacturer": device.get("manufacturer"),
                "model": device.get("model"),
            }
            row: dict[str, Any] | None = stored_devices.get(key)
            bootstrap_row = device["protocol"] in bootstrapped_now
            if row is None:
                stored_devices[key] = {
                    "source": device["protocol"],
                    **observed,
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    # Devices recorded by their source's bootstrap predate the
                    # audit and are trusted; later ones wait for the user.
                    "trusted": bootstrap_row,
                    "alerted": bootstrap_row or key in settled,
                }
                changed = True
                continue
            if row.get("alerted", True) is False and key in settled:
                row["alerted"] = True
                changed = True
            if any(row.get(k) != observed[k] for k in _ROW_FIELDS):
                row.update(observed)
                changed = True
            last = dt_util.parse_datetime(str(row.get("last_seen") or ""))
            if (
                last is None
                or dt_util.as_utc(now) - dt_util.as_utc(last) >= LAST_SEEN_REFRESH
            ):
                row["last_seen"] = now_iso
                changed = True

        for key in list(stored_devices):
            if key not in seen:
                del stored_devices[key]
                changed = True

        announcements = [
            BootstrapSummary(
                source=source,
                device_count=sum(1 for d in devices if d["protocol"] == source),
            )
            for source in bootstrapped_now
        ]
        if not (changed or self._dirty):
            return []
        saved = await self.async_save()
        self._dirty = not saved
        if not saved:
            self._pending_announcements.extend(announcements)
            return []
        result = [*self._pending_announcements, *announcements]
        self._pending_announcements = []
        return result

    async def async_set_trusted(
        self, device_ids: Iterable[str], *, trusted: bool
    ) -> list[str]:
        """
        Mark the rows for *device_ids* (registry ids or keys) trusted or not.

        Returns the keys whose flag changed. Unknown ids are ignored: a device
        is recorded by the next Sentinel run, not by this call. Trusting a
        device also settles its pending alert: the user has recognized it.
        Raises ``HomeAssistantError`` when the change could not be saved (it
        stays in memory and is retried by the next commit).
        """
        wanted = set(device_ids)
        changed: list[str] = []
        dirty = False
        for key, row in self._data["devices"].items():
            if key not in wanted and row.get("ha_device_id") not in wanted:
                continue
            if trusted and row.get("alerted", True) is False:
                row["alerted"] = True
                dirty = True
            if bool(row.get("trusted")) != trusted:
                row["trusted"] = trusted
                changed.append(key)
        if (changed or dirty) and not await self.async_save():
            self._dirty = True
            msg = "The device inventory could not be saved; try again."
            raise HomeAssistantError(msg)
        return sorted(changed)
