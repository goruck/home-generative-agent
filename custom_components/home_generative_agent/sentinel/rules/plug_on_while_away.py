"""Rule: plugs on the user's list are still on after everyone left."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.util import dt as dt_util

from custom_components.home_generative_agent.const import (
    RECOMMENDED_SENTINEL_AWAY_PLUG_DELAY_MIN as DEFAULT_DELAY_MIN,
)
from custom_components.home_generative_agent.sentinel.models import (
    AnomalyFinding,
    FindingAction,
    build_anomaly_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
        SnapshotEntity,
    )

LOGGER = logging.getLogger(__name__)

_PLUG_DOMAIN = "switch"
_HOME = "home"
_NOT_KNOWN = frozenset({STATE_UNAVAILABLE, STATE_UNKNOWN})


class PlugOnWhileAwayRule:
    """
    Report listed plugs that are on once everyone has been away a while.

    The first rule whose finding names its own action: ``switch.turn_off`` on
    those plugs (#789). Sentinel runs it by itself only when automatic actions
    are on and every guardrail passes; otherwise the finding is an alert.

    One finding per run names every listed plug that is on, so several plugs
    are one alert (and one action) rather than alerts a type cooldown spaces
    out. Nothing off the list is reported, an empty list (the default) turns
    the rule off, and entries that are not ``switch`` entities are ignored. A
    plug the user excluded for this rule is left out of the finding, not the
    whole finding dropped.

    A trip starts when every person is known to be somewhere other than
    ``home`` (a named zone counts as away) and ends when someone is home. A
    person who is ``unknown`` or ``unavailable`` pauses the trip: nothing is
    reported meanwhile, and the trip goes on once they are known to be away
    again, so a tracker blip does not start a new trip and a new alert. A
    home with no person entities is never known to be empty.

    The trip's start is the latest change among the person entities when the
    rule first sees the home empty: the last one to leave. After a Home
    Assistant restart that is the restart time, so the time away is a lower
    bound; the alert says "at least". A plug switched on after everyone left
    is reported like one left on: the home has been empty that long.
    """

    rule_id = "plug_on_while_away"
    requires: frozenset[str] = frozenset()
    cooldown_minutes = 0

    def __init__(
        self,
        plugs: Iterable[object] = (),
        delay_min: int = DEFAULT_DELAY_MIN,
        is_entity_excluded: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Keep the user's plug list and how long the home must be empty."""
        self.plugs: frozenset[str] = frozenset(
            plug.lower()
            for plug in plugs
            if isinstance(plug, str) and plug.lower().startswith(f"{_PLUG_DOMAIN}.")
        )
        self._delay = timedelta(minutes=max(0, delay_min))
        self._is_entity_excluded = is_entity_excluded
        # When the current trip started, or None between trips.
        self._away_since: datetime | None = None
        self._warned_no_people = False

    def due_at(self) -> datetime | None:
        """Return when the current trip reaches the delay, or None between trips."""
        if not self.plugs or self._away_since is None:
            return None
        return self._away_since + self._delay

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return one finding for the listed plugs still on while away."""
        if not self.plugs:
            return []
        now = dt_util.as_utc(
            dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        )
        if not self._known_empty(snapshot, now) or self._away_since is None:
            return []
        away = now - self._away_since
        if away < self._delay:
            return []

        on = sorted(
            (
                entity
                for entity in snapshot["entities"]
                if entity["entity_id"] in self.plugs
                and entity["state"] == STATE_ON
                and not self._excluded(entity["entity_id"])
            ),
            key=lambda entity: entity["entity_id"],
        )
        if not on:
            return []
        return [self._finding(on, away)]

    def _known_empty(self, snapshot: FullStateSnapshot, now: datetime) -> bool:
        """
        Return True when everyone is known to be away, keeping the trip.

        Someone home, or no person entities at all, ends the trip. A person
        who is unknown or unavailable pauses it: False, the start kept.
        """
        people = [e for e in snapshot["entities"] if e["domain"] == "person"]
        if not people and not self._warned_no_people:
            self._warned_no_people = True
            LOGGER.warning(
                "Sentinel's plug rule watches %d plug(s) but the home has no "
                "person entities, so it can never tell that everyone is away.",
                len(self.plugs),
            )
        states = {person["state"] for person in people}
        if not people or _HOME in states:
            self._away_since = None
            return False
        if states & _NOT_KNOWN:
            return False
        if self._away_since is None:
            self._away_since = min(now, _last_departure(people) or now)
        return True

    def _excluded(self, entity_id: str) -> bool:
        return bool(
            self._is_entity_excluded
            and self._is_entity_excluded(entity_id, self.rule_id)
        )

    def _finding(self, on: list[SnapshotEntity], away: timedelta) -> AnomalyFinding:
        entity_ids = [entity["entity_id"] for entity in on]
        away_since = self._away_since or dt_util.utcnow()
        evidence = {
            "entity_ids": entity_ids,
            "friendly_names": [
                entity["friendly_name"] or entity["entity_id"] for entity in on
            ],
            "away_min": int(away.total_seconds() / 60),
            "threshold_min": int(self._delay.total_seconds() / 60),
            "away_since": away_since.isoformat(),
        }
        return AnomalyFinding(
            # One anomaly per set of plugs per trip: the minutes away change
            # every run, and a renamed area or plug is the same anomaly.
            anomaly_id=build_anomaly_id(
                self.rule_id,
                entity_ids,
                {"entity_ids": entity_ids, "away_since": evidence["away_since"]},
            ),
            type=self.rule_id,
            severity="medium",
            confidence=0.9,
            triggering_entities=entity_ids,
            evidence=evidence,
            suggested_actions=["turn_off_plug"],
            is_sensitive=False,
            actions=(FindingAction("switch.turn_off", tuple(entity_ids)),),
            standing_condition=True,
        )


def _last_departure(people: list[SnapshotEntity]) -> datetime | None:
    """Return the latest change among the person entities, if any parses."""
    changed = [
        dt_util.as_utc(parsed)
        for person in people
        if (parsed := dt_util.parse_datetime(person["last_changed"] or "")) is not None
    ]
    return max(changed) if changed else None
