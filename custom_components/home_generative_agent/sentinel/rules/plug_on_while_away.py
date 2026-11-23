"""Rule: a plug on the user's list is still on after everyone left."""

from __future__ import annotations

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
    from collections.abc import Iterable
    from datetime import datetime

    from custom_components.home_generative_agent.snapshot.schema import (
        FullStateSnapshot,
    )

_PLUG_DOMAIN = "switch"
_NOT_KNOWN = frozenset({STATE_UNAVAILABLE, STATE_UNKNOWN})
# Evidence left out of the anomaly id: the name is cosmetic and the minutes
# away change every run, which would make each run a new anomaly.
_UNSTABLE_EVIDENCE_KEYS = frozenset({"friendly_name", "away_min"})


class PlugOnWhileAwayRule:
    """
    Report a listed plug that is on once everyone has been away a while.

    The first rule whose finding names its own action: ``switch.turn_off`` on
    that plug (#789). Sentinel runs it by itself only when automatic actions
    are on and every guardrail passes; otherwise the finding is an alert.

    Nothing off the list is ever reported, and an empty list (the default)
    turns the rule off. Entries that are not ``switch`` entities are ignored.

    "Everyone away" needs at least one person entity, and every person in a
    known state other than ``home``: a home with no person entities, or one
    whose tracker is unavailable or unknown, is not known to be empty. The
    time away is measured by observation, from the first run that saw the
    home empty, so it restarts with the engine; a person arriving, or one
    becoming unknown, ends the episode. A plug switched on after everyone
    left is reported like one left on: the home has been empty that long.
    """

    rule_id = "plug_on_while_away"
    requires: frozenset[str] = frozenset()
    cooldown_minutes = 0

    def __init__(
        self,
        plugs: Iterable[object] = (),
        delay_min: int = DEFAULT_DELAY_MIN,
    ) -> None:
        """Keep the user's plug list and how long the home must be empty."""
        self._plugs = frozenset(
            plug.lower()
            for plug in plugs
            if isinstance(plug, str) and plug.lower().startswith(f"{_PLUG_DOMAIN}.")
        )
        self._delay_min = max(0, delay_min)
        # When the home was first seen empty in the current episode.
        self._away_since: datetime | None = None

    def evaluate(self, snapshot: FullStateSnapshot) -> list[AnomalyFinding]:
        """Return a finding for each listed plug still on while away."""
        if not self._plugs:
            return []
        now = dt_util.as_utc(
            dt_util.parse_datetime(snapshot["derived"]["now"]) or dt_util.utcnow()
        )
        if not _everyone_away(snapshot):
            self._away_since = None
            return []
        if self._away_since is None:
            self._away_since = now
        away_since = self._away_since
        away = now - away_since
        if away < timedelta(minutes=self._delay_min):
            return []

        findings: list[AnomalyFinding] = []
        for entity in snapshot["entities"]:
            entity_id = entity["entity_id"]
            if entity_id not in self._plugs or entity["state"] != STATE_ON:
                continue
            evidence = {
                "entity_id": entity_id,
                "area": entity["area"],
                "friendly_name": entity["friendly_name"] or None,
                "away_min": int(away.total_seconds() / 60),
                "threshold_min": self._delay_min,
                "away_since": away_since.isoformat(),
            }
            findings.append(
                AnomalyFinding(
                    anomaly_id=build_anomaly_id(
                        self.rule_id,
                        [entity_id],
                        {
                            key: value
                            for key, value in evidence.items()
                            if key not in _UNSTABLE_EVIDENCE_KEYS
                        },
                    ),
                    type=self.rule_id,
                    severity="medium",
                    confidence=0.9,
                    triggering_entities=[entity_id],
                    evidence=evidence,
                    suggested_actions=["turn_off_plug"],
                    is_sensitive=False,
                    actions=(FindingAction("switch.turn_off", (entity_id,)),),
                    standing_condition=True,
                )
            )
        return findings


def _everyone_away(snapshot: FullStateSnapshot) -> bool:
    """Return True when every person is known to be somewhere else."""
    people = [e for e in snapshot["entities"] if e["domain"] == "person"]
    return bool(people) and all(
        person["state"] != "home" and person["state"] not in _NOT_KNOWN
        for person in people
    )
