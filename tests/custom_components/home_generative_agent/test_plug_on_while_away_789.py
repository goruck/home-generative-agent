# ruff: noqa: S101
"""
A listed plug still on after everyone left (#789).

The first rule whose finding names its own action. Automatic actions are
still forced off, so what users get is the alert; these tests also pin what
the action gate will see once they are on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.const import (
    ACTION_POLICY_AUTO_EXECUTE,
    ACTION_POLICY_PROMPT_USER,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES,
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED,
    CONF_SENTINEL_AWAY_PLUG_DELAY_MIN,
    CONF_SENTINEL_AWAY_PLUGS,
    CONF_SENTINEL_DAILY_DIGEST_ENABLED,
    CONF_SENTINEL_DAILY_DIGEST_TIME,
    CONF_SENTINEL_ENABLED,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.sentinel.execution import (
    SentinelExecutionService,
)
from custom_components.home_generative_agent.sentinel.models import FindingAction
from custom_components.home_generative_agent.sentinel.notifier import (
    _plug_on_while_away_mobile_message,
)
from custom_components.home_generative_agent.sentinel.rules.plug_on_while_away import (
    PlugOnWhileAwayRule,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

from .test_sentinel_autonomy_settings import _settings
from .test_subentries import DummyEntry, DummySubentry, _make_sentinel_flow_for_mode

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_START = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
_DRIER = "switch.drier"


def _entity(entity_id: str, state: str, changed: datetime) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "domain": entity_id.split(".", 1)[0],
        "state": state,
        "friendly_name": entity_id.split(".", 1)[1].replace("_", " ").title(),
        "area": "Utility",
        "attributes": {},
        "last_changed": changed.isoformat(),
        "last_updated": changed.isoformat(),
    }


def _snapshot(
    minutes: int,
    *,
    people: dict[str, str] | None = None,
    plugs: dict[str, str] | None = None,
) -> FullStateSnapshot:
    """Return the home *minutes* after 12:00; people and plugs changed at 09:00."""
    now = _START + timedelta(minutes=minutes)
    long_ago = _START - timedelta(hours=3)
    people = {"person.ann": "not_home"} if people is None else people
    plugs = {_DRIER: "on"} if plugs is None else plugs
    home = sorted(p for p, state in people.items() if state == "home")
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": now.isoformat(),
            "entities": [
                _entity(entity_id, state, long_ago)
                for entity_id, state in {**people, **plugs}.items()
            ],
            "camera_activity": [],
            "derived": {
                "now": now.isoformat(),
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": bool(home),
                "people_home": home,
                "people_away": sorted(set(people) - set(home)),
                "last_motion_by_area": {},
            },
        }
    )


def _rule(delay_min: int = 10, plugs: Any = (_DRIER,)) -> PlugOnWhileAwayRule:
    return PlugOnWhileAwayRule(plugs=plugs, delay_min=delay_min)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_listed_plug_still_on_is_reported_once_the_home_was_empty_long_enough() -> (
    None
):
    """Not before the delay; the time away is counted from the first empty run."""
    rule = _rule(delay_min=10)

    assert rule.evaluate(_snapshot(0)) == []
    assert rule.evaluate(_snapshot(9)) == []
    findings = rule.evaluate(_snapshot(10))

    assert [f.type for f in findings] == ["plug_on_while_away"]
    finding = findings[0]
    assert finding.triggering_entities == [_DRIER]
    assert finding.evidence["away_min"] == 10
    assert finding.evidence["threshold_min"] == 10
    assert finding.is_sensitive is False


def test_the_finding_names_its_own_action_and_only_its_own_plug() -> None:
    """The one call Sentinel may make: turn that plug off."""
    rule = _rule(delay_min=0, plugs=(_DRIER, "switch.iron"))
    plugs = {_DRIER: "on", "switch.iron": "on", "switch.freezer": "on"}

    findings = rule.evaluate(_snapshot(0, plugs=plugs))

    assert {f.triggering_entities[0] for f in findings} == {_DRIER, "switch.iron"}
    for finding in findings:
        assert finding.actions == (
            FindingAction("switch.turn_off", (finding.triggering_entities[0],)),
        )
        assert finding.standing_condition is True


def test_nothing_off_the_list_is_ever_reported() -> None:
    """The list is the opt-in; an empty one turns the rule off."""
    plugs = {_DRIER: "on", "switch.freezer": "on"}

    assert _rule(delay_min=0, plugs=()).evaluate(_snapshot(0, plugs=plugs)) == []
    listed = _rule(delay_min=0).evaluate(_snapshot(0, plugs=plugs))
    assert [f.triggering_entities for f in listed] == [[_DRIER]]


@pytest.mark.parametrize("state", ["off", "unavailable", "unknown"])
def test_a_plug_that_is_not_on_is_not_reported(state: str) -> None:
    """Only a plug Home Assistant says is on."""
    assert _rule(delay_min=0).evaluate(_snapshot(0, plugs={_DRIER: state})) == []


@pytest.mark.parametrize(
    "people",
    [
        {},  # a home with no person entities is not known to be empty
        {"person.ann": "home"},
        {"person.ann": "not_home", "person.bob": "home"},
        {"person.ann": "not_home", "person.bob": "unknown"},
        {"person.ann": "not_home", "person.bob": "unavailable"},
    ],
)
def test_the_home_must_be_known_to_be_empty(people: dict[str, str]) -> None:
    """A tracker that is unknown or unavailable is not a person who left."""
    assert _rule(delay_min=0).evaluate(_snapshot(0, people=people)) == []


def test_a_named_zone_counts_as_away() -> None:
    """A person at "work" is away; only `home` is home."""
    people = {"person.ann": "work", "person.bob": "not_home"}
    assert len(_rule(delay_min=0).evaluate(_snapshot(0, people=people))) == 1


def test_someone_coming_home_ends_the_episode() -> None:
    """The next time everyone leaves, the delay starts again."""
    rule = _rule(delay_min=10)
    rule.evaluate(_snapshot(0))
    assert rule.evaluate(_snapshot(10))

    assert rule.evaluate(_snapshot(11, people={"person.ann": "home"})) == []
    assert rule.evaluate(_snapshot(12)) == []
    assert rule.evaluate(_snapshot(21)) == []
    assert rule.evaluate(_snapshot(22))


def test_an_unknown_tracker_ends_the_episode_too() -> None:
    """Minutes with an unknown tracker do not count as minutes away."""
    rule = _rule(delay_min=10)
    rule.evaluate(_snapshot(0))

    assert rule.evaluate(_snapshot(5, people={"person.ann": "unknown"})) == []
    assert rule.evaluate(_snapshot(10)) == []


def test_the_same_episode_is_the_same_anomaly() -> None:
    """One alert per plug per time everyone leaves, not one per run."""
    rule = _rule(delay_min=0)
    first = rule.evaluate(_snapshot(0))[0]
    later = rule.evaluate(_snapshot(30))[0]
    assert later.anomaly_id == first.anomaly_id
    assert later.evidence["away_min"] == 30

    rule.evaluate(_snapshot(31, people={"person.ann": "home"}))
    again = rule.evaluate(_snapshot(60))[0]
    assert again.anomaly_id != first.anomaly_id


@pytest.mark.parametrize(
    "plugs",
    [None, "switch.drier", [1, None], ["light.lamp", "sensor.drier_power"]],
)
def test_a_list_that_is_not_switches_watches_nothing(plugs: Any) -> None:
    """A hand-edited setting must not break the run or watch another domain."""
    rule = PlugOnWhileAwayRule(plugs=plugs or (), delay_min=0)
    snapshot = _snapshot(0, plugs={_DRIER: "on", "light.lamp": "on"})
    assert rule.evaluate(snapshot) == []


# ---------------------------------------------------------------------------
# The action gate: a standing condition is not stale data
# ---------------------------------------------------------------------------

_ACTING: dict[str, Any] = {
    CONF_SENTINEL_AUTO_EXECUTION_ENABLED: True,
    CONF_SENTINEL_AUTO_EXECUTE_ALLOWED_SERVICES: ["switch.turn_off"],
}


def test_a_plug_on_for_hours_is_not_stale_data() -> None:
    """How long it has been on is the condition, not the age of the reading."""
    finding = _rule(delay_min=0).evaluate(_snapshot(0))[0]
    svc = SentinelExecutionService(_ACTING)

    # The plug last changed three hours ago, far past the 30-minute threshold.
    result = svc.evaluate_canary(finding, _snapshot(0), 2, _START)

    assert result.data_quality == "fresh"
    assert result.action_policy_path == ACTION_POLICY_AUTO_EXECUTE


@pytest.mark.parametrize("state", ["unavailable", "unknown", None])
def test_a_plug_home_assistant_no_longer_hears_from_is_not_acted_on(
    state: str | None,
) -> None:
    """Found on, then gone before the gate: the alert is sent, nothing runs."""
    finding = _rule(delay_min=0).evaluate(_snapshot(0))[0]
    svc = SentinelExecutionService(_ACTING)
    later = _snapshot(1, plugs={} if state is None else {_DRIER: state})

    result = svc.evaluate_canary(finding, later, 2, _START)

    assert result.action_policy_path == ACTION_POLICY_PROMPT_USER
    assert result.block_reason == "data_quality_unavailable"


def test_an_ordinary_finding_is_still_judged_by_its_age() -> None:
    """Only a rule that says so skips the age test."""
    from dataclasses import replace  # noqa: PLC0415

    finding = replace(
        _rule(delay_min=0).evaluate(_snapshot(0))[0], standing_condition=False
    )
    svc = SentinelExecutionService(_ACTING)

    result = svc.evaluate_canary(finding, _snapshot(0), 2, _START)

    assert result.block_reason == "data_quality_stale"


# ---------------------------------------------------------------------------
# The alert text
# ---------------------------------------------------------------------------


def test_the_alert_names_the_plug_and_the_time_away() -> None:
    """Fixed copy: the figures come from the finding, not from a model."""
    rule = _rule(delay_min=10)
    rule.evaluate(_snapshot(0))
    finding = rule.evaluate(_snapshot(25))[0]

    assert _plug_on_while_away_mobile_message(finding) == (
        "Drier is still on and everyone has been away 25 min."
    )


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


def test_the_default_watches_nothing() -> None:
    """Upgrading turns nothing on."""
    options = resolve_runtime_options(cast("Any", DummyEntry()))
    assert options[CONF_SENTINEL_AWAY_PLUGS] == []
    assert options[CONF_SENTINEL_AWAY_PLUG_DELAY_MIN] == 10


@pytest.mark.asyncio
async def test_the_form_stores_the_plug_list_and_the_delay(
    hass: HomeAssistant,
) -> None:
    """Switches only, each once, and the delay as a whole number."""
    flow = _make_sentinel_flow_for_mode(hass, DummyEntry())
    result = await flow.async_step_settings(
        _settings(
            **{
                CONF_SENTINEL_AWAY_PLUGS: [_DRIER, "switch.iron", _DRIER],
                CONF_SENTINEL_AWAY_PLUG_DELAY_MIN: 15.0,
            }
        )
    )
    assert result.get("type") == "create_entry"
    data = result.get("data") or {}
    assert data[CONF_SENTINEL_AWAY_PLUGS] == [_DRIER, "switch.iron"]
    assert data[CONF_SENTINEL_AWAY_PLUG_DELAY_MIN] == 15


@pytest.mark.asyncio
async def test_emptying_the_plug_list_turns_the_rule_off(hass: HomeAssistant) -> None:
    """An emptied selector sends no key; the stored list must not come back."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True, CONF_SENTINEL_AWAY_PLUGS: [_DRIER]},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    updates: list[Any] = []
    flow.async_update_and_abort = lambda *_a, **kw: (  # type: ignore[assignment]
        updates.append(kw) or {"type": "abort", "reason": "reconfigure_successful"}
    )
    submitted = _settings()
    submitted.pop(CONF_SENTINEL_AWAY_PLUGS, None)

    await flow.async_step_settings(submitted)

    assert updates[0]["data"][CONF_SENTINEL_AWAY_PLUGS] == []


@pytest.mark.asyncio
async def test_basic_setup_keeps_the_plug_list(hass: HomeAssistant) -> None:
    """The list is what an automatic action may touch: a re-run keeps it."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_AWAY_PLUGS: [_DRIER],
            CONF_SENTINEL_AWAY_PLUG_DELAY_MIN: 25,
        },
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    updates: list[Any] = []
    flow.async_update_and_abort = lambda *_a, **kw: (  # type: ignore[assignment]
        updates.append(kw) or {"type": "abort", "reason": "reconfigure_successful"}
    )
    await flow.async_step_setup_mode({"setup_mode": "basic"})
    await flow.async_step_basic_settings(
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_DAILY_DIGEST_ENABLED: False,
            CONF_SENTINEL_DAILY_DIGEST_TIME: "08:00:00",
        }
    )
    assert updates[0]["data"][CONF_SENTINEL_AWAY_PLUGS] == [_DRIER]
    assert updates[0]["data"][CONF_SENTINEL_AWAY_PLUG_DELAY_MIN] == 25
