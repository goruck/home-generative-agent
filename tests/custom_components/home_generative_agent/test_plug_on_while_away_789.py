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
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

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
    CONF_SENTINEL_QUIET_HOURS_START,
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
    SUBENTRY_TYPE_SENTINEL,
)
from custom_components.home_generative_agent.core.subentry_resolver import (
    resolve_runtime_options,
)
from custom_components.home_generative_agent.sentinel.engine import SentinelEngine
from custom_components.home_generative_agent.sentinel.execution import (
    SentinelExecutionService,
)
from custom_components.home_generative_agent.sentinel.models import FindingAction
from custom_components.home_generative_agent.sentinel.notifier import (
    _persistent_message,
    _plug_on_while_away_mobile_message,
)
from custom_components.home_generative_agent.sentinel.rules.plug_on_while_away import (
    PlugOnWhileAwayRule,
)
from custom_components.home_generative_agent.sentinel.suppression import (
    SuppressionState,
    register_finding,
    should_suppress,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    validate_snapshot,
)

from .test_sentinel_autonomy_level import DummyAudit, DummyNotifier, DummySuppression
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
    left: int = 0,
    start: datetime = _START,
) -> FullStateSnapshot:
    """
    Return the home *minutes* after *start*.

    The people last changed *left* minutes after *start* (when they left, for
    people away); the plugs three hours before it.
    """
    now = start + timedelta(minutes=minutes)
    people = {"person.ann": "not_home"} if people is None else people
    plugs = {_DRIER: "on"} if plugs is None else plugs
    home = sorted(p for p, state in people.items() if state == "home")
    entities = [
        _entity(person, state, start + timedelta(minutes=left))
        for person, state in people.items()
    ] + [
        _entity(plug, state, start - timedelta(hours=3))
        for plug, state in plugs.items()
    ]
    return validate_snapshot(
        {
            "schema_version": 1,
            "generated_at": now.isoformat(),
            "entities": entities,
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


def _rule(
    delay_min: int = 10, plugs: Any = (_DRIER,), **kwargs: Any
) -> PlugOnWhileAwayRule:
    return PlugOnWhileAwayRule(plugs=plugs, delay_min=delay_min, **kwargs)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_listed_plug_still_on_is_reported_once_the_home_was_empty_long_enough() -> (
    None
):
    """Not before the delay, counted from when the last person left."""
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


def test_the_trip_starts_when_the_last_person_left_not_when_sentinel_looked() -> None:
    """After a reload the rule knows the home has been empty for hours."""
    rule = _rule(delay_min=10)
    people = {"person.ann": "not_home", "person.bob": "work"}

    findings = rule.evaluate(_snapshot(180, people=people, left=0))

    assert findings[0].evidence["away_min"] == 180
    assert findings[0].evidence["away_since"] == _START.isoformat()


def test_the_trip_starts_with_the_last_person_to_leave() -> None:
    """Not the first: the home was occupied until the last one went."""
    snapshot = _snapshot(30, people={"person.ann": "not_home"}, left=0)
    snapshot["entities"].append(
        cast("Any", _entity("person.bob", "work", _START + timedelta(minutes=25)))
    )
    rule = _rule(delay_min=10)

    assert rule.evaluate(snapshot) == []
    assert rule.due_at() == _START + timedelta(minutes=35)


def test_a_last_change_in_the_future_starts_the_trip_now() -> None:
    """Clock skew must not make the home empty for longer than it was."""
    rule = _rule(delay_min=0)
    findings = rule.evaluate(_snapshot(0, left=90))
    assert findings[0].evidence["away_since"] == _START.isoformat()


def test_a_person_with_no_tracker_is_named_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Home Assistant holds them unknown for good, which pauses every trip."""
    snapshot = _snapshot(0, people={"person.ann": "not_home", "person.kid": "unknown"})
    for entity in snapshot["entities"]:
        if entity["entity_id"] == "person.kid":
            entity["attributes"] = {"device_trackers": []}
    rule = _rule(delay_min=0)
    with caplog.at_level("WARNING"):
        assert rule.evaluate(snapshot) == []
        assert rule.evaluate(snapshot) == []
    assert caplog.text.count("person.kid has no device tracker") == 1


def test_one_area_is_kept_in_the_evidence() -> None:
    """For correlation with the room's other findings and the explanation."""
    finding = _rule(delay_min=0).evaluate(_snapshot(0))[0]
    assert finding.evidence["area"] == "Utility"
    assert finding.evidence["areas"] == ["Utility"]


def test_several_plugs_are_one_alert_and_one_action() -> None:
    """A type cooldown must not space them out: one finding names them all."""
    rule = _rule(delay_min=0, plugs=(_DRIER, "switch.iron"))
    plugs = {_DRIER: "on", "switch.iron": "on", "switch.freezer": "on"}

    findings = rule.evaluate(_snapshot(0, plugs=plugs))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.triggering_entities == [_DRIER, "switch.iron"]
    assert finding.actions == (
        FindingAction("switch.turn_off", (_DRIER, "switch.iron")),
    )
    assert finding.standing_condition is True


def test_each_area_gets_its_own_alert_for_its_own_recipient() -> None:
    """Plugs in one room share an alert; another room gets its own."""
    snapshot = _snapshot(
        0, plugs={_DRIER: "on", "switch.iron": "on", "switch.kettle": "on"}
    )
    for entity in snapshot["entities"]:
        if entity["entity_id"] == "switch.kettle":
            entity["area"] = "Kitchen"
    rule = _rule(delay_min=0, plugs=(_DRIER, "switch.iron", "switch.kettle"))

    findings = rule.evaluate(snapshot)

    assert sorted(
        (f.evidence.get("area"), f.triggering_entities) for f in findings
    ) == [("Kitchen", ["switch.kettle"]), ("Utility", [_DRIER, "switch.iron"])]
    assert len({f.anomaly_id for f in findings}) == 2


def test_a_second_areas_alert_is_not_held_by_the_type_cooldown() -> None:
    """Held by each plug's own cooldown, so areas are not spaced out."""
    snapshot = _snapshot(0, plugs={_DRIER: "on", "switch.kettle": "on"})
    for entity in snapshot["entities"]:
        if entity["entity_id"] == "switch.kettle":
            entity["area"] = "Kitchen"
    state = SuppressionState()
    first, second = _rule(delay_min=0, plugs=(_DRIER, "switch.kettle")).evaluate(
        snapshot
    )
    register_finding(state, first, _START)

    decision = should_suppress(
        state, second, _START, timedelta(minutes=30), timedelta(minutes=30)
    )

    assert not decision.suppress


def test_nothing_off_the_list_is_ever_reported() -> None:
    """The list is the opt-in; an empty one turns the rule off."""
    plugs = {_DRIER: "on", "switch.freezer": "on"}

    assert _rule(delay_min=0, plugs=()).evaluate(_snapshot(0, plugs=plugs)) == []
    assert _rule(delay_min=0, plugs=()).due_at() is None
    listed = _rule(delay_min=0).evaluate(_snapshot(0, plugs=plugs))
    assert [f.triggering_entities for f in listed] == [[_DRIER]]


def test_an_excluded_plug_is_left_out_not_the_whole_alert() -> None:
    """The engine drops a finding when any of its devices is excluded."""
    excluded = {"switch.iron"}
    rule = _rule(
        delay_min=0,
        plugs=(_DRIER, "switch.iron"),
        is_entity_excluded=lambda entity_id, _type: entity_id in excluded,
    )
    plugs = {_DRIER: "on", "switch.iron": "on"}

    findings = rule.evaluate(_snapshot(0, plugs=plugs))

    assert [f.triggering_entities for f in findings] == [[_DRIER]]


@pytest.mark.parametrize("state", ["off", "unavailable", "unknown"])
def test_a_plug_that_is_not_on_is_not_reported(state: str) -> None:
    """Only a plug Home Assistant says is on."""
    assert _rule(delay_min=0).evaluate(_snapshot(0, plugs={_DRIER: state})) == []


@pytest.mark.parametrize(
    "people",
    [
        {"person.ann": "home"},
        {"person.ann": "not_home", "person.bob": "home"},
        {"person.ann": "not_home", "person.bob": "unknown"},
        {"person.ann": "not_home", "person.bob": "unavailable"},
    ],
)
def test_the_home_must_be_known_to_be_empty(people: dict[str, str]) -> None:
    """A tracker that is unknown or unavailable is not a person who left."""
    assert _rule(delay_min=0).evaluate(_snapshot(0, people=people)) == []


def test_a_home_with_no_people_is_never_empty_and_says_so_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Otherwise the rule would do nothing with nothing in the log."""
    rule = _rule(delay_min=0)
    with caplog.at_level("WARNING"):
        assert rule.evaluate(_snapshot(0, people={})) == []
        assert rule.evaluate(_snapshot(5, people={})) == []
    assert caplog.text.count("no person entities") == 1


def test_a_named_zone_counts_as_away() -> None:
    """A person at "work" is away; only `home` is home."""
    people = {"person.ann": "work", "person.bob": "not_home"}
    assert len(_rule(delay_min=0).evaluate(_snapshot(0, people=people))) == 1


def test_someone_coming_home_ends_the_trip() -> None:
    """The next time everyone leaves, the delay starts again from then."""
    rule = _rule(delay_min=10)
    rule.evaluate(_snapshot(0))
    assert rule.evaluate(_snapshot(10))

    assert rule.evaluate(_snapshot(11, people={"person.ann": "home"}, left=11)) == []
    assert rule.due_at() is None
    assert rule.evaluate(_snapshot(12, left=12)) == []
    assert rule.evaluate(_snapshot(21, left=12)) == []
    assert rule.evaluate(_snapshot(22, left=12))


def test_an_unknown_tracker_pauses_the_trip() -> None:
    """A blip reports nothing meanwhile and does not start a new trip."""
    rule = _rule(delay_min=10)
    first = rule.evaluate(_snapshot(0)) or rule.evaluate(_snapshot(10))

    assert rule.evaluate(_snapshot(15, people={"person.ann": "unknown"}, left=15)) == []
    again = rule.evaluate(_snapshot(16, left=16))

    assert again[0].anomaly_id == first[0].anomaly_id
    assert again[0].evidence["away_min"] == 16


def test_the_same_trip_is_the_same_anomaly() -> None:
    """One alert per set of plugs per trip, not one per run."""
    rule = _rule(delay_min=0)
    first = rule.evaluate(_snapshot(0))[0]
    later = rule.evaluate(_snapshot(30))[0]
    assert later.anomaly_id == first.anomaly_id
    assert later.evidence["away_min"] == 30

    rule.evaluate(_snapshot(31, people={"person.ann": "home"}, left=31))
    again = rule.evaluate(_snapshot(60, left=32))[0]
    assert again.anomaly_id != first.anomaly_id


def test_the_rule_says_when_its_wait_ends() -> None:
    """For the engine's one-shot wake-up."""
    rule = _rule(delay_min=10)
    assert rule.due_at() is None
    rule.evaluate(_snapshot(3))
    assert rule.due_at() == _START + timedelta(minutes=10)


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
        "Drier is still on and everyone has been away at least 25 min."
    )


def test_a_long_list_of_names_never_cuts_off_the_time_away() -> None:
    """Fewer names and a count, not the end of the sentence."""
    plugs = {f"switch.plug_with_a_long_name_{i}": "on" for i in range(8)}
    finding = _rule(delay_min=0, plugs=tuple(plugs)).evaluate(
        _snapshot(25, plugs=plugs)
    )[0]
    msg = _plug_on_while_away_mobile_message(finding, cap=120)
    assert len(msg) <= 120
    assert msg.endswith("everyone has been away at least 25 min.")
    assert "+" in msg


def test_several_plugs_are_named_in_one_alert() -> None:
    """One push for all of them."""
    finding = _rule(delay_min=0, plugs=(_DRIER, "switch.iron")).evaluate(
        _snapshot(25, plugs={_DRIER: "on", "switch.iron": "on"})
    )[0]
    assert _plug_on_while_away_mobile_message(finding) == (
        "Drier, Iron are still on and everyone has been away at least 25 min."
    )


@pytest.mark.parametrize(
    ("language", "one", "two"),
    [
        (
            "cs",
            "Drier: stále zapnuto, všichni jsou pryč nejméně 25 min.",
            "Drier, Iron: stále zapnuto, všichni jsou pryč nejméně 25 min.",
        ),
        (
            "zh-Hans",
            "Drier仍开着，所有人已离家至少 25 分钟。",
            "Drier, Iron仍开着，所有人已离家至少 25 分钟。",
        ),
    ],
)
def test_the_alert_follows_the_home_assistant_language(
    language: str, one: str, two: str
) -> None:
    """The words around the names and figures are translated."""
    hass = MagicMock()
    hass.config.language = language
    single = _rule(delay_min=0).evaluate(_snapshot(25))[0]
    double = _rule(delay_min=0, plugs=(_DRIER, "switch.iron")).evaluate(
        _snapshot(25, plugs={_DRIER: "on", "switch.iron": "on"})
    )[0]
    assert _plug_on_while_away_mobile_message(single, hass) == one
    assert _plug_on_while_away_mobile_message(double, hass) == two


def test_the_persistent_record_keeps_the_figures_without_an_explanation() -> None:
    """With the explainer off, the record must not fall back to generic text."""
    finding = _rule(delay_min=0).evaluate(_snapshot(25))[0]
    assert _persistent_message(None, finding) == (
        "Drier is still on and everyone has been away at least 25 min."
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


@pytest.mark.asyncio
async def test_an_emptied_list_stays_empty_after_another_fields_error(
    hass: HomeAssistant,
) -> None:
    """The redisplay must not bring the stored plugs back."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {CONF_SENTINEL_ENABLED: True, CONF_SENTINEL_AWAY_PLUGS: [_DRIER]},
    )
    entry = DummyEntry()
    entry.subentries[existing.subentry_id] = existing
    flow = _make_sentinel_flow_for_mode(hass, entry)
    submitted = _settings(**{CONF_SENTINEL_QUIET_HOURS_START: "22"})
    submitted.pop(CONF_SENTINEL_AWAY_PLUGS, None)

    form = await flow.async_step_settings(submitted)

    assert cast("Any", form).get("errors") == {"base": "quiet_hours_incomplete"}
    schema = cast("Any", form).get("data_schema")
    suggested = {
        key.schema: key.description["suggested_value"]
        for key in schema.schema
        if key.description and "suggested_value" in key.description
    }
    assert suggested.get(CONF_SENTINEL_AWAY_PLUGS, []) == []
    # No default either: an emptied selector must stay empty.
    field = cast(
        "Any", next(key for key in schema.schema if key == CONF_SENTINEL_AWAY_PLUGS)
    )
    assert field.default is vol.UNDEFINED


@pytest.mark.asyncio
async def test_basic_setup_keeps_the_delay_with_no_plugs_picked(
    hass: HomeAssistant,
) -> None:
    """Emptying the list for a while must not reset the chosen wait."""
    existing = DummySubentry(
        "sent1",
        SUBENTRY_TYPE_SENTINEL,
        "Sentinel",
        {
            CONF_SENTINEL_ENABLED: True,
            CONF_SENTINEL_AWAY_PLUGS: [],
            CONF_SENTINEL_AWAY_PLUG_DELAY_MIN: 60,
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
    assert updates[0]["data"][CONF_SENTINEL_AWAY_PLUG_DELAY_MIN] == 60


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _engine(hass: Any, options: dict[str, object]) -> SentinelEngine:
    return SentinelEngine(
        hass=hass,
        options=options,
        suppression=DummySuppression(),
        notifier=cast("Any", DummyNotifier()),
        audit_store=cast("Any", DummyAudit()),
        explainer=None,
        entry_id="plug-entry",
    )


@pytest.mark.parametrize(
    ("options", "plugs", "delay"),
    [
        (
            {
                CONF_SENTINEL_AWAY_PLUGS: [_DRIER, "light.lamp"],
                CONF_SENTINEL_AWAY_PLUG_DELAY_MIN: 25,
            },
            {_DRIER},
            25,
        ),
        ({}, set(), 10),
        (
            {
                CONF_SENTINEL_AWAY_PLUGS: "switch.drier",
                CONF_SENTINEL_AWAY_PLUG_DELAY_MIN: "x",
            },
            set(),
            10,
        ),
    ],
)
def test_the_engine_builds_the_rule_from_the_settings(
    options: dict[str, object], plugs: set[str], delay: int
) -> None:
    """A wrong key here would make the rule silently do nothing."""
    engine = _engine(object(), options)
    rule = engine._plug_rule
    assert rule in engine._rules
    assert rule.plugs == plugs
    assert rule._delay == timedelta(minutes=delay)


def _state_event(entity_id: str, state: str, *, old: str = "off") -> Any:
    new_state = MagicMock()
    new_state.state = state
    new_state.attributes = {}
    old_state = MagicMock()
    old_state.state = old
    event = MagicMock()
    event.data = {
        "entity_id": entity_id,
        "new_state": new_state,
        "old_state": old_state,
    }
    return event


def test_an_excluded_plug_turning_on_does_not_wake_sentinel() -> None:
    """The #481 contract: an excluded entity wakes nothing."""
    engine = _engine(
        object(),
        {
            CONF_SENTINEL_AWAY_PLUGS: [_DRIER],
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {"plug_on_while_away": [_DRIER]},
        },
    )
    engine._on_state_changed(_state_event(_DRIER, "on"))
    assert engine._trigger_scheduler._queue == []


def test_a_listed_plug_turning_on_wakes_sentinel() -> None:
    """Not one off the list, not one turning off, not an update of one on."""
    engine = _engine(object(), {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    queue = engine._trigger_scheduler._queue

    engine._on_state_changed(_state_event("switch.freezer", "on"))
    engine._on_state_changed(_state_event(_DRIER, "off"))
    engine._on_state_changed(_state_event(_DRIER, "on", old="on"))
    assert queue == []

    engine._on_state_changed(_state_event(_DRIER, "on"))
    assert [record.anomaly_type for record in queue] == ["plug_on_while_away"]


@pytest.mark.asyncio
async def test_sentinel_wakes_when_the_wait_ends(hass: HomeAssistant) -> None:
    """One timer per trip, at the end of the wait, gone when someone is home."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    queue = engine._trigger_scheduler._queue
    start = dt_util.utcnow()

    engine._plug_rule.evaluate(_snapshot(0, start=start))
    engine._schedule_plug_wake()
    assert engine._plug_wake_at == start + timedelta(minutes=10)

    engine._plug_rule.evaluate(_snapshot(1, people={"person.ann": "home"}, start=start))
    engine._schedule_plug_wake()
    assert engine._plug_wake is None

    engine._plug_rule.evaluate(_snapshot(2, left=2, start=start))
    engine._schedule_plug_wake()
    due = start + timedelta(minutes=12)
    assert engine._plug_wake_at == due

    async_fire_time_changed(hass, due + timedelta(seconds=1))
    await hass.async_block_till_done()
    assert [record.anomaly_type for record in queue] == ["plug_on_while_away"]
    assert engine._plug_wake is None


@pytest.mark.asyncio
async def test_a_stopped_engine_leaves_no_timer(hass: HomeAssistant) -> None:
    """The timer must not outlive a reload."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    engine._plug_rule.evaluate(_snapshot(0, start=dt_util.utcnow()))
    engine._schedule_plug_wake()
    assert engine._plug_wake is not None

    await engine.stop()

    assert engine._plug_wake is None


def test_the_engine_passes_the_users_exclusions_to_the_rule() -> None:
    """An excluded plug is left out of the alert, as the rule tests show."""
    engine = _engine(
        object(),
        {
            CONF_SENTINEL_AWAY_PLUGS: [_DRIER, "switch.iron"],
            CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {
                "plug_on_while_away": ["switch.iron"]
            },
        },
    )
    assert engine._plug_rule._excluded("switch.iron")
    assert not engine._plug_rule._excluded(_DRIER)


@pytest.mark.asyncio
async def test_a_run_sets_the_timer(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wake-up comes from the run that saw everyone leave."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    cast("Any", engine)._suppression._read_only = False
    start = dt_util.utcnow()

    async def _fake_build(_hass: Any, **_kwargs: Any) -> FullStateSnapshot:
        return _snapshot(0, start=start)

    monkeypatch.setattr(
        "custom_components.home_generative_agent.sentinel.engine."
        "async_build_full_state_snapshot",
        _fake_build,
    )
    try:
        await engine._run_once()
        assert engine._plug_wake_at == start + timedelta(minutes=10)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_a_run_left_behind_by_a_stop_sets_no_timer(hass: HomeAssistant) -> None:
    """After stop() begins, a run still going must not arm a new wake."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    engine._plug_rule.evaluate(_snapshot(0, start=dt_util.utcnow()))
    engine._stopped = True

    engine._schedule_plug_wake()

    assert engine._plug_wake is None


@pytest.mark.asyncio
async def test_a_wake_that_fires_after_a_stop_does_nothing(hass: HomeAssistant) -> None:
    """The callback checks the stop itself, in case its cancel came too late."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    start = dt_util.utcnow()
    engine._plug_rule.evaluate(_snapshot(0, start=start))
    engine._schedule_plug_wake()
    engine._stopped = True

    async_fire_time_changed(hass, start + timedelta(minutes=10, seconds=1))
    await hass.async_block_till_done()

    assert engine._trigger_scheduler._queue == []


@pytest.mark.asyncio
async def test_a_wait_that_ended_while_the_run_was_busy_still_wakes(
    hass: HomeAssistant,
) -> None:
    """Due after the snapshot was taken but before scheduling: wake at once."""
    engine = _engine(hass, {CONF_SENTINEL_AWAY_PLUGS: [_DRIER]})
    start = dt_util.utcnow() - timedelta(minutes=10, seconds=30)
    # The snapshot was taken 9 min 59 s into the trip; the run ends after 10.
    engine._plug_rule._away_since = start
    engine._plug_rule.evaluated_at = start + timedelta(minutes=9, seconds=59)

    engine._schedule_plug_wake()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=1))
    await hass.async_block_till_done()

    assert [r.anomaly_type for r in engine._trigger_scheduler._queue] == [
        "plug_on_while_away"
    ]


def test_a_plug_that_comes_on_later_is_not_held_by_the_first_ones_cooldown() -> None:
    """One alert names several plugs; it is held only while all of them are."""
    state = SuppressionState()
    rule = _rule(delay_min=0, plugs=(_DRIER, "switch.iron"))
    first = rule.evaluate(_snapshot(0))[0]
    register_finding(state, first, _START)

    later = rule.evaluate(_snapshot(35, plugs={_DRIER: "on", "switch.iron": "on"}))[0]
    entity_cooldown = timedelta(minutes=60)
    held_alone = should_suppress(
        state, first, _START + timedelta(minutes=35), timedelta(0), entity_cooldown
    )
    with_new = should_suppress(
        state, later, _START + timedelta(minutes=35), timedelta(0), entity_cooldown
    )

    assert held_alone.reason_code == "entity_cooldown"
    assert not with_new.suppress
