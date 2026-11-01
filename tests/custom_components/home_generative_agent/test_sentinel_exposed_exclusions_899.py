# ruff: noqa: S101
"""
Exclusions on the sensitive-exposure aggregate apply per entity (#899).

Before the fix the engine dropped the whole ``ha_sensitive_entity_exposed_
without_pin`` finding when any listed entity was excluded, and an exclusion
on a member did not reach the group holding it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from custom_components.home_generative_agent.agent.rule_entities import AuditReach
from custom_components.home_generative_agent.const import (
    CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS,
)
from custom_components.home_generative_agent.sentinel.engine import (
    _parse_rule_entity_exclusions,
)
from custom_components.home_generative_agent.sentinel.rules.ha_sensitive_entity_exposed_without_pin import (
    HaSensitiveEntityExposedWithoutPinRule,
)
from custom_components.home_generative_agent.snapshot.network import (
    HaNativeInputs,
    NetworkBuildContext,
    ha_cap,
    ha_native_adapter,
)
from custom_components.home_generative_agent.snapshot.schema import (
    FullStateSnapshot,
    SnapshotEntity,
    validate_snapshot,
)

from .test_sentinel_rule_exclusions import (
    DummyNotifier,
    _make_engine,
    _patch_snapshot,
)

if TYPE_CHECKING:
    from custom_components.home_generative_agent.sentinel.models import (
        AnomalyFinding,
    )

RULE_ID = "ha_sensitive_entity_exposed_without_pin"
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _snapshot(
    exposed: dict[str, list[str]],
    groups: dict[str, list[str]] | None = None,
    partial: list[str] | None = None,
) -> FullStateSnapshot:
    ha: dict[str, Any] = {
        "exposed_sensitive_entities": exposed,
        "critical_action_pin_enabled": False,
    }
    if groups is not None:
        ha["exposed_sensitive_groups"] = groups
    if partial is not None:
        ha["exposed_sensitive_groups_partial"] = partial
    capabilities = sorted(ha_cap(k) for k in ha)
    return validate_snapshot(
        {
            "schema_version": 2,
            "generated_at": "2026-11-01T12:00:00+00:00",
            "entities": [],
            "camera_activity": [],
            "derived": {
                "now": "2026-11-01T12:00:00+00:00",
                "timezone": "UTC",
                "is_night": False,
                "anyone_home": True,
                "people_home": ["Lindo"],
                "people_away": [],
                "last_motion_by_area": {},
            },
            "network": {
                "capabilities": capabilities,
                "sources": dict.fromkeys(capabilities, "ha_native"),
                "clients": [],
                "posture": {},
                "ha_security": ha,
                "counters": {},
            },
        }
    )


def _rule(exclusions: dict[str, list[str]]) -> HaSensitiveEntityExposedWithoutPinRule:
    parsed = _parse_rule_entity_exclusions(exclusions)

    def _excluded(entity_id: str, anomaly_type: str) -> bool:
        return any(
            key in parsed and parsed[key].matches(entity_id)
            for key in ("*", anomaly_type)
        )

    return HaSensitiveEntityExposedWithoutPinRule(is_entity_excluded=_excluded)


def _only(findings: list[AnomalyFinding]) -> AnomalyFinding:
    assert len(findings) == 1
    return findings[0]


ALEXA_LOCK_AND_GROUP = {"cloud.alexa": ["cover.all_covers", "lock.front_door"]}
GROUP_HOLDS = {"cover.all_covers": ["cover.garage", "cover.shed"]}


@pytest.mark.parametrize("key", ["*", RULE_ID])
def test_an_excluded_group_leaves_the_rest_of_the_report(key: str) -> None:
    """Excluding the group removes the group, not the exposed lock."""
    finding = _only(
        _rule({key: ["cover.all_covers"]}).evaluate(
            _snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS)
        )
    )
    assert finding.triggering_entities == ["lock.front_door"]
    assert finding.evidence["exposures"] == {"cloud.alexa": ["lock.front_door"]}
    assert "groups" not in finding.evidence
    assert "cover.all_covers" not in finding.evidence["summary"]


def test_a_glob_exclusion_removes_only_what_it_matches() -> None:
    """``cover.*`` quiets every cover; the lock is still reported."""
    finding = _only(
        _rule({"*": ["cover.*"]}).evaluate(_snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS))
    )
    assert finding.evidence["exposures"] == {"cloud.alexa": ["lock.front_door"]}


def test_a_group_counts_as_excluded_when_every_member_it_holds_is() -> None:
    """A member exclusion reaches the group that is the only route to it."""
    rule = _rule({RULE_ID: ["cover.garage", "cover.shed"]})
    finding = _only(rule.evaluate(_snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS)))
    assert finding.evidence["exposures"] == {"cloud.alexa": ["lock.front_door"]}
    only_group = rule.evaluate(
        _snapshot({"cloud.alexa": ["cover.all_covers"]}, GROUP_HOLDS)
    )
    assert only_group == []


def test_a_group_holding_some_excluded_members_names_only_the_rest() -> None:
    """The group stays reported and lists only the members still in scope."""
    finding = _only(
        _rule({RULE_ID: ["cover.garage"]}).evaluate(
            _snapshot({"cloud.alexa": ["cover.all_covers"]}, GROUP_HOLDS)
        )
    )
    assert finding.triggering_entities == ["cover.all_covers"]
    assert finding.evidence["groups"] == {"cover.all_covers": ["cover.shed"]}
    assert finding.evidence["summary"].endswith("cover.all_covers holds cover.shed.")
    assert "cover.garage" not in finding.evidence["summary"]


def test_an_excluded_member_exposed_directly_and_through_a_group() -> None:
    """Both routes to an excluded member are quiet; nothing else is."""
    rule = _rule({RULE_ID: ["cover.garage"]})
    exposed = {
        "cloud.alexa": ["cover.all_covers", "cover.garage"],
        "conversation": ["cover.garage", "lock.front_door"],
    }
    finding = _only(
        rule.evaluate(_snapshot(exposed, {"cover.all_covers": ["cover.garage"]}))
    )
    assert finding.evidence["exposures"] == {"conversation": ["lock.front_door"]}


def test_a_group_with_unknown_members_counts_by_its_own_id() -> None:
    """With no member list, only the group's own exclusion removes it."""
    rule = _rule({RULE_ID: ["cover.garage"]})
    finding = _only(rule.evaluate(_snapshot({"cloud.alexa": ["cover.all_covers"]})))
    assert finding.evidence["exposures"] == {"cloud.alexa": ["cover.all_covers"]}
    assert "groups" not in finding.evidence


def _adapter_snapshot(
    exposed: dict[str, list[str]], reach: AuditReach
) -> FullStateSnapshot:
    """Build the snapshot the rule reads through the real adapter."""
    entities = [
        SnapshotEntity(
            entity_id=entity_id,
            domain=entity_id.split(".", 1)[0],
            state="closed",
            friendly_name=None,
            area=None,
            attributes=attributes,
            last_changed="2026-11-01T12:00:00+00:00",
            last_updated="2026-11-01T12:00:00+00:00",
        )
        for entity_id, attributes in (
            ("cover.garage", {"device_class": "garage"}),
            ("cover.partial", {}),
        )
    ]
    result = ha_native_adapter(
        HaNativeInputs(now=NOW, exposed=exposed, sensitive_reach=reach),
        entities,
        NetworkBuildContext(),
    )
    ha = result.ha_security
    return _snapshot(
        ha["exposed_sensitive_entities"],
        ha.get("exposed_sensitive_groups"),
        ha.get("exposed_sensitive_groups_partial"),
    )


def test_a_partly_read_group_is_not_dropped_by_member_exclusions() -> None:
    """
    A group whose membership was only partly read may hold more (fail closed).

    Excluding every member the audit could read must not hide the group: an
    unread member may be an unexcluded lock or garage door.
    """
    snapshot = _adapter_snapshot(
        {"cloud.alexa": ["cover.partial"]},
        AuditReach(
            groups={"cover.partial": ["cover.garage"]},
            unreadable_groups=["cover.partial"],
        ),
    )
    rule = _rule({RULE_ID: ["cover.garage"]})
    finding = _only(rule.evaluate(snapshot))
    assert finding.evidence["exposures"] == {"cloud.alexa": ["cover.partial"]}
    assert "groups" not in finding.evidence
    # Its own id still removes it.
    assert _rule({RULE_ID: ["cover.partial"]}).evaluate(snapshot) == []
    # A fully read group with every member excluded is still dropped.
    complete = _adapter_snapshot(
        {"cloud.alexa": ["cover.partial"]},
        AuditReach(groups={"cover.partial": ["cover.garage"]}),
    )
    assert rule.evaluate(complete) == []


def test_everything_excluded_emits_nothing() -> None:
    """No exposure left in scope means no finding."""
    rule = _rule({"*": ["cover.all_covers", "lock.front_door"]})
    assert rule.evaluate(_snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS)) == []


def test_no_predicate_keeps_the_rule_unchanged() -> None:
    """The rule built without the engine reports every exposure."""
    finding = _only(
        HaSensitiveEntityExposedWithoutPinRule().evaluate(
            _snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS)
        )
    )
    assert finding.triggering_entities == ["cover.all_covers", "lock.front_door"]


@pytest.mark.asyncio
async def test_the_engine_keeps_the_report_when_a_group_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The engine wires its predicate into the rule.

    Without it the generic filter drops the aggregate because the excluded
    group is a triggering entity, and the lock exposure is never told.
    """
    _patch_snapshot(monkeypatch, _snapshot(ALEXA_LOCK_AND_GROUP, GROUP_HOLDS))
    engine = _make_engine(
        {CONF_SENTINEL_RULE_ENTITY_EXCLUSIONS: {"*": ["cover.all_covers"]}}
    )
    await engine._run_once()
    notifier = cast("DummyNotifier", cast("Any", engine)._notifier)
    findings = [
        cast("AnomalyFinding", call["finding"])
        for call in notifier.calls
        if cast("AnomalyFinding", call["finding"]).type == RULE_ID
    ]
    assert len(findings) == 1
    assert findings[0].evidence["exposures"] == {"cloud.alexa": ["lock.front_door"]}
