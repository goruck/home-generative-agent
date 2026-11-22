"""Shared test helper: build a finding's callable actions."""

from __future__ import annotations

import re

from custom_components.home_generative_agent.sentinel.models import FindingAction

_SERVICE = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")


def service_actions(
    names: list[str] | None, entity_ids: list[str]
) -> tuple[FindingAction, ...]:
    """
    Return an action for each ``domain.service`` in *names*.

    Advice such as ``check_camera`` is left out, as a rule would leave it in
    ``suggested_actions`` only. A service acts on the *entity_ids* of its own
    domain (``homeassistant.*`` on all of them); when none is of its domain
    it gets a stand-in target, since a call on another domain is malformed.
    With no *entity_ids* the action has no target.
    """
    actions = []
    for name in names or []:
        if not _SERVICE.fullmatch(name):
            continue
        domain = name.split(".", 1)[0]
        targets = [
            e for e in entity_ids if domain in ("homeassistant", e.split(".", 1)[0])
        ]
        if entity_ids and not targets:
            targets = [f"{domain}.stand_in"]
        actions.append(FindingAction(name, tuple(targets)))
    return tuple(actions)
