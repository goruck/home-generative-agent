"""Shared test helper: build a finding's callable actions."""

from __future__ import annotations

import re

from custom_components.home_generative_agent.sentinel.models import FindingAction

_SERVICE = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")


def service_actions(
    names: list[str] | None, entity_ids: list[str]
) -> tuple[FindingAction, ...]:
    """
    Return an action on *entity_ids* for each ``domain.service`` in *names*.

    Advice such as ``check_camera`` is left out, as a rule would leave it in
    ``suggested_actions`` only.
    """
    return tuple(
        FindingAction(name, tuple(entity_ids))
        for name in names or []
        if _SERVICE.fullmatch(name)
    )
