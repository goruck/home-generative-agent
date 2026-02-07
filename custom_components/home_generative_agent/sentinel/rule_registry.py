"""Registry for generated sentinel rules."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_rule_registry"
LOGGER = logging.getLogger(__name__)


class RuleRegistry:
    """Persist enabled dynamic rules."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store(hass, STORE_VERSION, STORE_KEY)
        self._rules: list[dict[str, Any]] = []

    async def async_load(self) -> None:
        """Load registry from storage."""
        try:
            data = await self._store.async_load()
        except (HomeAssistantError, OSError, ValueError):
            return
        if isinstance(data, list):
            self._rules = list(data)

    async def async_save(self) -> None:
        """Persist registry."""
        try:
            await self._store.async_save(self._rules)
        except (HomeAssistantError, OSError, ValueError):
            return

    def list_rules(self) -> list[dict[str, Any]]:
        """Return rules list."""
        return list(self._rules)

    def find_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Return a rule by id."""
        for rule in self._rules:
            if rule.get("rule_id") == rule_id:
                return rule
        return None

    async def async_add_rule(self, rule: dict[str, Any]) -> bool:
        """Add a new rule to registry."""
        rule_id = rule.get("rule_id")
        if not rule_id:
            LOGGER.debug("Skipping rule registry add: missing rule_id.")
            return False
        if self.find_rule(str(rule_id)):
            LOGGER.info("Rule registry ignored duplicate rule %s.", rule_id)
            return False
        self._rules.append(rule)
        await self.async_save()
        LOGGER.info("Rule registry added dynamic rule %s.", rule_id)
        return True

    async def async_remove_rule(self, rule_id: str) -> bool:
        """Remove rule by id."""
        for idx, rule in enumerate(self._rules):
            if rule.get("rule_id") == rule_id:
                self._rules.pop(idx)
                await self.async_save()
                return True
        return False
