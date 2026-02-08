"""Registry for generated sentinel rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

STORE_VERSION = 1
STORE_KEY = "home_generative_agent_sentinel_rule_registry"
LOGGER = logging.getLogger(__name__)


class RuleRegistry:
    """Persist enabled dynamic rules."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize dynamic rule storage."""
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
        for rule in self._rules:
            if not isinstance(rule, dict):
                continue
            rule.setdefault("enabled", True)

    async def async_save(self) -> None:
        """Persist registry."""
        try:
            await self._store.async_save(self._rules)
        except (HomeAssistantError, OSError, ValueError):
            return

    def list_rules(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        """Return rules list, optionally including disabled rules."""
        if include_disabled:
            return list(self._rules)
        return [rule for rule in self._rules if bool(rule.get("enabled", True))]

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
        stored_rule = dict(rule)
        stored_rule.setdefault("enabled", True)
        self._rules.append(stored_rule)
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

    async def async_set_rule_enabled(self, rule_id: str, *, enabled: bool) -> bool:
        """Set rule enabled state by id."""
        for rule in self._rules:
            if rule.get("rule_id") != rule_id:
                continue
            if bool(rule.get("enabled", True)) == enabled:
                return True
            rule["enabled"] = enabled
            await self.async_save()
            LOGGER.info(
                "Rule registry %s dynamic rule %s.",
                "activated" if enabled else "deactivated",
                rule_id,
            )
            return True
        return False
