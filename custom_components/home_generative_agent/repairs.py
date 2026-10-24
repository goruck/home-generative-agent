"""Fix flows for Home Generative Agent's repair issues."""

from __future__ import annotations

from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.components.repairs import ConfirmRepairFlow, RepairsFlow
from homeassistant.helpers import issue_registry as ir

from .core.embedding_record import ISSUE_EMBEDDING_MODEL_CHANGED, async_start_reembed

if TYPE_CHECKING:
    from homeassistant.components.repairs import RepairsFlowResult
    from homeassistant.core import HomeAssistant


class ReembedStoreFlow(RepairsFlow):
    """Re-embed the memory store with the current embedding model (#855)."""

    def __init__(self, entry_id: str | None) -> None:
        """Remember which entry's store to re-embed."""
        self._entry_id = entry_id

    async def async_step_init(
        self,
        user_input: dict[str, str] | None = None,  # noqa: ARG002
    ) -> RepairsFlowResult:
        """
        Show what the re-embed does.

        Home Assistant opens a fix flow with ``{"issue_id": ...}`` as input:
        passing it on would read as the user's confirmation and start hours
        of re-embedding without showing the form.
        """
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        """Start the re-embed once confirmed."""
        if user_input is None:
            issue = ir.async_get(self.hass).async_get_issue(self.handler, self.issue_id)
            if issue is None or not issue.translation_placeholders:
                # Gone since the dialog opened (a reload cleared it).
                return self.async_abort(reason="not_ready")
            return self.async_show_form(
                step_id="confirm",
                data_schema=vol.Schema({}),
                description_placeholders=issue.translation_placeholders,
            )
        entry = (
            self.hass.config_entries.async_get_entry(self._entry_id)
            if self._entry_id
            else None
        )
        if entry is None or not await async_start_reembed(self.hass, entry):
            return self.async_abort(reason="not_ready")
        return self.async_create_entry(data={})


async def async_create_fix_flow(
    hass: HomeAssistant,  # noqa: ARG001
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create the fix flow for a fixable issue."""
    if issue_id.startswith(f"{ISSUE_EMBEDDING_MODEL_CHANGED}_"):
        entry_id = (data or {}).get("entry_id")
        return ReembedStoreFlow(entry_id if isinstance(entry_id, str) else None)
    return ConfirmRepairFlow()
