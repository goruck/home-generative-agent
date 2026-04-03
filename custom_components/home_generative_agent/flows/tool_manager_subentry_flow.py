"""Tool Configuration Manager subentry flow."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
)

from ..const import (  # noqa: TID252
    CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
    CONF_INSTRUCTION_RELEVANCE_THRESHOLD,
    CONF_INSTRUCTION_RETRIEVAL_LIMIT,
    CONF_INSTRUCTIONS_CONFIG,
    CONF_TOOL_RELEVANCE_THRESHOLD,
    CONF_TOOL_RETRIEVAL_LIMIT,
    RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
    RECOMMENDED_INSTRUCTION_RELEVANCE_THRESHOLD,
    RECOMMENDED_INSTRUCTION_RETRIEVAL_LIMIT,
    RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
    RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
    SUBENTRY_TYPE_TOOL_MANAGER,
)

_LOGGER = logging.getLogger(__name__)

# Constants for this flow's data model
CONF_TOOL_PROVIDERS = "tool_providers"
CONF_TOOLS_CONFIG = "tools"


def _current_subentry(flow: ConfigSubentryFlow) -> ConfigSubentry | None:
    """Return the currently edited subentry."""
    entry = flow._get_entry()
    subentry_id = getattr(flow, "_subentry_id", None)
    if not subentry_id:
        subentry_id = flow.context.get("subentry_id")
    if subentry_id:
        return entry.subentries.get(subentry_id)
    if flow.source == SOURCE_RECONFIGURE:
        matches = [
            subentry
            for subentry in entry.subentries.values()
            if subentry.subentry_type == SUBENTRY_TYPE_TOOL_MANAGER
        ]
        if len(matches) == 1:
            return matches[0]
    return None


class ToolManagerSubentryFlow(ConfigSubentryFlow):
    """Config flow for Tool Manager."""

    def __init__(self) -> None:
        """Initialize."""
        self._payload: dict[str, Any] = {}
        self._provider_to_edit: str | None = None
        self._tool_to_edit: str | None = None
        self._instruction_to_edit: str | None = None

    def _schedule_reload(self) -> None:
        entry = self._get_entry()
        self.hass.async_create_task(
            self.hass.config_entries.async_reload(entry.entry_id)
        )

    def _trigger_reindex(self) -> None:
        """Tell the RAG engine to reindex tools because tags changed."""
        entry = self._get_entry()
        entry.runtime_data.tools_version_hash = "FORCED_REINDEX"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Start of flow."""
        current = _current_subentry(self)
        if not self._payload:
            self._payload = {
                CONF_TOOL_RETRIEVAL_LIMIT: RECOMMENDED_TOOL_RETRIEVAL_LIMIT,
                CONF_TOOL_RELEVANCE_THRESHOLD: RECOMMENDED_TOOL_RELEVANCE_THRESHOLD,
                CONF_INSTRUCTION_RETRIEVAL_LIMIT: RECOMMENDED_INSTRUCTION_RETRIEVAL_LIMIT,
                CONF_INSTRUCTION_RELEVANCE_THRESHOLD: RECOMMENDED_INSTRUCTION_RELEVANCE_THRESHOLD,
                CONF_INSTRUCTION_RAG_INTENT_WEIGHT: RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
                CONF_TOOL_PROVIDERS: {},
                CONF_TOOLS_CONFIG: {},
                CONF_INSTRUCTIONS_CONFIG: {},
            }
            if current:
                self._payload.update(current.data)

        if user_input is not None:
            # Save global settings
            self._payload[CONF_TOOL_RETRIEVAL_LIMIT] = int(
                user_input[CONF_TOOL_RETRIEVAL_LIMIT]
            )
            self._payload[CONF_TOOL_RELEVANCE_THRESHOLD] = float(
                user_input[CONF_TOOL_RELEVANCE_THRESHOLD]
            )
            self._payload[CONF_INSTRUCTION_RETRIEVAL_LIMIT] = int(
                user_input[CONF_INSTRUCTION_RETRIEVAL_LIMIT]
            )
            self._payload[CONF_INSTRUCTION_RELEVANCE_THRESHOLD] = float(
                user_input[CONF_INSTRUCTION_RELEVANCE_THRESHOLD]
            )
            self._payload[CONF_INSTRUCTION_RAG_INTENT_WEIGHT] = float(
                user_input[CONF_INSTRUCTION_RAG_INTENT_WEIGHT]
            )

            action = user_input.get("next_action", "save")
            if action == "save":
                return self.async_create_or_update()

            if action.startswith("edit_provider_"):
                self._provider_to_edit = action.replace("edit_provider_", "")
                return await self.async_step_provider_editor()

            if action.startswith("edit_tool_"):
                self._tool_to_edit = action.replace("edit_tool_", "")
                return await self.async_step_tool_editor()

            if action == "add_instruction":
                return await self.async_step_instruction_name()

            if action.startswith("edit_instruction_"):
                self._instruction_to_edit = action.replace("edit_instruction_", "")
                return await self.async_step_instruction_editor()

            return self.async_create_or_update()

        # Build options for next actions
        actions = [SelectOptionDict(label="Save and Finish", value="save")]

        # Add provider options
        apis = llm.async_get_apis(self.hass)
        providers = [llm.LLM_API_ASSIST] + [
            api.id for api in apis if api.id != llm.LLM_API_ASSIST
        ]
        providers.append("langchain_internal")  # Add custom internal tool provider

        for p in providers:
            actions.append(
                SelectOptionDict(
                    label=f"Configure Provider: {p}", value=f"edit_provider_{p}"
                )
            )

        # Add instruction options
        actions.append(
            SelectOptionDict(label="Add New Prompt and Tag", value="add_instruction")
        )
        instructions = self._payload.get(CONF_INSTRUCTIONS_CONFIG, {})
        for i_name in sorted(instructions.keys()):
            actions.append(
                SelectOptionDict(
                    label=f"Configure Instruction: {i_name}",
                    value=f"edit_instruction_{i_name}",
                )
            )

        # We could also list tools, but maybe too many. We'll list a group for now.
        # RAG tools dynamic generation would be amazing, but for now we let users pick from active ones.
        # RAG tools dynamic generation
        active_tools = set()
        llm_context = llm.LLMContext(
            platform="home_generative_agent",
            context=None,
            language=None,
            assistant="conversation",
            device_id=None,
        )
        for api in apis:
            try:
                inst = await llm.async_get_api(self.hass, api.id, llm_context)
                for t in inst.tools:
                    active_tools.add(t.name)
            except Exception:
                pass
        active_tools.update(
            [
                "get_and_analyze_camera_image",
                "get_camera_last_events",
                "upsert_memory",
                "get_entity_history",
                "confirm_sensitive_action",
                "alarm_control",
                "resolve_entity_ids",
                "write_yaml_file",
                "get_available_tools",
                "add_automation",
            ]
        )

        for t in sorted(active_tools):
            actions.append(
                SelectOptionDict(label=f"Configure Tool: {t}", value=f"edit_tool_{t}")
            )

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_TOOL_RETRIEVAL_LIMIT,
                    default=self._payload.get(CONF_TOOL_RETRIEVAL_LIMIT),
                ): NumberSelector(NumberSelectorConfig(min=1, max=50, step=1)),
                vol.Required(
                    CONF_TOOL_RELEVANCE_THRESHOLD,
                    default=self._payload.get(CONF_TOOL_RELEVANCE_THRESHOLD),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
                vol.Required(
                    CONF_INSTRUCTION_RETRIEVAL_LIMIT,
                    default=self._payload.get(CONF_INSTRUCTION_RETRIEVAL_LIMIT),
                ): NumberSelector(NumberSelectorConfig(min=1, max=50, step=1)),
                vol.Required(
                    CONF_INSTRUCTION_RELEVANCE_THRESHOLD,
                    default=self._payload.get(CONF_INSTRUCTION_RELEVANCE_THRESHOLD),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
                vol.Required(
                    CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
                    default=self._payload.get(
                        CONF_INSTRUCTION_RAG_INTENT_WEIGHT,
                        RECOMMENDED_INSTRUCTION_RAG_INTENT_WEIGHT,
                    ),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.01)),
                vol.Required("next_action", default="save"): SelectSelector(
                    SelectSelectorConfig(
                        options=actions,
                        mode=SelectSelectorMode.DROPDOWN,
                        sort=False,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            description_placeholders={
                "tool_manager_title": "RAG Tool & Prompt Configuration"
            },
        )

    async def async_step_provider_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            providers = self._payload.setdefault(CONF_TOOL_PROVIDERS, {})
            providers[self._provider_to_edit] = {
                "enabled": user_input.get("enabled", True),
                "prompt": user_input.get("prompt", ""),
                "tags": user_input.get("tags", ""),
            }
            return await self.async_step_user()

        current_cfg = self._payload.get(CONF_TOOL_PROVIDERS, {}).get(
            self._provider_to_edit, {}
        )
        schema = vol.Schema(
            {
                vol.Required(
                    "enabled", default=current_cfg.get("enabled", True)
                ): BooleanSelector(),
                vol.Optional(
                    "prompt",
                    default=current_cfg.get("prompt", ""),
                    description={"suggested_value": current_cfg.get("prompt", "")},
                ): TemplateSelector(),
                vol.Optional(
                    "tags",
                    default=current_cfg.get("tags", ""),
                    description={"suggested_value": current_cfg.get("tags", "")},
                ): TextSelector(),
            }
        )

        return self.async_show_form(
            step_id="provider_editor",
            data_schema=schema,
            description_placeholders={
                "provider_name": self._provider_to_edit or "",
                "provider_injection_label": "Provider Context (Tier 2 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this tool (e.g., 'chill, movie night, dim').",
            },
        )

    async def async_step_tool_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            tools = self._payload.setdefault(CONF_TOOLS_CONFIG, {})
            tools[self._tool_to_edit] = {
                "enabled": user_input.get("enabled", True),
                "prompt": user_input.get("prompt", ""),
                "tags": user_input.get("tags", ""),
            }
            return await self.async_step_user()

        current_cfg = self._payload.get(CONF_TOOLS_CONFIG, {}).get(
            self._tool_to_edit, {}
        )
        schema = vol.Schema(
            {
                vol.Required(
                    "enabled", default=current_cfg.get("enabled", True)
                ): BooleanSelector(),
                vol.Optional(
                    "prompt",
                    default=current_cfg.get("prompt", ""),
                    description={"suggested_value": current_cfg.get("prompt", "")},
                ): TemplateSelector(),
                vol.Optional(
                    "tags",
                    default=current_cfg.get("tags", ""),
                    description={"suggested_value": current_cfg.get("tags", "")},
                ): TextSelector(),
            }
        )

        return self.async_show_form(
            step_id="tool_editor",
            data_schema=schema,
            description_placeholders={
                "tool_name": self._tool_to_edit or "",
                "tool_injection_label": "Specific Tool Instructions (Tier 3 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this tool (e.g., 'chill, movie night, dim').",
            },
        )

    async def async_step_instruction_name(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            self._instruction_to_edit = user_input["name"]
            return await self.async_step_instruction_editor()

        return self.async_show_form(
            step_id="instruction_name",
            data_schema=vol.Schema({vol.Required("name"): TextSelector()}),
            description_placeholders={"instruction_name_title": "Instruction Name"},
        )

    async def async_step_instruction_editor(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            instructions = self._payload.setdefault(CONF_INSTRUCTIONS_CONFIG, {})
            if user_input.get("delete_entry"):
                instructions.pop(self._instruction_to_edit, None)
            else:
                instructions[self._instruction_to_edit] = {
                    "enabled": user_input.get("enabled", True),
                    "prompt": user_input.get("prompt", ""),
                    "tags": user_input.get("tags", ""),
                }
            return await self.async_step_user()

        current_cfg = self._payload.get(CONF_INSTRUCTIONS_CONFIG, {}).get(
            self._instruction_to_edit, {}
        )
        schema = vol.Schema(
            {
                vol.Required(
                    "enabled", default=current_cfg.get("enabled", False)
                ): BooleanSelector(),
                vol.Optional(
                    "prompt",
                    default=current_cfg.get("prompt", ""),
                    description={"suggested_value": current_cfg.get("prompt", "")},
                ): TemplateSelector(),
                vol.Optional(
                    "tags",
                    default=current_cfg.get("tags", ""),
                    description={"suggested_value": current_cfg.get("tags", "")},
                ): TextSelector(),
                vol.Optional("delete_entry", default=False): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="instruction_editor",
            data_schema=schema,
            description_placeholders={
                "instruction_name": self._instruction_to_edit or "",
                "instruction_injection_label": "Instruction Text (Tier 1.5 Prompt)",
                "semantic_tags_hint": "Comma-separated keywords to help the AI find this instruction.",
            },
        )

    def async_create_or_update(self) -> SubentryFlowResult:
        current = _current_subentry(self)
        self._trigger_reindex()
        self._schedule_reload()

        if current:
            return self.async_update_and_abort(
                self._get_entry(), current, data=self._payload, title="RAG Tool Manager"
            )

        if self.source == SOURCE_RECONFIGURE:
            self._source = SOURCE_USER
            self.context["source"] = SOURCE_USER

        return self.async_create_entry(title="RAG Tool Manager", data=self._payload)

    async_step_reconfigure = async_step_user
