"""Helper utilities for conversation processing."""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

import yaml

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)
_CODE_FENCE_MIN_LINES = 3
_YAML_LIST_INDENT = 2
_YAML_NESTED_INDENT = 4
_ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_ENTITY_ID_TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b")
_ENTITY_ID_MATCH_SCORE_MIN = 0.6
_ENTITY_ID_TOKEN_OVERLAP_WEIGHT = 0.2


class _IndentDumper(yaml.SafeDumper):
    """Force YAML lists to indent under their parent key."""

    def increase_indent(
        self,
        flow: bool = False,  # noqa: FBT001, FBT002
        indentless: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> Any:
        return super().increase_indent(flow, False)  # noqa: FBT003


def _strip_code_fence(content: str) -> str:
    """Remove a surrounding Markdown code fence if present."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < _CODE_FENCE_MIN_LINES or not lines[0].startswith("```"):
        return stripped
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].startswith("```"):
            return "\n".join(lines[1:idx]).strip()
    return stripped


def _extract_json_block(content: str) -> str | None:
    """Extract the first complete JSON object/array from content."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(content[idx:])
        except json.JSONDecodeError:
            continue
        return content[idx : idx + end]
    return None


def _load_json_payload(content: str) -> tuple[Any | None, str | None]:
    """Load JSON from content, optionally extracting a JSON block."""
    candidate = _strip_code_fence(content)
    if not candidate:
        return None, None
    if candidate[0] not in "{[":
        return None, candidate

    try:
        return json.loads(candidate), candidate
    except json.JSONDecodeError as err:
        extracted = _extract_json_block(candidate)
        if extracted and extracted != candidate:
            try:
                return json.loads(extracted), extracted
            except json.JSONDecodeError:
                pass
        _LOGGER.warning(
            "Schema-first JSON parsing failed: %s; content=%r", err, candidate[:500]
        )
        return None, candidate


def _resolve_entity_id(entity_id: str, hass: HomeAssistant) -> str:
    """Try to resolve a suggested entity_id to an existing entity_id."""
    if not _ENTITY_ID_PATTERN.match(entity_id):
        return entity_id
    if hass.states.get(entity_id):
        return entity_id

    domain, object_id = entity_id.split(".", 1)
    prefix = f"{domain}."
    candidates = [
        state.entity_id
        for state in hass.states.async_all()
        if state.entity_id.startswith(prefix)
    ]
    if not candidates:
        return entity_id

    def score_match(candidate: str) -> float:
        candidate_obj = candidate.split(".", 1)[1]
        ratio = difflib.SequenceMatcher(None, object_id, candidate_obj).ratio()
        target_tokens = {token for token in object_id.split("_") if token}
        candidate_tokens = {token for token in candidate_obj.split("_") if token}
        overlap = 0.0
        if target_tokens:
            overlap = len(target_tokens & candidate_tokens) / len(target_tokens)
        return ratio + (overlap * _ENTITY_ID_TOKEN_OVERLAP_WEIGHT)

    tokens = [token for token in object_id.split("_") if token]
    if tokens:
        token_matches = [
            candidate
            for candidate in candidates
            if all(token in candidate.split(".", 1)[1] for token in tokens)
        ]
        if token_matches:
            best_match = max(token_matches, key=score_match)
            if score_match(best_match) >= _ENTITY_ID_MATCH_SCORE_MIN:
                return best_match

    scored = max(candidates, key=score_match)
    if score_match(scored) >= _ENTITY_ID_MATCH_SCORE_MIN:
        return scored

    close = difflib.get_close_matches(
        entity_id, candidates, n=1, cutoff=_ENTITY_ID_MATCH_SCORE_MIN
    )
    return close[0] if close else entity_id


def _fix_dashboard_entities(payload: dict[str, Any], hass: HomeAssistant) -> bool:
    """Update DashboardSpec entity_ids when a close existing match is found."""
    if not isinstance(payload.get("views"), list):
        return False

    changed = False

    def update_entity(value: str) -> str:
        nonlocal changed
        resolved = _resolve_entity_id(value, hass)
        if resolved != value:
            _LOGGER.debug("Resolved dashboard entity_id %s -> %s", value, resolved)
            changed = True
        return resolved

    def update_entity_rows(entities: list[Any]) -> None:
        for idx, entity in enumerate(entities):
            if isinstance(entity, str):
                entities[idx] = update_entity(entity)
                continue
            if isinstance(entity, dict):
                entity_id = entity.get("entity")
                if isinstance(entity_id, str):
                    entity["entity"] = update_entity(entity_id)

    def update_cards(cards: list[Any]) -> None:
        for card in cards:
            if not isinstance(card, dict):
                continue
            entity_id = card.get("entity")
            if isinstance(entity_id, str):
                card["entity"] = update_entity(entity_id)

            entities = card.get("entities")
            if isinstance(entities, list):
                update_entity_rows(entities)

            nested_cards = card.get("cards")
            if isinstance(nested_cards, list):
                update_cards(nested_cards)

    for view in payload.get("views", []):
        if not isinstance(view, dict):
            continue
        cards = view.get("cards")
        if isinstance(cards, list):
            update_cards(cards)

    return changed


def _maybe_fix_dashboard_entities(content: str, hass: HomeAssistant) -> str:
    payload, candidate = _load_json_payload(content)
    if payload is None or candidate is None:
        return content

    if not isinstance(payload, dict):
        return content
    if not _fix_dashboard_entities(payload, hass):
        return content

    return json.dumps(payload, ensure_ascii=True, separators=(",", ": "))


def _fix_entity_ids_in_text(content: str, hass: HomeAssistant) -> str:
    """Replace entity_id-like tokens in text with existing entity_ids when possible."""

    def replace(match: re.Match[str]) -> str:
        entity_id = match.group(0)
        resolved = _resolve_entity_id(entity_id, hass)
        if resolved != entity_id:
            _LOGGER.debug("Resolved text entity_id %s -> %s", entity_id, resolved)
        return resolved

    return _ENTITY_ID_TOKEN_PATTERN.sub(replace, content)


def _is_dashboard_request(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "dashboard" in lowered or "lovelace" in lowered


def _convert_schema_json_to_yaml(  # noqa: PLR0911, PLR0912
    content: str,
    enabled: bool,  # noqa: FBT001
) -> str:
    """Convert schema-first JSON to YAML when enabled; otherwise passthrough."""
    if not enabled:
        return content
    if "```yaml" in content:
        inner = _strip_code_fence(content)
        fixed = _fix_automation_yaml_indentation(inner)
        return f"```yaml\n{fixed}\n```"
    candidate = _strip_code_fence(content)
    if not candidate:
        return content
    payload: Any | None = None
    if candidate[0] in "{[":
        payload, _ = _load_json_payload(candidate)
        if payload is None:
            _LOGGER.warning("Schema-first JSON parsing failed; trying YAML fallback.")
            try:
                payload = yaml.safe_load(candidate)
            except yaml.YAMLError as err:
                _LOGGER.warning("Schema-first YAML fallback failed: %s", err)
                payload = None
            if payload is None or isinstance(payload, str):
                return (
                    "Schema-first JSON parsing failed. "
                    "Please respond with valid JSON only."
                )
    else:
        try:
            payload = yaml.safe_load(candidate)
        except yaml.YAMLError:
            payload = None
        if payload is None or isinstance(payload, str):
            return content
    if isinstance(payload, dict) and "yaml" in payload:
        raw_yaml = payload.get("yaml")
        if isinstance(raw_yaml, str):
            raw_yaml = raw_yaml.replace("\\n", "\n").strip()
            try:
                yaml_payload = yaml.safe_load(raw_yaml)
            except yaml.YAMLError:
                yaml_payload = None
            if yaml_payload is not None and not isinstance(yaml_payload, str):
                yaml_payload = _normalize_automation_payload(yaml_payload)
                yaml_text = cast("Any", yaml.dump)(
                    yaml_payload,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                    Dumper=_IndentDumper,
                ).rstrip()
                yaml_text = _fix_automation_yaml_indentation(yaml_text)
                return f"```yaml\n{yaml_text}\n```"
            return f"```yaml\n{raw_yaml}\n```"
    payload = _normalize_automation_payload(payload)
    yaml_text = cast("Any", yaml.dump)(
        payload,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        Dumper=_IndentDumper,
    ).rstrip()
    yaml_text = _fix_automation_yaml_indentation(yaml_text)
    return f"```yaml\n{yaml_text}\n```"


def _normalize_automation_payload(payload: Any) -> Any:  # noqa: PLR0912
    """Heuristically nest common automation fields under trigger/action."""
    if isinstance(payload, list) and payload:
        payload[0] = _normalize_automation_payload(payload[0])
        return payload
    if not isinstance(payload, dict):
        return payload
    if "trigger" in payload:
        if isinstance(payload["trigger"], dict):
            payload["trigger"] = [payload["trigger"]]
        if isinstance(payload.get("trigger"), list):
            trigger_items = payload["trigger"]
            if trigger_items:
                first = trigger_items[0]
                if isinstance(first, dict):
                    for key in ("entity_id", "to", "from", "for", "attribute"):
                        if key in payload and key not in first:
                            first[key] = payload.pop(key)
    if "action" in payload:
        if isinstance(payload["action"], dict):
            payload["action"] = [payload["action"]]
        if isinstance(payload.get("action"), list):
            action_items = payload["action"]
            if action_items:
                first = action_items[0]
                if isinstance(first, dict):
                    for key in ("target", "data", "data_template"):
                        if key in payload and key not in first:
                            first[key] = payload.pop(key)
                    if "alias" in first and "alias" not in payload:
                        payload["alias"] = first.pop("alias")
    if "condition" in payload and isinstance(payload["condition"], dict):
        payload["condition"] = [payload["condition"]]
    return _reorder_automation_payload(payload)


def _reorder_automation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Put common automation keys in a stable, HA-friendly order."""
    preferred = [
        "alias",
        "description",
        "trigger",
        "condition",
        "action",
        "mode",
        "max",
        "id",
    ]
    reordered: dict[str, Any] = {}
    for key in preferred:
        if key in payload:
            reordered[key] = payload[key]
    for key, value in payload.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def _fix_automation_yaml_indentation(yaml_text: str) -> str:
    """Ensure trigger/action/condition list items are indented under their keys."""
    if "trigger:" not in yaml_text and "action:" not in yaml_text:
        return yaml_text
    lines = yaml_text.splitlines()
    out: list[str] = []
    in_block: str | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and stripped.endswith(":"):
            key = stripped[:-1]
            in_block = key if key in {"trigger", "action", "condition"} else None
            out.append(line)
            continue
        if in_block:
            if stripped.startswith("- ") and indent < _YAML_LIST_INDENT:
                out.append((" " * _YAML_LIST_INDENT) + stripped)
                continue
            if (
                stripped
                and indent < _YAML_NESTED_INDENT
                and not stripped.startswith("- ")
            ):
                out.append((" " * _YAML_NESTED_INDENT) + stripped)
                continue
        if indent == 0 and stripped and not stripped.startswith("- "):
            in_block = None
        out.append(line)
    return "\n".join(out)
