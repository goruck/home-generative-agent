"""Prompt templates for LLM discovery (advisory only)."""

SYSTEM_PROMPT = (
    "You are an assistant suggesting potential anomaly rule ideas. "
    "You must not invent facts. Only use the provided snapshot. "
    "Output JSON only, matching the schema exactly. "
    "Do not include any extra keys or prose."
)

USER_PROMPT_TEMPLATE = (
    "User goal: propose candidate anomaly ideas from the snapshot.\n"
    "Snapshot (authoritative): {snapshot}.\n"
    "Already active rule IDs: {active_rule_ids}.\n"
    "Existing semantic keys (do not duplicate): {existing_semantic_keys}.\n"
    "Return JSON with fields: schema_version, generated_at, model, candidates[].\n"
    "Each candidates[] item MUST include keys: candidate_id, title, summary, "
    "evidence_paths, pattern, confidence_hint. Optional: suggested_type.\n"
    "No other keys are allowed.\n"
    "schema_version MUST be the integer 1.\n"
    "confidence_hint MUST be a number between 0.0 and 1.0 (not a word).\n"
    "evidence_paths MUST be snapshot paths (e.g., derived.is_night, "
    "entities[entity_id=lock.front_door].state).\n"
    "Only return novel candidates that are not already covered by active rules "
    "or existing semantic keys.\n"
    "Do NOT output user queries or commands; only anomaly ideas.\n"
    "Return at most 3 candidates. "
    "Each candidate MUST include no more than 8 evidence_paths."
)
