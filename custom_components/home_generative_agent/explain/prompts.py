"""Prompt templates for LLM explanation layer."""

SYSTEM_PROMPT = (
    "You are an assistant explaining a home anomaly to a user. "
    "You must not invent facts. Only use the evidence provided. "
    "If evidence is insufficient, say so briefly."
)

USER_PROMPT_TEMPLATE = (
    "Explain the anomaly type: {anomaly_type}.\n"
    "Severity: {severity}.\n"
    "Evidence (authoritative): {evidence}.\n"
    "Suggested actions (semantic only): {suggested_actions}.\n"
    "Return a short explanation and rank the suggested actions."
)
