# Implementation Plan: Custom OpenAI Base URL & Tool Calling Improvements

**Version:** 1.0  
**Date:** November 9, 2025  
**Status:** Planning Phase - Awaiting Confirmation

---

## Objective Summary

Extend the Home Generative Agent config flow to support:
1. **Custom OpenAI Base URL** for both chat and embeddings (compatibility with llama.cpp/custom OpenAI APIs)
2. **Harden & Improve Tool Calling** mechanism for reliability and better error handling

---

## Part 1: Custom OpenAI Base URL Support

### 1.1 Problem Analysis

**Current State:**
- Ollama ✅ supports configurable `base_url` (CONF_OLLAMA_URL)
- OpenAI ❌ hardcoded to `https://api.openai.com`
  - `ChatOpenAI()` instantiation (line 318-327 in `__init__.py`) lacks `base_url` parameter
  - `OpenAIEmbeddings()` instantiation (line 370-377 in `__init__.py`) lacks `base_url` parameter
  - Health check URL hardcoded (line 207 in `core/utils.py`)

**Use Case:**
- llama.cpp with OpenAI-compatible API running on `http://localhost:8000`
- Custom OpenAI proxies or internal CDN endpoints

### 1.2 Implementation Scope

#### A. Constants (`const.py`)
```python
# Add new config key
CONF_OPENAI_BASE_URL = "openai_base_url"
RECOMMENDED_OPENAI_BASE_URL = "https://api.openai.com/v1"  # Default to official
```

#### B. Validation (`core/utils.py`)
```python
async def validate_openai_base_url(
    hass: HomeAssistant, 
    base_url: str | None
) -> None:
    """Validate OpenAI-compatible base URL."""
    if not base_url:
        return  # Optional field
    
    # Normalize URL (add /v1 suffix if missing)
    # Check connectivity with HEAD request
    # Raise InvalidAuthError or CannotConnectError
```

#### C. Config Flow (`config_flow.py`)
- **Stage 1 (User Step):** Add optional field for OpenAI base URL
- **Validation:** Call `validate_openai_base_url()` in `_run_validations_user()`
- **Translation:** Update `translations/en.json`

#### D. Provider Instantiation (`__init__.py`)
- Extract `base_url` from config (lines 318-327 for ChatOpenAI)
- Pass to `ChatOpenAI(api_key=..., base_url=base_url, ...)`
- Do same for `OpenAIEmbeddings` (lines 370-377)
- Update health check to use custom base_url

### 1.3 Changes Summary

| File | Changes | Lines |
|------|---------|-------|
| `const.py` | Add `CONF_OPENAI_BASE_URL`, `RECOMMENDED_OPENAI_BASE_URL` | End of config section |
| `core/utils.py` | Add `validate_openai_base_url()` function | After `validate_openai_key()` |
| `core/utils.py` | Update `openai_healthy()` to use custom base_url | ~line 207 |
| `config_flow.py` | Add OpenAI base URL field to user step schema | ~line 100-110 |
| `config_flow.py` | Add validation call in `_run_validations_user()` | ~line 390-435 |
| `__init__.py` | Pass `base_url` to ChatOpenAI instantiation | ~line 318-327 |
| `__init__.py` | Pass `base_url` to OpenAIEmbeddings instantiation | ~line 370-377 |
| `translations/en.json` | Add label for openai_base_url field | ~line 14 |

---

## Part 2: Tool Calling Improvements

### 2.1 Current Issues Identified

**From CODEBASE_ANALYSIS.md & Code Review:**

1. **Minimal Error Distinction** (graph.py lines 294-299)
   - Catches `HomeAssistantError` and `ValidationError` generically
   - Doesn't distinguish between:
     - Input validation failures (bad tool args)
     - Execution failures (service unavailable)
     - Parameter type mismatches
     - Tool not found

2. **No Tool Schema Validation**
   - Tool arguments not validated before invocation
   - Type mismatches detected only at runtime
   - No validation of required vs optional parameters

3. **Limited Logging & Observability**
   - Only basic debug log at line 279: `LOGGER.debug("Tool call: %s(%s)", ...)`
   - No tracking of tool success/failure rates
   - No performance metrics (response time)

4. **No Rate Limiting**
   - Multiple rapid tool calls not throttled
   - Could cause resource exhaustion or API rate limits

5. **Unsafe Tool Response Handling**
   - Tool responses not validated against expected schema
   - No sanitization of responses before feeding to LLM
   - Could cause context pollution or injection attacks

### 2.2 Implementation Scope

#### A. Enhanced Error Handling (`agent/graph.py`)

**New Error Classes** (to distinguish error types):
```python
class ToolExecutionError(Exception):
    """Tool executed but failed to produce valid output."""
    error_type: str  # "validation", "execution", "not_found", "timeout"
    tool_name: str
    details: dict
```

**Updated `_call_tools()` node** (lines 257-318):
- Catch specific exceptions with proper handling
- Return error type to agent (e.g., "Tool parameter 'entity_id' must be string, got int")
- Different retry guidance based on error type

#### B. Tool Schema Validation (`agent/tools.py` or new module)

**New Helper Function:**
```python
def _validate_tool_params(
    tool_schema: dict,
    provided_args: dict,
) -> tuple[bool, str | None]:
    """
    Validate tool arguments against OpenAI schema.
    Returns: (is_valid, error_message)
    """
    # Check required parameters
    # Validate types (coerce where possible)
    # Return helpful error message if invalid
```

**Integration Point:**
- Call before tool invocation in `_call_tools()` (lines 290-313)
- Return validation error with guidance to agent

#### C. Tool Call Tracking & Logging

**Tool Metrics Collector** (new file: `agent/tool_metrics.py`):
```python
@dataclass
class ToolCallMetrics:
    tool_name: str
    success: bool
    error_type: str | None
    duration_ms: float
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

class ToolMetricsCollector:
    def record_call(self, metrics: ToolCallMetrics) -> None:
        """Track tool call for observability."""
        LOGGER.info(
            "Tool call: %s (success=%s, duration=%dms, retries=%d)",
            metrics.tool_name,
            metrics.success,
            metrics.duration_ms,
            metrics.retry_count,
        )
```

#### D. Rate Limiting

**Simple Token Bucket** (in `agent/graph.py`):
```python
class ToolCallRateLimiter:
    def __init__(self, max_calls_per_minute: int = 30):
        self.max_calls = max_calls_per_minute
        self.call_times: deque = deque(maxlen=max_calls_per_minute)
    
    async def check_rate_limit(self) -> bool:
        """Return True if call allowed, False if rate limit exceeded."""
        # Remove calls older than 1 minute
        # Check if we've hit the limit
```

#### E. Tool Response Validation

**Response Sanitization:**
```python
def _sanitize_tool_response(
    response: Any,
    max_length: int = 16000,  # Prevent context bloat
) -> str:
    """
    Validate and sanitize tool response.
    - Check response is JSON-serializable
    - Limit response size
    - Escape harmful content
    - Return as string for ToolMessage
    """
```

### 2.3 Changes Summary

| File | Changes | Scope |
|------|---------|-------|
| `agent/graph.py` | Create error type enum | New section ~line 1 |
| `agent/graph.py` | Update `_call_tools()` with enhanced error handling | Lines 257-318 |
| `agent/graph.py` | Add rate limiter instance to graph config | ~line 100 |
| `agent/tools.py` | Add `_validate_tool_params()` helper | New function |
| `agent/tools.py` | Add `_sanitize_tool_response()` helper | New function |
| `agent/tool_metrics.py` | New file for tool call tracking | New module |
| `const.py` | Add tool-related constants (rate limits, timeouts) | End of file |
| `conversation.py` | Initialize rate limiter and pass to config | ~line 280-312 |

---

## Part 3: Ollama Compatibility for llama.cpp

### 3.1 Verification

✅ **Already Supported:**
- Ollama provider with configurable base_url
- Can point to llama.cpp OpenAI-compatible endpoint
- Works for chat, VLM, summarization, embeddings

⚠️ **After OpenAI Base URL Implementation:**
- Users can choose between:
  - **Ollama provider** → `base_url: http://localhost:8000` (existing)
  - **OpenAI provider** → `base_url: http://localhost:8000` (new) if llama.cpp has `/v1/chat/completions`

### 3.2 Recommendation

Document that llama.cpp can be used via:
1. **Ollama interface** (recommended if using Ollama server)
2. **OpenAI-compatible endpoint** with custom base_url (for llama.cpp direct)

---

## Implementation Phases

### Phase 1: Config Flow Extension (2-3 hours)
1. Add constants
2. Add validation function
3. Update config_flow UI
4. Update translations
5. Test config entry creation/update

### Phase 2: OpenAI Provider Integration (1-2 hours)
1. Update ChatOpenAI instantiation
2. Update OpenAIEmbeddings instantiation
3. Update health check function
4. Test with custom base_url

### Phase 3: Tool Calling Improvements (3-4 hours)
1. Implement error classification
2. Add schema validation
3. Add metrics tracking
4. Implement rate limiting
5. Add response sanitization
6. Test error recovery and chain-of-thought

### Phase 4: Testing & Documentation (1-2 hours)
1. Unit tests for new functions
2. Integration tests (health checks, tool calling)
3. README updates
4. Changelog entry

---

## Risk Assessment

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking change to `__init__.py` | High | Backward compatible (default to official URL) |
| Tool validation too strict | Medium | Gradual rollout, clear error messages |
| Rate limiting blocks legitimate use | Medium | Conservative defaults (30/min), configurable |
| Custom base_url breaks auth | Medium | Validation test before accepting |

---

## Success Criteria

### Part 1: Custom OpenAI Base URL ✓
- [ ] Config flow accepts optional base_url
- [ ] Health check uses custom base_url
- [ ] ChatOpenAI receives base_url parameter
- [ ] Embeddings use custom base_url
- [ ] Works with llama.cpp endpoint
- [ ] Backwards compatible (defaults to official)

### Part 2: Tool Calling Improvements ✓
- [ ] Error types clearly distinguished in logs
- [ ] Invalid tool arguments rejected with helpful guidance
- [ ] Tool response size limited (<16KB)
- [ ] Rate limiting prevents rapid-fire calls
- [ ] Tool metrics recorded and logged
- [ ] Agent retries with corrections (unchanged behavior)

---

## Next Steps (Pending Your Confirmation)

1. **Do you approve this plan?**
2. **Any adjustments needed?**
3. **Priority order preferences?** (Config URL first? Tool hardening first?)
4. **Should we add config.yaml examples for llama.cpp?**

Once confirmed, I will:
1. Implement Part 1 (Config Flow + OpenAI Base URL)
2. Implement Part 2 (Tool Calling Improvements)
3. Create comprehensive tests
4. Provide integration documentation

---

**Status:** ⏳ Awaiting Your Confirmation
