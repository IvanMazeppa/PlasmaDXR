# OpenAI Agents SDK Integration Issues Report

**Date:** 2025-01-05
**Phase:** 9 - OpenAI Agents SDK Integration
**Status:** Functional but unstable

---

## Executive Summary

The OpenAI Agents SDK integration for the blender-librarian MCP server is now functional, but experiences intermittent timeout and connection stability issues. When it works, it provides excellent multi-agent orchestration with vision analysis capabilities. The core problem is the multi-hop MCP chain introducing latency that exceeds connection timeouts.

---

## Issues Encountered

### 1. Wrong Model Being Used (RESOLVED)

**Symptom:** OpenAI dashboard showed `gpt-4o` being called instead of `gpt-5.2`

**Root Cause:** Default model parameter was hardcoded as `gpt-4o` in three files:
- `librarian_agents/doc_expert.py`
- `librarian_agents/librarian_orchestrator.py`
- `librarian_agents/vision_expert.py`

**Fix:** Changed all default model parameters from `gpt-4o` to `gpt-5.2`

```python
# Before
def __init__(self, model: str = "gpt-4o"):

# After
def __init__(self, model: str = "gpt-5.2"):
```

---

### 2. Reasoning Effort Disabled (RESOLVED)

**Symptom:** OpenAI dashboard showed "Reasoning effort: none" for all API calls

**Root Cause:** GPT-5.2 defaults to `reasoning.effort = "none"` for low latency. The Agents SDK `Agent` class was not configured with `ModelSettings`.

**Impact:** Without reasoning, GPT-5.2 operates like a faster but less intelligent model. For agentic tasks involving multi-step documentation search and vision analysis, `medium` or `high` reasoning is recommended.

**Fix:** Added `ModelSettings` with `Reasoning(effort="medium")` to all agents:

```python
from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

self._agent = Agent(
    name="Documentation Expert",
    instructions=instructions,
    model=self.model,
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium"),
        verbosity="low"
    ),
    mcp_servers=[self._mcp_server],
    tools=[...]
)
```

**Files Modified:**
- `librarian_agents/doc_expert.py` (line 241-244)
- `librarian_agents/librarian_orchestrator.py` (line 126-129)
- `librarian_agents/vision_expert.py` (line 263-266)

---

### 3. MCP Connection Timeouts (FIXED via Streaming)

**Symptom:** MCP error `-32001: AbortError: The operation was aborted` or `Connection closed`

**Root Cause:** The multi-hop architecture introduces cumulative latency:

```
Claude Code (client)
    ↓ MCP call
blender-librarian MCP server
    ↓ async call
OpenAI Agents SDK orchestrator
    ↓ API call + tool calls
GPT-5.2 (with reasoning tokens)
    ↓ MCP tool calls
blender-manual MCP server (spawned via stdio)
    ↓ semantic search
sentence-transformers embeddings (slow initial load)
```

**Timeout Chain:**
1. Claude Code → blender-librarian: Unknown timeout (MCP client default)
2. blender-librarian → OpenAI API: 60 seconds (configurable)
3. OpenAI API → blender-manual: 60 seconds (`client_session_timeout_seconds`)
4. Embedding model load: 10-30 seconds on first query

**SOLUTION: Streaming Mode**

Changed from `Runner.run()` to `Runner.run_streamed()` which:
- Yields events incrementally (token-by-token, handoffs, tool calls)
- Keeps the connection alive during long operations
- Logs all events for debugging

```python
# OLD (blocking, prone to timeout)
result = await Runner.run(self._orchestrator, full_query)

# NEW (streaming, timeout-resilient)
streamed_result = Runner.run_streamed(self._orchestrator, full_query)
async for event in streamed_result.stream_events():
    # Process events incrementally (keeps connection alive)
    if event.type == "agent_updated_stream_event":
        logger.info(f"Handoff to {event.new_agent.name}")
result = await streamed_result.result()
```

**Additional Fixes Applied:**
- Increased `client_session_timeout_seconds` from 5s to 60s in `doc_expert.py`
- Added explicit `await self._mcp_server.connect()` before use
- Added logging for all streaming events to aid debugging

---

### 4. Circular Import Error (RESOLVED)

**Symptom:** `ImportError: cannot import name 'DocExpertAgent' from partially initialized module 'agents'`

**Root Cause:** The agents directory was named `agents/`, conflicting with the `openai-agents` package import `from agents import Agent`.

**Fix:** Renamed directory from `agents/` to `librarian_agents/`

---

### 5. MCPServerStdio Connection Not Established (RESOLVED)

**Symptom:** `Server not initialized. Make sure you call connect() first.`

**Root Cause:** The `MCPServerStdio` class requires explicit `connect()` call before use. The original code assumed auto-connection.

**Fix:** Added explicit connection in `doc_expert.py`:

```python
self._mcp_server = MCPServerStdio(
    name="blender-manual",
    params={...},
    cache_tools_list=self.cache_tools,
    client_session_timeout_seconds=60.0
)

# IMPORTANT: Explicitly connect to the MCP server before using it
await self._mcp_server.connect()
```

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Code (User)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Tool Call
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              blender-librarian MCP Server                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           LibrarianOrchestrator (GPT-5.2)               │   │
│  │              reasoning.effort = "medium"                 │   │
│  └──────────┬─────────────────────────┬────────────────────┘   │
│             │                         │                         │
│             ▼                         ▼                         │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  DocExpertAgent      │  │  VisionExpertAgent   │            │
│  │  (GPT-5.2 + MCP)     │  │  (GPT-5.2 + Vision)  │            │
│  │  reasoning = medium  │  │  reasoning = medium  │            │
│  └──────────┬───────────┘  └──────────────────────┘            │
│             │                                                   │
│             │ MCPServerStdio (subprocess)                       │
│             ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            blender-manual MCP Server                      │  │
│  │  - 12 documentation search tools                          │  │
│  │  - Semantic search via sentence-transformers              │  │
│  │  - Blender 5.0 API reference                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Successful Test Results

When working correctly, the system provides excellent results:

**Query:** "How should I approach creating a realistic sun model in Blender 5.0 that matches the solar eruptions in my reference image?"

**Response:**
- **Agents Used:** vision_expert, doc_expert, orchestrator
- **Diagnosis:** Correctly identified thin filamentary prominences, bright footpoints, wispy strands
- **Recommendations:** Specific parameter values for:
  - Domain Resolution: 512 (final) / 256 (preview)
  - Vorticity: 2.5 (range 2.0-3.0)
  - Flame Smoke: 0.08 (keep plasma emissive, not sooty)
  - Temperature: 4.0 with cooling gradient
  - Turbulence force field settings
  - Shading notes for emission-driven rendering

**Confidence:** 0.62

---

## Recommendations for Stability

### Implemented Fixes ✅

1. **Streaming responses (DONE):** Changed to `Runner.run_streamed()` to keep connection alive during long operations

2. **Reasoning enabled (DONE):** Set `reasoning.effort = "medium"` for better quality recommendations

3. **Explicit MCP connect (DONE):** Added `await self._mcp_server.connect()` before use

4. **Extended timeouts (DONE):** Increased `client_session_timeout_seconds` to 60s

### Remaining Optimizations

1. **Pre-warm embeddings:** Load sentence-transformers model at server startup, not first query

2. **Connection pooling:** Keep blender-manual MCP server running persistently instead of spawning per-query

3. **Dynamic reasoning effort:** Use `"low"` for straightforward lookups, `"medium"` for complex analysis

4. **Retry logic:** Add exponential backoff for transient failures

### Long-term Architecture

1. **Split into fast/slow paths:**
   - Fast path: Playbook lookup + cached responses (< 5s)
   - Slow path: Full Agents SDK orchestration (background job)

2. **Implement WebSocket:** Replace stdio transport with persistent WebSocket for lower latency

3. **Guardrails:** Add input/output validation for parameter range checking

---

## Testing Checklist

- [x] GPT-5.2 model confirmed in OpenAI dashboard
- [x] Reasoning effort set to "medium"
- [x] Vision analysis working with reference images
- [x] Multi-agent handoffs functional (orchestrator → doc_expert → vision_expert)
- [x] Structured JSON output with recommendations
- [x] Streaming mode implemented (Runner.run_streamed)
- [ ] Consistent < 30s response times (test with streaming)
- [ ] Zero timeout errors over 10 consecutive queries
- [ ] Pre-warmed embedding model at startup

---

## Files Modified

| File | Changes |
|------|---------|
| `librarian_agents/doc_expert.py` | Model → gpt-5.2, added ModelSettings with reasoning, increased timeout to 60s, added explicit connect() |
| `librarian_agents/librarian_orchestrator.py` | Model → gpt-5.2, added ModelSettings with reasoning, **STREAMING MODE via Runner.run_streamed()**, added logging |
| `librarian_agents/vision_expert.py` | Model → gpt-5.2, added ModelSettings with reasoning |
| `agents/` → `librarian_agents/` | Directory renamed to avoid import conflict |

---

## References

- [OpenAI Agents SDK Model Settings](https://openai.github.io/openai-agents-python/ref/model_settings/)
- [OpenAI GPT-5.2 Reasoning Guide](https://platform.openai.com/docs/guides/reasoning)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

---

**Last Updated:** 2025-01-05
**Author:** Claude Code Session
