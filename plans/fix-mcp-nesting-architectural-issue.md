# Implementation Plan: Fix MCP Nesting Architectural Issue

**Date:** 2026-01-09
**Status:** READY FOR IMPLEMENTATION
**Priority:** CRITICAL - Core functionality blocked
**Estimated Effort:** 4-6 hours

---

## Executive Summary

The `blender-vfx-orchestrator` MCP server cannot spawn child MCP client connections to `blender-manual` from within MCP tool handlers due to `anyio` TaskGroup conflicts. This blocks the `create_asset` tool entirely.

**Solution:** Convert the 12 `blender-manual` MCP tools to OpenAI Agents SDK `function_tool` wrappers, eliminating MCP transport for inter-agent communication.

---

## Problem Statement

### The Error
```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

### Root Cause
When `create_asset` is called:
1. Claude Code calls `create_asset` tool via MCP stdio transport
2. FastMCP receives request in its `anyio` task group
3. Tool handler calls `await get_orchestrator()`
4. Orchestrator initializes `DocsExpertConnectionPool`
5. Pool calls `MCPServerStdio.connect()` from OpenAI Agents SDK
6. `MCPServerStdio` creates a NEW `anyio` TaskGroup for subprocess management
7. **CRASH**: The child TaskGroup's cancel scope conflicts with parent

### Current Impact
- `create_asset` hangs for 5+ minutes then crashes
- DocsExpert (documentation search) completely disabled
- Self-improvement/learning loop broken
- Workaround: `ENABLE_DOCS_EXPERT=0` (disables feature entirely)

---

## Solution Architecture

### Approach: OpenAI Agents SDK Subagent (Solution 4)

Convert `blender-manual` functions to direct Python imports wrapped with `@function_tool` decorator instead of MCP transport.

```
BEFORE (MCP Nesting - BROKEN):
┌─────────────────────────────────────────────────────────┐
│ Claude Code (MCP Host)                                   │
│   └─> blender-vfx-orchestrator (MCP Server)             │
│         └─> DocsExpert Agent                            │
│               └─> MCPServerStdio (FAILS HERE)           │
│                     └─> blender-manual (MCP Server)     │
└─────────────────────────────────────────────────────────┘

AFTER (Direct Import - WORKS):
┌─────────────────────────────────────────────────────────┐
│ Claude Code (MCP Host)                                   │
│   └─> blender-vfx-orchestrator (MCP Server)             │
│         └─> DocsExpert Agent                            │
│               └─> function_tool wrappers (in-process)   │
│                     └─> blender_docs_tools.py (import)  │
└─────────────────────────────────────────────────────────┘
```

### Why This Solution

1. **Alignment with existing architecture**: Orchestrator already uses OpenAI Agents SDK
2. **No transport conflicts**: `function_tool` runs in-process, no subprocess spawning
3. **Performance**: In-process function calls are faster than MCP stdio IPC
4. **Simplicity**: No connection pools, no subprocess lifecycle management

---

## Implementation Steps

### Phase 1: Extract Shared Search Module (1-2 hours)

**Create new file:** `agents/blender-vfx-orchestrator/shared/blender_docs_tools.py`

This module contains the search logic extracted from `blender-manual/blender_server.py` as standalone functions wrapped with `@function_tool`.

#### 1.1 Copy Core Data Structures and Helpers

Extract from `blender_server.py`:
- `VDB_KEYWORDS` dictionary
- `search_index` list and type definitions
- `compute_score()` function
- `format_results()` function
- `create_snippet()` function
- `load_cache()` function
- `build_index()` function (for fallback/rebuild)
- Semantic search embedding functions (optional)

#### 1.2 Convert 12 Tools to function_tool Wrappers

| Original MCP Tool | New function_tool | Notes |
|-------------------|-------------------|-------|
| `search_manual` | `search_blender_manual` | General keyword search |
| `search_tutorials` | `search_blender_tutorials` | Tutorial/learning resources |
| `browse_hierarchy` | `browse_blender_hierarchy` | Directory tree navigation |
| `search_vdb_workflow` | `search_vdb_workflow` | VDB/NanoVDB specialized |
| `search_python_api` | `search_blender_python_api` | bpy.ops/types docs |
| `search_nodes` | `search_blender_nodes` | Shader/compositor nodes |
| `search_modifiers` | `search_blender_modifiers` | Modifier documentation |
| `read_page` | `read_blender_page` | Full page content |
| `list_api_modules` | `list_blender_api_modules` | API module listing |
| `search_bpy_operators` | `search_bpy_operators` | bpy.ops.* search |
| `search_bpy_types` | `search_bpy_types` | bpy.types.* search |
| `search_semantic` | `search_blender_semantic` | AI embedding search |

#### 1.3 Module Structure

```python
"""
Blender Documentation Search Tools for OpenAI Agents SDK.

Extracted from blender-manual MCP server for in-process use.
Avoids MCP nesting conflicts by using function_tool wrappers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import function_tool

# Configuration
BLENDER_MANUAL_PATH = Path(__file__).parent.parent.parent / "blender-manual"
MANUAL_DIR = BLENDER_MANUAL_PATH / "manual"
PYTHON_API_DIR = BLENDER_MANUAL_PATH / "python_api"
CACHE_FILE = BLENDER_MANUAL_PATH / "manual_index.json"

# Global search index (loaded on first use)
_search_index: List[Dict[str, Any]] = []
_index_loaded = False

# ... helper functions ...

@function_tool
def search_blender_manual(
    query: str,
    limit: int = 5,
    offset: int = 0,
    compact: bool = False
) -> str:
    """
    Search the Blender 5.0 manual and Python API for relevant documentation.

    Args:
        query: Search keywords (e.g., "volume rendering", "export openvdb")
        limit: Maximum results to return (default 5)
        offset: Skip first N results for pagination
        compact: If True, return minimal output (titles/paths only)

    Returns:
        JSON with matching documentation pages, scores, and snippets
    """
    _ensure_index_loaded()
    # ... implementation ...
```

#### 1.4 Index Loading Strategy

Two options:

**Option A: Lazy Loading (Recommended)**
- Load index from cache on first tool call
- ~2 second delay on first search
- No startup overhead

**Option B: Eager Loading**
- Load during orchestrator initialization
- Consistent latency on all calls
- 2 second added startup time

### Phase 2: Update DocsExpert Agent (1 hour)

**Modify:** `agents/blender-vfx-orchestrator/specialized_agents/docs_expert.py`

#### 2.1 Remove MCP Client Code

Delete:
- `DocsExpertAgent` class (uses MCPServerStdio)
- `DocsExpertConnectionPool` class
- `create_docs_expert()` function
- `create_docs_expert_pooled()` function
- `get_docs_connection_pool()` function

#### 2.2 Create New Agent Factory

```python
"""
Documentation Expert Agent using OpenAI Agents SDK.

Uses direct function_tool imports instead of MCP transport
to avoid task group conflicts.
"""

from __future__ import annotations

import os
from typing import Optional

from agents import Agent, ModelSettings
from openai.types.shared import Reasoning

from ..shared.blender_docs_tools import (
    search_blender_manual,
    search_blender_tutorials,
    browse_blender_hierarchy,
    search_vdb_workflow,
    search_blender_python_api,
    search_blender_nodes,
    search_blender_modifiers,
    read_blender_page,
    list_blender_api_modules,
    search_bpy_operators,
    search_bpy_types,
    search_blender_semantic,
)

# Local tools (parameter validation)
from .docs_expert_tools import (
    validate_parameter_range,
    get_parameter_defaults,
)


DOC_EXPERT_INSTRUCTIONS = """..."""  # Keep existing instructions


def create_docs_expert(custom_instructions: str = "") -> Agent:
    """
    Create a documentation expert agent with direct function tools.

    No MCP transport - all tools run in-process.
    """
    instructions = DOC_EXPERT_INSTRUCTIONS
    if custom_instructions:
        instructions = instructions + "\n\n" + custom_instructions

    return Agent(
        name="Documentation Expert",
        instructions=instructions,
        model=os.getenv("DOC_EXPERT_MODEL", "gpt-5.2"),
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium"),
            verbosity="low"
        ),
        tools=[
            # Blender documentation search (12 tools)
            search_blender_manual,
            search_blender_tutorials,
            browse_blender_hierarchy,
            search_vdb_workflow,
            search_blender_python_api,
            search_blender_nodes,
            search_blender_modifiers,
            read_blender_page,
            list_blender_api_modules,
            search_bpy_operators,
            search_bpy_types,
            search_blender_semantic,
            # Parameter validation (2 tools)
            validate_parameter_range,
            get_parameter_defaults,
        ],
    )
```

### Phase 3: Update Orchestrator (30 min)

**Modify:** `agents/blender-vfx-orchestrator/orchestrator.py`

#### 3.1 Remove ENABLE_DOCS_EXPERT Flag

```python
# DELETE this line:
ENABLE_DOCS_EXPERT = os.environ.get("ENABLE_DOCS_EXPERT", "0") == "1"
```

#### 3.2 Simplify Initialize Method

```python
async def initialize(self) -> None:
    if self._initialized:
        return

    print("[Orchestrator] Initializing...", file=sys.stderr)

    # Create all 5 specialized agents (all synchronous now)
    self._script_writer = create_script_writer()
    self._executor = create_executor()
    self._quality_analyst = create_quality_analyst()
    self._learning_agent = create_learning_agent()
    self._docs_expert = create_docs_expert()  # No await needed!

    # Build handoffs list (all 5 agents)
    handoffs_list = [
        handoff(self._script_writer, ...),
        handoff(self._executor, ...),
        handoff(self._quality_analyst, ...),
        handoff(self._learning_agent, ...),
        handoff(self._docs_expert, ...),  # Always enabled now
    ]

    self._orchestrator = Agent(
        name="Blender VFX Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=os.getenv("ORCHESTRATOR_MODEL", "gpt-5.2"),
        handoffs=handoffs_list,
    )

    self._initialized = True
    print("[Orchestrator] Initialization complete", file=sys.stderr)
```

#### 3.3 Remove MCP Connection Cleanup

In `close()` method:
```python
async def close(self) -> None:
    """Clean up resources."""
    # No MCP connections to close anymore
    self._initialized = False
```

### Phase 4: Update MCP Server Entry Point (15 min)

**Modify:** `agents/blender-vfx-orchestrator/server.py`

#### 4.1 Remove DocsExpert Connection Pool References

```python
# DELETE these imports:
from orchestrator import ENABLE_DOCS_EXPERT

# DELETE this from lifespan:
if ENABLE_DOCS_EXPERT:
    try:
        from specialized_agents.docs_expert import get_docs_connection_pool
        pool = get_docs_connection_pool()
        if pool.is_connected:
            await pool.close()
    except Exception:
        pass

# DELETE from get_status():
docs_connected = False
if ENABLE_DOCS_EXPERT:
    ...
```

#### 4.2 Simplify get_status Tool

```python
@mcp.tool()
async def get_status(ctx: Context = None) -> str:
    """Get orchestrator and budget status."""
    # ... budget code stays same ...

    result = {
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "initialized": is_orchestrator_initialized(),
        "budget": { ... },
        "docs_expert_enabled": True,  # Always enabled now
        "project_root": str(PROJECT_ROOT),
    }
    return json.dumps(result, indent=2)
```

### Phase 5: Testing (1 hour)

#### 5.1 Unit Tests

Create `agents/blender-vfx-orchestrator/tests/test_docs_tools.py`:

```python
import pytest
from shared.blender_docs_tools import (
    search_blender_manual,
    search_vdb_workflow,
    read_blender_page,
)

def test_search_manual_basic():
    result = search_blender_manual("volume rendering")
    data = json.loads(result)
    assert "results" in data
    assert len(data["results"]) > 0

def test_search_vdb_workflow():
    result = search_vdb_workflow("export openvdb")
    data = json.loads(result)
    assert "results" in data

def test_read_page():
    result = read_blender_page("render/cycles/world_settings.html")
    data = json.loads(result)
    assert "content" in data or "error" in data
```

#### 5.2 Integration Tests

```bash
# Test via Claude Code MCP
cd agents/blender-vfx-orchestrator
python test_mcp_tools.py

# Expected results:
# - get_status returns in <50ms ✓
# - list_sessions returns in <50ms ✓
# - create_asset initializes orchestrator without hanging ✓
# - create_asset can use docs expert for searches ✓
```

#### 5.3 End-to-End Test

```python
# test_e2e.py
import asyncio
from orchestrator import get_orchestrator, create_vfx_asset
from models.shared_context import AssetRequest, EffectType

async def test_full_pipeline():
    """Test full asset creation with docs expert queries."""
    session = await create_vfx_asset(
        asset_name="test_explosion",
        description="bright orange mushroom cloud",
        effect_type="pyro",
        max_iterations=1,  # Just test one iteration
    )

    assert session.status != "failed"
    print(f"Session: {session.session_id}")
    print(f"Score: {session.best_score}")

asyncio.run(test_full_pipeline())
```

---

## File Changes Summary

| File | Action | Changes |
|------|--------|---------|
| `shared/blender_docs_tools.py` | CREATE | 12 function_tool wrappers + search logic |
| `shared/__init__.py` | CREATE | Module exports |
| `specialized_agents/docs_expert.py` | MODIFY | Remove MCP client, use function_tools |
| `specialized_agents/docs_expert_tools.py` | CREATE | Extract validate/defaults tools |
| `orchestrator.py` | MODIFY | Remove ENABLE_DOCS_EXPERT, simplify init |
| `server.py` | MODIFY | Remove connection pool references |
| `tests/test_docs_tools.py` | CREATE | Unit tests for doc tools |

---

## Rollback Plan

If issues arise:
1. Revert to `ENABLE_DOCS_EXPERT=0` (feature disabled)
2. Git revert commits
3. DocsExpert can be re-enabled via MCP when running standalone (not as MCP server)

---

## Success Criteria

1. ✅ `get_status` returns `docs_expert_enabled: true`
2. ✅ `create_asset` completes without hanging
3. ✅ DocsExpert can search documentation during stuck detection
4. ✅ Full VFX asset generation loop works end-to-end
5. ✅ No `anyio` TaskGroup errors in logs

---

## Dependencies

- OpenAI Agents SDK (`openai-agents` package)
- Existing `blender-manual` cache file (`manual_index.json`)
- Optional: sentence-transformers for semantic search

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cache file missing | Low | Medium | Fallback to build_index() |
| Import path issues | Medium | Low | Careful path configuration |
| Semantic search unavailable | Low | Low | Graceful degradation |
| Performance regression | Low | Low | In-process faster than IPC |

---

## Timeline

| Phase | Estimated | Actual |
|-------|-----------|--------|
| Phase 1: Extract Search Module | 2 hours | TBD |
| Phase 2: Update DocsExpert | 1 hour | TBD |
| Phase 3: Update Orchestrator | 30 min | TBD |
| Phase 4: Update Server | 15 min | TBD |
| Phase 5: Testing | 1 hour | TBD |
| **Total** | **4-6 hours** | TBD |

---

## Appendix: Tool Signatures Reference

All 12 tools to be converted:

```python
# Core Search Tools
search_manual(query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> str
search_tutorials(topic: str, technique: Optional[str] = None, limit: int = 5, compact: bool = False) -> str
browse_hierarchy(path: Optional[str] = None) -> str
search_vdb_workflow(query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> str

# API Documentation Tools
search_python_api(operation: str, limit: int = 5, compact: bool = False, api_only: bool = False) -> str
list_api_modules(category: Optional[str] = None, limit: int = 30) -> str
search_bpy_operators(category: str, operation: Optional[str] = None, limit: int = 5) -> str
search_bpy_types(typename: str, limit: int = 5) -> str

# Node/Modifier Tools
search_nodes(node_type: str, category: Optional[str] = None, limit: int = 5, compact: bool = False) -> str
search_modifiers(modifier_name: Optional[str] = None, limit: int = 5, compact: bool = False) -> str

# Content Tools
read_page(path: str, max_length: int = 4000, source: str = "auto") -> str
search_semantic(query: str, limit: int = 5, compact: bool = False) -> str
```

---

## SpecFlow Gap Analysis Results

The plan underwent SpecFlow analysis which identified 12 gaps (5 critical, 5 important, 2 low). The following sections address these gaps.

### Critical Gaps Addressed

#### Gap 3.1 & 3.2: Error Handling Strategy

All function_tool wrappers must use graceful error handling:

```python
@function_tool
def search_blender_manual(query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> str:
    """Search the Blender 5.0 manual."""
    try:
        _ensure_index_loaded()
        results = _search_index(query, limit, offset)
        return json.dumps({
            "results": results,
            "query": query,
            "count": len(results)
        })
    except FileNotFoundError as e:
        return json.dumps({
            "error": "Documentation index not found",
            "detail": str(e),
            "fallback_suggestion": "Use default Blender parameters or consult online docs"
        })
    except Exception as e:
        return json.dumps({
            "error": "Search failed",
            "detail": str(e),
            "query": query
        })
```

**Decision:** Option A - Return JSON with `{"error": "...", "fallback_suggestion": "..."}`. This allows agents to gracefully degrade without crashing.

#### Gap 3.3: Timeout Configuration

Add per-tool timeouts using `asyncio.wait_for`:

```python
TOOL_TIMEOUTS = {
    "search_blender_manual": 5,
    "search_blender_tutorials": 5,
    "browse_blender_hierarchy": 3,
    "search_vdb_workflow": 5,
    "search_blender_python_api": 5,
    "search_blender_nodes": 5,
    "search_blender_modifiers": 5,
    "read_blender_page": 10,
    "list_blender_api_modules": 5,
    "search_bpy_operators": 5,
    "search_bpy_types": 5,
    "search_blender_semantic": 30,  # Loads embedding model
}

async def with_timeout(coro, timeout_seconds: float, fallback_message: str):
    """Wrap coroutine with timeout and graceful fallback."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return json.dumps({
            "error": "Tool call timed out",
            "timeout_seconds": timeout_seconds,
            "fallback_suggestion": fallback_message
        })
```

#### Gap 3.4: Session State Migration

Add schema versioning to session state:

```python
# In utils/session_persistence.py
SESSION_SCHEMA_VERSION = 2  # Increment when breaking changes made

def migrate_session_state(session: SessionState) -> SessionState:
    """Migrate old session states to current schema."""
    if session.schema_version < 2:
        # Convert any MCP tool references to function_tool references
        # Tool names are the same, just transport changed
        session.schema_version = 2
    return session
```

**Decision:** No breaking changes to tool names or signatures. Sessions can resume without migration.

#### Gap 3.6: Integration Test Coverage

Add specific test cases in `tests/test_mcp_nesting_fix.py`:

```python
import pytest
from unittest.mock import patch

def test_no_mcp_nesting_in_docs_expert():
    """Verify DocsExpert never spawns child MCP client."""
    with patch("mcp.client.stdio.stdio_client") as mock_mcp:
        from orchestrator import get_orchestrator
        import asyncio
        asyncio.run(get_orchestrator())
        assert mock_mcp.call_count == 0, "DocsExpert spawned MCP client!"

def test_function_tool_error_handling():
    """Verify graceful error handling in wrappers."""
    from shared.blender_docs_tools import search_blender_manual
    with patch("shared.blender_docs_tools._search_index", side_effect=FileNotFoundError):
        result = search_blender_manual("test query")
        data = json.loads(result)
        assert "error" in data
        assert "fallback_suggestion" in data

def test_docs_expert_creation_is_synchronous():
    """Verify DocsExpert creation doesn't require await."""
    from specialized_agents.docs_expert import create_docs_expert
    agent = create_docs_expert()  # No await!
    assert agent is not None
    assert "Documentation Expert" in agent.name
```

#### Gap 3.8: Rollback Strategy with Feature Flag

Add feature flag for quick rollback:

```python
# In orchestrator.py
import os

# Feature flag for rollback capability
USE_IN_PROCESS_DOCS = os.environ.get("BLENDER_VFX_DOCS_TRANSPORT", "in_process") == "in_process"

async def initialize(self) -> None:
    # ... other agents ...

    if USE_IN_PROCESS_DOCS:
        # New: In-process function_tool wrappers (default)
        from specialized_agents.docs_expert import create_docs_expert
        self._docs_expert = create_docs_expert()
    else:
        # Fallback: MCP transport (only for standalone testing, NOT MCP server)
        print("[Orchestrator] WARNING: MCP transport for docs only works standalone", file=sys.stderr)
        from specialized_agents.docs_expert_mcp import create_docs_expert_mcp
        self._docs_expert = await create_docs_expert_mcp()
```

#### Gap 3.9: Dependency Validation

Add dependency checks at module load:

```python
# In shared/blender_docs_tools.py
import sys

# Validate Python version
if sys.version_info < (3, 10):
    raise RuntimeError("blender_docs_tools requires Python 3.10+")

# Validate optional dependencies
_SEMANTIC_SEARCH_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("sentence-transformers not installed, semantic search disabled")

# Required dependencies
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise RuntimeError("beautifulsoup4 required: pip install beautifulsoup4")
```

### Important Gaps Addressed

#### Gap 3.5: Cache Strategy

Use per-module lazy loading, not global LRU cache:

```python
# Module-level state (not per-session, but safe)
_search_index: List[Dict[str, Any]] = []
_index_loaded = False
_embeddings: Optional[np.ndarray] = None
_embedding_model: Optional[Any] = None

def _ensure_index_loaded():
    """Lazy load index on first use."""
    global _search_index, _index_loaded
    if _index_loaded:
        return

    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            _search_index = cache.get("pages", [])
    else:
        # Build index from HTML files (slow but only once)
        _search_index = _build_index()
        _save_cache(_search_index)

    _index_loaded = True
```

**Decision:** Module-level state is acceptable since:
- Index is read-only after load
- Same results regardless of caller
- No user-specific data in cache

#### Gap 3.10: Structured Logging

Add logging to all function_tool wrappers:

```python
import logging
import sys

logger = logging.getLogger("blender_docs_tools")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    "[%(name)s][%(levelname)s] %(message)s"
))
logger.addHandler(handler)

@function_tool
def search_blender_manual(query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> str:
    """Search the Blender 5.0 manual."""
    logger.info(f"search_blender_manual(query='{query[:50]}...', limit={limit})")
    try:
        result = _search_impl(query, limit, offset, compact)
        logger.info(f"search_blender_manual returned {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"search_blender_manual failed: {e}")
        raise
```

#### Gap 3.12: Input Validation

Add Pydantic validation for all tool inputs:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class SearchManualArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(5, ge=1, le=30)
    offset: int = Field(0, ge=0)
    compact: bool = False

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Remove potential path traversal
        return v.replace("..", "").replace("/", " ").strip()

@function_tool
def search_blender_manual(query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> str:
    """Search the Blender 5.0 manual."""
    args = SearchManualArgs(query=query, limit=limit, offset=offset, compact=compact)
    return _search_impl(args.query, args.limit, args.offset, args.compact)
```

### Updated File Changes Summary

| File | Action | Changes |
|------|--------|---------|
| `shared/blender_docs_tools.py` | CREATE | 12 function_tool wrappers + error handling + logging + validation |
| `shared/__init__.py` | CREATE | Module exports |
| `specialized_agents/docs_expert.py` | MODIFY | Remove MCP client, use function_tools, add feature flag |
| `specialized_agents/docs_expert_tools.py` | CREATE | Extract validate/defaults tools |
| `specialized_agents/docs_expert_mcp.py` | CREATE | Fallback MCP implementation (for standalone use only) |
| `orchestrator.py` | MODIFY | Remove ENABLE_DOCS_EXPERT, add USE_IN_PROCESS_DOCS flag |
| `server.py` | MODIFY | Remove connection pool references |
| `tests/test_docs_tools.py` | CREATE | Unit tests for doc tools |
| `tests/test_mcp_nesting_fix.py` | CREATE | Integration tests for MCP nesting fix |

### Critical Questions Answered

| Question | Answer | Rationale |
|----------|--------|-----------|
| Q1: Error handling strategy | Option A: Return JSON with `{"error": ...}` | Allows graceful degradation |
| Q2: Session state migration | No migration needed | Tool names unchanged, only transport changed |
| Q3: Dependency conflicts | Validate at module load | Early failure with clear error message |
| Q4: Cache scope | Module-level (shared) | Read-only index, no user data |
| Q5: Timeouts | Per-tool config (3-30s) | Prevents hung agents |
| Q6: Logging format | Structured with tool name prefix | Clear differentiation from MCP calls |

---

**Document Author:** Claude Code
**Last Updated:** 2026-01-09
