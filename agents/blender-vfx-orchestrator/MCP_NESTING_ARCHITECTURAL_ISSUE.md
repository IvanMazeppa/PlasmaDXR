# MCP Nesting Architectural Issue - Technical Analysis

**Date:** 2026-01-09
**Status:** BLOCKING - Prevents `create_asset` from functioning
**Priority:** CRITICAL - Core functionality disabled

---

## Executive Summary

The `blender-vfx-orchestrator` MCP server cannot spawn child MCP client connections (to `blender-manual`) from within MCP tool handlers. This is a fundamental architectural conflict with the `anyio` async task group system used by the `mcp` Python SDK.

**The Error:**
```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

**Impact:** The `create_asset` tool hangs for 5+ minutes then crashes. Without `blender-manual` integration, the self-improvement/learning loop is disabled.

---

## System Architecture

### Current Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code / Cursor                         │
│                    (MCP Host Process)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ stdio transport
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              blender-vfx-orchestrator (MCP Server)              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FastMCP Server                        │   │
│  │  Tools: get_status, create_asset, list_sessions, etc.   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ Tool handler calls                │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BlenderVFXOrchestrator                      │   │
│  │  - OpenAI Agents SDK orchestration                       │   │
│  │  - DocsExpertConnectionPool                              │   │
│  │  - MCPConnectionPool                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ Attempts to spawn child MCP       │
│                              ▼ client (FAILS HERE)              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           stdio_client (mcp SDK)                         │   │
│  │  - Creates asyncio subprocess                            │   │
│  │  - Enters anyio TaskGroup                                │   │
│  │  - CONFLICT: Nested task group in different task         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼ (Never reaches)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              blender-manual (MCP Server)                 │   │
│  │  - 4227 Blender documentation pages                      │   │
│  │  - Semantic search with embeddings                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### The Problem

When `create_asset` is called:

1. **Claude Code** calls `create_asset` tool via MCP stdio transport
2. **FastMCP** receives request in its `anyio` task group
3. **Tool handler** calls `await get_orchestrator()`
4. **Orchestrator** initializes `DocsExpertConnectionPool`
5. **DocsExpertConnectionPool** calls `stdio_client()` from `mcp` SDK
6. **stdio_client** creates a NEW `anyio` TaskGroup for subprocess management
7. **CRASH**: The child TaskGroup's cancel scope is in a different task than the parent

---

## Root Cause Analysis

### The `anyio` Cancel Scope Constraint

The `anyio` library enforces a strict rule: **cancel scopes must be exited in the same task they were entered**. This is a safety mechanism to prevent resource leaks and ensure proper cleanup.

```python
# This is NOT allowed in anyio:
async def outer_task():
    async with anyio.create_task_group() as tg:
        # Starting a child process here that creates its own task group
        # The child's task group is in outer_task's context
        # But when the child tries to manage its own cancel scopes,
        # they may run in different internal tasks
        await start_child_mcp_client()  # CRASH
```

### Code Flow Leading to Crash

**File: `server.py`**
```python
@mcp.tool()
async def create_asset(...) -> str:
    """MCP tool handler - runs in FastMCP's task group."""
    orchestrator = await get_orchestrator()  # Triggers initialization
    return await create_vfx_asset(orchestrator, ...)
```

**File: `orchestrator.py`**
```python
async def get_orchestrator() -> BlenderVFXOrchestrator:
    if _orchestrator is None:
        _orchestrator = BlenderVFXOrchestrator()
        await _orchestrator.initialize()  # Triggers MCP client spawn
    return _orchestrator
```

**File: `orchestrator.py` - `initialize()`**
```python
async def initialize(self):
    # This creates child MCP client connections
    self._docs_expert_pool = DocsExpertConnectionPool()
    await self._docs_expert_pool.connect()  # CRASH POINT
```

**File: `librarian_agents/doc_expert.py`**
```python
class DocsExpertConnectionPool:
    async def connect(self):
        # Uses mcp SDK's stdio_client which creates TaskGroup
        self._client = await stdio_client(
            StdioServerParameters(
                command="uv",
                args=["run", "blender-manual-server"],
            )
        )
        # ^^^ This spawns subprocess with its own anyio task group
        # The task group conflict occurs here
```

### The `mcp` SDK's `stdio_client`

The `mcp` Python SDK's `stdio_client` function (from `mcp.client.stdio`) uses `anyio` internally:

```python
# Simplified view of what stdio_client does:
async def stdio_client(params: StdioServerParameters):
    async with anyio.create_task_group() as tg:
        # Spawn subprocess for child MCP server
        process = await anyio.open_process(...)

        # Background task for reading stdout
        tg.start_soon(read_stdout_task)

        # Background task for writing stdin
        tg.start_soon(write_stdin_task)

        # ... client operations ...
```

When this runs inside an MCP tool handler (which is already inside FastMCP's task group), the nested task groups conflict.

---

## Evidence from Logs

### Log File: `2026-01-07T02-01-46-239Z.jsonl`

**Status tools work fine (no MCP client spawn):**
```json
{"timestamp": "2026-01-07T02:01:46.528Z", "method": "tools/call", "params": {"name": "get_status"}}
{"timestamp": "2026-01-07T02:01:46.539Z", "result": {"content": [{"type": "text", "text": "{\"orchestrator_version\": \"1.0.0\", \"initialized\": false, ...}"}]}}
// Duration: 11ms ✓
```

**`create_asset` triggers blender-manual initialization:**
```json
{"timestamp": "2026-01-07T02:01:54.892Z", "method": "tools/call", "params": {"name": "create_asset", "arguments": {...}}}
// blender-manual starts (4227 pages, embeddings loading)
{"timestamp": "2026-01-07T02:02:01.200Z", "level": "INFO", "message": "blender-manual server started"}
```

**Then 5-minute hang followed by crash:**
```json
{"timestamp": "2026-01-07T02:07:XX.XXX", "level": "ERROR", "message": "RuntimeError: Attempted to exit cancel scope in a different task than it was entered in"}
```

---

## Affected Code Locations

| File | Class/Function | Issue |
|------|----------------|-------|
| `server.py` | `create_asset()` | Tool handler that triggers initialization |
| `server.py` | `resume_session()` | Also triggers initialization |
| `orchestrator.py` | `get_orchestrator()` | Singleton that creates MCP connections |
| `orchestrator.py` | `BlenderVFXOrchestrator.initialize()` | Spawns DocsExpert pool |
| `librarian_agents/doc_expert.py` | `DocsExpertConnectionPool.connect()` | Calls `stdio_client()` |
| `mcp_connection_pool.py` | `MCPConnectionPool` | Generic MCP client spawner |

---

## Potential Solutions

### Solution 1: Eliminate MCP Client Spawning (Direct Import)

**Approach:** Import `blender-manual` functions directly instead of spawning as child MCP server.

**Pros:**
- No subprocess management
- No task group conflicts
- Simpler architecture

**Cons:**
- Loses MCP transport abstraction
- Tight coupling between servers
- May have import conflicts

**Implementation:**
```python
# Instead of MCP client connection:
# from mcp.client.stdio import stdio_client

# Direct import:
from agents.blender_manual.server import search_manual, search_vdb_workflow

class DocsExpertDirectImport:
    async def search(self, query: str) -> str:
        return await search_manual(query)
```

### Solution 2: HTTP Transport Instead of stdio

**Approach:** Run `blender-manual` as HTTP server, connect via HTTP client.

**Pros:**
- No subprocess spawning from tool handler
- HTTP clients don't have task group issues
- Standard REST patterns

**Cons:**
- Requires running blender-manual separately
- Additional complexity (port management, lifecycle)
- Changes deployment model

**Implementation:**
```python
# blender-manual as HTTP server (separate process)
# uvicorn blender_manual.server:app --port 8001

class DocsExpertHTTP:
    async def search(self, query: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/search",
                json={"query": query}
            )
            return response.json()
```

### Solution 3: Pre-spawn MCP Clients Before Tool Registration

**Approach:** Initialize all MCP client connections during server startup, before FastMCP registers tools.

**Pros:**
- MCP connections established outside tool handler context
- Reuses existing MCP architecture

**Cons:**
- Startup delay (must wait for all connections)
- Connections may timeout during idle periods
- Complex lifecycle management

**Implementation:**
```python
# server.py - Initialize before mcp.tool() decorators
_pre_initialized_clients = {}

async def pre_initialize():
    """Called before FastMCP starts accepting requests."""
    pool = DocsExpertConnectionPool()
    await pool.connect()  # Safe here - not inside tool handler
    _pre_initialized_clients['docs_expert'] = pool

# Main entry point
if __name__ == "__main__":
    asyncio.run(pre_initialize())
    mcp.run()
```

### Solution 4: OpenAI Agents SDK Subagent (Recommended)

**Approach:** Convert `blender-manual` to an OpenAI Agents SDK subagent that runs in-process.

**Pros:**
- Native integration with existing Agents SDK architecture
- No MCP client spawning needed
- Agents SDK handles tool routing
- Aligns with existing 5-agent orchestration design

**Cons:**
- Requires rewriting blender-manual as Agent
- Need to port search functions to Agent tools

**Implementation:**
```python
from agents import Agent, function_tool

# blender-manual as Agents SDK agent (not MCP server)
blender_docs_agent = Agent(
    name="BlenderDocsAgent",
    instructions="Search Blender documentation to answer questions.",
    tools=[
        function_tool(search_manual),
        function_tool(search_vdb_workflow),
        function_tool(search_python_api),
    ]
)

# In orchestrator.py
class BlenderVFXOrchestrator:
    def __init__(self):
        self.docs_agent = blender_docs_agent  # No MCP spawn

    async def search_docs(self, query: str) -> str:
        result = await Runner.run(self.docs_agent, query)
        return result.final_output
```

### Solution 5: Separate Process with IPC

**Approach:** Run MCP client management in a separate process, communicate via queue.

**Pros:**
- Complete isolation from FastMCP's task groups
- Can use any MCP transport

**Cons:**
- Complex IPC setup
- Serialization overhead
- Process lifecycle management

---

## Recommended Solution

**Solution 4 (OpenAI Agents SDK Subagent)** is recommended because:

1. **Alignment with existing architecture**: The orchestrator already uses OpenAI Agents SDK with 5 specialized agents
2. **No transport conflicts**: Agents SDK handles tool invocation internally
3. **Simplicity**: The `blender-manual` functionality (search functions) can be wrapped as `function_tool` decorators
4. **Performance**: In-process function calls are faster than IPC

### Migration Path

1. **Extract search functions** from `blender-manual` server.py
2. **Create `BlenderDocsAgent`** with those functions as tools
3. **Replace `DocsExpertConnectionPool`** with direct agent reference
4. **Remove MCP client spawning** code entirely

---

## Testing Verification

After implementing the fix, verify with:

```bash
# 1. Run test script
cd agents/blender-vfx-orchestrator
python test_mcp_tools.py

# 2. Test via Claude Code
# Call get_status - should return in <20ms
# Call create_asset - should NOT hang, should complete or show progress
```

**Success Criteria:**
- `get_status` returns in <50ms ✓ (already fixed)
- `list_sessions` returns in <50ms ✓ (already fixed)
- `create_asset` initializes orchestrator without hanging
- `create_asset` can search blender docs for parameter help
- Full asset generation loop completes

---

## Files to Modify

| File | Change Required |
|------|-----------------|
| `orchestrator.py` | Remove `DocsExpertConnectionPool`, add `BlenderDocsAgent` |
| `librarian_agents/doc_expert.py` | Convert to Agents SDK agent with function tools |
| `mcp_connection_pool.py` | Remove or deprecate (no longer needed) |
| `server.py` | Update tool handlers to use new docs agent |
| `requirements.txt` | Ensure `openai-agents` is the only MCP-related dep |

---

## Appendix: Error Stack Trace

```
Traceback (most recent call last):
  File "orchestrator.py", line XX, in initialize
    await self._docs_expert_pool.connect()
  File "librarian_agents/doc_expert.py", line XX, in connect
    self._client = await stdio_client(params)
  File "mcp/client/stdio.py", line XX, in stdio_client
    async with anyio.create_task_group() as tg:
  File "anyio/_backends/_asyncio.py", line XX, in __aexit__
    raise RuntimeError(
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

---

**Document Version:** 1.0
**Author:** Claude Code
**For:** Multi-agent debugging deployment
