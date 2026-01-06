# MCP Cancel Scope Fix: Nested Async Generator Issue

**Date:** 2026-01-06
**Issue:** `search_docs_with_agents_sdk` MCP tool always aborted with error -32001
**Resolution Time:** ~6 hours (would have been ~30 minutes with proper logs)
**Root Cause:** anyio cancel scope crossing task boundaries when MCPServerStdio spawned inside MCP tool

---

## TL;DR

**Problem:** MCP tool worked on OpenAI dashboard but always timed out in Claude Code CLI.

**Root Cause:** Spawning `MCPServerStdio` subprocess inside an MCP async tool causes anyio cancel scope issues.

**Fix:** Add FastMCP `lifespan` handler to pre-warm the MCP connection pool at server startup.

**Key Lesson:** MCP server logs are at `~/.cache/claude-cli-nodejs/-<project-path>/mcp-logs-<server-name>/` - check these FIRST when debugging MCP issues!

---

## Symptoms

1. Calling `search_docs_with_agents_sdk` via Claude Code CLI always failed with:
   ```
   MCP error -32001: AbortError: The operation was aborted.
   ```

2. The OpenAI dashboard showed successful API calls completing in ~30-40 seconds

3. The tool appeared to hang indefinitely until manually stopped

4. Other MCP tools in the same server worked fine (non-async, simpler tools)

---

## Failed Debugging Attempts

Before finding the actual logs, we tried several fixes that didn't work:

### 1. Internal Timeout (25s)
Added `asyncio.wait_for()` with 25-second timeout to return before MCP's ~30s timeout.
- **Result:** Still failed - the hang wasn't in our code

### 2. Progress Notifications
Added `ctx.report_progress()` calls throughout the tool to keep the MCP connection alive.
- **Result:** Still failed - progress notifications don't help with cancel scope issues

### 3. Streaming Mode
Used `Runner.run_streamed()` instead of `Runner.run()` to yield events incrementally.
- **Result:** Still failed - the issue was in subprocess spawning, not streaming

### 4. Request Timeout Parameter
Tried `FastMCP("blender-librarian", request_timeout=120)`.
- **Result:** `TypeError` - MCP SDK 1.25.0 doesn't support this parameter

---

## The Breakthrough: Finding MCP Server Logs

**Critical Discovery:** Claude Code CLI caches MCP server logs at:

```
~/.cache/claude-cli-nodejs/-<escaped-project-path>/mcp-logs-<server-name>/
```

For this project:
```
~/.cache/claude-cli-nodejs/-home-maz3ppa-projects-PlasmaDXR/mcp-logs-blender-librarian/
```

The logs are `.jsonl` files with timestamped entries showing:
- Tool call start/end times
- Actual error messages and stack traces
- Debug output from the server

---

## What the Logs Revealed

```json
{"debug":"Tool 'search_docs_with_agents_sdk' failed after 403s: MCP error -32001: AbortError"}
{"error":"RuntimeError: Attempted to exit cancel scope in a different task than it was entered in"}
{"error":"RuntimeError: Attempted to exit a cancel scope that isn't the current tasks's current cancel scope"}
```

**Key Insight:** The tool ran for **403 seconds** (6.7 minutes!) before aborting. The OpenAI API work completed quickly (~39 seconds), but the cleanup hung.

The stack trace pointed to:
```
File ".../mcp/client/stdio/__init__.py", line 189, in stdio_client
    yield read_stream, write_stream
GeneratorExit

RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

---

## Root Cause Analysis

### The Architecture

```
Claude Code CLI
    └── blender-librarian MCP Server (FastMCP)
            └── search_docs_with_agents_sdk (async tool)
                    └── LibrarianOrchestrator.initialize()
                            └── create_doc_expert_pooled()
                                    └── MCPConnectionPool.get_server()
                                            └── MCPServerStdio (spawns subprocess)
                                                    └── blender-manual MCP Server
```

### The Problem

When `MCPServerStdio` is created **inside** an MCP async tool:

1. The subprocess spawns with its own async context
2. The `stdio_client` async generator yields streams for communication
3. When the tool completes, Python tries to clean up the async generator
4. The cleanup runs in a different task than the one that created it
5. anyio's cancel scope validation fails with `RuntimeError`

### Why It Worked in Direct Tests

When calling the async function directly with `asyncio.run()`:
- The subprocess runs in the same task context
- Cleanup works (though still shows warnings)
- Results are returned before the error

When called via MCP:
- FastMCP runs tools in its own task context
- The subprocess is in a different task hierarchy
- Cancel scope validation fails catastrophically

---

## The Fix

### Solution: Pre-warm MCP Connections at Server Startup

FastMCP supports a `lifespan` parameter for startup/shutdown logic. By spawning the MCPServerStdio subprocess **before** any tools are called, it runs in the main server task context.

### Implementation

```python
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context

@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Pre-warm MCP connection pool at startup."""
    _logger.info("Pre-warming MCP connection pool...")

    try:
        from librarian_agents.doc_expert import get_connection_pool
        pool = get_connection_pool()
        await pool.get_server()  # Spawns subprocess in main task
        _logger.info("MCP connection pool pre-warmed successfully")
    except Exception as e:
        _logger.warning(f"Failed to pre-warm (will retry on first use): {e}")

    yield  # Server runs here

    # Shutdown cleanup
    try:
        pool = get_connection_pool()
        await pool.close()
    except Exception as e:
        _logger.warning(f"Error closing pool: {e}")

# Create server with lifespan handler
mcp = FastMCP("blender-librarian", lifespan=_lifespan)
```

### Why This Works

1. Server starts → lifespan runs → MCPServerStdio subprocess spawns
2. Subprocess is now in the main server task context
3. Tool calls reuse the existing connection (no new subprocess)
4. No cancel scope boundary crossing
5. Cleanup works correctly when server shuts down

---

## Verification

After implementing the fix and restarting the MCP server:

```python
# Documentation Expert - SUCCESS
result = await search_docs_with_agents_sdk(
    query="What is flame_smoke used for?",
    effect_type="sun"
)
# Returns valid JSON with answer, confidence 0.92

# Vision Expert - SUCCESS
result = await search_docs_with_agents_sdk(
    query="What's wrong with this render?",
    include_vision=True,
    render_path="/path/to/render.png"
)
# Returns diagnosis, recommendations, confidence 0.66
```

---

## Key Lessons Learned

### 1. Always Check MCP Server Logs First

```bash
# Location pattern:
~/.cache/claude-cli-nodejs/-<escaped-project-path>/mcp-logs-<server-name>/

# Example:
ls ~/.cache/claude-cli-nodejs/-home-maz3ppa-projects-PlasmaDXR/mcp-logs-blender-librarian/
```

These logs show the **actual** error, not just "operation aborted".

### 2. Subprocess Spawning in Async Tools is Dangerous

Never spawn long-lived subprocesses inside MCP async tools. Use:
- Lifespan handlers for initialization
- Connection pools initialized at startup
- Persistent connections rather than per-request spawning

### 3. anyio Cancel Scope Errors Point to Task Boundary Issues

When you see:
```
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

Look for:
- Async generators crossing task boundaries
- Subprocess spawning inside async contexts
- Context managers used across different tasks

### 4. MCP Timeout ≠ Your Code's Timeout

The MCP -32001 error doesn't mean your code timed out. It means:
- The MCP protocol connection was aborted
- Could be client-side timeout, server crash, or cleanup failure
- Check server logs for the real error

---

## Files Changed

- `agents/blender-librarian/server.py`
  - Added `logging` import
  - Added `asynccontextmanager` import
  - Added `_lifespan()` async context manager
  - Changed `FastMCP("blender-librarian")` to `FastMCP("blender-librarian", lifespan=_lifespan)`

---

## References

- [FastMCP Lifespan Documentation](https://github.com/jlowin/fastmcp)
- [anyio Cancel Scope Documentation](https://anyio.readthedocs.io/en/stable/cancellation.html)
- [MCP SDK stdio_client Implementation](https://github.com/modelcontextprotocol/python-sdk)

---

**Document Author:** Claude Code
**Reviewed By:** Ben
**Status:** Fix Verified Working
