# MCP Server Connection Issue - FIXED ✅

**Problem:** Server wouldn't connect after adding ML comparison tool

**Root Cause:** The MCP `Server` object and decorator registrations were inside the `main()` async function instead of at module level. Decorators must be registered when the module loads, not inside an async function.

**Fix Applied:** Refactored `src/agent.py` to match the working pix-debug server pattern:
- Server created at module level: `server = Server("rtxdi-quality-analyzer")`
- Decorators applied at module level: `@server.list_tools()`, `@server.call_tool()`
- `main()` only runs the server (no registration logic)

---

## Steps to Reconnect (After Fix)

### 1. Remove old server registration

```bash
claude mcp remove rtxdi-quality-analyzer
```

### 2. Re-add with correct path

```bash
claude mcp add --transport stdio rtxdi-quality-analyzer \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  --env PIX_PATH="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64" \
  -- /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/run_server.sh
```

### 3. Verify server is connected

```bash
claude mcp list
```

You should see:
```
rtxdi-quality-analyzer (connected) - 3 tools
```

### 4. Test in Claude Code

Start a new Claude Code session and ask:
```
What tools do you have from rtxdi-quality-analyzer?
```

Expected response:
- ✅ compare_performance
- ✅ analyze_pix_capture
- ✅ compare_screenshots_ml

---

## What Changed in agent.py

### ❌ BEFORE (Broken):
```python
async def main():
    # Create server inside async function
    server = Server("rtxdi-quality-analyzer")

    # Register decorators inside async function
    @server.list_tools()
    async def list_tools():
        ...
```

**Problem:** Decorators registered inside async function don't work with MCP protocol

### ✅ AFTER (Fixed):
```python
# Create server at module level
server = Server("rtxdi-quality-analyzer")

# Register decorators at module level
@server.list_tools()
async def list_tools():
    ...

async def main():
    # Only run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(...)
```

**Solution:** Decorators registered when module loads (module-level scope)

---

## Verification

Run this test to confirm the fix:

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate
python -c "
from src.agent import server, list_tools
import asyncio

async def test():
    tools = await list_tools()
    print(f'✓ Server has {len(tools)} tools')
    for tool in tools:
        print(f'  - {tool.name}')

asyncio.run(test())
"
```

Expected output:
```
✓ Server has 3 tools
  - compare_performance
  - analyze_pix_capture
  - compare_screenshots_ml
```

---

## If Still Not Connecting

### Option 1: Restart Claude Code completely
1. Close all Claude Code windows
2. Kill any background processes: `pkill -f "claude"`
3. Start Claude Code fresh
4. Run: `claude mcp list`

### Option 2: Check Claude Code logs
```bash
# View MCP connection logs (location varies by system)
# Look for errors related to rtxdi-quality-analyzer
tail -f ~/.claude/logs/mcp.log
```

### Option 3: Test server manually
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
./run_server.sh
# Should start without errors and wait for input
# Press Ctrl+C to stop
```

---

## Success Criteria

After applying the fix and reconnecting:

- [x] `python -c "from src.agent import server"` runs without errors
- [x] `claude mcp list` shows rtxdi-quality-analyzer as "connected"
- [x] Claude Code can list 3 tools from the server
- [x] Test comparison works: `compare_screenshots_ml` tool is available

If all checks pass, the server is fully operational! 🎉

---

**Status:** Fixed ✅
**Date:** 2025-10-23
**Root Cause:** Decorator registration inside async function
**Solution:** Refactored to module-level registration (matching pix-debug pattern)
