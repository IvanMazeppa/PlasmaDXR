# MCP Server Connection Debugging Summary

**Date:** 2025-10-23
**Status:** ❌ NOT RESOLVED - Connection still failing
**Current SDK:** 0.1.3 (also tested 0.1.1, 0.1.4)

---

## Timeline of Events

### ✅ Original Working State
- **Date:** Earlier session
- **Tools:** 2 tools (compare_performance, analyze_pix_capture)
- **SDK Version:** 0.1.4
- **Status:** WORKING - Server connected successfully
- **Launcher:** `run_server.sh` used `python -m src.agent`

### ❌ What We Changed (When it Broke)
1. Added ML comparison tool (3rd tool with LPIPS)
2. Changed `run_server.sh` from `python -m src.agent` to `python "$SCRIPT_DIR/src/agent.py"` (to match pix-debug)
3. This caused `ImportError: attempted relative import with no known parent package`

### 🔧 Attempted Fixes (None Worked)

**Fix #1: Module-level server registration**
- Moved `Server()` and decorators from inside `main()` to module level
- Matched pix-debug structure exactly
- **Result:** Import error persisted

**Fix #2: Absolute imports instead of relative**
- Changed from `from .tools.X import Y` to `from src.tools.X import Y`
- Added `sys.path.insert(0, str(SCRIPT_DIR.parent))`
- **Result:** Import errors gone, BUT server times out on connection (30 seconds)

**Fix #3: SDK downgrade**
- Tried SDK 0.1.1 (matches working pix-debug)
- Tried SDK 0.1.3 (latest stable before 0.1.4)
- **Result:** Still times out

**Fix #4: Revert to 2-tool version**
- Removed ML tool completely
- Back to original 2-tool config
- **Result:** STILL doesn't work! 🚩 **KEY INSIGHT**

---

## Critical Observation (User's Insight)

**If the original 2-tool version worked with SDK 0.1.4, reverting to that exact code should work.**

**Since reverting doesn't work, we didn't truly revert everything. Something else changed.**

---

## Error States Observed

### State 1: Import Error (Fixed)
```
ImportError: attempted relative import with no known parent package
```
- **Cause:** Changed from `python -m src.agent` to direct script execution
- **Fixed by:** Absolute imports + sys.path manipulation
- **Log location:** `/home/maz3ppa/.cache/claude-cli-nodejs/-mnt-d-Users-dilli-AndroidStudioProjects-PlasmaDX-Clean/mcp-logs-rtxdi-quality-analyzer/2025-10-23T16-20-55-008Z.txt`

### State 2: Connection Timeout (Current Issue)
```
Connection timeout triggered after 27395ms (limit: 30000ms)
Connection to MCP server "rtxdi-quality-analyzer" timed out after 30000ms
```
- **Cause:** Unknown - server starts but doesn't respond to MCP initialize handshake
- **Not fixed by:** SDK downgrade, reverting to 2 tools, absolute imports
- **Log location:** `/home/maz3ppa/.cache/claude-cli-nodejs/-mnt-d-Users-dilli-AndroidStudioProjects-PlasmaDX-Clean/mcp-logs-rtxdi-quality-analyzer/2025-10-23T19-02-25-610Z.txt`

---

## Current Server Configuration

**File:** `run_server.sh`
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/src/agent.py"
```

**File:** `src/agent.py` (current state)
- Uses absolute imports: `from src.tools.X import Y`
- Has `sys.path.insert(0, str(SCRIPT_DIR.parent))` at top
- Server and decorators at module level (not in main())
- Currently has 2 tools (ML tool removed during debugging)

**MCP Config:** `/home/maz3ppa/.claude.json`
```json
{
  "type": "stdio",
  "command": "bash",
  "args": [
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/run_server.sh"
  ],
  "env": {
    "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean",
    "PIX_PATH": "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
  }
}
```

---

## Working Reference: pix-debug Server

**SDK Version:** 0.1.1
**Launcher:** `run_mcp_server.sh`
```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/mcp_server.py"
```

**Key Differences from rtxdi-quality-analyzer:**
1. Uses older SDK (0.1.1 vs our 0.1.3/0.1.4)
2. Has NO `cd` command (we tried with/without)
3. Server file is `mcp_server.py` (single file, not src/agent.py)
4. Uses direct imports (no src/ subdirectory)

**MCP Config for pix-debug:**
```json
{
  "type": "stdio",
  "command": "bash",
  "args": [
    "/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/run_mcp_server.sh"
  ],
  "env": {}
}
```

---

## Potential Root Causes (To Investigate)

### Theory 1: Module Structure Difference
- pix-debug: Flat structure (`mcp_server.py` at root)
- rtxdi-quality-analyzer: Package structure (`src/agent.py` with `src/tools/`)
- **Test:** Flatten structure, move everything to root-level single file

### Theory 2: Import Path Contamination
- `sys.path.insert(0, ...)` might be interfering with MCP internals
- **Test:** Remove sys.path manipulation, use PYTHONPATH instead

### Theory 3: Async/Await Issue in Tools
- Original 2 tools had stub async functions
- ML tool uses actual async operations (LPIPS model loading)
- **Test:** Check if any tool hangs during initialization

### Theory 4: run_server.sh Changed but Not Reverted
- We changed from `python -m src.agent` to direct execution
- Did we ever successfully test reverting back to `python -m src.agent`?
- **Test:** Try original module execution syntax

### Theory 5: Virtual Environment Corruption
- Multiple SDK version changes may have corrupted venv
- **Test:** Delete and recreate venv from scratch

### Theory 6: Hidden File Changes
- `.env`, `__pycache__`, `.pyc` files might have stale data
- **Test:** Clean all cached files, restart fresh

---

## Diagnostic Commands

### Check Server Starts Without Errors
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate
timeout 3 python src/agent.py 2>&1
# Should show no errors, just timeout
```

### Check Imports Work
```bash
source venv/bin/activate
python -c "from src.agent import server, list_tools; import asyncio; asyncio.run(list_tools())"
# Should list 2 or 3 tools
```

### Test MCP Handshake Manually
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | timeout 5 ./run_server.sh 2>&1
# Should see JSON response (currently: no response = timeout)
```

### Compare Package Versions
```bash
# Working pix-debug
cd /mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4
source venv/bin/activate && pip list | grep -E "(mcp|claude-agent-sdk)"

# Broken rtxdi-quality-analyzer
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate && pip list | grep -E "(mcp|claude-agent-sdk)"
```

### View Latest Error Logs
```bash
cat /home/maz3ppa/.cache/claude-cli-nodejs/-mnt-d-Users-dilli-AndroidStudioProjects-PlasmaDX-Clean/mcp-logs-rtxdi-quality-analyzer/*.txt | tail -100
```

---

## Files to Check in Next Session

**Essential Files:**
1. `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/src/agent.py`
2. `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/run_server.sh`
3. `/home/maz3ppa/.claude.json` (search for "rtxdi-quality-analyzer")

**Reference (Working):**
1. `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/mcp_server.py`
2. `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/run_mcp_server.sh`

**Backup Files (Versions Created During Debugging):**
1. `src/agent_old.py` - Original with relative imports
2. `src/agent_3tools.py` - Version with ML tool
3. `src/agent_test2tools.py` - Test 2-tool version

---

## Recommended Next Steps

### Option A: Nuclear Option - Start Fresh
1. Delete entire rtxdi-quality-analyzer directory
2. Copy working pix-debug structure
3. Add tools one by one
4. Test connection after each change

### Option B: Systematic Binary Search
1. Create EXACT copy of pix-debug server (rename it, test it connects)
2. Change ONE thing at a time toward rtxdi-quality-analyzer config
3. Identify exact change that breaks connection

### Option C: Module Execution (Revert to Original)
1. Change `run_server.sh` back to `python -m src.agent`
2. Use relative imports (original structure)
3. Test if original execution method works

### Option D: Flatten Structure
1. Merge all code into single `mcp_server.py` file (like pix-debug)
2. Remove src/ package structure
3. Use direct imports, no package

---

## Questions to Answer

1. **What was the EXACT `run_server.sh` content when it originally worked?**
   - Did it use `python -m src.agent` or `python src/agent.py`?

2. **Does pix-debug still work?**
   - If we modify pix-debug to use package structure, does IT break?

3. **Is there a backup of the working version?**
   - Git history? Other sessions?

4. **Can we create a minimal reproducer?**
   - Smallest possible MCP server that exhibits same timeout

5. **What does Claude Code expect from stdio servers?**
   - Does it require immediate response to initialize?
   - Is there a protocol version issue?

---

## Technical Details

**Python Version:** 3.12.3
**MCP Protocol Version:** 2024-11-05
**Claude Code Version:** Unknown (check with `claude --version`)

**Package Versions (Current):**
- claude-agent-sdk: 0.1.3
- mcp: 1.18.0
- anyio: 4.11.0
- httpx: 0.28.1

**Package Versions (Working pix-debug):**
- claude-agent-sdk: 0.1.1
- mcp: 1.18.0
- anyio: 4.11.0
- httpx: 0.28.1

---

## Success Criteria

Server is FIXED when:
- ✅ `claude mcp list` shows rtxdi-quality-analyzer as "connected"
- ✅ No timeout errors in logs
- ✅ Can invoke tools from Claude Code session
- ✅ Both 2-tool and 3-tool (with ML) versions work

---

**Last Updated:** 2025-10-23 20:10 UTC
**Context Remaining:** 5%
**Status:** UNRESOLVED - Connection timeout persists across all attempted fixes

**Key Insight:** If reverting to original 2-tool code doesn't work, we didn't truly revert. Something external changed (venv state? cached files? MCP config?).
