# MCP Server Debug Information

**Status:** Connection timeout (30s) - Server starts but doesn't respond to MCP protocol

---

## Questions for Opus 4.1 / SDK Docs Research

1. **Server initialization pattern:**
   - Should `Server()` be created at module level or inside `main()`?
   - Should decorators be at module level or inside `main()`?
   - Does SDK 0.1.4 have different requirements than 0.1.1?

2. **stdio_server context manager:**
   - Is there a specific initialization sequence required?
   - Are there any async context manager issues in SDK 0.1.4?

3. **Known issues:**
   - Are there known bugs in SDK 0.1.4 stdio transport?
   - Are there compatibility issues with Python 3.12.3?

---

## What Works

✅ Python imports (all tools load successfully)
✅ Server starts without errors
✅ Virtual environment setup correct
✅ All dependencies installed (including LPIPS weights ~528MB)
✅ MCP config syntax correct (matches working pix-debug server)

---

## What Doesn't Work

❌ MCP protocol handshake times out after 27-30 seconds
❌ No JSON-RPC response to initialize message
❌ Connection fails in Claude Code

---

## Tested Configurations

### Config 1: Module-level server (SDK 0.1.3)
```python
server = Server("rtxdi-quality-analyzer")

@server.list_tools()
async def list_tools():
    ...

async def main():
    async with stdio_server() as (read, write):
        await server.run(...)
```
**Result:** Timeout

### Config 2: Function-level server (SDK 0.1.1)
```python
async def main():
    server = Server("rtxdi-quality-analyzer")

    @server.list_tools()
    async def list_tools():
        ...

    async with stdio_server() as (read, write):
        await server.run(...)
```
**Result:** Timeout

### Config 3: GitHub backup structure (SDK 0.1.4)
- Class-based analyzer
- Function-level server inside main()
- Decorators inside main()
**Result:** Timeout

---

## Working Reference: pix-debug

**Location:** `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/`

**SDK:** 0.1.1
**Structure:** Flat file (single `mcp_server.py`), module-level server
**Status:** WORKING ✅

**Key differences:**
1. Single file vs package structure
2. No class wrapper (direct functions)
3. SDK 0.1.1 (we're on 0.1.4)

---

## Files to Check

**Current state:**
- `src/agent.py` - Main server (317 lines, function-level server)
- `src/tools/ml_visual_comparison.py` - LPIPS ML tool (working locally)
- `src/tools/performance_comparison.py` - Performance analyzer
- `src/tools/pix_analysis.py` - PIX capture analyzer
- `run_server.sh` - Uses `python -m src.agent`

**Backups:**
- `src/agent_old.py` - Module-level server version
- `src/agent_3tools.py` - Earlier test version
- `test_minimal_mcp.py` - Minimal test server

---

## MCP Config

**Location:** `/home/maz3ppa/.claude.json`

```json
{
  "rtxdi-quality-analyzer": {
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
}
```

---

## Next Steps

### Option A: Research with Opus 4.1
1. Fetch latest Claude Agent SDK 0.1.4 documentation
2. Check for known issues or required patterns
3. Verify correct server initialization sequence
4. Look for Python 3.12 compatibility issues

### Option B: Copy pix-debug exactly
1. Flatten structure (single file, no package)
2. Downgrade to SDK 0.1.1
3. Remove class wrapper
4. Test if basic connection works

### Option C: Incremental debugging
1. Test minimal server with 0 tools
2. Add tools one by one
3. Identify which change breaks connection

### Option D: Check for environment issues
1. Test in fresh virtual environment
2. Clear all `__pycache__` directories
3. Remove and recreate MCP config from scratch
4. Check for port conflicts or stdio issues

---

## Diagnostic Commands

```bash
# Test server starts
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate
timeout 3 python -m src.agent

# Test imports
python -c "from src.agent import main; print('OK')"

# Check SDK version
pip list | grep claude-agent-sdk

# View latest logs
cat /home/maz3ppa/.cache/claude-cli-nodejs/-mnt-d-Users-dilli-AndroidStudioProjects-PlasmaDX-Clean/mcp-logs-rtxdi-quality-analyzer/*.txt | tail -50

# Compare with working pix-debug
cd /mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4
source venv/bin/activate
pip list | grep claude-agent-sdk
```

---

**Last Updated:** 2025-10-23 21:30 UTC
