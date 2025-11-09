# DXR Shadow Engineer - Troubleshooting Guide

## Fixed: Virtual Environment Path Issue

**Error**: `-bash: /venv/bin/activate: No such file or directory`

**Problem**: You used an absolute path `/venv/` instead of relative path `venv/`

**Solution**:
```bash
# WRONG (looks in root directory /venv/)
source /venv/bin/activate

# CORRECT (looks in current directory ./venv/)
source venv/bin/activate

# Or use full path
source /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-shadow-engineer/venv/bin/activate
```

## Installation Steps (Correct Order)

```bash
# 1. Navigate to project directory
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-shadow-engineer

# 2. Activate virtual environment (relative path!)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python3 -c "from mcp.server import Server; print('MCP installed successfully')"

# 5. Test server locally
python3 dxr_shadow_server.py
# Should start without errors (waits for stdin from Claude Code)
```

## MCP vs Claude Agent SDK - Why We Use MCP

### You Asked: "Why not use claude-agent-sdk?"

**Answer**: For your use case (MCP servers within Claude Code), the **mcp package is correct**. Here's why:

| Feature | `mcp` package | `claude-agent-sdk` |
|---------|---------------|---------------------|
| **Purpose** | MCP servers (passive tools) | Standalone agents (active) |
| **Transport** | stdio (stdin/stdout) | API calls |
| **API Key** | ❌ Not needed | ✅ Required |
| **Use Case** | Tools for Claude Code | Independent agents |
| **Your Setup** | ✅ Perfect fit | ❌ Wrong architecture |

### Key Differences

**MCP Package (`mcp==1.21.0`)**:
- ✅ Works with Claude Max subscription (no API key!)
- ✅ Passive tool server (waits for Claude Code to call it)
- ✅ Communicates via stdio (standard input/output)
- ✅ Matches your existing servers (rtxdi-quality-analyzer, pix-debug)
- ✅ Simple architecture for tool-based servers

**Claude Agent SDK (`claude-agent-sdk`)**:
- ❌ Requires API key (separate from Max subscription)
- ❌ Standalone agent with own query loop
- ❌ Makes direct API calls to Anthropic
- ❌ Wrong architecture for MCP servers
- ❌ Overkill for your use case

### When to Use Each

Use **mcp package** when:
- Building tools for Claude Code
- You have Claude Max subscription
- You want passive tools (Claude calls you)
- Example: Shadow research, PCSS analysis, shader generation

Use **claude-agent-sdk** when:
- Building standalone agents
- Agent needs to maintain conversation state
- Agent runs independently from Claude Code
- Example: Autonomous coding assistant, chatbot

### Your rtxdi-quality-analyzer Uses MCP

Looking at your existing server:
```python
# rtxdi-quality-analyzer/rtxdi_server.py
from mcp.server import Server  # ← Uses mcp package, not claude-agent-sdk
server = Server("rtxdi-quality-analyzer")
```

The dxr-shadow-engineer follows the same pattern - proven and working!

## ⚠️ CRITICAL FIX: Windows Line Endings

**Error**: `$'\r': command not found` in MCP logs

**Problem**: The `run_server.sh` script has Windows-style line endings (CRLF) instead of Unix-style (LF). This is a common issue when creating files on Windows filesystems in WSL.

**Solution**:
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-shadow-engineer

# Fix line endings (use either dos2unix or sed)
dos2unix run_server.sh
# OR
sed -i 's/\r$//' run_server.sh

# Verify fix (should show "ASCII text executable", not "CRLF")
file run_server.sh

# Make executable
chmod +x run_server.sh

# Test server (should start without errors)
timeout 2 ./run_server.sh
# Expected: Times out after 2 seconds (server waiting for stdin)
```

**This fix has been applied to your server** ✅

## Connection Issues

If the server won't connect to Claude Code, check these:

### 1. Verify MCP Config File

Your MCP config should be at one of these locations:
- `~/.config/claude-code/mcp.json`
- Or wherever Claude Code stores settings on your system

Check the config:
```json
{
  "mcpServers": {
    "dxr-shadow-engineer": {
      "command": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-shadow-engineer/run_server.sh",
      "env": {
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
      }
    }
  }
}
```

### 2. Test Server Standalone

```bash
# Navigate to directory
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-shadow-engineer

# Activate venv
source venv/bin/activate

# Run server (should start without errors)
python3 dxr_shadow_server.py

# You should see no output (server waits for stdin)
# Press Ctrl+C to exit

# If you see errors, check:
# - Missing dependencies? Run: pip install -r requirements.txt
# - Import errors? Run: python3 -c "from mcp.server import Server"
```

### 3. Check Launch Script Permissions

```bash
# Make script executable
chmod +x run_server.sh

# Test it
./run_server.sh
# Should start server without errors
```

### 4. Verify Dependencies Installed

```bash
# Activate venv
source venv/bin/activate

# Check mcp package
pip show mcp

# Expected output:
# Name: mcp
# Version: 1.21.0
# ...

# Test import
python3 -c "from mcp.server import Server; print('✅ MCP works')"
```

### 5. Claude Code Restart

After adding to MCP config:
1. Restart Claude Code completely
2. Check Claude Code logs for errors
3. Try using a tool: "Research shadow techniques for volumetric particles"

## Common Errors

### "ModuleNotFoundError: No module named 'mcp'"

**Solution**: Dependencies not installed or venv not activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied: ./run_server.sh"

**Solution**: Script not executable
```bash
chmod +x run_server.sh
```

### "Server not found in Claude Code"

**Solution**: MCP config not loaded
1. Check MCP config file path
2. Verify JSON syntax (use jsonlint.com)
3. Restart Claude Code
4. Check Claude Code logs

### "Connection refused" or "Server timeout"

**Solution**: Server not starting properly
```bash
# Test server manually
python3 dxr_shadow_server.py

# Check for errors
# If errors appear, dependencies might be missing
```

## Quick Test Commands

Once everything is set up, test with these queries in Claude Code:

1. **Research Test**:
   ```
   Research the best shadow technique for volumetric particles with DXR 1.1
   ```

2. **PCSS Analysis Test**:
   ```
   Analyze my current PCSS implementation
   ```

3. **Shader Generation Test**:
   ```
   Generate raytraced shadow shader code for the Balanced preset
   ```

4. **Performance Test**:
   ```
   Analyze shadow performance for 10K particles with 13 lights
   ```

5. **Comparison Test**:
   ```
   Compare PCSS vs raytraced inline shadows
   ```

## Still Having Issues?

If the server still won't connect:

1. **Check Claude Code version**: Ensure you have latest version with MCP support
2. **Check Python version**: `python3 --version` (should be 3.10+)
3. **Check permissions**: Ensure run_server.sh is executable
4. **Check logs**: Look for Claude Code logs (usually in ~/.local/share/claude-code/)
5. **Try rtxdi-quality-analyzer**: If that works but dxr-shadow-engineer doesn't, compare configs

## Manual Installation (If pip fails)

If pip install is hanging or failing:

```bash
# Install packages one by one
pip install mcp==1.21.0
pip install python-dotenv==1.0.0
pip install rich==13.7.0

# Skip optional packages if needed
# (requests, beautifulsoup4, pandas, numpy are optional for basic functionality)
```

The server will work with just `mcp` and `python-dotenv` - other packages are for future web research features.

---

**Last Updated**: 2025-11-09
**Status**: All fixes applied, ready to test
