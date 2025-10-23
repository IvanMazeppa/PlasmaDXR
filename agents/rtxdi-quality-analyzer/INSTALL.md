# RTXDI Quality Analyzer - Installation Guide

## ✅ SDK Integration Complete

The agent now has full MCP server integration using the Claude Agent SDK. You can add it to Claude Code using the `claude mcp add` command.

---

## Quick Installation

### Step 1: Test the Agent (Optional but Recommended)

First, verify the agent works:

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate
python -m src.agent
```

You should see the MCP server start (it will wait for input on stdin). Press `Ctrl+C` to stop it. If there are no errors, you're ready to add it to Claude Code!

---

### Step 2: Add to Claude Code

**IMPORTANT:** The latest `claude mcp add` syntax requires:
1. `--transport stdio` flag
2. `--` (double dash) to separate Claude's flags from the server command

Choose **ONE** of the following methods:

#### **Option 1: Using Wrapper Script (EASIEST - Recommended)**

```bash
claude mcp add --transport stdio rtxdi-quality-analyzer \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  --env PIX_PATH="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64" \
  -- /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/run_server.sh
```

The wrapper script handles directory changes and venv activation automatically.

---

#### **Option 2: Using bash -c (Alternative)**

```bash
claude mcp add --transport stdio rtxdi-quality-analyzer \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  --env PIX_PATH="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64" \
  -- bash -c "cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer && source venv/bin/activate && python -m src.agent"
```

---

#### **Option 3: From Agent Directory (If already there)**

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer

claude mcp add --transport stdio rtxdi-quality-analyzer \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  --env PIX_PATH="/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64" \
  -- ./venv/bin/python -m src.agent
```

---

### Step 3: Verify Installation

Check that the server was added:

```bash
claude mcp list
```

You should see `rtxdi-quality-analyzer` in the list.

---

### Step 4: Test from Claude Code

Open a new Claude Code session and try:

```
Please use the compare_performance tool to analyze my RTXDI performance
```

or

```
Analyze the latest PIX capture for bottlenecks
```

Claude Code should automatically route these requests to your agent! 🎉

---

## Troubleshooting

### "Command not found: claude"

Make sure Claude Code CLI is installed:
```bash
npm install -g @anthropic-ai/claude-code
```

### "Server failed to start"

Check the logs:
```bash
claude mcp logs rtxdi-quality-analyzer
```

Common issues:
- Python path incorrect (use full path to venv Python)
- Missing dependencies (activate venv and run `pip install -r requirements.txt`)
- Wrong working directory (must be the agent's root directory)

### "Tool not found"

The agent is running but Claude Code can't see the tools. This usually means:
- The server didn't start correctly (check logs)
- The MCP protocol handshake failed (restart Claude Code)

### Testing Manually

You can test the server directly:

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer
source venv/bin/activate
python -m src.agent
```

Then send a JSON-RPC request on stdin:
```json
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

You should see a list of available tools.

---

## Removing the Server

If you need to remove or reinstall:

```bash
claude mcp remove rtxdi-quality-analyzer
```

Then add it again with the updated command.

---

## What's Registered

When you add the MCP server, Claude Code can now use two tools:

### 1. `compare_performance`
Compares FPS, frame times, and GPU metrics between legacy renderer, RTXDI M4, and RTXDI M5.

**Example usage in Claude Code:**
```
Compare performance between all my renderer modes
```

### 2. `analyze_pix_capture`
Analyzes PIX GPU captures to identify RTXDI bottlenecks.

**Example usage in Claude Code:**
```
Analyze the latest PIX capture for performance issues
```

---

## Next Steps

Now that your agent is registered:

1. **Try the tools** from a Claude Code session
2. **Implement real parsing** in `src/utils/metrics_parser.py` and `src/utils/pix_parser.py`
3. **Add more tools** as needed (screenshot capture, image comparison, etc.)

The stub implementations will return sample data for now. Replace them with real parsing logic to get actual results!

---

## Authentication Note

**You don't need an API key!** The agent runs within Claude Code's context and automatically inherits your Claude Max authentication. The `.env` file only needs project paths:

```bash
PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
PIX_PATH=/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64
```

No `ANTHROPIC_API_KEY` required! 🎉
