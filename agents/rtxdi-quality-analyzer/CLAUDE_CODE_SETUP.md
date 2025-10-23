# Claude Code MCP Server Setup

This guide explains how to register the RTXDI Quality Analyzer as an MCP server in Claude Code.

## Authentication

**You do NOT need an API key!**

Since you're using Claude Code with your Claude Max subscription, authentication is handled automatically. The agent runs within Claude Code's context and inherits your authentication.

---

## Setup Steps

### 1. Create `.env` File (No API Key Needed)

```bash
cd agents/rtxdi-quality-analyzer
cp .env.example .env
```

Edit `.env` and set project paths:
```bash
PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
PIX_PATH=/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64
```

**Note:** Leave `ANTHROPIC_API_KEY` commented out - you don't need it!

---

### 2. Register MCP Server in Claude Code

Claude Code uses a configuration file to register MCP servers. The location depends on your OS:

**Linux/WSL2:**
```
~/.claude/mcp_servers.json
```

**Windows:**
```
%APPDATA%\.claude\mcp_servers.json
```

**macOS:**
```
~/Library/Application Support/Claude/mcp_servers.json
```

---

### 3. Add Your Agent to MCP Config

Edit `mcp_servers.json` and add your agent:

```json
{
  "mcpServers": {
    "rtxdi-quality-analyzer": {
      "command": "python",
      "args": [
        "-m",
        "src.agent"
      ],
      "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer",
      "env": {
        "PYTHONPATH": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer",
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean",
        "PIX_PATH": "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
      }
    }
  }
}
```

**Important:** Make sure to use the virtual environment's Python if needed:
```json
{
  "mcpServers": {
    "rtxdi-quality-analyzer": {
      "command": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer/venv/bin/python",
      "args": ["-m", "src.agent"],
      "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer",
      "env": {
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean",
        "PIX_PATH": "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
      }
    }
  }
}
```

---

### 4. Restart Claude Code

After updating the MCP config, restart Claude Code to load the new server.

---

### 5. Test the Agent

In your Claude Code session, you should now be able to use the agent's tools:

```
Please analyze the latest PIX capture for RTXDI bottlenecks
```

or

```
Compare performance between legacy renderer, RTXDI M4, and RTXDI M5
```

Claude Code will automatically route these requests to your agent's tools!

---

## How It Works

```
┌─────────────────┐
│   Claude Code   │  (Your Claude Max subscription)
│   (authenticated)│
└────────┬────────┘
         │ Starts and connects to...
         ▼
┌─────────────────────────────────┐
│  RTXDI Quality Analyzer Agent   │  (No API key needed!)
│  (MCP Server)                   │
│                                 │
│  Tools:                         │
│  - compare_performance          │
│  - analyze_pix_capture          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PlasmaDX-Clean Project         │
│  - Parse logs/                  │
│  - Analyze PIX/Captures/        │
│  - Generate reports             │
└─────────────────────────────────┘
```

---

## Troubleshooting

### "Agent not found"
- Check that the path in `mcp_servers.json` is correct
- Verify the virtual environment Python path
- Restart Claude Code after config changes

### "Import errors"
- Ensure virtual environment is activated
- Check `PYTHONPATH` is set correctly in MCP config
- Verify all dependencies are installed: `pip list | grep claude-agent-sdk`

### "Permission denied"
- On WSL2, ensure paths use `/mnt/c/...` format (not `C:\...`)
- Check file permissions: `chmod +x venv/bin/python`

---

## Testing Without Claude Code

You can test the agent standalone before registering it:

```bash
# Activate virtual environment
source venv/bin/activate

# Test CLI mode (no Claude needed)
python -m src.cli interactive

# Test agent startup (will show initialization)
python -m src.agent
```

---

## Next Steps After Registration

Once your agent is registered with Claude Code:

1. **SDK Integration** - Implement the Agent SDK tool decorators (see verifier report)
2. **Real Parsing** - Replace stub implementations with actual log/PIX parsing
3. **Test Tools** - Use the agent from Claude Code sessions to validate functionality

The agent will automatically inherit your Claude Code authentication - no API key required!
