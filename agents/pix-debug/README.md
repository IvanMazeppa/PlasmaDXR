# PIX Debugging Agent v4

**Status**: ✅ Production Ready  
**SDK Version**: 0.1.6  
**Last Updated**: 2025-11-01

A Model Context Protocol (MCP) server providing comprehensive DirectX 12 / DXR debugging tools for PlasmaDX-Clean volumetric particle renderer. Designed for autonomous debugging of Volumetric ReSTIR shader execution issues, GPU hangs, and rendering artifacts.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Setup](#setup)
- [Usage Examples](#usage-examples)
- [Known Issues](#known-issues)
- [Architecture](#architecture)
- [Development](#development)
- [Changelog](#changelog)

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested with 3.12.3)
- WSL2 or Linux environment
- PlasmaDX-Clean project at `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean`
- PIX for Windows at `/mnt/c/Program Files/Microsoft PIX/2509.25/`
- Claude Code with MCP support

### Installation

```bash
# Clone or navigate to agent directory
cd /mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your paths (usually defaults work)

# Test the server
python mcp_server.py
```

### Connecting to Claude Code

Add to your Claude Code MCP settings (`.claude/mcp_settings.json`):

```json
{
  "mcpServers": {
    "pix-debug": {
      "command": "python",
      "args": ["/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/mcp_server.py"],
      "cwd": "/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4",
      "env": {}
    }
  }
}
```

Then in Claude Code:

```
/mcp reconnect pix-debug
```

---

## Available Tools

See [TOOLS.md](TOOLS.md) for complete documentation of all 9 diagnostic tools.

**Quick reference**:

| Tool | Purpose | Status |
|------|---------|--------|
| `diagnose_gpu_hang` | Find crash thresholds | ✅ Working |
| `analyze_dxil_root_signature` | Detect root signature mismatches | ✅ Working |
| `validate_shader_execution` | Confirm shaders execute | ✅ Working |
| `capture_buffers` | Dump GPU buffers | ⚠️ Manual trigger |
| `analyze_particle_buffers` | Validate particle data | ✅ Working |
| `analyze_restir_reservoirs` | Parse ReSTIR data | ⚠️ Outdated format |
| `pix_capture` | Create .wpix captures | ⚠️ Manual launch |
| `pix_list_captures` | List PIX captures | ✅ Working |
| `diagnose_visual_artifact` | Automated artifact diagnosis | ✅ Working |

---

## Setup

### Environment Variables

Edit `.env` file:

```env
# PIX Tool Path
PIXTOOL_PATH="/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe"

# PlasmaDX Project Path
PLASMA_DX_PATH="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"

# Buffer Dump Directory (relative to PLASMA_DX_PATH)
BUFFER_DUMP_DIR="${PLASMA_DX_PATH}/PIX/buffer_dumps"

# PIX Captures Directory (relative to PLASMA_DX_PATH)
PIX_CAPTURES_DIR="${PLASMA_DX_PATH}/PIX/Captures"
```

### Dependencies

```txt
# requirements.txt
anthropic-sdk==0.1.6
python-dotenv==1.0.0
numpy==1.24.0
pyautogui==0.9.54  # For keyboard automation (future)
```

### Directory Structure

```
pix-debugging-agent-v4/
├── mcp_server.py          # Main MCP server
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your configuration (git-ignored)
├── README.md             # This file
├── TOOLS.md              # Complete tool documentation
├── CHANGELOG.md          # Version history
└── venv/                 # Python virtual environment
```

---

## Usage Examples

### Example 1: Debug Shader Execution Failure

**Scenario**: PopulateVolumeMip2 diagnostic counters all zero.

```
User: "PopulateVolumeMip2 diagnostic counters are all zero"

Agent workflow:
1. validate_shader_execution → Confirms NOT_EXECUTING
2. analyze_dxil_root_signature → Checks resource bindings
3. Provides specific fix
```

### Example 2: Find GPU Hang Threshold

**Scenario**: Crash at unknown particle count.

```
User: "Find the particle count that causes PopulateVolumeMip2 to hang"

Agent: diagnose_gpu_hang with test_threshold=true
Result: "2044 works, 2045 crashes"
```

### Example 3: Root Signature Mismatch

**Scenario**: Shader resources optimized away.

```
User: "GenerateCandidates shader isn't executing"

Agent workflow:
1. analyze_dxil_root_signature
2. Finds missing t0/t1/t2 resources  
3. Reads shader source
4. Discovers Phase 1 stub code
5. Recommends C++ root signature fix
```

---

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for complete list.

### Critical Issues

1. **PopulateVolumeMip2 Not Executing** (🔄 Active)
   - Root signature correct but shader doesn't run
   - Diagnostic counters all zero
   - Needs PIX capture analysis

2. **analyze_restir_reservoirs Outdated** (⚠️ Update needed)
   - Designed for 32-byte format (deprecated)
   - Current: 64-byte Volumetric ReSTIR format

3. **Duplicate Tool Registrations** (⚠️ Cleanup needed)
   - `analyze_dxil_root_signature` registered twice
   - No functional impact

---

## Architecture

### Hybrid Debugging Approach

**Fast Path (Buffer Dumps)**:
- 2-second captures
- Direct GPU→CPU readback
- Statistical analysis

**Deep Inspection (PIX Captures)**:
- 5-10 minute captures
- Full GPU state timeline
- Shader debugging

### Integration with PlasmaDX

**Command-Line Flags**:
- `--restir`: Enable Volumetric ReSTIR
- `--particles N`: Set particle count
- `--dump-buffers N`: Dump at frame N (future)

**Log Format**:

```
[22:36:31] [INFO] ========== PopulateVolumeMip2 Diagnostic Counters ==========
[22:36:31] [INFO]   [0] Total threads executed: 0
[22:36:31] [INFO]   [1] Early returns: 0
[22:36:31] [INFO]   [2] Total voxel writes: 0
```

---

## Development

### Adding New Tools

1. Register in `list_tools()`
2. Implement handler in `call_tool()`
3. Create async function
4. Document in TOOLS.md

### Testing

```bash
# Reconnect MCP
/mcp reconnect pix-debug

# Test tool
"Use diagnose_gpu_hang to test at 100 particles"
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

**Latest (v0.1.6 - 2025-11-01)**:
- Added `analyze_dxil_root_signature` tool
- Added `validate_shader_execution` tool
- Fixed GenerateCandidates root signature mismatch

---

## Resources

- [Claude Agent SDK Docs](https://docs.anthropic.com/en/api/agent-sdk)
- [MCP Protocol Spec](https://modelcontextprotocol.io)
- [PIX Documentation](https://devblogs.microsoft.com/pix/)
- [TOOLS.md](TOOLS.md) - Complete tool reference
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) - Bug tracker

---

## License

MIT

---

**Maintained by**: Claude Code automated debugging sessions  
**Primary Use Case**: Volumetric ReSTIR shader execution debugging  
**Status**: Production-ready for DXR debugging workflows
