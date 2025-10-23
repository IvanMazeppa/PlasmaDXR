# Agent SDK Migration Roadmap

**Goal:** Migrate existing custom agents to Claude Agent SDK for better integration with Claude Code and consistent MCP server architecture.

**Current Status:**
- ✅ **rtxdi-quality-analyzer** - Migrated to Agent SDK (2025-10-23)
- ✅ **pix-debug** - Already MCP server (native implementation)
- ⏳ **Remaining agents** - To be evaluated for migration

---

## Existing Agents Inventory

### ✅ Production MCP Servers (Already Working)

**1. rtxdi-quality-analyzer**
- **Location:** `agents/rtxdi-quality-analyzer/`
- **Status:** Agent SDK migration complete
- **Tools:** 2 (compare_performance, analyze_pix_capture)
- **Focus:** High-level RTXDI performance analysis

**2. pix-debug**
- **Location:** `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/`
- **Status:** Native MCP implementation (working well)
- **Tools:** 6 (capture_buffers, analyze_restir_reservoirs, analyze_particle_buffers, pix_capture, pix_list_captures, diagnose_visual_artifact)
- **Focus:** Low-level GPU debugging and buffer analysis
- **Recommendation:** Keep as-is - working well, no need to migrate

---

## Candidate Agents for Agent SDK Migration

Based on PlasmaDX development needs, here are specialized agents that could be created:

### 🔄 Agent #1: shader-analyzer (High Priority)

**Purpose:** HLSL shader analysis, compilation checking, and optimization suggestions

**Tools:**
1. `analyze_shader_errors` - Parse DXC compilation errors with context
2. `check_shader_performance` - Identify expensive operations (branching, texture samples)
3. `validate_root_signature` - Check root signature vs shader binding compatibility
4. `suggest_optimizations` - Shader optimization recommendations

**Why Needed:**
- Shader compilation errors are cryptic
- Need context-aware error explanations
- Optimization opportunities hard to spot manually

**Effort:** 3-4 hours
**Impact:** High - Shaders are 50% of dev time

---

### 🔄 Agent #2: build-analyzer (Medium Priority)

**Purpose:** MSBuild analysis, dependency tracking, and build optimization

**Tools:**
1. `analyze_build_errors` - Parse MSBuild errors with file:line context
2. `check_build_performance` - Identify slow compilation units
3. `validate_dependencies` - Check for missing/outdated dependencies
4. `suggest_build_optimizations` - Incremental build improvements

**Why Needed:**
- MSBuild errors can be obscure
- Build times can slow iteration
- Dependency issues hard to diagnose

**Effort:** 2-3 hours
**Impact:** Medium - Quality of life improvement

---

### 🔄 Agent #3: config-manager (Low Priority)

**Purpose:** Configuration file management and validation

**Tools:**
1. `validate_config` - Check JSON config against schema
2. `compare_configs` - Diff between config files
3. `suggest_config` - Recommend config based on use case
4. `merge_configs` - Intelligent config merging

**Why Needed:**
- Many config files (scenarios, presets, builds)
- Easy to have typos or invalid values
- Hard to remember all options

**Effort:** 2 hours
**Impact:** Low - Nice to have

---

## Migration Template

For each agent to be migrated, follow this process:

### Step 1: Create Project Structure (10 minutes)

```bash
cd agents/
mkdir <agent-name>
cd <agent-name>

# Create structure
mkdir -p src/{tools,utils}
touch src/__init__.py
touch src/agent.py
touch src/cli.py
touch src/tools/__init__.py
touch src/utils/__init__.py

# Create configs
touch .env.example
touch .gitignore
touch README.md
touch INSTALL.md
touch requirements.txt
touch run_server.sh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies (5 minutes)

```bash
# requirements.txt
cat > requirements.txt << EOF
# Claude Agent SDK
claude-agent-sdk==0.1.4

# Environment management
python-dotenv==1.0.0

# CLI utilities
rich==13.7.0

# Type checking (development)
mypy==1.7.1
EOF

pip install -r requirements.txt
```

### Step 3: Create MCP Server (30 minutes)

**File:** `src/agent.py`

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

load_dotenv()

# Create MCP server
server = Server("<agent-name>")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="tool_name",
            description="Tool description",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": []
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "tool_name":
        # Implement tool logic
        result = await tool_implementation(arguments)

        return [TextContent(
            type="text",
            text=result
        )]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Create Wrapper Script (5 minutes)

**File:** `run_server.sh`

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate
exec python -m src.agent
```

```bash
chmod +x run_server.sh
```

### Step 5: Register with Claude Code (5 minutes)

```bash
claude mcp add --transport stdio <agent-name> \
  --env PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean \
  -- /full/path/to/agents/<agent-name>/run_server.sh
```

### Step 6: Test and Verify (10 minutes)

```bash
# Test server starts
./run_server.sh

# Verify in Claude Code
claude mcp list

# Test tool invocation
# In Claude Code session: "Use <tool-name> to..."
```

---

## Migration Priority

### **High Priority (Do Next):**

1. **shader-analyzer** - Immediate value, high impact
   - Estimated time: 3-4 hours
   - ROI: Very high (saves hours per day)

### **Medium Priority (Later):**

2. **build-analyzer** - Quality of life improvement
   - Estimated time: 2-3 hours
   - ROI: Medium (occasional use)

### **Low Priority (If Time Permits):**

3. **config-manager** - Nice to have
   - Estimated time: 2 hours
   - ROI: Low (manual config management works)

---

## Alternative: Extend Existing Agents

Instead of creating new agents, consider adding tools to existing ones:

### **Add to rtxdi-quality-analyzer:**
- `analyze_shader_compilation` - HLSL error parsing
- `validate_rtxdi_config` - Config file validation
- `compare_frame_captures` - Visual regression testing

### **Add to pix-debug:**
- `analyze_shader_profile` - Shader performance from PIX
- `validate_pipeline_state` - PSO validation
- `check_resource_barriers` - Resource state debugging

**Pros:**
- Less overhead (fewer servers to manage)
- Better tool discoverability
- Shared code and utilities

**Cons:**
- Agents become more complex
- Harder to maintain focused purpose

**Recommendation:** Start with extending existing agents, create new ones only if they get too large (>10 tools)

---

## Testing Checklist

For each migrated agent:

- [ ] Server starts without errors
- [ ] All tools listed in `claude mcp list`
- [ ] Tools callable from Claude Code
- [ ] Error handling works (try invalid inputs)
- [ ] Documentation complete (README, INSTALL)
- [ ] .env.example has all required vars
- [ ] Virtual environment dependencies correct
- [ ] Wrapper script executable and working
- [ ] Type hints throughout (mypy passes)
- [ ] No syntax errors (python -m py_compile)

---

## Common Patterns

### **Pattern 1: File Analysis Tools**
```python
async def analyze_file(file_path: str) -> str:
    """Analyze a file and return findings"""
    if not Path(file_path).exists():
        return f"Error: File not found: {file_path}"

    # Parse file
    # Analyze content
    # Format results

    return formatted_results
```

### **Pattern 2: Comparison Tools**
```python
async def compare_items(item_a: str, item_b: str) -> str:
    """Compare two items and show differences"""
    # Parse both items
    # Compute diff
    # Format side-by-side comparison

    return comparison_report
```

### **Pattern 3: Validation Tools**
```python
async def validate_config(config_path: str) -> str:
    """Validate configuration file"""
    # Load config
    # Check against schema
    # Report errors with line numbers

    return validation_report
```

---

## Resources

**Agent SDK Documentation:**
- Overview: https://docs.claude.com/en/api/agent-sdk/overview
- Python Reference: https://docs.claude.com/en/api/agent-sdk/python
- Custom Tools Guide: https://docs.claude.com/en/api/agent-sdk/custom-tools
- MCP Integration: https://docs.claude.com/en/api/agent-sdk/mcp

**Examples:**
- rtxdi-quality-analyzer: `agents/rtxdi-quality-analyzer/`
- pix-debug: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/`

**Tools:**
- Claude MCP CLI: `claude mcp --help`
- Agent SDK: `pip show claude-agent-sdk`
- MCP Protocol: https://modelcontextprotocol.io/

---

## Next Steps

1. **Review candidate agents** - Decide which to build first
2. **Start with shader-analyzer** - Highest ROI
3. **Follow migration template** - Use proven process
4. **Test thoroughly** - Use testing checklist
5. **Document** - Update this roadmap with learnings

---

**Last Updated:** 2025-10-23
**Status:** rtxdi-quality-analyzer complete, 3 candidates identified
**Owner:** Ben + Claude Code
