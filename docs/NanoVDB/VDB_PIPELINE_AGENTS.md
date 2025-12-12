# VDB Pipeline Agent Ecosystem

**Version:** 1.1
**Status:** Production + Proposed Extensions
**Last Updated:** 2025-12-11

---

## Overview

The VDB pipeline uses a specialized agent ecosystem spanning both the **Blender worktree** (asset creation) and **NanoVDB worktree** (rendering). This document describes existing agents and proposes new specialized agents to improve the pipeline.

---

## Agent Architecture for Claude Max Subscribers

**Important:** This project uses a hybrid approach optimized for **Claude Max subscribers** who want to avoid paying twice (subscription + API keys).

### Agent Types Explained

| Type | Requires API Key | How It Works | Best For |
|------|------------------|--------------|----------|
| **MCP Server** | No | Python server exposing tools via Model Context Protocol | Specialized tools (conversion, analysis, profiling) |
| **Legacy Agent Prompt** | No | Markdown file with expertise/instructions loaded by Claude Code | Domain expertise (recipes, diagnostics, scripting) |
| **Agent SDK** | **Yes** | Full autonomous agents via `@anthropic-ai/agent-sdk` | Complex orchestration (only `mission-control` uses this) |

### Why This Architecture?

As of December 2025, the **Claude Agent SDK still requires API keys** - it does not work with Claude Max OAuth tokens. This was confirmed via:
- Official Agent SDK documentation
- GitHub issues #11, #6536 discussing OAuth token support

**Our Solution:**
- **MCP Servers** provide specialized tools (no API key needed)
- **Legacy Agent Prompts** provide domain expertise (no API key needed)
- Only **`mission-control`** uses Agent SDK (accepts API key cost for strategic orchestration)

### Practical Impact

When you see "Agent Spec" in this document, it means a **Legacy Agent Prompt** (`AGENT_PROMPT.md` file) - NOT an Agent SDK agent. These work entirely within your Claude Max subscription.

---

## Current Agent Ecosystem

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     BLENDER WORKTREE AGENTS                               │
│                   (Asset Creation & Documentation)                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ blender-manual (MCP Server)                                         │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/blender-manual/                                    │ │
│  │                                                                     │ │
│  │ 12 Tools:                                                           │ │
│  │ - search_manual          - search_vdb_workflow                      │ │
│  │ - search_tutorials       - search_python_api                        │ │
│  │ - browse_hierarchy       - search_nodes                             │ │
│  │ - read_page              - search_modifiers                         │ │
│  │ - search_semantic        - list_api_modules                         │ │
│  │ - search_bpy_operators   - search_bpy_types                         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ blender-scripting (Legacy Agent Prompt)                             │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/blender-scripting/AGENT_PROMPT.md                  │ │
│  │                                                                     │ │
│  │ Capabilities:                                                       │ │
│  │ - Generate bpy Python scripts from natural language                 │ │
│  │ - Debug script errors with context-aware suggestions                │ │
│  │ - Teach Blender-specific patterns (context, modes, data paths)      │ │
│  │ - Look up API via blender-manual MCP tools                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ celestial-body-curator (Legacy Agent Prompt)                        │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/celestial-body-curator/AGENT_PROMPT.md             │ │
│  │                                                                     │ │
│  │ Responsibilities:                                                   │ │
│  │ - Author/maintain recipe library (docs/blender_recipes/)            │ │
│  │ - Bridge astrophysical accuracy with Blender workflows              │ │
│  │ - Standardize VDB export settings for each celestial type           │ │
│  │ - Cross-reference with PlasmaDX material types                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ blender-diagnostics (Legacy Agent Prompt)                           │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/blender-diagnostics/AGENT_PROMPT.md                │ │
│  │                                                                     │ │
│  │ Capabilities:                                                       │ │
│  │ - Analyze scene diagnostic output                                   │ │
│  │ - Troubleshoot simulation/export issues                             │ │
│  │ - Provide step-by-step fix guidance                                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    NANOVDB WORKTREE AGENTS                                │
│                   (Rendering & Integration)                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ gaussian-analyzer (MCP Server)                                      │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/gaussian-analyzer/                                 │ │
│  │                                                                     │ │
│  │ Tools:                                                              │ │
│  │ - analyze_gaussian_parameters     - simulate_material_properties    │ │
│  │ - estimate_performance_impact     - compare_rendering_techniques    │ │
│  │ - validate_particle_struct                                          │ │
│  │                                                                     │ │
│  │ VDB Pipeline Use:                                                   │ │
│  │ - Validate material properties before loading in PlasmaDX           │ │
│  │ - Estimate FPS impact of volumetric assets                          │ │
│  │ - Compare rendering approaches (ray marching vs billboard)          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ materials-council (MCP Server)                                      │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/materials-council/                                 │ │
│  │                                                                     │ │
│  │ Tools:                                                              │ │
│  │ - generate_material_shader        - generate_particle_struct        │ │
│  │ - generate_material_config        - create_test_scenario            │ │
│  │ - generate_imgui_controls         - validate_file_syntax            │ │
│  │                                                                     │ │
│  │ VDB Pipeline Use:                                                   │ │
│  │ - Map Blender volume properties to PlasmaDX materials               │ │
│  │ - Generate shader code for new material types                       │ │
│  │ - Create test scenarios for validation                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ dxr-volumetric-pyro-specialist (MCP Server)                         │ │
│  │ Status: PRODUCTION                                                  │ │
│  │ Location: agents/dxr-volumetric-pyro-specialist/                    │ │
│  │                                                                     │ │
│  │ Tools:                                                              │ │
│  │ - research_pyro_techniques        - design_explosion_effect         │ │
│  │ - design_fire_effect              - estimate_pyro_performance       │ │
│  │ - compare_pyro_techniques                                           │ │
│  │                                                                     │ │
│  │ VDB Pipeline Use:                                                   │ │
│  │ - Design supernova/explosion effects for recipes                    │ │
│  │ - Specify temporal dynamics for animated VDB sequences              │ │
│  │ - Estimate performance of complex pyro effects                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed New Agents

### 1. vdb-pipeline-orchestrator

**Purpose:** Coordinate the full VDB asset pipeline from Blender to PlasmaDX.

**Type:** Legacy Agent Prompt (no API key required)

**Responsibilities:**
- Guide users through complete asset creation workflow
- Route queries to appropriate specialized agents
- Track asset status (created, exported, converted, tested)
- Validate pipeline stages before proceeding

**Interaction Model:**

```
User: "I want to create a nebula for my scene"
       │
       ▼
vdb-pipeline-orchestrator
       │
       ├──► celestial-body-curator: "Get hydrogen_cloud recipe"
       ├──► blender-scripting: "Generate creation script"
       ├──► (User executes in Blender)
       ├──► vdb-converter: "Convert .vdb to .nvdb"
       ├──► gaussian-analyzer: "Validate material properties"
       └──► materials-council: "Map to PlasmaDX material type"
```

**Proposed Location:** `agents/vdb-pipeline-orchestrator/AGENT_PROMPT.md`

---

### 2. vdb-converter-agent

**Purpose:** Automate VDB format conversion and validation.

**Type:** MCP Server (no API key required)

**Proposed Tools:**

| Tool | Purpose |
|------|---------|
| `convert_vdb_to_nvdb` | Convert OpenVDB to NanoVDB format |
| `batch_convert` | Convert directory of VDB files |
| `inspect_vdb` | Analyze VDB file contents (grids, bounds, size) |
| `validate_nvdb` | Verify NanoVDB file is loadable by PlasmaDX |
| `estimate_memory` | Estimate GPU memory for grid |
| `suggest_resolution` | Recommend resolution based on target FPS |

**Technical Implementation:**

```python
# agents/vdb-converter/vdb_converter_server.py
@server.tool("convert_vdb_to_nvdb")
async def convert_vdb_to_nvdb(input_path: str, output_path: str = None):
    """Convert OpenVDB file to NanoVDB format."""
    # Use existing scripts/convert_vdb_to_nvdb.py logic
    ...

@server.tool("inspect_vdb")
async def inspect_vdb(path: str):
    """Return grid information for VDB file."""
    # Use existing scripts/inspect_vdb.py logic
    ...
```

**Proposed Location:** `agents/vdb-converter/`

---

### 3. nanovdb-performance-profiler

**Purpose:** Profile and optimize NanoVDB rendering performance.

**Type:** MCP Server (no API key required)

**Proposed Tools:**

| Tool | Purpose |
|------|---------|
| `profile_render_pass` | Measure NanoVDB render time |
| `analyze_ray_efficiency` | Check ray marching efficiency |
| `suggest_step_size` | Recommend optimal step size |
| `compare_settings` | A/B test different parameters |
| `identify_bottleneck` | Find performance bottlenecks |

**Integration with PIX:**

```python
@server.tool("capture_nanovdb_frame")
async def capture_nanovdb_frame():
    """Capture PIX frame focused on NanoVDB dispatch."""
    # Trigger PIX capture during NanoVDB::Render()
    ...
```

**Proposed Location:** `agents/nanovdb-profiler/`

---

### 4. blender-scene-validator

**Purpose:** Validate Blender scenes before export to catch common issues.

**Type:** MCP Server (no API key required, runs in Blender Python context)

**Proposed Tools:**

| Tool | Purpose |
|------|---------|
| `validate_domain_settings` | Check fluid domain is correctly configured |
| `check_cache_path` | Verify cache directory exists and is writable |
| `validate_frame_range` | Ensure frame range matches bake settings |
| `check_resolution` | Warn if resolution is too high for real-time |
| `estimate_bake_time` | Estimate simulation bake duration |
| `preflight_export` | Full validation before baking |

**Proposed Location:** `agents/blender-scene-validator/`

---

## Agent Interaction Patterns

### Pattern 1: Recipe-Driven Asset Creation

```
celestial-body-curator
    │
    ├──► Provides recipe with settings
    │
    └──► blender-scripting
              │
              ├──► Generates Python script
              │
              └──► blender-manual (MCP)
                        │
                        └──► API documentation lookup
```

### Pattern 2: Conversion Pipeline

```
vdb-converter-agent
    │
    ├──► convert_vdb_to_nvdb
    │
    ├──► gaussian-analyzer
    │         │
    │         └──► Validate material properties
    │
    └──► nanovdb-profiler
              │
              └──► Estimate performance impact
```

### Pattern 3: Troubleshooting Flow

```
User: "My VDB won't load in PlasmaDX"
    │
    ▼
blender-diagnostics
    │
    ├──► Scene validation (check export settings)
    │
    └──► vdb-converter
              │
              ├──► inspect_vdb (check grid contents)
              │
              └──► validate_nvdb (check file integrity)
```

---

## Property Mapping Reference

### Blender → PlasmaDX

| Blender Property | PlasmaDX Property | Conversion |
|------------------|-------------------|------------|
| Volume Density | `densityScale` | `densityScale = density * 0.4` |
| Anisotropy | `scatteringCoeff` (phase) | Direct map (-1 to +1) |
| Emission Strength | `emissionStrength` | `emission = strength * 0.25` |
| Volume Color | Procedural noise color | Tone-mapped |
| Absorption Color | Inverted for absorption | `1 - color` |
| Temperature | `TemperatureToColor()` | Blackbody curve |

### Material Type Mapping

| Celestial Body | Recipe | PlasmaDX Material |
|----------------|--------|-------------------|
| Emission Nebula | `hydrogen_cloud.md` | GAS_CLOUD |
| Dark Nebula | `dark_nebula.md` | DUST |
| Supernova | `supernova_remnant.md` | PLASMA |
| Stellar Corona | `accretion_corona.md` | PLASMA |
| Protoplanetary Disk | `protoplanetary_disk.md` | GAS_CLOUD |

---

## Implementation Priority

### Phase 1: Immediate (High Value, Low Effort)

| Agent | Status | Effort | Value |
|-------|--------|--------|-------|
| blender-manual | DONE | - | High |
| blender-scripting | DONE | - | High |
| celestial-body-curator | DONE | - | High |
| blender-diagnostics | DONE | - | Medium |

### Phase 2: Short-Term (Filling Gaps)

| Agent | Status | Effort | Value |
|-------|--------|--------|-------|
| vdb-converter-agent | PROPOSED | Medium | High |
| blender-scene-validator | PROPOSED | Medium | Medium |

### Phase 3: Optimization (After Core Workflow Stable)

| Agent | Status | Effort | Value |
|-------|--------|--------|-------|
| nanovdb-profiler | PROPOSED | High | Medium |
| vdb-pipeline-orchestrator | PROPOSED | High | Medium |

---

## Configuration

### MCP Server Registration (No API Key)

Add MCP servers to `.claude/settings.json`. These run locally and don't require API keys:

```json
{
  "mcpServers": {
    "blender-manual": {
      "command": "python",
      "args": ["agents/blender-manual/blender_server.py"]
    },
    "vdb-converter": {
      "command": "python",
      "args": ["agents/vdb-converter/vdb_converter_server.py"]
    }
  }
}
```

### Legacy Agent Prompt Registration (No API Key)

Add agent prompts to `.claude/commands/` as markdown files or reference them in CLAUDE.md. These provide domain expertise without requiring API keys.

Example: `agents/celestial-body-curator/AGENT_PROMPT.md`

### Agent SDK (Requires API Key)

Only `mission-control` uses the Claude Agent SDK, which requires an API key:

```bash
# mission-control is the ONLY Agent SDK agent
# It provides strategic orchestration across multiple councils
# Uses API key authentication (separate from Max subscription)
cd agents/mission-control
ANTHROPIC_API_KEY=your_key node index.js
```

**Cost consideration:** The Agent SDK incurs per-token API costs separate from your Claude Max subscription. Use sparingly for high-value orchestration tasks.

---

## Related Documentation

- [NanoVDB System Overview](./NANOVDB_SYSTEM_OVERVIEW.md)
- [Two-Worktree Workflow](./TWO_WORKTREE_WORKFLOW.md)
- [Blender Workflow Spec](../BLENDER_PLASMADX_WORKFLOW_SPEC.md)
- [Recipe Library](../blender_recipes/README.md)

---

*Document maintained by: Claude Code Agent Ecosystem*
