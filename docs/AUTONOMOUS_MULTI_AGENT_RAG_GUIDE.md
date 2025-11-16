# Autonomous Multi-Agent RAG Workflow - Complete Guide

**Version:** 1.0
**Last Updated:** 2025-11-15
**Status:** Phase 0.1 Complete (Probe-Grid Specialist Operational)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Agent Hierarchy](#agent-hierarchy)
4. [MCP Tools Reference](#mcp-tools-reference)
5. [Workflow Examples](#workflow-examples)
6. [Technical Architecture](#technical-architecture)
7. [Troubleshooting](#troubleshooting)
8. [Current Issues & Roadmap](#current-issues--roadmap)

---

## Quick Start

### For First-Time Users

**What is this system?**
A multi-agent diagnostic and development system for PlasmaDX that uses:
- **RAG (Retrieval Augmented Generation)** for intelligent log/buffer analysis
- **Specialized MCP agents** for rendering diagnostics, performance analysis, and implementation
- **LangGraph workflows** for self-correcting diagnostic loops

**How do I use it?**

1. **Start Claude Code** (agents auto-connect via `~/.claude.json`)
2. **Ask a diagnostic question:**
   ```
   "Why is probe grid lighting so dim?"
   "Diagnose black dots at far distances"
   "Compare RTXDI M4 vs M5 performance"
   ```
3. **Claude launches appropriate agents** automatically
4. **Review agent findings** and apply fixes

### Active Agents (2025-11-15)

| Agent | Status | Purpose |
|-------|--------|---------|
| **path-and-probe** | ✅ Operational | Probe-grid lighting diagnostics (6 tools) |
| **log-analysis-rag** | ✅ Operational | RAG-based log/buffer analysis (6 tools) |
| **dxr-image-quality-analyst** | ✅ Operational | Visual quality analysis (5 tools) |
| **pix-debug** | ⚠️ Outdated | GPU debugging (needs update for probe-grid) |

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code (CLI)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │   User Query: "Why is probe grid dim?"                │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          Multi-Agent Orchestration Layer              │  │
│  │  (Analyzes query → Routes to specialists)             │  │
│  └───────────────────────────────────────────────────────┘  │
│         │              │              │              │       │
│         ▼              ▼              ▼              ▼       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Tier 3  │  │  Tier 3  │  │  Tier 4  │  │  Tier 4  │   │
│  │  path-   │  │   rt-    │  │   log-   │  │   dxr-   │   │
│  │  probe   │  │ lighting │  │ analysis │  │  image-  │   │
│  │ (active) │  │(planned) │  │   -rag   │  │ quality  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         │              │              │              │       │
│         └──────────────┴──────────────┴──────────────┘       │
│                            ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         PlasmaDX-Clean Rendering Engine               │  │
│  │  - PIX captures (.wpix)                               │  │
│  │  - Buffer dumps (.bin)                                │  │
│  │  - Application logs                                   │  │
│  │  - Screenshots (F2)                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Tier 3 Rendering Specialists** (Domain experts)
   - Deep knowledge of specific RT subsystems
   - Can read code, analyze shaders, diagnose issues
   - Example: `path-and-probe` (probe-grid expert)

2. **Tier 4 Diagnostics** (Cross-cutting analysis)
   - RAG-powered semantic search (log-analysis-rag)
   - ML-based visual quality (dxr-image-quality-analyst)
   - Performance profiling (pix-debug)

3. **MCP Tools** (Agent capabilities)
   - Launch PlasmaDX.exe with configs
   - Capture PIX GPU traces
   - Analyze buffer dumps
   - Compare screenshots with ML (LPIPS)
   - Query logs with RAG

---

## Agent Hierarchy

### Tier 3: Rendering Specialists

#### **path-and-probe** ✅ ACTIVE
**Expertise:** Probe-grid lighting system (current active lighting)
**Files:** `src/lighting/ProbeGridSystem.{h,cpp}`, `shaders/probe_grid/update_probes.hlsl`

**Tools (6):**
1. `analyze_probe_grid` - Config & performance analysis
2. `validate_probe_coverage` - Particle distribution coverage check
3. `diagnose_interpolation` - Detect black dots/banding artifacts
4. `optimize_update_pattern` - Performance tuning for target FPS
5. `validate_sh_coefficients` - Spherical harmonics data integrity
6. `compare_vs_restir` - Benchmark vs shelved ReSTIR

**Common Use Cases:**
```bash
# Black dots at far distances
mcp__path-and-probe__diagnose_interpolation(symptom="black dots at far distances")

# Optimize for 120 FPS
mcp__path-and-probe__optimize_update_pattern(target_fps=120, particle_count=10000)

# Validate probe coverage
mcp__path-and-probe__validate_probe_coverage(particle_bounds="[-1500, +1500]")
```

#### **rt-lighting-engineer** ⏳ PLANNED
**Expertise:** Particle-to-particle RT lighting, emission models
**Files:** `particle_gaussian_raytrace.hlsl`, `RTLightingSystem_RayQuery.{h,cpp}`
**Status:** Needs creation (Phase 0.3)

#### **sampling-and-distribution** ⏳ PLANNED
**Expertise:** RTXDI/ReSTIR light sampling strategies
**Files:** `rtxdi_raygen.hlsl`, temporal accumulation shaders
**Status:** Needs creation (ReSTIR currently shelved)

---

### Tier 4: Diagnostics

#### **log-analysis-rag** ✅ ACTIVE
**Expertise:** RAG-powered semantic search of logs/PIX/buffer dumps
**Database:** 374+ documents (ChromaDB + FAISS hybrid retrieval)

**Tools (6):**
1. `ingest_logs` - Index logs/PIX/buffers into RAG database
2. `diagnose_issue` - Run full LangGraph self-correcting workflow
3. `query_logs` - Direct hybrid retrieval (BM25 + FAISS)
4. `analyze_pix_capture` - Extract PIX GPU capture metadata
5. `read_buffer_dump` - Parse binary GPU buffer dumps
6. `route_to_specialist` - Recommend which specialist agent to use

**Example Workflow:**
```bash
# Ingest new logs
mcp__log-analysis-rag__ingest_logs(path="PIX/buffer_dumps/2025_11_15")

# Run diagnostic
mcp__log-analysis-rag__diagnose_issue(question="Why are probe buffers zeroed out?")

# Query specific logs
mcp__log-analysis-rag__query_logs(semantic_query="probe grid dispatch parameters")
```

#### **dxr-image-quality-analyst** ✅ ACTIVE
**Expertise:** Visual quality analysis, ML-based screenshot comparison

**Tools (5):**
1. `list_recent_screenshots` - List screenshots by date (newest first)
2. `compare_performance` - Compare legacy, RTXDI M4, M5 performance
3. `analyze_pix_capture` - Analyze PIX captures for bottlenecks
4. `compare_screenshots_ml` - ML-powered LPIPS perceptual similarity (~92% human correlation)
5. `assess_visual_quality` - AI vision analysis for volumetric quality (7 dimensions)

**Quality Rubric (7 Dimensions):**
- Volumetric depth
- Rim lighting
- Temperature gradient
- RTXDI stability
- Shadows
- Scattering
- Temporal stability

---

## MCP Tools Reference

### Launching PlasmaDX

**Agent:** `pix-debug` (or custom diagnostic agent - TBD)

```bash
# Launch with specific config
mcp__pix-debug__diagnose_gpu_hang(
  particle_count=10000,
  render_mode="gaussian",
  timeout_seconds=10
)
```

**Manual Launch:**
```bash
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json
```

### Buffer Dumps

**Issue:** Automated buffer dump (Ctrl+D) currently broken ⚠️
**Workaround:** Manual PIX capture via GPU Capture UI

**Planned Fix:** Create diagnostic agent that can:
1. Launch PlasmaDX.exe
2. Wait for initialization
3. Send runtime commands (Ctrl+D or config-based trigger)
4. Capture buffer dumps automatically
5. Analyze dumps with log-analysis-rag

### Screenshot Capture

**In-App:** Press **F2** during rendering
**Saves to:** `build/bin/Debug/screenshots/screenshot_YYYY-MM-DD_HH-MM-SS.bmp`
**Metadata:** Matching `.json` file with render settings

**ML Comparison:**
```bash
mcp__dxr-image-quality-analyst__compare_screenshots_ml(
  before_path="screenshots/screenshot_2025-11-15_17-59-25.bmp",
  after_path="screenshots/screenshot_2025-11-15_18-30-00.bmp",
  save_heatmap=true
)
```

---

## Workflow Examples

### Example 1: Diagnose Dim Probe Grid

**User Query:** "Probe grid lighting is extremely dim"

**Agent Workflow:**
1. **log-analysis-rag** ingests recent logs/buffers
2. **path-and-probe** analyzes probe grid configuration
3. **dxr-image-quality-analyst** captures screenshot, assesses quality
4. **Diagnosis:** Dispatch mismatch (only 32³ probes updated, not 48³)
5. **Fix Applied:** Dynamic dispatch calculation
6. **Validation:** Rebuild, test, confirm all 110,592 probes updating

### Example 2: Compare RTXDI M4 vs M5

**User Query:** "Compare RTXDI M4 and M5 performance"

**Agent Workflow:**
1. **dxr-image-quality-analyst** parses performance logs
2. **Tool:** `compare_performance(legacy_log, rtxdi_m4_log, rtxdi_m5_log)`
3. **Result:** FPS comparison, bottleneck analysis, quality rubric scores
4. **Recommendation:** Which version to use based on target FPS

### Example 3: Black Dots at Far Distances

**User Query:** "Black dots appearing at far distances"

**Agent Workflow:**
1. **log-analysis-rag** queries: "AABB generation sizing issue"
2. **path-and-probe** diagnoses: `diagnose_interpolation(symptom="black dots")`
3. **Findings:**
   - Probe out-of-bounds (particles beyond [-1500, +1500] grid)
   - AABB sizing issue in `particle_gaussian_raytrace.hlsl:602-610`
4. **Fix:** Add distance filtering logic to AABB shader
5. **Validation:** LPIPS screenshot comparison (before/after)

---

## Technical Architecture

### RAG System (log-analysis-rag)

**Vector Database:** ChromaDB + FAISS
**Retrieval Strategy:** Hybrid BM25 (keyword) + FAISS (semantic)
**Embedding Model:** sentence-transformers (384-dim)
**LangGraph Workflow:** Self-correcting diagnostic loop with confidence thresholding

**Indexed Documents:**
- Application logs (txt)
- PIX captures (wpix metadata)
- Buffer dumps (bin → parsed to text)
- Shader source (hlsl)
- C++ source (cpp, h)

**Query Flow:**
```
User Query → Embedding → [BM25 + FAISS] → Top-K Documents → LLM Analysis → Self-Correction Loop → Final Diagnosis
```

### MCP Server Architecture

**Standard MCP Pattern:**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("agent-name")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [...]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Route to tool implementations
    ...

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                        server.create_initialization_options())
```

**Connection:** stdio transport via `~/.claude.json`

---

## Troubleshooting

### Agent Won't Connect

**Symptom:** MCP server timeout or "Agent not found"

**Solutions:**
1. **Restart Claude Code** (servers only connect on startup, not runtime reconnect)
2. **Check `~/.claude.json`** for correct path to `run_server.sh`
3. **Verify virtual env:** `source agents/<agent>/venv/bin/activate`
4. **Test manually:** `cd agents/<agent> && bash run_server.sh`

### Buffer Dump Automation Broken ⚠️

**Current Issue:** Ctrl+D buffer dump not working (user must manually extract via PIX)

**Workaround:**
1. Open PIX GPU Capture UI
2. Capture frame manually
3. Export buffers via PIX interface
4. Save to `PIX/buffer_dumps/<date>/`

**Planned Fix:** Create diagnostic agent (see Current Issues section)

### pix-debug Diagnosing Wrong System ⚠️

**Issue:** pix-debug still diagnoses ReSTIR (shelved), not probe-grid (active)

**Workaround:** Use `path-and-probe` instead for probe-grid diagnostics

**Fix Required:** Update pix-debug or deprecate (Phase 0.2)

---

## Current Issues & Roadmap

### 🔴 CRITICAL ISSUES

#### 1. Buffer Dump Automation Broken ⚠️
**Impact:** HIGH - Agents can't autonomously capture diagnostic data
**Status:** Workaround via manual PIX capture
**Priority:** **IMMEDIATE**

**Required Solution:**
- Create diagnostic agent that can:
  1. Launch PlasmaDX.exe with specified config
  2. Wait for initialization (detect via log parsing or timeout)
  3. Send runtime commands (Ctrl+D or config-based trigger)
  4. Capture buffer dumps to `PIX/buffer_dumps/`
  5. Capture screenshot to `screenshots/`
  6. Analyze dumps with log-analysis-rag

**Proposed Agent:** `diagnostic-runner` or enhance existing `pix-debug`

#### 2. Probe Grid Still Extremely Dim
**Impact:** HIGH - Core rendering system not working despite dispatch fix
**Status:** Dispatch fix applied (110,592 probes updating), but still dim
**Priority:** **IMMEDIATE**

**Possible Causes:**
- Probe intensity too low (current: 800.0, try 2000.0?)
- Shader bug in irradiance accumulation
- Buffers actually writing but values incorrect
- Need buffer dump validation to confirm

**Next Steps:**
1. Fix buffer dump automation (prerequisite)
2. Capture probe buffer dump
3. Validate SH coefficients non-zero
4. Tune probe intensity if needed

#### 3. Config System Scattered & Broken ⚠️
**Impact:** MEDIUM - Agents can't select configs for different tasks
**Status:** Configs scattered (root vs `configs/`), system "useless" after updates
**Priority:** **MINOR** (user says can wait unless critical)

**Current State:**
```
Root level (scattered):
- config_dev.json
- config_dev.json.original
- config_pix_analysis.json
- config_pix_close.json
- config_pix_far.json
- config_pix_inside.json
- config_pix_veryclose.json

Organized directory:
- configs/ (proper location)
```

**Required Fix:**
1. Consolidate all configs to `configs/` directory
2. Create config profiles for agent tasks:
   - `configs/agents/diagnostic_default.json`
   - `configs/agents/probe_grid_focus.json`
   - `configs/agents/performance_test.json`
3. Update config loading to search `configs/` first
4. Document config selection API for agents

---

### ✅ COMPLETED (Phase 0.1)

- ✅ Created `path-and-probe` specialist (6 tools operational)
- ✅ Fixed `log-analysis-rag` MCP server connection
- ✅ Created `MULTI_AGENT_ROADMAP.md`
- ✅ Fixed probe grid dispatch mismatch (32³ → 48³)
- ✅ Simplified lighting system (disabled ambient, physical emission, dynamic emission)
- ✅ Created this consolidated guide

---

### 📋 ROADMAP (Phase 0.2-0.4)

**Phase 0.2: Infrastructure Fixes**
- [ ] Fix buffer dump automation (create diagnostic-runner agent)
- [ ] Update pix-debug for probe-grid or deprecate
- [ ] Consolidate config system (if deemed critical)

**Phase 0.3: Rendering Specialists**
- [ ] Create `rt-lighting-engineer` specialist
- [ ] Test full collaboration workflow:
  - path-and-probe → diagnoses probe-grid issue
  - log-analysis-rag → confirms with RAG data
  - rt-lighting-engineer → implements fix
  - dxr-image-quality-analyst → validates fix

**Phase 0.4: Quality & Optimization**
- [ ] Fix probe grid dim lighting issue
- [ ] Implement ML visual quality feedback loop
- [ ] Add automated regression testing

---

## Related Documentation

**Core References:**
- `MULTI_AGENT_ROADMAP.md` - Development priorities
- `AGENT_HIERARCHY_AND_ROLES.md` - Detailed agent specs
- `SESSION_HANDOFF_2025-11-15.md` - Latest session findings

**Technical Deep Dives:**
- `agents/AGENT_MCP_TOOL_SPECS.md` - MCP tool API specs
- `agents/AGENT_SDK_MIGRATION_ROADMAP.md` - Migration from old SDK
- `NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md` - NVIDIA RAG architecture
- `RUNBOOK_MULTI_AGENT_RAG.md` - Operational runbook

**Research:**
- `docs/research/NVIDIA_Log_Analysis_Multi-Agent_RAG.pdf` - Original NVIDIA paper
- `agents/ML_VISUAL_ANALYSIS_DESIGN.md` - ML quality analysis design

---

**Last Updated:** 2025-11-15
**Maintainer:** Claude Code Sessions
**Version:** 1.0 (Phase 0.1 Complete)
