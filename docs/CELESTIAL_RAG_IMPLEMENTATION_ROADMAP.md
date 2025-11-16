# Celestial Agent RAG - Implementation Roadmap

**Date:** 2025-11-16
**Status:** Ready for Implementation
**Based on:** NVIDIA Multi-Agent RAG + Claude Agent SDK Architecture

---

## Executive Summary

This roadmap outlines the implementation of a **hierarchical, autonomous multi-agent RAG system** for PlasmaDX-Clean, transitioning from the current ad-hoc agent usage to a NVIDIA-style orchestrated workflow with Claude Agent SDK integration.

**Current State:**
- ✅ 7+ specialized agents operational (dxr-image-quality-analyst, log-analysis-rag, path-and-probe, etc.)
- ✅ MCP servers running via stdio (external processes)
- ✅ 53 research documents indexed
- ✅ LPIPS visual quality analysis working
- ✅ PIX/log ingestion functional
- ⚠️ **NO orchestration** - user manually coordinates agents
- ⚠️ **NO persistence** - context lost between sessions
- ⚠️ **NO automated QA loops** - manual screenshot/buffer capture

**Target State:**
- 🎯 Mission-control orchestrator managing all agents
- 🎯 Automated nightly QA pipeline (build → run → capture → analyze → report)
- 🎯 Session persistence via RAG store + SESSION_SUMMARY files
- 🎯 Council-based specialization (rendering, materials, physics, diagnostics)
- 🎯 Evidence-driven decisions with brutal honesty feedback
- 🎯 Celestial rendering evolution (gas clouds, stars, supernovae, rocky/icy bodies)

---

## Architecture Overview

### 4-Tier Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    STRATEGIC TIER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ mission-     │  │ knowledge-   │  │ backlog-     │  │
│  │ control      │  │ steward      │  │ scribe       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                     SYSTEMS TIER                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ RENDERING COUNCIL                                 │   │
│  │  • rt-lighting-architect (RayQuery)              │   │
│  │  • shadow-systems-engineer (PCSS → DXR inline)   │   │
│  │  • rtxdi-director (M5/M6)                        │   │
│  │  • restir-specialist (legacy ReSTIR)             │   │
│  │  • probe-grid-engineer (ambient volumes)         │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ MATERIALS COUNCIL                                 │   │
│  │  • gaussian-analyzer (structure/perf)            │   │
│  │  • material-system-engineer (file ops/codegen)   │   │
│  │  • volumetric-pyro-specialist (temporal pyro)    │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ PHYSICS COUNCIL                                   │   │
│  │  • pinn-integration-agent                        │   │
│  │  • particle-physics-shader-agent                 │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ DIAGNOSTICS COUNCIL                               │   │
│  │  • dxr-image-quality-analyst (LPIPS + 7-dim QA)  │   │
│  │  • log-analysis-rag (logs + PIX embeddings)      │   │
│  │  • pix-debugger-vX (buffer validation)           │   │
│  │  • metrics-runner (performance/telemetry)        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   OPERATIONS TIER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ particle-    │  │ config-      │  │ asset-       │  │
│  │ pipeline-    │  │ curator      │  │ metadata-    │  │
│  │ runner       │  │              │  │ agent        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Division of Labor

**Mission-Control** (in-process SDK server):
- Reads CLAUDE.md + session docs
- Dispatches work to councils via `dispatch_plan(task, council)`
- Records decisions: `record_decision(decision, rationale, artifacts)`
- Publishes status: `publish_status()` → `docs/sessions/SESSION_<date>.md`
- Enforces dependencies, resolves tool conflicts

**Knowledge-Steward** (in-process SDK server):
- Ingests docs/logs/PIX/screenshots into RAG store (Chroma DB)
- Fetches context: `fetch_context(domain="rtxdi")` to reload state
- Tags artifacts for semantic search
- Maintains vector DB entries for persistence

**Backlog-Scribe** (in-process SDK server):
- Auto-updates `NEXT_STEPS.md`, `docs/sessions/SESSION_*.md`
- Tracks open questions, blocking issues
- Links to evidence (PIX captures, buffer dumps, screenshots)

**Council Leads** (external stdio servers):
- Coordinate specialists within their domain
- Report to mission-control
- Prevent "too many cooks" by enforcing single point of contact per task

---

## Implementation Phases

### Phase 0: Foundation (Week 1) ✅ PARTIALLY COMPLETE

**Goal:** Config system + first skill working

**Completed:**
- ✅ Lighting config system (multi-light, probe-grid, RTXDI control)
- ✅ Scenario configs (multi_light_only, probe_grid_only, hybrid_lighting)
- ✅ First skill: `lighting-quality-comparison.md`
- ✅ Config loading via `--config=configs/scenarios/*.json`

**Remaining:**
- ⏳ Test scenario configs with actual runtime
- ⏳ Validate lighting-quality-comparison skill workflow

---

### Phase 1: Strategic Tier (Week 2)

**Goal:** Mission-control, knowledge-steward, backlog-scribe operational

#### 1.1 Mission-Control Agent

**Location:** `agents/mission-control/`

**Tools to Implement:**
```python
@tool("dispatch_plan", "Route work to specialized councils")
async def dispatch_plan(plan: str, priority: str) -> dict:
    # Parse plan, assign to councils, write to SESSION_<date>.md
    # Return: { "assignments": [{"council": "rendering", "task": "..."}] }

@tool("record_decision", "Log decision with rationale and artifacts")
async def record_decision(decision: str, rationale: str, artifacts: list) -> None:
    # Append to SESSION_<date>.md with timestamp
    # Link to PIX captures, buffer dumps, screenshots

@tool("publish_status", "Generate status report")
async def publish_status() -> dict:
    # Query all councils, aggregate status
    # Return: { "overall": "...", "councils": {...} }

@tool("handoff_to_agent", "Transfer task to specific agent")
async def handoff_to_agent(agent: str, task: str, context: dict) -> dict:
    # Prepare context, call agent's entry point
```

**SDK Configuration:**
```python
mission_control = create_sdk_mcp_server(
    name="mission-control",
    version="1.0.0",
    tools=[dispatch_plan, record_decision, publish_status, handoff_to_agent]
)
```

**Deliverables:**
- [ ] `agents/mission-control/server.py` (SDK in-process server)
- [ ] `agents/mission-control/README.md` (tool schemas, usage)
- [ ] Tool implementations for dispatch, record, publish, handoff
- [ ] Integration test: dispatch simple task to diagnostics council

#### 1.2 Knowledge-Steward Agent

**Location:** `agents/knowledge-steward/`

**Tools to Implement:**
```python
@tool("ingest_artifact", "Add document/log/screenshot to RAG store")
async def ingest_artifact(path: str, type: str, metadata: dict) -> None:
    # Embed with ChromaDB, tag with metadata
    # Types: "doc", "log", "pix", "screenshot", "buffer_dump"

@tool("fetch_context", "Retrieve context for domain")
async def fetch_context(domain: str, limit: int = 10) -> dict:
    # Query RAG store for relevant context
    # Domains: "rtxdi", "probe-grid", "gaussian", "shadows", etc.
    # Return: { "documents": [...], "artifacts": [...] }

@tool("semantic_search", "Search across all artifacts")
async def semantic_search(query: str, limit: int = 5) -> dict:
    # Semantic search across vector DB
```

**Deliverables:**
- [ ] `agents/knowledge-steward/server.py` (SDK in-process server)
- [ ] ChromaDB integration (local persistent storage)
- [ ] Artifact tagging schema (domain, date, agent, type)
- [ ] Integration test: ingest SESSION_HANDOFF_2025-11-15.md, fetch context for "probe-grid"

#### 1.3 Backlog-Scribe Agent

**Location:** `agents/backlog-scribe/`

**Tools to Implement:**
```python
@tool("update_next_steps", "Update NEXT_STEPS.md with new tasks")
async def update_next_steps(tasks: list) -> None:
    # Parse NEXT_STEPS.md, add/update tasks, maintain priorities

@tool("log_session_event", "Append event to SESSION_<date>.md")
async def log_session_event(event: str, severity: str, context: dict) -> None:
    # Timestamp + event + context (e.g., "RTXDI M5 patchwork persists")

@tool("link_evidence", "Link PIX/log/screenshot to session")
async def link_evidence(session_id: str, artifact_path: str, description: str) -> None:
    # Add artifact link to SESSION_<date>.md with description
```

**Deliverables:**
- [ ] `agents/backlog-scribe/server.py` (SDK in-process server)
- [ ] Auto-create `docs/sessions/SESSION_<YYYY-MM-DD>.md` on first call
- [ ] Integration test: log 3 events, link 2 PIX captures

**Phase 1 Success Criteria:**
- ✅ Mission-control can dispatch tasks to existing agents
- ✅ Knowledge-steward ingests 10+ documents into RAG store
- ✅ Backlog-scribe maintains SESSION_<date>.md with timestamps
- ✅ `/mcp` shows 3 new in-process servers connected

---

### Phase 2: Operations Tier (Week 3)

**Goal:** Automated build/run/capture pipeline

#### 2.1 Particle-Pipeline-Runner Agent

**Location:** `agents/particle-pipeline-runner/`

**Tools to Implement:**
```python
@tool("rebuild_shaders", "Recompile all shaders")
async def rebuild_shaders() -> dict:
    # Call MSBuild on CompileShaders target
    # Return: { "success": bool, "errors": [...] }

@tool("run_plasmadx", "Launch application with config")
async def run_plasmadx(config: str, duration_seconds: int = 120) -> dict:
    # Launch exe with --config=<path>
    # Wait duration_seconds, then terminate gracefully
    # Return: { "pid": int, "log_path": str }

@tool("capture_screenshot", "Trigger F2 screenshot programmatically")
async def capture_screenshot(tag: str = "") -> dict:
    # Send F2 keypress to PlasmaDX window
    # Return: { "path": "screenshots/screenshot_<timestamp>.bmp", "tag": tag }

@tool("collect_logs", "Copy latest log to session directory")
async def collect_logs(session_id: str) -> dict:
    # Find latest log in build/bin/Debug/logs/
    # Copy to docs/sessions/<session_id>/logs/
    # Return: { "log_path": str, "size_kb": int }

@tool("run_pix_capture", "Trigger PIX GPU capture")
async def run_pix_capture(frame: int = 120, preset: str = "quick") -> dict:
    # Use pixtool or programmatic PIX trigger
    # Return: { "capture_path": "PIX/Captures/...", "duration_ms": int }
```

**Deliverables:**
- [ ] `agents/particle-pipeline-runner/server.py` (SDK in-process server)
- [ ] Windows automation for F2 keypress (AutoHotkey or SendInput)
- [ ] PIX capture integration (pixtool wrapper or WinPixEventRuntime)
- [ ] Integration test: full pipeline (rebuild → run 30s → screenshot → logs)

#### 2.2 Config-Curator Agent

**Location:** `agents/config-curator/`

**Tools:**
```python
@tool("create_scenario", "Generate scenario config JSON")
async def create_scenario(name: str, lighting: dict, effects: dict) -> dict:
    # Generate JSON config from template
    # Validate against schema
    # Save to configs/scenarios/<name>.json

@tool("validate_config", "Check config syntax and completeness")
async def validate_config(path: str) -> dict:
    # Parse JSON, check required fields
    # Return: { "valid": bool, "errors": [...] }
```

**Deliverables:**
- [ ] `agents/config-curator/server.py`
- [ ] Config templates for common scenarios
- [ ] JSON schema validation

#### 2.3 Asset-Metadata-Agent

**Location:** `agents/asset-metadata-agent/`

**Tools:**
```python
@tool("capture_camera_pose", "Record camera position/rotation")
async def capture_camera_pose() -> dict:
    # Extract from latest log or runtime memory
    # Return: { "position": [x,y,z], "rotation": [yaw,pitch], "timestamp": ... }

@tool("manipulate_imgui_control", "Change ImGui slider/checkbox")
async def manipulate_imgui_control(control: str, value: float) -> None:
    # Use memory editing or config reload to change runtime setting
    # Examples: "probe_grid_intensity", "shadow_rays_enabled"

@tool("traverse_to_position", "Move camera to specific position")
async def traverse_to_position(position: list, duration_seconds: float = 2.0) -> None:
    # Interpolate camera movement over duration
```

**Deliverables:**
- [ ] `agents/asset-metadata-agent/server.py`
- [ ] Runtime memory manipulation (CheatEngine-style or shared memory)
- [ ] Camera macro system

**Phase 2 Success Criteria:**
- ✅ Pipeline-runner can build, run, and capture screenshots autonomously
- ✅ Config-curator generates valid scenario configs
- ✅ Asset-metadata-agent can move camera and capture poses
- ✅ Full end-to-end test: `mission-control dispatch_plan("capture lighting comparison")` → pipeline-runner executes

---

### Phase 3: Council Organization (Week 4)

**Goal:** Create council lead agents that coordinate specialists

#### 3.1 Rendering-Council Lead

**Location:** `agents/rendering-council/`

**Subordinate Agents:**
- rt-lighting-architect (NEW - focuses on particle_gaussian_raytrace.hlsl lighting)
- shadow-systems-engineer (EXISTING - dxr-shadow-engineer)
- rtxdi-director (NEW - RTXDI M5/M6 specialist)
- restir-specialist (NEW - legacy ReSTIR research)
- probe-grid-engineer (EXISTING - path-and-probe)

**Tools:**
```python
@tool("assign_rt_task", "Assign task to RT specialist")
async def assign_rt_task(specialist: str, task: str, priority: str) -> dict:
    # Dispatch to rt-lighting-architect, shadow-engineer, etc.
    # Track status, aggregate reports

@tool("summarize_status", "Get rendering council status")
async def summarize_status() -> dict:
    # Query all subordinate agents
    # Return: { "specialists": {...}, "blockers": [...], "progress": "..." }
```

**Deliverables:**
- [ ] `agents/rendering-council/server.py` (stdio server)
- [ ] Tool routing to subordinate agents
- [ ] Status aggregation logic
- [ ] Integration test: assign task to shadow-engineer via council

#### 3.2 Materials-Council Lead

**Subordinate Agents:**
- gaussian-analyzer (EXISTING)
- material-system-engineer (EXISTING)
- volumetric-pyro-specialist (EXISTING - dxr-volumetric-pyro-specialist)

**Deliverables:**
- [ ] `agents/materials-council/server.py`
- [ ] Coordination logic for struct extension → codegen → validation

#### 3.3 Physics-Council Lead

**Subordinate Agents:**
- pinn-integration-agent (NEW)
- particle-physics-shader-agent (NEW)

**Deliverables:**
- [ ] `agents/physics-council/server.py`
- [ ] Hybrid physics coordination (PINN + shader)

#### 3.4 Diagnostics-Council Lead

**Subordinate Agents:**
- dxr-image-quality-analyst (EXISTING)
- log-analysis-rag (EXISTING)
- pix-debugger-vX (EXISTING - pix-debug)
- metrics-runner (NEW - performance telemetry)

**Deliverables:**
- [ ] `agents/diagnostics-council/server.py`
- [ ] QA workflow orchestration

**Phase 3 Success Criteria:**
- ✅ All 4 councils operational
- ✅ Mission-control can dispatch to councils instead of individual agents
- ✅ Councils handle specialist coordination internally
- ✅ Status aggregation working (mission-control queries councils, councils query specialists)

---

### Phase 4: Automated QA Loop (Week 5)

**Goal:** Nightly autonomous pipeline

#### 4.1 Nightly Pipeline

**Schedule:** UTC 02:00 (cron job or Windows Task Scheduler)

**Workflow:**
```
1. mission-control.dispatch_plan("nightly QA run")
2. particle-pipeline-runner.rebuild_shaders()
3. particle-pipeline-runner.run_plasmadx(config="baseline", duration=120)
4. particle-pipeline-runner.capture_screenshot(tag="nightly_baseline")
5. particle-pipeline-runner.collect_logs(session_id=<today>)
6. particle-pipeline-runner.run_pix_capture(frame=120, preset="quick")
7. log-analysis-rag.ingest_logs(include_pix=true)
8. dxr-image-quality-analyst.assess_visual_quality(screenshot_path=<latest>)
9. knowledge-steward.ingest_artifact(path=<screenshot>, type="screenshot")
10. mission-control.publish_status() → docs/sessions/SESSION_<date>.md
11. If QA regression detected → backlog-scribe.log_session_event("regression", "high", {...})
```

**Deliverables:**
- [ ] Nightly pipeline script (Python orchestrator using ClaudeSDKClient)
- [ ] Cron/Task Scheduler configuration
- [ ] Email/Slack alerts on regression (optional)
- [ ] QA threshold configs (LPIPS > 0.30 = alert, FPS < 90 = alert)

#### 4.2 Quality Gates

**Implemented in mission-control hooks:**
```python
async def qa_gate_hook(input_data, tool_use_id, context):
    # Check if LPIPS change > 0.30
    # Check if FPS regression > 15%
    # If violated, require approval or auto-reject
```

**Deliverables:**
- [ ] Hook implementations in `agents/mission-control/hooks.py`
- [ ] Threshold configurations
- [ ] Alert mechanisms

**Phase 4 Success Criteria:**
- ✅ Nightly pipeline runs autonomously
- ✅ SESSION_<date>.md generated daily
- ✅ Regressions detected and logged
- ✅ Artifacts ingested into RAG store
- ✅ Quality gates enforced

---

### Phase 5: Celestial Rendering Evolution (Weeks 6-8)

**Goal:** Expand beyond plasma-only rendering to diverse celestial bodies

#### 5.1 Material Type Expansion

**New Material Types:**
1. **GAS_CLOUD** - Nebula wisps, stellar nurseries
   - Scattering: 0.9 (high scattering)
   - Absorption: 0.1 (translucent)
   - Procedural noise: Perlin 3-octave turbulence
   - Color: Temperature-dependent (3000K-15000K)

2. **STAR_MAIN_SEQUENCE** - Solar-like stars
   - Emission: 10.0 (self-luminous)
   - Temperature: 5000K-10000K
   - Corona effects via emission multiplier

3. **STAR_GIANT** - Red/blue giants
   - Temperature: 3000K-20000K
   - Size: 3-5× larger radius
   - Pulsating animation

4. **NEUTRON_STAR** - Ultra-dense remnants
   - Temperature: 1,000,000K+
   - Extreme emission (100.0)
   - Tiny radius, intense brightness

5. **SUPERNOVA** - Explosive events (temporal)
   - Temperature: 200,000K peak → exponential decay
   - Expansion: Quadratic acceleration
   - Duration: 3-5 seconds
   - Particle budget: +2000 temporary particles

6. **DUST** - Interstellar dust clouds
   - Opacity: 0.8 (opaque)
   - Scattering: 0.3 (low)
   - Brownian noise animation

7. **ROCKY** - Asteroids, planets
   - Surface scattering (not volumetric)
   - Albedo-based reflection
   - Normal mapping via noise

8. **ICY** - Comets, icy moons
   - High albedo (0.9)
   - Specular highlights
   - Sublimation effects near heat sources

#### 5.2 Material System Implementation

**Deliverables:**
- [ ] Extend `MaterialTypeProperties` struct in Config.h
- [ ] Add material-specific shaders in `shaders/materials/`
- [ ] ImGui controls for each material type
- [ ] JSON presets: `configs/materials/gas_cloud.json`, etc.
- [ ] Performance budgets validated by gaussian-analyzer

**Agent Workflow:**
1. `mission-control` dispatches to `materials-council`
2. `materials-council` coordinates:
   - `gaussian-analyzer` validates struct changes, estimates perf impact
   - `material-system-engineer` generates HLSL shaders + C++ code
   - `volumetric-pyro-specialist` designs temporal dynamics (for supernova, etc.)
3. `diagnostics-council` runs QA:
   - `dxr-image-quality-analyst` validates visual quality
   - `metrics-runner` checks FPS impact
4. `mission-control` records decision, publishes to SESSION_<date>.md

#### 5.3 Procedural Noise Stack

**Implementation:**
```hlsl
// shaders/materials/noise_stack.hlsl

struct NoiseLayerConfig {
    float frequency;
    float amplitude;
    uint octaves;
    float temporal_rate;
    uint noise_type;  // 0=Perlin, 1=Simplex, 2=Worley
};

float3 SampleNoiseStack(float3 position, float time, NoiseLayerConfig configs[4]) {
    float3 displacement = float3(0, 0, 0);
    for (uint i = 0; i < 4; i++) {
        if (configs[i].octaves == 0) continue;
        float noise = SampleNoise(configs[i].noise_type, position, time, configs[i]);
        displacement += noise * configs[i].amplitude;
    }
    return displacement;
}
```

**Deliverables:**
- [ ] Noise library: Perlin, Simplex, Worley implementations
- [ ] Noise stack compositor
- [ ] Per-material noise presets
- [ ] Performance optimization (noise caching)

#### 5.4 Temporal Animation System

**For dynamic materials (supernova, stellar flare, pulsating stars):**

```cpp
// src/materials/TemporalAnimation.h

struct TemporalEvent {
    uint8_t type;           // 0=supernova, 1=flare, 2=pulse
    float start_time;       // Event trigger time
    float duration;         // Event lifetime
    float3 epicenter;       // Event origin
    bool active;
};

class TemporalAnimationManager {
    void Update(float delta_time);
    void TriggerSupernova(float3 position);
    void TriggerStellarFlare(float3 position);
    std::vector<TemporalEvent> GetActiveEvents() const;
};
```

**Deliverables:**
- [ ] Temporal event manager (C++)
- [ ] Shader integration (pass events via constant buffer)
- [ ] ImGui controls for triggering events manually
- [ ] Automated event scheduling (for nightly QA demos)

**Phase 5 Success Criteria:**
- ✅ 8 material types implemented and validated
- ✅ Procedural noise stack working (3-4 layers max)
- ✅ Temporal animations (supernova, flare) running at 90+ FPS
- ✅ Presets for common celestial phenomena (nebula, star cluster, etc.)
- ✅ QA passing for all material types
- ✅ Documentation: `CELESTIAL_MATERIALS_GUIDE.md`

---

## SDK Integration Details

### Claude Agent SDK Configuration

**File:** `.claude/agent_config.py`

```python
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
from agents.mission_control.server import mission_control
from agents.knowledge_steward.server import knowledge_steward
from agents.backlog_scribe.server import backlog_scribe
from agents.particle_pipeline_runner.server import pipeline_runner

PROJECT_ROOT = Path("/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

def get_agent_options():
    return ClaudeAgentOptions(
        cwd=PROJECT_ROOT,
        permission_mode="manual",  # Require approval for edits
        allowed_tools=[
            # Mission-control
            "mcp__mission-control__dispatch_plan",
            "mcp__mission-control__record_decision",
            "mcp__mission-control__publish_status",

            # Knowledge-steward
            "mcp__knowledge-steward__ingest_artifact",
            "mcp__knowledge-steward__fetch_context",

            # Pipeline-runner
            "mcp__pipeline-runner__rebuild_shaders",
            "mcp__pipeline-runner__run_plasmadx",
            "mcp__pipeline-runner__capture_screenshot",

            # Existing external agents
            "mcp__dxr-image-quality-analyst__*",
            "mcp__log-analysis-rag__*",
            "mcp__path-and-probe__*",
            # ... add all external agent tools
        ],
        mcp_servers={
            # In-process SDK servers
            "mission-control": mission_control,
            "knowledge-steward": knowledge_steward,
            "backlog-scribe": backlog_scribe,
            "pipeline-runner": pipeline_runner,

            # External stdio servers
            "dxr-image-quality-analyst": {
                "type": "stdio",
                "command": "bash",
                "args": ["-lc", "cd agents/dxr-image-quality-analyst && ./run_server.sh"],
                "cwd": str(PROJECT_ROOT),
                "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
            },
            "log-analysis-rag": {
                "type": "stdio",
                "command": "bash",
                "args": ["-lc", "cd agents/log-analysis-rag && ./run_server.sh"],
                "cwd": str(PROJECT_ROOT),
                "env": {"PROJECT_ROOT": str(PROJECT_ROOT)}
            },
            # ... add all external agents
        },
        hooks={
            "PreToolUse": [guard_bash, enforce_background_execution],
            "PostToolUse": [log_tool_usage]
        }
    )
```

### Hooks for Safety

```python
# agents/mission_control/hooks.py

async def guard_bash(input_data, tool_use_id, context):
    """Prevent destructive bash commands"""
    if input_data.get("tool_name") != "Bash":
        return {}

    cmd = input_data.get("tool_input", {}).get("command", "")
    forbidden = ["rm -rf", "format", "diskpart", "dd if="]

    if any(x in cmd for x in forbidden):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Dangerous command blocked: {cmd}"
            }
        }
    return {}

async def enforce_background_execution(input_data, tool_use_id, context):
    """Ensure long-running processes use background execution"""
    if input_data.get("tool_name") != "Bash":
        return {}

    cmd = input_data.get("tool_input", {}).get("command", "")
    long_running = ["pix_capture", "MSBuild", "./PlasmaDX-Clean.exe"]

    if any(x in cmd for x in long_running) and "--background" not in cmd and "&" not in cmd:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Long-running process must use background execution: {cmd}"
            }
        }
    return {}

async def log_tool_usage(output_data, tool_use_id, context):
    """Log all tool uses for audit trail"""
    tool_name = output_data.get("tool_name")
    # Write to audit log: logs/tool_audit_<date>.log
    # Include: timestamp, tool_name, tool_input, result
    return {}
```

---

## File Structure

```
PlasmaDX-Clean/
├── agents/
│   ├── mission-control/          # NEW - Strategic orchestrator
│   │   ├── server.py
│   │   ├── hooks.py
│   │   └── README.md
│   ├── knowledge-steward/         # NEW - RAG curator
│   │   ├── server.py
│   │   ├── chroma_db/            # ChromaDB storage
│   │   └── README.md
│   ├── backlog-scribe/            # NEW - Session documenter
│   │   ├── server.py
│   │   └── README.md
│   ├── particle-pipeline-runner/  # NEW - Build/run automation
│   │   ├── server.py
│   │   ├── automation/           # Windows automation scripts
│   │   └── README.md
│   ├── config-curator/            # NEW - Config generation
│   ├── asset-metadata-agent/      # NEW - Runtime control
│   ├── rendering-council/         # NEW - Council lead
│   ├── materials-council/         # NEW - Council lead
│   ├── physics-council/           # NEW - Council lead
│   ├── diagnostics-council/       # NEW - Council lead
│   ├── dxr-image-quality-analyst/ # EXISTING
│   ├── log-analysis-rag/          # EXISTING
│   ├── path-and-probe/            # EXISTING
│   ├── pix-debug/                 # EXISTING
│   └── ... (other existing agents)
│
├── docs/
│   ├── sessions/                  # NEW - Daily session summaries
│   │   ├── SESSION_2025-11-16.md
│   │   ├── SESSION_2025-11-17.md
│   │   └── ...
│   ├── research/                  # Existing research library
│   ├── CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md  # THIS FILE
│   └── ... (existing docs)
│
├── .claude/
│   ├── agent_config.py            # NEW - SDK configuration
│   ├── mcp_settings.json          # Updated with all agents
│   └── skills/
│       └── lighting-quality-comparison.md  # EXISTING
│
├── configs/
│   ├── scenarios/                 # EXISTING (recently added)
│   │   ├── multi_light_only.json
│   │   ├── probe_grid_only.json
│   │   └── hybrid_lighting.json
│   └── materials/                 # NEW - Material presets
│       ├── gas_cloud.json
│       ├── supernova.json
│       └── ... (8 material types)
│
├── shaders/
│   └── materials/                 # NEW - Per-material shaders
│       ├── noise_stack.hlsl
│       ├── gas_cloud_raytrace.hlsl
│       ├── supernova_raytrace.hlsl
│       └── ...
│
└── scripts/
    └── nightly_pipeline.py        # NEW - Automated QA pipeline
```

---

## Success Metrics

### Phase 1 (Strategic Tier)
- ✅ 3 in-process SDK servers operational (mission-control, knowledge-steward, backlog-scribe)
- ✅ RAG store ingests 50+ documents
- ✅ SESSION_<date>.md auto-generated with decisions + artifacts

### Phase 2 (Operations Tier)
- ✅ Pipeline-runner autonomously builds, runs, captures
- ✅ End-to-end test: dispatch task → autonomous execution → result in SESSION_<date>.md

### Phase 3 (Councils)
- ✅ 4 council leads operational
- ✅ Mission-control dispatches to councils (not individual agents)
- ✅ Status aggregation working (councils → specialists)

### Phase 4 (QA Loop)
- ✅ Nightly pipeline runs without human intervention
- ✅ Regressions detected and logged
- ✅ Quality gates enforced (LPIPS, FPS thresholds)

### Phase 5 (Celestial Rendering)
- ✅ 8 material types implemented
- ✅ Procedural noise stack working
- ✅ Temporal animations (supernova) at 90+ FPS
- ✅ Presets for common phenomena

---

## Risk Mitigation

### Risk 1: SDK Learning Curve
**Mitigation:** Start with simple in-process servers (mission-control tool with 1 function), iterate

### Risk 2: External Agent Integration
**Mitigation:** Keep existing stdio servers, wrap in SDK config, migrate gradually

### Risk 3: Performance Overhead
**Mitigation:** Use in-process servers for lightweight tasks, stdio for heavy ML/GPU

### Risk 4: Context Loss
**Mitigation:** Persistent RAG store + SESSION_<date>.md files survive context resets

### Risk 5: Agent Complexity Explosion
**Mitigation:** Council leads prevent direct agent coordination, enforce single point of contact

---

## Next Actions

**Immediate (This Week):**
1. Test scenario configs (`multi_light_only.json`) with runtime
2. Validate lighting-quality-comparison skill workflow
3. Start Phase 1: Create `agents/mission-control/` directory structure

**Short-Term (Weeks 2-3):**
4. Implement mission-control tool set (dispatch, record, publish)
5. Set up ChromaDB for knowledge-steward
6. Build particle-pipeline-runner automation

**Medium-Term (Weeks 4-5):**
7. Create council lead agents
8. Deploy nightly QA pipeline
9. Validate end-to-end autonomous workflow

**Long-Term (Weeks 6-8):**
10. Material type expansion (8 types)
11. Procedural noise stack
12. Temporal animations (supernova, flares)
13. Comprehensive QA validation

---

**Status:** Ready for Implementation
**Priority:** HIGH - Foundation for all future autonomous work
**Estimated Completion:** 8 weeks (with parallel agent development)
**Dependencies:** Claude Agent SDK, ChromaDB, Windows automation (AutoHotkey/SendInput)

---

**References:**
- `docs/research/CELESTIAL_AGENT_ECOSYSTEM_PLAN.md`
- `docs/NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md`
- `docs/IMPLEMENTATION_PLAN_CLAUDE_SDK.md`
- `docs/RUNBOOK_MULTI_AGENT_RAG.md`
- Claude Agent SDK: https://github.com/anthropics/claude-agent-sdk-python
