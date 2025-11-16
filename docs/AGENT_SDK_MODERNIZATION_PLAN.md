# Agent SDK Modernization & Documentation Consolidation Plan

**Branch**: 0.17.0 - Fresh Start
**Date**: 2025-11-16
**Goal**: Unified Agent SDK ecosystem with streamlined documentation

---

## Executive Summary

**Problem**: 85.5KB of multi-agent documentation spread across 8 files with significant overlap, using outdated Agent SDK patterns.

**Solution**:
1. Consolidate 8 docs → 2 master docs (save ~70KB, ~80% reduction)
2. Update mission-control to latest Agent SDK (Skills, setting_sources)
3. Create Agent Skills for high-level workflows
4. Deprecate outdated "agents" (MCP tool servers are correct, but some autonomous agent attempts were wrong)

**Estimated Time**: 4-6 hours

---

## Current State Analysis

### Documentation Audit (85.5KB total)

| Document | Size | Status | Action |
|----------|------|--------|--------|
| AUTONOMOUS_MULTI_AGENT_RAG_GUIDE.md | 23KB | 🟢 Keep | **MASTER DOC #1** - Consolidate others into this |
| MULTI_AGENT_PYRO_PLANNING_EXERCISE.md | 15KB | 🔴 Archive | Planning exercise, not architecture |
| COUNCIL_AGENT_IMPLEMENTATION_PLAN.md | 14KB | 🟡 Merge | Merge into MASTER #1 |
| NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md | 12KB | 🔴 Archive | Specific workflow example, now outdated |
| MULTI_AGENT_ROADMAP.md | 9.7KB | 🟡 Merge | Merge roadmap into CELESTIAL_RAG |
| AGENT_HIERARCHY_AND_ROLES.md | 5.1K | 🟢 Keep | **MASTER DOC #2** - Quick reference |
| IMPLEMENTATION_PLAN_CLAUDE_SDK.md | 4.6KB | 🔴 Delete | Outdated (incorrect MCP server approach) |
| RUNBOOK_MULTI_AGENT_RAG.md | 2.1KB | 🔴 Delete | Covered by mission-control README |

**Result After Consolidation**:
- **2 master docs**: `AUTONOMOUS_MULTI_AGENT_RAG_GUIDE.md` (comprehensive), `AGENT_HIERARCHY_AND_ROLES.md` (quick ref)
- **1 roadmap**: `CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md` (updated with latest status)
- **Total**: ~35KB (59% reduction from 85.5KB)

### Agent SDK Usage Audit

**Current mission-control.py** (OUTDATED):
```python
# Missing latest features:
options = ClaudeAgentOptions(
    cwd=str(PROJECT_ROOT),
    mcp_servers={...},
    # ❌ Missing: setting_sources for Skills
    # ❌ Missing: Skill tool in allowed_tools
    system_prompt=...,
)
```

**Updated mission-control.py** (MODERN):
```python
options = ClaudeAgentOptions(
    cwd=str(PROJECT_ROOT),

    # ✅ Enable Agent Skills
    setting_sources=["user", "project"],

    # ✅ MCP specialist tool servers
    mcp_servers={
        "dxr-image-quality-analyst": {...},
        "gaussian-analyzer": {...},
        # ...
    },

    # ✅ Allow Skills + specialist tools
    allowed_tools=[
        "Skill",  # NEW: Enable Agent Skills
        "mcp__dxr-image-quality-analyst__*",
        "mcp__gaussian-analyzer__*",
        # ...
    ],

    system_prompt=...,
)
```

---

## Modernization Strategy

### Phase 1: Documentation Consolidation (2 hours)

#### Step 1.1: Create Master Architecture Doc (1 hour)

**Target**: `AUTONOMOUS_MULTI_AGENT_RAG_GUIDE.md` (consolidate to ~30KB)

**Merge in**:
- COUNCIL_AGENT_IMPLEMENTATION_PLAN.md → Section "Council Agents"
- MULTI_AGENT_ROADMAP.md → Section "Roadmap"
- Keep only latest, accurate information

**New structure**:
```markdown
# Autonomous Multi-Agent RAG System

## Quick Start
- How to use mission-control (5 min read)
- Running tests
- Common workflows

## Architecture
- Mission-Control (Autonomous orchestrator with ClaudeSDKClient)
- Council Agents (Future: Rendering, Materials, Physics, Diagnostics)
- Specialist MCP Tool Servers (6 servers, 38+ tools)
- Agent Skills (High-level reusable workflows)

## Mission-Control Agent ✅ OPERATIONAL
- What it is (autonomous AI with ClaudeSDKClient)
- Proof of autonomous reasoning (test results)
- Usage examples
- How to extend

## Agent Skills (NEW)
- When to use Skills vs MCP tools
- Creating custom skills
- Example: Visual Quality Assessment Skill

## Council Agents (PLANNED)
- Why councils? (domain specialization)
- Implementation plan
- Estimated effort: 8-12 hours

## Specialist MCP Tool Servers ✅ CORRECT
- dxr-image-quality-analyst (5 tools)
- gaussian-analyzer (5 tools)
- material-system-engineer (9 tools)
- path-and-probe (6 tools)
- log-analysis-rag (6 tools)
- pix-debug (7 tools)
- dxr-shadow-engineer (research)

## Roadmap
- Phase 1: Mission-Control ✅ COMPLETE
- Phase 2: Agent Skills (THIS PHASE)
- Phase 3: Council Agents (PLANNED)
- Phase 4: Nightly Autonomous QA (FUTURE)

## Migration Guide
- From old "agents" to unified ecosystem
- What changed and why
- What's salvageable (80-90%)
```

#### Step 1.2: Update Quick Reference (30 min)

**Target**: `AGENT_HIERARCHY_AND_ROLES.md` (keep at ~5KB)

**Purpose**: Fast lookup for "what agent handles what domain"

**Structure**:
```markdown
# Agent Hierarchy - Quick Reference

## Tier 1: Strategic Orchestration
- **mission-control**: Autonomous strategic orchestrator
  - Location: agents/mission-control/autonomous_agent.py
  - Coordinates: All councils and specialist tools
  - Status: ✅ OPERATIONAL

## Tier 2: Domain Councils (PLANNED)
- **rendering-council**: Rendering, lighting, shadows, visual quality
- **materials-council**: Gaussians, particles, shaders, material types
- **physics-council**: PINN ML, simulation, dynamics
- **diagnostics-council**: Logs, PIX, crashes, debugging

## Tier 3: Specialist MCP Tool Servers ✅ CORRECT
[Table with server, tools count, status, purpose]

## Agent Skills (NEW)
[Table with skill name, description, when to use]

## Decision Tree
User query → Mission-Control → [Skill | Council | Direct MCP Tool]
```

#### Step 1.3: Archive/Delete Outdated Docs (30 min)

**Archive** (move to `docs/archive/`):
- MULTI_AGENT_PYRO_PLANNING_EXERCISE.md (planning exercise, not architecture)
- NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md (specific workflow, now outdated)
- MULTI_AGENT_ROADMAP.md (merged into CELESTIAL_RAG)
- COUNCIL_AGENT_IMPLEMENTATION_PLAN.md (merged into AUTONOMOUS guide)

**Delete** (obsolete):
- IMPLEMENTATION_PLAN_CLAUDE_SDK.md (incorrect approach, already replaced)
- RUNBOOK_MULTI_AGENT_RAG.md (covered by mission-control README)

---

### Phase 2: Mission-Control SDK Update (1 hour)

#### Step 2.1: Add Skills Support (30 min)

**File**: `agents/mission-control/autonomous_agent.py`

**Changes**:
```python
def create_options(self) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),

        # NEW: Enable Agent Skills from filesystem
        setting_sources=["user", "project"],

        mcp_servers={...},  # Existing specialist servers

        allowed_tools=[
            # NEW: Enable Skills
            "Skill",

            # Existing specialist tools
            "mcp__dxr-image-quality-analyst__list_recent_screenshots",
            "mcp__dxr-image-quality-analyst__compare_screenshots_ml",
            # ... (rest of tools)
        ],

        system_prompt="""You are Mission-Control, autonomous strategic orchestrator.

**Your Capabilities:**
1. **Agent Skills** - High-level reusable workflows (e.g., "Visual Quality Assessment")
2. **Council Agents** - Domain specialists (Rendering, Materials, Physics, Diagnostics) [FUTURE]
3. **Specialist MCP Tools** - Low-level tool coordination (38+ tools across 6 servers)

**Decision Framework:**
- User asks high-level workflow → Use Agent Skill
- User asks domain-specific question → Coordinate specialist MCP tools
- Complex cross-domain tasks → Use multiple specialists + synthesize

**Example:**
User: "Analyze visual quality"
You: Use "Visual Quality Assessment" Skill (if exists) OR coordinate dxr-image-quality-analyst tools

Remember: You have AUTONOMOUS AI REASONING - make strategic decisions!
""",
    )
```

#### Step 2.2: Test Skills Integration (30 min)

```bash
cd agents/mission-control
python autonomous_agent.py "Test Skills integration - do you see any Agent Skills?"
```

**Expected**: Agent reports available Skills (or "no Skills found yet")

---

### Phase 3: Create Agent Skills (2-3 hours)

#### Step 3.1: Create Skills Directory Structure

```bash
mkdir -p .claude/skills/visual-quality-assessment
mkdir -p .claude/skills/gpu-crash-diagnosis
mkdir -p .claude/skills/material-type-analysis
```

#### Step 3.2: Visual Quality Assessment Skill (1 hour)

**File**: `.claude/skills/visual-quality-assessment/SKILL.md`

```markdown
---
description: Comprehensive visual quality analysis using LPIPS ML comparison, PIX captures, and probe grid diagnostics. Detects rendering regressions, lighting issues, and visual artifacts.
keywords: visual quality, LPIPS, screenshot comparison, rendering regression, visual artifacts, probe grid, lighting
---

# Visual Quality Assessment Skill

## What This Does

Autonomously analyzes visual rendering quality using:
1. LPIPS ML-powered screenshot comparison (~92% human correlation)
2. PIX GPU capture analysis for bottlenecks
3. Probe grid lighting configuration diagnostics
4. Visual artifact detection

## When to Use

- "Analyze visual quality"
- "Compare screenshots"
- "Why does rendering look different?"
- "Detect visual regression"
- "LPIPS comparison"

## Workflow

### Step 1: List Recent Screenshots
Use `mcp__dxr-image-quality-analyst__list_recent_screenshots` to find latest 2 screenshots.

### Step 2: LPIPS Comparison
Use `mcp__dxr-image-quality-analyst__compare_screenshots_ml` with:
- before_path: Older screenshot
- after_path: Newer screenshot
- save_heatmap: true

### Step 3: Analyze Results
**Quality Gate**: LPIPS similarity ≥ 85% (target)

If LPIPS < 85%:
- 🚨 **CRITICAL DEGRADATION** - Humans will notice
- Investigate probe grid (mcp__path-and-probe__analyze_probe_grid)
- Check PIX captures (mcp__dxr-image-quality-analyst__analyze_pix_capture)
- Search logs for shader/config changes

If LPIPS ≥ 85%:
- ✅ **ACCEPTABLE QUALITY** - Minor differences only

### Step 4: Generate Report
Provide:
- LPIPS similarity % (and vs 85% threshold)
- Overall similarity %
- Visual difference interpretation
- Recommended next steps (if degradation detected)
- Link to difference heatmap

## Example Output

**CRITICAL DEGRADATION DETECTED**

LPIPS Similarity: 69.29% (❌ 33.7% below 85% threshold)
Overall Similarity: 51.21%

Evidence:
- 87.66% of pixels changed significantly
- Mean absolute difference: 11%
- Max difference: 122% (clipping/saturation issues)

Interpretation:
Catastrophic visual regression between Nov 2-4. Either major rendering feature broke or scene config changed radically.

Recommended Actions:
1. View difference heatmap: PIX/heatmaps/diff_*.png
2. Search logs Nov 2-4 for shader errors, probe grid changes, RTXDI failures
3. Check PIX captures for GPU errors

## Quality Gates

- **85% LPIPS threshold**: Minimum acceptable perceptual similarity
- **90% LPIPS target**: Ideal quality (imperceptible differences)
- **<70% LPIPS**: Critical degradation requiring immediate investigation
```

#### Step 3.3: GPU Crash Diagnosis Skill (1 hour)

**File**: `.claude/skills/gpu-crash-diagnosis/SKILL.md`

```markdown
---
description: Autonomous GPU crash and TDR (Timeout Detection and Recovery) diagnosis using PIX captures, buffer dumps, and log analysis. Identifies DXIL issues, resource state errors, and atomic contention.
keywords: GPU crash, TDR, hang, timeout, PIX, buffer dump, DXIL, shader execution, atomic contention
---

# GPU Crash Diagnosis Skill

## What This Does

Autonomously diagnoses GPU crashes and hangs using:
1. PIX .wpix capture analysis
2. Buffer dump validation
3. DXIL root signature inspection
4. Shader execution validation
5. Log-based pattern matching

## When to Use

- "GPU crash"
- "TDR timeout"
- "Application hangs"
- "Driver error"
- "Diagnose crash at X particles"

## Workflow

### Step 1: Gather Evidence
1. Check latest logs for crash patterns
2. Identify particle count threshold (if crash is count-related)
3. List recent PIX captures

### Step 2: PIX Capture Analysis
Use `mcp__pix-debug__analyze_pix_capture`:
- Analyzes GPU timeline
- Identifies long-running dispatches
- Detects resource state errors

### Step 3: Buffer Validation
Use `mcp__pix-debug__analyze_particle_buffers`:
- Validates particle data integrity
- Checks for NaN/Inf values
- Verifies buffer sizes

### Step 4: Shader Execution Validation
Use `mcp__pix-debug__validate_shader_execution`:
- Checks if compute shaders actually ran
- Validates diagnostic counters
- Detects silent execution failures

### Step 5: DXIL Root Signature Analysis (if needed)
Use `mcp__pix-debug__analyze_dxil_root_signature`:
- Disassembles DXIL shader
- Extracts root signature
- Compares to expected bindings

### Step 6: Synthesize Diagnosis
Provide:
- Root cause hypothesis
- Evidence supporting diagnosis
- Particle count threshold (if applicable)
- Recommended fix
- Confidence level (high/medium/low)

## Example Output

**GPU HANG DIAGNOSIS**

Root Cause: Atomic contention in Volumetric ReSTIR at ≥2045 particles

Evidence:
- TDR timeout at exactly 2045 particles (consistent threshold)
- PIX capture shows PopulateVolumeMip2 dispatch: 3.1 seconds (vs 0.5ms normal)
- Buffer validation: 5.35 particles/voxel (high density → atomic contention)
- No DXIL root signature issues detected

Explanation:
Volumetric ReSTIR uses atomics for reservoir updates. At 5.35 particles/voxel, atomic contention causes cascading delays, exceeding Windows TDR 3-second timeout.

Recommended Fix:
1. Replace Volumetric ReSTIR with probe grid (zero atomics)
2. OR reduce dispatch granularity (smaller thread groups)
3. OR implement spatial batching to reduce density

Confidence: HIGH (consistent reproduction, clear atomic bottleneck)
```

#### Step 3.4: Material Type Analysis Skill (1 hour)

**File**: `.claude/skills/material-type-analysis/SKILL.md`

```markdown
---
description: Analyze 3D Gaussian particle structure for material type expansion. Recommends particle struct modifications, shader changes, and performance impact for supporting diverse celestial materials (stars, gas clouds, dust, rocky/icy bodies).
keywords: material types, particle structure, 3D Gaussian, celestial materials, shader generation, performance impact
---

# Material Type Analysis Skill

## What This Does

Autonomously analyzes particle structure for material type expansion:
1. Current particle struct analysis (gaussian-analyzer)
2. Shader modification recommendations
3. Performance impact estimation
4. Material type property recommendations
5. Phase 1 vs Phase 2 implementation paths

## When to Use

- "Analyze particle structure for materials"
- "Material type expansion"
- "How to add new material types?"
- "Celestial material diversity"
- "Particle struct modifications"

## Workflow

### Step 1: Analyze Current Gaussian Structure
Use `mcp__gaussian-analyzer__analyze_gaussian_parameters`:
- analysis_depth: "comprehensive"
- focus_area: "all"

Returns:
- Current particle struct size (32 bytes)
- Shader analysis
- Material support status

### Step 2: Search Particle Struct Definition
Use `mcp__material-system-engineer__search_codebase`:
- Pattern: "struct.*Particle|ParticleData"
- File glob: "**/*.h"

### Step 3: Estimate Performance Impact
Use `mcp__gaussian-analyzer__estimate_performance_impact`:
- particle_struct_bytes: 48 or 64 (Phase 1 vs Phase 2)
- material_types_count: 8
- shader_complexity: "moderate"

Returns:
- Memory impact @ 100K particles
- Estimated FPS reduction
- Optimization opportunities

### Step 4: Generate Recommendations
Provide:
- **Phase 1** (minimal, 48 bytes): Add albedo + materialType
- **Phase 2** (full, 64 bytes): Add roughness, metallic, overrides
- Material type recommendations (8 types: PLASMA, STAR_MAIN_SEQUENCE, GAS_CLOUD, etc.)
- Shader modification requirements
- Performance trade-offs

## Example Material Types (8 total)

0. **PLASMA_BLOB**: Hot volumetric plasma, high emission, forward scattering
1. **STAR_MAIN_SEQUENCE**: 5000-10000K, spherical, minimal elongation
2. **STAR_GIANT**: 3000-5000K, large radius, diffuse edges
3. **GAS_CLOUD**: Wispy, backward scattering (g=-0.3), albedo-based
4. **DUST_PARTICLE**: Dense, high absorption, isotropic scattering
5. **ROCKY_BODY**: Hybrid surface/volume, reflected only, high albedo
6. **ICY_BODY**: Very high albedo (0.8-0.95), specular reflection

## Performance Impact Summary

**Phase 1** (48 bytes):
- Memory: +1.5 MB @ 100K particles
- FPS impact: ~10% reduction
- Benefit: 8 material types with constant buffer properties

**Phase 2** (64 bytes):
- Memory: +3.0 MB @ 100K particles
- FPS impact: ~15-18% reduction
- Benefit: Per-particle customization, hybrid rendering
```

---

### Phase 4: Update CELESTIAL_RAG Roadmap (30 min)

**File**: `docs/CELESTIAL_RAG_IMPLEMENTATION_ROADMAP.md`

**Updates**:
1. Mark Phase 1 (Mission-Control) as ✅ COMPLETE with proof
2. Add Phase 2: Agent Skills
3. Add Phase 3: Council Agents (from COUNCIL_AGENT_IMPLEMENTATION_PLAN.md)
4. Update timelines based on actual completion

**New Phase 2**:
```markdown
### Phase 2: Agent Skills (Week 3) - IN PROGRESS

**Goal**: Create reusable high-level workflow Skills

**Skills to Implement:**

1. ✅ **Visual Quality Assessment** (complete)
   - LPIPS ML comparison
   - PIX analysis
   - Probe grid diagnostics
   - Quality gate enforcement (85% threshold)

2. ✅ **GPU Crash Diagnosis** (complete)
   - PIX capture analysis
   - Buffer validation
   - DXIL root signature inspection
   - TDR diagnosis

3. ✅ **Material Type Analysis** (complete)
   - Particle struct analysis
   - Performance impact estimation
   - Material type recommendations

4. ⏳ **Rendering Optimization** (planned)
   - Bottleneck identification
   - FPS profiling
   - Optimization recommendations

**Deliverables:**
- [x] .claude/skills/visual-quality-assessment/SKILL.md
- [x] .claude/skills/gpu-crash-diagnosis/SKILL.md
- [x] .claude/skills/material-type-analysis/SKILL.md
- [ ] .claude/skills/rendering-optimization/SKILL.md
- [x] Mission-control updated with Skills support
- [ ] Integration tests for Skills

**Status**: 3/4 Skills complete, mission-control updated
```

---

## Implementation Timeline

### Immediate (Next 2 hours)

1. ✅ Documentation audit complete
2. ⏳ Consolidate 8 docs → 2 master docs (2 hours)
3. ⏳ Archive/delete outdated docs (30 min)

### Short-Term (4-6 hours this week)

4. ⏳ Update mission-control with Skills support (1 hour)
5. ⏳ Create 3 Agent Skills (3 hours):
   - Visual Quality Assessment
   - GPU Crash Diagnosis
   - Material Type Analysis
6. ⏳ Test Skills integration (1 hour)
7. ⏳ Update CELESTIAL_RAG roadmap (30 min)

### Medium-Term (1-2 weeks)

8. ⏳ Create Rendering Optimization Skill
9. ⏳ Build council agents (if needed after testing Skills)
10. ⏳ Nightly autonomous QA pipeline (Skills-based)

---

## Success Criteria

### Documentation Consolidation

- ✅ Reduced from 85.5KB → ~35KB (59% reduction)
- ✅ 8 files → 2 master docs + 1 roadmap
- ✅ Outdated docs archived or deleted
- ✅ No duplicate information

### Agent SDK Modernization

- ✅ Mission-control uses latest SDK patterns
- ✅ Skills support enabled (setting_sources, Skill tool)
- ✅ 3+ Agent Skills operational
- ✅ Skills documentation complete

### Unified Ecosystem

- ✅ Clear hierarchy: Skills → Autonomous Agents → MCP Tools
- ✅ Each layer has defined purpose
- ✅ No confusion about "what's an agent"
- ✅ Migration path documented

---

## What NOT to Change

### Keep As-Is ✅

1. **Specialist MCP Tool Servers** - All 6 servers are correct
2. **Mission-Control Architecture** - Autonomous agent pattern works
3. **CLAUDE.md** - Project context and collaboration preferences
4. **Session Handoffs** - Historical session documentation

### Archive (Don't Delete) 📦

1. Planning exercises (MULTI_AGENT_PYRO_PLANNING_EXERCISE.md)
2. Historical workflow examples (NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md)
3. Outdated roadmaps (MULTI_AGENT_ROADMAP.md - before consolidation)

**Reason**: Historical context useful for understanding decisions, but not primary documentation.

---

## Next Steps

### Immediate Actions

1. Review this plan with user for approval
2. Execute Phase 1: Documentation consolidation
3. Execute Phase 2: Mission-Control SDK update
4. Execute Phase 3: Create first 3 Agent Skills

### Validation

After each phase, test:
- Documentation: Can user find info quickly? Is it accurate?
- SDK update: Does mission-control still work? Skills detected?
- Skills: Do Skills trigger autonomously? Do they work correctly?

---

**Last Updated**: 2025-11-16
**Branch**: 0.17.0
**Status**: Ready for implementation
**Estimated Total Time**: 4-6 hours
