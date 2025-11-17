# Agent Architecture Specification - PlasmaDX-Clean

**Date:** 2025-11-17
**Status:** DEFINITIVE SPEC (Post-Testing & Decision)
**Branch:** 0.17.2

---

## Executive Summary

After extensive testing and evaluation, the PlasmaDX-Clean agent architecture uses **Legacy Agents** (`.claude/agents/`) for daily development work, with **Agent SDK** reserved for automation tasks only. This provides £0 daily cost while maintaining full autonomous capabilities.

**Key Decision:** Agent SDK councils are NOT used for daily work - too expensive ($0.79+ per session) for equivalent capability to free legacy agents.

---

## Architecture Overview

### Tier 1: You (Ben) - Supervised-Autonomous Authority
- **Role:** Final authority, strategic direction, quality standards
- **Communication Style:** Brutal honesty preferred (no sugarcoating)
- **Approval Required For:**
  - Architecture changes
  - Performance trade-offs >5% FPS regression
  - Quality compromises (LPIPS < 0.85)
- **Autonomous Otherwise:** Agents handle routine analysis, bug fixes, testing

### Tier 2: Legacy Agents (`.claude/agents/`) - Daily Development
- **Cost:** £0 (included in Claude Code Max subscription)
- **Format:** Markdown files with YAML frontmatter
- **Capabilities:** Full context windows, separate conversations, resumable sessions
- **Tools:** Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, TodoWrite, Task
- **When to Use:** All daily development work, bug fixes, feature implementation, analysis

**Active Legacy Agents:**
1. **gaussian-volumetric-rendering-specialist** (`.claude/agents/gaussian-volumetric-rendering-specialist.md`)
   - 3D Gaussian volumetric rendering expert
   - Debugs anisotropic stretching, transparency, cube artifacts
   - 6-phase workflow: Analysis → Research → Design → Implementation → Validation → Documentation
   - Access to all 8 MCP servers (38+ tools)

**Planned Legacy Agents (Create as needed):**
2. **rendering-quality-specialist** - Visual quality analysis, LPIPS validation, probe grid optimization
3. **material-type-specialist** - Material system design, particle structure modifications
4. **performance-diagnostics-specialist** - Performance profiling, bottleneck analysis, optimization

### Tier 3: MCP Tool Servers (Specialists) - £0 Cost
- **Role:** Provide domain-specialized tools (NO AI reasoning, just tools)
- **Cost:** £0 (run locally, no API calls)
- **Connection:** stdio transport via `~/.claude.json`

**Operational Servers (7 total, 38+ tools):**

| Server | Tools | Status | Purpose |
|--------|-------|--------|---------|
| **dxr-image-quality-analyst** | 5 | ✅ Connected | LPIPS ML, visual quality, PIX analysis |
| **gaussian-analyzer** | 5 | ✅ Connected | 3D Gaussian analysis, material simulation, performance impact |
| **material-system-engineer** | 9 | ✅ Connected | Codebase ops, shader generation, struct validation |
| **path-and-probe** | 6 | ✅ Connected | Probe grid analysis, SH validation, interpolation diagnostics |
| **log-analysis-rag** | 6 | ⚠️ Needs fix | RAG log search, anomaly detection, routing to specialists |
| **pix-debug** | 7 | ⚠️ Needs fix | GPU hang diagnosis, buffer validation, DXIL analysis |
| **dxr-shadow-engineer** | Research | ✅ Available | Shadow technique research, RT shadow generation |

**Total:** 38+ tools covering rendering, materials, physics, diagnostics domains

---

## Agent SDK - Automation Only (NOT Daily Use)

### Mission-Control Autonomous Agent
**Location:** `agents/mission-control/autonomous_agent.py`
**Cost:** Pay per use (e.g., $0.79 per session)
**Status:** ✅ OPERATIONAL but NOT for daily work

**Proven Capabilities:**
- ✅ Independent AI reasoning with ClaudeSDKClient
- ✅ Strategic decision-making and tool coordination
- ✅ Multi-tool orchestration (demonstrated)
- ✅ Strategic pivots when approaches fail
- ✅ Evidence-based recommendations

**When to Use (Rare):**
- ✅ CI/CD automation (nightly builds, tests, QA)
- ✅ Multi-day persistent analysis (weekly health checks)
- ✅ Automated regression testing pipelines
- ❌ **NOT for daily ad-hoc analysis** (use legacy agents instead)

**Test Results (2025-11-17):**
- $0.79 cost for Gaussian artifact analysis
- Quality: Equivalent to legacy agent (£0)
- Verdict: Not worth daily cost

### Rendering-Council (Proof-of-Concept)
**Location:** `agents/rendering-council/rendering_council_agent.py`
**Status:** ✅ Built, tested, but NOT used daily (too expensive)

**Proof-of-concept demonstrated:**
- Agent SDK autonomous agents work
- Same analytical quality as legacy agents
- BUT: $0.79+ per session not justified for daily work

**Verdict:** Keep code for future automation, don't use daily.

---

## Decision Framework

### Use Legacy Agents When:
- ✅ Daily bug fixes and feature development
- ✅ Ad-hoc analysis and diagnostics
- ✅ Code refactoring and shader optimization
- ✅ Visual quality validation
- ✅ Performance profiling
- ✅ Material system design

**Rationale:** £0 cost, same quality, simpler workflow.

### Use Agent SDK When:
- ✅ CI/CD nightly automation (build → test → analyze → report)
- ✅ Weekly autonomous health checks
- ✅ Multi-day persistent analysis requiring session memory
- ❌ **NEVER for daily ad-hoc work** (legacy agents are free and equivalent)

**Rationale:** Automation justifies API cost, daily work doesn't.

---

## Tool Access Matrix

### Legacy Agents Have Access To:
1. **Claude Code Built-In Tools:**
   - Read, Write, Edit (file operations)
   - Bash (git, build, testing)
   - Glob, Grep (search)
   - WebSearch, WebFetch (research)
   - TodoWrite (task tracking)
   - Task (launch subagents)

2. **All MCP Tool Servers:**
   - `mcp__dxr-image-quality-analyst__*` (5 tools)
   - `mcp__gaussian-analyzer__*` (5 tools)
   - `mcp__material-system-engineer__*` (9 tools)
   - `mcp__path-and-probe__*` (6 tools)
   - `mcp__log-analysis-rag__*` (6 tools)
   - `mcp__pix-debug__*` (7 tools)
   - `mcp__dxr-shadow-engineer__*` (research tools)

3. **Subagent Launching:**
   - Can launch Task tool with specialized subagent types
   - Example: `Task(subagent_type="Explore", prompt="Find all RTXDI shaders")`

### Agent SDK Has Access To:
- Same MCP tool servers as legacy agents
- Persistent session memory (cross-session context)
- HTTP communication (for multi-agent networks - future)

---

## Implementation Patterns

### Creating a New Legacy Agent

**File:** `.claude/agents/your-agent-name.md`

**Template:**
```markdown
---
name: your-agent-name
description: Brief description of expertise
tools: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, TodoWrite, Task
color: cyan
---

# Your Agent Name - Expertise Area

## Core Responsibilities

[What this agent handles]

## Workflow Phases

### Phase 1: Problem Analysis
[How to analyze issues in this domain]

### Phase 2: Research & Diagnosis
[Research patterns, use MCP tools]

### Phase 3: Solution Design
[Design fixes, propose options]

### Phase 4: Implementation
[Apply fixes, test changes]

### Phase 5: Validation
[Validate with LPIPS, performance metrics, etc.]

### Phase 6: Documentation
[Update session logs, document decisions]

## MCP Tools Reference

### gaussian-analyzer (5 tools)
- `analyze_gaussian_parameters` - [when to use]
- `simulate_material_properties` - [when to use]
[... etc for relevant tools]

## Example Workflows

### Example 1: [Common Task Name]
**User asks:** "[typical user query]"

**Your workflow:**
1. [Step 1]
2. [Step 2]
3. [Deliver results]

## Quality Gates

- **LPIPS threshold:** ≥ 0.85 for visual changes
- **FPS threshold:** >5% regression requires user approval
- **Build health:** Must compile without errors
```

### Launching a Legacy Agent

```bash
# In Claude Code, just reference by name:
@gaussian-volumetric-rendering-specialist "Analyze anisotropic stretching bug"

# Or use /agents command:
/agents
# Then select agent from list
```

---

## MCP Tool Usage Examples

### Visual Quality Analysis
```bash
# List recent screenshots
mcp__dxr-image-quality-analyst__list_recent_screenshots(limit=10)

# LPIPS comparison
mcp__dxr-image-quality-analyst__compare_screenshots_ml(
  before_path="/path/to/before.png",
  after_path="/path/to/after.png",
  save_heatmap=true
)

# Visual quality assessment
mcp__dxr-image-quality-analyst__assess_visual_quality(
  screenshot_path="/path/to/screenshot.png"
)
```

### Gaussian Structure Analysis
```bash
# Analyze particle structure
mcp__gaussian-analyzer__analyze_gaussian_parameters(
  analysis_depth="comprehensive",
  focus_area="all"
)

# Simulate material changes
mcp__gaussian-analyzer__simulate_material_properties(
  material_type="GAS_CLOUD",
  properties={"opacity": 0.5, "scattering_coefficient": 0.8}
)

# Performance impact estimation
mcp__gaussian-analyzer__estimate_performance_impact(
  particle_struct_bytes=64,
  material_types_count=8,
  shader_complexity="moderate"
)
```

### Probe Grid Diagnostics
```bash
# Analyze grid configuration
mcp__path-and-probe__analyze_probe_grid(
  include_performance=true
)

# Diagnose interpolation artifacts
mcp__path-and-probe__diagnose_interpolation(
  symptom="black dots at far distances"
)

# Optimize update pattern
mcp__path-and-probe__optimize_update_pattern(
  target_fps=120,
  particle_count=10000
)
```

---

## Quality Gates & Standards

### Visual Quality
- **LPIPS threshold:** ≥ 0.85 (minimum acceptable perceptual similarity)
- **LPIPS target:** ≥ 0.90 (ideal quality, imperceptible differences)
- **Critical degradation:** < 0.70 LPIPS requires immediate investigation

### Performance
- **Target:** 165 FPS @ 10K particles with RT lighting
- **Acceptable:** 142 FPS @ 10K particles with RT lighting + shadows
- **Regression limit:** <5% FPS loss acceptable without approval
- **Major regression:** >5% FPS loss requires user approval

### Code Quality
- **Build health:** Must compile without errors
- **Shader compilation:** All shaders must compile to DXIL
- **Test passing:** Core functionality tests must pass

### Communication Style
**Per user's autism support needs:**
- ✅ **Brutal honesty** - No sugarcoating, direct feedback
- ✅ **Specific numbers** - "LPIPS 0.34 vs 0.92" not "quality degraded"
- ✅ **Clear next steps** - Actionable recommendations, not vague suggestions
- ✅ **Admit mistakes** - When agent was wrong, say so explicitly
- ❌ **No deflection** - Answer questions directly, don't dismiss

---

## Current Status (2025-11-17)

### Operational
- ✅ 1 legacy agent (gaussian-volumetric-rendering-specialist)
- ✅ 4/7 MCP tool servers connected
- ✅ Mission-Control Agent SDK (for automation only)
- ✅ Screenshot conversion tool (BMP → PNG, 93-99% compression)

### Needs Fixing
- ⚠️ log-analysis-rag connection issue (virtual env)
- ⚠️ pix-debug connection issue (virtual env)
- ⚠️ Anisotropic stretching still not working (velocity formula fixed in code, but not visible)

### Planned (Create as needed)
- ⏳ rendering-quality-specialist legacy agent
- ⏳ material-type-specialist legacy agent
- ⏳ performance-diagnostics-specialist legacy agent

---

## What Was Archived/Deprecated

### Abandoned Approaches (WRONG)
- ❌ Mission-control as MCP server (WRONG - should be Agent SDK agent)
- ❌ Council agents for daily work (too expensive, not justified)
- ❌ Agent Skills (complex, Agent SDK modernization not needed for daily work)

### Successful Approaches (KEEP)
- ✅ Legacy agents for daily work (£0 cost, full capabilities)
- ✅ MCP tool servers for specialists (correct as tools-only)
- ✅ Agent SDK for automation only (when cost justified)

### Archived Documents
**Moved to `docs/archive/`:**
- MULTI_AGENT_PYRO_PLANNING_EXERCISE.md (planning exercise)
- NVIDIA_MULTI_AGENT_RAG_WORKFLOW.md (specific workflow example)
- IMPLEMENTATION_PLAN_CLAUDE_SDK.md (outdated MCP approach)
- RUNBOOK_MULTI_AGENT_RAG.md (covered by this spec)
- COUNCIL_AGENT_IMPLEMENTATION_PLAN.md (not doing councils for daily work)
- AGENT_SDK_MODERNIZATION_PLAN.md (not needed for legacy agents)
- MULTI_AGENT_ROADMAP copy.md (duplicate)

**Reason:** Historical context useful but not primary architecture.

---

## File Locations

### Legacy Agents
- `.claude/agents/` - All legacy agent markdown files
- Example: `.claude/agents/gaussian-volumetric-rendering-specialist.md`

### MCP Tool Servers
- `agents/dxr-image-quality-analyst/` - Visual quality analysis tools
- `agents/gaussian-analyzer/` - 3D Gaussian structure tools
- `agents/material-system-engineer/` - Codebase operation tools
- `agents/path-and-probe/` - Probe grid diagnostic tools
- `agents/log-analysis-rag/` - RAG-based log analysis tools
- `agents/pix-debug/` - GPU debugging tools
- `agents/dxr-shadow-engineer/` - Shadow research tools

### Agent SDK (Automation Only)
- `agents/mission-control/` - Autonomous strategic orchestrator
- `agents/rendering-council/` - Proof-of-concept (not used daily)

### Tools & Scripts
- `tools/convert_screenshots.py` - BMP → PNG conversion (93-99% compression)

### Documentation
- `docs/AGENT_ARCHITECTURE_SPECIFICATION.md` - THIS FILE (DEFINITIVE)
- `docs/CLAUDE.md` - Project context, user preferences, collaboration style
- `docs/sessions/SESSION_*.md` - Historical session documentation
- `docs/archive/` - Archived/deprecated documentation

---

## Migration from Old Documentation

### If You're Reading Old Docs

**OLD (WRONG):**
- Mission-control is an MCP server
- Council agents for daily work
- Agent SDK for everything
- 8 documents with 85.5KB of overlapping info

**NEW (CORRECT):**
- Mission-control is Agent SDK agent (automation only)
- Legacy agents for daily work (£0 cost)
- MCP servers are tools only (no AI reasoning)
- 1 document (this file) with clear decisions

### If You See References to "Councils"

**Context:** Early planning assumed Agent SDK councils would be used daily.

**Reality:** After testing ($0.79 per session), decided not worth cost for daily work.

**Current Approach:** Legacy agents provide equivalent capability at £0 cost.

---

## Next Steps

### Immediate (This Session)
1. ✅ Audit documentation
2. ✅ Create master spec (this file)
3. ⏳ Archive old docs to `docs/archive/`
4. ⏳ Fix MCP connection issues (log-analysis-rag, pix-debug)

### Short-Term (This Week)
5. ⏳ Debug anisotropic stretching (still broken despite code fixes)
6. ⏳ Create 2-3 more legacy agents as needed
7. ⏳ Test full workflow: Agent → MCP tools → validation

### Long-Term (Automation)
8. ⏳ Set up nightly CI/CD with Mission-Control Agent SDK
9. ⏳ Automated regression testing pipeline
10. ⏳ Session memory system (RAG across historical decisions)

---

## Success Criteria

### Architecture
- ✅ Clear tier separation (You → Legacy Agents → MCP Tools)
- ✅ Cost optimization (£0 daily work, pay only for automation)
- ✅ No confusion about "what's an agent"

### Documentation
- ✅ 1 definitive spec (this file)
- ✅ Old docs archived (historical context preserved)
- ✅ No duplicate or contradictory information

### Workflow
- ✅ Legacy agents handle 100% of daily development
- ✅ Agent SDK used only for justified automation
- ✅ All MCP tool servers connected and operational

---

## Frequently Asked Questions

### Why not use Agent SDK for daily work?
**Cost:** $0.79+ per session adds up quickly.
**Benefit:** No quality advantage over free legacy agents.
**Decision:** Only use Agent SDK when automation justifies API cost.

### Why not build council agents?
**Reason:** Testing showed equivalent capability at £0 with legacy agents.
**Complexity:** Councils add HTTP communication layer without benefit.
**Decision:** Keep proof-of-concept code, don't use daily.

### Why keep MCP tool servers if agents do the work?
**Separation:** MCP servers provide specialized tools (NO AI reasoning).
**Reusability:** Both legacy agents AND Agent SDK can use same MCP tools.
**Design:** Clean separation of concerns (tools vs reasoning).

### Can I create new legacy agents?
**Yes!** Create `.claude/agents/your-agent-name.md` with YAML frontmatter.
**Template:** See "Implementation Patterns" section above.
**Tools:** Full access to Claude Code tools + all MCP servers.

### When should I use Agent SDK?
**Only for automation:**
- Nightly CI/CD builds & tests
- Weekly autonomous health checks
- Multi-day persistent analysis
**NEVER for daily ad-hoc work** (use legacy agents instead).

---

**Last Updated:** 2025-11-17
**Maintainer:** Ben (user) + Claude Code sessions
**Status:** DEFINITIVE POST-TESTING SPEC

**This is the single source of truth for agent architecture. Old docs are archived.**
