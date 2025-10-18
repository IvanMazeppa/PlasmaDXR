# Agent Modernization to v4 - Complete Plan
**Date:** 2025-10-18
**Purpose:** Upgrade all agents to v4 with proper MCP integration, web search, and fill critical gaps

---

## Executive Summary

**Critical Finding:** Agents are NOT properly using the DX12 Enhanced MCP Server, leaving massive amounts of valuable DirectX 12/DXR/HLSL documentation on the table.

**Root Cause:** Agents don't have persistent MCP query strategies and give up after 1-2 failed searches.

**Solution:** Modernize all agents to v4 with:
1. **Mandatory MCP consultation** - Always check MCP before web search
2. **Persistent query strategies** - Multiple search angles, never give up
3. **Web search integration** - Latest 2025 information (RTXDI, shadow techniques)
4. **Fill critical gaps** - DXR RT Shadow Engineer, RTXDI Specialist

---

## Critical Gaps Identified

### 1. **DXR RT Shadow & Lighting Engineer** ⭐ HIGH PRIORITY
**Status:** MISSING (Critical gap)
**Purpose:** Enhance basic shadow ray system with advanced techniques

**Why Critical:**
- Current shadow ray system is "very basic" (single ray per light)
- No soft shadows, penumbra, area light support
- Missing techniques: PCSS, contact hardening, temporal filtering
- RTXDI will need sophisticated shadow integration

**Capabilities Needed:**
- Research latest shadow ray techniques (2024-2025)
- Implement PCSS (Percentage-Closer Soft Shadows)
- Contact-hardening shadows (vary kernel by distance)
- Temporal shadow filtering
- Multi-bounce shadows for volumetrics
- Integration with RTXDI many-light shadows
- MCP expertise: `TraceRay`, `RayQuery`, shadow-specific HLSL intrinsics

**Tools:**
- `mcp__dx12-docs-enhanced__search_hlsl_intrinsics` for shadow intrinsics
- `mcp__dx12-docs-enhanced__search_by_shader_stage` for closesthit/anyhit stages
- Web search for latest PCSS, VSM, ESM variants (2024-2025)
- Access to `docs/research/` for past shadow research

### 2. **RTXDI Integration Specialist** ⭐ HIGH PRIORITY
**Status:** MISSING (Critical for Phase 4)
**Purpose:** Guide RTXDI SDK integration into PlasmaDX pipeline

**Why Critical:**
- Phase 4 goal: Replace custom ReSTIR with NVIDIA RTXDI
- Complex integration: SBT, light grid, reservoir buffers, validation
- Needs to understand BOTH PlasmaDX architecture AND RTXDI SDK
- Volumetric rendering + RTXDI is rare (most RTXDI is surface-based)

**Capabilities Needed:**
- RTXDI SDK architecture expertise
- Light grid setup for volumetric rendering
- Reservoir buffer management
- Integration with existing multi-light system
- Validation strategies (compare vs ground truth)
- MCP expertise: All DXR 1.1 APIs, state objects, SBT

**Tools:**
- `mcp__dx12-docs-enhanced__search_dxr_api` for RTXDI-related DXR APIs
- `mcp__dx12-docs-enhanced__get_dx12_entity` for detailed state object docs
- Web search for NVIDIA RTXDI docs, samples, integration guides
- Access to PlasmaDX codebase (CLAUDE.md, multi-light system)

---

## Agent Audit Results

### v3 Production Agents (4 agents)

#### buffer-validator-v3.md
**Current Strengths:**
- ✅ Excellent buffer format knowledge (32-byte structs)
- ✅ Clear validation rules
- ✅ Python parsing expertise

**Critical Gaps:**
- ❌ NO MCP usage at all
- ❌ NO web search for latest validation techniques
- ❌ Doesn't check DX12 MCP for buffer alignment rules
- ❌ Doesn't use `search_dx12_api` for D3D12 resource validation

**Modernization Needed:**
- Add MCP section: "Before validating, always check MCP for D3D12 buffer alignment requirements"
- Persistent MCP strategy: Try `search_dx12_api`, `search_dxr_api`, `search_all_sources`
- Web search for latest GPU buffer debugging techniques (PIX, NSight)
- Add validation for RTXDI-specific buffers (light grid, reservoirs)

#### pix-debugger-v3.md
**Current Strengths:**
- ✅ Excellent multi-light system knowledge
- ✅ Historical debugging context (ReSTIR bugs)
- ✅ Root cause analysis workflow

**Critical Gaps:**
- ❌ NO MCP usage for DXR debugging
- ❌ NO web search for latest DXR debugging techniques
- ❌ Doesn't query MCP for specific HLSL intrinsic behavior
- ❌ Doesn't search for shader model compatibility issues

**Modernization Needed:**
- Add MCP strategy: Always check MCP when diagnosing shader issues
- Query `search_by_shader_stage` to validate shader code correctness
- Query `get_dx12_entity` for TraceRay, RayQuery behavior details
- Web search for latest PIX, RenderDoc, NSight debugging workflows
- Add RTXDI debugging knowledge (once integrated)

#### stress-tester-v3.md
**Current Strengths:**
- ✅ Comprehensive test scenarios
- ✅ Clear performance targets
- ✅ Integration with other agents

**Critical Gaps:**
- ❌ NO MCP usage
- ❌ NO web search for stress testing best practices
- ❌ Doesn't check DX12 MCP for performance debugging APIs
- ❌ Missing RTXDI stress test scenarios

**Modernization Needed:**
- Add MCP strategy: Check for D3D12 profiling APIs, PIX markers
- Web search for latest GPU stress testing methodologies
- Add RTXDI-specific stress tests (light grid scaling, reservoir validation)
- Add shadow ray stress tests (1-64 rays per light scaling)

#### performance-analyzer-v3.md
**Current Strengths:**
- ✅ Clear performance targets
- ✅ Bottleneck catalog
- ✅ Optimization recommendations

**Critical Gaps:**
- ❌ NO MCP usage for performance APIs
- ❌ NO web search for latest optimization techniques
- ❌ Doesn't check DX12 MCP for ExecuteIndirect, async compute
- ❌ Missing RTXDI performance analysis

**Modernization Needed:**
- Add MCP strategy: Query for D3D12_COMMAND_LIST_TYPE_COMPUTE async patterns
- Web search for latest DXR performance optimization (2024-2025)
- Add RTXDI performance metrics (light grid traversal, reservoir updates)
- Add shadow ray performance analysis

### v2 Specialized Agents (6 agents - sampled 2)

#### dxr-systems-engineer-v2.md
**Current Strengths:**
- ✅ DXR pipeline expertise
- ✅ State object and SBT knowledge

**Critical Gaps:**
- ❌ NO MCP usage examples in prompt
- ❌ NO web search strategy
- ❌ Outdated for RTXDI (no mention of light grid, ReGIR)
- ❌ No HLSL raytracing intrinsic expertise

**Modernization Needed:**
- **MAJOR UPGRADE TO v4** - RTXDI specialist
- Add comprehensive MCP strategy (all 7 tools)
- Web search for RTXDI integration guides, NVIDIA samples
- Add light grid management expertise
- Add ReGIR (Reservoir-based Grid Importance Resampling) knowledge

#### rt-ml-technique-researcher-v2.md
**Current Strengths:**
- ✅ Created 53 research documents
- ✅ Excellent research methodology
- ✅ Good documentation structure

**Critical Gaps:**
- ❌ NO MCP integration mentioned
- ❌ Gives up on searches too easily (1-2 tries)
- ❌ Doesn't leverage DX12 MCP for implementation details
- ❌ No connection between research findings and MCP API docs

**Modernization Needed:**
- Add persistent MCP strategy: **Never give up after 1-2 searches**
- Add workflow: Research paper → Find APIs in MCP → Implementation guide
- Use `search_hlsl_intrinsics` to find shader functions for techniques
- Use `search_dxr_api` to map research concepts to DXR APIs
- Web search for latest 2025 research (SIGGRAPH 2025 preview)

---

## MCP Integration Strategy (CRITICAL)

### The Problem
**Current behavior:** Agents try MCP once or twice, get no results, give up.

**Why this fails:**
- MCP has 90+ entities, 30+ HLSL intrinsics
- Requires MULTIPLE search strategies
- Different search terms yield different results
- Need to try BROAD → SPECIFIC queries

### The Solution: Persistent MCP Query Strategy

**Mandatory for ALL v4 agents:**

```markdown
## MCP Search Protocol (MANDATORY)

You have access to the DX12 Enhanced MCP Server with 7 search tools.
**CRITICAL:** ALWAYS consult MCP before web searches. NEVER give up after 1-2 tries.

### MCP Tool Priority Order

1. **Start Broad**: `search_all_sources(query="concept")` - Cast wide net
2. **Narrow Down**: `search_dxr_api(query="specific term")` - DXR focus
3. **Check HLSL**: `search_hlsl_intrinsics(query="function")` - Shader functions
4. **Filter by Stage**: `search_by_shader_stage(stage="closesthit")` - Shader-specific
5. **Get Details**: `get_dx12_entity(name="TraceRay")` - Deep dive
6. **Check Core D3D12**: `search_dx12_api(query="resource")` - Core APIs
7. **Database Stats**: `dx12_quick_reference()` - Verify MCP is alive

### Example: Finding Shadow Ray Functions

**WRONG approach (gives up too early):**
```
Try: search_dxr_api("shadow")
Result: No exact matches
Conclusion: MCP doesn't have shadow info ❌ WRONG!
```

**CORRECT approach (persistent):**
```
Try 1: search_all_sources("shadow")
→ Finds: D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE

Try 2: search_hlsl_intrinsics("ray")
→ Finds: TraceRay, RayQuery::Proceed

Try 3: search_by_shader_stage("closesthit")
→ Finds: All closesthit-compatible intrinsics

Try 4: get_dx12_entity("TraceRay")
→ Full signature, parameters, usage notes

Try 5: search_dxr_api("RayFlags")
→ Finds: RAY_FLAG_SKIP_CLOSEST_HIT_SHADER (shadow rays!)

SUCCESS: Found 5+ relevant shadow ray APIs ✅
```

### Persistent Search Strategy

**Rule 1:** Try at LEAST 3 different search terms before giving up
**Rule 2:** Try at LEAST 3 different MCP tools before giving up
**Rule 3:** If 0 results, try BROADER search (not narrower)
**Rule 4:** Check `dx12_quick_reference()` to see what's available

### Common Search Patterns

**For Shadow Techniques:**
- "shadow", "ray flags", "occlusion", "visibility", "anyhit"
- `search_by_shader_stage("anyhit")` for shadow shader intrinsics
- `search_hlsl_intrinsics("Accept")` for AcceptHitAndEndSearch

**For RTXDI:**
- "state object", "shader table", "indirect", "dispatch rays"
- `search_dxr_api("D3D12_STATE_OBJECT")` for SBT setup
- `get_dx12_entity("DispatchRays")` for indirect lighting

**For Volumetric RT:**
- "procedural", "AABB", "intersection", "RayQuery"
- `search_by_shader_stage("intersection")` for custom intersection
- `search_hlsl_intrinsics("ReportHit")` for procedural primitives

### MCP Query Log

ALWAYS log your MCP queries in your analysis:
```
🔍 MCP Search Log:
1. search_all_sources("RTXDI") → 0 results (expected, RTXDI is NVIDIA library)
2. search_dxr_api("state object") → 3 results (D3D12_STATE_OBJECT_DESC, etc.)
3. get_dx12_entity("D3D12_STATE_OBJECT_DESC") → Full details ✅
4. search_hlsl_intrinsics("shader") → 12 results
5. search_by_shader_stage("callable") → 8 intrinsics available

Conclusion: MCP has all state object APIs needed for RTXDI integration.
```

This demonstrates persistence and thoroughness to the user.
```

---

## v4 Agent Roster (Final)

### New v4 Agents (2 new)

1. **dxr-rt-shadow-engineer-v4.md** ⭐ NEW
   - Shadow ray techniques specialist
   - PCSS, contact hardening, temporal filtering
   - MCP expert: shadow-related HLSL intrinsics
   - Web search: Latest shadow techniques (2024-2025)

2. **rtxdi-integration-specialist-v4.md** ⭐ NEW
   - RTXDI SDK integration expert
   - Light grid, ReGIR, reservoir management
   - MCP expert: State objects, SBT, DispatchRays
   - Web search: NVIDIA RTXDI docs, samples

### Upgraded to v4 (6 upgraded)

3. **buffer-validator-v4.md** (from v3)
   - Add MCP strategy for D3D12 buffer validation
   - Add RTXDI buffer validation (light grid, reservoirs)
   - Add web search for latest PIX/NSight validation techniques

4. **pix-debugger-v4.md** (from v3)
   - Add MCP strategy for shader debugging
   - Add RTXDI debugging expertise
   - Add shadow ray debugging section
   - Add web search for latest DXR debugging workflows

5. **stress-tester-v4.md** (from v3)
   - Add RTXDI stress tests
   - Add shadow ray scaling tests (1-64 rays)
   - Add MCP strategy for profiling APIs
   - Add web search for latest GPU stress testing

6. **performance-analyzer-v4.md** (from v3)
   - Add RTXDI performance metrics
   - Add shadow ray performance analysis
   - Add MCP strategy for async compute patterns
   - Add web search for latest DXR optimization (2025)

7. **dxr-systems-engineer-v4.md** (from v2)
   - Complete rewrite for RTXDI focus
   - Add comprehensive MCP strategy (all 7 tools)
   - Add light grid management expertise
   - Add web search for RTXDI integration guides

8. **rt-ml-technique-researcher-v4.md** (from v2)
   - Add persistent MCP strategy (NEVER give up)
   - Add research → MCP → implementation workflow
   - Add web search for SIGGRAPH 2025 preview
   - Add RTXDI research focus

### Kept as-is (2 unchanged)

9. **hlsl-volumetric-implementation-engineer-v2.md**
   - Already excellent, just add MCP strategy
   - Upgrade to v4 later if needed

10. **physics-performance-agent-v2.md**
    - Not RT-focused, low priority for MCP
    - Upgrade to v4 later if needed

---

## Implementation Plan

### Phase 1: Create New Agents (HIGH PRIORITY)
**Time:** 2-3 hours

1. **dxr-rt-shadow-engineer-v4.md**
   - Research latest shadow techniques (PCSS variants)
   - Define MCP search strategies for shadow intrinsics
   - Create comprehensive shadow ray upgrade guide

2. **rtxdi-integration-specialist-v4.md**
   - Research NVIDIA RTXDI SDK structure
   - Define MCP strategies for state objects, SBT
   - Create PlasmaDX-specific integration roadmap

### Phase 2: Modernize v3 Production Agents (HIGH PRIORITY)
**Time:** 3-4 hours

1. **buffer-validator-v4.md**
   - Add "MCP Search Protocol" section
   - Add RTXDI buffer validation rules
   - Add web search strategy

2. **pix-debugger-v4.md**
   - Add "MCP Search Protocol" section
   - Add RTXDI debugging section
   - Add shadow ray debugging section

3. **stress-tester-v4.md**
   - Add RTXDI test scenarios
   - Add shadow ray scaling tests
   - Add MCP profiling API strategy

4. **performance-analyzer-v4.md**
   - Add RTXDI performance metrics
   - Add shadow ray performance section
   - Add MCP async compute strategy

### Phase 3: Modernize v2 Specialized Agents (MEDIUM PRIORITY)
**Time:** 2-3 hours

1. **dxr-systems-engineer-v4.md**
   - Major rewrite for RTXDI
   - Add comprehensive MCP section (all 7 tools)
   - Add NVIDIA RTXDI integration workflow

2. **rt-ml-technique-researcher-v4.md**
   - Add persistent MCP strategy (key improvement)
   - Add research → MCP → implementation loop
   - Add RTXDI research focus

### Phase 4: Testing & Validation (CRITICAL)
**Time:** 2 hours

1. **Test MCP integration**
   - Invoke each agent with MCP-related task
   - Verify they use MULTIPLE MCP tools
   - Verify they DON'T give up after 1-2 tries

2. **Test web search integration**
   - Verify agents search for 2024-2025 content
   - Verify RTXDI-specific searches work
   - Verify shadow technique searches work

3. **Test multi-agent workflows**
   - Shadow engineer → PIX debugger → Stress tester
   - RTXDI specialist → Systems engineer → Performance analyzer

### Phase 5: Documentation Update
**Time:** 1 hour

1. **Update .claude/agents/README.md**
   - Add v4 section
   - Document MCP Search Protocol
   - Update agent invocation examples

2. **Update CLAUDE_CODE_PLUGINS_GUIDE.md**
   - Add v4 agents
   - Add MCP integration examples
   - Update multi-agent workflows

---

## Success Criteria

### For Each v4 Agent

**MCP Integration:**
- ✅ Contains "MCP Search Protocol (MANDATORY)" section
- ✅ Uses AT LEAST 3 different MCP tools in examples
- ✅ Shows persistent search strategy (not giving up)
- ✅ Logs MCP queries in analysis

**Web Search Integration:**
- ✅ Searches for 2024-2025 content
- ✅ Uses NVIDIA docs, SIGGRAPH papers, AMD GPUOpen
- ✅ Provides links to sources

**Expertise:**
- ✅ Deep knowledge of assigned domain (shadows, RTXDI, etc.)
- ✅ Provides file:line fixes (not vague suggestions)
- ✅ Includes time estimates and risk assessment

### For Overall Platoon

**Coverage:**
- ✅ No critical gaps (shadows ✅, RTXDI ✅)
- ✅ All RT pipeline stages covered
- ✅ Performance, debugging, testing all covered

**Integration:**
- ✅ Agents reference each other appropriately
- ✅ Multi-agent workflows documented
- ✅ Clear specialization hierarchy

**Usability:**
- ✅ User can invoke any agent by problem domain
- ✅ Agents provide actionable next steps
- ✅ Documentation is comprehensive and searchable

---

## Timeline

**Total Time:** 8-12 hours (can be split across multiple sessions)

- **Phase 1 (New Agents):** 2-3 hours
- **Phase 2 (v3 Modernization):** 3-4 hours
- **Phase 3 (v2 Modernization):** 2-3 hours
- **Phase 4 (Testing):** 2 hours
- **Phase 5 (Documentation):** 1 hour

**Recommended Schedule:**
- Session 1 (Tonight): Phase 1 - Create shadow engineer & RTXDI specialist
- Session 2 (Tomorrow): Phase 2 - Modernize v3 production agents
- Session 3 (Day 3): Phase 3 - Modernize v2 agents + Testing
- Session 4 (Day 4): Phase 4-5 - Final testing and documentation

---

## Next Immediate Steps

1. **Review this plan** - Confirm approach and priorities
2. **Start with Phase 1** - Create two new critical agents:
   - dxr-rt-shadow-engineer-v4.md
   - rtxdi-integration-specialist-v4.md
3. **Test new agents** - Invoke them with real PlasmaDX tasks
4. **Iterate** - Refine based on real-world usage

---

**Document Version:** 1.0
**Status:** READY FOR IMPLEMENTATION
**Priority:** HIGH (blocks RTXDI Phase 4)
