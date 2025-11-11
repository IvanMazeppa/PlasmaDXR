# 3D Gaussian System Integration Guide

Complete guide for using the **3d-gaussian-volumetric-engineer** agent with the **gaussian-analyzer** MCP server to extend PlasmaDX's particle system beyond plasma-only rendering.

## Quick Start (5 Minutes)

### 1. Setup MCP Server

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/gaussian-analyzer

# Install dependencies
./run_server.sh  # Will auto-create venv and install packages

# Verify server starts (Ctrl+C to exit)
# Should see MCP protocol messages, no errors
```

### 2. Configure Claude Code

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "gaussian-analyzer": {
      "command": "bash",
      "args": ["-c", "cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/gaussian-analyzer && ./run_server.sh"]
    }
  }
}
```

### 3. Launch Agent

```bash
# In Claude Code terminal or chat
/agent 3d-gaussian-volumetric-engineer
```

### 4. First Analysis

```
User: "Analyze the current Gaussian particle structure and propose extensions for supporting diverse celestial bodies (stars, gas clouds, dust)"

Agent: Will automatically:
1. Read current ParticleSystem.h/cpp and shader files
2. Call analyze_gaussian_parameters (MCP tool)
3. Present comprehensive analysis with recommendations
4. Propose 48-byte structure with 8 material types
```

---

## Complete Workflow Examples

### Example 1: Adding Gas Cloud Material

**Goal:** Extend particles to support wispy gas clouds with backward scattering

**User Request:**
```
"I want to add support for wispy gas cloud particles with low opacity and backward scattering. Analyze the feasibility, estimate performance impact, and propose implementation."
```

**Agent Workflow:**

**Phase 1: Analysis**
```markdown
Agent calls: analyze_gaussian_parameters(analysis_depth="detailed", focus_area="materials")

Result:
- Current: 32 bytes (PLASMA only)
- Missing: albedo, materialType, per-type scattering
- Recommendation: 48-byte structure with constant buffer materials
```

**Phase 2: Simulation**
```markdown
Agent calls: simulate_material_properties(
    material_type="GAS_CLOUD",
    properties={
        "opacity": 0.3,
        "scattering_coefficient": 1.5,
        "emission_multiplier": 0.1,
        "albedo_rgb": [0.6, 0.7, 0.9],
        "phase_function_g": -0.3  # Backward scattering
    }
)

Result:
- Visual: Wispy, diffuse edges, glow around backlighting
- Performance: ~108 FPS @ 10K particles (-10% from baseline)
- Shader: Needs albedo lookup + phase function modification
```

**Phase 3: Performance Validation**
```markdown
Agent calls: estimate_performance_impact(
    particle_struct_bytes=48,
    material_types_count=5,
    shader_complexity="moderate"
)

Result:
- Memory: +50% (3.1 MB → 4.6 MB @ 100K particles)
- Shader: +12 ALU ops per intersection
- Total impact: ~12% FPS reduction
- Verdict: ✅ Acceptable (stays within 90-120 FPS target @ 10K)
```

**Phase 4: Implementation Proposal**
```cpp
// Agent proposes this struct modification
struct ParticleData {
    XMFLOAT3 position;      // 12 bytes (legacy)
    XMFLOAT3 velocity;      // 12 bytes (legacy)
    float temperature;      // 4 bytes (legacy)
    float radius;           // 4 bytes (legacy)
    XMFLOAT3 albedo;        // 12 bytes (NEW)
    uint32_t materialType;  // 4 bytes (NEW) - GAS_CLOUD = 3
};  // Total: 48 bytes (16-byte aligned ✅)

// Agent proposes shader modifications
// In particle_gaussian_raytrace.hlsl:
MaterialProperties matProps = g_materialProperties[particle.materialType];
float3 scatteredColor = particle.albedo * matProps.scattering_coeff;
float phaseValue = HenyeyGreenstein(matProps.phase_g, cosTheta);
```

**Phase 5: Validation**
```markdown
Agent calls: validate_particle_struct(
    struct_definition=<proposed struct>,
    check_backward_compatibility=True
)

Result:
✅ 16-byte aligned
✅ Backward compatible (legacy fields in same positions)
✅ No critical issues
Recommendation: Proceed with implementation
```

**Agent Final Output:**
```markdown
## Implementation Plan

1. Modify ParticleSystem.h (add albedo + materialType)
2. Create particle_types.hlsl (MaterialProperties constant buffer)
3. Update particle_gaussian_raytrace.hlsl (material lookup + albedo application)
4. Build and test with GAS_CLOUD type
5. Capture screenshots and use rtxdi-quality-analyzer for ML comparison
6. Validate FPS meets target (108+ FPS @ 10K particles)

Estimated effort: 2-3 days
Risk: Low (backward compatible, moderate performance impact)
```

---

### Example 2: Comparing Volumetric vs Hybrid Rendering

**Goal:** Decide whether to implement hybrid surface/volume rendering for rocky bodies

**User Request:**
```
"Compare pure volumetric Gaussian rendering vs hybrid surface/volume approach. I want to understand the trade-offs before committing to an implementation."
```

**Agent Workflow:**

**Phase 1: Technique Comparison**
```markdown
Agent calls: compare_rendering_techniques(
    techniques=["pure_volumetric_gaussian", "hybrid_surface_volume", "adaptive_lod"],
    criteria=["performance", "visual_quality", "implementation_complexity", "material_flexibility"]
)

Result table:
| Technique | Performance | Quality | Complexity | Flexibility | Total |
|-----------|-------------|---------|------------|-------------|-------|
| Pure Volumetric | 8/10 (108 FPS) | 9/10 | 9/10 | 8/10 | 34/40 |
| Hybrid Surface/Volume | 6/10 (90 FPS) | 10/10 | 5/10 | 10/10 | 31/40 |
| Adaptive LOD | 9/10 (115 FPS) | 8/10 | 6/10 | 9/10 | 32/40 |
```

**Phase 2: Material Simulation (Both Approaches)**
```markdown
# Pure volumetric for "rocky" particles
Agent calls: simulate_material_properties(
    material_type="ROCKY_BODY",
    render_mode="volumetric_only"
)

Result:
- Visual: Can approximate rocky appearance with low emission + high albedo
- Limitation: No true surface reflection (all scattering-based)
- Performance: Good (standard volumetric pipeline)

# Hybrid approach
Agent calls: simulate_material_properties(
    material_type="ROCKY_BODY",
    render_mode="hybrid_surface_volume"
)

Result:
- Visual: True surface reflection + subsurface scattering
- Benefit: Physically accurate for solid bodies
- Cost: +15 ALU ops (normal approximation + BRDF)
```

**Phase 3: Performance Analysis**
```markdown
Agent calls: estimate_performance_impact(
    particle_struct_bytes=64,  # Hybrid needs more fields
    shader_complexity="complex"
)

Result:
- Hybrid impact: ~25% FPS reduction (120 → 90 FPS @ 10K)
- Still meets target: ✅ (90 FPS > 90 FPS minimum)
- But: Less headroom for 100K particles
```

**Agent Recommendation:**
```markdown
## Recommended Approach: Phased Implementation

**Phase 1 (Now):** Pure Volumetric (48 bytes)
- Implement 8 material types with volumetric rendering
- Covers 80% of use cases (stars, gas, nebulae)
- Fast to implement (1-2 weeks)
- Low risk (~10% FPS impact)

**Phase 2 (Later, if needed):** Hybrid Rendering (64 bytes)
- Add hybrid mode for rocky/icy bodies
- Only use hybrid when materialType requires it
- 4-6 weeks implementation
- Requires extensive testing

**Reasoning:**
- Most celestial bodies are volumetric (stars, gas, nebulae)
- Rocky/icy bodies are minority use case
- Can approximate rocky appearance with volumetric (good enough for now)
- Defer complexity until proven necessary
```

---

## Tool Usage Patterns

### Pattern 1: Exploratory Analysis

Use when you don't know what's possible with Gaussian particles.

```markdown
User: "What can 3D Gaussian particles actually do in terms of material properties?"

Agent workflow:
1. analyze_gaussian_parameters (comprehensive)
   → Understand current capabilities and limitations

2. compare_rendering_techniques (all techniques)
   → See full landscape of options

3. simulate_material_properties (multiple materials)
   → Test GAS_CLOUD, STAR, DUST with different properties
   → Understand visual appearance range

4. Report findings with examples and screenshots
```

### Pattern 2: Validation & Optimization

Use when you have a proposal and need validation.

```markdown
User: "I want to implement a 56-byte particle structure with 12 material types. Is this feasible?"

Agent workflow:
1. validate_particle_struct (check GPU alignment)
   → 56 bytes not 16-byte aligned ❌
   → Recommend 64 bytes with padding

2. estimate_performance_impact (64 bytes, 12 types)
   → ~20% FPS reduction
   → Still acceptable but close to limit

3. Recommendation: Reduce to 48 bytes + 8 types
   → Better performance, still covers all use cases
```

### Pattern 3: Implementation Planning

Use when ready to implement and need step-by-step plan.

```markdown
User: "Create a complete implementation plan for adding 5 material types"

Agent workflow:
1. analyze_gaussian_parameters (detailed)
   → Identify files to modify

2. simulate_material_properties (each of 5 types)
   → Determine material property values

3. validate_particle_struct (proposed 48-byte struct)
   → Ensure GPU compatibility

4. estimate_performance_impact (full analysis)
   → Confirm acceptable performance

5. Generate step-by-step implementation plan with code snippets
```

---

## Integration with Other Tools

### Working with RTXDI Quality Analyzer

**Before implementing changes:**
```markdown
# Capture baseline screenshot
[In PlasmaDX: Press F2]

# Use Gaussian Analyzer to design changes
/agent 3d-gaussian-volumetric-engineer
"Design material system for gas clouds"
[Agent uses gaussian-analyzer MCP tools]

# Implement proposed changes
[Modify code based on agent's plan]

# Build and capture new screenshot
[Press F2 again]

# Compare with ML
mcp__rtxdi-quality-analyzer__compare_screenshots_ml(
    before_path="screenshots/baseline.bmp",
    after_path="screenshots/gas_cloud_test.bmp"
)

# If LPIPS score shows significant difference:
mcp__rtxdi-quality-analyzer__assess_visual_quality(
    screenshot_path="screenshots/gas_cloud_test.bmp"
)
```

### Working with PIX Debugging Agent

**For performance validation:**
```markdown
# After implementing material changes

1. Use Gaussian Analyzer to estimate performance
   estimate_performance_impact(48 bytes, moderate complexity)
   Estimate: ~108 FPS @ 10K

2. Run actual build and capture PIX trace
   [Build Debug build, run with --pix-capture]

3. Use PIX Debugging Agent to analyze
   /agent pix-debugging-agent
   "Analyze latest PIX capture for material lookup overhead"

4. Compare actual vs estimated
   Estimated: 108 FPS
   Actual: 112 FPS ✅ (better than expected!)
```

---

## Common Workflows

### Workflow A: Research → Design → Implement

```markdown
Day 1: Research Phase
- Launch 3d-gaussian-volumetric-engineer agent
- Request: "Research what material properties are possible with Gaussian particles"
- Agent uses WebSearch + analyze_gaussian_parameters
- Output: Research report with citations and feasibility analysis

Day 2-3: Design Phase
- Request: "Design a material type system with 8 types"
- Agent uses simulate_material_properties for each type
- Agent uses compare_rendering_techniques to validate approach
- Output: Material type specification with properties and shader modifications

Day 4-5: Validation Phase
- Request: "Validate the proposed design and estimate performance"
- Agent uses validate_particle_struct + estimate_performance_impact
- Output: Validation report + performance projections

Day 6-10: Implementation Phase
- Follow agent's step-by-step implementation plan
- Build incrementally, test after each file modification
- Use rtxdi-quality-analyzer for visual validation

Day 11-12: Optimization Phase
- Request: "Optimize the material system for better performance"
- Agent analyzes bottlenecks, proposes optimizations
- Implement optimizations, validate with PIX captures
```

### Workflow B: Quick Prototype → Iterate

```markdown
Hour 1: Quick Analysis
- analyze_gaussian_parameters (quick mode)
- "Add albedo + materialType fields (48 bytes)"

Hour 2-4: Minimal Implementation
- Modify ParticleSystem.h (add 2 fields)
- Create simple material constant buffer
- Add basic material lookup in shader

Hour 5: Test & Validate
- Build, run, capture screenshots
- Use rtxdi-quality-analyzer ML comparison
- If good → expand to more materials
- If bad → iterate with agent's help
```

---

## Troubleshooting

### "MCP tool not found" errors

```bash
# Check MCP server is configured
cat .claude/settings.json | grep gaussian-analyzer

# Test server manually
cd agents/gaussian-analyzer
./run_server.sh
# Should start without errors
```

### Agent not using tools automatically

```markdown
# Be explicit in requests
❌ "I want to add materials"
✅ "Analyze the current particle structure and propose extensions for materials"

# Agent needs context to know when to use tools
✅ "Use the gaussian-analyzer tools to design a material system"
```

### Tool returns "file not found"

```bash
# Verify PROJECT_ROOT in .env
cat agents/gaussian-analyzer/.env

# Should point to PlasmaDX-Clean root
PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
```

### Performance estimates don't match reality

```markdown
# Estimates are based on assumptions
# Always validate with real measurements:

1. Use estimate_performance_impact (prediction)
2. Implement changes
3. Use rtxdi-quality-analyzer compare_performance (actual)
4. If mismatch > 20%, investigate with PIX captures
```

---

## Best Practices

### 1. Start with Analysis Phase

Don't jump straight to implementation. Use `analyze_gaussian_parameters` first to understand the current system.

### 2. Simulate Before Implementing

Use `simulate_material_properties` to test visual appearance before writing code.

### 3. Validate Structures Early

Use `validate_particle_struct` to catch alignment issues before spending time on implementation.

### 4. Compare Techniques Before Committing

Use `compare_rendering_techniques` to understand trade-offs between approaches.

### 5. Collaborate with Quality Analyzer

Use gaussian-analyzer for **design**, use rtxdi-quality-analyzer for **validation**.

### 6. Iterate Based on Real Data

Estimates are useful for planning, but always validate with actual measurements and PIX captures.

---

## Next Steps

After setup and first analysis:

1. **Review agent's recommendations** (read full analysis output)
2. **Ask clarifying questions** ("Why 48 bytes instead of 64?")
3. **Request specific simulations** ("Test GAS_CLOUD with different phase functions")
4. **Get implementation plan** ("Create step-by-step plan for 5 material types")
5. **Implement incrementally** (one file at a time, test after each change)
6. **Validate with ML tools** (rtxdi-quality-analyzer screenshot comparison)
7. **Measure real performance** (PIX captures, FPS logs)
8. **Iterate and optimize** (use agent's optimization recommendations)

---

## Support & Resources

- **Agent definition**: `.claude/agents/3d-gaussian-volumetric-engineer.md`
- **MCP server**: `agents/gaussian-analyzer/`
- **Main project docs**: `CLAUDE.md`, `MASTER_ROADMAP_V2.md`
- **Related agents**:
  - `celestial-rendering-specialist` (implementation specialist)
  - `pix-debugging-agent` (performance debugging)
  - `rtxdi-integration-specialist-v4` (RTXDI expertise)

Good luck extending your Gaussian particle system! 🚀
