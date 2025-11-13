# Gaussian-Analyzer Agent - Quick Start Guide

## What This Agent Does

The **3D Gaussian Volumetric Engineer** agent transforms PlasmaDX-Clean's single-material (plasma-only) particle system into a heterogeneous celestial body rendering system. It analyzes, designs, and validates material systems for diverse astrophysical phenomena (stars, gas clouds, dust, supernovae, etc.) while maintaining 90-120 FPS performance targets.

## Prerequisites

### 1. MCP Server Running
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/gaussian-analyzer
source venv/bin/activate
python3 gaussian_server.py
```

### 2. Claude Agent SDK Configuration
Ensure MCP server is registered in your Claude settings:
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

## Using the Agent (Claude Agent SDK)

### Method 1: Direct Agent SDK Launch

1. **Load the prompt:**
   - Open `AGENT_PROMPT.md` in your Claude Agent SDK interface
   - Copy the entire prompt content
   - Paste into the system prompt field

2. **Start conversation:**
   ```
   "Analyze the current 3D Gaussian particle structure and propose a material system for supporting gas clouds, different star types, and dust particles."
   ```

3. **Agent workflow:**
   - Calls `mcp__gaussian-analyzer__analyze_gaussian_parameters` (comprehensive analysis)
   - Calls `mcp__gaussian-analyzer__simulate_material_properties` (for each material type)
   - Calls `mcp__gaussian-analyzer__estimate_performance_impact` (48-byte structure)
   - Calls `mcp__gaussian-analyzer__validate_particle_struct` (GPU alignment check)
   - Provides complete architectural proposal with implementation roadmap

### Method 2: Interactive Q&A

**Example conversation flow:**

**You:** "What's missing from the current particle system for rendering nebulae?"

**Agent:**
- Runs `analyze_gaussian_parameters` (quick, materials focus)
- Reports: "Missing albedo RGB (scattering color), material type field, per-material opacity/scattering coefficients"
- Proposes: "Nebulae need backward scattering (g=-0.3), low opacity (0.2-0.4), blue/purple albedo"

**You:** "Show me the performance impact of adding those properties."

**Agent:**
- Runs `estimate_performance_impact` (48-byte struct, 8 materials, moderate complexity)
- Reports: "48 bytes × 10K = 480KB (+50% memory), material lookup adds 3 ALU ops per ray, estimated FPS: 112-115 (from 120 baseline)"

**You:** "Validate the struct for GPU compatibility."

**Agent:**
- Runs `validate_particle_struct` with proposed struct
- Reports: "PASS - 48 bytes = 3×16 alignment, fields correctly aligned, backward compatibility requires migration"

**You:** "Compare pure volumetric vs hybrid rendering approaches."

**Agent:**
- Runs `compare_rendering_techniques` (volumetric, hybrid, adaptive LOD)
- Reports: "Pure volumetric: Best for Phase 1 (90-120 FPS, excellent material flexibility), Hybrid: Phase 3 optimization (110-140 FPS, surface detail for rocky bodies)"

## Common Request Patterns

### 1. Initial Material System Design
```
"Design a material system for PlasmaDX that supports main sequence stars,
giant stars, gas clouds, dust particles, and the existing plasma. Maintain
90-120 FPS @ 10K particles."
```

**Expected output:**
- Comprehensive current state analysis
- 48-byte particle structure proposal
- 5 material type definitions with properties
- HLSL shader modifications
- Performance estimates (FPS, memory)
- 3-phase implementation roadmap

### 2. Specific Material Exploration
```
"I want wispy blue gas clouds that scatter light backward. What material
properties should I use? Show me the HLSL code changes needed."
```

**Expected output:**
- `simulate_material_properties` for GAS_CLOUD
- Specific property values (opacity: 0.3, g: -0.3, albedo: [0.6, 0.7, 0.9])
- HLSL code snippet for scatter calculation
- Visual appearance prediction (wispy, diffuse, blue-tinted)
- Performance impact (2-3 extra ALU ops)

### 3. Performance Validation
```
"Will adding 8 material types with albedo RGB drop me below 90 FPS @ 10K
particles on RTX 4060 Ti?"
```

**Expected output:**
- `estimate_performance_impact` (48-byte, 8 materials, moderate)
- Detailed FPS estimates at 10K/50K/100K
- Bottleneck analysis (memory bandwidth vs ALU)
- Optimization recommendations if below 90 FPS
- Red/yellow/green performance verdict

### 4. Advanced Techniques Research
```
"Research shows 85% of Gaussians are mergeable in dense regions. Can we use
adaptive merging/coalescing to boost performance? What's the implementation
strategy?"
```

**Expected output:**
- Reference to BalanceGS research (Oct 2024)
- Similarity-based merging algorithm proposal
- HLSL compute shader for merging pass
- Performance estimate (+15-25% FPS in dense accretion disk)
- Phase 3 roadmap integration

### 5. Exotic Effects (Supernovae, Prominences)
```
"How would I implement supernova explosion particles with procedural noise-
driven emission and radial expansion? Show the particle type definition and
shader changes."
```

**Expected output:**
- `simulate_material_properties` for SUPERNOVA material
- Particle struct additions (explosion_time, expansion_rate)
- HLSL procedural noise function (Perlin/Simplex)
- Radial velocity calculation, temperature decay over time
- PyroFX research references (EmberGen, Houdini GPU Pyro)
- Performance warning if "complex" tier shader

## Integration with Other Agents

### Validation Workflow (with dxr-image-quality-analyst)

1. **Design Phase (You):**
   ```
   "Propose a material system for gas clouds and stars."
   ```
   - gaussian-analyzer agent designs 48-byte structure, provides HLSL code

2. **Implementation Phase:**
   - Implement proposed changes in C++ and HLSL
   - Build project, run PlasmaDX-Clean

3. **Validation Phase (Image Quality Analyst):**
   ```
   "Assess the visual quality of the new gas cloud particles. F2 to capture screenshot."
   ```
   - Press F2 to capture screenshot
   - dxr-image-quality-analyst uses AI vision to analyze volumetric depth, scattering quality, temperature gradient
   - Provides brutal honesty feedback: "Gas clouds lack volumetric depth - scattering coefficient too low (0.8, should be 1.5-2.0)"

4. **Iteration Phase (You):**
   ```
   "Image analyst says scattering coefficient should be 1.5-2.0. Re-simulate
   GAS_CLOUD material with scattering_coefficient: 1.8. What's the visual
   difference?"
   ```
   - gaussian-analyzer re-runs `simulate_material_properties`
   - Predicts improved volumetric depth, increased wispy appearance
   - Provides updated HLSL constant value

5. **Performance Validation (Image Quality Analyst):**
   ```
   "Check the performance impact of the changes. Analyze application logs."
   ```
   - dxr-image-quality-analyst parses logs, reports: "FPS: 114 (from 120, -5% acceptable)"

## Tool Usage Examples

### Tool 1: analyze_gaussian_parameters
```
Agent: "I'm calling analyze_gaussian_parameters with comprehensive depth
to understand the current particle structure limitations."

Tool call:
mcp__gaussian-analyzer__analyze_gaussian_parameters(
    analysis_depth="comprehensive",
    focus_area="all"
)

Output:
- Current 32-byte structure: position (12B), velocity (12B), temperature (4B), radius (4B)
- Missing for celestial diversity: material_type (4B), albedo_rgb (12B), scattering/absorption coefficients
- Shader analysis: No material branching, single opacity model (Beer-Lambert only)
- Performance baseline: 120 FPS @ 10K particles
- Recommendation: 48-byte structure with 8 material types, material lookup texture
```

### Tool 2: simulate_material_properties
```
Agent: "Testing GAS_CLOUD material with backward scattering for wispy appearance."

Tool call:
mcp__gaussian-analyzer__simulate_material_properties(
    material_type="GAS_CLOUD",
    properties={
        "opacity": 0.3,
        "scattering_coefficient": 1.5,
        "emission_multiplier": 0.1,
        "albedo_rgb": [0.6, 0.7, 0.9],
        "phase_function_g": -0.3
    },
    render_mode="volumetric_only"
)

Output:
- Visual prediction: Wispy, diffuse, blue-tinted clouds with soft edges
- Scattering dominates (coefficient 1.5 >> emission 0.1×)
- Backward phase (g=-0.3) creates glow-around-edges effect
- Shader ALU: +3 ops (material lookup + phase calculation)
- HLSL snippet: "float phase = HenyeyGreenstein(cosTheta, -0.3);"
```

### Tool 3: estimate_performance_impact
```
Agent: "Calculating FPS impact of 48-byte structure with 8 material types."

Tool call:
mcp__gaussian-analyzer__estimate_performance_impact(
    particle_struct_bytes=48,
    material_types_count=8,
    shader_complexity="moderate",
    particle_counts=[10000, 50000, 100000]
)

Output:
- Memory: 480KB (from 320KB, +50%)
- Shader overhead: Material lookup (2 ALU) + branch (1 ALU) = 3 ALU ops
- FPS estimates:
  - 10K particles: 112-115 FPS (from 120, -4% to -7%)
  - 50K particles: 58-62 FPS (from 65, -5% to -11%)
  - 100K particles: 28-31 FPS (from 32, -9% to -13%)
- Bottleneck: Memory bandwidth (48-byte fetch vs 32-byte)
- Verdict: GREEN for 10K target (>90 FPS maintained)
```

### Tool 4: compare_rendering_techniques
```
Agent: "Comparing pure volumetric vs hybrid surface-volume approaches."

Tool call:
mcp__gaussian-analyzer__compare_rendering_techniques(
    techniques=["pure_volumetric_gaussian", "hybrid_surface_volume"],
    criteria=["performance", "visual_quality", "material_flexibility"]
)

Output:
Comparison Matrix:
                           Pure Volumetric    Hybrid Surface-Volume
Performance (FPS)          112-115           110-140
Visual Quality             Excellent         Outstanding
Material Flexibility       Perfect           Good (surfaces need meshes)
Implementation Complexity  Low               High
Memory Usage               480KB             680KB (+ mesh data)

Recommendation: Pure volumetric for Phase 1 (simpler, faster to implement),
hybrid for Phase 3 (rocky planets need surface detail)
```

### Tool 5: validate_particle_struct
```
Agent: "Validating proposed 48-byte particle structure for GPU compatibility."

Tool call:
mcp__gaussian-analyzer__validate_particle_struct(
    struct_definition="""
    struct ParticleData {
        XMFLOAT3 position;      // 12 bytes
        XMFLOAT3 velocity;      // 12 bytes
        float temperature;      // 4 bytes
        float radius;           // 4 bytes
        XMFLOAT3 albedo;        // 12 bytes (NEW)
        uint32_t materialType;  // 4 bytes (NEW)
    };
    """,
    check_backward_compatibility=True,
    check_gpu_alignment=True
)

Output:
Field Analysis:
- position (XMFLOAT3): 12 bytes, offset 0, ✓ aligned
- velocity (XMFLOAT3): 12 bytes, offset 12, ✓ aligned
- temperature (float): 4 bytes, offset 24, ✓ aligned
- radius (float): 4 bytes, offset 28, ✓ aligned
- albedo (XMFLOAT3): 12 bytes, offset 32, ✓ aligned
- materialType (uint32_t): 4 bytes, offset 44, ✓ aligned
Total: 48 bytes

GPU Alignment: ✓ PASS (48 bytes = 3 × 16-byte alignment)
Backward Compatibility: ❌ FAIL (32-byte legacy data cannot auto-migrate)
  -> Mitigation: Provide migration script (zero-fill albedo, set materialType=PLASMA)

Final Verdict: PASS with migration required
```

## Troubleshooting

### "MCP server not responding"
```bash
# Check if server is running
ps aux | grep gaussian_server

# Restart server
cd agents/gaussian-analyzer
pkill -f gaussian_server.py
./run_server.sh
```

### "Tool returns empty results"
```bash
# Verify PROJECT_ROOT in .env
cat .env
# Should show: PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

# Check key files exist
ls src/particles/ParticleSystem.h
ls shaders/particles/particle_gaussian_raytrace.hlsl
```

### "Agent doesn't use tools"
- Ensure prompt explicitly states "You have access to 5 computational tools via gaussian-analyzer MCP server"
- Start requests with action verbs: "Analyze...", "Simulate...", "Validate...", "Compare..."
- Check MCP server logs for connection issues

### "Performance estimates seem wrong"
- Tool uses baseline FPS from CLAUDE.md (120 FPS @ 10K particles)
- Estimates are ±5% confidence without real PIX data
- Use dxr-image-quality-analyst for actual measured FPS validation

## Next Steps

1. **Start simple:** "Analyze the current particle structure and identify gaps"
2. **Iterate designs:** Use simulate_material_properties to experiment
3. **Validate performance:** Always run estimate_performance_impact before proposing
4. **Check GPU compatibility:** Run validate_particle_struct on final struct
5. **Implement changes:** Follow agent's HLSL snippets and struct definitions
6. **Validate visually:** Use dxr-image-quality-analyst with F2 screenshots
7. **Measure performance:** Use dxr-image-quality-analyst to parse application logs

**Ready to transform PlasmaDX into a heterogeneous celestial rendering powerhouse!**
