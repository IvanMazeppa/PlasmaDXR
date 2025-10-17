# Session Summary: Sphere Boundary Mystery Investigation

**Date:** 2025-10-16
**Status:** CRITICAL BLOCKER - Must fix before RTXDI
**Context at End:** 5%

---

## Summary

Multi-light system (Phase 3.5) is technically functional but has a **hard spherical boundary at ~300 units** where lights completely fail to illuminate particles beyond this radius, despite particles existing far beyond this limit.

**Critical Discovery:** This is NOT:
- ❌ Particle physics constraints (constraints are disabled: `m_constraintShape = 0`)
- ❌ Light position culling (lights work at 2000+ units distance inside the sphere)
- ❌ Simple attenuation (attenuation formula doesn't create hard cutoffs)
- ❌ Particle density falloff (thousands of particles visible beyond the boundary)

**What we know for certain:**
- ✅ Particles CAN and DO drift beyond 300 units (no snap-back, user confirms)
- ✅ Lights work perfectly at 2000 units distance - but ONLY for particles inside the sphere
- ✅ Particles outside sphere are clearly visible but receive NO green tint from green light
- ✅ The boundary is perfectly spherical, centered at origin

---

## Evidence (Screenshots Provided)

User provided 3 test screenshots with single green light at different positions:

1. **Position (20, 0, 0):** Bright green sphere, particles well-lit inside boundary
2. **Position (500, 0, 0):** Middle screenshot - green sphere still visible, clear boundary
3. **Position (2000, 0, 0):** Faint yellowish-green but STILL WORKS inside sphere

**Key Observation:** "particles are nearby but outside of the sphere and should be lighting up green but they aren't"

The light WORKS at extreme distances, but ONLY affects particles within the ~300-unit sphere from origin.

---

## Technical Investigation Results

### Physics System Analysis

**ParticleSystem.h:72**
```cpp
uint32_t m_constraintShape = 0;  // 0=NONE - constraints DISABLED
```

**ParticleSystem.cpp:216** (Hardcoded but unused)
```cpp
constants.constraintRadius = 50.0f;  // Not applied (shape=0)
```

**Conclusion:** Physics constraints are OFF. Particles freely drift to 600+ units.

### Acceleration Structure Analysis

**All particles included in BLAS:**
- `m_particleCount = 10000` (full count)
- `geomDesc.AABBs.AABBCount = m_particleCount` (line 374)
- AABB generation has NO distance filtering (verified in `generate_particle_aabbs.hlsl`)
- `ComputeGaussianAABB()` uses only `p.position ± maxRadius` (lines 73-83 in gaussian_common.hlsl)

**BLAS build process (RTLightingSystem_RayQuery.cpp:369-398):**
1. GenerateAABBs() - Dispatches for ALL 10,000 particles
2. BuildBLAS() - Uses `m_particleCount` AABBs
3. BuildTLAS() - Single instance, identity transform

**No obvious culling found in code.**

### Config File Check

**config.json line 54:**
```json
"outerRadius": 300.0
```

This defines SPAWN RANGE (10-300 units), not a hard boundary. User confirms particles regularly drift beyond this.

---

## The Mystery: Why 300?

The 300-unit sphere matches `outerRadius: 300.0`, but:
- Physics constraints are disabled
- No AABB filtering by distance
- No ray TMax limits at 300 units
- User observes particles at 400-600+ units routinely

**Hypothesis:** DXR acceleration structure builder may be computing conservative world-space bounds for TLAS, possibly using initial particle spawn distribution (10-300 units) and NOT updating for particles that drift beyond.

**Alternative Hypothesis:** Hidden DXR internal optimization culling primitives beyond a certain distance from TLAS center.

---

## Three Approaches to Solving This

### Approach 1: Diagnostic Logging (Fastest - 1 hour)

Add logging to verify BLAS/TLAS contains particles beyond 300 units:

**Files to modify:**
- `src/lighting/RTLightingSystem_RayQuery.cpp:350-367` (GenerateAABBs)
- `shaders/dxr/generate_particle_aabbs.hlsl:39` (AABB computation)

**What to log:**
1. Min/max particle positions in AABB generation
2. Number of particles with |position| > 300
3. BLAS bounds after build (if D3D12 exposes this)
4. Sample 10 AABBs from particles at different distances

**Implementation:**
```cpp
// In GenerateAABBs, after dispatch:
// Add buffer readback to check min/max AABB bounds
// Log: "AABB range: min=({},{},{}), max=({},{},{})"
// Log: "Particles beyond 300 units: {}/{}"
```

### Approach 2: PIX GPU Debugging (Most thorough - 2 hours)

Deploy PIX debugging agent with focus on acceleration structure analysis:

**Workflow:**
1. Build DebugPIX configuration
2. Run: `./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json`
3. Auto-capture at frame 120
4. Analyze capture:
   - Inspect BLAS/TLAS bounds in GPU memory
   - Trace shadow rays from light at (500, 0, 0)
   - Verify ray TMax values
   - Check if rays even START for distant particles

**PIX Agent Previous Success:**
- Found multi-light shader math bug (lerp 40x too weak) in 13 minutes
- Successfully analyzed buffer contents and shader execution

**Time Investment:** 20-minute time box for agent execution + analysis

### Approach 3: Claude Code Plugins (NEW - Experimental)

**Available plugins installed:**
- `feature-dev` - Guided feature development with codebase analysis
- `agent-sdk-dev` - SDK app development
- `pr-review-toolkit` - Code review agents

**Potential workflow:**
```
/feature-dev "Investigate DXR acceleration structure sphere boundary issue at 300 units"
```

The feature-dev plugin can:
- Analyze existing BLAS/TLAS implementation patterns
- Search for hidden distance limits or culling
- Propose architectural fixes
- Guide debugging strategy

**Documentation:** Read `CLAUDE_CODE_PLUGINS_GUIDE.md` for plugin usage

---

## Remaining Multi-Light Issues (Secondary Priority)

From `MULTI_LIGHT_FIXES_NEEDED.md`:

1. **Light radius control** - Shader doesn't use `light.radius` parameter (line 726)
2. **RT lighting toggle** - Can't fully disable particle-to-particle RT lighting
3. **Make RT lighting + 1 primary light default**

**Status:** System is WORKING, these are polish items. Fix sphere boundary first.

---

## Critical Decision Point

**Must fix sphere boundary before proceeding to RTXDI.** RTXDI relies on the same BLAS/TLAS infrastructure. If there's a fundamental ray traversal or bounds issue, RTXDI will inherit the bug.

**User's concern:** "i'm not sure what's happening, it's a bit disconcerting when the physics and lighting mechanics aren't fully understood."

**Recommendation:** Use Approach 2 (PIX agent) first, as it has proven track record and will definitively show:
- What's in the acceleration structure
- Where rays are actually going
- Whether the bug is in build or traversal

If PIX analysis is inconclusive, try Approach 3 (Claude Code plugins) for deeper codebase pattern analysis.

---

## Files Modified This Session

### Core Changes
- `src/core/Application.cpp:194-195` - Removed auto-init 13 lights (start with 0)
- `src/core/Application.cpp:1973` - Expanded light position range to ±2000
- `src/core/Application.cpp:1915-2050` - Added comprehensive multi-light ImGui panel

### Shader Fix
- `shaders/particles/particle_gaussian_raytrace.hlsl:747` - Fixed multi-light contribution (removed weak lerp)

**Critical lesson learned:** MSBuild doesn't always trigger shader recompilation! Always verify `.dxil` timestamp after shader changes.

### Log Spam Fix
- `src/particles/ParticleRenderer_Gaussian.cpp:485-490` - Throttled "Updated light buffer" log

---

## Next Session Instructions

### 1. Read These Documents First

**REQUIRED READING:**
```
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\CLAUDE_CODE_PLUGINS_GUIDE.md
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\MULTI_LIGHT_FIXES_NEEDED.md
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\CLAUDE.md
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\MASTER_ROADMAP_V2.md
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\ROADMAP_CELESTIAL_BODIES.md
D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\SHADOW_RTXDI_IMPLEMENTATION_ROADMAP.md
```

### 2. Context from Previous Session

User feedback: "clearly there's work to be done to improve the physical accuracy which improving the image quality"

**Known UI limitations:**
- Can't fully disable original RT lighting (only dim it)
- This interferes with multi-light testing
- RT lighting toggle needed (see MULTI_LIGHT_FIXES_NEEDED.md)

### 3. Decision Tree for Next Actions

**Option A: Deploy PIX Agent (Recommended)**
```bash
# Time box: 20 minutes
# Focus: Acceleration structure bounds analysis
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --config=configs/agents/pix_agent.json
```

**Option B: Use Claude Code Plugins**
```
/feature-dev "Analyze DXR BLAS/TLAS sphere boundary limitation at 300 units from origin"
```

**Option C: Manual Diagnostic Logging**
- Add AABB bounds verification to GenerateAABBs()
- Log particle positions beyond 300 units
- Verify shadow ray TMax values

### 4. Success Criteria

- [ ] Identify root cause of 300-unit sphere boundary
- [ ] Verify lights can illuminate particles at 400+ units
- [ ] No regression in lighting quality inside 300 units
- [ ] Document findings for RTXDI integration

### 5. After Sphere Boundary Fix

Apply the 3 remaining polish fixes from `MULTI_LIGHT_FIXES_NEEDED.md`:
1. Light radius shader implementation (5 min)
2. RT lighting toggle (15 min)
3. Investigate if outerRadius config should be increased

Then proceed with RTXDI roadmap.

---

## Technical Notes for Debugging

### Shadow Ray Code Path (Multi-Light)

**Shader:** `particle_gaussian_raytrace.hlsl:715-757`

```hlsl
for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
    Light light = g_lights[lightIdx];
    float3 lightDir = normalize(light.position - pos);
    float lightDist = length(light.position - pos);

    // Line 731: Shadow ray cast
    float shadowTerm = CastShadowRay(pos, lightDir, lightDist);
}
```

**CastShadowRay (line 119):**
```hlsl
float CastShadowRay(float3 origin, float3 direction, float maxDist) {
    shadowRay.TMax = maxDist;  // Uses full distance to light
    query.TraceRayInline(g_particleBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
}
```

**No hard 300-unit limit in shadow ray code.**

### Acceleration Structure Build Chain

1. **AABB Generation** (`generate_particle_aabbs.hlsl:39`)
   - Dispatches: `(10000 + 255) / 256 = 40` thread groups
   - Each thread computes 1 AABB
   - Output: 24 bytes per particle (minXYZ, maxXYZ)

2. **BLAS Build** (`RTLightingSystem_RayQuery.cpp:369`)
   - Input: All 10,000 AABBs
   - Geometry type: `D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS`
   - Flags: `D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE`

3. **TLAS Build** (`RTLightingSystem_RayQuery.cpp:405`)
   - Single instance
   - Identity transform (no translation/rotation/scale)
   - References entire BLAS

**Hypothesis:** D3D12 may compute TLAS world-space bounds conservatively, and the BVH internal nodes might have incorrect bounds for distant particles.

---

## User Quotes (Context for Next Session)

> "i've NEVER seen this snapping back behaviour. it's always been the case that they naturally drift out, and if i give them enough time they will drift a very long way away from the origin."

> "i'm not sure what happening, it's a bit disconcerting when the physics and lighting mechanics aren't fully understood."

> "no, there's definitely a sphere shaped boundary which the lights can't traverse. i've been setting the colour to green for a single light and tried to move it out to light up particles that have drifted, once it passes that bondary region it just disappears despite there being particles out there to illuminate."

> "do you think we should perform a debugging effort, perhaps deploy the agents - the PIX agent has proved to be invaluable with its unique abilities. the agent [...] let us to a series of breakthrough"

---

## Build Commands

**Compile shaders manually:**
```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
  -T cs_6_5 -E main \
  shaders/particles/particle_gaussian_raytrace.hlsl \
  -Fo shaders/particles/particle_gaussian_raytrace.dxil
```

**Build project:**
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /nologo /v:m
```

**Build DebugPIX for agent:**
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" \
  PlasmaDX-Clean.sln /p:Configuration=DebugPIX /p:Platform=x64 /nologo /v:m
```

---

## Final Status

**Multi-Light System:** ✅ WORKING (with sphere boundary limitation)
**Physics System:** ✅ Particles drift freely
**Sphere Boundary:** ❌ BLOCKING ISSUE - must fix before RTXDI
**Secondary Polish Items:** 📝 Documented in MULTI_LIGHT_FIXES_NEEDED.md

**Recommended First Action:** Deploy PIX agent with 20-minute time box to analyze acceleration structure bounds.

---

**Session ended at 5% context.**
**Next session: Start with PIX agent or Claude Code plugins to solve sphere boundary mystery.**
