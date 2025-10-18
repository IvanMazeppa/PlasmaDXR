# RTXDI Integration - Milestone 1 Complete ✅

**Date**: 2025-10-18
**Status**: SDK Linked, Infrastructure Created, Ready for Week 1 Day 2

---

## What Was Accomplished

### ✅ Milestone 1: RTXDI SDK Linked (6 hours estimated, completed in session)

**Tasks Completed**:
1. ✅ Cloned NVIDIA RTXDI runtime library to `external/RTXDI-Runtime/`
2. ✅ CMake integration (add_subdirectory + include paths + linking)
3. ✅ Created `RTXDILightingSystem.h/cpp` skeleton class
4. ✅ Build verification: SUCCESS (RTXDI static library linked)
5. ✅ Validated PCSS and multi-light systems working correctly
6. ✅ Fixed multi-light initialization bug (13 lights loaded at startup)

**Files Added**:
- `external/RTXDI-Runtime/` - NVIDIA RTXDI runtime library
- `src/lighting/RTXDILightingSystem.h` - RTXDI wrapper class (header)
- `src/lighting/RTXDILightingSystem.cpp` - RTXDI wrapper class (implementation)

**Files Modified**:
- `CMakeLists.txt` - Added RTXDI subdirectory, include paths, and linking
- `src/core/Application.cpp` - Fixed multi-light initialization (line 195)

**Build Status**: ✅ SUCCESS

---

## Timeline: Milestone 1 → First RTXDI Visual Test

**User Question**: "After setting up the SDK how much more work is involved before we can test out RTXDI for the first time?"

**Answer**: **~32 hours of work** from Milestone 1 complete to first visual test.

---

### Milestone 2: Light Grid Construction (+10 hours from M1)

**What You'll See**: PIX capture shows RTXDI light grid filled with 13 lights

**Tasks**:
1. Create light grid buffer (GPU memory)
   - Size: 27,000 cells (30×30×30 grid for 600-unit world)
   - Cell size: 20 units
   - Max lights per cell: 16
   - Total memory: ~6 MB

2. Write light grid build compute shader
   - File: `shaders/rtxdi/light_grid_build_cs.hlsl`
   - Dispatch: One thread per cell (30×30×30 = 27K threads)
   - Populates cell with nearby lights (distance-based)

3. Upload lights to RTXDI format
   - Convert PlasmaDX lights → RTXDI light structure
   - Upload to GPU buffer each frame (or once if static)

4. Integrate with Application.cpp
   - Call `RTXDILightingSystem::UpdateLightGrid()` each frame
   - Pass current light array

**Deliverable**: PIX shows light grid buffer with 13 lights distributed across cells

**Test Command**:
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --rtxdi --dump-buffers 120
# Then inspect light_grid.bin in PIX/buffer_dumps/frame_120/
```

---

### Milestone 3: DXR Pipeline Setup (+14 hours from M2, 24 total)

**What You'll See**: DispatchRays() runs without crashing (black screen OK at this stage)

**Tasks**:
1. Create DXR state object (6 hours)
   - Raygen shader: `shaders/rtxdi/rtxdi_raygen.hlsl`
   - Miss shader: `shaders/rtxdi/rtxdi_miss.hlsl`
   - ClosestHit shader: `shaders/rtxdi/rtxdi_closesthit.hlsl`
   - Callable shader: `shaders/rtxdi/volumetric_brdf.hlsl` (Henyey-Greenstein)
   - State object type: `D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE`

2. Build Shader Binding Table (SBT) (4 hours)
   - Raygen record (1 entry)
   - Miss record (1 entry)
   - HitGroup record (1 entry for procedural Gaussians)
   - Callable record (1 entry for volumetric phase function)
   - SBT layout: RayGen | Miss | HitGroup | Callable

3. Write raygen shader stub (2 hours)
   - TraceRay() call (replaces RayQuery)
   - RTXDI sampling stub (returns dummy light for now)
   - Output to UAV

4. DispatchRays() integration (2 hours)
   - Replace current RayQuery dispatch with DispatchRays()
   - D3D12_DISPATCH_RAYS_DESC setup
   - Width × Height rays (1920 × 1080 = 2M rays)

**Deliverable**: DispatchRays() executes, raygen shader runs (output black for now)

**Major Change**: This switches from **RayQuery** (inline) → **TraceRay** (traditional DXR)
- Multi-light system stays RayQuery (no changes)
- RTXDI uses TraceRay (requires state object + SBT)

---

### Milestone 4: First Visual RTXDI Test (+8 hours from M3, **32 total**) 🎯

**What You'll See**: **WORKING RTXDI RENDERING!** Lights illuminating particles via importance sampling

**Tasks**:
1. Reservoir buffer setup (2 hours)
   - 2× buffers for ping-pong (current + previous)
   - Size: 1920 × 1080 × 64 bytes = 126 MB per buffer
   - Format: `RTXDI_DIReservoir` structure (SDK-defined)
   - UAV flags for read/write

2. Initial RTXDI sampling (4 hours)
   - Implement `RTXDI_SampleLights()` in raygen shader
   - Read light grid (cell lookup based on ray hit position)
   - Weighted reservoir sampling (importance = luminance × attenuation)
   - Update reservoir with selected light

3. Temporal reuse (1 hour)
   - Read previous frame reservoir
   - Validate (position delta < threshold)
   - Merge if valid (reservoir combine operation)
   - Write to current reservoir

4. Connect to Gaussian volumetric renderer (1 hour)
   - Output selected light to UAV
   - Volumetric ray marching reads RTXDI light samples
   - Apply same Henyey-Greenstein phase function
   - Use existing PCSS shadow system (no changes needed!)

**Deliverable**: **FIRST RTXDI VISUAL TEST**

**Expected Result**:
- RTXDI selects important lights (not all 13 every frame)
- Temporal reuse working (previous frame influence visible)
- Particles illuminated with smart light selection
- PCSS shadows working with RTXDI (same shadow system, different light source)

**Expected Performance**: ~100-110 FPS @ 10K particles
- RTXDI overhead: +3ms (ReSTIR sampling + temporal reuse)
- No spatial reuse yet (Week 3)
- Still 88% of >100 FPS target

**Test Command**:
```bash
./build/Debug/PlasmaDX-Clean.exe --rtxdi --physics=simple-orbit
# Use simple-orbit to avoid sphere clumping for lighting demo
```

**Visual Comparison**:
```bash
# Multi-light (all 13 lights, brute force)
./build/Debug/PlasmaDX-Clean.exe --multi-light

# RTXDI (smart selection, temporal reuse)
./build/Debug/PlasmaDX-Clean.exe --rtxdi
```

---

### Full Feature Complete (+64 hours from M4, 96 total)

**Week 3-4**: Add spatial reuse, visibility reuse, optimization

**Spatial Reuse** (Week 3):
- Merge with 4-8 neighbor reservoirs
- 1-2 spatial resampling passes
- +30-40% quality improvement
- -10% FPS cost

**Visibility Reuse** (Week 3):
- Cache shadow ray results
- Reuse cached visibility from neighbors
- Reduces shadow rays from M → 1 per pixel
- +15-20% FPS gain

**Optimization** (Week 4):
- Static light grid (rebuild only on light change): +5% FPS
- Reduce spatial passes if FPS drops below target
- Fine-tune reservoir parameters (M count, weight thresholds)

**Final Expected Performance**: 105-115 FPS @ 10K particles + 100 lights

---

## Parallel Track: Physics Modes

**Status**: Can be implemented in parallel with RTXDI Week 1-2

While RTXDI infrastructure is being built (Milestones 2-3, no visual changes yet), the physics mode system can be developed independently.

### Proposed Physics Mode Enum

```cpp
enum class PhysicsMode {
    FreeFloat,        // No forces, no constraints (best for lighting tests)
    SimpleOrbit,      // Keplerian orbits around origin (clean, no clumping)
    AccretionDisk,    // Current system (gravity + constraints + viscosity)
    GalaxyCollision,  // Dual mass centers
    NBodyGravity,     // Full N-body simulation (expensive)
    Custom            // User-defined via config
};
```

### Command-Line Support

```bash
--physics=free-float      # Particles float, no clumping
--physics=simple-orbit    # Clean Keplerian orbits (RECOMMENDED for RTXDI testing)
--physics=accretion-disk  # Current system
--physics=galaxy          # Dual galaxy collision
```

### Config Support

```json
{
  "physics": {
    "mode": "simple-orbit",
    "gravity": 1.0,
    "centralMass": 1000.0,
    "enableConstraints": false,
    "enableViscosity": false
  }
}
```

### Why This Helps RTXDI Testing

**Current Problem**: Schwarzschild black hole + spherical constraints → all particles clump in sphere → lighting can't reach interior

**Solution**: Use `--physics=simple-orbit` for RTXDI testing
- Particles orbit in clean rings (like Saturn's rings)
- No sphere clumping
- Lights can illuminate from all angles
- Temperature/density gradients visible
- Perfect for demonstrating multi-light vs RTXDI comparison

**Porting from PlasmaVulkan**:
The PlasmaVulkan project already has this refined physics system with multiple modes. Port the orbit logic and mode selection system to PlasmaDX.

**Estimated Time**: 8-10 hours
- Port physics mode enum: 1 hour
- Implement simple-orbit mode: 3 hours
- Add free-float mode: 2 hours
- Command-line + config integration: 2 hours
- Testing and tuning: 2 hours

**Benefit**: Clean lighting demos for Milestone 4 testing

---

## Current State Summary

### What's Working ✅

1. **PCSS Soft Shadows**: Complete, all 3 variants (Performance/Balanced/Quality)
2. **Multi-Light System**: Fixed initialization bug, 13 lights loaded at startup
3. **RTXDI SDK**: Linked and ready to use
4. **RTXDILightingSystem Class**: Skeleton created, build verified

### What's Next ⏳

**Immediate** (Next Session):
1. Milestone 2: Light grid construction (+10 hours)
2. Add `--rtxdi` command-line flag (30 minutes)
3. Add lighting system selector in Application.cpp (1 hour)

**This Week**:
1. Milestone 3: DXR pipeline setup (+14 hours)
2. Milestone 4: First RTXDI visual test (+8 hours)
3. **Total to First Test**: 32 hours from Milestone 1

**Parallel Track** (Optional):
1. Port PlasmaVulkan physics modes (8-10 hours)
2. Add `--physics=simple-orbit` for clean lighting demos
3. Test both multi-light and RTXDI with clean particle distribution

---

## Dual-Lighting-Path Architecture

**Design Decision**: Keep multi-light system as separate path, add RTXDI as parallel option

**Rationale**:
1. Performance comparison (multi-light vs RTXDI)
2. Correctness validation (multi-light as ground truth)
3. Fallback safety (if RTXDI has issues)
4. Educational value (see different approaches)
5. Research platform (A/B testing)

**Command-Line Selection** (similar to --billboard / --gaussian):
```bash
--multi-light  # Current system (default)
--rtxdi        # NVIDIA RTXDI (new)
```

**Both systems will**:
- Share PCSS shadow infrastructure (no duplication)
- Share volumetric Gaussian renderer
- Share particle physics
- Use different light sampling strategies

**Multi-Light** (good for <20 lights):
- Brute-force loop through all lights
- Simple, predictable
- 115-120 FPS @ 13 lights

**RTXDI** (good for 100+ lights):
- Importance sampling with ReSTIR
- Temporal + spatial reuse
- 105-115 FPS @ 100 lights

---

## Files Structure

```
external/
  RTXDI-Runtime/          # NVIDIA RTXDI SDK runtime library
    Include/Rtxdi/        # Headers (.h, .hlsli)
    Source/               # Implementation (.cpp)
    CMakeLists.txt        # Static library build

src/lighting/
  RTLightingSystem.cpp         # Existing RT particle-to-particle lighting (RayQuery)
  RTXDILightingSystem.h/cpp    # NEW: NVIDIA RTXDI wrapper (TraceRay)
  AccelerationStructure.cpp    # Shared BLAS/TLAS (both systems use this)

shaders/rtxdi/                 # NEW: RTXDI shaders (to be created in Milestone 2-4)
  light_grid_build_cs.hlsl     # Milestone 2: Light grid construction
  rtxdi_raygen.hlsl            # Milestone 3: Raygen with RTXDI sampling
  rtxdi_miss.hlsl              # Milestone 3: Miss shader
  rtxdi_closesthit.hlsl        # Milestone 3: ClosestHit with volume integration
  volumetric_brdf.hlsl         # Milestone 4: Callable shader (Henyey-Greenstein)
```

---

## Testing Strategy

### Milestone 2 Validation (Light Grid)

**PIX Capture**:
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --rtxdi --dump-buffers 120
```

**Buffer Validation**:
```python
# PIX/scripts/analysis/validate_light_grid.py
import struct

with open('PIX/buffer_dumps/frame_120/light_grid.bin', 'rb') as f:
    data = f.read()

# Each cell: 16 light indices (uint32) + 16 weights (float)
cell_size = (16 * 4) + (16 * 4)  # 128 bytes per cell
cell_count = len(data) // cell_size

print(f"Light grid cells: {cell_count}")
print(f"Expected: 27000 (30×30×30)")

# Validate light distribution
lights_per_cell = []
for i in range(cell_count):
    cell_data = data[i*cell_size:(i+1)*cell_size]
    light_indices = struct.unpack('16I', cell_data[:64])
    # Count non-zero light indices
    count = sum(1 for idx in light_indices if idx > 0)
    lights_per_cell.append(count)

print(f"Avg lights per cell: {sum(lights_per_cell) / len(lights_per_cell):.2f}")
print(f"Max lights in cell: {max(lights_per_cell)}")
```

### Milestone 3 Validation (DXR Pipeline)

**Check for crashes**:
```bash
./build/Debug/PlasmaDX-Clean.exe --rtxdi
# Should NOT crash on DispatchRays()
# Black screen is OK at this stage
```

**Log validation**:
```
[INFO] RTXDI: State object created (raygen + miss + closesthit + callable)
[INFO] RTXDI: Shader binding table built (4 records)
[INFO] RTXDI: DispatchRays() executing (1920 × 1080 rays)
```

### Milestone 4 Validation (First Visual Test)

**Visual Inspection**:
- Particles should be illuminated by RTXDI-selected lights
- Temporal reuse visible (smooth transitions frame-to-frame)
- PCSS shadows working

**Performance Check**:
```bash
# Should see ~100-110 FPS @ 10K particles
# Drop from multi-light (115-120 FPS) is expected (RTXDI overhead)
```

**Comparison Test**:
```bash
# Test multi-light
./build/Debug/PlasmaDX-Clean.exe --multi-light --physics=simple-orbit

# Test RTXDI
./build/Debug/PlasmaDX-Clean.exe --rtxdi --physics=simple-orbit

# Side-by-side PIX captures
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --multi-light --dump-buffers 120
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --rtxdi --dump-buffers 120
```

**Buffer Validation** (use buffer-validator-v3 agent):
```
@buffer-validator-v3 validate PIX/buffer_dumps/frame_120/rtxdi_reservoirs.bin
# Should see: valid reservoir data, no NaN/Inf, reasonable weight values
```

---

## Known Challenges and Solutions

### Challenge 1: RayQuery → TraceRay Migration

**Problem**: Multi-light uses RayQuery (inline ray tracing), RTXDI requires TraceRay (traditional DXR)

**Solution**: Keep both pipelines
- Multi-light: RayQuery (no changes)
- RTXDI: TraceRay (new state object + SBT)

**Why RTXDI needs TraceRay**: RTXDI uses callable shaders for material evaluation (Henyey-Greenstein phase function). Callable shaders require TraceRay pipeline.

### Challenge 2: Volumetric Adaptation

**Problem**: RTXDI designed for surfaces with normals. PlasmaDX has volumetric Gaussians (no surfaces).

**Solution**: Pseudo-surface abstraction
- Use view direction as "normal": `float3 pseudoNormal = -rayDir;`
- RTXDI uses this for importance sampling hints only
- Actual scattering computed in callable shader (volumetric phase function)

**Why this works**: Volumetric scattering is view-dependent (not normal-dependent). Henyey-Greenstein uses `dot(lightDir, viewDir)`, not normal.

### Challenge 3: No Surface Hit Position

**Problem**: RTXDI expects hit position on surface. Volumetric ray marching has integration path through volume.

**Solution**: Use ray march entry point as "hit position"
- Report first Gaussian intersection as hit
- RTXDI samples lights at that position
- Volume integration happens in ClosestHit shader (16 steps along ray)

**Result**: RTXDI selects important lights, volumetric rendering handles scattering.

---

## Documentation Updates Needed

### CLAUDE.md

**Add after Multi-Light System section** (~line 550):

```markdown
## RTXDI Integration (Phase 4 - Active Development)

**Status**: Milestone 1 Complete (SDK Linked) - Week 1 Day 2 In Progress

**Timeline**: ~32 hours to first visual test (Milestone 4)

**Components**:
- Light Grid (ReGIR) - Spatial acceleration for 100+ lights
- Reservoir Sampling - Importance-weighted light selection
- Temporal Reuse - Merge with previous frame
- Spatial Reuse - Merge with neighbor pixels (Week 3)

**Integration Strategy**: Parallel path to multi-light system
- Command-line: `--multi-light` (default) vs `--rtxdi` (new)
- Both share PCSS shadow system
- Both share volumetric Gaussian renderer
- Both share particle physics

**Performance Targets**:
- Milestone 4 (basic temporal): 100-110 FPS @ 10K particles
- Full feature (spatial reuse): 105-115 FPS @ 100 lights

**Milestones**:
1. ✅ SDK Linked (Milestone 1 - COMPLETE)
2. ⏳ Light Grid Construction (Milestone 2 - Next)
3. ⏳ DXR Pipeline Setup (Milestone 3)
4. ⏳ First Visual Test (Milestone 4)
5. ⏳ Spatial Reuse + Optimization (Week 3-4)

See `.claude/RTXDI_WEEK1_MILESTONE1_COMPLETE.md` for complete integration plan.
```

### README.md

**Update features section**:

```markdown
### Ray Traced Lighting & Shadows

- **Dual Lighting Systems** (Runtime Selectable)
  - **Multi-Light System** (Phase 3.5) ✅ COMPLETE
    - 13-light "Stellar Ring" configuration
    - Manual light placement and color control
    - Perfect for <20 lights
    - 115-120 FPS @ 10K particles

  - **RTXDI Integration** (Phase 4) ⏳ IN PROGRESS
    - Milestone 1: SDK Linked ✅
    - Milestone 4 Target: First visual test (~32 hours)
    - NVIDIA RTX Direct Illumination
    - ReSTIR GI (spatial + temporal reuse)
    - Scales to 100+ lights
    - 100-115 FPS @ 10K particles + 100 lights

- **PCSS Soft Shadows** ✅ COMPLETE
  - 3 quality presets (Performance/Balanced/Quality)
  - Percentage-Closer Soft Shadows with Poisson disk sampling
  - Temporal filtering (1-ray convergence to 8-ray quality)
  - Works with both Multi-Light and RTXDI
```

---

## Next Session Agenda

**Immediate Tasks**:
1. Add `--rtxdi` command-line flag to Application.cpp
2. Add `LightingSystem` enum (MultiLight vs RTXDI)
3. Start Milestone 2: Light grid buffer creation

**Milestone 2 Breakdown** (10 hours):
- Hour 1-2: Create light grid buffer + descriptor
- Hour 3-4: Write light grid build compute shader
- Hour 5-6: Integrate RTXDI light data structures
- Hour 7-8: Dispatch light grid build each frame
- Hour 9: PIX capture validation
- Hour 10: Buffer dump and Python validation script

**Goal for Next Session**: Complete Milestone 2 (light grid construction)

**Stretch Goal**: Start Milestone 3 (state object creation for DXR pipeline)

---

## Conclusion

**Milestone 1 Status**: ✅ **COMPLETE**

**What Was Achieved**:
- RTXDI SDK successfully linked
- RTXDILightingSystem class created
- Build verified (no errors)
- Multi-light bug fixed (13 lights at startup)
- PCSS validation complete (shadows working)

**Timeline Clarity**:
- **To First Test**: 32 hours from Milestone 1
- **Milestone 2**: +10 hours (light grid)
- **Milestone 3**: +14 hours (DXR pipeline)
- **Milestone 4**: +8 hours (**first visual RTXDI test!**)

**Strategic Decisions**:
- Dual-path architecture (multi-light + RTXDI)
- Physics modes system (port from PlasmaVulkan)
- Both work in parallel tracks

**Ready to Proceed**: YES! 🚀

---

**Document Version**: 1.0
**Last Updated**: 2025-10-18
**Next Milestone**: Light Grid Construction (Milestone 2, +10 hours)
