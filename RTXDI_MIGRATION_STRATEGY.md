# RTXDI Migration Strategy - Legacy ReSTIR Cleanup

**Date**: 2025-10-18
**Branch**: 0.7.5 → 0.8.0
**Decision Point**: Before starting Milestone 4 (reservoir sampling)

---

## Executive Summary

**Recommendation**: **Clean removal of legacy ReSTIR before implementing RTXDI reservoir sampling**

This prevents architectural confusion, reduces code complexity, and ensures RTXDI M4 has a clean foundation.

---

## Current Architecture Overlap

### Legacy ReSTIR (Custom Implementation - DEPRECATED)

**Location**: `ParticleRenderer_Gaussian` class

**Buffers** (138 MB total @ 1080p):
```cpp
// src/particles/ParticleRenderer_Gaussian.h:138-144
Microsoft::WRL::ComPtr<ID3D12Resource> m_reservoirBuffer[2];  // 2× 63MB
D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirSRV[2];
D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirSRVGPU[2];
D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirUAV[2];
D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirUAVGPU[2];
uint32_t m_currentReservoirIndex = 0;
```

**Controls** (in RenderConstants):
```cpp
uint32_t useReSTIR;                     // Toggle
uint32_t restirInitialCandidates;       // M = 16-32
uint32_t frameIndex;                    // Temporal validation
float restirTemporalWeight;             // 0-1
```

**Shader Integration**:
- `particle_gaussian_raytrace.hlsl` - Has ReSTIR sampling code
- Reads from `g_prevReservoirs` (t5)
- Writes to `g_currReservoirs` (u2)

**Status**:
- ⚠️ DEPRECATED per CLAUDE.md:185
- ⚠️ "Known Issues: Weight calculation edge cases, temporal reuse validation bugs"
- ⚠️ Never worked correctly (from CLAUDE.md context)

---

### RTXDI System (New Production Implementation)

**Location**: `RTXDILightingSystem` class

**Architecture**:
- Light grid (27,000 cells, 3.375 MB)
- DXR pipeline (raygen + miss shaders)
- Future: RTXDI reservoir sampling (M4)

**Status**:
- ✅ M2.1 Complete: Light grid buffers
- ✅ M2.2 Complete: Compute shader
- ✅ M2.3 Complete: Buffer validation
- ✅ M3 Complete: DXR pipeline (shaders not loading - need fix)
- ⏸️ M4 Pending: Reservoir sampling

**Goal**: Replace legacy ReSTIR entirely with production-grade NVIDIA RTXDI

---

## The Problem: Architectural Confusion

### Issue 1: Two Reservoir Systems

**If we keep legacy ReSTIR during RTXDI M4 development:**
- 2 separate reservoir buffer systems (legacy 138MB + RTXDI future ~64MB = 202MB)
- Confusing naming: `m_reservoirBuffer[2]` vs future RTXDI reservoirs
- Risk of using wrong buffers in shaders
- Increased debugging complexity

### Issue 2: Shader Resource Conflicts

**Current Gaussian shader bindings**:
```hlsl
// Legacy ReSTIR
Texture2D<float4> g_prevReservoirs : register(t5);  // Legacy
RWTexture2D<float4> g_currReservoirs : register(u2); // Legacy

// Future RTXDI (will need same registers!)
StructuredBuffer<Reservoir> g_rtxdiReservoirs : register(t?);  // Conflict!
```

**Result**: Register conflicts, binding confusion, hard-to-debug issues

### Issue 3: Code Maintenance Burden

**Legacy ReSTIR code locations**:
1. `ParticleRenderer_Gaussian.h` - 7 member variables
2. `ParticleRenderer_Gaussian.cpp` - Buffer creation, descriptor allocation
3. `RenderConstants` struct - 4 parameters
4. `particle_gaussian_raytrace.hlsl` - ReSTIR sampling logic
5. `Application.cpp` - ImGui controls for `useReSTIR` toggle

**Keeping this during M4 development**:
- Extra mental overhead
- Risk of accidentally using legacy path
- Confusion for future developers
- Harder PIX debugging (two reservoir systems visible)

---

## Recommended Migration Strategy

### Phase 1: Fix Shader Loading (5 minutes)

**Problem**: RTXDI shaders not loading (logs/PlasmaDX-Clean_20251018_230411.log:132)

```
[ERROR] Failed to open rtxdi_raygen.dxil
[ERROR] Failed to create DXR pipeline
```

**Root Cause**: Shaders compiled to wrong location or not compiled at all

**Fix**: Verify shader files exist and are in correct path

```bash
# Check shader existence
ls -lh build/Debug/shaders/rtxdi/*.dxil

# Expected:
#   light_grid_build_cs.dxil (7.6 KB)
#   rtxdi_raygen.dxil (5.4 KB)
#   rtxdi_miss.dxil (2.5 KB)

# If missing, manually compile (from previous session):
dxc.exe -T lib_6_3 shaders/rtxdi/rtxdi_raygen.hlsl -Fo build/Debug/shaders/rtxdi/rtxdi_raygen.dxil
dxc.exe -T lib_6_3 shaders/rtxdi/rtxdi_miss.hlsl -Fo build/Debug/shaders/rtxdi/rtxdi_miss.dxil
```

**Validation**: Re-run with `"--rtxdi"` flag (explicit quotes), check logs for:
```
[INFO] RTXDI DispatchRays executed (1920x1080)
```

**Branch**: Save as 0.7.6 after fix verified

---

### Phase 2: Remove Legacy ReSTIR (30 minutes)

**Use rtxdi-integration-specialist-v4 or code-reviewer agent**

**Files to Modify**:

1. **`src/particles/ParticleRenderer_Gaussian.h`** (remove 7 members):
   ```cpp
   // DELETE these lines:
   Microsoft::WRL::ComPtr<ID3D12Resource> m_reservoirBuffer[2];
   D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirSRV[2];
   D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirSRVGPU[2];
   D3D12_CPU_DESCRIPTOR_HANDLE m_reservoirUAV[2];
   D3D12_GPU_DESCRIPTOR_HANDLE m_reservoirUAVGPU[2];
   uint32_t m_currentReservoirIndex = 0;

   // DELETE these methods:
   ID3D12Resource* GetCurrentReservoirs() const;
   ID3D12Resource* GetPrevReservoirs() const;
   ```

2. **`src/particles/ParticleRenderer_Gaussian.h`** (RenderConstants):
   ```cpp
   // DELETE these lines (around line 61-65):
   uint32_t useReSTIR;
   uint32_t restirInitialCandidates;
   uint32_t frameIndex;
   float restirTemporalWeight;
   ```

3. **`src/particles/ParticleRenderer_Gaussian.cpp`**:
   - Remove reservoir buffer creation (Initialize method)
   - Remove descriptor allocation for reservoirs
   - Remove ping-pong swap logic (Render method)
   - Remove reservoir buffer cleanup (destructor)

4. **`src/core/Application.cpp`**:
   - Remove ImGui controls for `useReSTIR` toggle
   - Remove `restirInitialCandidates` slider
   - Remove `restirTemporalWeight` slider
   - Search for "ReSTIR" and remove all UI code

5. **`shaders/particles/particle_gaussian_raytrace.hlsl`**:
   - Remove `g_prevReservoirs` binding (t5)
   - Remove `g_currReservoirs` binding (u2)
   - Remove ReSTIR sampling logic (search for "ReSTIR" comments)
   - Remove candidate sampling loop
   - Remove reservoir update code

6. **Root Signature** (ParticleRenderer_Gaussian.cpp):
   - Remove descriptor table entries for t5, u2
   - Update parameter count (currently 10, will be 8)

**Validation After Removal**:
```bash
# Should still run with multi-light system
build\Debug\PlasmaDX-Clean.exe

# Check logs for NO ReSTIR references:
grep -i "restir" logs/PlasmaDX-Clean_*.log
# Should return NOTHING

# Verify multi-light still works (13 lights, 120 FPS)
```

**Branch**: Save as 0.7.7 after validation

---

### Phase 3: Implement RTXDI Reservoir Sampling (M4) - 6-8 hours

**NOW you have clean foundation for RTXDI!**

**Deploy rtxdi-integration-specialist-v4 with this prompt**:

```
RTXDI Milestone 4: Reservoir Sampling Integration

Context:
- M1-M3 complete: Light grid (27k cells), DXR pipeline operational
- Legacy ReSTIR removed (clean slate)
- Target: First visual difference with RTXDI vs multi-light

Tasks:
1. Create RTXDI reservoir buffers (NOT reusing old ReSTIR buffers)
   - Structure: Based on RTXDI SDK spec
   - Size: 1920×1080 per buffer
   - Ping-pong: 2 buffers for temporal reuse

2. Extend rtxdi_raygen.hlsl with reservoir sampling:
   - Sample light from grid cell (weighted random)
   - Store selected light in reservoir
   - Temporal reuse (merge with previous frame)

3. Integrate with Gaussian renderer:
   - Pass RTXDI reservoir output to particle_gaussian_raytrace.hlsl
   - Replace multi-light loop with single RTXDI sample
   - Add --rtxdi vs --multi-light comparison

4. Performance validation:
   - Target: 115-120 FPS (within 5% of baseline)
   - Measure RTXDI overhead

Use 7+ MCP queries for RTXDI SDK reservoir structure, temporal reuse algorithm, etc.
```

**Expected Outcome**:
- Clean RTXDI reservoir implementation
- No confusion with old ReSTIR code
- First visual test: RTXDI matches multi-light quality
- Performance within target

**Branch**: Save as 0.8.0 (first visual test milestone!)

---

## Why Remove Legacy ReSTIR Now?

### Argument FOR Removal

1. **Clean Architecture**:
   - No naming conflicts (`m_reservoirBuffer` available for RTXDI)
   - No shader register conflicts (t5, u2 free)
   - Single source of truth for reservoir sampling

2. **Easier Debugging**:
   - PIX captures show only one reservoir system
   - No confusion about which buffers are active
   - Clear mental model: "RTXDI only"

3. **Faster M4 Development**:
   - No risk of accidentally using legacy code
   - Agent can use standard variable names
   - Cleaner diffs for code review

4. **Lower Memory Usage**:
   - Remove 138 MB of legacy buffers
   - Make room for RTXDI buffers (~64 MB)
   - Net savings: 74 MB

5. **CLAUDE.md Compliance**:
   - CLAUDE.md:216: "Phase 4: Remove custom ReSTIR code"
   - This IS Phase 4 (we're starting M4 now)
   - Following the documented plan

### Argument AGAINST Removal (and rebuttals)

1. **"What if we need to reference legacy implementation?"**
   - Rebuttal: It never worked correctly, documented as broken
   - Git history preserves it (branch 0.7.5 and earlier)
   - RTXDI is production-grade, no need for buggy reference

2. **"Removal takes time away from M4"**
   - Rebuttal: 30 minutes vs potential 4-8 hours of confusion
   - Clean foundation makes M4 FASTER
   - Reduces risk of architectural mistakes

3. **"Maybe we want to compare implementations?"**
   - Rebuttal: Can't compare - legacy doesn't work correctly
   - Visual comparison will be RTXDI vs multi-light (not vs ReSTIR)
   - Multi-light is the real baseline

### Recommendation: **REMOVE IT**

**Rationale**: The 30-minute cost of removal is FAR outweighed by:
- Architectural clarity
- Reduced debugging complexity
- Faster M4 development
- Compliance with documented migration plan

---

## Alternative: Keep But Disable

**If you're hesitant to delete**, we could:

1. **Completely disable via preprocessor**:
   ```cpp
   #define ENABLE_LEGACY_RESTIR 0  // Set to 0 to disable

   #if ENABLE_LEGACY_RESTIR
   // All legacy ReSTIR code here
   #endif
   ```

2. **Remove from shaders** (higher conflict risk):
   - Still remove from `particle_gaussian_raytrace.hlsl`
   - Keep C++ code disabled

**Downsides**:
- Still have naming conflicts
- Code clutter (disabled code visible)
- Temptation to re-enable instead of fixing RTXDI

**Verdict**: Not recommended. If keeping, better to just delete and rely on git history.

---

## Recommended Timeline

| Phase | Task | Time | Branch |
|-------|------|------|--------|
| 1 | Fix shader loading (M3) | 5 min | 0.7.6 |
| 2 | Test M3 with working shaders | 5 min | 0.7.6 |
| 3 | Remove legacy ReSTIR | 30 min | 0.7.7 |
| 4 | Test multi-light still works | 10 min | 0.7.7 |
| 5 | **START M4**: RTXDI reservoir sampling | 6-8 hours | 0.8.0 |

**Total overhead from cleanup**: 50 minutes
**Time saved during M4**: 2-4 hours (conservative estimate)
**Net benefit**: 1.5-3.5 hours saved + cleaner architecture

---

## Agent Deployment Strategy

### Phase 1-2: Fix Shader Loading

**DIY** (5 minutes) or deploy **pix-debugger-v3** if stuck

### Phase 3: Remove Legacy ReSTIR

**Option A**: Manual removal (faster if confident)

**Option B**: Deploy **code-reviewer agent** (safer):
```
Review and remove all legacy ReSTIR code from PlasmaDX-Clean

Context:
- Legacy ReSTIR marked DEPRECATED in CLAUDE.md:185
- Never worked correctly, being replaced by RTXDI
- Need clean architecture before RTXDI M4

Files to clean:
1. src/particles/ParticleRenderer_Gaussian.h (remove reservoir members)
2. src/particles/ParticleRenderer_Gaussian.cpp (remove buffer creation)
3. src/core/Application.cpp (remove ImGui controls)
4. shaders/particles/particle_gaussian_raytrace.hlsl (remove sampling code)

Search for: "ReSTIR", "reservoir", "m_reservoirBuffer", "useReSTIR"

Validation:
- Build succeeds
- Multi-light system still works (13 lights, 120 FPS)
- No "ReSTIR" references in logs
```

### Phase 4: RTXDI M4

**Deploy rtxdi-integration-specialist-v4** (see Phase 3 above for full prompt)

---

## Success Criteria

### After Phase 2 (Shader Fix)
- ✅ RTXDI shaders load successfully
- ✅ Log shows: "RTXDI DispatchRays executed (1920x1080)"
- ✅ No fallback to multi-light
- ✅ 115-120 FPS maintained

### After Phase 3 (Legacy Removal)
- ✅ No `m_reservoirBuffer` in code (except git history)
- ✅ No ReSTIR references in shaders
- ✅ Multi-light still works (13 lights visible)
- ✅ 120 FPS with multi-light
- ✅ Build succeeds with zero warnings
- ✅ `grep -r "ReSTIR" src/` returns nothing (except comments about RTXDI vs legacy)

### After Phase 5 (M4 Complete)
- ✅ RTXDI reservoir sampling operational
- ✅ Visual difference: `--rtxdi` vs `--multi-light`
- ✅ Performance within 5% of multi-light baseline
- ✅ Clean reservoir architecture (RTXDI only)
- ✅ Ready for M5 (spatial reuse)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Break multi-light during removal | Low | High | Test after each file change |
| Accidentally remove RTXDI code | Very Low | Medium | Search for "RTXDI" separately |
| Miss some legacy references | Medium | Low | Use grep -r extensively |
| Shader compilation issues | Low | Medium | Have dxc.exe commands ready |
| Forget to update root signature | Medium | High | Use code-reviewer agent |

**Overall Risk**: **LOW** - Removal is straightforward, well-scoped

---

## Questions for User

Before proceeding, please confirm:

1. **Do you want to remove legacy ReSTIR now** (my recommendation)?
   - Alternative: Keep but disable with preprocessor
   - Alternative: Keep as-is and work around conflicts

2. **Which agent to use for removal?**
   - Manual (you're confident, want speed)
   - code-reviewer (safer, automated)
   - Both (manual + agent validation)

3. **Should we fix shader loading first** (Phase 1-2)?
   - This is blocking M3 testing
   - 5-10 minutes to fix
   - Then we can see RTXDI pipeline actually run

4. **Timeline preference**:
   - Full cleanup today (Phases 1-3, ~50 min)
   - Just fix shaders today, defer removal
   - Skip cleanup, proceed to M4 with conflicts

---

## My Recommended Answer

**"Yes, remove legacy ReSTIR now (Phase 3) after fixing shaders (Phase 1-2)"**

**Reasoning**:
- 30-minute investment prevents 2-4 hours of M4 confusion
- Clean architecture from day 1
- Follows documented migration plan (CLAUDE.md:216)
- Git preserves history if ever needed
- Makes M4 agent deployment cleaner

**Execution**:
1. Fix shader loading (5 min)
2. Test M3 works (5 min)
3. Deploy code-reviewer to remove legacy ReSTIR (30 min)
4. Test multi-light still works (10 min)
5. Save as branch 0.7.7
6. **THEN** start M4 with rtxdi-integration-specialist-v4

**Total time to clean foundation**: 50 minutes
**Ready for M4**: Clean, tested, no conflicts

---

**Decision Point**: Your call! What's your preference?
