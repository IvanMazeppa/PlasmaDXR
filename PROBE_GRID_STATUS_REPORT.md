# Probe Grid Implementation - Status Report & Roadmap

**Date:** 2025-11-03 21:30
**Branch:** 0.13.1
**Context:** Pivoting from Volumetric ReSTIR to Hybrid Probe Grid after atomic contention analysis

---

## Executive Summary

**Problem:** Volumetric ReSTIR crashes at ≥2045 particles due to atomic contention (5.35 particles/voxel with 32³ volume).

**Solution:** Hybrid Probe Grid - pre-compute lighting at sparse grid points, particles interpolate via trilinear sampling.

**Status:** 55% complete (6/11 tasks done)
- ✅ Data structures and GPU buffers
- ✅ Shader implementation (update_probes.hlsl)
- ✅ Pipeline creation (root signature + PSO)
- ⏳ Dispatch logic (IN PROGRESS)
- ⏳ Particle query integration (PENDING)
- ⏳ Application wiring (PENDING)

**Expected Result:** 90-110 FPS at 10K particles with zero atomic contention

---

## Background: Why We Pivoted

### Volumetric ReSTIR Failure Analysis

**Timeline:**
- **2025-11-02:** Discovered GPU TDR crash at exactly 2045 particles (3-second timeout)
- **2025-11-03:** Diagnosed root cause: atomic contention in InterlockedMax() operations
- **2025-11-03 20:15:** Pivoted to Probe Grid approach after analysis showed unfixable architectural issue

**Root Cause:**
```
32³ volume = 32,768 voxels
2044 particles × 84.4 voxels/particle = 175,204 voxel writes
175,204 ÷ 32,768 = 5.35 particles per voxel (average)
```

**Why It Failed:**
- Multiple GPU threads calling `InterlockedMax()` on same voxel
- Atomic operations serialize → threads wait in queue
- At 2045 particles, worst-case voxel exceeds 2-second TDR timeout → crash
- **Critical insight:** Reducing resolution made it WORSE (fewer voxels = more overlap)

**See:** `VOLUMETRIC_RESTIR_ATOMIC_CONTENTION_ANALYSIS.md` for complete technical analysis

---

## Probe Grid Architecture

### Core Concept

**Grid Creation (once) → Probe Update (every 4 frames) → Particle Query (every frame)**

Pre-compute lighting at 32,768 sparse probe points, particles interpolate between nearest 8 probes using trilinear interpolation.

### Key Architectural Advantages

1. **Zero Atomic Operations**
   - Each probe writes to ITS OWN memory location
   - No inter-thread conflicts possible
   - No contention regardless of particle count

2. **Scalable Design**
   - Performance independent of particle density
   - 2045 particles? 10,000 particles? Same cost!
   - O(1) per-particle query (8 probe reads)

3. **Temporal Amortization**
   - Update 1/4 of probes per frame
   - Spreads 524,288 rays across 4 frames
   - Amortized cost: 131,072 rays/frame (~0.5-1.0ms)

4. **Clean Separation**
   - Probe updates: RayQuery + lighting accumulation
   - Particle queries: Pure read-only texture sampling
   - No shared state between passes

### Grid Parameters (32³ Configuration)

```
Grid Size:        32³ = 32,768 probes
World Coverage:   [-1500, +1500] units per axis (3000-unit cube)
Grid Spacing:     93.75 units per probe (3000 ÷ 32)
Probe Size:       128 bytes (16-byte aligned)
Memory Footprint: 4.06 MB (vs Volumetric ReSTIR's 29.88 MB)
```

**Probe Data Structure:**
```cpp
struct Probe {
    float3 position;           // 12 bytes - world-space location
    uint32_t lastUpdateFrame;  // 4 bytes - temporal tracking
    float3 irradiance[9];      // 108 bytes - spherical harmonics L2
    uint32_t padding[1];       // 4 bytes - GPU alignment
};  // Total: 128 bytes
```

### Performance Estimates

**Probe Update Cost (every 4 frames):**
- 8,192 probes × 64 rays = 524,288 rays
- Amortized: 131,072 rays/frame
- Expected: 0.5-1.0ms per frame

**Particle Query Cost (every frame):**
- 10K particles × 8 probe reads = 80,000 reads
- Cache-friendly trilinear interpolation
- Expected: 0.2-0.3ms per frame

**Total Overhead:** ~0.7-1.3ms per frame
**Expected FPS:** 90-110 FPS @ 10K particles (vs current crash at 2045)

---

## Implementation Progress (6/11 Tasks Complete)

### ✅ Task 1: ProbeGridSystem Class Structure
**Status:** COMPLETE
**Files:** `src/lighting/ProbeGridSystem.h`, `src/lighting/ProbeGridSystem.cpp`

**Key Methods:**
- `Initialize()` - Creates buffers, descriptors, pipelines
- `UpdateProbes()` - Per-frame probe lighting update
- `GetProbeBufferSRV()` - Descriptor for particle shader sampling
- `GetProbeGridParams()` - Grid configuration for shader binding

**CMake Integration:** ✅ Added to sources/headers in CMakeLists.txt

---

### ✅ Task 2: Probe Data Structure
**Status:** COMPLETE
**Location:** `ProbeGridSystem.h:81-92`

**Design Decisions:**
- 128-byte alignment for GPU cache efficiency
- Spherical harmonics L2 (9 RGB coefficients)
- MVP uses SH[0] only (DC term), higher-order terms reserved for Phase 2
- `lastUpdateFrame` enables temporal amortization tracking

---

### ✅ Task 3: GPU Buffer Allocation
**Status:** COMPLETE
**Location:** `ProbeGridSystem.cpp:42-153`

**Resources Created:**
- Probe buffer: 4.06 MB structured buffer (32,768 × 128 bytes)
- Constant buffer: 256 bytes (ProbeUpdateConstants)
- SRV/UAV descriptors via ResourceManager

**Verification:**
```bash
# Build log shows:
Probe buffer created successfully!
  Total probes: 32768
  Probe size: 128 bytes
  Buffer size: 4.06 MB
```

---

### ✅ Task 4: Descriptor Heap Setup
**Status:** COMPLETE
**Location:** `ProbeGridSystem.cpp:84-117`

**Descriptors Allocated:**
- `m_probeBufferSRV` (D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
  - For particle shader sampling (trilinear interpolation)
  - GPU handle stored for fast binding
  
- `m_probeBufferUAV` (D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
  - For probe update shader writes
  - GPU handle stored for fast binding

---

### ✅ Task 5: update_probes.hlsl Shader
**Status:** COMPLETE
**Location:** `shaders/probe_grid/update_probes.hlsl` (compiled to .dxil)

**Shader Features:**

**1. Fibonacci Sphere Ray Distribution**
```hlsl
float3 FibonacciSphere(uint index, uint totalRays) {
    // Evenly distributed rays on unit sphere
    // Based on Hannay & Nye (1999) golden ratio spiral
}
```
- 64 rays per probe (configurable)
- Even distribution prevents clustering artifacts
- Deterministic (same rays every update)

**2. RayQuery Inline Ray Tracing**
```hlsl
RayQuery<RAY_FLAG_NONE> q;
q.TraceRayInline(g_particleTLAS, RAY_FLAG_NONE, 0xFF, ray);
q.Proceed();
if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    // Accumulate lighting from hit particle
}
```
- **CRITICAL:** No atomic operations!
- Each probe independently traces rays
- Uses existing particle TLAS (no duplicate infrastructure)

**3. Temperature-Based Blackbody Emission**
```hlsl
float3 BlackbodyColor(float temperature);
float BlackbodyIntensity(float temperature);
```
- Wien's law approximation (800K-30,000K range)
- Cool red (1000K) → Yellow (3000K) → White (6000K) → Hot blue (15000K+)
- Stefan-Boltzmann law: intensity ∝ T⁴

**4. Temporal Amortization**
```hlsl
uint updateSlot = g_frameIndex % g_updateInterval;
if ((probeLinearIdx % g_updateInterval) != updateSlot) {
    return; // Not this probe's turn
}
```
- Frame 0: probes 0, 4, 8, 12, ...
- Frame 1: probes 1, 5, 9, 13, ...
- Frame 2: probes 2, 6, 10, 14, ...
- Frame 3: probes 3, 7, 11, 15, ...

**Thread Configuration:**
- `[numthreads(8, 8, 8)]` = 512 threads per group
- Dispatch: 4×4×4 thread groups = 64 groups
- Total: 512 × 64 = 32,768 threads (one per probe)

**Build Verification:**
```bash
# CMakeLists.txt line 311-319:
Compiling update_probes.hlsl
# Output: shaders/probe_grid/update_probes.dxil (compiled successfully)
```

---

### ✅ Task 6: Root Signature + PSO
**Status:** COMPLETE
**Location:** `ProbeGridSystem.cpp:156-250`

**Root Signature (5 parameters):**
```cpp
rootParams[0]: b0 - ProbeUpdateConstants (CBV)
rootParams[1]: t0 - Particle buffer (SRV, structured buffer)
rootParams[2]: t1 - Light buffer (SRV, structured buffer)
rootParams[3]: t2 - Particle TLAS (SRV, acceleration structure)
rootParams[4]: u0 - Probe buffer (UAV, structured buffer)
```

**Pipeline State:**
- Compute shader: update_probes.dxil
- Root signature: m_updateProbeRS
- PSO name: "ProbeGridSystem::UpdateProbePSO"

**Build Verification:**
```bash
# Build succeeded (2025-11-03 21:15):
ProbeGridSystem.cpp compiled successfully
PlasmaDX-Clean.exe linked successfully
```

---

## Remaining Tasks (5/11)

### ⏳ Task 7: Implement Probe Update Dispatch (IN PROGRESS)
**Status:** Stub implementation exists, needs completion
**Location:** `ProbeGridSystem.cpp:252-268` (UpdateProbes method)

**What Needs to be Done:**
1. Upload ProbeUpdateConstants to constant buffer
2. Set compute root signature and PSO
3. Bind 5 shader resources (constant buffer, particles, lights, TLAS, probes)
4. Dispatch 4×4×4 thread groups
5. UAV barrier after dispatch

**Pseudocode:**
```cpp
void ProbeGridSystem::UpdateProbes(...) {
    if (!m_initialized || !m_enabled) return;
    
    // 1. Upload constants
    ProbeUpdateConstants constants = {
        .gridMin = m_gridMin,
        .gridSpacing = m_gridSpacing,
        .gridSize = m_gridSize,
        .raysPerProbe = m_raysPerProbe,
        .particleCount = particleCount,
        .lightCount = lightCount,
        .frameIndex = frameIndex,
        .updateInterval = m_updateInterval
    };
    // Map and copy to m_updateConstantBuffer
    
    // 2. Set pipeline state
    commandList->SetPipelineState(m_updateProbePSO.Get());
    commandList->SetComputeRootSignature(m_updateProbeRS.Get());
    
    // 3. Bind resources
    commandList->SetComputeRootConstantBufferView(0, constantBufferGPUAddress);
    commandList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(2, lightBuffer->GetGPUVirtualAddress());
    commandList->SetComputeRootShaderResourceView(3, particleTLAS->GetGPUVirtualAddress());
    commandList->SetComputeRootUnorderedAccessView(4, m_probeBuffer->GetGPUVirtualAddress());
    
    // 4. Dispatch 4×4×4 thread groups (8×8×8 threads each = 512 threads/group)
    commandList->Dispatch(4, 4, 4);
    
    // 5. UAV barrier
    CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_probeBuffer.Get());
    commandList->ResourceBarrier(1, &uavBarrier);
    
    m_frameCount++;
}
```

**Reference:** See `VolumetricReSTIRSystem.cpp:854-918` for similar dispatch pattern

---

### ⏳ Task 8: Particle Shader Integration (PENDING)
**Status:** Not started
**Files to Modify:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**What Needs to be Done:**

**1. Add probe grid constant buffer:**
```hlsl
cbuffer ProbeGridParams : register(b4) {  // Next available slot
    float3 g_probeGridMin;
    float g_probeGridSpacing;
    uint g_probeGridSize;
    uint g_probeGridEnabled;
    uint padding0;
    uint padding1;
};
```

**2. Add probe buffer SRV:**
```hlsl
struct Probe {
    float3 position;
    uint lastUpdateFrame;
    float3 irradiance[9];
    uint padding[1];
};
StructuredBuffer<Probe> g_probeGrid : register(t7);  // Next available slot
```

**3. Implement trilinear interpolation:**
```hlsl
float3 SampleProbeGrid(float3 worldPos) {
    // 1. Convert world pos to grid coordinates
    float3 gridPos = (worldPos - g_probeGridMin) / g_probeGridSpacing;
    int3 baseProbe = floor(gridPos);
    float3 blend = frac(gridPos);
    
    // 2. Trilinear interpolation between 8 nearest probes
    float3 irradiance = float3(0, 0, 0);
    for (int z = 0; z <= 1; z++) {
        for (int y = 0; y <= 1; y++) {
            for (int x = 0; x <= 1; x++) {
                int3 probeIdx = baseProbe + int3(x, y, z);
                
                // Bounds check
                if (all(probeIdx >= 0) && all(probeIdx < g_probeGridSize)) {
                    uint idx = probeIdx.x + probeIdx.y * g_probeGridSize + 
                               probeIdx.z * g_probeGridSize * g_probeGridSize;
                    
                    // Trilinear weights
                    float3 weight = lerp(1.0 - blend, blend, float3(x, y, z));
                    float w = weight.x * weight.y * weight.z;
                    
                    // Sample probe irradiance (SH[0] for MVP)
                    irradiance += g_probeGrid[idx].irradiance[0] * w;
                }
            }
        }
    }
    
    return irradiance;
}
```

**4. Integrate into main shader:**
```hlsl
// In main particle rendering loop:
if (g_probeGridEnabled) {
    float3 probeIrradiance = SampleProbeGrid(particleWorldPos);
    finalColor += probeIrradiance * particleDensity * scatteringAlbedo;
}
```

**Reference:** See `PROBE_GRID_IMPLEMENTATION_OUTLINE.md` lines 87-121

---

### ⏳ Task 9: Application Integration (PENDING)
**Status:** Not started
**Files to Modify:** `src/core/Application.h`, `src/core/Application.cpp`

**What Needs to be Done:**

**1. Add ProbeGridSystem member:**
```cpp
// Application.h
#include "lighting/ProbeGridSystem.h"

class Application {
    // ...
private:
    std::unique_ptr<ProbeGridSystem> m_probeGridSystem;
};
```

**2. Initialize in Application::Initialize():**
```cpp
// After RTLightingSystem initialization
m_probeGridSystem = std::make_unique<ProbeGridSystem>();
if (!m_probeGridSystem->Initialize(m_device.get(), m_resourceManager.get())) {
    LOG_ERROR("Failed to initialize ProbeGridSystem");
    return false;
}
LOG_INFO("ProbeGridSystem initialized successfully");
```

**3. Update in Render() loop:**
```cpp
// In Application::Render(), before particle rendering:
if (m_probeGridSystem && m_probeGridSystem->IsEnabled()) {
    m_probeGridSystem->UpdateProbes(
        commandList,
        m_rtLightingSystem->GetParticleTLAS(),  // Reuse existing TLAS
        m_particleSystem->GetParticleBuffer(),
        m_particleSystem->GetParticleCount(),
        m_lightBuffer.Get(),
        m_lights.size(),
        m_frameCount
    );
}
```

**4. Pass probe buffer to Gaussian renderer:**
```cpp
// In ParticleRenderer_Gaussian::Render():
if (m_probeGridEnabled) {
    ProbeGridSystem::ProbeGridParams params = m_probeGridSystem->GetProbeGridParams();
    // Upload params to constant buffer
    // Bind probe buffer SRV
}
```

**Reference:** See how VolumetricReSTIRSystem is integrated in Application.cpp

---

### ⏳ Task 10: ImGui Controls (PENDING)
**Status:** Not started
**Location:** Add to Application::RenderImGui() in `src/core/Application.cpp`

**What Needs to be Done:**

```cpp
// In "Rendering Features" section
if (ImGui::CollapsingHeader("Probe Grid Lighting")) {
    bool enabled = m_probeGridSystem->IsEnabled();
    if (ImGui::Checkbox("Enable Probe Grid", &enabled)) {
        m_probeGridSystem->SetEnabled(enabled);
    }
    
    if (enabled) {
        uint32_t raysPerProbe = m_probeGridSystem->GetRaysPerProbe();
        if (ImGui::SliderInt("Rays Per Probe", reinterpret_cast<int*>(&raysPerProbe), 16, 256)) {
            m_probeGridSystem->SetRaysPerProbe(raysPerProbe);
        }
        
        uint32_t updateInterval = m_probeGridSystem->GetUpdateInterval();
        if (ImGui::SliderInt("Update Interval (frames)", reinterpret_cast<int*>(&updateInterval), 1, 8)) {
            m_probeGridSystem->SetUpdateInterval(updateInterval);
        }
        
        ImGui::Text("Grid: %u³ probes", m_probeGridSystem->GetGridSize());
        ImGui::Text("Spacing: %.2f units", m_probeGridSystem->GetProbeGridParams().gridSpacing);
        ImGui::Text("Memory: 4.06 MB");
    }
}
```

**Controls to Expose:**
- Enable/disable toggle
- Rays per probe (16-256, default 64)
- Update interval (1-8 frames, default 4)
- Grid info display (read-only)

---

### ⏳ Task 11: Testing & Validation (PENDING)
**Status:** Not started
**Critical Tests:**

**1. Particle Count Scaling Test:**
- Start: 1000 particles
- Increment: +500 particles
- Watch for: Crossing 2045 threshold
- **Expected:** NO CRASH (unlike Volumetric ReSTIR)

**2. Performance Benchmark:**
```
2000 particles:  120+ FPS (baseline)
2045 particles:  120+ FPS (critical threshold - should NOT crash!)
5000 particles:  100+ FPS
10000 particles: 90-110 FPS (target)
```

**3. Visual Quality Assessment:**
- Compare screenshots vs inline RayQuery
- Check for smooth lighting gradients
- Verify no visible probe boundaries (trilinear interpolation working)
- Look for temporal stability (no flickering between updates)

**4. MCP Screenshot Comparison:**
```bash
# Capture before/after screenshots
F2 key in-app

# Use MCP tool for ML comparison
mcp__rtxdi-quality-analyzer__compare_screenshots_ml(
    before_path="/path/to/rayquery_baseline.bmp",
    after_path="/path/to/probe_grid.bmp"
)
```

**5. Stress Tests:**
- 2045 particles for 5 minutes (stability)
- Rapid particle count changes (1K → 10K → 1K)
- Camera movement stress (fast panning/zooming)

---

## Critical Files Reference

### Implementation Files (Modified/Created)
```
src/lighting/ProbeGridSystem.h          ✅ Complete
src/lighting/ProbeGridSystem.cpp        ⏳ UpdateProbes() stub needs completion
shaders/probe_grid/update_probes.hlsl   ✅ Complete
shaders/particles/particle_gaussian_raytrace.hlsl  ⏳ Needs SampleProbeGrid()
src/core/Application.h                  ⏳ Needs ProbeGridSystem member
src/core/Application.cpp                ⏳ Needs init + render integration
CMakeLists.txt                          ✅ Complete (shader build added)
```

### Documentation Files (For Reference)
```
PROBE_GRID_IMPLEMENTATION_OUTLINE.md    - Original 7-day plan
VOLUMETRIC_RESTIR_ATOMIC_CONTENTION_ANALYSIS.md  - Why we pivoted
VOLUMETRIC_RESTIR_RESOLUTION_FIX.md     - Logger fixes + shader staleness
VOLUMETRIC_SCATTERING_OPTIONS_RESEARCH.md  - RTXGI/RTX Volumetrics research
```

---

## Expected Outcomes

### Performance Targets
```
Particle Count | Current (ReSTIR) | Probe Grid Target | Status
---------------|------------------|-------------------|--------
1,000          | 120 FPS          | 120 FPS          | Baseline
2,044          | 120 FPS (freeze) | 120 FPS          | Critical test
2,045          | CRASH (TDR)      | 120 FPS          | ✨ SUCCESS METRIC
5,000          | N/A              | 100+ FPS         | Target
10,000         | N/A              | 90-110 FPS       | Target
```

### Quality Expectations
- **Smooth lighting gradients** - No visible probe boundaries
- **Temporally stable** - No flickering between 4-frame updates
- **Physically plausible** - Blackbody emission matches particle temperature
- **Comparable to RayQuery** - LPIPS similarity >0.85 (85% match)

### Validation Criteria
✅ No crashes at 2045+ particles (atomic contention eliminated)
✅ 90+ FPS at 10K particles (performance target met)
✅ Smooth lighting (trilinear interpolation working)
✅ Temporal stability (no flickering artifacts)

---

## Next Session Action Plan

### Immediate Tasks (Session Start)
1. **Complete UpdateProbes() dispatch** (~30 minutes)
   - Upload constants
   - Bind resources
   - Dispatch compute
   - Add UAV barrier

2. **Add SampleProbeGrid() to Gaussian shader** (~45 minutes)
   - Trilinear interpolation implementation
   - Constant buffer setup
   - Integration with existing lighting

3. **Wire into Application** (~30 minutes)
   - Initialize ProbeGridSystem
   - Call UpdateProbes() in render loop
   - Pass probe buffer to Gaussian renderer

### Testing Phase (After Integration)
4. **First test run** - 2045 particles, watch for crash
5. **Performance benchmark** - Measure FPS at 1K/2K/5K/10K particles
6. **Visual comparison** - Screenshot before/after, ML comparison

### Stretch Goals (If Time Permits)
7. **ImGui controls** - User-facing toggles and sliders
8. **Optimization pass** - Profile with PIX, identify bottlenecks
9. **Documentation update** - CLAUDE.md, README.md

---

## Recovery Instructions (If Context Lost)

**If you're reading this after auto-compact:**

1. **Read this document first** - Complete status snapshot
2. **Check branch:** Should be on `0.13.1`
3. **Verify build:** Run MSBuild, confirm clean compile
4. **Next task:** Implement `UpdateProbes()` dispatch in `ProbeGridSystem.cpp:252`
5. **Reference files:**
   - `VolumetricReSTIRSystem.cpp:854-918` (dispatch pattern)
   - `PROBE_GRID_IMPLEMENTATION_OUTLINE.md` (original plan)
   - This document (current status)

**Key Context:**
- **We pivoted from Volumetric ReSTIR** (atomic contention unfixable)
- **Probe Grid is 55% done** (infrastructure complete, needs integration)
- **Goal:** 90+ FPS at 10K particles (vs current crash at 2045)
- **Architecture:** Zero atomics = zero contention = scalable

---

## Success Metrics (Final Validation)

When implementation is complete, validate with these tests:

### Test 1: The 2045 Particle Test (Critical)
```bash
# Start application
./build/bin/Debug/PlasmaDX-Clean.exe

# Set particle count to 2045
# Enable Probe Grid in ImGui
# Watch for: NO CRASH, smooth rendering
# Expected: 120 FPS, stable frame times
```
**Pass Criteria:** No crash, no freeze, stable 120 FPS

### Test 2: Performance Scaling
```bash
# Test at: 1K, 2K, 5K, 10K particles
# Record FPS for each
# Plot scaling curve
```
**Pass Criteria:** 90+ FPS at 10K particles

### Test 3: Visual Quality (MCP Tool)
```bash
# Capture screenshot with probe grid
# Capture screenshot with inline RayQuery
# Run ML comparison
```
**Pass Criteria:** LPIPS similarity >0.85

---

**Current Branch:** 0.13.1
**Last Build:** 2025-11-03 21:15 ✅ SUCCESS
**Next Milestone:** Complete dispatch + integration (Tasks 7-9)
**Estimated Time to Completion:** 2-3 hours

