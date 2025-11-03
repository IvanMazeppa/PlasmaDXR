# Probe Grid - Quick Start Guide

**Branch:** 0.13.1 | **Status:** 55% Complete | **Next:** UpdateProbes() dispatch

---

## TL;DR

**Problem:** Volumetric ReSTIR crashes at 2045 particles (atomic contention)
**Solution:** Probe grid (zero atomics, zero contention)
**Status:** Infrastructure done, needs dispatch + integration
**ETA:** 2-3 hours to completion

---

## What's Done ✅

1. ✅ ProbeGridSystem.h/cpp (class structure)
2. ✅ Probe struct (128 bytes, SH L2 ready)
3. ✅ GPU buffers (4.06 MB, 32,768 probes)
4. ✅ Descriptors (SRV/UAV allocated)
5. ✅ update_probes.hlsl (Fibonacci sphere + RayQuery)
6. ✅ Root signature + PSO (5-parameter binding)

**Build Status:** ✅ Compiles cleanly (2025-11-03 21:15)

---

## What's Next ⏳

### Task 7: Complete UpdateProbes() Dispatch
**File:** `src/lighting/ProbeGridSystem.cpp:252-268`
**Pattern:** Copy from `VolumetricReSTIRSystem.cpp:854-918`

```cpp
void ProbeGridSystem::UpdateProbes(...) {
    // 1. Upload constants to m_updateConstantBuffer
    // 2. SetPipelineState + SetComputeRootSignature
    // 3. Bind 5 resources (CB, particles, lights, TLAS, probes)
    // 4. Dispatch(4, 4, 4)
    // 5. UAV barrier
}
```

### Task 8: Add SampleProbeGrid() to Gaussian Shader
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl`

```hlsl
// Add:
cbuffer ProbeGridParams : register(b4) { ... }
StructuredBuffer<Probe> g_probeGrid : register(t7);

float3 SampleProbeGrid(float3 worldPos) {
    // Trilinear interpolation between 8 nearest probes
}
```

### Task 9: Wire into Application
**Files:** `src/core/Application.h/cpp`

```cpp
// Application.h: Add member
std::unique_ptr<ProbeGridSystem> m_probeGridSystem;

// Application.cpp: Initialize + call UpdateProbes()
```

---

## Test Plan

1. **2045 Particle Test** - Should NOT crash (unlike ReSTIR)
2. **10K Performance** - Target 90-110 FPS
3. **Visual Quality** - LPIPS >0.85 vs RayQuery baseline

---

## Quick Reference

**Probe Grid Parameters:**
- Grid: 32³ = 32,768 probes
- World: [-1500, +1500] per axis
- Spacing: 93.75 units
- Rays: 64 per probe
- Update: 1/4 probes per frame

**Key Architecture:**
- Zero atomics → zero contention
- Each probe owns memory slot
- Trilinear particle queries
- 0.7-1.3ms total overhead

---

## Files to Edit (Next Steps)

```
MUST EDIT:
✏️ src/lighting/ProbeGridSystem.cpp (complete UpdateProbes)
✏️ shaders/particles/particle_gaussian_raytrace.hlsl (add SampleProbeGrid)
✏️ src/core/Application.h (add member)
✏️ src/core/Application.cpp (init + render loop)

REFERENCE:
📖 PROBE_GRID_STATUS_REPORT.md (full details)
📖 PROBE_GRID_IMPLEMENTATION_OUTLINE.md (original plan)
📖 VolumetricReSTIRSystem.cpp (dispatch pattern)
```

---

**If context lost:** Read PROBE_GRID_STATUS_REPORT.md first!
