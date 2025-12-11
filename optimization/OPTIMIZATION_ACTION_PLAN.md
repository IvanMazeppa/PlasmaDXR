# Multi-Agent Optimization Action Plan
**Generated:** 2025-12-06
**Target:** PlasmaDX-Clean RT Volumetric Renderer
**Current Performance:** 142 FPS @ 1080p, 100K particles
**Target Performance:** 280+ FPS with PINN ML Physics

---

## Executive Summary

Multi-agent performance analysis identified **4 critical optimization domains** with projected **+97% FPS improvement** (142 → 280 FPS). All 4 optimization agents completed successfully with 92.9% cost efficiency.

**Key Findings:**
- ✅ No stale shaders detected (good build hygiene)
- ⚠️ BLAS rebuild bottleneck: 2.1ms/frame (primary target)
- ⚠️ Memory bandwidth inefficiencies: 8 UAV barriers/frame
- ✅ Context window usage reasonable: 6.9K tokens (CLAUDE.md)

---

## Optimization Priorities (High → Low Impact)

### 🎯 Priority 1: BLAS/TLAS Acceleration Structure Optimization
**Projected Gain:** +41 FPS (+29%)
**Domain:** Raytracing Traversal
**Status:** ✅ IMPLEMENTED (Phases 1 & 2 Complete)

> **Implementation Notes:**
> - Phase 1 (BLAS Update): Implemented 2025-12-08 - see `IMPLEMENTED_2025-12-08.md`
> - Phase 2 (Frustum Culling): Implemented 2025-12-11 - see `FRUSTUM_CULLING_IMPLEMENTATION.md`

#### Current Bottleneck
- Full BLAS rebuild every frame: 2.1ms
- 100K procedural primitives (AABBs) regenerated each frame
- TLAS rebuild includes all instances (no culling)

#### Optimization Strategy

**Phase 1: BLAS Update (Week 1)** ✅ DONE 2025-12-08
```cpp
// src/lighting/RTLightingSystem.cpp
D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

// First build: Full build
BuildRaytracingAccelerationStructure(..., buildFlags, ...);

// Subsequent frames: Update
buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
BuildRaytracingAccelerationStructure(..., buildFlags, ..., pSourceAccelerationStructure);
```

**Expected:** 2.1ms → 0.8ms (-62%), +20 FPS

**Phase 2: Frustum Culling (Week 2)** ✅ DONE 2025-12-11
```cpp
// GPU-side frustum culling in AABB generation shader
// Particles outside frustum output degenerate AABBs (min > max)
// See: shaders/dxr/generate_particle_aabbs.hlsl
// See: FRUSTUM_CULLING_IMPLEMENTATION.md for full details
```

**Expected:** Additional +21 FPS (+15%), **Total: +41 FPS**

**Phase 3: Distance-Based LOD (Week 3)**
- Far particles (>500m): Reduce density 50%
- Mid particles (200-500m): Reduce density 25%
- Near particles (<200m): Full density

**Implementation Files:**
- `src/lighting/RTLightingSystem.h/cpp`
- `src/particles/ParticleRenderer_Gaussian.cpp`
- `shaders/dxr/generate_particle_aabbs.hlsl`

---

### 🎯 Priority 2: Memory Bandwidth Optimization
**Projected Gain:** +8-12 FPS (+6-8%)
**Domain:** Memory Bandwidth
**Status:** 🔄 Ready to Implement

#### Current Bottleneck
- 8 UAV barriers per frame (unnecessary GPU stalls)
- Froxel grid: 7 MB (R32_FLOAT) → can halve to 3.6 MB (R16_FLOAT)
- Froxel density injection has race condition (`+=` not atomic)

#### Optimization Strategy

**Phase 1: UAV Barrier Batching**
```cpp
// Instead of individual barriers
D3D12_RESOURCE_BARRIER barriers[] = {
    CD3DX12_RESOURCE_BARRIER::UAV(m_froxelDensity.Get()),
    CD3DX12_RESOURCE_BARRIER::UAV(m_froxelLighting.Get()),
    CD3DX12_RESOURCE_BARRIER::UAV(m_particleBuffer.Get())
};
m_commandList->ResourceBarrier(_countof(barriers), barriers);
```

**Expected:** 8 barriers → 4 barriers, +3-5 FPS

**Phase 2: Froxel Format Optimization**
```hlsl
// shaders/froxel/inject_density.hlsl
RWTexture3D<float16_t> g_froxelDensity;  // Was R32_FLOAT, now R16_FLOAT
```

**Expected:** 7 MB → 3.6 MB (-49% memory), +2-3 FPS

**Phase 3: Atomic Density Injection**
```hlsl
// Fix race condition in inject_density.hlsl
uint packedDensity = asuint(densityToAdd);
InterlockedAdd(g_froxelDensityAtomic[voxelIndex], packedDensity);
```

**Expected:** Eliminate visual artifacts, +2-4 FPS

**Implementation Files:**
- `src/rendering/FroxelSystem.h/cpp`
- `shaders/froxel/inject_density.hlsl`
- `shaders/froxel/light_voxels.hlsl`

---

### 🎯 Priority 3: PINN ML Physics Integration
**Projected Gain:** +80-100 FPS (+56-70%)
**Domain:** Physics Simulation
**Status:** 🔄 Python ✅ Complete, C++ Integration Pending

#### Current Status
- Python training complete: `ml/models/pinn_accretion_disk.onnx` (50K params)
- Network: 7D input → 5×128 hidden (Tanh) → 3D force output
- Inference speed: 5-10× faster than GPU compute shader

#### Integration Strategy

**Phase 1: ONNX Runtime Integration (Week 1)**
```cpp
// src/ml/PINNPhysicsSystem.h
class PINNPhysicsSystem {
    Ort::Session* m_session;
    Ort::MemoryInfo m_memoryInfo;

    void Initialize(const std::string& modelPath);
    void InferForces(const ParticleState* input, float3* forces, size_t count);
};
```

**Phase 2: Hybrid Mode (Week 2)**
```cpp
// src/particles/ParticleSystem.cpp
void ParticleSystem::UpdatePhysics(float deltaTime) {
    if (m_config.enablePINN) {
        // PINN for far particles (r > 100m)
        m_pinnSystem->InferForces(m_farParticles, m_farForces, m_farCount);

        // Traditional shader for near ISCO (high precision needed)
        RunPhysicsShader(m_nearParticles, m_nearCount);
    } else {
        RunPhysicsShader(m_allParticles, m_particleCount);
    }
}
```

**Expected:** +80-100 FPS at 100K particles, **Total FPS: 220-240**

**Implementation Files:**
- `src/ml/PINNPhysicsSystem.h/cpp` (NEW)
- `src/particles/ParticleSystem.h/cpp`
- `CMakeLists.txt` (ONNX Runtime linkage)

---

### 🎯 Priority 4: Shader Build System Hardening
**Projected Gain:** 0 FPS (Quality of Life)
**Domain:** Developer Experience
**Status:** ✅ No immediate issues, but preventative measures recommended

#### Current Status
- ✅ No stale shaders detected (good build hygiene)
- ⚠️ Manual dxc compilation still required occasionally
- ⚠️ No pre-commit validation for shader staleness

#### Hardening Strategy

**Phase 1: CMake Timestamp Validation**
```cmake
# CMakeLists.txt
add_custom_command(
    OUTPUT ${DXIL_OUTPUT}
    COMMAND ${DXC_EXE} -T cs_6_5 -E main ${HLSL_INPUT} -Fo ${DXIL_OUTPUT}
    DEPENDS ${HLSL_INPUT}
    COMMENT "Compiling shader: ${HLSL_INPUT}"
)
# CMake will auto-rebuild if .hlsl is newer
```

**Phase 2: Pre-Build Validation Hook**
```python
# tools/validate_shaders.py
def check_stale_shaders():
    for hlsl in Path("shaders").rglob("*.hlsl"):
        dxil = Path("build/bin/Debug/shaders") / hlsl.relative_to("shaders").with_suffix(".dxil")
        if dxil.exists() and hlsl.stat().st_mtime > dxil.stat().st_mtime:
            raise RuntimeError(f"Stale shader: {hlsl}")
```

**Phase 3: Runtime Validation (Debug builds only)**
```cpp
// src/utils/ShaderManager.cpp (Debug only)
#ifdef _DEBUG
void ShaderManager::LoadShader(const std::string& path) {
    auto hlslPath = std::filesystem::path(path).replace_extension(".hlsl");
    auto dxilPath = std::filesystem::path(path);

    if (std::filesystem::last_write_time(hlslPath) > std::filesystem::last_write_time(dxilPath)) {
        LogWarning("Stale shader detected: " + path);
    }
}
#endif
```

**Implementation Files:**
- `CMakeLists.txt`
- `tools/validate_shaders.py` (NEW)
- `src/utils/ShaderManager.cpp`

---

### 🎯 Priority 5: Context Window Optimization
**Projected Gain:** 0 FPS (AI Development Efficiency)
**Domain:** Claude Code Context Management
**Status:** ✅ Currently healthy (6.9K tokens), proactive optimization recommended

#### Current Status
- CLAUDE.md: 27 KB (~6.9K tokens)
- ✅ Below 10K token target
- ⚠️ Large historical docs still in main directory

#### Optimization Strategy

**Phase 1: Archive Completed Phases**
```bash
mkdir -p docs/archive/completed_phases
mv PARTICLE_FLASHING_ROOT_CAUSE_ANALYSIS.md docs/archive/
mv PCSS_IMPLEMENTATION_SUMMARY.md docs/archive/completed_phases/
mv DYNAMIC_EMISSION_IMPLEMENTATION.md docs/archive/completed_phases/
```

**Phase 2: Compress Historical Context**
- `PARTICLE_FLASHING_ROOT_CAUSE_ANALYSIS.md` (14K words) → Executive summary (500 words)
- Keep detailed analysis in archive with link

**Phase 3: MCP Server Delegation**
- Use `dxr-shadow-engineer` MCP server for shadow-related queries
- Use `gaussian-analyzer` for particle rendering questions
- Reduce shadow/Gaussian implementation details in CLAUDE.md

**Expected:** 6.9K → 5K tokens (-27%), faster Claude Code responses

**Implementation Files:**
- `CLAUDE.md`
- `docs/archive/` (NEW directory structure)

---

## Implementation Timeline

### Week 1: Quick Wins
- [x] Multi-agent analysis (COMPLETED)
- [ ] BLAS update implementation
- [ ] UAV barrier batching
- [ ] Froxel R16 format optimization
- **Expected:** 142 FPS → 165 FPS (+23 FPS)

### Week 2: Medium-Complexity Optimizations
- [ ] Frustum culling for TLAS
- [ ] Atomic froxel density injection
- [ ] ONNX Runtime integration (PINN)
- **Expected:** 165 FPS → 210 FPS (+45 FPS)

### Week 3: Advanced Optimizations
- [ ] Hybrid PINN + traditional physics
- [ ] Distance-based LOD system
- [ ] Shader build system hardening
- **Expected:** 210 FPS → 280 FPS (+70 FPS)

### Week 4: Validation & Tuning
- [ ] Performance regression testing
- [ ] Quality validation (visual artifacts)
- [ ] Context window cleanup
- [ ] Documentation updates
- **Target:** 280+ FPS sustained ✅

---

## Monitoring & Validation

### Performance Metrics Dashboard
```bash
# Real-time monitoring
python optimization/performance_dashboard.py /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean --monitor

# Analyze optimization report
python optimization/performance_dashboard.py . --analyze-report optimization/reports/optimization_report_*.json
```

### Validation Checklist
- [ ] FPS targets met (see timeline above)
- [ ] Visual quality unchanged (no new artifacts)
- [ ] GPU memory usage stable (<8 GB)
- [ ] No device removal errors
- [ ] PIX captures show expected optimization (BLAS update vs rebuild)
- [ ] PINN physics matches traditional accuracy (max 2% deviation)

### Rollback Plan
Each optimization phase should be implemented in separate feature branches:
- `optimize/blas-update`
- `optimize/memory-bandwidth`
- `optimize/pinn-integration`

If performance regresses, revert specific branch and investigate.

---

## Cost-Benefit Analysis

| Optimization | Dev Time | FPS Gain | Complexity | Risk |
|--------------|----------|----------|------------|------|
| BLAS Update | 1 week | +20 FPS | Medium | Low |
| Frustum Culling | 1 week | +21 FPS | Medium | Low |
| UAV Batching | 2 days | +5 FPS | Low | Very Low |
| Froxel R16 | 1 day | +3 FPS | Low | Low |
| PINN Integration | 2 weeks | +90 FPS | High | Medium |
| Distance LOD | 1 week | +15 FPS | Medium | Low |

**Total Dev Time:** 6-7 weeks
**Total FPS Gain:** +138 FPS (142 → 280)
**ROI:** 19.7 FPS/week

---

## Success Criteria

### Mandatory
✅ Achieve 280 FPS @ 1080p, 100K particles
✅ Visual quality maintained (no new artifacts)
✅ No device removal errors
✅ PINN physics accuracy within 2% of traditional

### Stretch Goals
🎯 300 FPS with DLSS Performance mode
🎯 Support 200K particles @ 165 FPS
🎯 Context window <5K tokens
🎯 Zero stale shader incidents

---

## Multi-Agent Token Budget

**Total Budget:** 50,000 tokens
**Used (Analysis):** 3,550 tokens (7.1%)
**Reserved (Implementation):** 20,000 tokens (40%)
**Available:** 26,450 tokens (52.9%)

**Cost Efficiency:** 92.9% ✅

---

## References

- Multi-Agent Optimization Report: `optimization/reports/optimization_report_1765015743.json`
- Performance Baselines: `CLAUDE.md` (Performance Targets section)
- BLAS Optimization: DirectX 12 Raytracing Spec (Section 4.5.2)
- PINN Training: `ml/PINN_README.md`
- MCP Servers: `agents/*/README.md`

---

**Generated by:** Multi-Agent Optimization Toolkit v1.0
**Next Review:** After Week 1 implementation
**Contact:** Ben (Project Lead)
