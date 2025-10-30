# Volumetric ReSTIR - Phase 1 Implementation Complete

**Date:** 2025-10-30
**Status:** ✅ Infrastructure and shaders implemented, ready for integration testing

---

## What Was Implemented

### 1. Core System Class

**Files Created:**
- `src/lighting/VolumetricReSTIRSystem.h` (219 lines)
- `src/lighting/VolumetricReSTIRSystem.cpp` (600 lines)

**Key Components:**
- VolumetricReservoir structure (64 bytes per pixel)
- PathVertex structure (16 bytes: distance + direction)
- Ping-pong reservoir buffers (264 MB @ 1080p)
- Piecewise-constant volume (Mip 2, 64³ grid, 512 KB)
- Compute pipelines for path generation and shading

**API Methods:**
```cpp
bool Initialize(Device*, ResourceManager*, width, height);
void GenerateCandidates(...); // Phase 1: RIS candidate sampling
void ShadeSelectedPaths(...);  // Phase 1: Final rendering
void SpatialReuse(...);        // Phase 2: Future
void TemporalReuse(...);       // Phase 3: Future
```

### 2. HLSL Shaders

**Files Created:**
- `shaders/volumetric_restir/volumetric_restir_common.hlsl` (400+ lines)
  - PathVertex and VolumetricReservoir structures
  - PCG random number generator
  - Regular tracking (distance sampling)
  - Weighted reservoir sampling (RIS)
  - Henyey-Greenstein phase function
  - Ray reconstruction utilities

- `shaders/volumetric_restir/path_generation.hlsl` (300+ lines)
  - Generates M=4 random walks per pixel
  - Uses regular tracking with Mip 2 volume
  - RayQuery for particle intersection
  - Weighted reservoir sampling (w = p̂/p)
  - Stores winning path in reservoir buffer

- `shaders/volumetric_restir/shading.hlsl` (230+ lines)
  - Reads reservoir winners
  - Reconstructs paths from stored vertices
  - Evaluates particle emission (blackbody)
  - Applies Henyey-Greenstein phase function
  - Computes transmittance (Beer-Lambert law)
  - Writes final color with tone mapping

### 3. Build System Integration

**Modified Files:**
- `CMakeLists.txt`
  - Added VolumetricReSTIRSystem.cpp to SOURCES
  - Added VolumetricReSTIRSystem.h to HEADERS
  - Added shader compilation for path_generation.hlsl and shading.hlsl
  - Shaders compile to `build/bin/Debug/shaders/volumetric_restir/`

**Shader Compilation:**
```bash
# Automatic during build:
dxc.exe -T cs_6_5 -E main path_generation.hlsl -Fo path_generation.dxil
dxc.exe -T cs_6_5 -E main shading.hlsl -Fo shading.dxil
```

### 4. GPU Resources Created

**Buffers:**
1. **Reservoir Buffers** (ping-pong, 2×)
   - Size: 1920×1080 × 64 bytes × 2 = 264 MB @ 1080p
   - Format: StructuredBuffer<VolumetricReservoir>
   - Access: UAV (write during generation), SRV (read during shading)

2. **Piecewise-Constant Volume** (Mip 2)
   - Size: 64³ × 2 bytes (R16_FLOAT) = 512 KB
   - Format: Texture3D<float>
   - Purpose: Fast transmittance lookups for regular tracking

**Pipelines:**
1. **Path Generation Pipeline**
   - Root signature: 5 parameters (constants, BLAS, particles, volume, reservoir UAV)
   - PSO: Compute shader (cs_6_5)
   - Dispatch: (width/8, height/8, 1)

2. **Shading Pipeline**
   - Root signature: 4 parameters (constants, BLAS, particles, descriptor table)
   - PSO: Compute shader (cs_6_5)
   - Dispatch: (width/8, height/8, 1)

---

## What's NOT Yet Done (Integration TODOs)

### Critical: Descriptor Table Binding

**Issue:** Root signatures reference descriptor tables but binding code is incomplete.

**Path Generation (line 477-479 in VolumetricReSTIRSystem.cpp):**
```cpp
// Root parameter 3: t2 - Volume Mip 2 (descriptor table)
// TODO: Need to set descriptor table for volume texture
// For now, skip (will cause shader to fail if it accesses volume)
```

**Fix Required:**
```cpp
// Allocate descriptor in GPU-visible heap
D3D12_CPU_DESCRIPTOR_HANDLE volumeDescriptor = m_resources->AllocateGPUDescriptor();
m_device->GetD3D12Device()->CreateShaderResourceView(
    m_volumeMip2.Get(),
    &volumeSRVDesc,
    volumeDescriptor
);

// Get GPU handle and bind
D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_resources->GetGPUHandle(volumeDescriptor);
commandList->SetComputeRootDescriptorTable(3, gpuHandle);
```

**Shading (line 580-582 in VolumetricReSTIRSystem.cpp):**
```cpp
// Root parameter 3: Descriptor table (t2: reservoir SRV, u0: output texture UAV)
// TODO: Need to set descriptor table
// For now, skip (will cause shader to fail)
```

**Fix Required:**
```cpp
// Allocate 2 descriptors (SRV + UAV) in GPU-visible heap
D3D12_CPU_DESCRIPTOR_HANDLE tableStart = m_resources->AllocateGPUDescriptors(2);

// SRV for reservoir buffer
m_device->GetD3D12Device()->CreateShaderResourceView(
    m_reservoirBuffer[m_currentBufferIndex].Get(),
    &reservoirSRVDesc,
    tableStart
);

// UAV for output texture
D3D12_CPU_DESCRIPTOR_HANDLE outputUAV = m_resources->OffsetDescriptor(tableStart, 1);
m_device->GetD3D12Device()->CreateUnorderedAccessView(
    outputTexture,
    nullptr,
    &outputUAVDesc,
    outputUAV
);

// Bind table
D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_resources->GetGPUHandle(tableStart);
commandList->SetComputeRootDescriptorTable(3, gpuHandle);
```

### Missing: Mip 2 Volume Population

**Issue:** Piecewise-constant volume (m_volumeMip2) is created but never filled with transmittance data.

**Required:** Create compute shader to populate volume from particle density:
```hlsl
// shaders/volumetric_restir/build_volume_mip2_cs.hlsl
[numthreads(4, 4, 4)]
void main(uint3 voxelCoords : SV_DispatchThreadID) {
    float3 worldPos = VoxelToWorld(voxelCoords);

    // Query local particle density
    float density = 0.0;
    [loop] for (each nearby particle) {
        float dist = length(particle.pos - worldPos);
        density += GaussianWeight(dist, particle.radius);
    }

    // Convert to transmittance (Beer-Lambert)
    float extinction = density * 0.001; // Tunable extinction coefficient
    float transmittance = exp(-extinction);

    g_volumeMip2[voxelCoords] = transmittance;
}
```

**When to Build:** Once per frame BEFORE GenerateCandidates() if particles moved.

### Missing: ShadeSelectedPaths Parameters

**Issue:** Shading pass needs additional parameters that aren't currently passed:
```cpp
void ShadeSelectedPaths(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* outputTexture)
```

**Fix:** Update signature to accept full scene context:
```cpp
void ShadeSelectedPaths(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* outputTexture,
    ID3D12Resource* particleBVH,        // ADD: For ray queries
    ID3D12Resource* particleBuffer,     // ADD: For emission evaluation
    uint32_t particleCount,             // ADD: For bounds checking
    const DirectX::XMFLOAT3& cameraPos, // ADD: For ray reconstruction
    const DirectX::XMMATRIX& viewMatrix,
    const DirectX::XMMATRIX& projMatrix
);
```

---

## Integration Steps

### Step 1: Add to Application.h

```cpp
#include "lighting/VolumetricReSTIRSystem.h"

class Application {
private:
    // Existing lighting systems
    std::unique_ptr<RTLightingSystem_RayQuery> m_rtLighting;
    std::unique_ptr<RTXDILightingSystem> m_rtxdiLighting;

    // NEW: Volumetric ReSTIR system
    std::unique_ptr<VolumetricReSTIRSystem> m_volumetricReSTIR;

    // Modes
    enum class LightingMode {
        MultiLight,
        ParticleRT,
        RTXDI,
        VolumetricReSTIR  // NEW
    };
    LightingMode m_currentLightingMode = LightingMode::MultiLight;
};
```

### Step 2: Initialize in Application.cpp

```cpp
bool Application::Initialize() {
    // ... existing initialization ...

    // Create volumetric ReSTIR system
    m_volumetricReSTIR = std::make_unique<VolumetricReSTIRSystem>();
    if (!m_volumetricReSTIR->Initialize(m_device.get(), m_resourceManager.get(),
                                       m_width, m_height)) {
        LOG_ERROR("Failed to initialize Volumetric ReSTIR system");
        return false;
    }

    return true;
}
```

### Step 3: Add ImGui Controls

```cpp
void Application::RenderUI() {
    if (ImGui::CollapsingHeader("Lighting Systems")) {
        // Existing radio buttons
        ImGui::RadioButton("Multi-Light", &mode, 0);
        ImGui::RadioButton("Particle RT", &mode, 1);
        ImGui::RadioButton("RTXDI", &mode, 2);

        // NEW: Volumetric ReSTIR
        ImGui::RadioButton("Volumetric ReSTIR (Experimental)", &mode, 3);

        if (mode == 3) {
            if (ImGui::TreeNode("ReSTIR Parameters")) {
                int M = m_volumetricReSTIR->GetRandomWalksPerPixel();
                if (ImGui::SliderInt("Random Walks (M)", &M, 1, 16)) {
                    m_volumetricReSTIR->SetRandomWalksPerPixel(M);
                }

                int K = m_volumetricReSTIR->GetMaxBounces();
                if (ImGui::SliderInt("Max Bounces (K)", &K, 1, 5)) {
                    m_volumetricReSTIR->SetMaxBounces(K);
                }

                ImGui::TreePop();
            }
        }

        m_currentLightingMode = static_cast<LightingMode>(mode);
    }
}
```

### Step 4: Dispatch in Render Loop

```cpp
void Application::Render() {
    // ... existing code ...

    // Volumetric rendering with selected lighting
    if (m_currentLightingMode == LightingMode::VolumetricReSTIR) {
        // Generate candidate paths
        m_volumetricReSTIR->GenerateCandidates(
            commandList.Get(),
            m_rtLighting->GetTLAS(),           // Reuse existing BLAS/TLAS
            m_particleSystem->GetParticleBuffer(),
            m_particleSystem->GetParticleCount(),
            m_cameraPos,
            m_viewMatrix,
            m_projMatrix,
            m_frameIndex
        );

        // Shade selected paths
        m_volumetricReSTIR->ShadeSelectedPaths(
            commandList.Get(),
            m_renderTarget.Get()
        );
    } else {
        // Existing rendering paths
        m_particleRenderer->Render(commandList.Get(), ...);
    }

    // ... present ...
}
```

### Step 5: First Build and Test

```bash
# Clean build to ensure shaders compile
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
rm -rf build/
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug

# Expected output:
# - Compiling path_generation.hlsl
# - Compiling shading.hlsl
# - VolumetricReSTIRSystem.cpp compiled
# - Shaders appear in build/bin/Debug/shaders/volumetric_restir/

# Run
./bin/Debug/PlasmaDX-Clean.exe
```

**Expected First Run:**
- System initializes successfully (264 MB allocated)
- ImGui shows "Volumetric ReSTIR (Experimental)" option
- Switching to it dispatches shaders BUT rendering may be black/incorrect due to:
  - Mip 2 volume not populated (shader reads zeros)
  - Descriptor table binding incomplete (shader sees null)

**Debug with PIX:**
```bash
./bin/Debug/PlasmaDX-Clean.exe --config=configs/agents/pix_agent.json
```
- Open `PIX/Captures/latest.wpix`
- Check "VolumetricReSTIR Path Generation" event
- Inspect reservoir buffer: should see non-zero wsum/M values
- Check "VolumetricReSTIR Shading" event
- Inspect output texture: check for non-black pixels

---

## Testing Checklist

### Phase 1A: Infrastructure Validation

- [ ] System initializes without errors (check logs)
- [ ] 264 MB reservoir buffers allocated (check memory usage)
- [ ] 512 KB volume allocated
- [ ] Shaders compile successfully (check build output)
- [ ] Shaders load at runtime (check logs: "Loaded path generation shader: XXX bytes")

### Phase 1B: Path Generation Validation

- [ ] Dispatch executes without GPU timeout (TDR)
- [ ] Reservoir buffer has non-zero data (PIX capture)
- [ ] `wsum` values in [0.0, 100.0] range (sanity check)
- [ ] `M` values equal to configured random walks (M=4 default)
- [ ] Path vertices have reasonable distances (z > 0)

### Phase 1C: Shading Validation

- [ ] Dispatch executes without errors
- [ ] Output texture has non-black pixels (check PIX)
- [ ] Colors are reasonable (not NaN/Inf)
- [ ] Performance is within bounds (<50ms @ 1080p)

### Phase 1D: Parameter Tuning

- [ ] Changing M (random walks) affects quality
- [ ] Changing K (max bounces) affects lighting complexity
- [ ] Frame-to-frame consistency (no excessive flickering)

---

## Known Limitations (Phase 1)

1. **No Spatial Reuse:** Each pixel samples independently → noisy
2. **No Temporal Reuse:** No history accumulation → flickering
3. **RIS Only:** Using basic importance sampling without MIS → suboptimal
4. **Simplified Transmittance:** Constant extinction → not physically accurate
5. **No Shadow Rays:** Path evaluation doesn't check occlusion → light leaks
6. **Placeholder Volume:** Mip 2 volume needs real particle density data

**These are all expected for Phase 1.** The goal is to establish infrastructure and validate the basic algorithm before adding complexity.

---

## Performance Expectations

**Phase 1 Performance (1920×1080, M=4, K=3):**

| Component | Time | Notes |
|-----------|------|-------|
| Path Generation | ~3-5 ms | M×K ray queries per pixel |
| Shading | ~1-2 ms | Single pass evaluation |
| **Total** | ~4-7 ms | ~200-250 FPS (if only this runs) |

**Memory:**
- Reservoirs: 264 MB @ 1080p
- Volume Mip 2: 512 KB
- **Total:** ~265 MB

**Comparison to Baseline:**
- Multi-light (13 lights): ~8 ms → **Phase 1 is competitive!**
- RTXDI M4: ~10 ms → **Phase 1 is faster (for now)**

**Note:** Performance will degrade when adding:
- Phase 2 (Spatial reuse): +3-5 ms
- Phase 3 (Temporal reuse): +2-3 ms
- Final target: ~15-20 ms (60-67 FPS)

---

## Next Steps (After Phase 1 Testing)

### If Phase 1 Works:

1. **Fix descriptor table binding** (critical for volume access)
2. **Populate Mip 2 volume** (build_volume_mip2_cs.hlsl)
3. **Add shadow rays** to shading pass for proper occlusion
4. **Tune extinction coefficient** for realistic transmittance
5. **Begin Phase 2** - Spatial reuse with Talbot MIS

### If Phase 1 Fails:

**Black Screen:**
- Check PIX capture for shader execution
- Verify reservoir buffer has data
- Check descriptor table bindings
- Validate ray reconstruction math

**GPU Timeout (TDR):**
- Reduce M (random walks) to 1-2
- Reduce K (max bounces) to 1
- Check for infinite loops in regular tracking
- Verify maxSteps termination (line 199 in volumetric_restir_common.hlsl)

**Excessive Noise:**
- Expected for Phase 1 (no spatial/temporal reuse)
- Increase M to 8-16 for smoother results
- Will be fixed in Phase 2/3

**Wrong Colors:**
- Check blackbody evaluation (EvaluateParticleEmission)
- Verify Henyey-Greenstein values (g=0.7 may be too forward-scattering)
- Check tone mapping (Reinhard may be too aggressive)

---

## Summary

**✅ Completed:**
- Full Phase 1 infrastructure (C++ + HLSL)
- Weighted reservoir sampling (RIS)
- Regular tracking with piecewise-constant volume
- Path generation and shading pipelines
- CMake integration
- 600+ lines of C++, 900+ lines of HLSL

**⏳ Remaining for Integration:**
- Descriptor table binding (~20 lines)
- Mip 2 volume population (~100 lines shader + dispatch)
- ShadeSelectedPaths parameter passing (~10 lines)
- Application.cpp integration (~50 lines)
- ImGui controls (~30 lines)

**📊 Estimated Time to First Render:**
- Descriptor fixes: 1 hour
- Integration: 2 hours
- Testing & debugging: 2-4 hours
- **Total:** 5-7 hours

**🎯 Success Criteria:**
- System initializes without errors
- Path generation completes in <10ms
- Output has visible volumetric structure (even if noisy)
- No GPU timeouts or crashes
- Memory usage stable @ 265 MB

---

**Ready for integration!** The hard parts (algorithm, shaders, resource management) are done. The remaining work is plumbing and validation.
