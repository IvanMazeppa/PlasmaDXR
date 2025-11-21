# Froxel Volumetric Fog Integration Status - 2025-11-21

## Executive Summary

**Status:** Core infrastructure complete (60% done), runtime integration pending

**Performance Goal:** 21 FPS (god rays) → ~100 FPS (froxels)
**Technique:** 3-pass froxel grid (density injection → voxel lighting → grid sampling)

---

## ✅ Completed (Phase 1: Infrastructure)

### 1. Core Files Created

All core shader and C++ files are implemented and compile successfully:

**Shaders:**
- ✅ `shaders/froxel/inject_density.hlsl` - Pass 1: Particle density injection (256 threads/group)
- ✅ `shaders/froxel/light_voxels.hlsl` - Pass 2: Voxel lighting (8×8×8 threads/group)
- ✅ `shaders/froxel/sample_froxel_grid.hlsl` - Pass 3: Grid sampling helper (included in Gaussian shader)

**C++ Infrastructure:**
- ✅ `src/rendering/FroxelSystem.h` - Complete class with full API
- ✅ `src/rendering/FroxelSystem.cpp` - Pipeline states, resource creation, descriptors

**Build System:**
- ✅ Added to `CMakeLists.txt` (sources, headers, shader compilation)
- ✅ Build verified: **Zero compilation errors**

### 2. FroxelSystem Implementation Details

**Pipeline States Created:**
```cpp
// Root Signature: Inject Density
//   b0: FroxelParams constant buffer
//   t0: Particle buffer (SRV)
//   u0: Density grid (UAV)
ComPtr<ID3D12RootSignature> m_injectDensityRootSig;
ComPtr<ID3D12PipelineState> m_injectDensityPSO;

// Root Signature: Light Voxels
//   b0: FroxelParams constant buffer
//   t0: Density grid (SRV)
//   t1: Light buffer (SRV)
//   t2: Particle BVH (acceleration structure)
//   u0: Lighting grid (UAV)
ComPtr<ID3D12RootSignature> m_lightVoxelsRootSig;
ComPtr<ID3D12PipelineState> m_lightVoxelsPSO;
```

**Resources Created:**
```cpp
// 3D Textures (160×90×64 = 921,600 voxels)
ComPtr<ID3D12Resource> m_densityGrid;   // R16_FLOAT (~1.76 MB)
ComPtr<ID3D12Resource> m_lightingGrid;  // R16G16B16A16_FLOAT (~7.04 MB)

// Descriptor Views (4 total)
D3D12_CPU_DESCRIPTOR_HANDLE m_densityGridUAV;  // Write in Pass 1
D3D12_CPU_DESCRIPTOR_HANDLE m_densityGridSRV;  // Read in Pass 2
D3D12_CPU_DESCRIPTOR_HANDLE m_lightingGridUAV; // Write in Pass 2
D3D12_CPU_DESCRIPTOR_HANDLE m_lightingGridSRV; // Read in Pass 3 (Gaussian shader)

// Constant Buffer (persistently mapped)
ComPtr<ID3D12Resource> m_constantBuffer;
void* m_constantBufferMapped;  // FroxelSystem::GridParams
```

**Grid Configuration:**
```cpp
struct GridParams {
    DirectX::XMFLOAT3 gridMin;      // [-1500, -1500, -1500]
    DirectX::XMFLOAT3 gridMax;      // [1500, 1500, 1500]
    DirectX::XMUINT3 gridDimensions; // [160, 90, 64] = 921,600 voxels
    float densityMultiplier;         // Global fog density scale
    DirectX::XMFLOAT3 voxelSize;    // Computed: (gridMax - gridMin) / gridDimensions
};
```

### 3. Application.h Integration

**Member Variable Added:**
```cpp
std::unique_ptr<FroxelSystem> m_froxelSystem;  // Froxel volumetric fog (Phase 5 - replaces god rays)
```

**Runtime Controls Added:**
```cpp
bool m_enableFroxelFog = false;          // Toggle froxel volumetric fog (F7 to toggle)
float m_froxelDensityMultiplier = 1.0f;  // Fog density multiplier (0.1-5.0)
bool m_debugFroxelVisualization = false; // Debug visualization mode
```

**Include Added:**
```cpp
#include "../rendering/FroxelSystem.h"
```

---

## ⏳ Pending (Phase 2: Runtime Integration)

### 4. Initialize FroxelSystem in Application.cpp

**Location:** `Application::Initialize()` (around line 200-300, after other subsystems)

**Code to Add:**
```cpp
// === Initialize Froxel Volumetric Fog System (Phase 5) ===
m_froxelSystem = std::make_unique<FroxelSystem>(m_device.get(), m_resources.get());
if (!m_froxelSystem->Initialize(m_width, m_height)) {
    LOG_ERROR("Failed to initialize Froxel system");
    return false;
}
LOG_INFO("Froxel system initialized (160×90×64 voxels, 921,600 total)");
```

**Why This Location:**
- After Device, ResourceManager, and ParticleSystem initialization
- Before renderer initialization (Gaussian renderer will need to bind froxel grid SRV)
- Matches pattern of other subsystems (ProbeGridSystem, VolumetricReSTIR)

### 5. Add Froxel Passes to Render Pipeline

**Location:** `Application::Render()` or `Application::RenderFrame()` (before Gaussian renderer call)

**Code to Add:**
```cpp
// === Froxel Volumetric Fog Rendering (Phase 5) ===
if (m_enableFroxelFog && m_froxelSystem) {
    PIXBeginEvent(commandList, PIX_COLOR_INDEX(10), "Froxel Volumetric Fog");

    // Pass 1: Clear density grid (UAV clear)
    m_froxelSystem->ClearGrid(commandList);

    // Barrier: Ensure clear completes before density injection
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(
        m_froxelSystem->GetDensityGrid()
    );
    commandList->ResourceBarrier(1, &barrier);

    // Pass 2: Inject particle density into froxel grid
    m_froxelSystem->InjectDensity(
        commandList,
        m_particleSystem->GetParticleBuffer(),
        m_activeParticleCount
    );

    // Barrier: UAV (write) → SRV (read) for density grid
    CD3DX12_RESOURCE_BARRIER densityBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_froxelSystem->GetDensityGrid(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    commandList->ResourceBarrier(1, &densityBarrier);

    // Pass 3: Calculate voxel lighting
    m_froxelSystem->LightVoxels(
        commandList,
        m_particleSystem->GetParticleBuffer(),
        m_activeParticleCount,
        m_gaussianRenderer->GetLightBuffer(),  // 13 lights
        static_cast<uint32_t>(m_lights.size()),
        m_rtLighting->GetTLAS()  // Particle BVH for shadow rays
    );

    // Barrier: UAV (write) → SRV (read) for lighting grid
    CD3DX12_RESOURCE_BARRIER lightingBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_froxelSystem->GetLightingGrid(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );
    commandList->ResourceBarrier(1, &lightingBarrier);

    PIXEndEvent(commandList);
}

// Pass 4: Render particles (existing Gaussian renderer)
// Gaussian shader will sample m_froxelSystem->GetLightingGridSRV() for volumetric fog
m_gaussianRenderer->Render(...);

// Barrier: Transition lighting grid back to UAV for next frame
if (m_enableFroxelFog && m_froxelSystem) {
    CD3DX12_RESOURCE_BARRIER lightingBackBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_froxelSystem->GetLightingGrid(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    commandList->ResourceBarrier(1, &lightingBackBarrier);
}
```

**Critical Notes:**
- **Barrier Order:** Clear → UAV barrier → Inject → UAV→SRV → Light → UAV→SRV → Sample
- **Resource States:** Density and lighting grids start in UAV state, transition to SRV for reading
- **PIX Events:** Wrap in PIXBeginEvent/PIXEndEvent for GPU debugging

### 6. Implement Dispatch Calls in FroxelSystem.cpp

**Current Status:** Dispatch calls are stubbed out (commented)

**InjectDensity() Implementation:**
```cpp
void FroxelSystem::InjectDensity(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount)
{
    if (!m_initialized) {
        LOG_ERROR("FroxelSystem not initialized");
        return;
    }

    // Update particle count
    m_gridParams.particleCount = particleCount;

    // Upload constant buffer
    memcpy(m_constantBufferMapped, &m_gridParams, sizeof(GridParams));

    // Set pipeline state
    commandList->SetPipelineState(m_injectDensityPSO.Get());
    commandList->SetComputeRootSignature(m_injectDensityRootSig.Get());

    // Bind resources
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // TODO: Bind particle buffer SRV (t0) - need descriptor table
    // commandList->SetComputeRootDescriptorTable(1, particleBufferSRV);

    // TODO: Bind density grid UAV (u0) - need descriptor table
    D3D12_GPU_DESCRIPTOR_HANDLE densityUAVGPU = m_resources->GetGPUHandle(m_densityGridUAV);
    commandList->SetComputeRootDescriptorTable(2, densityUAVGPU);

    // Dispatch density injection compute shader
    // Thread groups: (particleCount + 255) / 256
    uint32_t threadGroups = (particleCount + 255) / 256;
    commandList->Dispatch(threadGroups, 1, 1);

    LOG_DEBUG("Injected density for {} particles ({} thread groups)", particleCount, threadGroups);
}
```

**LightVoxels() Implementation:**
```cpp
void FroxelSystem::LightVoxels(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* particleBuffer,
    uint32_t particleCount,
    ID3D12Resource* lightBuffer,
    uint32_t lightCount,
    ID3D12Resource* particleBVH)
{
    if (!m_initialized) {
        LOG_ERROR("FroxelSystem not initialized");
        return;
    }

    // Update light count in constant buffer
    // NOTE: This is a hack - need to extend GridParams struct
    // For now, assume lightCount is already set elsewhere

    // Upload constant buffer
    memcpy(m_constantBufferMapped, &m_gridParams, sizeof(GridParams));

    // Set pipeline state
    commandList->SetPipelineState(m_lightVoxelsPSO.Get());
    commandList->SetComputeRootSignature(m_lightVoxelsRootSig.Get());

    // Bind resources
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // TODO: Bind density grid SRV (t0) - need descriptor table
    D3D12_GPU_DESCRIPTOR_HANDLE densitySRVGPU = m_resources->GetGPUHandle(m_densityGridSRV);
    commandList->SetComputeRootDescriptorTable(1, densitySRVGPU);

    // TODO: Bind light buffer SRV (t1) - need descriptor table
    // commandList->SetComputeRootDescriptorTable(2, lightBufferSRV);

    // Bind particle BVH (t2) - acceleration structure uses root descriptor
    commandList->SetComputeRootShaderResourceView(3, particleBVH->GetGPUVirtualAddress());

    // TODO: Bind lighting grid UAV (u0) - need descriptor table
    D3D12_GPU_DESCRIPTOR_HANDLE lightingUAVGPU = m_resources->GetGPUHandle(m_lightingGridUAV);
    commandList->SetComputeRootDescriptorTable(4, lightingUAVGPU);

    // Dispatch voxel lighting compute shader
    // Thread groups: 8×8×8 per group covers entire grid
    uint32_t groupsX = (m_gridParams.gridDimensions.x + 7) / 8;  // 20
    uint32_t groupsY = (m_gridParams.gridDimensions.y + 7) / 8;  // 12
    uint32_t groupsZ = (m_gridParams.gridDimensions.z + 7) / 8;  // 8
    commandList->Dispatch(groupsX, groupsY, groupsZ);

    LOG_DEBUG("Lit {} voxels with {} lights ({} thread groups)",
              m_gridParams.gridDimensions.x * m_gridParams.gridDimensions.y * m_gridParams.gridDimensions.z,
              lightCount,
              groupsX * groupsY * groupsZ);
}
```

**TODO: Descriptor Table Binding**
- Need to get GPU descriptor handles for particle buffer, light buffer
- May need to expose these from ParticleSystem and ParticleRenderer_Gaussian

### 7. Modify Gaussian Raytrace Shader

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**Step 1: Add froxel grid resources (near top of file, after existing textures)**
```hlsl
// === Froxel Volumetric Fog (Phase 5 - replaces god rays) ===
Texture3D<float4> g_froxelLightingGrid : register(t10);
SamplerState g_linearClampSampler : register(s0);
```

**Step 2: Add froxel parameters to GaussianConstants cbuffer**
```hlsl
cbuffer GaussianConstants : register(b0)
{
    // ... existing constants ...

    // === Froxel Grid Parameters ===
    float3 froxelGridMin;             // [-1500, -1500, -1500]
    float3 froxelGridMax;             // [1500, 1500, 1500]
    uint3 froxelGridDimensions;       // [160, 90, 64]
    uint useFroxelFog;                // Toggle: 0=god rays, 1=froxels
};
```

**Step 3: Include froxel sampling helper**
```hlsl
// After other includes (around line 100)
#include "froxel/sample_froxel_grid.hlsl"
```

**Step 4: Replace god ray call (search for "RayMarchAtmosphericFog", around line 1702)**
```hlsl
// OLD (god rays - expensive):
/*
if (godRayDensity > 0.001) {
    float3 atmosphericFog = RayMarchAtmosphericFog(...);
    finalColor += atmosphericFog * 0.1;
}
*/

// NEW (froxels - fast):
if (useFroxelFog != 0) {
    float3 atmosphericFog = RayMarchFroxelGrid(
        cameraPos,
        ray.Direction,
        3000.0,              // Max ray distance
        godRayDensity        // Reuse existing density parameter (or create new)
    );
    finalColor += atmosphericFog * 0.1;
}
```

**Step 5: Update ParticleRenderer_Gaussian to bind froxel grid**

In `ParticleRenderer_Gaussian::Render()`:
```cpp
// Bind froxel lighting grid SRV (t10)
if (m_useFroxelFog && froxelSystem) {
    D3D12_GPU_DESCRIPTOR_HANDLE froxelSRV = m_resources->GetGPUHandle(
        froxelSystem->GetLightingGridSRV()
    );
    commandList->SetComputeRootDescriptorTable(FROXEL_SRV_SLOT, froxelSRV);
}
```

**Step 6: Update RenderConstants to include froxel parameters**
```cpp
struct RenderConstants {
    // ... existing fields ...

    // Froxel grid parameters
    DirectX::XMFLOAT3 froxelGridMin;
    float padding_froxel1;
    DirectX::XMFLOAT3 froxelGridMax;
    float padding_froxel2;
    DirectX::XMUINT3 froxelGridDimensions;
    uint32_t useFroxelFog;  // 0=disabled, 1=enabled
};
```

### 8. Add ImGui Controls

**Location:** `Application::RenderImGui()` (add new collapsing header)

**Code to Add:**
```cpp
// === Froxel Volumetric Fog Controls ===
if (ImGui::CollapsingHeader("Froxel Volumetric Fog (Phase 5)")) {
    // Toggle froxel fog
    if (ImGui::Checkbox("Enable Froxel Fog (F7)", &m_enableFroxelFog)) {
        LOG_INFO("Froxel fog: {}", m_enableFroxelFog ? "ENABLED" : "DISABLED");
    }

    if (m_enableFroxelFog && m_froxelSystem) {
        ImGui::Indent();

        // Density multiplier
        float densityMult = m_froxelSystem->GetGridParams().densityMultiplier;
        if (ImGui::SliderFloat("Fog Density", &densityMult, 0.1f, 5.0f, "%.2f")) {
            m_froxelSystem->SetDensityMultiplier(densityMult);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Global fog density multiplier (0.1 = subtle, 5.0 = thick)");
        }

        // Debug visualization
        if (ImGui::Checkbox("Debug Visualization", &m_debugFroxelVisualization)) {
            m_froxelSystem->EnableDebugVisualization(m_debugFroxelVisualization);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show density/lighting heat map for debugging");
        }

        ImGui::Separator();

        // Grid info (read-only)
        const auto& params = m_froxelSystem->GetGridParams();
        ImGui::Text("Grid: %dx%dx%d = %d voxels",
                    params.gridDimensions.x,
                    params.gridDimensions.y,
                    params.gridDimensions.z,
                    params.gridDimensions.x * params.gridDimensions.y * params.gridDimensions.z);

        ImGui::Text("Voxel Size: %.2f × %.2f × %.2f units",
                    params.voxelSize.x, params.voxelSize.y, params.voxelSize.z);

        float densityMB = (params.gridDimensions.x * params.gridDimensions.y *
                          params.gridDimensions.z * 2) / (1024.0f * 1024.0f);
        float lightingMB = (params.gridDimensions.x * params.gridDimensions.y *
                           params.gridDimensions.z * 8) / (1024.0f * 1024.0f);
        ImGui::Text("GPU Memory: %.2f MB (density) + %.2f MB (lighting)", densityMB, lightingMB);

        ImGui::Unindent();
    }
    else if (!m_froxelSystem) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "FroxelSystem not initialized!");
    }
}

// Comparison with god rays
if (m_enableFroxelFog || m_godRayDensity > 0.001f) {
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Note: God rays and froxels are mutually exclusive");
    if (m_enableFroxelFog && m_godRayDensity > 0.001f) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "WARNING: Both enabled - froxels will take priority");
    }
}
```

**Keyboard Shortcut (optional):**
```cpp
// In Application::OnKeyPress()
case VK_F7:
    m_enableFroxelFog = !m_enableFroxelFog;
    LOG_INFO("Froxel fog: {}", m_enableFroxelFog ? "ENABLED" : "DISABLED");
    break;
```

---

## 🧪 Testing Plan

### Phase 1: Verify Density Injection

**Steps:**
1. Enable froxel fog via ImGui or F7
2. Enable debug visualization: `m_debugFroxelVisualization = true`
3. Call `DebugVisualizeFroxelDensity()` in Gaussian shader instead of normal rendering
4. **Expected:** Colored heat map around particles (blue = low, red = high)
5. **If black:** Density injection not working (check dispatch, UAV binding)

**Debug Checklist:**
- [ ] Froxel system initialized successfully (check logs)
- [ ] InjectDensity() dispatched (check PIX capture event timeline)
- [ ] Thread group count correct: `(particleCount + 255) / 256`
- [ ] Density grid UAV bound correctly (check PIX resource bindings)
- [ ] Constant buffer uploaded (check froxelGridMin/Max/Dimensions)

### Phase 2: Verify Voxel Lighting

**Steps:**
1. Disable density debug visualization
2. Call `DebugVisualizeFroxelLighting()` in Gaussian shader
3. **Expected:** Colored fog matching light colors, shadows visible
4. **If uniform color:** Lighting not working (check light buffer binding, BVH)

**Debug Checklist:**
- [ ] LightVoxels() dispatched after InjectDensity()
- [ ] Thread group count correct: `(20, 12, 8)` for 160×90×64 grid
- [ ] Light buffer bound correctly (check SRV)
- [ ] Particle BVH bound (acceleration structure)
- [ ] Lighting grid UAV bound correctly

### Phase 3: Performance Validation

**Comparison Test:**
1. Disable froxels, enable god rays: `m_godRayDensity = 1.0`, `m_enableFroxelFog = false`
2. Record FPS: **~21-29 FPS** (baseline from previous test)
3. Disable god rays, enable froxels: `m_godRayDensity = 0.0`, `m_enableFroxelFog = true`
4. Record FPS: **Target ~100 FPS**
5. **Expected speedup:** 4-5× FPS gain

**PIX Profiling:**
1. Capture frame with PIX
2. Find froxel passes in event timeline
3. Check GPU times:
   - InjectDensity: **Target ~0.5ms**
   - LightVoxels: **Target ~3-5ms**
   - Total froxel overhead: **~6ms** (vs ~48ms for god rays)

**Performance Checklist:**
- [ ] InjectDensity dispatch time < 1ms
- [ ] LightVoxels dispatch time < 6ms
- [ ] Total froxel overhead < 8ms
- [ ] FPS improvement: 21 FPS → 90-100+ FPS

### Phase 4: Visual Quality Comparison

**Screenshot Comparison:**
1. God rays enabled: Capture screenshot (`m_godRayDensity = 1.0`)
2. Froxels enabled: Capture screenshot (`m_enableFroxelFog = true`)
3. Compare side-by-side

**Quality Checklist:**
- [ ] Fog concentrated around particles (not uniform)
- [ ] Multi-light colors visible in fog
- [ ] Shadows visible (darker regions where particles block light)
- [ ] Smooth gradients (no hard voxel edges from trilinear interpolation)
- [ ] Depth perception (fog thicker in dense particle regions)

**Known Differences:**
- Froxels: Fog only where particles are (spatially accurate)
- God rays: Uniform fog everywhere (less physically accurate but more cinematic)
- Froxels: Sharper shadows (single shadow ray per voxel)
- God rays: Softer shadows (multiple samples per pixel)

---

## 🐛 Known Issues and Troubleshooting

### Issue 1: Black Screen / No Fog Visible

**Possible Causes:**
- Density injection not running (check dispatch count in PIX)
- UAV/SRV bindings incorrect (check descriptor tables)
- Constant buffer not uploaded (check gridMin/Max/Dimensions)
- Fog density too low (increase `m_froxelDensityMultiplier`)

**Debug Steps:**
1. Enable PIX capture: `--config=configs/agents/pix_agent.json`
2. Check event timeline for "Froxel Volumetric Fog" marker
3. Inspect density grid UAV after InjectDensity dispatch
4. Verify non-zero values in density grid texture

### Issue 2: Uniform Fog Color (Not Multi-Light)

**Possible Causes:**
- Light buffer not bound to LightVoxels shader
- Light count incorrect in constant buffer
- Particle BVH not bound (shadows not working)

**Debug Steps:**
1. Check PIX resource bindings for LightVoxels dispatch
2. Verify light buffer SRV descriptor is valid
3. Dump light buffer to check data (should have 13 lights × 64 bytes)

### Issue 3: Blocky Artifacts / Hard Voxel Edges

**Possible Causes:**
- Trilinear interpolation not enabled in sampler
- Wrong sampler state bound (point filtering instead of linear)

**Fix:**
```hlsl
// In sample_froxel_grid.hlsl, verify:
SamplerState g_linearClampSampler : register(s0);  // Must be D3D12_FILTER_MIN_MAG_MIP_LINEAR
```

### Issue 4: Performance Not Improving

**Possible Causes:**
- God rays still enabled (`m_godRayDensity > 0.001`)
- Froxel grid resolution too high (try 120×64×48 instead of 160×90×64)
- Shader not using froxel grid (still using old god ray path)

**Debug Steps:**
1. Verify `useFroxelFog` constant in shader is 1
2. Check shader code - ensure RayMarchFroxelGrid() is called, not RayMarchAtmosphericFog()
3. Reduce grid resolution if needed (trade quality for performance)

### Issue 5: Crash on Initialization

**Possible Causes:**
- Descriptor heap full (too many UAVs/SRVs allocated)
- GPU out of memory (9MB of 3D textures)

**Fix:**
```cpp
// Check ResourceManager descriptor heap capacity
// May need to increase heap size in ResourceManager::CreateDescriptorHeap()
```

---

## 📊 Expected Performance Metrics

### Baseline (God Rays - Expensive)

- **FPS:** 21-29 FPS @ 1440p
- **Frame Time:** ~48ms
- **Operations:** 1.5 BILLION ops/frame
  - 3.7M pixels × 32 steps × 13 lights × RayQuery

### Target (Froxels - Efficient)

- **FPS:** 90-100 FPS @ 1440p
- **Frame Time:** ~10ms
- **Operations:** 12 MILLION ops/frame
  - 921K voxels × 13 lights
- **Speedup:** **750× fewer operations, 5× FPS gain**

### Breakdown by Pass

| Pass | Resolution | Operations | GPU Time | Bottleneck |
|------|-----------|-----------|----------|-----------|
| **Inject Density** | 10K particles → 921K voxels | 10K × 8 neighbors | ~0.5ms | Memory bandwidth (UAV atomic adds) |
| **Light Voxels** | 921K voxels × 13 lights | 12M light evaluations | ~3-5ms | ALU (lighting math + shadow rays) |
| **Sample Grid** | 3.7M pixels × 32 samples | 118M texture samples | ~1-2ms | Texture cache (hardware trilinear) |
| **Total** | - | - | **~6ms** | - |

---

## 🔄 Integration Workflow Summary

**Recommended Order:**

1. ✅ **DONE:** Create core files, build system integration
2. ⏳ **NEXT:** Initialize FroxelSystem in Application.cpp
3. ⏳ **NEXT:** Implement dispatch calls in FroxelSystem.cpp (InjectDensity, LightVoxels)
4. ⏳ **NEXT:** Add froxel passes to render pipeline with barriers
5. ⏳ **NEXT:** Modify Gaussian shader to sample froxel grid
6. ⏳ **NEXT:** Add ImGui controls for runtime tuning
7. 🧪 **TEST:** Verify density injection (debug visualization)
8. 🧪 **TEST:** Verify voxel lighting (debug visualization)
9. 🧪 **TEST:** Performance validation (PIX profiling)
10. 🧪 **TEST:** Visual quality comparison (screenshots)

**Estimated Time Remaining:** 2-3 hours for full integration

---

## 📝 Code Snippet Quick Reference

### Get Particle Buffer SRV (for InjectDensity binding)
```cpp
// In ParticleSystem.h, add accessor:
D3D12_CPU_DESCRIPTOR_HANDLE GetParticleBufferSRV() const { return m_particleSRV; }

// In FroxelSystem::InjectDensity():
D3D12_GPU_DESCRIPTOR_HANDLE particleSRV = m_resources->GetGPUHandle(
    particleSystem->GetParticleBufferSRV()
);
commandList->SetComputeRootDescriptorTable(1, particleSRV);
```

### Get Light Buffer SRV (for LightVoxels binding)
```cpp
// In ParticleRenderer_Gaussian.h, accessor already exists:
D3D12_GPU_DESCRIPTOR_HANDLE GetLightSRV() const { return m_lightSRVGPU; }

// In FroxelSystem::LightVoxels():
commandList->SetComputeRootDescriptorTable(2, lightSRV);  // Pass as parameter
```

### Resource Barrier Helpers
```cpp
// UAV barrier (ensure previous UAV writes complete)
CD3DX12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(resource);
commandList->ResourceBarrier(1, &uavBarrier);

// UAV → SRV transition
CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(
    resource,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
);
commandList->ResourceBarrier(1, &toSRV);

// SRV → UAV transition (for next frame)
CD3DX12_RESOURCE_BARRIER toUAV = CD3DX12_RESOURCE_BARRIER::Transition(
    resource,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS
);
commandList->ResourceBarrier(1, &toUAV);
```

---

## 🎯 Success Criteria

**Froxel system is fully functional when:**

1. ✅ Build compiles with zero errors
2. ⏳ Application initializes FroxelSystem without errors
3. ⏳ Density injection produces visible heat map in debug mode
4. ⏳ Voxel lighting shows multi-light colors and shadows
5. ⏳ Froxel fog visible in Gaussian renderer (Pass 3)
6. ⏳ FPS improves from 21 FPS to 90-100+ FPS
7. ⏳ ImGui controls work (density slider, debug toggle)
8. ⏳ No visual artifacts (blocky voxels, black screen, etc.)

**When all criteria met:** Froxel system replaces god rays as the primary volumetric fog solution.

---

**Last Updated:** 2025-11-21 02:30
**Status:** Infrastructure complete (60%), runtime integration pending (40%)
**Next Session:** Continue with Application.cpp initialization and render pipeline integration
