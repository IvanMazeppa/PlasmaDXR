# RTXPT Feature Cherry-Picking Roadmap

**Project**: PlasmaDX-Clean
**Source**: NVIDIA RTXPT v1.7.0 (Downloaded: /mnt/d/users/dilli/androidstudioprojects/rtxpt)
**Author**: Roadmap generated 2025-10-25
**Timeline**: 10-week phased implementation

---

## Overview

This roadmap outlines selective integration of RTXPT features without adopting the full Donut/NVRHI framework. Each phase is independent and builds on the previous, allowing flexible prioritization.

**Philosophy**: Cherry-pick production-ready SDKs and techniques, maintain direct D3D12 control.

---

## Complete Timeline

| Phase | Feature | Benefit | Timeline | FPS Impact |
|-------|---------|---------|----------|------------|
| **Phase 1** | Balanced GI | Color bleeding, soft shadows | 1-2 days | 99 FPS |
| **Phase 2** | Temporal Accumulation | Reduce noise, restore FPS | 2-3 days | 115 FPS |
| **Phase 3** | NRD Denoising | High quality at 1 ray/particle | 2-3 weeks | 120+ FPS |
| **Phase 4** | Full Multi-Bounce GI | 3-4 bounces, realistic GI | 1-2 weeks | 90-100 FPS |
| **Phase 5** | DLSS via Streamline | 4K rendering, upscaling | 2-3 weeks | 120+ FPS @ 4K |

---

## Phase 1: Balanced GI (Week 1)

**Status**: See `BALANCED_GI_IMPLEMENTATION_PLAN.md`

**Deliverable**: Single-bounce indirect lighting @ 99 FPS

**Reference**: None required (custom implementation)

---

## Phase 2: Temporal Accumulation (Week 2)

**Goal**: Reduce rays/particle by accumulating over time
**Reference**: `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/Shaders/PathTracer/PathTracerHelpers.hlsli`

### What RTXPT Does

RTXPT accumulates samples across frames using exponential moving average:

```hlsl
// From PathTracerHelpers.hlsli (conceptual, not direct copy)
float3 AccumulateTemporalSamples(float3 currentSample, float3 previousAccumulated, uint frameCount)
{
    float alpha = 1.0 / min(frameCount + 1, maxSampleCount);
    return lerp(previousAccumulated, currentSample, alpha);
}
```

### Your Implementation

**Reuse existing RTXDI M5 temporal accumulation infrastructure!**

You already have:
- Ping-pong buffers (m_accumulatedBuffer[2])
- Temporal blend shader
- Camera movement detection
- Reset logic

**Modify**: `src/lighting/RTLightingSystem_RayQuery.cpp`

```cpp
// Add temporal accumulation for indirect lighting
// (Copy pattern from RTXDILightingSystem::DispatchTemporalAccumulation)

void RTLightingSystem_RayQuery::AccumulateIndirectTemporal(
    ID3D12GraphicsCommandList* cmdList,
    const DirectX::XMFLOAT3& cameraPos,
    uint32_t frameIndex)
{
    // Reduce indirect rays to 4/particle (instead of 8)
    m_indirectRaysPerParticle = 4;

    // Dispatch indirect with low ray count (noisy)
    DispatchIndirectLighting(cmdList, particleBuffer);

    // Temporal accumulation compute shader
    // Input: current frame indirect (4 rays, noisy)
    // Input: previous frame accumulated (smooth)
    // Output: blended result
    // Blend formula: lerp(prev, curr, 0.125) = 8-frame accumulation
}
```

**New shader**: `shaders/dxr/particle_indirect_temporal.hlsl`

```hlsl
// Read current noisy indirect
// Read previous accumulated
// Detect camera movement (reset if moved)
// Blend: output = lerp(previous, current, 0.125)
```

**Performance**: 115 FPS (only -5 FPS vs. baseline)
**Quality**: Same as 8 rays/frame, but 67ms convergence time

**Timeline**: 2-3 days (reuse RTXDI M5 patterns)

---

## Phase 3: NRD Denoising Integration (Weeks 3-5)

**Goal**: Use NVIDIA NRD for production-quality denoising
**Reference**: `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/NRD/`

### What is NRD?

**NVIDIA Real-Time Denoiser** - Standalone SDK for ray-traced lighting:
- **ReLAX**: Best for diffuse GI (your use case)
- **ReBLUR**: Best for reflections/specular
- **Supports**: DX12, Vulkan, standalone (no Donut/NVRHI required!)

**Location in RTXPT**:
- Headers: `rtxpt/External/NRD/Include/`
- Binaries: `rtxpt/NRD.dll`
- Shaders: `rtxpt/ShaderPrecompiled/nrd/dxil/Nrd/Shaders/`

### Week 3: Download & Build NRD SDK

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/external
git clone https://github.com/NVIDIA-RTX/NRD.git
cd NRD
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Add to CMakeLists.txt**:
```cmake
# NRD SDK
set(NRD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NRD")
if(EXISTS "${NRD_DIR}/Include")
    target_include_directories(${PROJECT_NAME} PRIVATE ${NRD_DIR}/Include)
    target_link_directories(${PROJECT_NAME} PRIVATE ${NRD_DIR}/Lib)
    target_link_libraries(${PROJECT_NAME} NRD.lib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_NRD)
endif()
```

**Copy DLL**:
```bash
cp external/NRD/Bin/Release/NRD.dll build/Debug/
```

### Week 4: Prepare Denoiser Inputs

NRD ReLAX requires specific input buffers:

1. **Radiance**: Your indirect lighting (already have)
2. **Hit Distance**: Ray hit T (store in alpha channel)
3. **Normal**: Particle velocity as proxy normal
4. **Linear Depth**: Camera space Z

**Modify indirect shader** to store hit distance:

```hlsl
// Store hit distance for denoiser (in alpha channel)
float hitDistance = query.CommittedRayT();
g_indirectLighting[particleIdx] = float4(finalIndirect, hitDistance);
```

**Create additional buffers**:

```cpp
// Normal buffer (R16G16B16A16_FLOAT)
// Store particle velocity normalized as "normal"
ComPtr<ID3D12Resource> m_normalBuffer;

// Depth buffer (R32_FLOAT)
// Store camera-space Z distance
ComPtr<ID3D12Resource> m_depthBuffer;
```

**Generate normal/depth in separate compute pass**:

```hlsl
// particle_generate_denoiser_inputs.hlsl
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Particle p = g_particles[dispatchThreadID.x];

    // Normal: Use velocity direction
    float3 normal = normalize(p.velocity);
    g_normals[dispatchThreadID.x] = float4(normal, 0);

    // Depth: Transform to camera space
    float3 viewPos = mul(float4(p.position, 1.0), g_viewMatrix).xyz;
    g_depth[dispatchThreadID.x] = viewPos.z;
}
```

### Week 5: Integrate ReLAX Denoiser

**Reference implementation**: `rtxpt/Rtxpt/NRD/DenoiserNRD.hlsli`

**Create NRD wrapper class**:

```cpp
// src/utils/NRDDenoiser.h
#pragma once

#include <NRD.h>
#include <d3d12.h>

class NRDDenoiser {
public:
    bool Initialize(ID3D12Device* device, uint32_t width, uint32_t height);
    void Denoise(ID3D12GraphicsCommandList* cmdList,
                ID3D12Resource* radianceHitDist,
                ID3D12Resource* normal,
                ID3D12Resource* depth,
                ID3D12Resource* output);
    void Shutdown();

private:
    nrd::Denoiser* m_denoiser = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};
```

**Initialize denoiser**:

```cpp
bool NRDDenoiser::Initialize(ID3D12Device* device, uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;

    // Setup denoiser descriptor
    nrd::DenoiserCreationDesc denoiserDesc = {};
    denoiserDesc.requestedMethods = { nrd::Method::RELAX_DIFFUSE };
    denoiserDesc.enableValidation = false;

    // Create denoiser instance
    nrd::Result result = nrd::CreateDenoiser(denoiserDesc, m_denoiser);
    if (result != nrd::Result::SUCCESS) {
        return false;
    }

    // Get required memory sizes
    const nrd::DenoiserDesc& desc = nrd::GetDenoiserDesc(*m_denoiser);

    // Allocate resources based on desc.permanentPoolSize, desc.transientPoolSize
    // (Implementation details in NRD SDK documentation)

    return true;
}
```

**Per-frame denoising**:

```cpp
void NRDDenoiser::Denoise(...)
{
    // Setup common settings
    nrd::CommonSettings commonSettings = {};
    commonSettings.frameIndex = m_frameIndex;
    commonSettings.viewToClipMatrix = /* ... */;
    commonSettings.worldToViewMatrix = /* ... */;

    // Setup ReLAX-specific settings
    nrd::RelaxDiffuseSettings relaxSettings = {};
    relaxSettings.prepassBlurRadius = 30.0f;
    relaxSettings.diffuseMaxAccumulatedFrameNum = 32;

    // Dispatch denoiser
    nrd::SetMethodSettings(*m_denoiser, nrd::Method::RELAX_DIFFUSE, &relaxSettings);
    nrd::Denoise(...);
}
```

**Integration in render loop**:

```cpp
// After indirect lighting dispatch
m_rtLighting->DispatchIndirectLighting(cmdList, particleBuffer);

// Generate denoiser inputs (normals, depth)
GenerateDenoiserInputs(cmdList);

// Denoise indirect lighting
m_nrdDenoiser->Denoise(cmdList,
                       m_rtLighting->GetIndirectLightingBuffer(),
                       m_normalBuffer,
                       m_depthBuffer,
                       m_denoisedIndirectBuffer);

// Use denoised result in Gaussian renderer
```

**Performance**: 120+ FPS (1 ray/particle vs 8 rays)
**Quality**: Superior to 16 rays/particle without denoising

**Timeline**: 2-3 weeks including learning NRD API

---

## Phase 4: Full Multi-Bounce GI (Weeks 6-7)

**Goal**: Add 2nd and 3rd bounce for realistic multi-bounce GI
**Reference**: `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/Shaders/PathTracer/PathTracer.hlsli`

### RTXPT Multi-Bounce Architecture

RTXPT uses bounce counters and recursive TraceRay:

```hlsl
// From PathTracer.hlsli:42-47
inline bool HasFinishedSurfaceBounces(uint vertexIndex, uint diffuseBounces)
{
    if (Bridge::getMaxBounceLimit() < vertexIndex)
        return true;
    return diffuseBounces > Bridge::getMaxDiffuseBounceLimit();
}
```

**Key concept**: Track vertex index and diffuse bounce count, terminate when limits reached.

### Your Implementation (RayQuery Iterative)

Since RayQuery doesn't support recursion, use **multiple compute passes**:

**Pass 1**: Direct lighting (1st bounce)
**Pass 2**: Indirect bounce 1 (2nd bounce) - reads direct
**Pass 3**: Indirect bounce 2 (3rd bounce) - reads indirect1

**Create 3rd pass shader**: `shaders/dxr/particle_indirect_bounce2_cs.hlsl`

```hlsl
cbuffer IndirectBounce2Constants : register(b0)
{
    uint particleCount;
    uint raysPerParticle;      // 4 for 3rd bounce
    float maxLightingDistance;
    float indirectIntensity;
};

StructuredBuffer<Particle> g_particles : register(t0);
RaytracingAccelerationStructure g_particleBVH : register(t1);

// INPUT: First bounce indirect lighting
StructuredBuffer<float4> g_indirectBounce1 : register(t2);

// OUTPUT: Second bounce indirect lighting
RWStructuredBuffer<float4> g_indirectBounce2 : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Same structure as indirect bounce 1
    // But reads from g_indirectBounce1 instead of g_directLighting

    // When ray hits particle:
    float3 hitIndirect1Light = g_indirectBounce1[hitParticleIdx].rgb;

    // Apply BRDF and accumulate
    accumulatedBounce2 += hitIndirect1Light * brdf * attenuation;
}
```

**Modify ComputeLighting pipeline**:

```cpp
void RTLightingSystem_RayQuery::ComputeLighting(...)
{
    // Build TLAS (once)
    GenerateAABBs(cmdList, particleBuffer);
    BuildBLAS(cmdList);
    BuildTLAS(cmdList);

    // Pass 1: Direct (16 rays/particle)
    DispatchRayQueryLighting(cmdList, particleBuffer);

    // Pass 2: Indirect 1 (8 rays/particle)
    DispatchIndirectLighting(cmdList, particleBuffer);

    // Pass 3: Indirect 2 (4 rays/particle)
    DispatchIndirectBounce2(cmdList, particleBuffer);
}
```

**Combine in Gaussian renderer**:

```hlsl
// Final composite
float3 directLighting = g_particleLighting[idx].rgb;
float3 indirectBounce1 = g_particleIndirectLighting1[idx].rgb;
float3 indirectBounce2 = g_particleIndirectLighting2[idx].rgb;

// Energy conservation: reduce contribution per bounce
float3 finalColor = baseColor + directLighting + indirectBounce1 + indirectBounce2 * 0.5;
```

**With NRD denoising** (1 ray/bounce):
- Pass 1: Direct (1 ray) → denoise
- Pass 2: Indirect1 (1 ray) → denoise
- Pass 3: Indirect2 (1 ray) → denoise

**Performance**: 90-100 FPS with denoising, 60-70 FPS without
**Quality**: Photorealistic multi-bounce GI

**Timeline**: 1-2 weeks

---

## Phase 5: DLSS Integration via Streamline (Weeks 8-10)

**Goal**: 4K rendering with DLSS upscaling
**Reference**: `rtxpt/sl.*.dll`, `rtxpt/nvngx_dlss.dll`

### What is Streamline?

**NVIDIA Streamline** - SDK wrapper for DLSS, Reflex, Frame Generation:
- **DLSS Super Resolution**: Render 1080p → upscale to 4K
- **DLSS Frame Generation**: Generate intermediate frames (DLSS 3.5)
- **Standalone**: Can integrate without Donut/NVRHI

**RTXPT DLLs** (copy to your project):
- `sl.common.dll` - Core Streamline runtime
- `sl.dlss.dll` - DLSS plugin
- `nvngx_dlss.dll` - DLSS neural network weights

### Week 8: Add Streamline SDK

**Copy DLLs from RTXPT**:

```bash
cp /mnt/d/users/dilli/androidstudioprojects/rtxpt/sl.*.dll /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/Debug/
cp /mnt/d/users/dilli/androidstudioprojects/rtxpt/nvngx_dlss.dll /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/Debug/
```

**Download Streamline SDK headers**:

```bash
cd external
git clone https://github.com/NVIDIAGameWorks/Streamline.git
```

**Add to CMakeLists.txt**:

```cmake
# Streamline SDK
target_include_directories(${PROJECT_NAME} PRIVATE external/Streamline/include)
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_STREAMLINE)
```

### Week 9: DLSS Integration

**Initialize Streamline**:

```cpp
#include "sl.h"
#include "sl_dlss.h"

// In Application::Initialize()
sl::Preferences pref = {};
pref.showConsole = false;
pref.logLevel = sl::LogLevel::eDefault;
pref.pathsToPlugins = { "." };  // DLLs in exe directory

sl::Result result = slInit(pref);
if (result != sl::Result::eOk) {
    LOG_ERROR("Failed to initialize Streamline");
    return false;
}

// Enable DLSS feature
sl::Feature features[] = { sl::kFeatureDLSS };
slSetFeatureLoaded(sl::kFeatureDLSS, true);
```

**Configure DLSS mode**:

```cpp
// Setup DLSS options
sl::DLSSOptions dlssOptions = {};
dlssOptions.mode = sl::DLSSMode::eQuality;  // 1080p → 4K (3840×2160)
dlssOptions.outputWidth = 3840;
dlssOptions.outputHeight = 2160;
dlssOptions.colorBuffersHDR = sl::Boolean::eTrue;  // HDR rendering

sl::DLSSOptimalSettings optimalSettings = {};
slDLSSGetOptimalSettings(dlssOptions, &optimalSettings);

LOG_INFO("DLSS render resolution: {}×{}", optimalSettings.renderWidth, optimalSettings.renderHeight);
// Expect: 1920×1080 for Quality mode
```

**Per-frame upscaling**:

```cpp
// Render loop
void Application::Render()
{
    // 1. Render at internal resolution (1080p)
    m_gaussianRenderer->Render(cmdList, 1920, 1080);

    // 2. Setup DLSS inputs
    sl::Resource colorInput = {};
    colorInput.type = sl::ResourceType::eTex2d;
    colorInput.native = m_hdrRenderTarget.Get();  // 1080p HDR buffer

    sl::Resource colorOutput = {};
    colorOutput.type = sl::ResourceType::eTex2d;
    colorOutput.native = m_dlssOutputBuffer.Get();  // 4K HDR buffer

    sl::Resource depth = {};
    depth.type = sl::ResourceType::eTex2d;
    depth.native = m_depthBuffer.Get();

    sl::Resource motionVectors = {};
    motionVectors.type = sl::ResourceType::eTex2d;
    motionVectors.native = m_motionVectorBuffer.Get();  // Need to generate

    // 3. Invoke DLSS
    sl::DLSSState dlssState = {};
    dlssState.colorIn = colorInput;
    dlssState.colorOut = colorOutput;
    dlssState.depth = depth;
    dlssState.mvec = motionVectors;

    slDLSSSetState(dlssState, cmdList);

    // 4. Present upscaled 4K result
    Present(m_dlssOutputBuffer);
}
```

**Motion vectors** (required for DLSS):

Create compute shader to generate camera motion:

```hlsl
// motion_vector_gen_cs.hlsl
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 uv = dispatchThreadID.xy / float2(width, height);

    // Reproject current pixel to previous frame
    float3 worldPos = ReconstructWorldPos(uv, depth);
    float4 prevClipPos = mul(float4(worldPos, 1.0), g_prevViewProj);
    float2 prevUV = (prevClipPos.xy / prevClipPos.w) * 0.5 + 0.5;

    // Motion vector = current - previous
    float2 motion = uv - prevUV;
    g_motionVectors[dispatchThreadID.xy] = float4(motion, 0, 0);
}
```

### Week 10: Polish & Optimization

**ImGui controls**:

```cpp
const char* dlssModes[] = { "Off", "Quality", "Balanced", "Performance", "Ultra Performance" };
int currentMode = 1;  // Quality

ImGui::Combo("DLSS Mode", &currentMode, dlssModes, 5);

if (currentMode == 0) {
    // Render at native 4K (no DLSS)
} else {
    // Apply DLSS with selected mode
}
```

**Performance monitoring**:

```cpp
// Query DLSS stats
sl::DLSSStats stats = {};
slDLSSGetStats(&stats);

ImGui::Text("DLSS: %.1f FPS (render: %d×%d, output: %d×%d)",
            stats.fps,
            stats.renderWidth, stats.renderHeight,
            stats.outputWidth, stats.outputHeight);
```

**Performance**: 120+ FPS @ 4K (rendering at 1080p internally)
**Quality**: Near-native 4K with AI upscaling

**Timeline**: 2-3 weeks including motion vector implementation

---

## Advanced Optimizations (Optional, Week 11+)

### Shader Execution Reordering (SER)

**Reference**: `rtxpt/ShaderDynamic/Source/External/NVAPI/nvShaderExtnEnums.h`

RTXPT uses NVAPI for SER to improve ray coherence:

```hlsl
// Requires NVAPI extension
#include "nvHLSLExtns.h"

// In ray tracing loop
NvReorderThread(hitShaderIndex, coherenceHint);
```

**Benefit**: 24-40% speedup on RTX 40-series
**Complexity**: Requires NVAPI integration, vendor-specific

### Opacity Micromaps (OMM)

**Reference**: `rtxpt/omm-lib.dll`

For alpha-tested geometry (if you add particle billboards):

```cpp
// Build OMM from alpha texture
// Attach to BLAS
// Automatic faster alpha testing
```

**Benefit**: 2-3× faster for alpha-tested geometry
**Use case**: If you add billboard particles with alpha cutout

### ReSTIR GI (RTXDI Alternative)

**Reference**: `rtxpt/Rtxpt/RTXDI/` (ReSTIR GI shaders)

Instead of multi-bounce compute passes, use RTXDI's ReSTIR GI:

```hlsl
// Reservoir-based indirect illumination
// Spatial + temporal reuse
// 1 ray/pixel with reservoir sampling
```

**Benefit**: Potentially better quality than manual multi-bounce at same cost
**Complexity**: High - requires understanding ReSTIR paper

---

## Summary: What Can Be Directly Reused from RTXPT

### ✅ Can Reuse Directly (Copy/Paste)

1. **NRD SDK** - Standalone library, drop-in integration
   - DLLs: `NRD.dll`
   - Headers: `external/NRD/Include/`
   - No Donut dependency

2. **Streamline DLLs** - Just copy files
   - `sl.*.dll`, `nvngx_dlss.dll`
   - Headers from GitHub

3. **Shader patterns** - Algorithm concepts
   - Bounce counting logic (PathTracer.hlsli:42-47)
   - Fibonacci sampling (already using)
   - BRDF formulas (can reference)

### ❌ Cannot Reuse Directly (Donut/NVRHI Coupled)

1. **Path tracer shaders** - Heavily use Donut resource binding
2. **C++ framework** - Tied to NVRHI command buffers
3. **Asset pipeline** - glTF loader requires Donut

### 🔄 Reuse with Adaptation

1. **NRD integration patterns** - Adapt to D3D12 (no NVRHI)
2. **Multi-bounce logic** - Port iterative approach for RayQuery
3. **DLSS setup code** - Adapt to your swap chain

---

## Learning Resources from RTXPT v1.7.0

**For NRD Integration**:
- `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/NRD/DenoiserNRD.hlsli` - Denoiser setup
- `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/CMakeLists.txt` (line 150-200) - NRD linking

**For Multi-Bounce Logic**:
- `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/Shaders/PathTracer/PathTracer.hlsli` - Bounce counting
- `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/Shaders/PathTracer/PathTracerNEE.hlsli` - Next event estimation

**For DLSS/Streamline**:
- Search RTXPT source for "Streamline" initialization
- DLL files in root directory provide working binaries

**For BRDF Models**:
- `/mnt/d/users/dilli/androidstudioprojects/rtxpt/Rtxpt/Shaders/PathTracer/Rendering/Materials/BxDF.hlsli` - BRDF implementations

---

## Recommended Prioritization

### Conservative Path (Stability Focus)
1. Phase 1 (Week 1) - Balanced GI
2. Phase 2 (Week 2) - Temporal accumulation
3. **STOP HERE** - Evaluate before proceeding
4. Phase 3 (Weeks 3-5) - NRD (if needed)

### Aggressive Path (Feature Focus)
1. Phases 1-2 (Weeks 1-2) - Balanced GI + temporal
2. Phase 3 (Weeks 3-5) - NRD denoising
3. Phase 4 (Weeks 6-7) - Multi-bounce GI
4. Phase 5 (Weeks 8-10) - DLSS

### Hybrid Path (Recommended)
1. Phases 1-2 (Weeks 1-2) - Balanced GI + temporal → **115 FPS**
2. **Evaluate visual quality** - Is multi-bounce needed?
3. If yes → Phase 4 (multi-bounce) then Phase 3 (NRD)
4. If no → Phase 5 (DLSS for 4K support)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**RTXPT Version**: v1.7.0
**Status**: Ready for phased implementation
