# DLSS Phase 3 Integration Roadmap

**Branch:** 0.11.3
**Date:** 2025-10-29
**Status:** In Progress (Motion Vector Buffer Created ✅)

---

## Phase 3 Goal

Integrate DLSS Ray Reconstruction denoising to replace temporal filtering in the PCSS shadow system. Target: 1-2 rays/light with DLSS quality matching 8-ray traditional shadows.

---

## ✅ Completed Tasks

### 1. Motion Vector Buffer Created
**Files Modified:**
- `src/particles/ParticleRenderer_Gaussian.h` (lines 200-205)
- `src/particles/ParticleRenderer_Gaussian.cpp` (lines 194-257)

**What Was Done:**
- Added `m_motionVectorBuffer` (RG16_FLOAT format)
- Created SRV and UAV descriptors
- Buffer size: 4MB @ 1920x1080 (32-bit per pixel)
- Follows same pattern as shadow buffers

**Log Output:**
```
[INFO] Creating motion vector buffer for DLSS...
[INFO]   Resolution: 1920x1080 pixels
[INFO]   Format: RG16_FLOAT (32-bit per pixel, 2 components)
[INFO]   Buffer size: 7 MB
[INFO] Created motion vector buffer: SRV=0x..., UAV=0x...
```

---

## ⏳ Remaining Tasks (4-5 hours total)

### Task 2: Motion Vector Compute Shader (1 hour)

**Goal:** Pre-compute screen-space motion vectors from particle velocities

**Create New File:** `shaders/particles/compute_motion_vectors.hlsl`

**Implementation:**
```hlsl
// Motion vector compute shader
// Converts particle world-space velocities to screen-space motion vectors

struct Particle {
    float3 position;    // World position
    float temperature;
    float3 velocity;    // World velocity (units/second)
    float density;
};

StructuredBuffer<Particle> g_particles : register(t0);
RWTexture2D<float2> g_motionVectors : register(u0);  // Output MV buffer

cbuffer MotionVectorConstants : register(b0) {
    row_major float4x4 viewProj;
    row_major float4x4 prevViewProj;
    float3 cameraPos;
    float deltaTime;
    uint screenWidth;
    uint screenHeight;
    uint particleCount;
    float padding;
};

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // Get pixel coordinates
    uint2 pixelCoord = DTid.xy;
    if (pixelCoord.x >= screenWidth || pixelCoord.y >= screenHeight)
        return;

    // Convert pixel to NDC
    float2 uv = (float2(pixelCoord) + 0.5f) / float2(screenWidth, screenHeight);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    // Find closest particle to this pixel (simple approach)
    // For better quality, could ray march and find first hit
    float minDist = 1e10;
    float3 closestVelocity = float3(0, 0, 0);
    float3 closestPos = float3(0, 0, 0);

    // OPTIMIZATION: Could use spatial acceleration here
    for (uint i = 0; i < particleCount; i++) {
        Particle p = g_particles[i];

        // Project particle to screen
        float4 clipPos = mul(viewProj, float4(p.position, 1.0f));
        clipPos /= clipPos.w;

        float2 screenPos = float2(clipPos.x, -clipPos.y);
        float dist = length(screenPos - ndc);

        if (dist < minDist) {
            minDist = dist;
            closestVelocity = p.velocity;
            closestPos = p.position;
        }
    }

    // Compute motion vector from velocity
    // Current position
    float4 currClip = mul(viewProj, float4(closestPos, 1.0f));
    currClip /= currClip.w;

    // Previous position (position - velocity * deltaTime)
    float3 prevPos = closestPos - closestVelocity * deltaTime;
    float4 prevClip = mul(prevViewProj, float4(prevPos, 1.0f));
    prevClip /= prevClip.w;

    // Motion vector in screen space (pixels)
    float2 motionVec = (currClip.xy - prevClip.xy) * float2(screenWidth, screenHeight) * 0.5f;

    // Write to output
    g_motionVectors[pixelCoord] = motionVec;
}
```

**C++ Integration (ParticleRenderer_Gaussian.cpp):**
```cpp
// Add to class members (ParticleRenderer_Gaussian.h):
#ifdef ENABLE_DLSS
Microsoft::WRL::ComPtr<ID3D12PipelineState> m_motionVectorPSO;
Microsoft::WRL::ComPtr<ID3D12RootSignature> m_motionVectorRootSig;
#endif

// Add to Initialize() after CreatePipeline():
#ifdef ENABLE_DLSS
if (!CreateMotionVectorPipeline()) {
    LOG_ERROR("Failed to create motion vector compute pipeline");
    return false;
}
#endif

// New function CreateMotionVectorPipeline() - follows CreatePipeline() pattern
```

**Dispatch Before Gaussian Render:**
```cpp
// In Render() function, before main Gaussian dispatch:
#ifdef ENABLE_DLSS
if (m_dlssSystem && m_dlssFeatureCreated) {
    // PIX event marker
    PIXBeginEvent(cmdList, 0, "Compute Motion Vectors");

    // Set pipeline
    cmdList->SetPipelineState(m_motionVectorPSO.Get());
    cmdList->SetComputeRootSignature(m_motionVectorRootSig.Get());

    // Bind resources
    cmdList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(1, particleBuffer->GetGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(2, m_motionVectorUAVGPU);

    // Dispatch (8x8 thread groups)
    uint32_t dispatchX = (m_screenWidth + 7) / 8;
    uint32_t dispatchY = (m_screenHeight + 7) / 8;
    cmdList->Dispatch(dispatchX, dispatchY, 1);

    // UAV barrier
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_motionVectorBuffer.Get());
    cmdList->ResourceBarrier(1, &barrier);

    PIXEndEvent(cmdList);
}
#endif
```

---

### Task 3: Implement EvaluateRayReconstruction() (30 min)

**File:** `src/dlss/DLSSSystem.cpp`

**Current Stub (line 182+):**
```cpp
bool DLSSSystem::EvaluateRayReconstruction(
    ID3D12GraphicsCommandList* cmdList,
    const RayReconstructionParams& params
) {
    // TODO: Implement
    return false;
}
```

**Full Implementation:**
```cpp
bool DLSSSystem::EvaluateRayReconstruction(
    ID3D12GraphicsCommandList* cmdList,
    const RayReconstructionParams& params
) {
    if (!m_rrFeature || !cmdList) {
        LOG_ERROR("DLSS: Feature not created or invalid command list");
        return false;
    }

    // Allocate evaluation parameters
    NVSDK_NGX_Parameter* evalParams = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_AllocateParameters(&evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR("DLSS: Failed to allocate eval parameters");
        return false;
    }

    // Set input/output resources (using typed setters!)
    NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_Color,
                                         params.inputNoisySignal);
    NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_Output,
                                         params.outputDenoisedSignal);
    NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_MotionVectors,
                                         params.inputMotionVectors);

    // Optional inputs (can be nullptr)
    if (params.inputDiffuseAlbedo) {
        NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_GBuffer_Albedo,
                                             params.inputDiffuseAlbedo);
    }
    if (params.inputNormals) {
        NVSDK_NGX_Parameter_SetD3d12Resource(evalParams, NVSDK_NGX_Parameter_GBuffer_Normals,
                                             params.inputNormals);
    }

    // Set jitter offsets (we don't use TAA, so 0)
    NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Jitter_Offset_X,
                             params.jitterOffsetX);
    NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Jitter_Offset_Y,
                             params.jitterOffsetY);

    // Motion vector scale (1.0 = already in pixel space)
    NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);

    // Denoiser strength
    NVSDK_NGX_Parameter_SetF(evalParams, NVSDK_NGX_Parameter_Denoise, m_denoiserStrength);

    // Reset flag (set to 1 on scene changes, 0 for temporal coherence)
    NVSDK_NGX_Parameter_SetI(evalParams, NVSDK_NGX_Parameter_Reset, 0);

    // Evaluate Ray Reconstruction
    result = NVSDK_NGX_D3D12_EvaluateFeature(cmdList, m_rrFeature, evalParams, nullptr);

    NVSDK_NGX_D3D12_DestroyParameters(evalParams);

    if (NVSDK_NGX_FAILED(result)) {
        char hexStr[16];
        sprintf_s(hexStr, "%08X", static_cast<uint32_t>(result));
        LOG_ERROR("DLSS: Evaluation failed: 0x{}", hexStr);
        return false;
    }

    return true;
}
```

---

### Task 4: Integrate DLSS into Gaussian Renderer (1 hour)

**File:** `src/particles/ParticleRenderer_Gaussian.cpp` in `Render()` function

**Current Shadow Pipeline (around line 450+):**
```cpp
// Current: Shadow rays → temporal blend → final shadow
// After 8-ray shadow pass, ping-pong temporal filtering happens
```

**New DLSS Pipeline:**
```cpp
#ifdef ENABLE_DLSS
if (m_enableDLSS && m_dlssSystem && m_dlssFeatureCreated) {
    // === DLSS RAY RECONSTRUCTION PIPELINE ===
    PIXBeginEvent(cmdList, 0, "DLSS Ray Reconstruction");

    // Transition shadow buffer to SRV for DLSS input
    CD3DX12_RESOURCE_BARRIER barriers[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_shadowBuffer[m_currentShadowIndex].Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        ),
        CD3DX12_RESOURCE_BARRIER::Transition(
            m_motionVectorBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        )
    };
    cmdList->ResourceBarrier(2, barriers);

    // Prepare DLSS parameters
    DLSSSystem::RayReconstructionParams dlssParams = {};
    dlssParams.inputNoisySignal = m_shadowBuffer[m_currentShadowIndex].Get();
    dlssParams.inputMotionVectors = m_motionVectorBuffer.Get();
    dlssParams.outputDenoisedSignal = m_shadowBuffer[1 - m_currentShadowIndex].Get();
    dlssParams.width = m_screenWidth;
    dlssParams.height = m_screenHeight;
    dlssParams.jitterOffsetX = 0.0f;  // No TAA
    dlssParams.jitterOffsetY = 0.0f;

    // Call DLSS Ray Reconstruction
    if (!m_dlssSystem->EvaluateRayReconstruction(
            static_cast<ID3D12GraphicsCommandList*>(cmdList),
            dlssParams)) {
        LOG_ERROR("DLSS evaluation failed, falling back to temporal filtering");
        // Fall through to temporal filtering
    } else {
        // DLSS succeeded - swap buffers
        m_currentShadowIndex = 1 - m_currentShadowIndex;

        // Transition back to UAV for next frame
        CD3DX12_RESOURCE_BARRIER backBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_shadowBuffer[m_currentShadowIndex].Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        cmdList->ResourceBarrier(1, &backBarrier);
    }

    PIXEndEvent(cmdList);
} else
#endif
{
    // Original temporal filtering path
    // ... existing code ...
}
```

---

### Task 5: Add ImGui Controls (30 min)

**File:** `src/core/Application.cpp`

**Add to class members (Application.h):**
```cpp
#ifdef ENABLE_DLSS
bool m_enableDLSS = false;                    // Toggle DLSS denoising
float m_dlssDenoiserStrength = 1.0f;          // 0.0-2.0
#endif
```

**ImGui Controls (add after existing DLSS toggle):**
```cpp
#ifdef ENABLE_DLSS
if (m_dlssSystem) {
    ImGui::Separator();
    ImGui::Text("Shadow Denoising Mode");

    // Radio buttons for mode selection
    static int denoisingMode = 0;  // 0=Temporal, 1=DLSS
    if (ImGui::RadioButton("Temporal Filtering (Default)", &denoisingMode, 0)) {
        m_enableDLSS = false;
        LOG_INFO("Shadow denoising: Temporal filtering");
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("DLSS Ray Reconstruction", &denoisingMode, 1)) {
        m_enableDLSS = true;
        LOG_INFO("Shadow denoising: DLSS-RR enabled");
    }

    if (m_enableDLSS) {
        ImGui::Indent();
        if (ImGui::SliderFloat("DLSS Denoiser Strength", &m_dlssDenoiserStrength, 0.0f, 2.0f)) {
            m_dlssSystem->SetDenoiserStrength(m_dlssDenoiserStrength);
        }
        ImGui::Text("Expected: 2-4x faster than 8-ray shadows");
        ImGui::Unindent();
    }
}
#endif
```

**Pass enableDLSS flag to renderer:**
```cpp
// In Render() function when setting constants:
#ifdef ENABLE_DLSS
constants.enableDLSS = m_enableDLSS ? 1 : 0;
#endif
```

---

### Task 6: Testing & Benchmarking (1 hour)

**Visual Quality Test:**
1. Run with 1 ray/light + Temporal filtering (baseline)
2. Run with 1 ray/light + DLSS
3. Run with 8 rays/light (ground truth)
4. Compare shadow quality - DLSS should match 8-ray

**Performance Benchmark:**
```
Test Conditions: 10K particles, 13 lights, 1920x1080

Configuration               | FPS    | Shadow Quality
----------------------------|--------|----------------
1 ray + Temporal (current)  | 115    | Noisy
1 ray + DLSS (new)          | 120    | Clean (matches 8-ray)
8 rays (ground truth)       | 60     | Clean
```

**Expected DLSS Benefits:**
- Same performance as 1-ray temporal (~115-120 FPS)
- Visual quality matches 8-ray traditional
- Faster convergence (2-3 frames vs 8 frames for temporal)
- Better temporal stability (AI-powered)

---

## Implementation Order (Next Session)

1. **Start here:** Create motion vector compute shader (1 hour)
2. Compile and integrate motion vector dispatch (30 min)
3. Implement EvaluateRayReconstruction() (30 min)
4. Integrate DLSS into Gaussian renderer (1 hour)
5. Add ImGui controls (30 min)
6. Test and benchmark (1 hour)

**Total:** 4.5 hours estimated

---

## Key Files Reference

**Headers:**
- `src/particles/ParticleRenderer_Gaussian.h` - Motion vector buffer members
- `src/dlss/DLSSSystem.h` - RayReconstructionParams struct
- `src/core/Application.h` - DLSS toggle controls

**Implementation:**
- `src/particles/ParticleRenderer_Gaussian.cpp` - Motion vector creation, DLSS integration
- `src/dlss/DLSSSystem.cpp` - EvaluateRayReconstruction() implementation
- `src/core/Application.cpp` - ImGui controls

**Shaders:**
- `shaders/particles/compute_motion_vectors.hlsl` - NEW FILE (create this)
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Reads denoised shadows

---

## Success Criteria

✅ Motion vectors generated correctly (verify in PIX)
✅ DLSS evaluation runs without errors
✅ Shadow quality matches 8-ray traditional
✅ Performance maintains 115-120 FPS @ 10K particles
✅ No crashes or GPU timeouts
✅ ImGui toggle works correctly

---

**Last Updated:** 2025-10-29
**Next Session Start:** Task 2 - Motion Vector Compute Shader
**Estimated Completion:** 4-5 hours
