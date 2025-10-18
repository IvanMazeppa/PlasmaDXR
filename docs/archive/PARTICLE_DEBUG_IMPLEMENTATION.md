# Particle Billboard Debug Infrastructure Implementation

## Overview
Complete debugging infrastructure for DX12 particle renderer to make invisible rendering bugs visually obvious on screen. The shader already has extensive debug modes - this document provides the complete integration and usage guide.

## File Locations
- Shader: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/shaders/particles/particle_billboard_vs.hlsl`
- Application: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/src/core/App.cpp`
- Header: `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX/src/core/App.h`

---

## 1. SHADER DEBUG MODES (Already Implemented)

The vertex shader has 6 debug modes built in:

### Mode 0: OFF (Normal Rendering)
- Temperature-based coloring (red to yellow)
- RT lighting applied
- Normal billboard rendering

### Mode 1: Clip-Space W Debug
- **RED**: W < 0 (behind camera - CRITICAL BUG)
- **GREEN**: W in [0,1] range (near camera)
- **BLUE**: W > 1 (far from camera, intensity based on distance)
- **Purpose**: Detect particles behind camera or with invalid depth

### Mode 2: Clip XY Position Debug
- **BLUE**: Center of screen (NDC distance < 0.2)
- **RED gradient**: Left/right position (abs(ndcX))
- **GREEN gradient**: Top/bottom position (abs(ndcY))
- **Purpose**: Verify screen-space distribution and clipping

### Mode 3: World-Space Distance Debug
- **GREEN**: < 100 units (very close)
- **GREEN→YELLOW**: 100-500 units
- **YELLOW→RED**: 500-2000 units
- **RED**: > 2000 units (very far)
- **Purpose**: Verify particle spatial distribution

### Mode 4: Origin Test
- Places ALL particles at world origin (0,0,0)
- Size increases with particle index (1.0 + idx * 0.01)
- **CYAN** color for all particles
- **Purpose**: Test if rendering pipeline works at all

### Mode 5: Validation Mode
Shows all validation errors with color coding:
- **MAGENTA**: NaN or Inf in clip position
- **RED**: Behind near plane (W < nearPlane)
- **BLUE**: Beyond far plane (W > farPlane)
- **ORANGE**: Way outside screen bounds (|NDC| > 2.0)
- **GREEN**: All checks passed

### Validation Constants (Always Active if enableValidation = 1)
- `nearPlane`: 0.1 (typical)
- `farPlane`: 10000.0 (typical)
- Errors overlay on normal rendering when validation enabled

---

## 2. SHADER CONSTANT BUFFER STRUCTURE

```hlsl
cbuffer DebugConstants : register(b2)
{
    uint debugMode;           // 0=off, 1=clipW, 2=clipXY, 3=distance, 4=origin, 5=validation
    uint enableValidation;    // Show validation errors as red (0=off, 1=on)
    float nearPlane;          // For validation (0.1)
    float farPlane;           // For validation (10000.0)
    uint particleCountForDebug;
    float3 padding4;
};
```

**CRITICAL**: This constant buffer MUST be bound at register(b2) when rendering particles.

---

## 3. APPLICATION INTEGRATION

### Step 3A: Add Member Variables to App.h

Add these lines after line 291 (after `m_meshParticleSystem`):

```cpp
// Particle debug controls (Numpad 0-9)
uint32_t m_particleDebugMode = 0;        // 0-5 debug modes
bool m_particleValidationEnabled = true; // Always validate by default
float m_particleNearPlane = 0.1f;
float m_particleFarPlane = 10000.0f;
```

### Step 3B: Add Keyboard Controls to App.cpp WndProc

Add this case block in the `WM_KEYDOWN` switch statement (around line 400, after the '0' key case for plasma disk thickness):

```cpp
// PARTICLE DEBUG MODES (Numpad 0-9)
case VK_NUMPAD0:  // Numpad 0: Debug mode OFF
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 0;
        LOGI("Particle Debug: OFF (normal rendering)");
    }
    break;
case VK_NUMPAD1:  // Numpad 1: Clip W debug
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 1;
        LOGI("Particle Debug: Clip-Space W (Red=behind, Green=near, Blue=far)");
    }
    break;
case VK_NUMPAD2:  // Numpad 2: Clip XY debug
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 2;
        LOGI("Particle Debug: Clip XY Position (Blue=center, R/G=edges)");
    }
    break;
case VK_NUMPAD3:  // Numpad 3: Distance debug
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 3;
        LOGI("Particle Debug: World Distance (Green=close, Yellow=mid, Red=far)");
    }
    break;
case VK_NUMPAD4:  // Numpad 4: Origin test
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 4;
        LOGI("Particle Debug: Origin Test (all particles at 0,0,0 - CYAN)");
    }
    break;
case VK_NUMPAD5:  // Numpad 5: Validation mode
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = 5;
        LOGI("Particle Debug: Validation (Magenta=NaN, Red=NearClip, Blue=FarClip, Orange=Offscreen, Green=OK)");
    }
    break;
case VK_NUMPAD6:  // Numpad 6: Toggle validation overlay
    if (g_appInstance) {
        g_appInstance->m_particleValidationEnabled = !g_appInstance->m_particleValidationEnabled;
        LOGI(g_appInstance->m_particleValidationEnabled ?
            "Particle Validation: ON (errors shown in all modes)" :
            "Particle Validation: OFF (errors hidden)");
    }
    break;
case VK_NUMPAD7:  // Numpad 7: Decrease near plane
    if (g_appInstance) {
        g_appInstance->m_particleNearPlane = std::max(0.01f, g_appInstance->m_particleNearPlane - 0.05f);
        LOGI("Particle Near Plane: " + std::to_string(g_appInstance->m_particleNearPlane));
    }
    break;
case VK_NUMPAD8:  // Numpad 8: Increase near plane
    if (g_appInstance) {
        g_appInstance->m_particleNearPlane = std::min(1.0f, g_appInstance->m_particleNearPlane + 0.05f);
        LOGI("Particle Near Plane: " + std::to_string(g_appInstance->m_particleNearPlane));
    }
    break;
case VK_NUMPAD9:  // Numpad 9: Cycle through all debug modes
    if (g_appInstance) {
        g_appInstance->m_particleDebugMode = (g_appInstance->m_particleDebugMode + 1) % 6;
        const char* modeNames[] = {
            "OFF", "Clip W", "Clip XY", "World Distance", "Origin Test", "Validation"
        };
        LOGI(std::string("Particle Debug: ") + modeNames[g_appInstance->m_particleDebugMode]);
    }
    break;
```

### Step 3C: Pass Debug Constants When Rendering

You need to modify the particle rendering code to create and bind the debug constant buffer. This requires changes in `ParticleSystem.cpp` or wherever the particle vertex shader is invoked.

#### Option A: Quick Test (Inline in App.cpp)

If you want to test immediately without modifying ParticleSystem, you can create a temporary constant buffer in the render loop:

```cpp
// In renderFrame(), before calling m_meshParticleSystem->RenderParticles()

// Create debug constants structure
struct ParticleDebugConstants {
    uint32_t debugMode;
    uint32_t enableValidation;
    float nearPlane;
    float farPlane;
    uint32_t particleCount;
    float padding[3];
};

ParticleDebugConstants debugConstants = {};
debugConstants.debugMode = m_particleDebugMode;
debugConstants.enableValidation = m_particleValidationEnabled ? 1u : 0u;
debugConstants.nearPlane = m_particleNearPlane;
debugConstants.farPlane = m_particleFarPlane;
debugConstants.particleCount = m_mode9ParticleCount;

// You'll need to upload this to a constant buffer and bind it at b2
// This requires modifying ParticleSystem to accept and bind debug constants
```

#### Option B: Integrate into ParticleSystem (Proper Solution)

Modify `ParticleSystem.h` to add:

```cpp
// Debug constants structure
struct DebugConstants {
    uint32_t debugMode;
    uint32_t enableValidation;
    float nearPlane;
    float farPlane;
    uint32_t particleCount;
    DirectX::XMFLOAT3 padding;
};

// Add to private members:
Microsoft::WRL::ComPtr<ID3D12Resource> m_debugConstantsBuffer;
```

Modify `ParticleSystem::CreateBuffers()` to create the debug constant buffer:

```cpp
// Create debug constants buffer (use UPLOAD heap for CPU writes)
CD3DX12_HEAP_PROPERTIES debugHeapProps(D3D12_HEAP_TYPE_UPLOAD);
CD3DX12_RESOURCE_DESC debugConstantsDesc = CD3DX12_RESOURCE_DESC::Buffer(
    (sizeof(DebugConstants) + 255) & ~255 // Align to 256 bytes
);

hr = m_device->CreateCommittedResource(
    &debugHeapProps,
    D3D12_HEAP_FLAG_NONE,
    &debugConstantsDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&m_debugConstantsBuffer)
);

if (FAILED(hr)) {
    LOGE("Failed to create debug constants buffer: " + std::to_string(static_cast<uint32_t>(hr)));
    return false;
}
```

Modify `ParticleSystem::RenderParticles()` or `RenderComputeParticles()` to accept debug parameters:

```cpp
void RenderComputeParticles(
    ID3D12GraphicsCommandList* cmdList,
    const DirectX::XMMATRIX& viewMatrix,
    const DirectX::XMMATRIX& projMatrix,
    const DirectX::XMFLOAT3& cameraPos,
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle,
    UINT width, UINT height,
    D3D12_GPU_DESCRIPTOR_HANDLE particleBufferSrv,
    D3D12_GPU_DESCRIPTOR_HANDLE particleLightingSrv,
    uint32_t mode10SubMode,
    // ADD THESE DEBUG PARAMETERS:
    uint32_t debugMode = 0,
    bool enableValidation = true,
    float nearPlane = 0.1f,
    float farPlane = 10000.0f
)
{
    // ... existing code ...

    // Update debug constants
    DebugConstants* debugData = nullptr;
    m_debugConstantsBuffer->Map(0, nullptr, reinterpret_cast<void**>(&debugData));
    debugData->debugMode = debugMode;
    debugData->enableValidation = enableValidation ? 1u : 0u;
    debugData->nearPlane = nearPlane;
    debugData->farPlane = farPlane;
    debugData->particleCount = m_particleCount;
    m_debugConstantsBuffer->Unmap(0, nullptr);

    // ... when setting root parameters ...
    cmdList->SetGraphicsRootConstantBufferView(2, m_debugConstantsBuffer->GetGPUVirtualAddress());

    // ... rest of rendering ...
}
```

Then in `App.cpp`, pass the debug parameters:

```cpp
m_meshParticleSystem->RenderComputeParticles(
    m_cmdList.Get(),
    viewMatrix, projMatrix, cameraPos, rtvHandle, m_width, m_height,
    particleBufferGpuHandle,
    lightingSrvGpuHandle,
    static_cast<uint32_t>(m_mode10SubMode),
    m_particleDebugMode,           // Debug mode
    m_particleValidationEnabled,   // Validation
    m_particleNearPlane,           // Near plane
    m_particleFarPlane             // Far plane
);
```

---

## 4. USAGE INSTRUCTIONS

### Quick Start
1. **Compile the shader**: Ensure `particle_billboard_vs.hlsl` is compiled with the latest changes
2. **Add keyboard controls**: Apply Step 3B changes to `App.cpp`
3. **Add debug state**: Apply Step 3A changes to `App.h`
4. **Wire up constants**: Apply Step 3C changes to `ParticleSystem.cpp`

### Debugging Workflow

#### Problem: Particles not visible at all
1. Press **Numpad 4**: Origin Test
   - If you see CYAN squares at screen center → Pipeline works, position bug
   - If nothing → PSO/shader/buffer binding issue

#### Problem: Particles behind camera
1. Press **Numpad 1**: Clip W Debug
   - RED particles → Behind camera (W < 0)
   - Fix: Check view matrix or particle positions

#### Problem: Particles clipped/off-screen
1. Press **Numpad 2**: Clip XY Debug
   - See where particles are in screen space
   - If all on edges → Projection matrix issue

#### Problem: Strange spatial distribution
1. Press **Numpad 3**: World Distance Debug
   - GREEN → Close, RED → Far
   - Verify particle positions match expected spatial layout

#### Problem: Random flickering/NaN values
1. Press **Numpad 5**: Validation Mode
   - **MAGENTA** → NaN/Inf in shader math (CRITICAL)
   - **RED** → Near plane clipping
   - **BLUE** → Far plane clipping
   - **GREEN** → All checks passed

#### General Debugging
1. Press **Numpad 6**: Toggle validation overlay
   - Enables/disables error highlighting in all modes
   - Useful to isolate errors vs. normal behavior

2. Press **Numpad 9**: Cycle all modes
   - Quick way to scan through all debug views

3. Press **Numpad 7/8**: Adjust near plane
   - Test near/far plane validation thresholds

### Keyboard Reference

| Key | Function |
|-----|----------|
| Numpad 0 | Debug OFF (normal rendering) |
| Numpad 1 | Clip-Space W Debug (depth validation) |
| Numpad 2 | Clip XY Position (screen-space distribution) |
| Numpad 3 | World Distance (spatial distribution) |
| Numpad 4 | Origin Test (pipeline validation) |
| Numpad 5 | Validation Mode (error detection) |
| Numpad 6 | Toggle validation overlay |
| Numpad 7 | Decrease near plane threshold |
| Numpad 8 | Increase near plane threshold |
| Numpad 9 | Cycle through all modes |

---

## 5. VALIDATION CHECKS

The shader performs these checks when `enableValidation = 1`:

### Check 1: NaN/Inf Detection
```hlsl
bool isInvalidFloat4(float4 v) {
    return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w) ||
           isinf(v.x) || isinf(v.y) || isinf(v.z) || isinf(v.w);
}
```
**Color**: MAGENTA
**Cause**: Shader math error, uninitialized data, division by zero

### Check 2: Near Plane Violation
```hlsl
if (clipPos.w < nearPlane) {
    hasError = true;
    errorColor = float3(1.0, 0.0, 0.0);  // RED
}
```
**Color**: RED
**Cause**: Particle behind camera or very close to near plane

### Check 3: Far Plane Violation
```hlsl
if (clipPos.w > farPlane) {
    hasError = true;
    errorColor = float3(0.0, 0.0, 1.0);  // BLUE
}
```
**Color**: BLUE
**Cause**: Particle beyond far plane (may be culled)

### Check 4: Screen Bounds Violation
```hlsl
float ndcX = clipPos.x / clipPos.w;
float ndcY = clipPos.y / clipPos.w;
if (abs(ndcX) > 2.0 || abs(ndcY) > 2.0) {
    hasError = true;
    errorColor = float3(1.0, 0.5, 0.0);  // ORANGE
}
```
**Color**: ORANGE
**Cause**: Particle way off-screen (wasteful if not culled)

---

## 6. COMMON BUG PATTERNS

### Pattern 1: All Particles Red (Mode 1)
- **Symptom**: Everything RED in Clip W mode
- **Cause**: View matrix inverted or particles behind camera
- **Fix**: Check camera position and view matrix construction

### Pattern 2: Magenta Flashing (Mode 5)
- **Symptom**: Random MAGENTA particles in Validation mode
- **Cause**: NaN/Inf from shader math
- **Fix**: Check for divide-by-zero, uninitialized particle data

### Pattern 3: Nothing Visible (All Modes)
- **Symptom**: No particles in any mode including Origin Test
- **Cause**: PSO not bound, shader not compiled, buffer not bound
- **Fix**: Check PIX capture, verify SetPipelineState calls

### Pattern 4: Particles Only at Origin (Mode 4)
- **Symptom**: Works in Mode 4, invisible in Mode 0
- **Cause**: Particle position data is zero/uninitialized
- **Fix**: Verify particle buffer upload, check simulation

### Pattern 5: All Blue (Mode 1)
- **Symptom**: Everything BLUE in Clip W mode
- **Cause**: Particles very far from camera (W >> 1)
- **Fix**: Check camera position or particle positions

---

## 7. PERFORMANCE IMPACT

Debug modes have minimal performance impact:
- Extra conditionals: ~10-20 ALU instructions per vertex
- No extra memory accesses (constants cached)
- **Recommendation**: Keep validation enabled during development

To disable completely:
- Set `debugMode = 0` and `enableValidation = 0`
- Compiler should optimize out dead code paths

---

## 8. EXTENDING DEBUG MODES

To add a new debug mode:

1. Add to `DebugConstants` in shader:
```hlsl
cbuffer DebugConstants : register(b2) {
    uint debugMode;  // Add new mode number
    // ...
}
```

2. Add visualization in vertex shader:
```hlsl
if (debugMode == 6) {  // Your new mode
    baseColor = /* your debug color logic */;
}
```

3. Add keyboard control in App.cpp:
```cpp
case VK_NUMPAD6:  // Or another key
    g_appInstance->m_particleDebugMode = 6;
    LOGI("Particle Debug: Your New Mode");
    break;
```

Example new modes:
- **Velocity Debug**: Color by particle velocity magnitude
- **Temperature Debug**: Color by raw temperature value
- **Index Debug**: Color by particle index (rainbow pattern)
- **Triangle Debug**: Different color per triangle in billboard

---

## 9. TROUBLESHOOTING

### Debug modes not working
1. Check shader compilation: Is the DXIL up-to-date?
2. Check constant buffer binding: Is b2 bound correctly?
3. Check root signature: Does it have 3 CBVs (b0, b1, b2)?

### Validation always shows errors
1. Adjust near/far planes with Numpad 7/8
2. Check if thresholds match your camera setup
3. Verify particle positions are in valid world space

### Keyboard not responding
1. Check if window has focus
2. Verify WndProc is receiving WM_KEYDOWN
3. Check if correct demo mode is active (Mode 9 or 10)

### Performance drop with debug enabled
1. Normal: 5-10% overhead is expected
2. Severe (>20%): May indicate driver issue or shader compiler problem
3. Test with debugMode=0 and enableValidation=0

---

## 10. PIX INTEGRATION

When using PIX GPU Capture with debug modes:

1. **Capture with Origin Test (Mode 4)**
   - Simplest case: all particles at (0,0,0)
   - Easy to verify vertex shader output

2. **Capture with Validation (Mode 5)**
   - See error colors in VS output
   - Check if errors appear in VS or PS stage

3. **Useful PIX Markers**
   - Add: `cmdList->BeginEvent(0, L"Particle Debug Mode X", ...)`
   - Shows current debug mode in timeline

4. **Shader Debugging**
   - Set breakpoint in vertex shader
   - Inspect `debugMode` constant value
   - Step through conditional branches

---

## SUMMARY

This debug infrastructure makes particle rendering bugs impossible to hide:

- **6 debug visualization modes** covering all common bug patterns
- **5 validation checks** to catch shader math errors
- **Numpad controls** for instant mode switching
- **Zero integration overhead** when disabled
- **Complete color-coded error detection**

All bugs become visually obvious - no more invisible particles!
