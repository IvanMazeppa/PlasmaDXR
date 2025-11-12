# Material System Implementation Progress

**Date:** 2025-11-12
**Branch:** `feature/gaussian-material-system`
**Status:** Phase 2 Complete ✅ | Phase 3 Pending ⏳

---

## Problem Statement

**Current Issue:** Muted, brown particle appearance due to:
- Physical emission DISABLED (`strength: 0.00`)
- All particles identical (no material diversity)
- 8 blue lights scattering off uniform particles → muddy brown color
- Lack of vibrant colors and self-emission

**Goal:** Transform into spectacular galactic core with:
- Vibrant, diverse colors (hot oranges, brilliant whites, wispy blues)
- Material-specific emission (stars glow 8× brighter than plasma)
- Heterogeneous particle types with distinct visual properties
- 90-120 FPS performance target maintained

---

## Sprint 1 Overview

**Total Time:** 10-12 hours (3 phases)

### ✅ Phase 1: Particle Structure Extension (COMPLETE)
**Time:** 2-3 hours
**Commits:** `724c5bc` - Extend Particle structure to 48 bytes

**Changes:**
- Extended Particle struct from 32 → 48 bytes (16-byte aligned)
- Added `float3 albedo` (12 bytes) - Surface/volume color
- Added `uint materialType` (4 bytes) - Enum index (0-4)
- Backward compatible: First 32 bytes unchanged

**Files Modified:**
- `src/particles/ParticleSystem.h` - Added ParticleMaterialType enum, extended struct
- `src/particles/ParticleSystem.cpp` - Initialize albedo/materialType in UploadParticleData
- `shaders/particles/particle_physics.hlsl` - Extended shader struct, GPU initialization
- `shaders/particles/gaussian_common.hlsl` - Updated shared Particle definition

**Validation:** ✅ Compilation successful, shaders compiled, no errors

---

### ✅ Phase 2: Material Constant Buffer System (COMPLETE)
**Time:** 3-4 hours
**Commits:** `45e46f8` - Implement Material Constant Buffer system

**Changes:**
- Created MaterialTypeProperties struct (64 bytes per material)
- Created MaterialPropertiesConstants (320 bytes for 5 materials)
- Initialized 5 vibrant material presets with spectacular colors
- Created GPU constant buffer and uploaded to GPU
- Added constant buffer declaration to Gaussian raytrace shader

**5 Material Presets Defined:**

| Type | Albedo RGB | Opacity | Emission× | Scattering | Phase G | Description |
|------|------------|---------|-----------|------------|---------|-------------|
| 0: PLASMA | (1.0, 0.4, 0.1) | 0.6 | 2.5× | 1.5 | 0.3 | Hot orange/red, legacy default |
| 1: STAR | (1.0, 0.95, 0.7) | 0.9 | **8.0×** | 0.5 | 0.0 | Brilliant white-yellow, intense glow |
| 2: GAS_CLOUD | (0.4, 0.6, 0.95) | 0.3 | 0.8× | 2.5 | -0.4 | Wispy blue/purple, backward scatter |
| 3: ROCKY_BODY | (0.35, 0.32, 0.3) | 1.0 | 0.05× | 0.3 | 0.2 | Deep grey, minimal emission |
| 4: ICY_BODY | (0.9, 0.95, 1.0) | 0.85 | 0.3× | 3.0 | -0.6 | Bright blue-white, reflective |

**Files Modified:**
- `src/particles/ParticleSystem.h` - Added MaterialTypeProperties, MaterialPropertiesConstants structs, buffer member, accessor
- `src/particles/ParticleSystem.cpp` - InitializeMaterialProperties(), CreateMaterialPropertiesBuffer()
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Added cbuffer MaterialProperties : register(b1)

**GPU Integration:**
- Buffer created: 320 bytes in UPLOAD heap
- Mapped and uploaded material data
- Accessible via `m_particleSystem->GetMaterialPropertiesBuffer()`

**Validation:** ✅ Compilation successful, buffer created, logged initialization

---

## ⏳ Phase 3: Material-Aware Emission + ImGui (PENDING)

**Estimated Time:** 4-6 hours
**Status:** Ready to begin

### Objectives

1. **Shader Integration** (2-3 hours):
   - Read material properties from constant buffer in Gaussian raytrace shader
   - Use particle's `materialType` to index into `g_materials[]`
   - Apply material-specific albedo, emission, opacity, scattering
   - Replace hardcoded plasma values with material lookups

2. **Root Signature Update** (30 minutes):
   - Add material constant buffer to ParticleRenderer_Gaussian root signature
   - Current: 10 parameters (b0, t0-t8, u0, u2)
   - New: 11 parameters (+b1 for material properties)

3. **Gaussian Renderer Integration** (1 hour):
   - Bind material buffer in Render() function
   - `SetComputeRootConstantBufferView()` for b1
   - Get buffer from ParticleSystem

4. **ImGui Controls** (1-2 hours):
   - Add "Material System" collapsible section
   - Material type selector dropdown (PLASMA, STAR, GAS_CLOUD, ROCKY, ICY)
   - Per-material property sliders:
     - Albedo RGB (0.0-1.0)
     - Opacity (0.0-1.0)
     - Emission multiplier (0.0-10.0)
     - Scattering coefficient (0.0-5.0)
     - Phase G (-1.0 to 1.0)
   - Enable/disable material system toggle
   - "Reset to Defaults" button

5. **Validation & Testing** (1 hour):
   - Capture screenshots with F2
   - ML comparison with baseline (LPIPS threshold)
   - Performance check (target: <10% FPS regression)
   - Visual verification of color diversity

---

## Phase 3 Implementation Plan

### Step 1: Update Gaussian Raytrace Shader (60 minutes)

**File:** `shaders/particles/particle_gaussian_raytrace.hlsl`

**Location:** Main ray marching loop (~line 800-900)

**Current Code Pattern:**
```hlsl
// Current: Uses particle.albedo directly (all particles have same default albedo)
float3 particleColor = particle.albedo;  // (1.0, 0.4, 0.1) - hot plasma orange
float opacity = 0.6;  // Hardcoded
float emissionStrength = 2.5;  // Hardcoded
```

**New Code Pattern:**
```hlsl
// New: Lookup material properties from constant buffer
MaterialTypeProperties material = g_materials[particle.materialType];

// Use material-specific properties
float3 particleColor = material.albedo;  // Per-material color
float opacity = material.opacity;        // Per-material opacity
float emissionStrength = material.emissionMultiplier;  // Per-material emission

// Optional: Override particle.albedo with material albedo
// This ensures per-particle albedo is used if set, otherwise material default
float3 finalColor = lerp(material.albedo, particle.albedo, 0.5);  // Blend option
```

**Search Pattern:** Look for emission calculations, albedo usage, opacity calculations in main shader loop

---

### Step 2: Update ParticleRenderer_Gaussian Root Signature (30 minutes)

**File:** `src/particles/ParticleRenderer_Gaussian.cpp`

**Function:** `CreateComputePipeline()` or equivalent initialization

**Current Root Signature:**
```cpp
// b0: GaussianConstants (main constant buffer)
// t0: g_particles
// t1: g_rtLighting
// t2: g_particleBVH
// t4: g_lights
// t5: g_prevShadow
// t6: g_rtxdiOutput
// t7: g_probeGrid
// t8: g_shadowDepth
// u0: g_output
// u2: g_currShadow
```

**New Root Signature:**
```cpp
// Add to root parameters array:
rootParams[N].InitAsConstantBufferView(1);  // b1: MaterialProperties (320 bytes)
```

**Index:** Find next available parameter index (likely 10 or 11)

---

### Step 3: Bind Material Buffer in Render Function (20 minutes)

**File:** `src/particles/ParticleRenderer_Gaussian.cpp`

**Function:** `Render()` or `RenderGaussians()`

**Add Before Dispatch:**
```cpp
// Bind material properties constant buffer (b1)
ID3D12Resource* materialBuffer = m_particleSystem->GetMaterialPropertiesBuffer();
cmdList->SetComputeRootConstantBufferView(
    MATERIAL_BUFFER_ROOT_INDEX,  // Root parameter index for b1
    materialBuffer->GetGPUVirtualAddress()
);
```

**Location:** After binding GaussianConstants (b0), before DispatchRays() or Dispatch()

---

### Step 4: Add ImGui Controls (90-120 minutes)

**File:** `src/core/Application.cpp` (or wherever ImGui UI is rendered)

**Function:** `RenderImGui()` or equivalent

**New UI Section:**
```cpp
if (ImGui::CollapsingHeader("Material System")) {
    ImGui::Text("Sprint 1 MVP - 5 Material Types");
    ImGui::Separator();

    // Material type selector (for testing - future: per-particle assignment)
    static int currentMaterial = 0;
    const char* materialNames[] = {
        "0: PLASMA (Hot orange/red)",
        "1: STAR (Brilliant white-yellow)",
        "2: GAS_CLOUD (Wispy blue/purple)",
        "3: ROCKY_BODY (Deep grey)",
        "4: ICY_BODY (Bright blue-white)"
    };
    if (ImGui::Combo("Preview Material", &currentMaterial, materialNames, 5)) {
        // Optional: Update a test particle to this material type
    }

    ImGui::Separator();
    ImGui::Text("Material Properties (Selected: %s)", materialNames[currentMaterial]);

    // Get current material properties (need accessor in ParticleSystem)
    auto& mat = m_particleSystem->GetMaterialProperties().materials[currentMaterial];

    // Albedo RGB
    if (ImGui::ColorEdit3("Albedo", &mat.albedo.x)) {
        m_particleSystem->UpdateMaterialProperties();  // Upload to GPU
    }

    // Opacity
    if (ImGui::SliderFloat("Opacity", &mat.opacity, 0.0f, 1.0f)) {
        m_particleSystem->UpdateMaterialProperties();
    }

    // Emission multiplier
    if (ImGui::SliderFloat("Emission Multiplier", &mat.emissionMultiplier, 0.0f, 10.0f)) {
        m_particleSystem->UpdateMaterialProperties();
    }

    // Scattering coefficient
    if (ImGui::SliderFloat("Scattering", &mat.scatteringCoefficient, 0.0f, 5.0f)) {
        m_particleSystem->UpdateMaterialProperties();
    }

    // Phase function G
    if (ImGui::SliderFloat("Phase G", &mat.phaseG, -1.0f, 1.0f)) {
        m_particleSystem->UpdateMaterialProperties();
    }

    ImGui::Separator();

    // Reset to defaults
    if (ImGui::Button("Reset All Materials to Defaults")) {
        m_particleSystem->InitializeMaterialProperties();  // Re-initialize defaults
        m_particleSystem->UpdateMaterialProperties();      // Upload to GPU
    }
}
```

**Required ParticleSystem Methods:**
```cpp
// In ParticleSystem.h (public section):
MaterialPropertiesConstants& GetMaterialProperties() { return m_materialProperties; }
void UpdateMaterialProperties();  // Re-upload buffer to GPU

// In ParticleSystem.cpp:
void ParticleSystem::UpdateMaterialProperties() {
    // Re-map and upload material data
    void* mappedData = nullptr;
    m_materialPropertiesBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, &m_materialProperties, sizeof(MaterialPropertiesConstants));
    m_materialPropertiesBuffer->Unmap(0, nullptr);
}
```

---

## Files Requiring Modification (Phase 3)

### High Priority (Core Functionality)
1. ✅ `shaders/particles/particle_gaussian_raytrace.hlsl` - Material lookup in shader
2. ✅ `src/particles/ParticleRenderer_Gaussian.cpp` - Root signature + buffer binding
3. ✅ `src/particles/ParticleRenderer_Gaussian.h` - Root parameter index constant
4. ✅ `src/particles/ParticleSystem.h` - Add UpdateMaterialProperties(), GetMaterialProperties()
5. ✅ `src/particles/ParticleSystem.cpp` - Implement UpdateMaterialProperties()

### Medium Priority (UI Controls)
6. ✅ `src/core/Application.cpp` - ImGui controls for material system

### Low Priority (Optional Enhancements)
7. ⏳ Per-particle material assignment system (future: heterogeneous particles)
8. ⏳ Material presets save/load (JSON serialization)

---

## Testing Plan (Phase 3)

### Checkpoint 3.1: Shader Integration (After Step 1-3)
**Expected:** Particles now use material-specific colors and emission

**Test:**
```bash
# 1. Build project
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build

# 2. Run application
./build/bin/Debug/PlasmaDX-Clean.exe

# 3. Verify:
# - Particles have orange/red color (PLASMA default)
# - No visual regression (LPIPS < 0.02 vs baseline)
# - FPS within 10% of baseline (80-103 FPS expected)

# 4. Capture screenshot
# (Press F2 in application)

# 5. ML comparison
```

**Validation via MCP:**
```python
compare_screenshots_ml(
    before_path="build/bin/Debug/screenshots/screenshot_2025-11-12_03-00-00.bmp",
    after_path="build/bin/Debug/screenshots/screenshot_AFTER_PHASE3.bmp",
    save_heatmap=True
)
# Expected: LPIPS < 0.02 (visually identical - backward compatible)
```

### Checkpoint 3.2: ImGui Controls (After Step 4)
**Expected:** Runtime material adjustment working

**Test:**
1. Open ImGui (F1)
2. Expand "Material System" section
3. Select "1: STAR" material
4. Adjust emission multiplier slider (0 → 10)
5. Observe real-time particle brightness change
6. Press F2 to capture screenshot
7. Verify visually distinct from PLASMA (LPIPS > 0.3)

### Checkpoint 3.3: Performance Validation
**Expected:** < 10% FPS regression

**Test:**
```bash
# Run with performance logging
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/scenarios/close_distance.json

# Check logs:
# Baseline:    80-98 FPS (from metadata)
# Phase 3:     72-103 FPS acceptable (<10% regression)
# If < 72 FPS: Performance issue, investigate
```

---

## Known Challenges & Solutions

### Challenge 1: Root Signature Parameter Limit
**Issue:** DirectX 12 root signature has 64 DWORD limit
**Current Usage:** ~40 DWORDs (estimate)
**Material Buffer:** +320 bytes via CBV (doesn't count against DWORD limit)
**Solution:** ✅ Use ConstantBufferView (CBV), not root constants

### Challenge 2: Shader Hot Reload
**Issue:** DXC doesn't always auto-recompile shaders on change
**Solution:** Manually trigger rebuild or delete `.dxil` files:
```bash
rm build/bin/Debug/shaders/particles/particle_gaussian_raytrace.dxil
MSBuild build/CompileShaders.vcxproj /t:Rebuild
```

### Challenge 3: ImGui Update Frequency
**Issue:** Slider widgets call setters every frame (60× per second)
**Solution:** ✅ Use `if (ImGui::SliderFloat(...))` to detect changes only
```cpp
// WRONG (calls every frame)
ImGui::SliderFloat("Value", &value, 0, 1);
UpdateMaterialProperties();

// CORRECT (calls only on change)
if (ImGui::SliderFloat("Value", &value, 0, 1)) {
    UpdateMaterialProperties();
}
```

---

## Baseline Screenshots (Pre-Sprint 1)

**Location:** `build/bin/Debug/screenshots/`

**Files:**
1. `screenshot_2025-11-12_03-00-00.bmp` - Close-up view (192.4 units)
2. `screenshot_2025-11-12_03-00-21.bmp` - Wide view (811.9 units)
3. `screenshot_2025-11-12_03-00-49.bmp` - Edge view (1761.9 units)

**Metadata:**
- FPS: 80-103 FPS (Medium quality preset, 10K particles)
- Rendering: Gaussian volumetric, MultiLight system (8 lights)
- Physical emission: DISABLED (strength: 0.00)
- Issue: Muted, brown appearance

**Validation Thresholds:**
- Backward compatibility: LPIPS < 0.02 (visually identical to baseline)
- Material distinctiveness: LPIPS > 0.3 (clearly different materials)
- Performance: < 10% FPS regression acceptable

---

## Git Workflow

**Branch:** `feature/gaussian-material-system`
**Created from:** `0.15.5` (or `0.15.8` - verify with `git log --oneline`)

**Commits So Far:**
```
45e46f8 - feat: Implement Material Constant Buffer system (Phase 2)
724c5bc - feat: Extend Particle structure to 48 bytes (Phase 1)
d75bc89 - docs: Add Sprint 1 Material System implementation plan
```

**Merge Strategy:**
- After Phase 3 complete and validated
- Merge to `main` or current development branch
- Tag as `v0.16.0` (Material System MVP)

---

## Next Session Action Items

### Immediate Tasks (Phase 3 Start)

1. **Update Gaussian raytrace shader** (60 min):
   - Find main ray marching loop
   - Add material property lookup: `MaterialTypeProperties mat = g_materials[particle.materialType]`
   - Replace hardcoded values with `mat.albedo`, `mat.emissionMultiplier`, etc.
   - Test compilation

2. **Find ParticleRenderer_Gaussian files** (10 min):
   - Locate `src/particles/ParticleRenderer_Gaussian.cpp`
   - Locate `src/particles/ParticleRenderer_Gaussian.h`
   - Identify root signature creation function
   - Identify render/dispatch function

3. **Update root signature** (30 min):
   - Add parameter for b1 (MaterialProperties CBV)
   - Define root parameter index constant
   - Recompile and verify no errors

4. **Bind material buffer** (20 min):
   - In Render() function, add `SetComputeRootConstantBufferView()`
   - Get buffer from ParticleSystem
   - Bind before dispatch

5. **Rebuild and test** (30 min):
   - Full rebuild
   - Launch application
   - Verify particles render (no crashes)
   - Capture screenshot
   - Check FPS

6. **Add ImGui controls** (90 min):
   - Implement UI section in Application.cpp
   - Add ParticleSystem::UpdateMaterialProperties()
   - Test runtime material adjustment
   - Capture screenshots with different materials

### Success Criteria

✅ **Phase 3 Complete When:**
- Particles use material-specific colors (orange PLASMA visible)
- FPS within 10% of baseline (72-103 FPS acceptable)
- ImGui controls functional (sliders update materials in real-time)
- ML comparison confirms backward compatibility (LPIPS < 0.02)
- Screenshot shows vibrant colors (not muted brown)
- No crashes, no visual artifacts

---

## Contact & Resources

**User:** Ben
**Project:** PlasmaDX-Clean (DirectX 12 volumetric particle renderer)
**Hardware:** RTX 4060 Ti, 1080p
**Target FPS:** 90-120 FPS @ 10K particles

**Key Documentation:**
- `SPRINT_1_MATERIAL_SYSTEM_IMPLEMENTATION.md` - Original implementation plan
- `CLAUDE.md` - Project architecture and guidelines
- `MASTER_ROADMAP_V2.md` - Long-term roadmap
- `RTXDI_QUALITY_ANALYZER_FIXES.md` - MCP tool documentation

**MCP Servers:**
- `gaussian-analyzer` - Material system computational analysis
- `rtxdi-quality-analyzer` - ML screenshot comparison (LPIPS), performance analysis

---

**Last Updated:** 2025-11-12
**Document Version:** 1.0
**Status:** Ready for Phase 3 implementation
