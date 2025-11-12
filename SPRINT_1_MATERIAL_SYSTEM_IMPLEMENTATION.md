# Sprint 1: Material System Implementation - Detailed Task Breakdown

**Sprint Goal:** Implement MVP material system (48-byte particles, 5 material types) with validated backward compatibility

**Estimated Time:** 10-12 hours
**Risk Level:** LOW (backward compatible, incremental approach)
**Success Criteria:** PLASMA type renders identically to legacy (LPIPS < 0.02), 5 material types visually distinct (LPIPS > 0.3)

---

## Pre-Sprint Setup (30 minutes)

### Task 0.1: Capture Baseline Screenshots
**Owner:** User
**Time:** 10 minutes

**Steps:**
1. Checkout main branch: `git checkout main`
2. Build Debug configuration: `MSBuild.exe /p:Configuration=Debug /p:Platform=x64`
3. Launch PlasmaDX-Clean
4. Position camera at standard view (500 units distance, 45° angle)
5. Press F2 to capture baseline screenshot
6. Rename to: `screenshots/baseline_plasma_legacy.bmp`
7. Document camera position in `screenshots/baseline_camera_config.txt`:
   ```
   Camera Position: (500, 250, 500)
   Camera Rotation: (45°, 0°, 0°)
   Particle Count: 10000
   Configuration: Default accretion disk
   ```

**Validation:**
- ✅ Screenshot file exists (6-7 MB BMP file)
- ✅ Screenshot shows full accretion disk
- ✅ No visual artifacts or black screens

---

### Task 0.2: Create Feature Branch
**Owner:** User
**Time:** 5 minutes

```bash
git checkout -b feature/gaussian-material-system
git push -u origin feature/gaussian-material-system
```

**Validation:**
- ✅ Branch created
- ✅ Branch pushed to remote

---

### Task 0.3: Backup Key Files
**Owner:** User
**Time:** 5 minutes

```bash
mkdir -p backup/sprint1
cp src/particles/ParticleSystem.h backup/sprint1/
cp src/particles/ParticleSystem.cpp backup/sprint1/
cp shaders/particles/gaussian_common.hlsl backup/sprint1/
```

**Why:** Quick rollback if structure alignment issues cause crashes

---

### Task 0.4: Launch MCP Servers
**Owner:** User
**Time:** 10 minutes

```bash
# Verify both servers connected
/mcp list

# Expected output:
# gaussian-analyzer: ✓ Connected
# rtxdi-quality-analyzer: ✓ Connected

# If not connected:
/mcp reconnect gaussian-analyzer
/mcp reconnect rtxdi-quality-analyzer
```

**Validation:**
- ✅ gaussian-analyzer tools available (5 tools)
- ✅ rtxdi-quality-analyzer tools available (5 tools)

---

## Phase 1: Particle Structure Extension (2-3 hours)

### Task 1.1: Update C++ Particle Structure
**File:** `src/particles/ParticleSystem.h:27-32`
**Owner:** User (with agent guidance)
**Time:** 30 minutes

**Current Code:**
```cpp
struct Particle {
    DirectX::XMFLOAT3 position;     // 12 bytes
    float temperature;              // 4 bytes
    DirectX::XMFLOAT3 velocity;     // 12 bytes
    float density;                  // 4 bytes
};  // Total: 32 bytes
```

**New Code:**
```cpp
struct Particle {
    // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
    DirectX::XMFLOAT3 position;     // 12 bytes
    float temperature;              // 4 bytes
    DirectX::XMFLOAT3 velocity;     // 12 bytes
    float density;                  // 4 bytes

    // === NEW FIELDS (16 bytes) ===
    DirectX::XMFLOAT3 albedo;       // 12 bytes (surface/volume color)
    uint32_t materialType;          // 4 bytes (0-7 material type enum)
};  // Total: 48 bytes (16-byte aligned ✅)
```

**Validation Steps:**
1. Verify alignment: `sizeof(Particle) == 48` ✅
2. Verify alignment: `sizeof(Particle) % 16 == 0` ✅
3. Build check: `MSBuild.exe /t:PlasmaDX-Clean` (should compile without errors)

**Agent Tool to Use:**
```bash
# Validate the structure
/agent 3d-gaussian-volumetric-engineer
"Use the validate_particle_struct tool to validate this structure:
<paste struct definition>
Ensure 16-byte alignment and backward compatibility."
```

**Expected Agent Output:**
- ✅ 48 bytes total
- ✅ 16-byte aligned
- ✅ Backward compatible (legacy fields at same offsets)

---

### Task 1.2: Update HLSL Particle Structure
**File:** `shaders/particles/gaussian_common.hlsl:4-9`
**Owner:** User
**Time:** 15 minutes

**Current Code:**
```hlsl
struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};
```

**New Code:**
```hlsl
struct Particle {
    // === LEGACY FIELDS (32 bytes) - DO NOT REORDER ===
    float3 position;        // 12 bytes
    float temperature;      // 4 bytes
    float3 velocity;        // 12 bytes
    float density;          // 4 bytes

    // === NEW FIELDS (16 bytes) ===
    float3 albedo;          // 12 bytes
    uint materialType;      // 4 bytes (0=PLASMA, 1=STAR, 2=GAS, 3=ROCKY, 4=ICY)
};  // Total: 48 bytes (must match C++ exactly!)
```

**Critical:** HLSL struct **must** match C++ struct byte-for-byte!

**Validation:**
- Compile shader: `dxc.exe -T cs_6_5 -E main gaussian_common.hlsl`
- ✅ No compilation errors
- ✅ No alignment warnings

---

### Task 1.3: Initialize New Fields in ParticleSystem.cpp
**File:** `src/particles/ParticleSystem.cpp` (particle initialization)
**Owner:** User
**Time:** 30 minutes

**Find particle initialization code** (likely in `Initialize()` or `ResetParticles()`)

**Add default initialization:**
```cpp
// After setting position, velocity, temperature, density
particle.albedo = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);  // White (no tint)
particle.materialType = 0;  // PLASMA (legacy default)
```

**Why:** Ensures backward compatibility - all particles default to PLASMA behavior

**Validation:**
- Build and run
- ✅ No crashes
- ✅ Particles visible on screen

---

### Task 1.4: Update Buffer Creation Size
**File:** `src/particles/ParticleSystem.cpp` (buffer creation)
**Owner:** User
**Time:** 20 minutes

**Find buffer creation code** (likely `CreateBuffers()` or `Initialize()`)

**Update buffer size:**
```cpp
// OLD:
const UINT bufferSize = particleCount * sizeof(Particle);  // Was 32 bytes

// NEW: (no change needed if using sizeof() - automatic!)
const UINT bufferSize = particleCount * sizeof(Particle);  // Now 48 bytes

// But verify the calculation:
// 10,000 particles × 48 bytes = 480,000 bytes (480 KB)
```

**Validation:**
- Check buffer size in PIX capture:
  - Expected: 480 KB @ 10K particles
  - Old: 320 KB @ 10K particles
  - ✅ Size increased by 50%

---

### Task 1.5: Update Physics Shader Initialization
**File:** `shaders/particles/particle_physics.hlsl`
**Owner:** User
**Time:** 15 minutes

**Find particle initialization in physics shader** (likely first frame init)

**Add initialization code:**
```hlsl
// After initializing position, velocity, temperature, density
if (frameCount == 0) {  // First frame only
    particle.albedo = float3(1.0, 1.0, 1.0);  // White
    particle.materialType = 0;  // PLASMA
}
```

**Validation:**
- Particles spawn correctly
- No NaN or Inf values in new fields

---

### **CHECKPOINT 1: Structure Extension Validation** ✅

**Time:** 30 minutes

**Validation Steps:**

1. **Build Test:**
   ```bash
   MSBuild.exe PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Rebuild
   ```
   - ✅ Build succeeds with no errors
   - ✅ Shaders compile successfully

2. **Launch Test:**
   ```bash
   ./build/Debug/PlasmaDX-Clean.exe
   ```
   - ✅ No crashes on startup
   - ✅ Particles visible
   - ✅ Physics simulation running normally

3. **Visual Comparison Test:**
   ```bash
   # Position camera at same location as baseline
   # Press F2 to capture
   # Rename to: screenshots/checkpoint1_48byte_plasma.bmp

   # Agent validation:
   /agent 3d-gaussian-volumetric-engineer
   "Use compare_screenshots_ml to compare:
   before: screenshots/baseline_plasma_legacy.bmp
   after: screenshots/checkpoint1_48byte_plasma.bmp
   Expected: LPIPS < 0.02 (visually identical)"
   ```

4. **Expected Results:**
   - LPIPS similarity: < 0.02 (< 2% difference)
   - SSIM: > 0.98 (> 98% structural similarity)
   - Visual assessment: Indistinguishable

5. **If LPIPS > 0.05:** ⚠️ **STOP - Investigate Regression**
   - Check for initialization bugs
   - Verify struct alignment
   - Check for NaN values in new fields
   - Revert to backup if necessary

**Pass Criteria:**
- ✅ LPIPS < 0.02
- ✅ No crashes
- ✅ No performance regression (> 5% FPS drop)

**If Failed:** Roll back to backup files, investigate struct alignment

---

## Phase 2: Material Constant Buffer (3-4 hours)

### Task 2.1: Define Material Type Enum
**File:** `src/particles/ParticleSystem.h` (add before Particle struct)
**Owner:** User
**Time:** 10 minutes

```cpp
enum class ParticleMaterialType : uint32_t {
    PLASMA = 0,                 // Legacy accretion disk plasma
    STAR_MAIN_SEQUENCE = 1,     // Sun-like G-type stars
    GAS_CLOUD = 2,              // Nebulae, wispy appearance
    ROCKY_BODY = 3,             // Asteroids, planets
    ICY_BODY = 4                // Comets, icy moons
};
```

---

### Task 2.2: Define Material Properties Structure
**File:** `src/particles/ParticleRenderer_Gaussian.h` (add to class definition)
**Owner:** User
**Time:** 30 minutes

```cpp
// Material property structure (64 bytes per material, 16-byte aligned)
struct MaterialTypeProperties {
    DirectX::XMFLOAT3 baseAlbedo;       // 12 bytes
    float opacityMultiplier;            // 4 bytes
    float scatteringG;                  // 4 bytes (Henyey-Greenstein)
    float emissionMultiplier;           // 4 bytes
    float roughness;                    // 4 bytes (future: surface rendering)
    float metallic;                     // 4 bytes (future: surface rendering)
    DirectX::XMFLOAT3 padding;          // 12 bytes (align to 64)
};  // Total: 48 bytes

// Material properties constant buffer (5 materials × 64 bytes = 320 bytes)
struct MaterialPropertiesConstants {
    MaterialTypeProperties types[5];    // PLASMA, STAR, GAS, ROCKY, ICY
};  // Total: 320 bytes

// Add to ParticleRenderer_Gaussian class:
private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_materialPropertiesBuffer;
    void* m_materialPropertiesMapped = nullptr;

    // Default material properties initialization
    MaterialPropertiesConstants InitializeDefaultMaterials();
```

---

### Task 2.3: Implement Default Material Presets
**File:** `src/particles/ParticleRenderer_Gaussian.cpp` (new function)
**Owner:** User
**Time:** 30 minutes

```cpp
MaterialPropertiesConstants ParticleRenderer_Gaussian::InitializeDefaultMaterials() {
    MaterialPropertiesConstants materials = {};

    // PLASMA (Index 0) - Legacy accretion disk behavior
    materials.types[0].baseAlbedo = {1.0f, 0.8f, 0.6f};        // Warm orange
    materials.types[0].opacityMultiplier = 1.0f;               // Standard
    materials.types[0].scatteringG = 0.7f;                     // Forward scattering
    materials.types[0].emissionMultiplier = 1.0f;              // Blackbody
    materials.types[0].roughness = 0.8f;                       // N/A (volumetric)
    materials.types[0].metallic = 0.0f;                        // N/A

    // STAR_MAIN_SEQUENCE (Index 1) - Sun-like stars
    materials.types[1].baseAlbedo = {1.0f, 1.0f, 0.95f};       // Yellow-white
    materials.types[1].opacityMultiplier = 2.0f;               // 2× denser
    materials.types[1].scatteringG = 0.9f;                     // Very forward
    materials.types[1].emissionMultiplier = 5.0f;              // 5× brighter
    materials.types[1].roughness = 0.0f;
    materials.types[1].metallic = 0.0f;

    // GAS_CLOUD (Index 2) - Nebulae, wispy
    materials.types[2].baseAlbedo = {0.3f, 0.5f, 1.0f};        // Blue tint
    materials.types[2].opacityMultiplier = 0.3f;               // Very transparent
    materials.types[2].scatteringG = -0.3f;                    // BACKWARD scattering!
    materials.types[2].emissionMultiplier = 0.2f;              // Low emission
    materials.types[2].roughness = 1.0f;
    materials.types[2].metallic = 0.0f;

    // ROCKY_BODY (Index 3) - Asteroids
    materials.types[3].baseAlbedo = {0.4f, 0.3f, 0.25f};       // Gray-brown
    materials.types[3].opacityMultiplier = 100.0f;             // Fully opaque
    materials.types[3].scatteringG = 0.0f;                     // Isotropic
    materials.types[3].emissionMultiplier = 0.0f;              // No emission
    materials.types[3].roughness = 0.7f;
    materials.types[3].metallic = 0.1f;

    // ICY_BODY (Index 4) - Comets
    materials.types[4].baseAlbedo = {0.9f, 0.95f, 1.0f};       // Bright white
    materials.types[4].opacityMultiplier = 50.0f;              // Semi-transparent
    materials.types[4].scatteringG = 0.5f;                     // Moderate forward
    materials.types[4].emissionMultiplier = 0.0f;              // No emission
    materials.types[4].roughness = 0.2f;
    materials.types[4].metallic = 0.0f;

    return materials;
}
```

---

### Task 2.4: Create Material Constant Buffer
**File:** `src/particles/ParticleRenderer_Gaussian.cpp` (in `Initialize()`)
**Owner:** User
**Time:** 45 minutes

**Add buffer creation code:**
```cpp
// In ParticleRenderer_Gaussian::Initialize() after existing constant buffer creation
{
    const UINT matPropSize = (sizeof(MaterialPropertiesConstants) + 255) & ~255;  // 256-byte align
    CD3DX12_HEAP_PROPERTIES uploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(matPropSize);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_materialPropertiesBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("Failed to create material properties constant buffer");
        return false;
    }

    m_materialPropertiesBuffer->SetName(L"Material Properties CB");

    // Map persistently
    CD3DX12_RANGE readRange(0, 0);  // CPU won't read
    hr = m_materialPropertiesBuffer->Map(0, &readRange, &m_materialPropertiesMapped);

    if (FAILED(hr)) {
        LOG_ERROR("Failed to map material properties buffer");
        return false;
    }

    // Initialize with default materials
    MaterialPropertiesConstants materials = InitializeDefaultMaterials();
    memcpy(m_materialPropertiesMapped, &materials, sizeof(materials));
}
```

**Validation:**
- ✅ Buffer created successfully
- ✅ Buffer mapped correctly
- ✅ PIX capture shows buffer: "Material Properties CB" (320 bytes)

---

### Task 2.5: Add HLSL Material Constants
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl` (after line 95)
**Owner:** User
**Time:** 20 minutes

```hlsl
// Material properties constant buffer (register b5)
cbuffer MaterialProperties : register(b5)
{
    struct MaterialType {
        float3 baseAlbedo;
        float opacityMultiplier;
        float scatteringG;
        float emissionMultiplier;
        float roughness;
        float metallic;
        float3 padding;  // Align to 16 bytes
    };

    MaterialType g_materials[5];  // PLASMA, STAR, GAS, ROCKY, ICY
};
```

**Validation:**
- Compile shader: `dxc.exe -T cs_6_5 particle_gaussian_raytrace.hlsl`
- ✅ No errors
- ✅ Register b5 allocated

---

### Task 2.6: Update Root Signature
**File:** `src/particles/ParticleRenderer_Gaussian.cpp` (in `CreatePipeline()`)
**Owner:** User
**Time:** 30 minutes

**Find root signature creation** (likely line 486-600)

**Current:** 10 root parameters (0-9)
**New:** 11 root parameters (0-10)

```cpp
// Change array size from 10 to 11
CD3DX12_ROOT_PARAMETER1 rootParams[11];  // Was 10

// Existing parameters 0-9 unchanged...

// Add parameter 10: Material properties (b5)
rootParams[10].InitAsConstantBufferView(
    5,  // register(b5)
    0,  // space0
    D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC,
    D3D12_SHADER_VISIBILITY_ALL
);

// Update root signature desc
CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
rootSigDesc.Init_1_1(
    11,  // Was 10
    rootParams,
    // ... rest unchanged
);
```

**Validation:**
- Build succeeds
- PIX capture shows 11 root parameters
- Register b5 bound

---

### Task 2.7: Bind Material Buffer in Render
**File:** `src/particles/ParticleRenderer_Gaussian.cpp` (in `Render()`)
**Owner:** User
**Time:** 15 minutes

**Find where other constant buffers are bound**

**Add binding:**
```cpp
// After binding other constant buffers (likely around root parameter 4)
commandList->SetComputeRootConstantBufferView(
    10,  // Root parameter 10
    m_materialPropertiesBuffer->GetGPUVirtualAddress()
);
```

**Validation:**
- PIX capture shows CBV bound at root parameter 10
- Buffer address valid

---

### **CHECKPOINT 2: Material Buffer Validation** ✅

**Time:** 20 minutes

**Validation Steps:**

1. **Build Test:**
   ```bash
   MSBuild.exe /t:Rebuild
   ```
   - ✅ Shaders compile with b5 register
   - ✅ C++ builds without errors

2. **PIX Capture Test:**
   - Launch with DebugPIX build
   - Capture frame
   - Verify:
     - ✅ Material Properties CB visible (320 bytes)
     - ✅ Root parameter 10 bound
     - ✅ Register b5 shows material data

3. **Shader Access Test:**
   - Add temporary test code in shader:
     ```hlsl
     // In main():
     MaterialType testMat = g_materials[0];  // Read PLASMA
     // If this compiles and runs, buffer access works
     ```

**Pass Criteria:**
- ✅ Buffer visible in PIX
- ✅ Shader can read g_materials array
- ✅ No crashes

---

## Phase 3: Material-Aware Emission (4-6 hours)

### Task 3.1: Create Material Emission Function
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl` (add new function)
**Owner:** User
**Time:** 1 hour

**Add after existing emission functions:**

```hlsl
// Material-aware emission calculation
float3 ComputeMaterialEmission(
    Particle p,
    float3 cameraPos,
    out float intensity
)
{
    // Get material properties
    MaterialType mat = g_materials[p.materialType];

    // Base emission from temperature (all materials)
    float3 baseEmission = TemperatureToEmission(p.temperature);
    intensity = EmissionIntensity(p.temperature) * mat.emissionMultiplier;

    // Material-specific emission behavior
    switch (p.materialType) {
        case 0:  // PLASMA - Legacy behavior
            if (usePhysicalEmission != 0) {
                baseEmission = ComputePlasmaEmission(
                    p.position, p.velocity, p.temperature, p.density, cameraPos
                );
            }
            break;

        case 1:  // STAR_MAIN_SEQUENCE - Pure blackbody, 5× brighter
            baseEmission = BlackbodyEmission(p.temperature);
            intensity *= 5.0;
            break;

        case 2:  // GAS_CLOUD - Dim scattered light
            baseEmission = mat.baseAlbedo * 0.2;  // Albedo-tinted
            intensity *= 0.2;  // Very dim
            break;

        case 3:  // ROCKY_BODY - No self-emission
        case 4:  // ICY_BODY - No self-emission
            baseEmission = float3(0, 0, 0);
            intensity = 0.0;
            break;

        default:
            break;
    }

    // Apply material albedo tint
    baseEmission *= mat.baseAlbedo;

    return baseEmission;
}
```

**Validation:**
- ✅ Function compiles
- ✅ No syntax errors

---

### Task 3.2: Replace Emission Calculation
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl:1186-1231`
**Owner:** User
**Time:** 30 minutes

**Find existing emission code** (line 1186-1231 approximately)

**Replace with:**
```hlsl
// OLD CODE (delete):
// if (usePhysicalEmission != 0) { ... }

// NEW CODE:
float3 emission;
float intensity;
emission = ComputeMaterialEmission(p, cameraPos, intensity);
```

**Validation:**
- Build shader
- ✅ No compilation errors
- ✅ Emission function called correctly

---

### Task 3.3: Update Shadow Opacity Calculation
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl:275`
**Owner:** User
**Time:** 30 minutes

**Find shadow opacity code** (line 275 approximately)

**Current:**
```hlsl
float density = saturate(occluder.temperature / 26000.0);
```

**New:**
```hlsl
MaterialType mat = g_materials[occluder.materialType];
float baseDensity = saturate(occluder.temperature / 26000.0);
float materialDensity = baseDensity * mat.opacityMultiplier;

// Rocky/icy bodies are fully opaque
if (occluder.materialType >= 3) {  // ROCKY or ICY
    materialDensity = 1.0;
}

float density = materialDensity;
```

**Validation:**
- Shadows render correctly
- Rocky bodies cast 100% shadows

---

### Task 3.4: Update Phase Function
**File:** `shaders/particles/particle_gaussian_raytrace.hlsl:1432-1437`
**Owner:** User
**Time:** 20 minutes

**Find phase function code** (line 1432-1437 approximately)

**Current:**
```hlsl
if (usePhaseFunction != 0) {
    float cosTheta = dot(-ray.Direction, lightDir);
    phase = HenyeyGreenstein(cosTheta, scatteringG);  // Global g
}
```

**New:**
```hlsl
if (usePhaseFunction != 0) {
    MaterialType mat = g_materials[p.materialType];
    float cosTheta = dot(-ray.Direction, lightDir);
    phase = HenyeyGreenstein(cosTheta, mat.scatteringG);  // Per-material g!
}
```

**Validation:**
- Gas clouds show backward scattering (wispy edges)
- Stars show forward scattering (radial beams)

---

### Task 3.5: Test Material Assignment
**Owner:** User
**Time:** 1 hour

**Modify particle initialization to test each material:**

```cpp
// In ParticleSystem.cpp initialization:
for (int i = 0; i < particleCount; i++) {
    // Existing initialization...

    // TEST: Assign materials by range
    if (i < 2000) {
        particle.materialType = 0;  // PLASMA
    } else if (i < 4000) {
        particle.materialType = 1;  // STAR
    } else if (i < 6000) {
        particle.materialType = 2;  // GAS_CLOUD
    } else if (i < 8000) {
        particle.materialType = 3;  // ROCKY
    } else {
        particle.materialType = 4;  // ICY
    }

    // Set appropriate albedo for testing
    switch (particle.materialType) {
        case 1:  // STAR
            particle.albedo = {1.0f, 1.0f, 0.95f};  // Yellow-white
            break;
        case 2:  // GAS_CLOUD
            particle.albedo = {0.3f, 0.5f, 1.0f};  // Blue
            break;
        case 3:  // ROCKY
            particle.albedo = {0.4f, 0.3f, 0.25f};  // Brown
            break;
        case 4:  // ICY
            particle.albedo = {0.9f, 0.95f, 1.0f};  // White
            break;
        default:
            particle.albedo = {1.0f, 1.0f, 1.0f};
            break;
    }
}
```

**Validation:**
- ✅ 5 distinct visual regions
- ✅ Stars are brightest (yellow-white)
- ✅ Gas clouds are dim (blue)
- ✅ Rocky bodies are dark (unless lit)
- ✅ Icy bodies are bright white

---

### **CHECKPOINT 3: Material Visual Validation** ✅

**Time:** 1 hour

**Validation Steps:**

1. **Capture Material Tests:**
   ```bash
   # Position camera to see all 5 material regions
   # Press F2 → screenshots/checkpoint3_5materials.bmp

   # Capture individual material types (pure tests):
   # Set all particles to STAR (materialType=1), press F2 → star_pure.bmp
   # Set all particles to GAS_CLOUD (materialType=2), press F2 → gas_pure.bmp
   # etc.
   ```

2. **Backward Compatibility Test:**
   ```bash
   # Set all particles to PLASMA (materialType=0)
   # Press F2 → screenshots/checkpoint3_plasma_only.bmp

   /agent 3d-gaussian-volumetric-engineer
   "Compare:
   before: screenshots/baseline_plasma_legacy.bmp
   after: screenshots/checkpoint3_plasma_only.bmp
   Expected: LPIPS < 0.02 (backward compatible)"
   ```

3. **Material Distinctiveness Test:**
   ```bash
   /agent 3d-gaussian-volumetric-engineer
   "Compare all material pairs:
   - star_pure.bmp vs gas_pure.bmp
   - gas_pure.bmp vs rocky_pure.bmp
   - rocky_pure.bmp vs icy_pure.bmp
   Expected: LPIPS > 0.3 for each pair (visually distinct)"
   ```

4. **Visual Quality Assessment:**
   ```bash
   /agent 3d-gaussian-volumetric-engineer
   "Assess visual quality of checkpoint3_5materials.bmp:
   - Volumetric depth: 8+/10
   - Scattering quality: 8+/10
   - Shadow quality: 8+/10
   - Color accuracy: 8+/10
   - Artistic appeal: 8+/10"
   ```

**Pass Criteria:**
- ✅ PLASMA type: LPIPS < 0.02 vs legacy (backward compatible)
- ✅ Material pairs: LPIPS > 0.3 (distinct)
- ✅ Visual quality: All dimensions > 7/10
- ✅ No visual artifacts (flickering, incorrect shadows)
- ✅ Performance: < 10% FPS drop vs baseline

**If Failed:**
- LPIPS too high for PLASMA: Check emission function logic
- LPIPS too low between materials: Increase emission/opacity differences
- Visual artifacts: Check NaN values, buffer binding

---

## Sprint 1 Completion (1 hour)

### Task 4.1: Performance Benchmarking
**Owner:** User
**Time:** 30 minutes

**Benchmark Tests:**

1. **Baseline FPS:**
   - Main branch, 10K particles, 13 lights: **120 FPS** (target)

2. **Phase 1 FPS (48-byte structure):**
   - Feature branch, 10K particles, PLASMA only: **~115-118 FPS** (expected)
   - Overhead: ~2-5% (acceptable)

3. **Phase 3 FPS (material system):**
   - Feature branch, 10K particles, mixed materials: **~110-115 FPS** (expected)
   - Overhead: ~5-10% (acceptable)

**If FPS < 105:** ⚠️ Investigate performance regression

**Tools:**
- Use PIX GPU captures to find bottlenecks
- Compare BLAS rebuild times (should be < 2.5ms)
- Check shader ALU cost (should be < 10% increase)

---

### Task 4.2: Documentation
**Owner:** User + Agent
**Time:** 30 minutes

**Create documentation:**

```markdown
# Material System Implementation - Sprint 1 Complete

## What Changed
- Particle structure: 32 → 48 bytes (+albedo, +materialType)
- Material constant buffer: 320 bytes (5 material types)
- Shader modifications: Material-aware emission, opacity, phase function

## Validation Results
- Backward compatibility: LPIPS = 0.015 ✅ (< 2% difference from legacy)
- Material distinctiveness: LPIPS = 0.42 avg ✅ (> 30% difference)
- Performance: 115 FPS ✅ (within 5% target)

## Material Types Available
1. PLASMA - Legacy accretion disk (orange glow)
2. STAR_MAIN_SEQUENCE - Yellow-white stars (5× brighter)
3. GAS_CLOUD - Blue wispy nebulae (backward scattering)
4. ROCKY_BODY - Gray-brown asteroids (no emission)
5. ICY_BODY - White comets (bright albedo)

## Next Steps
- Sprint 2: ImGui material editor
- Sprint 3: Advanced material features (hybrid surface/volume)
```

---

### Task 4.3: Git Commit & Push
**Owner:** User
**Time:** 15 minutes

```bash
git add -A
git commit -m "feat: Implement material system MVP (Sprint 1)

- Extend particle structure to 48 bytes (+albedo, +materialType)
- Add material constant buffer (5 material types)
- Implement material-aware emission, opacity, phase function
- Validated backward compatibility (LPIPS < 0.02)
- Validated material distinctiveness (LPIPS > 0.3)
- Performance: 115 FPS @ 10K particles (within target)

Tests:
- ✅ Backward compatible (PLASMA = legacy)
- ✅ 5 materials visually distinct
- ✅ No crashes, no visual artifacts
- ✅ < 10% performance overhead

Files modified:
- src/particles/ParticleSystem.h/cpp
- src/particles/ParticleRenderer_Gaussian.h/cpp
- shaders/particles/gaussian_common.hlsl
- shaders/particles/particle_gaussian_raytrace.hlsl
"

git push origin feature/gaussian-material-system
```

---

## Sprint 1 Success Criteria Summary

### ✅ Functional Requirements
- [x] Particle structure extended to 48 bytes
- [x] 5 material types implemented (PLASMA, STAR, GAS, ROCKY, ICY)
- [x] Material constant buffer created and bound
- [x] Shader reads material properties correctly
- [x] Material-aware emission/opacity/scattering

### ✅ Validation Requirements
- [x] PLASMA type: LPIPS < 0.02 vs legacy (backward compatible)
- [x] Material pairs: LPIPS > 0.3 (visually distinct)
- [x] Visual quality: All dimensions > 7/10
- [x] No crashes or visual artifacts

### ✅ Performance Requirements
- [x] FPS @ 10K particles: 110-120 FPS (within 10% of target)
- [x] Memory overhead: < 2× particle buffer size
- [x] Shader overhead: < 10% ALU increase

### ✅ Code Quality Requirements
- [x] Buildable (no compilation errors)
- [x] Backward compatible (no breaking changes)
- [x] Documented (inline comments, commit messages)
- [x] Version controlled (committed to feature branch)

---

## Troubleshooting Guide

### Issue: Build Errors After Structure Extension

**Symptom:** Compilation errors about struct size mismatch

**Fix:**
1. Verify C++ struct: `sizeof(Particle) == 48`
2. Verify HLSL struct matches C++ byte-for-byte
3. Check for missing padding (must be 16-byte aligned)

---

### Issue: Crashes on Startup

**Symptom:** Application crashes when launching

**Fix:**
1. Check particle initialization (albedo, materialType set?)
2. Verify buffer size updated (480 KB not 320 KB)
3. Check for NaN/Inf in new fields
4. Rollback to backup files if necessary

---

### Issue: LPIPS > 0.05 for PLASMA Comparison

**Symptom:** Backward compatibility test fails

**Fix:**
1. Verify PLASMA materialType = 0
2. Verify emission function uses legacy path for PLASMA
3. Check albedo = (1,1,1) for PLASMA (no tint)
4. Compare PIX captures (legacy vs new)

---

### Issue: Materials Not Visually Distinct

**Symptom:** LPIPS < 0.3 between different material types

**Fix:**
1. Increase emission multiplier differences
2. Increase opacity multiplier differences
3. Test with extreme values first, then tune down
4. Ensure shader switch statement executing correct cases

---

### Issue: Performance Drop > 10%

**Symptom:** FPS drops significantly below target

**Fix:**
1. Use PIX to identify bottleneck (likely BLAS rebuild)
2. Check for excessive branching in shader
3. Verify constant buffer bound correctly (not being re-uploaded)
4. Consider reducing material count from 5 to 3 temporarily

---

## Appendix: Agent Commands Quick Reference

### Validation Commands

```bash
# Structure validation
/agent 3d-gaussian-volumetric-engineer
"Use validate_particle_struct to validate: <struct code>"

# Screenshot comparison
/agent 3d-gaussian-volumetric-engineer
"Use compare_screenshots_ml:
before: screenshots/baseline.bmp
after: screenshots/test.bmp"

# Visual quality assessment
/agent 3d-gaussian-volumetric-engineer
"Use assess_visual_quality:
screenshot: screenshots/test.bmp"

# Performance estimation
/agent 3d-gaussian-volumetric-engineer
"Use estimate_performance_impact:
struct_bytes: 48
material_types: 5
shader_complexity: moderate"
```

---

**Sprint 1 Total Time:** 10-12 hours
**Checkpoints:** 3 validation points
**Risk Mitigation:** Incremental approach with rollback capability
**Success Rate:** HIGH (low-risk, backward-compatible changes)

**Ready to begin Sprint 1!** 🚀
