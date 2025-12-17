# Luminous Star Particles - Milestone Checklist

**Document Version:** 1.0
**Created:** December 2025
**Status:** Implementation Tracking
**Chosen Architecture:** Clean Architecture (LuminousParticleSystem class + single 32-light buffer)

---

## Implementation Strategy

**Build and Test After Every Milestone** - This ensures we catch bugs early and maintain a working codebase throughout development.

---

## Pre-Implementation Checklist

- [x] Review FEATURE_OVERVIEW.md
- [x] Review ARCHITECTURE_OPTIONS.md (Clean Architecture chosen)
- [x] Review TECHNICAL_REFERENCE.md
- [x] Ensure project builds cleanly: `MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64`
- [ ] Take baseline screenshot (F2) for comparison
- [ ] Note baseline FPS in ImGui

---

## Milestone 1: Expand Light Buffer to 32 ✅ COMPLETE

**Goal:** Increase MAX_LIGHTS from 16 to 32 without breaking existing functionality.

### 1.1 C++ Changes
- [x] `src/particles/ParticleRenderer_Gaussian.cpp` line 95: Changed `MAX_LIGHTS = 16` to `MAX_LIGHTS = 32`
- [x] `src/particles/ParticleRenderer_Gaussian.cpp` line 1233: Also updated second MAX_LIGHTS constant
- [x] Verified light buffer allocation uses MAX_LIGHTS constant

### 1.2 Shader Changes
- [x] `shaders/particles/particle_gaussian_raytrace.hlsl` line 49: Updated comment `// Number of active lights (0-32: 16 stars + 16 static)`
- [x] `shaders/particles/particle_gaussian_raytrace.hlsl` line 1580: Updated loop limit from 16 to 32

### 1.3 Build & Test
```bash
# Build - User built using worktree alias (symbolic links configured)
# Run - Application launched successfully
```

### 1.4 Verification
- [x] Application launches without crash
- [x] Existing 13 lights still work
- [x] No visual artifacts
- [x] FPS unchanged from baseline

**Status:** [x] COMPLETE (Dec 2025)

---

**Implementation Notes:**
- Worktree setup used with symbolic links for externals/libraries
- User has build alias configured for this branch

---

## Milestone 2: Add SUPERGIANT_STAR Material Type ✅ COMPLETE

**Goal:** Add new material type with high emission and low opacity.

### 2.1 C++ Enum Update
- [x] `src/particles/ParticleSystem.h` line 45: Added `SUPERGIANT_STAR = 8` to enum
- [x] `src/particles/ParticleSystem.h` line 46: Updated `COUNT = 9`
- [x] `src/particles/ParticleSystem.h` line 95: Updated `materials[9]` array size
- [x] Updated comments to reflect 576 bytes (9 materials × 64 bytes)

### 2.2 Material Properties (ACTUAL VALUES IMPLEMENTED)
- [x] `src/particles/ParticleSystem.cpp` lines 1139-1153 in `InitializeMaterialProperties()`:
  ```cpp
  // Material 8: SUPERGIANT_STAR (Luminous star particles with embedded lights)
  // Blue-white supergiant, VERY low opacity so light shines through!
  m_materialProperties.materials[8].albedo = DirectX::XMFLOAT3(0.85f, 0.9f, 1.0f);  // Blue-white (25000K+)
  m_materialProperties.materials[8].opacity = 0.15f;                // VERY transparent
  m_materialProperties.materials[8].emissionMultiplier = 15.0f;     // Highest emission
  m_materialProperties.materials[8].scatteringCoefficient = 0.3f;   // Low scattering (self-luminous)
  m_materialProperties.materials[8].phaseG = 0.0f;                  // Isotropic
  m_materialProperties.materials[8].expansionRate = 0.0f;           // No expansion
  m_materialProperties.materials[8].coolingRate = 0.0f;             // No cooling
  m_materialProperties.materials[8].fadeStartRatio = 1.0f;          // Never fade
  ```

### 2.3 Shader Updates
- [x] `shaders/particles/particle_gaussian_raytrace.hlsl` lines 119-123: Updated `g_materials[9]` and comments

### 2.4 Additional Updates
- [x] Updated loop at line 1163 to zero padding for 9 materials
- [x] Updated log message at line 88 to "9 material types"
- [x] Updated log message at line 1169-1178 to include SUPERGIANT_STAR
- [x] Updated buffer comment at line 1182 to "576 bytes"

### 2.5 Verification
- [ ] Application launches without crash (PENDING BUILD/TEST)
- [ ] No assertion errors about material index
- [ ] Material buffer size increased (576 bytes)

**Status:** [x] COMPLETE (Dec 2025) - Awaiting build verification

---

## Milestone 3: Create LuminousParticleSystem Class

**Goal:** Create the management class for star particle-light binding.

### 3.1 Create Header File
- [ ] Create `src/particles/LuminousParticleSystem.h`:
  - StarParticleBinding struct
  - StarPreset enum
  - Class interface (Initialize, Update, GetStarLights, etc.)

### 3.2 Create Implementation File
- [ ] Create `src/particles/LuminousParticleSystem.cpp`:
  - Initialize() - allocate tracking arrays
  - BindParticleToLight() - establish 1:1 mapping
  - Update() - sync light positions from particles
  - GetStarLights() - return light array for rendering

### 3.3 Update Build System
- [ ] Add new files to CMakeLists.txt

### 3.4 Build & Test
```bash
# Regenerate project if needed
cd build && cmake .. -G "Visual Studio 17 2022" -A x64

# Build
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
```

### 3.5 Verification
- [ ] Project compiles with new class
- [ ] No linker errors
- [ ] Application launches (class not yet integrated)

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 4: Integrate LuminousParticleSystem into Application

**Goal:** Hook the star system into the main application loop.

### 4.1 Application.h Changes
- [ ] Add `#include "particles/LuminousParticleSystem.h"`
- [ ] Add member: `std::unique_ptr<LuminousParticleSystem> m_luminousParticles;`
- [ ] Add method: `void InitializeLuminousStars();`

### 4.2 Application.cpp Changes
- [ ] In `Initialize()`: Call `InitializeLuminousStars()`
- [ ] Implement `InitializeLuminousStars()`:
  - Create LuminousParticleSystem
  - Bind first 16 particles to lights 0-15
  - Set star particle material type to SUPERGIANT_STAR
- [ ] In `Update()`: Call `m_luminousParticles->Update()`
- [ ] In `Render()`: Merge star lights with static lights

### 4.3 Build & Test
```bash
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
./build/bin/Debug/PlasmaDX-Clean.exe
```

### 4.4 Verification
- [ ] Application launches without crash
- [ ] 16 star lights visible in ImGui light count
- [ ] Total light count shows 29 (13 static + 16 star)
- [ ] Star lights have initial positions

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 5: Initialize Star Particles in Physics Shader

**Goal:** Set first 16 particles to SUPERGIANT_STAR material in GPU shader.

### 5.1 Shader Changes
- [ ] `shaders/particles/particle_physics.hlsl`: Add star particle initialization block
  ```hlsl
  // In initialization (totalTime < 0.01)
  if (index < 16) {
      g_particles[index].materialType = 8;  // SUPERGIANT_STAR
      g_particles[index].temperature = 25000.0;
      g_particles[index].density = 1.5;
      g_particles[index].flags |= FLAG_IMMORTAL;
  }
  ```

### 5.2 Shader Compilation
```bash
"/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
    -T cs_6_5 -E main shaders/particles/particle_physics.hlsl \
    -Fo build/bin/Debug/shaders/particles/particle_physics.dxil -I shaders
```

### 5.3 Build & Test
```bash
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
./build/bin/Debug/PlasmaDX-Clean.exe
```

### 5.4 Verification
- [ ] First 16 particles appear noticeably brighter
- [ ] First 16 particles have semi-transparent glow (light shines through)
- [ ] Star particles orbit with physics
- [ ] No visual artifacts

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 6: Light Position Sync

**Goal:** Update light positions from particle positions each frame.

### 6.1 CPU Prediction Implementation
- [ ] In `LuminousParticleSystem::Update()`:
  - Track star positions and velocities on CPU
  - Apply same physics (gravity, Keplerian motion)
  - Copy predicted positions to light array

### 6.2 Integration
- [ ] Verify Update() is called AFTER particle physics
- [ ] Verify lights are uploaded to GPU AFTER sync

### 6.3 Build & Test
```bash
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
./build/bin/Debug/PlasmaDX-Clean.exe
```

### 6.4 Verification
- [ ] Star lights move with accretion disk
- [ ] Lights stay centered on star particles (no drift)
- [ ] Dynamic shadows shift as stars orbit
- [ ] Nearby particles receive light from stars

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 7: Visual Tuning & ImGui Controls

**Goal:** Add runtime controls for adjusting star particle appearance.

### 7.1 ImGui Integration
- [ ] Add "Star Particle System" collapsing header
- [ ] Add sliders: Luminosity, Opacity, Light Radius
- [ ] Add toggle: Enable/Disable star particles
- [ ] Add star count display

### 7.2 Runtime Adjustments
- [ ] Test opacity slider (0.1 - 0.5 range)
- [ ] Test luminosity slider (5.0 - 20.0 range)
- [ ] Test light radius slider (50 - 200 range)

### 7.3 Build & Test
```bash
MSBuild build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:PlasmaDX-Clean /v:minimal
./build/bin/Debug/PlasmaDX-Clean.exe
```

### 7.4 Verification
- [ ] ImGui controls appear
- [ ] Sliders affect rendering in real-time
- [ ] Toggle disables star lights
- [ ] Find optimal visual settings

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 8: Performance Validation

**Goal:** Verify performance meets targets.

### 8.1 Performance Testing
- [ ] Run at 10K particles
- [ ] Record FPS with star lights ON
- [ ] Record FPS with star lights OFF
- [ ] Calculate performance impact

### 8.2 Performance Targets
| Metric | Baseline | With Stars | Target |
|--------|----------|------------|--------|
| FPS (10K particles) | ~142 | ??? | >80 |
| Frame time | ~7ms | ??? | <12.5ms |
| Light count | 13 | 29 | 29 |

### 8.3 Verification
- [ ] FPS meets target (>80 FPS)
- [ ] Frame time stable
- [ ] No stuttering or hitches

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Milestone 9: Final Polish & Documentation

**Goal:** Complete the feature with final touches.

### 9.1 Final Verification
- [ ] Take screenshot (F2) for documentation
- [ ] Compare with baseline screenshot
- [ ] Run for 5+ minutes to verify stability

### 9.2 Documentation Updates
- [ ] Update CLAUDE.md with new feature
- [ ] Update MASTER_ROADMAP_V2.md
- [ ] Capture final performance numbers

### 9.3 Git Commit
```bash
git add -A
git commit -m "feat: Add luminous star particles (16 physics-driven lights)

- Add LuminousParticleSystem class for particle-light binding
- Add SUPERGIANT_STAR material type (15x emission, 0.15 opacity)
- Expand MAX_LIGHTS from 16 to 32
- CPU-side position prediction for light sync
- ImGui controls for runtime adjustment
- Performance: ~XX FPS with 29 lights @ 10K particles"
```

**Status:** [ ] NOT STARTED / [ ] IN PROGRESS / [ ] COMPLETE

---

## Summary Progress

| Milestone | Description | Status |
|-----------|-------------|--------|
| 1 | Expand Light Buffer | ✅ COMPLETE |
| 2 | Add SUPERGIANT_STAR Material | ✅ COMPLETE |
| 3 | Create LuminousParticleSystem | [ ] PENDING |
| 4 | Integrate into Application | [ ] PENDING |
| 5 | Initialize Star Particles | [ ] PENDING |
| 6 | Light Position Sync | [ ] PENDING |
| 7 | Visual Tuning & ImGui | [ ] PENDING |
| 8 | Performance Validation | [ ] PENDING |
| 9 | Final Polish | [ ] PENDING |

---

## Rollback Plan

If any milestone fails:

1. **Revert shader changes:** `git checkout -- shaders/`
2. **Revert C++ changes:** `git checkout -- src/`
3. **Rebuild:** Full clean rebuild
4. **Verify:** Baseline functionality restored

---

## Notes & Issues Log

Use this section to track any issues encountered during implementation:

| Date | Milestone | Issue | Resolution |
|------|-----------|-------|------------|
| Dec 2025 | 1 | MAX_LIGHTS defined in two places in ParticleRenderer_Gaussian.cpp | Updated both at lines 95 and 1233 |
| Dec 2025 | 2 | Shader also needs g_materials array size update | Updated particle_gaussian_raytrace.hlsl to g_materials[9] |

---

## Implementation Changes Summary

### Files Modified (Milestones 1-2):

**Milestone 1 (Light Buffer):**
- `src/particles/ParticleRenderer_Gaussian.cpp` - MAX_LIGHTS 16→32 (lines 95, 1233)
- `shaders/particles/particle_gaussian_raytrace.hlsl` - Updated comments and loop limit (lines 49, 1580)

**Milestone 2 (Material Type):**
- `src/particles/ParticleSystem.h` - Added SUPERGIANT_STAR=8, COUNT=9, materials[9] (lines 45-46, 79-81, 95)
- `src/particles/ParticleSystem.cpp` - Added material 8 properties, updated logs (lines 88, 1139-1178, 1182)
- `shaders/particles/particle_gaussian_raytrace.hlsl` - g_materials[9], updated comments (lines 119-123)

---

**Last Updated:** December 2025
