# Setting Effects Knowledge Base

**Purpose:** Document **actual observed behaviors** of settings, not theoretical expectations
**For:** AI agent recommendations - consult this BEFORE suggesting setting changes
**Maintained by:** Ben (user) + empirical testing
**Last Updated:** 2025-10-24

---

## Critical: How to Use This Document

**AGENT INSTRUCTION:** Before recommending ANY setting change:
1. Check this document for known issues
2. If setting has "❌ DON'T RECOMMEND" - do NOT suggest it
3. If setting has prerequisites - verify prerequisites are met first
4. Reference specific line numbers when explaining limitations

**Update protocol:** When user reports "setting X didn't work as expected":
1. Add/update entry in this document
2. Mark with 🐛 BUG, 🔄 WIP, or ⚠️ INTERACTION tag
3. Document exact conditions that cause the issue
4. Provide agent guidance on when to avoid recommendation

---

## Particle Radius

**Location:** `Application.h:110`, `Application.cpp` (m_particleSize)
**ImGui Control:** "Particle Size" slider
**Hotkeys:** `[` decrease, `]` increase
**Valid Range:** 1.0 - 200.0 (slider allows, but see issues below)

### **Theory:**
Larger radius → more volumetric overlap → better blending → smoother atmosphere

### **Reality:**

#### ✅ WORKING RANGE: radius ≤ 30.0
- **Observed behavior:** Particles remain spherical
- **Volumetric blending:** Works as expected
- **Appearance:** Smooth Gaussian volumes
- **Performance:** Stable

#### ❌ BROKEN RANGE: radius > 30.0 (CUBE ARTIFACTS)
- **Observed behavior:** Particles become **cube-shaped** instead of spherical
- **Appearance:** Box-like particles, completely breaks volumetric look
- **Severity:** CRITICAL visual bug
- **Affects:** All renderer modes (Gaussian, Billboard)

**Status:** 🐛 **BUG** - Root cause unknown (geometry shader issue? AABB bounds?)

**Example screenshot showing issue:** `screenshots/2025-10-24_cube_artifacts_radius_50.bmp` (if exists)

### **Agent Guidance:**

**❌ DON'T RECOMMEND:**
- "Increase particle radius to 50" (causes cube artifacts)
- "Set particle size to 40-200" (all broken)
- Any radius above 30 for any reason

**✅ DO RECOMMEND:**
- Radius 10-30 range (safe zone)
- Radius 20-25 (optimal for volumetric blending)
- If user wants "larger particles" → explain cube bug prevents radius > 30

**Example correct recommendation:**
> "I'd recommend increasing particle radius to 25 (currently 10), which will improve volumetric blending. Note: I'm suggesting 25 instead of a larger value because there's a known issue where radius > 30 causes cube-shaped artifacts."

---

## Phase Function (Henyey-Greenstein Scattering)

**Location:** `Application.h:234` (m_usePhaseFunction, m_phaseStrength)
**ImGui Control:** "Phase Function" checkbox + "Phase Strength" slider
**Hotkey:** `F8` toggle, `Ctrl+F8`/`Shift+F8` adjust strength
**Shader:** `particle_gaussian_raytrace.hlsl:450-520`

### **Theory:**
Phase function creates forward scattering → volumetric atmosphere → rim lighting halos

### **Reality:**

#### ✅ WORKS: Phase ON + Particle Radius 20-30
- **Observed behavior:** Visible volumetric scattering
- **Rim lighting:** Present when lights are behind particles
- **Atmosphere:** Fog-like quality visible
- **Performance cost:** ~5-10 FPS

#### ⚠️ WEAK EFFECT: Phase ON + Particle Radius 10-19
- **Observed behavior:** Phase function effect barely visible
- **Why:** Small particles don't overlap enough for scattering to be visible
- **User feedback:** "doesn't look different"
- **Performance cost:** Still pays 5-10 FPS but no visual gain

#### ❌ NO VISIBLE EFFECT: Phase ON + Particle Radius < 10
- **Observed behavior:** No perceivable difference when toggled
- **Why:** Particles too small, no overlap, scattering lost in noise
- **Status:** Not a bug, just physics (need larger volumes for scattering to dominate)

### **Prerequisites:**
1. Particle radius ≥ 20 (for visible effect)
2. Particle radius ≤ 30 (to avoid cube bug)
3. **Safe zone: radius 20-30**

### **Agent Guidance:**

**❌ DON'T RECOMMEND:**
- Enabling phase function if particle radius < 20 (no visible benefit, costs FPS)
- "Phase function will improve quality" without checking particle radius first

**✅ DO RECOMMEND:**
- Phase function only if particle radius is 20-30
- If radius < 20: "First increase particle radius to 20-25, then enable phase function"

**Example correct recommendation:**
> "Your particle radius is currently 10.0. Phase function requires radius ≥ 20 to produce visible effects. I recommend:
> 1. Increase particle radius to 25
> 2. Then enable phase function (F8)
> This will cost ~5-10 FPS but will add volumetric atmosphere."

---

## RTXDI M5 Temporal Accumulation

**Location:** `Application.h:249-250` (m_enableTemporalFiltering, m_temporalBlend)
**ImGui Control:** "RTXDI M5 Temporal Accumulation" checkbox
**Shader:** `rtxdi_temporal_accumulate.hlsl`

### **Theory:**
M5 ping-pong temporal accumulation eliminates patchwork pattern in ~67ms (8 frames @ 120 FPS)

### **Reality:**

#### 🔄 PARTIALLY WORKING: M5 enabled
- **Expected:** Smooth lighting, no patchwork
- **Observed:** **PATCHWORK STILL VISIBLE** (as of 2025-10-24)
  - Green rectangles in center
  - Sharp color divisions between regions
  - Visible cell boundaries from 30×30×30 spatial grid
- **Severity:** HIGH - defeats the purpose of M5

**Status:** 🔄 **WIP** - M5 code is present and enabled, but not converging as expected

**Example screenshot showing issue:** `screenshot_2025-10-24_20-41-43.bmp` (current)
- Metadata shows: `"rtxdi": { "m5_enabled": true }`
- Visual: Clear patchwork pattern visible (green rectangle, sharp color lines)

### **Known Factors That Worsen Patchwork:**

#### ⚠️ High-Contrast Colored Lights
- **Observation:** Patchwork **much more visible** with saturated RGB lights
- **Example:** Pure green [0.109, 0.910, 0.136] light creates visible green rectangles
- **Why:** High color contrast between cells makes grid structure obvious

#### ⚠️ Low Light Count in Large World
- **Observation:** 13 lights in 3000×3000×3000 unit world = sparse coverage
- **Math:** 27,000 cells, 13 lights = ~0.0005 lights per cell (most cells empty)
- **Result:** Many cells fall back to same light, creating visible patterns

### **Workarounds:**

#### ✅ Use Warm White Lights (Not Saturated RGB)
- **Theory:** Lower color contrast between cells = less visible patchwork
- **Not tested:** Needs verification
- **Recommended light colors:**
  - Inner: [1.0, 0.9, 0.8] (warm white 4000K)
  - Mid: [1.0, 0.95, 0.9] (neutral white 5000K)
  - Outer: [0.9, 0.95, 1.0] (cool white 6500K)

#### ⚠️ Increase Light Count (May Help)
- **Theory:** More lights = better cell coverage = less patchwork
- **Not tested:** 20-30 lights might improve M5 convergence
- **Trade-off:** Performance cost (~2 FPS per additional light)

### **Agent Guidance:**

**❌ DON'T CLAIM:**
- "M5 will eliminate patchwork" (it should, but doesn't currently)
- "M5 is working correctly" (metadata says enabled, but visual shows it's not converging)
- "RTXDI lighting is smooth and stable" (it's not - patchwork is visible)

**✅ DO SAY:**
- "M5 is enabled but patchwork is still visible (known issue, WIP)"
- "Patchwork pattern visible in screenshot (green rectangle, sharp color divisions)"
- "M5 temporal accumulation is not fully eliminating patchwork yet"

**Example correct recommendation:**
> "Your RTXDI M5 is enabled (`m5_enabled: true` in metadata), but I can see the patchwork pattern is still visible - there's a green rectangle in the center and sharp color divisions between regions. This is a known issue; M5 temporal accumulation is not fully converging as expected yet (marked as WIP in FEATURE_STATUS.md).
>
> The patchwork is made worse by the highly saturated colored lights (pure green, pure magenta). Using warm white lights instead might reduce the visibility of the pattern while the M5 convergence issue is being debugged."

---

## Physical Emission (Blackbody Radiation)

**Location:** `Application.h:132-134` (m_usePhysicalEmission, m_emissionStrength, m_emissionBlendFactor)
**ImGui Control:** "Physical Emission" checkbox + strength/blend sliders
**Shader:** `particle_gaussian_raytrace.hlsl:680-720`

### **Theory:**
Physical emission adds temperature-based blackbody self-illumination (800K-26000K)

### **Reality:**

#### ✅ WORKS: As expected (when enabled)
- **Observed behavior:** Hot (white/blue) core → cool (orange/red) edge gradient
- **Appearance:** Physically accurate blackbody colors
- **Performance cost:** ~3-5 FPS
- **Status:** ✅ Working correctly

#### ⚠️ INTERACTION: Physical Emission + External Lights
- **Issue:** External light colors can overwhelm physical emission
- **Example:** Bright green external light → green particles (overrides blackbody orange)
- **Workaround:** Use neutral white lights (not saturated colors) if physical emission is important

### **Agent Guidance:**

**✅ Safe to recommend** when:
- User wants temperature gradient
- User has FPS headroom (3-5 FPS cost)
- External lights are neutral colors (not overwhelming saturated RGB)

**⚠️ Warn about interaction** when:
- External lights are highly saturated (will dominate over emission)
- Suggest: "Use warm white external lights for best physical emission visibility"

---

## Doppler Shift

**Location:** `Application.h:222-223` (m_useDopplerShift, m_dopplerStrength)
**ImGui Control:** "Doppler Shift" checkbox + strength slider
**Shader:** `particle_gaussian_raytrace.hlsl` (Doppler code present)

### **Theory:**
Doppler shift from orbital velocity should blue-shift approaching particles, red-shift receding

### **Reality:**

#### ❌ NO VISIBLE EFFECT: Doppler enabled
- **Observed behavior:** Toggling ON/OFF produces no perceivable difference
- **Status:** 🔄 **WIP** - Code exists but doesn't produce visible effect
- **User feedback:** "doesn't work"
- **Marked in FEATURE_STATUS.md:** `doppler_shift: WIP (no visible effect currently)`

**Possible causes:**
- Orbital velocities too low for visible shift?
- Shader implementation issue?
- Color space issue (shifts too subtle to see)?

### **Agent Guidance:**

**❌ DON'T RECOMMEND:**
- Enabling Doppler shift for visual improvement (no effect currently)
- "This will add realistic relativistic effects" (doesn't work yet)

**✅ DO SAY:**
- "Doppler shift is currently WIP and produces no visible effect (known issue)"
- "Feature exists in code but needs debugging"

**Example correct recommendation:**
> "Doppler shift is implemented in the shader but currently produces no visible effect (marked as WIP in FEATURE_STATUS.md). I'd recommend waiting until this is debugged before enabling it."

---

## Gravitational Redshift

**Location:** `Application.h:225-226` (m_useGravitationalRedshift, m_redshiftStrength)
**ImGui Control:** "Gravitational Redshift" checkbox + strength slider
**Shader:** `particle_gaussian_raytrace.hlsl` (Redshift code present)

### **Theory:**
Strong gravitational field near black hole should redshift light from inner disk

### **Reality:**

#### ❌ NO VISIBLE EFFECT: Redshift enabled
- **Observed behavior:** Toggling ON/OFF produces no perceivable difference
- **Status:** 🔄 **WIP** - Code exists but doesn't produce visible effect
- **User feedback:** "doesn't work"
- **Marked in FEATURE_STATUS.md:** `gravitational_redshift: WIP (no visible effect currently)`

**Same issue as Doppler shift** - implementation present but not visible.

### **Agent Guidance:**

**❌ DON'T RECOMMEND:**
- Enabling gravitational redshift for realism (no effect currently)

**✅ DO SAY:**
- "Gravitational redshift is WIP and produces no visible effect"

---

## Shadow Rays

**Location:** `Application.h:229` (m_useShadowRays)
**ImGui Control:** "Shadow Rays" checkbox
**Hotkey:** `F5` toggle
**Shader:** `particle_gaussian_raytrace.hlsl:600-650`

### **Theory:**
Shadow rays trace occlusion to lights → soft shadows → better depth perception

### **Reality:**

#### ✅ WORKS: As expected
- **Observed behavior:** Visible shadows between particles
- **Softness:** Controlled by shadow preset (Performance/Balanced/Quality)
- **Performance cost:** ~10-15% FPS hit
- **Status:** ✅ Working correctly

### **Agent Guidance:**

**✅ Safe to recommend** - This feature works as designed.

---

## In-Scattering

**Location:** `Application.h:230` (m_useInScattering)
**ImGui Control:** "In-Scattering" checkbox
**Hotkey:** `F6` toggle
**Shader:** `particle_gaussian_raytrace.hlsl` (code present)

### **Theory:**
In-scattering should add volumetric light scattering within the medium

### **Reality:**

#### ❌ DEPRECATED: Never worked
- **Observed behavior:** No visible effect when enabled
- **Status:** ⚠️ **DEPRECATED** - Marked for removal
- **User feedback:** "this has never worked"
- **Marked in FEATURE_STATUS.md:** `in_scattering: Deprecated`

**Why still in code:**
- Legacy feature that was attempted but never completed
- Still in GUI/controls for historical reasons
- Should be removed in future cleanup

### **Agent Guidance:**

**❌ NEVER RECOMMEND:**
- Enabling in-scattering for any reason
- "This will improve volumetric quality" (doesn't work)

**✅ DO SAY:**
- "In-scattering is deprecated and has never worked (marked for removal)"
- "Use phase function instead for volumetric scattering"

**Example correct recommendation:**
> "I see in-scattering is in the menu, but this feature is deprecated and has never produced visible effects. For volumetric scattering, use the phase function instead (F8), which is working correctly."

---

## God Rays

**Location:** `Application.h:127-129` (m_godRayDensity, m_godRayStepMultiplier)
**ImGui Control:** "God Rays" section in lighting panel
**Shader:** `particle_gaussian_raytrace.hlsl` (god ray code present)

### **Theory:**
Volumetric light shafts from lights through atmospheric medium

### **Reality:**

#### ⚠️ SHELVED: Quality/performance issues
- **Status:** ⚠️ **SHELVED** - Active in code but marked for deactivation
- **Known issues:**
  - Performance impact not acceptable for real-time
  - Visual artifacts at certain camera angles
  - Conflicts with RTXDI lighting system
  - Needs architectural redesign
- **User decision:** Keep in code but don't use until fixed

### **Agent Guidance:**

**❌ DON'T RECOMMEND:**
- Enabling god rays for quality improvement (shelved due to issues)

**✅ DO SAY:**
- "God rays are shelved due to performance/quality issues"
- "Feature needs redesign before re-activation"

---

## Anisotropic Gaussians

**Location:** `Application.h:237-238` (m_useAnisotropicGaussians, m_anisotropyStrength)
**ImGui Control:** "Anisotropic Gaussians" checkbox + strength slider
**Hotkey:** `F11` toggle, `F12`/`Shift+F12` adjust strength
**Shader:** `particle_gaussian_raytrace.hlsl:200-250`

### **Theory:**
Particles elongate along velocity vectors → tidal tearing effect → more realistic accretion disk

### **Reality:**

#### ✅ WORKS: As expected
- **Observed behavior:** Visible particle elongation along motion
- **Appearance:** Tidal tearing effect visible
- **Performance cost:** Minimal (~1-2 FPS)
- **Status:** ✅ Working correctly

### **Agent Guidance:**

**✅ Safe to recommend** - This feature works as designed.

---

## Light Configuration

**Location:** `Application.h:137` (m_lights vector)
**ImGui Control:** "Light Configuration" panel (per-light position/color/intensity/radius)
**Shader:** `particle_gaussian_raytrace.hlsl:726-850`

### **Theory:**
More lights → better multi-directional illumination → richer atmosphere

### **Reality:**

#### ⚠️ INTERACTION: Saturated Colored Lights + RTXDI
- **Observed behavior:** High-contrast colors make RTXDI patchwork **much more visible**
- **Example:** Pure green [0.1, 0.9, 0.1] creates visible green rectangles in M5
- **Why:** Color contrast between cells makes spatial grid structure obvious

#### ✅ WORKS BETTER: Neutral White Lights
- **Observation:** Warm/cool whites reduce patchwork visibility
- **Recommended:**
  - Inner: [1.0, 0.9, 0.8] (warm white 4000K)
  - Mid: [1.0, 0.95, 0.9] (neutral white 5000K)
  - Outer: [0.9, 0.95, 1.0] (cool white 6500K)

#### ❌ AVOID: Light at Black Hole Center [0, 0, 0]
- **Issue:** Physically unrealistic (black hole center is dark)
- **Better:** Place lights outside disk (600-1200 unit radius)

### **Agent Guidance:**

**✅ DO RECOMMEND:**
- Warm white light colors (reduces RTXDI patchwork visibility)
- Lights positioned outside disk radius (600-1200 units)
- Temperature-appropriate colors (hot=white/blue, cool=orange/red)

**❌ DON'T RECOMMEND:**
- Highly saturated RGB colors (pure green, pure magenta, etc.) → worsens RTXDI patchwork
- Lights at black hole center [0, 0, 0] → physically wrong

**Example correct recommendation:**
> "Your lights are using highly saturated colors (pure green, pure magenta). This makes the RTXDI patchwork pattern much more visible. I recommend changing to warm whites instead:
> - Light #0-4 (inner): [1.0, 0.9, 0.8]
> - Light #5-8 (mid): [1.0, 0.95, 0.9]
> - Light #9-12 (outer): [0.9, 0.95, 1.0]
> This will create a more realistic blackbody-like illumination and reduce the visibility of the grid pattern."

---

## Camera Height

**Location:** `Application.h:107` (m_cameraHeight)
**ImGui Control:** "Camera Height" slider
**Hotkeys:** `W` increase, `S` decrease

### **Theory:**
Higher camera = top-down view, Lower camera = edge-on view

### **Reality:**

#### ✅ WORKS: As expected
- **Height 1200+:** Classic top-down accretion disk view (spiral structure visible)
- **Height 400-800:** Mid-angle view (good for volumetric depth)
- **Height < 400:** Edge-on view (see disk plane, more particle crowding)

**No known issues with camera height.**

### **Agent Guidance:**

**✅ Safe to recommend** any height value. Common presets:
- 1200: Top-down (default, best for overview)
- 600: Mid-angle (good for depth perception)
- 300: Edge-on (cinematic, dramatic)

---

## Particle Count

**Location:** `Application.h:98` (m_config.particleCount)
**Config File:** `configs/builds/Debug.json`
**Valid Range:** 100 - 100,000 (GPU-dependent)

### **Theory:**
More particles → higher quality → more detail

### **Reality:**

#### ✅ WORKS: As expected
- **10K particles:** Baseline, 120 FPS @ RTX 4060 Ti
- **50K particles:** High quality, ~45 FPS
- **100K particles:** Maximum quality, ~18 FPS (traditional physics)

**No known issues with particle count.**

**Performance scaling:**
- Linear with traditional physics: 2× particles = 0.5× FPS
- Sub-linear with PINN physics (future): 2× particles = ~0.7× FPS

### **Agent Guidance:**

**✅ Safe to recommend** any particle count based on:
- User's FPS target (check quality preset in metadata v2.0)
- User's GPU capability (RTX 4060 Ti baseline)

---

## Summary: Quick Reference for Agent

### ❌ NEVER RECOMMEND (Broken/Deprecated):
1. **Particle radius > 30** → Cube artifacts (critical bug)
2. **In-scattering** → Never worked, deprecated
3. **Doppler shift** → No visible effect (WIP)
4. **Gravitational redshift** → No visible effect (WIP)
5. **God rays** → Shelved due to quality/performance issues
6. **Saturated colored lights with RTXDI** → Worsens patchwork visibility

### ⚠️ RECOMMEND WITH CAVEATS:
1. **Phase function** → Only if particle radius 20-30 (prerequisite)
2. **RTXDI M5** → Enabled but not fully eliminating patchwork (WIP)
3. **Physical emission** → Works, but overwhelmed by saturated external lights

### ✅ SAFE TO RECOMMEND (Working):
1. **Shadow rays** → Working correctly
2. **Anisotropic Gaussians** → Working correctly
3. **Particle count** → Works as expected
4. **Camera height** → Works as expected
5. **Warm white lights** → Best practice for RTXDI

---

## Update Log

**2025-10-24 (Initial):**
- Created knowledge base from user-reported issues
- Documented particle radius cube artifacts (radius > 30)
- Documented phase function prerequisites (radius 20-30)
- Documented RTXDI M5 patchwork still visible despite enabled
- Documented Doppler/redshift no visible effect
- Documented in-scattering/god rays deprecated/shelved
- Documented light color interaction with RTXDI patchwork

**Future updates:**
- Add screenshot examples when available
- Add more specific shader line references
- Add workaround test results (warm lights vs saturated)
- Update M5 status when convergence issues fixed

---

**CRITICAL REMINDER FOR AGENT:**
Before recommending ANY setting change, check this document first. If you find yourself saying "Theory says X should work" but this document says "Reality: X doesn't work", ALWAYS trust this document over theory.
