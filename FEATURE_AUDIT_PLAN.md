# Feature Audit Plan - Making AI Recommendations Accurate

**Date:** 2025-10-24
**Problem:** Phase 1 metadata doesn't capture enough context, leading to incorrect/confusing recommendations

---

## The Problem

**Current issues with Phase 1 metadata:**
1. **RTXDI status confusing** - Says "M5 enabled" but RTXDI lighting system not active (using Multi-Light instead)
2. **FPS targets wrong** - Assumes 120 FPS universal target, but depends on quality preset (30/60/120/165 FPS)
3. **Deprecated features included** - In-scattering in metadata but never worked
4. **Missing critical settings** - Physical effects (emission, Doppler), light colors, physics system state
5. **No feature status** - Can't tell what's working vs WIP vs deprecated

**Result:** Agent gives well-meaning but incorrect advice based on incomplete/wrong information.

---

## Solution: Multi-Layered Audit System

### Layer 1: Feature Status Documentation (Manual)
### Layer 2: Enhanced Metadata Schema (Automated)
### Layer 3: Automated Code Audit (Semi-Automated)
### Layer 4: Runtime Validation Tests (Automated)

---

## Layer 1: FEATURE_STATUS.md Documentation

**Purpose:** Single source of truth for what works, what doesn't, and what's expected

**Format:**
```markdown
# Feature Status Matrix

## Lighting Systems

### Multi-Light System (Phase 3.5)
**Status:** ✅ COMPLETE
**Working:** Yes, fully functional
**Description:** Brute-force multi-light rendering, scales to ~20 lights
**Performance:** Good (<20 lights), poor (>20 lights)
**Controls:**
  - ImGui: Light configuration panel
  - Hotkeys: None
**Metadata fields:** `light_count`, `light_colors[]`, `light_positions[]`
**Tests:** ✅ Verified working with 13 lights @ 120 FPS

### RTXDI Lighting System (Phase 4)
**Status:** 🔄 WIP
**Working:** Partially (M4 works, M5 incomplete)
**Description:** NVIDIA RTXDI weighted reservoir sampling
**Known issues:**
  - M5 temporal accumulation not converging
  - Ping-pong buffer logic unfinished
**Controls:**
  - ImGui: "Enable RTXDI" checkbox
  - Hotkeys: F7 to toggle
**Metadata fields:** `rtxdi_enabled`, `rtxdi_m5_enabled`, `temporal_blend_factor`
**Tests:** ⚠️ M4 works, M5 needs debugging

---

## Physical Effects

### Blackbody Emission
**Status:** ✅ COMPLETE
**Working:** Yes
**Description:** Temperature-based color emission (800K-26000K)
**Controls:**
  - ImGui: "Use Physical Emission" checkbox
  - ImGui: "Emission Strength" slider (0.0-5.0)
  - ImGui: "Emission Blend Factor" slider (0.0-1.0)
**Metadata fields:** `use_physical_emission`, `emission_strength`, `emission_blend_factor`
**Tests:** ✅ Verified working

### Doppler Shift
**Status:** ✅ COMPLETE
**Working:** Yes
**Description:** Relativistic color shift based on particle velocity
**Controls:**
  - ImGui: "Use Doppler Shift" checkbox
  - ImGui: "Doppler Strength" slider (0.0-5.0)
**Metadata fields:** `use_doppler_shift`, `doppler_strength`
**Tests:** ✅ Verified working

### In-Scattering
**Status:** ❌ DEPRECATED / NON-FUNCTIONAL
**Working:** No - never completed
**Description:** Volumetric in-scattering (attempted implementation)
**Known issues:**
  - Shader code incomplete
  - Performance unacceptable when attempted
  - Marked for removal
**Controls:**
  - ImGui: "Use In-Scattering" checkbox (REMOVE THIS)
  - Hotkey: F6 (REMOVE THIS)
**Metadata fields:** `use_in_scattering` (SHOULD BE REMOVED)
**Action:** Remove from GUI and metadata in next cleanup pass

---

## Quality Presets

### Performance Targets
**Status:** ✅ DEFINED
**Quality levels and target FPS:**
  - Maximum Quality: Any FPS (video/screenshots, not realtime)
  - Ultra: 30 FPS target
  - High: 60 FPS target
  - Medium: 120 FPS target
  - Low: 165 FPS target

**Current detection:** Not implemented
**Needed:** Auto-detect quality preset based on settings
**Metadata field needed:** `quality_preset`, `target_fps`
```

**Action:** Create this file systematically for EVERY feature in the codebase.

---

## Layer 2: Enhanced Metadata Schema v2.0

**Expand ScreenshotMetadata struct to include:**

```cpp
struct ScreenshotMetadata {
    // Schema versioning
    std::string schemaVersion = "2.0";

    // === RENDERING CONFIGURATION ===

    // Active lighting system
    enum class LightingSystemType {
        MultiLight,  // Phase 3.5 brute force
        RTXDI       // Phase 4 ReSTIR
    };
    LightingSystemType activeLightingSystem;

    // RTXDI-specific (only if RTXDI active)
    struct RTXDIConfig {
        bool m4Enabled;
        bool m5Enabled;
        bool m5Converged;  // NEW: Track if actually converged
        float temporalBlendFactor;
        int framesUntilConvergence;  // NEW: Estimate based on FPS
    } rtxdi;

    // Light configuration
    struct LightConfig {
        int count;
        std::vector<XMFLOAT3> colors;     // NEW: Actual light colors
        std::vector<XMFLOAT3> positions;  // NEW: Light positions
        std::vector<float> intensities;   // NEW: Light intensities
    } lights;

    // Shadow configuration
    struct ShadowConfig {
        std::string preset;  // "Performance", "Balanced", "Quality", "Custom"
        int raysPerLight;
        bool temporalFilteringEnabled;
        float temporalBlendFactor;
    } shadows;

    // === QUALITY PRESET ===

    enum class QualityPreset {
        Maximum,  // Any FPS - video/screenshots
        Ultra,    // 30 FPS target
        High,     // 60 FPS target
        Medium,   // 120 FPS target
        Low       // 165 FPS target
    };
    QualityPreset qualityPreset;
    float targetFPS;

    // === PHYSICAL EFFECTS ===

    struct PhysicalEffects {
        // Emission
        bool usePhysicalEmission;
        float emissionStrength;
        float emissionBlendFactor;

        // Relativistic effects
        bool useDopplerShift;
        float dopplerStrength;
        bool useGravitationalRedshift;
        float redshiftStrength;

        // Phase function
        bool usePhaseFunction;
        float phaseStrength;

        // Anisotropic Gaussians
        bool useAnisotropicGaussians;
        float anisotropyStrength;
    } physicalEffects;

    // === FEATURE STATUS FLAGS ===

    struct FeatureStatus {
        // Working features
        bool multiLightWorking = true;
        bool shadowRaysWorking = true;
        bool phaseFunctionWorking = true;
        bool physicalEmissionWorking = true;
        bool dopplerShiftWorking = true;

        // WIP features
        bool rtxdiM5Working = false;  // Under development

        // Deprecated/non-functional
        bool inScatteringDeprecated = true;  // Never worked
        bool godRaysDeprecated = true;       // Shelved in Phase 3.7
    } featureStatus;

    // === PARTICLES ===

    struct ParticleConfig {
        int count;
        float radius;
        float gravityStrength;
        bool physicsEnabled;

        // Physics system details
        float innerRadius;
        float outerRadius;
        float diskThickness;
    } particles;

    // === PERFORMANCE ===

    struct Performance {
        float fps;
        float frameTime;
        float targetFPS;
        float fpsRatio;  // current / target (1.0 = on target)

        // Bottleneck hints
        bool likelyGPUBound;
        bool likelyCPUBound;
    } performance;

    // === CAMERA ===

    struct CameraState {
        XMFLOAT3 position;
        XMFLOAT3 lookAt;
        float distance;
        float height;
        float angle;
        float pitch;
    } camera;

    // === ML/QUALITY ===

    struct MLQuality {
        bool pinnEnabled;
        std::string modelPath;
        bool adaptiveQualityEnabled;
    } mlQuality;

    // === METADATA ===

    std::string timestamp;
    std::string configFile;
};
```

**Benefits:**
- Agent knows exactly what lighting system is active
- Agent knows quality target (30/60/120/165 FPS)
- Agent knows what features are deprecated/non-functional
- Agent can see all physical effects and their settings
- Agent can see actual light colors/positions

---

## Layer 3: Automated Code Audit

**Use grep/glob to verify feature implementation status:**

```python
# audit_features.py

import os
import subprocess
from pathlib import Path

class FeatureAuditor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

    def audit_feature(self, feature_name, search_patterns):
        """
        Audit a feature by searching for implementation patterns

        Returns:
        - files_found: List of files implementing this feature
        - controls_found: ImGui controls for this feature
        - hotkeys_found: Keyboard shortcuts
        - shader_code: Related shader files
        - status: COMPLETE/WIP/DEPRECATED/NON_FUNCTIONAL
        """
        results = {
            'name': feature_name,
            'cpp_files': [],
            'shader_files': [],
            'imgui_controls': [],
            'hotkeys': [],
            'status': 'UNKNOWN'
        }

        # Search C++ implementation
        for pattern in search_patterns['cpp']:
            files = self._grep_pattern(pattern, "*.cpp")
            results['cpp_files'].extend(files)

        # Search shader implementation
        for pattern in search_patterns['hlsl']:
            files = self._grep_pattern(pattern, "*.hlsl")
            results['shader_files'].extend(files)

        # Search ImGui controls
        imgui_pattern = f'ImGui.*{feature_name}'
        results['imgui_controls'] = self._grep_pattern(imgui_pattern, "*.cpp")

        # Search hotkeys
        hotkey_pattern = f'OnKeyPress.*{feature_name}'
        results['hotkeys'] = self._grep_pattern(hotkey_pattern, "*.cpp")

        # Determine status
        results['status'] = self._determine_status(results)

        return results

    def _grep_pattern(self, pattern, file_glob):
        """Search for pattern in files matching glob"""
        cmd = f"grep -r '{pattern}' --include='{file_glob}' {self.project_root}"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            return [line.split(':')[0] for line in output.strip().split('\n')]
        except subprocess.CalledProcessError:
            return []

    def _determine_status(self, results):
        """Heuristic to determine feature status"""
        if not results['cpp_files'] and not results['shader_files']:
            return 'NON_FUNCTIONAL'

        if results['imgui_controls'] and results['cpp_files'] and results['shader_files']:
            return 'COMPLETE'

        if results['cpp_files'] and not results['shader_files']:
            return 'WIP'

        # Check for "TODO" or "FIXME" in related files
        # (More sophisticated analysis)

        return 'UNKNOWN'

# Example usage
auditor = FeatureAuditor('/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean')

# Audit in-scattering feature
in_scatter_audit = auditor.audit_feature(
    'InScattering',
    {
        'cpp': ['m_useInScattering', 'InScatter'],
        'hlsl': ['InScatter', 'in_scatter']
    }
)

print(f"In-Scattering Status: {in_scatter_audit['status']}")
print(f"  CPP files: {len(in_scatter_audit['cpp_files'])}")
print(f"  Shader files: {len(in_scatter_audit['shader_files'])}")
print(f"  ImGui controls: {len(in_scatter_audit['imgui_controls'])}")
print(f"  Hotkeys: {len(in_scatter_audit['hotkeys'])}")

# If status is NON_FUNCTIONAL or WIP, flag for review
```

**Run this script to generate:**
- `FEATURE_STATUS.md` - Markdown documentation
- `feature_audit.json` - Machine-readable status

---

## Layer 4: Runtime Validation Tests

**Create automated tests to verify features actually work:**

```cpp
// tests/FeatureValidationTests.cpp

class FeatureValidationTests {
public:
    // Test 1: Verify physical emission changes particle color
    bool TestPhysicalEmission() {
        // Capture frame with emission OFF
        auto frame1 = CaptureFrame();

        // Enable physical emission
        m_usePhysicalEmission = true;

        // Capture frame with emission ON
        auto frame2 = CaptureFrame();

        // Verify frames are different (emission had effect)
        float diff = ComputeFrameDifference(frame1, frame2);
        return diff > 0.1f;  // >10% difference expected
    }

    // Test 2: Verify RTXDI actually changes lighting
    bool TestRTXDI() {
        // Use Multi-Light system
        m_lightingSystem = LightingSystem::MultiLight;
        auto frame1 = CaptureFrame();

        // Switch to RTXDI
        m_lightingSystem = LightingSystem::RTXDI;
        auto frame2 = CaptureFrame();

        // Verify different (RTXDI had effect)
        float diff = ComputeFrameDifference(frame1, frame2);
        return diff > 0.05f;  // >5% difference expected
    }

    // Test 3: Verify in-scattering does nothing (deprecated)
    bool TestInScatteringDeprecated() {
        // Capture with in-scattering OFF
        m_useInScattering = false;
        auto frame1 = CaptureFrame();

        // Capture with in-scattering "ON"
        m_useInScattering = true;
        auto frame2 = CaptureFrame();

        // Verify frames are identical (feature doesn't work)
        float diff = ComputeFrameDifference(frame1, frame2);
        return diff < 0.001f;  // <0.1% difference = no effect
    }
};
```

**Run these tests automatically on each build to generate:**
- `test_results.json` - Feature validation results
- Update `FEATURE_STATUS.md` based on test results

---

## Implementation Plan

### Phase 1: Documentation (Manual - 2-3 hours)

**Task:** Create `FEATURE_STATUS.md` by going through Application.h/cpp

**Process:**
1. List every `m_use*` bool flag
2. List every ImGui control
3. List every hotkey (OnKeyPress)
4. Document expected behavior
5. Mark status: ✅ COMPLETE / 🔄 WIP / ❌ DEPRECATED

**Output:** Single source of truth document

### Phase 2: Enhanced Metadata (Automated - 3-4 hours)

**Task:** Expand ScreenshotMetadata struct (v2.0)

**Changes:**
1. Add `activeLightingSystem` enum
2. Add `qualityPreset` and `targetFPS`
3. Add `PhysicalEffects` struct with all toggles
4. Add `LightConfig` with colors/positions
5. Add `FeatureStatus` flags
6. Update `GatherScreenshotMetadata()` to populate new fields
7. Update `SaveScreenshotMetadata()` to serialize new fields

**Output:** Comprehensive metadata JSON

### Phase 3: Automated Audit (Semi-Automated - 2-3 hours)

**Task:** Create Python audit script

**Process:**
1. Grep for all `m_use*` flags in Application.h
2. Search for ImGui controls for each flag
3. Search for hotkey bindings
4. Search for shader usage
5. Generate `feature_audit.json`
6. Cross-reference with `FEATURE_STATUS.md`
7. Flag mismatches (feature in code but marked deprecated)

**Output:** `feature_audit.json` + validation report

### Phase 4: Runtime Tests (Automated - 4-5 hours)

**Task:** Create automated validation tests

**Process:**
1. Implement `FeatureValidationTests` class
2. Add tests for each feature
3. Run on CI/CD (or manually)
4. Update `FEATURE_STATUS.md` with test results
5. Flag features that fail tests

**Output:** `test_results.json` + updated status doc

---

## Agent Integration

**Once complete, the agent will:**

1. **Read FEATURE_STATUS.md** before giving advice
2. **Check metadata schema v2.0** for actual config
3. **Know quality targets** (30/60/120/165 FPS based on preset)
4. **Avoid recommending deprecated features** (no more "enable in-scattering")
5. **Give accurate FPS expectations** ("At Ultra quality, 30 FPS is target, not 120")
6. **Understand lighting system** ("You're using Multi-Light, not RTXDI, so M5 status is irrelevant")

**Example improved recommendation:**

**Before (Phase 1):**
> "Your FPS is 34 - should be 120. Major bottleneck!"

**After (Phase 2 + audit):**
> "Your FPS is 34.1 at Ultra quality (target: 30 FPS). You're **exceeding target by 13%** - performance is good!
>
> Note: Your metadata shows `rtxdi_m5_enabled: true` but `activeLightingSystem: MultiLight`. The M5 setting is irrelevant when using Multi-Light system. No RTXDI fixes needed."

---

## Recommended Approach

**Start with Phase 1 (Manual Documentation):**

I can help create `FEATURE_STATUS.md` by:
1. Reading through `Application.h` and extracting all flags
2. Searching for ImGui controls
3. Searching for hotkey bindings
4. Creating comprehensive documentation

**Then Phase 2 (Enhanced Metadata):**

Update the C++ metadata struct and capture logic to include:
- Active lighting system
- Quality preset
- Target FPS
- All physical effects
- Feature status flags

**Would you like me to:**
1. **Start with Phase 1** - Create comprehensive FEATURE_STATUS.md?
2. **Jump to Phase 2** - Design and implement metadata v2.0?
3. **Create audit script** - Python tool to auto-generate status?
4. **All of the above** - Complete audit system?

This will make the agent recommendations **10× more accurate and useful**.
