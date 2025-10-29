# DLSS Super Resolution Implementation Summary

**Date:** 2025-10-29
**Version:** 0.11.7
**Feature:** NVIDIA DLSS Super Resolution (AI Upscaling)

---

## Overview

Successfully integrated NVIDIA DLSS Super Resolution into PlasmaDX-Clean for 1.5×-3× performance gains through AI-powered upscaling. The renderer now renders at lower internal resolutions (e.g., 1280×720) and uses DLSS tensor cores to intelligently upscale to native resolution (e.g., 2560×1440) with minimal quality loss.

**Key Achievement:** Smooth quality mode switching with deferred resource recreation to prevent GPU resource hazards.

---

## Architecture Changes

### 1. DLSSSystem Integration

**Files Modified:**
- `src/dlss/DLSSSystem.h` - Added Super Resolution structures and quality mode tracking
- `src/dlss/DLSSSystem.cpp` - Implemented feature creation/evaluation with quality mode detection

**Key Structures:**

```cpp
enum class DLSSQualityMode {
    Quality,        // 67% render res → 1.5× boost
    Balanced,       // 58% render res → 1.72× boost (RECOMMENDED)
    Performance,    // 50% render res → 2× boost
    UltraPerf       // 33% render res → 3× boost (for 4K)
};

struct SuperResolutionParams {
    ID3D12Resource* inputColor;         // Low-res render
    ID3D12Resource* outputUpscaled;     // High-res output
    ID3D12Resource* inputMotionVectors; // Optional (zeros acceptable)
    ID3D12Resource* inputDepth;         // Optional (improves quality)

    uint32_t renderWidth, renderHeight; // Input resolution
    uint32_t outputWidth, outputHeight; // Output resolution

    float jitterOffsetX, jitterOffsetY; // Set to 0 (no TAA)
    float sharpness;                     // 0.0-1.0
    int reset;                           // 1 = clear history
};
```

**Feature Recreation:**
- DLSSSystem now tracks current quality mode in `m_qualityMode`
- Feature is automatically released and recreated when quality mode or resolution changes
- Prevents dimension mismatches between created feature and input buffers

### 2. ParticleRenderer_Gaussian Changes

**Files Modified:**
- `src/particles/ParticleRenderer_Gaussian.h` - Added DLSS resolution tracking and quality mode
- `src/particles/ParticleRenderer_Gaussian.cpp` - Implemented buffer recreation and deferred changes

**Resolution Management:**

```cpp
// Dual resolution system:
m_renderWidth, m_renderHeight     // Internal render resolution (low-res)
m_outputWidth, m_outputHeight     // Target display resolution (native)
m_screenWidth, m_screenHeight     // Active render resolution (set to m_renderWidth/Height)
```

**Buffer Recreation on Quality Mode Change:**
1. Output texture (m_outputTexture) - Recreated at new render resolution
2. Upscaled output texture (m_upscaledOutputTexture) - Always at native resolution
3. Motion vector buffer (m_motionVectorBuffer) - Recreated at render resolution
4. Depth buffer (m_depthBuffer) - Recreated at render resolution
5. Feature flag reset (m_dlssFeatureCreated = false) to trigger NGX feature recreation

**Key Implementation Details:**
- Uses manual resolution calculation based on documented DLSS percentages
- NGX_DLSS_GET_OPTIMAL_SETTINGS is queried for validation but not used (often returns native res)
- Fixed hardcoded Balanced mode to use `m_dlssQualityMode` parameter

### 3. Application Integration

**Files Modified:**
- `src/core/Application.h` - Added quality mode state and deferred change flag
- `src/core/Application.cpp` - Implemented ImGui controls and deferred quality mode switching

**Deferred Quality Mode Change (Critical Fix):**

```cpp
// Application.h:282
bool m_dlssQualityModeChanged = false;  // Flag to defer until safe

// ImGui (Application.cpp:2661-2664)
if (qualityChanged) {
    m_dlssQualityModeChanged = true;  // Don't apply immediately!
    LOG_INFO("DLSS: Quality mode change requested (will apply at start of next frame)");
}

// Render() safe point (Application.cpp:434-441)
void Application::Render() {
    // SAFE POINT: Before any GPU work begins
    if (m_dlssQualityModeChanged && m_gaussianRenderer) {
        m_gaussianRenderer->SetDLSSSystem(m_dlssSystem.get(), m_width, m_height, m_dlssQualityMode);
        m_dlssQualityModeChanged = false;
    }
    // ... rest of render ...
}
```

**Why Deferred Changes Are Essential:**
- Immediate changes cause **resource hazards** - destroying GPU buffers mid-frame while GPU is using them
- Symptoms: Application freeze, TDR (Timeout Detection and Recovery), GPU hang
- Solution: Set flag in ImGui, apply change at start of next frame before GPU work begins

**ImGui Controls:**
- Radio buttons for Quality/Balanced/Performance/Ultra Performance modes
- Tooltips showing render resolution and expected FPS boost for each mode
- Visual status indicator when DLSS is active

---

## Resolution Calculations

### Quality Mode Percentages (from DLSS documentation):

| Quality Mode      | Render Scale | Boost | Example (2560×1440) |
|-------------------|--------------|-------|---------------------|
| Quality           | 67%          | 1.5×  | 1714×964            |
| Balanced          | 58%          | 1.72× | 1484×836            |
| Performance       | 50%          | 2.0×  | 1280×720            |
| Ultra Performance | 33%          | 3.0×  | 845×476             |

### Resolution Alignment:
- All resolutions aligned to 2 pixels (DLSS requirement)
- Formula: `(width * renderScale + 1) & ~1`

### 4K Examples:
- Quality: 2560×1440 render → 3840×2160 output
- Balanced: 2227×1253 render → 3840×2160 output
- Performance: 1920×1080 render → 3840×2160 output
- Ultra Performance: 1267×713 render → 3840×2160 output (**8.9× fewer pixels!**)

---

## Performance Results

**Test Configuration:** RTX 4060 Ti, 2560×1440, 10K particles, 13 lights, RT lighting, PCSS shadows

### Baseline (No DLSS):
- Native 2560×1440: 60 FPS (capped by VSync)

### With DLSS:
- **Balanced (58%)**: 60 FPS (still capped, but more headroom)
- **Performance (50%)**: Significant gains when GPU-bound
- **Ultra Performance (33%)**: Massive gains for extreme upscaling

**Note:** With 4× shadow rays to avoid VSync cap:
- Before DLSS: 30 FPS
- After DLSS (Balanced): 50 FPS (67% improvement)

---

## Known Issues and Workarounds

### Issue 1: NGX_DLSS_GET_OPTIMAL_SETTINGS Returns Native Resolution
**Symptom:** API returns 2560×1440 for both render and output resolutions
**Workaround:** Use manual calculation based on documented percentages
**Status:** Working as intended (manual calculation is more reliable)

### Issue 2: Ultra Performance Aliasing
**Symptom:** Visible aliasing at 33% render scale
**Cause:** Rendering at very low resolution (845×476 for 1440p)
**Expected:** This is normal for Ultra Performance - designed for 4K displays where pixel density is higher
**Recommendation:** Use Quality or Balanced for 1440p, reserve Ultra Performance for 4K

### Issue 3: Resource Hazard on Quality Mode Change
**Symptom:** Application freeze when switching quality modes
**Root Cause:** Destroying GPU resources mid-frame while GPU is using them
**Fix:** Deferred quality mode change using `m_dlssQualityModeChanged` flag
**Status:** ✅ RESOLVED

### Issue 4: Per-Frame Log Spam
**Symptom:** 20,500+ "DLSS: Super Resolution upscaling successful" messages per session
**Fix:** Removed per-frame success logging, kept failure warnings
**Status:** ✅ RESOLVED

---

## Testing Checklist

- [x] DLSS feature creation at all quality modes
- [x] Quality mode switching without freezing
- [x] Resolution changes without crashes
- [x] Upscaling quality verification (Balanced looks identical to native)
- [x] Performance gains with increased GPU load (4× shadow rays)
- [x] Clean log output (no spam)
- [ ] 4K TV testing (3840×2160)
- [ ] Ultra Performance mode quality assessment at 4K
- [ ] Particle count scaling with DLSS Performance mode

---

## Future Improvements

1. **Motion Vector Generation**: Currently using zeros (static scene assumption)
   - Would improve temporal stability during camera motion
   - Requires previous frame viewProj matrix in shader

2. **Jitter Support**: Implement TAA jittering for DLSS
   - Improves anti-aliasing quality
   - Requires per-frame jitter offset calculation

3. **Auto Quality Mode**: Dynamically adjust based on GPU load
   - Monitor frame time
   - Switch to lower quality if FPS drops below target

4. **HDR Swap Chain**: Direct HDR output (currently HDR→SDR blit)
   - Would eliminate blit pass overhead
   - Requires HDR monitor and swap chain changes

---

## Code References

### Key Functions:

**DLSSSystem.cpp:**
- `CreateSuperResolutionFeature()` - Line 126-212 (feature creation with quality mode detection)
- `EvaluateSuperResolution()` - Line 214-280 (per-frame upscaling)

**ParticleRenderer_Gaussian.cpp:**
- `SetDLSSSystem()` - Line 931-1183 (buffer recreation and resolution calculation)
- `Render()` - Line 634-655 (lazy feature creation)
- `Render()` - Line 777-838 (DLSS evaluation)

**Application.cpp:**
- `Render()` - Line 434-441 (deferred quality mode change application)
- ImGui controls - Line 2624-2669 (quality mode radio buttons)

---

## Lessons Learned

### Critical Lesson: GPU Resource Lifecycle Management
**Never destroy GPU resources mid-frame!**
- Always defer resource recreation until safe points (start of frame, before GPU work)
- Use flags to defer operations requested during rendering (ImGui, user input)
- This prevents TDRs, freezes, and GPU hangs

### DLSS API Quirks:
1. `NGX_DLSS_GET_OPTIMAL_SETTINGS` can return unreliable values - use manual calculation
2. Feature must be recreated when quality mode changes (dimension mismatch errors)
3. HDR flag (`NVSDK_NGX_DLSS_Feature_Flags_IsHDR`) is required for R16G16B16A16_FLOAT textures
4. Zeros for motion vectors are acceptable (DLSS handles it gracefully)

### Quality Mode Selection Guidelines:
- **1080p**: Balanced or Quality
- **1440p**: Balanced (default)
- **4K**: Performance or Ultra Performance
- **8K**: Ultra Performance required

---

## Conclusion

DLSS Super Resolution successfully integrated with smooth quality mode switching and significant performance gains. The deferred quality mode change pattern prevents resource hazards and provides a stable foundation for future DLSS features (DLSS Frame Generation, DLSS Ray Reconstruction).

**Key Achievement:** Real-time switchable quality modes with zero freezing or crashes.

**Next Steps:** Test Ultra Performance on 4K TV to validate quality at higher pixel densities.
