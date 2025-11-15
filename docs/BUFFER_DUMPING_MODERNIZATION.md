# Buffer Dumping System Modernization

**Date:** 2025-11-15
**Status:** ✅ Complete
**Build:** Debug x64 (verified)

---

## Problem Statement

The legacy buffer dumping system (from Phase 1, ~2 years old) only dumped hardcoded buffers:
- `g_particles.bin`
- `g_rtLighting.bin`
- Optionally 1-2 RTXDI buffers

**Impact:** Missing critical diagnostic data for:
- RTXDI M5 temporal accumulation debugging
- PCSS shadow temporal buffers (ping-pong)
- DLSS motion vectors and upscaled output
- Volumetric ReSTIR reservoirs
- Material properties buffer (Phase 8)

---

## Solution

Refactored `Application::DumpGPUBuffers()` to intelligently dump **8-12 buffers** based on active rendering systems.

### Conditional Buffer Dumping

**1. Core Buffers** (always dumped if they exist):
```cpp
g_particles.bin              // Particle state (position, velocity, temperature, emission)
g_materialProperties.bin     // Material types (Phase 8 - Celestial Body System)
```

**2. Lighting System Buffers** (conditional on `m_lightingSystem`):
```cpp
// MultiLight path
g_rtLighting_MultiLight.bin

// RTXDI path (calls RTXDILightingSystem::DumpBuffers())
g_rtxdi_currentReservoirs.bin
g_rtxdi_accumulated.bin
g_rtxdi_debugOutput.bin

// Volumetric ReSTIR path
g_volumetricReSTIR_reservoirs.bin
```

**3. Renderer Buffers** (Gaussian renderer only):
```cpp
// Shadow buffers (if m_useShadowRays)
g_pcss_shadowHistory0.bin    // Ping-pong temporal accumulation
g_pcss_shadowHistory1.bin

// Depth and lighting
g_depthBuffer.bin            // Screen-space depth (used for shadows)
g_gaussian_lights.bin        // Light buffer for probe grid

// DLSS buffers (if ENABLE_DLSS)
g_dlss_motionVectors.bin
g_dlss_upscaledOutput.bin
```

---

## Enhanced Metadata JSON

**Before:**
```json
{
  "frame": 120,
  "timestamp": "2025-11-15 14:32:10",
  "camera_position": [500.0, 200.0, 300.0],
  "particle_count": 10000,
  "render_mode": "Gaussian"
}
```

**After:**
```json
{
  "frame": 120,
  "timestamp": "2025-11-15 14:32:10",
  "camera": {
    "position": [500.0, 200.0, 300.0],
    "distance": 612.45,
    "angle": 0.52,
    "pitch": 0.34,
    "height": 0.0
  },
  "particles": {
    "count": 10000,
    "size": 15.0,
    "adaptive_radius_enabled": true
  },
  "rendering": {
    "mode": "Gaussian",
    "lighting_system": "MultiLight",
    "light_count": 13,
    "shadow_preset": "Performance",
    "shadow_rays_per_light": 1,
    "use_shadow_rays": true,
    "use_in_scattering": true,
    "use_phase_function": true,
    "rt_lighting_strength": 2.0,
    "rt_emission_suppression": 0.7
  },
  "emission": {
    "strength": 0.25,
    "threshold_kelvin": 22000.0,
    "temporal_rate": 0.03
  },
  "features": {
    "dlss_enabled": true,
    "adaptive_quality_enabled": false,
    "probe_grid_enabled": false
  },
  "performance": {
    "current_fps": 118.5
  }
}
```

**New Sections:**
- **camera**: Full camera state (position, angle, pitch, height)
- **particles**: Particle configuration (count, size, adaptive radius)
- **rendering**: Complete rendering state (lighting system, shadow preset, light count)
- **emission**: Dynamic emission settings (strength, threshold, temporal rate)
- **features**: Active feature flags (DLSS, adaptive quality, probe grid)
- **performance**: Current FPS

---

## Code Changes

### Files Modified

**`src/core/Application.cpp`** (165 lines changed):
- Lines 1880-1962: Refactored `DumpGPUBuffers()` (87 lines)
  - Conditional dumping based on active systems
  - Buffer count tracking
  - Descriptive buffer names
- Lines 2094-2171: Enhanced `WriteMetadataJSON()` (78 lines)
  - Structured JSON output
  - All rendering state captured

**`src/particles/ParticleRenderer_Gaussian.h`** (18 lines added):
- Lines 165-182: Added 5 new getter methods
  - `GetShadowBuffer(int index)` - PCSS ping-pong buffers (R16_FLOAT)
  - `GetDepthBuffer()` - Screen-space depth buffer
  - `GetMotionVectorBuffer()` - DLSS motion vectors (if ENABLE_DLSS)
  - `GetUpscaledOutputBuffer()` - DLSS upscaled output (if ENABLE_DLSS)

**Backup Created:**
- `src/core/Application.cpp.backup-buffer-dump` (full file backup before changes)

---

## Build Verification

```bash
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build /nologo /v:minimal /clp:ErrorsOnly
```

**Result:** ✅ Build succeeded (0 errors, 0 warnings)

---

## Testing Instructions

### Test Different Lighting Systems

```bash
# Multi-light path (default)
./build/Debug/PlasmaDX-Clean.exe --multi-light --dump-buffers 120

# RTXDI path (3 additional buffers)
./build/Debug/PlasmaDX-Clean.exe --rtxdi --dump-buffers 120

# Volumetric ReSTIR path (experimental)
./build/Debug/PlasmaDX-Clean.exe --volumetric-restir --dump-buffers 120
```

### Verify Metadata JSON

```bash
cat PIX/buffer_dumps/metadata.json | jq .
```

**Expected fields:**
- `camera` object with position, angle, pitch
- `rendering.lighting_system` (MultiLight, RTXDI, or VolumetricReSTIR)
- `rendering.shadow_preset` (Performance, Balanced, Quality)
- `emission` object with strength, threshold, temporal_rate
- `performance.current_fps`

### Verify Buffer Count

**Before refactor:** 2-3 buffers (~30 MB)
**After refactor:** 8-12 buffers (~100-200 MB)

```bash
ls -lh PIX/buffer_dumps/*.bin
```

**Expected buffers (MultiLight + shadows + DLSS):**
```
g_particles.bin               (10K particles × 64 bytes = 640 KB)
g_materialProperties.bin      (if Phase 8 implemented)
g_rtLighting_MultiLight.bin   (lighting buffer)
g_pcss_shadowHistory0.bin     (1920×1080 R16 = 4 MB)
g_pcss_shadowHistory1.bin     (1920×1080 R16 = 4 MB)
g_depthBuffer.bin             (1920×1080 depth)
g_gaussian_lights.bin         (13 lights × 64 bytes)
g_dlss_motionVectors.bin      (if DLSS enabled)
g_dlss_upscaledOutput.bin     (if DLSS enabled)
```

---

## Impact

### Debugging Benefits

1. **RTXDI M5 Temporal Accumulation:**
   - Now dumps reservoirs, accumulated buffers, debug output
   - Can analyze patchwork pattern issues with actual GPU data

2. **PCSS Shadow Quality:**
   - Ping-pong temporal buffers now dumped
   - Can verify temporal convergence and stability

3. **DLSS Integration:**
   - Motion vectors and upscaled output now available
   - Can debug DLSS quality issues and upscaling artifacts

4. **Volumetric ReSTIR:**
   - Reservoir buffer dumping enabled
   - Critical for experimental VolumetricReSTIR debugging

### Log Analysis RAG Integration

**Direct benefit:** More comprehensive buffer dumps → better diagnostic data for RAG ingestion
- Enhanced metadata provides full system context
- All buffers available for PIX capture analysis
- Improved correlation between logs, screenshots, and buffer states

---

## Future Enhancements

### Probe Grid Buffers (TODO)
When `m_probeGridSystem` is active, dump:
```cpp
g_probeGrid_irradiance.bin   // SH coefficients
g_probeGrid_visibility.bin   // Visibility data
```

**Requires:** Exposing `GetProbeBuffer()` from `ProbeGridSystem` to `Application`

### Buffer Size Tracking
Add to metadata JSON:
```json
"buffers": [
  {"name": "g_particles.bin", "size_bytes": 640000, "format": "Custom struct"},
  {"name": "g_pcss_shadowHistory0.bin", "size_bytes": 4147200, "format": "R16_FLOAT"}
]
```

### Differential Dumps
Optionally dump only buffers that changed since last dump (delta compression)

---

## Related Documentation

- `SESSION_SUMMARY_2025-11-15.md` - Full session notes (log-analysis-rag agent)
- `PIX/docs/QUICK_REFERENCE.md` - PIX debugging workflow
- `MASTER_ROADMAP_V2.md` - Phase roadmap (see Phase 4: RTXDI M5)
- `PCSS_IMPLEMENTATION_SUMMARY.md` - Shadow temporal accumulation details

---

**Author:** Claude Code (Sonnet 4.5)
**Last Updated:** 2025-11-15
**Version:** 1.0
