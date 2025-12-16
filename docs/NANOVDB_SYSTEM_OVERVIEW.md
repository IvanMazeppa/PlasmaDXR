# NanoVDB System Overview

**Version:** 1.0
**Status:** Production Ready
**Last Updated:** 2025-12-11

---

## Executive Summary

The NanoVDB system in PlasmaDX-Clean provides GPU-accelerated volumetric rendering for celestial bodies like nebulae, gas clouds, and animated smoke/fire effects. It supports both procedural fog generation and file-loaded volumetric grids.

### Key Capabilities

| Feature | Status | Description |
|---------|--------|-------------|
| Procedural Fog Spheres | Production | Runtime-generated amorphous gas clouds |
| File Loading (.nvdb) | Production | Load NanoVDB grids from disk |
| Animation Sequences | Production | Multi-frame volumetric animations |
| Runtime Controls | Production | ImGui sliders for all parameters |
| Depth Occlusion | Production | Volumes respect existing geometry |
| Multi-Light Support | Production | Up to 16 lights with scattering |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ASSET CREATION                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐  │
│  │   Blender 5.0   │      │  Python Script  │      │  nanovdb_convert │  │
│  │   Mantaflow     │ ───► │ convert_vdb_    │ ───► │  (Alternative)   │  │
│  │   Simulation    │      │ to_nvdb.py      │      │                  │  │
│  └─────────────────┘      └─────────────────┘      └──────────────────┘  │
│          │                        │                         │            │
│          └────────────────────────┼─────────────────────────┘            │
│                                   │                                      │
│                                   ▼                                      │
│                          ┌────────────────┐                              │
│                          │  .nvdb Files   │                              │
│                          │  (NanoVDB)     │                              │
│                          └───────┬────────┘                              │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PLASMADX RUNTIME                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      NanoVDBSystem (C++)                            │  │
│  │  Location: src/rendering/NanoVDBSystem.h/cpp                        │  │
│  │                                                                     │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │  │
│  │  │ LoadFromFile()  │  │ CreateFogSphere │  │ LoadAnimation       │ │  │
│  │  │ - .nvdb parsing │  │ - Procedural    │  │ Sequence()          │ │  │
│  │  │ - GPU upload    │  │   fog generation│  │ - Multi-frame       │ │  │
│  │  │ - SRV creation  │  │ - No GPU memory │  │   animation         │ │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │  │
│  │                                                                     │  │
│  │  Runtime Controls:                                                  │  │
│  │  - Density Scale     - Emission Strength    - Step Size            │  │
│  │  - Absorption Coeff  - Scattering Coeff     - Max Ray Distance     │  │
│  │  - Grid Position     - Grid Scale           - Animation FPS        │  │
│  │                                                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                   │                                      │
│                                   ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                   nanovdb_raymarch.hlsl                             │  │
│  │  Location: shaders/volumetric/nanovdb_raymarch.hlsl                 │  │
│  │                                                                     │  │
│  │  Features:                                                          │  │
│  │  - PNanoVDB.h integration (HLSL-compatible NanoVDB access)          │  │
│  │  - Trilinear interpolation for smooth density sampling              │  │
│  │  - FBM noise for amorphous gas appearance                           │  │
│  │  - Curl noise for fluid advection animation                         │  │
│  │  - Beer-Lambert absorption                                          │  │
│  │  - Henyey-Greenstein phase function for scattering                  │  │
│  │  - Blackbody temperature-to-color conversion                        │  │
│  │  - Depth buffer occlusion                                           │  │
│  │                                                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## File Locations

### Core Implementation

| File | Purpose |
|------|---------|
| `src/rendering/NanoVDBSystem.h` | Class declaration, API |
| `src/rendering/NanoVDBSystem.cpp` | Implementation |
| `shaders/volumetric/nanovdb_raymarch.hlsl` | GPU ray marching shader |
| `external/nanovdb/` | NanoVDB library (header-only) |

### Conversion Scripts

| File | Purpose |
|------|---------|
| `scripts/convert_vdb_to_nvdb.py` | Python VDB→NanoVDB converter |
| `scripts/blender_vdb_to_nvdb.py` | Blender-specific export script |
| `scripts/inspect_vdb.py` | VDB file inspection tool |

### Asset Storage

| Directory | Contents |
|-----------|----------|
| `VDBs/NanoVDB/` | Production .nvdb files |
| `VDBs/NanoVDB/chimney_smoke/` | Animated smoke sequence |
| `VDBs/Blender_projects/` | Source Blender files |
| `VDBs/Clouds/` | Cloud assets |
| `VDBs/Smoke/` | Smoke assets |

---

## API Reference

### Initialization

```cpp
// Initialize the system
bool Initialize(Device* device, ResourceManager* resources,
                uint32_t screenWidth, uint32_t screenHeight);

// Shutdown and release resources
void Shutdown();
```

### Grid Creation/Loading

```cpp
// Create procedural fog sphere (no GPU memory, shader-based)
bool CreateFogSphere(float radius, DirectX::XMFLOAT3 center,
                     float voxelSize = 5.0f, float halfWidth = 3.0f);

// Load single NanoVDB file
bool LoadFromFile(const std::string& filepath);

// Load animation sequence
bool LoadAnimationSequence(const std::vector<std::string>& filepaths);
size_t LoadAnimationFromDirectory(const std::string& directory,
                                  const std::string& pattern = "*.nvdb");
```

### Rendering

```cpp
void Render(
    ID3D12GraphicsCommandList* commandList,
    const DirectX::XMMATRIX& viewProj,
    const DirectX::XMFLOAT3& cameraPos,
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAV,
    D3D12_GPU_DESCRIPTOR_HANDLE lightSRV,
    D3D12_GPU_DESCRIPTOR_HANDLE depthSRV,
    uint32_t lightCount,
    ID3D12DescriptorHeap* descriptorHeap,
    uint32_t renderWidth,
    uint32_t renderHeight,
    float time = 0.0f);
```

### Runtime Parameters

```cpp
// Enable/Disable
void SetEnabled(bool enabled);
bool IsEnabled() const;

// Density and Appearance
void SetDensityScale(float scale);      // Default: 1.0
void SetEmissionStrength(float s);       // Default: 0.5
void SetAbsorptionCoeff(float coeff);    // Default: 0.1
void SetScatteringCoeff(float coeff);    // Default: 0.5

// Ray Marching
void SetMaxRayDistance(float dist);      // Default: 2000.0
void SetStepSize(float size);            // Default: 5.0

// Grid Positioning
void SetGridCenter(const DirectX::XMFLOAT3& center);
void ScaleGridBounds(float scale);

// Animation
void SetAnimationPlaying(bool playing);
void SetAnimationSpeed(float fps);       // Default: 24.0
void SetAnimationLoop(bool loop);        // Default: true
void SetAnimationFrame(size_t frame);
void UpdateAnimation(float deltaTime);   // Call each frame
```

---

## Performance Characteristics

### Memory Usage

| Grid Type | Size | GPU Memory |
|-----------|------|------------|
| Procedural Sphere | N/A | 0 MB (shader-based) |
| 128³ density grid | ~2 MB | ~2 MB |
| 256³ density grid | ~16 MB | ~16 MB |
| cloud_01.nvdb | 31 MB | ~31 MB |
| Animation (30 frames) | 30× single | 30× single |

### Performance Impact

| Scenario | FPS Impact | Notes |
|----------|------------|-------|
| Procedural fog (default) | -2-5 FPS | Noise computation in shader |
| Small .nvdb (128³) | -1-3 FPS | Fast sampling |
| Large .nvdb (256³+) | -5-15 FPS | Memory bandwidth limited |
| Animation playback | +memory only | Frame switching is fast |

---

## Integration Points

### With Gaussian Particle Renderer

The NanoVDB system composites **additively** with the existing Gaussian particle render:

```hlsl
// In nanovdb_raymarch.hlsl
float4 existingColor = g_output[DTid.xy];
float3 finalColor = existingColor.rgb + volumeColor.rgb;
g_output[DTid.xy] = float4(finalColor, finalAlpha);
```

### With DLSS

Render dimensions are passed to account for DLSS upscaling:

```cpp
void Render(..., uint32_t renderWidth, uint32_t renderHeight, ...);
// Uses renderWidth/renderHeight, not native resolution
```

### With Multi-Light System

Up to 16 lights are sampled for volumetric scattering:

```hlsl
for (uint i = 0; i < lightCount && i < 16; i++) {
    Light light = g_lights[i];
    // Henyey-Greenstein phase function
    float phase = HenyeyGreenstein(cosTheta, phaseG);
    totalLight += light.color * attenuation * phase * scatteringCoeff;
}
```

---

## Related Documentation

- [Two-Worktree Workflow](./TWO_WORKTREE_WORKFLOW.md) - Development architecture
- [VDB Pipeline Agents](./VDB_PIPELINE_AGENTS.md) - AI agent ecosystem
- [Blender Integration](../BLENDER_PLASMADX_WORKFLOW_SPEC.md) - Asset creation workflow

---

*Document maintained by: Claude Code Agent Ecosystem*
