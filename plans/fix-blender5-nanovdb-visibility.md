# Fix: Blender 5 NanoVDB Assets Not Rendering

**Type:** bug fix + enhancement
**Priority:** High
**Created:** 2025-12-17
**Status:** Ready for Implementation

---

## Overview

PlasmaDX-Clean's NanoVDB volumetric rendering system works for procedural fog and some downloaded .nvdb files, but **Blender 5-created NanoVDB assets are invisible**. This plan addresses the root causes and implements fixes to enable the full Blender 5 → PlasmaDX workflow.

## Problem Statement

Despite having a working NanoVDB raymarching pipeline, volumes exported from Blender 5 do not render:

- **Symptom:** Debug mode shows cyan pixels ("inside AABB but no density") instead of green ("density found")
- **Working:** Procedural fog sphere renders correctly
- **Working:** Some downloaded test .nvdb files render correctly
- **Broken:** Blender 5-exported .nvdb files appear invisible

## Root Cause Analysis

### Cause 1: Grid Type Mismatch (CRITICAL)

**Evidence:**
- `shaders/volumetric/nanovdb_raymarch.hlsl:101` only accepts `gridType == 1u` (PNANOVDB_GRID_TYPE_FLOAT)
- Blender 5 OpenVDB cache defaults to **Half (16-bit)** precision for smaller file sizes
- Half precision creates grids of type `PNANOVDB_GRID_TYPE_HALF (9)` or `PNANOVDB_GRID_TYPE_FP16 (15)`
- Shader returns `density = 0.0` for any non-FLOAT grid → invisible volume

**Blender Manual Reference:** `physics/fluid/type/domain/cache.html`
- Precision options: Full (32-bit), Half (16-bit), Mini (8-bit)

### Cause 2: Wrong Grid Selected (CRITICAL)

**Evidence:**
- `src/rendering/NanoVDBSystem.cpp:131` hardcodes grid index 0: `readGrid<HostBuffer>(filepath, 0, 1)`
- Blender exports multi-grid VDB files (density, temperature, velocity)
- If density grid is at index 1+, the system loads the wrong grid

**Current behavior:** Loads grid[0] regardless of name or type

### Cause 3: Axis/Scale Mismatch (SECONDARY)

**Evidence:**
- Blender uses Z-up coordinate system; many engines use Y-up
- Blender simulations often have small world bounds (single-digit meters)
- PlasmaDX scene scale is hundreds/thousands of units

**Current mitigation:** ImGui controls for scale/center exist but don't fix type/selection issues

---

## Acceptance Criteria

### Primary Goal A: Fix Blender 5 Assets Not Rendering
- [ ] At least one Blender 5-exported volumetric asset renders reliably in PlasmaDX-Clean
- [ ] Both static and animated sequences work
- [ ] Correct density values are sampled (not all zeros)

### Primary Goal B: World-Space Multi-Volume System
- [ ] Multiple NanoVDB volumes can be placed simultaneously
- [ ] Per-volume transforms (position/scale)
- [ ] Per-volume render parameters
- [ ] Per-volume animation sequences
- [ ] ImGui menu to manage volumes

---

## Technical Approach

### Phase 1: Diagnostics & Logging (Day 1)

**Objective:** Surface grid metadata to understand exactly what's being loaded.

#### 1.1 Enhance NanoVDBSystem::LoadFromFile logging

**File:** `src/rendering/NanoVDBSystem.cpp:112-230`

Add logging for:
- Grid name (via `floatGrid->gridName()`)
- Grid type (via `gridData->mGridType`)
- Grid value type (FLOAT, HALF, FP16, etc.)
- Active voxel count
- World bounds
- Voxel size

```cpp
// After loading, log comprehensive grid info
LOG_INFO("[NanoVDB] Grid metadata:");
LOG_INFO("  Name: {}", gridName);
LOG_INFO("  Type: {} ({})", gridTypeStr, gridType);
LOG_INFO("  Bounds: ({:.2f},{:.2f},{:.2f}) to ({:.2f},{:.2f},{:.2f})",
         worldBBox.min()[0], worldBBox.min()[1], worldBBox.min()[2],
         worldBBox.max()[0], worldBBox.max()[1], worldBBox.max()[2]);
LOG_INFO("  Voxel size: {:.4f}", voxelSize);
LOG_INFO("  Active voxels: {}", activeVoxels);
```

#### 1.2 Add ImGui readout

**File:** `src/core/Application.cpp` (NanoVDB UI section)

Display in ImGui:
- Loaded grid name
- Grid type (with warning if not FLOAT)
- Active voxel count
- File path

```cpp
if (m_nanoVDBSystem->HasFileGrid()) {
    ImGui::Text("Grid: %s", m_nanoVDBSystem->GetGridName().c_str());
    ImGui::Text("Type: %s", m_nanoVDBSystem->GetGridTypeName().c_str());
    if (m_nanoVDBSystem->GetGridType() != 1) {
        ImGui::TextColored(ImVec4(1,0.3,0.3,1), "WARNING: Not FLOAT type!");
    }
}
```

### Phase 2: Fix Grid Selection (Day 1-2)

**Objective:** Load the correct grid by name, not hardcoded index.

#### 2.1 Implement grid enumeration

**File:** `src/rendering/NanoVDBSystem.cpp`

Add method to list all grids in file:

```cpp
struct GridInfo {
    std::string name;
    uint32_t type;
    uint64_t voxelCount;
    int index;
};

std::vector<GridInfo> NanoVDBSystem::EnumerateGrids(const std::string& filepath) {
    // Use nanovdb::io::readGridMetaData() to enumerate without loading full data
    // Return list of all grids with name, type, index
}
```

#### 2.2 Implement grid selection by name

**File:** `src/rendering/NanoVDBSystem.cpp:131`

Replace hardcoded index 0 with intelligent selection:

```cpp
int gridIndex = SelectGridByName(filepath, "density");  // Prefer "density"
if (gridIndex < 0) {
    gridIndex = SelectFirstFloatGrid(filepath);  // Fallback to first FLOAT grid
}
if (gridIndex < 0) {
    LOG_ERROR("[NanoVDB] No suitable density grid found!");
    return false;
}

nanovdb::GridHandle<HostBuffer> handle =
    nanovdb::io::readGrid<HostBuffer>(filepath, gridIndex, 1);
```

#### 2.3 Add ImGui grid selector

**File:** `src/core/Application.cpp`

When multi-grid file loaded, show dropdown:

```cpp
if (gridList.size() > 1) {
    if (ImGui::BeginCombo("Select Grid", currentGridName.c_str())) {
        for (const auto& grid : gridList) {
            bool selected = (grid.index == currentGridIndex);
            if (ImGui::Selectable(grid.name.c_str(), selected)) {
                m_nanoVDBSystem->SelectGrid(grid.index);
            }
        }
        ImGui::EndCombo();
    }
}
```

### Phase 3: Fix Grid Type Support (Day 2-3)

**Objective:** Support HALF/FP16 grids from Blender's Half precision exports.

#### Option A: Shader Support (Recommended)

**File:** `shaders/volumetric/nanovdb_raymarch.hlsl:91-139`

Add HALF grid sampling using PNanoVDB helpers:

```hlsl
float SampleNanoVDBDensity(float3 worldPos) {
    pnanovdb_grid_handle_t grid;
    grid.address = pnanovdb_address_null();

    pnanovdb_uint32_t gridType = pnanovdb_grid_get_grid_type(g_gridBuffer, grid);

    // Support FLOAT (1), HALF (9), FP16 (15)
    if (gridType != PNANOVDB_GRID_TYPE_FLOAT &&
        gridType != PNANOVDB_GRID_TYPE_HALF &&
        gridType != PNANOVDB_GRID_TYPE_FP16) {
        return 0.0;
    }

    // ... tree traversal code ...

    // Read value based on type
    float density;
    if (gridType == PNANOVDB_GRID_TYPE_FLOAT) {
        density = pnanovdb_read_float(g_gridBuffer, valueAddr);
    } else if (gridType == PNANOVDB_GRID_TYPE_HALF || gridType == PNANOVDB_GRID_TYPE_FP16) {
        density = pnanovdb_read_half(g_gridBuffer, valueAddr);
    }

    return max(density, 0.0);
}
```

**PNanoVDB.h reference:** Line 1102 defines `PNANOVDB_GRID_TYPE_HALF = 9`, Line 1108 defines `PNANOVDB_GRID_TYPE_FP16 = 15`

#### Option B: CPU Transcoding (Alternative)

If shader complexity is a concern, transcode on CPU during load:

```cpp
// If grid is HALF, transcode to FLOAT before GPU upload
if (gridType == NANOVDB_GRID_TYPE_HALF) {
    handle = TranscodeToFloat(handle);  // Convert buffer
}
```

**Recommendation:** Option A (shader support) is preferred for flexibility and performance.

### Phase 4: Error Feedback (Day 3)

**Objective:** Provide actionable feedback when files fail to load.

#### 4.1 Add validation and error messages

**File:** `src/rendering/NanoVDBSystem.cpp`

```cpp
bool NanoVDBSystem::LoadFromFile(const std::string& filepath) {
    // Pre-validate
    auto grids = EnumerateGrids(filepath);
    if (grids.empty()) {
        m_lastError = "No grids found in file";
        LOG_ERROR("[NanoVDB] {}: {}", m_lastError, filepath);
        return false;
    }

    // Find suitable grid
    int gridIndex = SelectDensityGrid(grids);
    if (gridIndex < 0) {
        m_lastError = "No density grid found. Available: " + GridListToString(grids);
        LOG_ERROR("[NanoVDB] {}", m_lastError);
        return false;
    }

    // Check type support
    if (!IsGridTypeSupported(grids[gridIndex].type)) {
        m_lastError = fmt::format("Unsupported grid type: {} ({}). Re-export with Full precision.",
                                  GridTypeName(grids[gridIndex].type), grids[gridIndex].type);
        LOG_ERROR("[NanoVDB] {}", m_lastError);
        return false;
    }

    // Proceed with load...
}
```

#### 4.2 Display errors in ImGui

**File:** `src/core/Application.cpp`

```cpp
if (!m_nanoVDBSystem->GetLastError().empty()) {
    ImGui::TextColored(ImVec4(1,0.3,0.3,1), "Error: %s",
                       m_nanoVDBSystem->GetLastError().c_str());
}
```

#### 4.3 Expand debug visualization

**File:** `shaders/volumetric/nanovdb_raymarch.hlsl:559-580`

Add new debug colors:
- Magenta: Grid type unsupported
- Yellow: Multi-grid file, using fallback
- Orange: Transform/scale warning

```hlsl
if (debugMode == 1) {
    if (gridType != PNANOVDB_GRID_TYPE_FLOAT &&
        gridType != PNANOVDB_GRID_TYPE_HALF) {
        g_output[DTid.xy] = float4(1.0, 0.0, 1.0, 1.0);  // Magenta = unsupported type
        return;
    }
    // ... existing debug logic ...
}
```

### Phase 5: Multi-Volume System (Day 4-5)

**Objective:** Support multiple NanoVDB volumes simultaneously.

#### 5.1 Create NanoVDBVolumeInstance

**File:** `src/rendering/NanoVDBVolumeInstance.h` (NEW)

```cpp
struct NanoVDBVolumeInstance {
    std::string name;
    std::string filepath;

    // GPU resources
    ComPtr<ID3D12Resource> gridBuffer;
    D3D12_GPU_DESCRIPTOR_HANDLE srvGPU;

    // Transform
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 scale;
    DirectX::XMFLOAT3 rotation;

    // Render parameters
    float densityScale = 1.0f;
    float emissionStrength = 0.5f;
    float absorptionCoeff = 0.1f;

    // Animation state
    std::vector<AnimationFrame> animFrames;
    size_t currentFrame = 0;
    float animFPS = 24.0f;
    bool animPlaying = false;
    bool animLoop = true;

    // Bounds
    DirectX::XMFLOAT3 worldMin, worldMax;

    // State
    bool enabled = true;
    bool visible = true;
};
```

#### 5.2 Create NanoVDBVolumeManager

**File:** `src/rendering/NanoVDBVolumeManager.h` (NEW)

```cpp
class NanoVDBVolumeManager {
public:
    bool AddVolume(const std::string& name, const std::string& filepath);
    bool RemoveVolume(const std::string& name);

    void RenderAll(ID3D12GraphicsCommandList* cmdList, ...);
    void UpdateAnimations(float deltaTime);

    NanoVDBVolumeInstance* GetVolume(const std::string& name);
    const std::vector<NanoVDBVolumeInstance>& GetVolumes() const;

private:
    std::vector<NanoVDBVolumeInstance> m_volumes;
    size_t m_maxVolumes = 16;
};
```

#### 5.3 Extend ImGui controls

**File:** `src/core/Application.cpp`

```cpp
if (ImGui::CollapsingHeader("NanoVDB Volumes")) {
    // Add volume button
    if (ImGui::Button("Add Volume...")) {
        // File dialog
    }

    // Per-volume controls
    for (auto& vol : m_volumeManager->GetVolumes()) {
        if (ImGui::TreeNode(vol.name.c_str())) {
            ImGui::Checkbox("Enabled", &vol.enabled);
            ImGui::DragFloat3("Position", &vol.position.x, 1.0f);
            ImGui::DragFloat3("Scale", &vol.scale.x, 0.1f, 0.01f, 100.0f);
            ImGui::SliderFloat("Density", &vol.densityScale, 0.1f, 10.0f);

            if (vol.animFrames.size() > 0) {
                ImGui::Text("Animation: %zu frames", vol.animFrames.size());
                ImGui::SliderInt("Frame", (int*)&vol.currentFrame, 0, vol.animFrames.size()-1);
                ImGui::Checkbox("Play", &vol.animPlaying);
            }

            if (ImGui::Button("Remove")) {
                m_volumeManager->RemoveVolume(vol.name);
            }
            ImGui::TreePop();
        }
    }
}
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/rendering/NanoVDBSystem.cpp` | Modify | Add grid enumeration, name-based selection, type detection, error handling |
| `src/rendering/NanoVDBSystem.h` | Modify | Add GetGridName(), GetGridType(), GetLastError(), EnumerateGrids() |
| `shaders/volumetric/nanovdb_raymarch.hlsl` | Modify | Add HALF/FP16 grid type support, expand debug colors |
| `src/core/Application.cpp` | Modify | Enhanced ImGui controls, error display, grid selector |
| `src/rendering/NanoVDBVolumeInstance.h` | Create | Per-volume instance data structure |
| `src/rendering/NanoVDBVolumeManager.h` | Create | Multi-volume management class |
| `src/rendering/NanoVDBVolumeManager.cpp` | Create | Multi-volume implementation |

---

## Validation Checklist

### Phase 1: Diagnostics
- [ ] Load a Blender-exported .nvdb and verify grid metadata is logged
- [ ] Confirm ImGui shows grid name and type
- [ ] If type is HALF/FP16, confirm warning is displayed

### Phase 2: Grid Selection
- [ ] Load a multi-grid .nvdb file (density + temperature)
- [ ] Confirm "density" grid is selected automatically
- [ ] Confirm ImGui dropdown shows all grids

### Phase 3: Type Support
- [ ] Export from Blender with Half precision
- [ ] Load in PlasmaDX and confirm volume renders (not invisible)
- [ ] Compare visual quality to Full precision export

### Phase 4: Error Feedback
- [ ] Load an unsupported file (e.g., Vec3f velocity grid only)
- [ ] Confirm error message appears in console and ImGui
- [ ] Confirm debug visualization shows magenta for unsupported types

### Phase 5: Multi-Volume
- [ ] Add 3 volumes with independent positions
- [ ] Confirm all 3 render simultaneously
- [ ] Load animated sequence on one volume while others are static
- [ ] Confirm animation plays independently

---

## Blender Export Guide (User Documentation)

### Recommended Export Settings

1. **File → Export → OpenVDB**
2. **Precision: Full (32-bit)** - Critical for compatibility
3. **Grid Selection:** Ensure "density" grid is included
4. **Coordinate System:** Leave as Blender default (Z-up)

### Conversion to NanoVDB

```bash
# Using PlasmaDX conversion script
python scripts/convert_vdb_to_nvdb.py smoke.vdb smoke.nvdb --grid density

# Or using nanovdb_convert tool
nanovdb_convert smoke.vdb smoke.nvdb
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cyan debug color | Grid type not FLOAT | Re-export with Full precision |
| Volume too small/large | Scale mismatch | Adjust scale slider in ImGui |
| Volume invisible | Wrong grid selected | Use grid selector dropdown |
| No grids found | Corrupt or empty file | Re-export from Blender |

---

## References

### Internal Documentation
- `docs/NanoVDB/BLENDER_5_TO_NANOVDB_FINDINGS_2025-12-17.md`
- `docs/NanoVDB/COMPOUND_ENGINEERING_PROMPT_BLENDER_NANOVDB_WORLDSPACE.md`
- `docs/NanoVDB/NANOVDB_UNIFIED_ROADMAP_V1.md`

### Key Code Locations
- Loader: `src/rendering/NanoVDBSystem.cpp:112-230`
- Shader sampling: `shaders/volumetric/nanovdb_raymarch.hlsl:91-139`
- Grid types: `shaders/nanovdb/PNanoVDB.h:1093-1120`
- Conversion script: `scripts/convert_vdb_to_nvdb.py`

### External References
- Blender Cache Docs: `physics/fluid/type/domain/cache.html`
- NVIDIA NanoVDB: https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb
- PNanoVDB Header: Portable NanoVDB for GPU shaders

---

## Commit Strategy

Small, reviewable commits in this order:

1. `feat(nanovdb): add grid metadata logging and ImGui display`
2. `feat(nanovdb): implement grid enumeration and name-based selection`
3. `feat(nanovdb): add HALF/FP16 grid type support in shader`
4. `feat(nanovdb): add error feedback and expanded debug visualization`
5. `feat(nanovdb): implement NanoVDBVolumeInstance and VolumeManager`
6. `feat(nanovdb): add multi-volume ImGui controls`
7. `docs(nanovdb): add Blender export guide and troubleshooting`

Each commit should be independently buildable and testable.
