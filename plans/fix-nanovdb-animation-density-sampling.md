# fix: NanoVDB Animation Density Sampling Returns Zero

**Type:** Bug Fix
**Priority:** High
**Created:** 2025-12-23
**Status:** Ready for Implementation

---

## Overview

NanoVDB volumetric animation loads successfully (131 frames, bounds extracted, frames cycle) but rendering produces no visible output. Debug mode shows **cyan** indicating "Ray hits AABB but no density (sampling issue)".

**Root Cause Identified:** Coordinate space mismatch between C++ bounds extraction and HLSL shader sampling.

---

## Problem Statement

### Symptom
- Animation loads correctly (131 frames detected)
- Bounds extracted: `(0, 11, 0)` to `(256, 252, 328)`
- Camera at `(800, 1200, 0)` looking at `(0, 0, 0)`
- Debug visualization shows **cyan** = rays hit AABB but density sampling returns 0

### Root Cause Analysis

**The core mismatch:**

| Component | Bounds Calculation Method | Coordinate System |
|-----------|---------------------------|-------------------|
| C++ Animation Load | `indexBBox.min/max() * voxelSize` | Manual world calculation |
| HLSL Shader Sampling | `pnanovdb_grid_world_to_indexf()` | NanoVDB's internal map transform |

The C++ code uses **simple multiplication** which ignores any translation/rotation in the grid's map transform. The shader uses NanoVDB's **actual transform** which may include offsets.

**Result:** Ray marches through the AABB (because bounds are computed one way), but when sampling density, the coordinates map to empty space (because transform is computed differently).

### Evidence

From `NanoVDBSystem.cpp`:
```cpp
// Animation load (lines 1016-1026) - BROKEN
auto indexBBox = floatGrid->indexBBox();
float voxelSize = static_cast<float>(floatGrid->voxelSize()[0]);
float minX = indexBBox.min()[0] * voxelSize;  // Simple multiply - WRONG
```

From `nanovdb_raymarch.hlsl`:
```hlsl
// Shader sampling (line 159) - Uses actual transform
pnanovdb_vec3_t indexVec = pnanovdb_grid_world_to_indexf(g_gridBuffer, grid, worldVec);
```

---

## Proposed Solution

### Phase 1: Core Fix (This PR)

Replace manual bounds calculation with NanoVDB's pre-computed `mWorldBBox`:

```cpp
// BEFORE (broken):
auto indexBBox = floatGrid->indexBBox();
float voxelSize = static_cast<float>(floatGrid->voxelSize()[0]);
m_gridWorldMin = {
    indexBBox.min()[0] * voxelSize,
    indexBBox.min()[1] * voxelSize,
    indexBBox.min()[2] * voxelSize
};

// AFTER (fixed):
const nanovdb::GridData* gridData = handle.gridData();
auto worldBBox = gridData->mWorldBBox;
m_gridWorldMin = {
    static_cast<float>(worldBBox.mCoord[0][0]),
    static_cast<float>(worldBBox.mCoord[0][1]),
    static_cast<float>(worldBBox.mCoord[0][2])
};
m_gridWorldMax = {
    static_cast<float>(worldBBox.mCoord[1][0]),
    static_cast<float>(worldBBox.mCoord[1][1]),
    static_cast<float>(worldBBox.mCoord[1][2])
};
```

### Phase 2: Validation & Edge Cases

Add bounds validation to catch corrupt data:

```cpp
// Validate worldBBox is not inf/NaN
bool ValidateWorldBBox(const nanovdb::BBox<nanovdb::Vec3d>& bbox) {
    for (int i = 0; i < 3; i++) {
        if (!std::isfinite(bbox.mCoord[0][i]) || !std::isfinite(bbox.mCoord[1][i])) {
            LOG_ERROR("[NanoVDB] Invalid worldBBox: component {} is inf/NaN", i);
            return false;
        }
    }
    if (bbox.mCoord[0][0] >= bbox.mCoord[1][0]) {
        LOG_ERROR("[NanoVDB] Invalid worldBBox: zero or negative volume");
        return false;
    }
    return true;
}
```

### Phase 3: Debug Visualization

Add index-space coordinate visualization to shader debug mode:

```hlsl
// In debug mode, visualize where we're actually sampling
if (debugMode == 2) {  // New mode: Index Space Visualization
    pnanovdb_vec3_t indexVec = pnanovdb_grid_world_to_indexf(g_gridBuffer, grid, worldVec);
    g_output[DTid.xy] = float4(
        frac(indexVec.x / 10.0),  // R = index X mod 10
        frac(indexVec.y / 10.0),  // G = index Y mod 10
        frac(indexVec.z / 10.0),  // B = index Z mod 10
        1.0
    );
}
```

---

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NanoVDB Animation Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LoadAnimationSequence()                                         │
│  ├── For each frame:                                            │
│  │   ├── nanovdb::io::readGrid(filepath)                        │
│  │   ├── Extract gridData->mWorldBBox  ◄── FIX HERE            │
│  │   ├── ValidateWorldBBox()           ◄── ADD                  │
│  │   ├── ComputeUnionBounds()          ◄── FUTURE (Phase 2)    │
│  │   └── Upload to GPU buffer                                   │
│  │                                                               │
│  AdvanceFrame()                                                  │
│  ├── Update m_gridWorldMin/Max from cached bounds               │
│  ├── Update shader constants (NanoVDBConstants cbuffer)         │
│  └── Dispatch nanovdb_raymarch.hlsl                             │
│                                                                  │
│  nanovdb_raymarch.hlsl                                          │
│  ├── Ray-AABB intersection (uses m_gridWorldMin/Max)            │
│  ├── Ray march through volume                                   │
│  ├── pnanovdb_grid_world_to_indexf()  ◄── Must match bounds!   │
│  └── Sample density via tree traversal                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Core Fix (2 hours)

**Tasks:**
- [ ] Modify `LoadAnimationSequence()` to use `gridData->mWorldBBox`
- [ ] Add `ValidateWorldBBox()` helper function
- [ ] Test with 131-frame gasoline explosion
- [ ] Verify cyan debug color disappears
- [ ] Verify density sampling produces visible output

**Files to modify:**
- `src/rendering/NanoVDBSystem.cpp:1016-1043`
- `src/rendering/NanoVDBSystem.h` (add validation helper)

#### Phase 2: Edge Cases & Robustness (4 hours)

**Tasks:**
- [ ] Handle empty frame 0 (auto-skip to first non-empty frame)
- [ ] Add per-frame bounds caching (avoid re-extracting each playback)
- [ ] Implement bounds union option (stable camera framing)
- [ ] Add BLAS rebuild optimization (only on significant bounds change)
- [ ] Handle multi-grid VDB files (select "density" grid by name)

**Files to modify:**
- `src/rendering/NanoVDBSystem.cpp`
- `src/rendering/NanoVDBSystem.h`

#### Phase 3: Debug & Polish (3 hours)

**Tasks:**
- [ ] Add index-space visualization mode (shader debug mode 2)
- [ ] Add bounds wireframe overlay
- [ ] Improve ImGui display (show current frame bounds, grid type)
- [ ] Add precision warning for large coordinates (>100K from origin)

**Files to modify:**
- `shaders/volumetric/nanovdb_raymarch.hlsl:731-769`
- `src/rendering/NanoVDBSystem.cpp` (ImGui section)

---

## Acceptance Criteria

### Functional Requirements

- [ ] Animation frames render visible volumetric density (not black/cyan)
- [ ] Debug mode shows green/yellow (density found) instead of cyan
- [ ] Bounds extraction uses `mWorldBBox` for consistent coordinate system
- [ ] Invalid bounds (inf/NaN) are detected and logged with error message
- [ ] Empty frame 0 is handled gracefully (skip or user notification)

### Non-Functional Requirements

- [ ] Animation playback maintains 60+ FPS after fix
- [ ] No memory leaks from bounds caching
- [ ] Backwards compatible with single-file VDB loading

### Quality Gates

- [ ] Tested with 131-frame gasoline explosion animation
- [ ] Tested with CloudPack single-frame volumes
- [ ] No GPU hangs or TDR crashes
- [ ] Debug visualization modes all functional

---

## Success Metrics

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Debug color | Cyan (no density) | Green/Yellow (density found) |
| Visible output | Black screen | Volumetric rendering |
| Frame bounds | Manual calculation | `mWorldBBox` from grid |
| Coordinate consistency | Mismatch | Unified system |

---

## Dependencies & Prerequisites

**Required:**
- NanoVDB header library (already integrated at `external/nanovdb/`)
- Existing `NanoVDBSystem` class (functional except bounds extraction)
- 131-frame gasoline explosion test asset (`assets/volumes/explosion/`)

**No external dependencies needed.**

---

## Risk Analysis & Mitigation

### High Risk: BLAS/TLAS Rebuild

**Risk:** Changing bounds may require BLAS rebuild, causing 2.1ms stall per frame.

**Mitigation:** Compare new bounds to previous; only rebuild if >1% volume change.

### Medium Risk: Multi-Grid VDB Files

**Risk:** VDB files may have multiple grids (density, velocity, heat). Selecting wrong grid returns zeros.

**Mitigation:** Enumerate grids, match "density" (case-insensitive), fallback to first FogVolume type.

### Low Risk: Precision Loss

**Risk:** Large world coordinates (>100K) may cause float precision issues in shader.

**Mitigation:** Detect and warn in ImGui; offer auto-center option.

---

## Future Considerations

1. **Async animation loading** - Current 4-minute load time blocks UI
2. **Frame bounds interpolation** - Smooth camera framing across varying bounds
3. **NanoVDB + Gaussian hybrid** - Render both systems in same scene
4. **RT lighting for NanoVDB** - Currently only Gaussians are lit by multi-light system

---

## Documentation Plan

**Files to update after implementation:**
- `CLAUDE.md` - Update NanoVDB status from "rendering incomplete" to "working"
- `docs/PROJECT_STATUS_DEC2025.md` - Mark animation rendering as fixed
- `README.md` - Add NanoVDB animation as feature

---

## References & Research

### Internal References

**Key files identified:**
- Bounds extraction bug: `src/rendering/NanoVDBSystem.cpp:1016-1026`
- Shader sampling: `shaders/volumetric/nanovdb_raymarch.hlsl:148-165`
- Constants struct: `src/rendering/NanoVDBSystem.h:349-376`
- Debug visualization: `shaders/volumetric/nanovdb_raymarch.hlsl:731-769`

### External References

**Official Documentation:**
- [NanoVDB: A GPU-Friendly and Portable VDB Data Structure (NVIDIA)](https://research.nvidia.com/labs/prl/publication/nanovdb/)
- [OpenVDB FAQ - NanoVDB](https://www.openvdb.org/documentation/doxygen/NanoVDB_FAQ.html)
- [GitHub Discussion: Creating GridBuffer with PNanoVDB (DX12/HLSL)](https://github.com/AcademySoftwareFoundation/openvdb/discussions/1307)

**Best Practices:**
- [Accelerating OpenVDB on GPUs with NanoVDB (NVIDIA Blog)](https://developer.nvidia.com/blog/accelerating-openvdb-on-gpus-with-nanovdb/)
- [Unreal VDB Plugin Troubleshooting](https://github.com/eidosmontreal/unreal-vdb/blob/main/HELPME.md)

**Implementation Examples:**
- [NanoVDB Hello World Examples](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/doc/nanovdb/HelloWorld.md)
- [PNanoVDB.h Source Code](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/PNanoVDB.h)

---

## MVP Implementation

### NanoVDBSystem.cpp (Phase 1 Fix)

```cpp
// Location: src/rendering/NanoVDBSystem.cpp
// Replace lines 1016-1043 with:

bool NanoVDBSystem::ExtractAnimationFrameBounds(
    const nanovdb::GridHandle<>& handle,
    XMFLOAT3& outMin,
    XMFLOAT3& outMax)
{
    const nanovdb::GridData* gridData = handle.gridData();
    if (!gridData) {
        LOG_ERROR("[NanoVDB] Failed to get grid data for bounds extraction");
        return false;
    }

    // Use NanoVDB's pre-computed world bounding box
    // This matches the coordinate system used by pnanovdb_grid_world_to_indexf()
    auto worldBBox = gridData->mWorldBBox;

    // Validate bounds are finite
    for (int i = 0; i < 3; i++) {
        if (!std::isfinite(worldBBox.mCoord[0][i]) ||
            !std::isfinite(worldBBox.mCoord[1][i])) {
            LOG_ERROR("[NanoVDB] worldBBox contains inf/NaN at component {}", i);
            return false;
        }
    }

    // Validate non-zero volume
    if (worldBBox.mCoord[0][0] >= worldBBox.mCoord[1][0] ||
        worldBBox.mCoord[0][1] >= worldBBox.mCoord[1][1] ||
        worldBBox.mCoord[0][2] >= worldBBox.mCoord[1][2]) {
        LOG_ERROR("[NanoVDB] worldBBox has zero or negative volume");
        return false;
    }

    outMin = {
        static_cast<float>(worldBBox.mCoord[0][0]),
        static_cast<float>(worldBBox.mCoord[0][1]),
        static_cast<float>(worldBBox.mCoord[0][2])
    };
    outMax = {
        static_cast<float>(worldBBox.mCoord[1][0]),
        static_cast<float>(worldBBox.mCoord[1][1]),
        static_cast<float>(worldBBox.mCoord[1][2])
    };

    LOG_INFO("[NanoVDB] Extracted world bounds: ({:.1f}, {:.1f}, {:.1f}) to ({:.1f}, {:.1f}, {:.1f})",
             outMin.x, outMin.y, outMin.z, outMax.x, outMax.y, outMax.z);

    return true;
}
```

### nanovdb_raymarch.hlsl (Debug Enhancement)

```hlsl
// Location: shaders/volumetric/nanovdb_raymarch.hlsl
// Add after line 769 (new debug mode):

// Debug Mode 2: Index Space Visualization
if (debugMode == 2) {
    if (RayAABBIntersection(rayOrigin, rayDir, gridWorldMin, gridWorldMax, tMin, tMax)) {
        float3 testPos = rayOrigin + rayDir * ((tMin + tMax) * 0.5);

        // Transform to index space
        pnanovdb_vec3_t worldVec = { testPos.x, testPos.y, testPos.z };
        pnanovdb_vec3_t indexVec = pnanovdb_grid_world_to_indexf(g_gridBuffer, grid, worldVec);

        // Visualize index coordinates as colors (periodic every 10 units)
        g_output[DTid.xy] = float4(
            frac(indexVec.x / 10.0),  // R = X position
            frac(indexVec.y / 10.0),  // G = Y position
            frac(indexVec.z / 10.0),  // B = Z position
            1.0
        );
    } else {
        g_output[DTid.xy] = float4(0.2, 0, 0, 1);  // Dark red = ray missed AABB
    }
    return;
}
```

---

## Edge Case Test Matrix

| Test Case | Input | Expected Output | Priority |
|-----------|-------|-----------------|----------|
| Single valid frame | 1-frame VDB | Renders correctly | P0 |
| 131-frame animation | Gasoline explosion | All frames visible | P0 |
| Empty frame 0 | Typical sim export | Skip/warn, render frame 1+ | P1 |
| Invalid bounds | inf/NaN in mWorldBBox | Error logged, frame skipped | P1 |
| Multi-grid VDB | density + velocity | Select "density" grid | P1 |
| Large coordinates | Grid at 1e6 | Precision warning | P2 |
| Varying frame bounds | Expanding smoke | Camera framing decision | P2 |

---

*Plan created: 2025-12-23*
*Research agents: repo-research-analyst, best-practices-researcher, framework-docs-researcher*
*Analysis: spec-flow-analyzer*
