# Descriptor Heap Free-List Pool Implementation

**Date:** 2025-12-19
**Status:** Complete - Integrated into DLSS resize path
**Build Verified:** Yes

---

## Overview

Implemented a free-list allocator for D3D12 descriptor heaps to prevent descriptor exhaustion during DLSS resize cycles and other buffer recreation operations.

### Problem Solved

Previously, descriptors were allocated linearly and never reclaimed. During window resize or DLSS quality mode changes, old descriptors were abandoned while new ones were allocated, eventually exhausting the heap.

### Solution

O(1) free-list pool that reclaims freed descriptors for reuse:

```
Before (linear allocation):
[0][1][2][3][4][5][6] → [7][8]...  (never reclaims slots 0-6)

After (free-list pool):
[0][1][2][3][4][5][6]
     ↓ Free slot 2
[0][1][ ][3][4][5][6] → freeList: [2]
     ↓ Allocate new
[0][1][X][3][4][5][6] → freeList: [] (reused slot 2!)
```

---

## Files Modified

### `src/utils/ResourceManager.h`

**Changes:**
- Added `#include <vector>` for free list
- Extended `DescriptorHeapInfo` struct with:
  - `std::vector<uint32_t> freeList` - Stack of freed descriptor indices
  - `uint32_t allocatedCount` - Currently active descriptors
  - `uint32_t totalAllocations` - Lifetime allocation count
  - `uint32_t totalFrees` - Lifetime free count
  - `uint32_t reuseCount` - Times free list provided a descriptor

**New Public Methods:**
```cpp
// Free a descriptor by CPU handle
void FreeDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle);

// Free a descriptor by index (direct)
void FreeDescriptorByIndex(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t index);

// Get index from CPU handle
uint32_t GetDescriptorIndex(D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle);

// Get usage statistics
struct DescriptorHeapStats {
    uint32_t totalDescriptors;    // Heap capacity
    uint32_t allocatedCount;      // Currently allocated (active)
    uint32_t freeListSize;        // Available in free list
    uint32_t highWaterMark;       // Maximum ever allocated
    uint32_t totalAllocations;    // Lifetime allocation count
    uint32_t totalFrees;          // Lifetime free count
    uint32_t reuseCount;          // Times free list was used
};
DescriptorHeapStats GetDescriptorStats(D3D12_DESCRIPTOR_HEAP_TYPE type) const;
```

### `src/utils/ResourceManager.cpp`

**Changes:**
- Added `#include <cstdint>` for `UINT32_MAX`
- Modified `AllocateDescriptor()` to check free list first
- Implemented 4 new methods with:
  - Double-free detection (warns and skips)
  - Bounds validation
  - Statistics tracking

---

## Usage Examples

### Freeing Descriptors During Buffer Recreation

```cpp
// When recreating a buffer with SRV/UAV, free the old descriptor first:
if (m_outputSRV.ptr != 0) {
    m_resources->FreeDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, m_outputSRV);
}
m_outputSRV = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
```

### Monitoring Heap Health (ImGui or Logging)

```cpp
auto stats = m_resources->GetDescriptorStats(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
LOG_INFO("Descriptors: {}/{} active, {} in free list, {} reused",
         stats.allocatedCount, stats.totalDescriptors,
         stats.freeListSize, stats.reuseCount);

// Or in ImGui:
ImGui::Text("Descriptors: %u/%u (free: %u, reused: %u)",
            stats.allocatedCount, stats.totalDescriptors,
            stats.freeListSize, stats.reuseCount);
```

### Batch Free (e.g., System Shutdown)

```cpp
// Free multiple descriptors
for (auto& handle : descriptorsToFree) {
    m_resources->FreeDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, handle);
}
```

---

## Integration Points

### DLSS Resize Path (`ParticleRenderer_Gaussian.cpp`) - INTEGRATED

The `SetDLSSSystem()` method recreates multiple buffers during DLSS quality mode changes.
All descriptors now call `FreeDescriptor()` before reallocating (10 total):

| Descriptor | Line | Status |
|------------|------|--------|
| `m_upscaledOutputSRV` | 1462-1465 | Integrated |
| `m_upscaledOutputUAV` | 1479-1483 | Integrated |
| `m_shadowSRV[i]` (x2) | 1592-1596 | Integrated |
| `m_shadowUAV[i]` (x2) | 1610-1614 | Integrated |
| `m_colorSRV[i]` (x2) | 1661-1665 | Integrated |
| `m_colorUAV[i]` (x2) | 1679-1683 | Integrated |
| `m_shadowDepthUAV` | 1731-1735 | Integrated |
| `m_shadowDepthSRV` | 1752-1756 | Integrated |

### Other Potential Integration Points

1. **NanoVDBSystem** - When animation frames are loaded/unloaded
2. **RTXDILightingSystem** - If temporal buffers are resized
3. **ProbeGridSystem** - If grid size is changed at runtime

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| AllocateDescriptor (from free list) | O(1) | Pop from vector back |
| AllocateDescriptor (fresh slot) | O(1) | Increment currentIndex |
| FreeDescriptor | O(n) | Double-free check scans free list |
| FreeDescriptorByIndex | O(n) | Double-free check scans free list |
| GetDescriptorIndex | O(1) | Pointer arithmetic |
| GetDescriptorStats | O(1) | Direct member access |

**Note:** The O(n) double-free check is a safety feature. For performance-critical code paths where double-free is impossible, you can skip it by directly manipulating the free list (not recommended).

---

## Safety Features

1. **Double-Free Detection:** Warns and skips if descriptor already in free list
2. **Bounds Validation:** Rejects indices beyond high water mark
3. **Null Handle Check:** Silently ignores null handles in FreeDescriptor
4. **Heap Type Validation:** Logs errors for unknown heap types

---

## Testing Recommendations

1. **Resize Stress Test:** Rapidly toggle DLSS quality modes
2. **Long Session Test:** Run for 30+ minutes with periodic resizes
3. **Monitor Stats:** Add ImGui panel to watch allocatedCount and reuseCount
4. **Memory Validation:** Run with D3D12 Debug Layer enabled

---

## Future Improvements

1. **Ring Buffer Upload Heap:** Similar pooling for upload buffer allocations
2. **Light Dirty Tracking:** Only upload changed lights
3. **PSO Disk Cache:** Serialize compiled PSOs to reduce cold start time
4. **Resource State Tracker:** Eliminate redundant barriers

---

## Related Files

- `src/utils/ResourceManager.h` - Header with new API
- `src/utils/ResourceManager.cpp` - Implementation
- `src/particles/ParticleRenderer_Gaussian.cpp` - Primary consumer (DLSS resize)
- `docs/DESCRIPTOR_HEAP_POOL_IMPLEMENTATION.md` - This document
