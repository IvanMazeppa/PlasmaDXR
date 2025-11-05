# Diagnostic Tests for 2045 Particle Crash
## RTX 4060 Ti (Ada Lovelace) - Hardware/Driver Bug Investigation

### Test A: Binary Search for Exact Threshold
Find the EXACT particle count where it breaks:
```cpp
// Test these specific counts:
2040 → Expected: Works
2041 → Expected: Works
2042 → Expected: Works
2043 → Expected: Works
2044 → Expected: Works (confirmed)
2045 → Expected: CRASH (confirmed)
2046 → Expected: CRASH
2047 → Expected: CRASH
2048 → Expected: CRASH (or works if 11-bit boundary theory)
2049 → Expected: CRASH
```

### Test B: Different BLAS Build Flags
Try different acceleration structure build preferences:
```cpp
// In BuildBLAS(), change:
blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;

// To:
blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
// Or:
blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
```

**Theory**: Different flags create different BVH structures, might avoid the bug.

### Test C: Multiple TLAS Instances (Same BLAS)
Instead of 1 instance with 2045 primitives, try:
```cpp
// Create 2 instances pointing to SAME BLAS
// Instance 0: Identity transform
// Instance 1: Small offset transform (0.001, 0, 0)
// Both reference the same BLAS with 2045 particles
```

**Theory**: Multiple instances might change TLAS traversal path.

### Test D: AABB Size Manipulation
Try different AABB generation strategies:
```cpp
// In generate_aabbs.hlsl:
// Option 1: Tighter bounds
aabbMin = particle.position - radius * 0.99;  // Slightly smaller
aabbMax = particle.position + radius * 0.99;

// Option 2: Looser bounds  
aabbMin = particle.position - radius * 1.01;  // Slightly larger
aabbMax = particle.position + radius * 1.01;

// Option 3: Axis-aligned offset
aabbMin = particle.position - radius + float3(0.001, 0, 0);
aabbMax = particle.position + radius + float3(0.001, 0, 0);
```

**Theory**: Different AABB sizes might change BVH construction.

### Test E: Primitive Reordering
Shuffle the primitives before building BLAS:
```cpp
// Generate AABBs in random order instead of linear
// Use reverse order (2044 → 0)
// Use morton code ordering (spatial coherence)
```

**Theory**: Primitive ordering affects BVH quality and depth.

### Test F: RayQuery Traversal Flags
Try different ray flags in the shader:
```cpp
// In update_probes.hlsl:
RayQuery<RAY_FLAG_NONE> q;  // Current

// Try:
RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
// Or:
RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;  // Should hit nothing
// Or:
RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> q;  // Shouldn't matter for AABBs
```

**Theory**: Different traversal flags might avoid the buggy code path.

### Test G: Scratch Buffer Size Override
Force larger scratch buffer allocation:
```cpp
// In BuildBLAS():
prebuildInfo.ScratchDataSizeInBytes *= 2;  // Double the scratch buffer
// Or:
prebuildInfo.ScratchDataSizeInBytes = (1 << 20);  // Force 1MB minimum
```

**Theory**: Driver might be underestimating scratch buffer needs.

### Test H: Add Dummy Triangles
Mix procedural primitives with triangles:
```cpp
// Add a single degenerate triangle to the BLAS
// This changes it from pure procedural to mixed geometry
D3D12_RAYTRACING_GEOMETRY_DESC triangleGeom = {};
triangleGeom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
// ... setup degenerate triangle at origin ...
```

**Theory**: Pure procedural BLAS might have different limits than mixed.

### Test I: Memory Barrier Paranoia
Add excessive barriers:
```cpp
// Before BLAS build:
cmdList->ResourceBarrier(barrier_uav_on_aabb);
cmdList->ResourceBarrier(barrier_uav_on_scratch);

// After BLAS build:
cmdList->ResourceBarrier(barrier_uav_on_blas);

// Before TLAS build:
cmdList->ResourceBarrier(barrier_uav_on_instance);

// After TLAS build:  
cmdList->ResourceBarrier(barrier_uav_on_tlas);

// Before RayQuery dispatch:
cmdList->ResourceBarrier(barrier_transition_tlas_to_srv);
```

**Theory**: Race condition in driver, excessive barriers might avoid it.

### Test J: Driver Downgrade Test
Try older NVIDIA drivers:
- 531.00 (minimum for DXR 1.1)
- 545.00 (before recent updates)
- 560.00 (before current 566.14)

**Theory**: Recent driver regression specific to Ada Lovelace.

## Data to Collect

For crash case (2045 particles), capture:
1. **Windows Event Log** - TDR timeout details
2. **NVIDIA Nsight Aftermath** - GPU crash dump (if enabled)
3. **PIX GPU Capture** - Up to crash point
4. **GPU-Z Sensors** - Memory/temperature at crash
5. **Driver Verifier** - Check for kernel violations

## Hardware-Specific Workarounds

### For Ada Lovelace (RTX 40-series) specifically:
```cpp
// Detect Ada Lovelace
if (IsAdaLovelaceGPU()) {
    // Option 1: Hard limit at 2044
    particleCount = min(particleCount, 2044);
    
    // Option 2: Use different build flags
    if (particleCount >= 2000) {
        blasFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
    }
    
    // Option 3: Force BLAS update instead of rebuild
    if (particleCount >= 2000 && previousBLAS) {
        blasInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
}
```

## Contact NVIDIA DevTech

Given this is likely a driver/hardware bug, consider:
1. Post on **NVIDIA Developer Forums** with this data
2. File bug report through **NVIDIA Developer Program**
3. Contact **NVIDIA DevTech** if you have access
4. Try **Intel Arc GPU** or **AMD RX 7000** to confirm it's NVIDIA-specific

## Alternative Renderer Architecture

If we can't fix the root cause, consider:
1. **Instanced Mesh Rendering** instead of ray tracing for particles
2. **Compute Shader Rasterization** (software renderer)
3. **Hybrid Approach**: RT for <2000 particles, raster for rest
4. **Level-of-Detail**: Only trace nearest 2000 particles
