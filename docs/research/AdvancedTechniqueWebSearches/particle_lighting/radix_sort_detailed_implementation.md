# Radix Sort + Spatial Binning: Detailed Implementation Guide

## Source
- Primary: AMD FidelityFX Parallel Sort (https://gpuopen.com/fidelityfx-parallel-sort/)
- Reference: NVIDIA GPU Gems 3 Chapter 39 (Parallel Prefix Sum)
- Research: "Fast Data Parallel Radix Sort Implementation in DirectX 11 Compute Shader"

## Summary

This technique completely eliminates atomic operations by sorting particles spatially, enabling deterministic neighbor access. After sorting, particles in the same spatial region are consecutive in memory, allowing direct buffer reads/writes without synchronization.

**Key Innovation:** Combines Morton code (Z-order curve) spatial hashing with GPU radix sort to create a cache-coherent, atomic-free particle interaction system.

## Implementation Details

### Algorithm Overview

```
1. Compute Morton codes for each particle (3D position → 1D key)
2. Radix sort particles by Morton code using FidelityFX Parallel Sort
3. Build cell start/end index buffer using parallel prefix sum
4. Neighbor search: Read from sorted array using cell indices
5. Accumulate lighting via direct buffer writes to particle's own index
```

### Step 1: Morton Code Encoding (Compute Shader)

```hlsl
// Morton code: Interleave bits of 3D grid coordinates
// Result: Particles close in 3D space have similar Morton codes

cbuffer GridParams : register(b0)
{
    float3 gridMin;
    float3 gridMax;
    uint3 gridDimensions; // e.g., 128x128x128
    float cellSize;
};

StructuredBuffer<Particle> particles : register(t0);
RWStructuredBuffer<uint> mortonKeys : register(u0);
RWStructuredBuffer<uint> particleIndices : register(u1); // Original index (payload)

// Expand 10-bit integer into 30 bits by inserting 2 zeros after each bit
uint ExpandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Compute Morton code from 3D coordinates (each coordinate: 10 bits max)
uint Morton3D(uint3 coord)
{
    uint xx = ExpandBits(coord.x);
    uint yy = ExpandBits(coord.y);
    uint zz = ExpandBits(coord.z);
    return xx | (yy << 1) | (zz << 2);
}

[numthreads(256, 1, 1)]
void ComputeMortonCodes(uint3 DTid : SV_DispatchThreadID)
{
    uint particleID = DTid.x;
    if (particleID >= numParticles) return;

    Particle p = particles[particleID];

    // Quantize position to grid cell
    float3 normalized = (p.position - gridMin) / (gridMax - gridMin);
    uint3 gridCoord = clamp(uint3(normalized * gridDimensions), 0, gridDimensions - 1);

    // Encode to Morton code
    mortonKeys[particleID] = Morton3D(gridCoord);
    particleIndices[particleID] = particleID; // Store original index
}
```

### Step 2: Radix Sort Using FidelityFX

```cpp
// C++ host code (simplified integration)

#include "ffx_parallelsort.h"

// Setup (once at initialization)
FfxParallelSortContext sortContext;
FfxParallelSortContextDescription sortDesc = {};
sortDesc.maxEntries = 100000; // Max particles
sortDesc.backendInterface = ffxGetInterfaceDX12(); // DX12 backend

ffxParallelSortContextCreate(&sortContext, &sortDesc);

// Per-frame sorting
FfxParallelSortDispatchDescription dispatchDesc = {};
dispatchDesc.commandList = commandList;
dispatchDesc.keyBuffer = mortonKeysBuffer; // Input: unsorted Morton codes
dispatchDesc.payloadBuffer = particleIndicesBuffer; // Input: original indices
dispatchDesc.numKeys = currentParticleCount;

ffxParallelSortContextDispatch(&sortContext, &dispatchDesc);

// After dispatch:
// - mortonKeysBuffer contains sorted Morton codes
// - particleIndicesBuffer contains corresponding original particle indices
```

### Step 3: Build Cell Start/End Indices (Compute Shader)

```hlsl
// After sorting, identify where each grid cell begins/ends in sorted array

StructuredBuffer<uint> sortedMortonKeys : register(t0);
RWStructuredBuffer<uint2> cellRanges : register(u0); // x=start, y=count

[numthreads(256, 1, 1)]
void BuildCellRanges(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= numParticles) return;

    uint currentKey = sortedMortonKeys[i];
    uint prevKey = (i > 0) ? sortedMortonKeys[i - 1] : 0xFFFFFFFF;
    uint nextKey = (i < numParticles - 1) ? sortedMortonKeys[i + 1] : 0xFFFFFFFF;

    // Start of new cell?
    if (currentKey != prevKey) {
        cellRanges[currentKey].x = i; // Start index
    }

    // End of cell?
    if (currentKey != nextKey) {
        uint start = cellRanges[currentKey].x;
        cellRanges[currentKey].y = (i + 1) - start; // Count
    }
}
```

### Step 4: Neighbor Search and Lighting Accumulation (Compute Shader)

```hlsl
// Final pass: Read neighbors from sorted array, accumulate lighting

StructuredBuffer<uint> sortedMortonKeys : register(t0);
StructuredBuffer<uint> sortedIndices : register(t1); // Original particle IDs
StructuredBuffer<uint2> cellRanges : register(t2);
StructuredBuffer<Particle> particles : register(t3);
RWStructuredBuffer<float3> lightingAccum : register(u0); // Output per particle

[numthreads(256, 1, 1)]
void ComputeParticleLighting(uint3 DTid : SV_DispatchThreadID)
{
    uint sortedIndex = DTid.x;
    if (sortedIndex >= numParticles) return;

    uint particleID = sortedIndices[sortedIndex]; // Original particle ID
    Particle p = particles[particleID];

    // Get this particle's grid cell
    float3 normalized = (p.position - gridMin) / (gridMax - gridMin);
    uint3 gridCoord = clamp(uint3(normalized * gridDimensions), 0, gridDimensions - 1);

    float3 totalLight = float3(0, 0, 0);

    // Check 27 neighboring cells (3x3x3 around this particle)
    for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        int3 neighborCoord = int3(gridCoord) + int3(dx, dy, dz);

        // Bounds check
        if (any(neighborCoord < 0) || any(neighborCoord >= int3(gridDimensions)))
            continue;

        // Get Morton code for neighbor cell
        uint neighborMorton = Morton3D(uint3(neighborCoord));

        // Get range of particles in this cell
        uint2 range = cellRanges[neighborMorton];
        uint start = range.x;
        uint count = range.y;

        // Iterate over particles in neighbor cell
        for (uint i = 0; i < count; i++) {
            uint neighborSortedIndex = start + i;
            uint neighborID = sortedIndices[neighborSortedIndex];

            if (neighborID == particleID) continue; // Skip self

            Particle neighbor = particles[neighborID];

            // Distance check
            float dist = length(neighbor.position - p.position);
            if (dist > p.influenceRadius) continue;

            // Accumulate lighting (example: inverse square falloff)
            float3 emission = GetEmission(neighbor.temperature);
            float falloff = 1.0 / (1.0 + dist * dist);
            totalLight += emission * falloff;
        }
    }}}

    // DIRECT WRITE (no atomics needed - each thread writes to unique index)
    lightingAccum[particleID] = totalLight;
}
```

## Data Structures

### GPU Buffers
```cpp
// Input
StructuredBuffer<Particle> particles; // Original particle data (100k)

// Intermediate (Morton encoding)
Buffer<uint> mortonKeys;              // Morton codes (100k)
Buffer<uint> particleIndices;         // Original indices (100k)

// After sorting
Buffer<uint> sortedMortonKeys;        // Sorted Morton codes (100k)
Buffer<uint> sortedIndices;           // Sorted original indices (100k)

// Cell lookup
Buffer<uint2> cellRanges;             // Start/count per cell (max: 1M for 100^3 grid)

// Output
Buffer<float3> lightingAccum;         // Accumulated lighting per particle (100k)
```

### CPU Setup
```cpp
struct Particle {
    float3 position;
    float3 velocity;
    float temperature;
    float emissionIntensity;
    float influenceRadius;
    // ... other fields
};

// Grid parameters (tune based on particle density)
const uint3 gridDimensions = uint3(128, 128, 128); // 2M cells
const float cellSize = (diskRadius * 2.0f) / 128.0f; // Adjust to average influence radius
```

## Pipeline Integration

### Frame Structure
```cpp
void RenderFrame()
{
    // 1. Update particles (physics, temperature, etc.)
    UpdateParticlesComputeShader();

    // 2. Compute Morton codes
    ComputeMortonCodesComputeShader();

    // 3. Sort by Morton codes
    FfxParallelSortDispatch();

    // 4. Build cell ranges
    BuildCellRangesComputeShader();

    // 5. Compute lighting (neighbor search + accumulation)
    ComputeParticleLightingComputeShader();

    // 6. Render particles with lighting
    RenderParticles();
}
```

### Resource Transitions
```cpp
// After physics update
TransitionBarrier(particlesBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

// After Morton code computation
TransitionBarrier(mortonKeysBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

// After sort (FidelityFX handles internal barriers)
// No explicit barrier needed

// After cell range build
TransitionBarrier(cellRangesBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

// After lighting computation
TransitionBarrier(lightingAccumBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
```

## Performance Metrics

### Expected Timings (RTX 4060 Ti, 100k particles, 128^3 grid)
- **Morton encoding:** 0.1-0.2ms
- **Radix sort (FidelityFX):** 0.8-1.2ms
- **Cell range build:** 0.1-0.15ms
- **Neighbor search + lighting:** 1.0-2.0ms (depends on avg neighbors per particle)
- **Total:** 2.0-3.55ms

### Memory Usage
- Particle buffer: 100k * 64 bytes = 6.4 MB
- Morton keys: 100k * 4 bytes = 400 KB
- Sorted indices: 100k * 4 bytes = 400 KB
- Cell ranges: 2M * 8 bytes = 16 MB
- Lighting output: 100k * 12 bytes = 1.2 MB
- **Total:** ~24.4 MB

### Bottlenecks
- **Memory bandwidth:** Reordering 100k particles every frame
- **Sort complexity:** O(n log n), but well-optimized in FidelityFX
- **Neighbor search:** O(27 * avg_particles_per_cell) per particle

## Hardware Requirements

### Minimum GPU
- DirectX 12.0 with Shader Model 6.0
- Wave intrinsics support (WaveActivePrefixSum, etc.)
- 4GB VRAM (comfortable for 100k particles)

### Optimal GPU
- RDNA2+ or Ampere+ architecture
- Shader Model 6.6 (64-bit atomics if needed for extensions)
- 8GB+ VRAM (headroom for larger particle counts)

## Implementation Complexity

### Estimated Dev Time
- **Day 1-2:** Integrate FidelityFX SDK, setup buffers
- **Day 3:** Implement Morton encoding shader
- **Day 4:** Implement cell range build shader
- **Day 5:** Implement neighbor search + lighting shader
- **Total:** 5 days (assumes familiarity with DX12 compute)

### Risk Level: MEDIUM
**Risks:**
- FidelityFX SDK integration may have quirks
- Morton encoding needs careful bit manipulation
- Cell range build has edge cases (empty cells, boundary particles)
- Neighbor search can be cache-unfriendly if grid too sparse

**Mitigations:**
- Start with FidelityFX samples (ParallelSort sample included)
- Test Morton encoding with simple visualization
- Add debug output for cell ranges
- Profile with PIX to identify cache misses

### Dependencies
- AMD FidelityFX SDK (https://github.com/GPUOpen-Effects/FidelityFX-SDK)
- DirectX 12 Agility SDK (already in your project)
- Visual Studio 2019+ with C++17

## Related Techniques

### Alternatives to FidelityFX Sort
- **NVIDIA CUB RadixSort** (CUDA-based, requires porting to HLSL)
- **WaveActivePrefixSum-based sort** (simpler, but slower for 100k+ elements)
- **Bitonic sort** (easier to implement, but O(n log^2 n) complexity)

### Extensions
- **Temporal coherence:** Re-sort only when particles move > threshold
- **Multi-resolution grids:** Coarse grid for distant particles, fine for close-up
- **GPU BVH:** Replace uniform grid with bounding volume hierarchy (more complex)

## Notes for PlasmaDX Integration

### Specific Considerations
1. **Accretion disk shape:** Disk is thin (z-axis sparse) - use anisotropic grid (e.g., 256x256x32)
2. **Temperature-based emission:** Already computed in your shaders - reuse directly
3. **Existing particle system:** Integrate as additional compute pass after physics update
4. **Camera culling:** Sort only visible particles (reduce n from 100k to ~30-50k)

### Integration Points
- **After:** `ParticleUpdateComputeShader` (physics)
- **Before:** `ParticleRenderPass` (billboards/visualization)
- **Parallel with:** Any post-processing (independent compute queue)

### Optimization Opportunities
- **Async compute:** Run sort on compute queue while rendering previous frame
- **Indirect dispatch:** Use `DispatchIndirect` to skip empty cells
- **Half-precision:** Use `float16` for lighting accumulation (2x memory bandwidth)

## Failure Cases and Limitations

### When This Technique Struggles
1. **Highly dynamic scenes:** If particles teleport/respawn frequently, re-sorting overhead dominates
2. **Non-uniform density:** Sparse regions waste iterations in neighbor search
3. **Very large influence radius:** If radius > 3*cellSize, need to check more than 27 cells

### Mitigations
- **Spatial-temporal hybrid:** Re-sort every N frames, use previous frame's sorted order
- **Adaptive grid:** Split dense cells, merge sparse cells (adds complexity)
- **Radius culling:** Cap influence radius at 2*cellSize, fade lighting at boundary

## Testing and Validation

### Unit Tests
1. **Morton encoding:** Verify nearby 3D points have similar codes
2. **Sort correctness:** Check sortedMortonKeys are monotonically increasing
3. **Cell ranges:** Validate start indices don't overlap, counts sum to numParticles
4. **Neighbor search:** Compare against brute-force O(n^2) for small test case

### Profiling Checkpoints
- **PIX/NSight captures:** Identify shader bottlenecks
- **Occupancy metrics:** Ensure wave/thread utilization > 80%
- **Memory bandwidth:** Check if sort is memory-bound (likely)

### Debug Visualization
- **Color particles by Morton code:** Should show spatial clustering
- **Render grid cells:** Visualize particle distribution
- **Highlight neighbors:** Draw lines between particles in same cell

---

**Status:** [Production-Ready]
**Maturity:** Used in SPH fluid simulations, N-body physics, photon mapping
**Recommended:** Yes, for long-term robust solution after validating with texture splatting
