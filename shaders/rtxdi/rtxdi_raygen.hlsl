// RTXDI Raygen Shader
// Milestone 4: Reservoir Sampling (Weighted Light Selection)
//
// Purpose: Sample the light grid and select ONE optimal light per pixel
//
// Phase: RESERVOIR SAMPLING
// - Select light from grid cell using weighted random selection
// - Store selected light data for Gaussian renderer
// - FIRST VISUAL TEST: Compare RTXDI vs brute-force multi-light

// Light structure (matches multi-light system: 32 bytes)
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
};

// Light grid cell structure (128 bytes)
struct LightGridCell {
    uint lightIndices[16];    // Which lights affect this cell (64 bytes)
    float lightWeights[16];   // Importance weights (64 bytes)
};

// Ray payload (16 bytes)
struct RayPayload {
    float3 debugColor;  // Debug visualization (cell coordinates)
    uint selectedLight; // Index of selected light
};

// Grid constants
cbuffer GridConstants : register(b0) {
    uint g_screenWidth;
    uint g_screenHeight;
    uint g_gridCellsX;      // 30
    uint g_gridCellsY;      // 30
    uint g_gridCellsZ;      // 30
    float g_worldMin;       // -1500.0f (actual grid bounds)
    float g_cellSize;       // 100.0f (actual cell size)
    uint g_frameIndex;      // For temporal random variation
};

// Resources
StructuredBuffer<LightGridCell> g_lightGrid : register(t0);  // Light grid
StructuredBuffer<Light> g_lights : register(t1);             // Light buffer
RWTexture2D<float4> g_output : register(u0);                 // Debug output

// ============================================================================
// Random Number Generation (PCG hash for deterministic per-pixel randomness)
// ============================================================================

uint PCGHash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float [0, 1) from pixel coordinate and frame index
float Random(uint2 pixelCoord, uint frameIndex) {
    uint seed = pixelCoord.y * g_screenWidth + pixelCoord.x;
    seed = PCGHash(seed + frameIndex * 1000);
    return float(seed) / 4294967296.0;  // 2^32
}

// ============================================================================
// Weighted Reservoir Sampling
// ============================================================================

// Select light from cell using weighted random selection
// Returns light index (0-15) or 0xFFFFFFFF if no lights in cell
uint SelectLightFromCell(LightGridCell cell, float randomValue) {
    // Calculate total weight
    float weightSum = 0.0;
    uint validLights = 0;

    for (uint i = 0; i < 16; i++) {
        if (cell.lightIndices[i] != 0xFFFFFFFF && cell.lightWeights[i] > 0.0) {
            weightSum += cell.lightWeights[i];
            validLights++;
        }
    }

    // No lights in cell - return invalid index
    if (validLights == 0 || weightSum <= 0.0) {
        return 0xFFFFFFFF;
    }

    // Weighted random selection
    float target = randomValue * weightSum;
    float accumulated = 0.0;

    for (uint i = 0; i < 16; i++) {
        if (cell.lightIndices[i] != 0xFFFFFFFF && cell.lightWeights[i] > 0.0) {
            accumulated += cell.lightWeights[i];
            if (accumulated >= target) {
                return cell.lightIndices[i];  // Return ACTUAL light index (not cell slot)
            }
        }
    }

    // Fallback: return first valid light (should never reach here)
    for (uint i = 0; i < 16; i++) {
        if (cell.lightIndices[i] != 0xFFFFFFFF) {
            return cell.lightIndices[i];
        }
    }

    return 0xFFFFFFFF;
}

// ============================================================================
// Grid Helper Functions
// ============================================================================

// Calculate 1D cell index from 3D coordinates
uint FlattenCellID(uint3 cellID) {
    return cellID.z * (g_gridCellsX * g_gridCellsY) + cellID.y * g_gridCellsX + cellID.x;
}

// Calculate world position from pixel coordinates
// SIMPLIFIED FIX (2025-10-19): Map to accretion disk plane at z=0
// This approximation works because:
// 1. Particles are in -300 to +300 range (disk)
// 2. Lights are in -300 to +300 range (RTXDI presets scaled down)
// 3. Grid cells at z=0 contain the relevant lights
float3 PixelToWorldPosition(uint2 pixelCoord) {
    // Normalize pixel to [0, 1]
    float2 uv = float2(pixelCoord) / float2(g_screenWidth, g_screenHeight);

    // Map to accretion disk plane at z=0 (-300 to +300)
    // This matches the particle distribution and RTXDI light positions
    float diskRange = 600.0f;  // Accretion disk spans 600 units (-300 to +300)
    float3 worldPos;
    worldPos.x = -300.0f + uv.x * diskRange;  // -300 to +300
    worldPos.y = -300.0f + (1.0 - uv.y) * diskRange;  // -300 to +300 (flip Y)
    worldPos.z = 0.0f;  // Disk plane

    // NOTE: This simple mapping ignores camera perspective, but works for our use case:
    // - Camera looks down at disk from above
    // - Pixels roughly map to disk positions in X/Y
    // - All relevant lights are in the -300 to +300 range at various Z heights
    // - Grid cells containing lights will be found via WorldToGridCell()

    return worldPos;
}

// Get grid cell containing world position
uint3 WorldToGridCell(float3 worldPos) {
    uint3 cellID;

    // Clamp to world bounds
    float3 clampedPos = clamp(worldPos, g_worldMin, g_worldMin + g_cellSize * float(g_gridCellsX));

    // Calculate cell coordinates
    cellID.x = uint((clampedPos.x - g_worldMin) / g_cellSize);
    cellID.y = uint((clampedPos.y - g_worldMin) / g_cellSize);
    cellID.z = uint((clampedPos.z - g_worldMin) / g_cellSize);

    // Clamp to grid dimensions
    cellID.x = min(cellID.x, g_gridCellsX - 1);
    cellID.y = min(cellID.y, g_gridCellsY - 1);
    cellID.z = min(cellID.z, g_gridCellsZ - 1);

    return cellID;
}

[shader("raygeneration")]
void RayGen() {
    // Get pixel coordinates
    uint2 pixelCoord = DispatchRaysIndex().xy;

    // Calculate world position from pixel
    float3 worldPos = PixelToWorldPosition(pixelCoord);

    // Determine which grid cell contains this position
    uint3 cellID = WorldToGridCell(worldPos);

    // Sample light grid
    uint flatCellIdx = FlattenCellID(cellID);
    LightGridCell cell = g_lightGrid[flatCellIdx];

    // ============================================================================
    // MILESTONE 4: Weighted Reservoir Sampling
    // ============================================================================

    // Generate random value for this pixel (varies per frame for temporal variation)
    float randomValue = Random(pixelCoord, g_frameIndex);

    // Select light from cell using weighted random selection
    uint selectedLightIndex = SelectLightFromCell(cell, randomValue);

    // Output format:
    // R: Selected light index (0-15) or 0xFFFFFFFF if no lights
    // G: Cell index (for debugging)
    // B: Number of lights in cell (for debugging)
    // A: Total weight sum (for debugging)

    float4 output;
    output.r = asfloat(selectedLightIndex);  // Store uint as float bits
    output.g = float(flatCellIdx);           // Cell index for debugging
    output.b = 0.0;                          // Reserved
    output.a = 1.0;                          // Alpha

    // Count lights in cell (for debugging)
    uint lightCount = 0;
    for (uint i = 0; i < 16; i++) {
        if (cell.lightIndices[i] != 0xFFFFFFFF && cell.lightWeights[i] > 0.0) {
            lightCount++;
        }
    }
    output.b = float(lightCount);

    // Write selected light data
    g_output[pixelCoord] = output;

    // ============================================================================
    // Next Step (Milestone 4 integration):
    // - Gaussian renderer reads this buffer
    // - Extracts selectedLightIndex from R channel (asuint)
    // - Looks up actual light data from g_lights buffer
    // - Uses THAT light instead of looping all 13 lights
    // ============================================================================
}
