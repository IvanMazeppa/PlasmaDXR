// RTXDI Raygen Shader
// Milestone 3: Light Grid Sampling Infrastructure
//
// Purpose: Sample the light grid to determine which lights affect each pixel
//
// Current Phase: DEBUG VISUALIZATION
// - Output cell index as RGB color
// - Validate light grid sampling works correctly
// - NO visual impact on main renderer yet (that's Milestone 4)

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
    float g_worldMin;       // -300.0f
    float g_cellSize;       // 20.0f
    uint g_padding;
};

// Resources
StructuredBuffer<LightGridCell> g_lightGrid : register(t0);  // Light grid
StructuredBuffer<Light> g_lights : register(t1);             // Light buffer
RWTexture2D<float4> g_output : register(u0);                 // Debug output

// Calculate 1D cell index from 3D coordinates
uint FlattenCellID(uint3 cellID) {
    return cellID.z * (g_gridCellsX * g_gridCellsY) + cellID.y * g_gridCellsX + cellID.x;
}

// Calculate world position from pixel coordinates
// For now, use simple mapping to disk plane (z = 0)
// TODO Milestone 4: Use camera matrices for proper 3D projection
float3 PixelToWorldPosition(uint2 pixelCoord) {
    // Normalize pixel to [0, 1]
    float2 uv = float2(pixelCoord) / float2(g_screenWidth, g_screenHeight);

    // Map to world space disk plane (-300 to +300)
    float worldRange = 600.0f; // g_worldMax - g_worldMin
    float3 worldPos;
    worldPos.x = g_worldMin + uv.x * worldRange;
    worldPos.y = g_worldMin + (1.0 - uv.y) * worldRange; // Flip Y
    worldPos.z = 0.0f; // Disk plane

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

    // === DEBUG VISUALIZATION ===
    // Output cell index as RGB color
    // R = cellX / 30, G = cellY / 30, B = cellZ / 30
    float3 debugColor;
    debugColor.r = float(cellID.x) / float(g_gridCellsX);
    debugColor.g = float(cellID.y) / float(g_gridCellsY);
    debugColor.b = float(cellID.z) / float(g_gridCellsZ);

    // For populated cells, visualize light count
    uint lightCount = 0;
    for (uint i = 0; i < 16; i++) {
        if (cell.lightWeights[i] > 0.0) {
            lightCount++;
        }
    }

    // Modulate brightness by light count (0-16 lights)
    float brightness = float(lightCount) / 16.0f;
    debugColor *= 0.5 + 0.5 * brightness; // Range: [0.5, 1.0]

    // Write debug output
    g_output[pixelCoord] = float4(debugColor, 1.0);

    // === MILESTONE 4 TODO ===
    // - Select light from cell using RTXDI sampling
    // - Store selected light in output buffer
    // - Pass to Gaussian renderer for lighting calculation
}
