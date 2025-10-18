// RTXDI Light Grid Build Compute Shader
// Populates a spatial acceleration structure (light grid) for efficient many-light sampling
//
// Grid Structure:
// - 30x30x30 cells (27,000 total)
// - Cell size: 20 units
// - World bounds: -300 to +300 on all axes
// - Max 16 lights per cell
//
// Algorithm:
// - Each thread processes one grid cell
// - Tests all lights for intersection with cell AABB
// - Stores light indices and weights for lights affecting the cell
// - Sorts by weight (brightest first) if >16 lights

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

// Grid constants
cbuffer GridConstants : register(b0) {
    uint g_gridCellsX;          // 30
    uint g_gridCellsY;          // 30
    uint g_gridCellsZ;          // 30
    uint g_lightCount;          // Number of active lights (0-16)

    float g_worldMin;           // -300.0f
    float g_worldMax;           // 300.0f
    float g_cellSize;           // 20.0f
    uint g_maxLightsPerCell;    // 16
};

// Input: Current lights
StructuredBuffer<Light> g_lights : register(t0);

// Output: Light grid
RWStructuredBuffer<LightGridCell> g_lightGrid : register(u0);

// Calculate 1D cell index from 3D coordinates
uint FlattenCellID(uint3 cellID) {
    return cellID.z * (g_gridCellsX * g_gridCellsY) + cellID.y * g_gridCellsX + cellID.x;
}

// Calculate cell's world-space AABB bounds
void GetCellBounds(uint3 cellID, out float3 cellMin, out float3 cellMax) {
    cellMin = float3(
        g_worldMin + float(cellID.x) * g_cellSize,
        g_worldMin + float(cellID.y) * g_cellSize,
        g_worldMin + float(cellID.z) * g_cellSize
    );
    cellMax = cellMin + g_cellSize;
}

// Test if light affects this cell (sphere-AABB intersection)
bool LightIntersectsCell(Light light, float3 cellMin, float3 cellMax) {
    // Find closest point on AABB to sphere center
    float3 closestPoint = clamp(light.position, cellMin, cellMax);

    // Distance from sphere center to closest point
    float distSq = dot(light.position - closestPoint, light.position - closestPoint);

    // Intersects if distance < radius
    // Add small epsilon to avoid missing lights on cell boundaries
    float radiusSq = light.radius * light.radius;
    return distSq <= (radiusSq + 1.0);
}

// Calculate importance weight for light at cell center
float CalculateLightWeight(Light light, float3 cellCenter) {
    float dist = length(light.position - cellCenter);

    // Attenuation (1/d^2 with small offset to avoid infinity)
    float attenuation = 1.0 / (1.0 + dist * dist * 0.01);

    // Weight = luminance * attenuation
    // Luminance approximation: 0.2126*R + 0.7152*G + 0.0722*B
    float luminance = dot(light.color, float3(0.2126, 0.7152, 0.0722));

    return luminance * light.intensity * attenuation;
}

// Insertion sort for weight-based prioritization (max 16 elements, very efficient)
void SortLightsByWeight(inout uint indices[16], inout float weights[16], uint count) {
    for (uint i = 1; i < count; i++) {
        uint keyIdx = indices[i];
        float keyWeight = weights[i];
        int j = int(i) - 1;

        // Move elements greater than key to one position ahead
        while (j >= 0 && weights[j] < keyWeight) {
            indices[j + 1] = indices[j];
            weights[j + 1] = weights[j];
            j--;
        }

        indices[j + 1] = keyIdx;
        weights[j + 1] = keyWeight;
    }
}

[numthreads(8, 8, 8)]
void main(uint3 cellID : SV_DispatchThreadID) {
    // Bounds check (30x30x30 grid)
    if (cellID.x >= g_gridCellsX || cellID.y >= g_gridCellsY || cellID.z >= g_gridCellsZ) {
        return;
    }

    // Calculate cell bounds
    float3 cellMin, cellMax;
    GetCellBounds(cellID, cellMin, cellMax);
    float3 cellCenter = (cellMin + cellMax) * 0.5;

    // Temporary storage for lights affecting this cell
    uint localIndices[16];
    float localWeights[16];
    uint localCount = 0;

    // Initialize arrays
    for (uint i = 0; i < 16; i++) {
        localIndices[i] = 0;
        localWeights[i] = 0.0;
    }

    // Test all lights for intersection with this cell
    for (uint lightIdx = 0; lightIdx < g_lightCount; lightIdx++) {
        Light light = g_lights[lightIdx];

        // Test sphere-AABB intersection
        if (LightIntersectsCell(light, cellMin, cellMax)) {
            // Calculate importance weight
            float weight = CalculateLightWeight(light, cellCenter);

            if (localCount < g_maxLightsPerCell) {
                // Add to list
                localIndices[localCount] = lightIdx;
                localWeights[localCount] = weight;
                localCount++;
            } else {
                // Find minimum weight in current list
                uint minIdx = 0;
                float minWeight = localWeights[0];
                for (uint j = 1; j < g_maxLightsPerCell; j++) {
                    if (localWeights[j] < minWeight) {
                        minWeight = localWeights[j];
                        minIdx = j;
                    }
                }

                // Replace if this light is more important
                if (weight > minWeight) {
                    localIndices[minIdx] = lightIdx;
                    localWeights[minIdx] = weight;
                }
            }
        }
    }

    // Sort lights by weight (brightest first)
    if (localCount > 1) {
        SortLightsByWeight(localIndices, localWeights, localCount);
    }

    // Write to output light grid
    uint flatCellIdx = FlattenCellID(cellID);
    LightGridCell outputCell;

    for (uint i = 0; i < 16; i++) {
        outputCell.lightIndices[i] = localIndices[i];
        outputCell.lightWeights[i] = localWeights[i];
    }

    g_lightGrid[flatCellIdx] = outputCell;
}
