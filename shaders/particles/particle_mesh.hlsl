// DirectX 12 Mesh Shader for NASA-quality accretion disk particles
// Creates camera-facing billboards for each particle with temperature-based coloring

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

struct RenderConstants {
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float3 cameraPos;
    float particleSize;
    float temperatureScale;
    float colorTempOffset;  // Runtime color adjustment
    float colorTempScale;   // Runtime color scaling
    float padding;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float3 color : COLOR0;
    float alpha : COLOR1;
    float3 worldPos : TEXCOORD1;  // World position for shadow mapping
    float temperature : TEXCOORD2;  // Temperature for emission calculation (Mode 9.2)
    float3 lighting : TEXCOORD3;  // Particle-to-particle lighting (Mode 9.2)
};

// Mode 9 sub-mode flag for shadow map support
cbuffer ModeParams : register(b1) {
    uint mode9SubMode;  // 0=Baseline, 1+=Shadow modes
    float3 modePadding;
};

// Particle lighting structure (Mode 9.2+)
struct ParticleLighting {
    float4 color;  // rgb = additive lighting, w = unused
};

StructuredBuffer<Particle> particles : register(t0);
Texture2D<float> shadowMap : register(t1);  // Shadow map (Mode 9.1+)
StructuredBuffer<ParticleLighting> particleLighting : register(t2);  // Particle lighting (Mode 9.2+)
SamplerState shadowSampler : register(s0);
ConstantBuffer<RenderConstants> renderConstants : register(b0);

// Temperature to color mapping optimized for galaxy colors
// Reduces blue dominance and creates warmer, more galactic appearance
float3 TemperatureToColor(float temperature) {
    // Apply runtime adjustable offset and scale before normalization
    float adjustedTemp = (temperature + renderConstants.colorTempOffset) * renderConstants.colorTempScale;

    // Normalize temperature to 0-1 range (800K to 26000K)
    float t = saturate((adjustedTemp - 800.0) / 25200.0);

    // Galaxy-inspired color gradient:
    // Red (cool/outer) -> Orange -> Yellow -> White (hot/core)
    // This matches typical galaxy color distributions
    float3 color;

    if (t < 0.25) {
        // Deep red to orange-red (0.0 to 0.25) - outer galaxy regions
        float blend = t / 0.25;
        color = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
    } else if (t < 0.5) {
        // Orange-red to orange (0.25 to 0.5) - mid regions
        float blend = (t - 0.25) / 0.25;
        color = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
    } else if (t < 0.75) {
        // Orange to yellow-white (0.5 to 0.75) - inner regions
        float blend = (t - 0.5) / 0.25;
        color = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
    } else {
        // Yellow-white to pure white (0.75 to 1.0) - hot cores
        float blend = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
    }

    return color;
}

[NumThreads(32, 1, 1)]
[OutputTopology("triangle")]
void main(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    out vertices VertexOutput verts[128],    // 32 particles * 4 vertices = 128
    out indices uint3 tris[64]               // 32 particles * 2 triangles = 64
) {
    uint particleIndex = groupId.x * 32 + localId.x;
    uint vertexIndex = localId.x * 4;  // 4 vertices per particle
    uint triangleIndex = localId.x * 2; // 2 triangles per particle

    // Check if we have a valid particle
    if (particleIndex >= 100000) { // Hardcoded particle count for now
        return;
    }

    SetMeshOutputCounts(128, 64);

    Particle p = particles[particleIndex];

    // Mode 9.2+: ALWAYS apply lighting (no mode check - mesh shader can't read root constants!)
    // DIAGNOSTIC: Bypass mode check entirely
    float3 lighting = float3(0.0, 100.0, 0.0);  // Hardcoded green for ALL modes

    // Calculate camera-facing billboard vectors (matches Vulkan reference)
    float3 worldPos = p.position;
    float3 toCamera = renderConstants.cameraPos - worldPos;
    float3 forward = normalize(toCamera);
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);

    // Scale based on particle size with subtle temperature variation (hotter = slightly larger)
    float tempScale = saturate((p.temperature - 800.0) / 25200.0); // 0 to 1
    float scale = renderConstants.particleSize * (1.0 + tempScale * 0.2); // Only 20% size increase for hottest particles
    right *= scale;
    up *= scale;

    // Generate temperature-based color
    float3 color = TemperatureToColor(p.temperature);
    float alpha = saturate(p.density * 0.8); // Alpha based on density

    // Create 4 vertices for the billboard quad
    float4x4 viewProj = mul(renderConstants.viewMatrix, renderConstants.projMatrix);

    // Bottom-left
    float3 pos0 = worldPos - right - up;
    verts[vertexIndex + 0].position = mul(float4(pos0, 1.0), viewProj);
    verts[vertexIndex + 0].texCoord = float2(0.0, 1.0);
    verts[vertexIndex + 0].color = color;
    verts[vertexIndex + 0].alpha = alpha;
    verts[vertexIndex + 0].worldPos = worldPos;
    verts[vertexIndex + 0].temperature = p.temperature;
    verts[vertexIndex + 0].lighting = lighting;

    // Bottom-right
    float3 pos1 = worldPos + right - up;
    verts[vertexIndex + 1].position = mul(float4(pos1, 1.0), viewProj);
    verts[vertexIndex + 1].texCoord = float2(1.0, 1.0);
    verts[vertexIndex + 1].color = color;
    verts[vertexIndex + 1].alpha = alpha;
    verts[vertexIndex + 1].worldPos = worldPos;
    verts[vertexIndex + 1].temperature = p.temperature;
    verts[vertexIndex + 1].lighting = lighting;

    // Top-left
    float3 pos2 = worldPos - right + up;
    verts[vertexIndex + 2].position = mul(float4(pos2, 1.0), viewProj);
    verts[vertexIndex + 2].texCoord = float2(0.0, 0.0);
    verts[vertexIndex + 2].color = color;
    verts[vertexIndex + 2].alpha = alpha;
    verts[vertexIndex + 2].worldPos = worldPos;
    verts[vertexIndex + 2].temperature = p.temperature;
    verts[vertexIndex + 2].lighting = lighting;

    // Top-right
    float3 pos3 = worldPos + right + up;
    verts[vertexIndex + 3].position = mul(float4(pos3, 1.0), viewProj);
    verts[vertexIndex + 3].texCoord = float2(1.0, 0.0);
    verts[vertexIndex + 3].color = color;
    verts[vertexIndex + 3].alpha = alpha;
    verts[vertexIndex + 3].worldPos = worldPos;
    verts[vertexIndex + 3].temperature = p.temperature;
    verts[vertexIndex + 3].lighting = lighting;

    // Create 2 triangles for the quad
    // Triangle 1: bottom-left, bottom-right, top-left
    tris[triangleIndex + 0] = uint3(
        vertexIndex + 0,
        vertexIndex + 1,
        vertexIndex + 2
    );

    // Triangle 2: bottom-right, top-right, top-left
    tris[triangleIndex + 1] = uint3(
        vertexIndex + 1,
        vertexIndex + 3,
        vertexIndex + 2
    );
}

// Pixel shader output structure for Multiple Render Targets (Mode 9.2)
struct PSOutput {
    float4 color : SV_Target0;      // Particle color with lighting/shadows
    float4 emission : SV_Target1;   // Emission intensity for hot particles (Mode 9.2+)
};

// Pixel shader for particle rendering
PSOutput PSMain(VertexOutput input) {
    PSOutput output;
    // Create spherical particle shape using texture coordinates
    float2 center = input.texCoord - 0.5;
    float distFromCenter = length(center) * 2.0; // Scale to 0-1 range

    // Discard pixels outside circle for hard edge
    if (distFromCenter > 1.0) discard;

    // Simulate 3D sphere lighting with sqrt falloff (like a real sphere)
    float sphereZ = sqrt(max(0.0, 1.0 - distFromCenter * distFromCenter));

    // Smooth edge fadeout
    float edgeFade = 1.0 - smoothstep(0.8, 1.0, distFromCenter);

    // Combine sphere lighting with edge fade
    float intensity = sphereZ * edgeFade;
    float alpha = intensity * input.alpha;

    // Apply temperature-based color with 3D sphere lighting
    float3 color = input.color;

    // Add bright center (hot core)
    float hotSpot = pow(1.0 - distFromCenter, 3.0);
    color = lerp(color, color * 1.5, hotSpot * 0.5);

    // Apply sphere shading
    color *= (0.6 + intensity * 0.8);

    // Mode 9.1+: Apply DXR shadow map
    float shadowFactor = 1.0;
    if (mode9SubMode >= 1) {
        // Project particle world position to shadow map UV space
        // Shadow map covers -100 to +100 in XZ plane (orthographic)
        float2 shadowUV = (input.worldPos.xz + 100.0) / 200.0;
        shadowUV.y = 1.0 - shadowUV.y;  // Flip Y for D3D texture coordinates

        // Sample shadow map (1.0 = lit, 0.0 = occluded)
        if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0) {
            shadowFactor = shadowMap.SampleLevel(shadowSampler, shadowUV, 0);

            // Apply shadow darkening (20% brightness in shadow, 100% in light)
            // shadowFactor: 0.0 = occluded (in shadow), 1.0 = lit (no occlusion)
            color = lerp(color * 0.2, color, shadowFactor);
        }
    }

    // Calculate emission for Mode 9.2+
    float emissionStrength = 0.0;
    if (mode9SubMode >= 2 && input.temperature > 15000.0) {
        // Only hot particles emit light (>15000K threshold)
        float normalizedTemp = saturate((input.temperature - 15000.0) / 11000.0); // 15000K-26000K range
        emissionStrength = pow(normalizedTemp, 2.0) * 5.0; // Exponential falloff, scale factor 5.0
    }

    // Output to dual render targets
    output.color = float4(color, alpha);
    output.emission = float4(input.color * emissionStrength, emissionStrength);

    return output;
}