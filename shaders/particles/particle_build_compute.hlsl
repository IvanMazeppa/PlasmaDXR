// Compute shader to build particle vertex buffer with RT lighting
// Replaces mesh shader to avoid compatibility issues

struct Particle {
    float3 position;
    float temperature;
    float3 velocity;
    float density;
};

struct ParticleLighting {
    float4 color;  // rgb = additive lighting, w = unused
};

struct ParticleVertex {
    float4 position;      // Clip space position
    float2 texCoord;      // UV for billboard
    float4 color;         // Base color + lighting (pre-multiplied)
    float alpha;          // Particle alpha
};

struct RenderConstants {
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float3 cameraPos;
    float particleSize;
    float temperatureScale;
    float colorTempOffset;
    float colorTempScale;
    float padding;
};

cbuffer BuildParams : register(b0) {
    uint particleCount;
    uint mode9SubMode;  // 0=Baseline, 2=ParticleRelight
    float2 buildPadding;
};

StructuredBuffer<Particle> particles : register(t0);
StructuredBuffer<ParticleLighting> particleLighting : register(t1);
ConstantBuffer<RenderConstants> renderConstants : register(b1);

RWStructuredBuffer<ParticleVertex> outputVertices : register(u0);

// Temperature to color mapping
float3 TemperatureToColor(float temperature) {
    float adjustedTemp = (temperature + renderConstants.colorTempOffset) * renderConstants.colorTempScale;
    float t = saturate((adjustedTemp - 800.0) / 25200.0);

    float3 color;
    if (t < 0.25) {
        float blend = t / 0.25;
        color = lerp(float3(0.5, 0.1, 0.05), float3(1.0, 0.3, 0.1), blend);
    } else if (t < 0.5) {
        float blend = (t - 0.25) / 0.25;
        color = lerp(float3(1.0, 0.3, 0.1), float3(1.0, 0.6, 0.2), blend);
    } else if (t < 0.75) {
        float blend = (t - 0.5) / 0.25;
        color = lerp(float3(1.0, 0.6, 0.2), float3(1.0, 0.95, 0.7), blend);
    } else {
        float blend = (t - 0.75) / 0.25;
        color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), blend);
    }
    return color;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIndex = dispatchThreadID.x;

    if (particleIndex >= particleCount)
        return;

    Particle p = particles[particleIndex];

    // Read RT lighting (Mode 9.2+)
    float3 lighting = float3(0.0, 0.0, 0.0);
    if (mode9SubMode >= 2) {
        lighting = particleLighting[particleIndex].color.rgb;
    }

    // Base color from temperature
    float3 baseColor = TemperatureToColor(p.temperature);

    // FINAL COLOR: base + RT lighting
    float3 finalColor = baseColor + lighting;

    // Alpha from density
    float alpha = saturate(p.density * 0.8);

    // Billboard vectors (camera-facing)
    float3 worldPos = p.position;
    float3 toCamera = renderConstants.cameraPos - worldPos;
    float3 forward = normalize(toCamera);
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);

    float tempScale = saturate((p.temperature - 800.0) / 25200.0);
    float scale = renderConstants.particleSize * (1.0 + tempScale * 0.2);
    right *= scale;
    up *= scale;

    // Build 4 vertices for billboard quad
    float4x4 viewProj = mul(renderConstants.viewMatrix, renderConstants.projMatrix);
    uint vertexBase = particleIndex * 4;

    // Bottom-left
    float3 pos0 = worldPos - right - up;
    outputVertices[vertexBase + 0].position = mul(float4(pos0, 1.0), viewProj);
    outputVertices[vertexBase + 0].texCoord = float2(0.0, 1.0);
    outputVertices[vertexBase + 0].color = float4(finalColor, 1.0);
    outputVertices[vertexBase + 0].alpha = alpha;

    // Bottom-right
    float3 pos1 = worldPos + right - up;
    outputVertices[vertexBase + 1].position = mul(float4(pos1, 1.0), viewProj);
    outputVertices[vertexBase + 1].texCoord = float2(1.0, 1.0);
    outputVertices[vertexBase + 1].color = float4(finalColor, 1.0);
    outputVertices[vertexBase + 1].alpha = alpha;

    // Top-right
    float3 pos2 = worldPos + right + up;
    outputVertices[vertexBase + 2].position = mul(float4(pos2, 1.0), viewProj);
    outputVertices[vertexBase + 2].texCoord = float2(1.0, 0.0);
    outputVertices[vertexBase + 2].color = float4(finalColor, 1.0);
    outputVertices[vertexBase + 2].alpha = alpha;

    // Top-left
    float3 pos3 = worldPos - right + up;
    outputVertices[vertexBase + 3].position = mul(float4(pos3, 1.0), viewProj);
    outputVertices[vertexBase + 3].texCoord = float2(0.0, 0.0);
    outputVertices[vertexBase + 3].color = float4(finalColor, 1.0);
    outputVertices[vertexBase + 3].alpha = alpha;
}
