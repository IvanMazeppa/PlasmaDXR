// Fixed mesh shader that reads from pre-merged particle buffer
// This WORKS AROUND the driver bug by reading from a single buffer

struct ParticleWithLighting {
    float3 position;
    float mass;
    float3 velocity;
    float temperature;
    float4 color;
    float3 rtLighting;  // Pre-merged RT lighting
    float pad;
};

struct Vertex {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
    float alpha : COLOR1;
};

struct Primitive {
    uint dummyField : DUMMY;
};

struct Constants {
    float4x4 viewProj;
    float3 cameraPos;
    float time;
    float3 cameraUp;
    float particleSize;
    float3 cameraRight;
    float pad;
};

// Single buffer access - this should work on buggy drivers!
StructuredBuffer<ParticleWithLighting> g_particlesWithLighting : register(t0);
ConstantBuffer<Constants> g_constants : register(b0);

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    out vertices Vertex verts[4],
    out primitives Primitive prims[2],
    out indices uint3 indices[2]
) {
    uint particleIndex = gid;

    // Read from single pre-merged buffer (works around driver bug)
    ParticleWithLighting p = g_particlesWithLighting[particleIndex];

    // Calculate billboard orientation
    float3 toCamera = normalize(g_constants.cameraPos - p.position);
    float3 right = normalize(cross(g_constants.cameraUp, toCamera));
    float3 up = cross(toCamera, right);

    // TEST: Make RT lighting visible as green
    float3 finalColor = p.color.rgb + p.rtLighting;

    // If RT lighting is working, particles should turn green
    // because we set rtLighting.g = 100.0 in the RT shader

    float size = g_constants.particleSize;

    // Generate 4 vertices for billboard
    float3 positions[4] = {
        p.position - right * size - up * size,
        p.position + right * size - up * size,
        p.position - right * size + up * size,
        p.position + right * size + up * size
    };

    float2 uvs[4] = {
        float2(0, 1),
        float2(1, 1),
        float2(0, 0),
        float2(1, 0)
    };

    // Output vertices
    SetMeshOutputCounts(4, 2);

    for (uint i = 0; i < 4; ++i) {
        verts[i].position = mul(float4(positions[i], 1.0), g_constants.viewProj);
        verts[i].texCoord = uvs[i];
        verts[i].color = float4(finalColor, 1.0);
        verts[i].alpha = p.color.a;
    }

    // Output primitives (triangles)
    prims[0].dummyField = 0;
    prims[1].dummyField = 0;

    // Output indices
    indices[0] = uint3(0, 1, 2);
    indices[1] = uint3(1, 3, 2);
}