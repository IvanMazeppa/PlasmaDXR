# DirectX 12 DXR 1.1 Particle Rendering Best Practices (2025)

## Target Hardware
- **GPU**: RTX 4060 Ti (34 RT cores, DXR Tier 1.1 support)
- **Architecture**: Ada Lovelace
- **Memory**: 8/16GB GDDR6 with 128-bit bus
- **L2 Cache**: 32MB (compensates for narrower bus)

## Executive Summary
Based on latest research (January 2025), optimal particle rendering on RTX 4060 Ti leverages:
1. **DrawIndexedInstanced** with structured buffers for 100K+ particles
2. **Inline raytracing (RayQuery)** for particle lighting/shadows
3. **Billboard vertex generation** in vertex shader (avoid geometry shader)
4. **Double-buffered GPU particle simulation** via compute shaders

## 1. Buffer Layout for 100K+ Particles

### Optimal Structured Buffer Design
```hlsl
// Particle data structure (16-byte aligned for optimal GPU access)
struct Particle {
    float3 position;     // 12 bytes
    float  lifetime;     // 4 bytes
    float3 velocity;     // 12 bytes
    float  size;         // 4 bytes
    float4 color;        // 16 bytes (RGBA)
    float2 textureIndex; // 8 bytes (atlas coords)
    float2 padding;      // 8 bytes (maintain 64-byte alignment)
}; // Total: 64 bytes per particle

// In HLSL
StructuredBuffer<Particle> ParticleData : register(t0);
```

### Buffer Creation Pattern (DirectX 12)
```cpp
// Create structured buffer for 100K particles
const UINT particleCount = 100000;
const UINT particleStride = sizeof(Particle); // 64 bytes

D3D12_RESOURCE_DESC bufferDesc = {};
bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
bufferDesc.Width = particleCount * particleStride;
bufferDesc.Height = 1;
bufferDesc.DepthOrArray = 1;
bufferDesc.MipLevels = 1;
bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
bufferDesc.SampleDesc.Count = 1;
bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

// Create in default heap for GPU-only access
device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    nullptr,
    IID_PPV_ARGS(&particleBuffer));

// Create SRV for vertex shader access
D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
srvDesc.Format = DXGI_FORMAT_UNKNOWN;
srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
srvDesc.Buffer.FirstElement = 0;
srvDesc.Buffer.NumElements = particleCount;
srvDesc.Buffer.StructureByteStride = particleStride;
srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
```

## 2. Billboard Vertex Generation Pattern

### Vertex Shader Implementation
```hlsl
// Optimized billboard vertex shader using DrawIndexedInstanced
struct VSInput {
    uint vertexID : SV_VertexID;
    uint instanceID : SV_InstanceID;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float4 color : COLOR0;
    float opacity : OPACITY;
};

cbuffer PerFrame : register(b0) {
    float4x4 viewProj;
    float3 cameraRight;
    float3 cameraUp;
    float3 cameraPos;
    float time;
}

PSInput VSMain(VSInput input) {
    PSInput output;

    // Fetch particle data using instance ID
    Particle p = ParticleData[input.instanceID];

    // Calculate particle age for fade-out
    float age = time - p.lifetime;
    float normalizedAge = saturate(age / MAX_LIFETIME);

    // Billboard quad corners (optimize with bit operations)
    const float2 corners[4] = {
        float2(-0.5f, -0.5f),
        float2( 0.5f, -0.5f),
        float2(-0.5f,  0.5f),
        float2( 0.5f,  0.5f)
    };

    // Generate billboard vertex
    float2 corner = corners[input.vertexID];
    float3 worldPos = p.position;
    worldPos += (cameraRight * corner.x + cameraUp * corner.y) * p.size;

    // Transform to clip space
    output.position = mul(float4(worldPos, 1.0f), viewProj);

    // Pass texture coordinates
    output.texCoord = corner + 0.5f; // Remap to [0,1]

    // Apply color and fade
    output.color = p.color;
    output.opacity = 1.0f - normalizedAge; // Fade out with age

    return output;
}
```

### DrawIndexedInstanced Call Pattern
```cpp
// Use a minimal index buffer for quad (6 indices)
const UINT indices[] = { 0, 1, 2, 2, 1, 3 };

// Draw all particles with single call
commandList->DrawIndexedInstanced(
    6,              // IndexCountPerInstance (quad)
    particleCount,  // InstanceCount (100K particles)
    0,              // StartIndexLocation
    0,              // BaseVertexLocation
    0               // StartInstanceLocation
);
```

## 3. RayQuery Performance Optimization

### Inline Raytracing for Particle Lighting
```hlsl
// Optimized RayQuery for particle shadow/occlusion
// Use in pixel shader or compute shader for particle lighting

[numthreads(64, 1, 1)]
void ComputeParticleLighting(uint3 tid : SV_DispatchThreadID) {
    uint particleID = tid.x;
    if (particleID >= particleCount) return;

    Particle p = ParticleData[particleID];

    // Configure RayQuery with template parameters for optimization
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;

    // Setup shadow ray (simplified)
    RayDesc ray;
    ray.Origin = p.position;
    ray.Direction = normalize(lightDirection);
    ray.TMin = 0.01f; // Bias to avoid self-intersection
    ray.TMax = lightDistance;

    // Trace ray (inline, no dynamic shader dispatch)
    query.TraceRayInline(
        SceneTLAS,      // Top-level acceleration structure
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xFF,           // Instance mask
        ray);

    // Process results efficiently
    query.Proceed();

    // Simple binary shadow
    float shadow = (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.3f : 1.0f;

    // Update particle lighting (write to UAV)
    ParticleLighting[particleID] = shadow;
}
```

### RayQuery Optimization Tips for RTX 4060 Ti
1. **Use RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH** for shadow rays
2. **Limit ray distance** to reduce BVH traversal cost
3. **Use template parameters** to enable compiler optimizations
4. **Batch rays in compute shader** for better GPU utilization
5. **Consider LOD system** - trace rays for nearby particles only

## 4. Common Pitfalls and Solutions

### Pitfall 1: Wrong Billboard Orientation
**Problem**: Billboards face wrong direction or appear flipped
**Solution**:
```hlsl
// Correct order of operations:
// 1. Apply particle position
// 2. Apply any particle rotation
// 3. THEN align to camera (billboard)
float3 worldPos = p.position; // Step 1
worldPos = RotateParticle(worldPos, p.rotation); // Step 2
worldPos += (cameraRight * corner.x + cameraUp * corner.y) * p.size; // Step 3
```

### Pitfall 2: Poor Performance with DrawInstanced
**Problem**: Using DrawInstanced instead of DrawIndexedInstanced
**Solution**: Always use DrawIndexedInstanced with shared index buffer
```cpp
// GOOD: Share index buffer across all particles
commandList->DrawIndexedInstanced(6, particleCount, 0, 0, 0);

// BAD: Individual draw calls or DrawInstanced without indices
for (int i = 0; i < particleCount; i++) {
    commandList->DrawInstanced(4, 1, 0, i); // DON'T DO THIS!
}
```

### Pitfall 3: Inefficient Buffer Updates
**Problem**: CPU-GPU sync stalls when updating particle data
**Solution**: Use double-buffering with compute shaders
```hlsl
// Double-buffer pattern for GPU particle simulation
RWStructuredBuffer<Particle> ParticlesCurrent : register(u0);
RWStructuredBuffer<Particle> ParticlesNext : register(u1);

[numthreads(256, 1, 1)]
void UpdateParticles(uint3 tid : SV_DispatchThreadID) {
    Particle p = ParticlesCurrent[tid.x];

    // Update physics
    p.velocity += gravity * deltaTime;
    p.position += p.velocity * deltaTime;
    p.lifetime += deltaTime;

    // Write to next buffer
    ParticlesNext[tid.x] = p;
}
// Swap buffers each frame on CPU side
```

### Pitfall 4: Resource State Mismanagement
**Problem**: DX12 resource state errors
**Solution**: Proper state transitions
```cpp
// Transition for compute update
commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
    particleBuffer.Get(),
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

// Update particles...

// Transition for rendering
commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
    particleBuffer.Get(),
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));
```

## 5. Performance Metrics and Expectations

### RTX 4060 Ti Performance Targets
- **100K particles @ 1080p**: 60+ FPS with basic shading
- **100K particles with RayQuery shadows**: 45-60 FPS
- **Memory usage**: ~6.4MB for particle data (64 bytes × 100K)
- **Draw call overhead**: <0.1ms for single DrawIndexedInstanced

### Optimization Hierarchy (Most to Least Impact)
1. **Use single DrawIndexedInstanced call** (10x improvement)
2. **GPU particle simulation** (5x improvement over CPU)
3. **Inline raytracing for shadows** (2x faster than dynamic shading)
4. **Proper buffer alignment** (15-20% improvement)
5. **Template RayQuery flags** (10-15% improvement)

## 6. Future-Proofing (DXR 1.2 Preview)

### Coming in April 2025
- **Opacity Micromaps (OMM)**: Up to 2.3x performance for alpha-tested particles
- **Shader Execution Reordering (SER)**: Up to 2x performance by reducing divergence
- **Cooperative Vectors**: 10x speedup for neural texture compression

### Preparation Steps
1. Structure code to easily integrate OMM for particle transparency
2. Design shaders to benefit from SER (group similar materials)
3. Consider neural rendering for particle LOD systems

## Implementation Checklist

- [ ] Create 64-byte aligned particle structure
- [ ] Implement structured buffer with proper SRV/UAV views
- [ ] Use DrawIndexedInstanced with shared 6-index quad buffer
- [ ] Generate billboards in vertex shader (avoid geometry shader)
- [ ] Implement double-buffered GPU particle update
- [ ] Add RayQuery for particle shadows/occlusion
- [ ] Profile with PIX for Windows
- [ ] Test with 10K, 50K, 100K particles for scaling
- [ ] Implement LOD system for distant particles
- [ ] Add frustum culling in compute shader

## References
- Microsoft DirectX Raytracing Specs (2025)
- NVIDIA RTX Best Practices Guide
- DirectX 12 Ultimate Features Documentation
- Intel Arc Graphics RT Developer Guide