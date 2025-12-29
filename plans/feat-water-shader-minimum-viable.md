# Minimum Viable Water Shader with RT Lighting

## Overview

Add support for rendering water meshes (from Blender liquid simulations) with RT lighting integration. Extends the existing DXR 1.1 RayQuery infrastructure—no new class hierarchies.

**Estimated Effort:** 6-9 hours
**Prerequisites:** Existing DXR 1.1 RayQuery infrastructure, TLAS/BLAS building

---

## Design Principles

1. **Extend, don't duplicate** - Reuse ground plane BLAS pattern from `RTLightingSystem_RayQuery.cpp:1384-1481`
2. **Static first, animate later** - Prove RT integration works before adding animation
3. **Simple file format** - Raw binary export from Blender, no gzip/parsing complexity
4. **Single-bounce optics** - 1 reflection + 1 refraction ray, no recursion

---

## Phase 1: Static Water Mesh (3 hours)

### 1.1 Blender Export Script

**File:** `scripts/export_water_mesh.py`

Export mesh as raw binary (no parsing needed in C++):

```python
import bpy
import struct

def export_water_frame(filepath, obj):
    """Export mesh as raw binary: header + vertices + indices"""
    mesh = obj.data
    mesh.calc_loop_triangles()

    vertices = []
    for v in mesh.vertices:
        vertices.extend([v.co.x, v.co.y, v.co.z])
        vertices.extend([v.normal.x, v.normal.y, v.normal.z])

    indices = []
    for tri in mesh.loop_triangles:
        indices.extend(tri.vertices)

    with open(filepath, 'wb') as f:
        # Header: vertex count, index count
        f.write(struct.pack('II', len(mesh.vertices), len(indices)))
        # Vertices: interleaved position + normal (6 floats each)
        f.write(struct.pack(f'{len(vertices)}f', *vertices))
        # Indices: uint32
        f.write(struct.pack(f'{len(indices)}I', *indices))

# Usage: export_water_frame("water_mesh.bin", bpy.context.active_object)
```

### 1.2 Mesh Loading in Application

**Modified:** `src/core/Application.h`

```cpp
// Add to Application class (no new classes needed)
struct WaterMeshData {
    std::vector<float> vertices;  // Interleaved pos + normal (6 floats per vertex)
    std::vector<uint32_t> indices;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
};

WaterMeshData m_waterMesh;
ComPtr<ID3D12Resource> m_waterVertexBuffer;
ComPtr<ID3D12Resource> m_waterIndexBuffer;
bool m_waterEnabled = false;
```

**Modified:** `src/core/Application.cpp`

```cpp
void Application::LoadWaterMesh(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return;

    // Read header
    file.read(reinterpret_cast<char*>(&m_waterMesh.vertexCount), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&m_waterMesh.indexCount), sizeof(uint32_t));

    // Read vertices (6 floats per vertex: pos + normal)
    m_waterMesh.vertices.resize(m_waterMesh.vertexCount * 6);
    file.read(reinterpret_cast<char*>(m_waterMesh.vertices.data()),
              m_waterMesh.vertices.size() * sizeof(float));

    // Read indices
    m_waterMesh.indices.resize(m_waterMesh.indexCount);
    file.read(reinterpret_cast<char*>(m_waterMesh.indices.data()),
              m_waterMesh.indices.size() * sizeof(uint32_t));

    // Upload to GPU buffers (same pattern as particle buffers)
    CreateWaterGPUBuffers();
    m_waterEnabled = true;
}
```

### 1.3 Triangle BLAS (Reuse Ground Plane Pattern)

**Modified:** `src/lighting/RTLightingSystem_RayQuery.cpp`

Follow existing ground plane BLAS pattern (lines 1384-1481):

```cpp
void RTLightingSystem_RayQuery::AddWaterMesh(
    ID3D12Resource* vertexBuffer,
    ID3D12Resource* indexBuffer,
    uint32_t vertexCount,
    uint32_t indexCount) {

    // Triangle geometry desc (same as ground plane)
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometryDesc.Triangles.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress();
    geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(float) * 6;  // pos + normal
    geometryDesc.Triangles.VertexCount = vertexCount;
    geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometryDesc.Triangles.IndexBuffer = indexBuffer->GetGPUVirtualAddress();
    geometryDesc.Triangles.IndexCount = indexCount;
    geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

    // Build BLAS (copy ground plane pattern)
    // ... (existing BLAS build code)

    m_hasWaterMesh = true;
}
```

### 1.4 TLAS Instance

Add water instance alongside particles:

```cpp
// In BuildTLAS()
if (m_hasWaterMesh) {
    D3D12_RAYTRACING_INSTANCE_DESC waterInstance = {};
    waterInstance.InstanceID = INSTANCE_ID_WATER;  // Define as 2 (0=particles, 1=ground)
    waterInstance.InstanceMask = 0xFF;
    waterInstance.InstanceContributionToHitGroupIndex = 0;
    waterInstance.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    SetIdentityTransform(waterInstance.Transform);
    waterInstance.AccelerationStructure = m_waterBLAS->GetGPUVirtualAddress();
    instances.push_back(waterInstance);
}
```

---

## Phase 2: Water Shader (3-4 hours)

### 2.1 Water Material Constants

**Modified:** `shaders/dxr/common.hlsl`

```hlsl
// Instance IDs
static const uint INSTANCE_ID_PARTICLES = 0;
static const uint INSTANCE_ID_GROUND = 1;
static const uint INSTANCE_ID_WATER = 2;

// Water material (hardcoded for MVP)
static const float WATER_IOR = 1.33;
static const float3 WATER_COLOR = float3(0.8, 0.9, 1.0);
static const float3 WATER_ABSORPTION = float3(0.45, 0.09, 0.06);
```

### 2.2 Water Shading Functions

**File:** `shaders/materials/water_material.hlsl`

```hlsl
#include "../dxr/common.hlsl"

// Fresnel-Schlick approximation
float FresnelSchlick(float cosTheta, float ior) {
    float r0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    return r0 + (1.0 - r0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Snell's law refraction (returns reflection if total internal reflection)
float3 RefractOrReflect(float3 incident, float3 normal, float eta) {
    float cosI = dot(-incident, normal);
    float sin2T = eta * eta * (1.0 - cosI * cosI);

    if (sin2T > 1.0) {
        return reflect(incident, normal);  // Total internal reflection
    }

    float cosT = sqrt(1.0 - sin2T);
    return eta * incident + (eta * cosI - cosT) * normal;
}

// Single-bounce reflection ray
float3 TraceReflection(float3 origin, float3 direction,
                       RaytracingAccelerationStructure tlas) {
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin + direction * 0.01;  // Offset to avoid self-hit
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 10000.0;

    q.TraceRayInline(tlas, RAY_FLAG_NONE,
                     ~(1 << INSTANCE_ID_WATER),  // Exclude water from reflection
                     ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Hit scene geometry - return simple shaded color
        // (Integrate with existing lighting later)
        return float3(0.3, 0.3, 0.35);
    }

    // Sky/environment
    return lerp(float3(0.5, 0.7, 1.0), float3(0.1, 0.2, 0.4),
                saturate(direction.y * 0.5 + 0.5));
}

// Single-bounce refraction ray with absorption
float3 TraceRefraction(float3 origin, float3 direction,
                       RaytracingAccelerationStructure tlas) {
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin + direction * 0.01;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 10000.0;

    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 hitColor = float3(0, 0, 0);
    float hitDistance = ray.TMax;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        hitDistance = q.CommittedRayT();
        hitColor = float3(0.2, 0.2, 0.2);  // Scene geometry
    } else {
        hitColor = float3(0.1, 0.15, 0.2);  // Environment through water
    }

    // Beer-Lambert absorption
    float3 absorption = exp(-WATER_ABSORPTION * hitDistance * 0.1);
    return hitColor * absorption * WATER_COLOR;
}

// Main water shading (called from particle_gaussian_raytrace.hlsl)
float3 ShadeWater(float3 hitPos, float3 normal, float3 viewDir,
                  RaytracingAccelerationStructure tlas) {

    // Ensure normal faces viewer
    float3 N = dot(normal, viewDir) > 0 ? normal : -normal;
    float NdotV = saturate(dot(N, viewDir));

    // Fresnel determines reflection vs refraction ratio
    float fresnel = FresnelSchlick(NdotV, WATER_IOR);

    // Reflection (single bounce)
    float3 reflectDir = reflect(-viewDir, N);
    float3 reflection = TraceReflection(hitPos, reflectDir, tlas);

    // Refraction (single bounce)
    float eta = 1.0 / WATER_IOR;  // Air to water
    float3 refractDir = RefractOrReflect(-viewDir, N, eta);
    float3 refraction = TraceRefraction(hitPos, refractDir, tlas);

    // Combine
    return lerp(refraction, reflection, fresnel);
}
```

### 2.3 Integration with Main Renderer

**Modified:** `shaders/particles/particle_gaussian_raytrace.hlsl`

Add water hit handling in main ray loop:

```hlsl
#include "../materials/water_material.hlsl"

// In main ray march loop, after RayQuery proceeds:
if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    uint instanceID = q.CommittedInstanceID();

    if (instanceID == INSTANCE_ID_WATER) {
        // Water hit - shade with reflection/refraction
        float3 hitPos = rayOrigin + rayDir * q.CommittedRayT();
        float3 hitNormal = GetTriangleNormal(q);  // Extract from barycentrics
        float3 waterColor = ShadeWater(hitPos, hitNormal, -rayDir, g_SceneTLAS);

        // Blend with existing accumulation
        accumulatedColor += waterColor * (1.0 - accumulatedAlpha);
        accumulatedAlpha = 1.0;
    }
    // ... existing particle/ground handling
}
```

---

## Phase 3: ImGui Controls (30 min)

**Modified:** `src/core/Application.cpp`

```cpp
// In RenderUI()
if (ImGui::CollapsingHeader("Water Mesh")) {
    ImGui::Checkbox("Enabled", &m_waterEnabled);

    if (ImGui::Button("Load Mesh...")) {
        // Simple file dialog or hardcoded test path
        LoadWaterMesh("assets/water/water_mesh.bin");
    }

    ImGui::Text("Vertices: %u", m_waterMesh.vertexCount);
    ImGui::Text("Triangles: %u", m_waterMesh.indexCount / 3);
}
```

---

## Phase 4: Animation (Optional, 2 hours)

Only implement after static mesh works:

```cpp
// Simple frame array (no interpolation, no double-buffering)
struct WaterAnimation {
    std::vector<WaterMeshData> frames;
    int currentFrame = 0;
    float playbackSpeed = 1.0f;
    float frameAccumulator = 0.0f;
};

void Application::UpdateWaterAnimation(float deltaTime) {
    if (m_waterAnimation.frames.empty()) return;

    m_waterAnimation.frameAccumulator += deltaTime * m_waterAnimation.playbackSpeed * 24.0f;

    if (m_waterAnimation.frameAccumulator >= 1.0f) {
        m_waterAnimation.frameAccumulator -= 1.0f;
        m_waterAnimation.currentFrame =
            (m_waterAnimation.currentFrame + 1) % m_waterAnimation.frames.size();

        // Rebuild BLAS with new frame (same as particle BLAS rebuild pattern)
        RebuildWaterBLAS(m_waterAnimation.currentFrame);
    }
}
```

---

## Render Pipeline

```
Existing:
  Particles → RT Lighting → DLSS → Present

With Water:
  Particles + Water Mesh → RT Lighting (unified TLAS) → DLSS → Present

TLAS Contents:
  Instance 0: Particle BLAS (procedural AABBs)
  Instance 1: Ground BLAS (triangles) - if present
  Instance 2: Water Mesh BLAS (triangles)
```

---

## Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| **M1** | Static mesh visible | Water mesh renders (solid color, no shading) |
| **M2** | Fresnel reflection | See environment reflected on water surface |
| **M3** | Refraction + absorption | See through water with blue tint at depth |
| **M4** | Animation (optional) | Water animates frame-by-frame |

---

## Files Summary

### New Files (2)
```
scripts/export_water_mesh.py          - Blender export script
shaders/materials/water_material.hlsl - Water shading functions
```

### Modified Files (4)
```
src/core/Application.h                  - WaterMeshData struct, buffers
src/core/Application.cpp                - LoadWaterMesh(), ImGui, animation
src/lighting/RTLightingSystem_RayQuery.h   - AddWaterMesh(), m_waterBLAS
src/lighting/RTLightingSystem_RayQuery.cpp - Water BLAS build (copy ground plane)
shaders/particles/particle_gaussian_raytrace.hlsl - Water instance ID handling
shaders/dxr/common.hlsl                 - INSTANCE_ID_WATER constant
```

---

## What's NOT in MVP (Future Enhancements)

- Caustics
- Foam/spray particles
- Chromatic dispersion
- Subsurface scattering
- Procedural normal maps/ripples
- Double-buffered async loading
- Frame interpolation
- BOBJ format support
- Recursive ray bounces (depth > 1)
