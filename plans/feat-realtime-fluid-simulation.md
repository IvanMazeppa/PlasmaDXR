# Real-Time Fluid Simulation with RT Lighting

## Overview

Implement a fully interactive fluid simulation where users can pour water, interact with liquid physics, and see real-time RT reflections/refractions. This is significantly more complex than baked simulations but enables dynamic, game-ready water effects.

**Estimated Effort:** 6-8 weeks for production quality
**Quick Prototype (using NVIDIA Flex):** 1-2 weeks

---

## Industry Examples

| Title | Technique | Performance | Notes |
|-------|-----------|-------------|-------|
| NVIDIA Flex Demos | Position Based Dynamics (PBD) | 60 FPS, 100K particles | Reference implementation |
| Control (Remedy) | SPH + RT reflections | 30-60 FPS | AAA game quality |
| Portal RTX | Pre-baked + RT | 60 FPS | Hybrid approach |
| Sea of Thieves | Height-field ocean | 60 FPS | 2D wave simulation (not full 3D) |
| Teardown | Voxel-based fluid | 60 FPS | Simplified but fast |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Frame Pipeline (~8ms total)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. PHYSICS COMPUTE (1-2ms)                              │    │
│  │     ├─ Particle emission from user input                 │    │
│  │     ├─ Neighbor search (spatial hashing)                 │    │
│  │     ├─ Density/pressure calculation                      │    │
│  │     ├─ Force accumulation (pressure, viscosity, gravity) │    │
│  │     ├─ Velocity integration                              │    │
│  │     └─ Collision detection (via TLAS ray queries)        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  2. SURFACE RECONSTRUCTION (0.5-1ms)                     │    │
│  │     ├─ Compute density field from particles              │    │
│  │     ├─ Marching Cubes isosurface extraction              │    │
│  │     ├─ Normal calculation                                │    │
│  │     └─ Output: Vertex buffer + Index buffer              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  3. ACCELERATION STRUCTURE UPDATE (0.3-0.5ms)            │    │
│  │     ├─ BLAS refit (faster than full rebuild)             │    │
│  │     └─ TLAS instance update                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  4. RT RENDERING (2-4ms)                                 │    │
│  │     ├─ Primary rays → water surface hits                 │    │
│  │     ├─ Reflection rays                                   │    │
│  │     ├─ Refraction rays with absorption                   │    │
│  │     └─ Shadow rays to light sources                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5. POST-PROCESSING & PRESENT (1-2ms)                    │    │
│  │     ├─ DLSS upscaling                                    │    │
│  │     ├─ Tone mapping                                      │    │
│  │     └─ Present                                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Total: 5-10ms = 100-200 FPS headroom @ 1080p                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: GPU Fluid Solver (2-3 weeks)

### 1.1 Choose Simulation Method

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **SPH** (Smoothed Particle Hydrodynamics) | Physically accurate, handles splashes | Complex to stabilize | Realistic water |
| **PBD** (Position Based Dynamics) | Stable, controllable, fast | Less physically accurate | Games, interactive |
| **FLIP/APIC** | Very accurate, handles thin sheets | Slow, complex | Offline rendering |
| **MPM** (Material Point Method) | Handles multiple materials | Very complex | Snow, sand, mud |

**Recommendation:** Start with **PBD** for stability and speed, upgrade to **SPH** for realism later.

### 1.2 Particle Data Structure

```hlsl
struct FluidParticle {
    float3 position;
    float  density;
    float3 velocity;
    float  pressure;
    float3 force;
    float  mass;
    uint   cellHash;      // For spatial hashing
    uint   particleIndex; // Original index for sorting
};

// GPU buffers
RWStructuredBuffer<FluidParticle> g_Particles;
RWStructuredBuffer<uint> g_CellStart;    // Spatial hash grid
RWStructuredBuffer<uint> g_CellEnd;
RWStructuredBuffer<uint> g_SortedIndices;
```

### 1.3 Spatial Hashing (Neighbor Search)

The most performance-critical part of fluid simulation:

```hlsl
// Hash function for 3D grid cell
uint HashCell(int3 cell) {
    const uint p1 = 73856093;
    const uint p2 = 19349663;
    const uint p3 = 83492791;
    return ((uint)(cell.x * p1) ^ (uint)(cell.y * p2) ^ (uint)(cell.z * p3)) % HASH_TABLE_SIZE;
}

// Compute shader: Assign particles to cells
[numthreads(256, 1, 1)]
void CSAssignCells(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_ParticleCount) return;

    float3 pos = g_Particles[idx].position;
    int3 cell = int3(floor(pos / CELL_SIZE));
    uint hash = HashCell(cell);

    g_Particles[idx].cellHash = hash;
    g_Particles[idx].particleIndex = idx;

    // Atomic increment for counting sort
    InterlockedAdd(g_CellCount[hash], 1);
}
```

### 1.4 PBD Fluid Solver

```hlsl
// Constants
static const float REST_DENSITY = 1000.0;  // kg/m³ for water
static const float VISCOSITY = 0.01;
static const float3 GRAVITY = float3(0, -9.81, 0);
static const float PARTICLE_RADIUS = 0.05;
static const float SMOOTH_RADIUS = PARTICLE_RADIUS * 4.0;

// SPH Kernel functions
float Poly6Kernel(float r, float h) {
    if (r >= h) return 0;
    float x = (h * h - r * r);
    return 315.0 / (64.0 * PI * pow(h, 9)) * x * x * x;
}

float3 SpikyGradient(float3 r, float dist, float h) {
    if (dist >= h || dist < 0.0001) return float3(0, 0, 0);
    float x = h - dist;
    return -45.0 / (PI * pow(h, 6)) * x * x * normalize(r);
}

// Main solver iteration
[numthreads(256, 1, 1)]
void CSSolveConstraints(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_ParticleCount) return;

    FluidParticle p = g_Particles[idx];

    // Calculate density
    float density = 0;
    for (uint n = 0; n < neighborCount; n++) {
        uint neighborIdx = GetNeighbor(idx, n);
        float3 r = p.position - g_Particles[neighborIdx].position;
        float dist = length(r);
        density += p.mass * Poly6Kernel(dist, SMOOTH_RADIUS);
    }
    p.density = density;

    // Calculate pressure (Tait equation of state)
    float pressureCoeff = 1000.0;  // Stiffness
    p.pressure = pressureCoeff * (pow(p.density / REST_DENSITY, 7) - 1);

    // Calculate forces
    float3 pressureForce = float3(0, 0, 0);
    float3 viscosityForce = float3(0, 0, 0);

    for (uint n = 0; n < neighborCount; n++) {
        uint neighborIdx = GetNeighbor(idx, n);
        FluidParticle neighbor = g_Particles[neighborIdx];

        float3 r = p.position - neighbor.position;
        float dist = length(r);

        // Pressure force
        float pressureTerm = (p.pressure + neighbor.pressure) / (2 * neighbor.density);
        pressureForce -= neighbor.mass * pressureTerm * SpikyGradient(r, dist, SMOOTH_RADIUS);

        // Viscosity force
        float3 velDiff = neighbor.velocity - p.velocity;
        viscosityForce += VISCOSITY * neighbor.mass * velDiff / neighbor.density
                         * ViscosityLaplacian(dist, SMOOTH_RADIUS);
    }

    // Total force
    p.force = pressureForce + viscosityForce + p.mass * GRAVITY;

    g_Particles[idx] = p;
}

// Integration step
[numthreads(256, 1, 1)]
void CSIntegrate(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_ParticleCount) return;

    FluidParticle p = g_Particles[idx];

    // Semi-implicit Euler integration
    p.velocity += (p.force / p.mass) * g_DeltaTime;
    p.position += p.velocity * g_DeltaTime;

    // Collision with scene geometry (via ray queries)
    p.position = ResolveCollisions(p.position, p.velocity);

    g_Particles[idx] = p;
}
```

### 1.5 Collision Detection via RT

Use DXR RayQuery for particle-scene collisions:

```hlsl
float3 ResolveCollisions(float3 position, inout float3 velocity) {
    // Cast ray in velocity direction
    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = normalize(velocity);
    ray.TMin = 0;
    ray.TMax = length(velocity) * g_DeltaTime + PARTICLE_RADIUS;

    q.TraceRayInline(g_SceneTLAS, RAY_FLAG_NONE, COLLISION_MASK, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float hitT = q.CommittedRayT();
        float3 hitNormal = GetTriangleNormal(q);

        // Push particle out of collision
        position = ray.Origin + ray.Direction * (hitT - PARTICLE_RADIUS * 1.1);

        // Reflect velocity with damping
        float restitution = 0.3;
        velocity = reflect(velocity, hitNormal) * restitution;
    }

    return position;
}
```

---

## Phase 2: Surface Reconstruction (1 week)

### 2.1 Density Field Generation

Convert particles to a 3D density grid:

```hlsl
// 3D texture for density field
RWTexture3D<float> g_DensityField;

[numthreads(8, 8, 8)]
void CSComputeDensityField(uint3 id : SV_DispatchThreadID) {
    float3 worldPos = GridToWorld(id);
    float density = 0;

    // Sample nearby particles
    int3 cellMin = int3(floor((worldPos - SMOOTH_RADIUS) / CELL_SIZE));
    int3 cellMax = int3(floor((worldPos + SMOOTH_RADIUS) / CELL_SIZE));

    for (int x = cellMin.x; x <= cellMax.x; x++) {
        for (int y = cellMin.y; y <= cellMax.y; y++) {
            for (int z = cellMin.z; z <= cellMax.z; z++) {
                uint hash = HashCell(int3(x, y, z));
                uint start = g_CellStart[hash];
                uint end = g_CellEnd[hash];

                for (uint i = start; i < end; i++) {
                    FluidParticle p = g_Particles[g_SortedIndices[i]];
                    float dist = length(worldPos - p.position);
                    density += Poly6Kernel(dist, SMOOTH_RADIUS);
                }
            }
        }
    }

    g_DensityField[id] = density;
}
```

### 2.2 Marching Cubes

Extract triangle mesh from density field:

```hlsl
// Marching Cubes lookup tables (256 cases)
StructuredBuffer<uint> g_TriTable;    // 256 * 16 entries
StructuredBuffer<uint> g_EdgeTable;   // 256 entries

// Output buffers
RWStructuredBuffer<float3> g_Vertices;
RWStructuredBuffer<float3> g_Normals;
RWStructuredBuffer<uint> g_Indices;
RWByteAddressBuffer g_Counter;  // Atomic counter for vertices

static const float ISO_LEVEL = 0.5;  // Surface threshold

[numthreads(4, 4, 4)]
void CSMarchingCubes(uint3 id : SV_DispatchThreadID) {
    // Sample 8 corners of cube
    float values[8];
    values[0] = g_DensityField[id + uint3(0, 0, 0)];
    values[1] = g_DensityField[id + uint3(1, 0, 0)];
    values[2] = g_DensityField[id + uint3(1, 1, 0)];
    values[3] = g_DensityField[id + uint3(0, 1, 0)];
    values[4] = g_DensityField[id + uint3(0, 0, 1)];
    values[5] = g_DensityField[id + uint3(1, 0, 1)];
    values[6] = g_DensityField[id + uint3(1, 1, 1)];
    values[7] = g_DensityField[id + uint3(0, 1, 1)];

    // Determine cube configuration (256 possibilities)
    uint cubeIndex = 0;
    for (int i = 0; i < 8; i++) {
        if (values[i] < ISO_LEVEL) cubeIndex |= (1 << i);
    }

    // Skip if cube is entirely inside or outside surface
    if (cubeIndex == 0 || cubeIndex == 255) return;

    // Get triangle configuration from lookup table
    uint edgeFlags = g_EdgeTable[cubeIndex];

    // Interpolate vertices on edges
    float3 edgeVertices[12];
    // ... (interpolation code)

    // Output triangles
    for (int t = 0; g_TriTable[cubeIndex * 16 + t] != 255; t += 3) {
        uint baseIndex;
        g_Counter.InterlockedAdd(0, 3, baseIndex);

        for (int v = 0; v < 3; v++) {
            uint edge = g_TriTable[cubeIndex * 16 + t + v];
            g_Vertices[baseIndex + v] = edgeVertices[edge];
            g_Normals[baseIndex + v] = ComputeNormal(id, edge);
        }

        g_Indices[baseIndex + 0] = baseIndex + 0;
        g_Indices[baseIndex + 1] = baseIndex + 1;
        g_Indices[baseIndex + 2] = baseIndex + 2;
    }
}
```

### 2.3 Alternative: Screen-Space Surface

Faster but view-dependent (good for prototyping):

```hlsl
// Render particles as spheres to depth buffer
// Then blur/smooth the depth
// Calculate normals from depth gradients
// Shade as water surface

// Pros: Very fast (0.2ms), no mesh generation
// Cons: View-dependent artifacts, no refraction through thick water
```

---

## Phase 3: User Interaction (3-5 days)

### 3.1 Particle Emission

```cpp
class FluidEmitter {
public:
    void EmitAtCursor(float3 worldPos, float3 velocity) {
        // Spawn particles in a small radius
        for (int i = 0; i < m_particlesPerEmit; i++) {
            FluidParticle p;
            p.position = worldPos + RandomInSphere(m_emitRadius);
            p.velocity = velocity + RandomInSphere(m_velocityVariance);
            p.mass = m_particleMass;
            p.density = 0;
            p.pressure = 0;
            p.force = float3(0, 0, 0);

            m_particles.push_back(p);
        }
    }

    void UpdateFromMouse(float2 mousePos, float2 mouseDelta, Camera& camera) {
        if (IsMouseDown(MOUSE_LEFT)) {
            float3 worldPos = camera.ScreenToWorld(mousePos);
            float3 velocity = camera.ScreenToWorldDirection(mouseDelta) * m_pourSpeed;
            EmitAtCursor(worldPos, velocity);
        }
    }

private:
    float m_emitRadius = 0.1f;
    float m_velocityVariance = 0.5f;
    float m_particleMass = 0.001f;
    float m_pourSpeed = 5.0f;
    int m_particlesPerEmit = 50;
};
```

### 3.2 ImGui Controls

```cpp
void FluidSystem::RenderUI() {
    if (ImGui::CollapsingHeader("Fluid Simulation")) {
        ImGui::SliderFloat("Viscosity", &m_viscosity, 0.0f, 0.1f);
        ImGui::SliderFloat("Stiffness", &m_stiffness, 100.0f, 10000.0f);
        ImGui::SliderFloat("Rest Density", &m_restDensity, 500.0f, 2000.0f);

        ImGui::Separator();
        ImGui::Text("Emitter");
        ImGui::SliderFloat("Pour Speed", &m_emitter.pourSpeed, 1.0f, 20.0f);
        ImGui::SliderInt("Particles/Frame", &m_emitter.particlesPerEmit, 10, 200);

        ImGui::Separator();
        ImGui::Text("Statistics");
        ImGui::Text("Particles: %d", m_particleCount);
        ImGui::Text("Physics: %.2f ms", m_physicsTime);
        ImGui::Text("Surface: %.2f ms", m_surfaceTime);

        if (ImGui::Button("Clear Fluid")) {
            ClearAllParticles();
        }
        if (ImGui::Button("Reset Scene")) {
            ResetToInitialState();
        }
    }
}
```

---

## Phase 4: RT Integration (1 week)

### 4.1 BLAS Update Strategy

```cpp
void FluidSystem::UpdateAccelerationStructure(ID3D12GraphicsCommandList4* cmdList) {
    // Option 1: Full rebuild (simple but slower)
    // m_meshBLAS->Build(cmdList, m_vertexBuffer, m_indexBuffer, m_vertexCount, m_indexCount);

    // Option 2: BLAS refit (faster for deforming geometry)
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.Inputs = m_blasInputs;
    buildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    buildDesc.SourceAccelerationStructureData = m_blas->GetGPUVirtualAddress();  // Previous frame
    buildDesc.DestAccelerationStructureData = m_blas->GetGPUVirtualAddress();

    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    // UAV barrier before use
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_blas.Get());
    cmdList->ResourceBarrier(1, &barrier);
}
```

### 4.2 Water Material (Same as MVP)

The water shader from the MVP plan works identically for real-time fluid.

---

## Phase 5: Optimization (1-2 weeks)

### 5.1 Performance Targets

| Component | Target | Technique |
|-----------|--------|-----------|
| Neighbor Search | < 0.5ms | Sorted spatial hash, GPU radix sort |
| Physics Solve | < 1.0ms | Jacobi iteration, async compute |
| Surface Recon | < 0.5ms | LOD grid, frustum culling |
| BLAS Update | < 0.3ms | Refit instead of rebuild |
| RT Rendering | < 3.0ms | Ray budget, early termination |
| **Total** | **< 6ms** | **166 FPS @ 1080p** |

### 5.2 Particle Count Scaling

| Particles | Expected FPS (RTX 4060 Ti) | Visual Quality |
|-----------|---------------------------|----------------|
| 10K | 120+ FPS | Small splashes |
| 50K | 90 FPS | Medium water body |
| 100K | 60 FPS | Large pool |
| 200K | 30 FPS | Cinematic only |

### 5.3 Async Compute

Run physics on async compute queue while rendering previous frame:

```
Frame N:
  Graphics Queue: Render water (Frame N-1 physics)
  Compute Queue:  Simulate physics (Frame N)

This hides physics latency behind render time.
```

---

## Quick Prototype Option: NVIDIA Flex (1-2 weeks)

If you want to prototype quickly before building custom solver:

### Using NVIDIA Flex

```cpp
// 1. Initialize Flex library
NvFlexLibrary* lib = NvFlexInit();
NvFlexSolver* solver = NvFlexCreateSolver(lib, maxParticles);

// 2. Set solver parameters
NvFlexParams params;
params.gravity[1] = -9.81f;
params.viscosity = 0.01f;
params.cohesion = 0.05f;
NvFlexSetParams(solver, &params);

// 3. Each frame
void Update() {
    // Add particles from user input
    if (mouseDown) {
        EmitParticles(cursorPos);
    }

    // Step simulation
    NvFlexUpdateSolver(solver, deltaTime, 1, false);

    // Get particle positions for rendering
    NvFlexGetParticles(solver, positions, nullptr);

    // Run marching cubes on positions
    GenerateSurface(positions);
}
```

**Trade-offs:**
- ✅ Production-quality physics out of the box
- ✅ Well-optimized for GPU
- ✅ Includes surface reconstruction
- ❌ External dependency (NVIDIA GameWorks)
- ❌ Less control over simulation details
- ❌ Licensing considerations for commercial use

---

## Implementation Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | GPU Fluid Solver | Particles move with gravity, collide with floor |
| 3 | Neighbor Search + Pressure | Particles clump like liquid, splashes work |
| 4 | Surface Reconstruction | Triangle mesh generated from particles |
| 5 | RT Integration | Reflections and refractions working |
| 6 | User Interaction | Pour water with mouse, clear/reset |
| 7-8 | Optimization & Polish | Stable 60 FPS, good visual quality |

---

## Success Criteria

- [ ] User can click and drag to pour water into scene
- [ ] Water collides with scene geometry (bowl, floor)
- [ ] Surface has real-time RT reflections and refractions
- [ ] Splashes and ripples look natural
- [ ] Stable 60 FPS with 50K particles at 1080p
- [ ] ImGui controls for viscosity, emission rate, etc.

---

## Future Enhancements

- **Foam particles:** Secondary particles at high-velocity impacts
- **Bubbles:** Air entrainment in turbulent regions
- **Multiple fluids:** Oil and water don't mix
- **Temperature:** Hot and cold water, steam generation
- **Viscous fluids:** Honey, lava, slime
- **Destruction:** Fluid erodes/destroys voxel terrain
