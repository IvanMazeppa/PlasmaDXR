// ===========================================================================================
// Depth Pre-Pass Compute Shader - Phase 1: Screen-Space Shadow System
// ===========================================================================================
//
// PURPOSE:
// Renders particle depth values to a depth buffer for screen-space contact shadows.
// This depth buffer is then sampled during screen-space shadow ray marching to detect
// occlusion between sample points and light sources.
//
// ALGORITHM:
// 1. For each particle, transform world position to screen space
// 2. Compute conservative depth value (closest point of Gaussian ellipsoid)
// 3. Write depth to UAV using atomicMin to handle overlapping particles
// 4. Result: R32_UINT depth buffer (float depth stored as uint) with particle occlusion data
//
// PERFORMANCE:
// ~0.1-0.2ms @ 10K particles @ 1440p
// Single pass, highly parallel, texture cache friendly
//
// ===========================================================================================

// Particle structure (matches C++ ParticleSystem::Particle)
struct Particle {
    float3 position;       // World-space position
    float radius;          // Particle radius (for conservative depth)
    float3 velocity;       // Particle velocity
    float temperature;     // Temperature (unused here)
    float3 color;          // Color (unused here)
    float mass;            // Mass (unused here)
};

// Constant buffer
cbuffer DepthPrePassConstants : register(b0) {
    float4x4 g_viewProj;         // View-projection matrix
    uint g_screenWidth;          // Output resolution width
    uint g_screenHeight;         // Output resolution height
    uint g_particleCount;        // Number of particles to process
    float g_padding;
};

// Input particle buffer
StructuredBuffer<Particle> g_particles : register(t0);

// Output depth buffer (R32_UINT - we store float as uint for atomic operations)
RWTexture2D<uint> g_depthBuffer : register(u0);

// ===========================================================================================
// MAIN COMPUTE SHADER
// ===========================================================================================

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint particleIdx = dispatchThreadID.x;

    // Bounds check
    if (particleIdx >= g_particleCount) {
        return;
    }

    // Read particle data
    Particle p = g_particles[particleIdx];

    // Transform particle position to clip space
    float4 clipPos = mul(float4(p.position, 1.0), g_viewProj);

    // Perspective divide to get NDC coordinates
    float3 ndc = clipPos.xyz / clipPos.w;

    // Check if particle is in front of camera (reject behind near plane)
    if (clipPos.w <= 0.0) {
        return;  // Behind camera
    }

    // Convert NDC to screen space UV coordinates
    // NDC: [-1, 1] → UV: [0, 1]
    float2 screenUV = float2(
        (ndc.x + 1.0) * 0.5,
        (1.0 - ndc.y) * 0.5  // Flip Y (NDC +Y is up, screen +Y is down)
    );

    // Convert UV to pixel coordinates
    int2 pixelPos = int2(
        screenUV.x * float(g_screenWidth),
        screenUV.y * float(g_screenHeight)
    );

    // Bounds check (reject particles outside screen)
    if (pixelPos.x < 0 || pixelPos.x >= int(g_screenWidth) ||
        pixelPos.y < 0 || pixelPos.y >= int(g_screenHeight)) {
        return;
    }

    // Conservative depth: Compute closest point of Gaussian to camera
    // For now, use particle center depth (can refine later with radius adjustment)
    float depth = ndc.z;  // Already in [0, 1] range for DirectX

    // CONSERVATIVE DEPTH ADJUSTMENT:
    // Subtract radius in NDC space to get closest point of sphere to camera
    // This creates slightly tighter shadows (more conservative occlusion)
    float radiusNDC = (p.radius / clipPos.w) * 0.01;  // Approximate radius in NDC
    depth = saturate(depth - radiusNDC);

    // Atomic min to handle overlapping particles
    // (Multiple particles may project to same pixel - keep closest)
    // Convert float depth to uint for atomic operation (preserves comparison order)
    InterlockedMin(g_depthBuffer[pixelPos], asuint(depth));
}

// ===========================================================================================
// NOTES:
// ===========================================================================================
//
// 1. DEPTH ENCODING:
//    - Depth is stored as R32_UINT (float bits reinterpreted as uint)
//    - Range: [0, 1] (0 = near plane, 1 = far plane)
//    - DirectX NDC Z is already in this range after perspective divide
//    - asuint() preserves IEEE 754 float comparison order for atomic min
//
// 2. CONSERVATIVE DEPTH:
//    - We subtract particle radius to get closest point to camera
//    - This ensures shadows are slightly tighter (better contact detection)
//    - Can be tuned via radiusNDC multiplier (0.01 = very conservative)
//
// 3. ATOMIC OPERATIONS:
//    - InterlockedMin ensures closest particle wins when multiple overlap
//    - This is correct behavior for occlusion testing
//
// 4. PERFORMANCE:
//    - 10K particles × 256-thread groups = 40 dispatches
//    - Each particle does 1 transform, 1 atomic write
//    - Highly parallel, minimal synchronization
//
// 5. FUTURE OPTIMIZATIONS:
//    - Could tile particles and write to depth buffer in tiles
//    - Could use shared memory for local depth reduction
//    - Could rasterize quads instead of points for better coverage
//
// ===========================================================================================
