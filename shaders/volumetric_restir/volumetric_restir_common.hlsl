/**
 * Volumetric ReSTIR Common Utilities
 *
 * Shared data structures and helper functions for volumetric path tracing
 * with spatiotemporal reservoir resampling.
 *
 * Based on "Fast Volume Rendering with Spatiotemporal Reservoir Resampling"
 * (Lin, Wyman, Yuksel 2021)
 */

#ifndef VOLUMETRIC_RESTIR_COMMON_HLSL
#define VOLUMETRIC_RESTIR_COMMON_HLSL

//=============================================================================
// Data Structures (must match C++ layout exactly)
//=============================================================================

/**
 * Path vertex: (distance, direction) pair
 *
 * Stores path implicitly as distances and directions from camera.
 * Reconstructing position: x_i = x_{i-1} + z_i * ω_i
 */
struct PathVertex {
    float z;           // Distance from previous vertex (4 bytes)
    float3 omega;      // Direction to next vertex (12 bytes)
                       // Total: 16 bytes
};

/**
 * Volumetric Reservoir (64 bytes per pixel)
 *
 * Stores the winning path from weighted reservoir sampling (RIS).
 * Layout optimized for GPU access (aligned to 64 bytes).
 */
struct VolumetricReservoir {
    uint pathLength;           // Number of scattering events k (0 = direct emission, 1+ = scattering)
    PathVertex vertices[3];    // Up to K=3 bounces (48 bytes)
    float wsum;                // Cumulative weight sum W = Σ w_i
    float M;                   // Total candidates seen (for MIS)
    uint flags;                // Bit 0: isScatteringPath (vs pure emission)
                               // Bits 1-31: reserved for future use
    uint padding;              // Align to 64 bytes
};

/**
 * Constants for path generation shader
 */
struct PathGenerationConstants {
    uint screenWidth;
    uint screenHeight;
    uint particleCount;
    uint randomWalksPerPixel;  // M (default: 4)

    uint maxBounces;           // K (default: 3)
    uint frameIndex;           // For temporal randomization
    uint padding0;
    uint padding1;

    float3 cameraPos;
    float padding2;

    float4x4 viewMatrix;
    float4x4 projMatrix;
    float4x4 invViewProjMatrix;
};

//=============================================================================
// Random Number Generation (PCG)
//=============================================================================

/**
 * PCG random number generator state
 * (Permuted Congruential Generator)
 */
struct PCGState {
    uint state;
};

/**
 * Initialize PCG state from pixel coordinates and frame index
 */
PCGState InitPCG(uint2 pixel, uint frame) {
    PCGState rng;
    rng.state = pixel.x + pixel.y * 1920u + frame * (1920u * 1080u);
    rng.state = rng.state * 747796405u + 2891336453u;
    return rng;
}

/**
 * Generate random uint32 [0, 2^32-1]
 */
uint PCGRandom(inout PCGState rng) {
    uint state = rng.state;
    rng.state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * Generate random float [0, 1)
 */
float PCGRandomFloat(inout PCGState rng) {
    return float(PCGRandom(rng)) * (1.0 / 4294967296.0);
}

/**
 * Generate random float in range [min, max)
 */
float PCGRandomRange(inout PCGState rng, float minVal, float maxVal) {
    return minVal + PCGRandomFloat(rng) * (maxVal - minVal);
}

/**
 * Generate random direction on unit sphere (uniform sampling)
 */
float3 PCGRandomDirection(inout PCGState rng) {
    float z = PCGRandomRange(rng, -1.0, 1.0);
    float a = PCGRandomRange(rng, 0.0, 2.0 * 3.14159265359);
    float r = sqrt(1.0 - z * z);
    return float3(r * cos(a), r * sin(a), z);
}

//=============================================================================
// Regular Tracking (Distance Sampling with Piecewise-Constant Volume)
//=============================================================================

/**
 * Sample distance along ray using piecewise-constant transmittance volume (Mip 2)
 *
 * Uses closed-form sampling of exponential distribution within voxels.
 * Returns sampled distance z and corresponding PDF p(z).
 *
 * @param origin Ray origin
 * @param direction Ray direction (normalized)
 * @param volumeMip2 3D transmittance texture (Mip 2, coarse grid)
 * @param maxDistance Maximum ray distance (far clip)
 * @param rng Random number generator state
 * @param[out] sampledDistance Sampled distance z
 * @param[out] pdf Probability density p(z)
 * @return true if valid sample, false if ray escaped volume
 */
bool SampleDistanceRegular(
    float3 origin,
    float3 direction,
    Texture3D<uint> volumeMip2,  // Changed from float - stored as uint for atomics
    SamplerState volumeSampler,
    float maxDistance,
    inout PCGState rng,
    out float sampledDistance,
    out float pdf)
{
    // Accumulate transmittance and distance
    float t = 0.0;
    float accumulatedTransmittance = 1.0;

    const float voxelSize = 100.0; // World-space voxel size (adjust to match volume bounds)
    const uint maxSteps = 100;     // Prevent infinite loops

    for (uint step = 0; step < maxSteps; step++) {
        float3 samplePos = origin + direction * t;

        // Convert world position to volume texture coordinates [0,1]³
        // Assuming volume covers [-1500, +1500] in each axis (3000-unit range)
        float3 volumeUVW = (samplePos + 1500.0) / 3000.0;

        // Check if ray escaped volume
        if (any(volumeUVW < 0.0) || any(volumeUVW > 1.0)) {
            pdf = 0.0;
            return false;
        }

        // Sample transmittance from Mip 2 (coarse grid, cheap lookup)
        // Volume stored as UINT for atomic operations - use Load() instead of SampleLevel()
        // Convert UVW [0,1] to voxel coordinates [0, 63]
        int3 voxelCoords = int3(volumeUVW * 64.0);
        voxelCoords = clamp(voxelCoords, int3(0, 0, 0), int3(63, 63, 63));

        uint extinctionUint = volumeMip2.Load(int4(voxelCoords, 0));
        float extinction = 1.0 - asfloat(extinctionUint);
        extinction = max(extinction, 0.001); // Prevent division by zero

        // Sample distance within this voxel using exponential distribution
        float u = PCGRandomFloat(rng);
        float dt = -log(1.0 - u) / extinction;

        t += dt;

        // Check if we exceeded max distance
        if (t >= maxDistance) {
            pdf = 0.0;
            return false;
        }

        // Update accumulated transmittance
        accumulatedTransmittance *= exp(-extinction * dt);

        // Russian roulette: decide whether to scatter here
        float scatterProb = 1.0 - exp(-extinction * dt);
        if (PCGRandomFloat(rng) < scatterProb) {
            // Scatter at this point
            sampledDistance = t;
            pdf = scatterProb * accumulatedTransmittance;
            return true;
        }

        // Continue ray marching (didn't scatter)
    }

    // Exceeded max steps without scattering
    pdf = 0.0;
    return false;
}

//=============================================================================
// Weighted Reservoir Sampling (RIS)
//=============================================================================

/**
 * Update reservoir with new candidate
 *
 * Standard weighted reservoir sampling algorithm.
 * Maintains running weight sum and randomly replaces sample based on weight ratio.
 *
 * @param reservoir Current reservoir state (modified in-place)
 * @param candidatePath Candidate path to consider
 * @param weight Importance weight w = p̂(x) / p(x)
 * @param rng Random number generator
 */
void UpdateReservoir(
    inout VolumetricReservoir reservoir,
    PathVertex candidateVertices[3],
    uint candidateLength,
    uint candidateFlags,
    float weight,
    inout PCGState rng)
{
    reservoir.wsum += weight;
    reservoir.M += 1.0;

    // Randomly replace current sample with probability w_i / W
    float replaceProbability = weight / reservoir.wsum;
    if (PCGRandomFloat(rng) < replaceProbability) {
        // Accept candidate
        reservoir.pathLength = candidateLength;

        [unroll]
        for (uint i = 0; i < 3; i++) {
            reservoir.vertices[i] = candidateVertices[i];
        }

        reservoir.flags = candidateFlags;
    }
}

/**
 * Initialize empty reservoir
 */
VolumetricReservoir InitReservoir() {
    VolumetricReservoir r;
    r.pathLength = 0;

    [unroll]
    for (uint i = 0; i < 3; i++) {
        r.vertices[i].z = 0.0;
        r.vertices[i].omega = float3(0, 0, 0);
    }

    r.wsum = 0.0;
    r.M = 0.0;
    r.flags = 0;
    r.padding = 0;

    return r;
}

//=============================================================================
// Phase Function (Henyey-Greenstein)
//=============================================================================

/**
 * Henyey-Greenstein phase function
 *
 * Anisotropic scattering model for participating media.
 * g = 0: isotropic, g > 0: forward scattering, g < 0: back scattering
 *
 * @param cosTheta Cosine of angle between incident and scattered directions
 * @param g Anisotropy parameter [-1, 1]
 * @return Phase function value (normalized to integrate to 1 over sphere)
 */
float HenyeyGreenstein(float cosTheta, float g) {
    float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * 3.14159265359 * pow(denom, 1.5));
}

/**
 * Sample direction from Henyey-Greenstein phase function
 *
 * @param incidentDir Incident direction (toward sample point)
 * @param g Anisotropy parameter
 * @param rng Random number generator
 * @return Scattered direction (away from sample point)
 */
float3 SampleHenyeyGreenstein(float3 incidentDir, float g, inout PCGState rng) {
    float cosTheta;

    if (abs(g) < 0.001) {
        // Isotropic: uniform sphere sampling
        cosTheta = 1.0 - 2.0 * PCGRandomFloat(rng);
    } else {
        // Anisotropic: importance sample HG
        float u = PCGRandomFloat(rng);
        float sqrTerm = (1.0 - g * g) / (1.0 + g - 2.0 * g * u);
        cosTheta = (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g);
    }

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * 3.14159265359 * PCGRandomFloat(rng);

    // Construct local coordinate frame around incident direction
    float3 w = -incidentDir; // Flip to point away from incident
    float3 u_axis = abs(w.x) > 0.1 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 v = normalize(cross(w, u_axis));
    u_axis = cross(v, w);

    // Transform sampled direction to world space
    float3 scattered = sinTheta * cos(phi) * u_axis +
                       sinTheta * sin(phi) * v +
                       cosTheta * w;

    return normalize(scattered);
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * Reconstruct world position from pixel coordinates
 *
 * @param pixelCoords Pixel coordinates (0,0) = top-left
 * @param screenSize Screen dimensions (width, height)
 * @param invViewProj Inverse view-projection matrix
 * @return World-space ray origin and direction
 */
void ReconstructRay(
    uint2 pixelCoords,
    uint2 screenSize,
    float4x4 invViewProj,
    out float3 rayOrigin,
    out float3 rayDirection)
{
    // Normalized device coordinates [-1, 1]
    float2 ndc = (float2(pixelCoords) + 0.5) / float2(screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y for D3D

    // Unproject to world space
    float4 nearPoint = mul(invViewProj, float4(ndc, 0.0, 1.0));
    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));

    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    rayOrigin = nearPoint.xyz;
    rayDirection = normalize(farPoint.xyz - nearPoint.xyz);
}

#endif // VOLUMETRIC_RESTIR_COMMON_HLSL
