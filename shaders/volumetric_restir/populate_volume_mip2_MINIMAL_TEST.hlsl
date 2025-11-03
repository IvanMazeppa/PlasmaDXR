/**
 * MINIMAL TEST SHADER - PopulateVolumeMip2 Diagnostic
 *
 * Tests basic compute dispatch infrastructure:
 * - Does shader execute at all?
 * - Do diagnostic counter writes work?
 * - Is the PSO/RS binding correct?
 *
 * This shader does ONLY:
 * 1. Thread 0 writes sentinel 0xDEADBEEF to counter[0]
 * 2. All threads increment counter[0] via InterlockedAdd
 * 3. Thread 0 writes final count to counter[1]
 *
 * Expected results with 2045 particles, 33 thread groups:
 * - counter[0] = 0xDEADBEEF + 2079 = 0xDEADBEEF + 0x81F = 0xDEAE370E
 * - counter[1] = Should contain final count (0xDEAE370E)
 *
 * If counters are zero → Shader never executes (infrastructure problem)
 * If counters have values → Shader executes (main shader logic problem)
 */

//=============================================================================
// Constant Buffer
//=============================================================================

cbuffer PopulationConstants : register(b0) {
    uint g_particleCount;
    uint g_volumeResolution;
    uint g_padding0;
    uint g_padding1;
    float3 g_worldMin;
    float g_padding2;
    float3 g_worldMax;
    float g_padding3;
    float g_extinctionScale;
    float g_padding4;
    float g_padding5;
    float g_padding6;
};

//=============================================================================
// Resources
//=============================================================================

// Particle data (not used in minimal test, but needed for root signature match)
struct Particle {
    float3 position;
    float radius;
    float3 velocity;
    float temperature;
    float3 color;
    float lifetime;
    float3 emissiveColor;
    float emissiveStrength;
    float3 gaussianScale;
    float padding;
};

StructuredBuffer<Particle> g_particles : register(t0);

// Volume texture (not used in minimal test, but needed for root signature match)
RWTexture3D<uint> g_volumeTexture : register(u0);

// Diagnostic counters (ONLY thing we use in minimal test)
RWByteAddressBuffer g_diagnosticCounters : register(u1);

//=============================================================================
// Main Compute Shader - MINIMAL TEST
//=============================================================================

[numthreads(63, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint threadIdx = dispatchThreadID.x;

    // CRITICAL: Dummy reads to prevent DXC from optimizing out resources
    // Root signature expects 4 parameters, DXIL must have all 4 resources
    uint dummyValue = 0;
    if (threadIdx == 999999) {  // Never true, but prevents optimization
        dummyValue = g_particleCount;  // Read cb0
        dummyValue += g_particles[0].radius;  // Read t0
        dummyValue += g_volumeTexture[uint3(0,0,0)];  // Read u0
    }

    // Thread 0: Write sentinel value to prove shader executed
    // Use particle count as unique identifier per run
    if (threadIdx == 0) {
        uint sentinel = 0xDEAD0000 | (g_particleCount & 0xFFFF);  // 0xDEAD + particle count
        g_diagnosticCounters.Store(0, sentinel + dummyValue);
    }

    // ALL threads: Increment counter[0]
    // This should add total thread count to the sentinel value
    uint dummy;
    g_diagnosticCounters.InterlockedAdd(0, 1, dummy);

    // Thread 0: Write final count to counter[1] for verification
    if (threadIdx == 0) {
        // Small delay to let other threads' increments complete
        GroupMemoryBarrierWithGroupSync();

        uint finalCount = g_diagnosticCounters.Load(0);
        g_diagnosticCounters.Store(4, finalCount);  // offset 4 = counter[1]
    }
}
