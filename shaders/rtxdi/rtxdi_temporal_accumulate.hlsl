// RTXDI Temporal Accumulation Shader
// Milestone 5: Smooth patchwork pattern via temporal sample accumulation
// Algorithm: Exponential Moving Average (EMA)

cbuffer AccumulationConstants : register(b0) {
    uint g_screenWidth;
    uint g_screenHeight;
    uint g_frameIndex;              // Current frame number
    uint g_maxSamples;              // Target sample count (8 or 16)
    float g_resetThreshold;         // Camera movement threshold (10.0 units)
    float3 g_cameraPos;             // Current camera position
    float3 g_prevCameraPos;         // Previous camera position
    uint g_forceReset;              // Force reset flag (1=reset, 0=normal)
    row_major float4x4 g_viewProj;      // Current ViewProj (for unprojection)
    row_major float4x4 g_prevViewProj;  // Previous ViewProj (for reprojection)
};

// Inputs
Texture2D<float4> g_rtxdiOutput : register(t0);      // Current frame RTXDI (raw samples)
Texture2D<float4> g_prevAccumulated : register(t1); // Previous frame accumulated

// Output
RWTexture2D<float4> g_currAccumulated : register(u0);

// Calculate world position from pixel coordinates (Planar Z=0 assumption)
float3 PixelToWorldPosition(uint2 pixelCoord) {
    float2 uv = (float2(pixelCoord) + 0.5) / float2(g_screenWidth, g_screenHeight);
    float diskRange = 600.0f;  // Accretion disk spans 600 units (-300 to +300)
    float3 worldPos;
    worldPos.x = -300.0f + uv.x * diskRange;
    worldPos.y = -300.0f + (1.0 - uv.y) * diskRange;
    worldPos.z = 0.0f;  // Disk plane
    return worldPos;
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint2 pixelCoord = DTid.xy;

    // Bounds check
    if (pixelCoord.x >= g_screenWidth || pixelCoord.y >= g_screenHeight) {
        return;
    }

    // === REPROJECTION (Fix for "Patchwork") ===
    // Find where this pixel was in the previous frame
    uint2 prevPixelCoord = pixelCoord; // Default to current if reprojection fails
    bool reprojected = false;

    // 1. Reconstruct world position (Planar Assumption)
    float3 worldPos = PixelToWorldPosition(pixelCoord);

    // 2. Project to previous clip space
    float4 prevClipPos = mul(float4(worldPos, 1.0), g_prevViewProj);

    // 3. Convert to screen space
    if (prevClipPos.w > 0.0) {
        float3 prevNDC = prevClipPos.xyz / prevClipPos.w;
        float2 prevUV = float2(
            (prevNDC.x + 1.0) * 0.5,
            (1.0 - prevNDC.y) * 0.5  // Flip Y
        );

        // Check if valid UV
        if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
            prevPixelCoord = uint2(prevUV * float2(g_screenWidth, g_screenHeight));
            reprojected = true;
        }
    }

    // Read current frame RTXDI output (raw 1-sample-per-pixel)
    float4 currentSample = g_rtxdiOutput[pixelCoord];
    uint currentLightIndex = asuint(currentSample.r);

    // Read previous accumulated result (from REPROJECTED coordinate)
    // Use Point Sampler load to avoid filtering indices
    float4 prevAccum = g_prevAccumulated[prevPixelCoord];
    uint prevSampleCount = uint(prevAccum.g);
    uint prevFrameID = uint(prevAccum.a);

    // === RESET CONDITIONS ===

    // 1. Force reset (user toggled RTXDI, changed presets, etc.)
    bool shouldReset = g_forceReset != 0;

    // 2. Reprojection failure (off-screen)
    if (!reprojected) {
        shouldReset = true;
    }

    // 3. Frame discontinuity (frame counter jumped, app paused, etc.)
    if (g_frameIndex != prevFrameID + 1 && prevFrameID != 0) {
        shouldReset = true;
    }

    // Note: We REMOVED the "Camera moved significantly" check because reprojection handles movement!
    // We effectively INCREASED the reset threshold to infinity, relying on reprojection validity.

    // 4. Current sample is invalid (no lights in cell)
    if (currentLightIndex == 0xFFFFFFFF) {
        // Keep previous accumulated result (don't corrupt with invalid samples)
        // Write to CURRENT pixel (we are filling holes) using reprojected history
        g_currAccumulated[pixelCoord] = prevAccum;
        return;
    }

    // === ACCUMULATION LOGIC ===

    float4 outputSample;

    if (shouldReset || prevSampleCount == 0) {
        // RESET: Start fresh with current sample
        outputSample.r = currentSample.r;       // Current light index
        outputSample.g = 1.0;                   // Sample count = 1
        outputSample.b = 0.0;                   // Reserved
        outputSample.a = float(g_frameIndex);   // Frame ID
    } else {
        // ACCUMULATE: Blend with exponential moving average

        // Clamp sample count to max (prevents alpha from becoming too small)
        uint clampedSampleCount = min(prevSampleCount, g_maxSamples);

        // EMA blend factor: alpha = 1 / (sampleCount + 1)
        // Early frames: high alpha (fast convergence)
        // Late frames: low alpha (stable, smooth)
        float alpha = 1.0 / float(clampedSampleCount + 1);

        // LIGHT INDEX BLENDING STRATEGY:
        // Probabilistic switch (discrete light index)
        // - With probability alpha, use current sample
        // - With probability (1-alpha), use previous sample
        // Over time, this converges to the most frequently sampled light

        // Generate per-pixel random value (deterministic, varies per frame)
        uint seed = pixelCoord.y * g_screenWidth + pixelCoord.x;
        seed = (seed * 747796405u + 2891336453u) + g_frameIndex;
        float randomValue = float(seed % 10000) / 10000.0;

        if (randomValue < alpha) {
            // Use current sample
            outputSample.r = currentSample.r;
        } else {
            // Keep previous sample
            outputSample.r = prevAccum.r;
        }

        // Update sample count
        outputSample.g = min(float(clampedSampleCount + 1), float(g_maxSamples));

        // Reserved channels
        outputSample.b = 0.0;
        outputSample.a = float(g_frameIndex);
    }

    // Write accumulated result
    g_currAccumulated[pixelCoord] = outputSample;
}
