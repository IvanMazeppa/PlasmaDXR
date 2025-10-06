# Particle Rendering Debug Plan - Diagonal Shape Issue

## Problem Summary
- **Symptom**: Particles render as diagonal green shapes instead of quads
- **Context**: After forcing instance 0 to render, 6 vertices form a diagonal line instead of 2 triangles
- **Window**: 2882x1668 (DPI scaled from 1920x1080)
- **Performance**: 1200+ FPS with physics off, RT lighting 14 seconds per 60 frames

## Root Cause Analysis

### Issue Identified
The vertex shader has incorrect vertex-to-corner mapping logic. Lines 59-65 in `particle_billboard_vs.hlsl`:
```hlsl
// INCORRECT MAPPING - creates diagonal pattern
if (vertIdx == 0) cornerIdx = 0;      // BL
else if (vertIdx == 1) cornerIdx = 1; // BR
else if (vertIdx == 2) cornerIdx = 2; // TL
else if (vertIdx == 3) cornerIdx = 2; // TL (second triangle)
else if (vertIdx == 4) cornerIdx = 1; // BR (second triangle)
else cornerIdx = 3;                    // TR
```

This creates vertices: BL, BR, TL, TL, BR, TR
- Triangle 1: BL -> BR -> TL (diagonal)
- Triangle 2: TL -> BR -> TR (diagonal)

Both triangles form diagonals instead of filling the quad.

## Step-by-Step Debugging Plan

### 1. Shader Modifications to Isolate Issue

#### A. Debug Vertex Shader with Fixed Positions
Create `particle_billboard_vs_debug.hlsl`:
```hlsl
PixelInput main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    // Force instance 0 for testing
    if (instanceID != 0) {
        PixelInput nullOut;
        nullOut.position = float4(0, 0, 0, 0);
        nullOut.texCoord = float2(0, 0);
        nullOut.color = float4(0, 0, 0, 0);
        nullOut.lighting = float4(0, 0, 0, 0);
        nullOut.alpha = 0;
        return nullOut;
    }

    // Hardcoded clip space positions for testing
    float4 positions[6] = {
        float4(-0.5, -0.5, 0.5, 1.0),  // 0: BL
        float4( 0.5, -0.5, 0.5, 1.0),  // 1: BR
        float4(-0.5,  0.5, 0.5, 1.0),  // 2: TL
        float4(-0.5,  0.5, 0.5, 1.0),  // 3: TL
        float4( 0.5, -0.5, 0.5, 1.0),  // 4: BR
        float4( 0.5,  0.5, 0.5, 1.0)   // 5: TR
    };

    PixelInput output;
    output.position = positions[vertexID];
    output.texCoord = float2(0.5, 0.5);
    output.color = float4(0, 1, 0, 1);  // Green
    output.lighting = float4(0, 0, 0, 0);
    output.alpha = 1.0;
    return output;
}
```

#### B. Corrected Corner Mapping
Fix the vertex indexing:
```hlsl
// CORRECT MAPPING for two CCW triangles
uint cornerIdx;
if (vertIdx == 0) cornerIdx = 0;      // BL
else if (vertIdx == 1) cornerIdx = 1; // BR
else if (vertIdx == 2) cornerIdx = 3; // TR (first triangle: BL-BR-TR)
else if (vertIdx == 3) cornerIdx = 0; // BL (second triangle)
else if (vertIdx == 4) cornerIdx = 3; // TR (second triangle)
else cornerIdx = 2;                    // TL (second triangle: BL-TR-TL)
```

### 2. Additional Logging

#### A. Vertex Position Logger
Add to vertex shader:
```hlsl
// Debug output first 6 vertices of instance 0
if (instanceID == 0 && vertexID < 6) {
    // Use InterlockedAdd to write to debug buffer
    RWStructuredBuffer<float4> debugBuffer : register(u0);
    debugBuffer[vertexID] = float4(worldPos, 1.0);
    debugBuffer[vertexID + 6] = clipPos;
}
```

#### B. C++ Side Logging
Add to `ParticleRenderer_Billboard.cpp`:
```cpp
void LogVertexDebugInfo(ID3D12GraphicsCommandList* cmdList, uint32_t frameNum) {
    if (frameNum == 0 || frameNum == 60 || frameNum == 120) {
        LOG_INFO("=== Vertex Debug Frame {} ===", frameNum);
        LOG_INFO("Draw call: DrawInstanced(6, {}, 0, 0)", m_particleCount);
        LOG_INFO("Expected triangles:");
        LOG_INFO("  Triangle 0: V0(BL) -> V1(BR) -> V2(TR)");
        LOG_INFO("  Triangle 1: V3(BL) -> V4(TR) -> V5(TL)");
    }
}
```

### 3. Test Cases

#### Test A: Single Particle at Origin
```cpp
// In ParticleSystem initialization
if (testMode == TestMode::SingleAtOrigin) {
    particles[0].position = float3(0, 0, 0);
    particles[0].temperature = 500.0f;
    particles[0].density = 1.0f;
    particleCount = 1;
}
```

#### Test B: Grid of 4 Particles
```cpp
if (testMode == TestMode::Grid2x2) {
    particles[0].position = float3(-10, 0, 0);
    particles[1].position = float3( 10, 0, 0);
    particles[2].position = float3(-10, 10, 0);
    particles[3].position = float3( 10, 10, 0);
    particleCount = 4;
}
```

#### Test C: Hardcoded Clip Space Quad
Direct clip space coordinates bypassing transforms.

### 4. PIX Graphics Debugger Commands

#### A. Capture Setup
```cpp
// Add PIX markers
PIXBeginEvent(cmdList, PIX_COLOR_INDEX(0), "ParticleBillboardRender");
PIXSetMarker(cmdList, PIX_COLOR_INDEX(1), "Instance0Debug");

// Before draw call
PIXBeginEvent(cmdList, PIX_COLOR_INDEX(2), "DrawBillboards_%d", m_particleCount);
cmdList->DrawInstanced(6, m_particleCount, 0, 0);
PIXEndEvent(cmdList);

PIXEndEvent(cmdList);
```

#### B. PIX Analysis Commands
1. **Capture Frame**: `WinPixEventRuntime.CaptureFrame()`
2. **Vertex Shader Debugger**:
   - Set breakpoint at vertex 0 of instance 0
   - Step through corner index calculation
   - Inspect clip space positions
3. **Mesh Viewer**:
   - Examine primitive topology
   - Verify triangle winding order
4. **Pipeline State Inspector**:
   - Verify cull mode is NONE
   - Check primitive topology is TRIANGLELIST

### 5. Fallback Rendering Approaches

#### A. Index Buffer Method
```cpp
// Create index buffer for quads
uint16_t indices[] = {
    0, 1, 2,  // Triangle 1: BL-BR-TR
    0, 2, 3   // Triangle 2: BL-TR-TL
};
CreateIndexBuffer(indices, 6);

// Use DrawIndexedInstanced instead
cmdList->DrawIndexedInstanced(6, m_particleCount, 0, 0, 0);
```

#### B. Triangle Strip Method
```cpp
// Change topology
psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE_STRIP;
cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

// Draw 4 vertices as strip (automatic triangulation)
cmdList->DrawInstanced(4, m_particleCount, 0, 0);
```

#### C. Geometry Shader Expansion
```hlsl
[maxvertexcount(6)]
void GS_ExpandPoint(point PointInput input[1],
                    inout TriangleStream<PixelInput> output)
{
    // Expand single point to 6 vertices (2 triangles)
    // Guarantees correct triangle formation
}
```

## Implementation Priority

1. **IMMEDIATE**: Fix vertex corner mapping (5 minutes)
2. **HIGH**: Add debug shader with hardcoded positions (10 minutes)
3. **HIGH**: Add PIX markers and capture frame (5 minutes)
4. **MEDIUM**: Implement index buffer fallback (20 minutes)
5. **LOW**: Add geometry shader expansion (30 minutes)

## Validation Checklist

- [ ] Single particle renders as square quad
- [ ] Quad fills completely (no diagonal gaps)
- [ ] Multiple particles don't overlap incorrectly
- [ ] Camera-facing billboards rotate properly
- [ ] RT lighting applies correctly
- [ ] Performance remains > 1000 FPS

## Expected Resolution

The primary issue is the vertex ordering creating two overlapping diagonal triangles instead of a filled quad. The fix is straightforward:
1. Correct the corner index mapping
2. Ensure CCW winding order
3. Verify with PIX that triangles form correctly

This should immediately resolve the diagonal shape issue and restore proper particle rendering.