"""
Shader Code Generation Tool
Generate HLSL shader code for DXR 1.1 inline RayQuery shadow systems
"""

from typing import Dict, Optional, List


async def generate_shadow_shader(
    technique: str = "inline_rayquery",
    quality_preset: str = "balanced",
    integration: str = "gaussian_renderer",
    features: Optional[List[str]] = None
) -> Dict:
    """
    Generate HLSL shader code for shadow implementation

    Args:
        technique: Shadow technique (inline_rayquery, rtxdi_integrated, hybrid)
        quality_preset: Quality level (performance, balanced, quality)
        integration: Where to integrate (gaussian_renderer, rtxdi_raygen, standalone)
        features: Optional features (temporal_accumulation, volumetric_attenuation, soft_shadows)

    Returns:
        Dict with generated shader code and integration instructions
    """

    if features is None:
        features = ["temporal_accumulation", "volumetric_attenuation"]

    results = {
        "technique": technique,
        "quality_preset": quality_preset,
        "integration": integration,
        "features": features,
        "shader_code": {},
        "root_signature": {},
        "integration_steps": []
    }

    # Generate shader code based on technique
    if technique == "inline_rayquery":
        results["shader_code"]["hlsl"] = generate_inline_rayquery_code(
            quality_preset, features
        )
        results["root_signature"] = generate_root_signature_inline_rayquery()
        results["integration_steps"] = generate_integration_steps_inline_rayquery(integration)

    elif technique == "rtxdi_integrated":
        results["shader_code"]["hlsl"] = generate_rtxdi_integrated_code(
            quality_preset, features
        )
        results["root_signature"] = generate_root_signature_rtxdi()
        results["integration_steps"] = generate_integration_steps_rtxdi(integration)

    elif technique == "hybrid":
        results["shader_code"]["hlsl"] = generate_hybrid_code(
            quality_preset, features
        )
        results["root_signature"] = generate_root_signature_hybrid()
        results["integration_steps"] = generate_integration_steps_hybrid(integration)

    return results


def generate_inline_rayquery_code(quality_preset: str, features: List[str]) -> str:
    """Generate inline RayQuery shadow shader code"""

    # Determine ray count based on quality preset
    ray_count_map = {
        "performance": 1,
        "balanced": 4,
        "quality": 8
    }
    ray_count = ray_count_map.get(quality_preset, 4)

    # Build shader code
    code = f'''//==============================================================================
// DXR 1.1 Inline RayQuery Shadow System
// Quality Preset: {quality_preset.upper()} ({ray_count} rays per light)
// Features: {", ".join(features)}
//==============================================================================

// Shadow configuration (root constants b10)
cbuffer ShadowConstants : register(b10)
{{
    uint  g_shadowRaysPerLight;  // Configurable ray count
    float g_shadowBias;          // Self-intersection avoidance
}};

// Shadow buffers (reuse PCSS ping-pong buffers)
Texture2D<float>   g_prevShadow : register(t5);  // Previous frame
RWTexture2D<float> g_currShadow : register(u2);  // Current frame

// TLAS (reuse from RTLightingSystem)
RaytracingAccelerationStructure g_TLAS : register(t3);

// Particle buffer (for volumetric attenuation)
StructuredBuffer<Particle> g_particles : register(t0);

//------------------------------------------------------------------------------
// Volumetric Shadow Occlusion
// Computes shadow visibility from particle to light with volumetric attenuation
//------------------------------------------------------------------------------
float ComputeVolumetricShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float3 lightDir,
    float  lightDistance)
{{
    // Setup shadow ray
    RayDesc shadowRay;
    shadowRay.Origin = particlePos + lightDir * g_shadowBias;  // Avoid self-intersection
    shadowRay.Direction = lightDir;
    shadowRay.TMin = g_shadowBias;
    shadowRay.TMax = lightDistance - g_shadowBias;

    // Initialize ray query for inline raytracing
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;

    query.TraceRayInline(
        g_TLAS,                    // Reuse existing acceleration structure
        RAY_FLAG_NONE,
        0xFF,                      // Instance inclusion mask
        shadowRay
    );

    // Accumulate shadow opacity through volume
    float shadowOpacity = 0.0;
    int hitCount = 0;

    while (query.Proceed())
    {{
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {{
            uint particleIndex = query.CandidatePrimitiveIndex();
            Particle occluder = g_particles[particleIndex];

            // Calculate intersection distance through particle
            float tHit = query.CandidateTriangleRayT();
            float distThroughParticle = min(occluder.radius * 2.0, shadowRay.TMax - tHit);

'''

    # Add volumetric attenuation if enabled
    if "volumetric_attenuation" in features:
        code += '''            // Beer-Lambert law: I = I0 * exp(-density * distance)
            // Use temperature as proxy for density (hotter = denser)
            float density = saturate(occluder.temperature / 26000.0);
            float attenuation = 1.0 - exp(-density * distThroughParticle * 0.5);

            // Accumulate opacity
            shadowOpacity += attenuation * (1.0 - shadowOpacity);  // Blend with existing opacity

'''
    else:
        code += '''            // Binary occlusion (no volumetric attenuation)
            shadowOpacity = 1.0;

'''

    code += '''            hitCount++;

            // Early out if fully occluded
            if (shadowOpacity >= 0.99 || hitCount >= 8)
            {{
                query.Abort();
                break;
            }}
        }}
    }}

    query.CommitProceduralPrimitiveHit(shadowOpacity);

    // Return shadow factor (0 = fully shadowed, 1 = fully lit)
    return 1.0 - saturate(shadowOpacity);
}}

'''

    # Add multi-sample soft shadows if quality > performance
    if ray_count > 1:
        code += f'''//------------------------------------------------------------------------------
// Soft Shadow Sampling (Multi-ray)
// Casts {ray_count} rays with jittered offsets for soft penumbra
//------------------------------------------------------------------------------
float ComputeSoftShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float3 lightDir,
    float  lightDistance,
    uint2  pixelCoord,
    uint   frameCount)
{{
    float shadowSum = 0.0;

    // Poisson disk offsets for {ray_count}-sample soft shadows
    static const float2 poissonDisk[{ray_count}] = {{
'''
        # Generate Poisson disk samples
        if ray_count == 4:
            code += '''        float2(-0.94201624, -0.39906216),
        float2( 0.94558609, -0.76890725),
        float2(-0.09418410, -0.92938870),
        float2( 0.34495938,  0.29387760)
'''
        elif ray_count == 8:
            code += '''        float2(-0.94201624, -0.39906216),
        float2( 0.94558609, -0.76890725),
        float2(-0.09418410, -0.92938870),
        float2( 0.34495938,  0.29387760),
        float2(-0.91588581,  0.45771432),
        float2(-0.81544232, -0.87912464),
        float2(-0.38277543,  0.27676845),
        float2( 0.97484398,  0.75648379)
'''

        code += '''    }};

    // Build tangent space for light direction
    float3 tangent = abs(lightDir.y) < 0.999 ?
                     normalize(cross(lightDir, float3(0, 1, 0))) :
                     normalize(cross(lightDir, float3(1, 0, 0)));
    float3 bitangent = cross(lightDir, tangent);

    // Light radius for penumbra (adjust for soft shadow width)
    float lightRadius = 50.0;  // TODO: Make configurable per light

    // Temporal rotation for sample distribution (reduces noise)
    float rotation = (frameCount % 16) * (3.14159265 / 8.0);
    float cosRot = cos(rotation);
    float sinRot = sin(rotation);

    [unroll]
    for (uint i = 0; i < g_shadowRaysPerLight; i++)
    {{
        // Apply temporal rotation to Poisson disk sample
        float2 offset = poissonDisk[i % {ray_count}];
        float2 rotatedOffset = float2(
            offset.x * cosRot - offset.y * sinRot,
            offset.x * sinRot + offset.y * cosRot
        );

        // Jitter light position for area light effect
        float3 jitteredLightPos = lightPos +
                                   tangent * (rotatedOffset.x * lightRadius) +
                                   bitangent * (rotatedOffset.y * lightRadius);

        float3 toJitteredLight = jitteredLightPos - particlePos;
        float3 jitteredLightDir = normalize(toJitteredLight);
        float jitteredLightDist = length(toJitteredLight);

        // Cast shadow ray
        float shadowFactor = ComputeVolumetricShadowOcclusion(
            particlePos,
            jitteredLightPos,
            jitteredLightDir,
            jitteredLightDist
        );

        shadowSum += shadowFactor;
    }}

    // Average shadow factor across samples
    return shadowSum / float(g_shadowRaysPerLight);
}}

'''
    else:
        # Single ray version (performance preset)
        code += '''//------------------------------------------------------------------------------
// Single Shadow Ray (Performance Preset)
// Alias for single-ray shadow computation
//------------------------------------------------------------------------------
float ComputeSoftShadowOcclusion(
    float3 particlePos,
    float3 lightPos,
    float3 lightDir,
    float  lightDistance,
    uint2  pixelCoord,
    uint   frameCount)
{
    return ComputeVolumetricShadowOcclusion(particlePos, lightPos, lightDir, lightDistance);
}

'''

    # Add temporal accumulation if enabled
    if "temporal_accumulation" in features:
        code += '''//------------------------------------------------------------------------------
// Temporal Shadow Accumulation
// Blends current shadow with previous frame for temporal stability
//------------------------------------------------------------------------------
float ApplyTemporalAccumulation(
    uint2 pixelCoord,
    float currentShadow)
{
    // Read previous frame shadow
    float prevShadow = g_prevShadow[pixelCoord];

    // Temporal blend (67ms convergence like PCSS)
    // 0.1 blend factor = 10% new sample, 90% history
    float blendFactor = 0.1;
    float finalShadow = lerp(prevShadow, currentShadow, blendFactor);

    // Write to current shadow buffer (will be prev next frame)
    g_currShadow[pixelCoord] = finalShadow;

    return finalShadow;
}

'''

    # Add main integration point
    code += '''//------------------------------------------------------------------------------
// Main Shadow Integration
// Call this in particle_gaussian_raytrace.hlsl after light contribution
//------------------------------------------------------------------------------
float3 ApplyShadowsToLighting(
    float3 unshadowedColor,
    float3 particlePos,
    Light  light,
    uint2  pixelCoord,
    uint   frameCount)
{
    // Calculate light direction and distance
    float3 toLight = light.position - particlePos;
    float  lightDistance = length(toLight);
    float3 lightDir = toLight / lightDistance;

    // Compute shadow occlusion
'''

    if ray_count > 1 or "soft_shadows" in features:
        code += '''    float shadowFactor = ComputeSoftShadowOcclusion(
        particlePos,
        light.position,
        lightDir,
        lightDistance,
        pixelCoord,
        frameCount
    );
'''
    else:
        code += '''    float shadowFactor = ComputeVolumetricShadowOcclusion(
        particlePos,
        light.position,
        lightDir,
        lightDistance
    );
'''

    if "temporal_accumulation" in features:
        code += '''
    // Apply temporal accumulation
    shadowFactor = ApplyTemporalAccumulation(pixelCoord, shadowFactor);
'''

    code += '''
    // Apply shadow to lighting
    return unshadowedColor * shadowFactor;
}

//==============================================================================
// End of Shadow System
//==============================================================================
'''

    return code


def generate_root_signature_inline_rayquery() -> Dict:
    """Generate root signature for inline RayQuery shadows"""
    return {
        "description": "Root signature extension for inline RayQuery shadows",
        "new_parameters": [
            {
                "index": 10,
                "type": "32BitConstants",
                "num_dwords": 2,
                "register": "b10",
                "shader_visibility": "ALL",
                "contents": ["shadowRaysPerLight", "shadowBias"]
            }
        ],
        "reused_parameters": [
            {
                "index": "t5",
                "name": "g_prevShadow",
                "type": "Texture2D<float>",
                "source": "PCSS ping-pong buffer (reused)"
            },
            {
                "index": "u2",
                "name": "g_currShadow",
                "type": "RWTexture2D<float>",
                "source": "PCSS ping-pong buffer (reused)"
            },
            {
                "index": "t3",
                "name": "g_TLAS",
                "type": "RaytracingAccelerationStructure",
                "source": "RT lighting system (reused)"
            }
        ],
        "code_cpp": '''// Add to root signature creation (after existing 10 parameters)
CD3DX12_ROOT_PARAMETER1 shadowParams;
shadowParams.InitAsConstants(
    2,                           // 2 DWORDs (shadowRaysPerLight, shadowBias)
    10,                          // Register b10
    0,                           // Space 0
    D3D12_SHADER_VISIBILITY_ALL
);
rootParams[10] = shadowParams;

// Root signature now has 11 parameters (still < 64 DWORD limit)
'''
    }


def generate_integration_steps_inline_rayquery(integration: str) -> List[Dict]:
    """Generate integration steps for inline RayQuery shadows"""

    if integration == "gaussian_renderer":
        return [
            {
                "step": 1,
                "title": "Update particle_gaussian_raytrace.hlsl",
                "actions": [
                    "Copy generated shadow functions to top of shader (after common includes)",
                    "Locate multi-light loop (around line 726)",
                    "After computing light contribution, call ApplyShadowsToLighting()",
                    "Replace unshadowed lighting with shadowed result"
                ],
                "code_example": '''// In multi-light loop (particle_gaussian_raytrace.hlsl:726)
for (uint lightIdx = 0; lightIdx < g_lightCount; lightIdx++)
{
    Light light = g_lights[lightIdx];

    // Compute unshadowed lighting (existing code)
    float3 unshadowedColor = ComputeLighting(light, rayOrigin, viewDir);

    // NEW: Apply shadow system
    float3 shadowedColor = ApplyShadowsToLighting(
        unshadowedColor,
        rayOrigin,
        light,
        DispatchRaysIndex().xy,
        g_frameCount
    );

    finalColor += shadowedColor;
}
'''
            },
            {
                "step": 2,
                "title": "Update root signature (C++)",
                "actions": [
                    "Open ParticleRenderer_Gaussian.cpp",
                    "Locate root signature creation (~line 150)",
                    "Add shadow constants parameter (index 10)",
                    "Recompile application"
                ],
                "code_example": '''// In ParticleRenderer_Gaussian::CreateRootSignature()
CD3DX12_ROOT_PARAMETER1 rootParams[11];  // Was 10, now 11

// ... existing parameters ...

// NEW: Shadow constants (index 10)
rootParams[10].InitAsConstants(
    2,                           // shadowRaysPerLight, shadowBias
    10,                          // Register b10
    0,
    D3D12_SHADER_VISIBILITY_ALL
);
'''
            },
            {
                "step": 3,
                "title": "Upload shadow constants",
                "actions": [
                    "In ParticleRenderer_Gaussian::Render()",
                    "After uploading lighting constants",
                    "Upload shadow configuration",
                    "Pass shadowRaysPerLight from ImGui"
                ],
                "code_example": '''// In ParticleRenderer_Gaussian::Render()

// Upload shadow constants
struct ShadowConstants {
    uint32_t shadowRaysPerLight;  // From m_shadowRaysPerLight
    float shadowBias;             // 0.01 default
} shadowConsts = {
    m_shadowRaysPerLight,  // 1/4/8 based on quality preset
    0.01f
};

cmdList->SetComputeRoot32BitConstants(
    10,                        // Shadow constants parameter
    2,                         // 2 DWORDs
    &shadowConsts,
    0
);
'''
            },
            {
                "step": 4,
                "title": "Add ImGui controls",
                "actions": [
                    "Open Application.cpp",
                    "Locate shadow controls section (ImGui)",
                    "Update to control raytraced shadow rays",
                    "Keep Performance/Balanced/Quality presets"
                ],
                "code_example": '''// In Application::RenderImGui()

ImGui::SeparatorText("Shadow Quality");

static const char* shadowPresets[] = { "Performance", "Balanced", "Quality" };
static int currentPreset = 0;  // Default: Performance

if (ImGui::Combo("Shadow Preset", &currentPreset, shadowPresets, 3))
{
    // Map preset to ray count
    int rayCounts[] = { 1, 4, 8 };
    m_gaussianRenderer->SetShadowRaysPerLight(rayCounts[currentPreset]);
}

ImGui::Text("Rays per light: %d", m_gaussianRenderer->GetShadowRaysPerLight());
'''
            },
            {
                "step": 5,
                "title": "Recompile shaders",
                "actions": [
                    "Rebuild project to compile particle_gaussian_raytrace.hlsl",
                    "Verify DXIL output in build/bin/Debug/shaders/",
                    "Check for compile errors in build log"
                ],
                "command": '''MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:CompileShaders
'''
            },
            {
                "step": 6,
                "title": "Test and profile",
                "actions": [
                    "Run application with Performance preset (1 ray/light)",
                    "Verify FPS >= 115 @ 10K particles",
                    "Test Balanced (4 rays) and Quality (8 rays) presets",
                    "Compare shadow quality to PCSS baseline"
                ],
                "expected_results": {
                    "Performance": "115-120 FPS (matches PCSS Performance)",
                    "Balanced": "90-100 FPS (matches PCSS Balanced)",
                    "Quality": "60-75 FPS (matches PCSS Quality)"
                }
            }
        ]

    return []


def generate_rtxdi_integrated_code(quality_preset: str, features: List[str]) -> str:
    """Generate RTXDI-integrated shadow shader code"""
    return '''//==============================================================================
// RTXDI-Integrated Shadow System
// Combines RTXDI light selection with shadow ray casting
//==============================================================================

// Add to rtxdi_raygen.hlsl after weighted light selection

// After selecting light via RTXDI reservoir sampling:
uint selectedLightIndex = SelectLightFromReservoir(reservoir);
Light selectedLight = g_lights[selectedLightIndex];

// Cast shadow ray for selected light
float3 particlePos = worldRayOrigin();
float3 toLight = selectedLight.position - particlePos;
float lightDist = length(toLight);
float3 lightDir = toLight / lightDist;

// Inline ray query for shadow
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
RayDesc shadowRay;
shadowRay.Origin = particlePos + lightDir * 0.01;
shadowRay.Direction = lightDir;
shadowRay.TMin = 0.01;
shadowRay.TMax = lightDist - 0.01;

shadowQuery.TraceRayInline(g_TLAS, RAY_FLAG_NONE, 0xFF, shadowRay);
shadowQuery.Proceed();

float shadowFactor = shadowQuery.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;

// Write to RTXDI output with shadow in alpha channel
float4 rtxdiOutput = float4(
    selectedLightIndex,  // R: Light index
    0,                    // G: Reserved
    0,                    // B: Reserved
    shadowFactor          // A: Shadow visibility
);

g_rtxdiOutput[DispatchRaysIndex().xy] = rtxdiOutput;

// In gaussian renderer, read shadow from RTXDI output alpha channel
'''


def generate_root_signature_rtxdi() -> Dict:
    """Generate root signature for RTXDI-integrated shadows"""
    return {
        "description": "RTXDI-integrated shadows use existing RTXDI output buffer",
        "new_parameters": [],
        "reused_parameters": [
            {
                "index": "t6",
                "name": "g_rtxdiOutput",
                "type": "Texture2D<float4>",
                "source": "RTXDI M5 output (alpha channel = shadow)"
            }
        ],
        "code_cpp": "// No root signature changes needed - reuses RTXDI output buffer"
    }


def generate_integration_steps_rtxdi(integration: str) -> List[Dict]:
    """Generate integration steps for RTXDI-integrated shadows"""
    return [
        {
            "step": 1,
            "title": "Modify rtxdi_raygen.hlsl",
            "actions": [
                "Add shadow ray after light selection",
                "Store shadow in alpha channel of RTXDI output",
                "Minimal change to existing raygen shader"
            ],
            "code_example": "// See generated HLSL code above"
        }
    ]


def generate_hybrid_code(quality_preset: str, features: List[str]) -> str:
    """Generate hybrid shadow shader code"""
    return "// Hybrid shadow technique - combines raytraced + shadow maps (future optimization)"


def generate_root_signature_hybrid() -> Dict:
    """Generate root signature for hybrid shadows"""
    return {"description": "Hybrid shadows - future optimization"}


def generate_integration_steps_hybrid(integration: str) -> List[Dict]:
    """Generate integration steps for hybrid shadows"""
    return []


async def format_shader_generation_report(results: Dict) -> str:
    """Format shader generation results as markdown report"""

    report = f"""# Shadow Shader Code Generation

## Configuration

- **Technique**: {results['technique']}
- **Quality Preset**: {results['quality_preset']}
- **Integration Point**: {results['integration']}
- **Features**: {', '.join(results['features'])}

---

## Generated Shader Code

```hlsl
{results['shader_code'].get('hlsl', 'No HLSL code generated')}
```

---

## Root Signature Changes

**Description**: {results['root_signature'].get('description', 'N/A')}

"""

    # New parameters
    new_params = results['root_signature'].get('new_parameters', [])
    if new_params:
        report += "### New Parameters\n\n"
        for param in new_params:
            report += f"""**Index {param['index']}**: `{param['type']}`
- Register: `{param['register']}`
- Size: {param['num_dwords']} DWORDs
- Visibility: {param['shader_visibility']}
- Contents: {', '.join(param['contents'])}

"""

    # Reused parameters
    reused_params = results['root_signature'].get('reused_parameters', [])
    if reused_params:
        report += "### Reused Parameters\n\n"
        for param in reused_params:
            report += f"""**{param['name']}** (`{param['index']}`): `{param['type']}`
- Source: {param['source']}

"""

    # C++ code
    cpp_code = results['root_signature'].get('code_cpp')
    if cpp_code:
        report += f"""### C++ Implementation

```cpp
{cpp_code}
```

"""

    # Integration steps
    steps = results.get('integration_steps', [])
    if steps:
        report += "---\n\n## Integration Steps\n\n"

        for step in steps:
            report += f"""### Step {step['step']}: {step['title']}

**Actions**:
"""
            for action in step['actions']:
                report += f"- {action}\n"

            if 'code_example' in step:
                report += f"""
**Code Example**:

```hlsl
{step['code_example']}
```
"""

            if 'command' in step:
                report += f"""
**Command**:

```bash
{step['command']}
```
"""

            if 'expected_results' in step:
                report += "\n**Expected Results**:\n"
                for preset, fps in step['expected_results'].items():
                    report += f"- {preset}: {fps}\n"

            report += "\n---\n\n"

    return report
