### DirectX Raytracing Pipeline and Shader Binding Table: A Practical Guide

This guide is a practical, production-focused walkthrough of building a robust DirectX Raytracing (DXR) pipeline and Shader Binding Table (SBT) with Direct3D 12. It covers correct implementation, common pitfalls, performance and memory considerations, and image-quality advice. Code snippets are in C++ and HLSL and match the DXR 1.0/1.1 model with the Agility SDK.

### What you will build

- **A raytracing state object** that encapsulates your raytracing shaders and configuration
- **A Shader Binding Table** with the correct layout, alignment, and local root data
- **A dispatch path** that launches rays with a compact payload and efficient hit groups

### Prerequisites and capability checks

- Enable the D3D12 debug layer in development.
- Verify raytracing tier support and feature options.

```cpp
// Capability check
ComPtr<ID3D12Device> device = CreateDevice();
D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
ThrowIfFailed(device->CheckFeatureSupport(
    D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));

if (options5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
    throw std::runtime_error("DXR not supported on this device/driver");
}
```

### Overview: Raytracing State Object (RTPSO)

The raytracing pipeline is created via a `D3D12_STATE_OBJECT_DESC` containing subobjects:

- **DXIL libraries**: Compiled shader libraries (e.g., `lib_6_6`) exporting shader entry points
- **Hit groups**: Logical groups that bind closest-hit, any-hit, and/or intersection shaders to a single export
- **Shader config**: Maximum ray payload size and attribute size
- **Global/Local root signatures**: Descriptor bindings; local root data is appended to SBT records
- **Associations**: Map local root signatures and shader config to specific shader exports or hit groups
- **Pipeline config**: Maximum trace recursion depth (and optionally config1 with DXR 1.1)

#### Compiling DXR shaders

Compile raytracing shaders into a DXIL library. You can embed them or load from disk.

```bash
# Example dxc invocations (adjust paths/defines)
dxc -T lib_6_6 -E LibMain -Fo RayLib.dxil RayLib.hlsl -enable-16bit-types -Zi -Qembed_debug
```

Typical HLSL entry attributes:

- `[shader("raygeneration")]` ray-gen
- `[shader("closesthit")]` closest-hit
- `[shader("anyhit")]` any-hit (alpha testing, early accept/reject)
- `[shader("miss")]` miss
- `[shader("intersection")]` for procedural primitives (AABBs)

#### Creating the state object

```cpp
// Helpers for building subobjects omitted for brevity
CD3D12_STATE_OBJECT_DESC soDesc(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);

// 1) DXIL library with exported shaders
D3D12_DXIL_LIBRARY_DESC dxilLib{}; 
dxilLib.DXILLibrary = {pLibBytecode, libBytecodeSize};
// Define D3D12_EXPORT_DESCs for each export (e.g., L"RayGen", L"Miss", L"CHS", L"AHS")
// Associate dxilLib.ExportDesc and NumExports

// 2) Hit group
D3D12_HIT_GROUP_DESC hitGroup{};
hitGroup.HitGroupExport = L"MyHitGroup";
hitGroup.Type            = D3D12_HIT_GROUP_TYPE_TRIANGLES;
hitGroup.ClosestHitShaderImport = L"CHS";
// Optional: hitGroup.AnyHitShaderImport = L"AHS";

// 3) Global & local root signatures
ComPtr<ID3D12RootSignature> globalRS = CreateGlobalRootSignature(device);
ComPtr<ID3D12RootSignature> localRS  = CreateLocalRootSignature(device);

// 4) Shader config (payload + attribute)
D3D12_RAYTRACING_SHADER_CONFIG shaderCfg{};
shaderCfg.MaxPayloadSizeInBytes   = 12;   // keep tiny: e.g., 3 floats (rgb) or packed
shaderCfg.MaxAttributeSizeInBytes = 8;    // triangles use 2 floats (barycentrics) = 8 bytes

// 5) Pipeline config
D3D12_RAYTRACING_PIPELINE_CONFIG pipelineCfg{};
pipelineCfg.MaxTraceRecursionDepth = 1;   // start with 1; raise only when needed

// 6) Subobject-to-exports associations
// - Associate localRS with L"MyHitGroup" and/or specific shader exports
// - Associate shaderCfg with all relevant exports

// 7) Assemble subobjects into soDesc (DXIL, HitGroup, GlobalRS, LocalRS, Assoc, ShaderCfg, PipelineCfg)

ComPtr<ID3D12StateObject> rtStateObject;
ThrowIfFailed(device->CreateStateObject(&soDesc, IID_PPV_ARGS(&rtStateObject)));

// State object properties for identifiers & stack sizes
ComPtr<ID3D12StateObjectProperties> soProps;
ThrowIfFailed(rtStateObject.As(&soProps));
```

With DXR 1.1, you can create **collections** (`D3D12_STATE_OBJECT_TYPE_COLLECTION`) and use `D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE` to link collections for modularity, or incrementally extend pipelines with `D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE` via the Add-To-State-Object path (ATSO) in the Agility SDK.

### Shader Binding Table (SBT)

The SBT maps shader records to geometry and ray types. Each record is:

- `ShaderIdentifier` (32 bytes)
- Optional `LocalRootArguments` (application-defined data for the local root signature)

The SBT is split into up to four regions used by `D3D12_DISPATCH_RAYS_DESC`:

- **RayGen**: 1 record (exactly one per dispatch)
- **Miss**: N records (ray-type variants)
- **HitGroup**: M records (per geometry/material/ray-type)
- **Callable**: K records (optional)

Key alignment rules (constants defined by D3D12):

- `D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT` = 64 bytes alignment for region start addresses
- `D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT` = 32 bytes alignment for each record (stride)

Your strides must be a multiple of 32 and large enough to contain the identifier plus local root data. Region start addresses must be 64-byte aligned. The RayGen region must have **exactly one** record per dispatch.

#### Building the SBT

```cpp
struct SbtRecordBuilder {
    std::vector<uint8_t> storage;

    static constexpr UINT ShaderIdSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES; // 32

    static void WriteRecord(
        uint8_t* dst,
        const void* shaderId,
        const void* localArgs,
        UINT localArgsSize)
    {
        memcpy(dst, shaderId, ShaderIdSize);
        if (localArgs && localArgsSize) {
            memcpy(dst + ShaderIdSize, localArgs, localArgsSize);
        }
    }
};

// Example SBT layout creation
auto shaderIdRayGen   = soProps->GetShaderIdentifier(L"RayGen");
auto shaderIdMiss0    = soProps->GetShaderIdentifier(L"Miss");
auto shaderIdHitGroup = soProps->GetShaderIdentifier(L"MyHitGroup");

// Sizes including local root data
UINT rayGenRecordSize = Align(SbtRecordBuilder::ShaderIdSize + rayGenLocalSize,  D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
UINT missRecordSize   = Align(SbtRecordBuilder::ShaderIdSize + missLocalSize,    D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
UINT hitRecordSize    = Align(SbtRecordBuilder::ShaderIdSize + hitLocalSize,     D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

UINT rayGenTableSize  = Align(rayGenRecordSize * 1,   D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
UINT missTableSize    = Align(missRecordSize   * numMiss, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
UINT hitTableSize     = Align(hitRecordSize    * numHits, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

UINT sbtSize = rayGenTableSize + missTableSize + hitTableSize; // + callable if used

ComPtr<ID3D12Resource> sbt;
ThrowIfFailed(device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
    D3D12_HEAP_FLAG_NONE,
    &CD3DX12_RESOURCE_DESC::Buffer(sbtSize),
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&sbt)));

uint8_t* mapped = nullptr;
sbt->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
uint64_t cursor = 0;

// RayGen (1)
SbtRecordBuilder::WriteRecord(mapped + cursor, shaderIdRayGen, rayGenLocalData, rayGenLocalSize);
cursor += rayGenTableSize; // advance to next region start, already 64B aligned

// Miss (N)
for (UINT i = 0; i < numMiss; ++i) {
    SbtRecordBuilder::WriteRecord(mapped + cursor + i * missRecordSize,
                                  soProps->GetShaderIdentifier(missExportNames[i]),
                                  missLocalData(i), missLocalSize);
}
cursor += missTableSize;

// HitGroup (M)
for (UINT i = 0; i < numHits; ++i) {
    SbtRecordBuilder::WriteRecord(mapped + cursor + i * hitRecordSize,
                                  shaderIdHitGroup,
                                  hitLocalData(i), hitLocalSize);
}

sbt->Unmap(0, nullptr);
```

Note that `LocalRootArguments` must match the `LocalRootSignature` associated with the export, and the amount of data you copy must not exceed the size defined by that root signature. Keep local root data as small as possible.

#### Dispatching rays

```cpp
D3D12_DISPATCH_RAYS_DESC drd{};

// Compute GPU virtual addresses
auto base = sbt->GetGPUVirtualAddress();
D3D12_GPU_VIRTUAL_ADDRESS rgStart = base + 0;
D3D12_GPU_VIRTUAL_ADDRESS msStart = base + rayGenTableSize;
D3D12_GPU_VIRTUAL_ADDRESS hgStart = base + rayGenTableSize + missTableSize;

drd.RayGenerationShaderRecord.StartAddress = rgStart;
drd.RayGenerationShaderRecord.SizeInBytes  = rayGenRecordSize; // one record only

drd.MissShaderTable.StartAddress  = msStart;
drd.MissShaderTable.SizeInBytes   = missTableSize;
drd.MissShaderTable.StrideInBytes = missRecordSize;

drd.HitGroupTable.StartAddress  = hgStart;
drd.HitGroupTable.SizeInBytes   = hitTableSize;
drd.HitGroupTable.StrideInBytes = hitRecordSize;

drd.Width  = dispatchWidth;
drd.Height = dispatchHeight;
drd.Depth  = 1;

// Bind global root signature + descriptor heaps as usual
commandList->SetPipelineState1(rtStateObject.Get());
commandList->DispatchRays(&drd);
```

### HLSL: minimal shader set

Keep payloads as small as possible. For shadows, a 1-byte or 1-int occluded flag is sufficient. For primary visibility, a compact `float3` radiance or packed `uint` can work.

```hlsl
// raytypes.hlsl
struct Payload {
    float3 radiance;
};

[shader("raygeneration")]
void RayGen()
{
    uint2 pix = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    float2 uv = (float2(pix) + 0.5) / float2(dim);

    RayDesc ray;
    ray.Origin = CameraPosition;
    ray.Direction = ComputePrimaryRayDirection(uv);
    ray.TMin = 0.0;
    ray.TMax = 1e27;

    Payload p; p.radiance = 0;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, p);
    Output[pix] = float4(p.radiance, 1);
}

[shader("closesthit")]
void CHS(inout Payload p, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 bc = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y,
                       attr.barycentrics.x,
                       attr.barycentrics.y);
    float3 n = normalize(mul(float4(Interpolate(Normal0, Normal1, Normal2, bc), 0), ObjToWorld).xyz);
    float3 l = normalize(LightDirection);
    float3 albedo = Interpolate(Albedo0, Albedo1, Albedo2, bc);
    p.radiance = albedo * max(0, dot(n, l));
}

[shader("miss")]
void Miss(inout Payload p)
{
    p.radiance = SkyColor;
}
```

If you use alpha-tested geometry, put the test in an **any-hit** shader and return `IgnoreHit()` for transparent texels. Avoid expensive work in any-hit; it runs frequently.

### Root signatures: global vs local

- **Global root signature** holds scene-global descriptor heaps, TLAS SRV (`SRV(SceneBVH)`), and constant buffers/textures shared across shaders.
- **Local root signature** binds per-draw/per-geometry data (e.g., material index, vertex/index buffers) and is associated with exports or hit groups via subobject associations.

Keep local root parameters minimal. Common patterns:

- Root constants for small values (material index, flags)
- Root descriptor tables for a compact `StructuredBuffer<Material>` indexed by the material index
- SRVs/UAVs in global space when they are shared by most shaders

### Shader config and attribute size

- `MaxPayloadSizeInBytes`: total size of your HLSL payload struct. Keep it small (e.g., 4–16 bytes for shadows/reflections) to reduce register pressure and spills.
- `MaxAttributeSizeInBytes`: triangles use 8 bytes (barycentrics). Procedural intersections can use up to 32 bytes. Value must be a multiple of 4 up to 32.

### Pipeline config and recursion

Start with `MaxTraceRecursionDepth = 1`. Raise only when you genuinely need recursive TraceRay (e.g., reflections → shadows). Many effects can be done with `1` using separate passes and screen-space denoisers.

With DXR 1.1, use `ID3D12StateObjectProperties::GetShaderStackSize` to query stack requirements and set `SetPipelineStackSize` to the sum of the worst-case stacks across your call chain. Underestimating stack size can cause GPU faults; overestimating wastes memory.

### Performance best practices

- **Keep payloads tiny**: smaller payloads generate less register pressure and fewer spills. Pack data (e.g., `float16_t`, RGBe, octahedron normals) when possible.
- **Minimize local root data**: large local roots bloat SBT records and memory bandwidth.
- **Avoid any-hit unless necessary**: move most material logic to closest-hit; use any-hit for alpha test and fast accept/reject.
- **Specialize pipelines by ray purpose**: e.g., a shadow-only pipeline with minimal payload and attribute size; a reflection pipeline with different shaders.
- **Batch hit groups** by material type to improve instruction cache locality; reuse hit groups across geometries when possible.
- **Prefer inline raytracing** for simple queries inside raster passes (e.g., ambient occlusion or shadow visibility) if it reduces state changes and improves coherence.
- **Cache pipelines** and use collections: compiling big state objects on the render thread is expensive. Build at load time and serialize if feasible.
- **Use conservative ray flags** (`RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`, `RAY_FLAG_SKIP_CLOSEST_HIT_SHADER`) when applicable to reduce shader invocations.
- **Use instance masks** to cull ray types at TLAS level (e.g., shadow rays skip decals). Instance contribution to hit group index can be used to select per-object materials efficiently.

### Memory and layout tips for SBT

- Use **upload heaps** for frequently rebuilt SBTs (dynamic scenes). For static or rarely changing SBTs, use default heaps and update with staging.
- Ensure table base addresses are 64-byte aligned and strides are multiples of 32 bytes.
- RayGen region: exactly one record per dispatch.
- Keep per-record local data tightly packed; align to 4 bytes. The record stride must be large enough for identifier + local root data, then aligned to 32.
- Consider separating SBTs by ray type if it simplifies your scene binding logic.

### Common validation errors and fixes

- **Export not found**: `State object does not contain any exports named 'X'` → Ensure your DXIL library exports match the names you use in hit groups/SBT. Export renaming via `D3D12_EXPORT_DESC` must be consistent.
- **SBT stride too small**: `StrideInBytes smaller than shader id + local root` → Compute stride as `Align(IdentifierSize + LocalRootSize, 32)`.
- **Region address not aligned**: Start addresses for `MissShaderTable`, `HitGroupTable`, etc., must be 64-byte aligned.
- **Local root mismatch**: Writing more bytes than the local root signature defines, or using a different layout than expected by the HLSL code. Keep a single definition of the local root structure and reuse it for CPU packing.
- **Attribute size mismatch**: Triangles require 8 bytes; if you set more than 8 without procedural intersections, validation may warn or fail.
- **Stack size issues**: Crashes or GPU page faults after adding recursion. Query stacks and call `SetPipelineStackSize` appropriately.
- **Any-hit overuse**: Severe performance drop. Keep any-hit minimal and avoid heavy texture fetches.

### Debugging and profiling

- **PIX**: Inspect the raytracing dispatch, SBT regions, root signatures, descriptor heaps. Use the Shader Table view to validate identifiers and local data.
- **Nsight Graphics / Radeon GPU Profiler**: Attribute and payload pressure, shader invocation counts, and hot spots. Optimize the most-invoked shaders first (often any-hit and closest-hit).
- **D3D12 debug layer**: Keep it on during development; filter noisy messages but do not disable raytracing-related warnings.
- **SBT dumps**: Add a developer toggle to dump SBT records (hex + decoded structs) to verify offsets and sizes.

### Image quality best practices

- **Ray budgets per effect**: Shadows can often be 1 ray/pixel; reflections may use half or quarter resolution with temporal accumulation.
- **Jitter + TAA**: Jitter your camera rays and integrate with TAA to reduce noise. Use blue-noise or Owen-scrambled sequences for improved convergence.
- **Denoisers**: Integrate high-quality spatiotemporal denoisers (e.g., NRD) to trade rays for quality. Provide G-buffer normals, roughness, and motion vectors.
- **Clamp and variance-guided filtering**: Clamp outliers and use per-pixel variance to drive kernel sizes.
- **Hybrid approaches**: Combine screen-space techniques (SSR/SSAO) with RT fallbacks to reduce ray counts.

### Minimal end-to-end checklist

- [ ] Device supports DXR tier ≥ 1.0
- [ ] DXIL library compiled, exports named consistently
- [ ] State object includes: DXIL, hit groups, shader config, root signatures, associations, pipeline config
- [ ] Global RS binds TLAS SRV and scene globals; Local RS binds per-geometry data
- [ ] Payload/attribute sizes minimized
- [ ] SBT: 64B-aligned region starts, 32B-aligned strides; RayGen count = 1
- [ ] DispatchRays dimensions match target output
- [ ] Debug layer clean; PIX/Nsight capture shows expected invocations

### Extended example: tiny material index via local root

```cpp
// Local root with a single 32-bit constant holding a material index
CD3DX12_ROOT_PARAMETER1 params[1];
params[0].InitAsConstants(/*Num32BitValues*/1, /*Register*/0, /*Space*/0, D3D12_SHADER_VISIBILITY_ALL);

D3D12_VERSIONED_ROOT_SIGNATURE_DESC localDesc{};
localDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
localDesc.Desc_1_1 = { _countof(params), params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE };

ComPtr<ID3DBlob> sigBlob, errBlob;
ThrowIfFailed(D3D12SerializeVersionedRootSignature(&localDesc, &sigBlob, &errBlob));
ComPtr<ID3D12RootSignature> localRS;
ThrowIfFailed(device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&localRS)));

// Associate localRS with MyHitGroup via a SUBOBJECT_TO_EXPORTS_ASSOCIATION
```

```hlsl
// HLSL side reads the material index from root constants
cbuffer LocalRoot : register(b0)
{
    uint gMaterialIndex;
};

[shader("closesthit")]
void CHS(inout Payload p, in BuiltInTriangleIntersectionAttributes attr)
{
    Material m = Materials[gMaterialIndex];
    // shade using m
}
```

This pattern keeps SBT records extremely small (identifier + 4 bytes), enabling large scenes with many hit records while maintaining memory efficiency.

### Practical SBT sizing

As a rule of thumb for a single ray type:

- RayGen: 1 × (32 + small local) → often 32–64 bytes (stride aligned to 32)
- Miss: few records × (32 + small local) → ≤ a few KB total
- HitGroup: number of materials/instances × (32 + 4–16 bytes) → can dominate; design to minimize this

For dynamic scenes, rebuild the HitGroup region each frame using an upload buffer. Avoid per-frame RTPSO rebuilds; those are expensive.

### Pitfalls to avoid

- **Mismatched export names** between DXIL library, hit groups, SBT, and HLSL `[shader]` entry points
- **Oversized payloads** causing register spills and performance cliffs
- **Overusing recursion** instead of multi-pass approaches
- **Any-hit doing heavy work** (texture sampling, loops) → move to closest-hit or precompute in raster
- **Large local root signatures** (arrays of descriptors) bloating SBT
- **Incorrect TLAS instance masks** leading to unnecessary shader invocations

### Where to go next

- Add a callable region for reusable utilities (e.g., BRDF sampling) if it simplifies shader code.
- Experiment with **DXR 1.1** features: pipeline libraries, collections, and `AddToStateObject` for incremental builds.
- Integrate **denoisers** and **temporal sampling** to reduce ray counts while improving image stability.

### References (authoritative and recommended)

- Microsoft Docs — DirectX Raytracing: [DirectX Raytracing (DXR) Overview](https://learn.microsoft.com/windows/win32/direct3d12/directx-raytracing)
- Microsoft Docs — State Subobjects: [D3D12 State Object and Subobjects](https://learn.microsoft.com/windows/win32/direct3d12/intro-to-raytracing-state-objects)
- Microsoft Docs — Shader Table: [Shader Table Layout and DispatchRays](https://learn.microsoft.com/windows/win32/direct3d12/directx-raytracing-shader-table)
- DirectX-Graphics-Samples — DXR Samples: [D3D12 Raytracing Samples](https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/Samples/Desktop/D3D12Raytracing)
- NVIDIA — Best Practices for RTX: [RTX Best Practices Guide](https://developer.nvidia.com/blog/rtx-best-practices)
- Intel — Real-Time Ray Tracing in Games: [Developer Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/real-time-ray-tracing-in-games.html)

---

By following this guide, you will have a correct, validation-friendly DXR pipeline and a compact, well-aligned SBT that scales to large scenes while keeping performance high and image quality stable.


