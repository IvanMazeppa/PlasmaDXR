### DirectX Raytracing Acceleration Structures in D3D12: Build, Update, and Compaction

This is a practical, production-focused guide to building, updating, and compacting DirectX Raytracing (DXR) acceleration structures with Direct3D 12. It emphasizes correctness, performance, memory efficiency, and image-quality considerations. Code samples are in C++ with notes for HLSL where relevant.

### What you will implement

- **Bottom-Level Acceleration Structures (BLAS)** per mesh or geometry set
- **Top-Level Acceleration Structures (TLAS)** over BLAS instances
- **Update paths** for dynamic content (when beneficial and supported)
- **Compaction** to reduce memory footprint for static content
- **Robust synchronization and alignment** practices to avoid GPU faults and validation errors

### Prerequisites and constants you must know

- Enable the D3D12 debug layer in development; capture with PIX/Nsight during bring-up.
- Capability check: `D3D12_FEATURE_D3D12_OPTIONS5` for `RaytracingTier` ≥ 1.0.
- Key DXR constants (defined in `d3d12.h`):
  - **`D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT = 256`** (result and scratch buffers must be 256B aligned)
  - **`D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT = 16`** (TLAS instance descriptor array base must be 16B aligned)
  - Triangle attributes are 8 bytes (barycentrics), informing shader config, not AS builds

### BLAS vs TLAS: what goes where

- **BLAS (bottom-level)**: Built from geometry (triangle meshes or procedural AABBs). One BLAS typically corresponds to one mesh (or a cluster of submeshes that always animate together). BLAS quality directly affects traversal cost; more expensive to build than TLAS.
- **TLAS (top-level)**: A scene-level hierarchy of BLAS instances. Each instance carries a transform, mask, culling/opacity flags, and indices that connect to your SBT. TLAS is cheap to rebuild compared to BLAS.

Implementation strategy:

- Static meshes → build BLAS once, compact, and reuse.
- Skinned/deforming meshes → consider BLAS updates (if vertex topology is stable) or full rebuilds if deformation is large and frequent.
- Rigid transforms per-object → rebuild TLAS each frame (fast) or use TLAS update when you specifically architect for it; many engines simply rebuild TLAS.

### Choosing BLAS partitioning

- **Per-mesh BLAS** is a sensible default: keeps update/build units manageable.
- **Per-material BLAS** can reduce SBT hit record pressure (fewer material variants per BLAS) and improve instruction-cache locality when hit groups are material-specialized.
- Avoid giant “mega BLAS” that mixes unrelated meshes; it hurts update selectivity and compaction efficiency.

### BLAS build: end-to-end steps

1) Describe geometry with `D3D12_RAYTRACING_GEOMETRY_DESC` (triangles):

- Index format: `DXGI_FORMAT_R16_UINT` or `DXGI_FORMAT_R32_UINT`.
- Vertex format: usually `DXGI_FORMAT_R32G32B32_FLOAT` for positions; specify `VertexBuffer.StartAddress`, `StrideInBytes`, and `VertexCount`.
- Optional per-geometry transform: `Triangles.Transform3x4` (GPU VA to a 3x4 row-major transform). Prefer leaving this null and use TLAS instance transforms for rigid motion.
- Geometry flags: `D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE` to skip any-hit; `NO_DUPLICATE_ANYHIT_INVOCATION` when alpha-test any-hit cannot cull early (rare).

2) Query sizes with `GetRaytracingAccelerationStructurePrebuildInfo`:

- Fill `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` (Type = BLAS, num descs, flags, layout) and call `GetRaytracingAccelerationStructurePrebuildInfo`.
- Allocate both result and scratch with sizes ≥ returned values and aligned to 256 bytes.

3) Allocate resources:

- Result (BLAS): default heap, `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE`.
- Scratch: default heap, `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` (UAV). You can reuse a large scratch buffer across builds if you size it to the maximum needed.

4) Build:

- Record `BuildRaytracingAccelerationStructure` into a compute or direct queue command list.
- Insert a UAV barrier on the BLAS result after the build before using it (for tracing or as input to a TLAS build).

Example (minimal C++):

```cpp
// Geometry desc (one triangle mesh)
D3D12_RAYTRACING_GEOMETRY_DESC g{};
g.Type                                    = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
g.Flags                                   = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE; // skip any-hit
auto& tri                                  = g.Triangles;
tri.IndexCount                             = indexCount;
tri.IndexFormat                            = DXGI_FORMAT_R32_UINT;
tri.IndexBuffer                            = indexBuffer->GetGPUVirtualAddress();
tri.VertexCount                            = vertexCount;
tri.VertexFormat                           = DXGI_FORMAT_R32G32B32_FLOAT; // position
tri.VertexBuffer.StartAddress              = vertexBuffer->GetGPUVirtualAddress();
tri.VertexBuffer.StrideInBytes             = sizeof(Vertex);
tri.Transform3x4                           = 0; // prefer TLAS instance transform

D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
inputs.Type                                = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
inputs.Flags                               = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
inputs.DescsLayout                         = D3D12_ELEMENTS_LAYOUT_ARRAY;
inputs.NumDescs                            = 1;
inputs.pGeometryDescs                      = &g;

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes{};
device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &sizes);

// Allocate result and scratch with 256B alignment (sizes.ResultDataMaxSizeInBytes, sizes.ScratchDataSizeInBytes)

D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
build.Inputs                               = inputs;
build.DestAccelerationStructureData        = blasResult->GetGPUVirtualAddress();
build.ScratchAccelerationStructureData     = scratch->GetGPUVirtualAddress();

cmdList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
CD3DX12_RESOURCE_BARRIER uav = CD3DX12_RESOURCE_BARRIER::UAV(blasResult.Get());
cmdList->ResourceBarrier(1, &uav);
```

Build flags you will use most:

- **`PREFER_FAST_TRACE`** for static or frequently traced content (most cases).
- **`PREFER_FAST_BUILD`** for very dynamic content where rebuild dominates cost.
- **`ALLOW_UPDATE`** if you will update this BLAS later (adds build time and larger memory footprint).
- **`ALLOW_COMPACTION`** to later shrink memory via post-build info and copy-compact.
- **`MINIMIZE_MEMORY`** when VRAM-constrained; can increase trace time.

### TLAS build: instances and SBT linkage

The TLAS references an array of `D3D12_RAYTRACING_INSTANCE_DESC` structures stored in GPU-visible memory.

Important fields to fill per instance:

- **`Transform`**: 3x4 matrix in row-major layout. This is the upper 3 rows of a typical 4x4 affine matrix; the implicit last row is `[0 0 0 1]`. Supplying column-major here will produce incorrect placement.
- **`InstanceID`**: 24-bit user value accessible in HLSL via `InstanceID()`. Great for indexing per-instance material/transform buffers.
- **`InstanceMask`**: 8-bit mask ANDed with ray’s visibility mask to include/exclude instances per ray type (e.g., shadows skip decals).
- **`InstanceContributionToHitGroupIndex`**: offset added to the hit-sbt index when invoking closest/any-hit for this instance. Use this to select the right material record.
- **`Flags`**: per-instance overrides: `FORCE_OPAQUE`, `FORCE_NON_OPAQUE`, `TRIANGLE_CULL_DISABLE`, `TRIANGLE_FRONT_COUNTERCLOCKWISE`.
- **`AccelerationStructure`**: GPU VA of the BLAS result.

Workflow:

1) Build or update BLAS for all meshes you plan to instance.
2) Fill the instance descriptor array in an upload or default heap buffer aligned to 16 bytes.
3) Build the TLAS (similar call as BLAS but with `Type = TOP_LEVEL` and `Inputs.InstanceDescs = GPUVA`).
4) Insert a UAV barrier on the TLAS result after the build.

Example (C++):

```cpp
// Fill instance descs (row-major 3x4 transforms)
std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(numInstances);
for (uint32_t i = 0; i < numInstances; ++i) {
    auto& d = instances[i];
    memset(&d, 0, sizeof(d));
    // Row-major 3x4
    memcpy(d.Transform, &rowMajor3x4[0][0], sizeof(float) * 12);
    d.InstanceID                             = instanceId[i] & 0xFFFFFF;
    d.InstanceMask                           = 0xFF; // visible to all rays by default
    d.InstanceContributionToHitGroupIndex    = hitGroupBase[i];
    d.Flags                                  = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
    d.AccelerationStructure                  = blasResults[i]->GetGPUVirtualAddress();
}

// Upload to GPU (ensure base address is 16B aligned)
ComPtr<ID3D12Resource> instanceBuffer = CreateUploadBuffer(instances.data(), instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
inputs.Type                = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
inputs.Flags               = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
inputs.NumDescs            = numInstances;
inputs.DescsLayout         = D3D12_ELEMENTS_LAYOUT_ARRAY;
inputs.InstanceDescs       = instanceBuffer->GetGPUVirtualAddress();

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes{};
device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &sizes);

// Allocate TLAS result and scratch (default heap)

D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
build.Inputs               = inputs;
build.DestAccelerationStructureData    = tlasResult->GetGPUVirtualAddress();
build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

cmdList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(tlasResult.Get()));
```

### Updates: when and how

BLAS updates reduce cost when geometry connectivity is unchanged and vertex positions move slightly (e.g., skinning). Requirements:

- The original BLAS must have been built with **`ALLOW_UPDATE`**.
- For update, set `build.SourceAccelerationStructureData` to the prior BLAS result, `DestAccelerationStructureData` to the new (or same) result, and add **`PERFORM_UPDATE`** to build flags.
- Use `PrebuildInfo.UpdateScratchDataSizeInBytes` for the scratch buffer size for updates.

Guidance:

- Use BLAS updates when deformation is moderate and you amortize the initial ALLOW_UPDATE cost across many frames.
- If the mesh topology changes (different index/vertex counts), you must rebuild the BLAS.
- For TLAS, both rebuild and update paths exist, but many engines simply rebuild TLAS every frame because it’s fast and simpler. Use TLAS update only when you benchmark a clear win in your content.

### Compaction: shrinking static acceleration structures

Compaction reduces memory usage while preserving trace performance for structures built with **`ALLOW_COMPACTION`**.

Steps:

1) After building, generate post-build info for the BLAS/TLAS using `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` with `COMPACTED_SIZE` query.
2) Read back the compacted size (CPU or GPU readback) and allocate a new result buffer sized to that value (256B aligned).
3) Issue `CopyRaytracingAccelerationStructure` with `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT` into the new buffer.
4) Insert a UAV barrier on the compacted result before use; then release the original.

Important constraint:

- A compacted acceleration structure is final; **do not expect to update it later**. If you need updates in the future, keep the original un-compacted structure or plan to rebuild.

### Synchronization, resource states, and queues

- Build writes occur via UAV; always insert a **UAV barrier** on the AS result before using it (for tracing or as an input to another build).
- Result buffers should be in `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` when used for tracing; scratch remains `UNORDERED_ACCESS`.
- You can build on the **compute queue** while raster work runs on the graphics queue. Use fences to synchronize if a TLAS build consumes BLAS built on another queue.
- Reuse a large scratch buffer across builds; grow conservatively to the max of requested sizes to avoid frequent reallocations.

### Instance transforms: row-major 3x4 layout

- The `D3D12_RAYTRACING_INSTANCE_DESC::Transform` is a 3x4 **row-major** matrix. In memory it is laid out as 12 floats: rows 0..2 of a 4x4 affine matrix; the omitted row is `[0 0 0 1]`.
- If your math library produces column-major matrices, **transpose** before writing into `Transform`.
- Non-uniform scales and handedness flips affect triangle winding. Combine with `TRIANGLE_FRONT_COUNTERCLOCKWISE` instance flag if you need to flip the cull sense.

### SBT linkage via instance contribution index

- `InstanceContributionToHitGroupIndex` is added to the hit-sbt index to select a material record. Typical patterns:
  - One record per material: set the contribution to the material base for this instance.
  - One record per (material × ray-type): interleave or block by ray-type and adjust contribution per dispatch.
- Keep your SBT compact. Using a material index in local root data often yields smaller SBTs than per-primitive records.

### Performance heuristics and content advice

- Prefer **`PREFER_FAST_TRACE`** for most BLAS that are traced many times per frame.
- Use **`PREFER_FAST_BUILD`** for very dynamic meshes (explosions, crowds) where rebuild time dominates.
- Avoid overusing **any-hit**. Mark geometry **opaque** whenever possible to skip any-hit invocations.
- Use **instance masks** to limit which objects certain ray types can hit (e.g., shadow rays skip particles/decals).
- Partition BLAS by **material families** if it improves shader I-cache locality.
- If VRAM is tight, compact static BLAS; keep dynamic BLAS un-compacted for flexibility.

### Common pitfalls and fixes

- **Wrong transform layout**: The TLAS instance `Transform` is row-major 3x4; writing column-major matrices produces incorrect placements.
- **Alignment errors**: Result and scratch buffers must be 256B aligned; TLAS instance array base must be 16B aligned.
- **Forgetting UAV barriers**: Always add a UAV barrier on the AS result after `BuildRaytracingAccelerationStructure` and after `CopyRaytracingAccelerationStructure`.
- **Misusing update/compaction**: Updates require the initial **`ALLOW_UPDATE`**. Compacted structures should be treated as immutable.
- **Geometry desc mistakes**: Vertex stride/format must match your position buffer. Bad formats lead to traversal crashes.
- **Too many materials in SBT**: Prefer passing a small material index via local roots instead of duplicating large SBT records.

### Minimal bring-up checklist

- [ ] Device supports DXR ≥ 1.0; debug layer enabled
- [ ] BLAS built with correct geometry formats; flags chosen per content
- [ ] TLAS instance descs: row-major 3x4 transforms; masks/IDs set; 16B aligned buffer
- [ ] Build sizes queried; buffers sized and 256B aligned
- [ ] UAV barriers after builds and copies
- [ ] Optional: post-build compaction for static BLAS/TLAS
- [ ] PIX/Nsight capture validates TLAS/BLAS contents and instance transforms

### Example: BLAS update vs rebuild trade-off

If a skinned mesh updates 10,000 vertices per frame but keeps topology fixed, building with `ALLOW_UPDATE` and performing updates can save ~30–50% of BLAS cost vs full rebuild, depending on GPU/driver and mesh size. However, the initial BLAS (with ALLOW_UPDATE) is slower to build and larger. Benchmark your content; if the character is on-screen rarely, a rebuild may be cheaper overall.

### TLAS per-frame rebuild pattern

Many engines rebuild TLAS every frame:

- Regenerate instance descs from current rigid transforms.
- Rebuild TLAS with `PREFER_FAST_TRACE`.
- Use a persistent scratch buffer sized to your worst case.

This is simple, robust, and typically fast enough (TLAS rebuild is usually a small fraction of frame time).

### References

- Microsoft Docs — DirectX Raytracing: `https://learn.microsoft.com/windows/win32/direct3d12/directx-raytracing`
- Microsoft Docs — Acceleration Structures: `https://learn.microsoft.com/windows/win32/direct3d12/acceleration-structure-build`
- Microsoft Docs — Raytracing State Objects: `https://learn.microsoft.com/windows/win32/direct3d12/intro-to-raytracing-state-objects`
- DirectX-Graphics-Samples — DXR Samples: `https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/Samples/Desktop/D3D12Raytracing`
- NVIDIA — Best Practices for RTX: `https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/`

---

With these practices, your acceleration structures will be correct, compact, and fast—forming a solid foundation for shadows, reflections, global illumination, and other ray-traced effects.


