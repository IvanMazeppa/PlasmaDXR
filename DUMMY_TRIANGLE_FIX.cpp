// Quick fix to test: Add dummy triangle to BLAS
// This converts pure procedural BLAS to mixed geometry BLAS
// Theory: Ada Lovelace might have a bug with pure procedural BLAS at 2045+ primitives

void RTLightingSystem_RayQuery::BuildBLAS(ID3D12GraphicsCommandList4* cmdList) {
    // ... existing AABB generation ...

    // Create a dummy degenerate triangle at origin (won't affect rendering)
    struct Vertex {
        float x, y, z;
    };
    
    // Upload buffer for triangle (3 vertices, degenerate at origin)
    Vertex dummyTriangle[3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},  
        {0.0f, 0.0f, 0.0f}
    };
    
    // Create vertex buffer
    ComPtr<ID3D12Resource> dummyVB;
    CD3DX12_HEAP_PROPERTIES uploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(dummyTriangle));
    
    device->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &vbDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&dummyVB));
    
    // Copy triangle data
    void* mapped;
    dummyVB->Map(0, nullptr, &mapped);
    memcpy(mapped, dummyTriangle, sizeof(dummyTriangle));
    dummyVB->Unmap(0, nullptr);
    
    // Build BLAS with BOTH geometries
    D3D12_RAYTRACING_GEOMETRY_DESC geomDescs[2] = {};
    
    // Geometry 0: Your existing AABBs
    geomDescs[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    geomDescs[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDescs[0].AABBs.AABBCount = m_particleCount;  // 2045 AABBs
    geomDescs[0].AABBs.AABBs.StartAddress = m_aabbBuffer->GetGPUVirtualAddress();
    geomDescs[0].AABBs.AABBs.StrideInBytes = 24;
    
    // Geometry 1: Dummy triangle (degenerate, won't be hit by rays)
    geomDescs[1].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomDescs[1].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geomDescs[1].Triangles.VertexBuffer.StartAddress = dummyVB->GetGPUVirtualAddress();
    geomDescs[1].Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
    geomDescs[1].Triangles.VertexCount = 3;
    geomDescs[1].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomDescs[1].Triangles.IndexCount = 0;  // Non-indexed
    geomDescs[1].Triangles.Transform3x4 = 0;  // No transform
    
    // Build BLAS with mixed geometry
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    blasInputs.NumDescs = 2;  // TWO geometries now!
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputs.pGeometryDescs = geomDescs;
    
    // ... rest of BLAS build code ...
    
    LOG_INFO("Built MIXED geometry BLAS: {} AABBs + 1 dummy triangle", m_particleCount);
}

// In your shader, primitive index 0-2044 = particles, index 2045+ = triangle (ignored)
// The triangle is degenerate (zero area) so rays won't hit it
