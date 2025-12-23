#include "NanoVDBSystem.h"
#include "../core/Device.h"
#include "../utils/ResourceManager.h"
#include "../utils/Logger.h"
#include "../utils/d3dx12/d3dx12.h"

#include <d3dcompiler.h>  // For D3DReadFileToBlob
#include <filesystem>     // For directory iteration (animation loading)
#include <algorithm>      // For std::sort
#include <fstream>        // For file reading

// NanoVDB file I/O
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/io/IO.h>

using namespace DirectX;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

std::string NanoVDBSystem::GridTypeToString(uint32_t gridType) {
    switch (gridType) {
        case GRID_TYPE_UNKNOWN: return "Unknown";
        case GRID_TYPE_FLOAT:   return "Float (32-bit)";
        case GRID_TYPE_DOUBLE:  return "Double (64-bit)";
        case GRID_TYPE_INT16:   return "Int16";
        case GRID_TYPE_INT32:   return "Int32";
        case GRID_TYPE_INT64:   return "Int64";
        case GRID_TYPE_VEC3F:   return "Vec3f";
        case GRID_TYPE_VEC3D:   return "Vec3d";
        case GRID_TYPE_MASK:    return "Mask";
        case GRID_TYPE_HALF:    return "Half (16-bit float)";
        case GRID_TYPE_UINT32:  return "UInt32";
        case GRID_TYPE_BOOL:    return "Bool";
        case GRID_TYPE_RGBA8:   return "RGBA8";
        case GRID_TYPE_FP4:     return "FP4 (4-bit)";
        case GRID_TYPE_FP8:     return "FP8 (8-bit)";
        case GRID_TYPE_FP16:    return "FP16 (16-bit quantized)";
        case GRID_TYPE_FPN:     return "FPN (N-bit)";
        case GRID_TYPE_VEC4F:   return "Vec4f";
        case GRID_TYPE_VEC4D:   return "Vec4d";
        default:                return "Unknown (" + std::to_string(gridType) + ")";
    }
}

NanoVDBSystem::~NanoVDBSystem() {
    Shutdown();
}

bool NanoVDBSystem::Initialize(Device* device, ResourceManager* resources,
                                uint32_t screenWidth, uint32_t screenHeight) {
    m_device = device;
    m_resources = resources;
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;

    LOG_INFO("[NanoVDB] Initializing volumetric rendering system...");

    // Create constant buffer (256-byte aligned for D3D12)
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(256);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create constant buffer");
        return false;
    }

    // Create compute pipeline
    if (!CreateComputePipeline()) {
        LOG_ERROR("[NanoVDB] Failed to create compute pipeline");
        return false;
    }

    LOG_INFO("[NanoVDB] System initialized successfully");
    return true;
}

void NanoVDBSystem::Shutdown() {
    m_gridBuffer.Reset();
    m_constantBuffer.Reset();
    m_rootSignature.Reset();
    m_pipelineState.Reset();
    m_hasGrid = false;
    m_hasFileGrid = false;
    m_enabled = false;

    LOG_INFO("[NanoVDB] System shutdown");
}

bool NanoVDBSystem::CreateFogSphere(float radius, DirectX::XMFLOAT3 center,
                                     float voxelSize, float halfWidth) {
    LOG_INFO("[NanoVDB] Creating procedural fog sphere: radius={:.1f}, center=({:.1f}, {:.1f}, {:.1f})",
             radius, center.x, center.y, center.z);

    // For the prototype, we don't actually create a NanoVDB grid
    // Instead, the shader uses procedural density calculation
    // This lets us test the rendering pipeline before integrating full NanoVDB

    // Store sphere parameters for shader
    m_sphereCenter = center;
    m_sphereRadius = radius;

    // Store grid bounds (used by shader for AABB culling)
    float boundsPadding = halfWidth * voxelSize;
    m_gridWorldMin = { center.x - radius - boundsPadding,
                       center.y - radius - boundsPadding,
                       center.z - radius - boundsPadding };
    m_gridWorldMax = { center.x + radius + boundsPadding,
                       center.y + radius + boundsPadding,
                       center.z + radius + boundsPadding };

    // Estimate voxel count for display purposes
    float gridSize = 2.0f * (radius + boundsPadding);
    int voxelsPerSide = static_cast<int>(gridSize / voxelSize);
    m_voxelCount = static_cast<uint32_t>(voxelsPerSide * voxelsPerSide * voxelsPerSide);

    // Mark as having a grid (even though it's procedural)
    m_hasGrid = true;
    m_gridSizeBytes = m_voxelCount * sizeof(float);  // Estimate

    LOG_INFO("[NanoVDB] Procedural fog sphere created!");
    LOG_INFO("  Bounds: [({:.1f}, {:.1f}, {:.1f}) - ({:.1f}, {:.1f}, {:.1f})]",
             m_gridWorldMin.x, m_gridWorldMin.y, m_gridWorldMin.z,
             m_gridWorldMax.x, m_gridWorldMax.y, m_gridWorldMax.z);
    LOG_INFO("  Estimated voxels: {} (procedural - no GPU memory used)", m_voxelCount);

    return true;
}

// ============================================================================
// PHASE 2: GRID ENUMERATION
// ============================================================================

std::vector<NanoVDBSystem::GridInfo> NanoVDBSystem::EnumerateGrids(const std::string& filepath) {
    std::vector<GridInfo> grids;

    try {
        // Read metadata for all grids in the file
        auto metaList = nanovdb::io::readGridMetaData(filepath);

        for (uint32_t i = 0; i < metaList.size(); ++i) {
            const auto& meta = metaList[i];

            GridInfo info;
            info.name = meta.gridName;
            info.type = static_cast<uint32_t>(meta.gridType);
            info.typeName = GridTypeToString(info.type);
            info.index = i;
            info.isCompatible = (info.type == GRID_TYPE_FLOAT ||
                                 info.type == GRID_TYPE_HALF ||
                                 info.type == GRID_TYPE_FP16);

            grids.push_back(info);
        }

        LOG_INFO("[NanoVDB] Enumerated {} grids in file:", grids.size());
        for (const auto& g : grids) {
            const char* compat = g.isCompatible ? "COMPATIBLE" : "incompatible";
            LOG_INFO("[NanoVDB]   [{}] '{}' - {} ({})",
                     g.index, g.name.empty() ? "<unnamed>" : g.name, g.typeName, compat);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("[NanoVDB] Failed to enumerate grids: {}", e.what());
    }

    return grids;
}

// ============================================================================
// LOAD FROM FILE - Original overload (backward compatible)
// ============================================================================

bool NanoVDBSystem::LoadFromFile(const std::string& filepath) {
    // Call the new overload with empty gridName (auto-select)
    return LoadFromFile(filepath, "");
}

// ============================================================================
// LOAD FROM FILE - Phase 2: With grid name selection
// ============================================================================

bool NanoVDBSystem::LoadFromFile(const std::string& filepath, const std::string& gridName) {
    LOG_INFO("[NanoVDB] ========================================");
    LOG_INFO("[NanoVDB] Loading grid from file: {}", filepath.c_str());
    LOG_INFO("[NanoVDB] ========================================");

    // Clear previous error state
    m_lastError.clear();
    m_gridName.clear();
    m_gridType = GRID_TYPE_UNKNOWN;
    m_gridTypeName.clear();

    // Check file extension - we only support .nvdb (NanoVDB native format)
    // OpenVDB (.vdb) files need to be converted first
    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    if (ext == "vdb") {
        m_lastError = "OpenVDB (.vdb) files not directly supported. Convert to .nvdb first.";
        LOG_ERROR("[NanoVDB] {}", m_lastError);
        LOG_ERROR("[NanoVDB] Conversion options:");
        LOG_ERROR("[NanoVDB]   - python scripts/convert_vdb_to_nvdb.py input.vdb output.nvdb");
        LOG_ERROR("[NanoVDB]   - Use Blender's OpenVDB export with NanoVDB option");
        return false;
    }

    try {
        // ================================================================
        // PHASE 2: ENUMERATE AND SELECT GRID
        // ================================================================
        m_availableGrids = EnumerateGrids(filepath);

        if (m_availableGrids.empty()) {
            m_lastError = "No grids found in file";
            LOG_ERROR("[NanoVDB] {}", m_lastError);
            return false;
        }

        // Select grid by name or use smart defaults
        int selectedIndex = -1;

        if (!gridName.empty()) {
            // User specified a grid name - find exact match
            for (const auto& g : m_availableGrids) {
                if (g.name == gridName) {
                    selectedIndex = static_cast<int>(g.index);
                    LOG_INFO("[NanoVDB] Selected grid by name: '{}'", gridName);
                    break;
                }
            }
            if (selectedIndex < 0) {
                // Try case-insensitive match
                std::string lowerName = gridName;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                for (const auto& g : m_availableGrids) {
                    std::string lowerGridName = g.name;
                    std::transform(lowerGridName.begin(), lowerGridName.end(), lowerGridName.begin(), ::tolower);
                    if (lowerGridName == lowerName) {
                        selectedIndex = static_cast<int>(g.index);
                        LOG_INFO("[NanoVDB] Selected grid by name (case-insensitive): '{}'", g.name);
                        break;
                    }
                }
            }
            if (selectedIndex < 0) {
                LOG_WARN("[NanoVDB] Requested grid '{}' not found", gridName);
            }
        }

        // If no name specified or not found, use smart defaults
        if (selectedIndex < 0) {
            // Priority 1: Grid named "density" (common Blender convention)
            for (const auto& g : m_availableGrids) {
                if (g.name == "density" || g.name == "Density") {
                    selectedIndex = static_cast<int>(g.index);
                    LOG_INFO("[NanoVDB] Auto-selected grid: 'density' (index {})", selectedIndex);
                    break;
                }
            }

            // Priority 2: First compatible grid (FLOAT, HALF, or FP16)
            if (selectedIndex < 0) {
                for (const auto& g : m_availableGrids) {
                    if (g.isCompatible) {
                        selectedIndex = static_cast<int>(g.index);
                        LOG_INFO("[NanoVDB] Auto-selected first compatible grid: '{}' (index {})",
                                 g.name.empty() ? "<unnamed>" : g.name, selectedIndex);
                        break;
                    }
                }
            }

            // Priority 3: First grid (with warning)
            if (selectedIndex < 0) {
                selectedIndex = 0;
                LOG_WARN("[NanoVDB] No compatible grids found, using first grid (index 0)");
            }
        }

        m_selectedGridIndex = static_cast<uint32_t>(selectedIndex);

        // Load the selected grid
        LOG_INFO("[NanoVDB] Loading grid index {}...", selectedIndex);
        nanovdb::GridHandle<nanovdb::HostBuffer> handle =
            nanovdb::io::readGrid<nanovdb::HostBuffer>(filepath, selectedIndex, 1);

        if (!handle) {
            m_lastError = "Failed to read grid from file (handle is empty)";
            LOG_ERROR("[NanoVDB] {}", m_lastError);
            return false;
        }

        // Get buffer info
        uint64_t bufferSize = handle.bufferSize();
        const void* bufferData = handle.data();

        LOG_INFO("[NanoVDB] Grid loaded: {} bytes ({:.2f} MB)",
                 bufferSize, bufferSize / (1024.0 * 1024.0));

        // ================================================================
        // PHASE 1: ENHANCED GRID METADATA EXTRACTION
        // ================================================================
        // Access raw GridData to get grid type regardless of template type
        const nanovdb::GridData* gridData = reinterpret_cast<const nanovdb::GridData*>(bufferData);
        if (gridData) {
            // Extract grid type (critical for shader compatibility)
            m_gridType = static_cast<uint32_t>(gridData->mGridType);
            m_gridTypeName = GridTypeToString(m_gridType);

            LOG_INFO("[NanoVDB] ----------------------------------------");
            LOG_INFO("[NanoVDB] GRID METADATA (Phase 1 Diagnostics)");
            LOG_INFO("[NanoVDB] ----------------------------------------");
            LOG_INFO("[NanoVDB]   Grid Type ID: {} ({})", m_gridType, m_gridTypeName);
            LOG_INFO("[NanoVDB]   Grid Class: {}", static_cast<int>(gridData->mGridClass));

            // Extract world bounding box from raw GridData
            auto worldBBox = gridData->mWorldBBox;
            m_gridWorldMin = {
                static_cast<float>(worldBBox.mCoord[0][0]),
                static_cast<float>(worldBBox.mCoord[0][1]),
                static_cast<float>(worldBBox.mCoord[0][2])
            };
            m_gridWorldMax = {
                static_cast<float>(worldBBox.mCoord[1][0]),
                static_cast<float>(worldBBox.mCoord[1][1]),
                static_cast<float>(worldBBox.mCoord[1][2])
            };

            LOG_INFO("[NanoVDB]   World Bounds: ({:.2f}, {:.2f}, {:.2f}) to ({:.2f}, {:.2f}, {:.2f})",
                     m_gridWorldMin.x, m_gridWorldMin.y, m_gridWorldMin.z,
                     m_gridWorldMax.x, m_gridWorldMax.y, m_gridWorldMax.z);

            // Calculate bounds size for scale diagnostics
            float sizeX = m_gridWorldMax.x - m_gridWorldMin.x;
            float sizeY = m_gridWorldMax.y - m_gridWorldMin.y;
            float sizeZ = m_gridWorldMax.z - m_gridWorldMin.z;
            LOG_INFO("[NanoVDB]   Bounds Size: {:.2f} x {:.2f} x {:.2f} units", sizeX, sizeY, sizeZ);

            // Check if bounds are suspiciously small (common Blender issue)
            if (sizeX < 10.0f && sizeY < 10.0f && sizeZ < 10.0f) {
                LOG_WARN("[NanoVDB]   WARNING: Bounds are very small (<10 units)!");
                LOG_WARN("[NanoVDB]   This is common for Blender exports. Use ScaleGridBounds() to enlarge.");
            }
        }

        // Try to access as float grid (most common for density fields)
        const nanovdb::NanoGrid<float>* floatGrid = handle.grid<float>();
        if (floatGrid) {
            m_gridName = floatGrid->gridName();
            LOG_INFO("[NanoVDB]   Grid Name: '{}'", m_gridName.empty() ? "<unnamed>" : m_gridName);

            // Get voxel size
            float voxelSize = static_cast<float>(floatGrid->voxelSize()[0]);
            LOG_INFO("[NanoVDB]   Voxel Size: {:.6f}", voxelSize);

            // Get active voxel count
            uint64_t activeVoxels = floatGrid->activeVoxelCount();
            m_voxelCount = static_cast<uint32_t>(std::min(activeVoxels, static_cast<uint64_t>(UINT32_MAX)));
            LOG_INFO("[NanoVDB]   Active Voxels: {}", activeVoxels);

            // Check for empty grid
            if (activeVoxels == 0) {
                LOG_WARN("[NanoVDB]   WARNING: Grid has ZERO active voxels!");
                LOG_WARN("[NanoVDB]   The volume will render as completely transparent.");
            }
        } else {
            // Non-float grid - try to extract name from raw data
            LOG_WARN("[NanoVDB]   Grid is NOT a float type (type={}: {})", m_gridType, m_gridTypeName);

            // Try to get grid name from nanovdb::GridData (it stores the name)
            // The grid name is stored as a C-string at offset gridData->mGridName
            m_gridName = "unknown";  // Default if we can't extract

            // Check shader compatibility
            if (m_gridType == GRID_TYPE_HALF || m_gridType == GRID_TYPE_FP16) {
                LOG_WARN("[NanoVDB]   This is a 16-bit float grid (common from Blender Half/Mini precision).");
                LOG_WARN("[NanoVDB]   Phase 3 will add shader support for HALF/FP16 grids.");
                LOG_WARN("[NanoVDB]   CURRENT STATUS: Shader will render this as DENSITY=0 (invisible).");
            } else {
                LOG_ERROR("[NanoVDB]   Grid type {} is NOT supported by the shader!", m_gridTypeName);
                LOG_ERROR("[NanoVDB]   Only FLOAT (1), HALF (9), and FP16 (15) grids can be rendered.");
            }
        }

        // ================================================================
        // SHADER COMPATIBILITY CHECK
        // ================================================================
        LOG_INFO("[NanoVDB] ----------------------------------------");
        LOG_INFO("[NanoVDB] SHADER COMPATIBILITY CHECK");
        LOG_INFO("[NanoVDB] ----------------------------------------");

        bool shaderCompatible = (m_gridType == GRID_TYPE_FLOAT);
        bool futureCompatible = (m_gridType == GRID_TYPE_HALF || m_gridType == GRID_TYPE_FP16);

        if (shaderCompatible) {
            LOG_INFO("[NanoVDB]   Status: COMPATIBLE (Float grid)");
        } else if (futureCompatible) {
            LOG_WARN("[NanoVDB]   Status: PARTIALLY COMPATIBLE (HALF/FP16 - needs Phase 3 shader update)");
            LOG_WARN("[NanoVDB]   The shader currently only supports FLOAT grids.");
            LOG_WARN("[NanoVDB]   Grid will load but may render as invisible until shader is updated.");
            // Don't set error - allow loading to continue for testing
        } else {
            m_lastError = "Grid type " + m_gridTypeName + " is not supported. Convert to Float grid.";
            LOG_ERROR("[NanoVDB]   Status: INCOMPATIBLE - {}", m_lastError);
            // Still allow upload for debugging purposes, but warn the user
        }

        LOG_INFO("[NanoVDB] ----------------------------------------");

        // Upload to GPU
        if (!UploadGridToGPU(bufferData, bufferSize)) {
            m_lastError = "Failed to upload grid to GPU";
            LOG_ERROR("[NanoVDB] {}", m_lastError);
            return false;
        }

        // Calculate and store original grid center (needed for repositioning)
        m_originalGridCenter = {
            (m_gridWorldMin.x + m_gridWorldMax.x) * 0.5f,
            (m_gridWorldMin.y + m_gridWorldMax.y) * 0.5f,
            (m_gridWorldMin.z + m_gridWorldMax.z) * 0.5f
        };

        // Set procedural sphere center to grid center for fallback
        m_sphereCenter = m_originalGridCenter;
        float dx = m_gridWorldMax.x - m_gridWorldMin.x;
        float dy = m_gridWorldMax.y - m_gridWorldMin.y;
        float dz = m_gridWorldMax.z - m_gridWorldMin.z;
        m_sphereRadius = 0.5f * std::sqrt(dx*dx + dy*dy + dz*dz);

        // Reset grid offset and scale since this is a fresh load
        m_gridOffset = { 0.0f, 0.0f, 0.0f };
        m_gridScale = 1.0f;  // Reset cumulative scale factor

        m_hasGrid = true;
        m_hasFileGrid = true;  // Mark as file-loaded grid
        m_gridSizeBytes = bufferSize;
        m_loadedFilePath = filepath;

        LOG_INFO("[NanoVDB] ========================================");
        LOG_INFO("[NanoVDB] Grid loaded and uploaded successfully!");
        LOG_INFO("[NanoVDB] ========================================");
        return true;

    } catch (const std::exception& e) {
        m_lastError = std::string("Exception: ") + e.what();
        LOG_ERROR("[NanoVDB] Failed to load grid: {}", e.what());
        return false;
    }
}

bool NanoVDBSystem::UploadGridToGPU(const void* gridData, uint64_t sizeBytes) {
    if (!gridData || sizeBytes == 0) {
        LOG_ERROR("[NanoVDB] UploadGridToGPU: Invalid grid data");
        return false;
    }

    LOG_INFO("[NanoVDB] Uploading {} bytes ({:.2f} MB) to GPU...",
             sizeBytes, sizeBytes / (1024.0 * 1024.0));

    // Release any existing grid buffer
    m_gridBuffer.Reset();

    // Create a default heap buffer for the grid data
    // We use a ByteAddressBuffer (raw buffer) for flexible access
    D3D12_HEAP_PROPERTIES defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeBytes);

    HRESULT hr = m_device->GetDevice()->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,  // Initial state for upload
        nullptr,
        IID_PPV_ARGS(&m_gridBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create GPU buffer for grid data (HRESULT: 0x{:08X})", hr);
        return false;
    }

    m_gridBuffer->SetName(L"NanoVDB Grid Buffer");

    // Create an upload buffer to transfer data to GPU
    ComPtr<ID3D12Resource> uploadBuffer;
    D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    hr = m_device->GetDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create upload buffer (HRESULT: 0x{:08X})", hr);
        m_gridBuffer.Reset();
        return false;
    }

    // Map the upload buffer and copy data
    void* mappedData = nullptr;
    hr = uploadBuffer->Map(0, nullptr, &mappedData);
    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to map upload buffer (HRESULT: 0x{:08X})", hr);
        m_gridBuffer.Reset();
        return false;
    }

    memcpy(mappedData, gridData, sizeBytes);
    uploadBuffer->Unmap(0, nullptr);

    // Create a command list to perform the copy
    ComPtr<ID3D12CommandAllocator> cmdAllocator;
    ComPtr<ID3D12GraphicsCommandList> cmdList;

    hr = m_device->GetDevice()->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&cmdAllocator)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create command allocator for upload");
        m_gridBuffer.Reset();
        return false;
    }

    hr = m_device->GetDevice()->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        cmdAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&cmdList)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create command list for upload");
        m_gridBuffer.Reset();
        return false;
    }

    // Copy from upload buffer to default buffer
    cmdList->CopyBufferRegion(m_gridBuffer.Get(), 0, uploadBuffer.Get(), 0, sizeBytes);

    // Transition to shader resource state
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_gridBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &barrier);

    // Close and execute
    cmdList->Close();

    ID3D12CommandList* cmdLists[] = { cmdList.Get() };
    m_device->GetCommandQueue()->ExecuteCommandLists(1, cmdLists);

    // Wait for upload to complete
    // Create a fence to synchronize
    ComPtr<ID3D12Fence> fence;
    hr = m_device->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create fence for upload sync");
        // Continue anyway - buffer might still work
    } else {
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        m_device->GetCommandQueue()->Signal(fence.Get(), 1);
        if (fence->GetCompletedValue() < 1) {
            fence->SetEventOnCompletion(1, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        CloseHandle(fenceEvent);
    }

    // Create SRV for the grid buffer as StructuredBuffer<uint>
    // PNanoVDB.h expects StructuredBuffer<uint> (pnanovdb_buf_t) in HLSL mode
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;  // Structured buffers use UNKNOWN format
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = static_cast<UINT>(sizeBytes / 4);  // Number of uint32 elements
    srvDesc.Buffer.StructureByteStride = sizeof(uint32_t);  // 4 bytes per element
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;  // Not a raw buffer

    // Allocate descriptor from ResourceManager
    m_gridBufferSRV_CPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_device->GetDevice()->CreateShaderResourceView(m_gridBuffer.Get(), &srvDesc, m_gridBufferSRV_CPU);
    m_gridBufferSRV_GPU = m_resources->GetGPUHandle(m_gridBufferSRV_CPU);

    LOG_INFO("[NanoVDB] Grid buffer uploaded successfully!");
    LOG_INFO("[NanoVDB] Buffer elements: {} uint32s", sizeBytes / 4);

    return true;
}

bool NanoVDBSystem::CreateComputePipeline() {
    LOG_INFO("[NanoVDB] Creating compute pipeline...");

    // Root signature with:
    // [0] CBV - NanoVDBConstants (b0)
    // [1] SRV table - Grid buffer (t0) - NanoVDB raw data
    // [2] SRV table - Light buffer (t1)
    // [3] SRV table - Depth buffer (t2)
    // [4] UAV table - Output texture (u0)

    CD3DX12_DESCRIPTOR_RANGE1 gridSrvRange;
    gridSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // t0: grid buffer

    CD3DX12_DESCRIPTOR_RANGE1 lightSrvRange;
    lightSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // t1: light buffer

    CD3DX12_DESCRIPTOR_RANGE1 depthSrvRange;
    depthSrvRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // t2: depth buffer

    CD3DX12_DESCRIPTOR_RANGE1 uavRange;
    uavRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // u0: output

    CD3DX12_ROOT_PARAMETER1 rootParams[5];
    rootParams[0].InitAsConstantBufferView(0);               // b0: constants
    rootParams[1].InitAsDescriptorTable(1, &gridSrvRange);   // t0: grid
    rootParams[2].InitAsDescriptorTable(1, &lightSrvRange);  // t1: lights
    rootParams[3].InitAsDescriptorTable(1, &depthSrvRange);  // t2: depth
    rootParams[4].InitAsDescriptorTable(1, &uavRange);       // u0: output

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc;
    rootSigDesc.Init_1_1(5, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> signature, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rootSigDesc, &signature, &error);
    if (FAILED(hr)) {
        if (error) {
            LOG_ERROR("[NanoVDB] Root signature serialization failed: {}",
                      static_cast<const char*>(error->GetBufferPointer()));
        }
        return false;
    }

    hr = m_device->GetDevice()->CreateRootSignature(
        0,
        signature->GetBufferPointer(),
        signature->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSignature)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create root signature");
        return false;
    }

    // Load compute shader
    ComPtr<ID3DBlob> shaderBlob;
    hr = D3DReadFileToBlob(L"build/bin/Debug/shaders/volumetric/nanovdb_raymarch.dxil", &shaderBlob);
    if (FAILED(hr)) {
        LOG_INFO("[NanoVDB] Shader not found at build path, trying alternate...");
        hr = D3DReadFileToBlob(L"shaders/volumetric/nanovdb_raymarch.dxil", &shaderBlob);
        if (FAILED(hr)) {
            LOG_INFO("[NanoVDB] Shader not found, pipeline creation deferred");
            // This is expected on first run before shader is compiled
            return true;  // Return true to allow initialization to complete
        }
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.CS = { shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize() };

    hr = m_device->GetDevice()->CreateComputePipelineState(
        &psoDesc,
        IID_PPV_ARGS(&m_pipelineState)
    );

    if (FAILED(hr)) {
        LOG_ERROR("[NanoVDB] Failed to create compute pipeline state");
        return false;
    }

    LOG_INFO("[NanoVDB] Compute pipeline created successfully");
    return true;
}

void NanoVDBSystem::Render(
    ID3D12GraphicsCommandList* commandList,
    const DirectX::XMMATRIX& viewProj,
    const DirectX::XMFLOAT3& cameraPos,
    D3D12_GPU_DESCRIPTOR_HANDLE outputUAV,
    D3D12_GPU_DESCRIPTOR_HANDLE lightSRV,
    D3D12_GPU_DESCRIPTOR_HANDLE depthSRV,
    uint32_t lightCount,
    ID3D12DescriptorHeap* descriptorHeap,
    uint32_t renderWidth,
    uint32_t renderHeight,
    float time) {

    if (!m_enabled || !m_hasGrid || !m_pipelineState) {
        return;  // Nothing to render
    }

    // Update constant buffer with actual render dimensions (important for DLSS)
    NanoVDBConstants constants = {};

    // Calculate inverse view-projection
    // NOTE: DirectXMath is row-major, HLSL with row_major keyword expects row-major
    // Do NOT transpose - store directly for row_major HLSL compatibility
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);
    XMStoreFloat4x4(&constants.invViewProj, invViewProj);

    constants.cameraPos = cameraPos;
    constants.densityScale = m_densityScale;
    constants.gridWorldMin = m_gridWorldMin;
    constants.emissionStrength = m_emissionStrength;
    constants.gridWorldMax = m_gridWorldMax;
    constants.absorptionCoeff = m_absorptionCoeff;
    constants.sphereCenter = m_sphereCenter;
    constants.scatteringCoeff = m_scatteringCoeff;
    constants.sphereRadius = m_sphereRadius;
    constants.maxRayDistance = m_maxRayDistance;
    constants.stepSize = m_stepSize;
    constants.lightCount = lightCount;
    constants.screenWidth = renderWidth;   // Use actual render dimensions
    constants.screenHeight = renderHeight; // (may differ from native due to DLSS)
    constants.time = time;                 // Animation time for procedural effects
    constants.debugMode = m_debugMode ? 1 : 0;
    constants.useGridBuffer = (m_hasFileGrid && m_gridBuffer) ? 1 : 0;
    constants.gridOffset = m_gridOffset;  // Transform sampling position back to original grid space
    constants.gridType = m_gridType;      // Pass grid type to shader (FLOAT=1, HALF=9, FP16=15)
    constants.materialType = static_cast<uint32_t>(m_materialType);  // Material behavior (SMOKE=0, FIRE=1, etc.)
    constants.enableShadows = m_enableShadows ? 1 : 0;  // Phase 1 shadow rays
    constants.shadowSteps = m_shadowSteps;              // Shadow march steps (8-32)
    constants.albedo = m_albedo;          // Base color for scattering/emission tint
    constants.gridScale = m_gridScale;    // Cumulative scale factor for coordinate transform
    constants.originalGridCenter = m_originalGridCenter;  // Original grid center before scaling

    // Map and update constants
    void* mappedData = nullptr;
    m_constantBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, &constants, sizeof(constants));
    m_constantBuffer->Unmap(0, nullptr);

    // Set descriptor heap (required for shader-visible descriptors)
    ID3D12DescriptorHeap* heaps[] = { descriptorHeap };
    commandList->SetDescriptorHeaps(1, heaps);

    // Set pipeline state
    commandList->SetComputeRootSignature(m_rootSignature.Get());
    commandList->SetPipelineState(m_pipelineState.Get());

    // Bind resources:
    // [0] CBV - NanoVDBConstants (b0)
    // [1] SRV table - Grid buffer (t0) - NanoVDB raw data
    // [2] SRV table - Light buffer (t1)
    // [3] SRV table - Depth buffer (t2)
    // [4] UAV table - Output texture (u0)
    commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Bind grid buffer if we have one loaded from file
    if (m_gridBuffer) {
        commandList->SetComputeRootDescriptorTable(1, m_gridBufferSRV_GPU);
    }

    commandList->SetComputeRootDescriptorTable(2, lightSRV);
    commandList->SetComputeRootDescriptorTable(3, depthSRV);
    commandList->SetComputeRootDescriptorTable(4, outputUAV);

    // Dispatch compute shader using actual render dimensions
    uint32_t groupsX = (renderWidth + 7) / 8;
    uint32_t groupsY = (renderHeight + 7) / 8;
    commandList->Dispatch(groupsX, groupsY, 1);

    // Log first few frames for verification
    static int renderCount = 0;
    renderCount++;
    if (renderCount <= 3) {
        LOG_INFO("[NanoVDB] Rendered frame (groups: {} x {}, res: {}x{}, time: {:.2f})",
                 groupsX, groupsY, renderWidth, renderHeight, time);
    }
}

// ============================================================================
// ANIMATION SUPPORT IMPLEMENTATION
// ============================================================================

bool NanoVDBSystem::LoadAnimationSequence(const std::vector<std::string>& filepaths) {
    if (filepaths.empty()) {
        LOG_ERROR("[NanoVDB] LoadAnimationSequence: No filepaths provided");
        return false;
    }

    LOG_INFO("[NanoVDB] Loading animation sequence with {} frames...", filepaths.size());

    // Clear existing animation frames
    m_animFrames.clear();
    m_animCurrentFrame = 0;
    m_animAccumulator = 0.0f;

    size_t loadedCount = 0;
    auto* d3dDevice = m_device->GetDevice();

    for (size_t i = 0; i < filepaths.size(); ++i) {
        const auto& filepath = filepaths[i];

        try {
            // Read file
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                LOG_WARN("[NanoVDB] Failed to open frame {}: {}", i, filepath);
                continue;
            }

            std::streamsize fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> fileData(fileSize);
            if (!file.read(fileData.data(), fileSize)) {
                LOG_WARN("[NanoVDB] Failed to read frame {}: {}", i, filepath);
                continue;
            }

            const void* bufferData = fileData.data();
            uint64_t bufferSize = static_cast<uint64_t>(fileSize);

            // Create GPU buffer for this frame
            AnimationFrame frame;
            frame.sizeBytes = bufferSize;

            D3D12_HEAP_PROPERTIES defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

            HRESULT hr = d3dDevice->CreateCommittedResource(
                &defaultHeapProps,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&frame.buffer));

            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create GPU buffer for frame {}", i);
                continue;
            }

            // Create upload buffer
            D3D12_HEAP_PROPERTIES uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
            ComPtr<ID3D12Resource> uploadBuffer;
            hr = d3dDevice->CreateCommittedResource(
                &uploadHeapProps,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer));

            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create upload buffer for frame {}", i);
                continue;
            }

            // Copy data to upload buffer
            void* mappedData = nullptr;
            uploadBuffer->Map(0, nullptr, &mappedData);
            memcpy(mappedData, bufferData, bufferSize);
            uploadBuffer->Unmap(0, nullptr);

            // Create command allocator and list for upload (same pattern as UploadGridToGPU)
            ComPtr<ID3D12CommandAllocator> cmdAllocator;
            ComPtr<ID3D12GraphicsCommandList> cmdList;

            hr = d3dDevice->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&cmdAllocator));
            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create command allocator for frame {}", i);
                continue;
            }

            hr = d3dDevice->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                cmdAllocator.Get(), nullptr,
                IID_PPV_ARGS(&cmdList));
            if (FAILED(hr)) {
                LOG_WARN("[NanoVDB] Failed to create command list for frame {}", i);
                continue;
            }

            // Copy and transition
            cmdList->CopyBufferRegion(frame.buffer.Get(), 0, uploadBuffer.Get(), 0, bufferSize);

            CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
                frame.buffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmdList->ResourceBarrier(1, &barrier);
            cmdList->Close();

            // Execute and wait
            ID3D12CommandList* cmdLists[] = { cmdList.Get() };
            m_device->GetCommandQueue()->ExecuteCommandLists(1, cmdLists);

            // Create fence and wait for upload
            ComPtr<ID3D12Fence> fence;
            hr = d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
            if (SUCCEEDED(hr)) {
                HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                m_device->GetCommandQueue()->Signal(fence.Get(), 1);
                if (fence->GetCompletedValue() < 1) {
                    fence->SetEventOnCompletion(1, fenceEvent);
                    WaitForSingleObject(fenceEvent, INFINITE);
                }
                CloseHandle(fenceEvent);
            }

            // Create SRV for this frame (StructuredBuffer<uint> like existing code)
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_UNKNOWN;
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Buffer.FirstElement = 0;
            srvDesc.Buffer.NumElements = static_cast<UINT>(bufferSize / sizeof(uint32_t));
            srvDesc.Buffer.StructureByteStride = sizeof(uint32_t);
            srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

            frame.srvCPU = m_resources->AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            d3dDevice->CreateShaderResourceView(frame.buffer.Get(), &srvDesc, frame.srvCPU);
            frame.srvGPU = m_resources->GetGPUHandle(frame.srvCPU);

            m_animFrames.push_back(std::move(frame));
            loadedCount++;

            if (loadedCount % 5 == 0 || i == filepaths.size() - 1) {
                LOG_INFO("[NanoVDB] Loaded {}/{} frames...", loadedCount, filepaths.size());
            }

        } catch (const std::exception& e) {
            LOG_WARN("[NanoVDB] Exception loading frame {}: {}", i, e.what());
            continue;
        }
    }

    if (loadedCount > 0) {
        LOG_INFO("[NanoVDB] Animation loaded: {} frames ({:.1f} MB total)",
                 loadedCount,
                 m_animFrames.size() > 0 ?
                     (m_animFrames[0].sizeBytes * loadedCount) / (1024.0 * 1024.0) : 0.0);

        // Use first frame as current grid
        if (!m_animFrames.empty()) {
            m_gridBuffer = m_animFrames[0].buffer;
            m_gridBufferSRV_GPU = m_animFrames[0].srvGPU;
            m_gridBufferSRV_CPU = m_animFrames[0].srvCPU;
            m_gridSizeBytes = m_animFrames[0].sizeBytes;
            m_hasGrid = true;
            m_hasFileGrid = true;

            // Extract grid bounds from a NON-EMPTY frame using NanoVDB
            // NOTE: Frame 0 is often empty in animation sequences - search for first frame with data
            // Use gridData() to get type-agnostic GridData* - works for ALL grid types
            try {
                bool boundsExtracted = false;

                // Try frames until we find one with valid voxel data
                // Start from frame 1 (skip frame 0 which is often empty), then try others
                std::vector<size_t> framesToTry;
                if (filepaths.size() > 1) framesToTry.push_back(1);  // Try frame 1 first
                if (filepaths.size() > 10) framesToTry.push_back(10); // Then frame 10
                framesToTry.push_back(0);  // Finally try frame 0
                if (filepaths.size() > 50) framesToTry.push_back(50); // Also try mid-sequence

                for (size_t frameIdx : framesToTry) {
                    if (frameIdx >= filepaths.size()) continue;

                    LOG_INFO("[NanoVDB] Trying to extract bounds from frame {}...", frameIdx);

                    nanovdb::GridHandle<nanovdb::HostBuffer> handle =
                        nanovdb::io::readGrid<nanovdb::HostBuffer>(filepaths[frameIdx], 0, 0);
                    if (!handle) continue;

                    const nanovdb::GridData* gridData = handle.gridData();
                    if (!gridData) continue;

                    // Log grid type for debugging (only once)
                    if (!boundsExtracted) {
                        m_gridType = static_cast<uint32_t>(gridData->mGridType);
                        m_gridTypeName = GridTypeToString(m_gridType);
                        LOG_INFO("[NanoVDB] Animation grid type: {} ({})", m_gridType, m_gridTypeName);
                    }

                    // Try to get float grid and check voxel count
                    const nanovdb::NanoGrid<float>* floatGrid = handle.grid<float>();
                    if (!floatGrid) {
                        LOG_WARN("[NanoVDB] Frame {} is not a float grid", frameIdx);
                        continue;
                    }

                    uint64_t voxelCount = floatGrid->activeVoxelCount();
                    if (voxelCount == 0) {
                        LOG_WARN("[NanoVDB] Frame {} has 0 active voxels, skipping...", frameIdx);
                        continue;
                    }

                    LOG_INFO("[NanoVDB] Frame {} has {} active voxels", frameIdx, voxelCount);

                    // Get bounds from gridData->mWorldBBox (matches shader's pnanovdb_grid_world_to_indexf)
                    // NOTE: Using mWorldBBox instead of indexBBox * voxelSize ensures coordinate
                    // consistency between C++ bounds and HLSL sampling transform.
                    // See: docs/PROJECT_STATUS_DEC2025.md for root cause analysis.
                    auto worldBBox = gridData->mWorldBBox;

                    float minX = static_cast<float>(worldBBox.mCoord[0][0]);
                    float minY = static_cast<float>(worldBBox.mCoord[0][1]);
                    float minZ = static_cast<float>(worldBBox.mCoord[0][2]);
                    float maxX = static_cast<float>(worldBBox.mCoord[1][0]);
                    float maxY = static_cast<float>(worldBBox.mCoord[1][1]);
                    float maxZ = static_cast<float>(worldBBox.mCoord[1][2]);

                    // Validate bounds - check for inf/NaN (common in corrupt VDB files)
                    bool boundsValid = std::isfinite(minX) && std::isfinite(minY) && std::isfinite(minZ) &&
                                       std::isfinite(maxX) && std::isfinite(maxY) && std::isfinite(maxZ) &&
                                       maxX > minX && maxY > minY && maxZ > minZ;

                    if (boundsValid) {
                        m_gridWorldMin = { minX, minY, minZ };
                        m_gridWorldMax = { maxX, maxY, maxZ };
                        m_voxelCount = static_cast<uint32_t>(voxelCount);
                        m_gridName = floatGrid->gridName();

                        LOG_INFO("[NanoVDB] Animation bounds from frame {} (mWorldBBox): ({:.2f},{:.2f},{:.2f}) to ({:.2f},{:.2f},{:.2f})",
                                 frameIdx, minX, minY, minZ, maxX, maxY, maxZ);
                        boundsExtracted = true;
                        break;
                    } else {
                        LOG_WARN("[NanoVDB] Frame {} has invalid worldBBox (inf/NaN or zero volume), trying next frame...", frameIdx);
                    }
                }

                if (!boundsExtracted) {
                    // Fallback: use a reasonable default size centered at origin
                    LOG_WARN("[NanoVDB] Could not extract valid bounds from any frame - using default 200x200x200");
                    m_gridWorldMin = { -100.0f, -100.0f, -100.0f };
                    m_gridWorldMax = { 100.0f, 100.0f, 100.0f };
                }

                // Store original center for repositioning
                m_originalGridCenter = {
                    (m_gridWorldMin.x + m_gridWorldMax.x) * 0.5f,
                    (m_gridWorldMin.y + m_gridWorldMax.y) * 0.5f,
                    (m_gridWorldMin.z + m_gridWorldMax.z) * 0.5f
                };
                m_gridOffset = { 0.0f, 0.0f, 0.0f };
                m_gridScale = 1.0f;  // Reset cumulative scale factor

                // Log final bounds
                float sizeX = m_gridWorldMax.x - m_gridWorldMin.x;
                float sizeY = m_gridWorldMax.y - m_gridWorldMin.y;
                float sizeZ = m_gridWorldMax.z - m_gridWorldMin.z;
                LOG_INFO("[NanoVDB] Animation bounds size: {:.2f} x {:.2f} x {:.2f} units", sizeX, sizeY, sizeZ);

                // Warn if bounds are very small (in world units, not the huge INT_MAX values)
                if (sizeX < 10.0f && sizeY < 10.0f && sizeZ < 10.0f && sizeX > 0 && sizeY > 0 && sizeZ > 0) {
                    LOG_WARN("[NanoVDB] WARNING: Animation bounds are very small (<10 units)!");
                    LOG_WARN("[NanoVDB] Use ScaleGridBounds() or the Scale slider to enlarge.");
                }
            } catch (const std::exception& e) {
                LOG_WARN("[NanoVDB] Could not extract animation bounds: {}", e.what());
                // Use default bounds on error
                m_gridWorldMin = { -100.0f, -100.0f, -100.0f };
                m_gridWorldMax = { 100.0f, 100.0f, 100.0f };
            }
        }

        // AUTO-START animation playback on load
        // This provides better UX - users expect animation to play immediately
        m_animPlaying = true;
        m_animCurrentFrame = 0;
        m_animAccumulator = 0.0f;
        LOG_INFO("[NanoVDB] Animation auto-started (FPS: {:.1f}, Loop: {})",
                 m_animFPS, m_animLoop ? "ON" : "OFF");

        return true;
    }

    LOG_ERROR("[NanoVDB] Failed to load any animation frames");
    return false;
}

size_t NanoVDBSystem::LoadAnimationFromDirectory(const std::string& directory, const std::string& pattern) {
    namespace fs = std::filesystem;

    std::vector<std::string> filepaths;

    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".nvdb") {
                    filepaths.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[NanoVDB] Failed to enumerate directory {}: {}", directory, e.what());
        return 0;
    }

    if (filepaths.empty()) {
        LOG_WARN("[NanoVDB] No .nvdb files found in {}", directory);
        return 0;
    }

    // Sort files by name (assumes numbered sequence like smoke_0001.nvdb)
    std::sort(filepaths.begin(), filepaths.end());

    LOG_INFO("[NanoVDB] Found {} .nvdb files in {}", filepaths.size(), directory);

    if (LoadAnimationSequence(filepaths)) {
        return m_animFrames.size();
    }
    return 0;
}

void NanoVDBSystem::SetAnimationFrame(size_t frame) {
    if (m_animFrames.empty()) return;

    frame = frame % m_animFrames.size();  // Clamp/wrap
    m_animCurrentFrame = frame;

    // Update current grid buffer to this frame
    m_gridBuffer = m_animFrames[frame].buffer;
    m_gridBufferSRV_GPU = m_animFrames[frame].srvGPU;
    m_gridBufferSRV_CPU = m_animFrames[frame].srvCPU;
    m_gridSizeBytes = m_animFrames[frame].sizeBytes;
}

void NanoVDBSystem::UpdateAnimation(float deltaTime) {
    if (!m_animPlaying || m_animFrames.empty() || m_animFPS <= 0.0f) {
        return;
    }

    m_animAccumulator += deltaTime;
    float frameTime = 1.0f / m_animFPS;

    while (m_animAccumulator >= frameTime) {
        m_animAccumulator -= frameTime;

        size_t nextFrame = m_animCurrentFrame + 1;

        if (nextFrame >= m_animFrames.size()) {
            if (m_animLoop) {
                nextFrame = 0;
            } else {
                m_animPlaying = false;
                return;
            }
        }

        SetAnimationFrame(nextFrame);
    }
}
