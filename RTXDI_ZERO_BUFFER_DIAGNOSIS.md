# RTXDI Zero Buffer Diagnosis - Complete Investigation

**Status**: Light grid buffer all zeros despite successful initialization and dispatch
**Date**: 2025-10-18
**Milestone**: 2.3 (Light Grid Buffer Validation) - BLOCKED

---

## Evidence Summary

### What's Working ✅

1. **RTXDI Initialization**: Successful (logs confirm)
   ```
   [21:18:32] [INFO] RTXDI Lighting System initialized successfully!
   [21:18:32] [INFO]   Light grid: 30x30x30 cells (27,000 total)
   [21:18:32] [INFO]   Ready for 100+ light scaling
   ```

2. **Multi-Light System**: 13 lights created
   ```
   [21:18:32] [INFO] Initialized multi-light system: 13 lights
   ```

3. **UpdateLightGrid() Called Every Frame**:
   ```
   [21:18:32] [INFO] RTXDI Light Grid updated (frame 0, 13 lights)
   [21:18:32] [INFO] RTXDI Light Grid updated (frame 1, 13 lights)
   [21:18:32] [INFO] RTXDI Light Grid updated (frame 2, 13 lights)
   ...
   ```

4. **No Errors, No Crashes**: 120 FPS stable performance

5. **Code Structure Correct**:
   - `Application.cpp:204` - InitializeLights() creates 13 lights
   - `Application.cpp:232` - RTXDI initialized AFTER lights created
   - `Application.cpp:396` - UpdateLightGrid() called with m_lights.data() and m_lights.size()
   - `RTXDILightingSystem.cpp:233-345` - Upload and dispatch implementation looks correct

### What's Failing ❌

1. **Light Grid Buffer All Zeros**: PIX capture shows g_lightGrid = all zeros
2. **Lights Buffer All Zeros**: PIX capture shows g_lights = all zeros (CRITICAL!)

---

## Root Cause Hypothesis

**CRITICAL FINDING**: The lights buffer (g_lights) uploaded to GPU is all zeros!

### Evidence from Log

Line 89 (during early Gaussian renderer initialization, BEFORE multi-light system):
```
[21:18:32] [INFO] Updated light buffer: 0 lights
```

Line 162 (first frame, AFTER multi-light init):
```
[21:18:32] [INFO] Updated light buffer: 13 lights
```

**Analysis**: The "0 lights" log is harmless (from Gaussian renderer initialization before multi-light system exists). The real issue is that the **GPU buffer g_lights contains zeros despite successful upload logs**.

### Possible Causes (Priority Order)

1. **Upload Heap Stale Data** (MOST LIKELY)
   - Upload heap allocated from ResourceManager
   - Data copied to upload heap
   - CopyBufferRegion executed
   - **BUT**: Upload heap might not be flushed before copy
   - **OR**: Command list not executed before compute shader reads

2. **Command List Synchronization**
   - UpdateLightGrid() uses same command list as rendering
   - Barrier transitions may be out of order
   - Upload copy might not finish before compute shader dispatch

3. **Resource State Mismatch**
   - Light buffer transitions: COMMON → COPY_DEST → NON_PIXEL_SHADER_RESOURCE
   - Compute shader expects t0 (SRV state)
   - Barrier might not have taken effect

4. **memcpy to Upload Heap Failed Silently**
   - uploadAllocation.cpuAddress might be invalid
   - No validation that memcpy succeeded
   - Upload heap might be read-only or unmapped

---

## Validation Plan

### Step 1: Add Debug Logging to UpdateLightGrid() (5 min)

**File**: `src/lighting/RTXDILightingSystem.cpp:252`

**Add after memcpy**:
```cpp
// Copy light data to upload heap
memcpy(uploadAllocation.cpuAddress, lights, lightDataSize);

// VALIDATION: Verify data was copied correctly
const float* uploadedData = reinterpret_cast<const float*>(uploadAllocation.cpuAddress);
LOG_INFO("  Uploaded light 0: pos=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
         uploadedData[0], uploadedData[1], uploadedData[2], uploadedData[3]);

// VALIDATION: Verify source data is non-zero
const float* sourceData = reinterpret_cast<const float*>(lights);
LOG_INFO("  Source light 0: pos=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
         sourceData[0], sourceData[1], sourceData[2], sourceData[3]);
```

**Expected Output**:
- If source data is zeros → `m_lights` vector is not populated correctly
- If upload data is zeros → upload heap allocation failed
- If both are non-zero → Copy/barrier/sync issue

### Step 2: Force GPU Sync After Upload (10 min)

**File**: `src/lighting/RTXDILightingSystem.cpp:275`

**Add after transition to NON_PIXEL_SHADER_RESOURCE**:
```cpp
// Transition to NON_PIXEL_SHADER_RESOURCE for compute shader read
barrier = CD3DX12_RESOURCE_BARRIER::Transition(
    m_lightBuffer.Get(),
    D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
);
commandList->ResourceBarrier(1, &barrier);

// **NEW**: Force GPU flush to ensure upload completes before compute shader
auto hr = commandList->Close();
if (FAILED(hr)) {
    LOG_ERROR("Failed to close command list: 0x{:08X}", static_cast<uint32_t>(hr));
    return;
}

m_device->ExecuteCommandList();
m_device->WaitForGPU();  // **CRITICAL**: Wait for upload to finish

// Re-open command list for compute dispatch
m_device->ResetCommandList();
commandList = m_device->GetCommandList();  // Get fresh command list
```

**Warning**: This will hurt performance (GPU stall every frame), but will prove if synchronization is the issue.

### Step 3: Dump Buffers with Ctrl+D (2 min)

**Action**:
1. Run app with `--rtxdi` flag
2. Wait 10 frames
3. Press **Ctrl+D** to trigger buffer dump
4. Check `PIX/buffer_dumps/frame_XXX/`

**Expected Files**:
- `g_lights.bin` (512 bytes = 16 × 32 bytes)
- `g_lightGrid.bin` (3,456,000 bytes = 27,000 × 128 bytes)
- `metadata.json`

**Run Validation Script**:
```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
python PIX/scripts/validate_rtxdi_buffers.py PIX/buffer_dumps/frame_XXX
```

**Expected Output**:
```
=== Validating Lights Buffer: PIX/buffer_dumps/frame_XXX/g_lights.bin ===
Buffer size: 512 bytes (16 lights max)

Light 0:
  Position: (150.00, 0.00, 0.00)   ← Should be NON-ZERO!
  Intensity: 8.00
  Color: (1.000, 0.600, 0.300)
  Radius: 200.00

Light 1:
  Position: (128.64, 79.55, 0.00)
  Intensity: 8.00
  ...

=== Summary: 13 valid lights out of 16 ===
✅ SUCCESS: All 13 lights present
```

**If lights buffer is all zeros**:
- Problem is in upload path (Step 1-2 needed)

**If lights buffer is valid but light grid is zeros**:
- Problem is in compute shader dispatch or resource binding

### Step 4: Validate Compute Shader Root Signature Binding (5 min)

**File**: `src/lighting/RTXDILightingSystem.cpp:313-319`

**Add validation logs**:
```cpp
// 4. Set grid constants (b0)
struct GridConstants {
    // ... (existing code)
} constants;

constants.gridCellsX = GRID_CELLS_X;
constants.gridCellsY = GRID_CELLS_Y;
constants.gridCellsZ = GRID_CELLS_Z;
constants.lightCount = lightCount;
constants.worldMin = WORLD_MIN;
constants.worldMax = WORLD_MAX;
constants.cellSize = CELL_SIZE;
constants.maxLightsPerCell = MAX_LIGHTS_PER_CELL;

LOG_INFO("  Compute constants: gridCells=({},{},{}), lightCount={}, cellSize={:.1f}",
         constants.gridCellsX, constants.gridCellsY, constants.gridCellsZ,
         constants.lightCount, constants.cellSize);

commandList->SetComputeRoot32BitConstants(0, 8, &constants, 0);

// 5. Bind light buffer SRV (t0)
D3D12_GPU_VIRTUAL_ADDRESS lightBufferGPU = m_lightBuffer->GetGPUVirtualAddress();
LOG_INFO("  Light buffer GPU address: 0x{:016X}", lightBufferGPU);
commandList->SetComputeRootShaderResourceView(1, lightBufferGPU);

// 6. Bind light grid UAV (u0)
D3D12_GPU_VIRTUAL_ADDRESS lightGridGPU = m_lightGridBuffer->GetGPUVirtualAddress();
LOG_INFO("  Light grid GPU address: 0x{:016X}", lightGridGPU);
commandList->SetComputeRootUnorderedAccessView(2, lightGridGPU);
```

**Expected Output**:
```
[INFO]   Compute constants: gridCells=(30,30,30), lightCount=13, cellSize=20.0
[INFO]   Light buffer GPU address: 0x00000265AB3C0000  ← Non-null
[INFO]   Light grid GPU address: 0x00000265AB800000   ← Non-null
```

**If addresses are null (0x0)**:
- Buffers not created correctly (should have failed earlier)

### Step 5: Add PIX Event Markers (2 min)

**File**: `src/lighting/RTXDILightingSystem.cpp:238-345`

**Wrap UpdateLightGrid() with PIX markers**:
```cpp
void RTXDILightingSystem::UpdateLightGrid(const void* lights, uint32_t lightCount,
                                          ID3D12GraphicsCommandList* commandList) {
    // ... (existing early returns)

    PIXBeginEvent(commandList, PIX_COLOR_INDEX(1), "RTXDI Light Grid Build");

    // === Milestone 2.2 Implementation: Light Grid Build ===

    // 1. Upload lights to GPU buffer (CPU → GPU) using ResourceManager
    PIXBeginEvent(commandList, PIX_COLOR_INDEX(2), "Upload Lights to GPU");
    {
        // ... (existing upload code)
    }
    PIXEndEvent(commandList);  // Upload Lights

    // 2. Transition light grid to UNORDERED_ACCESS
    PIXBeginEvent(commandList, PIX_COLOR_INDEX(3), "Resource Barriers");
    {
        // ... (existing barrier code)
    }
    PIXEndEvent(commandList);  // Resource Barriers

    // 3-7. Compute shader dispatch
    PIXBeginEvent(commandList, PIX_COLOR_INDEX(4), "Compute Dispatch (Light Grid)");
    {
        // ... (existing dispatch code)
    }
    PIXEndEvent(commandList);  // Compute Dispatch

    PIXEndEvent(commandList);  // RTXDI Light Grid Build
}
```

**Usage**: Open PIX capture, verify dispatch executes and see GPU time breakdown

---

## Action Plan (Execution Order)

### Immediate Actions (30 min total)

1. **Add Debug Logging** (Step 1) - Verify data is non-zero at upload time
2. **Add GPU Sync** (Step 2) - Prove synchronization hypothesis
3. **Run App with RTXDI** - Collect logs with new validation
4. **Dump Buffers** (Step 3) - Ctrl+D, run validation script

### Expected Outcomes

**Scenario A**: Lights buffer dump shows all zeros
- **Root Cause**: Upload heap allocation or memcpy failed
- **Fix**: Replace upload heap path with explicit upload buffer creation
- **Time**: 30 min

**Scenario B**: Lights buffer dump shows valid data, light grid still zeros
- **Root Cause**: Compute shader didn't execute or binding failed
- **Fix**: Check dispatch parameters, root signature binding
- **Time**: 45 min

**Scenario C**: Adding GPU sync fixes the issue (both buffers valid after sync)
- **Root Cause**: Command list synchronization issue
- **Fix**: Restructure UpdateLightGrid() to use separate command list for upload
- **Time**: 60 min

---

## Quick Start Command Sequence

```bash
# 1. Build and run with RTXDI
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
MSBuild PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64
./build/Debug/PlasmaDX-Clean.exe --rtxdi

# 2. Wait 10 frames, press Ctrl+D

# 3. Run validation script (app will create buffer dumps)
python PIX/scripts/validate_rtxdi_buffers.py PIX/buffer_dumps/frame_XXX

# 4. Check validation output for root cause
```

---

## Files to Modify (If Needed)

### Critical Path (Most Likely Fix)

**File**: `src/lighting/RTXDILightingSystem.cpp`
- **Lines 240-276**: Upload heap path (add validation, add GPU sync)
- **Lines 313-319**: Root signature binding (add logging)

### Secondary Investigation

**File**: `src/core/Application.cpp`
- **Line 396**: UpdateLightGrid() call site (verify m_lights is populated)

**File**: `shaders/rtxdi/light_grid_build_cs.hlsl`
- **Lines 134-166**: Light iteration loop (verify g_lights is bound correctly)

---

## Success Criteria

**Milestone 2.3 Complete** when:
1. ✅ `g_lights.bin` shows 13 non-zero lights
2. ✅ `g_lightGrid.bin` shows non-zero weights in center cells
3. ✅ Cell (15,15,15) has 1-13 light indices
4. ✅ Weights are sorted descending
5. ✅ No NaN/Inf values

**Next Milestone**: 2.4 - DXR Pipeline Setup (callable shaders, state object)

---

## Estimated Time to Resolution

**Best Case**: 30 min (synchronization fix)
**Likely Case**: 1 hour (upload heap issue)
**Worst Case**: 2 hours (shader binding issue)

**Begin with Step 1 (debug logging) immediately.**
