# CRITICAL: GPU Device Removed During PINN Readback

## Error Details

**HRESULT:** 0x887A0005 (decimal: 2289696773)
**Error:** `DXGI_ERROR_DEVICE_REMOVED` - GPU crashed/reset
**Trigger:** `m_readbackBuffer->Map()` call in `ReadbackParticleData()`

## What This Means

The GPU is **crashing** during the particle buffer readback operation. This triggers Windows TDR (Timeout Detection and Recovery) which removes the D3D12 device.

This is NOT a simple mapping error - the GPU is actually hanging/crashing.

## Likely Causes

### 1. Resource State Mismatch (Most Likely)
The particle buffer transition might be invalid:
- Buffer is `UNORDERED_ACCESS` normally
- We transition to `COPY_SOURCE`
- Copy to readback buffer
- Transition back to `UNORDERED_ACCESS`

**Problem:** The buffer might be in use by a compute shader or RT pipeline when we try to transition it.

### 2. UAV Barrier Missing
Between compute shader writing to particle buffer and our copy operation, we might need a UAV barrier.

### 3. Command List Re-use Issue
`m_device->ResetCommandList()` after ExecuteCommandList might be resetting state incorrectly.

### 4. GPU Still Using Buffer
Despite `WaitForGPU()`, the buffer might still be referenced by:
- RT BLAS/TLAS (uses particle buffer for AABBs)
- Gaussian renderer (reads particles)
- Physics compute shader

## Immediate Fix: DON'T Map Readback Buffers Mid-Frame

The current architecture is fundamentally flawed:
```
GPU particles → [CRASH HERE] → CPU inference → GPU upload
```

**Problem:** We're trying to readback a buffer that's actively used by rendering/RT systems.

## Recommended Solution: CPU-Only PINN Mode

When PINN is enabled, keep particles on CPU entirely:

```cpp
// PINN initialization:
1. Do ONE initial readback to get particle state
2. Switch to CPU-only particle storage
3. Disable GPU physics compute shader
4. Each frame:
   - PINN inference on CPU particles
   - Upload CPU particles to GPU (upload-only, safe)
   - GPU renders from upload buffer (read-only, safe)
```

**Benefits:**
- ✅ No mid-frame readback (no crash)
- ✅ Faster (no GPU→CPU transfer)
- ✅ Simpler architecture
- ✅ PINN can run while GPU renders previous frame

**Changes needed:**
```cpp
// In ParticleSystem::SetPINNEnabled(true):
1. Wait for all GPU work to finish
2. Do ONE readback to m_cpuPositions/m_cpuVelocities
3. Set flag: m_particlesOnCPU = true
4. Disable GPU physics dispatch

// In Update():
if (m_usePINN && m_particlesOnCPU) {
    UpdatePhysics_PINN_CPUOnly();  // No readback, just PINN inference
    UploadParticleData();          // Upload to GPU for rendering
} else {
    UpdatePhysics_GPU();           // Normal GPU compute
}
```

## Why Current Approach Fails

D3D12 doesn't allow:
```cpp
GPU writes to buffer → Immediate readback → Continue using on GPU
```

While buffer is in use:
- RT system references it (BLAS AABBs)
- Gaussian renderer reads it
- Next physics dispatch will write to it

Attempting to Map() while GPU has active references = **DEVICE_REMOVED**

## Next Session: Implement CPU-Only PINN

1. Add flag: `bool m_particlesOnCPU` to ParticleSystem
2. Modify `SetPINNEnabled()` to do ONE safe readback at toggle time
3. Keep particles on CPU while PINN active
4. Only upload to GPU (never readback)
5. Switch back to GPU when PINN disabled

This is the correct architecture for ML physics anyway.
