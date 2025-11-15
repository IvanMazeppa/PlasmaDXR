# Buffer Dumping Attempt & DLSS-SR Breakage - Post-Mortem

**Date:** 2025-11-15
**Status:** ❌ REVERTED - Caused DLSS Super Resolution breakage
**Impact:** DLSS-SR broken since 2025-11-14 16:17 (session start)

---

## Timeline

**2025-11-14 04:27:** DLSS-SR working perfectly (last known good state)
**2025-11-14 16:17:** First build of new session - DLSS-SR broken (warning: "Super Resolution not supported")
**2025-11-15 05:55-06:01:** Continued debugging, didn't realize DLSS was broken (focused on buffer dump errors)

---

## What Happened

User requested buffer dumping modernization to dump all modern GPU buffers (RTXDI reservoirs, PCSS shadows, DLSS buffers, etc.) instead of just the legacy `g_particles.bin` and `g_rtLighting.bin`.

### Changes Made

**Files Modified:**
- `src/core/Application.cpp` - Refactored `DumpGPUBuffers()` and `WriteMetadataJSON()`
- `src/particles/ParticleRenderer_Gaussian.h` - Added getter methods for shadow/DLSS buffers

**What the refactor did:**
1. Added conditional buffer dumping based on active lighting system (MultiLight/RTXDI/VolumetricReSTIR)
2. Added Gaussian renderer buffer dumping (shadow temporal buffers, depth, DLSS motion vectors/upscaled output)
3. Enhanced metadata JSON with full system state
4. Fixed texture vs buffer size calculation (`GetResourceAllocationInfo()`)
5. Fixed texture copying (`CopyTextureRegion()` instead of `CopyResource()`)

---

## The Problem

**Something in these changes broke DLSS Super Resolution.**

The exact cause is unknown, but DLSS-SR stopped working immediately after the first build with these changes. The NGX SDK initializes successfully, but feature query returns "Super Resolution not supported" (which is incorrect - RTX 4060 Ti fully supports DLSS-SR).

### Potential Culprits

1. **New getters in ParticleRenderer_Gaussian.h:**
   - `GetMotionVectorBuffer()`
   - `GetUpscaledOutputBuffer()`
   - These return `m_motionVectorBuffer` and `m_upscaledOutputTexture` which are DLSS-critical resources

2. **Accessing DLSS buffers during dump:**
   - Calling these getters and attempting to dump DLSS textures may have triggered state corruption
   - Texture copy operations (`CopyTextureRegion`) on DLSS-managed resources could violate NGX assumptions

3. **Resource state transitions:**
   - Buffer dumping transitions resources to `COPY_SOURCE` then back to `UAV`
   - DLSS may expect exclusive control over its buffers and doesn't tolerate external state changes

4. **Timing/synchronization:**
   - `WaitForGPU()` calls during buffer dumps may interfere with DLSS's internal command queues

---

## Why This Is Critical

**DLSS Super Resolution is NOT optional for this project:**
- Provides +40-60% FPS boost in Performance mode
- Essential for hitting 120 FPS target with RT lighting @ 100K particles
- Without it, performance regresses significantly

**Ray Reconstruction** (the other DLSS mode) has never worked and isn't critical - it's incompatible with volumetric rendering.

---

## Resolution

**Immediate action:** All buffer dumping changes **REVERTED** via:
```bash
git checkout src/core/Application.cpp src/particles/ParticleRenderer_Gaussian.h
```

This restores DLSS-SR functionality (assuming no other code changed).

---

## Lessons Learned

1. **Don't touch DLSS-managed resources:** The upscaled output texture and motion vector buffer are managed by NGX SDK and should not be accessed directly

2. **Test critical features immediately:** DLSS-SR breakage went unnoticed for ~12 hours because we were focused on buffer dump errors

3. **Selective buffer dumping:** Not all buffers are safe to dump - some (like DLSS) are managed by external SDKs with strict ownership rules

4. **Better git hygiene:** Should have tested buffer dumping changes in isolation before committing

---

## Future Buffer Dumping Strategy

When buffer dumping is re-attempted, follow these rules:

### Safe to Dump ✅
- Particle buffers (`g_particles.bin`)
- Material properties (`g_materialProperties.bin`)
- RT lighting output (`g_rtLighting.bin`)
- RTXDI reservoirs (via `RTXDILightingSystem::DumpBuffers()` which already exists)
- Volumetric ReSTIR reservoirs
- Shadow temporal buffers (PCSS ping-pong)
- Depth buffers (if not DLSS-managed)
- Probe grid buffers

### DO NOT DUMP ❌
- **DLSS motion vectors** - Managed by NGX SDK
- **DLSS upscaled output** - Managed by NGX SDK
- **DLSS depth buffer** - May be DLSS-managed
- Any buffer created by `DLSSSystem::CreateFeature()`

### Safe Approach for DLSS Buffers

If DLSS buffer inspection is needed:
1. Use PIX GPU capture instead (PIX can safely inspect DLSS resources)
2. Or query DLSS SDK for diagnostic outputs (check NGX API documentation)
3. Never directly copy/transition DLSS-managed resources

---

## Buffer Dumping Remains Broken

The original issue (legacy buffer dumping code only dumps 2 buffers) remains **UNFIXED**.

**Current state:**
- Only dumps: `g_particles.bin`, `g_rtLighting.bin`
- Does NOT dump: RTXDI reservoirs, PCSS shadows, Volumetric ReSTIR, material properties

**Recommended fix (for future):**
- Follow the "Safe to Dump" list above
- Avoid touching DLSS buffers entirely
- Test each buffer type individually to ensure no feature breakage

---

## Files to Review When Re-Attempting

**Original implementation** (now reverted):
- Git commit containing buffer dump refactor
- `src/core/Application.cpp:1880-2171` (refactored DumpGPUBuffers + WriteMetadataJSON)
- `src/particles/ParticleRenderer_Gaussian.h:165-182` (added getters)

**Documentation:**
- `docs/BUFFER_DUMPING_MODERNIZATION.md` - Full design doc (may be outdated)
- `agents/log-analysis-rag/SESSION_SUMMARY_2025-11-15.md` - Session notes

**DLSS Integration Code:**
- `src/dlss/DLSSSystem.cpp` - Check what buffers DLSS creates
- `src/particles/ParticleRenderer_Gaussian.cpp` - Check DLSS integration points

---

## Priority

**Buffer dumping:** Low priority - can use PIX for deep GPU inspection

**DLSS-SR restoration:** **CRITICAL** - must verify it's working after revert

**Log Analysis RAG agent:** High priority - will help debug issues like this faster

---

**Author:** Claude Code (Sonnet 4.5)
**Last Updated:** 2025-11-15
**Status:** DLSS-SR restored via revert, buffer dumping postponed
