# RTXDI Milestone 3: Test Plan

**Quick validation guide for DXR pipeline implementation**

---

## Test 1: Shader Compilation Verification

**What**: Verify all RTXDI shaders compiled successfully

**Command**:
```bash
ls -lh build/Debug/shaders/rtxdi/*.dxil
```

**Expected output**:
```
light_grid_build_cs.dxil  (7.6 KB)
rtxdi_raygen.dxil         (5.5 KB)
rtxdi_miss.dxil           (2.6 KB)
```

**Status**: ✅ PASSED (verified 2025-10-18 22:38)

---

## Test 2: Runtime Execution (No Crashes)

**What**: Run application with RTXDI lighting system

**Command**:
```bash
./build/Debug/PlasmaDX-Clean.exe --rtxdi
```

**Expected console output** (first 5 frames):
```
RTXDI Light Grid updated (frame 1, 13 lights)
RTXDI DispatchRays executed (1920x1080)
RTXDI Light Grid updated (frame 2, 13 lights)
RTXDI DispatchRays executed (1920x1080)
[... frames 3-5 ...]
```

**Success criteria**:
- [ ] Application starts without crashes
- [ ] FPS stable (115-120 FPS expected)
- [ ] Console shows "DispatchRays executed" for first 5 frames
- [ ] No D3D12 debug layer errors

**Status**: ⏸️ PENDING (requires build + run)

---

## Test 3: PIX GPU Capture

**What**: Verify DXR pipeline appears in PIX timeline

**Command**:
```bash
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --rtxdi
# Capture frame 120 (or manual Ctrl+D)
```

**PIX timeline verification**:
1. Find "UpdateLightGrid" compute dispatch
2. **NEW**: Find "DispatchRays" event immediately after
3. Verify existing RT lighting system still runs

**Event hierarchy**:
```
Frame N
└─ Render()
   ├─ UpdateLightGrid (Compute: 4×4×4 groups)
   ├─ DispatchRays (NEW - DXR: 1920×1080×1 rays)  <--- VERIFY THIS
   ├─ RTLighting::ComputeLighting (Compute)
   └─ ParticleRenderer::Render (Graphics)
```

**Detailed checks**:
- [ ] DispatchRays event exists
- [ ] Event shows 1920×1080 ray dimensions
- [ ] SBT buffer visible in resources (128 bytes)
- [ ] Debug output UAV bound to u0
- [ ] Light grid SRV bound to t0 (3.375 MB)
- [ ] Lights SRV bound to t1 (512 bytes)

**Status**: ⏸️ PENDING (requires PIX capture)

---

## Test 4: Resource Validation

**What**: Verify all DXR resources created correctly

**Where**: Check initialization logs in `logs/PlasmaDX-Clean_*.log`

**Expected log entries**:
```
Creating DXR pipeline...
Debug output buffer created: 1920x1080 (R32G32B32A32_FLOAT)
Loaded shaders: raygen=XXXX bytes, miss=XXXX bytes
DXR global root signature created
DXR state object created (5 subobjects)
Shader binding table created: 128 bytes (raygen=64, miss=64)
DXR pipeline created successfully!
```

**Success criteria**:
- [ ] All "created successfully" messages appear
- [ ] No "Failed to create" errors
- [ ] State object has 5 subobjects
- [ ] SBT size = 128 bytes

**Status**: ⏸️ PENDING (requires log review)

---

## Test 5: Performance Baseline

**What**: Measure FPS impact of Milestone 3

**Baseline** (without RTXDI, frame 100):
- 10K particles: ~120 FPS (8.3 ms/frame)
- 100K particles: ~105 FPS (9.5 ms/frame)

**Expected** (with RTXDI M3, frame 100):
- 10K particles: 115-118 FPS (8.5-8.7 ms/frame)
- 100K particles: 100-103 FPS (9.7-10.0 ms/frame)

**Acceptable overhead**: <5% (0.2-0.4 ms/frame)

**How to measure**:
1. Run baseline: `./build/Debug/PlasmaDX-Clean.exe` (no --rtxdi)
2. Note FPS after 10 seconds
3. Run RTXDI: `./build/Debug/PlasmaDX-Clean.exe --rtxdi`
4. Note FPS after 10 seconds
5. Calculate: `overhead = (baseline - rtxdi) / baseline * 100`

**Status**: ⏸️ PENDING (requires runtime measurement)

---

## Test 6: Debug Buffer Verification (Advanced)

**What**: Verify debug output buffer is being written

**Method 1: PIX GPU capture**
1. Capture frame
2. Find DispatchRays event
3. Click "Pipeline State" → "Output Resources"
4. Find "RTXDI Debug Output" UAV (u0)
5. View resource contents (should show non-zero RGB values)

**Method 2: Manual readback** (for advanced debugging)
```cpp
// Add to RTXDILightingSystem::DumpBuffers()
// Copy m_debugOutputBuffer to readback heap
// Write to file: PIX/buffer_dumps/g_debugOutput.bin
// Analyze: 1920×1080 pixels × 16 bytes/pixel = 31.6 MB
```

**Expected values**:
- R channel: 0.0 - 1.0 (cell X / 30)
- G channel: 0.0 - 1.0 (cell Y / 30)
- B channel: 0.0 - 1.0 (cell Z / 30)
- A channel: 1.0 (always opaque)

**Status**: ⏸️ PENDING (requires PIX or custom dump)

---

## Quick Smoke Test (5 minutes)

**Minimal validation to verify Milestone 3 works:**

```bash
# 1. Verify shaders compiled
ls build/Debug/shaders/rtxdi/*.dxil | wc -l
# Expected: 3 (light_grid_build_cs, rtxdi_raygen, rtxdi_miss)

# 2. Run for 10 seconds
timeout 10 ./build/Debug/PlasmaDX-Clean.exe --rtxdi
# Expected: No crashes, clean exit

# 3. Check logs
grep "DispatchRays" logs/PlasmaDX-Clean_*.log | head -5
# Expected: 5 lines showing "DispatchRays executed"

# 4. Check for errors
grep "ERROR" logs/PlasmaDX-Clean_*.log | grep -i rtxdi
# Expected: No output (no RTXDI errors)
```

**If all 4 pass: Milestone 3 is working correctly ✅**

---

## Common Issues

### Issue: "Failed to open rtxdi_raygen.dxil"
**Solution**: Working directory is wrong. Run from project root, not build/Debug/

### Issue: QueryInterface to ID3D12GraphicsCommandList4 fails
**Solution**: DXR 1.1 not supported. Update GPU drivers (NVIDIA 531.00+, AMD 23.1.1+)

### Issue: DispatchRays crashes with access violation
**Solution**: SBT alignment incorrect. Verify records are 32-byte aligned, tables 256-byte aligned

### Issue: PIX shows no DispatchRays event
**Solution**: RTXDI lighting system not active. Verify `--rtxdi` flag passed to application

---

## Success Declaration

**Milestone 3 is VALIDATED when:**
- ✅ Test 1 (Shader compilation): PASSED
- [ ] Test 2 (Runtime execution): PASSED
- [ ] Test 3 (PIX capture): PASSED
- [ ] Test 4 (Resource validation): PASSED
- [ ] Test 5 (Performance baseline): PASSED (overhead <5%)

**Current status**: 1/5 tests passed (shader compilation only)

**Next action**: Build project and run Tests 2-5

---

**Test plan version**: 1.0
**Last updated**: 2025-10-18 22:45 UTC
