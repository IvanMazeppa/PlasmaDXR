# Known Issues & Bug Tracker

**Last Updated**: 2025-11-01

Active bugs, limitations, and workarounds for PIX Debugging Agent v4.

---

## Table of Contents

- [Critical Issues](#critical-issues)
- [Tool Limitations](#tool-limitations)
- [Code Quality Issues](#code-quality-issues)
- [Resolved Issues](#resolved-issues)

---

## Critical Issues

### 1. PopulateVolumeMip2 Not Executing

**Status**: 🔄 **ACTIVE INVESTIGATION**
**Severity**: Critical
**Affects**: Volumetric ReSTIR Phase 1
**First Reported**: 2025-10-29
**Last Updated**: 2025-11-01

#### Symptoms
- Shader dispatches successfully (logged)
- PSO and root signature creation succeed
- Diagnostic counters ALL ZERO (shader never executes)
- No D3D12 validation errors
- No GPU hang or TDR

#### Evidence
```
[22:36:31] [INFO] ========== PopulateVolumeMip2 Diagnostic Counters ==========
[22:36:31] [INFO]   [0] Total threads executed: 0  ← Should ALWAYS be non-zero
[22:36:31] [INFO]   [1] Early returns: 0
[22:36:31] [INFO]   [2] Total voxel writes: 0
[22:36:31] [INFO]   [3] Max voxels per particle: 0
```

#### What We've Checked
✅ Root signature: **PERFECT MATCH** (4/4 resources verified via analyze_dxil_root_signature)
- cb0: PopulationConstants ✓
- t0: g_particles ✓
- u0: g_volumeTexture ✓
- u1: g_diagnosticCounters ✓

✅ PSO creation: **SUCCESS** (logged)

✅ Shader compilation: **SUCCESS** (6096 bytes loaded)

✅ Dispatch parameters: **VALID** ((100+63)/64 = 2 thread groups)

✅ Resource bindings: **ALL 4 PARAMETERS BOUND** (lines 898-907 in VolumetricReSTIRSystem.cpp)

#### What's Different from Working Shaders
| Aspect | PopulateVolumeMip2 | GenerateCandidates (fixed) | ShadeSelectedPaths (working) |
|--------|-------------------|---------------------------|----------------------------|
| Root signature | Correct | Fixed (simplified) | Correct |
| Execution | ❌ Never runs | ✅ Runs (stub) | ✅ Runs |
| Diagnostic line 158 | Never reached | N/A | N/A |

#### Hypothesis
This is a **silent D3D12 failure** - the GPU receives the dispatch but silently discards it. Possible causes:

1. **Parameter order mismatch** (not detected by presence check)
   - C++ binds: param[0], param[1], param[2], param[3]
   - Shader expects: Same resources but different order?

2. **Descriptor type mismatch**
   - u0: C++ uses descriptor table, shader expects root descriptor?
   - u1: C++ uses root descriptor, shader expects descriptor table?

3. **Resource state issue**
   - UAV barrier timing?
   - Transition barrier missing?

4. **Pipeline state issue**
   - Wrong PSO bound before dispatch?
   - RS/PSO mismatch?

#### Next Steps
1. **PIX capture analysis** (capture available: volumetric_restir_debug_2.wpix)
   - Inspect actual bound resources
   - Check resource states
   - Verify PSO/RS at dispatch point

2. **Manual verification** of C++ parameter order vs DXIL expectations

3. **Descriptor type audit**:
   - u0 (g_volumeTexture): Descriptor table (required for typed 3D UAV)
   - u1 (g_diagnosticCounters): Root descriptor
   - Verify this matches shader expectations

#### Workaround
None - shader is essential for Volumetric ReSTIR Phase 1

---

### 2. GenerateCandidates Root Signature Mismatch (RESOLVED ✅)

**Status**: ✅ **RESOLVED**
**Severity**: Critical (was)
**Affects**: Volumetric ReSTIR Phase 1
**Resolved**: 2025-11-01

#### Original Issue
Phase 1 stub code returned false immediately, never using particle or volume data. DXC optimizer removed unused resources (t0, t1, t2), causing root signature mismatch.

#### Root Cause
```hlsl
// path_generation.hlsl lines 264-272
// PHASE 1 STUB: Generate empty paths to test infrastructure
pathLength = 0;
flags = 0;
return false;  // ← Never uses g_particles, g_volumeTexture, g_particleBVH!
```

DXC saw resources declared but never used → optimized them away → DXIL only has cb0 + u0.

#### Fix Applied
Simplified C++ root signature to match Phase 1 stub shader:
```cpp
// VolumetricReSTIRSystem.cpp:528-534
CD3DX12_ROOT_PARAMETER1 rootParams[2];  // Was 5
rootParams[0].InitAsConstantBufferView(0, 0);      // cb0
rootParams[1].InitAsUnorderedAccessView(0, 0);     // u0
// Removed: t0 (BLAS), t1 (particles), t2 (volume)
```

#### Status
✅ **RESOLVED** - Shader now executes (stub returns empty reservoirs as expected)

#### TODO for Phase 2
When enabling full random walk:
1. Uncomment lines 275-315 in path_generation.hlsl
2. Re-add t0/t1/t2 to C++ root signature
3. Re-add SetComputeRootShaderResourceView() calls
4. Rebuild and verify

---

## Tool Limitations

### 3. analyze_restir_reservoirs Outdated Format

**Status**: ⚠️ **UPDATE NEEDED**
**Severity**: Medium
**Affects**: analyze_restir_reservoirs tool

#### Issue
Tool designed for old 32-byte ReSTIR reservoir format (deprecated). Current Volumetric ReSTIR uses 64-byte format.

**Old Format** (32 bytes):
```cpp
struct ReSTIRReservoir {
    float3 lightPos;     // 12 bytes
    float weightSum;     // 4 bytes
    uint M;              // 4 bytes
    float W;             // 4 bytes
    uint8_t padding[8];  // 8 bytes
};  // Total: 32 bytes
```

**New Format** (64 bytes):
```cpp
struct VolumetricReservoir {
    uint selectedPath;       // 4 bytes
    PathVertex vertices[3];  // 48 bytes (3× 16-byte vertices)
    float weightSum;         // 4 bytes
    float W;                 // 4 bytes
    uint M;                  // 4 bytes
};  // Total: 64 bytes
```

#### Impact
- Tool cannot parse current reservoir buffers
- Statistical analysis not available for Volumetric ReSTIR
- Workaround: Use PIX GUI to inspect buffers manually

#### Fix Required
Update `analyze_restir_reservoirs()` function in mcp_server.py:
1. Parse 64-byte structures
2. Extract PathVertex data
3. Compute new statistics (path length, vertex positions)

#### Priority
Low - Volumetric ReSTIR Phase 1 uses stub shader (empty reservoirs)
Medium - Required for Phase 2-3 debugging

---

### 4. pix_capture Requires Manual Launch

**Status**: ⚠️ **LIMITATION**
**Severity**: Low
**Affects**: pix_capture tool

#### Issue
Cannot automatically launch PlasmaDX with PIX attached. User must manually start PlasmaDX before calling tool.

#### Workaround
1. Manually launch PlasmaDX
2. Call `pix_capture` tool
3. PIX attaches to running process
4. Capture created successfully

#### Root Cause
`pixtool.exe` command-line options don't support launching arbitrary executables with PIX attached.

#### Possible Solutions
1. Investigate `pixtool.exe` advanced options
2. Use PIX SDK for programmatic control
3. Create wrapper script that launches both

#### Priority
Low - Workaround is straightforward

---

### 5. capture_buffers Requires Manual Trigger

**Status**: ⚠️ **LIMITATION**
**Severity**: Low
**Affects**: capture_buffers tool

#### Issue
Currently requires manual Ctrl+D key press to trigger buffer dump. Cannot automatically capture at specific frame.

#### Workaround
1. Launch PlasmaDX
2. Press Ctrl+D at desired frame
3. Tool finds and analyzes the dump

#### Possible Solutions
1. Implement `--dump-buffers N` flag in PlasmaDX (frame-based auto-dump)
2. Use keyboard automation (pyautogui already imported)
3. Add MCP-triggered dump via named pipe/file signal

#### Priority
Medium - Would improve automation workflow

---

## Code Quality Issues

### 6. Duplicate Tool Registrations

**Status**: ⚠️ **CLEANUP NEEDED**
**Severity**: Low (cosmetic)
**Affects**: mcp_server.py

#### Issue
`analyze_dxil_root_signature` registered twice in `list_tools()`:
- Once at line 111-121
- Again at line 150-160 (duplicate)

Creates confusing tool count: 11 registrations vs 9 unique tools.

#### Impact
- No functional impact (MCP handles duplicates gracefully)
- Confusing when debugging tool list
- Wastes memory (minimal)

#### Fix Required
Remove duplicate registration (lines 150-160 or 111-121)

#### Priority
Low - No functional impact

---

### 7. Missing Error Handling in DXIL Analysis

**Status**: ⚠️ **IMPROVEMENT NEEDED**
**Severity**: Low
**Affects**: analyze_dxil_root_signature

#### Issue
Limited error handling for edge cases:
- Invalid DXIL file (corrupted bytecode)
- dxc.exe not found at expected path
- Malformed DXIL output

#### Impact
- Tool may crash instead of providing useful error message
- Hard to diagnose if dxc.exe path is wrong

#### Fix Required
Add comprehensive try/catch blocks and error messages

#### Priority
Low - Works correctly for valid inputs

---

## Resolved Issues

### ✅ 1. GenerateCandidates Root Signature Mismatch
**Resolved**: 2025-11-01
**Fix**: Simplified C++ root signature to match Phase 1 stub shader
See [Issue #2](#2-generatecandidates-root-signature-mismatch-resolved-) above

### ✅ 2. Working Directory Issues (Shader Loading)
**Resolved**: 2025-10-31
**Fix**: Launch PlasmaDX from build/bin/Debug/ directory
**Details**: See CHANGELOG.md v0.1.5

### ✅ 3. SDK 0.1.1 → 0.1.6 Upgrade
**Resolved**: 2025-10-31
**Fix**: Added explicit system prompt required by SDK 0.1.6
**Details**: See CHANGELOG.md v0.1.5

### ✅ 4. diagnose_gpu_hang Process Orphaning
**Resolved**: 2025-10-31
**Fix**: Use `taskkill.exe /F` for forced termination
**Details**: Prevents orphaned PlasmaDX processes

---

## Issue Submission Guidelines

### Reporting New Issues

Please include:
1. **Symptom**: Clear description of unexpected behavior
2. **Tool Affected**: Which MCP tool (if applicable)
3. **Steps to Reproduce**: Exact commands/workflow
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Logs**: Relevant error messages or log excerpts
7. **Environment**: PlasmaDX version, particle count, etc.

### Issue Priority Levels

- **Critical**: Blocks all debugging (e.g., shader won't execute)
- **High**: Severely limits functionality (e.g., tool crashes frequently)
- **Medium**: Workaround exists but inconvenient
- **Low**: Cosmetic or rare edge case

---

## Debugging Resources

- **PIX Captures**: `PIX/Captures/` directory
- **Application Logs**: `build/bin/Debug/logs/`
- **Buffer Dumps**: `PIX/buffer_dumps/`
- **Tool Documentation**: See [TOOLS.md](TOOLS.md)

---

**Maintained by**: Claude Code automated debugging sessions
**Report Issues**: Via Claude Code conversation
**Last Review**: 2025-11-01
