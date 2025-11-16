# PIX Debugging Tools Reference

Complete reference for all 9 MCP diagnostic tools.

**Last Updated**: 2025-11-01

---

## Quick Reference

| Tool | Status | Best For |
|------|--------|----------|
| [diagnose_gpu_hang](#1-diagnose_gpu_hang) | ✅ Working | Finding crash thresholds |
| [analyze_dxil_root_signature](#2-analyze_dxil_root_signature) | ✅ Working | Root signature mismatches |
| [validate_shader_execution](#3-validate_shader_execution) | ✅ Working | Confirming shader execution |
| [analyze_particle_buffers](#4-analyze_particle_buffers) | ✅ Working | Particle data validation |
| [diagnose_visual_artifact](#5-diagnose_visual_artifact) | ✅ Working | Rendering artifacts |
| [pix_list_captures](#6-pix_list_captures) | ✅ Working | Listing PIX captures |
| [capture_buffers](#7-capture_buffers) | ⚠️ Manual | GPU buffer dumps |
| [pix_capture](#8-pix_capture) | ⚠️ Manual | Creating PIX captures |
| [analyze_restir_reservoirs](#9-analyze_restir_reservoirs) | ⚠️ Outdated | ReSTIR analysis (old format) |

---

## 1. diagnose_gpu_hang

### Purpose
Autonomous GPU hang/TDR crash diagnosis for Volumetric ReSTIR shaders.

### What It Does
1. Launches PlasmaDX-Clean with `--restir` flag
2. Monitors for crashes/hangs with configurable timeout
3. Captures and analyzes application logs
4. Identifies exact crash thresholds (e.g., "2044 works, 2045 crashes")
5. Provides detailed failure analysis

### Parameters

```python
diagnose_gpu_hang(
    particle_count=2045,           # int, default: 2045
    render_mode="volumetric_restir", # str, default: "volumetric_restir"
    timeout_seconds=10,            # int, default: 10
    test_threshold=False,          # bool, default: False
    capture_logs=True              # bool, default: True
)
```

**Parameters**:
- `particle_count`: Number of particles to test
- `render_mode`: "volumetric_restir", "gaussian", "multi_light"
- `timeout_seconds`: How long to wait before considering hung (default: 10)
- `test_threshold`: Test multiple counts around threshold (2040, 2044, 2045, 2048, 2050)
- `capture_logs`: Capture logs for analysis (default: true)

### Examples

**Single test**:
```
In Claude Code:
  "Use diagnose_gpu_hang to test at 2045 particles"
```

**Threshold testing**:
```
In Claude Code:
  "Use diagnose_gpu_hang to find the crash threshold at 2045 particles"
```

**With custom timeout**:
```
In Claude Code:
  "Use diagnose_gpu_hang to test at 10000 particles with 20 second timeout"
```

### Output Format

```json
{
  "status": "hung",  // "timeout", "hung", "completed", "crashed"
  "particle_count": 2045,
  "runtime_seconds": 10.5,
  "logs_captured": true,
  "restir_enabled": true,
  "frame_count": 2,
  "analysis": {
    "crash_detected": true,
    "last_operation": "PopulateVolumeMip2 dispatch",
    "recommendations": [
      "Create PIX capture at 2045 particles",
      "Check PopulateVolumeMip2 diagnostic counters"
    ]
  }
}
```

### Status Meanings

- **timeout**: Process exceeded timeout (normal operation, no crash)
- **hung**: Process actually hung (GPU TDR, infinite loop)
- **completed**: Process exited cleanly
- **crashed**: Process returned non-zero exit code

### Known Issues
- None

---

## 2. analyze_dxil_root_signature

### Purpose
Detect root signature mismatches causing silent shader execution failures.

### What It Does
1. Runs `dxc.exe -dumpbin` on compiled DXIL shader
2. Extracts resource bindings (cbuffers, SRVs, UAVs)
3. Compares to expected bindings for Volumetric ReSTIR shaders
4. Identifies missing resources or wrong register slots
5. Provides specific C++ fix recommendations

### Parameters

```python
analyze_dxil_root_signature(
    dxil_path="build/bin/Debug/shaders/volumetric_restir/populate_volume_mip2.dxil",
    shader_name="PopulateVolumeMip2"  # Optional
)
```

**Parameters**:
- `dxil_path` (required): Path to compiled .dxil file
- `shader_name` (optional): One of "PopulateVolumeMip2", "GenerateCandidates", "ShadeSelectedPaths"

### Expected Bindings

**PopulateVolumeMip2**:
```
cb0: PopulationConstants
t0:  g_particles
u0:  g_volumeTexture
u1:  g_diagnosticCounters
```

**GenerateCandidates**:
```
cb0: PathGenerationConstants
u0:  g_reservoirs
(Phase 1 stub - t0/t1/t2 removed temporarily)
```

**ShadeSelectedPaths**:
```
cb0: ShadingConstants
t0:  g_particleBVH
t1:  g_particles
t2:  g_reservoirs
u0:  g_outputTexture
```

### Examples

**Via Python**:
```python
analyze_dxil_root_signature(
    dxil_path="build/bin/Debug/shaders/volumetric_restir/populate_volume_mip2.dxil",
    shader_name="PopulateVolumeMip2"
)
```

**In Claude Code**:
```
"Analyze the root signature for PopulateVolumeMip2"
```

### Output Format

```json
{
  "shader_name": "PopulateVolumeMip2",
  "resource_bindings": {
    "cbuffers": [{"name": "PopulationConstants", "register": "cb0"}],
    "srvs": [{"name": "g_particles", "register": "t0"}],
    "uavs": [
      {"name": "g_volumeTexture", "register": "u0"},
      {"name": "g_diagnosticCounters", "register": "u1"}
    ]
  },
  "summary": {
    "total_cbuffers": 1,
    "total_srvs": 1,
    "total_uavs": 2,
    "issues_found": 0
  },
  "issues": [],
  "recommendations": [
    "✅ All expected resources found in shader",
    "If shader still not executing:",
    "1. Verify C++ root signature matches this layout",
    "2. Check parameter ORDER (not just presence)"
  ]
}
```

### Common Issues Detected

**Missing Resource**:
```json
{
  "issues": [{
    "severity": "error",
    "resource": "g_diagnosticCounters",
    "expected": "UAV at u1",
    "actual": "NOT FOUND",
    "impact": "Shader will not execute - root signature mismatch"
  }]
}
```

**DXC Optimizer Removed Resource** (Phase 1 stub):
- Shader declares resource but never uses it
- DXC removes it from compiled DXIL
- C++ still provides it → mismatch → silent failure

### Known Issues
- None

---

## 3. validate_shader_execution

### Purpose
Confirm shaders are actually executing (not just dispatching).

### What It Does
1. Parses application logs for diagnostic counters
2. Analyzes PopulateVolumeMip2 execution status
3. Detects execution states:
   - **NOT_EXECUTING**: All counters zero (shader never runs)
   - **EXECUTING_BUT_NO_OUTPUT**: Counters non-zero but no voxel writes
   - **EXECUTING_NORMALLY**: Counters and voxel writes both non-zero
4. Provides severity-based recommendations

### Parameters

```python
validate_shader_execution(
    log_path=None,      # Optional: path to log file
    buffer_dir=None     # Optional: path to buffer dumps
)
```

**Parameters**:
- `log_path` (optional): Path to specific log file (uses latest if not provided)
- `buffer_dir` (optional): Path to buffer dump directory for additional validation

### Examples

**Default (latest log)**:
```
In Claude Code:
  "Validate that PopulateVolumeMip2 is executing"
```

**Specific log**:
```python
validate_shader_execution(
    log_path="build/bin/Debug/logs/PlasmaDX-Clean_20251101_223628.log"
)
```

### Output Format

```json
{
  "status": "NOT_EXECUTING",
  "diagnostic_counters": {
    "total_threads_executed": 0,
    "early_returns": 0,
    "total_voxel_writes": 0,
    "max_voxels_per_particle": 0
  },
  "analysis": {
    "shader_dispatched": true,
    "shader_executed": false,
    "severity": "critical"
  },
  "recommendations": [
    "❌ CRITICAL: Shader dispatches but never executes",
    "Next steps:",
    "1. Run analyze_dxil_root_signature to check resource bindings",
    "2. Verify PSO creation succeeded",
    "3. Check for D3D12 validation errors"
  ]
}
```

### Execution States

| State | Counters[0] | Writes | Meaning |
|-------|-------------|--------|---------|
| NOT_EXECUTING | 0 | 0 | Shader never runs (root signature mismatch likely) |
| EXECUTING_BUT_NO_OUTPUT | >0 | 0 | Shader runs but writes nothing (bounds check issue) |
| EXECUTING_NORMALLY | >0 | >0 | Shader working correctly |

### Known Issues
- None

---

## 4. analyze_particle_buffers

### Purpose
Validate particle buffer data integrity.

### What It Does
1. Reads particle buffer binary files (g_particles.bin)
2. Parses particle structures (position, velocity, temperature, etc.)
3. Validates data ranges (checks for NaN, Inf, out-of-bounds)
4. Computes statistics (min/max/avg)

### Parameters

```python
analyze_particle_buffers(
    particles_path="PIX/buffer_dumps/g_particles.bin",
    expected_count=100  # Optional
)
```

**Parameters**:
- `particles_path` (required): Path to g_particles.bin
- `expected_count` (optional): Expected particle count for validation

### Examples

```python
analyze_particle_buffers(
    particles_path="PIX/buffer_dumps/g_particles.bin",
    expected_count=100
)
```

### Output Format

```json
{
  "particle_count": 100,
  "buffer_size_bytes": 4800,
  "validation": {
    "has_nan": false,
    "has_inf": false,
    "out_of_bounds": 0
  },
  "statistics": {
    "position_range": {"min": [-1500, -1500, -1500], "max": [1500, 1500, 1500]},
    "temperature_range": {"min": 800.0, "max": 26000.0},
    "avg_velocity": 45.2
  },
  "issues": []
}
```

### Known Issues
- None

---

## 5. diagnose_visual_artifact

### Purpose
Automated diagnosis of rendering issues from symptom description.

### What It Does
1. Analyzes symptom description using AI
2. Provides hypothesis based on known issues
3. Suggests diagnostic checks
4. Optionally analyzes buffer dumps for confirmation

### Parameters

```python
diagnose_visual_artifact(
    symptom="black dots appearing when camera is 800 units away",
    buffer_dump_dir=None  # Optional
)
```

**Parameters**:
- `symptom` (required): Description of visual artifact
- `buffer_dump_dir` (optional): Path to buffer dumps for analysis

### Examples

```
In Claude Code:
  "Diagnose visual artifact: black dots at far camera distances"
```

### Output Format

```json
{
  "symptom": "black dots at far camera distances",
  "hypothesis": "Low M values in ReSTIR reservoirs at far distances",
  "confidence": "high",
  "suggested_checks": [
    "Analyze ReSTIR reservoir M values",
    "Check spatial/temporal reuse at far distances",
    "Verify visibility scaling"
  ],
  "known_similar_issues": [
    "ReSTIR black dots bug (2025-10-13) - visibility scaling fix"
  ]
}
```

### Known Issues
- None

---

## 6. pix_list_captures

### Purpose
List available PIX .wpix captures with metadata.

### What It Does
1. Scans `PIX/Captures/` directory
2. Lists .wpix files with size and modification date
3. Helps identify recent captures for analysis

### Parameters

None

### Examples

```
In Claude Code:
  "List available PIX captures"
```

### Output Format

```json
{
  "captures_directory": "PIX/Captures",
  "captures": [
    {
      "filename": "volumetric_restir_debug_2.wpix",
      "size_mb": 5.9,
      "modified": "2025-11-01 22:15:33",
      "age_hours": 1.2
    }
  ],
  "total_captures": 1,
  "total_size_mb": 5.9
}
```

### Known Issues
- None

---

## 7. capture_buffers

### Purpose
Trigger in-app GPU buffer dump from PlasmaDX.

### What It Does
1. Signals PlasmaDX to dump GPU buffers
2. Saves particle data, ReSTIR reservoirs, lighting buffers
3. Creates metadata.json with frame info

### Parameters

```python
capture_buffers(
    frame=120,         # Optional: frame number
    mode="gaussian",   # Optional: rendering mode
    output_dir=None    # Optional: custom output directory
)
```

**Parameters**:
- `frame` (optional): Frame number to capture at
- `mode` (optional): Rendering mode ("gaussian", "volumetric_restir", etc.)
- `output_dir` (optional): Custom output directory

### Examples

```
In Claude Code:
  "Capture buffers at frame 120 in volumetric_restir mode"
```

### Output Format

```json
{
  "output_directory": "PIX/buffer_dumps/2025-11-01_22-30-15",
  "files_created": [
    "g_particles.bin",
    "g_reservoirs.bin",
    "metadata.json"
  ],
  "metadata": {
    "frame": 120,
    "particle_count": 100,
    "timestamp": "2025-11-01 22:30:15"
  }
}
```

### Known Issues

**Status**: ⚠️ Manual trigger required

**Issue**: Currently requires manual Ctrl+D key press to trigger dump

**Workaround**:
1. Launch PlasmaDX
2. Press Ctrl+D at desired frame
3. Tool will find and analyze the dump

**Future**: Implement automated frame capture

---

## 8. pix_capture

### Purpose
Create PIX .wpix GPU capture using pixtool.exe.

### What It Does
1. Uses pixtool.exe to create GPU capture
2. Saves .wpix file to PIX/Captures/
3. Optionally opens capture in PIX GUI

### Parameters

```python
pix_capture(
    frames=1,          # int, default: 1
    output_name=None,  # Optional: filename
    auto_open=False    # bool, default: False
)
```

**Parameters**:
- `frames`: Number of frames to capture
- `output_name` (optional): Custom filename (auto-generated if not provided)
- `auto_open`: Open capture in PIX GUI after creation

### Examples

```
In Claude Code:
  "Create a PIX capture with 2 frames"
```

### Output Format

```json
{
  "capture_file": "PIX/Captures/volumetric_restir_2025-11-01_22-45-12.wpix",
  "size_mb": 6.2,
  "frames_captured": 2,
  "pix_opened": false
}
```

### Known Issues

**Status**: ⚠️ Requires manual PlasmaDX launch

**Issue**: Cannot automatically launch PlasmaDX with PIX attached

**Workaround**:
1. Manually launch PlasmaDX
2. Call pix_capture tool
3. PIX will attach to running process

**Future**: Investigate pixtool command-line options for automated launch

---

## 9. analyze_restir_reservoirs

### Purpose
Parse and analyze ReSTIR reservoir buffers (OLD FORMAT).

### What It Does
1. Reads binary reservoir buffer files
2. Parses 32-byte reservoir structures (DEPRECATED FORMAT)
3. Computes statistics (W, M, weightSum)
4. Identifies issues (low M values, invalid data)

### Parameters

```python
analyze_restir_reservoirs(
    current_path="PIX/buffer_dumps/g_currentReservoirs.bin",
    sample_size=1000  # Optional
)
```

**Parameters**:
- `current_path` (required): Path to g_currentReservoirs.bin
- `sample_size` (optional): Number of reservoirs to sample (default: 1000)

### Output Format

```json
{
  "reservoir_count": 230400,
  "sample_size": 1000,
  "statistics": {
    "avg_M": 2.1,
    "avg_W": 45.3,
    "avg_weightSum": 8.2
  },
  "issues": [
    {
      "severity": "warning",
      "issue": "Low average M value (2.1 vs expected >8)",
      "recommendation": "Check spatial/temporal reuse"
    }
  ]
}
```

### Known Issues

**Status**: ⚠️ OUTDATED - Needs update for new format

**Issue**: Designed for old 32-byte ReSTIR reservoir format (deprecated)

**Current Format**: 64-byte Volumetric ReSTIR reservoirs
- 3× PathVertex (48 bytes)
- State data (16 bytes)

**Recommendation**: Update tool or mark as deprecated until Volumetric ReSTIR Phase 2-3 complete

---

## Tool Selection Guide

### "My shader isn't executing"
1. `validate_shader_execution` - Confirm execution status
2. `analyze_dxil_root_signature` - Check resource bindings
3. Fix root signature based on recommendations

### "Application crashes at unknown particle count"
1. `diagnose_gpu_hang` with `test_threshold=true`
2. Identifies threshold (e.g., "2044 works, 2045 crashes")
3. `pix_capture` at crash threshold
4. Manual PIX analysis or next debugging iteration

### "Visual artifact (black dots, flickering, etc.)"
1. `diagnose_visual_artifact` with symptom description
2. Follow suggested checks
3. `analyze_particle_buffers` if particle data suspected
4. `analyze_restir_reservoirs` if ReSTIR related (old format only)

### "Need to inspect GPU state"
1. `pix_list_captures` - Find recent captures
2. `pix_capture` - Create new capture if needed
3. Manual PIX GUI inspection
4. Future: Automated .wpix parsing

---

**Last Updated**: 2025-11-01
**Total Tools**: 9 (7 fully functional, 2 with limitations)
