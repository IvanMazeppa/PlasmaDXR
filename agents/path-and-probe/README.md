# Path & Probe Specialist

**Tier 3 Rendering Specialist** - Probe-Grid Lighting System Expert

## Purpose

Diagnose and optimize the **active probe-grid lighting system** (replaced Volumetric ReSTIR).

- **Architecture:** 32³ probe grid (32,768 probes), [-1500, +1500] world coverage
- **Storage:** Spherical Harmonics L2 (27 floats/probe = 3.35 MB)
- **Update:** Every 4 frames (temporal amortization)
- **Query:** Trilinear interpolation (8 nearest probes)
- **Performance:** ~1.1ms/frame total (0.8ms update + 0.3ms query)

## Tools (6)

| Tool | Purpose |
|------|---------|
| `analyze_probe_grid` | Config & performance analysis |
| `validate_probe_coverage` | Particle distribution coverage check |
| `diagnose_interpolation` | Detect visual artifacts/banding |
| `optimize_update_pattern` | Performance tuning for target FPS |
| `validate_sh_coefficients` | Spherical harmonics data integrity |
| `compare_vs_restir` | Benchmark vs shelved ReSTIR |

## Quick Start

```bash
# Test the agent
mcp__path-and-probe__analyze_probe_grid

# Diagnose lighting quality issues
mcp__path-and-probe__diagnose_interpolation(symptom="rim lighting inferior to multi-light")

# Optimize for 120 FPS
mcp__path-and-probe__optimize_update_pattern(target_fps=120, particle_count=10000)

# Compare to multi-light quality (use lighting-quality-comparison skill instead!)
```

## Common Use Cases

**Lighting Quality Comparison (Primary Use):**
```
Use the lighting-quality-comparison skill!
Just say: "Compare probe-grid to multi-light quality"

The skill will automatically:
  - Run analyze_probe_grid
  - Capture screenshots
  - Run LPIPS ML comparison
  - Query RAG for historical data
  - Provide actionable recommendations
```

**Rim Lighting Quality Issues:**
```
diagnose_interpolation → Check interpolation quality at particle edges
validate_probe_coverage → Verify probe density for far particles
validate_sh_coefficients → Check spherical harmonics accuracy
```

**Performance Issues:**
```
analyze_probe_grid → Review current config
optimize_update_pattern → Get recommendations
compare_vs_restir → Validate architecture choice
```

## Integration

Works with:
- **log-analysis-rag** - Ingests probe-grid logs for semantic search
- **dxr-image-quality-analyst** - Visual quality validation
- **pix-debug** - Buffer validation (when updated for probe-grid)

## Architecture Context

**Why Probe-Grid Replaced ReSTIR:**
- ReSTIR crashed at ≥2045 particles (atomic contention, 5.35 particles/voxel)
- Probe-grid: Zero atomics, scales to 100K+ particles
- See: `PROBE_GRID_STATUS_REPORT.md`

**Files:**
- `src/lighting/ProbeGridSystem.{h,cpp}`
- `shaders/probe_grid/update_probes.hlsl`
- `shaders/particles/particle_gaussian_raytrace.hlsl` (query logic)
