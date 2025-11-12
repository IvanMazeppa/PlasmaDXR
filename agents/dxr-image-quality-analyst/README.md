# RTXDI Quality Analyzer

**Comprehensive performance and quality analysis agent for DirectX 12 RTXDI ray tracing**

Built with the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) for autonomous diagnostic analysis of the PlasmaDX-Clean RTXDI implementation.

---

## Overview

The RTXDI Quality Analyzer is a specialized diagnostic agent that helps answer the critical question: **"Why isn't RTXDI outperforming the legacy renderer?"**

### Core Features

**1. Performance Comparison Tool**
- Compare FPS, frame times, and GPU metrics between:
  - Legacy renderer
  - RTXDI M4 (weighted reservoir sampling)
  - RTXDI M5 (temporal accumulation)
- Identify which renderer is fastest and by how much
- Generate detailed performance deltas and percentage changes

**2. PIX Capture Analysis Tool**
- Parse PIX .wpix captures for RTXDI-specific metrics:
  - Light grid utilization and spatial coverage
  - Temporal accumulation overhead
  - Reservoir buffer usage
  - Ray tracing dispatch costs
  - BLAS/TLAS rebuild times
- Identify primary bottlenecks with root cause analysis
- Provide actionable optimization recommendations with file:line references

### Extension Points (Future)

- Screenshot capture automation
- Image comparison utilities (SSIM, PSNR, diff heatmaps)
- Automated benchmark sweeps
- Before/after comparison reports

---

## Installation

### Prerequisites

- Python 3.10+ (3.10, 3.11, 3.12, or 3.13)
- PlasmaDX-Clean project at `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean`
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

```bash
# Navigate to the agent directory
cd agents/rtxdi-quality-analyzer

# Create virtual environment
python3 -m venv venv

# Activate virtual environment (WSL2/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here
PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
PIX_PATH=/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64

# Optional (defaults shown)
BUILD_DIR=${PROJECT_ROOT}/build
PIX_CAPTURES_DIR=${PROJECT_ROOT}/PIX/Captures
PIX_BUFFER_DUMPS_DIR=${PROJECT_ROOT}/PIX/buffer_dumps
LOGS_DIR=${PROJECT_ROOT}/logs
```

---

## Usage

The agent can run in two modes:

### 1. Standalone CLI Mode

Test tools independently without MCP integration:

```bash
# Activate virtual environment
source venv/bin/activate

# Performance comparison
python -m src.cli performance \
  --legacy logs/legacy.log \
  --rtxdi-m4 logs/rtxdi_m4.log \
  --rtxdi-m5 logs/rtxdi_m5.log

# PIX capture analysis
python -m src.cli pix --capture PIX/Captures/latest.wpix

# PIX analysis without buffer dumps
python -m src.cli pix --capture PIX/Captures/latest.wpix --no-buffers

# Interactive diagnostic mode
python -m src.cli interactive
```

**Interactive Mode Example:**

```
Query: Why isn't RTXDI faster than the legacy renderer?
[Runs performance comparison and provides detailed analysis]

Query: Analyze the latest PIX capture for bottlenecks
[Parses PIX capture and identifies primary bottleneck]
```

### 2. MCP Server Mode (Claude Code Integration)

Connect the agent to Claude Code sessions via MCP protocol:

```bash
# Start MCP server
python -m src.agent
```

Then in your Claude Code session, use the agent's tools via MCP.

---

## Example Outputs

### Performance Comparison Report

```
================================================================================
RTXDI PERFORMANCE COMPARISON REPORT
================================================================================

METRICS SUMMARY
--------------------------------------------------------------------------------

LEGACY:
  FPS: 120.0 (min: 95.0, max: 142.0)
  Frame Time: 8.30ms (p95: 10.20ms, p99: 12.80ms)
  Particles: 10,000 | Lights: 13
  Resolution: 1920x1080
  GPU Timings:
    - blas_rebuild: 2.10ms
    - tlas_rebuild: 0.80ms
    - rt_lighting: 3.20ms

RTXDI_M4:
  FPS: 115.0 (min: 90.0, max: 135.0)
  Frame Time: 8.70ms (p95: 11.10ms, p99: 13.50ms)
  ...

================================================================================
COMPARISONS
--------------------------------------------------------------------------------

Legacy vs RTXDI M4:
  RTXDI M4 is 4.2% slower than legacy
  FPS: 120.0 → 115.0 (-5.0, -4.2%)
  Frame Time: 8.30ms → 8.70ms (+0.40ms, +4.8%)

================================================================================
FASTEST: LEGACY @ 120.0 FPS
================================================================================

RECOMMENDATIONS
--------------------------------------------------------------------------------

⚠️ RTXDI M4 is slower than legacy renderer. Primary causes likely include:
  - BLAS rebuild is expensive (2.1ms). Consider BLAS updates instead of full rebuild
    (src/lighting/RTLightingSystem_RayQuery.cpp:456)
  - RT lighting is expensive (3.2ms). Consider adaptive ray sampling or spatial reuse (RTXDI M6)
  - Profile with PIX to identify exact bottleneck. Use: @pix-debugging-agent
```

### PIX Analysis Report

```
================================================================================
PIX CAPTURE ANALYSIS - RTXDI BOTTLENECK REPORT
================================================================================

Capture File: PIX/Captures/RTXDI_M4_2025-10-22.wpix

RTXDI METRICS
--------------------------------------------------------------------------------

Light Grid:
  Size: 30×30×30 cells
  Coverage: 78.5%
  Avg lights/cell: 2.3
  Max lights/cell: 8

GPU Timings:
  temporal_accumulation: 1.20ms
  ray_dispatch: 2.80ms
  blas_rebuild: 2.10ms
  tlas_rebuild: 0.80ms
  ────────────────────────────────────────
  TOTAL RTXDI OVERHEAD: 7.90ms

================================================================================
PRIMARY BOTTLENECK
--------------------------------------------------------------------------------

🔴 BLAS rebuild: 2.10ms (26.6% of total overhead)

Consider BLAS updates instead of full rebuild, or implement particle LOD culling

================================================================================
OPTIMIZATION RECOMMENDATIONS
--------------------------------------------------------------------------------

🔴 BLAS rebuild is primary bottleneck (2.1ms)
   Fix: Implement BLAS updates instead of full rebuild
   File: src/lighting/RTLightingSystem_RayQuery.cpp:456
   Expected impact: +25% FPS (reduce from 2.1ms to ~0.5ms)
   Estimated time: 4-6 hours
```

---

## Architecture

### Project Structure

```
rtxdi-quality-analyzer/
├── src/
│   ├── agent.py                    # MCP server mode entry point
│   ├── cli.py                      # Standalone CLI entry point
│   ├── tools/
│   │   ├── performance_comparison.py  # Performance metrics comparison
│   │   └── pix_analysis.py           # PIX capture analysis
│   └── utils/
│       ├── metrics_parser.py         # Parse FPS, frame times, GPU metrics
│       └── pix_parser.py             # Parse .wpix captures
├── requirements.txt
├── .env.example
└── README.md
```

### Tool Descriptions

**`performance_comparison.py`**
- Parses log files from legacy, RTXDI M4, and RTXDI M5 renders
- Compares FPS, frame times, and GPU component timings
- Generates delta reports and identifies fastest renderer
- Provides optimization recommendations

**`pix_analysis.py`**
- Parses PIX .wpix captures using pixtool.exe
- Extracts RTXDI-specific metrics (light grid, reservoirs, timings)
- Identifies primary performance bottleneck
- Provides actionable fixes with file:line references

---

## Integration with PlasmaDX-Clean

This agent integrates with the existing PlasmaDX-Clean infrastructure:

### Log Files

Expected log format in `logs/`:
```
[INFO] FPS: 120.5 (avg: 118.2, min: 95.3, max: 142.7)
[INFO] Frame time: 8.3ms (avg: 8.5ms, p95: 10.2ms, p99: 12.8ms)
[INFO] BLAS rebuild: 2.1ms
[INFO] TLAS rebuild: 0.8ms
[INFO] RT lighting: 3.2ms
```

### PIX Captures

Expected PIX capture location: `PIX/Captures/*.wpix`

Uses `pixtool.exe` to export event lists for timing analysis.

### Buffer Dumps

Expected buffer dumps in `PIX/buffer_dumps/`:
- `g_particles.bin` (32 bytes/particle)
- `g_lights.bin` (32 bytes/light)
- `g_currentReservoirs.bin` (32 bytes/pixel)
- `g_prevReservoirs.bin` (32 bytes/pixel)
- `g_rtLighting.bin` (16 bytes/pixel)

---

## Development

### Type Checking

```bash
# Run mypy for type checking
mypy src/
```

### Adding New Features

The agent is designed for iterative enhancement. To add new features:

1. **Screenshot Capture Automation**
   - Add `src/tools/screenshot_capture.py`
   - Integrate with PlasmaDX window capture API

2. **Image Comparison Utilities**
   - Add `src/utils/image_comparison.py`
   - Use Pillow + scikit-image for SSIM, PSNR, diff heatmaps
   - Uncomment relevant dependencies in `requirements.txt`

3. **Automated Benchmarks**
   - Add `src/tools/benchmark_suite.py`
   - Sweep particle counts, light counts, resolutions
   - Generate before/after comparison reports

### Extension Points

Look for `# TODO:` comments in the code for extension points:
- `metrics_parser.py`: Log file parsing (currently stub)
- `pix_parser.py`: PIX capture parsing (currently stub)
- `agent.py`: MCP server setup (needs `createSdkMcpServer()` integration)

---

## Troubleshooting

### "No PIX capture file found"

Ensure PIX captures exist in `PIX/Captures/` or provide explicit path:
```bash
python -m src.cli pix --capture /path/to/capture.wpix
```

### "Failed to parse log file"

Check log format matches expected structure. Currently using stub implementation - implement full parser in `metrics_parser.py`.

### Type errors during development

Run type checking to verify:
```bash
mypy src/ --ignore-missing-imports
```

---

## Resources

- [Claude Agent SDK Documentation](https://docs.claude.com/en/api/agent-sdk/overview)
- [Python SDK Reference](https://docs.claude.com/en/api/agent-sdk/python)
- [PlasmaDX-Clean CLAUDE.md](../../CLAUDE.md)
- [RTXDI Implementation Summary](../../RTXDI_IMPLEMENTATION_SUMMARY.md)

---

## Current Status

**Version:** 0.1.0

**Implemented:**
- ✅ Project structure and virtual environment
- ✅ Performance comparison tool (stub implementation)
- ✅ PIX analysis tool (stub implementation)
- ✅ Standalone CLI mode
- ✅ Type hints throughout
- ✅ Rich console output

**In Progress:**
- 🔄 Full log file parsing implementation
- 🔄 PIX capture parsing with pixtool.exe
- 🔄 MCP server integration

**Planned:**
- ⏳ Screenshot capture automation
- ⏳ Image comparison utilities (SSIM, PSNR)
- ⏳ Automated benchmark sweeps
- ⏳ Report generation with before/after comparisons

---

**Last Updated:** 2025-10-22
**Author:** Claude Code Agent SDK
**License:** Same as PlasmaDX-Clean project
