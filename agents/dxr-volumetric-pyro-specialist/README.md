# DXR Volumetric Pyro Specialist - MCP Server

A specialized design consultant for pyrotechnic and explosive volumetric effects in DirectX 12 DXR 1.1 raytraced particle renderer.

## Overview

This MCP server provides tools for designing and analyzing volumetric pyro effects (explosions, fire, smoke) for real-time rendering at 90-120 FPS.

**Position in Multi-Agent Pipeline:**
- **Receives from**: `gaussian-analyzer` (baseline volumetric analysis, performance estimates)
- **Provides to**: `material-system-engineer` (detailed pyro specs for code generation)
- **Validated by**: `dxr-image-quality-analyst` (visual quality assessment, FPS measurement)

## Tools Provided

### 1. `research_pyro_techniques`
Research cutting-edge volumetric pyro techniques (NVIDIA RTX Volumetrics, GPU pyro solvers, neural rendering).

### 2. `design_explosion_effect`
Design complete explosion specifications (supernova, stellar flare, accretion burst) with:
- Temporal dynamics (expansion curves, temperature decay, opacity fade)
- Material properties (scattering, absorption, emission, phase function)
- Procedural noise parameters (SimplexNoise3D configuration)
- Color profiles (temperature → RGB blackbody)
- Performance estimates (FPS impact, ALU operations)

### 3. `design_fire_effect`
Design fire/smoke effects (stellar fire, nebula wisps, plasma jets) with:
- Turbulence parameters (Perlin noise configuration)
- Scattering profiles and opacity curves
- Temperature gradients and color mapping

### 4. `estimate_pyro_performance`
Estimate FPS impact of pyro effects:
- ALU operation counts per particle
- Memory bandwidth requirements
- Temporal buffer overhead
- FPS predictions on RTX 4060 Ti @ 1080p

### 5. `compare_pyro_techniques`
Compare implementation approaches (particle-based vs OpenVDB vs hybrid vs GPU pyro solvers) across:
- Visual quality
- Performance (FPS at different particle counts)
- Implementation complexity
- Memory usage

## Installation

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-volumetric-pyro-specialist

# The run_server.sh script handles everything:
./run_server.sh
```

This will:
1. Create virtual environment
2. Install dependencies (mcp, python-dotenv, rich, requests, etc.)
3. Launch MCP server

## MCP Server Configuration

Already configured in your `.claude.json`:

```json
{
  "mcpServers": {
    "dxr-volumetric-pyro-specialist": {
      "type": "stdio",
      "command": "bash",
      "args": [
        "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-volumetric-pyro-specialist/run_server.sh"
      ],
      "env": {
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
      }
    }
  }
}
```

## Usage Examples

### Design a Supernova Explosion
```
Use the dxr-volumetric-pyro-specialist tool design_explosion_effect to create a supernova with:
- Duration: 5 seconds
- Max radius: 500 meters
- Peak temperature: 100000K
- Particle budget: 10000
```

### Estimate Performance Impact
```
Use estimate_pyro_performance to check FPS impact of a moderate complexity effect with 10K particles and 3 noise octaves
```

### Compare Approaches
```
Use compare_pyro_techniques to compare particle_gaussian vs hybrid vs openvdb for visual quality and performance
```

## Output Format

All tools return detailed specifications with:
- **HLSL code snippets** (temporal dynamics, noise configuration, material properties)
- **Shader integration points** (which files to modify, buffer requirements)
- **Performance estimates** (FPS impact, ALU counts, memory bandwidth)
- **Validation criteria** (for dxr-image-quality-analyst to verify)

## Technical Domain

- **Graphics API**: DirectX 12 DXR 1.1 inline ray tracing (RayQuery API)
- **Shaders**: HLSL Shader Model 6.5+ compute shaders
- **Rendering**: 3D Gaussian volumetric rendering (ray-ellipsoid intersection)
- **Physics**: Beer-Lambert absorption, Henyey-Greenstein scattering, blackbody emission
- **Target Performance**: 90-120 FPS on RTX 4060 Ti (1920×1080, 10K particles)

## Celestial Effects Specialization

Designed for astrophysical visualization:
- Supernovae (Type II core-collapse)
- Solar prominences and stellar flares
- Accretion disk fires and bursts
- Nebula wisps and dust clouds
- Explosive phenomena

## Collaboration Model

**NO TOOL DUPLICATION** - specialized design specifications ONLY:

- ✅ **This agent**: Design pyro effect specifications (temporal curves, material properties, noise params)
- ❌ **Defers to material-system-engineer**: File operations, code generation, HLSL shader writing
- ❌ **Defers to gaussian-analyzer**: Generic material analysis, particle structure validation
- ❌ **Defers to dxr-image-quality-analyst**: Visual validation, screenshot comparison, FPS measurement

## Dependencies

- `mcp==1.21.0` - MCP Server SDK
- `python-dotenv==1.0.0` - Environment variables
- `rich==13.7.0` - Beautiful report formatting
- `requests==2.31.0` - Web research
- `beautifulsoup4==4.12.2` - HTML parsing
- `pandas==2.1.3` - Data analysis
- `numpy==1.26.2` - Numerical computations

## Testing

The server provides 5 specialized tools for pyro effects design. Test with:

```bash
# In Claude Code, invoke a tool:
@dxr-volumetric-pyro-specialist design_explosion_effect with effect_type="supernova"
```

## Version History

- **1.0.0** (2025-11-14): Initial MCP server implementation
  - 5 specialized pyro design tools
  - Integration with gaussian-analyzer and material-system-engineer
  - Performance estimation for RTX 4060 Ti
  - Celestial effects specialization (supernovae, stellar flares)

## Architecture

```
dxr-volumetric-pyro-specialist/
├── pyro_server.py              # Main MCP server
├── run_server.sh               # Server launcher
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
└── src/
    └── tools/
        ├── pyro_research.py          # Research cutting-edge techniques
        ├── explosion_designer.py     # Design explosion effects
        ├── fire_designer.py          # Design fire/smoke effects
        ├── performance_estimator.py  # Estimate FPS impact
        └── technique_comparator.py   # Compare approaches
```

## License

Part of the PlasmaDX-Clean project.

---

**Created by**: Claude Code (rebuilt as MCP server from Agent SDK prototype)
**Date**: 2025-11-14
**MCP SDK Version**: 1.21.0
