# DXR Volumetric Pyro Specialist Agent

A specialized design consultant for pyrotechnic and explosive volumetric effects in DirectX 12 DXR 1.1 raytraced particle renderer.

## Overview

This agent translates high-level effect requests (e.g., "add supernova explosions", "create solar flares") into implementation-ready material specifications for volumetric 3D Gaussian particle systems.

**Position in Multi-Agent Pipeline:**
- **Receives from**: `gaussian-analyzer` (baseline volumetric analysis, performance estimates)
- **Provides to**: `material-system-engineer` (detailed pyro specs for code generation)
- **Validated by**: `dxr-image-quality-analyst` (visual quality assessment, FPS measurement)

## Key Capabilities

1. **Research Cutting-Edge Techniques**: NVIDIA RTX Volumetrics, neural rendering, GPU pyro solvers
2. **Design Explosion Dynamics**: Temporal expansion curves, temperature decay, shockwave propagation
3. **Design Fire/Smoke Materials**: Scattering coefficients, opacity profiles, turbulence specifications
4. **Generate Procedural Noise Parameters**: SimplexNoise, Perlin, Worley for flickering/turbulence
5. **Estimate Performance Impact**: FPS impact of animated volumetric effects, ALU operation counts
6. **Compare Implementation Approaches**: Particle-based vs OpenVDB vs hybrid

## Technical Domain

- **Graphics API**: DirectX 12 DXR 1.1 inline ray tracing (RayQuery API)
- **Shaders**: HLSL Shader Model 6.5+ compute shaders
- **Rendering**: 3D Gaussian volumetric rendering (ray-ellipsoid intersection)
- **Physics**: Beer-Lambert absorption, Henyey-Greenstein scattering, blackbody emission
- **Target Performance**: 90-120 FPS on RTX 4060 Ti (1920×1080, 10K particles)

## Installation

### Prerequisites

- Python 3.10 or higher (tested with Python 3.12)
- Claude Code MAX subscription (session-based authentication, no API key needed)
- WSL2 environment (Ubuntu on Windows)

### Setup

1. **Navigate to agent directory:**
   ```bash
   cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-volumetric-pyro-specialist
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and set PROJECT_ROOT to your PlasmaDX-Clean directory
   ```

5. **Verify installation:**
   ```bash
   python3 -c "import claude_agent_sdk; print(f'SDK version: {claude_agent_sdk.__version__}')"
   ```

   Expected output: `SDK version: 0.1.6` (or later)

## Usage

### Interactive Mode

Run the agent in interactive consultation mode:

```bash
python3 main.py
```

Example queries:
- "Design a supernova explosion effect"
- "Create solar flare material specifications"
- "Compare particle-based vs OpenVDB for nebula wisps"
- "Estimate FPS impact of adding fire effects to 10K particles"

### Single Query Mode

Run a single design consultation:

```bash
python3 main.py "Design a supernova explosion effect with realistic temperature gradients"
```

### MCP Server Mode

Launch as an MCP server for integration with other agents:

```bash
./run_server.sh
```

Add to `~/.claude/mcp_settings.json`:
```json
{
  "mcpServers": {
    "dxr-pyro-specialist": {
      "command": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/dxr-volumetric-pyro-specialist/run_server.sh",
      "args": [],
      "env": {}
    }
  }
}
```

## Agent Configuration

### System Prompt

The agent uses a specialized system prompt that defines:
- Role as pyro effects design consultant
- Position in multi-agent pipeline
- Technical domain expertise (DXR 1.1, HLSL, volumetric rendering)
- Communication style (brutal honesty, quantitative analysis)
- Output format for pyro specifications

### Allowed Tools

- `WebSearch`: Research cutting-edge volumetric pyro techniques
- `WebFetch`: Fetch technical documentation (NVIDIA RTX, papers, blog posts)
- `Read`: Read project files (CLAUDE.md, shader code for context)

### Model Configuration

- **Model**: `claude-sonnet-4-5-20250929` (latest with vision capabilities)
- **Permission Mode**: `bypassPermissions` (autonomous operation in pipeline)
- **Settings Sources**: Project and user settings from `.claude/` directories

## Example Output

When you ask for a pyro effect design, the agent provides:

1. **Effect Overview**: Name, visual description, celestial context
2. **Temporal Dynamics**: Expansion curve, temperature decay, opacity fade
3. **Material Properties**: Scattering, absorption, emission, phase function
4. **Procedural Noise**: Algorithm, frequency, amplitude, octaves
5. **Color Profile**: Temperature gradient (K → RGB)
6. **Performance Estimate**: FPS impact, ALU ops/particle, memory bandwidth
7. **Shader Integration Points**: Which compute shader, which pass, buffer requirements
8. **Validation Criteria**: Expected visual characteristics

## Collaboration Model

**NO TOOL DUPLICATION** - this agent focuses on design specifications ONLY:

- **Defer to material-system-engineer**: File operations, code generation, C++ structs, HLSL shaders
- **Defer to gaussian-analyzer**: Generic material analysis, particle structure validation
- **Defer to dxr-image-quality-analyst**: Visual validation, screenshot comparison, FPS measurement

## Performance Targets

- **Particle budget**: 10K-100K particles (volumetric 3D Gaussians)
- **Frame budget**: 8-11ms per frame (90-120 FPS target)
- **Target hardware**: RTX 4060 Ti (8GB VRAM, ~22 TFLOPS FP32)

## Development

### Running Tests

```bash
# Basic import test
python3 -c "from main import PYRO_SPECIALIST_SYSTEM_PROMPT; print('✓ Imports successful')"

# Run agent with test query
python3 main.py "Design a minimal test explosion"
```

### Debugging

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python3 main.py
```

### Agent SDK Version

This agent uses **claude-agent-sdk 0.1.6** (latest as of 2025-11-14) with:
- Enhanced vision capabilities for screenshot analysis
- Latest Claude Sonnet 4.5 model
- Improved session management
- Better error handling

Check for updates:
```bash
pip install --upgrade claude-agent-sdk
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'claude_agent_sdk'`:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

If you see `KeyError: 'PROJECT_ROOT'`:
```bash
cp .env.example .env
# Edit .env and set PROJECT_ROOT
```

### MCP Server Issues

If the MCP server won't start:
```bash
# Check if virtual environment exists
ls venv/

# Recreate if needed
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Related Documentation

- [Claude Agent SDK - Python Reference](https://docs.claude.com/en/api/agent-sdk/python)
- [PlasmaDX-Clean CLAUDE.md](../../CLAUDE.md) - Project overview and guidelines
- [Material System Engineer Agent](../material-system-engineer/) - Code generation partner
- [Gaussian Analyzer Agent](../gaussian-analyzer/) - Volumetric analysis partner
- [DXR Image Quality Analyst](../dxr-image-quality-analyst/) - Visual validation partner

## Version History

- **1.0.0** (2025-11-14): Initial release
  - Specialized system prompt for pyro effects design
  - Interactive and single-query modes
  - MCP server integration
  - Latest Claude Agent SDK 0.1.6 with vision capabilities

## License

Part of the PlasmaDX-Clean project. See main repository for license information.

## Author

Created by Claude Agent SDK for the PlasmaDX-Clean autonomous agent pipeline.
