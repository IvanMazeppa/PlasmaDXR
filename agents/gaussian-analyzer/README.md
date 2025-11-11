# Gaussian Analyzer MCP Server

Specialized MCP server providing computational tools for 3D Gaussian volumetric particle analysis and optimization. Works in collaboration with the `3d-gaussian-volumetric-engineer` Claude Code agent to analyze particle structures, simulate material properties, and estimate performance impacts.

## Overview

This MCP server provides 5 specialized tools for Gaussian particle system analysis:

1. **analyze_gaussian_parameters** - Analyze current particle structure and identify gaps
2. **simulate_material_properties** - Test how material changes affect rendering
3. **estimate_performance_impact** - Calculate FPS impact of proposed changes
4. **compare_rendering_techniques** - Compare volumetric rendering approaches
5. **validate_particle_struct** - Validate GPU compatibility of particle structures

## Installation

### Prerequisites

- Python 3.10+
- PlasmaDX-Clean project

### Setup

```bash
cd agents/gaussian-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env to set PROJECT_ROOT if needed
nano .env
```

### Running the Server

**Option 1: Use run script (recommended)**

```bash
./run_server.sh
```

**Option 2: Manual start**

```bash
source venv/bin/activate
python3 gaussian_server.py
```

## Tool Documentation

### 1. analyze_gaussian_parameters

Analyzes the current 3D Gaussian particle structure and proposes extensions for diverse celestial body rendering.

**Parameters:**
- `analysis_depth` (optional): `"quick"`, `"detailed"` (default), or `"comprehensive"`
- `focus_area` (optional): `"structure"`, `"shaders"`, `"materials"`, `"performance"`, or `"all"` (default)

**Example Usage:**

```python
# Via MCP tool call
mcp__gaussian-analyzer__analyze_gaussian_parameters(
    analysis_depth="comprehensive",
    focus_area="all"
)
```

**Output:**
- Current particle structure analysis (32-byte baseline)
- Missing properties for celestial diversity
- Shader pipeline analysis
- Material type proposals (8 types recommended)
- Performance impact estimates
- Recommendations for 48-byte vs 64-byte structures

### 2. simulate_material_properties

Simulates how proposed material property changes would affect rendering appearance and performance.

**Parameters:**
- `material_type` (required): Material type to simulate
  - `"PLASMA_BLOB"`, `"STAR_MAIN_SEQUENCE"`, `"STAR_GIANT"`, `"STAR_HYPERGIANT"`
  - `"GAS_CLOUD"`, `"DUST_PARTICLE"`, `"ROCKY_BODY"`, `"ICY_BODY"`, `"CUSTOM"`
- `properties` (optional): Dict of material properties
  - `opacity`: 0.0-1.0
  - `scattering_coefficient`: 0.0+
  - `emission_multiplier`: 0.0+
  - `albedo_rgb`: [r, g, b] (0.0-1.0 each)
  - `phase_function_g`: -1.0 to 1.0
- `render_mode` (optional): `"volumetric_only"` (default), `"hybrid_surface_volume"`, `"comparison"`

**Example Usage:**

```python
# Simulate wispy gas cloud
mcp__gaussian-analyzer__simulate_material_properties(
    material_type="GAS_CLOUD",
    properties={
        "opacity": 0.3,
        "scattering_coefficient": 1.5,
        "emission_multiplier": 0.1,
        "albedo_rgb": [0.6, 0.7, 0.9],
        "phase_function_g": -0.3  # Backward scattering for wispy appearance
    },
    render_mode="volumetric_only"
)
```

**Output:**
- Visual appearance prediction (emission vs scattering dominated)
- Performance impact estimate (shader ALU operations)
- Required shader modifications (HLSL code snippets)
- Recommendations for implementation

### 3. estimate_performance_impact

Calculates estimated FPS impact of proposed particle structure or shader modifications.

**Parameters:**
- `particle_struct_bytes` (required): Proposed struct size (32-128 bytes)
- `material_types_count` (optional): Number of material types (default: 5)
- `shader_complexity` (optional): `"minimal"`, `"moderate"` (default), or `"complex"`
- `particle_counts` (optional): Array of particle counts to test (default: [10000, 50000, 100000])

**Example Usage:**

```python
# Estimate 48-byte structure with 8 materials
mcp__gaussian-analyzer__estimate_performance_impact(
    particle_struct_bytes=48,
    material_types_count=8,
    shader_complexity="moderate",
    particle_counts=[10000, 50000, 100000]
)
```

**Output:**
- Memory impact analysis (current vs proposed)
- Shader complexity overhead
- Projected FPS at various particle counts
- Performance targets validation (90-120 FPS @ 10K)
- Bottleneck analysis
- Optimization recommendations

### 4. compare_rendering_techniques

Compares different volumetric rendering approaches across multiple criteria.

**Parameters:**
- `techniques` (optional): Array of techniques to compare (default: volumetric + hybrid)
  - `"current_implementation"` (32-byte PLASMA baseline)
  - `"pure_volumetric_gaussian"` (48-byte multi-material)
  - `"hybrid_surface_volume"` (64-byte + surface rendering)
  - `"billboard_impostors"` (sprite-based)
  - `"adaptive_lod"` (distance-based simplification)
- `criteria` (optional): Array of comparison criteria
  - `"performance"`, `"visual_quality"`, `"memory_usage"`
  - `"implementation_complexity"`, `"material_flexibility"`

**Example Usage:**

```python
# Compare volumetric vs hybrid approaches
mcp__gaussian-analyzer__compare_rendering_techniques(
    techniques=["pure_volumetric_gaussian", "hybrid_surface_volume", "adaptive_lod"],
    criteria=["performance", "visual_quality", "material_flexibility"]
)
```

**Output:**
- Side-by-side comparison tables
- Score matrices (performance, quality, complexity, flexibility)
- Overall rankings
- Strategic recommendations
- Best approach for different priorities

### 5. validate_particle_struct

Validates proposed particle structure for GPU alignment, size constraints, and backward compatibility.

**Parameters:**
- `struct_definition` (required): C++ struct code (string)
- `check_backward_compatibility` (optional): Check against 32-byte legacy (default: True)
- `check_gpu_alignment` (optional): Validate 16-byte alignment (default: True)

**Example Usage:**

```python
# Validate proposed 48-byte structure
struct_code = """
struct ParticleData {
    XMFLOAT3 position;      // 12 bytes
    XMFLOAT3 velocity;      // 12 bytes
    float temperature;      // 4 bytes
    float radius;           // 4 bytes
    XMFLOAT3 albedo;        // 12 bytes (NEW)
    uint32_t materialType;  // 4 bytes (NEW)
};
"""

mcp__gaussian-analyzer__validate_particle_struct(
    struct_definition=struct_code,
    check_backward_compatibility=True,
    check_gpu_alignment=True
)
```

**Output:**
- Field-by-field analysis (type, size, offset)
- GPU alignment validation (16-byte boundary check)
- Backward compatibility check (vs 32-byte legacy)
- Common issues detection (float3 vs XMFLOAT3, bool types, etc.)
- Final verdict (PASS/FAIL with actionable fixes)

## Integration with Claude Code

### MCP Server Configuration

Add to your Claude Code MCP settings (`.claude/settings.json` or `~/.config/claude/settings.json`):

```json
{
  "mcpServers": {
    "gaussian-analyzer": {
      "command": "bash",
      "args": ["-c", "cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/gaussian-analyzer && ./run_server.sh"],
      "env": {
        "PROJECT_ROOT": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
      }
    }
  }
}
```

### Using with 3D Gaussian Volumetric Engineer Agent

The `3d-gaussian-volumetric-engineer` agent is designed to work seamlessly with this MCP server:

```bash
# Launch the agent
/agent 3d-gaussian-volumetric-engineer

# Agent will automatically use these tools during its workflow
# Phase 1: analyze_gaussian_parameters (structure analysis)
# Phase 2: simulate_material_properties (experimentation)
# Phase 3: estimate_performance_impact (architecture design)
# Phase 4: validate_particle_struct (prototyping)
# Phase 5: compare_rendering_techniques (optimization)
```

### Collaboration with RTXDI Quality Analyzer

These tools complement the existing `rtxdi-quality-analyzer` MCP server:

**Gaussian Analyzer (this server):**
- Structural analysis and design proposals
- Performance estimation and simulation
- GPU compatibility validation

**RTXDI Quality Analyzer:**
- Visual quality assessment (ML-powered screenshot comparison)
- Performance measurement (actual FPS from logs)
- PIX capture analysis (real bottleneck identification)

**Workflow:**
1. Use `gaussian-analyzer` to design material system (Phase 1-3)
2. Implement proposed changes
3. Use `rtxdi-quality-analyzer` to validate visual quality (screenshots)
4. Use `rtxdi-quality-analyzer` to measure actual performance (logs)
5. Iterate based on real measurements

## Development

### Project Structure

```
gaussian-analyzer/
├── gaussian_server.py          # Main MCP server
├── run_server.sh              # Launcher script
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
└── src/
    └── tools/
        ├── parameter_analyzer.py      # Tool 1: Structure analysis
        ├── material_simulator.py      # Tool 2: Material simulation
        ├── performance_estimator.py   # Tool 3: Performance estimation
        ├── technique_comparator.py    # Tool 4: Technique comparison
        └── struct_validator.py        # Tool 5: Structure validation
```

### Adding New Tools

1. Create tool function in `src/tools/your_tool.py`
2. Import in `gaussian_server.py`
3. Add to `@server.list_tools()` decorator
4. Add handler in `@server.call_tool()` decorator
5. Update this README with tool documentation

### Testing Tools

```bash
# Activate virtual environment
source venv/bin/activate

# Test individual tool (example)
python3 -c "
import asyncio
from src.tools.parameter_analyzer import analyze_gaussian_parameters

async def test():
    result = await analyze_gaussian_parameters(
        '/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean',
        analysis_depth='detailed',
        focus_area='all'
    )
    print(result)

asyncio.run(test())
"
```

## Troubleshooting

### Server won't start

```bash
# Check Python version (requires 3.10+)
python3 --version

# Reinstall dependencies
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### "PROJECT_ROOT not found" errors

```bash
# Set PROJECT_ROOT in .env file
echo "PROJECT_ROOT=/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean" > .env
```

### MCP connection issues

```bash
# Check if server is running
ps aux | grep gaussian_server

# Test server directly
python3 gaussian_server.py
# Should not error, will wait for MCP client connection
```

### Tool returns empty results

- Verify `PROJECT_ROOT` points to correct directory
- Check that key files exist:
  - `src/particles/ParticleSystem.h`
  - `shaders/particles/particle_gaussian_raytrace.hlsl`
  - `shaders/particles/gaussian_common.hlsl`

## Examples

See the `3d-gaussian-volumetric-engineer` agent documentation for complete workflow examples using these tools.

**Quick example:**

```bash
# 1. Launch agent
/agent 3d-gaussian-volumetric-engineer

# 2. Request analysis
"Analyze the current Gaussian particle structure and propose extensions for supporting gas clouds and stars"

# Agent will:
# - Call analyze_gaussian_parameters (comprehensive analysis)
# - Call simulate_material_properties for GAS_CLOUD and STAR types
# - Call estimate_performance_impact for 48-byte structure
# - Call validate_particle_struct for proposed structure
# - Present complete architectural proposal
```

## License

Part of the PlasmaDX-Clean project. See main repository for license information.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review main project CLAUDE.md for context
3. See rtxdi-quality-analyzer for similar MCP server reference
