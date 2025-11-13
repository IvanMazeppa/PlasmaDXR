# Material System Engineer

**Autonomous implementation agent** for PlasmaDX-Clean material system design and build orchestration.

## Overview

The Material System Engineer is a specialized AI agent that transforms material system designs into working code. It handles:

- **File Operations**: Read/write codebase files with automatic backup
- **Code Generation**: Generate HLSL shaders, C++ structs, material configs
- **Integration**: Create test scenarios, ImGui controls, and validation configs
- **Build Orchestration**: Provide compilation instructions (Phase A: manual, Phase B: automatic)

## Quick Start

### 1. Setup

```bash
# Navigate to agent directory
cd agents/material-system-engineer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Verify PROJECT_ROOT path in .env
nano .env  # Should be: /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
```

### 2. Run MCP Server

```bash
# Using launcher script (recommended)
./run_server.sh

# Or manually
source venv/bin/activate
python material_engineer_server.py
```

### 3. Connect from Claude Code

Add to your Claude Code MCP settings (`~/.claude/mcp_settings.json`):

```json
{
  "mcpServers": {
    "material-system-engineer": {
      "command": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/material-system-engineer/run_server.sh"
    }
  }
}
```

---

## Tool Reference

### File Operations

**read_codebase_file(file_path)**
- Read any project file (shader/C++/header/JSON)
- Returns file contents with size and line count
- Example: `read_codebase_file("src/particles/ParticleSystem.h")`

**write_codebase_file(file_path, content, create_backup=true)**
- Write file with automatic timestamped backup to `.backups/`
- Creates parent directories if needed
- Example: `write_codebase_file("src/particles/ParticleSystem.h", new_content)`

**search_codebase(pattern, file_glob="**/*", max_results=100)**
- Search for pattern across codebase (regex supported)
- Filter by file pattern (e.g., "**/*.hlsl", "**/*.cpp")
- Returns matches with file paths and line numbers
- Example: `search_codebase("MaterialType", "**/*.{h,cpp}")`

### Code Generation

**generate_material_shader(material_type, properties, base_shader_template="volumetric_raytracing")**
- Generate complete HLSL shader code for material type
- Includes ray marching, phase functions, Beer-Lambert absorption
- Properties: opacity, scattering_coefficient, emission_multiplier, albedo_rgb, phase_function_g
- Example:
  ```python
  generate_material_shader(
      "GAS_CLOUD",
      {
          "opacity": 0.3,
          "scattering_coefficient": 1.5,
          "emission_multiplier": 0.1,
          "albedo_rgb": [0.6, 0.7, 0.9],
          "phase_function_g": -0.3
      }
  )
  ```

**generate_particle_struct(base_struct, new_fields, target_alignment=16)**
- Generate C++ particle struct with GPU alignment
- Automatically calculates padding for 16-byte alignment
- Validates struct size and provides compile-time assertions
- Example:
  ```python
  generate_particle_struct(
      "current_48byte",
      [
          {"type": "float", "name": "roughness", "size_bytes": 4, "comment": "Surface roughness (0-1)"},
          {"type": "float", "name": "metallic", "size_bytes": 4, "comment": "Metallic property (0-1)"}
      ]
  )
  ```

**generate_material_config(material_definitions, output_format="json")**
- Generate material property configuration files
- Formats: json, cpp_array, hlsl_constants
- Example:
  ```python
  generate_material_config(
      [
          {
              "type": "GAS_CLOUD",
              "opacity": 0.3,
              "scattering_coefficient": 1.5,
              "emission_multiplier": 0.1,
              "albedo_rgb": [0.6, 0.7, 0.9],
              "phase_function_g": -0.3,
              "description": "Wispy gas cloud with backward scattering"
          }
      ],
      "json"
  )
  ```

### Integration Tools

**create_test_scenario(name, particle_count, material_distribution, lighting_preset="stellar_ring", camera_distance=800.0, output_format="json")**
- Generate test scenario for material system validation
- Material distribution must sum to 1.0
- Creates JSON config or markdown documentation
- Example:
  ```python
  create_test_scenario(
      "gas_cloud_test",
      10000,
      {
          "GAS_CLOUD": 0.3,
          "STAR_MAIN_SEQUENCE": 0.2,
          "PLASMA": 0.5
      }
  )
  ```

**generate_imgui_controls(material_types, output_format="cpp")**
- Generate ImGui UI code for material property editing
- Creates sliders, color pickers, preset buttons
- Formats: cpp (code), markdown (documentation)
- Example: `generate_imgui_controls(["PLASMA", "GAS_CLOUD", "STAR"])`

**validate_file_syntax(file_path, file_content, file_type="auto")**
- Basic syntax validation for generated code
- Checks braces, parentheses, JSON validity
- File types: auto (detect from extension), cpp, hlsl, json
- Example: `validate_file_syntax("ParticleSystem.h", content)`

---

## Collaboration with Other Agents

### Material System Engineer (YOU)
**Role**: Implementation & build orchestration
**Tools**: File ops, code generation, integration
**Outputs**: Working code, configs, test scenarios

### gaussian-analyzer
**Role**: Design & architectural analysis
**Tools**: Structure analysis, material simulation, performance estimation
**Outputs**: Design recommendations, performance predictions

### dxr-image-quality-analyst
**Role**: Visual quality assessment
**Tools**: Screenshot analysis, LPIPS comparison, performance measurement
**Outputs**: Quality grades (7 dimensions), brutal feedback

### Workflow Example

```
User: "Add gas cloud and star materials"
  ↓
material-system-engineer (YOU):
  1. Read current codebase (read_codebase_file)
  2. Request gaussian-analyzer design recommendations
  3. Generate code based on design
  4. Write files with backups
  5. Create test scenario
  6. Provide compilation instructions
  ↓
User: Compiles and runs
  ↓
dxr-image-quality-analyst:
  1. Assess visual quality (screenshot)
  2. Measure FPS
  3. Provide brutal feedback
  ↓
material-system-engineer (YOU):
  1. Iterate based on feedback
  2. Adjust material properties
  3. Regenerate code
```

---

## Phase A vs Phase B Capabilities

### Current: Phase A (File Operations + Code Generation)

✅ Read/write codebase files
✅ Generate HLSL shaders
✅ Generate C++ structs
✅ Generate material configs
✅ Create test scenarios
✅ Generate ImGui controls
✅ Validate syntax
❌ Compile shaders automatically
❌ Build projects automatically
❌ Run tests automatically

**Phase A Limitation**: Manual compilation required

```bash
# User must run these commands manually
dxc.exe -T cs_6_5 -E main shaders/particles/particle_gaussian_raytrace.hlsl -Fo build/Debug/shaders/.../particle_gaussian_raytrace.dxil
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64 /t:Build
```

### Future: Phase B (Compilation + Validation)

Phase B will add 4 new tools:

🔮 **compile_shader**: Automatic DXC compilation with error reporting
🔮 **build_project**: Automatic MSBuild with build log parsing
🔮 **patch_code**: Surgical code modifications with auto-rollback
🔮 **run_tests**: Execute validation tests automatically

**Phase B Advantage**: Full autonomous iteration loop
1. Generate code
2. Compile automatically
3. If errors → fix and recompile
4. If success → run tests
5. Report results

No manual intervention required!

---

## File Backup System

All file writes create automatic timestamped backups:

```
.backups/
├── ParticleSystem.h.2025-11-13-14-32-01.backup
├── ParticleSystem.h.2025-11-13-15-45-23.backup
├── particle_gaussian_raytrace.hlsl.2025-11-13-14-33-15.backup
└── ...
```

**Restore from backup:**
```bash
# List backups
ls -lt .backups/

# Restore specific backup
cp .backups/ParticleSystem.h.2025-11-13-14-32-01.backup src/particles/ParticleSystem.h
```

---

## Directory Structure

```
material-system-engineer/
├── material_engineer_server.py  # Main MCP server
├── AGENT_PROMPT.md              # Agent system instructions
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── run_server.sh                # Server launcher script
├── src/
│   └── tools/
│       ├── __init__.py
│       ├── file_operations.py    # read_file, write_file, search
│       ├── code_generator.py     # generate_shader, generate_struct, generate_config
│       └── integration_tools.py  # test_scenario, imgui_controls, validate_syntax
└── venv/                        # Python virtual environment
```

---

## Troubleshooting

### Server Won't Start

```bash
# Check Python version (need 3.8+)
python3 --version

# Verify dependencies installed
pip list | grep mcp

# Check .env file exists and has correct path
cat .env

# Run with verbose output
python material_engineer_server.py
```

### Tool Execution Errors

**File not found errors:**
- Verify PROJECT_ROOT in .env is correct
- Use relative paths from project root (not absolute paths)

**Syntax validation errors:**
- Run validate_file_syntax on generated code before writing
- Check for mismatched braces/parentheses

**Backup errors:**
- Ensure write permissions on project directory
- .backups/ directory is created automatically

### Performance Issues

**Slow file searches:**
- Use specific file_glob patterns (e.g., "**/*.hlsl" not "**/*")
- Limit max_results to 50-100 for faster results

**Large generated files:**
- Generated shaders can be 500+ lines
- Use validate_file_syntax before writing to catch errors early

---

## Development & Extension

### Adding New Tools (Phase B Preview)

To add compilation tools in Phase B:

1. Create `src/tools/build_operations.py`
2. Implement `compile_shader()` function
3. Add tool registration in `material_engineer_server.py`
4. Update AGENT_PROMPT.md with new capabilities

Example template:

```python
# src/tools/build_operations.py

async def compile_shader(
    shader_path: str,
    entry_point: str = "main",
    target: str = "cs_6_5"
) -> str:
    """
    Compile HLSL shader with DXC

    Returns compilation result with errors/warnings
    """
    # Implementation here
    pass
```

### Testing Tools

```python
# Test file operations
python -c "
import asyncio
from src.tools.file_operations import read_codebase_file

result = asyncio.run(read_codebase_file(
    '/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean',
    'src/particles/ParticleSystem.h'
))
print(result)
"
```

---

## License

Part of PlasmaDX-Clean project. See project root LICENSE file.

## Contact

For issues, feature requests, or collaboration:
- Open issue in PlasmaDX-Clean repository
- Contact via project communication channels

---

**Version**: 1.0.0 (Phase A)
**Last Updated**: 2025-11-13
**Status**: Production-ready (Phase A tools)
**Next Milestone**: Phase B (Compilation + Validation tools)
