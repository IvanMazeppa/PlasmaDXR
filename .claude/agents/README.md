# PlasmaDX Agent Library

Consolidated agent repository with version tracking and specialization hierarchy.

## Agent Versions

### v3 - Production Agents (Latest)
**Status:** Active development, current production use
**Created:** 2025-10-17
**Expertise:** DirectX 12, DXR 1.1, buffer validation, debugging, stress testing, performance analysis

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| `buffer-validator-v3.md` | GPU buffer validation | Binary format validation, NaN/Inf detection, statistical analysis |
| `pix-debugger-v3.md` | Root cause analysis | Multi-light debugging, RT pipeline expertise, file:line fixes |
| `stress-tester-v3.md` | Comprehensive testing | Particle/light scaling, distance scenarios, regression detection |
| `performance-analyzer-v3.md` | Performance profiling | Bottleneck identification, PIX integration, optimization recommendations |

### v2 - Specialized Agents (Active)
**Status:** Active, specialized for specific tasks
**Created:** 2025-09-30
**Expertise:** Mesh shaders, DXR systems, HLSL volumetrics, physics, research

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| `dx12-mesh-shader-engineer-v2.md` | Mesh/Amplification shaders | Mesh shader pipeline, Ada Lovelace workarounds |
| `dxr-graphics-debugging-engineer-v2.md` | DXR rendering diagnosis | Black screens, artifacts, missing effects |
| `dxr-systems-engineer-v2.md` | DXR infrastructure | BLAS/TLAS, SBT, RT tier detection |
| `hlsl-volumetric-implementation-engineer-v2.md` | Volumetric shaders | Ray marching, Beer-Lambert, phase functions |
| `physics-performance-agent-v2.md` | Physics optimization | GPU compute, Keplerian dynamics, temperature models |
| `rt-ml-technique-researcher-v2.md` | Research & innovation | Web search, technique analysis, documentation creation |

**Note:** `rt-ml-technique-researcher-v2` created the 53 research documents in `docs/research/`.

### v1 - Original Agents (Archived)
**Status:** Reference only, superseded by v2
**Created:** 2025-09-23 to 2025-09-30

| Agent | Superseded By |
|-------|--------------|
| `dx12-mesh-shader-engineer.md` | `dx12-mesh-shader-engineer-v2.md` |
| `dxr-graphics-debugging-engineer.md` | `dxr-graphics-debugging-engineer-v2.md` |
| `physics-performance-agent.md` | `physics-performance-agent-v2.md` |

## Usage

### Invoking Agents (Claude Code)

```bash
# v3 Production Agents
@buffer-validator-v3 validate PIX/buffer_dumps/frame_120/g_particles.bin
@pix-debugger-v3 analyze "light radius control has no effect"
@stress-tester-v3 run particle-scaling
@performance-analyzer-v3 profile build/Debug/PlasmaDX-Clean.exe

# v2 Specialized Agents
@rt-ml-technique-researcher-v2 research "NVIDIA RTXDI integration guide"
@physics-performance-agent-v2 optimize particle physics shader
@dxr-systems-engineer-v2 implement BLAS update optimization
```

### Multi-Agent Workflows

**Example 1: Debug Multi-Light Issue**
```bash
# 1. Capture + validate
./build/DebugPIX/PlasmaDX-Clean-PIX.exe --dump-buffers 120
@buffer-validator-v3 validate PIX/buffer_dumps/frame_120/g_lights.bin

# 2. Root cause analysis
@pix-debugger-v3 analyze "lights disappear beyond 300 units"

# 3. Apply fix (pix-debugger-v3 provides file:line and exact code)
```

**Example 2: Research + Implement New Feature**
```bash
# 1. Research phase
@rt-ml-technique-researcher-v2 research "DXR denoising NRD integration"

# 2. Implementation phase
@dxr-systems-engineer-v2 implement NRD denoiser integration

# 3. Validation phase
@performance-analyzer-v3 profile build/Debug/PlasmaDX-Clean.exe
```

## Agent Development Guidelines

### Creating v4 Agents

When creating next-generation agents:

1. **Copy from v3** - Start with proven v3 agent as template
2. **Increment version** - Use `-v4` suffix
3. **Document changes** - Update this README with v4 section
4. **Update plugin.json** - Add v4 agents to manifest
5. **Keep v3 active** - Don't delete previous versions until v4 is proven

### Specialization Hierarchy

```
v3 Production Agents (4)
├── General-purpose debugging/testing
├── Buffer validation expertise
└── Multi-light system knowledge

v2 Specialized Agents (6)
├── Mesh shader specialists
├── Research agents (created docs/research/)
└── Physics optimization experts

v1 Original Agents (3)
└── Archived for reference
```

## Resources Available to Agents

### Research Library
- **Location:** `../../docs/research/AdvancedTechniqueWebSearches/`
- **Count:** 53 markdown documents
- **Topics:** DXR, ML lighting, optimizations, ReSTIR, denoising
- **Creator:** `rt-ml-technique-researcher-v2.md`

### Project Documentation
- **Location:** `../../` (project root)
- **Key files:**
  - `CLAUDE.md` - Current project state, architecture, known issues
  - `MASTER_ROADMAP_V2.md` - Development phases, current priorities
  - `MULTI_LIGHT_FIXES_NEEDED.md` - Active bug list with fixes
  - `CLAUDE_CODE_PLUGINS_GUIDE.md` - Plugin system documentation

### Historical Analysis
- **Location:** `../../docs/archive/`
- **Files:** Driver analysis, mesh shader viability, particle debug architecture
- **Use:** Reference for past decisions and deprecated approaches

## Notes

- **MCP Integration (Future):** Planned watched folder automation for research docs
- **Version Strategy:** Keep all versions for forensic analysis and rollback capability
- **Research Workflow:** v2 researcher creates docs → v3 production agents consume them
- **Testing Strategy:** v3 agents handle testing, v2 agents handle specialized implementation

---

**Last Updated:** 2025-10-17
**Plugin Version:** 3.0.0
**Active Agents:** 10 (4 v3 + 6 v2)
