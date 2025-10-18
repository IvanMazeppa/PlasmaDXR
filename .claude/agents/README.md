# PlasmaDX Agent Library

Consolidated agent repository with version tracking and specialization hierarchy.

## Agent Versions

### v4 - RTXDI & Advanced RT (Latest)
**Status:** Active development, Phase 4 RTXDI integration
**Created:** 2025-10-18
**Expertise:** NVIDIA RTXDI, ReGIR, soft shadows, PCSS, volumetric ray tracing

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| `rtxdi-integration-specialist-v4.md` | RTXDI integration | ReSTIR → RTXDI migration, light grid, ReGIR, volumetric adaptation |
| `dxr-rt-shadow-engineer-v4.md` | Advanced shadows | PCSS, soft shadows, RTXDI visibility reuse, temporal filtering |

**Note:** These agents are globally registered in `~/.claude/agents/` for use across all projects.

### v3 - Production Agents
**Status:** Active, current production use
**Created:** 2025-10-17
**Expertise:** DirectX 12, DXR 1.1, buffer validation, debugging, stress testing, performance analysis

| Agent | Purpose | Key Capabilities |
|-------|---------|------------------|
| `buffer-validator-v3.md` | GPU buffer validation | Binary format validation, NaN/Inf detection, statistical analysis |
| `pix-debugger-v3.md` | Root cause analysis (DEPRECATED) | **⚠️ Superseded by PIX MCP Server** (see below) |
| `stress-tester-v3.md` | Comprehensive testing | Particle/light scaling, distance scenarios, regression detection |
| `performance-analyzer-v3.md` | Performance profiling | Bottleneck identification, PIX integration, optimization recommendations |

**⚠️ PIX Debugger v3 → MCP Server Migration:**
- The PIX debugging capabilities have been migrated to a **standalone MCP server**
- Location: `/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/`
- MCP Server: `pix-debug` (registered in Claude Code)
- Tools: `capture_buffers`, `analyze_restir_reservoirs`, `analyze_particle_buffers`, `pix_capture`, `pix_list_captures`, `diagnose_visual_artifact`
- Advantage: No API key needed, uses Claude.ai Max subscription directly

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
# v4 RTXDI & Advanced RT Agents
/rtxdi-integration-specialist-v4
/dxr-rt-shadow-engineer-v4

# v3 Production Agents
/buffer-validator-v3
/stress-tester-v3
/performance-analyzer-v3

# PIX MCP Server (replaces pix-debugger-v3)
# These tools are automatically available in any Claude Code session:
# - mcp__pix-debug__capture_buffers
# - mcp__pix-debug__analyze_restir_reservoirs
# - mcp__pix-debug__analyze_particle_buffers
# - mcp__pix-debug__pix_capture
# - mcp__pix-debug__pix_list_captures
# - mcp__pix-debug__diagnose_visual_artifact

# v2 Specialized Agents
/rt-ml-technique-researcher-v2
/physics-performance-agent-v2
/dxr-systems-engineer-v2
```

### Multi-Agent Workflows

**Example 1: RTXDI Integration (Phase 4)**
```bash
# 1. Start RTXDI integration specialist
/rtxdi-integration-specialist-v4

# Agent guides through:
# - Week 1: SDK setup + light grid construction
# - Week 2: Reservoir buffers + RTXDI sampling
# - Week 3: Temporal/spatial reuse + performance validation
# - Week 4: Cleanup + documentation

# 2. Test shadow integration
/dxr-rt-shadow-engineer-v4

# 3. Validate performance
/performance-analyzer-v3
```

**Example 2: Debug ReSTIR Issues with MCP Tools**
```bash
# Just ask Claude naturally - MCP tools are automatically available:
You: "Capture buffers at frame 120 and analyze the ReSTIR reservoirs"

# Claude automatically uses:
# - mcp__pix-debug__capture_buffers
# - mcp__pix-debug__analyze_restir_reservoirs

You: "I'm seeing black dots at far distances"

# Claude automatically uses:
# - mcp__pix-debug__diagnose_visual_artifact
# - Provides HLSL fix with sqrt(M) visibility scaling
```

**Example 3: Research + Implement New Feature**
```bash
# 1. Research phase
/rt-ml-technique-researcher-v2

# 2. Implementation phase
/dxr-systems-engineer-v2

# 3. Validation phase
/performance-analyzer-v3
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
v4 RTXDI & Advanced RT (2)
├── RTXDI integration specialist
└── Advanced shadow systems

v3 Production Agents (4)
├── General-purpose debugging/testing
├── Buffer validation expertise
├── Multi-light system knowledge
└── PIX debugger (→ migrated to MCP server)

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

**Last Updated:** 2025-10-18
**Plugin Version:** 4.0.0
**Active Agents:** 12 (2 v4 + 4 v3 + 6 v2)
**MCP Servers:** 1 (pix-debug with 6 tools)
