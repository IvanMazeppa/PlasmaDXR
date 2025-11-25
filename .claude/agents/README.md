# PlasmaDX-Clean Agent System

**Last Updated**: 2025-11-24
**Claude Code Version**: 2.x
**Agent Count**: 2 Councils + 3 Phase 5 agents + 7 Domain Specialists

---

## Council Architecture (Strategic Layer)

Councils are **strategic decision-makers** that coordinate specialists and enforce quality gates.

### **materials-council** ✅ ACTIVE
**Description**: Strategic orchestrator for material system decisions
**Use when**: Adding material types, modifying particle structure, setting material properties
**Capabilities**:
- Architectural decisions for material type systems
- Performance budget enforcement (<5% FPS regression)
- GPU alignment validation (16-byte requirement)
- Dispatches to `materials-and-structure-specialist` for implementation

### **physics-council** ✅ ACTIVE
**Description**: PINN integration and GPU physics decisions
**Use when**: ML physics, accretion disk dynamics, performance optimization
**Capabilities**:
- Physics model architecture decisions
- Conservation law enforcement
- Dispatches to `physics-animation-engineer` for implementation

### **rendering-council** ⏳ PLANNED
**Description**: Visual quality and rendering pipeline decisions
**Use when**: RTXDI issues, shadow quality, volumetric artifacts

### **diagnostics-council** ⏳ PLANNED
**Description**: PIX debugging and performance profiling coordination
**Use when**: GPU crashes, TDR hangs, bottleneck analysis

---

## Phase 5 Development Agents (Project-Specific)

### **phase-5-orchestrator**
**Description**: Coordinates Phase 5 Celestial Rendering implementation
**Use when**: Starting any Phase 5 milestone or needing multi-agent coordination
**Capabilities**:
- Launches specialized agents in optimal order
- Tracks progress across all 3 milestones
- Ensures integration between components

### **celestial-rendering-specialist**
**Description**: Particle type system implementation specialist
**Use when**: Implementing 5 material types with distinct visual properties
**Capabilities**:
- Expands particle structure (32→48 bytes)
- Creates material property constant buffers
- Integrates per-type rendering in Gaussian shader

### **physics-animation-engineer**
**Description**: Physics parameter implementation for animation controls
**Use when**: Implementing constraint shapes, alpha viscosity, temperature dynamics
**Capabilities**:
- Implements 4 constraint shapes (SPHERE, DISC, TORUS, ACCRETION_DISK)
- Adds Keplerian dynamics (black hole mass parameter)
- Implements Shakura-Sunyaev alpha viscosity
- Adds shear heating and radiative cooling

---

## Domain Specialist Agents

These agents handle domain-specific implementation work:

- **gaussian-volumetric-rendering-specialist** - 3D Gaussian rendering bugs, anisotropic stretching, cube artifacts
- **materials-and-structure-specialist** - Material system design, particle structure, GPU alignment
- **rendering-quality-specialist** - LPIPS validation, shadow quality, probe grid diagnostics
- **performance-diagnostics-specialist** - PIX captures, GPU hangs, FPS optimization
- **agentic-ecosystem-architect** - Meta-agent orchestrator, coordinates multi-agent workflows
- **pix-debugging-agent** - PIX capture analysis, buffer validation
- **rtxdi-integration-specialist-v4** - RTXDI lighting integration

---

## Multi-Agent Workflow Example

**Scenario**: Implement Particle Type System (Milestone 5.1)

1. **Orchestrate** (phase-5-orchestrator):
   - Reads PARTICLE_TYPE_SYSTEM_SPEC.md
   - Breaks down into 3 tasks
   - Launches celestial-rendering-specialist for each task

2. **Implement** (celestial-rendering-specialist):
   - Task 1: Expand particle buffer (32→48 bytes)
   - Task 2: Create material property system
   - Task 3: Integrate per-type rendering

3. **Validate** (pix-debugging-agent):
   - Dump particle buffer with `--dump-buffers`
   - Verify 48-byte structure
   - Check new fields (particleType, opacity, density, lifetime)

4. **Test** (performance-diagnostics-specialist):
   - Run with different type distributions (100% stars, 100% gas, etc.)
   - Measure performance impact (<5% overhead target)
   - Visual regression test (PLASMA_BLOB matches baseline)

---

## Best Practices

✅ **Read specs first** - Agents are more effective when specs are loaded in context
✅ **One agent at a time** - Avoid parallel agents working on same files
✅ **Validate incrementally** - Test after each agent completes a task
✅ **Use orchestrator for milestones** - Let phase-5-orchestrator manage complex workflows
✅ **Document changes** - Agents should update relevant .md files

❌ **Don't skip testing** - Always build and verify after agent modifications
❌ **Don't mix concerns** - Use specialized agents for their specific domains
❌ **Don't ignore warnings** - Agent warnings about compatibility issues are critical

---

## Agent Invocation

**Explicit invocation**:
```
@phase-5-orchestrator "Begin Milestone 5.1"
@celestial-rendering-specialist "Implement particle structure expansion"
@physics-animation-engineer "Add constraint shape system"
@materials-council "Review material type proposal"
@gaussian-volumetric-rendering-specialist "Debug anisotropic stretching"
```

**Implicit invocation** (Claude Code auto-selects based on task):
```
User: "Implement the 5 particle types for Phase 5"
→ Claude Code launches celestial-rendering-specialist
```

---

## Future Agents (TODO)

- **rendering-council** - Strategic visual quality decisions
- **diagnostics-council** - PIX debugging and performance profiling coordination
- **integration-tester** - End-to-end testing of Phase 5 components
- **animation-scenario-builder** - Creates animation presets and sequences

---

## MCP Tools Reference

Agent files document MCP (Model Context Protocol) tools for specialized functionality. These tools are available via configured MCP servers:

| MCP Server | Tools | Domain |
|------------|-------|--------|
| `gaussian-analyzer` | 5 | Particle structure analysis, rendering technique comparison |
| `material-system-engineer` | 9 | Code generation, shader generation, struct validation |
| `dxr-image-quality-analyst` | 5 | Screenshot comparison, LPIPS validation, visual quality |
| `path-and-probe` | 6 | Probe grid analysis, SH coefficient validation |
| `pix-debug` | 8 | GPU debugging, buffer analysis, TDR diagnosis |
| `dxr-shadow-engineer` | 5 | Shadow technique research and shader generation |
| `dxr-volumetric-pyro-specialist` | 5 | Volumetric effects, explosions, fire/smoke |
| `dx12-enhanced` | 6 | D3D12/DXR API reference and HLSL intrinsics |

**Usage in agents**: Agent files document MCP tool calls using notation like `mcp__server-name__tool_name()`. MCP tools are automatically available when MCP servers are configured and don't need to be declared in the agent's `tools:` field.
