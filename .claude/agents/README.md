# PlasmaDX-Clean Agent System

**Last Updated**: 2025-11-19
**Claude Code Version**: 2.x
**Agent Count**: 4 Councils + 3 Phase 5 agents + 5 global debugging agents

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

### **rendering-council** ⏳ PLANNED (Agent SDK)
**Description**: Visual quality and rendering pipeline decisions
**Use when**: RTXDI issues, shadow quality, volumetric artifacts

### **physics-council** ⏳ PLANNED
**Description**: PINN integration and GPU physics decisions
**Use when**: ML physics, accretion disk dynamics, performance optimization

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

## Global Debugging Agents (Available from ~/.claude/agents/)

These agents complement Phase 5 development with debugging and performance tools:

- **buffer-validator-v3** - GPU buffer validation from PIX dumps
- **pix-debugger-v3** - PIX capture root cause analysis
- **performance-analyzer-v3** - Performance profiling and bottleneck identification
- **stress-tester-v3** - Scalability testing (particle count, light count, camera distance)
- **physics-performance-agent-v2** - Physics compute shader optimization

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

3. **Validate** (buffer-validator-v3):
   - Dump particle buffer with `--dump-buffers`
   - Verify 48-byte structure
   - Check new fields (particleType, opacity, density, lifetime)

4. **Test** (stress-tester-v3):
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
```

**Implicit invocation** (Claude Code auto-selects based on task):
```
User: "Implement the 5 particle types for Phase 5"
→ Claude Code launches celestial-rendering-specialist
```

---

## Future Agents (TODO)

- **integration-tester** - End-to-end testing of Phase 5 components
- **animation-scenario-builder** - Creates animation presets and sequences
- **material-property-designer** - Interactive material property tuning
- **ux-improvement-specialist** - Milestone 5.3 UX streamlining

