# Mission-Control: Migration from MCP Server to Agent Skill

**Date**: 2025-11-16
**Status**: ✅ COMPLETE
**Migration Type**: MCP Server → Agent Skill

---

## Problem: MCP Server Connection Timeout

### Root Cause
The original `server.py` implementation had a fundamental architectural mismatch:

1. **Incorrect Approach**: Used `create_sdk_mcp_server()` to create an external MCP server
2. **Expected Behavior**: Claude Code tried to connect via JSON-RPC protocol
3. **Actual Behavior**: `server.py` ran as standalone Agent SDK application with `ClaudeSDKClient`
4. **Result**: 30-second connection timeout → MCP server unavailable

### Error Log Evidence
```
Connection to MCP server "mission-control" timed out after 30000ms
Error: write EPIPE
```

**Key Insight**: The Agent SDK's `create_sdk_mcp_server()` creates **in-process custom tool servers** for use WITHIN Agent SDK apps, NOT external MCP servers for Claude Code integration.

---

## Solution: Agent Skill Architecture

### Why Agent Skills?
Agent Skills are the **correct** approach for integrating strategic orchestration into Claude Code because:

1. **Native Integration**: Auto-discovered by Claude Code from `.claude/skills/` directory
2. **Prompt-Based**: Package strategic instructions as specialized prompts
3. **No External Server**: Run directly within Claude Code, no subprocess management
4. **Modular**: Progressive disclosure (metadata → instructions → resources)
5. **Executable Tools**: Optional executable code via scripts/forms
6. **Better for Strategic Orchestration**: Prompt templates align with Mission-Control's decision-making role

### Agent Skill Structure
```
.claude/skills/mission-control/
├── SKILL.md                    # Main skill definition (YAML frontmatter + instructions)
├── DECISION_FRAMEWORK.md       # Optional: Decision-making templates (future)
└── QUALITY_GATES.md            # Optional: Quality gate checklists (future)
```

### SKILL.md Architecture
```markdown
---
name: mission-control
description: When to use this skill (must be specific!)
---

# Main Content
- Strategic instructions
- Decision-making framework
- Quality gates
- Communication style (brutal honesty)
- Examples with evidence-based recommendations
```

---

## Migration Steps Completed

### ✅ Step 1: Create Agent Skill Directory
```bash
mkdir -p .claude/skills/mission-control
```

### ✅ Step 2: Write SKILL.md with Strategic Prompts
Created comprehensive SKILL.md with:
- Strategic coordination instructions
- Decision recording framework
- Quality gate enforcement (LPIPS, FPS)
- Human oversight (supervised autonomy)
- MCP tool integration (all 4 councils + specialist agents)
- Communication style (brutal honesty per CLAUDE.md)
- Examples (visual quality regression, performance optimization, probe grid coverage)

### ✅ Step 3: Remove MCP Server Configuration
No external MCP config file needed to remove (configuration was in-code only).

### ✅ Step 4: Preserve Existing Tools (for future use)
Original tool implementations (`record.py`, `dispatch.py`, `handoff.py`, `status.py`) preserved in `agents/mission-control/tools/` for potential future use as:
- Executable scripts called from Agent Skill
- Python functions for standalone CLI mode
- Reference implementations

---

## How to Use the Agent Skill

### In Claude Code
The skill is **automatically discovered** when Claude Code starts. To invoke:

**Implicit Invocation** (Recommended):
Claude Code will automatically activate the skill when you request strategic orchestration tasks:
- "Analyze the RTXDI visual quality regression"
- "Should we use PINN physics for 100K particles?"
- "Diagnose the probe grid black dots at far distances"

**Explicit Invocation**:
You can also explicitly request the skill:
- "Use mission-control skill to coordinate RTXDI debugging"
- "Invoke mission-control to make architecture decision"

### What the Skill Provides
1. **Strategic Analysis**: Holistic problem analysis across domains
2. **Specialist Coordination**: Routes tasks to councils (rendering, materials, physics, diagnostics)
3. **Evidence-Based Recommendations**: Quantified FPS, LPIPS, buffer validation
4. **Quality Gate Enforcement**: Pre-deployment validation (build, shaders, performance, buffers)
5. **Approval Workflow**: Major decisions require explicit user approval
6. **Brutal Honesty**: Direct, specific, quantified communication (per CLAUDE.md)

---

## Comparison: MCP Server vs Agent Skill

| Aspect | MCP Server (Old) | Agent Skill (New) |
|--------|------------------|-------------------|
| **Integration** | External JSON-RPC | Native Claude Code |
| **Discovery** | Manual config | Auto-discovered |
| **Architecture** | Standalone subprocess | In-process prompts |
| **Latency** | Network overhead | Zero overhead |
| **Maintenance** | Server + client code | Single SKILL.md |
| **Use Case** | Function calls | Strategic instructions |
| **Best For** | Data processing tools | Decision-making, coordination |

---

## Future Enhancements

### Phase 1: Enhanced Agent Skill (Optional Resources)
- **DECISION_FRAMEWORK.md**: Decision templates (analysis → recommendation → approval)
- **QUALITY_GATES.md**: Checklists for pre-deployment validation
- **EXAMPLES/**: Reference PIX captures, LPIPS comparisons, FPS regressions

### Phase 2: Executable Tools (Optional)
Convert existing tools to executable scripts callable from the Agent Skill:
```bash
.claude/skills/mission-control/scripts/record_decision.py
.claude/skills/mission-control/scripts/publish_status.py
```

### Phase 3: Multi-Skill Orchestration
Create additional skills for specialized workflows:
- `rtxdi-workflow`: RTXDI temporal accumulation debugging
- `pinn-integration`: PINN ML physics deployment workflow
- `visual-quality-gate`: LPIPS validation and regression detection

---

## Technical Details

### Agent Skill Discovery
Claude Code automatically scans `.claude/skills/` on startup. Skills must:
1. Have `SKILL.md` with valid YAML frontmatter
2. Include `name` and `description` fields
3. Description must specify **what** the skill does AND **when** to use it

### Skill Activation
Skills activate based on:
- **Explicit user request**: "Use mission-control skill..."
- **Implicit context matching**: Claude Code matches description to user intent
- **Allowed tools list**: Must include `"Skill"` in enabled tools (already configured)

### Progressive Disclosure
Agent Skills load content in 3 levels:
1. **Level 1 (Always)**: YAML metadata (~100 tokens)
2. **Level 2 (When activated)**: Main SKILL.md body (<5K tokens)
3. **Level 3 (On-demand)**: Additional resource files (scripts, templates, forms)

This keeps context usage efficient while providing deep expertise when needed.

---

## References

- **Agent Skills Documentation**: https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- **Claude Agent SDK**: https://github.com/anthropics/claude-agent-sdk-python
- **Original MCP Server Logs**: `mcp-logs-mission-control/2025-11-16T14-44-57-668Z.txt`
- **SKILL.md Location**: `.claude/skills/mission-control/SKILL.md`

---

**Migration Status**: ✅ COMPLETE
**Next Steps**: Test Agent Skill invocation in Claude Code
**Rollback Plan**: If Agent Skill fails, revert to standalone CLI mode: `python agents/mission-control/server.py`
