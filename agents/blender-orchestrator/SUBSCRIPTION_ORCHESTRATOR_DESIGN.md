# Blender VFX Orchestrator - Subscription-Based Design

**Date:** 2026-01-02
**Status:** RECOMMENDATION
**Goal:** Autonomous VFX asset generation using Claude Code subscription (not API keys)

---

## Executive Summary

**The Problem:** The current `blender-orchestrator` is built on the Claude Agent SDK, which **requires API keys** and cannot use subscription auth. This means you pay twice: Max subscription ($100-200/month) + API usage fees.

**The Solution:** Replace the Claude Agent SDK orchestrator with **Claude Code native capabilities**:
1. **Skills** - Define orchestrator behavior in `.claude/skills/`
2. **Task Tool** - Spawn subagents for parallel work
3. **MCP Servers** - Keep all 16 specialized tools (they work with both approaches)
4. **Slash Commands** - User-invocable entry points

This approach runs entirely on your Max subscription with no API key costs.

---

## Research Findings

### Claude Agent SDK Limitation (Confirmed)

From [GitHub Issue #5891](https://github.com/anthropics/claude-code/issues/5891):
> "The Claude Agent SDK is intended to be used with an API key. Claude Code can be used with an API key or with a subscription."

From [Agent SDK Documentation](https://platform.claude.com/docs/en/api/agent-sdk/overview):
> "Unless previously approved, we do not allow third party developers to offer Claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK."

**Conclusion:** There is no workaround. Agent SDK = API keys = additional cost.

### Claude Code Native Capabilities

What Claude Code provides natively (works with subscription):

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Skills** | Markdown instruction files in `.claude/skills/` | Define orchestrator behavior |
| **Task Tool** | Spawn subagents with specific prompts | Parallel evaluation, research |
| **MCP Servers** | External tool servers (already built) | All 16 agents work unchanged |
| **Slash Commands** | `/skill-name` invocation | Entry points for workflows |
| **Sessions** | Persistent conversation with resume | Long-running asset generation |
| **Hooks** | Pre/post tool execution | Guardrails, logging |

---

## Current Architecture vs Proposed Architecture

### Current (Agent SDK - Requires API Key)

```
┌─────────────────────────────────────────────────────────────┐
│                 test_explosion.py                            │
│                 (Python test script)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              BlenderOrchestratorAgent                        │
│              (orchestrator.py - 1400+ lines)                 │
│                                                              │
│  • Claude Agent SDK (ClaudeSDKClient)  ← REQUIRES API KEY   │
│  • WorkflowStateMachine                                      │
│  • AutonomyController                                        │
│  • TokenGuardrails                                           │
│  • SessionManager                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Servers (16)                          │
│  script-generator, blender-executor, asset-evaluator, ...   │
└─────────────────────────────────────────────────────────────┘
```

### Proposed (Claude Code Native - Uses Subscription)

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                           │
│                (Max Subscription Auth)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              /blender-orchestrator Skill                     │
│        (.claude/skills/blender-orchestrator/SKILL.md)        │
│                                                              │
│  • Workflow stages defined in markdown                       │
│  • Quality thresholds documented                             │
│  • Spawns Task subagents for parallel work                  │
│  • Calls MCP tools directly                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
┌───────────────────────┐   ┌───────────────────────┐
│    Task Subagent      │   │    Task Subagent      │
│  (Blender Execution)  │   │   (ML Evaluation)     │
└───────────────────────┘   └───────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Servers (16)                          │
│  script-generator, blender-executor, asset-evaluator, ...   │
│                                                              │
│  ✅ UNCHANGED - Same servers work with both approaches      │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Enhance Existing Skill (Low Effort, High Value)

The skill at `.claude/skills/blender-orchestrator/SKILL.md` already exists but isn't optimized for autonomous operation. Enhance it:

**Current SKILL.md gaps:**
- No explicit workflow stages
- No Task subagent spawning instructions
- No error recovery patterns
- Missing iteration loop logic

**Enhanced SKILL.md additions:**

```markdown
## Autonomous Workflow

When invoked, follow this workflow:

### Stage 1: Generate Script
1. Call `mcp__script-generator__list_techniques` to see available approaches
2. Call `mcp__script-generator__generate_script` with user's description
3. If generation fails, call `mcp__script-generator__modify_script` with fixes

### Stage 2: Execute Blender
1. Call `mcp__blender-executor__execute_blender_script` with the script path
2. If execution fails, call `mcp__blender-executor__parse_blender_errors`
3. Apply fixes and retry (max 3 attempts)

### Stage 3: Evaluate Quality
1. Call `mcp__asset-evaluator__evaluate_vfx_quality` on rendered output
2. If score < 60, identify issues from diagnostics
3. Call `mcp__iteration-controller__diagnose_vfx_issues`

### Stage 4: Iterate
1. If quality passed, report success
2. If quality failed AND iterations < 5:
   - Call `mcp__script-generator__modify_script` with diagnostic feedback
   - Return to Stage 2
3. If max iterations reached, report best result

### Parallel Evaluation (Use Task Tool)
When evaluating quality, spawn parallel subagents:
- One for VFX quality scoring
- One for temporal consistency
- One for ground truth comparison (if reference provided)
```

### Phase 2: Create Orchestrator Subagent Type (Medium Effort)

Add a custom subagent definition that Claude Code can spawn:

**File:** `.claude/agents.toml` (or similar config)

```toml
[blender-orchestrator]
description = "Autonomous VFX asset generator using Blender + ML evaluation"
system_prompt = """
You are an autonomous VFX asset orchestrator. Your goal is to create high-quality
NanoVDB volumetric assets by:
1. Generating Blender scripts from descriptions
2. Executing simulations
3. Evaluating quality with ML
4. Iterating until thresholds are met

You have access to these MCP tools:
- mcp__script-generator__* (script creation/modification)
- mcp__blender-executor__* (Blender execution)
- mcp__asset-evaluator__* (ML quality evaluation)
- mcp__iteration-controller__* (iteration logic)
- mcp__experiment-tracker__* (learning from attempts)
- mcp__blender-manual__* (API documentation)

Quality Thresholds:
- VFX Score: >= 60/100
- Temporal Consistency: >= 0.7
- Ground Truth (if reference): >= 0.65

You may iterate up to 5 times. Always explain your reasoning.
"""
allowed_tools = [
    "mcp__script-generator__*",
    "mcp__blender-executor__*",
    "mcp__asset-evaluator__*",
    "mcp__iteration-controller__*",
    "mcp__experiment-tracker__*",
    "mcp__blender-manual__*"
]
```

### Phase 3: Slash Command Entry Point

Create a user-invocable command:

**File:** `.claude/commands/create-vfx-asset.md`

```markdown
---
name: create-vfx-asset
description: Create a VFX asset using the Blender orchestrator
---

# VFX Asset Creation

Create a volumetric VFX asset using Blender + ML evaluation.

## Arguments
- `$1`: Asset name (e.g., "mushroom_explosion_v1")
- `$2`: Effect type (pyro, explosion, fire, smoke, nebula, sun)
- `$3`: Description of desired effect

## Workflow
1. Generate Blender script from description
2. Execute simulation
3. Evaluate with ML (VFX quality, temporal consistency)
4. Iterate until quality >= 60 or max 5 iterations

## Usage
```
/create-vfx-asset mushroom_v1 pyro "Rising mushroom cloud with orange fire"
```
```

---

## What Can Be Reused

### From Current Implementation (Keep)

| Component | Location | Reuse Strategy |
|-----------|----------|----------------|
| MCP Servers (all 16) | `agents/*/` | 100% reusable - unchanged |
| Quality thresholds | `SKILL.md` | Already documented |
| Workflow stages | `orchestrator.py` | Extract logic to SKILL.md |
| Autonomy levels | `autonomy.py` | Describe in SKILL.md prose |
| Guardrails | `guardrails.py` | Use Claude Code's built-in limits |

### From Current Implementation (Replace)

| Component | Current | Replacement |
|-----------|---------|-------------|
| `ClaudeSDKClient` | Agent SDK | Claude Code native |
| `_query_agent()` | SDK inner agent | Task tool subagents |
| `SessionManager` | Custom JSON | Claude Code sessions |
| `WorkflowStateMachine` | Python class | SKILL.md instructions |
| `create_options()` | SDK config | Not needed |

---

## Cost Comparison

### Current (Agent SDK)
- Max subscription: ~$100-200/month (shared with claude.ai)
- API usage: ~$5-20/session (based on troubleshooting report)
- **Total:** $100-200 fixed + $5-20 per heavy session

### Proposed (Claude Code Native)
- Max subscription: ~$100-200/month (same)
- API usage: $0 (runs on subscription)
- **Total:** $100-200 fixed only

**Savings:** $5-20+ per session (eliminates API costs entirely)

---

## Migration Path

### Option A: Gradual Migration (Recommended)

1. **Week 1:** Enhance SKILL.md with workflow instructions
2. **Week 2:** Test skill invocation manually in Claude Code
3. **Week 3:** Add slash command entry point
4. **Week 4:** Document and refine based on real usage

**Pros:** Low risk, can compare both approaches, no big bang

### Option B: Full Replacement

1. Archive `orchestrator.py` and related files
2. Create new SKILL.md with complete workflow
3. Test end-to-end
4. Remove Agent SDK dependencies

**Pros:** Cleaner result, no legacy code
**Cons:** Higher risk, more upfront work

---

## Limitations of Native Approach

### What You Lose

1. **Programmatic Control** - No Python API for external integration
2. **Precise Token Counting** - Claude Code doesn't expose token metrics
3. **Custom Hooks** - Limited to Claude Code's built-in hooks
4. **Headless Execution** - Requires Claude Code CLI session

### What You Gain

1. **No API Costs** - Runs entirely on subscription
2. **Simpler Architecture** - No SDK complexity
3. **Native Integration** - Works with all Claude Code features
4. **Immediate Availability** - No server startup/config issues

---

## Recommendation

**Start with Phase 1 (Enhanced SKILL.md)** - This is the lowest-effort, highest-value change:

1. Update `.claude/skills/blender-orchestrator/SKILL.md` with explicit workflow stages
2. Test by invoking the skill: "Use the blender-orchestrator skill to create a mushroom explosion"
3. Iterate on the instructions based on what works/fails

This lets you validate the approach before investing in slash commands or custom subagent types.

---

## Next Steps

1. [ ] Review this document for alignment with your goals
2. [ ] Decide on migration path (gradual vs full replacement)
3. [ ] Update SKILL.md with enhanced workflow instructions
4. [ ] Test invocation from Claude Code CLI
5. [ ] If successful, deprecate `orchestrator.py`

---

## Appendix: MCP Tools Reference

All 16 MCP servers continue to work unchanged. Quick reference:

### Core Pipeline
- `mcp__script-generator__generate_script` - Create Blender scripts
- `mcp__script-generator__modify_script` - Adjust based on feedback
- `mcp__blender-executor__execute_blender_script` - Run simulations
- `mcp__asset-evaluator__evaluate_vfx_quality` - ML quality scoring

### Quality Analysis
- `mcp__asset-evaluator__evaluate_ground_truth` - Compare to reference
- `mcp__asset-evaluator__analyze_temporal_quality` - Animation consistency
- `mcp__asset-evaluator__extract_vfx_diagnostics` - Detailed analysis

### Learning & Iteration
- `mcp__iteration-controller__diagnose_vfx_issues` - Identify problems
- `mcp__experiment-tracker__record_experiment_result` - Track attempts
- `mcp__experiment-tracker__suggest_experiments` - Get fix ideas

### Documentation
- `mcp__blender-manual__search_python_api` - API reference
- `mcp__blender-manual__search_tutorials` - How-to guides

---

**End of Document**
