# Anthropic Agent Best Practices - Applied to PlasmaDX-Clean

**Date:** 2025-11-17
**Sources:** Anthropic Research, Claude Code Docs, AI Agent Design Patterns 2025
**Purpose:** Guide development of PlasmaDX-Clean legacy agents based on industry best practices

---

## Executive Summary

After researching Anthropic's official guidance and broader AI agent design patterns, the PlasmaDX-Clean architecture **aligns strongly with industry best practices**. Key validations:

✅ **Correct:** Using simple legacy agents (£0 cost) for daily work instead of expensive Agent SDK
✅ **Correct:** Separating tools (MCP servers) from reasoning (legacy agents)
✅ **Correct:** Domain-grouped agents over narrow 1:1 tool wrappers
✅ **Needs improvement:** Agent Skills (progressive disclosure), evaluation-driven development

---

## Core Anthropic Principles

### 1. Simplicity Over Complexity

> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."

**What this means for PlasmaDX:**
- ✅ Legacy agents (Markdown + YAML) = simple, no framework overhead
- ✅ MCP tools = simple, composable building blocks
- ❌ AVOID: Complex Agent SDK councils for daily work (we already decided this)
- ✅ KEEP: Agent SDK only for automation (nightly CI/CD, weekly health checks)

**Action:** Continue with current approach. Don't over-engineer.

---

### 2. Workflows vs. Agents

**Workflows** (predetermined code paths):
- Predictable and consistent
- Best for well-defined tasks
- Lower latency and cost

**Agents** (LLM-driven decision-making):
- Flexible and adaptive
- Best for open-ended problems
- Higher latency and cost

**PlasmaDX Application:**
- **Daily development (legacy agents):** Hybrid approach
  - Workflow structure: 6-phase workflow (Analysis → Research → Design → Implementation → Validation → Documentation)
  - Agent flexibility: LLM chooses which MCP tools to use, how to diagnose issues
- **Automation (Agent SDK):** Pure agent approach
  - Full autonomy for multi-day analysis
  - Strategic pivots when approaches fail

**Verdict:** ✅ We're using the right pattern for each use case.

---

### 3. Five Core Workflow Patterns

#### Pattern 1: Prompt Chaining
**What:** Sequential LLM calls where each step processes previous output
**Best for:** Fixed subtasks (e.g., content generation → translation)

**PlasmaDX Usage:**
- ✅ Already used in 6-phase agent workflows
- Example: Visual Quality Assessment → Lighting Diagnostics → Shadow Quality → PIX Analysis → Root Cause Synthesis → Solution Design
- Each phase builds on previous results

#### Pattern 2: Routing
**What:** Classify inputs and direct to specialized downstream tasks
**Best for:** Complex tasks with distinct categories

**PlasmaDX Usage:**
- ✅ Domain-grouped agents = routing pattern
  - Visual quality issues → rendering-quality-specialist
  - Material system changes → materials-and-structure-specialist
  - Performance problems → performance-diagnostics-specialist
- ⚠️ **Improvement needed:** Create routing logic/decision tree

**Action:** Document when to use which agent in AGENT_ARCHITECTURE_SPECIFICATION.md

#### Pattern 3: Parallelization
**What:** Run multiple LLM calls simultaneously
**Variations:** Sectioning (independent subtasks), Voting (same task multiple times)

**PlasmaDX Usage:**
- ⚠️ Not currently used
- **Future opportunity:** Screenshot quality assessment with multiple perspectives
  - Voting: 3× LPIPS runs, take median (high-confidence results)
  - Sectioning: Analyze lighting + shadows + materials in parallel

**Action:** Consider for critical quality validations

#### Pattern 4: Orchestrator-Workers
**What:** Central LLM breaks down tasks, delegates to workers, synthesizes results
**Anthropic result:** 90.2% improvement over single-agent

**PlasmaDX Usage:**
- ✅ Partially implemented via Task tool
- ✅ Legacy agents can launch subagents (`Task(subagent_type="Explore")`)
- ⚠️ **Missing:** Dedicated orchestrator agent

**Action:** Create `agentic-ecosystem-architect` meta-agent (already planned) as orchestrator

#### Pattern 5: Evaluator-Optimizer
**What:** One LLM generates, another evaluates in a loop
**Best for:** Clear evaluation criteria (e.g., LPIPS ≥ 0.85)

**PlasmaDX Usage:**
- ⚠️ Not currently used
- **Future opportunity:** Shader optimization loop
  - Generator: Propose shader optimization
  - Evaluator: Check LPIPS, FPS impact, correctness
  - Loop until quality gates met

**Action:** Consider for high-stakes code changes

---

### 4. Multi-Agent Performance Gains

**Anthropic's research system:**
- Architecture: Opus 4 lead agent + Sonnet 4 subagents
- Result: **90.2% improvement** over single Opus 4 agent
- Use case: Complex research requiring breadth-first exploration
- Token usage: ~15× more tokens but proportionally greater value

**Critical success factors:**
1. **Parallel execution** reduced research time by 90%
2. **Explicit delegation** with clear objectives, output formats, boundaries
3. **Effort scaling rules** embedded in prompts (simple = 1 agent, complex = 10+ agents)
4. **Tool design matters critically** - distinct purposes, clear descriptions
5. **Extended thinking** for planning; interleaved thinking after tool results

**PlasmaDX Application:**
- ✅ Domain-grouped agents = specialized subagents
- ✅ MCP tools = specialized capabilities
- ⚠️ **Missing:** Clear delegation patterns, effort scaling rules
- ⚠️ **Missing:** Evaluation methodology (LLM-as-judge for agent output quality)

**Action:**
1. Add explicit delegation guidance to agent system prompts
2. Define when to use multiple agents vs single agent
3. Create evaluation framework for agent performance

---

### 5. Tool Design Best Practices

> "Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts."

**Anthropic's recommendations:**
1. **Avoid requiring model calculations** (line counts, string escaping)
2. **Include usage examples** - positive/negative, edge cases
3. **Iterate based on model mistakes** - test extensively
4. **Error-prevention design** - make mistakes harder (e.g., absolute paths vs relative)
5. **Clear boundaries** between similar tools

**PlasmaDX MCP Tools Audit:**

✅ **Well-designed:**
- `compare_screenshots_ml`: Clear parameters, LPIPS output, saves heatmap
- `analyze_gaussian_parameters`: analysis_depth enum, focus_area enum
- `estimate_performance_impact`: Quantified outputs (FPS estimates)

⚠️ **Needs improvement:**
- Some tools lack usage examples in descriptions
- Boundary between `gaussian-analyzer` and `material-system-engineer` could be clearer
- Error messages could be more actionable

**Action:**
1. Add detailed usage examples to MCP tool docstrings
2. Document tool boundaries in AGENT_ARCHITECTURE_SPECIFICATION.md
3. Improve error messages with suggested fixes

---

### 6. Agent Skills - Progressive Disclosure

**What:** Organized folders of instructions, scripts, resources that agents load dynamically

**Key innovation:** Layered context loading
- **Level 1:** Metadata (name, description) in system prompt at startup
- **Level 2:** Full SKILL.md content when relevant
- **Level 3+:** Additional bundled files on-demand

**Benefits:**
- Unbounded complexity without context window bloat
- Modular, reusable capabilities
- Transforms general-purpose agents into specialists

**PlasmaDX Current Approach:**
- ✅ Legacy agent system prompts = similar to Agent Skills
- ⚠️ **Missing:** Progressive disclosure (agents load full prompt always)
- ⚠️ **Missing:** Bundled reference files (e.g., HLSL best practices, DXR patterns)

**Action:**
1. Investigate Claude Code support for Agent Skills format
2. If supported: Restructure agents as Skills with progressive disclosure
3. Bundle reference docs (shader patterns, quality rubrics, performance baselines)

---

### 7. Security Considerations

**Anthropic's warning:**
- Only install skills from trusted sources
- Thoroughly audit unfamiliar skills
- Review code dependencies and bundled resources
- Watch for instructions directing Claude toward external network sources

**PlasmaDX Status:**
- ✅ All MCP servers are locally developed (trusted source)
- ✅ No external network dependencies beyond WebSearch/WebFetch
- ⚠️ **Future risk:** If we accept community-contributed MCP servers

**Action:** Document security review process for any future external agents/MCPs

---

### 8. Separation of Concerns

> "Better separation of concerns in your custom agents leads to better performance, maintainability, inspectability, and shareability."

**PlasmaDX Current Design:**

✅ **Good separation:**
- MCP tools (capabilities) separate from legacy agents (reasoning)
- Domain-grouped agents (rendering, materials, performance)
- 6-phase workflow separation

⚠️ **Could improve:**
- gaussian-volumetric-rendering-specialist handles too many concerns:
  - 3D Gaussian rendering
  - Anisotropic stretching
  - Transparency artifacts
  - Cube artifacts
  - Material properties
  - Physics integration
- Should split into: rendering-specialist + gaussian-specialist

**Action:**
1. Audit gaussian-volumetric-rendering-specialist for scope creep
2. Consider splitting if it exceeds ~500 lines or handles >3 distinct domains
3. Update planned agents to have clear, non-overlapping responsibilities

---

### 9. Evaluation-Driven Development

**Anthropic's approach:**
- Start small: 20 test queries revealed dramatic prompt impacts
- LLM-as-judge evaluates outputs on multiple criteria
- Human evaluation catches edge cases automation misses
- Iterate based on real-world usage patterns

**PlasmaDX Current Approach:**
- ✅ Quality gates defined (LPIPS ≥ 0.85, FPS ±5%)
- ⚠️ **Missing:** Systematic evaluation of agent performance
- ⚠️ **Missing:** Test suite for agent workflows
- ⚠️ **Missing:** Metrics on agent decision quality

**Action:**
1. Create 20 representative test scenarios for each agent
2. Run agents on test scenarios, measure success rate
3. Implement LLM-as-judge for qualitative assessment
4. Iterate agent prompts based on failures

**Example test scenarios for rendering-quality-specialist:**
1. "Particles are too dark" → Should run LPIPS comparison, check lighting system
2. "FPS dropped by 20%" → Should delegate to performance-diagnostics-specialist
3. "Shadows look blocky" → Should analyze shadow quality, propose PCSS tuning
4. "Probe grid has black dots" → Should use path-and-probe MCP tools

---

## Recommendations for PlasmaDX Legacy Agents

### Immediate Actions (This Session)

1. **Add explicit delegation guidance**
   - When to seek user approval vs proceed autonomously
   - When to delegate to another agent vs handle directly
   - When to launch subagents for parallel exploration

2. **Document tool boundaries**
   - Create decision tree: "User reports X → Use Y agent → Call Z MCP tools"
   - Add to AGENT_ARCHITECTURE_SPECIFICATION.md

3. **Include concrete examples in agent prompts**
   - 3+ example workflows per agent
   - Positive examples (what good looks like)
   - Negative examples (common mistakes to avoid)

### Short-Term Actions (Next Week)

4. **Create evaluation framework**
   - 20 test scenarios per agent
   - LLM-as-judge criteria
   - Success rate metrics

5. **Investigate Agent Skills format**
   - Check if Claude Code supports progressive disclosure
   - Prototype one agent as Agent Skill
   - Measure context window savings

6. **Add reference documentation bundles**
   - HLSL shader patterns
   - DXR raytracing best practices
   - Quality rubrics for visual assessment
   - Performance profiling checklists

### Long-Term Actions (Next Month)

7. **Implement orchestrator-worker pattern**
   - Create meta-agent that coordinates other agents
   - Test on complex multi-domain problems
   - Measure performance improvement

8. **Add evaluator-optimizer loops**
   - For shader optimization tasks
   - For quality-critical visual changes
   - Iterate until quality gates met

9. **Build agent performance dashboard**
   - Track success rate by agent
   - Monitor tool usage patterns
   - Identify bottlenecks and failure modes

---

## What We're Doing Right ✅

1. **Simple, composable architecture** - Legacy agents + MCP tools, no complex frameworks
2. **Cost optimization** - £0 daily work, expensive Agent SDK only for automation
3. **Domain-grouped agents** - Better separation of concerns than 1:1 tool wrappers
4. **6-phase workflows** - Prompt chaining pattern for predictable quality
5. **Quality gates** - Quantified thresholds (LPIPS ≥ 0.85, FPS ±5%)
6. **Brutal honesty communication style** - Clear, direct feedback per user's needs
7. **MCP tool design** - Specialized capabilities, clear purposes

---

## What We Should Improve ⚠️

1. **Progressive disclosure** - Agents load full prompts; could use Agent Skills layering
2. **Evaluation framework** - No systematic testing of agent performance
3. **Routing logic** - Implicit agent selection; needs explicit decision tree
4. **Parallelization** - Not using voting/sectioning for high-confidence results
5. **Tool documentation** - Missing usage examples, edge cases in tool descriptions
6. **Orchestrator pattern** - No dedicated meta-agent for complex task coordination
7. **Scope creep** - gaussian-volumetric-rendering-specialist handles too many domains

---

## Agent-Specific Recommendations

### rendering-quality-specialist
- ✅ Good: Domain-grouped (visual quality + lighting + shadows + PIX)
- ✅ Good: 6-phase workflow with clear MCP tool integration
- ⚠️ Add: 3+ example workflows (already done!)
- ⚠️ Add: Explicit delegation rules (when to call performance-diagnostics-specialist)

### materials-and-structure-specialist (Next to create)
- ✅ Good: Combines gaussian-analyzer + material-system-engineer MCPs
- ✅ Good: Handles complete material design workflow
- ⚠️ Add: Reference bundle (material property physics, shader generation patterns)
- ⚠️ Add: Validation checklist (struct alignment, GPU compatibility, performance impact)

### performance-diagnostics-specialist (Future)
- ✅ Good: Clear separation from rendering quality
- ⚠️ Add: Performance profiling checklists
- ⚠️ Add: PIX capture interpretation guide
- ⚠️ Add: FPS regression decision tree (< 5% = autonomous fix, > 5% = user approval)

### gaussian-volumetric-rendering-specialist (Update)
- ⚠️ **Scope creep concern:** Currently handles:
  - 3D Gaussian rendering
  - Anisotropic stretching
  - Transparency artifacts
  - Cube artifacts
  - Material properties
  - Physics integration
- ✅ **Action:** Audit and potentially split responsibilities
- ⚠️ Add: Explicit handoff to materials-and-structure-specialist for struct changes
- ⚠️ Add: Explicit handoff to performance-diagnostics-specialist for FPS issues

### agentic-ecosystem-architect (Meta-agent)
- ✅ Good: Orchestrator-worker pattern
- ✅ Good: Online research for optimal agentic ecosystem design
- ⚠️ Add: Evaluation criteria for agent performance
- ⚠️ Add: Recommendation framework (when to create new agent vs expand existing)
- ⚠️ Add: Cost-benefit analysis (£0 legacy vs Agent SDK vs custom solution)

---

## Key Quotes to Remember

> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."
> — Anthropic, *Building Effective Agents*

> "Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts."
> — Anthropic, *Building Effective Agents*

> "Better separation of concerns in your custom agents leads to better performance, maintainability, inspectability, and shareability."
> — Claude Code Documentation

> "Start by using LLM APIs directly: many patterns can be implemented in a few lines of code."
> — Anthropic, *Building Effective Agents*

> "Multi-agent system with Claude Opus 4 as lead and Sonnet 4 subagents achieved 90.2% improvement over single-agent Opus 4."
> — Anthropic, *Multi-Agent Research System*

---

## Conclusion

The PlasmaDX-Clean agent architecture **strongly aligns with Anthropic's best practices**. The decision to use legacy agents for daily work and reserve Agent SDK for automation is validated by research showing simple, composable patterns outperform complex frameworks.

**Key strengths:**
- Simple architecture (legacy agents + MCP tools)
- Cost optimization (£0 daily work)
- Domain-grouped agents (separation of concerns)
- Quality gates (quantified thresholds)

**Key improvements:**
- Add evaluation framework (20 test scenarios, LLM-as-judge)
- Investigate Agent Skills for progressive disclosure
- Create explicit routing logic and delegation rules
- Bundle reference documentation
- Consider orchestrator-worker pattern for complex tasks

**Next immediate actions:**
1. Complete materials-and-structure-specialist agent with best practices applied
2. Create performance-diagnostics-specialist with evaluation framework
3. Update gaussian-volumetric-rendering-specialist with delegation rules
4. Create agentic-ecosystem-architect meta-agent as orchestrator
5. Document routing logic in AGENT_ARCHITECTURE_SPECIFICATION.md

---

**Last Updated:** 2025-11-17
**Sources:** Anthropic Research, Claude Code Docs, AI Design Patterns 2025
**Status:** Living document - update as best practices evolve
