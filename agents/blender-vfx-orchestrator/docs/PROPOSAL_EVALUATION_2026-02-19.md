# Architecture Proposal Evaluation

**Date:** 2026-02-19
**Evaluator:** Claude Opus 4.6
**Purpose:** Score and compare architecture proposals against the Mission Statement, identify the best ideas from each, and determine what should feed into the S3 roadmap revision.

---

## Evaluation Criteria

Each proposal is scored 1-10 on 10 categories derived from the Mission Statement and practical constraints. Justification is required for every score.

| # | Category | Weight | What It Measures |
|---|----------|--------|------------------|
| 1 | **Mission Alignment** | High | Does it match the vision? Generality, adaptability, autonomy, honesty. |
| 2 | **Flexibility / Generality** | High | Can it handle novel prompts and new physics types (rigid body, particles, cloth)? |
| 3 | **Anti-Hallucination** | High | How well does it solve the API accuracy problem? Preventive, detective, or reactive? |
| 4 | **Self-Learning Design** | Medium | Is the learning system evidence-gated, practical, and connected to agent behavior? |
| 5 | **SDK Utilization** | Medium | How well does it leverage Agents SDK features (HITL, streaming, guardrails, sessions)? |
| 6 | **HITL Integration** | Medium | Quality of human-in-the-loop design. Progressive autonomy, intervention points. |
| 7 | **Practical Feasibility** | High | Can we build this given budget ($20/mo), team (Ben + AI), and current state? |
| 8 | **Code Specificity** | Medium | How concrete? Runnable examples vs. abstract descriptions? |
| 9 | **Budget Awareness** | Medium | Does it respect the $20/month constraint? Cost-conscious design? |
| 10 | **Observability** | Medium | Can we see what's happening, debug problems, track trends? |

**Scoring guide:**
- 1-3: Weak / missing / wrong
- 4-5: Present but underdeveloped
- 6-7: Solid, some gaps
- 8-9: Strong, well-thought-out
- 10: Exceptional, best-in-class idea

---

## Proposal A: "Architecture From Scratch"

**Source:** `ARCHITECTURE_FROM_SCRATCH_PROPOSAL.md`
**Author:** Claude Opus 4.6 (3-agent research team)
**Length:** ~1,285 lines
**Approach:** Clean-slate design centered on the "Introspection Sandbox" pattern — using a persistent live Blender process to validate every API attribute before execution.

### Scores

| # | Category | Score | Justification |
|---|----------|-------|---------------|
| 1 | Mission Alignment | **9** | Explicitly addresses generality (Explorer agent for novel physics), adaptability (Planner/Coder split for creative + technical separation), and earned autonomy. Directly references Mission Statement design principles. Templates positioned as accelerators, not the centerpiece. |
| 2 | Flexibility | **9** | Explorer agent with `run_diagnostic_script()`, `introspect_blender_type()`, and web search enables genuine research into unfamiliar physics systems. Phase 4 explicitly targets "brick wall collapsing" (rigid body, never seen before). Clear path from known effects to novel prompts. |
| 3 | Anti-Hallucination | **10** | The standout idea across all proposals. Three-layer system: (1) inject verified API context into prompt, (2) AST-parse + validate every attribute against live Blender `dir()`, (3) runtime error feedback loop. Persistent Blender process with caching. This is a whitelist approach — catches ALL invalid attributes, not just known ones. Fundamentally better than any regex/blocklist approach. |
| 4 | Self-Learning | **8** | Three tiers (session ephemeral, validated patterns, API ground truth). Evidence-gating with explicit promotion criteria (3+ observations, 70%+ success rate, 2+ unique prompts). Anti-learning rules prevent bad knowledge. Connected to agent behavior via dynamic instructions. Less detail on the observation → hypothesis → validated knowledge pipeline mechanics. |
| 5 | SDK Utilization | **6** | Uses agents-as-tools, dynamic instructions, structured output, guardrails, RunHooks correctly. But incorrectly states "The Agents SDK doesn't have built-in HITL" — it does (needs_approval + RunState). No mention of streaming. No mention of parallelism. No mention of Sessions API, call_model_input_filter, or ToolOutputTrimmer. |
| 6 | HITL | **5** | Acknowledges HITL importance. Progressive autonomy table (Guided → Assisted → Semi-Auto → Autonomous) is good. But implements with `input()` instead of SDK's actual HITL feature. Basic console prompt, not serializable/resumable state. |
| 7 | Feasibility | **7** | Well-phased build plan (Foundation → Loop → Learning → Exploration → Polish). But starting without templates is risky given 0/188 track record. 9+ week timeline may be optimistic. AST-based attribute validation has edge cases (dynamic attribute access, setattr, getattr) that aren't addressed. Introspection Sandbox itself is practical and buildable. |
| 8 | Code Specificity | **9** | Extensive code examples throughout: BlenderIntrospector class, validate_and_fix_script function, coder_dynamic_instructions, PipelineContext, QualityEvaluation, BudgetTracker, SessionState. Nearly every concept has runnable pseudocode. "What I'd explicitly NOT build" section is unusually honest. |
| 9 | Budget Awareness | **8** | Cost model per operation (Planner $0.05, Coder $0.08, Critic $0.01, Explorer $0.15). Budget-aware model routing (downgrade models when budget tight). Tiered evaluation (cheap deterministic checks early, ML only on later iterations). Per-session limits. |
| 10 | Observability | **6** | RunHooks for basic logging. Budget tracking. But no structured RunSummary equivalent. No mention of tracing beyond the hooks. No per-run decision record. Less systematic than Proposal B. |

**Total: 77/100**

### Key Strengths
1. **Introspection Sandbox** — the single best idea across all proposals
2. **Planner/Coder separation** — cleaner than plan-and-code-simultaneously
3. **Explorer Agent** — genuine adaptive research capability
4. **Code specificity** — nearly everything is shown with runnable examples
5. **Honest "what NOT to build" section** — prevents over-engineering

### Key Weaknesses
1. **Missed SDK HITL feature** — incorrectly says SDK lacks it
2. **No streaming or parallelism discussion** — both mentioned by Ben as important
3. **No templates at all** — risky given 0/188 track record with pure code gen
4. **Observability is underdeveloped** — needs structured run summaries
5. **AST validation edge cases** — dynamic attribute access patterns not addressed

### Unique Ideas Worth Taking
- Persistent Blender introspection process with caching
- 3-layer anti-hallucination (prevent → detect → learn)
- `# VERIFY:` comment pattern for uncertain attributes
- Tiered quality evaluation (cheap early, expensive late)
- Budget-aware model routing (downgrade when tight)
- Anti-learning rules (don't learn from single observations or interrupted runs)
- Diagnostic + rollback script generation (3 scripts per attempt)

---

## Proposal B: "Architecture Reboot Proposal"

**Source:** `ARCHITECTURE_REBOOT_PROPOSAL_2026-02-18.md`
**Author:** Codex (GPT-5)
**Length:** ~278 lines
**Approach:** Systems architecture proposal emphasizing control plane / execution plane separation, typed contracts between stages, and the Manager pattern with 10 specialist agents.

### Scores

| # | Category | Score | Justification |
|---|----------|-------|---------------|
| 1 | Mission Alignment | **8** | Good alignment: control plane vs execution plane maps to "LLM for creative decisions, deterministic for known problems." Evidence-based learning, fail-loud routing, earned autonomy. Less explicit about generality and flexibility compared to Proposal A. |
| 2 | Flexibility | **6** | Lists "add rigid body + particles" in Phase 2 but doesn't describe HOW. TechniqueResearchAgent exists but is barely developed. No Explorer equivalent with diagnostic scripting tools. No mechanism for runtime discovery of new physics systems. The generalization path exists on paper but lacks substance. |
| 3 | Anti-Hallucination | **7** | `BlenderCapabilityRegistry` with versioned feature flags is architecturally clean. `TruthProbe` that queries docs/MCP before uncertain API use is a good concept. Adapter pattern isolates deprecations to boundaries. But no runtime introspection (live `dir()` calls). Registry must be manually maintained per Blender version. Less powerful than Proposal A's whitelist approach. |
| 4 | Self-Learning | **7** | Three-tier memory (run/session/knowledge) is clean. Knowledge promotion rules are solid: repeated success on held-out prompts, statistical significance, no safety/cost regressions, human review for high-impact rules. But less concrete on implementation — no code examples, no promotion threshold numbers. |
| 5 | SDK Utilization | **9** | Best SDK utilization of any proposal. Explicitly names: `Agent.as_tool` (manager pattern), `needs_approval` + `RunState` (HITL), `call_model_input_filter` (context trimming), SDK `Sessions` (3 tiers), tracing with `workflow_name`/`group_id`/metadata, `run_streamed`, tool guardrails. Shows genuine familiarity with the full SDK feature set. Also references OpenAI API features (conversation state, webhooks, backgrounds). |
| 6 | HITL | **8** | Correctly identifies SDK HITL: `needs_approval` + `RunState` pause/resume. Dedicated `UserApprovalAgent` for HITL policy. Approval gates for destructive/expensive actions. Phase 3 explicitly gates autonomy reduction on measured reliability. The strongest HITL design of the proposals. |
| 7 | Feasibility | **8** | Most pragmatic migration advice: "Build a thin new kernel beside the old system and migrate lanes one by one." Phase 0 is schemas + state machine + tracing — the right foundation. More conservative timeline. Doesn't overpromise on novel capabilities early. But less specific about what each phase delivers. |
| 8 | Code Specificity | **4** | The weakest area. Almost no code examples. Agent names listed but not defined. Schemas named but not specified. Data contracts mentioned but not shown. Compare to Proposal A's extensive code blocks — this reads more like an architecture memo than an implementation guide. |
| 9 | Budget Awareness | **7** | Per-run budget envelope, per-stage spend caps, stop conditions for non-improving iterations. Model allocation policy per stage. But no specific cost model per operation, no model downgrade routing by budget level, no tiered evaluation strategy. |
| 10 | Observability | **8** | `DecisionRecord` per loop is mandatory. Mandatory transparency record. Tracing with `workflow_name`, `group_id`, metadata. Eval harness + baseline datasets. Continuous eval program across known, novel, and adversarial prompts. More systematic than Proposal A. |

**Total: 72/100**

### Key Strengths
1. **Best SDK utilization** — correctly uses HITL, sessions, streaming, call_model_input_filter
2. **Typed data contracts** — versioned schemas as the only message payloads between stages
3. **10 specialist agents** — more granular role separation than Proposal A's 4
4. **"Script Generation Policy"** — primary + diagnostic + rollback scripts per attempt
5. **Migration advice** — "thin new kernel beside old system" is pragmatic

### Key Weaknesses
1. **Very low code specificity** — reads like an architecture memo, not an implementation guide
2. **Flexibility is underdeveloped** — TechniqueResearchAgent exists in name only
3. **Anti-hallucination relies on pre-built registry** — no runtime introspection
4. **10 agents may be over-decomposed** — IntentPlannerAgent, SceneSpecAgent, and ScriptWriterAgent could be 2 agents rather than 3
5. **No Explorer equivalent** — no mechanism for "run a Blender script to discover APIs"

### Unique Ideas Worth Taking
- **Typed data contracts as the only message payloads** — eliminates "untyped prose" between stages
- **`TruthProbe`** — queries docs/MCP before uncertain API use (complements introspection)
- **Primary + diagnostic + rollback scripts** — three script variants per generation
- **`BlenderCapabilityRegistry`** — versioned feature flags per Blender version
- **Adapter pattern** — isolates Blender API to adapter boundaries
- **`ExperimentPolicyAgent`** — dedicated agent for deciding iteration strategy
- **Physical plausibility proxies** — domain-specific checks (bounds, continuity, temporal behavior) as a separate eval layer
- **`call_model_input_filter`** — context trimming/redaction hook from SDK

---

## Comparison Matrix

| Category | Proposal A (From Scratch) | Proposal B (Reboot) | Winner |
|----------|--------------------------|---------------------|--------|
| Mission Alignment | 9 | 8 | **A** |
| Flexibility | 9 | 6 | **A** |
| Anti-Hallucination | 10 | 7 | **A** |
| Self-Learning | 8 | 7 | **A** |
| SDK Utilization | 6 | 9 | **B** |
| HITL | 5 | 8 | **B** |
| Feasibility | 7 | 8 | **B** |
| Code Specificity | 9 | 4 | **A** |
| Budget Awareness | 8 | 7 | **A** |
| Observability | 6 | 8 | **B** |
| **TOTAL** | **77** | **72** | **A** |

### Pattern

**Proposal A excels at:** The core technical problem (hallucination, flexibility, code generation architecture). It's the stronger *technical* proposal.

**Proposal B excels at:** The operational infrastructure (SDK integration, HITL, observability, migration). It's the stronger *engineering practice* proposal.

**They're complementary, not competing.** The ideal system takes Proposal A's technical architecture (introspection sandbox, Explorer agent, Planner/Coder split) and wraps it in Proposal B's operational discipline (typed contracts, SDK HITL, tracing, migration strategy).

---

## Combined "Best Of" Ideas (Ranked by Impact)

| Rank | Idea | Source | Impact | Why |
|------|------|--------|--------|-----|
| 1 | Introspection Sandbox (live Blender `dir()` validation) | A | **Critical** | Eliminates entire category of hallucination bugs. Whitelist > blacklist. |
| 2 | Explorer Agent with diagnostic scripts | A | **Critical** | Enables novel physics types and runtime API discovery. Solves flexibility problem. |
| 3 | SDK HITL with `needs_approval` + `RunState` | B | **High** | Real pause/resume/approval flow. Could have saved many failed runs. |
| 4 | Typed data contracts between stages | B | **High** | Eliminates parsing ambiguity. Makes testing tractable. |
| 5 | Planner/Coder agent separation | A | **High** | Better creative reasoning + cleaner code generation. |
| 6 | 3-layer anti-hallucination (prevent → detect → learn) | A | **High** | Defense in depth. Most hallucinations caught before execution. |
| 7 | Tiered quality evaluation (cheap early, ML late) | A | **Medium** | Saves budget. Fast early iteration. |
| 8 | Primary + diagnostic + rollback scripts | B | **Medium** | Built-in recovery path. Reduced iteration cost on failures. |
| 9 | Budget-aware model routing | A | **Medium** | Extends budget life. Pragmatic. |
| 10 | Continuous eval program (known + novel + adversarial) | B | **Medium** | Regression detection. Prevents drift. |
| 11 | Physical plausibility proxies | B | **Medium** | Domain-specific checks catch physics nonsense cheaply. |
| 12 | Anti-learning rules | A | **Low** | Prevents bad knowledge contamination. Good discipline. |
| 13 | `call_model_input_filter` for context trimming | B | **Low** | Prevents context overflow in long sessions. |
| 14 | Adapter pattern for Blender API | B | **Low** | Clean but adds abstraction cost. |

---

## Proposal C: "Claude Opus Architecture Proposal"

**Source:** `CLAUDE_OPUS_ARCHITECTURE_PROPOSAL_2026-02-18.md`
**Author:** Claude Opus 4.6 (solo, high effort mode)
**Length:** ~1,293 lines
**Approach:** Literature-backed 3-agent architecture (Writer/Evaluator/Strategist) with deterministic safety layer, SQLite knowledge base, and context resets per iteration. Heavily references academic papers (AgentCoder, Hybrid Cascade, GEPA, Ralph).

### Scores

| # | Category | Score | Justification |
|---|----------|-------|---------------|
| 1 | Mission Alignment | **8** | Good separation of creative vs mechanical. Explicitly maps each S1.5 failure to a deterministic fixer. References mission statement principles. However, research/exploration is Phase 5 ("Polish"), not a core capability — at odds with mission statement's "Generality Over Specialization." Research Agent is marked "optional." |
| 2 | Flexibility | **7** | Research Agent exists with `run_diagnostic_script` and `search_blender_docs` tools — can investigate unfamiliar APIs. But it's on gpt-5-mini (cheapest model), called only when Strategist decides to invoke it, and is explicitly "optional." Novel physics types are deferred to Phase 5. System COULD handle "brick wall collapsing" eventually, but it's not a design priority. |
| 3 | Anti-Hallucination | **6** | Uses the same blacklist approach as the current system (`BLENDER_5_RENAMES` lookup table). Well-implemented but fundamentally the same pattern that grows linearly and misses novel hallucinations. Does inject doc search results into dynamic instructions (Layer 1 prevention), and has `run_diagnostic_script` on the Researcher. But no introspection sandbox, no live `dir()` validation, no AST-based attribute checking. The proposal explicitly relies on the lookup table + prompting, which is the approach that's been failing. |
| 4 | Self-Learning | **9** | The strongest self-learning design across all proposals. SQLite with FTS5 instead of vector store — pragmatic, zero API cost, debuggable. Four well-designed tables (techniques, code_patterns, failure_patterns, iteration_records). Dynamic instructions inject patterns from knowledge base. Clear "what to learn" and "what NOT to learn" sections. Simplified escape velocity (2 levels instead of 5). References GEPA and Ralph for academic grounding. Knowledge connected to agent behavior via dynamic instructions — the complete loop. |
| 5 | SDK Utilization | **8** | Strong and accurate. Uses: dynamic instructions (callable), `RunContextWrapper[T]`, `@input_guardrail`, `RunHooks`, `ModelSettings.reasoning` for o3, `agent.as_tool()`, `output_type` (Pydantic), `Tracing`. Appendix A lists 13 used features and 9 deliberately skipped with rationale. Misses SDK HITL (`needs_approval` + `RunState`) — implements custom exception instead. Also explicitly skips streaming and tool guardrails. |
| 6 | HITL | **6** | Clear intervention triggers table (novel effect, circuit breaker, plateau, budget, escalation). Custom exception-based pause/resume (`HumanInputRequired`). Session save/load for resume. Transparency report (`PipelineReport`). But incorrectly states "The SDK doesn't have a native pause and wait for human input mechanism" — it does (`needs_approval` + `RunState` serialization). Custom implementation is functional but less robust than SDK-native approach. |
| 7 | Feasibility | **9** | The most buildable proposal. 15-day plan in 5 phases, each with "Test:" checkpoints. Only 3 core agents (minimal complexity). Pipeline controller is "~80 lines." Clean file structure (Appendix C). SQLite over vector store reduces dependencies. Directly addresses every S1.5 failure with specific deterministic fixers. Appendix B maps each failure to its prevention mechanism. Academic references validate the agent count and pattern choices. |
| 8 | Code Specificity | **10** | The most code-complete proposal of all. Full Agent definitions with SDK patterns. Complete `fix_script()` with 6 sub-fixers. Camera validation snippet (38 lines, production-ready). Light energy clamping with regex. Collision effector injection. SQLite schema (4 tables). Complete pipeline controller (~80 lines). Error classification enum + recovery strategy with pattern matching. Circuit breaker. Budget hooks + guardrails. Pipeline report dataclass. File structure appendix. Nearly every concept is shown with runnable, copy-pasteable Python. |
| 9 | Budget Awareness | **9** | Best budget design across all proposals. Pre-flight cost estimation. Per-iteration cost: $0.06 → 66 assets/month at $20 budget (concrete, believable math). `BudgetHooks` via `RunHooks.on_llm_end`. `@input_guardrail` for budget check on expensive agents. gpt-5-mini for Evaluator and Researcher (cost-efficient). "What NOT to build" prevents over-spending on unnecessary agents. Per-call cost table for each model. |
| 10 | Observability | **7** | `PipelineReport` dataclass with comprehensive fields (session_id, iterations, scores, fixes, strategy reasoning, cost, duration). Tracing with `with trace(...)`. `RunHooks` for logging. Iteration summaries with compression. But no structured `RunSummary` for cross-run comparison. No explicit regression tracking dashboard. No reproducibility contract (seeds, versions, hashes). Less systematic than Proposal B's continuous eval program. |

**Total: 79/100**

### Key Strengths
1. **Most code-complete proposal** — nearly everything is runnable Python
2. **Most buildable** — 15-day plan, 3 agents, clear file structure, minimal dependencies
3. **Best self-learning design** — SQLite + FTS5, evidence-gated, connected to dynamic instructions
4. **Best budget math** — $0.06/iteration, 66 assets/month, concrete and believable
5. **Literature-backed** — AgentCoder, Hybrid Cascade, GEPA, Ralph validate design choices
6. **Context resets per iteration** — prevents context window bloat, validated by Ralph pattern
7. **Appendix B** — explicitly maps every S1.5 failure to its prevention mechanism

### Key Weaknesses
1. **Anti-hallucination is a blacklist** — same `BLENDER_5_RENAMES` approach that grows linearly and misses novel hallucinations. No introspection sandbox.
2. **Flexibility is secondary** — Research Agent is "optional," on cheapest model, deferred to Phase 5
3. **Missed SDK HITL** — incorrectly says SDK lacks native HITL, implements custom exception instead
4. **No streaming** — explicitly skipped ("no real-time UI") despite Ben identifying it as important
5. **No parallelism** — doc search, knowledge queries, and introspection could run concurrently
6. **No reproducibility contract** — no seed/model/version stamps on eval runs

### Unique Ideas Worth Taking
- **SQLite + FTS5 for knowledge base** — replaces vector store at small scale, zero API cost
- **Context resets per iteration** — fresh `Runner.run()` with compressed briefing, not raw history
- **Strategist curates Writer input** — Writer NEVER sees raw error logs, only curated briefing
- **Evaluator is stateless** — no anchoring bias, doesn't know iteration count or history
- **Circuit breaker** — 3 identical errors OR 3 iterations with <5 point change = halt
- **Error classification enum** — typed error routing (crash/timeout/syntax/attribute/physics/infra)
- **Pipeline is ~80 lines** — proves the orchestrator doesn't need to be 4,429 lines
- **"66 assets/month"** — first proposal to calculate throughput at budget constraint
- **Academic validation** — AgentCoder proves 3-agent pattern works (79.9% → 91.5% pass@1)

---

## Updated Comparison Matrix

| Category | A (From Scratch) | B (Reboot) | C (Opus Solo) | Best |
|----------|-----------------|------------|---------------|------|
| Mission Alignment | 9 | 8 | 8 | **A** |
| Flexibility | 9 | 6 | 7 | **A** |
| Anti-Hallucination | 10 | 7 | 6 | **A** |
| Self-Learning | 8 | 7 | 9 | **C** |
| SDK Utilization | 6 | 9 | 8 | **B** |
| HITL | 5 | 8 | 6 | **B** |
| Feasibility | 7 | 8 | 9 | **C** |
| Code Specificity | 9 | 4 | 10 | **C** |
| Budget Awareness | 8 | 7 | 9 | **C** |
| Observability | 6 | 8 | 7 | **B** |
| **TOTAL** | **77** | **72** | **79** | **C** |

### Three-Proposal Pattern

Each proposal leads in different areas:

**Proposal A (77)** leads: Mission Alignment, Flexibility, Anti-Hallucination
→ Best *technical architecture* — the introspection sandbox is irreplaceable

**Proposal B (72)** leads: SDK Utilization, HITL, Observability
→ Best *engineering practices* — typed contracts, SDK-native HITL, continuous eval

**Proposal C (79)** leads: Self-Learning, Feasibility, Code Specificity, Budget Awareness
→ Best *implementation blueprint* — most buildable, best budget math, strongest knowledge system

### The Ideal Combination

The highest-scoring system would combine:
- **From A:** Introspection Sandbox, Explorer Agent with diagnostic tools, 3-layer anti-hallucination
- **From B:** SDK HITL (`needs_approval` + `RunState`), typed data contracts, `call_model_input_filter`, continuous eval program, reproducibility contract
- **From C:** 3-agent core (Writer/Evaluator/Strategist), SQLite knowledge base, context resets, circuit breaker, error classification, ~80-line pipeline, budget math

This combination would score approximately: 9+9+10+9+9+8+9+10+9+8 = **90/100**

---

## Pending Evaluations

*Space reserved for additional proposals as they are submitted.*

### Proposal D: _(pending)_

---

## Notes for Roadmap Integration

After all proposals are evaluated, the "Combined Best Of" list will be reconciled with the current S3 roadmap v0.2. Key integration questions:

1. Does the Introspection Sandbox replace or supplement the current API fixer?
2. Where does the Explorer Agent fit in the phase plan?
3. How does Planner/Coder separation interact with the template system?
4. What SDK features are missing from the current codebase that must be added?
5. Does the template system (Lane A) coexist with introspection-validated code generation (Lane B)?
