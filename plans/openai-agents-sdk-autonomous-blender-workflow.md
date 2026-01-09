# OpenAI Agents SDK Autonomous Blender VFX Workflow

## Executive Summary

This plan consolidates the current fragmented MCP server architecture (11 servers, 201 tools) into a unified autonomous agent system using the OpenAI Agents SDK. The goal is to create a self-improving VFX asset generation pipeline that:

1. Deeply understands Blender 5.0 through documentation search
2. Autonomously generates, executes, evaluates, and iterates on VFX assets
3. Learns from successes/failures to escape local minima
4. Maintains state across context limits

**Target Architecture**: Single `BlenderVFXOrchestrator` agent that coordinates 5 specialized agents via handoffs, replacing manual tool orchestration with autonomous decision-making.

---

## Current State Analysis

### Existing MCP Servers (11 total, 201 tools)

| Server | Status | Tools | Pattern | Action |
|--------|--------|-------|---------|--------|
| `script-generator` | Working | 12 | FastMCP | Wrap as Agent tool |
| `blender-executor` | Working | 5 | FastMCP | Wrap as Agent tool |
| `asset-evaluator` | Working | 40+ | FastMCP | Wrap as Agent tool |
| `experiment-tracker` | Working | 12 | FastMCP | Wrap as Agent tool |
| `iteration-controller` | Working | 9 | FastMCP | **Deprecate** (orchestrator absorbs) |
| `blender-manual` | Working | 12 | FastMCP | MCP via MCPServerStdio |
| `blender-librarian` | Working | 8 | **OpenAI Agents SDK** | **Reference implementation** |
| `blender-orchestrator` | Partial | 4 | FastMCP + CLI | **Replace with new orchestrator** |
| `mission-control` | Broken | 4 | Claude Agent SDK | **Deprecate** |
| `log-analysis-rag` | Working | 5 | FastMCP | Optional integration |
| `pix-debug` | Working | 10 | FastMCP | Optional integration |

### Reference Implementation: `blender-librarian`

The `blender-librarian` server already demonstrates key patterns we'll extend:

```python
# From librarian_agents/librarian_orchestrator.py
class LibrarianOrchestrator:
    async def initialize(self, session_context: str = "") -> None:
        # Create specialized agents using pooled MCP connection
        self._doc_expert = await create_doc_expert_pooled(custom_instructions=session_context)
        self._vision_expert = create_vision_expert(custom_instructions=session_context)

        # Create orchestrator with handoffs to specialists
        self._orchestrator = Agent(
            name="Blender Librarian",
            instructions=instructions,
            model=self.model,
            handoffs=[
                handoff(agent=self._doc_expert, tool_name_override="consult_documentation_expert"),
                handoff(agent=self._vision_expert, tool_name_override="consult_vision_expert"),
            ]
        )
```

**Key Patterns to Reuse**:
1. `MCPConnectionPool` singleton for MCP server reuse
2. Dynamic reasoning effort estimation
3. Budget tracking with tiered limits
4. Playbook system for FREE cached fixes
5. Session state persistence via JSON

---

## Target Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BlenderVFXOrchestrator (GPT-5.2)                    │
│  - Session state management                                              │
│  - Quality gate enforcement (LPIPS ≥ 0.85, score ≥ 60)                  │
│  - Budget/autonomy control                                               │
│  - Iteration loop with escape hatch logic                               │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │ handoffs (autonomous delegation)
          ┌─────────────┼─────────────┬─────────────┬─────────────┐
          ▼             ▼             ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ScriptWriter │ │  Executor   │ │QualityAnalyst│ │LearningAgent│ │  DocsExpert │
│   Agent     │ │   Agent     │ │    Agent    │ │   Agent     │ │    Agent    │
│             │ │             │ │             │ │             │ │             │
│@function_tool│ │@function_tool│ │@function_tool│ │@function_tool│ │MCPServerStdio│
│- generate   │ │- execute    │ │- evaluate_v2│ │- query_kb   │ │- blender-   │
│- modify     │ │- parse_err  │ │- diagnose_v2│ │- record_fix │ │  manual     │
│- validate   │ │- list_runs  │ │- compare_v2 │ │- suggest    │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │               │               │               │               │
       └───────────────┴───────────────┴───────────────┴───────────────┘
                                       │
                              Shared Context (Pydantic)
                              - current_params
                              - iteration_history
                              - quality_scores
                              - learned_fixes
```

### Agent Specifications

#### 1. BlenderVFXOrchestrator (Main Agent)

**Model**: `gpt-5.2` with medium reasoning effort

**Responsibilities**:
- Receive asset generation requests
- Manage iteration loop (max 10 iterations)
- Enforce quality gates
- Detect stuck states and trigger escape strategies
- Budget enforcement ($20/month limit)
- Session persistence across context limits

**Instructions** (condensed):
```
You are an autonomous VFX asset generation orchestrator.

ITERATION LOOP:
1. Generate/modify script via ScriptWriterAgent
2. Execute via ExecutorAgent
3. Evaluate via QualityAnalystAgent
4. If passed → complete session
5. If failed → consult LearningAgent for fixes
6. If stuck (3+ iterations, <5% improvement) → consult DocsExpert for new approaches
7. Apply fixes and repeat

QUALITY GATES:
- LPIPS ≥ 0.85 (if reference provided)
- Overall score ≥ 60
- No critical issues

BUDGET LIMITS:
- $20/month total
- $10 vision analysis
- $8 documentation search
- $2 emergency buffer
```

#### 2. ScriptWriterAgent

**Model**: `gpt-4.1-mini` (fast, cost-effective)

**Tools** (wrapped from `script-generator` MCP):
- `generate_script(effect_type, description, technique_name)`
- `modify_script(script_path, modifications)`
- `validate_script(script_path)`
- `recommend_technique(effect_type, description)` - UCB1 algorithm

**Instructions**:
```
You generate Blender 5.0 Mantaflow Python scripts for VFX effects.

ALWAYS:
- Use recommend_technique() first for technique selection
- Validate scripts before returning
- Apply parameter ranges from Blender API

PARAMETER RANGES (enforced):
- turbulence: 0.0-1.0
- flame_smoke: 0.0-8.0
- burning_rate: 0.01-4.0
- domain_resolution: 32-512
```

#### 3. ExecutorAgent

**Model**: `gpt-4.1-mini`

**Tools** (wrapped from `blender-executor` MCP):
- `execute_blender_script(script_path, script_args, output_dir)`
- `parse_blender_errors(stderr, stdout)`
- `list_run_outputs(run_dir)`
- `get_latest_run()`

**Instructions**:
```
You execute Blender scripts and handle errors.

ON ERROR:
1. Parse error with parse_blender_errors()
2. Return structured error with suggested_fix
3. Common fixes:
   - "openvdb_cache_compress_type" → Check Blender 5.0 API changes
   - "poll failed" → Set active object and mode
```

#### 4. QualityAnalystAgent

**Model**: `gpt-4.1-mini` with vision capability

**Tools** (wrapped from `asset-evaluator` MCP):
- `evaluate_render_v2(render_path, reference_path, profile)` - Consolidated evaluation
- `diagnose_issues_v2(render_path, effect_type)` - VLM-powered diagnostics
- `compare_renders_v2(render_a, render_b)` - Iteration comparison

**Instructions**:
```
You evaluate VFX render quality using ML metrics.

METRICS (evaluate_render_v2):
- LPIPS: Perceptual similarity (lower = better)
- SigLIP 2: Semantic alignment
- TOPIQ: Image quality
- Feature CV: Texture variety
- DINOv2: Structural similarity (comprehensive profile)

PROFILES:
- quick: LPIPS + SigLIP (~2s)
- standard: + TOPIQ, feature_cv (~10s)
- comprehensive: + DINOv2, VLM diagnosis (~30s)
```

#### 5. LearningAgent

**Model**: `gpt-4.1-mini`

**Tools** (wrapped from `experiment-tracker` MCP):
- `query_knowledge_base(query)`
- `record_experiment_result(hypothesis, success, learnings)`
- `suggest_experiments(issue, current_params)`
- `get_warnings_before_change(parameter, change_type)`
- `add_manual_learning(parameter, rule, warning)`

**Instructions**:
```
You maintain the learning system for VFX generation.

BEFORE CHANGES:
- Always call get_warnings_before_change()
- Check knowledge base for similar issues

AFTER EXPERIMENTS:
- Record ALL results (success AND failure)
- Extract learnings for future sessions
- Add manual learnings for discovered gotchas
```

#### 6. DocsExpertAgent

**Model**: `gpt-5.2` (needs reasoning for synthesis)

**MCP Integration**: `MCPServerStdio` → `blender-manual` server

**Tools** (native MCP tools, 12 total):
- `search_semantic(query)` - Natural language search
- `search_vdb_workflow(query)` - VDB/volume topics
- `search_python_api(operation)` - bpy.ops, bpy.types
- `search_bpy_types(typename)` - Type properties
- `read_page(path)` - Full page content

**Instructions**:
```
You are a Blender 5.0 documentation expert.

SEARCH STRATEGY (in order):
1. search_semantic - Natural language questions
2. search_vdb_workflow - VDB/volume/caching
3. search_python_api - bpy.ops, bpy.types
4. search_bpy_types - Specific type properties
5. read_page - Full content for promising results

OUTPUT FORMAT:
- answer: 1-2 sentence summary
- modifications: dict of parameter changes
- rationale: Why these changes help
- confidence: 0.0-1.0 (0.3 if docs not found)
- citations: documentation paths used
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Objective**: Create base infrastructure and migrate existing patterns

#### Tasks

1. **Create unified agent package structure**
   ```
   agents/blender-vfx-orchestrator/
   ├── __init__.py
   ├── server.py                    # FastMCP + CLI entry point
   ├── orchestrator.py              # BlenderVFXOrchestrator class
   ├── agents/
   │   ├── __init__.py
   │   ├── script_writer.py         # ScriptWriterAgent
   │   ├── executor.py              # ExecutorAgent
   │   ├── quality_analyst.py       # QualityAnalystAgent
   │   ├── learning_agent.py        # LearningAgent
   │   └── docs_expert.py           # DocsExpertAgent (port from librarian)
   ├── tools/
   │   ├── __init__.py
   │   ├── script_generator_tools.py    # @function_tool wrappers
   │   ├── executor_tools.py
   │   ├── evaluator_tools.py
   │   └── experiment_tools.py
   ├── models/
   │   ├── __init__.py
   │   ├── shared_context.py        # SharedContext Pydantic model
   │   ├── asset_request.py
   │   ├── iteration_result.py
   │   └── session_state.py
   ├── utils/
   │   ├── __init__.py
   │   ├── mcp_connection_pool.py   # Port from librarian
   │   ├── budget_tracker.py        # Port from librarian
   │   └── session_persistence.py
   ├── requirements.txt
   ├── run_server.sh
   └── .env.example
   ```

2. **Port `MCPConnectionPool` from blender-librarian**
   - Singleton pattern for MCP server reuse
   - Pre-warm embeddings flag
   - 120s timeout for cold starts

3. **Port `BudgetTracker` from blender-librarian**
   - $20/month limit
   - Tiered allocation (vision/docs/buffer)
   - JSON persistence

4. **Create Pydantic models**
   ```python
   class SharedContext(BaseModel):
       session_id: str
       asset_name: str
       effect_type: str
       current_iteration: int = 0
       max_iterations: int = 10
       current_params: Dict[str, Any] = {}
       iteration_history: List[IterationResult] = []
       quality_scores: Dict[str, float] = {}
       learned_fixes: List[Dict[str, Any]] = []

   class IterationResult(BaseModel):
       iteration: int
       script_path: str
       render_path: str
       scores: Dict[str, float]
       passed: bool
       issues: List[str]
       fixes_applied: List[Dict[str, Any]]
   ```

5. **Write comprehensive tests**
   - Unit tests for each agent
   - Integration tests for handoff flows
   - Mock MCP servers for CI

**Deliverables**:
- [ ] Package structure created
- [ ] MCPConnectionPool ported and tested
- [ ] BudgetTracker ported and tested
- [ ] Pydantic models defined
- [ ] Basic test suite passing

### Phase 2: Agent Implementation (Week 2)

**Objective**: Implement all 5 specialized agents

#### Tasks

1. **Implement ScriptWriterAgent**
   ```python
   # agents/script_writer.py
   from agents import Agent, function_tool
   from ..tools.script_generator_tools import (
       generate_script, modify_script, validate_script, recommend_technique
   )

   SCRIPT_WRITER_INSTRUCTIONS = """..."""

   def create_script_writer_agent() -> Agent:
       return Agent(
           name="Script Writer",
           instructions=SCRIPT_WRITER_INSTRUCTIONS,
           model="gpt-4.1-mini",
           tools=[generate_script, modify_script, validate_script, recommend_technique]
       )
   ```

2. **Implement ExecutorAgent**
   - Wrap `blender-executor` MCP tools
   - Error parsing with suggested fixes
   - Run output management

3. **Implement QualityAnalystAgent**
   - Wrap `asset-evaluator` consolidated tools (v2 APIs)
   - Profile selection (quick/standard/comprehensive)
   - Issue diagnosis

4. **Implement LearningAgent**
   - Wrap `experiment-tracker` MCP tools
   - Knowledge base queries
   - Experiment recording

5. **Port DocsExpertAgent from blender-librarian**
   - Reuse `MCPConnectionPool`
   - Reuse search strategy instructions
   - Add validation tools

**Deliverables**:
- [ ] ScriptWriterAgent implemented and tested
- [ ] ExecutorAgent implemented and tested
- [ ] QualityAnalystAgent implemented and tested
- [ ] LearningAgent implemented and tested
- [ ] DocsExpertAgent ported and tested

### Phase 3: Orchestrator Implementation (Week 3)

**Objective**: Create the main orchestrator with handoff logic

#### Tasks

1. **Implement BlenderVFXOrchestrator**
   ```python
   # orchestrator.py
   from agents import Agent, handoff, Runner
   from agents.mcp import MCPServerStdio
   from .agents import (
       create_script_writer_agent,
       create_executor_agent,
       create_quality_analyst_agent,
       create_learning_agent,
       create_docs_expert_pooled
   )

   class BlenderVFXOrchestrator:
       async def initialize(self) -> None:
           # Create specialized agents
           self._script_writer = create_script_writer_agent()
           self._executor = create_executor_agent()
           self._quality_analyst = create_quality_analyst_agent()
           self._learning_agent = create_learning_agent()
           self._docs_expert = await create_docs_expert_pooled()

           # Create main orchestrator with handoffs
           self._orchestrator = Agent(
               name="Blender VFX Orchestrator",
               instructions=ORCHESTRATOR_INSTRUCTIONS,
               model="gpt-5.2",
               model_settings=ModelSettings(
                   reasoning=Reasoning(effort="medium")
               ),
               handoffs=[
                   handoff(self._script_writer, "delegate_to_script_writer"),
                   handoff(self._executor, "delegate_to_executor"),
                   handoff(self._quality_analyst, "delegate_to_quality_analyst"),
                   handoff(self._learning_agent, "delegate_to_learning_agent"),
                   handoff(self._docs_expert, "delegate_to_docs_expert"),
               ]
           )

       async def create_asset(self, request: AssetRequest) -> SessionResult:
           """Main entry point for asset generation."""
           context = SharedContext(
               session_id=generate_session_id(),
               asset_name=request.asset_name,
               effect_type=request.effect_type,
               max_iterations=request.max_iterations
           )

           # Build prompt with context
           prompt = self._build_generation_prompt(request, context)

           # Run orchestrator (autonomous iteration loop)
           result = await Runner.run(
               self._orchestrator,
               prompt,
               context={"shared": context.model_dump()}
           )

           return self._parse_result(result, context)
   ```

2. **Implement iteration loop logic in instructions**
   - Quality gate checks
   - Stuck detection (3+ iterations, <5% improvement)
   - Escape strategies (technique switch, docs consultation)

3. **Implement session persistence**
   - Save state after each iteration
   - Resume from saved state
   - Handle context limit gracefully

4. **Implement budget enforcement**
   - Track costs per operation
   - Warn at 80% budget
   - Hard stop at 100%

**Deliverables**:
- [ ] Orchestrator class implemented
- [ ] Handoff flow working end-to-end
- [ ] Session persistence working
- [ ] Budget enforcement working
- [ ] Integration tests passing

### Phase 4: Integration & Polish (Week 4)

**Objective**: MCP server exposure, CLI, and production hardening

#### Tasks

1. **Create FastMCP server wrapper**
   ```python
   # server.py
   from mcp.server.fastmcp import FastMCP
   from .orchestrator import BlenderVFXOrchestrator

   mcp = FastMCP("blender-vfx-orchestrator")
   orchestrator: Optional[BlenderVFXOrchestrator] = None

   @mcp.tool()
   async def create_asset(
       asset_name: str,
       effect_type: str,
       description: str,
       reference_path: str = "",
       max_iterations: int = 10
   ) -> str:
       """Create a VFX asset through autonomous iteration."""
       global orchestrator
       if not orchestrator:
           orchestrator = BlenderVFXOrchestrator()
           await orchestrator.initialize()

       request = AssetRequest(
           asset_name=asset_name,
           effect_type=effect_type,
           description=description,
           reference_path=reference_path,
           max_iterations=max_iterations
       )

       result = await orchestrator.create_asset(request)
       return result.model_dump_json()
   ```

2. **Create CLI interface**
   - `python server.py --mcp` for MCP mode
   - `python server.py create <name> <type> <description>` for direct use
   - `python server.py resume <session_id>` for resumption

3. **Add observability**
   - Structured logging with iteration context
   - Cost tracking per session
   - Performance metrics

4. **Production hardening**
   - Retry with exponential backoff
   - Timeout handling (25s per agent call)
   - Graceful degradation

5. **Documentation**
   - Update CLAUDE.md
   - Create OPERATOR_MANUAL.md
   - API reference

**Deliverables**:
- [ ] FastMCP server working
- [ ] CLI interface working
- [ ] Logging and metrics implemented
- [ ] Error handling robust
- [ ] Documentation complete

---

## Migration Strategy

### Deprecation Plan

| Server | Action | Timeline |
|--------|--------|----------|
| `iteration-controller` | Deprecate (absorbed by orchestrator) | Phase 3 |
| `blender-orchestrator` | Replace with new implementation | Phase 4 |
| `mission-control` | Remove (broken, Claude SDK) | Phase 1 |
| `blender-librarian` | Keep (reference, may merge later) | Post-Phase 4 |

### Backward Compatibility

During transition:
1. Keep existing MCP servers running
2. New orchestrator calls them via `@function_tool` wrappers
3. Gradual migration of direct MCP calls to orchestrator
4. Eventually deprecate direct MCP access

---

## Quality Gates

### Per-Iteration Quality Checks

```python
def check_quality_gates(result: IterationResult, reference_path: str = None) -> bool:
    # Must pass ALL gates
    gates = [
        result.scores.get("overall_score", 0) >= 60,
        len([i for i in result.issues if "critical" in i.lower()]) == 0,
    ]

    # LPIPS gate only if reference provided
    if reference_path:
        gates.append(result.scores.get("lpips", 1.0) <= 0.15)  # Lower is better

    return all(gates)
```

### Stuck Detection

```python
def is_stuck(history: List[IterationResult], window: int = 3) -> bool:
    if len(history) < window:
        return False

    recent = history[-window:]
    scores = [r.scores.get("overall_score", 0) for r in recent]

    # Less than 5% improvement over window
    improvement = max(scores) - min(scores)
    return improvement < 5.0
```

### Escape Strategies

When stuck:
1. **Technique Switch**: Use UCB1 to select untried technique
2. **Documentation Search**: Consult DocsExpert for new approaches
3. **Parameter Reset**: Return to defaults and try different direction
4. **Human Escalation**: After 2 failed escape attempts

---

## Cost Estimation

### Per-Session Costs (10 iterations)

| Agent | Calls | Tokens/Call | Cost/Call | Total |
|-------|-------|-------------|-----------|-------|
| Orchestrator (GPT-5.2) | 10 | 4000 | $0.08 | $0.80 |
| ScriptWriter (GPT-4.1-mini) | 10 | 2000 | $0.004 | $0.04 |
| Executor (GPT-4.1-mini) | 10 | 1000 | $0.002 | $0.02 |
| QualityAnalyst (GPT-4.1-mini) | 10 | 2000 | $0.004 | $0.04 |
| LearningAgent (GPT-4.1-mini) | 20 | 1500 | $0.003 | $0.06 |
| DocsExpert (GPT-5.2) | 3 | 5000 | $0.10 | $0.30 |
| **Total** | | | | **~$1.26/session** |

### Monthly Budget

- $20/month allows ~15 full sessions
- Or ~50 quick (3-iteration) sessions
- Vision analysis adds ~$0.50/session if used

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| MCP server timeouts | 120s timeout, pre-warm embeddings, connection pooling |
| GPT-5.2 hallucination | Parameter validation against Blender API ranges |
| Infinite iteration loops | Max 10 iterations, stuck detection, budget enforcement |
| Context limit during session | Session state persistence, resumption support |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| Cost overrun | Budget tracker with tiered limits, hard stops |
| Blender API changes | Version-specific parameter ranges, docs search |
| Quality regression | Comprehensive test suite, reference renders |

---

## Success Metrics

### Primary Metrics

1. **Automation Rate**: % of assets completed without human intervention
   - Target: 70% (up from ~30% with current system)

2. **Iteration Efficiency**: Average iterations to pass quality gates
   - Target: 4.5 iterations (down from 7+)

3. **Escape Success Rate**: % of stuck states successfully resolved
   - Target: 80%

### Secondary Metrics

- Cost per successful asset: <$2.00
- Time to first render: <60 seconds
- Session resumption success: 95%
- Budget utilization: 80-90%

---

## Open Questions (from SpecFlow Analysis)

### Critical (Must Resolve Before Phase 2)

1. **Multi-reference handling**: How to weight multiple reference images?
   - Proposed: Weighted average with user-specified weights

2. **Parallel execution**: Can agents run in parallel for speed?
   - Proposed: Sequential for Phase 1, parallel exploration in Phase 5

3. **Human-in-the-loop**: When exactly should orchestrator ask human?
   - Proposed: After 2 failed escape attempts, budget warnings at 80%

4. **Technique catalog expansion**: How to add new techniques?
   - Proposed: JSON catalog with UCB1 stats, manual or learned additions

5. **Cross-session learning**: How to share learnings between users?
   - Proposed: Local knowledge base initially, shared playbook later

### Important (Resolve Before Phase 4)

6. Graceful degradation when MCP servers unavailable
7. Rollback strategy for failed modifications
8. Version compatibility checks for Blender 5.x minor versions
9. Integration with existing PlasmaDX-Clean screenshot capture

---

## Appendix A: File Listing

### New Files to Create

```
agents/blender-vfx-orchestrator/
├── __init__.py
├── server.py
├── orchestrator.py
├── agents/__init__.py
├── agents/script_writer.py
├── agents/executor.py
├── agents/quality_analyst.py
├── agents/learning_agent.py
├── agents/docs_expert.py
├── tools/__init__.py
├── tools/script_generator_tools.py
├── tools/executor_tools.py
├── tools/evaluator_tools.py
├── tools/experiment_tools.py
├── models/__init__.py
├── models/shared_context.py
├── models/asset_request.py
├── models/iteration_result.py
├── models/session_state.py
├── utils/__init__.py
├── utils/mcp_connection_pool.py
├── utils/budget_tracker.py
├── utils/session_persistence.py
├── tests/__init__.py
├── tests/test_agents.py
├── tests/test_orchestrator.py
├── tests/test_integration.py
├── requirements.txt
├── run_server.sh
└── .env.example
```

### Files to Modify

- `CLAUDE.md` - Add new orchestrator documentation
- `.claude/settings.json` - Register new MCP server
- `docs/BLENDER_AUTONOMOUS_AGENT_ARCHITECTURE.md` - Mark as implemented

### Files to Deprecate

- `agents/iteration-controller/` - Absorbed by orchestrator
- `agents/mission-control/` - Broken, remove
- `agents/blender-orchestrator/` - Replaced by new implementation

---

## Appendix B: OpenAI Agents SDK Patterns

### Handoff Pattern (Recommended)

```python
from agents import Agent, handoff, Runner

specialist = Agent(name="Specialist", instructions="...", tools=[...])
orchestrator = Agent(
    name="Orchestrator",
    instructions="Delegate to specialist when needed",
    handoffs=[handoff(specialist, "consult_specialist")]
)

result = await Runner.run(orchestrator, "User request")
```

### Agents-as-Tools Pattern (Alternative)

```python
from agents import Agent, function_tool

specialist = Agent(name="Specialist", instructions="...", tools=[...])

@function_tool
async def consult_specialist(query: str) -> str:
    result = await Runner.run(specialist, query)
    return result.final_output

orchestrator = Agent(
    name="Orchestrator",
    tools=[consult_specialist]
)
```

### MCP Integration Pattern

```python
from agents import Agent
from agents.mcp import MCPServerStdio

mcp_server = MCPServerStdio(
    name="blender-manual",
    params={"command": "python", "args": ["server.py"]}
)
await mcp_server.connect()

agent = Agent(
    name="Docs Expert",
    mcp_servers=[mcp_server],  # Native MCP tool access
    tools=[custom_tool]        # Plus custom tools
)
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-07 | Claude Code | Initial plan based on architecture doc and SpecFlow analysis |
