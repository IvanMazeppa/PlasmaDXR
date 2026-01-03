# Multi-Agent Pipeline Improvement Plan

**Based on:** Key findings from `docs/MULTI_AGENT_PIPELINE_ANALYSIS.md`
**Goal:** Enhance pipeline autonomy, reliability, and learning capabilities
**Created:** 2026-01-03

---

## Summary of Key Findings

| # | Finding | Severity | Root Cause |
|---|---------|----------|------------|
| 1 | Orchestrator describes tools but may not execute them | HIGH | No execution verification layer |
| 2 | Technique selection is too random | MEDIUM | No success rate tracking per technique |
| 3 | Knowledge base exists but isn't automatically consulted | MEDIUM | Manual integration, not built into workflow |
| 4 | External research capability missing | LOW | No web research or technique discovery tools |
| 5 | Session resumption may be unreliable | LOW | State format compatibility issues |

---

## Phase 1: Orchestrator Reliability (Priority: HIGH)

**Problem:** The blender-orchestrator may describe MCP tool calls but not actually execute them.

### Tasks:

1. **Add MCP Tool Execution Verification**
   - File: `agents/blender-orchestrator/orchestrator.py`
   - Add wrapper function that verifies tool execution returned valid results
   - Log all tool calls with inputs/outputs for debugging
   - Implement retry logic with exponential backoff

2. **Implement Workflow Tracing**
   - File: `agents/blender-orchestrator/workflow_tracer.py` (new)
   - Create trace log for each session showing state transitions
   - Capture: timestamp, state, tool called, tool result, next state
   - Export as JSON for post-mortem analysis

3. **Add MCP Server Health Checks**
   - File: `agents/blender-orchestrator/health_check.py` (new)
   - Verify all 6 MCP servers are responsive before starting workflow
   - Test each server with minimal ping-style request
   - Report unhealthy servers in status output

### Success Criteria:
- Every MCP tool call has verified result or logged failure
- Session traces show complete workflow execution
- Health check runs at session start

---

## Phase 2: Intelligent Technique Selection (Priority: MEDIUM)

**Problem:** Technique selection uses simple keyword matching with random fallback. No learning from past success.

### Tasks:

1. **Add Technique Performance Table**
   - File: `agents/experiment-tracker/database.py`
   - New table: `technique_performance`
   - Columns: technique_name, effect_type, success_count, failure_count, avg_score, avg_iterations, last_used
   - Update after each completed session

2. **Implement Exploration/Exploitation Strategy**
   - File: `agents/script-generator/technique_selector.py` (new)
   - 80% exploitation: Use techniques with highest success rate for effect type
   - 20% exploration: Try underused or new techniques
   - Boost unexplored techniques to ensure coverage

3. **Add Technique Recommendation MCP Tool**
   - File: `agents/script-generator/server.py`
   - New tool: `recommend_technique(effect_type, description)`
   - Returns ranked list with confidence scores
   - Considers: effect type match, historical success, keyword relevance

### Success Criteria:
- Technique performance tracked across sessions
- Best techniques used 80% of the time
- New techniques still get tested (20% exploration)

---

## Phase 3: Automatic Knowledge Base Integration (Priority: MEDIUM)

**Problem:** experiment-tracker has warnings and suggestions but they're not automatically consulted.

### Tasks:

1. **Mandatory Warning Checks Before Parameter Changes**
   - File: `agents/blender-orchestrator/orchestrator.py`
   - Before any `modify_script()` call, query `get_warnings_before_change()`
   - Log warnings and adjust modifications if high-severity warnings found
   - Add warning check to DECIDE_NEXT_ACTION state

2. **Suggestion-Driven Iteration Loop**
   - File: `agents/iteration-controller/server.py`
   - Modify `diagnose_vfx_issues()` to first query knowledge base
   - If knowledge base has high-confidence suggestion (>0.6), use it
   - Fall back to heuristic modifications only when no knowledge available

3. **Cross-Session Knowledge Loading**
   - File: `agents/blender-orchestrator/orchestrator.py`
   - At session start, load relevant knowledge for effect type
   - Pre-populate known issues and solutions
   - Display in session initialization summary

### Success Criteria:
- Warnings checked before every parameter modification
- Knowledge base suggestions used when confidence > 0.6
- Past learnings applied to new sessions automatically

---

## Phase 4: External Research Integration (Priority: LOW)

**Problem:** System can't discover new techniques from external sources.

### Tasks:

1. **Add Research MCP Tool**
   - File: `agents/blender-manual/server.py`
   - New tool: `research_vfx_technique(query, sources)`
   - Sources: Blender docs, web search for tutorials
   - Return structured technique suggestions

2. **Implement Technique Import**
   - File: `agents/script-generator/technique_importer.py` (new)
   - Parse discovered technique descriptions into catalog format
   - Validate parameters against Blender 5.0 API ranges
   - Add as "experimental" techniques with exploration bonus

3. **Reference Image Collection**
   - Directory: `assets/reference_images/` structure
   - Organize by effect type and quality tier
   - Tag with metadata for ground truth evaluation

### Success Criteria:
- Can research new techniques from documentation
- Discovered techniques added to catalog for testing
- Reference images organized for evaluation

---

## Phase 5: Session Resumption Reliability (Priority: LOW)

**Problem:** Resuming interrupted sessions may fail due to state format issues.

### Tasks:

1. **State Schema Versioning**
   - File: `agents/iteration-controller/state_schema.py` (new)
   - Define versioned state schema with migration support
   - Validate state before load, migrate if needed
   - Log schema version in all saved states

2. **State Validation on Load**
   - File: `agents/iteration-controller/server.py`
   - Modify `load_iteration_state()` to validate all required fields
   - Check that referenced files (renders, scripts, VDBs) exist
   - Report what's missing and offer partial resume

3. **Periodic State Checkpoints**
   - File: `agents/blender-orchestrator/config.yaml`
   - Add auto-checkpoint after each successful stage
   - Keep last 3 checkpoints per session
   - Enable rollback to earlier checkpoint if needed

### Success Criteria:
- State schema versioned with migration support
- Missing files detected on resume
- Multiple checkpoints allow rollback

---

## Implementation Order

```
Week 1: Phase 1 (Orchestrator Reliability)
  - Day 1-2: MCP tool execution verification
  - Day 3-4: Workflow tracing
  - Day 5: Health checks

Week 2: Phase 2 (Technique Selection)
  - Day 1-2: Performance tracking table
  - Day 3-4: Exploration/exploitation strategy
  - Day 5: Recommendation tool

Week 3: Phase 3 (Knowledge Integration)
  - Day 1-2: Mandatory warning checks
  - Day 3-4: Suggestion-driven iteration
  - Day 5: Cross-session knowledge loading

Week 4: Phases 4-5 (Research + Resumption)
  - Day 1-2: Research tool
  - Day 3-4: State versioning and validation
  - Day 5: Testing and documentation
```

---

## Files to Modify/Create

### Modify:
- `agents/blender-orchestrator/orchestrator.py` - Phases 1, 3
- `agents/script-generator/server.py` - Phase 2
- `agents/iteration-controller/server.py` - Phases 3, 5
- `agents/experiment-tracker/database.py` - Phase 2
- `agents/blender-manual/server.py` - Phase 4
- `agents/blender-orchestrator/config.yaml` - Phase 5

### Create:
- `agents/blender-orchestrator/workflow_tracer.py` - Phase 1
- `agents/blender-orchestrator/health_check.py` - Phase 1
- `agents/script-generator/technique_selector.py` - Phase 2
- `agents/script-generator/technique_importer.py` - Phase 4
- `agents/iteration-controller/state_schema.py` - Phase 5

---

## Testing Strategy

1. **Unit Tests** - Each new module gets test file
2. **Integration Tests** - End-to-end workflow with mock MCP servers
3. **Regression Tests** - Ensure existing functionality not broken
4. **Manual Validation** - Run real asset generation after each phase

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Tool execution success rate | Unknown | >95% |
| Technique selection informed by history | 0% | 80% |
| Knowledge base consulted per iteration | ~0% | 100% |
| Session resume success rate | Unknown | >90% |
| Average iterations to pass quality | ~3-5 | <3 |

---

## Related Documents

- `docs/MULTI_AGENT_PIPELINE_ANALYSIS.md` - Full system analysis
- `.claude/skills/blender-orchestrator/SKILL.md` - Skill definition
- `agents/blender-orchestrator/config.yaml` - Configuration reference

---

## Notes

- Start with Phase 1 as it's foundational - other improvements meaningless if tools don't execute
- Phase 2 and 3 can be parallelized after Phase 1 complete
- Phases 4 and 5 are lower priority enhancements
- Consider adding telemetry/metrics collection for ongoing monitoring
