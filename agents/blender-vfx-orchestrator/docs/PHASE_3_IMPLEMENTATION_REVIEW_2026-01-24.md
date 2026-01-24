# Phase 3 Implementation Review (2026-01-24)

Goal: verify Phase 3 (structured outputs + turn budget alignment) and identify enforcement gaps or doc mismatches.

---

## Summary (Brutal Truth)
Phase 3 is **mostly correct**, and the two enforcement gaps were **fixed**:
- `doc_refs` are now validated via output guardrail.
- Research Agent hooks now align with the 4‑turn Phase 3 budget.

Remaining risk: Research “warnings” should be grounded in docs/patterns unless KB tools are added.

---

## Critical Findings

### 1) `doc_refs` is required in docs but not enforced
The schema uses a default empty list, and there is no guardrail or post‑check.

```97:109:agents/blender-vfx-orchestrator/orchestrator.py
class ResearchOutput(BaseModel):
    ...
    doc_refs: List[str] = Field(default_factory=list, description="Blender 5.0 documentation references used (URLs or section names)")
```

**Impact:** Research Agent can return empty `doc_refs` and still pass validation; downstream prompts treat it as acceptable.

**Fix (APPLIED 2026-01-24):** Output guardrail enforces `len(doc_refs) > 0`.

Docs affected:
- `AGENT_SPECIFICATION.md` (marks doc_refs REQUIRED)
- `SDK_ENFORCEMENT_PROTOCOL.md` (marks doc_refs REQUIRED)
- `PROMPT_SYSTEM_OPTIMIZATION_2026-01-23.md` (marks doc_refs REQUIRED)

---

### 2) Research hooks config does not match Phase 3 documentation
Phase 3 docs say Research Agent has 4 turns and tighter tool limits, but `create_research_hooks()` allows **10 turns** and 6 same‑tool calls.

```459:473:agents/blender-vfx-orchestrator/hooks/enforcement_hooks.py
config = EnforcementConfig(
    max_same_tool_calls=6,
    max_consecutive_same_tool=4,
    max_exempt_tool_calls=10,
    max_turns=10,
    hard_turn_limit=15,
    ...
)
```

**Impact:** Enforcement does not reflect the Phase 3 “tight budget” goal.

**Fix (APPLIED 2026-01-24):** `create_research_hooks()` aligned to 4 turns / tighter tool limits.

---

## High‑Priority Findings

### 3) Research warnings claim KB grounding without tool access
Research Agent instructions/docs list `warnings` “from knowledge base,” but Research Agent does **not** have KB tools.

**Impact:** `warnings` may become hallucinated or stale.

**Fix (APPLIED 2026-01-24):** Docs updated to “warnings from docs/patterns only” unless KB tool is added.

---

## Suggested Next Fixes (Minimal)
1. If you want KB-grounded warnings, add `query_knowledge_base` to Research Agent tools.

