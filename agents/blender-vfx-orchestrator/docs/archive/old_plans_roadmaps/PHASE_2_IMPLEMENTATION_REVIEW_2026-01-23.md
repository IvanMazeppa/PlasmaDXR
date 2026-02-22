# Phase 2 Implementation Review (2026-01-23)

Goal: review Phase 2 (self‑learning reuse + pattern outcomes) with a focus on mechanical correctness and actionable learning signals.

---

## Summary (Brutal Truth)
Phase 2 is **wired**, and two critical mechanical bugs were found and **fixed**:
1) Pattern outcomes were computed after `previous_score` was updated (delta ≈ 0).
2) Extracted patterns were overwriting the applied‑pattern ID (misattributed outcomes).

Remaining risk: **pattern parameter parsing is too naive** for real Blender scripts.

---

## Critical Findings

### 1) Pattern improvement is always ~0
`previous_score` is overwritten **before** outcome reporting, so improvement becomes `0.0` and patterns never get credited.

```2136:2138:agents/blender-vfx-orchestrator/orchestrator.py
                    previous_score = quality.overall_score
                    print(f"[Pipeline] Score: {quality.overall_score:.1f} | Passed: {quality.passed}", file=sys.stderr)
```

Outcome reporting uses `previous_score` after it has already been updated:
```2281:2292:agents/blender-vfx-orchestrator/orchestrator.py
                    if hasattr(context, 'last_applied_pattern_id') and context.last_applied_pattern_id:
                        try:
                            score_improvement = quality.overall_score - previous_score if quality else 0
                            pattern_success = score_improvement > 0
                            outcome_result = report_pattern_outcome(
                                pattern_id=context.last_applied_pattern_id,
                                success=pattern_success,
                                improvement=score_improvement,  # Include actual improvement value
                                notes=f"Issue: {quality.primary_issue or 'unknown'}, Score: {quality.overall_score:.1f}"
                            )
```

**Fix (APPLIED 2026-01-23):** Use `baseline_score_snapshot` for improvement.

---

### 2) Applied vs extracted patterns are conflated
`context.last_applied_pattern_id` is overwritten when a *new* pattern is extracted, even if it wasn’t applied yet.

```2265:2279:agents/blender-vfx-orchestrator/orchestrator.py
                    if learning.pattern_extracted and learning.pattern_id:
                        print(f"[Pipeline] Pattern extracted: {learning.pattern_id}", file=sys.stderr)
                        # Track patterns in session for future use
                        if not hasattr(session, 'extracted_patterns'):
                            session.extracted_patterns = []
                        session.extracted_patterns.append({
                            "pattern_id": learning.pattern_id,
                            "iteration": iteration,
                            "score_delta": quality.overall_score - previous_score if quality else 0,
                        })

                        # Store pattern_id to report outcome after next execution
                        context.last_applied_pattern_id = learning.pattern_id
```

**Impact:** A newly extracted pattern can be reported as “applied” and scored immediately, which corrupts the pattern library.

**Fix (APPLIED 2026-01-23):** Extracted patterns no longer overwrite `last_applied_pattern_id`.

Optional hardening:
- Add `last_extracted_pattern_id` if you want explicit audit separation.

---

## High‑Priority Findings

### 3) Pattern parameter parsing is too naive
The parser only captures `(\w+)` assignments; it will fail on nested attributes and indexed fields (common in Blender scripts).

```1617:1639:agents/blender-vfx-orchestrator/orchestrator.py
                                for line in pattern_to_apply.code_snippet.strip().split('\n'):
                                    line = line.strip()
                                    if '=' in line and not line.startswith('#'):
                                        # Extract parameter name and value
                                        # Handle patterns like "domain.param = value" or "obj.param = value"
                                        match = re.match(r'(?:\w+\.)?(\w+)\s*=\s*(.+)', line)
                                        if match:
                                            param_name = match.group(1)
                                            value_str = match.group(2).strip()
                                            # Try to parse the value
```

**Impact:** Patterns that use `node.inputs["Density"]`, tuples, vectors, or nested attributes will parse incorrectly and apply wrong changes.

**Fix:** Either:
- Require patterns to store structured `parameter_changes` metadata, or
- Parse with a small AST-based extractor instead of regex.

---

## Suggested Next Fixes (Minimal)
1. Store structured parameter changes inside pattern objects (or parse with AST).
2. Optional: add `last_extracted_pattern_id` for explicit audit trails.

---

## Phase 3 Implementation Complete (2026-01-24)

Phase 3 has been implemented with the following changes:

### ResearchOutput Structured Schema
- Added `doc_refs: List[str]` field (required) for Blender 5.0 documentation references
- Research Agent now uses `output_type=AgentOutputSchema(ResearchOutput, strict_json_schema=False)`
- Pipeline uses structured output directly instead of regex parsing for alternatives

### Turn Budget Alignment
- Prompt turn budgets now show "Target: X | Hard limit: Y"
- Research Agent: Target 4, Limit 4 (aligned)
- Script Writer: Target 5, Limit 15 (allows retries - documented)
- Quality Analyst: Target 3, Limit 6 (allows retries - documented)
- Learning Agent: Target 3-4, Limit 8 (allows pattern extraction - documented)

### Files Modified
- `orchestrator.py`: ResearchOutput schema, research agent output_type, pipeline structured output handling
- `tools/dynamic_instructions.py`: Turn budget documentation alignment
- `docs/AGENT_SPECIFICATION.md`: Added Research Agent spec with output schema
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`: Added Phase 3 section
- `docs/PROMPT_SYSTEM_OPTIMIZATION_2026-01-23.md`: Phase 3 status table

