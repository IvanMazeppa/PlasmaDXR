# Current State Audit and Recommendation - 2026-02-17

## Scope

This is a top-to-bottom audit of the current Blender VFX orchestrator state, plus an honest recommendation on whether to revise incrementally vs make wide-scale changes.

Audit basis:
- Direct code inspection in `agents/blender-vfx-orchestrator/*`
- Runtime/doc consistency checks
- Trace/test posture review
- External research from official docs and primary papers

---

## Honest Bottom Line

You are not imagining the regression.  
The architecture currently contains **real re-entry paths** for deprecated/hallucination-prone behavior.

My recommendation:
- Do a **wide-scale architectural cleanup** now.
- Do **not** throw everything away.
- Keep proven components; aggressively remove contradictory/legacy paths.

This is the highest-ROI route to "fix once and for all."

---

## Verified Findings

## Critical Findings (P0/P1)

1. Legacy fallback paths still active in runtime orchestration.
- Spec-first failure path falls back to original Script Writer:
  - `orchestrator.py:2448`
  - `orchestrator.py:2450`
  - `orchestrator.py:2452`
- Modification flows also fallback with explicit hallucination warning:
  - `orchestrator.py:2837`
  - `orchestrator.py:2995`

Impact:
- Even with spec-first enabled, runtime can still route into the exact behavior class you are trying to eliminate.

2. Execution truth contradiction in pipeline status handling.
- Comment says do not flip success for non-zero exit:
  - `orchestrator.py:3281`
- Code then forces `execution.success = True`:
  - `orchestrator.py:3293`

Impact:
- Learning/eval signals can be contaminated.
- Root-cause diagnosis becomes unreliable.

3. Potential serialization mismatch for execution time.
- `BlenderExecution` assignment uses fallback `execution.execution_time` lookup:
  - `orchestrator.py:3825`
- Primary field elsewhere is `execution_time_seconds`:
  - `orchestrator.py:169`

Impact:
- Missing/incorrect timing metrics in downstream artifacts.

---

## High Findings (P1/P2)

4. Guardrail coverage is narrower than intended.
- `validate_code_against_spec` explicitly reduced to deprecated blacklist checks only:
  - `api_spec_guardrails.py:430`
  - `api_spec_guardrails.py:472`

Impact:
- Unknown/hallucinated attributes can still pass if not on blacklist.

5. Research grounding is intentionally relaxed (S1-2).
- Empty/invalid `doc_refs` allowed if `api_modules` appears grounded:
  - `research_guardrails.py:120`
  - `research_guardrails.py:128`
  - `research_guardrails.py:148`

Impact:
- Better resilience, but weaker hard guarantees against bad grounding artifacts.

6. API fixer validation currently fail-open on unknown signatures.
- Vector-store signature miss => continue without blocking:
  - `blender_api_fixer.py:1681`
  - `blender_api_fixer.py:1684`

Impact:
- Hallucinated or drifted API calls can survive pre-exec checks.

7. Docs/runtime version truth drift exists.
- Runtime pins `openai-agents==0.9.0`:
  - `requirements.txt:11`
- Key docs still claim `0.8.3`:
  - `RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md:9`
  - `AI_OPERATION_MANUAL.md:12`
  - `SDK_ENFORCEMENT_PROTOCOL.md:7`

Impact:
- Human + agent operators can follow outdated constraints.

---

## Medium Findings (P2/P3)

8. Observability summaries are low-signal for many runs.
- Many trace summaries report `status: unknown` and `quality_score: null`:
  - e.g. `traces/p0_verification_20260202_005842_summary.json:8`
  - e.g. `traces/p0_verification_20260202_005842_summary.json:9`
- Analyzer defaults these fields and does not robustly derive terminal values:
  - `tools/trace_analyzer.py:71`
  - `tools/trace_analyzer.py:72`
  - `tools/trace_analyzer.py:223`

Impact:
- Hard to prove improvement or detect regressions quickly.

9. Test inventory is broad but heterogeneous.
- 37 local test files (excluding venv) include mixed true tests and benchmark/integration runners.

Impact:
- Some test names imply validation, but may not enforce strict regression gates.

---

## What Is Working and Should Be Preserved

1. Pipeline entrypoint is already routed to `create_asset_pipeline()` in server path.
- `server.py:183`
- `server.py:186`

2. Pre-execution API fixer is automatically applied.
- `blender_executor_tools.py:512`
- `blender_executor_tools.py:517`

3. Enforcement hooks concept is solid and should be retained (loop limits, doc-query requirements, turn budgets).
- `hooks/enforcement_hooks.py:7`
- `hooks/enforcement_hooks.py:8`
- `hooks/enforcement_hooks.py:9`

---

## External Research Synthesis

The following external guidance supports a structural cleanup approach:

1. Structured outputs and strict schemas should be used as hard contracts, not optional hints.
- OpenAI Structured Outputs announcement: strict schema adherence and reliability claims.  
  https://openai.com/index/introducing-structured-outputs/
- OpenAI Responses OpenAPI spec confirms strict schema controls in request/response/tool shapes.  
  https://api.openai.com/v1/responses

2. Eval-driven development is mandatory for reliability.
- OpenAI evaluation best practices explicitly call out:
  - "Adopt eval-driven development"
  - avoid "vibe-based evals"
  - calibrate with human feedback  
  https://platform.openai.com/docs/guides/evaluation-best-practices

3. Agent systems should start simple, instrument deeply, and scale complexity only when justified.
- OpenAI practical guide to building agents.  
  https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

4. Reliable benchmark methodology matters; naive benchmark chasing can mislead.
- SWE-bench Verified paper/launch context for stricter real-world evaluation assumptions.  
  https://openai.com/index/introducing-swe-bench-verified/

5. Safe autonomy requires explicit user control and tool boundaries.
- MCP spec security principles and user consent/control requirements.  
  https://modelcontextprotocol.io/specification/2025-06-18#user-consent-and-control

6. Self-improvement should be iterative and externally validated, not blindly self-applied.
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651
- Voyager: https://arxiv.org/abs/2305.16291
- Toolformer: https://arxiv.org/abs/2302.04761

7. Governance/security frameworks are relevant once autonomy expands.
- NIST AI RMF 1.0: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
- OWASP LLM Top 10 (2025): https://genai.owasp.org/llm-top-10/

---

## Recommendation: How to Proceed

## Opinion on Wide-Scale Changes

You should proceed with wide-scale changes now, but targeted:
- **Yes** to re-architecting safety-critical paths.
- **No** to discarding all existing work.
- Treat sunk-cost components as disposable if they violate runtime truth.

## Priority Order

1. Remove or hard-gate all legacy fallback paths.
2. Fix run-state truth semantics (separate process truth from artifact viability).
3. Introduce allowlist-based API truth validation (not blacklist-only).
4. Enforce eval gating in CI before every merge.
5. Repair observability so improvements/regressions are measurable.
6. Only then expand autonomous self-learning.

## What "Autonomous/Self-Improving" Should Mean Here

Not:
- model can rewrite itself in production without proof.

Yes:
- model proposes changes,
- changes run in sandbox evals,
- promotion requires measured gain and no critical regressions,
- rollback is automatic on regression.

---

## Concrete Next Moves (This Week)

1. Delete or hard-disable Script Writer fallback paths in `orchestrator.py`.
2. Patch execution success semantics and re-run pipeline truth tests.
3. Add strict API-allowlist guardrail for top 3 effect classes.
4. Align all version-truth docs with actual `requirements.txt`.
5. Patch trace analyzer to derive terminal status/score from event stream.
6. Define first golden eval suite and make it a merge blocker.

---

## Final Honest Assessment

You are at the right decision point.  
If you keep layering fixes on the current contradictory paths, you will keep rediscovering the same classes of failures.

If you apply the proposed selective reset now, this project can become stable enough for credible autonomy.

