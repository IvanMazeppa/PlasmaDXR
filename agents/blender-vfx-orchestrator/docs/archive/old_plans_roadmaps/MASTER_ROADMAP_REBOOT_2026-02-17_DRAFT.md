# Master Roadmap Reboot (Draft) - 2026-02-17

## Executive Position

Yes, a deep reset is warranted.

My recommendation is **not** a total rewrite. It is a **selective architectural reset**:
- Keep proven assets (pipeline entrypoint, artifact gates, Blender API fixer rules that are validated, executor scaffolding).
- Remove or hard-disable paths that reintroduce deprecated/hallucinated behavior.
- Rebuild truth/validation/eval loops as first-class infrastructure.

This avoids sunk-cost paralysis while preserving the parts that already work.

---

## Why This Reboot Now

The current system still has contradictory runtime paths that can bypass the intended spec-first guarantees:
- Spec-first failure falls back to legacy Script Writer in runtime code (`orchestrator.py:2448`, `orchestrator.py:2450`, `orchestrator.py:2452`).
- Additional modification paths still fallback to Script Writer and explicitly note hallucination risk (`orchestrator.py:2837`, `orchestrator.py:2995`).
- Execution status logic is internally contradictory (comment says preserve failed state, code forces success to continue evaluation) (`orchestrator.py:3281`, `orchestrator.py:3293`).
- Session serialization has a likely field mismatch risk around execution time (`orchestrator.py:3825`).

These are exactly the class of regressions you are reporting.

---

## External Grounding (Research Highlights)

The plan below aligns with current public guidance and primary research:
- OpenAI Structured Outputs: strict schema adherence for structured responses and tool args (`strict: true`), with JSON-schema-first control surfaces.  
  Source: https://openai.com/index/introducing-structured-outputs/
- OpenAI API OpenAPI spec (v2.3.0) shows strict schema controls in `responses` and function tooling defaults/semantics.  
  Source: https://api.openai.com/v1/responses
- OpenAI eval guidance: eval-driven development, task-specific evals, avoid "vibe-based evals", calibrate with humans.  
  Source: https://platform.openai.com/docs/guides/evaluation-best-practices
- OpenAI "A Practical Guide to Building Agents": begin simple, instrument heavily, add complexity only when needed.  
  Source: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- SWE-bench Verified work for reliable agent benchmarking methodology.  
  Source: https://openai.com/index/introducing-swe-bench-verified/
- MCP security model: explicit user consent/control, data privacy, tool safety boundaries.  
  Source: https://modelcontextprotocol.io/specification/2025-06-18#user-consent-and-control
- Core agent research:
  - ReAct (reason+act trajectories): https://arxiv.org/abs/2210.03629
  - Reflexion (verbal reinforcement): https://arxiv.org/abs/2303.11366
  - Self-Refine (iterative self-feedback): https://arxiv.org/abs/2303.17651
  - Voyager (continual skill library): https://arxiv.org/abs/2305.16291
  - Toolformer (self-supervised tool use): https://arxiv.org/abs/2302.04761
- Governance/security baselines:
  - NIST AI RMF 1.0: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
  - OWASP LLM Top 10 (2025): https://genai.owasp.org/llm-top-10/

---

## North-Star Outcomes

By end of reboot:
1. No deprecated or hallucinated Blender API use reaches execution.
2. No hidden fallback path can bypass spec-first constraints.
3. Every production change must pass an eval harness before merge.
4. Self-improvement is evidence-based (accepted only with measured gains and no regression).
5. The system can run autonomously with bounded risk and transparent traceability.

---

## Workstreams

## WS1 - Runtime Contract Lockdown (Week 1-2)

Goal: make invalid behavior structurally impossible.

Actions:
- Hard-disable all legacy Script Writer fallbacks behind a kill switch defaulting OFF in production.
- Enforce a single state machine path: `PLAN -> GENERATE -> VALIDATE -> EXECUTE -> EVALUATE -> DECIDE`.
- Replace contradictory execution status handling with dual-channel outcome:
  - `process_success` (exit code truth)
  - `artifact_viable` (render produced)
  without mutating truth flags.
- Fix model serialization mismatches (`execution_time_seconds` consistency end-to-end).

Exit criteria:
- Zero runtime path to deprecated handoff/legacy writer unless explicitly enabled for debug.
- Deterministic pass/fail semantics in logs and artifacts.

---

## WS2 - API Truth Layer (Week 2-4)

Goal: stop deprecated/hallucinated attributes at generation-time and pre-exec time.

Actions:
- Promote current blacklist guardrail to **allowlist + schema** checking for effect-specific APIs.
- Build a versioned "Blender API Truth Pack":
  - canonical attributes/ops by effect,
  - deprecated map,
  - enum constraints,
  - required object-state preconditions.
- Change vector-store validation from fail-open to tiered enforcement:
  - unknown API => block in strict mode, quarantine in debug mode.
- Add static script lints before execution:
  - invalid `bpy.types` attr usage
  - unknown `bpy.ops` args
  - missing mandatory preconditions (e.g., active object/mode for bake ops).

Exit criteria:
- "Deprecated attribute escaped to runtime" = 0 in continuous test runs.

---

## WS3 - Eval-First Engineering System (Week 2-5)

Goal: no more benchmark thrash without signal.

Actions:
- Create versioned eval suites:
  - `api_compliance_eval`
  - `execution_stability_eval`
  - `visual_quality_eval`
  - `cost_latency_eval`
- Build golden set (at least 50 representative tasks) with fixed seeds/configs.
- Add merge gate: PR fails if any key metric regresses beyond tolerance.
- Add nightly canary matrix across key effect families.

Exit criteria:
- Every roadmap claim backed by reproducible eval deltas.

---

## WS4 - Observability and Incident Forensics (Week 3-5)

Goal: every failure is diagnosable in one pass.

Actions:
- Fix trace summarizer to ingest terminal status, quality, iteration counts from actual events.
- Standardize run metadata schema across orchestrator, traces, artifacts.
- Add dashboard-ready rollups:
  - fallback usage,
  - hallucination catches,
  - gate failures by phase,
  - quality trend by branch/model/prompt version.
- Introduce incident template and "5 whys" postmortem automation.

Exit criteria:
- No more `status=unknown` / `quality_score=null` for completed runs.

---

## WS5 - Memory and Self-Improvement, Safely (Week 5-8)

Goal: autonomy that improves without destabilizing baseline quality.

Actions:
- Split memory into two classes:
  - `verified_rules` (human- or eval-validated; deployable)
  - `candidate_hypotheses` (experimental only).
- Add proposal protocol for self-modification:
  - hypothesis
  - expected impact
  - risk tags
  - automatic rollback condition
- Batch-test self-generated changes in sandbox evals only.
- Promote to `verified_rules` only after statistically significant improvement.

Exit criteria:
- Self-learning updates have audit trails and measured impact.
- Automatic rollback triggers on regression.

---

## WS6 - Claude Opus + Codex Collaboration Protocol (Week 1 onward)

Goal: reduce model-specific blind spots.

Execution model:
- Claude Opus owns:
  - adversarial critique,
  - failure mode hypothesis generation,
  - prompt/instruction conflict analysis.
- Codex owns:
  - deterministic implementation,
  - test harnesses and CI gates,
  - runtime instrumentation, enforcement wiring.

Required workflow per major change:
1. Claude proposes change + risk analysis.
2. Codex implements minimal patch + tests.
3. Claude reviews failure modes post-implementation.
4. Codex merges only if eval gates pass.

---

## 30/60/90 Day Milestones

### Day 30
- Legacy fallbacks disabled in production mode.
- API truth pack v1 enforced.
- Eval harness integrated into CI.

### Day 60
- Observability overhaul complete.
- Self-improvement loop running in sandbox mode only.
- Stable quality trend across nightly matrix.

### Day 90
- Controlled autonomous improvements promoted via evidence gate.
- Incident rate and benchmark regression rate materially reduced.

---

## Immediate 7-Day Action List

1. Remove/guard all legacy Script Writer fallback paths.
2. Resolve execution status truth contradiction.
3. Fix execution-time serialization mismatch.
4. Align all docs/version-truth files to actual runtime pin.
5. Implement minimal API allowlist checker for highest-frequency effects.
6. Add CI job: fail on deprecated Blender attribute patterns.
7. Patch trace analyzer to emit truthful terminal metrics.

---

## Decision Policy (What We Keep vs Replace)

Keep:
- `create_asset_pipeline` architecture and server entrypoint flow.
- Artifact discovery/gating framework.
- Proven Blender fixer patches that are evidence-backed.

Replace or heavily refactor:
- Any path that can route to legacy/hallucination-prone generation.
- Fail-open API validation behavior.
- Non-deterministic or contradictory run-state semantics.
- Benchmark scripts without strict pass/fail criteria.

---

## Final Note

This roadmap intentionally favors **structural guarantees over prompt-only fixes**.  
Prompting can improve behavior; only architecture can eliminate entire classes of failure.

