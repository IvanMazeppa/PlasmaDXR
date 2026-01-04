### Blender Librarian Agent (Blender 5.0.1 + Vision) — Design Position & Implementation Plan

**Document purpose**: Provide a cohesive, highly detailed position on how to build a “Blender Librarian” agent with **vision** capabilities that is **trained on Blender 5.0.1 documentation**, and can translate render/evaluation feedback into **actionable Blender changes** (shader nodes, render settings, simulation params, bpy API code edits) for your VFX VDB pipeline.

**Audience**: You (and other agent analyses you’ll collate) — intended as a practical engineering plan, not a marketing overview.

**Key constraint**: Limited budget (~$20 OpenAI). This pushes us toward **local retrieval + selective GPT-5.2 calls** only when needed.

---

### Executive summary (position)

My position is that the “best” Blender Librarian is **not** a single monolithic model “trained on Blender docs” in the classic fine-tune sense. Instead, the highest ROI approach is a **hybrid system**:

- **Local doc RAG (Blender manual + Python API)** for grounding and version accuracy
- **Vision-based diagnosis** (render vs reference) to identify *what visually differs*
- **A constrained “advice → patch” interface** that outputs *only* modifications your pipeline can safely apply (e.g., `script-generator.modify_script()` + limited code replacements)
- **A continuously-learning layer** fed by your own experiments (experiment-tracker) so it becomes project-specific over time

Fine-tuning (SFT/DPO) can help later, but it should be reserved for the **narrowest** parts (e.g., mapping from symptoms→actions / producing consistent JSON), after you’ve accumulated enough labeled examples from your pipeline.

This hybrid approach maximizes:
- **Correctness** (grounded in Blender 5.0.1 docs)
- **Cost efficiency** (docs are “free” locally; GPT-5.2 is used selectively)
- **Safety** (changes constrained; no runaway “creative rewriting” of scripts)
- **Iteration speed** (fast retrieval; minimal tokens; short feedback loops)

---

### Clarifying the phrase “trained on Blender 5.0.1 documentation”

This can mean three different things. Pick the one that matches your goals:

- **(A) Embedded / vectorized**: The documentation is chunked, embedded, and retrieved (RAG).  
  - **Pros**: Always accurate and up-to-date; cheap; easy to debug; easy to cite.  
  - **Cons**: The model must still “reason” using retrieved snippets.

- **(B) Fine-tuned**: You train a model on doc-derived examples (Q/A, tool-usage patterns, JSON outputs).  
  - **Pros**: More consistent outputs; better format compliance; less retrieval dependence.  
  - **Cons**: Requires data curation, eval harness, and can bake in errors; retraining needed when Blender changes.

- **(C) “Doc-aware at runtime”**: The model has no special training, but you give it doc excerpts every time.  
  - **Pros**: Simple.  
  - **Cons**: Costs tokens; brittle if you pass irrelevant snippets.

**Recommendation**: Start with **A + vision** (hybrid RAG+vision). Add **B** only after you have a sizeable corpus of *successful* internal experiments to fine-tune toward your pipeline’s “style of fixes.”

---

### What the Librarian must do (functional requirements)

At minimum, it must:

- **R1: Retrieve relevant Blender 5.0.1 docs** for the current situation, with *paths* and *titles* included.
- **R2: Diagnose the render mismatch** (current render vs reference) in a way that ties to Blender knobs.
- **R3: Propose concrete edits** that are safe and compatible with your pipeline:
  - simulation parameters (Mantaflow/domain/flow)
  - shader/node changes (Principled Volume, temperature→emission mapping)
  - render/color management settings (Filmic/AgX, exposure)
  - bpy API specifics for Blender 5.x
- **R4: Output structured, machine-actionable JSON** that downstream code can execute without guesswork.
- **R5: Provide confidence + alternatives** so the orchestrator can decide when to accept, ask approval, or stop.

---

### Non-goals (to keep it sane and affordable)

- **N1**: The librarian should not rewrite whole Blender scripts end-to-end every iteration.
- **N2**: It should not attempt to “solve Blender” with vague suggestions like “try increasing samples.”
- **N3**: It should not require constant OpenAI calls; most work should be local until escalation.

---

### Proposed architecture (hybrid RAG + vision + constrained patch outputs)

#### Components

- **Doc Retriever (local)**  
  Source: Blender 5.0.1 manual + Python API HTML.  
  Function: given a query, return top-k doc hits + minimal snippets + paths.  
  Implementation: your existing `blender-manual` MCP server is already this.

- **Vision Diagnoser (GPT-5.2 vision)**  
  Inputs: `render.png`, optional `reference.png`, optional ML metrics JSON.  
  Outputs: visual discrepancy list (brightness, hue, limb darkening, granulation texture, corona prominence), ranked by importance.

- **Advice Synthesizer (GPT-5.2)**  
  Inputs: doc hits + vision discrepancies + evaluator metrics + current parameters + pipeline constraints.  
  Outputs: a small, safe set of changes (≤ 2 params + ≤ 3 code replacements) with reasoning.

- **Patch Applicator (existing)**  
  Uses `script-generator.modify_script()` and validator.

- **Learning Recorder (existing)**  
  Uses experiment-tracker to log which librarian actions helped (or hurt).

#### Control flow (where librarian fits)

The librarian should be invoked only in escalation points:

- **Plateau**: no improvement for N iterations.
- **Repeated category failures**: e.g. persistent “WRONG COLOR” on `sun`.
- **Hard-to-translate metrics**: warm_ratio mismatch, limb darkening absence, blowouts.

This keeps token usage small and prevents “overfitting” to the librarian’s subjective advice.

---

### Why I recommend RAG+vision over a doc fine-tune (for your budget)

With ~$20:

- A full fine-tune is rarely cost-effective without:
  - a curated dataset (thousands+ high quality examples)
  - an eval harness to prevent regressions
  - iteration cycles that cost both time and money

RAG+vision provides:
- **Immediate gains**: better doc recall and better visual diagnosis now
- **Clear debugging**: you can inspect which docs were used
- **Cheaper iteration**: retrieval is local; GPT-5.2 used only at escalation

Fine-tuning becomes attractive later for:
- output format reliability (strict JSON)
- consistent “pipeline-native” terminology
- mapping from known failure patterns → recommended patches

---

### Data pipeline for “training on Blender docs” (RAG done right)

If you do RAG, do it with metadata so the model can cite and reason:

#### 1) Ingestion

- **Inputs**:
  - Blender Manual HTML pages
  - Blender Python API HTML pages
- **Normalization**:
  - strip nav/sidebars
  - preserve headings
  - preserve code blocks
  - extract stable page IDs/paths

#### 2) Chunking strategy (critical)

Avoid naive fixed-size chunks only. Prefer:

- Chunk by **heading sections** (H2/H3 blocks).
- Keep code blocks adjacent to the section they describe.
- Store:
  - `source`: manual / python_api
  - `path`: relative HTML path
  - `title`
  - `headers` (top 3-5)
  - `keywords` (e.g., “Principled Volume”, “AgX”, “color management”, “Mantaflow”)
  - `blender_version`: 5.0.1

#### 3) Retrieval

Use a **two-stage** approach if possible:

- Stage 1: BM25/keyword/heuristic scoring (fast)
- Stage 2: semantic re-rank (embedding similarity)

Your existing `blender-manual` index already supports this direction (keyword + optional semantic).

#### 4) Query templates (make retrieval reliable)

Instead of “volume looks wrong”, generate structured queries:

- “Principled Volume blackbody temperature attribute emission strength blowout”
- “color management Filmic vs AgX exposure volumetric”
- “Mantaflow fire temperature grid range exported to VDB”
- “bpy.types.ShaderNodeVolumePrincipled inputs blackbody_intensity”

The librarian can automatically emit these based on detected symptoms.

---

### Vision capability: how to make it practical (not vague)

Vision should not produce poetic descriptions; it should yield **measurable discrepancies** and **action hypotheses**.

#### Inputs

- Current render frame (PNG)
- Reference image (PNG), optional but recommended for “match ground truth”
- Your evaluator JSON (e.g., warm_ratio, brightness distribution, structural metrics)
- Optional: current script params and relevant extracted params from validation

#### Output schema (recommended)

The vision module should return:

- `discrepancies[]`: each with:
  - `type`: brightness | color_temperature | limb_darkening | texture_scale | corona_extent | prominence_coverage | noise_pattern | saturation | dynamic_range
  - `severity`: low/medium/high/critical
  - `evidence`: short, objective language
  - `likely_blender_causes[]`: e.g. exposure, blackbody intensity, density scaling, view transform

This becomes the bridge to doc retrieval + patching.

---

### “Advice → patch” interface: constrain outputs to keep it safe

Your pipeline can only reliably apply a limited set of script edits. Lean into that.

#### Safe modification contract (example)

Return only:
- `modifications`: keys compatible with your `script-generator.modify_script()`:
  - `resolution`, `frame_end`, `frame_start`
  - `turbulence` / `vorticity`
  - `temperature`
  - `domain_scale`
  - `custom_code` (string replacements)

If you need larger changes (shader node graph rewrites), represent them as **small “surgical” string replacements** in a well-known part of the script (templated insertion points are ideal).

#### Why this matters

Without constraints, a general model will:
- change too much at once
- break Blender context (operators poll failures)
- drift away from your pipeline conventions
- produce unrepeatable results

Small bounded changes + frequent evaluation is the winning strategy.

---

### Learning loop: how the librarian gets better over time

The librarian should “learn” in two ways:

#### 1) Retrieval learning (cheap)

Maintain a small table like:

- symptom → query expansions → best doc paths
- effect_type → preferred doc clusters

This can be simple JSON (versioned) and updated automatically after successful runs.

#### 2) Outcome learning (project-specific)

For each librarian recommendation, record:

- which discrepancies were targeted
- which modifications were applied
- quality deltas (VFX score, ground-truth similarity metrics)
- whether it helped or hurt

Later you can:
- train a small policy model (or even a rules layer) that predicts which fix to try first
- fine-tune a model to output better structured recommendations

---

### Recommended build phases (pragmatic roadmap)

#### Phase 0: Make docs actionable (no OpenAI spend)

- Ensure `blender-manual` semantic search is available (local embeddings)
- Create “symptom → query” templates
- Add a tiny “playbook” JSON for known solar issues (blackbody blowout, warm_ratio drift, limb darkening)

Exit criteria:
- Given a symptom string, the system consistently returns the right pages (Principled Volume, color management).

#### Phase 1: Add vision-assisted diagnosis (selective GPT-5.2 use)

- Vision call only on plateau or hard issues
- Output structured discrepancies
- Keep output short and actionable

Exit criteria:
- Vision identifies the correct failure class (e.g., blowout vs underexposure vs too-orange).

#### Phase 2: Add synthesis into safe patches

- Combine doc hits + discrepancies → `modifications` payload
- Apply ≤2 changes per iteration
- Always validate script post-edit

Exit criteria:
- Recommendations lead to positive score changes more often than random guessing.

#### Phase 3: Optional fine-tuning (only when you have data)

- Create dataset from experiment-tracker:
  - inputs: (issues, metrics, doc hits, params)
  - outputs: (best modifications + rationale)
- Fine-tune for JSON correctness and better fix ranking

Exit criteria:
- Reduced iterations-to-pass and fewer “wild” suggestions.

---

### Cost control (critical with $20)

To keep OpenAI usage inside budget:

- Only call GPT-5.2 when:
  - plateau (no improvement for N)
  - repeated same issue category
  - user explicitly requests deep diagnosis
- Cap output tokens.
- Pass compact doc hits (titles + paths) rather than full pages.
- Include only 1-2 images per call (render + reference).

---

### Licensing / compliance note (Blender manual)

Blender manual is **CC BY-SA 4.0**. A RAG index that stores excerpts/metadata typically remains compliant if:
- you preserve attribution (source + path)
- you do not redistribute the entire manual as proprietary content

If you ever publish model outputs trained from the docs, ensure your compliance story is clear (attribution, share-alike obligations where applicable).

---

### Concrete implementation blueprint (what I’d build)

#### A) “Blender Librarian” MCP server

Expose one tool:
- `advise_next_modifications(...) -> JSON`

Internally:
- Retrieve doc hits from `blender-manual`
- Optionally run vision diagnosis
- Produce constrained patch suggestions + doc paths used

#### B) Orchestrator integration

- On plateau / persistent issue:
  - call librarian
  - merge `modifications` into `script-generator.modify_script()`
  - validate → execute → evaluate

#### C) Template insertion points (recommended improvement)

Add explicit markers in generated scripts so code replacements are robust:

- `# <LIBRARIAN:SHADER_BLOCK_BEGIN>`
- `# <LIBRARIAN:SHADER_BLOCK_END>`

Then `custom_code` can target a stable block, rather than brittle string matching.

---

### What “success” looks like

For the sun case specifically:

- The librarian can say:
  - “This is blackbody blowout due to temperature attribute scale; fix by scaling temperature field before blackbody or using emission color instead.”
- It can cite:
  - Principled Volume docs and color management docs (paths)
- It proposes:
  - 1-2 precise edits (e.g., lower blackbody intensity, adjust view transform/exposure, clamp temperature mapping)
- Within 1-3 iterations, you see monotonic improvement in:
  - ground-truth similarity metrics
  - warm_ratio drifting toward target
  - no more viewport-vs-F12 surprises (or at least explained and mitigated)

---

### Closing recommendation

Build the librarian as a **tool-driven, doc-grounded, vision-assisted “patch suggester”**, not as a monolithic fine-tuned oracle.

Do RAG locally (fast + free), use vision sparingly (expensive but high leverage), and restrict outputs to safe patch contracts. Once you’ve accumulated enough successful examples in your experiment tracker, consider fine-tuning for format discipline and better fix-ranking.


