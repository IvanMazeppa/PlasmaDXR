# Phase 2B-5: HITL Framework — What It Does and How to Use It

## What Was Added

Five checkpoint types that can pause the pipeline and ask a human what to do:

| Checkpoint | Triggers When | Purpose |
|-----------|---------------|---------|
| **Quality Plateau** | Score unchanged 3+ iterations | "Should we keep trying?" |
| **Stall Detection** | Same issue 3x at escape >= 2 | "Stuck on the same problem" |
| **Budget Warning** | >80% of monthly budget spent | "Running low on API credits" |
| **Critical Issue** | Score=0 for 2+ consecutive iterations | "Nothing is rendering" |
| **Escalation (L4)** | Escape velocity hits Level 4 | "System admits it's stuck" |

When a checkpoint fires, the human chooses one of:

| Decision | What Happens |
|----------|-------------|
| **Continue** | Keep going as-is |
| **Switch Technique** | Forces a technique switch (escape level 2) |
| **Adjust Parameters** | (Placeholder — currently same as continue; future: accept param hints) |
| **Abort** | Pauses session immediately |
| **Increase Budget** | Adds $5 to vision budget |

### Autonomy Levels

Not all checkpoints fire at all times. An autonomy level (0-4) gates which ones can trigger:

| Level | Name | What Fires |
|-------|------|-----------|
| 0 | Guided | Everything (plateau + stall + budget + critical + L4) |
| 1 | Supervised | Stall + budget + critical + L4 |
| 2 | Semi-autonomous | Critical + L4 only |
| 3 | Autonomous | L4 only |
| 4 | Full autonomous | Nothing — pipeline never pauses |

Each preset has a default autonomy level (e.g., `quick_test` = 3, `budget_saver` = 0).

---

## Why Not SDK `needs_approval`?

The OpenAI Agents SDK `needs_approval` feature only works with `HostedMCPTool` — tools exposed via MCP servers. Our tools are all Python `@function_tool` decorators running in-process. The SDK's `RunResult` has no `interruptions` field for in-process tools.

The Architecture doc already anticipated this: *"We implement HITL at the pipeline level, not the agent level."*

So the HITL framework is pure Python — it lives in the orchestrator's iteration loop, not inside any agent. The agents don't know about it. The pipeline checks conditions between phases and decides whether to pause.

---

## How It Works in Practice

### Two Modes

**Interactive mode** (`hitl_interactive: true`):
The pipeline calls `input()` and blocks. You see a prompt in your terminal:

```
============================================================
  HITL CHECKPOINT: BUDGET_WARNING
============================================================
  Reason: Budget >80% spent: $16.50/$20.00

  iteration: 3
  score: 42.0
  best_score: 42.0
  techniques_tried: ['fire_mantaflow', 'fire_shader']
  budget_spent: 16.5
  budget_remaining: 3.5

  Options:
    [c] Continue as-is
    [s] Switch technique
    [a] Adjust parameters
    [q] Abort session
    [i] Increase budget

  Decision [c/s/a/q/i]: _
```

You type a letter, press Enter, and the pipeline continues (or pauses if you chose abort).

**Non-interactive mode** (`hitl_interactive: false`):
The pipeline saves the checkpoint to the session JSON file and sets status to `PAUSED`. The session file (in `build/orchestrator_state/`) will have a `pending_checkpoint` field like:

```json
{
  "session_id": "fire_001_20260223",
  "status": "paused",
  "pending_checkpoint": {
    "checkpoint_type": "budget_warning",
    "reason": "Budget >80% spent: $16.50/$20.00",
    "context": { "iteration": 3, "score": 42.0, ... },
    "timestamp": "2026-02-23T14:30:00",
    "decision": null,
    "human_notes": null
  }
}
```

To resume, you edit the JSON (set `"decision": "continue"` or `"switch"` or `"abort"` etc.) and call:

```python
result = await resume_vfx_session("fire_001_20260223")
```

The pipeline reads the pending checkpoint, applies the decision, clears it, and continues.

### Where Checkpoints Fire (Pipeline Integration)

```
Pipeline Loop
│
├── [Resume] check_pending()     ← On session resume, if pending_checkpoint exists
│
├── [Phase 0-3] Normal pipeline  ← Research → Generate → Execute → Evaluate
│
├── [Phase 3.95] check_post_eval()  ← After quality evaluation, before Learning Agent
│     └── Checks: budget, stall, critical, plateau (in priority order)
│
├── [Phase 4-5] Learning + Quality Gate
│
└── [Escape L4] check_escalation()  ← Replaces the old bare PAUSED+break
```

### Running Tests With HITL

For **automated tests** (CI, E2E scripts): Use `quick_test` preset (autonomy=3) or set `hitl_enabled: false`. The pipeline won't pause unless it hits L4 escalation.

For **manual testing** where you want to see checkpoints: Use `budget_saver` preset (autonomy=0, interactive=true). Every possible checkpoint will fire and prompt you.

For **production batch runs**: Use `production` preset (autonomy=2, interactive=false). Only critical failures pause, and they save to JSON for later review.

### How Session Save/Resume Already Works

This is the existing mechanism — HITL plugs into it:

1. After every iteration, the pipeline calls `save_session(session)` → writes to `build/orchestrator_state/session_{id}.json`
2. `session.status = SessionStatus.PAUSED` breaks the loop
3. `resume_vfx_session(session_id)` loads the JSON, calls `create_asset_pipeline(request, resume_session_id=session_id)`
4. The pipeline detects `is_resuming=True`, restores state, and re-enters the loop

HITL adds `pending_checkpoint` to this flow — it's just one more field in the session JSON that gets checked on resume.

---

## What HITL Controls (and Could Control)

Currently HITL is **decision-based**: the human picks from 5 fixed choices. But the architecture supports much more:

### What It Controls Now

| Decision | Pipeline Effect |
|----------|----------------|
| Continue | No-op, loop continues |
| Switch Technique | Sets `escape_level = SWITCH_TECHNIQUE`, forces research + new script |
| Abort | Sets `status = PAUSED`, breaks loop |
| Increase Budget | Adds $5 to vision budget category |
| Adjust Params | (Stub — continues as-is, future: accept param dict) |

### What It Could Control (Future Extensions)

The `HITLCheckpoint.context` dict and `human_notes` field are generic — they can carry anything:

1. **Parameter injection**: Human provides specific parameter overrides (e.g., `{"flame_smoke": 3.5, "resolution_max": 128}`) in `human_notes` or a `params` key in the checkpoint. The pipeline applies them to the next script modification.

2. **Technique forcing**: Instead of just "switch", let the human specify *which* technique (e.g., `"switch to fire_shader_emission"`).

3. **Reference image swap**: Human provides a new reference image path mid-session.

4. **Quality threshold adjustment**: Lower the bar from 60 to 45 if the effect type is just inherently hard.

5. **Iteration budget override**: "Give it 3 more iterations" even if `max_iterations` was hit.

6. **Custom instructions**: Free-text guidance injected into the next agent prompt (e.g., "try adding a wind force field" or "the fire color should be bluer").

All of these just need the orchestrator to read additional fields from the checkpoint decision and route them to the right pipeline variable. The HITL handler itself doesn't need to change — it's the orchestrator's response to decisions that gets richer.

---

## GUI Layer: Feasibility Assessment

### Short Answer

Highly feasible. The system already has all the backend pieces — session state as JSON, budget tracking, config presets, iteration history with scores. A GUI would just be a read/write layer over these existing data structures.

### What Already Exists (Backend)

| Data | Format | Location |
|------|--------|----------|
| Session state | JSON (Pydantic) | `build/orchestrator_state/session_*.json` |
| Budget status | Python dict | `BudgetTracker.get_status()` |
| Config presets | YAML | `config/presets.yaml` |
| Iteration history | JSON array in session | `session.iterations[]` |
| Quality scores | Nested in iteration | `iteration.quality.overall_score` |
| Render images | PNG files | `sessions/artifacts/*/renders/` |
| HITL checkpoints | JSON in session | `session.pending_checkpoint`, `session.hitl_history` |
| Pipeline monitor alerts | Python objects | `PipelineMonitor.check_after_*()` |
| Experiment knowledge | SQLite | `experiments.db` |

### Architecture Options

**Option A: Local Web Dashboard (Recommended for your case)**

A lightweight Python web app (FastAPI + HTMX or Streamlit) running on localhost alongside the orchestrator.

```
Browser (localhost:8080)
    │
    ├── Dashboard: Live session status, score chart, render thumbnails
    ├── HITL: Pending checkpoints → click to decide
    ├── Config: Edit presets, switch active preset
    ├── Sessions: Browse history, resume paused sessions
    └── Budget: Spending breakdown, category limits
    │
FastAPI Server
    │
    ├── Reads session JSONs from disk
    ├── Reads/writes presets.yaml
    ├── Calls BudgetTracker.get_status()
    ├── Writes decisions to pending_checkpoint
    └── Calls resume_vfx_session() via async endpoint
```

**Why this fits:**
- You're already running on WSL2 — a localhost web server is trivial
- No external dependencies, no auth needed
- HTMX gives you live updates without a JS framework
- FastAPI is already in your Python ecosystem
- Streamlit is even simpler if you don't care about custom UI

**Effort estimate:** ~500-800 lines for a functional MVP with:
- Session list + status overview
- Score progression chart (matplotlib or plotly)
- Render image gallery
- HITL checkpoint approval buttons
- Preset selector
- Budget display

**Option B: Terminal UI (Rich/Textual)**

A TUI using Python's `textual` library — runs in the same terminal as the orchestrator.

- Lighter than a web app
- No browser needed
- But harder to show images (render thumbnails)
- Less intuitive for config editing

**Option C: Desktop App (PyQt/Tkinter)**

Overkill for now. More effort, fewer benefits over a web dashboard on localhost.

### How HITL Would Work With a GUI

The non-interactive mode was designed exactly for this:

1. Pipeline hits a checkpoint → saves to `session.pending_checkpoint` → sets `PAUSED`
2. GUI polls session files (or gets a WebSocket push) → shows checkpoint card with context
3. Human clicks a decision button in the GUI
4. GUI writes decision to the session JSON's `pending_checkpoint.decision` field
5. GUI calls the resume endpoint → pipeline picks up where it left off

The interactive CLI `input()` mode becomes unnecessary — the GUI replaces it entirely. You'd run with `hitl_interactive: false` always and let the GUI handle the human interaction.

### Beyond HITL: Full Control System

A GUI naturally expands to cover everything:

| Feature | Backend Already Exists? | GUI Shows |
|---------|------------------------|-----------|
| Session monitoring | Yes (session JSONs) | Live status, score chart, ETA |
| HITL approvals | Yes (pending_checkpoint) | Checkpoint cards with action buttons |
| Render preview | Yes (PNG files on disk) | Image gallery, side-by-side comparison |
| Config management | Yes (presets.yaml) | Dropdown selector, inline editing |
| Budget tracking | Yes (BudgetTracker) | Pie chart, category breakdown, alerts |
| Experiment history | Yes (experiments.db) | Table of past runs, searchable |
| Pipeline monitor | Yes (PipelineMonitor) | Alert feed, oscillation warnings |
| Session resume | Yes (resume_vfx_session) | "Resume" button on paused sessions |
| Technique explorer | Partial (KB) | Browse techniques, success rates |
| Knowledge base | Yes (experiment tracker) | Search patterns, view trust levels |

The orchestrator already writes all this data — a GUI just needs to read it and present it. The hard part (the pipeline logic, state management, agent orchestration) is done.

### Recommended Path

1. **Now:** Use HITL as-is (CLI interactive mode for dev, non-interactive + JSON for batch)
2. **When you want a GUI:** Start with FastAPI + HTMX, ~2-3 days for MVP
3. **MVP scope:** Session list, score chart, HITL buttons, preset selector, budget display
4. **Later:** Add WebSocket for live updates, render gallery, technique browser

The system is already "GUI-ready" — it just needs the presentation layer.
