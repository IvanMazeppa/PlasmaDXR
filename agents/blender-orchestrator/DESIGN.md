# Blender Orchestrator Agent Design

## Overview

**blender-orchestrator** is an autonomous Claude Agent SDK agent that orchestrates the Blender VFX asset generation pipeline with adaptive autonomy and token guardrails.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    BlenderOrchestratorAgent                                │
│                    (Claude Agent SDK - Opus 4.5)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Trust Score   │    │  Autonomy       │    │  Token Guard    │        │
│  │   (0.0 - 1.0)   │◄──►│  Controller     │◄──►│  (limits/costs) │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                     │                      │                   │
│           └─────────────────────┴──────────────────────┘                   │
│                                 │                                          │
│                    ┌────────────▼────────────┐                            │
│                    │   Decision Engine       │                            │
│                    │   (autonomous/ask/deny) │                            │
│                    └────────────┬────────────┘                            │
│                                 │                                          │
│         ┌───────────────────────┼───────────────────────┐                 │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐       ┌─────────────┐        ┌─────────────┐            │
│  │  Workflow   │       │  MCP Tool   │        │  Session    │            │
│  │  Manager    │◄─────►│  Router     │◄──────►│  State      │            │
│  │             │       │             │        │             │            │
│  └─────────────┘       └─────────────┘        └─────────────┘            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        │                           │                            │
        ▼                           ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│script-generator│          │blender-executor│          │asset-evaluator │
└───────────────┘          └───────────────┘          └───────────────┘
        │                           │                            │
        ▼                           ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│blender-manual │          │iteration-ctrl │          │experiment-     │
└───────────────┘          └───────────────┘          │  tracker       │
                                                      └───────────────┘
```

## Adaptive Autonomy System

### Trust Score (0.0 - 1.0)

The agent's trust score determines how much autonomy it has:

| Trust Score | Autonomy Level | Behavior |
|-------------|----------------|----------|
| 0.0 - 0.3   | SUPERVISED     | Ask before every major step |
| 0.3 - 0.6   | GUIDED         | Ask before technique changes and session end |
| 0.6 - 0.8   | AUTONOMOUS     | Ask only for session end and error recovery |
| 0.8 - 1.0   | TRUSTED        | Fully autonomous, only notify on completion |

### Trust Score Adjustments

**Positive Events (increase trust):**
- Successful iteration (quality improved): +0.05
- Quality threshold met: +0.10
- Asset completed successfully: +0.15
- Human approval given: +0.02

**Negative Events (decrease trust):**
- Quality degraded: -0.05
- Blender execution error: -0.10
- Technique change required: -0.03
- Token limit exceeded: -0.15
- Human override/correction: -0.10
- Critical failure: -0.20

### Manual Override

Config file allows immediate autonomy level override:
```yaml
autonomy:
  override: "supervised"  # Force specific level regardless of trust score
  # Options: supervised, guided, autonomous, trusted, null (use trust score)
```

## Token Guardrails

### Per-Session Limits

```yaml
guardrails:
  token_limits:
    per_session: 100000      # Max tokens per asset generation session
    per_iteration: 20000     # Max tokens per iteration cycle
    per_tool_call: 5000      # Max tokens per MCP tool call

  cost_limits:
    per_session_usd: 5.00    # Max cost per session
    per_day_usd: 20.00       # Max daily cost
    warning_threshold: 0.80   # Warn at 80% of limit
```

### Token Tracking

The orchestrator tracks:
1. **Input tokens** - Prompts sent to Claude
2. **Output tokens** - Responses from Claude
3. **Estimated cost** - Based on Opus pricing ($15/1M input, $75/1M output)

### Limit Behaviors

When approaching limits:
- **80% warning**: Log warning, continue with caution
- **95% pause**: Ask human for approval to continue
- **100% hard stop**: End session gracefully, save state

## Workflow Stages

### 1. SESSION_START
- Initialize experiment-tracker session
- Record baseline parameters
- Set up logging

### 2. GENERATE_SCRIPT
- Query blender-manual for API validation
- Use script-generator with technique catalog
- Validate parameters against documented ranges

### 3. EXECUTE_BLENDER
- Run blender-executor with timeout
- Parse errors if execution fails
- On failure: consult blender-manual, fix, retry (max 3)

### 4. EVALUATE_QUALITY
- Use asset-evaluator VFX quality tools
- Check temporal consistency if animation
- Diagnose issues

### 5. DECIDE_NEXT_ACTION
This is where autonomy level matters:

| Decision | SUPERVISED | GUIDED | AUTONOMOUS | TRUSTED |
|----------|-----------|--------|------------|---------|
| Iterate (improve) | ASK | AUTO | AUTO | AUTO |
| Change params | ASK | ASK | AUTO | AUTO |
| Change technique | ASK | ASK | ASK | AUTO |
| End session (pass) | ASK | ASK | ASK | NOTIFY |
| End session (fail) | ASK | ASK | ASK | ASK |

### 6. RECORD_LEARNING
- Use experiment-tracker to record outcome
- Update knowledge base
- Adjust trust score

### 7. SESSION_END
- Save final state
- Export successful assets
- Generate session report

## Configuration File

```yaml
# agents/blender-orchestrator/config.yaml

orchestrator:
  name: "blender-vfx-orchestrator"
  version: "0.1.0"

autonomy:
  initial_trust_score: 0.2       # Start supervised
  override: null                  # null = use trust score, or "supervised"/"guided"/"autonomous"/"trusted"

  trust_adjustments:
    success: 0.05
    quality_pass: 0.10
    asset_complete: 0.15
    approval_given: 0.02
    quality_degrade: -0.05
    execution_error: -0.10
    technique_change: -0.03
    token_exceeded: -0.15
    human_override: -0.10
    critical_failure: -0.20

guardrails:
  token_limits:
    per_session: 100000
    per_iteration: 20000
    per_tool_call: 5000

  cost_limits:
    per_session_usd: 5.00
    per_day_usd: 20.00
    warning_threshold: 0.80

  max_iterations: 10             # Hard limit on iterations per asset
  max_retries: 3                 # Max retries on execution failure
  execution_timeout: 900         # 15 min max per Blender run

quality:
  vfx_pass_threshold: 70         # VFX quality score to pass
  vfx_good_threshold: 80         # Score considered "good"
  plateau_threshold: 5           # Score change < 5 for 3 iterations = plateau
  temporal_threshold: 0.70       # Temporal consistency threshold

workflow:
  validate_api: true             # Query blender-manual before generation
  use_knowledge_base: true       # Check experiment-tracker before params
  record_learnings: true         # Save outcomes to knowledge base

mcp_servers:
  script_generator:
    command: "./run_server.sh"
    cwd: "agents/script-generator"
  blender_executor:
    command: "./run_server.sh"
    cwd: "agents/blender-executor"
  asset_evaluator:
    command: "./run_server.sh"
    cwd: "agents/asset-evaluator"
  experiment_tracker:
    command: "./run_server.sh"
    cwd: "agents/experiment-tracker"
  blender_manual:
    command: "./run_server.sh"
    cwd: "agents/blender-manual"

logging:
  level: INFO
  session_logs: "build/orchestrator_logs/"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## State Persistence

### Session State (survives restarts)

```json
{
  "session_id": "sun_surface_2025_01",
  "asset_name": "sun_surface",
  "effect_type": "pyro",
  "description": "Realistic sun surface with prominences",

  "trust_score": 0.45,
  "tokens_used": 35000,
  "cost_usd": 1.75,

  "current_stage": "EVALUATE_QUALITY",
  "current_iteration": 3,
  "best_score": 62.5,
  "best_iteration": 2,

  "parameters": {
    "flame_max_temp": 6000,
    "turbulence": 0.4,
    "domain_scale": 8.0
  },

  "history": [
    {"iteration": 1, "score": 47.6, "action": "adjust_params"},
    {"iteration": 2, "score": 62.5, "action": "adjust_params"},
    {"iteration": 3, "score": null, "action": "in_progress"}
  ],

  "decisions_pending_approval": [],
  "last_updated": "2025-01-01T12:00:00Z"
}
```

### Daily Stats (for cost tracking)

```json
{
  "date": "2025-01-01",
  "total_tokens": 150000,
  "total_cost_usd": 7.50,
  "sessions_run": 3,
  "assets_completed": 2,
  "trust_score_trend": [0.2, 0.35, 0.45]
}
```

## Human Interaction Points

### Approval Request Format

When the orchestrator needs approval:

```
═══════════════════════════════════════════════════════════════════════════════
APPROVAL REQUESTED - Blender VFX Orchestrator
═══════════════════════════════════════════════════════════════════════════════

Session: sun_surface_2025_01
Iteration: 3/10
Trust Level: GUIDED (0.45)
Tokens Used: 35,000 / 100,000

DECISION: Change technique after 3 iterations with plateau

Current State:
- VFX Quality Score: 52.1 (target: 70)
- Previous Scores: 47.6 → 52.1 → 52.3 (plateau detected)
- Current Technique: rising_mushroom

Proposed Action:
- Switch to technique: "ground_burst" (different emission pattern)
- Rationale: Rising mushroom produces top-heavy effect, ground burst may
  distribute density more evenly for sun surface.

Options:
[1] APPROVE - Switch to ground_burst technique
[2] MODIFY  - Suggest different technique
[3] ADJUST  - Keep current technique, modify specific parameters
[4] ABORT   - End session, save current best

Your choice:
```

### Notification Format (no approval needed)

When autonomy level allows proceeding:

```
[ORCHESTRATOR] Session: sun_surface_2025_01, Iteration 4
               Score: 52.3 → 68.5 (+16.2)
               Action: Continuing iteration (autonomous)
               Tokens: 45,000/100,000 (45%)
```

## Error Handling

### Recoverable Errors
- Blender execution timeout → Retry with simpler settings
- Script syntax error → Query blender-manual, fix, retry
- Evaluation failed → Skip evaluation, ask human

### Critical Errors (always ask human)
- MCP server connection lost
- Token limit exceeded
- 3 consecutive failures
- Trust score dropped below 0.1

## API (for Claude Code integration)

The orchestrator exposes these entry points:

```python
# Start new asset generation
orchestrator.create_asset(
    asset_name="explosion_01",
    effect_type="pyro",
    description="Dramatic fireball explosion",
    semantic_query="a bright explosion with flames and smoke"
)

# Resume interrupted session
orchestrator.resume_session("explosion_01_2025_01")

# Manual trust score adjustment
orchestrator.set_trust_score(0.8)  # Promote to autonomous

# Force autonomy level
orchestrator.set_autonomy_override("supervised")  # Temporary override

# Get session status
status = orchestrator.get_status()

# Abort session
orchestrator.abort_session(save_best=True)
```

## Implementation Files

```
agents/blender-orchestrator/
├── config.yaml                 # Configuration file
├── server.py                   # Agent SDK entry point
├── orchestrator.py             # BlenderOrchestratorAgent class
├── autonomy.py                 # Trust score & autonomy controller
├── guardrails.py               # Token tracking & limits
├── workflow.py                 # Workflow stage management
├── state.py                    # Session state persistence
├── prompts.py                  # System prompts and templates
├── mcp_bridge.py               # MCP server connections
├── requirements.txt            # Python dependencies
├── run_server.sh               # Launch script
└── DESIGN.md                   # This document
```

## Development Phases

### Phase 1: Foundation
- [x] Design document (this file)
- [ ] Config schema and loader
- [ ] Base orchestrator class
- [ ] MCP server integration

### Phase 2: Core Logic
- [ ] Workflow state machine
- [ ] Trust score system
- [ ] Token guardrails
- [ ] Session persistence

### Phase 3: Claude Integration
- [ ] Agent SDK client setup
- [ ] System prompts
- [ ] can_use_tool callback
- [ ] Approval request formatting

### Phase 4: Testing
- [x] Unit tests for autonomy logic
- [ ] Integration tests with MCP servers
- [ ] End-to-end asset generation test

### Phase 5: Refinement
- [ ] Tune trust score adjustments
- [ ] Optimize token usage
- [ ] Add more workflow hooks

---

## Quick Start

### Installation

```bash
cd agents/blender-orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CLI Usage

```bash
# Check orchestrator status
./run_server.sh status

# Create a new VFX asset
./run_server.sh create explosion_01 pyro "A dramatic fireball explosion"

# Create with specific technique and resolution
./run_server.sh create sun_surface pyro "Realistic sun surface" \
    --resolution 128 --frames 100 --technique rising_mushroom

# List all sessions
./run_server.sh list

# Resume an interrupted session
./run_server.sh resume explosion_01_20250101_120000

# Set trust score manually (for testing)
./run_server.sh trust 0.8

# Override autonomy level
./run_server.sh autonomy supervised
```

### Python API Usage

```python
from orchestrator import BlenderOrchestratorAgent
from state import AssetRequest

# Initialize orchestrator
agent = BlenderOrchestratorAgent()

# Create an asset request
request = AssetRequest(
    asset_name="supernova_burst",
    effect_type="pyro",
    description="Expanding stellar explosion with hot core",
    resolution=96,
    frame_end=50
)

# Run with adaptive autonomy
import asyncio
result = asyncio.run(agent.create_asset(request))
print(f"Best score: {result['best_score']}")
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key"  # Required for Claude API
export PLASMADX_CONFIG="path/to/config.yaml"  # Optional config override
```

### File Structure

```
blender-orchestrator/
├── orchestrator.py    # Main BlenderOrchestratorAgent class
├── autonomy.py        # Trust score and autonomy level system
├── guardrails.py      # Token usage limits and cost tracking
├── workflow.py        # Workflow state machine
├── state.py           # Session persistence
├── server.py          # CLI entry point
├── config.yaml        # Default configuration
├── run_server.sh      # Shell launcher
└── DESIGN.md          # This document
```
