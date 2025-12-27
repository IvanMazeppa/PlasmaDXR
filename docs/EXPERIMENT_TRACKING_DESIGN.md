# Experiment Tracking System Design

## Problem Statement

The current iteration system evaluates quality but doesn't **learn**:
- No tracking of parameter changes → effect relationships
- No memory of what worked/failed across sessions
- No systematic exploration of parameter space
- Each iteration is blind to why previous ones succeeded/failed

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Experiment Tracking Layer                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Experiment  │    │  Knowledge   │    │  Hypothesis  │                   │
│  │   Database   │◄──►│    Base      │◄──►│   Engine     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    Experiment Runner                          │           │
│  │  - Parallel variation testing                                 │           │
│  │  - A/B comparisons                                           │           │
│  │  - Systematic parameter sweeps                               │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Existing Pipeline           │
                    │  - Script Generator           │
                    │  - Blender Executor           │
                    │  - Asset Evaluator            │
                    │  - Iteration Controller       │
                    └───────────────────────────────┘
```

---

## Core Components

### 1. Experiment Database

Stores every experiment with full context:

```python
@dataclass
class Experiment:
    id: str                          # Unique experiment ID
    session_id: str                  # Parent session
    timestamp: datetime

    # What changed
    parameter_changes: Dict[str, ParameterChange]
    hypothesis: str                  # "Increasing domain height will fix clipping"

    # Before state
    baseline_params: Dict[str, Any]
    baseline_scores: Dict[str, float]
    baseline_render: str             # Path to before image

    # After state
    result_params: Dict[str, Any]
    result_scores: Dict[str, float]
    result_render: str               # Path to after image

    # Analysis
    effects: List[ObservedEffect]    # What actually changed
    success: bool                    # Did it achieve the hypothesis goal?
    notes: str                       # Human/AI observations

@dataclass
class ParameterChange:
    parameter: str                   # "domain_scale"
    old_value: Any                   # 6.0
    new_value: Any                   # 10.0
    change_type: str                 # "increase", "decrease", "modify"
    magnitude: float                 # 1.67x

@dataclass
class ObservedEffect:
    metric: str                      # "framing_position"
    direction: str                   # "shifted_up"
    magnitude: float                 # How much
    expected: bool                   # Was this the intended effect?
    side_effect: bool                # Unintended consequence?
```

### 2. Knowledge Base

Accumulates learnings across all experiments:

```python
@dataclass
class ParameterKnowledge:
    parameter: str                   # "domain_scale"

    # Observed relationships
    effects: List[CausalRelationship]

    # Statistical summary
    experiments_count: int
    avg_effect_on_scores: Dict[str, float]

    # Learned rules
    rules: List[str]                 # "Increasing domain_scale shifts content upward"
    warnings: List[str]              # "Changes position, not just size"

    # Recommended ranges
    safe_range: Tuple[float, float]
    optimal_value: float
    confidence: float

@dataclass
class CausalRelationship:
    cause: str                       # "increase domain_scale"
    effect: str                      # "content shifts upward"
    confidence: float                # 0.85
    evidence_count: int              # 3 experiments support this
    counterexamples: int             # 0 experiments contradict
```

### 3. Hypothesis Engine

Generates and tests hypotheses systematically:

```python
class HypothesisEngine:
    def diagnose_issue(self, current_render: str, target_description: str) -> List[Hypothesis]:
        """
        Analyze current render and suggest hypotheses to test.

        Example output:
        [
            Hypothesis(
                issue="clipping at top edge",
                proposed_fix="increase domain height",
                parameter="domain_scale",
                suggested_change="+50%",
                confidence=0.7,
                risks=["may shift content position"]
            ),
            Hypothesis(
                issue="clipping at top edge",
                proposed_fix="move domain center down",
                parameter="domain_location_z",
                suggested_change="-2.0",
                confidence=0.6,
                risks=["may clip at bottom instead"]
            )
        ]
        """
        pass

    def design_experiment(self, hypothesis: Hypothesis) -> ExperimentPlan:
        """
        Create a controlled experiment to test the hypothesis.
        Includes control group (no change) and test group (with change).
        """
        pass

    def analyze_result(self, experiment: Experiment) -> ExperimentAnalysis:
        """
        Determine if hypothesis was supported, refuted, or inconclusive.
        Extract learnings for knowledge base.
        """
        pass
```

### 4. Experiment Runner

Executes experiments systematically:

```python
class ExperimentRunner:
    def run_parallel_variations(
        self,
        base_script: str,
        variations: List[Dict[str, Any]],
        evaluation_criteria: Dict[str, float]
    ) -> List[ExperimentResult]:
        """
        Run multiple variations in parallel and compare results.

        Example:
        variations = [
            {"domain_scale": 8.0, "domain_z": 0},    # Variation A
            {"domain_scale": 10.0, "domain_z": 0},   # Variation B
            {"domain_scale": 6.0, "domain_z": -2},   # Variation C
        ]

        Returns ranked results with analysis of which worked best and why.
        """
        pass

    def run_parameter_sweep(
        self,
        parameter: str,
        range_start: float,
        range_end: float,
        steps: int
    ) -> ParameterSweepResult:
        """
        Systematically test a range of values for one parameter.
        Builds understanding of parameter sensitivity.
        """
        pass

    def run_ab_test(
        self,
        control_params: Dict,
        test_params: Dict,
        change_description: str
    ) -> ABTestResult:
        """
        Controlled comparison of exactly one change.
        """
        pass
```

---

## Workflow Example

### Current (Blind Iteration)
```
1. Generate script with defaults
2. Run, evaluate → Score: 47.6
3. Human says "clipping at top"
4. Guess a fix (increase domain)
5. Run, evaluate → Score: 46.9
6. Human says "framing is worse now"
7. No learning recorded, start over
```

### Proposed (Learning Iteration)
```
1. Generate script with defaults
2. Run, evaluate → Score: 47.6, Record baseline

3. Diagnose: "clipping detected at top edge"

4. Hypothesis Engine generates options:
   - H1: Increase domain_scale (confidence: 0.6, risk: position shift)
   - H2: Lower domain_z position (confidence: 0.7, risk: bottom clip)
   - H3: Increase domain_scale AND lower position (confidence: 0.8)

5. Run parallel experiments for H1, H2, H3

6. Results:
   - H1: Clipping fixed BUT position shifted up (partial success)
   - H2: Clipping fixed, position maintained (success)
   - H3: Best result (success)

7. Record learnings to Knowledge Base:
   - "domain_scale increase alone shifts content upward"
   - "combine domain_scale increase with position adjustment"
   - "domain_z offset = -domain_scale_increase * 0.3 maintains framing"

8. Next session automatically applies this knowledge
```

---

## Database Schema (SQLite)

```sql
-- Experiments table
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp DATETIME,
    hypothesis TEXT,
    baseline_params JSON,
    result_params JSON,
    baseline_scores JSON,
    result_scores JSON,
    baseline_render TEXT,
    result_render TEXT,
    success BOOLEAN,
    notes TEXT
);

-- Parameter changes
CREATE TABLE parameter_changes (
    id INTEGER PRIMARY KEY,
    experiment_id TEXT,
    parameter TEXT,
    old_value TEXT,
    new_value TEXT,
    change_type TEXT,
    magnitude REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Observed effects
CREATE TABLE observed_effects (
    id INTEGER PRIMARY KEY,
    experiment_id TEXT,
    metric TEXT,
    direction TEXT,
    magnitude REAL,
    expected BOOLEAN,
    side_effect BOOLEAN,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Knowledge base
CREATE TABLE parameter_knowledge (
    parameter TEXT PRIMARY KEY,
    effects JSON,
    experiments_count INTEGER,
    avg_effects JSON,
    rules JSON,
    warnings JSON,
    safe_range JSON,
    optimal_value REAL,
    confidence REAL,
    updated_at DATETIME
);

-- Causal relationships
CREATE TABLE causal_relationships (
    id INTEGER PRIMARY KEY,
    cause TEXT,
    effect TEXT,
    confidence REAL,
    evidence_count INTEGER,
    counterexamples INTEGER
);
```

---

## New MCP Tools

### `experiment_diagnose`
Analyze current render and suggest experiments to run.

### `experiment_run_parallel`
Run multiple variations simultaneously and compare.

### `experiment_run_sweep`
Systematically test parameter ranges.

### `knowledge_query`
Query the knowledge base: "What happens when I increase domain_scale?"

### `knowledge_add_observation`
Manually add learnings from human observation.

---

## Integration with Existing System

The experiment tracking layer wraps the existing pipeline:

```python
# Before (current)
result = iteration_controller.create_asset(...)

# After (with experiment tracking)
result = experiment_tracker.create_asset_with_learning(
    ...,
    enable_parallel_exploration=True,
    hypothesis_mode="auto",  # or "manual"
    record_to_knowledge_base=True
)
```

---

## Implementation Priority

1. **Phase 1: Experiment Database** - Record everything
2. **Phase 2: Knowledge Base** - Accumulate learnings
3. **Phase 3: Hypothesis Engine** - Suggest experiments
4. **Phase 4: Parallel Runner** - Test multiple variations
5. **Phase 5: Cross-Session Learning** - Apply past learnings to new sessions

---

## Open Questions

1. **Storage format:** SQLite vs JSON files vs hybrid?
2. **Parallelization:** How many Blender instances can run simultaneously?
3. **Human-in-the-loop:** When should the system ask for human feedback?
4. **Confidence thresholds:** When is a learning "proven" enough to apply automatically?

---

## Example Knowledge Entry

After the domain_scale experiment:

```json
{
  "parameter": "domain_scale",
  "effects": [
    {
      "cause": "increase domain_scale without adjusting position",
      "effect": "content shifts upward in frame",
      "confidence": 1.0,
      "evidence_count": 1,
      "counterexamples": 0
    }
  ],
  "rules": [
    "When increasing domain_scale, also decrease domain_location_z proportionally",
    "Formula: new_z = old_z - (scale_increase * 0.3)"
  ],
  "warnings": [
    "Increasing domain_scale alone will shift content position"
  ]
}
```

This would prevent the same mistake in future sessions.
