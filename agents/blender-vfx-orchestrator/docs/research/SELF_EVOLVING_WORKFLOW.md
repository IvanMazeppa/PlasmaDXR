# Self-Evolving Agents Workflow

**Source:** [OpenAI Cookbook - Self-Evolving Agents](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)
**Relevance:** Continuous improvement through LLM-as-judge evaluation and prompt refinement

---

## Core Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 SELF-EVOLVING AGENT LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Baseline Agent generates output                             │
│  2. Evaluation measures performance against criteria            │
│  3. Feedback Collection identifies gaps and failure modes       │
│  4. Prompt Optimization refines instructions                    │
│  5. Updated Agent replaces previous when performance improves   │
│  6. GOTO 1 (continuous cycle)                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Exit Condition:** Performance exceeds target threshold (e.g., 0.8 score) OR retry limits reached.

---

## Human Feedback Integration

### Option 1: Manual Review via Platform UI

Users provide:
- Thumbs-up/down ratings
- Textual feedback on outputs

Platform then:
- Automatically generates improved prompts
- Based on accumulated annotations

**Pros:** High-quality signal
**Cons:** Requires human SME time

### Option 2: Automated LLM-as-Judge

When human review unavailable:
- LLM evaluates outputs against rubric
- System automatically triggers optimization
- Based on scoring thresholds

**Pros:** Fully autonomous
**Cons:** May miss nuanced issues

---

## LLM-as-Judge Evaluation

### Multi-Grader System

Use multiple graders for robust evaluation:

| Grader Type | Function | Threshold |
|-------------|----------|-----------|
| **Python (domain)** | Verify domain entities appear in output | 0.8 |
| **Python (format)** | Enforce structure/length requirements | 0.85 |
| **Text similarity** | Ensure semantic fidelity to intent | 0.85 |
| **Score model** | Capture nuanced quality signals | 0.85 |

### Lenient Pass Criteria

A "lenient pass" occurs when:
- 75% of graders pass, OR
- Average score exceeds 0.85

This prevents over-optimization on single criteria.

### Application to Your System

```python
from typing import List
from pydantic import BaseModel

class QualityGrader(BaseModel):
    """Individual grader result."""
    name: str
    score: float
    passed: bool
    feedback: str

class MultiGraderResult(BaseModel):
    """Combined result from all graders."""
    graders: List[QualityGrader]
    lenient_pass: bool
    aggregate_score: float

async def evaluate_script_quality(
    script: str,
    render_image: bytes,
    effect_type: str
) -> MultiGraderResult:
    """Evaluate script quality using multiple graders."""

    graders = []

    # Grader 1: API Correctness (Python-based)
    api_score = await grade_api_correctness(script)
    graders.append(QualityGrader(
        name="api_correctness",
        score=api_score,
        passed=api_score >= 0.9,
        feedback="API calls verified" if api_score >= 0.9 else "Unverified API calls found"
    ))

    # Grader 2: Visual Quality (LLM-as-judge)
    visual_result = await Runner.run(
        visual_grader_agent,
        [{"type": "image", "data": render_image}]
    )
    graders.append(QualityGrader(
        name="visual_quality",
        score=visual_result.score,
        passed=visual_result.score >= 0.6,
        feedback=visual_result.feedback
    ))

    # Grader 3: Effect Accuracy (LLM-as-judge)
    accuracy_result = await Runner.run(
        accuracy_grader_agent,
        f"Does this render match '{effect_type}'? Image attached."
    )
    graders.append(QualityGrader(
        name="effect_accuracy",
        score=accuracy_result.score,
        passed=accuracy_result.score >= 0.7,
        feedback=accuracy_result.feedback
    ))

    # Calculate aggregate
    scores = [g.score for g in graders]
    passed_count = sum(1 for g in graders if g.passed)
    aggregate = sum(scores) / len(scores)

    lenient_pass = (passed_count / len(graders) >= 0.75) or (aggregate >= 0.85)

    return MultiGraderResult(
        graders=graders,
        lenient_pass=lenient_pass,
        aggregate_score=aggregate
    )
```

---

## Versioned Prompt Tracking

Track prompt iterations with rollback capability:

```python
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

class PromptVersion(BaseModel):
    """Single prompt version."""
    version: int
    prompt: str
    model: str
    created_at: datetime
    performance_score: Optional[float] = None
    metadata: dict = {}

class VersionedPromptManager:
    """Manages prompt iterations with rollback capability."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.versions: List[PromptVersion] = []
        self.current_version: int = 0

    def add_version(
        self,
        prompt: str,
        model: str,
        metadata: dict = {}
    ) -> PromptVersion:
        """Create new version entry."""
        version = PromptVersion(
            version=len(self.versions) + 1,
            prompt=prompt,
            model=model,
            created_at=datetime.now(),
            metadata=metadata
        )
        self.versions.append(version)
        self.current_version = version.version
        return version

    def record_performance(self, version: int, score: float):
        """Record performance for a version."""
        self.versions[version - 1].performance_score = score

    def get_best_version(self) -> PromptVersion:
        """Get highest-performing version."""
        scored = [v for v in self.versions if v.performance_score is not None]
        if not scored:
            return self.versions[-1]
        return max(scored, key=lambda v: v.performance_score)

    def rollback(self, to_version: int) -> PromptVersion:
        """Rollback to specific version."""
        self.current_version = to_version
        return self.versions[to_version - 1]

    def get_current(self) -> PromptVersion:
        """Get current active prompt."""
        return self.versions[self.current_version - 1]
```

---

## Prompt Optimization Loop

```python
async def optimize_agent_prompt(
    agent: Agent,
    test_cases: List[TestCase],
    target_score: float = 0.8,
    max_attempts: int = 5
) -> Agent:
    """Automatically optimize agent prompt based on test performance."""

    prompt_manager = VersionedPromptManager(agent.name)
    prompt_manager.add_version(agent.instructions, agent.model)

    for attempt in range(max_attempts):
        # Run test cases
        results = []
        for case in test_cases:
            result = await Runner.run(agent, case.input)
            evaluation = await evaluate_output(result.final_output, case.expected)
            results.append(evaluation)

        # Calculate aggregate score
        avg_score = sum(r.score for r in results) / len(results)
        prompt_manager.record_performance(prompt_manager.current_version, avg_score)

        if avg_score >= target_score:
            print(f"Target reached at attempt {attempt + 1}: {avg_score:.2f}")
            break

        # Generate improved prompt
        failures = [r for r in results if r.score < target_score]
        improvement_prompt = f"""
        Current agent instructions:
        {agent.instructions}

        Failed on these cases:
        {[f.feedback for f in failures]}

        Generate improved instructions that address these failures.
        Keep the core purpose but fix the specific issues.
        """

        improved = await Runner.run(prompt_optimizer_agent, improvement_prompt)
        new_instructions = improved.final_output

        # Create new version
        prompt_manager.add_version(new_instructions, agent.model, {
            "attempt": attempt + 1,
            "previous_score": avg_score,
            "failure_count": len(failures)
        })

        # Update agent
        agent = Agent(
            name=agent.name,
            instructions=new_instructions,
            tools=agent.tools,
            model=agent.model,
        )

    # Return best-performing version
    best = prompt_manager.get_best_version()
    return Agent(
        name=agent.name,
        instructions=best.prompt,
        tools=agent.tools,
        model=best.model,
    )
```

---

## Continuous Monitoring

Periodically check for new data and trigger re-evaluation:

```python
import asyncio
from datetime import datetime, timedelta

class ContinuousMonitor:
    """Monitor agent performance and trigger retraining."""

    def __init__(
        self,
        agent: Agent,
        check_interval: timedelta = timedelta(hours=1),
        performance_threshold: float = 0.7
    ):
        self.agent = agent
        self.check_interval = check_interval
        self.threshold = performance_threshold
        self.last_check = datetime.now()

    async def run(self):
        """Main monitoring loop."""
        while True:
            await asyncio.sleep(self.check_interval.total_seconds())

            # Get recent performance data
            recent_results = await get_recent_results(self.agent.name)

            if not recent_results:
                continue

            avg_score = sum(r.score for r in recent_results) / len(recent_results)

            if avg_score < self.threshold:
                print(f"Performance degraded: {avg_score:.2f} < {self.threshold}")
                # Trigger retraining
                self.agent = await optimize_agent_prompt(
                    self.agent,
                    test_cases=generate_test_cases(recent_results),
                    target_score=self.threshold
                )
                print(f"Agent retrained. New version deployed.")

            self.last_check = datetime.now()
```

---

## Application to blender-vfx-orchestrator

### Immediate Integration Points

1. **Multi-grader evaluation** in `QualityAnalyst`:
   - API correctness grader (rule-based)
   - Visual quality grader (vision model)
   - Effect accuracy grader (LLM-as-judge)

2. **Versioned prompt tracking** for `ScriptWriter`:
   - Track which prompts produce successful scripts
   - Rollback if new prompt performs worse

3. **Automatic prompt refinement** when quality plateaus:
   - Analyze failure patterns
   - Generate targeted prompt improvements
   - Test and deploy if better

### Key Insight

> "The notebook is modular—feel free to run sections independently or sequentially as you adapt the retraining loop to your own agents."

Your existing `LearningAgent` and `experiment-tracker` can be enhanced with these patterns for continuous self-improvement.
