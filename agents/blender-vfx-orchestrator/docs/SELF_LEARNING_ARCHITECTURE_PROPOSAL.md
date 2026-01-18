# Self-Learning Architecture Proposal for Blender VFX Orchestrator

**Date:** 2026-01-16
**Status:** ✅ ALL STRATEGIES IMPLEMENTED
**Author:** Claude Code Analysis

---

## Implementation Status

| Strategy | Status | Date |
|----------|--------|------|
| **Strategy 1: Vector Store** | **✅ Complete** | 2026-01-17 |
| **Strategy 2: Knowledge Distillation** | **✅ Complete** | 2026-01-17 |
| **Strategy 3: Proactive Documentation Mining** | **✅ Complete** | 2026-01-16 |
| **Strategy 4: Code Pattern Memory** | **✅ Complete** | 2026-01-17 |
| **Strategy 5: Escape Velocity Mechanism** | **✅ Complete** | 2026-01-16 |

---

## Executive Summary

The blender-vfx-orchestrator agent system can generate valid Blender 5.0.1 scripts but lacks true self-learning capability. Analysis reveals the system captures knowledge but fails to synthesize it into actionable improvements. This document proposes 5 architectural strategies to enable autonomous self-learning behavior.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Current Architecture Review](#current-architecture-review)
4. [Proposed Strategies](#proposed-strategies)
   - [Strategy 1: Vector Store for Blender Documentation](#strategy-1-vector-store-for-blender-documentation) ✅
   - [Strategy 2: Knowledge Distillation Loop](#strategy-2-knowledge-distillation-loop) ✅
   - [Strategy 3: Proactive Documentation Mining](#strategy-3-proactive-documentation-mining) ✅
   - [Strategy 4: Code Pattern Memory](#strategy-4-code-pattern-memory) ✅
   - [Strategy 5: Escape Velocity Mechanism](#strategy-5-escape-velocity-mechanism) ✅
5. [Implementation Priority](#implementation-priority)
6. [Concrete Implementation Steps](#concrete-implementation-steps)
7. [Success Metrics](#success-metrics)
8. [Risks and Mitigations](#risks-and-mitigations)

---

## Problem Statement

### Observed Behavior
- Agent generates valid Blender scripts and iterates on quality
- When quality plateaus, agent falls back to familiar techniques
- Same types of fixes are attempted repeatedly
- Novel approaches from Blender documentation are rarely discovered
- Knowledge is recorded but doesn't transform future behavior

### Desired Behavior
- Agent learns new techniques from documentation autonomously
- Successful code patterns are extracted and reused
- Stuck detection triggers exploration of genuinely different approaches
- Knowledge compounds across sessions

---

## Root Cause Analysis

### Gap 1: UCB1 Selects But Doesn't Learn

**Location:** `specialized_agents/script_writer.py`

The UCB1 (Upper Confidence Bound) algorithm balances exploration/exploitation among *existing* techniques but never discovers techniques outside its hardcoded set.

```python
# Current: UCB1 selects from fixed technique list
SCRIPT_WRITER_INSTRUCTIONS = """
1. Generate new scripts using UCB1 technique selection for variety
"""
```

**Problem:** The technique library is static. No mechanism to add new techniques discovered from documentation or successful experiments.

### Gap 2: Docs Expert is Reactive, Not Proactive

**Location:** `orchestrator.py` lines 107-113

```python
# Current: Only consulted when explicitly "stuck"
## STUCK DETECTION
You are STUCK if:
- 3+ consecutive iterations with <5 point improvement
- Same primary issue persists for 3+ iterations
```

**Problem:** By the time stuck detection triggers, 3 iterations have been wasted on the same approach. Documentation mining should happen *before* repeated failures.

### Gap 3: Experiment Tracker Records But Doesn't Transform

**Location:** `tools/experiment_tracker_tools.py`

The experiment tracker stores:
- Parameter changes
- Score deltas
- Warnings and learnings (as text)

**Problem:** Learnings are stored as natural language text but never transformed into:
- Executable code patterns
- New techniques for UCB1
- Proactive script modifications

### Gap 4: No Feedback Loop to Script Generation

**Location:** `specialized_agents/learning_agent.py`

```python
# Current: Knowledge base warns but doesn't inject code
LEARNING_AGENT_INSTRUCTIONS = """
BEFORE MAKING CHANGES:
1. ALWAYS call get_warnings_before_change(parameter, change_type)
2. Check query_knowledge_base(issue) for similar past issues
"""
```

**Problem:** The knowledge base provides warnings (reactive) but never injects *working code snippets* (proactive).

### Gap 5: Same Primary Issue Loop

**Location:** `orchestrator.py` stuck detection

When the same primary issue persists, the agent tries variations of the same approach rather than fundamentally different solutions.

---

## Current Architecture Review

### Agent Structure

```
BlenderVFXOrchestrator (coordinator)
├── ScriptWriter      - Generates Blender Python scripts
│   └── Uses: UCB1 technique selection, parameter validation
├── Executor          - Runs scripts, parses errors
├── QualityAnalyst    - Vision + ML quality evaluation
│   └── Uses: GPT-4o vision (primary), LPIPS/SigLIP/TOPIQ (backup)
├── LearningAgent     - Experiment tracking, knowledge queries
│   └── Uses: experiment-tracker MCP server
└── DocsExpert        - Blender documentation search
    └── Uses: blender-manual MCP server (keyword search)
```

### Data Flow (Current)

```
Request → ScriptWriter → Executor → QualityAnalyst
                ↑                         ↓
                └──── LearningAgent ←─────┘
                           ↑
                      DocsExpert (only when stuck)
```

### Knowledge Persistence (Current)

| Component | Storage | Retrieval |
|-----------|---------|-----------|
| Experiment Tracker | SQLite DB | Keyword query |
| Session State | JSON files | Direct load |
| Blender Manual | Indexed files | Keyword search |
| Technique Library | Hardcoded | UCB1 selection |

---

## Proposed Strategies

### Strategy 1: Vector Store for Blender Documentation

#### Problem Addressed
The blender-manual MCP provides keyword search, which misses semantically related concepts. When searching for "turbulence", it won't find related concepts like "vorticity", "noise_strength", or "upres noise".

#### Solution
Create an OpenAI vector store containing:
- Blender 5.0 Python API documentation
- Mantaflow fluid simulation tutorials
- Successful script templates from past sessions
- Extracted code patterns from experiments

#### Implementation

**New File:** `tools/semantic_docs_tools.py`

```python
from agents import function_tool
from openai import OpenAI

# Vector store ID (created via setup script)
BLENDER_DOCS_VECTOR_STORE_ID = "vs_blender_5_0_docs"

@function_tool
async def semantic_search_blender_docs(
    query: str,
    max_results: int = 5,
    include_code_examples: bool = True
) -> str:
    """
    Search Blender documentation semantically using OpenAI vector stores.

    Unlike keyword search, this finds conceptually related documentation
    even when exact terms don't match. Use for:
    - Finding alternative approaches to a problem
    - Discovering related API functions
    - Finding code examples for specific effects

    Args:
        query: Natural language description of what you're looking for
               (e.g., "how to make smoke rise faster and dissipate")
        max_results: Maximum number of documentation chunks to return
        include_code_examples: Whether to prioritize results with code

    Returns:
        JSON with:
        - results: List of relevant documentation chunks
        - code_snippets: Extracted Python code examples
        - related_apis: List of bpy.types/bpy.ops references found
    """
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1",
        input=f"""Find Blender 5.0 documentation relevant to: {query}

        Focus on:
        1. Python API code examples
        2. Parameter ranges and defaults
        3. Related settings that affect this behavior
        """,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [BLENDER_DOCS_VECTOR_STORE_ID],
            "max_num_results": max_results
        }],
        include=["file_search_call.results"]
    )

    return format_search_results(response)


@function_tool
async def find_alternative_approaches(
    current_approach: str,
    issue: str,
    effect_type: str
) -> str:
    """
    Search for fundamentally different approaches to solve an issue.

    Use when current approach has plateaued. This searches for:
    - Different API functions that achieve similar results
    - Alternative simulation techniques
    - Workarounds used by other artists

    Args:
        current_approach: What you're currently doing
        issue: The problem you're trying to solve
        effect_type: Type of effect (pyro, explosion, fire, etc.)

    Returns:
        JSON with alternative approaches, each including:
        - approach_name: Short description
        - code_example: Working Python snippet
        - trade_offs: Pros and cons vs current approach
        - confidence: How likely this is to help (0-100)
    """
    # Semantic search with negative examples
    query = f"""
    For {effect_type} effects in Blender 5.0:

    Problem: {issue}

    Current approach (NOT working well): {current_approach}

    Find DIFFERENT approaches that solve the same problem.
    Exclude variations of the current approach.
    """

    # Implementation continues...
```

**Setup Script:** `scripts/create_blender_vector_store.py`

```python
"""
Create OpenAI vector store with Blender 5.0 documentation.

Run once during setup:
    python scripts/create_blender_vector_store.py

Sources indexed:
1. Blender 5.0 Python API reference (bpy.types, bpy.ops)
2. Mantaflow fluid simulation documentation
3. Physics simulation tutorials
4. Successful patterns from experiment tracker
"""

import os
from pathlib import Path
from openai import OpenAI

DOCS_DIR = Path("agents/blender-manual/docs")
PATTERNS_DIR = Path("agents/experiment-tracker/successful_patterns")

def create_vector_store():
    client = OpenAI()

    # Create vector store
    vector_store = client.vector_stores.create(
        name="blender_5_0_documentation",
        metadata={
            "version": "5.0.1",
            "created": "2026-01-16",
            "purpose": "VFX script generation"
        }
    )

    print(f"Created vector store: {vector_store.id}")

    # Upload documentation files
    for doc_file in DOCS_DIR.glob("**/*.md"):
        upload_file(client, vector_store.id, doc_file)

    for doc_file in DOCS_DIR.glob("**/*.rst"):
        upload_file(client, vector_store.id, doc_file)

    # Upload successful patterns
    for pattern_file in PATTERNS_DIR.glob("*.json"):
        upload_file(client, vector_store.id, pattern_file)

    print(f"Vector store ready: {vector_store.id}")
    print("Add this to your environment:")
    print(f"  BLENDER_DOCS_VECTOR_STORE_ID={vector_store.id}")

def upload_file(client, vector_store_id, file_path):
    # Upload and add to vector store
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="assistants")

    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file.id
    )
    print(f"  Uploaded: {file_path.name}")

if __name__ == "__main__":
    create_vector_store()
```

#### Integration Points

1. **DocsExpert Agent:** Add `semantic_search_blender_docs` as primary tool
2. **ScriptWriter Agent:** Add `find_alternative_approaches` for stuck situations
3. **Orchestrator:** Use semantic search in pre-iteration research phase

#### Expected Impact
- Discovers related concepts that keyword search misses
- Finds code examples for novel approaches
- Enables proactive exploration of alternatives

---

### Strategy 2: Knowledge Distillation Loop

#### Problem Addressed
Experiment outcomes are stored as metadata (parameters, scores, text descriptions) but never transformed into reusable code patterns that can be directly injected into future scripts.

#### Solution
After each successful experiment, extract the actual working code changes and generalize them into reusable patterns.

#### Implementation

**New File:** `tools/pattern_extractor.py`

```python
from agents import function_tool
from typing import Dict, List, Any
import json
import difflib

@function_tool
async def extract_successful_pattern(
    original_script: str,
    modified_script: str,
    issue_fixed: str,
    improvement_delta: float,
    effect_type: str
) -> str:
    """
    Extract a reusable pattern from a successful script modification.

    Analyzes the diff between original and modified scripts to identify
    the specific code changes that fixed the issue. Generalizes these
    into a reusable pattern that can be applied to future scripts.

    Args:
        original_script: Path to script before modification
        modified_script: Path to script after successful modification
        issue_fixed: Description of issue that was fixed
        improvement_delta: Score improvement (e.g., 15.0 for +15 points)
        effect_type: Type of effect (pyro, explosion, etc.)

    Returns:
        JSON with:
        - pattern_id: Unique identifier for this pattern
        - pattern_name: Human-readable name
        - issue_category: Generalized issue category
        - code_template: Parameterized code that can be applied
        - applicability: When this pattern should be used
        - confidence: Based on improvement delta
    """
    # Read both scripts
    with open(original_script) as f:
        original = f.read()
    with open(modified_script) as f:
        modified = f.read()

    # Generate diff
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        lineterm=''
    ))

    # Extract added/changed lines
    changes = extract_meaningful_changes(diff)

    # Generalize into pattern
    pattern = generalize_pattern(changes, issue_fixed, effect_type)

    # Store in pattern library
    pattern_id = store_pattern(pattern)

    return json.dumps({
        "success": True,
        "pattern_id": pattern_id,
        "pattern_name": pattern["name"],
        "code_template": pattern["template"],
        "applicability": pattern["when_to_use"],
        "confidence": min(100, improvement_delta * 5)  # Scale to 0-100
    })


@function_tool
async def apply_pattern_to_script(
    script_path: str,
    pattern_id: str,
    parameters: Dict[str, Any] = None
) -> str:
    """
    Apply a stored pattern to an existing script.

    Retrieves a pattern from the library and applies it to the script,
    substituting any provided parameters.

    Args:
        script_path: Path to script to modify
        pattern_id: ID of pattern to apply
        parameters: Optional parameter overrides for the pattern

    Returns:
        JSON with:
        - success: Whether pattern was applied
        - modified_script_path: Path to modified script
        - changes_made: Description of changes
    """
    pattern = load_pattern(pattern_id)

    with open(script_path) as f:
        script = f.read()

    # Find insertion point based on pattern's target
    modified = apply_pattern_template(script, pattern, parameters)

    # Save modified script
    output_path = script_path.replace(".py", f"_pattern_{pattern_id}.py")
    with open(output_path, "w") as f:
        f.write(modified)

    return json.dumps({
        "success": True,
        "modified_script_path": output_path,
        "changes_made": pattern["description"]
    })


@function_tool
async def search_patterns(
    issue: str,
    effect_type: str = "",
    min_confidence: float = 50.0
) -> str:
    """
    Search pattern library for patterns that might fix an issue.

    Args:
        issue: Description of the issue to fix
        effect_type: Optional filter by effect type
        min_confidence: Minimum confidence threshold (0-100)

    Returns:
        JSON with list of matching patterns, ranked by relevance
    """
    # Implementation: semantic search over pattern library
    pass
```

**Pattern Storage Structure:**

```json
{
  "pattern_id": "pat_20260116_001",
  "name": "increase_smoke_density_with_dissolve_compensation",
  "issue_category": "smoke_too_thin",
  "effect_types": ["pyro", "explosion", "fire"],
  "code_template": {
    "target": "domain_settings",
    "changes": [
      {
        "property": "flame_smoke",
        "operation": "multiply",
        "factor": "{density_multiplier}",
        "default": 1.5
      },
      {
        "property": "dissolve_speed",
        "operation": "multiply",
        "factor": "{dissolve_compensation}",
        "default": 0.8,
        "note": "Compensate for increased density"
      }
    ]
  },
  "when_to_use": "When smoke appears too thin or transparent",
  "confidence": 85,
  "source_experiments": ["exp_20260115_042", "exp_20260114_019"],
  "average_improvement": 12.5
}
```

#### Integration Points

1. **LearningAgent:** Call `extract_successful_pattern` after each successful experiment
2. **ScriptWriter:** Use `search_patterns` before generating modifications
3. **Orchestrator:** Apply high-confidence patterns automatically

---

### Strategy 3: Proactive Documentation Mining

#### Problem Addressed
DocsExpert only activates when stuck (3+ iterations). By then, multiple iterations have been wasted on the same failing approach.

#### Solution
Add a pre-iteration research phase that proactively searches for alternatives when early warning signs appear.

#### Implementation

**Modified Orchestrator Instructions:**

```python
ORCHESTRATOR_INSTRUCTIONS = """
...

## PRE-ITERATION RESEARCH (NEW)

BEFORE each modification iteration, check for early warning signs:

1. **Same Issue Persisting** (2+ iterations)
   - If the same primary issue appears twice, IMMEDIATELY consult DocsExpert
   - Search for alternative approaches before trying another variation

2. **Parameter at Boundary**
   - If a parameter is already at its safe range limit, don't push further
   - Search documentation for alternative parameters that affect the same outcome

3. **Quality Plateau** (< 3 point change)
   - If score changed < 3 points, the current approach may be exhausted
   - Research fundamentally different techniques

## RESEARCH WORKFLOW

When early warning triggered:
1. delegate_to_docs_expert with query: "alternative approaches for {issue}"
2. If alternatives found: try the most promising one
3. If no alternatives: escalate exploration level (see Escape Velocity)

## STUCK DETECTION (UPDATED)

You are approaching STUCK if:
- 2+ consecutive iterations with same primary issue (was 3+)
- Quality plateau (< 3 point change) for 2 iterations (was <5 for 3)

You are STUCK if:
- 3+ consecutive iterations with same primary issue
- Quality score unchanged for 3 iterations

...
"""
```

**New Pre-Iteration Research Tool:**

```python
@function_tool
async def pre_iteration_research(
    current_issue: str,
    current_approach: str,
    iteration_history: List[Dict],
    effect_type: str
) -> str:
    """
    Proactively research alternatives BEFORE attempting another iteration.

    Analyzes iteration history to detect early warning signs and
    searches for alternative approaches if patterns indicate
    the current approach is unlikely to succeed.

    Args:
        current_issue: The primary issue being addressed
        current_approach: Description of current approach
        iteration_history: List of recent iterations with scores/issues
        effect_type: Type of effect being created

    Returns:
        JSON with:
        - warning_level: "none", "early", "stuck"
        - should_research: bool
        - research_queries: Suggested searches if should_research
        - alternative_approaches: Pre-fetched alternatives if available
        - recommendation: What to do next
    """
    # Analyze history for warning signs
    warning_level = analyze_warning_signs(iteration_history, current_issue)

    if warning_level == "none":
        return json.dumps({
            "warning_level": "none",
            "should_research": False,
            "recommendation": "Proceed with planned modification"
        })

    # Proactive research
    alternatives = await find_alternative_approaches(
        current_approach=current_approach,
        issue=current_issue,
        effect_type=effect_type
    )

    return json.dumps({
        "warning_level": warning_level,
        "should_research": True,
        "alternative_approaches": alternatives,
        "recommendation": f"Consider alternative: {alternatives[0]['name']}" if alternatives else "Escalate exploration"
    })


def analyze_warning_signs(history: List[Dict], current_issue: str) -> str:
    """Detect early warning signs from iteration history."""
    if len(history) < 2:
        return "none"

    # Check for same issue persisting
    recent_issues = [h.get("primary_issue", "") for h in history[-3:]]
    if recent_issues.count(current_issue) >= 2:
        return "early" if recent_issues.count(current_issue) == 2 else "stuck"

    # Check for quality plateau
    recent_scores = [h.get("score", 0) for h in history[-3:]]
    if len(recent_scores) >= 2:
        delta = abs(recent_scores[-1] - recent_scores[-2])
        if delta < 3:
            return "early"

    return "none"
```

#### Integration Points

1. **Orchestrator:** Call `pre_iteration_research` before each script modification
2. **Session State:** Track primary issue persistence count
3. **DocsExpert:** Enhanced to prioritize alternative approaches

---

### Strategy 4: Code Pattern Memory

#### Problem Addressed
Knowledge base tracks parameter metadata but not actual working code. When a fix succeeds, we record "increased flame_smoke from 2.0 to 3.5" but not the actual Python code that did it.

#### Solution
Store successful code snippets alongside parameter changes, enabling semantic retrieval of working code.

#### Implementation

**New Storage Layer:** `utils/code_pattern_memory.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import hashlib

@dataclass
class CodePattern:
    """A successful code pattern extracted from an experiment."""
    pattern_id: str
    issue_category: str
    effect_types: List[str]
    code_snippet: str           # Actual Python code
    context_lines: int = 5      # Lines of context around the change
    parameters_affected: List[str] = field(default_factory=list)
    average_improvement: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    source_experiments: List[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for vector embedding."""
        return f"""
        Issue: {self.issue_category}
        Effect types: {', '.join(self.effect_types)}
        Parameters: {', '.join(self.parameters_affected)}
        Code:
        {self.code_snippet}
        """


class CodePatternMemory:
    """
    Persistent storage for successful code patterns.

    Unlike parameter-based knowledge, this stores actual working code
    that can be retrieved semantically and applied to new scripts.
    """

    def __init__(self, storage_dir: str = "data/code_patterns"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, CodePattern] = {}
        self._load_patterns()

    def record_successful_pattern(
        self,
        issue: str,
        code_snippet: str,
        effect_type: str,
        improvement: float,
        experiment_id: str,
        parameters: List[str]
    ) -> str:
        """
        Record a new successful code pattern.

        Returns:
            pattern_id of the stored pattern
        """
        # Generate pattern ID from content hash
        pattern_id = self._generate_id(code_snippet, issue)

        # Check if similar pattern exists
        existing = self._find_similar(code_snippet)
        if existing:
            # Update existing pattern
            existing.usage_count += 1
            existing.source_experiments.append(experiment_id)
            existing.average_improvement = (
                (existing.average_improvement * (existing.usage_count - 1) + improvement)
                / existing.usage_count
            )
            if effect_type not in existing.effect_types:
                existing.effect_types.append(effect_type)
            self._save_pattern(existing)
            return existing.pattern_id

        # Create new pattern
        pattern = CodePattern(
            pattern_id=pattern_id,
            issue_category=self._categorize_issue(issue),
            effect_types=[effect_type],
            code_snippet=code_snippet,
            parameters_affected=parameters,
            average_improvement=improvement,
            usage_count=1,
            success_rate=1.0,
            source_experiments=[experiment_id]
        )

        self.patterns[pattern_id] = pattern
        self._save_pattern(pattern)

        return pattern_id

    def retrieve_patterns_for_issue(
        self,
        issue: str,
        effect_type: Optional[str] = None,
        min_improvement: float = 5.0,
        limit: int = 5
    ) -> List[CodePattern]:
        """
        Retrieve patterns that might help with an issue.

        Uses semantic similarity to find relevant patterns,
        ranked by average improvement.
        """
        # Semantic search implementation
        candidates = []

        issue_category = self._categorize_issue(issue)

        for pattern in self.patterns.values():
            # Filter by effect type if specified
            if effect_type and effect_type not in pattern.effect_types:
                continue

            # Filter by minimum improvement
            if pattern.average_improvement < min_improvement:
                continue

            # Score by relevance
            score = self._relevance_score(pattern, issue_category, issue)
            if score > 0.3:  # Threshold
                candidates.append((score, pattern))

        # Sort by score * improvement (relevance weighted by effectiveness)
        candidates.sort(key=lambda x: x[0] * x[1].average_improvement, reverse=True)

        return [p for _, p in candidates[:limit]]

    def record_pattern_outcome(
        self,
        pattern_id: str,
        success: bool,
        improvement: float
    ):
        """Record the outcome of applying a pattern."""
        if pattern_id not in self.patterns:
            return

        pattern = self.patterns[pattern_id]
        pattern.usage_count += 1

        # Update success rate
        old_successes = pattern.success_rate * (pattern.usage_count - 1)
        new_successes = old_successes + (1 if success else 0)
        pattern.success_rate = new_successes / pattern.usage_count

        # Update average improvement
        if success:
            pattern.average_improvement = (
                (pattern.average_improvement * (pattern.usage_count - 1) + improvement)
                / pattern.usage_count
            )

        self._save_pattern(pattern)

    def _generate_id(self, code: str, issue: str) -> str:
        """Generate unique ID for pattern."""
        content = f"{code}:{issue}"
        return f"pat_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def _categorize_issue(self, issue: str) -> str:
        """Categorize issue into standard categories."""
        issue_lower = issue.lower()

        categories = {
            "too_dark": ["dark", "dim", "no light", "black"],
            "too_bright": ["bright", "overexposed", "blown out"],
            "lacks_density": ["thin", "sparse", "transparent", "faint"],
            "too_dense": ["thick", "opaque", "solid"],
            "clipping": ["clipping", "cut off", "edge", "boundary"],
            "wrong_color": ["color", "hue", "tint"],
            "lacks_detail": ["detail", "resolution", "blurry", "soft"],
            "wrong_scale": ["scale", "size", "proportion"],
            "animation_issue": ["timing", "speed", "frame", "motion"],
            "structure_issue": ["shape", "structure", "form", "pattern"]
        }

        for category, keywords in categories.items():
            if any(kw in issue_lower for kw in keywords):
                return category

        return "general"

    def _find_similar(self, code: str) -> Optional[CodePattern]:
        """Find existing pattern with similar code."""
        # Simplified: exact match check
        # Full implementation would use code similarity metrics
        for pattern in self.patterns.values():
            if self._code_similarity(pattern.code_snippet, code) > 0.9:
                return pattern
        return None

    def _code_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity between two code snippets."""
        # Simplified implementation
        from difflib import SequenceMatcher
        return SequenceMatcher(None, code1, code2).ratio()

    def _relevance_score(
        self,
        pattern: CodePattern,
        issue_category: str,
        issue_text: str
    ) -> float:
        """Score relevance of pattern to issue."""
        score = 0.0

        # Category match
        if pattern.issue_category == issue_category:
            score += 0.5

        # Keyword overlap
        issue_words = set(issue_text.lower().split())
        pattern_words = set(pattern.issue_category.replace("_", " ").split())
        overlap = len(issue_words & pattern_words)
        score += 0.3 * min(overlap / max(len(issue_words), 1), 1.0)

        # Success rate bonus
        score += 0.2 * pattern.success_rate

        return score

    def _load_patterns(self):
        """Load all patterns from storage."""
        for file in self.storage_dir.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                pattern = CodePattern(**data)
                self.patterns[pattern.pattern_id] = pattern

    def _save_pattern(self, pattern: CodePattern):
        """Save pattern to storage."""
        file = self.storage_dir / f"{pattern.pattern_id}.json"
        with open(file, "w") as f:
            json.dump(pattern.__dict__, f, indent=2)
```

#### Function Tool Wrappers

```python
# tools/code_pattern_tools.py

from agents import function_tool
from utils.code_pattern_memory import CodePatternMemory, get_pattern_memory

@function_tool
async def store_successful_code_pattern(
    issue: str,
    code_snippet: str,
    effect_type: str,
    improvement: float,
    experiment_id: str,
    parameters: str  # JSON array of parameter names
) -> str:
    """
    Store a successful code pattern for future reuse.

    Call this after a successful experiment to extract the working
    code changes and store them for semantic retrieval.

    Args:
        issue: The issue that was fixed
        code_snippet: The actual Python code that fixed it
        effect_type: Type of effect
        improvement: Score improvement achieved
        experiment_id: ID of the experiment
        parameters: JSON array of parameters affected

    Returns:
        JSON with pattern_id and confirmation
    """
    memory = get_pattern_memory()

    params = json.loads(parameters) if parameters else []

    pattern_id = memory.record_successful_pattern(
        issue=issue,
        code_snippet=code_snippet,
        effect_type=effect_type,
        improvement=improvement,
        experiment_id=experiment_id,
        parameters=params
    )

    return json.dumps({
        "success": True,
        "pattern_id": pattern_id,
        "message": f"Pattern stored: {pattern_id}"
    })


@function_tool
async def find_code_patterns_for_issue(
    issue: str,
    effect_type: str = "",
    min_improvement: float = 5.0
) -> str:
    """
    Find stored code patterns that might help with an issue.

    Searches the pattern library for working code that previously
    fixed similar issues. Returns actual code snippets that can
    be applied to the current script.

    Args:
        issue: Description of the issue to fix
        effect_type: Optional filter by effect type
        min_improvement: Minimum average improvement (default 5 points)

    Returns:
        JSON with list of patterns, each including:
        - pattern_id: Unique identifier
        - code_snippet: Working Python code
        - average_improvement: Historical effectiveness
        - usage_count: How often this pattern has been used
        - success_rate: Percentage of successful applications
    """
    memory = get_pattern_memory()

    patterns = memory.retrieve_patterns_for_issue(
        issue=issue,
        effect_type=effect_type or None,
        min_improvement=min_improvement
    )

    return json.dumps({
        "issue": issue,
        "patterns_found": len(patterns),
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "issue_category": p.issue_category,
                "code_snippet": p.code_snippet,
                "average_improvement": p.average_improvement,
                "usage_count": p.usage_count,
                "success_rate": p.success_rate
            }
            for p in patterns
        ]
    })
```

#### Integration Points

1. **LearningAgent:** Store patterns after successful experiments
2. **ScriptWriter:** Query patterns before generating modifications
3. **Orchestrator:** Consider pattern confidence when selecting approach

---

### Strategy 5: Escape Velocity Mechanism

#### Problem Addressed
When stuck, the agent tries small variations until max iterations. There's no mechanism to escalate exploration aggressiveness over time.

#### Solution
Implement escalating exploration that becomes increasingly aggressive as stuck duration increases.

#### Implementation

**Escape Velocity Levels:**

| Level | Trigger | Action | Description |
|-------|---------|--------|-------------|
| 0 | Normal | Standard modification | Minor parameter adjustment |
| 1 | Same issue 2x | Query knowledge base | Check for known alternatives |
| 2 | Same issue 3x | Force different technique | Don't modify - generate new script |
| 3 | Score plateau 3x | Mine documentation | Search for novel approaches |
| 4 | No progress 4x | Request guidance | Admit defeat or ask human |

**Session State Extension:**

```python
@dataclass
class StuckDetectionState:
    """Track stuck indicators for escape velocity."""
    same_issue_count: int = 0
    last_primary_issue: str = ""
    plateau_count: int = 0
    last_score: float = 0.0
    escape_level: int = 0
    techniques_tried: List[str] = field(default_factory=list)

    def update(self, iteration_result: IterationResult) -> int:
        """Update state and return new escape level."""
        # Check same issue
        if iteration_result.primary_issue == self.last_primary_issue:
            self.same_issue_count += 1
        else:
            self.same_issue_count = 1
            self.last_primary_issue = iteration_result.primary_issue

        # Check plateau
        score_delta = abs(iteration_result.score - self.last_score)
        if score_delta < 3.0:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
        self.last_score = iteration_result.score

        # Determine escape level
        if self.same_issue_count >= 4 or self.plateau_count >= 4:
            self.escape_level = 4
        elif self.same_issue_count >= 3 or self.plateau_count >= 3:
            self.escape_level = 3
        elif self.same_issue_count >= 2:
            self.escape_level = 2
        elif self.plateau_count >= 2:
            self.escape_level = 1
        else:
            self.escape_level = 0

        return self.escape_level
```

**Orchestrator Escape Logic:**

```python
ESCAPE_VELOCITY_INSTRUCTIONS = """
## ESCAPE VELOCITY PROTOCOL

Track your escape_level (0-4) based on stuck indicators.

### Level 0: Normal Operation
- Apply suggested modifications
- Record experiment outcome

### Level 1: Knowledge Check
- Query knowledge base for alternatives
- Check if current approach has historical failures
- If failure rate > 50%, escalate to Level 2

### Level 2: Force Technique Switch
- DO NOT modify current script
- Generate entirely new script with different technique
- Mark current technique as "tried and failed"

### Level 3: Documentation Mining
- Semantic search Blender docs for novel approaches
- Search for approaches NOT in current technique library
- Try first promising novel approach found

### Level 4: Human Escalation
- Report: "Exhausted autonomous options"
- Provide summary of all approaches tried
- Request human guidance or accept best result

### Escape Level Triggers

After each iteration, evaluate:
1. Is primary_issue same as last 2+ iterations? → Escalate
2. Is score delta < 3 points for 2+ iterations? → Escalate
3. Is escape_level already > 0? → Apply corresponding action

### Technique Tracking

Maintain list of tried techniques for current session:
- When switching technique (Level 2+), choose one NOT in list
- If all techniques tried, escalate to Level 3+
"""
```

**Function Tool for Escape Decision:**

```python
@function_tool
async def evaluate_escape_velocity(
    current_issue: str,
    current_score: float,
    iteration_history: str,  # JSON array of {issue, score, technique}
    techniques_available: str  # JSON array of technique names
) -> str:
    """
    Evaluate current stuck state and determine escape action.

    Analyzes iteration history to detect stuck patterns and
    recommends appropriate escape action based on severity.

    Args:
        current_issue: Current primary issue
        current_score: Current quality score
        iteration_history: JSON array of recent iterations
        techniques_available: JSON array of available techniques

    Returns:
        JSON with:
        - escape_level: 0-4 severity level
        - action: Recommended action
        - reason: Why this action was chosen
        - technique_suggestion: If switching, which to try
    """
    history = json.loads(iteration_history)
    techniques = json.loads(techniques_available)

    # Analyze history
    state = StuckDetectionState()
    for iteration in history:
        state.update(IterationResult(
            iteration=iteration.get("iteration", 0),
            score=iteration.get("score", 0),
            passed=iteration.get("passed", False),
            primary_issue=iteration.get("issue", "")
        ))

    # Get tried techniques
    tried = set(h.get("technique", "") for h in history)
    untried = [t for t in techniques if t not in tried]

    # Determine action
    if state.escape_level == 0:
        return json.dumps({
            "escape_level": 0,
            "action": "continue",
            "reason": "Normal progress"
        })

    elif state.escape_level == 1:
        return json.dumps({
            "escape_level": 1,
            "action": "check_knowledge",
            "reason": f"Same issue '{current_issue}' persisting",
            "queries": [
                f"alternative fixes for {current_issue}",
                f"different approach to {current_issue}"
            ]
        })

    elif state.escape_level == 2:
        if untried:
            return json.dumps({
                "escape_level": 2,
                "action": "switch_technique",
                "reason": "Current approach exhausted",
                "technique_suggestion": untried[0],
                "untried_count": len(untried)
            })
        else:
            # All techniques tried, escalate
            state.escape_level = 3

    if state.escape_level == 3:
        return json.dumps({
            "escape_level": 3,
            "action": "mine_documentation",
            "reason": "All known techniques tried",
            "search_queries": [
                f"Blender {current_issue} alternative solutions",
                f"Mantaflow {current_issue} workaround"
            ]
        })

    else:  # Level 4
        return json.dumps({
            "escape_level": 4,
            "action": "request_guidance",
            "reason": "Exhausted autonomous options",
            "summary": {
                "iterations": len(history),
                "techniques_tried": list(tried),
                "best_score": max(h.get("score", 0) for h in history),
                "persistent_issue": current_issue
            }
        })
```

---

## Implementation Priority

### Priority Matrix

| Strategy | Impact | Effort | Priority |
|----------|--------|--------|----------|
| Strategy 3: Proactive Mining | High | Medium | **1st** |
| Strategy 1: Vector Store | High | High | **2nd** |
| Strategy 5: Escape Velocity | Medium | Low | **3rd** |
| Strategy 2: Knowledge Distillation | High | High | **4th** |
| Strategy 4: Code Pattern Memory | Medium | Medium | **5th** |

### Recommended Implementation Order

1. **Phase 1: Quick Wins (1-2 days)**
   - Implement Strategy 3 (Proactive Documentation Mining)
   - Implement Strategy 5 (Escape Velocity)
   - These require minimal infrastructure changes

2. **Phase 2: Vector Store Infrastructure (3-5 days)**
   - Create Blender 5.0 vector store
   - Implement Strategy 1 (Semantic Search)
   - Integrate with DocsExpert

3. **Phase 3: Knowledge Transformation (5-7 days)**
   - Implement Strategy 2 (Knowledge Distillation)
   - Implement Strategy 4 (Code Pattern Memory)
   - Connect to LearningAgent and ScriptWriter

---

## Concrete Implementation Steps

### Step 1: Create Vector Store Setup Script

```bash
# Create scripts directory if needed
mkdir -p scripts

# Create setup script
cat > scripts/create_blender_vector_store.py << 'EOF'
# (Content from Strategy 1)
EOF

# Run setup (requires OPENAI_API_KEY)
python scripts/create_blender_vector_store.py
```

### Step 2: Add Pre-Iteration Research to Orchestrator

Modify `orchestrator.py`:
1. Add `pre_iteration_research` tool import
2. Update `ORCHESTRATOR_INSTRUCTIONS` with early warning detection
3. Call research before each modification

### Step 3: Implement Escape Velocity

Modify `models/shared_context.py`:
1. Add `StuckDetectionState` class
2. Add escape level tracking to `SessionState`

Modify `orchestrator.py`:
1. Add `evaluate_escape_velocity` tool
2. Update instructions with escape protocol

### Step 4: Create Code Pattern Memory

Create new files:
1. `utils/code_pattern_memory.py`
2. `tools/code_pattern_tools.py`

Modify `specialized_agents/learning_agent.py`:
1. Add pattern extraction to post-experiment workflow

### Step 5: Integrate Semantic Search

Create new files:
1. `tools/semantic_docs_tools.py`

Modify `specialized_agents/docs_expert.py`:
1. Add vector store search as primary tool
2. Fall back to keyword search if vector store unavailable

---

## Success Metrics

### Quantitative Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Iterations to pass | ~4.5 | < 3.5 | Average across sessions |
| Same-issue loops | ~2.3 | < 1.5 | Count before technique switch |
| Technique discovery | 0 | > 5/month | New patterns added to library |
| Knowledge reuse | ~15% | > 50% | Patterns applied from library |

### Qualitative Metrics

- Agent tries genuinely different approaches when stuck
- Novel techniques emerge from documentation mining
- Successful fixes are reused in future sessions
- Human intervention rate decreases

---

## Risks and Mitigations

### Risk 1: Vector Store Costs

**Risk:** OpenAI vector store incurs per-query costs.

**Mitigation:**
- Cache common queries locally
- Batch documentation uploads
- Use smaller model for embedding (text-embedding-3-small)
- Set query limits per session

### Risk 2: Pattern Over-Generalization

**Risk:** Extracted patterns might not apply broadly.

**Mitigation:**
- Track success rate per pattern
- Require minimum 2 successful uses before recommending
- Include context (effect_type) in pattern matching
- Decay confidence for old unused patterns

### Risk 3: Escape Velocity Over-Aggressiveness

**Risk:** System might abandon good approaches too quickly.

**Mitigation:**
- Require 2+ iterations before escalating
- Allow step-down if progress resumes
- Log escape decisions for review
- Human override capability

### Risk 4: Documentation Noise

**Risk:** Blender docs contain outdated or irrelevant content.

**Mitigation:**
- Filter to Blender 5.0+ content only
- Prioritize Python API docs over UI tutorials
- Add metadata tags for filtering
- Use high relevance threshold (0.7+)

---

## Appendix A: File Manifest

### New Files to Create

```
scripts/
  create_blender_vector_store.py    # Vector store setup

agents/blender-vfx-orchestrator/
  tools/
    semantic_docs_tools.py           # Strategy 1
    pattern_extractor.py             # Strategy 2
    code_pattern_tools.py            # Strategy 4
  utils/
    code_pattern_memory.py           # Strategy 4
  data/
    code_patterns/                   # Pattern storage directory
```

### Existing Files to Modify

```
agents/blender-vfx-orchestrator/
  orchestrator.py                    # Strategies 3, 5
  models/shared_context.py           # Strategy 5
  specialized_agents/
    docs_expert.py                   # Strategy 1
    learning_agent.py                # Strategy 2
    script_writer.py                 # Strategy 4
```

---

## Appendix B: Environment Variables

```bash
# Required for vector store
BLENDER_DOCS_VECTOR_STORE_ID=vs_xxxxx

# Optional tuning
ESCAPE_VELOCITY_THRESHOLD=2          # Iterations before escalation
PATTERN_MIN_CONFIDENCE=0.5           # Minimum pattern confidence
SEMANTIC_SEARCH_MAX_RESULTS=5        # Results per query
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-16 | Claude Code | Initial draft |

---

*End of Document*
