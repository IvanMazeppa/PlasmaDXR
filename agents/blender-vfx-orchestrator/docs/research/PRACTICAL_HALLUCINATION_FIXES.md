# Practical Hallucination Fixes for Developers

**Source:** [Reducing LLM Hallucinations - Zep Guide](https://www.getzep.com/ai-agents/reducing-llm-hallucinations/)
**Relevance:** Developer-focused techniques immediately applicable to your system

---

## The Core Problem

LLMs fabricate plausible-sounding content when they:
1. Don't have the information in training data
2. Have conflicting information (multiple Blender versions)
3. Are asked for specifics they can only guess at

**Your Case:** Blender 5.0 attributes didn't exist when models were trained. The model guesses based on patterns from Blender 2.x-4.x.

---

## Technique 1: Retrieval-Augmented Generation (RAG)

### What It Does

RAG "anchors responses to real sources" by fetching relevant reference data **before** generation.

### Your Current Implementation

```
User Request → semantic_search_blender_docs() → Results → ScriptWriter → Script
```

### The Problem

Agent can **ignore** search results or **not search at all**.

### The Fix: Make RAG Authoritative

```python
SCRIPTWRITER_INSTRUCTIONS = """
You are a Blender 5.0 script generator.

CRITICAL: The semantic_search results are the ONLY source of truth.
- If an attribute is NOT in the search results, it DOES NOT EXIST
- Never infer or guess attributes based on naming patterns
- Never use attributes you "remember" from training - they may be outdated
- If you need an attribute not in results, call semantic_search_blender_docs() again

When writing bpy.* calls:
1. VERIFY each attribute exists in search results
2. VERIFY the type matches (int vs float vs bool)
3. VERIFY the parent class is correct (DomainSettings vs FlowSettings)
4. If ANY verification fails, STOP and search for the correct attribute
"""
```

---

## Technique 2: Explicit "I Don't Know" Training

### What It Does

Teaches the model when to refuse rather than guess.

### Implementation

Add few-shot examples to ScriptWriter prompt:

```python
FEW_SHOT_EXAMPLES = """
<example>
User: Set the resolution_divisions to 128
Assistant: I cannot use 'resolution_divisions' because:
1. I searched for this attribute and it does not exist in Blender 5.0
2. The correct attribute is 'resolution_max' (int, 32-512)
3. I will use: domain_settings.resolution_max = 128
</example>

<example>
User: Enable use_adaptive_time_steps
Assistant: I cannot use 'use_adaptive_time_steps' because:
1. This attribute has a typo - Blender uses underscores differently
2. The correct attribute is 'use_adaptive_timesteps' (bool)
3. I will use: domain_settings.use_adaptive_timesteps = True
</example>

<example>
User: Set velocity_multi to 5.0
Assistant: I cannot use 'velocity_multi' because:
1. This attribute does not exist in FluidFlowSettings
2. The correct attribute is 'velocity_factor' (float, 0-1000)
3. I will use: flow_settings.velocity_factor = 5.0
</example>
"""
```

---

## Technique 3: Confidence Scoring

### What It Does

Inspects token probabilities to gauge if model was uncertain. Flat probability distribution = guessing.

### Implementation for Script Generation

```python
from pydantic import BaseModel
from typing import List

class APICallWithConfidence(BaseModel):
    """Single API call with confidence score."""
    code: str
    attribute: str
    confidence: float  # 0.0 to 1.0
    verification_status: str  # "verified", "unverified", "guessed"

class ScriptWithConfidence(BaseModel):
    """Script output with per-call confidence."""
    script: str
    api_calls: List[APICallWithConfidence]
    lowest_confidence: float
    recommendation: str

SCRIPTWRITER_INSTRUCTIONS = """
...existing instructions...

OUTPUT FORMAT:
For each bpy.* call you write, provide:
1. The code line
2. Your confidence (0.0-1.0) that this API exists in Blender 5.0
3. Verification status:
   - "verified" if you found it in semantic_search results
   - "unverified" if you couldn't verify but believe it exists
   - "guessed" if you're inferring from patterns

If ANY call has confidence < 0.8 or status != "verified":
- Flag it in your output
- Explain what search you would need to verify it
"""
```

### Using Confidence in Pipeline

```python
async def generate_with_confidence_check(prompt: str) -> str:
    result = await Runner.run(script_writer, prompt)
    output: ScriptWithConfidence = result.final_output

    # Check for low-confidence calls
    risky_calls = [c for c in output.api_calls if c.confidence < 0.8]

    if risky_calls:
        # Force verification
        for call in risky_calls:
            verified = await verify_api_call(call.attribute)
            if not verified:
                raise HallucinationDetected(f"Unverified API: {call.attribute}")

    return output.script
```

---

## Technique 4: Chain-of-Thought Verification

### What It Does

Forces step-by-step reasoning that exposes errors before final output.

### Implementation

```python
COT_VERIFICATION_PROMPT = """
Before writing the script, complete this verification chain:

STEP 1: List all attributes I plan to use
- Attribute 1: [name] on [class]
- Attribute 2: [name] on [class]
...

STEP 2: For each attribute, answer:
- Did I find this in semantic_search results? [YES/NO]
- If NO, what similar attributes exist?
- What is the correct type? [int/float/bool/enum]

STEP 3: Corrections needed
- [List any attributes that need correction]

STEP 4: Final attribute list
- [Only verified attributes]

STEP 5: Generate script using ONLY Step 4 attributes
"""
```

---

## Technique 5: Self-Consistency Sampling

### What It Does

Generate multiple outputs and keep only what appears in majority.

### Implementation

```python
async def generate_with_consistency_check(
    prompt: str,
    num_samples: int = 3,
    agreement_threshold: float = 0.67
) -> str:
    """Generate multiple scripts and extract consensus API calls."""

    # Generate multiple variations
    scripts = []
    for _ in range(num_samples):
        result = await Runner.run(script_writer, prompt, temperature=0.7)
        scripts.append(result.final_output)

    # Extract API calls from each
    all_api_calls = [extract_api_calls(s) for s in scripts]

    # Find consensus (appears in 2+ of 3 scripts)
    api_counts = Counter(call for calls in all_api_calls for call in calls)
    min_agreement = int(num_samples * agreement_threshold)
    consensus_apis = {api for api, count in api_counts.items() if count >= min_agreement}

    # Regenerate with only consensus APIs
    constrained_prompt = f"""
    {prompt}

    CONSTRAINT: You may ONLY use these verified API calls:
    {list(consensus_apis)}

    Any other API calls will cause execution failure.
    """

    final_result = await Runner.run(script_writer, constrained_prompt, temperature=0.0)
    return final_result.final_output
```

---

## Technique 6: Post-Processing Verification

### What It Does

Rule-based filtering after generation to catch obvious errors.

### Implementation

```python
import re
from typing import Tuple, List

# Known hallucination patterns
HALLUCINATION_PATTERNS = [
    (r"resolution_divisions", "resolution_max"),
    (r"use_adaptive_time_steps", "use_adaptive_timesteps"),
    (r"velocity_multi", "velocity_factor"),
    (r"\.time_scale\s*=", "# time_scale removed - use timesteps_maximum"),
    (r"absolute_density", "density + use_absolute flag"),
    (r"noise_scale\s*=\s*\d+\.\d+", "noise_scale = int(...)"),  # float to int
]

def verify_and_correct_script(script: str) -> Tuple[str, List[str]]:
    """
    Post-process script to catch and correct hallucinations.

    Returns: (corrected_script, list_of_corrections)
    """
    corrections = []

    for pattern, replacement in HALLUCINATION_PATTERNS:
        if re.search(pattern, script):
            corrections.append(f"Found '{pattern}' → corrected to '{replacement}'")

            # Apply correction
            if "noise_scale" in pattern:
                # Special handling for type conversion
                script = re.sub(
                    r"noise_scale\s*=\s*(\d+)\.\d+",
                    r"noise_scale = \1  # Converted to int",
                    script
                )
            else:
                script = re.sub(pattern, replacement, script)

    return script, corrections

# Use in pipeline
script = await generate_script(prompt)
corrected_script, corrections = verify_and_correct_script(script)

if corrections:
    log.warning(f"Auto-corrected {len(corrections)} hallucinations: {corrections}")
```

---

## Technique 7: Structured Output Constraints

### What It Does

Force output into schema that prevents certain hallucination types.

### Implementation

```python
from pydantic import BaseModel, validator
from typing import Literal

class VerifiedAttribute(BaseModel):
    """Attribute that must be verified."""
    name: str
    parent_class: Literal[
        "FluidDomainSettings",
        "FluidFlowSettings",
        "FluidEffectorSettings",
        "Scene",
        "Object"
    ]
    value_type: Literal["int", "float", "bool", "str", "enum"]
    value: str
    verification_source: str  # Must cite search result

    @validator("name")
    def check_not_hallucinated(cls, v):
        KNOWN_HALLUCINATIONS = [
            "resolution_divisions",
            "use_adaptive_time_steps",
            "velocity_multi",
            "time_scale",
        ]
        if v in KNOWN_HALLUCINATIONS:
            raise ValueError(f"'{v}' is a known hallucination. Use verified attribute.")
        return v

class StructuredScript(BaseModel):
    """Script with verified attributes only."""
    attributes_used: List[VerifiedAttribute]
    script_code: str

    @validator("script_code")
    def check_only_verified_attrs(cls, v, values):
        # Ensure script only uses declared attributes
        declared = {a.name for a in values.get("attributes_used", [])}
        used = extract_attribute_names(v)

        undeclared = used - declared
        if undeclared:
            raise ValueError(f"Script uses undeclared attributes: {undeclared}")
        return v
```

---

## Summary: Implementation Priority

| Priority | Technique | Effort | Impact |
|----------|-----------|--------|--------|
| P0 | Post-processing verification | Low | High - catches known patterns |
| P0 | Explicit "I don't know" examples | Low | Medium - prevents guessing |
| P1 | Make RAG authoritative | Medium | High - prevents ignoring docs |
| P1 | Confidence scoring | Medium | High - flags uncertainty |
| P2 | Chain-of-thought verification | Medium | Medium - exposes reasoning |
| P2 | Self-consistency sampling | High | Medium - expensive but robust |
| P3 | Structured output constraints | High | High - enforces at schema level |

**Start with P0 techniques** - they're low effort and catch the most common issues you're seeing.
