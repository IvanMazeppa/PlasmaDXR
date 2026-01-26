# Script Writer Overhaul Proposal

**Date:** 2026-01-25
**Status:** PARTIAL ⚠️ - Attribute + enum validation implemented, needs verification
**Problem:** Script Writer hallucinating Blender API attributes despite having doc tools
**Root Cause:** No structural enforcement - agent CAN skip verification

---

## Implementation Status (Updated 2026-01-26 03:40 UTC)

### Test Results

| Test | API Spec Agent | Code Writer | Executor | Overall |
|------|----------------|-------------|----------|---------|
| v5 (00:01) | ❌ doc_ref format wrong | N/A (fallback) | N/A | FAIL |
| v6 (00:12) | ✅ PASSED (18 attrs, 3 ops) | ❌ 1 ops violation | N/A | FAIL |
| v7 (00:39) | ✅ PASSED (7 attrs, 0 ops) | ✅ PASSED (7 attrs) | ⚠️ "ERROR: None" | PARTIAL |
| **v9 (02:45)** | ✅ PASSED (14 attrs, 3 ops) | ✅ PASSED (12 attrs) | ❌ Enum error | **FAIL** |
| **v10 (04:20)** | ❌ Loop detection | ⚠️ fallback | ❌ No camera | **FAIL** |

### v9 Detailed Results (2026-01-26 02:45 UTC)

**Trace:** `traces/e2e_test_v9_20260126_024559.jsonl` (226KB)
**Model:** gpt-5-mini | **Iterations:** 1

| Phase | Duration | Tool Calls | Status |
|-------|----------|------------|--------|
| Research Agent | 51.7s | 3 | ✅ |
| Technique Selector | 26.5s | 0 | ✅ |
| API Spec Agent | 223s | 17 (16 searches + 1 intent) | ✅ |
| Code Writer | 138s | 2 (write + validate) | ✅ |
| API Validator | <1s | 0 | ✅ (3/3 valid) |
| Executor | 36.7s | 2 | ❌ |

**Failure:**
```
TypeError: enum "FLOW" not found in ('INFLOW', 'OUTFLOW', 'GEOMETRY')
fset.flow_behavior = 'FLOW'  # ← Hallucinated value
```

### What's Working

1. **API Spec Agent** - Creates verified APISpec with proper doc_refs
2. **Code Writer guardrail** - Catches any APIs not in the spec
3. **Parallel tool calls** - 14 doc searches batched in single turn (~18s)
4. **Loop detection** - Stays under 20-call limit, no infinite loops
5. **Fallback path** - Falls back to original Script Writer if spec-first fails
6. **Safe ops allowlist** - Common bpy.ops exempted from strict validation

### ✅ Enum Value Validation (Implemented)

**The spec-first pipeline now validates enum values in the Code Writer guardrail.**

- ✅ Guardrail verifies `flow_behavior` attribute exists (doc_ref valid)
- ✅ Guardrail rejects invalid enum values like `'FLOW'`
- ⚠️ Needs verification in a successful test run

**Implemented Fix (Phase 12.1):**
```python
VALID_ENUM_VALUES = {
    "flow_behavior": ["INFLOW", "OUTFLOW", "GEOMETRY"],
    "flow_type": ["SMOKE", "FIRE", "BOTH"],
    "domain_type": ["GAS", "LIQUID"],
    # ... more enums
}
```

### Remaining Issues

1. **Enum validation verification** - Needs a successful spec-first run
2. **Spec-first doc search stabilization** - Bundle-first plan added; needs verification
3. **Render pipeline camera fix** - API fixer + instructions added; needs verification
4. **Executor error reporting** - Reports "ERROR: None" even when script succeeds
5. **Render generation** - Test scripts don't produce renders (setup only)

### Files Modified

- `specialized_agents/api_spec_agent.py` - Bundle-first doc search plan + enum extraction rules
- `guardrails/api_spec_guardrails.py` - Expanded safe_ops allowlist, added debug output, enum validation
- `hooks/enforcement_hooks.py` - Increased limits, added fallback hooks
- `tools/blender_api_fixer.py` - Inject camera setup when render has no camera
- `hooks/__init__.py` - Exported new hook function
- `orchestrator.py` - Uses fallback hooks for original Script Writer

### Documentation Updated (2026-01-26)

- `docs/VERSION_TRUTH.md` - Added VALID_ENUM_VALUES, INVALID ENUM VALUES, output file locations
- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md` - Added v9 findings, enum hallucination blocker
- `docs/ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md` - Phase 12 marked PARTIAL, added test findings

---

## Executive Summary

The Script Writer agent has access to documentation tools but **nothing forces it to use them**. The current architecture relies on:
1. Prompt instructions (ignored)
2. Post-hoc validation (catches only known errors)
3. Execution failure (too late)

**The fix is not better prompts or more corrections. The fix is making hallucination structurally impossible.**

This document proposes splitting the Script Writer into a **Spec-First Pipeline** where API verification happens BEFORE code generation, enforced by SDK guardrails.

**Tightened requirements in this proposal:**
- Doc refs must be **API docs** (not manual) and include the attribute/operation name.
- `bpy.ops.*` calls are explicitly verified, not just attributes.
- Code guardrail is deterministic by enforcing `dset`/`fset` variable naming.
- Guardrail examples follow SDK requirements (always return `output_info`).

---

## Problem Assessment

### Current Failure Mode

```
Script Writer receives: "Create fire effect"
Script Writer thinks: "I know FluidDomainSettings has timesteps_per_frame"  ← WRONG
Script Writer writes: dsettings.timesteps_per_frame = 5
API Validator checks: Not in KNOWN_API_CHANGES → assumes valid  ← WRONG
Executor runs: AttributeError: 'FluidDomainSettings' has no attribute 'timesteps_per_frame'
```

### Evidence from E2E Tests (2026-01-25)

| Run | Hallucinated Attribute | Correct Attribute | Result |
|-----|------------------------|-------------------|--------|
| 18:25 | `noise_scale = 1.0` | `noise_scale = 1` (int) | TypeError |
| 19:10 | `timesteps_per_frame` | `timesteps_maximum` | AttributeError |
| Earlier | `use_adaptive_time_steps` | `use_adaptive_timesteps` | AttributeError |
| Earlier | `resolution_divisions` | `resolution_max` | AttributeError |

**Pattern:** The agent invents plausible-sounding names instead of verifying against docs.

### Why Other LLMs Don't Have This Problem

When Claude or GPT-5 works in an IDE context:
1. They READ files before writing
2. They SEARCH docs when uncertain
3. They have an error feedback loop
4. They VERIFY before committing

The Script Writer agent skips all of this because **nothing enforces it**.

---

## Root Cause Analysis

### 1. Tools Available But Not Required

**File:** `orchestrator.py` (lines 1096-1110)

```python
self._script_writer = Agent[SharedContext](
    name="Script Writer",
    tools=[
        write_script,
        validate_script,
        modify_script,
        semantic_search_blender_docs,      # ← HAS THIS
        search_blender_api_by_intent,      # ← HAS THIS
    ],
    # No mechanism to REQUIRE tool usage before write_script
)
```

The agent HAS doc tools but CAN call `write_script` without using them.

### 2. RunHooks Enforcement Was Weakened

**File:** `hooks/enforcement_hooks.py`

```python
require_doc_query_before=[
    # NOTE: write_script and generate_script NO LONGER require doc query
    # because Research Agent already did the documentation research
],
```

This assumption is FALSE. Research Agent provides high-level approach, not attribute-level verification.

### 3. Validation Assumes Valid By Default

**File:** `specialized_agents/api_validator.py`

```python
validation = APICallValidation(
    api_call=call,
    is_valid=True,        # ← DEFAULT IS TRUE
    confidence=0.5,
)

# Only marks invalid if in KNOWN_API_CHANGES
for pattern, fix in KNOWN_API_CHANGES.items():
    if pattern in call:
        validation.is_valid = False
```

Unknown attributes are assumed valid. This is backwards.

### 4. Output Guardrail Checks Format, Not Content

**File:** `guardrails/script_guardrails.py`

```python
@output_guardrail
async def validate_script_output(ctx, agent, output):
    # Checks: script_path exists, technique_used provided
    # Does NOT check: Are the API calls in the script valid?
```

---

## Proposed Solution: Spec-First Pipeline

### Architecture Overview

```
CURRENT (broken):
┌─────────────────────────────────────────────────────────┐
│  Script Writer                                           │
│  - Receives research summary                             │
│  - Generates code from "memory"                          │
│  - Optionally uses doc tools (usually doesn't)           │
│  - Outputs code with hallucinated attributes             │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  API Validator                                           │
│  - Checks against known-bad list (incomplete)            │
│  - Assumes valid if not known-bad                        │
└─────────────────────────────────────────────────────────┘
          │
          ▼
     AttributeError


PROPOSED (spec-first):
┌─────────────────────────────────────────────────────────┐
│  PHASE A: API Spec Agent                                 │
│  - Receives research summary                             │
│  - MUST call doc tools (enforced by RunHooks)            │
│  - Outputs APISpec with verified attributes only         │
│  - Output guardrail rejects if any attr lacks doc_ref    │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE B: Code Writer Agent                              │
│  - Receives VERIFIED APISpec                             │
│  - Can ONLY use attributes in the spec                   │
│  - Output guardrail parses code, rejects unlisted attrs  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
     Verified Code (hallucination impossible)
```

### Key Insight

The agent cannot hallucinate an attribute that isn't in the verified spec. And the spec cannot include an attribute without a doc reference. This creates a **closed system** where hallucination is structurally impossible.

---

## Implementation Details

### 1. APISpec Schema (Pydantic)

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

class APIAttribute(BaseModel):
    """A single verified API attribute."""
    object_type: str = Field(description="e.g., FluidDomainSettings")
    attribute_name: str = Field(description="e.g., resolution_max")
    value_type: str = Field(description="e.g., int, float, bool, enum")
    doc_ref: str = Field(description="API documentation path - REQUIRED (python reference)")
    example_value: Optional[str] = Field(default=None, description="e.g., '128' or 'True'")

class APIOperation(BaseModel):
    """A verified bpy.ops call."""
    op_path: str = Field(description="e.g., bpy.ops.fluid.bake_data")
    doc_ref: str = Field(description="API documentation path - REQUIRED (python reference)")

class APISpec(BaseModel):
    """Verified API specification for script generation.

    Every attribute MUST have a doc_ref from Blender 5.0 documentation.
    This is enforced by output guardrail.
    """
    effect_type: str
    technique: str

    # Core API usage
    domain_attributes: List[APIAttribute] = Field(
        description="FluidDomainSettings attributes to use"
    )
    flow_attributes: List[APIAttribute] = Field(
        description="FluidFlowSettings attributes to use"
    )
    ops: List[APIOperation] = Field(
        default_factory=list,
        description="bpy.ops.* calls used by the script"
    )
    shader_nodes: List[str] = Field(
        default_factory=list,
        description="Shader node types to create"
    )

    # Documentation grounding
    approach_doc_refs: List[str] = Field(
        description="General approach documentation references"
    )

    @field_validator('domain_attributes', 'flow_attributes')
    @classmethod
    def all_attributes_have_doc_refs(cls, v):
        for attr in v:
            if not attr.doc_ref or attr.doc_ref.strip() == "":
                raise ValueError(f"Attribute {attr.attribute_name} missing doc_ref")
            if not attr.doc_ref.startswith("blender_python_reference_5_0/"):
                raise ValueError(f"{attr.attribute_name} doc_ref must be API doc path")
            if attr.attribute_name.lower() not in attr.doc_ref.lower():
                raise ValueError(f"{attr.attribute_name} not found in doc_ref")
        return v

    @field_validator('ops')
    @classmethod
    def all_ops_have_doc_refs(cls, v):
        for op in v:
            if not op.doc_ref or op.doc_ref.strip() == "":
                raise ValueError(f"Operation {op.op_path} missing doc_ref")
            if not op.op_path.startswith("bpy.ops."):
                raise ValueError(f"Operation must start with bpy.ops: {op.op_path}")
            if not op.doc_ref.startswith("blender_python_reference_5_0/"):
                raise ValueError(f"{op.op_path} doc_ref must be API doc path")
        return v
```

### 2. API Spec Agent

```python
from agents import Agent, AgentOutputSchema, Runner, function_tool
from agents import output_guardrail, GuardrailFunctionOutput

# Output guardrail ensures all attributes are verified
@output_guardrail
async def validate_api_spec(ctx, agent, output: APISpec) -> GuardrailFunctionOutput:
    """Verify every attribute has a valid doc_ref."""
    errors = []

    for attr in output.domain_attributes + output.flow_attributes:
        if not attr.doc_ref:
            errors.append(f"{attr.object_type}.{attr.attribute_name}: missing doc_ref")
        elif not attr.doc_ref.startswith("blender_python_reference_5_0/"):
            errors.append(f"{attr.object_type}.{attr.attribute_name}: doc_ref must be API doc path")
        elif attr.attribute_name.lower() not in attr.doc_ref.lower():
            errors.append(f"{attr.object_type}.{attr.attribute_name}: doc_ref missing attribute name")

    for op in output.ops:
        if not op.doc_ref:
            errors.append(f"{op.op_path}: missing doc_ref")
        elif not op.doc_ref.startswith("blender_python_reference_5_0/"):
            errors.append(f"{op.op_path}: doc_ref must be API doc path")

    if errors:
        return GuardrailFunctionOutput(
            output_info={"errors": errors, "reason": "Unverified attributes"},
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info={
            "verified_count": len(output.domain_attributes) + len(output.flow_attributes),
            "verified_ops": len(output.ops),
        },
        tripwire_triggered=False,
    )


api_spec_agent = Agent[SharedContext](
    name="API Spec Agent",
    instructions="""ROLE: Create verified API specification for Blender script.
INPUTS: effect_type, description, research_summary.
TOOLS: semantic_search_blender_docs, search_blender_api_by_intent, validate_parameter_range.
OUTPUT: APISpec (Pydantic schema with doc_refs for EVERY attribute).

## CRITICAL RULES
1. You MUST call semantic_search_blender_docs for EACH attribute you plan to use.
2. You MUST include the doc_ref from the search result.
3. Doc refs MUST be API docs (`blender_python_reference_5_0/...`), not manual pages.
4. You MUST include `bpy.ops.*` calls in the spec with doc refs (e.g., bake ops).
5. If you cannot find documentation for an attribute or op, DO NOT include it.
6. NEVER guess attribute names - only use EXACT names from documentation.

## Turn Budget
T1: semantic_search_blender_docs for domain settings (FluidDomainSettings)
T2: semantic_search_blender_docs for flow settings (FluidFlowSettings)
T3: search_blender_api_by_intent for any uncertain APIs
T4: Return APISpec with all verified attributes

## Output Contract
Every attribute in domain_attributes and flow_attributes MUST have:
- object_type: Exact class name from docs
- attribute_name: Exact attribute name from docs (CASE SENSITIVE)
- value_type: From docs (int, float, bool, enum)
- doc_ref: API doc path (REQUIRED - guardrail will reject without this)
Every op in ops MUST have:
- op_path: Exact bpy.ops path (e.g., bpy.ops.fluid.bake_data)
- doc_ref: API doc path (REQUIRED)

STOP after T4. Do NOT invent attributes.""",

    model="gpt-5.2",
    model_settings=ModelSettings(reasoning={"effort": "high"}),
    output_type=AgentOutputSchema(APISpec, strict_json_schema=False),
    output_guardrails=[validate_api_spec],
    tools=[
        semantic_search_blender_docs,
        search_blender_api_by_intent,
        validate_parameter_range,
    ],
)
```

### 3. Code Writer Agent

```python
from agents import Agent, AgentOutputSchema
import re

class VerifiedScriptOutput(BaseModel):
    """Script output that has been verified against API spec."""
    script_path: str
    technique_used: str
    apis_used: List[str] = Field(description="List of APIs used (must match spec)")


@output_guardrail
async def validate_code_against_spec(ctx, agent, output: VerifiedScriptOutput) -> GuardrailFunctionOutput:
    """Ensure code only uses APIs from the verified spec."""

    # Get the API spec from context (passed from previous phase)
    api_spec: APISpec = ctx.context.api_spec

    # Read the generated script
    script_content = Path(output.script_path).read_text()

    # Extract attribute assignments for enforced variable names
    # Required convention: dset = domain_settings, fset = flow_settings
    domain_pattern = r'\bdset\.(\w+)\s*='
    flow_pattern = r'\bfset\.(\w+)\s*='
    used_attributes = set(re.findall(domain_pattern, script_content))
    used_attributes.update(re.findall(flow_pattern, script_content))

    # Build set of allowed attributes from spec
    allowed_attributes = set()
    for attr in api_spec.domain_attributes + api_spec.flow_attributes:
        allowed_attributes.add(attr.attribute_name)

    # Extract bpy.ops calls
    ops_pattern = r'\bbpy\.ops\.[a-zA-Z_]+\.[a-zA-Z_]+'
    used_ops = set(re.findall(ops_pattern, script_content))

    # Find violations
    violations = used_attributes - allowed_attributes
    allowed_ops = {op.op_path for op in api_spec.ops}
    op_violations = used_ops - allowed_ops

    if violations or op_violations:
        return GuardrailFunctionOutput(
            output_info={
                "violations": list(violations),
                "op_violations": list(op_violations),
                "reason": "Code uses attributes or ops not in verified spec",
                "allowed": list(allowed_attributes),
                "allowed_ops": list(allowed_ops),
            },
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info={"apis_verified": len(used_attributes), "ops_verified": len(used_ops)},
        tripwire_triggered=False,
    )


code_writer_agent = Agent[SharedContext](
    name="Code Writer",
    instructions="""ROLE: Write Blender Python code using ONLY verified API attributes.
INPUTS: APISpec (verified attributes with doc_refs).
TOOLS: write_script, validate_script.
OUTPUT: VerifiedScriptOutput.

## CRITICAL RULES
1. You MUST ONLY use attributes listed in the APISpec.
2. Do NOT invent or guess ANY attribute names.
3. Copy attribute names EXACTLY from the spec (case-sensitive).
4. If you need an attribute not in the spec, STOP and report it.
5. Use variable names `dset` (domain_settings) and `fset` (flow_settings) so guardrails can verify usage.

## Turn Budget
T1: Write complete Blender Python code using ONLY spec attributes
T2: write_script(code=..., output_name=..., technique_name=...)
T3: validate_script(script_path)
T4: Return VerifiedScriptOutput

## Attribute Usage
For each attribute, copy EXACTLY from APISpec:
- domain_attributes: Use for FluidDomainSettings
- flow_attributes: Use for FluidFlowSettings
- shader_nodes: Use for node creation
- ops: Use ONLY the ops listed in APISpec

NEVER write an attribute not in the spec. The output guardrail will reject it.""",

    model="gpt-5.2",
    model_settings=ModelSettings(reasoning={"effort": "high"}),
    output_type=AgentOutputSchema(VerifiedScriptOutput, strict_json_schema=False),
    output_guardrails=[validate_code_against_spec],
    tools=[
        write_script,
        validate_script,
    ],
)
```

### 4. Pipeline Integration

```python
async def create_asset_pipeline_v2(self, request: AssetRequest) -> SessionState:
    """Spec-first pipeline that prevents hallucination."""

    # ... existing setup ...

    # PHASE 0: Research (unchanged)
    research_result = await Runner.run(
        self._research_agent,
        research_prompt,
        context=context,
        max_turns=4,
    )
    research_output: ResearchOutput = research_result.final_output

    # PHASE 0.5: Technique Selection (unchanged)
    # ...

    # PHASE 1A: API Specification (NEW)
    # Agent MUST verify every attribute against docs
    spec_prompt = f"""Create API specification for {request.effect_type} effect.

Research summary:
{research_output.recommended_approach}

Key parameters from research:
{research_output.key_parameters}

You MUST:
1. Search docs for EACH FluidDomainSettings attribute you plan to use
2. Search docs for EACH FluidFlowSettings attribute you plan to use
3. Include doc_ref for EVERY attribute
4. Only include attributes you found in Blender 5.0 documentation"""

    try:
        spec_result = await Runner.run(
            self._api_spec_agent,
            spec_prompt,
            context=context,
            max_turns=6,
        )
        api_spec: APISpec = spec_result.final_output

        # Store in context for code writer guardrail
        context.api_spec = api_spec

        print(f"[Pipeline] API Spec verified: {len(api_spec.domain_attributes)} domain attrs, "
              f"{len(api_spec.flow_attributes)} flow attrs", file=sys.stderr)

    except OutputGuardrailTripwireTriggered as e:
        # Spec failed verification - cannot proceed
        print(f"[Pipeline] API Spec REJECTED: {e.guardrail_result.output_info}", file=sys.stderr)
        return self._fail_session(session, "API specification failed verification")

    # PHASE 1B: Code Generation (using verified spec)
    code_prompt = f"""Write Blender Python script using ONLY these verified attributes:

Domain attributes (FluidDomainSettings):
{json.dumps([attr.model_dump() for attr in api_spec.domain_attributes], indent=2)}

Flow attributes (FluidFlowSettings):
{json.dumps([attr.model_dump() for attr in api_spec.flow_attributes], indent=2)}

Ops (bpy.ops calls):
{json.dumps([op.model_dump() for op in api_spec.ops], indent=2)}

COPY ATTRIBUTE NAMES EXACTLY. Do not invent any attributes."""

    try:
        code_result = await Runner.run(
            self._code_writer_agent,
            code_prompt,
            context=context,
            max_turns=8,
        )
        script_output: VerifiedScriptOutput = code_result.final_output

    except OutputGuardrailTripwireTriggered as e:
        # Code used unverified attributes
        violations = e.guardrail_result.output_info.get("violations", [])
        print(f"[Pipeline] Code REJECTED - used unverified attrs: {violations}", file=sys.stderr)
        return self._fail_session(session, f"Code used unverified attributes: {violations}")

    # PHASE 2: Execution (unchanged, but now with verified code)
    # ...
```

---

## SDK Alignment

### OpenAI Agents SDK Patterns Used

**Reference:** https://github.com/openai/openai-agents-python/tree/main/docs

#### 1. Structured Output (`output_type`)

From [agents.md](https://github.com/openai/openai-agents-python/blob/main/docs/agents.md):

```python
from agents import Agent, AgentOutputSchema
from pydantic import BaseModel

class APISpec(BaseModel):
    domain_attributes: List[APIAttribute]
    # ...

agent = Agent(
    output_type=AgentOutputSchema(APISpec, strict_json_schema=False),
)

result = await Runner.run(agent, prompt)
api_spec: APISpec = result.final_output  # Type-safe!
```

#### 2. Output Guardrails

From [guardrails.md](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md):

```python
from agents import output_guardrail, GuardrailFunctionOutput

@output_guardrail
async def validate_api_spec(ctx, agent, output: APISpec) -> GuardrailFunctionOutput:
    """Output guardrails validate agent output before it's accepted."""
    if not all(attr.doc_ref for attr in output.domain_attributes):
        return GuardrailFunctionOutput(
            output_info={"reason": "Missing doc_refs"},
            tripwire_triggered=True,  # Rejects the output
        )
    return GuardrailFunctionOutput(
        output_info={"status": "passed"},
        tripwire_triggered=False
    )

agent = Agent(
    output_guardrails=[validate_api_spec],
)
```

#### 2.1 Enforcing Doc Tool Usage (SDK-correct)
Use RunHooks or `ModelSettings.tool_choice` to ensure a doc tool is called before any spec output.
This prevents the spec agent from skipping doc tools.

```python
from agents import ModelSettings, RunHooks

class RequireDocsFirst(RunHooks):
    def __init__(self):
        self.doc_query_made = False

    async def on_tool_start(self, context, agent, tool):
        if tool.name in {"semantic_search_blender_docs", "search_blender_api_by_intent"}:
            self.doc_query_made = True
    async def on_agent_end(self, context, agent, output):
        if not self.doc_query_made:
            raise DocQueryRequiredError("Doc query required before spec output")

api_spec_agent = Agent(
    model_settings=ModelSettings(tool_choice="semantic_search_blender_docs"),
)

spec_result = await Runner.run(api_spec_agent, spec_prompt, hooks=RequireDocsFirst())
```

#### 3. Context Passing

From [context.md](https://github.com/openai/openai-agents-python/blob/main/docs/context.md):

```python
# Store verified spec in context for downstream agents
context.api_spec = api_spec

# Code writer guardrail accesses it
@output_guardrail
async def validate_code_against_spec(ctx, agent, output):
    api_spec = ctx.context.api_spec  # Access from context
```

#### 4. Exception Handling

From [agents.md](https://github.com/openai/openai-agents-python/blob/main/docs/agents.md):

```python
from agents import OutputGuardrailTripwireTriggered

try:
    result = await Runner.run(agent, prompt)
except OutputGuardrailTripwireTriggered as e:
    # Guardrail rejected the output
    errors = e.guardrail_result.output_info
    # Handle gracefully
```

---

## Alternative Approaches Considered

### Option B: RunHooks Enforcement (Simpler but Weaker)

```python
class StrictScriptWriterHooks(RunHooks):
    """Enforce doc verification before write_script."""

    def __init__(self):
        self.verified_attributes = set()

    async def on_tool_start(self, context, agent, tool):
        if tool.name == "write_script":
            code = tool.args.get("code", "")
            attrs_used = self._extract_attributes(code)
            unverified = attrs_used - self.verified_attributes

            if unverified:
                raise VerificationRequiredError(
                    f"BLOCKED: {unverified} not verified. "
                    f"Call verify_attribute() for each first."
                )

        if tool.name == "verify_attribute":
            attr = tool.args.get("attribute")
            # ... verification logic ...
            self.verified_attributes.add(attr)
```

**Pros:** Simpler, modifies existing agent
**Cons:** Agent can still hallucinate in the verify call; relies on prompt compliance

### Option C: Template-Based Generation (Most Restrictive)

```python
class FluidDomainConfig(BaseModel):
    """Pre-defined valid attributes - agent cannot invent new ones."""
    resolution_max: int = Field(ge=32, le=512)
    use_adaptive_timesteps: bool = True
    timesteps_maximum: int = Field(ge=1, le=100, default=4)
    # ... all valid Blender 5.0 attributes ...

# Agent fills template, cannot add attributes
```

**Pros:** Hallucination literally impossible
**Cons:** Inflexible, requires maintaining complete attribute list

---

## Comparison Matrix

| Criteria | Current | Option A (Spec-First) | Option B (RunHooks) | Option C (Templates) |
|----------|---------|----------------------|---------------------|---------------------|
| Prevents hallucination | No | Yes | Partial | Yes |
| SDK-aligned | Yes | Yes | Yes | Yes |
| Flexibility | High | High | High | Low |
| Implementation effort | - | Medium | Low | Medium |
| Maintenance burden | High (whack-a-mole) | Low | Medium | High |
| Guardrail enforcement | None | Strong | Medium | N/A |

**Recommendation:** Option A (Spec-First) provides the best balance of prevention strength and flexibility.

---

## Implementation Plan

### Phase 1: Core Infrastructure (1-2 days)

1. Create `APISpec` and `APIAttribute` Pydantic models
2. Create `validate_api_spec` output guardrail
3. Create `APISpecAgent` with structured output
4. Test spec generation in isolation

### Phase 2: Code Writer Integration (1 day)

1. Create `validate_code_against_spec` output guardrail
2. Modify Code Writer to receive APISpec
3. Update pipeline to use two-phase flow
4. Test end-to-end

### Phase 3: Migration (1 day)

1. Run parallel with old pipeline
2. Compare results
3. Switch over when stable
4. Document changes

---

## Success Criteria

1. **Zero AttributeErrors** from hallucinated attributes
2. **100% doc_ref coverage** in APISpec output
3. **Code guardrail catches** any unlisted attribute usage
4. **Trace shows** doc queries for each attribute in spec

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Doc search doesn't find valid attributes | Add fallback to API store; expand search queries |
| Spec phase adds latency | Acceptable tradeoff for correctness; cache common specs |
| Code writer still hallucinates | Guardrail catches it; strong prompt + examples |
| Breaking change to pipeline | Feature flag; gradual rollout |

---

## Conclusion

The current architecture cannot prevent hallucination because:
1. Verification is optional (prompt-based)
2. Validation assumes valid by default
3. No structural gate between "planning" and "writing"

The Spec-First approach makes hallucination **structurally impossible** by:
1. Requiring doc_ref for every attribute (guardrail-enforced)
2. Separating specification from implementation
3. Verifying code against spec before acceptance

This aligns with OpenAI Agents SDK patterns for structured output, guardrails, and context passing.

---

## Appendix: Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `models/api_spec.py` | CREATE | APISpec and APIAttribute Pydantic models |
| `guardrails/api_spec_guardrails.py` | CREATE | validate_api_spec, validate_code_against_spec |
| `specialized_agents/api_spec_agent.py` | CREATE | API Spec Agent definition |
| `specialized_agents/code_writer_agent.py` | CREATE | Code Writer Agent definition |
| `orchestrator.py` | MODIFY | Add spec-first pipeline flow |
| `tools/dynamic_instructions.py` | MODIFY | Add API Spec Agent instructions |

---

*End of Proposal*
