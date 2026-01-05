# Blender Librarian API Validation Report

**Date:** 2026-01-05
**Purpose:** Validate `plans/feat-blender-librarian-FINAL.md` against local OpenAI API documentation
**Status:** CORRECTIONS REQUIRED

---

## Summary

After reviewing the local OpenAI API documentation mirror against the Blender Librarian implementation plan, I found **3 issues** that need correction:

| Issue | Severity | Status |
|-------|----------|--------|
| Vision input format incorrect | HIGH | Fix Required |
| Missing GPT-5.2 optimization parameters | MEDIUM | Enhancement |
| Response usage tracking uncertain | LOW | Verify at runtime |

---

## Issue 1: Vision Input Format (HIGH SEVERITY)

### Current Plan (INCORRECT)
```python
# Lines 362-375 in feat-blender-librarian-FINAL.md
if image_base64:
    input_content = [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low"
            }
        }
    ]
else:
    input_content = user_prompt
```

### OpenAI Documentation (CORRECT)
From `docs/openai_api_documentation/Core_Concepts/IMAGES_AND_VISION.md`:

```python
# Responses API requires role and content structure
response = client.responses.create(
    model="gpt-4.1-mini",  # or gpt-5.2
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        ],
    }],
)
```

### Key Differences
1. **Type names**: `input_text` and `input_image` (not `text` and `image_url`)
2. **Structure**: Must wrap in `[{"role": "user", "content": [...]}]`
3. **Detail parameter**: Goes inside `input_image` dict, not nested `image_url`

### Corrected Code
```python
def call_gpt52_vision(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    image_base64: Optional[str] = None,
    reasoning_effort: str = "low",
    verbosity: str = "low"
) -> tuple[str, float]:
    """
    Call GPT-5.2 with optional image. Returns (response, estimated_cost).
    Uses Responses API (the recommended modern API).
    """
    # Build input content with correct Responses API format
    if image_base64:
        input_data = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"  # 85 tokens - cost effective
                }
            ]
        }]
    else:
        input_data = user_prompt  # Simple string for text-only

    # Use Responses API with GPT-5.2 optimizations
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,
        input=input_data,
        reasoning={"effort": reasoning_effort},  # GPT-5.2 optimization
        text={"verbosity": verbosity},           # GPT-5.2 optimization
        store=False,
        temperature=0.3
    )

    # ... rest of function
```

---

## Issue 2: Missing GPT-5.2 Optimization Parameters (MEDIUM SEVERITY)

### Current Plan
The plan doesn't use GPT-5.2's reasoning effort or verbosity controls.

### OpenAI Documentation
From `docs/openai_api_documentation/Get Started/USING_GPT_5.2.md`:

```python
# Reasoning effort levels: none, low, medium, high, xhigh
response = client.responses.create(
    model="gpt-5.2",
    input="...",
    reasoning={"effort": "low"}  # Lower = faster + cheaper
)

# Verbosity levels: low, medium, high
response = client.responses.create(
    model="gpt-5.2",
    input="...",
    text={"verbosity": "low"}  # Concise outputs = fewer tokens
)
```

### Recommendation
For Blender Librarian, use:
- **Vision diagnosis:** `reasoning={"effort": "low"}` - Visual analysis doesn't need deep reasoning
- **Doc synthesis:** `reasoning={"effort": "medium"}` - May need some reasoning for complex fixes
- **Both:** `text={"verbosity": "low"}` - We want JSON, not verbose explanations

This could reduce token usage by 20-40%, saving ~$2-4/month.

---

## Issue 3: Response Usage Tracking (LOW SEVERITY)

### Current Plan
```python
input_tokens = getattr(response, 'usage', {}).get('input_tokens', 500)
output_tokens = getattr(response, 'usage', {}).get('output_tokens', 200)
```

### Observation
The Responses API documentation shows `response.output_text` exists, but usage tracking structure may differ from Chat Completions. The current code has reasonable fallbacks (500/200 tokens), but should be verified at runtime.

### Recommendation
Test with actual API calls and adjust based on actual response structure. The current fallback values are reasonable for cost estimation.

---

## Required Changes to Plan

### File: `plans/feat-blender-librarian-FINAL.md`

**Change 1:** Replace lines 362-395 (the `call_gpt52_vision` function) with corrected version.

**Change 2:** Add reasoning/verbosity parameters to API calls on lines 570 and 666.

---

## Corrected Function (Full)

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_gpt52_vision(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    image_base64: Optional[str] = None,
    reasoning_effort: str = "low",
    verbosity: str = "low"
) -> tuple[str, float]:
    """
    Call GPT-5.2 with optional image. Returns (response, estimated_cost).
    Uses Responses API (the recommended modern API).

    Per OpenAI docs (2025): "While Chat Completions remains supported,
    Responses is recommended for all new projects."

    Key differences from Chat Completions:
    - Uses client.responses.create() instead of client.chat.completions.create()
    - Uses 'input' instead of 'messages'
    - Uses 'instructions' for system-level guidance
    - Response has 'output_text' helper instead of choices[0].message.content
    - Vision uses 'input_text' and 'input_image' types (not 'text' and 'image_url')
    - 3% improvement in SWE-bench, 40-80% better cache utilization
    """
    # Build input content with correct Responses API format
    if image_base64:
        # Vision requires structured input with role
        input_data = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"  # 85 tokens - cost effective
                }
            ]
        }]
    else:
        # Text-only can use simple string
        input_data = user_prompt

    # Use Responses API with GPT-5.2 optimizations
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt,  # System prompt goes in 'instructions'
        input=input_data,            # User content goes in 'input'
        reasoning={"effort": reasoning_effort},  # GPT-5.2: none/low/medium/high/xhigh
        text={"verbosity": verbosity},           # GPT-5.2: low/medium/high
        store=False,                 # Don't persist for privacy
        temperature=0.3              # Lower temp for consistent outputs
    )

    # Estimate cost (GPT-5.2 pricing: $1.75/1M input, $14/1M output)
    # Note: Responses API usage structure may differ - using fallbacks
    usage = getattr(response, 'usage', None)
    if usage:
        input_tokens = getattr(usage, 'input_tokens', 500)
        output_tokens = getattr(usage, 'output_tokens', 200)
    else:
        input_tokens = 500  # Conservative estimate
        output_tokens = 200

    cost = (input_tokens / 1000 * COST_INPUT_1K) + (output_tokens / 1000 * COST_OUTPUT_1K)
    if image_base64:
        cost += COST_IMAGE_LOW  # Add image cost (85 tokens for low detail)

    # Use output_text helper (Responses API convenience method)
    return response.output_text, cost
```

---

## Updated Call Sites

### diagnose_render_issue (line 570)
```python
response, cost = call_gpt52_vision(
    client,
    system_prompt,
    user_prompt,
    render_b64,
    reasoning_effort="low",    # Visual analysis
    verbosity="low"            # JSON output
)
```

### get_modification_advice (line 666)
```python
response, cost = call_gpt52_vision(
    client,
    system_prompt,
    user_prompt,
    reasoning_effort="medium",  # May need reasoning for fixes
    verbosity="low"             # JSON output
)
```

---

## Next Steps

1. **Apply corrections** to `plans/feat-blender-librarian-FINAL.md`
2. **Run plan_review** to validate the updated plan
3. **Implement** the corrected server.py

---

## References

- `docs/openai_api_documentation/Core_Concepts/RESPONSES_API.md` - Lines 1-200 (API structure)
- `docs/openai_api_documentation/Core_Concepts/IMAGES_AND_VISION.md` - Lines 200-400 (vision format)
- `docs/openai_api_documentation/Get Started/USING_GPT_5.2.md` - Lines 100-300 (reasoning/verbosity)
