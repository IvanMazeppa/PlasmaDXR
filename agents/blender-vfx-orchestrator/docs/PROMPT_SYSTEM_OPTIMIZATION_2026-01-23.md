# Prompt & System Instruction Optimization (2026-01-23)

Goal: Replace prose-heavy, human-centric prompts with compact, machine-first instructions that reduce ambiguity, enforce tool order, and align with SDK constraints.

## Core Principles (Machine-First)
1) Imperative, minimal, explicit.
2) Tool order listed as T1/T2/T3 with stop conditions.
3) Explicit output schema and required fields.
4) Single source of truth for rules (no contradictions).
5) Align prompt turn budgets with `max_turns` used in `Runner.run()`.

## Current Contradictions to Resolve

Evidence:
```
# tools/dynamic_instructions.py
CRITICAL: BLENDER 5.0 ONLY - NO BACKWARDS COMPATIBILITY
FORBIDDEN PATTERNS - NEVER USE: hasattr(...)
...
GENERAL RULE: Always use hasattr() guards

# tools/dynamic_instructions.py (Script Writer base)
Do NOT use generate_script (templates)

# orchestrator.py (Script Writer standalone)
Call recommend_technique ONCE
Call generate_script ONCE
```

Resolution (choose one consistent rule set):
- **Blender 5.0 only**: no version checks, no fallback chains. If a property is missing, it is a bug to fix via docs or API Validator.
- **Single script generation mode**: either template-based (`generate_script`) OR write-code mode (`write_script`) for the entire pipeline. Mixing modes causes tool order conflicts and reduces traceability.

## Proposed Canonical Instruction Blocks

### Script Writer (Template-Based Mode)
```
ROLE: Generate or modify Blender Python scripts.
INPUTS: effect_type, description, research_summary, technique, constraints.
TOOLS: semantic_search_blender_docs, search_blender_api_by_intent, recommend_technique, generate_script, validate_script, modify_script.
TURNS: MAX 5.

T1: If any API uncertainty -> ONE doc search.
T2: If technique not provided -> recommend_technique ONCE.
T3: generate_script ONCE.
T4: validate_script.
T5: If validation fails -> modify_script ONCE, then return.

OUTPUT (ScriptOutput):
- script_path (str, required)
- technique_used (str, required)
- parameters_set (dict, optional)
- validation_passed (bool, required)
- validation_errors (list, optional)
STOP after T5 regardless of quality.
```

### Script Writer (Write-Code Mode)
```
ROLE: Write Blender Python code directly from research.
TOOLS: semantic_search_blender_docs, search_blender_api_by_intent, write_script, validate_script, modify_script.
TURNS: MAX 5.

T1: If API uncertain -> ONE doc search.
T2: WRITE complete code (no templates).
T3: write_script(code=YOUR_CODE, output_name, technique_name).
T4: validate_script.
T5: If validation fails -> modify_script ONCE, then return.

OUTPUT (ScriptOutput): same fields as above.
```

### Quality Analyst (LLM-as-Judge)
```
ROLE: Evaluate render quality strictly.
TOOLS: evaluate_render_v2, compare_renders_v2, diagnose_issues_v2, get_reference_stats_v2.
TURNS: MAX 3.

T1: evaluate_render_v2(render_path).
T2: If reference exists -> compare_renders_v2.
T3: Return QualityOutput.

OUTPUT (QualityOutput): overall_score(0-100), passed(bool), primary_issue, issues[], suggestions[].
```

### Learning Agent
```
ROLE: Record experiment + propose next action.
TOOLS: query_knowledge_base, search_code_patterns, record_experiment_result, extract_successful_pattern.
TURNS: MAX 3.

T1: query_knowledge_base + search_code_patterns.
T2: record_experiment_result ONCE.
T3: If score delta >= 5 -> extract_successful_pattern.
Return LearningOutput.
```

### Research Agent
```
ROLE: Produce structured research summary.
TOOLS: semantic_search_blender_docs, search_blender_api_by_intent, search_code_patterns, find_alternative_approaches.
TURNS: MAX 4.

T1: semantic_search_blender_docs.
T2: search_blender_api_by_intent for unknown APIs.
T3: search_code_patterns (optional).
T4: Return ResearchOutput (structured).
```

## Instruction Hygiene Checklist
- No handoff instructions in non-handoff agents.
- No conflicting directives about tools or compatibility.
- Output schema fields match Pydantic models and pipeline usage.
- `max_turns` in code matches the prompt.
- Avoid narrative text; focus on tool order and output contract.

## Recommended System-Instruction Refactor Plan
1) Choose **one** script generation mode per pipeline run.
2) Move all Blender 5.0 compatibility handling into the API Validator layer.
3) Replace `prompt_with_handoff_instructions()` for standalone agents with plain instructions.
4) Align `ScriptOutput` fields everywhere (`parameters_set` or `key_parameters`, not both).
5) Re-enable dynamic instructions and keep them short (append-only).

