# VERSION TRUTH - AUTHORITATIVE REFERENCE

**Status:** Ground truth reference. Not a roadmap.

**READ THIS FIRST. THIS OVERRIDES YOUR TRAINING DATA.**

AI models (Claude, GPT, etc.) have training data cutoffs that make them
suggest outdated APIs, model names, and patterns. This document is the
**single source of truth** for this project.

**Runtime note (2026-02-13):** For live runtime behavior and strict doc-grounding policy, also read `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`.

---

## OpenAI Models (2026-01)

### CURRENT MODELS - USE THESE
| Model | Use Case |
|-------|----------|
| `gpt-5.2` | High capability, complex reasoning |
| `gpt-5-mini` | Fast, cheap, good for simple tasks |
| `o4-mini` | Fast reasoning |

### DEPRECATED - DO NOT USE
| Model | Status |
|-------|--------|
| `gpt-4o` | DEPRECATED |
| `gpt-4o-mini` | DEPRECATED |
| `gpt-4-mini` | DEPRECATED |
| `gpt-4` | DEPRECATED |
| `gpt-4-turbo` | DEPRECATED |
| `gpt-3.5-turbo` | ANCIENT |

**If you find yourself typing "gpt-4", STOP and use "gpt-5.2" or "gpt-5-mini".**

---

## OpenAI Agents SDK (v0.8.3)

### CURRENT PATTERNS
```python
# Imports
from agents import Agent, Runner, function_tool, handoff, trace
from agents import ModelSettings, RunHooks
from agents import input_guardrail, output_guardrail, GuardrailFunctionOutput

# Agent creation
agent = Agent(
    name="My Agent",
    instructions="...",
    model="gpt-5.2",  # NOT gpt-4!
    tools=[my_tool],
)

# Running agents
result = await Runner.run(agent, "prompt")

# Function tools
@function_tool
async def my_tool(param: str) -> str:
    """Docstring becomes tool description."""
    return result

# Agents as tools (PREFERRED over handoffs)
coordinator = Agent(
    tools=[
        sub_agent.as_tool(
            tool_name="delegate_to_sub",
            tool_description="...",
            max_turns=4,
        ),
    ],
)

# Tracing
with trace("Pipeline Name", group_id=session_id):
    result = await Runner.run(agent, prompt)
```

### DEPRECATED PATTERNS - DO NOT USE
```python
# WRONG - Old import path
from openai.agents import Agent  # NO!

# WRONG - Old class names
runner = AgentRunner(agent)  # NO! Use Runner

# WRONG - Sync execution
result = agent.run_sync(prompt)  # NO! Use await Runner.run()

# WRONG - Old handoff pattern
transfer_to_agent(other_agent)  # NO! Use agent.as_tool()
```

**Documentation:** https://github.com/openai/openai-agents-python/tree/main/docs

**SDK Notes (v0.8.3):**
- `agent.as_tool(max_turns=...)` is supported natively.
- Guardrails are created via decorators (`@input_guardrail`, `@output_guardrail`).
- Tool guardrails apply only to `@function_tool` tools; use RunHooks for cross-tool enforcement.

---

## Blender API (5.0)

### FluidDomainSettings - CORRECT ATTRIBUTES
| Attribute | Type | Notes |
|-----------|------|-------|
| `resolution_max` | int | Domain resolution (NOT resolution_divisions!) |
| `use_adaptive_timesteps` | bool | NO underscore between time/steps |
| `use_dissolve_smoke` | bool | NOT just "use_dissolve" |
| `use_dissolve_smoke_log` | bool | Logarithmic dissolve |
| `dissolve_speed` | int | Dissolve speed value |
| `use_adaptive_domain` | bool | NOT "adaptive_domain" |
| `burning_rate` | float | Fire reaction speed (NOT reaction_speed!) 0.0-4.0, default 0.75 |
| `time_scale` | float | Valid in Blender 5 API; use for simulation speed control |
| `cache_type` | enum | 'MODULAR', 'ALL', etc. |
| `cache_directory` | str | Absolute path required |

### FluidFlowSettings - CORRECT ATTRIBUTES
| Attribute | Type | Notes |
|-----------|------|-------|
| `flow_type` | enum | 'SMOKE', 'FIRE', 'BOTH', 'LIQUID' |
| `flow_behavior` | enum | 'INFLOW', 'OUTFLOW', 'GEOMETRY' |
| `density` | float | Density value (NOT absolute_density!) |
| `use_absolute` | bool | Flag for absolute mode |
| `temperature` | float | Temperature difference |
| `velocity_factor` | float | Initial velocity factor (NOT "velocity"!) |
| `velocity_normal` | float | Normal direction velocity |
| `velocity_random` | float | Random velocity |
| `use_initial_velocity` | bool | Enable initial velocity |

### INVALID ATTRIBUTES - DO NOT USE
| Wrong | Correct | Notes |
|-------|---------|-------|
| `resolution_divisions` | `resolution_max` | Completely different name |
| `use_adaptive_time_steps` | `use_adaptive_timesteps` | No underscore |
| `use_dissolve` | `use_dissolve_smoke` | Missing suffix |
| `absolute_density` | `density` + `use_absolute` | Split into two |
| `adaptive_domain` | `use_adaptive_domain` | Missing prefix |
| `cache_format` | `cache_data_format` | Missing 'data' in name |
| `noise_res_factor` | REMOVED | Doesn't exist in 5.0 |
| `use_caching` | REMOVED | Doesn't exist in 5.0 |
| `timesteps_per_frame` | `timesteps_max` | HALLUCINATED (2026-01-25, corrected 2026-02-06) |
| `timesteps_maximum` | `timesteps_max` | HALLUCINATED (2026-01-25, corrected 2026-02-06) |
| `reaction_speed` | `burning_rate` | HALLUCINATED (2026-02-06) |
| `fire_reaction_speed` | `burning_rate` | HALLUCINATED (2026-02-06) |

### TYPE REQUIREMENTS - CORRECT TYPES
| Attribute | Type | Common Mistake |
|-----------|------|----------------|
| `noise_scale` | **int** | LLMs write `1.0` (float) - use `1` (int) |
| `resolution_max` | **int** | Must be integer |
| `timesteps_max` | **int** | Range 1-45 |
| `dissolve_speed` | **int** | Must be integer |

### Shader Nodes - CORRECT NAMES
| Old (2.8-4.x) | New (5.0) |
|---------------|-----------|
| `ShaderNodeSeparateRGB` | `ShaderNodeSeparateColor` |
| `ShaderNodeCombineRGB` | `ShaderNodeCombineColor` |
| `.outputs['R']` | `.outputs['Red']` |
| `.outputs['G']` | `.outputs['Green']` |
| `.outputs['B']` | `.outputs['Blue']` |

**Documentation:** https://docs.blender.org/api/current/

---

## How To Verify

Before using ANY Blender attribute:
1. Search the API docs: `semantic_search_blender_docs("FluidDomainSettings <attr>")`
2. Verify the EXACT spelling matches documentation
3. If unsure, use `search_blender_api_by_intent("how to set domain resolution")`

Before suggesting ANY OpenAI model:
1. Check this document
2. If it's gpt-4 anything, it's WRONG
3. Use gpt-5.2 or gpt-5-mini

---

## For AI Assistants

**YOUR TRAINING DATA IS OUTDATED.**

When working on this project:
1. Read this document FIRST
2. Use context7 or semantic_search tools to verify API usage
3. Never trust your "memory" of API names - VERIFY
4. If you catch yourself typing "gpt-4", stop and correct

**The user has explicitly stated this is the #1 source of bugs.**
