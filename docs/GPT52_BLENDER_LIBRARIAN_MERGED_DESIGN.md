# GPT-5.2 Blender Librarian Agent - Comprehensive Design

**Merged from:** Claude Code planning session + GPT-5.2 high analysis
**Created:** 2026-01-04
**Status:** Planning
**Budget Constraint:** $20/month OpenAI

---

## Executive Summary

Create a **hybrid RAG + Vision** "Blender Librarian" agent that:
- Uses **local documentation retrieval** (free) for most queries
- Calls **GPT-5.2 Vision** only at escalation points (plateau, hard issues)
- Outputs **constrained, safe patches** compatible with existing pipeline
- **Learns from experiments** to improve over time

This is NOT a fine-tuned model. It's a **tool-driven, doc-grounded, vision-assisted patch suggester**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPT-5.2 BLENDER LIBRARIAN                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐     │
│  │  MCP Server     │    │  OpenAI Client  │    │  Budget Tracker      │     │
│  │  (agents/       │───▶│  (Responses API)│◀───│  ($20/month limit)   │     │
│  │  gpt-librarian/)│    └────────┬────────┘    └─────────────────────┘     │
│  └─────────────────┘             │                                          │
│           │                      │                                          │
│           ▼                      ▼                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐     │
│  │  Local Doc RAG  │    │  Vision API     │    │  Playbook JSON      │     │
│  │  (blender-manual│    │  (Render Critic)│    │  (Known Fixes)      │     │
│  │   + OpenAI VS)  │    └─────────────────┘    └─────────────────────┘     │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
          │ blender-    │  │ asset-      │  │ script-     │
          │ orchestrator│  │ evaluator   │  │ generator   │
          └─────────────┘  └─────────────┘  └─────────────┘
```

### Key Design Decisions

1. **Local-first retrieval** - Use existing blender-manual MCP server (free)
2. **Selective GPT-5.2** - Only call OpenAI at escalation points
3. **Constrained outputs** - Only modifications pipeline can safely apply
4. **Cached ground truth reports** - Vision analysis of reference images cached for reuse
5. **Documentation gets majority of budget** - Per user preference

---

## Non-Goals (Guardrails)

- **N1**: Do NOT rewrite whole Blender scripts every iteration
- **N2**: Do NOT give vague suggestions like "try increasing samples"
- **N3**: Do NOT require constant OpenAI calls; most work should be local
- **N4**: Do NOT produce unbounded "creative rewrites" of scripts

---

## MCP Server Structure

**Directory:** `agents/gpt-librarian/`

```
agents/gpt-librarian/
├── server.py              # MCP server with tools
├── openai_client.py       # OpenAI Responses API wrapper
├── vector_store.py        # Vector store management (optional)
├── budget_tracker.py      # $20/month budget enforcement
├── vision_analyzer.py     # Image analysis logic
├── query_templates.py     # Symptom → structured query mapping
├── playbook.json          # Known fixes for common issues
├── requirements.txt       # openai>=1.60.0, mcp>=1.0.0, tenacity
├── .env.example           # OPENAI_API_KEY template
└── run_server.sh          # Startup script
```

---

## MCP Tools Specification

### Tool 1: `query_blender_docs`

```python
@tool
def query_blender_docs(
    question: str,
    effect_type: str = None,  # Filter: pyro, nebula, sun, etc.
    include_api: bool = True,  # Include Python API docs
    max_results: int = 5
) -> dict:
    """
    Query Blender documentation with optional GPT-5.2 synthesis.

    Strategy:
    1. Check playbook.json for known symptom → fix mapping (FREE)
    2. Query local blender-manual MCP server (FREE)
    3. If results insufficient, synthesize with GPT-5.2 ($)

    Returns:
        {
            "answer": "Synthesized answer with parameter recommendations",
            "citations": [{"title": str, "path": str, "snippet": str}],
            "confidence": float,  # 0.0-1.0
            "source": "playbook" | "local_rag" | "gpt52_synthesis",
            "tokens_used": int,
            "budget_remaining": float
        }
    """
```

### Tool 2: `analyze_render_quality`

```python
@tool
def analyze_render_quality(
    render_path: str,
    effect_type: str,
    evaluation_metrics: dict = None,  # From asset-evaluator
    reference_path: str = None,  # Optional ground truth
    detail_level: str = "low"  # "low" (85 tokens) or "high" (variable)
) -> dict:
    """
    Analyze render using GPT-5.2 vision + existing ML scores.

    Returns structured discrepancies with Blender-specific causes.

    Returns:
        {
            "overall_score": int,  # 0-100
            "discrepancies": [
                {
                    "type": "brightness | color_temperature | limb_darkening | texture_scale | corona_extent | prominence_coverage | noise_pattern | saturation | dynamic_range",
                    "severity": "low | medium | high | critical",
                    "evidence": "Short, objective description",
                    "likely_blender_causes": ["exposure", "blackbody_intensity", "density_scaling"]
                }
            ],
            "parameter_suggestions": {
                "flame_smoke": {"current": 1.0, "suggested": 2.5, "rationale": str}
            },
            "doc_queries": [str],  # Structured queries for doc lookup
            "tokens_used": int
        }
    """
```

### Tool 3: `suggest_recreation_approach`

```python
@tool
def suggest_recreation_approach(
    reference_path: str,
    target_effect_type: str = None,  # Auto-detect if not provided
    current_params: dict = None,
    cache_report: bool = True  # Cache detailed report for reuse
) -> dict:
    """
    Analyze reference image and suggest Blender recreation approach.

    NOTE: Detailed reports are CACHED to save budget on repeat queries.

    Returns:
        {
            "detected_effect_type": str,
            "key_visual_properties": [str],
            "suggested_technique": str,  # From technique catalog
            "initial_parameters": dict,
            "implementation_steps": [str],
            "success_criteria": {
                "lpips_threshold": float,
                "clip_threshold": float,
                "vfx_threshold": int
            },
            "cached": bool,  # True if loaded from cache
            "cache_path": str  # Path to cached report
        }
    """
```

### Tool 4: `diagnose_iteration_failure`

```python
@tool
def diagnose_iteration_failure(
    render_path: str,
    previous_render_path: str = None,
    current_params: dict,
    current_scores: dict,  # From asset-evaluator
    iteration_history: list = None  # Previous attempts
) -> dict:
    """
    Diagnose why iterations aren't improving, suggest breakthrough changes.

    Returns:
        {
            "diagnosis": str,
            "root_cause": str,
            "is_stuck": bool,  # True if detected plateau
            "suggested_approach_change": str,  # Different technique
            "modifications": dict,  # Safe patch contract
            "doc_citations": [{"path": str, "title": str}],
            "research_needed": bool,
            "research_query": str  # If research_needed
        }
    """
```

### Tool 5: `get_budget_status`

```python
@tool
def get_budget_status() -> dict:
    """
    Get current budget consumption status.

    Returns:
        {
            "monthly_limit": 20.0,
            "spent_this_month": float,
            "remaining": float,
            "reset_date": str,
            "calls_today": int,
            "estimated_calls_remaining": int,
            "budget_allocation": {
                "documentation": 14.0,  # $14 for docs (majority)
                "vision": 6.0           # $6 for vision
            }
        }
    """
```

---

## Safe Modification Contract

The librarian can ONLY output modifications compatible with `script-generator.modify_script()`:

```python
SAFE_MODIFICATIONS = {
    # Direct parameters
    "resolution": int,      # Simulation resolution
    "frame_start": int,
    "frame_end": int,
    "turbulence": float,    # 0.0-1.0
    "vorticity": float,
    "temperature": float,
    "domain_scale": float,
    "density": float,

    # Code replacements (via template markers)
    "custom_code": {
        "search_pattern": str,  # Regex or literal
        "replacement": str
    }
}
```

### Template Insertion Points (Robust Code Replacement)

Generated scripts should include explicit markers:

```python
# <LIBRARIAN:PARAM_BLOCK_BEGIN>
flame_max_temp = 3.0
flame_smoke = 1.5
burning_rate = 0.75
# <LIBRARIAN:PARAM_BLOCK_END>

# <LIBRARIAN:SHADER_BLOCK_BEGIN>
mat = bpy.data.materials.new(name="VolumetricMaterial")
mat.use_nodes = True
# ... shader setup ...
# <LIBRARIAN:SHADER_BLOCK_END>
```

This makes `custom_code` replacements **robust instead of brittle string matching**.

---

## Structured Query Templates

Instead of vague queries, the librarian generates structured doc queries:

```python
QUERY_TEMPLATES = {
    "blackbody_blowout": [
        "Principled Volume blackbody temperature attribute emission strength blowout",
        "color management Filmic vs AgX exposure volumetric",
        "bpy.types.ShaderNodeVolumePrincipled inputs blackbody_intensity"
    ],
    "wrong_color_temperature": [
        "Mantaflow fire temperature grid range exported to VDB",
        "blackbody emission color temperature kelvin mapping",
        "Principled Volume temperature attribute range"
    ],
    "limb_darkening_missing": [
        "volumetric density falloff sphere edge",
        "gradient texture spherical coordinates density",
        "emission vs absorption balance volumetric"
    ],
    "texture_too_uniform": [
        "Mantaflow noise turbulence vorticity settings",
        "procedural noise texture detail scale",
        "domain resolution vs noise detail"
    ]
}
```

---

## Playbook JSON (Zero-Cost Fixes)

Known issues with documented fixes - checked BEFORE any OpenAI calls:

```json
{
  "sun": {
    "blackbody_blowout": {
      "symptoms": ["white core", "no color gradient", "overexposed center"],
      "cause": "blackbody_intensity too high or temperature field not clamped",
      "fixes": [
        {"param": "blackbody_intensity", "action": "reduce", "range": [0.5, 2.0]},
        {"param": "view_transform", "action": "set", "value": "AgX"},
        {"param": "exposure", "action": "reduce", "range": [-2.0, 0.0]}
      ],
      "doc_paths": [
        "render/color_management.html",
        "render/shader_nodes/shader/principled_volume.html"
      ]
    },
    "warm_ratio_too_low": {
      "symptoms": ["too blue", "not warm enough", "cold appearance"],
      "cause": "temperature field too high or wrong color mapping",
      "fixes": [
        {"param": "flame_max_temp", "action": "reduce", "range": [2.0, 4.0]},
        {"param": "blackbody_tint", "action": "set", "value": [1.0, 0.9, 0.7]}
      ]
    }
  },
  "pyro": {
    "too_dark": {
      "symptoms": ["insufficient brightness", "flame not visible"],
      "fixes": [
        {"param": "flame_max_temp", "action": "increase", "range": [3.0, 6.0]},
        {"param": "emission_strength", "action": "increase", "range": [2.0, 10.0]}
      ]
    }
  }
}
```

---

## Vision Discrepancy Schema

Detailed output format for vision analysis:

```python
@dataclass
class VisualDiscrepancy:
    type: Literal[
        "brightness",
        "color_temperature",
        "limb_darkening",
        "texture_scale",
        "corona_extent",
        "prominence_coverage",
        "noise_pattern",
        "saturation",
        "dynamic_range",
        "granulation",
        "edge_definition"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    evidence: str  # Short, objective description
    likely_blender_causes: List[str]
    suggested_params: Dict[str, Any]
```

---

## Two-Stage Retrieval Strategy

```python
def retrieve_docs(query: str, effect_type: str = None) -> List[DocHit]:
    """
    Two-stage retrieval for optimal results.

    Stage 1: BM25/keyword scoring (fast, local)
    Stage 2: Semantic re-rank with embeddings (accurate)
    """
    # Stage 1: Keyword search via blender-manual
    keyword_hits = blender_manual.search_manual(query, limit=20)

    # Stage 2: Semantic re-rank
    if len(keyword_hits) > 5:
        embeddings = get_embeddings([h.content for h in keyword_hits])
        query_embedding = get_embedding(query)
        scores = cosine_similarity(query_embedding, embeddings)
        keyword_hits = sorted(zip(keyword_hits, scores), key=lambda x: -x[1])[:5]

    return keyword_hits
```

---

## Symptom-Query Learning Table

Simple JSON that improves over time:

```json
{
  "symptom_mappings": {
    "too dark": {
      "query_expansions": ["emission strength", "blackbody intensity", "exposure"],
      "best_doc_paths": ["render/shader_nodes/shader/principled_volume.html"],
      "success_rate": 0.85
    },
    "procedural looking": {
      "query_expansions": ["turbulence", "vorticity", "noise scale", "resolution"],
      "best_doc_paths": ["physics/fluid/type/domain/settings.html"],
      "success_rate": 0.72
    }
  },
  "effect_type_clusters": {
    "sun": ["color_management", "principled_volume", "blackbody", "emission"],
    "pyro": ["mantaflow", "domain", "flow", "smoke", "fire"]
  }
}
```

Updated automatically after successful runs.

---

## OpenAI Integration

### Responses API (NOT Assistants - Deprecated)

```python
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

class BlenderLibrarianClient:
    def __init__(self, api_key: str, vector_store_id: str = None):
        self.client = OpenAI(api_key=api_key)
        self.vector_store_id = vector_store_id
        self.model = "gpt-5.2"  # Or gpt-4o-mini for budget mode

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(openai.RateLimitError)
    )
    def query_with_docs(self, question: str, doc_context: str) -> dict:
        """Query with local doc context (no vector store needed)."""
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": BLENDER_LIBRARIAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Documentation context:\n{doc_context}\n\nQuestion: {question}"}
            ]
        )
        return self._parse_response(response)

    def analyze_image(self, image_path: str, prompt: str, detail: str = "low") -> dict:
        """Analyze render with vision API."""
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode()

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": VFX_CRITIC_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": detail  # "low" = 85 tokens fixed
                            }
                        }
                    ]
                }
            ]
        )
        return self._parse_response(response)
```

---

## Budget Tracking

```python
class BudgetTracker:
    MONTHLY_LIMIT = 20.0
    DOC_BUDGET = 14.0   # $14 for documentation (majority per user preference)
    VISION_BUDGET = 6.0  # $6 for vision

    COSTS = {
        "input_per_1k": 0.00175,   # $1.75/M tokens
        "output_per_1k": 0.014,    # $14.00/M tokens
        "vision_low": 0.0001,      # ~85 tokens fixed
        "vision_high_base": 0.001,
    }

    def can_afford(self, estimated_cost: float, category: str = "doc") -> bool:
        budget = self.DOC_BUDGET if category == "doc" else self.VISION_BUDGET
        spent = self.state["spent_doc"] if category == "doc" else self.state["spent_vision"]
        return (spent + estimated_cost) <= budget

    def get_status(self) -> dict:
        return {
            "monthly_limit": self.MONTHLY_LIMIT,
            "budget_allocation": {
                "documentation": {"limit": self.DOC_BUDGET, "spent": self.state["spent_doc"]},
                "vision": {"limit": self.VISION_BUDGET, "spent": self.state["spent_vision"]}
            },
            "total_remaining": self.MONTHLY_LIMIT - self.state["spent_doc"] - self.state["spent_vision"],
            "estimated_calls_remaining": int((self.MONTHLY_LIMIT - self.state["total"]) / 0.08)
        }
```

---

## Ground Truth Report Caching

Per user preference, vision analyses of reference images should be cached:

```python
class GroundTruthCache:
    CACHE_DIR = Path("cache/ground_truth_reports")

    def get_or_analyze(self, reference_path: str, analyzer: Callable) -> dict:
        """Get cached report or create new one."""
        cache_key = self._hash_image(reference_path)
        cache_path = self.CACHE_DIR / f"{cache_key}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                report = json.load(f)
                report["cached"] = True
                report["cache_path"] = str(cache_path)
                return report

        # Generate new report (costs tokens)
        report = analyzer(reference_path)
        report["cached"] = False
        report["generated_at"] = datetime.now().isoformat()

        # Cache for future use
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _hash_image(self, path: str) -> str:
        """Content-based hash for cache key."""
        import hashlib
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:16]
```

---

## System Prompts

### Blender Librarian

```python
BLENDER_LIBRARIAN_SYSTEM_PROMPT = """You are a Blender VFX expert specialized in volumetric rendering, Mantaflow fluid simulation, and creating realistic celestial effects.

When answering questions:
1. Search the provided Blender 5.0.1 documentation for relevant information
2. Provide specific parameter names and value ranges
3. Include code examples for Python API when relevant
4. Cite documentation sources with page paths
5. Be concise but thorough

You are assisting with a DXR volumetric particle renderer that generates NanoVDB assets via Blender simulations.

Effect types: pyro (explosions, fire, smoke), nebula (gas clouds), sun/star (solar prominences), mesh physics (soft body, cloth).

IMPORTANT: Output ONLY modifications compatible with the safe modification contract. Do not suggest wholesale script rewrites."""
```

### VFX Critic

```python
VFX_CRITIC_SYSTEM_PROMPT = """You are a brutally honest VFX quality analyst. Analyze renders for:

1. BRIGHTNESS - Is the effect properly exposed? Not too dark/bright?
2. COLOR TEMPERATURE - Does it match the effect type?
3. STRUCTURE - Sufficient turbulence and detail? Or blobby/procedural?
4. COVERAGE - Does the effect fill the frame appropriately?
5. REALISM - Real footage or obviously synthetic?

For each issue:
- Describe specifically
- Severity: critical/high/medium/low
- Suggest specific Blender parameter changes

Return JSON with this schema:
{
    "overall_score": 0-100,
    "discrepancies": [
        {"type": str, "severity": str, "evidence": str, "likely_blender_causes": [str]}
    ],
    "modifications": {"param_name": {"from": X, "to": Y, "reason": str}}
}

Be direct. Sugar-coating wastes budget."""
```

---

## Implementation Phases

### Phase 0: Local-Only Foundation (No OpenAI Spend)

**Goal:** Handle common cases without any API cost

- [ ] Create `agents/gpt-librarian/` directory structure
- [ ] Implement `playbook.json` with known sun/pyro fixes
- [ ] Create `query_templates.py` for structured queries
- [ ] Implement `symptom_mappings.json` learning table
- [ ] Integration with existing `blender-manual` MCP server

**Exit Criteria:** Given a symptom string, system returns correct doc pages and parameter suggestions from playbook.

### Phase 1: Budget Infrastructure (Still No Spend)

- [ ] Implement `budget_tracker.py` with $14/$6 doc/vision split
- [ ] Create ground truth cache system
- [ ] Add `.env.example` with OPENAI_API_KEY
- [ ] MCP server skeleton with tool definitions

**Exit Criteria:** Budget tracking works, cache system functional.

### Phase 2: Documentation Synthesis

- [ ] Implement `openai_client.py` with Responses API
- [ ] Add `query_blender_docs` tool with escalation logic
- [ ] Test with 10 sample queries

**Exit Criteria:** Doc queries return relevant answers with citations, budget tracked correctly.

### Phase 3: Vision Integration

- [ ] Implement `vision_analyzer.py`
- [ ] Add `analyze_render_quality` tool
- [ ] Implement ground truth caching
- [ ] Test with renders from `build/vdb_output/`

**Exit Criteria:** Vision identifies correct failure class (blowout vs underexposure vs wrong color).

### Phase 4: Ground Truth Recreation

- [ ] Implement `suggest_recreation_approach` tool
- [ ] Cache NASA solar footage analyses
- [ ] Integration with technique catalog

**Exit Criteria:** Given reference image, produces actionable recreation plan.

### Phase 5: Integration & Learning

- [ ] Implement `diagnose_iteration_failure` tool
- [ ] Connect to blender-orchestrator workflow
- [ ] Auto-update symptom_mappings after successful runs
- [ ] Update operator manual

**Exit Criteria:** Recommendations lead to positive score changes more often than random.

---

## Cost Analysis

### Budget Allocation (Per User Preference)

| Category | Budget | Purpose |
|----------|--------|---------|
| Documentation | $14 | Query synthesis, research |
| Vision | $6 | Render analysis, ground truth |
| **Total** | **$20** | Monthly limit |

### Estimated Usage

| Use Case | Calls/Month | Avg Cost | Total |
|----------|-------------|----------|-------|
| Doc queries (with synthesis) | 100 | $0.05 | $5.00 |
| Doc queries (local only) | 200 | $0.00 | $0.00 |
| Vision analysis (low) | 50 | $0.08 | $4.00 |
| Vision analysis (high) | 10 | $0.20 | $2.00 |
| Ground truth (cached after first) | 5 | $0.30 | $1.50 |
| **Buffer** | - | - | $7.50 |

### Cost Optimization

1. **Playbook-first** - Check known fixes before any API call
2. **Local retrieval** - Use blender-manual for most queries
3. **Vision detail="low"** - 85 tokens fixed vs variable
4. **Cache ground truth** - Analyze reference images once
5. **gpt-4o-mini fallback** - For simple queries when budget tight

---

## Licensing Note

Blender manual is **CC BY-SA 4.0**. RAG index compliance:
- Preserve attribution (source + path)
- Do not redistribute entire manual as proprietary
- If publishing model outputs trained from docs, ensure attribution

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Budget adherence | ≤ $20/month | budget_tracker.get_status() |
| Playbook hit rate | >30% | Queries resolved without API |
| Doc query relevance | >90% | Human eval of 20 queries |
| Vision accuracy | >80% | Compare to human VFX critique |
| Iteration improvement | +15% | Compare with/without librarian |
| Cache hit rate | >50% | Ground truth reuse |

---

## Open Questions

### Resolved
1. **Budget allocation** - Documentation majority ($14), vision ($6) per user preference
2. **Ground truth caching** - Yes, cache detailed reports for reuse

### Still Open
3. **Fallback when budget exhausted** - Queue requests or fall back to blender-manual only?
4. **Auto-apply suggestions** - Require human approval or auto-apply safe modifications?
5. **gpt-4o-mini for simple queries** - Use cheaper model when confidence high?

---

## References

### Internal
- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Pipeline architecture
- `docs/BLENDER_VFX_PIPELINE_OPERATOR_MANUAL.md` - Workflow documentation
- `agents/blender-manual/blender_server.py` - Existing doc server
- `agents/asset-evaluator/server.py` - ML evaluation system

### External
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Blender Manual CC BY-SA 4.0](https://docs.blender.org/manual/en/latest/)

---

## Source Documents

This design merges insights from:
1. **Claude Code Planning Session** (2026-01-04) - Architecture, API specifics, budget tracking
2. **GPT-5.2 High Analysis** (`BLENDER_LIBRARIAN_AGENT_DESIGN_GPT52.md`) - Tactical implementation, template markers, playbook concept

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-04
