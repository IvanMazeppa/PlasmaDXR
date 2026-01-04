# feat: GPT-5.2 Blender Librarian Agent

**Type:** Enhancement
**Priority:** Medium
**Created:** 2026-01-04
**Status:** Planning

---

## Overview

Create an OpenAI GPT-5.2 powered "Blender Librarian" agent that provides intelligent Blender documentation search, vision-based VFX quality analysis, and targeted advice for recreating ground truth reference images. The agent works alongside existing ML evaluation systems (LPIPS, CLIP, VFX diagnostics) within a **$20/month budget constraint**.

---

## Problem Statement / Motivation

### Current State

The PlasmaDXR VFX pipeline has a comprehensive multi-agent system for generating volumetric VFX assets:

1. **script-generator** - Creates Blender Python scripts with technique selection (UCB1)
2. **blender-executor** - Runs Mantaflow simulations
3. **asset-evaluator** - ML-based quality evaluation (LPIPS, CLIP, VFX diagnostics)
4. **experiment-tracker** - Knowledge base for learnings
5. **blender-manual** - Indexed Blender 5.0 documentation (2,196+ pages) with semantic search

### Problems

1. **Limited Research Integration (Phase 7)** - When the pipeline gets stuck (2+ iterations without improvement), it only has access to local blender-manual searches, which may not provide actionable advice for complex VFX problems

2. **No Visual Intelligence** - The existing evaluators (LPIPS, CLIP) produce scores but cannot provide human-like reasoning about *why* a render looks wrong or *how* to fix it specifically

3. **Ground Truth Recreation Gap** - Users have reference images (NASA solar footage, VFX sequences) but no intelligent system to analyze them and suggest recreation approaches

4. **Knowledge Synthesis Missing** - Blender-manual returns raw documentation snippets; no agent synthesizes these into actionable parameter recommendations

### Why GPT-5.2?

- **Claude Agent SDK requires separate API billing** - User's Max subscription doesn't cover it
- **$20 OpenAI budget available** - Practical constraint we can work within
- **GPT-5.2 Vision API** - State-of-the-art for image analysis (halves error rates on visual reasoning)
- **OpenAI adopted MCP** - Our existing MCP servers will work seamlessly
- **Vector Stores** - First 1GB free, then $0.10/GB/day (cheap for documentation)

---

## Proposed Solution

### High-Level Architecture

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
│  │  Vector Store   │    │  Vision API     │    │  Function Calling   │     │
│  │  (Blender Docs) │    │  (Render Critic)│    │  (Tool Orchestration)│    │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘     │
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

### Core Capabilities

1. **Documentation Intelligence** - Query OpenAI vector store containing Blender docs, synthesize answers
2. **Vision-Based Quality Analysis** - Analyze renders with GPT-5.2 vision, provide reasoning for issues
3. **Ground Truth Advisor** - Compare reference images to renders, suggest recreation approaches
4. **Integrated Diagnostics** - Combine LPIPS/CLIP/VFX scores with vision reasoning

---

## Technical Approach

### MCP Server Structure

**New Directory:** `agents/gpt-librarian/`

```
agents/gpt-librarian/
├── server.py              # MCP server with tools
├── openai_client.py       # OpenAI Responses API wrapper
├── vector_store.py        # Vector store management
├── budget_tracker.py      # $20/month budget enforcement
├── vision_analyzer.py     # Image analysis logic
├── requirements.txt       # openai>=1.60.0, mcp>=1.0.0
├── .env.example           # OPENAI_API_KEY template
└── run_server.sh          # Startup script
```

### MCP Tools Specification

#### Tool 1: `query_blender_docs`

```python
@tool
def query_blender_docs(
    question: str,
    effect_type: str = None,  # Optional filter: pyro, nebula, sun, etc.
    include_api: bool = True,  # Include Python API docs
    max_results: int = 5
) -> dict:
    """
    Query Blender documentation with GPT-5.2 synthesis.

    Returns:
        {
            "answer": "Synthesized answer with parameter recommendations",
            "citations": [{"title": str, "path": str, "snippet": str}],
            "confidence": float,  # 0.0-1.0
            "tokens_used": int,
            "budget_remaining": float
        }
    """
```

#### Tool 2: `analyze_render_quality`

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
    Analyze render using GPT-5.2 vision + LPIPS/CLIP/VFX scores.

    Returns:
        {
            "quality_assessment": str,  # Overall evaluation
            "issues_identified": [
                {"issue": str, "location": str, "severity": str}
            ],
            "parameter_suggestions": {
                "flame_smoke": {"current": 1.0, "suggested": 2.5, "rationale": str}
            },
            "reasoning": str,  # Chain-of-thought explanation
            "tokens_used": int,
            "vision_cost": float
        }
    """
```

#### Tool 3: `suggest_recreation_approach`

```python
@tool
def suggest_recreation_approach(
    reference_path: str,
    target_effect_type: str = None,  # Auto-detect if not provided
    current_params: dict = None
) -> dict:
    """
    Analyze reference image and suggest Blender recreation approach.

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
            }
        }
    """
```

#### Tool 4: `diagnose_iteration_failure`

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
            "parameter_changes": dict,
            "research_needed": bool,
            "research_query": str  # If research_needed
        }
    """
```

#### Tool 5: `get_budget_status`

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
            "estimated_calls_remaining": int
        }
    """
```

### OpenAI Integration

#### Responses API Pattern (NOT Assistants - deprecated)

```python
# openai_client.py

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

class BlenderLibrarianClient:
    def __init__(self, api_key: str, vector_store_id: str):
        self.client = OpenAI(api_key=api_key)
        self.vector_store_id = vector_store_id
        self.model = "gpt-5.2"  # Or gpt-4o-mini for budget mode

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(openai.RateLimitError)
    )
    def query_with_docs(self, question: str, max_results: int = 5) -> dict:
        """Query docs via file_search tool in Responses API."""
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": BLENDER_LIBRARIAN_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": max_results
                }
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
                {
                    "role": "system",
                    "content": VFX_CRITIC_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": detail  # "low" = 85 tokens, "high" = variable
                            }
                        }
                    ]
                }
            ]
        )
        return self._parse_response(response)
```

### Vector Store Setup

**Option A: Reuse blender-manual index (Recommended)**

```python
# vector_store.py

class VectorStoreManager:
    def __init__(self, client: OpenAI):
        self.client = client
        self.vector_store_id = None

    def create_from_blender_manual(self, manual_path: str) -> str:
        """Create OpenAI vector store from existing blender-manual index."""

        # Load existing index (from agents/blender-manual/manual_index.json)
        with open(f"{manual_path}/manual_index.json") as f:
            index = json.load(f)

        # Create vector store
        vs = self.client.vector_stores.create(name="Blender 5.0 Documentation")

        # Convert to markdown files and upload
        temp_dir = Path("/tmp/blender_docs_upload")
        temp_dir.mkdir(exist_ok=True)

        files_to_upload = []
        for page in index["pages"]:
            md_path = temp_dir / f"{page['category']}_{page['subcategory']}_{page['path'].replace('/', '_')}.md"
            md_content = f"""# {page['title']}

**Category:** {page['category']}
**Subcategory:** {page['subcategory']}
**Keywords:** {', '.join(page.get('keywords', []))}

{page['content']}
"""
            md_path.write_text(md_content)
            files_to_upload.append(open(md_path, "rb"))

        # Batch upload (efficient)
        batch = self.client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id,
            files=files_to_upload
        )

        # Cleanup
        for f in files_to_upload:
            f.close()
        shutil.rmtree(temp_dir)

        self.vector_store_id = vs.id
        return vs.id
```

### Budget Tracking

```python
# budget_tracker.py

import json
from datetime import datetime
from pathlib import Path

class BudgetTracker:
    MONTHLY_LIMIT = 20.0  # $20/month

    # Estimated costs (GPT-5.2, Jan 2026)
    COSTS = {
        "input_per_1k": 0.00175,   # $1.75/M tokens
        "output_per_1k": 0.014,    # $14.00/M tokens
        "vision_low": 0.0001,      # ~85 tokens
        "vision_high_base": 0.001, # Base + per-tile
        "vision_high_tile": 0.0005,
    }

    def __init__(self, state_path: str = "budget_state.json"):
        self.state_path = Path(state_path)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path) as f:
                state = json.load(f)
                # Reset if new month
                if datetime.fromisoformat(state["period_start"]).month != datetime.now().month:
                    return self._new_period()
                return state
        return self._new_period()

    def _new_period(self) -> dict:
        return {
            "period_start": datetime.now().replace(day=1).isoformat(),
            "spent": 0.0,
            "calls": 0
        }

    def can_afford(self, estimated_cost: float) -> bool:
        return (self.state["spent"] + estimated_cost) <= self.MONTHLY_LIMIT

    def record_usage(self, input_tokens: int, output_tokens: int, vision_detail: str = None):
        cost = (input_tokens / 1000) * self.COSTS["input_per_1k"]
        cost += (output_tokens / 1000) * self.COSTS["output_per_1k"]

        if vision_detail == "low":
            cost += self.COSTS["vision_low"]
        elif vision_detail == "high":
            cost += self.COSTS["vision_high_base"]  # + tiles if measured

        self.state["spent"] += cost
        self.state["calls"] += 1
        self._save_state()
        return cost

    def get_status(self) -> dict:
        return {
            "monthly_limit": self.MONTHLY_LIMIT,
            "spent_this_month": round(self.state["spent"], 4),
            "remaining": round(self.MONTHLY_LIMIT - self.state["spent"], 4),
            "reset_date": (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).isoformat(),
            "calls_today": self.state["calls"],
            "estimated_calls_remaining": int((self.MONTHLY_LIMIT - self.state["spent"]) / 0.05)  # ~$0.05/call avg
        }
```

### System Prompts

```python
# prompts.py

BLENDER_LIBRARIAN_SYSTEM_PROMPT = """You are a Blender VFX expert specialized in volumetric rendering, Mantaflow fluid simulation, and creating realistic celestial effects.

When answering questions:
1. Search the provided Blender 5.0 documentation for relevant information
2. Provide specific parameter names and value ranges
3. Include code examples for Python API when relevant
4. Cite documentation sources with page paths
5. Be concise but thorough

You are assisting with a DXR volumetric particle renderer that generates NanoVDB assets via Blender simulations.

Effect types you specialize in: pyro (explosions, fire, smoke), nebula (gas clouds), sun/star (solar prominences), and mesh physics (soft body, cloth).

Budget constraint: User has limited API budget. Be efficient with responses."""

VFX_CRITIC_SYSTEM_PROMPT = """You are a brutally honest VFX quality analyst. Analyze renders for:

1. BRIGHTNESS - Is the effect properly exposed? Not too dark/bright?
2. COLOR - Does the color temperature match the effect type?
3. STRUCTURE - Is there sufficient turbulence and detail? Or is it blobby/procedural-looking?
4. COVERAGE - Does the effect fill the frame appropriately?
5. REALISM - Does it look like real footage or obviously synthetic?

For each issue found:
- Describe the problem specifically
- Indicate severity (critical/high/medium/low)
- Suggest specific Blender parameter changes

Return structured JSON:
{
    "overall_score": 0-100,
    "issues": [{"issue": str, "severity": str, "location": str}],
    "parameter_changes": {"param_name": {"from": X, "to": Y, "reason": str}}
}

Be direct. Sugar-coating wastes the user's time and budget."""
```

---

## Implementation Phases

### Phase 1: Foundation (3-4 hours)

**Tasks:**
- [ ] Create `agents/gpt-librarian/` directory structure
- [ ] Implement `openai_client.py` with Responses API wrapper
- [ ] Implement `budget_tracker.py` with $20/month enforcement
- [ ] Create `server.py` MCP server skeleton
- [ ] Add to `.claude/settings.local.json` permissions

**Deliverables:**
- Working MCP server that can make OpenAI API calls
- Budget tracking with persistence

### Phase 2: Vector Store Setup (2-3 hours)

**Tasks:**
- [ ] Implement `vector_store.py` with create/upload functions
- [ ] Script to convert blender-manual index to OpenAI vector store
- [ ] Test file_search retrieval quality
- [ ] Implement `query_blender_docs` tool

**Deliverables:**
- Blender 5.0 docs indexed in OpenAI vector store
- Working documentation query tool

### Phase 3: Vision Integration (3-4 hours)

**Tasks:**
- [ ] Implement `vision_analyzer.py`
- [ ] Create `analyze_render_quality` tool
- [ ] Integrate with existing asset-evaluator metrics
- [ ] Test with sample renders from `build/vdb_output/`

**Deliverables:**
- Vision-based render analysis
- Structured issue identification and parameter suggestions

### Phase 4: Ground Truth Recreation (2-3 hours)

**Tasks:**
- [ ] Implement `suggest_recreation_approach` tool
- [ ] Add reference image analysis capability
- [ ] Create approach document generation
- [ ] Test with NASA solar footage in `assets/reference_images/`

**Deliverables:**
- Ability to analyze reference images and suggest recreation approaches

### Phase 5: Integration & Testing (2-3 hours)

**Tasks:**
- [ ] Implement `diagnose_iteration_failure` tool
- [ ] Integration tests with blender-orchestrator workflow
- [ ] Budget optimization (use gpt-4o-mini for simple queries)
- [ ] Documentation and usage examples

**Deliverables:**
- Fully integrated librarian in iteration loop
- Operator manual updates

---

## Acceptance Criteria

### Functional Requirements

- [ ] `query_blender_docs` returns relevant answers with citations
- [ ] `analyze_render_quality` identifies issues visible in renders
- [ ] `suggest_recreation_approach` produces actionable recreation plans
- [ ] `diagnose_iteration_failure` helps break out of stuck loops
- [ ] All tools track and respect $20/month budget

### Non-Functional Requirements

- [ ] Average query latency < 5 seconds
- [ ] Budget tracking accurate to within 1%
- [ ] Graceful degradation when budget exhausted (queue or fallback)
- [ ] Rate limit handling with exponential backoff

### Quality Gates

- [ ] Vision analysis agrees with human assessment >80% of time
- [ ] Documentation queries return relevant content >90% of time
- [ ] Integration doesn't break existing blender-orchestrator workflow

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Monthly budget adherence | ≤ $20 | budget_tracker.get_status() |
| Query relevance | >90% | Human evaluation of 20 queries |
| Vision accuracy | >80% | Compare to human VFX critique |
| Iteration improvement rate | +15% | Compare with/without librarian |
| User satisfaction | Positive | Ben's feedback |

---

## Dependencies & Prerequisites

### Required

- OpenAI API key with $20/month budget
- Python 3.10+ (for openai SDK)
- Existing blender-manual index (`agents/blender-manual/manual_index.json`)

### Optional (Enhance Quality)

- Sample renders for vision testing (`build/vdb_output/`)
- Reference images (`assets/reference_images/star/`)

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Budget overrun | Medium | High | Hard limit in budget_tracker, alert at 80% |
| Vision API poor quality | Low | Medium | A/B test with human; fallback to LPIPS/CLIP only |
| Vector store retrieval miss | Medium | Medium | Combine with blender-manual local search |
| OpenAI API rate limits | Medium | Medium | Exponential backoff, queue during limits |
| Integration breaks orchestrator | Low | High | Loose coupling; librarian is optional |

---

## Cost Analysis

### Estimated Monthly Usage

| Use Case | Calls/Month | Avg Cost/Call | Total |
|----------|-------------|---------------|-------|
| Doc queries | 100 | $0.05 | $5.00 |
| Vision analysis (low) | 50 | $0.08 | $4.00 |
| Vision analysis (high) | 20 | $0.20 | $4.00 |
| Ground truth recreation | 10 | $0.30 | $3.00 |
| **Buffer** | - | - | $4.00 |
| **Total** | 180 | - | **$20.00** |

### Cost Optimization Strategies

1. **Use gpt-4o-mini for simple doc queries** - 10x cheaper
2. **Use vision detail="low" by default** - 85 tokens fixed
3. **Cache common queries** - Reduce repeated calls
4. **Batch vision analysis** - Analyze multiple iterations at once

---

## Alternative Approaches Considered

### 1. Extend blender-manual MCP Server

**Pros:** No new server, single source of truth
**Cons:** Couples OpenAI dependency to existing server, budget tracking more complex
**Decision:** Rejected - separation of concerns preferred

### 2. Use Claude API Instead

**Pros:** Already using Claude Code, familiar
**Cons:** Requires Claude Agent SDK with separate billing; user only has OpenAI budget
**Decision:** Rejected - budget constraint

### 3. Local LLM (Ollama/LMStudio)

**Pros:** No API cost, full control
**Cons:** No vision capability, worse quality, significant setup
**Decision:** Rejected - GPT-5.2 vision is key differentiator

### 4. NVIDIA NIM + LLM

**Pros:** Already have NVIDIA API for log-analysis-rag
**Cons:** NVIDIA doesn't have vision API comparable to GPT-5.2
**Decision:** Rejected - vision capability critical

---

## Future Considerations

### Phase 2 Enhancements (If Successful)

- **Fine-tuned model** for VFX-specific analysis
- **Multi-image comparison** in single call
- **Streaming responses** for long analyses
- **Feedback learning** from human ratings

### Integration Opportunities

- **experiment-tracker auto-logging** of librarian insights
- **mission-control routing** for council decisions
- **SKILL.md enhancement** with librarian consultation stage

---

## References & Research

### Internal References

- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Pipeline architecture
- `docs/BLENDER_VFX_PIPELINE_OPERATOR_MANUAL.md` - Workflow documentation
- `agents/blender-manual/blender_server.py` - Existing doc server
- `agents/asset-evaluator/server.py` - ML evaluation system

### External References

- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create)
- [OpenAI Vector Stores](https://platform.openai.com/docs/api-reference/vector-stores)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [GPT-5.2 Model Documentation](https://platform.openai.com/docs/models/gpt-5.2)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)

### Research Sources

- RAG Best Practices 2025 (firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- Multi-Agent LLM Architecture Guide (collabnix.com)
- Visual Agents at CVPR 2025 (voxel51.com)

---

## Open Questions for Clarification

### Critical (Must Answer Before Implementation)

1. **Budget allocation** - Should vision analysis have a separate budget cap (e.g., $10 vision, $10 docs)?
2. **Fallback behavior** - When budget exhausted, queue requests or fall back to blender-manual only?
3. **Integration depth** - Should librarian auto-apply suggestions or always require human approval?

### Important (Can Default If Not Answered)

4. **Model choice** - GPT-5.2 for all, or gpt-4o-mini for simple queries?
5. **Conversation history** - Retain per-asset or per-session?
6. **Ground truth iteration limit** - How many refinement cycles before human escalation?

---

## Appendix: User Flow Diagrams

### Flow 1: Documentation Query

```
User: "How do I increase turbulence in Mantaflow?"
    │
    ▼
[query_blender_docs]
    │
    ├─▶ Search OpenAI vector store
    │
    ├─▶ Retrieve top 5 relevant docs
    │
    ├─▶ GPT-5.2 synthesizes answer
    │
    ▼
Response: "Increase the 'vorticity' parameter in FluidDomainSettings...
          (per Fluid Domain docs, section 'Guides')"
```

### Flow 2: Vision Analysis in Iteration Loop

```
[blender-orchestrator iteration]
    │
    ├─▶ [blender-executor] runs simulation
    │
    ├─▶ [asset-evaluator] produces scores: VFX=45, LPIPS=0.72
    │
    ├─▶ Score below threshold → invoke librarian
    │
    ▼
[analyze_render_quality]
    │
    ├─▶ GPT-5.2 vision analyzes render
    │
    ├─▶ Cross-reference with evaluation metrics
    │
    ├─▶ Identify: "TOO DARK - flame_max_temp too low"
    │
    ▼
Response: {
    "issues": [{"issue": "Insufficient brightness", "severity": "high"}],
    "parameter_changes": {"flame_max_temp": {"from": 2.0, "to": 4.0}}
}
    │
    ▼
[script-generator] applies changes → next iteration
```

### Flow 3: Ground Truth Recreation

```
User: "Recreate this NASA solar prominence footage"
    │
    ├─▶ [suggest_recreation_approach]
    │       │
    │       ├─▶ GPT-5.2 vision analyzes reference
    │       │
    │       ├─▶ Identifies: red prominences, orange disk, ~5800K
    │       │
    │       ├─▶ Searches docs for prominence techniques
    │       │
    │       ▼
    │   Response: {
    │       "technique": "thermal_corona",
    │       "initial_parameters": {...},
    │       "success_criteria": {"lpips": 0.35, "vfx": 60}
    │   }
    │
    ├─▶ [script-generator] creates script
    │
    ├─▶ [blender-executor] runs simulation
    │
    ├─▶ [asset-evaluator] + [analyze_render_quality] evaluate
    │
    ├─▶ If not matching → refine (max 3 cycles)
    │
    ▼
Final asset or escalate to human
```

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-04
**Author:** Claude Code (Planning Session)
