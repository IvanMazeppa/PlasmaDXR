# Vector Store Guide

Status: authoritative
Last updated: 2026-03-16
Purpose: complete reference for how vector stores work in the VFX orchestrator, what's in them, and how to add content

---

## Why Vector Stores Matter

The Research Agent and Docs Expert can only discover techniques and API knowledge that exists in the vector stores. If a physics system, modifier workflow, or technique isn't represented, the agent will either hallucinate an approach or default to the simplest known method. Vector store coverage directly determines technique diversity and script quality.

---

## Architecture: Three Stores

The system uses three OpenAI vector stores, searched through `tools/semantic_docs_tools.py`.

| Store | Env Var | Store ID | Content | File Count |
|-------|---------|----------|---------|------------|
| **Manual Store** | `BLENDER_MANUAL_VECTOR_STORE_ID` | `vs_697564fbfc0c8191b2e44aa47dbaf482` | Raw Blender 5.0 manual pages | ~100 |
| **API Store** | `BLENDER_API_VECTOR_STORE_ID` | `vs_697571f0275c8191910fea0f2c8bdd3a` | Blender Python API reference | ~50 |
| **Rewritten Manual Store** | `BLENDER_REWRITTEN_MANUAL_STORE_ID` | `vs_699b7e6221bc81919ba2f4a1eae11588` | LLM-optimized rewrites + technique guides | **174** |

### Search Priority

When searching, the system auto-detects intent and routes queries:

1. **API queries** (e.g., `bpy.types.FluidDomainSettings`) → API Store only
2. **Manual/conceptual queries** (e.g., "how to create fire") → Rewritten Manual Store → Manual Store (fallback)
3. **Ambiguous queries** → All stores, results merged and ranked by relevance score

The Rewritten Manual Store is always searched first for manual queries because it has higher information density. The original Manual Store serves as fallback.

### How Search Works

```python
# tools/semantic_docs_tools.py

# 1. Query intent auto-detected from keywords
detected_intent = _classify_query_intent(query)  # 'api', 'manual', or 'both'

# 2. Appropriate store(s) selected
# 3. OpenAI vector_stores.search API called per store
response = client.vector_stores.search(
    vector_store_id=store_id,
    query=query,
    max_num_results=max_results
)

# 4. Results merged, ranked by score, content trimmed to 800 chars
# 5. If results are weak (score < threshold), SDK file_search fallback runs
```

---

## What's Currently in the Rewritten Manual Store (174 files)

### Breakdown by Category

| Category | Files | Coverage |
|----------|-------|----------|
| **Mantaflow (fluid/gas/liquid)** | ~25 | Strong — domain, flow, effector, cache, gas settings, liquid mesh/particles, adaptive domain, noise |
| **Rigid Body** | ~18 | Strong — properties, dynamics, collisions, constraints (all 8 types), world, tips |
| **Cloth** | ~10 | Strong — settings, collisions, shape, cache, field weights, physical properties, examples |
| **Particles** | ~20 | Strong — emitter (emission, velocity, rotation, physics types, render, children), hair (dynamics, shape, emission) |
| **Soft Body** | ~14 | Good — forces (interior/exterior), settings (edges, goal, solver, cache), collision, examples |
| **Force Fields** | ~14 | Strong — all types (wind, turbulence, vortex, drag, harmonic, magnetic, etc.) |
| **Dynamic Paint** | ~4 | Basic — canvas, brush, introduction |
| **Render/Lights** | ~6 | Basic — light objects, light linking, world, volume absorption/scatter shaders |
| **Modifiers** | ~10 | Basic — physics modifiers (cloth, collision, fluid, explode, ocean, particle, soft body) |
| **General** | ~5 | Physics introduction, baking, collision, simulation nodes |

### Technique Guides (seeded, 2026-03-15)

These are LLM-optimized technique documents with working Python code, decision guides, and parameter tables. They teach the agent HOW to create effects, not just what the API looks like.

| Seed Script | Technique Docs | Code Patterns | Topics |
|-------------|---------------|---------------|--------|
| `seed_destruction_techniques.py` | 6 | 4 | Glass shattering: cell fracture, constraints, GN fracture, particle debris, projectiles, combined hero |
| `seed_cloth_techniques.py` | 5 | 4 | Flags/banners, fabric draping, cloth interaction, cloth materials, overview |
| `seed_mantaflow_advanced_techniques.py` | 5 | 4 | Liquid sim, multi-flow, dissolve effects, OpenVDB export, overview |
| `seed_particle_techniques.py` | 5 | 4 | Emitter effects, force fields, instancing, collision, overview |
| `seed_mixed_physics_techniques.py` | 5 | 4 | RB+particles, fluid+RB, cloth+forces, scene physics setup, overview |
| `seed_modifier_techniques.py` | 6 | 4 | Smooth objects (Screw), hard surface (Boolean/Bevel), arrays, deformation, modifier stacking |
| **Total** | **32** | **24** | |

---

## How to Upload New Content

### Option 1: Seed Script (Recommended for Technique Guides)

All seed scripts follow the same pattern. Use any existing one as a template.

```bash
cd agents/blender-vfx-orchestrator

# Dry run — see what would be uploaded without spending API credits
venv/bin/python scripts/seed_modifier_techniques.py --dry-run

# Upload technique docs to vector store + seed code patterns
venv/bin/python scripts/seed_modifier_techniques.py

# Only upload docs (no code patterns)
venv/bin/python scripts/seed_modifier_techniques.py --docs-only

# Only seed code patterns (no vector store upload)
venv/bin/python scripts/seed_modifier_techniques.py --patterns-only
```

**Creating a new seed script:**

1. Copy an existing seed script (e.g., `seed_cloth_techniques.py`)
2. Replace `TECHNIQUE_DOCS` dict with your new documents
3. Replace `CODE_PATTERNS` list with your code snippets
4. Update the test query at the bottom
5. Run with `--dry-run` first to verify

**Document format for `TECHNIQUE_DOCS`:**

```python
TECHNIQUE_DOCS = {
    "my_technique_overview.md": """# My Technique in Blender 5.0
DocType: technique-guide
DocPath: techniques/my_domain/overview
DocVersion: 5.0.1
PhysicsDomain: rigid_body
---

## Overview
Dense, factual content here. No fluff.

## Technique Comparison Table
| Technique | Realism | Complexity | Best For |
|-----------|---------|------------|----------|
| ...       | ...     | ...        | ...      |

## Complete Python Code
```python
import bpy
# Working Blender 5.0 code here
```

NOTE: Use correct Blender 5.0 API names.
TECHNIQUE: Actionable advice for the agent.
""",
}
```

**Code pattern format for `CODE_PATTERNS`:**

```python
CODE_PATTERNS = [
    {
        "name": "my_pattern_name",
        "description": "What this pattern does",
        "effect_types": ["fire", "explosion"],  # Which effects this applies to
        "code": '''
import bpy
# Minimal working code snippet
''',
        "expected_improvement": 30,  # Estimated quality score improvement
    },
]
```

### Option 2: Direct Upload (for individual files)

```python
from openai import OpenAI
import os

client = OpenAI()
store_id = os.getenv("BLENDER_REWRITTEN_MANUAL_STORE_ID", "vs_699b7e6221bc81919ba2f4a1eae11588")

# Write content to a temp file
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
    f.write("# My Document\nDocType: technique-guide\n...")
    temp_path = f.name

# Upload
with open(temp_path, 'rb') as f:
    file_obj = client.files.create(file=f, purpose='assistants')
    client.vector_stores.files.create(vector_store_id=store_id, file_id=file_obj.id)

# Wait for processing
import time
while True:
    store = client.vector_stores.retrieve(store_id)
    if store.file_counts.completed == store.file_counts.total:
        break
    time.sleep(2)
```

### Option 3: Rewritten Manual Pipeline (for bulk manual rewrites)

The `rewritten_manual/` directory contains 131 LLM-optimized rewrites of Blender manual pages. These were generated by an offline LLM process that:

1. Fetched raw Blender manual HTML pages
2. Rewrote each page into dense, LLM-optimized markdown
3. Added frontmatter (DocType, DocPath, DocVersion, Model)
4. Uploaded all files to the Rewritten Manual Store

The rewrite summary is in `rewritten_manual/_rewrite_summary.json`.

To add more rewritten manual pages, follow the same pattern: fetch the HTML, rewrite to dense markdown with frontmatter, save to `rewritten_manual/`, and upload to the store.

---

## Document Frontmatter Standard

Every document in the vector store should have this frontmatter at the top:

```
# Document Title
DocType: technique-guide | manual-rewritten | api-reference
DocPath: techniques/physics_domain/topic_name
DocVersion: 5.0.1
PhysicsDomain: rigid_body | cloth | fluid | particles | soft_body | forces | modeling
---
```

- **DocType**: Tells the search system what kind of document this is
- **DocPath**: Hierarchical path for categorization and deduplication
- **DocVersion**: Blender version this applies to
- **PhysicsDomain**: Which physics system or domain this covers

---

## Code Pattern Memory (Separate System)

Code patterns are stored separately from vector stores, in the local code pattern memory system (`tools/code_pattern_tools.py`). They are small, proven code snippets that can be directly inserted into generated scripts.

**Current count:** 32 patterns (12 original + 24 from seed scripts, some overlap)

**How they're used:** The Research Agent and Learning Agent can retrieve patterns by effect type and insert them into Script Writer prompts. Each pattern has a success rate and improvement score.

**Storage:** JSON files in the orchestrator's data directory, not in OpenAI's cloud.

---

## What's Missing (Coverage Gaps)

### Weak or Missing Areas

| Area | Current State | Impact | Priority |
|------|--------------|--------|----------|
| **Dynamic Paint** | 4 basic manual pages, no technique guides | Low — rarely requested | Low |
| **Soft Body** | 14 manual pages, no technique guides | Medium — useful for deformable objects | Medium |
| **Geometry Nodes** | 1 manual page, mentioned in destruction techniques | High — procedural everything | High |
| **Camera/Composition** | 0 docs | Medium — affects every scene | Medium |
| **HDRI/Environment** | 0 docs | Medium — affects lighting quality | Medium |
| **Render Settings** | 0 docs | Low — mostly hardcoded in pipeline | Low |
| **Advanced Materials** | Basic in cloth/destruction seeds | Medium — affects visual quality | Medium |
| **Lighting Techniques** | 4 light docs, no technique guides | High — #1 visual quality factor | High |

### Recommended Next Seeds

1. **Lighting techniques** — 3-point lighting, rim lights, volumetric lighting, HDRI setup. Every scene needs good lighting and the agent currently guesses.
2. **Geometry Nodes** — procedural scatter, instancing, attribute-driven effects. The most powerful Blender feature for scene building.
3. **Camera and composition** — focal length, depth of field, rule of thirds framing. Affects every render.
4. **Soft body techniques** — bouncing, jiggling, deformable objects. Fills the last major physics gap.

---

## Verification and Testing

### Testing a Search Query

```bash
cd agents/blender-vfx-orchestrator
venv/bin/python -c "
from tools.semantic_docs_tools import _search_vector_store
results = _search_vector_store('cloth flag wind pinning', max_results=3, intent='manual')
for r in results:
    print(f'{r[\"score\"]:.3f} {r[\"filename\"]} — {r[\"content\"][:100]}...')
"
```

### Checking Store File Count

```bash
venv/bin/python -c "
from openai import OpenAI
client = OpenAI()
store = client.vector_stores.retrieve('vs_699b7e6221bc81919ba2f4a1eae11588')
print(f'Files: {store.file_counts.completed} completed, {store.file_counts.total} total')
print(f'Status: {store.status}')
"
```

### Listing All Files in a Store

```bash
venv/bin/python -c "
from openai import OpenAI
client = OpenAI()
files = client.vector_stores.files.list('vs_699b7e6221bc81919ba2f4a1eae11588', limit=200)
for f in files.data:
    print(f'{f.id} {f.status} {f.created_at}')
print(f'Total: {len(files.data)}')
"
```

---

## Cost Considerations

- **Vector store storage:** Charged per GB per day. 174 small markdown files is negligible (~$0.01/day).
- **Search queries:** Each `vector_stores.search()` call costs a small amount. Budget is tracked in the pipeline's budget tracker under "documentation" category ($8/month allocation).
- **File uploads:** One-time cost per file. Seed scripts upload 5-6 files each (~$0.01 per run).
- The search system has a file_search fallback that uses the Responses API — this is more expensive. It only triggers when direct search returns weak results (score < threshold).

---

## Key Files

| File | Purpose |
|------|---------|
| `tools/semantic_docs_tools.py` | Search implementation, intent classification, store routing |
| `scripts/seed_*.py` | Upload technique guides and code patterns |
| `scripts/wipe_kb.py` | Backup and wipe knowledge base (use with caution) |
| `rewritten_manual/` | 131 LLM-optimized Blender manual rewrites |
| `rewritten_manual/_rewrite_summary.json` | Metadata about the rewrite process |
