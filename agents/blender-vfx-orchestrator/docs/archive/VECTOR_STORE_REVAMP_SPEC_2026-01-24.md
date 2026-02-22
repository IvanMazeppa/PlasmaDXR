# Vector Store Revamp Spec (Blender Docs)

## Goal
Deliver reliable, Blender‑5‑only doc grounding for agents with:
- Stable `doc_refs` (not temp filenames)
- High‑signal retrieval for Mantaflow/volume workflows
- Deterministic coverage checks (manual + Python API)
- Minimal false negatives when agents query docs

## Current Pain Points (Observed)
- `doc_refs` are temp filenames from uploads, not stable doc paths.
- Vector store has “no results” for common queries, which triggers guardrails.
- Single mixed store makes intent queries noisy (manual vs API vs scripting).
- Upload converts large HTML into flattened text; important headers/paths get lost.
- Agents rely on result content without clear doc identity.

## Target Design (High‑Level)
**Two Stores + Deterministic Metadata**
1) **Manual Store** (`blender_manual_5_0`): Blender manual only  
2) **API Store** (`blender_api_5_0`): Python API reference only  

**Hard rule:** Every chunk must contain an explicit `DocPath:` (manual or API path), and `DocType:` (`manual|api`) in the header.

This enables:
- `doc_refs` = actual manual/API paths
- domain‑aware tool routing (manual vs API)
- “missing docs” detection is clean and actionable

## Document Format (Upload Artifact)
Use a uniform header for every chunk:
```
# <Title>
DocType: manual|api
DocPath: <relative/path.html>     # same as source path
DocVersion: 5.0.1
SourceRepo: agents/blender-manual
ChunkId: <path>#<section>#<n>
---
<chunk content>
```

### Why this matters
- Agents can reliably extract the real doc path
- `doc_refs` no longer point to temp filenames
- You can link results back to the local repo instantly

## Ingestion Pipeline (Proposed)
### 1) Pre‑processing
- Use local repo: `agents/blender-manual` (already in tree)
- Source HTML from:
  - Manual HTML
  - Python API HTML

### 2) Chunking Strategy
**Manual**
- Chunk by section heading (h2/h3).  
- Target chunk size: 500–1200 words.
- Keep “overview” chunks intact (don’t split index pages too aggressively).

**Python API**
- Chunk per class or function (bpy.types.*, bpy.ops.*).
- Preserve signatures and property lists.
- Target chunk size: 300–800 words.

### 3) Metadata & Header Injection
Inject the header above into each chunk so:
- `DocPath` is the canonical doc reference.
- `DocType` drives routing.
- `ChunkId` prevents duplicates and enables provenance.

### 4) Upload Strategy
Option A (recommended): **Two vector stores**  
- `BLENDER_MANUAL_VECTOR_STORE_ID`  
- `BLENDER_API_VECTOR_STORE_ID`  

Option B: Single store with strict `DocType` metadata  
- Requires explicit filtering in tools (less ideal).

### 5) Verification & Health Checks
Run a verification script after upload that:
- Counts docs uploaded per store (manual/API)
- Verifies each chunk contains `DocPath` and `DocType`
- Runs N fixed queries and checks non‑zero results:
  - `bpy.types.FluidDomainSettings`
  - `Mantaflow domain settings`
  - `Principled Volume`
  - `bpy.ops.fluid.bake_all`

## Retrieval Strategy (Tools)
**Replace “one generic search” with a two‑stage router:**
1) **Intent router** picks store:  
   - “bpy.” → API store  
   - “Mantaflow”, “volume”, “shader”, “workflow” → Manual store  
2) **Doc search bundle** (single tool call)  

### Example Query Plan
- Manual:
  - “Mantaflow domain settings”
  - “Principled Volume temperature attribute”
  - “volume rendering blackbody”
- API:
  - “bpy.types.FluidDomainSettings”
  - “bpy.types.FluidFlowSettings”
  - “bpy.ops.fluid.bake_all”

## Stability Improvements
- **Doc refs**: Always extracted from `DocPath` header.
- **Fallback**: If vector store returns zero, fall back to local `agents/blender-manual` index.
- **Version gating**: Explicitly embed `DocVersion: 5.0.1` in each chunk.
- **Anti‑pollution**: Reject any doc chunk that does not include version header.

## Local Fallback (Offline)
Use `agents/blender-manual` as a guaranteed fallback:
- If vector search returns 0, run local index search.
- Return doc refs from local paths (same `DocPath` format).

## Migration Plan
1) Build new chunker for manual + API (split by h2/h3 or class def).  
2) Upload to **two stores** with stable headers.  
3) Update tools to query stores by intent and extract `DocPath`.  
4) Add health‑check script (preflight) used in CI or before runs.  
5) Run a 2‑iteration shakedown and compare doc hit rate.

## Expected Impact
- **Doc hit rate** increases, fewer “no results found”.
- `doc_refs` become stable and meaningful for guardrails.
- Faster debugging (paths map to local manual/API HTML).
- Safer Blender‑5‑only enforcement.

## Notes on Existing Scripts
Current uploader (`scripts/upload_blender_docs_to_vectorstore.py`):
- Uploads full HTML converted to large text blobs.
- Uses temp `.md` filenames, which is why `doc_refs` are noisy.

This spec replaces that with **chunked, header‑rich uploads** and split stores.

