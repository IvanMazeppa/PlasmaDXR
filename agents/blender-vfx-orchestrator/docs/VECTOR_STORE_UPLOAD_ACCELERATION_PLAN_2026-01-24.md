# Vector Store Upload Acceleration Plan (2026-01-24)

## Purpose
Speed up Blender docs ingestion into OpenAI vector stores without sacrificing
doc‑grounding quality or traceability (stable `DocPath`, `DocType`).

This plan is tailored to the Blender manual + Python API scale and the
existing pipeline in `scripts/upload_blender_docs_to_vectorstore.py`.

---

## Current Bottlenecks (Observed / Likely)
- **Per‑file upload overhead:** thousands of small files = high latency + API overhead.
- **Sequential uploads:** limited concurrency; network latency dominates.
- **Doc refs lose meaning:** temp file names replace real doc paths.
- **Large HTML files truncated:** a hard cap reduces recall for long API pages.

---

## Fastest Strategy (Recommended)
### 1) Pre‑chunk + Bundle (Reduce file count 100×)
Instead of uploading thousands of tiny files, produce **bundles**:
- Manual bundles (e.g., `manual_bundle_0001.md`)
- API bundles (e.g., `api_bundle_0001.md`)

Each bundle contains many chunks, each with a header:
```
# <Title>
DocType: manual|api
DocPath: <relative/path.html>
DocVersion: 5.0.1
ChunkId: <path>#<section>#<n>
---
<chunk text>
```

**Result:** far fewer files → dramatically fewer API calls.

### 2) Use File Batches (Native Fast Path)
The OpenAI vector store API supports batch upload via
`vector_stores.file_batches.upload_and_poll` (up to ~500 files per batch).

**Why it matters:** The batch endpoint is optimized for large ingestion.

### 3) Parallelize by Batch (Not by File)
Run **N batch uploads in parallel** instead of N single file uploads.

Use a concurrency cap (e.g., 2–6) to avoid rate‑limit churn.

---

## Proposed Ingestion Pipeline
### Step A: Build Bundles (Local)
1) Parse HTML → text (manual + API separately).
2) Chunk by section (manual) or symbol (API).
3) Write bundles to `docs_bundles/`:
   - `manual_bundle_####.md`
   - `api_bundle_####.md`
4) Keep a manifest JSON with:
   - bundle filename
   - chunk count
   - total bytes
   - source type (manual/api)

### Step B: Upload Bundles in Batches
1) Create two vector stores:
   - `BLENDER_MANUAL_VECTOR_STORE_ID`
   - `BLENDER_API_VECTOR_STORE_ID`
2) Upload bundles using `upload_and_poll` in batches:
   - batch size: 100–500 files
   - concurrency: 2–6 batches at a time
3) Track file IDs per bundle in the manifest for resume support.

### Step C: Verify Coverage
Run a verification script after upload:
- Count bundles ingested per store
- Test 5–10 canonical queries
- Fail fast if queries return 0 results

---

## Implementation Notes (for Your Repo)
### 1) Update the uploader
Replace per‑file uploads with:
- bundle creation (manual + api)
- `vector_stores.file_batches.upload_and_poll`
- batch‑level concurrency (not file‑level)

### 2) Adjust the size cap
Your current script caps content at **512KB**. If this is a self‑imposed limit,
consider increasing bundle size (while staying under OpenAI’s file size limit).
This alone can reduce API calls by an order of magnitude.

### 3) Preserve DocRefs
Every chunk in a bundle should include `DocPath` so:
- `doc_refs` can always point to real manual/API paths
- guardrails won’t break on temp filenames

---

## Example: Batch Upload Skeleton (Pseudo)
```
files = list_bundle_paths()
batches = chunk(files, size=200)

async def upload_batch(bundle_paths, vector_store_id):
    return client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id,
        files=[open(p, "rb") for p in bundle_paths]
    )

await run_batches_in_parallel(batches, concurrency=4)
```

---

## Resume Strategy (Critical for Long Uploads)
- Write a manifest per bundle:
  - `bundle_name`, `hash`, `status`, `file_ids`
- If upload resumes, skip completed bundles.
- If a batch partially fails, retry only failed bundles.

---

## Expected Impact
- **10×–100× fewer API calls** (bundle vs file)
- **Much faster ingestion** (batch + parallel)
- **Stable doc_refs** (no temp filenames)
- **Lower failure rate** due to resumable batches

---

## Minimal Change Path (If You Don’t Want a Rewrite)
1) Keep current extraction logic.
2) Combine 50–200 small files into single “bundle” files.
3) Use `upload_and_poll` for those bundles only.
4) Parallelize on batch, not file.
