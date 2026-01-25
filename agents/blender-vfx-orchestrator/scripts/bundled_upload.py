#!/usr/bin/env python3
"""
Bundled Upload for Vector Store - 100x Faster.

Instead of uploading 13,000+ individual files, bundles ~100 chunks per file.
This reduces API calls from 13,000 to ~130.

Each bundle contains multiple chunks with headers preserved:
```
# <Title>
DocType: api
DocPath: bpy.types.FluidDomainSettings.html#flame_smoke
DocVersion: 5.0.1
ChunkId: unique-id
---
<content>

================================================================================

# <Next Title>
...
```
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from scripts.chunk_blender_docs import (
    chunk_manual_page,
    chunk_api_page,
    DEFAULT_DOCS_BASE as DOCS_BASE,
    DEFAULT_MANUAL_PATH as MANUAL_DIR,
    DEFAULT_API_PATH as API_DIR,
    DocChunk,
)

# Vector store IDs
MANUAL_STORE_ID = "vs_6975104199c08191acb1495c86d581ce"
API_STORE_ID = "vs_697512bf81c481919ae3b7a8ffb8223a"

# Bundle settings
CHUNKS_PER_BUNDLE = 100  # Combine 100 chunks into 1 file
BATCH_SIZE = 50  # Upload 50 bundles per batch (= 5000 chunks worth)
PARALLEL_BATCHES = 3  # Run 3 batch uploads in parallel

CHUNK_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"


def create_bundle_content(chunks: list[DocChunk]) -> str:
    """Combine multiple chunks into a single bundle file."""
    parts = []
    for chunk in chunks:
        parts.append(chunk.to_upload_text())
    return CHUNK_SEPARATOR.join(parts)


def generate_chunks(doc_type: str, doc_dir: Path) -> list[DocChunk]:
    """Generate all chunks for a doc type."""
    html_files = sorted(doc_dir.glob("**/*.html"))
    print(f"  Found {len(html_files)} HTML files")

    chunk_func = chunk_api_page if doc_type == "api" else chunk_manual_page
    all_chunks = []

    for i, html_file in enumerate(html_files):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(html_files)} files ({len(all_chunks)} chunks)...", flush=True)
        try:
            chunks = chunk_func(html_file, DOCS_BASE)
            all_chunks.extend(chunks)
        except Exception:
            pass

    print(f"  Generated {len(all_chunks)} total chunks")
    return all_chunks


def create_bundles(chunks: list[DocChunk], doc_type: str, temp_dir: Path) -> list[Path]:
    """Create bundle files from chunks."""
    bundles = []

    for i in range(0, len(chunks), CHUNKS_PER_BUNDLE):
        batch_chunks = chunks[i:i + CHUNKS_PER_BUNDLE]
        bundle_num = i // CHUNKS_PER_BUNDLE + 1

        bundle_content = create_bundle_content(batch_chunks)
        bundle_path = temp_dir / f"{doc_type}_bundle_{bundle_num:04d}.md"
        bundle_path.write_text(bundle_content, encoding='utf-8')
        bundles.append(bundle_path)

    return bundles


def upload_batch(client: OpenAI, store_id: str, bundle_paths: list[Path], batch_num: int, total_batches: int) -> dict:
    """Upload a batch of bundles."""
    start = time.time()

    # Upload files to OpenAI
    file_ids = []
    for bp in bundle_paths:
        try:
            with open(bp, 'rb') as f:
                uploaded = client.files.create(file=f, purpose='assistants')
                file_ids.append(uploaded.id)
        except Exception as e:
            print(f"    [!] Failed: {bp.name}: {e}", file=sys.stderr)

    if not file_ids:
        return {"ok": 0, "failed": len(bundle_paths)}

    # Add to vector store
    try:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=store_id,
            file_ids=file_ids
        )
        ok = batch.file_counts.completed
        failed = batch.file_counts.failed
        elapsed = time.time() - start

        chunks_uploaded = ok * CHUNKS_PER_BUNDLE
        print(f"  [Batch {batch_num}/{total_batches}] {ok} bundles ({chunks_uploaded} chunks) in {elapsed:.1f}s", flush=True)

        return {"ok": ok, "failed": failed}
    except Exception as e:
        print(f"  [Batch {batch_num}] ERROR: {e}", file=sys.stderr)
        return {"ok": 0, "failed": len(file_ids)}


def main():
    parser = argparse.ArgumentParser(description="Bundled upload - 100x faster")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--chunks-per-bundle", type=int, default=CHUNKS_PER_BUNDLE)
    args = parser.parse_args()

    chunks_per_bundle = args.chunks_per_bundle

    print("=" * 60)
    print("BUNDLED VECTOR STORE UPLOAD (100x Faster)")
    print("=" * 60)
    print(f"Chunks per bundle: {chunks_per_bundle}")
    print(f"Batch size: {BATCH_SIZE} bundles")
    print(f"Parallel batches: {PARALLEL_BATCHES}")
    print()

    client = OpenAI()

    # Determine what to upload
    if args.api_only:
        doc_types = [("api", API_DIR, API_STORE_ID)]
    elif args.manual_only:
        doc_types = [("manual", MANUAL_DIR, MANUAL_STORE_ID)]
    else:
        doc_types = [
            ("api", API_DIR, API_STORE_ID),
            ("manual", MANUAL_DIR, MANUAL_STORE_ID),
        ]

    for doc_type, doc_dir, store_id in doc_types:
        if not doc_dir.exists():
            print(f"\nSkipping {doc_type}: {doc_dir} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {doc_type.upper()}")
        print("=" * 60)

        start_time = time.time()

        # Generate chunks
        print("Step 1: Generating chunks...")
        chunks = generate_chunks(doc_type, doc_dir)

        if not chunks:
            print("  No chunks generated!")
            continue

        # Create bundles
        print(f"\nStep 2: Creating bundles ({chunks_per_bundle} chunks each)...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundles = create_bundles(chunks, doc_type, temp_path)
            print(f"  Created {len(bundles)} bundles")

            # Upload in batches
            print(f"\nStep 3: Uploading bundles...")
            total_ok = 0
            total_failed = 0

            # Split bundles into batches
            batch_groups = []
            for i in range(0, len(bundles), BATCH_SIZE):
                batch_groups.append(bundles[i:i + BATCH_SIZE])

            total_batches = len(batch_groups)
            print(f"  {total_batches} batches of up to {BATCH_SIZE} bundles each")

            # Upload batches in parallel
            with ThreadPoolExecutor(max_workers=PARALLEL_BATCHES) as executor:
                futures = {}
                for batch_idx, batch_bundles in enumerate(batch_groups):
                    future = executor.submit(
                        upload_batch,
                        client,
                        store_id,
                        batch_bundles,
                        batch_idx + 1,
                        total_batches
                    )
                    futures[future] = batch_idx

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        total_ok += result["ok"]
                        total_failed += result["failed"]
                    except Exception as e:
                        print(f"  Batch error: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        chunks_uploaded = total_ok * chunks_per_bundle
        rate = chunks_uploaded / elapsed if elapsed > 0 else 0

        print(f"\n{doc_type.upper()} COMPLETE:")
        print(f"  Bundles: {total_ok} uploaded, {total_failed} failed")
        print(f"  Chunks: ~{chunks_uploaded} (in {len(chunks)} total)")
        print(f"  Time: {elapsed/60:.1f} minutes ({rate:.0f} chunks/sec)")

    # Final status
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    for name, sid in [("API", API_STORE_ID), ("Manual", MANUAL_STORE_ID)]:
        try:
            vs = client.vector_stores.retrieve(sid)
            print(f"{name}: {vs.file_counts.completed} files ({vs.status})")
        except:
            pass

    print("=" * 60)


if __name__ == "__main__":
    main()
