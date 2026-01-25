#!/usr/bin/env python3
"""
Fast Individual File Upload - Maximum Parallelism.

Uploads individual chunk files using:
- 500 files per batch (OpenAI max)
- 6 parallel batch uploads
- 20 parallel file uploads per batch

Target: ~4,400 manual chunks in 20-30 minutes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

# Default store IDs (can be overridden via args)
MANUAL_STORE_ID = "vs_6975104199c08191acb1495c86d581ce"
API_STORE_ID = "vs_697512bf81c481919ae3b7a8ffb8223a"

# Aggressive parallelism settings
BATCH_SIZE = 500  # OpenAI max
PARALLEL_BATCHES = 6  # Concurrent batch uploads
FILE_UPLOAD_WORKERS = 20  # Parallel file uploads per batch

# Progress tracking
progress_lock = Lock()
total_uploaded = 0
total_failed = 0
start_time = None


def get_chunk_filename(chunk: DocChunk) -> str:
    """Get the filename for a chunk."""
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in chunk.chunk_id)[:80]
    return f"{safe_name}.md"


def upload_single_file(client: OpenAI, file_path: Path) -> str | None:
    """Upload a single file and return its ID."""
    try:
        with open(file_path, 'rb') as f:
            uploaded = client.files.create(file=f, purpose='assistants')
            return uploaded.id
    except Exception as e:
        return None


def upload_batch(
    client: OpenAI,
    store_id: str,
    chunks: list[DocChunk],
    batch_num: int,
    total_batches: int,
    temp_dir: Path
) -> dict:
    """Upload a batch of chunks with maximum parallelism."""
    global total_uploaded, total_failed, start_time

    batch_start = time.time()

    # Create temp files
    file_paths = []
    for chunk in chunks:
        filename = get_chunk_filename(chunk)
        file_path = temp_dir / f"b{batch_num}_{filename}"
        file_path.write_text(chunk.to_upload_text(), encoding='utf-8')
        file_paths.append(file_path)

    # Upload files in parallel
    file_ids = []
    failed_uploads = 0
    with ThreadPoolExecutor(max_workers=FILE_UPLOAD_WORKERS) as executor:
        futures = {executor.submit(upload_single_file, client, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                file_ids.append(result)
            else:
                failed_uploads += 1

    # Clean up temp files
    for fp in file_paths:
        try:
            fp.unlink()
        except:
            pass

    if not file_ids:
        with progress_lock:
            total_failed += len(chunks)
        return {"ok": 0, "failed": len(chunks)}

    # Add to vector store as batch
    try:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=store_id,
            file_ids=file_ids
        )
        ok = batch.file_counts.completed
        failed = batch.file_counts.failed + failed_uploads

        with progress_lock:
            total_uploaded += ok
            total_failed += failed
            elapsed = time.time() - start_time
            rate = total_uploaded / elapsed if elapsed > 0 else 0
            remaining = (total_batches - batch_num) * BATCH_SIZE
            eta = remaining / rate if rate > 0 else 0

        batch_time = time.time() - batch_start
        print(f"  [Batch {batch_num:2d}/{total_batches}] {ok:3d} ok, {failed} failed ({batch_time:.0f}s) | Total: {total_uploaded:,} ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)

        return {"ok": ok, "failed": failed}
    except Exception as e:
        print(f"  [Batch {batch_num}] ERROR: {e}", file=sys.stderr)
        with progress_lock:
            total_failed += len(file_ids)
        return {"ok": 0, "failed": len(file_ids)}


def main():
    global total_uploaded, total_failed, start_time

    parser = argparse.ArgumentParser(description="Fast individual file upload")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--store-id", type=str, help="Override store ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--parallel", type=int, default=PARALLEL_BATCHES)
    parser.add_argument("--workers", type=int, default=FILE_UPLOAD_WORKERS)
    args = parser.parse_args()

    batch_size = min(args.batch_size, 500)
    parallel_batches = args.parallel
    file_workers = args.workers

    print("=" * 70)
    print("FAST INDIVIDUAL FILE UPLOAD - MAXIMUM PARALLELISM")
    print("=" * 70)
    print(f"Batch size: {batch_size} files (OpenAI max: 500)")
    print(f"Parallel batches: {parallel_batches}")
    print(f"File upload workers: {file_workers}")
    print()

    client = OpenAI()

    # Determine what to upload
    if args.api_only:
        store_id = args.store_id or API_STORE_ID
        doc_types = [("api", API_DIR, store_id)]
    elif args.manual_only:
        store_id = args.store_id or MANUAL_STORE_ID
        doc_types = [("manual", MANUAL_DIR, store_id)]
    else:
        doc_types = [
            ("manual", MANUAL_DIR, args.store_id or MANUAL_STORE_ID),
        ]

    for doc_type, doc_dir, store_id in doc_types:
        if not doc_dir.exists():
            print(f"\nSkipping {doc_type}: {doc_dir} not found")
            continue

        # Reset counters
        total_uploaded = 0
        total_failed = 0
        start_time = time.time()

        print(f"\n{'=' * 70}")
        print(f"Uploading {doc_type.upper()} to {store_id}")
        print("=" * 70)

        # Generate chunks
        html_files = sorted(doc_dir.glob("**/*.html"))
        print(f"Step 1: Generating chunks from {len(html_files)} HTML files...")

        chunk_func = chunk_api_page if doc_type == "api" else chunk_manual_page
        all_chunks = []
        for i, html_file in enumerate(html_files):
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(html_files)} files ({len(all_chunks)} chunks)...", flush=True)
            try:
                chunks = chunk_func(html_file, DOCS_BASE)
                all_chunks.extend(chunks)
            except:
                pass

        print(f"  Generated {len(all_chunks):,} chunks")

        if not all_chunks:
            print("  No chunks to upload!")
            continue

        # Create batches
        batches = []
        for i in range(0, len(all_chunks), batch_size):
            batches.append(all_chunks[i:i + batch_size])

        total_batches = len(batches)
        print(f"\nStep 2: Uploading {len(all_chunks):,} chunks in {total_batches} batches...")
        print(f"  ({parallel_batches} batches in parallel, {file_workers} file uploads each)")
        print()

        # Upload with parallel batches
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
                futures = {}
                for batch_idx, batch_chunks in enumerate(batches):
                    future = executor.submit(
                        upload_batch,
                        client,
                        store_id,
                        batch_chunks,
                        batch_idx + 1,
                        total_batches,
                        temp_path
                    )
                    futures[future] = batch_idx

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  Batch error: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        rate = total_uploaded / elapsed if elapsed > 0 else 0

        print(f"\n{doc_type.upper()} COMPLETE:")
        print(f"  Uploaded: {total_uploaded:,}")
        print(f"  Failed: {total_failed}")
        print(f"  Time: {elapsed/60:.1f} minutes ({rate:.1f} files/sec)")

    # Final status
    print("\n" + "=" * 70)
    print("FINAL STORE STATUS")
    print("=" * 70)

    # Check the store we just uploaded to
    try:
        vs = client.vector_stores.retrieve(store_id)
        print(f"Store: {vs.name}")
        print(f"  Completed: {vs.file_counts.completed:,}")
        print(f"  In progress: {vs.file_counts.in_progress}")
        print(f"  Failed: {vs.file_counts.failed}")
    except Exception as e:
        print(f"Error checking store: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
