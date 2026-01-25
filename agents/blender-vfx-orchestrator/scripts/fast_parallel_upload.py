#!/usr/bin/env python3
"""
Fast Parallel Upload for Vector Store.

Uses concurrent batch uploads with larger batch sizes to dramatically speed up ingestion.
OpenAI supports up to 500 files per batch - we use 500.
Multiple batches are uploaded in parallel using ThreadPoolExecutor.

Target: 10,000+ files in ~30-60 minutes instead of hours.
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

# Vector store IDs
MANUAL_STORE_ID = "vs_6975104199c08191acb1495c86d581ce"
API_STORE_ID = "vs_697512bf81c481919ae3b7a8ffb8223a"

# Aggressive batch settings
BATCH_SIZE = 500  # Max allowed by OpenAI
PARALLEL_BATCHES = 4  # Number of batches to upload simultaneously
FILE_UPLOAD_WORKERS = 16  # Parallel file uploads within a batch

# Progress tracking
progress_lock = Lock()
total_uploaded = 0
total_failed = 0


def get_chunk_filename(chunk: DocChunk) -> str:
    """Get the filename for a chunk."""
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in chunk.chunk_id)[:80]
    return f"{safe_name}.md"


def load_already_uploaded(filepath: str = "/tmp/already_uploaded.txt") -> set:
    """Load set of already uploaded filenames."""
    try:
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def upload_single_file(client: OpenAI, file_path: Path) -> str | None:
    """Upload a single file and return its ID."""
    try:
        with open(file_path, 'rb') as f:
            uploaded = client.files.create(file=f, purpose='assistants')
            return uploaded.id
    except Exception as e:
        print(f"    [!] Failed to upload {file_path.name}: {e}", file=sys.stderr)
        return None


def upload_batch(
    client: OpenAI,
    store_id: str,
    chunks: list,
    batch_num: int,
    total_batches: int,
    temp_dir: Path
) -> dict:
    """Upload a batch of chunks with parallel file uploads."""
    global total_uploaded, total_failed

    start_time = time.time()

    # Create temp files in parallel
    file_paths = []
    for chunk in chunks:
        filename = get_chunk_filename(chunk)
        file_path = temp_dir / f"batch{batch_num}_{filename}"
        file_path.write_text(chunk.to_upload_text(), encoding='utf-8')
        file_paths.append(file_path)

    # Upload files to OpenAI in parallel
    file_ids = []
    with ThreadPoolExecutor(max_workers=FILE_UPLOAD_WORKERS) as executor:
        futures = {executor.submit(upload_single_file, client, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                file_ids.append(result)

    # Clean up temp files immediately
    for fp in file_paths:
        try:
            fp.unlink()
        except:
            pass

    if not file_ids:
        return {"ok": 0, "failed": len(chunks), "batch_num": batch_num}

    # Add to vector store as a batch
    try:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=store_id,
            file_ids=file_ids
        )
        ok = batch.file_counts.completed
        failed = batch.file_counts.failed

        elapsed = time.time() - start_time
        rate = ok / elapsed if elapsed > 0 else 0

        with progress_lock:
            total_uploaded += ok
            total_failed += failed
            print(f"  [Batch {batch_num}/{total_batches}] {ok} ok, {failed} failed ({rate:.1f}/s) | Total: {total_uploaded}", flush=True)

        return {"ok": ok, "failed": failed, "batch_num": batch_num}
    except Exception as e:
        print(f"  [Batch {batch_num}] ERROR: {e}", file=sys.stderr)
        return {"ok": 0, "failed": len(file_ids), "batch_num": batch_num}


def main():
    global total_uploaded, total_failed

    parser = argparse.ArgumentParser(description="Fast parallel upload of chunked docs")
    parser.add_argument("--api-only", action="store_true", help="Only upload API docs")
    parser.add_argument("--manual-only", action="store_true", help="Only upload Manual docs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Files per batch (max 500)")
    parser.add_argument("--parallel", type=int, default=PARALLEL_BATCHES, help="Parallel batch uploads")
    args = parser.parse_args()

    batch_size = min(args.batch_size, 500)  # OpenAI max
    parallel_batches = args.parallel

    print("=" * 60)
    print("FAST PARALLEL VECTOR STORE UPLOAD")
    print("=" * 60)
    print(f"Batch size: {batch_size} files")
    print(f"Parallel batches: {parallel_batches}")
    print(f"File upload workers: {FILE_UPLOAD_WORKERS}")
    print()

    client = OpenAI()
    already_uploaded = load_already_uploaded()
    print(f"Already uploaded: {len(already_uploaded)} files")

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

        # Reset counters
        total_uploaded = 0
        total_failed = 0
        start_time = time.time()

        # Get HTML files
        html_files = sorted(doc_dir.glob("**/*.html"))
        print(f"\n{'='*60}")
        print(f"Processing {len(html_files)} {doc_type.upper()} HTML files...")

        # Generate chunks
        all_chunks = []
        chunk_func = chunk_api_page if doc_type == "api" else chunk_manual_page
        for i, html_file in enumerate(html_files):
            if (i + 1) % 500 == 0:
                print(f"  Chunked {i+1}/{len(html_files)} files ({len(all_chunks)} chunks)...", flush=True)
            try:
                chunks = chunk_func(html_file, DOCS_BASE)
                all_chunks.extend(chunks)
            except Exception as e:
                pass  # Skip errors silently

        print(f"  Generated {len(all_chunks)} chunks")

        # Filter out already uploaded
        chunks_to_upload = []
        for chunk in all_chunks:
            filename = get_chunk_filename(chunk)
            if filename not in already_uploaded:
                chunks_to_upload.append(chunk)

        skipped = len(all_chunks) - len(chunks_to_upload)
        print(f"  Skipping {skipped} already uploaded")
        print(f"  Uploading {len(chunks_to_upload)} new chunks")

        if not chunks_to_upload:
            print("  Nothing to upload!")
            continue

        # Create batches
        batches = []
        for i in range(0, len(chunks_to_upload), batch_size):
            batches.append(chunks_to_upload[i:i + batch_size])

        total_batches = len(batches)
        print(f"  {total_batches} batches of up to {batch_size} files each")
        print()

        # Upload batches in parallel
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

                # Wait for all batches
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  Batch error: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        rate = total_uploaded / elapsed if elapsed > 0 else 0
        print(f"\n{doc_type.upper()} COMPLETE: {total_uploaded} uploaded, {total_failed} failed")
        print(f"Time: {elapsed/60:.1f} minutes ({rate:.1f} files/sec)")

    # Final status
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    for name, sid in [("API", API_STORE_ID), ("Manual", MANUAL_STORE_ID)]:
        try:
            vs = client.vector_stores.retrieve(sid)
            print(f"{name}: {vs.file_counts.completed:,} files ({vs.status})")
        except:
            pass

    print("=" * 60)


if __name__ == "__main__":
    main()
