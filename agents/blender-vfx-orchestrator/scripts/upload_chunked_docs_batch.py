#!/usr/bin/env python3
"""
Batch Upload Chunked Blender Documentation to OpenAI Vector Stores.

Uses OpenAI's batch file upload API for much faster uploads.
Uploads files in batches of 100 using file_batches.create_and_poll().

Usage:
    python scripts/upload_chunked_docs_batch.py --api-only
    python scripts/upload_chunked_docs_batch.py --manual-only
    python scripts/upload_chunked_docs_batch.py  # Both
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed")
    sys.exit(1)

# Import chunker
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.chunk_blender_docs import (
    chunk_manual_page,
    chunk_api_page,
    DocChunk,
    ChunkingStats,
    should_skip_file,
    DEFAULT_DOCS_BASE,
)

# Vector store IDs
MANUAL_STORE_ID = os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID", "vs_6975104199c08191acb1495c86d581ce")
API_STORE_ID = os.getenv("BLENDER_API_VECTOR_STORE_ID", "vs_697512bf81c481919ae3b7a8ffb8223a")

# Paths
DOCS_BASE = DEFAULT_DOCS_BASE

# Batch settings
BATCH_SIZE = 100  # Files per batch upload
UPLOAD_WORKERS = 8  # Parallel file creation threads


def create_temp_file(chunk: DocChunk, temp_dir: Path) -> Path:
    """Create a temp file for a chunk."""
    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in chunk.chunk_id)[:80]
    file_path = temp_dir / f"{safe_name}.md"
    file_path.write_text(chunk.to_upload_text(), encoding='utf-8')
    return file_path


def upload_file_to_openai(client: OpenAI, file_path: Path) -> str:
    """Upload a single file to OpenAI and return file_id."""
    with open(file_path, 'rb') as f:
        response = client.files.create(file=f, purpose="assistants")
    return response.id


def batch_upload_chunks(
    client: OpenAI,
    vector_store_id: str,
    chunks: List[DocChunk],
    doc_type: str
) -> dict:
    """
    Upload chunks using batch API for speed.
    """
    results = {"total": len(chunks), "success": 0, "failed": 0}

    print(f"\nUploading {len(chunks)} {doc_type} chunks to {vector_store_id}...")
    start_time = time.time()

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Process in batches
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"  Batch {batch_num}/{total_batches}: {len(batch_chunks)} chunks...")

            # Create temp files
            temp_files = []
            for chunk in batch_chunks:
                try:
                    temp_file = create_temp_file(chunk, temp_path)
                    temp_files.append(temp_file)
                except Exception as e:
                    print(f"    Error creating temp file: {e}", file=sys.stderr)
                    results["failed"] += 1

            if not temp_files:
                continue

            # Upload files in parallel
            file_ids = []
            with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
                future_to_file = {
                    executor.submit(upload_file_to_openai, client, f): f
                    for f in temp_files
                }

                for future in as_completed(future_to_file):
                    try:
                        file_id = future.result()
                        file_ids.append(file_id)
                    except Exception as e:
                        results["failed"] += 1
                        print(f"    Upload error: {e}", file=sys.stderr)

            # Add files to vector store in batch
            if file_ids:
                try:
                    batch = client.vector_stores.file_batches.create_and_poll(
                        vector_store_id=vector_store_id,
                        file_ids=file_ids
                    )
                    results["success"] += batch.file_counts.completed
                    results["failed"] += batch.file_counts.failed
                    print(f"    Batch complete: {batch.file_counts.completed} ok, {batch.file_counts.failed} failed")
                except Exception as e:
                    print(f"    Batch error: {e}", file=sys.stderr)
                    results["failed"] += len(file_ids)

            # Clean up temp files
            for f in temp_files:
                try:
                    f.unlink()
                except:
                    pass

            # Progress
            elapsed = time.time() - start_time
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - done) / rate if rate > 0 else 0
            print(f"    Total: {done}/{len(chunks)} ({rate:.1f}/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")
    return results


def process_and_upload(
    client: OpenAI,
    path: Path,
    base_path: Path,
    vector_store_id: str,
    doc_type: str
) -> dict:
    """Process and upload docs."""
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return {"total": 0, "success": 0, "failed": 0}

    html_files = [f for f in path.rglob("*.html") if not should_skip_file(f)]
    print(f"\nProcessing {len(html_files)} {doc_type} HTML files...")

    # Chunk files
    all_chunks = []
    for i, html_file in enumerate(html_files):
        try:
            if doc_type == "api":
                chunks = chunk_api_page(html_file, base_path)
            else:
                chunks = chunk_manual_page(html_file, base_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Chunk error {html_file.name}: {e}", file=sys.stderr)

        if (i + 1) % 200 == 0:
            print(f"  Chunked {i + 1}/{len(html_files)} files ({len(all_chunks)} chunks)...")

    print(f"  Generated {len(all_chunks)} chunks")

    # Upload
    return batch_upload_chunks(client, vector_store_id, all_chunks, doc_type)


def main():
    parser = argparse.ArgumentParser(description="Batch upload Blender docs")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--docs-base", type=Path, default=DOCS_BASE)
    args = parser.parse_args()

    print("=" * 60)
    print("Blender Documentation Batch Upload")
    print("=" * 60)

    client = OpenAI()
    results = {}

    if not args.api_only:
        results["manual"] = process_and_upload(
            client,
            args.docs_base / "blender_manual_html",
            args.docs_base,
            MANUAL_STORE_ID,
            "manual"
        )

    if not args.manual_only:
        results["api"] = process_and_upload(
            client,
            args.docs_base / "blender_python_reference_5_0",
            args.docs_base,
            API_STORE_ID,
            "api"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dtype, res in results.items():
        print(f"{dtype.upper()}: {res['success']}/{res['total']} ({res['failed']} failed)")

    # Final status
    print("\nFinal store status:")
    if not args.api_only:
        vs = client.vector_stores.retrieve(MANUAL_STORE_ID)
        print(f"  Manual: {vs.file_counts.total} files")
    if not args.manual_only:
        vs = client.vector_stores.retrieve(API_STORE_ID)
        print(f"  API: {vs.file_counts.total} files")


if __name__ == "__main__":
    main()
