#!/usr/bin/env python3
"""
Upload Chunked Blender Documentation to OpenAI Vector Stores.

Uploads pre-chunked documentation with stable DocPath headers to
separate Manual and API vector stores.

Usage:
    # Upload API docs only
    python scripts/upload_chunked_docs.py --api-only

    # Upload Manual docs only
    python scripts/upload_chunked_docs.py --manual-only

    # Upload both
    python scripts/upload_chunked_docs.py

    # Dry run (no actual uploads)
    python scripts/upload_chunked_docs.py --dry-run

Environment Variables:
    BLENDER_MANUAL_VECTOR_STORE_ID - Vector store ID for manual docs
    BLENDER_API_VECTOR_STORE_ID - Vector store ID for API docs
    OPENAI_API_KEY - OpenAI API key
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai package not installed. Run: pip install openai")
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

# Vector store IDs from environment
MANUAL_STORE_ID = os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID", "vs_6975104199c08191acb1495c86d581ce")
API_STORE_ID = os.getenv("BLENDER_API_VECTOR_STORE_ID", "vs_697512bf81c481919ae3b7a8ffb8223a")

# Paths
DOCS_BASE = DEFAULT_DOCS_BASE
MANUAL_PATH = DOCS_BASE / "blender_manual_html"
API_PATH = DOCS_BASE / "blender_python_reference_5_0"

# Upload settings
BATCH_SIZE = 50  # Chunks per batch before status update
RATE_LIMIT_DELAY = 0.05  # Seconds between uploads (to avoid rate limits)
MAX_RETRIES = 3


def upload_chunk(
    client: OpenAI,
    vector_store_id: str,
    chunk: DocChunk,
    dry_run: bool = False
) -> Optional[str]:
    """
    Upload a single chunk to the vector store.

    Returns file_id on success, None on failure.
    """
    if dry_run:
        return "dry_run_file_id"

    for attempt in range(MAX_RETRIES):
        try:
            # Create temporary file with chunk content
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.md',
                delete=False,
                encoding='utf-8'
            ) as tmp:
                tmp.write(chunk.to_upload_text())
                tmp_path = tmp.name

            try:
                # Upload file to OpenAI
                with open(tmp_path, 'rb') as f:
                    file_response = client.files.create(
                        file=f,
                        purpose="assistants"
                    )

                # Add to vector store
                client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_response.id
                )

                return file_response.id

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)  # Wait before retry
                continue
            print(f"  Error uploading {chunk.chunk_id[:50]}: {e}", file=sys.stderr)
            return None

    return None


def upload_chunks(
    client: OpenAI,
    vector_store_id: str,
    chunks: List[DocChunk],
    doc_type: str,
    dry_run: bool = False
) -> dict:
    """
    Upload a list of chunks to a vector store.

    Returns upload statistics.
    """
    results = {
        "total": len(chunks),
        "success": 0,
        "failed": 0,
        "file_ids": []
    }

    print(f"\nUploading {len(chunks)} {doc_type} chunks to {vector_store_id}...")

    start_time = time.time()

    for i, chunk in enumerate(chunks):
        file_id = upload_chunk(client, vector_store_id, chunk, dry_run)

        if file_id:
            results["success"] += 1
            results["file_ids"].append(file_id)
        else:
            results["failed"] += 1

        # Progress update
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(chunks):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i + 1}/{len(chunks)} ({results['success']} ok, {results['failed']} failed) "
                  f"[{rate:.1f}/s, ETA: {eta:.0f}s]")

        # Rate limiting
        if not dry_run:
            time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({results['success']} success, {results['failed']} failed)")

    return results


def process_and_upload(
    client: OpenAI,
    path: Path,
    base_path: Path,
    vector_store_id: str,
    doc_type: str,
    dry_run: bool = False
) -> dict:
    """
    Process HTML files and upload chunks to vector store.
    """
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        return {"total": 0, "success": 0, "failed": 0}

    # Collect HTML files
    html_files = [f for f in path.rglob("*.html") if not should_skip_file(f)]
    print(f"\nProcessing {len(html_files)} {doc_type} HTML files...")

    # Chunk all files
    all_chunks = []
    stats = ChunkingStats()

    for i, html_file in enumerate(html_files):
        try:
            if doc_type == "api":
                chunks = chunk_api_page(html_file, base_path)
            else:
                chunks = chunk_manual_page(html_file, base_path)

            for chunk in chunks:
                stats.add_chunk(chunk)
                all_chunks.append(chunk)

            stats.files_processed += 1

        except Exception as e:
            stats.files_errored += 1
            stats.errors.append(f"{html_file.name}: {str(e)[:50]}")

        # Progress
        if (i + 1) % 200 == 0:
            print(f"  Chunked {i + 1}/{len(html_files)} files ({len(all_chunks)} chunks)...")

    print(f"  Generated {len(all_chunks)} chunks from {stats.files_processed} files")

    # Upload chunks
    return upload_chunks(client, vector_store_id, all_chunks, doc_type, dry_run)


def verify_store(client: OpenAI, store_id: str, store_name: str) -> bool:
    """Verify vector store exists and is accessible."""
    try:
        vs = client.vector_stores.retrieve(store_id)
        print(f"  {store_name}: {vs.name} (status: {vs.status}, files: {vs.file_counts.total})")
        return True
    except Exception as e:
        print(f"  {store_name}: ERROR - {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload chunked Blender docs to OpenAI vector stores"
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Only upload manual documentation"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only upload API documentation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but don't actually upload"
    )
    parser.add_argument(
        "--docs-base",
        type=Path,
        default=DOCS_BASE,
        help="Base path for Blender documentation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Blender Documentation Vector Store Upload")
    print("=" * 60)
    print(f"Manual Store: {MANUAL_STORE_ID}")
    print(f"API Store: {API_STORE_ID}")
    print(f"Docs Base: {args.docs_base}")
    if args.dry_run:
        print("MODE: DRY RUN (no actual uploads)")
    print()

    # Initialize client
    client = OpenAI()

    # Verify stores
    print("Verifying vector stores...")
    stores_ok = True

    if not args.api_only:
        if not verify_store(client, MANUAL_STORE_ID, "Manual"):
            stores_ok = False

    if not args.manual_only:
        if not verify_store(client, API_STORE_ID, "API"):
            stores_ok = False

    if not stores_ok and not args.dry_run:
        print("\nError: One or more vector stores not accessible. Aborting.")
        sys.exit(1)

    # Process and upload
    results = {}

    if not args.api_only:
        manual_path = args.docs_base / "blender_manual_html"
        results["manual"] = process_and_upload(
            client, manual_path, args.docs_base,
            MANUAL_STORE_ID, "manual", args.dry_run
        )

    if not args.manual_only:
        api_path = args.docs_base / "blender_python_reference_5_0"
        results["api"] = process_and_upload(
            client, api_path, args.docs_base,
            API_STORE_ID, "api", args.dry_run
        )

    # Summary
    print()
    print("=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)

    total_success = 0
    total_failed = 0

    for doc_type, res in results.items():
        print(f"{doc_type.upper()}: {res['success']}/{res['total']} uploaded ({res['failed']} failed)")
        total_success += res['success']
        total_failed += res['failed']

    print(f"\nTOTAL: {total_success} uploaded, {total_failed} failed")

    # Final store status
    if not args.dry_run:
        print("\nFinal vector store status:")
        if not args.api_only:
            verify_store(client, MANUAL_STORE_ID, "Manual")
        if not args.manual_only:
            verify_store(client, API_STORE_ID, "API")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
