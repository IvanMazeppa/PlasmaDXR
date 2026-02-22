#!/usr/bin/env python3
"""
Upload rewritten Blender manual pages to an OpenAI vector store.

Takes the output of experiment_manual_rewrite.py (a directory of .md files)
and uploads them to a vector store for use by semantic_docs_tools.py.

Usage:
    # Upload all .md files from the rewritten output dir
    python scripts/upload_rewritten_manual.py --input ./rewritten_manual_test

    # Upload to an existing vector store
    python scripts/upload_rewritten_manual.py --input ./rewritten_manual_test --store-id vs_abc123

    # Dry run — show what would be uploaded
    python scripts/upload_rewritten_manual.py --input ./rewritten_manual_test --dry-run

    # Verify an existing store (no uploads)
    python scripts/upload_rewritten_manual.py --verify-only --store-id vs_abc123

    # Custom store name
    python scripts/upload_rewritten_manual.py --input ./rewritten_manual_test --store-name "blender-manual-fire-only"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Error: openai package required. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_STORE_NAME = "blender-manual-rewritten-physics"
RATE_LIMIT_DELAY = 0.1  # Seconds between uploads
MAX_RETRIES = 3
POLL_INTERVAL = 2.0  # Seconds between file status polls
POLL_TIMEOUT = 300.0  # Max seconds to wait for all files to process
TEST_QUERY = "fire simulation techniques mantaflow"


# ============================================================
# UPLOAD
# ============================================================

def collect_md_files(input_dir: Path) -> list[Path]:
    """Collect all .md files from input dir, excluding summary JSON."""
    files = sorted(input_dir.rglob("*.md"))
    # Filter out any hidden files or summary files
    files = [f for f in files if not f.name.startswith("_") and not f.name.startswith(".")]
    return files


def upload_file(client: OpenAI, vector_store_id: str, file_path: Path) -> str | None:
    """Upload a single .md file to the vector store. Returns file_id or None."""
    for attempt in range(MAX_RETRIES):
        try:
            with open(file_path, "rb") as f:
                file_response = client.files.create(file=f, purpose="assistants")

            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id,
            )
            return file_response.id

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            print(f"  Error uploading {file_path.name}: {e}", file=sys.stderr)
            return None

    return None


def poll_until_ready(client: OpenAI, vector_store_id: str) -> dict:
    """Poll vector store until all files are processed. Returns file_counts."""
    start = time.time()
    while time.time() - start < POLL_TIMEOUT:
        vs = client.vector_stores.retrieve(vector_store_id)
        counts = vs.file_counts
        in_progress = counts.in_progress
        if in_progress == 0:
            return {
                "completed": counts.completed,
                "failed": counts.failed,
                "cancelled": counts.cancelled,
                "total": counts.total,
            }
        elapsed = time.time() - start
        print(f"  Waiting for processing... {counts.completed}/{counts.total} ready "
              f"({in_progress} in progress, {elapsed:.0f}s elapsed)", flush=True)
        time.sleep(POLL_INTERVAL)

    print(f"  Warning: timed out after {POLL_TIMEOUT}s waiting for file processing", file=sys.stderr)
    vs = client.vector_stores.retrieve(vector_store_id)
    counts = vs.file_counts
    return {
        "completed": counts.completed,
        "failed": counts.failed,
        "cancelled": counts.cancelled,
        "total": counts.total,
    }


def verify_store(client: OpenAI, store_id: str) -> bool:
    """Verify store is accessible and run a test query."""
    print(f"\nVerifying store {store_id}...")

    # Check store exists
    try:
        vs = client.vector_stores.retrieve(store_id)
        print(f"  Name: {vs.name}")
        print(f"  Status: {vs.status}")
        print(f"  Files: {vs.file_counts.completed} completed, "
              f"{vs.file_counts.failed} failed, "
              f"{vs.file_counts.total} total")
    except Exception as e:
        print(f"  Error retrieving store: {e}", file=sys.stderr)
        return False

    if vs.file_counts.completed == 0:
        print("  Warning: no completed files in store — skipping test query")
        return True

    # Run test query
    print(f"  Test query: \"{TEST_QUERY}\"")
    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query=TEST_QUERY,
            max_num_results=3,
        )
        results = response.data
        if not results:
            print("  Warning: test query returned 0 results")
            return True

        print(f"  Results: {len(results)} hits")
        for i, result in enumerate(results):
            score = result.score or 0
            filename = result.filename or "unknown"
            preview = ""
            if result.content:
                preview = result.content[0].text[:120].replace("\n", " ")
            print(f"    [{i+1}] {filename} (score: {score:.3f})")
            print(f"        {preview}...")
        return True

    except Exception as e:
        print(f"  Test query error: {e}", file=sys.stderr)
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload rewritten Blender manual pages to an OpenAI vector store"
    )
    parser.add_argument("--input", type=str, help="Input directory containing rewritten .md files")
    parser.add_argument("--store-id", type=str, help="Existing vector store ID to upload to (creates new if omitted)")
    parser.add_argument("--store-name", type=str, default=DEFAULT_STORE_NAME, help=f"Name for new store (default: {DEFAULT_STORE_NAME})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without doing it")
    parser.add_argument("--verify-only", action="store_true", help="Only verify an existing store (requires --store-id)")
    args = parser.parse_args()

    # Validate args
    if args.verify_only:
        if not args.store_id:
            print("Error: --verify-only requires --store-id", file=sys.stderr)
            sys.exit(1)
        client = OpenAI()
        ok = verify_store(client, args.store_id)
        sys.exit(0 if ok else 1)

    if not args.input:
        print("Error: --input is required (unless using --verify-only)", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect files
    md_files = collect_md_files(input_dir)
    if not md_files:
        print(f"Error: no .md files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Upload Rewritten Manual to Vector Store")
    print("=" * 60)
    print(f"Input: {input_dir} ({len(md_files)} files)")
    total_bytes = sum(f.stat().st_size for f in md_files)
    print(f"Total size: {total_bytes / 1024:.1f} KB")
    if args.store_id:
        print(f"Target store: {args.store_id}")
    else:
        print(f"New store name: {args.store_name}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print()

    # Dry run — list files
    if args.dry_run:
        for f in md_files:
            rel = f.relative_to(input_dir)
            size = f.stat().st_size
            print(f"  {rel} ({size / 1024:.1f} KB)")
        print(f"\nTotal: {len(md_files)} files, {total_bytes / 1024:.1f} KB")
        print("No uploads performed (dry run).")
        return

    # Initialize client
    client = OpenAI()

    # Create or reuse vector store
    if args.store_id:
        store_id = args.store_id
        try:
            vs = client.vector_stores.retrieve(store_id)
            print(f"Using existing store: {vs.name} ({store_id})")
            print(f"  Current files: {vs.file_counts.total}")
        except Exception as e:
            print(f"Error: cannot access store {store_id}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Creating vector store: {args.store_name}")
        try:
            vs = client.vector_stores.create(name=args.store_name)
            store_id = vs.id
            print(f"  Created: {store_id}")
        except Exception as e:
            print(f"Error creating vector store: {e}", file=sys.stderr)
            sys.exit(1)

    # Upload files
    print(f"\nUploading {len(md_files)} files...")
    print("-" * 60)

    success_count = 0
    failed_count = 0
    file_ids = []
    start_time = time.time()

    for i, file_path in enumerate(md_files):
        rel = file_path.relative_to(input_dir)
        size_kb = file_path.stat().st_size / 1024
        print(f"[{i+1}/{len(md_files)}] {rel} ({size_kb:.1f} KB)...", end=" ", flush=True)

        file_id = upload_file(client, store_id, file_path)
        if file_id:
            success_count += 1
            file_ids.append(file_id)
            print(f"OK ({file_id})")
        else:
            failed_count += 1
            print("FAILED")

        time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - start_time
    print(f"\nUpload complete in {elapsed:.1f}s")
    print(f"  Success: {success_count}/{len(md_files)}")
    if failed_count:
        print(f"  Failed: {failed_count}")

    # Poll until files are processed
    if success_count > 0:
        print("\nWaiting for vector store to process files...")
        counts = poll_until_ready(client, store_id)
        print(f"  Final: {counts['completed']} completed, {counts['failed']} failed, {counts['total']} total")

    # Verify with test query
    verify_store(client, store_id)

    # Summary
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Vector Store ID: {store_id}")
    print(f"Files uploaded: {success_count}")
    print()
    print("To use this store in semantic_docs_tools.py, set:")
    print(f"  export BLENDER_REWRITTEN_MANUAL_STORE_ID={store_id}")


if __name__ == "__main__":
    main()
