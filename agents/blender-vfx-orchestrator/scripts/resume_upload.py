#!/usr/bin/env python3
"""
Resume upload of chunked Blender docs to vector store.

Reads list of already-uploaded files and skips them.

Usage:
    python scripts/resume_upload.py --api-only
    python scripts/resume_upload.py --manual-only
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent to path for imports
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

# Batch settings
BATCH_SIZE = 100
NUM_WORKERS = 8


def get_chunk_filename(chunk: DocChunk) -> str:
    """Get the filename that would be used for this chunk."""
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in chunk.chunk_id)[:80]
    return f"{safe_name}.md"


def load_already_uploaded(filepath: str = "/tmp/already_uploaded.txt") -> set:
    """Load set of already uploaded filenames."""
    try:
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def create_temp_file(chunk: DocChunk, temp_dir: Path) -> Path:
    """Create a temp file for a chunk."""
    filename = get_chunk_filename(chunk)
    file_path = temp_dir / filename
    file_path.write_text(chunk.to_upload_text(), encoding='utf-8')
    return file_path


def upload_batch(client: OpenAI, store_id: str, file_paths: list, batch_num: int, total_batches: int) -> dict:
    """Upload a batch of files."""
    print(f"  Batch {batch_num}/{total_batches}: {len(file_paths)} chunks...", flush=True)

    # Upload files to OpenAI
    file_ids = []
    for fp in file_paths:
        try:
            with open(fp, 'rb') as f:
                uploaded = client.files.create(file=f, purpose='assistants')
                file_ids.append(uploaded.id)
        except Exception as e:
            print(f"    Warning: Failed to upload {fp.name}: {e}", flush=True)

    if not file_ids:
        return {"ok": 0, "failed": len(file_paths)}

    # Add to vector store
    try:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=store_id,
            file_ids=file_ids
        )
        ok = batch.file_counts.completed
        failed = batch.file_counts.failed
        print(f"    Batch complete: {ok} ok, {failed} failed", flush=True)
        return {"ok": ok, "failed": failed}
    except Exception as e:
        print(f"    Batch error: {e}", flush=True)
        return {"ok": 0, "failed": len(file_ids)}


def main():
    parser = argparse.ArgumentParser(description="Resume upload of chunked docs")
    parser.add_argument("--api-only", action="store_true", help="Only upload API docs")
    parser.add_argument("--manual-only", action="store_true", help="Only upload Manual docs")
    args = parser.parse_args()

    print("=" * 60)
    print("Resume Blender Documentation Upload")
    print("=" * 60)

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

        # Get HTML files
        html_files = sorted(doc_dir.glob("**/*.html"))
        print(f"\nProcessing {len(html_files)} {doc_type} HTML files...")

        # Generate chunks - use DOCS_BASE as base_path to match original batch script
        all_chunks = []
        chunk_func = chunk_api_page if doc_type == "api" else chunk_manual_page
        for i, html_file in enumerate(html_files):
            if (i + 1) % 500 == 0:
                print(f"  Chunked {i+1}/{len(html_files)} files ({len(all_chunks)} chunks)...", flush=True)
            try:
                chunks = chunk_func(html_file, DOCS_BASE)  # Use DOCS_BASE for consistent filenames
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"    Warning: Failed to chunk {html_file.name}: {e}", flush=True)

        print(f"  Generated {len(all_chunks)} chunks")

        # Filter out already uploaded
        chunks_to_upload = []
        for chunk in all_chunks:
            filename = get_chunk_filename(chunk)
            if filename not in already_uploaded:
                chunks_to_upload.append(chunk)

        print(f"  Skipping {len(all_chunks) - len(chunks_to_upload)} already uploaded")
        print(f"  Uploading {len(chunks_to_upload)} new chunks to {store_id}")

        if not chunks_to_upload:
            print("  Nothing to upload!")
            continue

        # Upload in batches
        total_ok = 0
        total_failed = 0
        total_batches = (len(chunks_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for batch_idx in range(total_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(chunks_to_upload))
                batch_chunks = chunks_to_upload[start:end]

                # Create temp files in parallel
                file_paths = []
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {executor.submit(create_temp_file, c, temp_path): c for c in batch_chunks}
                    for future in as_completed(futures):
                        file_paths.append(future.result())

                # Upload batch
                result = upload_batch(client, store_id, file_paths, batch_idx + 1, total_batches)
                total_ok += result["ok"]
                total_failed += result["failed"]

                # Progress
                rate = total_ok / ((batch_idx + 1) * 60) if batch_idx > 0 else 0.5
                remaining = len(chunks_to_upload) - total_ok - total_failed
                eta = remaining / rate if rate > 0 else 0
                print(f"    Total: {total_ok}/{len(chunks_to_upload)} ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)

                # Clean up temp files
                for fp in file_paths:
                    fp.unlink(missing_ok=True)

        print(f"\n{doc_type.upper()} Upload complete: {total_ok} ok, {total_failed} failed")

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
