"""
Upload Blender Documentation to OpenAI Vector Store.

This script uploads Blender 5.0 documentation to an OpenAI vector store
for semantic search capabilities in the VFX orchestrator.

Usage:
    # Set your vector store ID and API key
    export OPENAI_API_KEY="your-api-key"
    export BLENDER_DOCS_VECTOR_STORE_ID="vs_696acc41b74c8191a8d6f614c0223923"

    # Run the upload
    python scripts/upload_blender_docs_to_vectorstore.py

Sources indexed:
1. Blender 5.0 Manual HTML (~2,196 files)
2. Blender Python API Reference (~2,063 files)
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Configuration
VECTOR_STORE_ID = os.getenv("BLENDER_DOCS_VECTOR_STORE_ID", "vs_696acc41b74c8191a8d6f614c0223923")
DOCS_BASE_PATH = Path("/home/maz3ppa/blender_documentation")
MANUAL_PATH = DOCS_BASE_PATH / "blender_manual_html"
API_REF_PATH = DOCS_BASE_PATH / "blender_python_reference_5_0"

# Upload settings
BATCH_SIZE = 100  # Files per batch
MAX_FILE_SIZE = 512 * 1024  # 512KB max per file for vector stores
MAX_WORKERS = 4  # Parallel upload threads

# Skip patterns
SKIP_PATTERNS = [
    "Zone.Identifier",  # Windows zone files
    "_static",
    "_sources",
    "genindex",
    "py-modindex",
    "search.html",
    "404.html",
]


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, preserving structure."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        self.in_style = False
        self.in_nav = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
        elif tag == "style":
            self.in_style = True
        elif tag == "nav":
            self.in_nav = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.text_parts.append("\n\n## ")
        elif tag in ("p", "div", "li"):
            self.text_parts.append("\n")
        elif tag == "br":
            self.text_parts.append("\n")
        elif tag == "code":
            self.text_parts.append("`")
        elif tag == "pre":
            self.text_parts.append("\n```\n")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        elif tag == "style":
            self.in_style = False
        elif tag == "nav":
            self.in_nav = False
        elif tag == "code":
            self.text_parts.append("`")
        elif tag == "pre":
            self.text_parts.append("\n```\n")

    def handle_data(self, data):
        if not self.in_script and not self.in_style and not self.in_nav:
            text = data.strip()
            if text:
                self.text_parts.append(text + " ")

    def get_text(self) -> str:
        return "".join(self.text_parts).strip()


def extract_text_from_html(html_path: Path) -> Optional[str]:
    """Extract clean text from HTML file."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        text = "\n".join(lines)

        # Add metadata header
        relative_path = html_path.relative_to(DOCS_BASE_PATH)
        category = relative_path.parts[1] if len(relative_path.parts) > 1 else "general"

        header = f"""# {html_path.stem.replace('_', ' ').replace('-', ' ').title()}
Source: Blender 5.0 Documentation
Category: {category}
Path: {relative_path}

---

"""
        return header + text

    except Exception as e:
        print(f"  Error extracting {html_path}: {e}")
        return None


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped."""
    path_str = str(file_path)
    for pattern in SKIP_PATTERNS:
        if pattern in path_str:
            return True
    return False


def get_all_html_files(base_path: Path) -> List[Path]:
    """Get all HTML files from a directory."""
    if not base_path.exists():
        print(f"Warning: Path does not exist: {base_path}")
        return []

    files = []
    for html_file in base_path.rglob("*.html"):
        if not should_skip_file(html_file):
            files.append(html_file)
    return files


def upload_file_to_vector_store(
    client: OpenAI,
    vector_store_id: str,
    text_content: str,
    original_path: Path
) -> Optional[str]:
    """Upload a single file to the vector store."""
    try:
        # Create a temporary file with the text content
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(text_content)
            tmp_path = tmp.name

        try:
            # Upload file to OpenAI
            with open(tmp_path, "rb") as f:
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
        print(f"  Error uploading {original_path.name}: {e}")
        return None


def upload_batch(
    client: OpenAI,
    vector_store_id: str,
    files: List[Path],
    batch_num: int,
    total_batches: int
) -> Dict[str, int]:
    """Upload a batch of files."""
    results = {"success": 0, "failed": 0, "skipped": 0}

    print(f"\nBatch {batch_num}/{total_batches} ({len(files)} files)")

    for i, file_path in enumerate(files):
        # Extract text
        text = extract_text_from_html(file_path)

        if not text:
            results["skipped"] += 1
            continue

        # Check size
        if len(text.encode("utf-8")) > MAX_FILE_SIZE:
            # Truncate if too large
            text = text[:MAX_FILE_SIZE - 1000] + "\n\n[Content truncated due to size limit]"

        # Upload
        file_id = upload_file_to_vector_store(
            client, vector_store_id, text, file_path
        )

        if file_id:
            results["success"] += 1
        else:
            results["failed"] += 1

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")

        # Rate limiting
        time.sleep(0.1)  # Small delay to avoid rate limits

    return results


def main():
    """Main upload function."""
    print("=" * 60)
    print("Blender Documentation Vector Store Upload")
    print("=" * 60)
    print(f"Vector Store ID: {VECTOR_STORE_ID}")
    print(f"Documentation Path: {DOCS_BASE_PATH}")
    print()

    # Initialize client
    client = OpenAI()

    # Verify vector store exists
    try:
        vs = client.vector_stores.retrieve(VECTOR_STORE_ID)
        print(f"Vector Store: {vs.name} (status: {vs.status})")
        print(f"Current file count: {vs.file_counts.total}")
    except Exception as e:
        print(f"Error: Could not access vector store: {e}")
        print("Make sure BLENDER_DOCS_VECTOR_STORE_ID is set correctly")
        sys.exit(1)

    # Collect all files
    print("\nCollecting HTML files...")
    manual_files = get_all_html_files(MANUAL_PATH)
    api_files = get_all_html_files(API_REF_PATH)

    all_files = manual_files + api_files
    print(f"  Manual files: {len(manual_files)}")
    print(f"  API Reference files: {len(api_files)}")
    print(f"  Total files: {len(all_files)}")

    if not all_files:
        print("No files found to upload!")
        sys.exit(1)

    # Confirm
    print(f"\nThis will upload {len(all_files)} files to the vector store.")
    response = input("Continue? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # Upload in batches
    total_results = {"success": 0, "failed": 0, "skipped": 0}
    batches = [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]

    print(f"\nUploading in {len(batches)} batches of up to {BATCH_SIZE} files each...")
    start_time = time.time()

    for batch_num, batch_files in enumerate(batches, 1):
        batch_results = upload_batch(
            client, VECTOR_STORE_ID, batch_files, batch_num, len(batches)
        )

        for key in total_results:
            total_results[key] += batch_results[key]

        print(f"  Batch complete: {batch_results}")

        # Longer pause between batches
        if batch_num < len(batches):
            time.sleep(1)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Successful: {total_results['success']}")
    print(f"Failed: {total_results['failed']}")
    print(f"Skipped: {total_results['skipped']}")
    print(f"Time elapsed: {elapsed:.1f} seconds")

    # Verify final state
    vs = client.vector_stores.retrieve(VECTOR_STORE_ID)
    print(f"\nVector Store final file count: {vs.file_counts.total}")
    print(f"Vector Store status: {vs.status}")

    print("\nDone! You can now use semantic search with the orchestrator.")
    print("Add to your .env:")
    print(f"  BLENDER_DOCS_VECTOR_STORE_ID={VECTOR_STORE_ID}")


if __name__ == "__main__":
    main()
