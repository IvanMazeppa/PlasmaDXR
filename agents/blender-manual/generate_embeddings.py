#!/usr/bin/env python3
"""
Generate embeddings for semantic search.
Run this OFFLINE before using semantic search feature.

Usage:
    ./venv/bin/python generate_embeddings.py
"""

import json
import sys
from pathlib import Path

# Check for required packages
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing required packages. Please run:")
    print("  ./venv/bin/pip install sentence-transformers numpy")
    sys.exit(1)

# Configuration
SERVER_DIR = Path(__file__).parent
CACHE_FILE = SERVER_DIR / "manual_index.json"
EMBEDDINGS_FILE = SERVER_DIR / "embeddings.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim embeddings
BATCH_SIZE = 32

def main():
    print("=" * 60)
    print("Blender Manual - Semantic Search Embedding Generator")
    print("=" * 60)

    # Load cache
    if not CACHE_FILE.exists():
        print(f"ERROR: Cache file not found: {CACHE_FILE}")
        print("Run the server once to build the index first.")
        sys.exit(1)

    print(f"Loading index from {CACHE_FILE}...")
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    search_index = cache_data.get("index", [])
    print(f"Loaded {len(search_index)} pages from cache.")

    # Load model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    print("(This may take a minute on first run to download the model)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully!")

    # Create texts for embedding
    print(f"\nGenerating embeddings for {len(search_index)} pages...")
    print("(This may take 2-5 minutes)")

    texts = []
    for item in search_index:
        # Combine title with first 500 chars of content for embedding
        embed_text = f"{item['title']}. {item['content'][:500]}"
        texts.append(embed_text)

    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"\nSaved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")
    print(f"File size: {EMBEDDINGS_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 60)
    print("SUCCESS! Semantic search is now available.")
    print("Restart the MCP server to enable semantic search.")
    print("=" * 60)

if __name__ == "__main__":
    main()
