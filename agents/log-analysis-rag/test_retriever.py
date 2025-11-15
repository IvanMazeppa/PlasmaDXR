#!/usr/bin/env python3
"""
Test the hybrid retriever with actual PlasmaDX logs
This verifies BM25 + FAISS retrieval works before building the full RAG system
"""

import os
import sys
from pathlib import Path

# Load .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tools.hybrid_retriever import PlasmaDXHybridRetriever

print("="*80)
print("PlasmaDX Hybrid Retriever Test")
print("="*80)

# Initialize retriever with PlasmaDX log directories
print("\n📂 Initializing retriever with log directories:")
log_dirs = [
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/logs",
    "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/PIX/buffer_dumps",
]

for log_dir in log_dirs:
    print(f"   - {log_dir}")

try:
    retriever = PlasmaDXHybridRetriever(
        log_dirs=log_dirs,
        embedding_model="nvidia/nv-embedqa-e5-v5",
        top_k=10
    )

    # Test queries (common PlasmaDX issues from your CLAUDE.md)
    test_queries = [
        "RTXDI M5 patchwork pattern temporal instability",
        "shader compilation error HLSL",
        "particle flickering far distance",
        "shadow quality PCSS temporal",
        "DXR raytracing performance bottleneck",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*80}")

        docs = retriever.retrieve(query)

        print(f"\n📊 Top 5 Results:")
        for j, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get('source', 'unknown')
            line = doc.metadata.get('line', '?')
            content = doc.page_content[:150].replace('\n', ' ')

            print(f"\n{j}. {Path(source).name}:{line}")
            print(f"   {content}...")

        if i < len(test_queries):
            print(f"\n{'─'*80}")

    print(f"\n{'='*80}")
    print("✅ Hybrid retriever test complete!")
    print("="*80)
    print("\n💡 Next step: Build LangGraph workflow nodes (retrieve → rerank → grade → generate)")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
