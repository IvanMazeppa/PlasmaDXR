#!/usr/bin/env python3
"""Quick NVIDIA API test with .env file support"""

import os
import sys

# Load .env file if it exists
def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded .env file")

load_env()

# Check API key
api_key = os.getenv("NVIDIA_API_KEY")
print(f"\n{'='*60}")
print("NVIDIA API Quick Test")
print(f"{'='*60}\n")

if not api_key:
    print("❌ ERROR: NVIDIA_API_KEY not found!")
    sys.exit(1)

print(f"✅ API Key found")
print(f"   Starts with: {api_key[:15]}...")
print(f"   Length: {len(api_key)} chars")

# Test imports
print(f"\n{'='*60}")
print("Testing Package Imports")
print(f"{'='*60}\n")

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
    print("✅ langchain-nvidia-ai-endpoints imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nInstalling required package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "langchain-nvidia-ai-endpoints"])
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
    print("✅ Package installed and imported")

# Test embedding API call
print(f"\n{'='*60}")
print("Testing NVIDIA Embeddings API")
print(f"{'='*60}\n")

try:
    embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
    test_text = "DirectX 12 raytracing shader error"

    print(f"Embedding text: '{test_text}'")
    embeddings = embedder.embed_query(test_text)

    print(f"✅ Embeddings API works!")
    print(f"   Vector dimension: {len(embeddings)}")
    print(f"   First 3 values: {embeddings[:3]}")

except Exception as e:
    print(f"❌ Embeddings API failed: {e}")
    print(f"\nError type: {type(e).__name__}")
    sys.exit(1)

# Test reranking API call
print(f"\n{'='*60}")
print("Testing NVIDIA Rerank API")
print(f"{'='*60}\n")

try:
    from langchain_core.documents import Document

    ranker = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3")

    query = "shader compilation error"
    docs = [
        Document(page_content="ERROR: HLSL syntax error in particle_physics.hlsl line 42"),
        Document(page_content="FPS: 120.5 frames per second"),
        Document(page_content="DXC compilation failed with exit code 1"),
    ]

    print(f"Query: '{query}'")
    print(f"Reranking {len(docs)} documents...")

    reranked = ranker.compress_documents(documents=docs, query=query)

    print(f"✅ Reranking API works!")
    print(f"   Top result: {reranked[0].page_content[:60]}...")

except Exception as e:
    print(f"❌ Reranking API failed: {e}")
    print(f"\nError type: {type(e).__name__}")
    # Don't exit - reranking is optional
    print("\nNote: Reranking failed but embeddings work - you can still use the RAG system")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}\n")
print("🎉 NVIDIA API is configured correctly!")
print("✅ You're ready to build the log-analysis-rag agent")
print(f"\n{'='*60}\n")
