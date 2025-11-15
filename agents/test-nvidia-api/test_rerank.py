#!/usr/bin/env python3
"""Test NVIDIA Rerank API"""

import os
import sys

# Load .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

print("="*60)
print("Testing NVIDIA Rerank API")
print("="*60 + "\n")

# Models to try
rerank_models = [
    "nvidia/nv-rerankqa-mistral-4b-v3",
    "nvidia/rerank-qa-mistral-4b",
    "nv-rerank-qa-mistral-4b:1",
]

query = "shader compilation error"
docs = [
    Document(page_content="ERROR: HLSL syntax error in particle_physics.hlsl line 42"),
    Document(page_content="FPS: 120.5 frames per second"),
    Document(page_content="DXC compilation failed with exit code 1"),
    Document(page_content="The weather is sunny today"),
]

for model_name in rerank_models:
    try:
        print(f"Testing: {model_name}")
        ranker = NVIDIARerank(model=model_name)

        reranked = ranker.compress_documents(documents=docs, query=query)

        print(f"✅ SUCCESS with {model_name}!")
        print(f"   Reranked {len(reranked)} documents")
        print(f"   Top result: {reranked[0].page_content[:60]}...")
        print()

        # Save working model
        with open('.working_rerank_model', 'w') as f:
            f.write(model_name)

        print(f"🎉 Found working rerank model: {model_name}")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Failed: {e}")
        print()
        continue

print("⚠️  Reranking not available, but embeddings still work!")
print("   You can still use the RAG system without reranking.")
