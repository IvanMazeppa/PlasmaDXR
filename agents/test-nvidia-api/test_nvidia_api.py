#!/usr/bin/env python3
"""
Test NVIDIA API Key and available models
Run this to verify your NVIDIA API setup is working
"""

import os
import sys

def check_api_key():
    """Verify NVIDIA_API_KEY is set"""
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        print("❌ ERROR: NVIDIA_API_KEY environment variable not set!")
        print("\nPlease set it in Windows:")
        print("  setx NVIDIA_API_KEY \"nvapi-your-key-here\"")
        return False

    if not api_key.startswith("nvapi-"):
        print("⚠️  WARNING: API key doesn't start with 'nvapi-'")
        print(f"   Key starts with: {api_key[:10]}...")
        return False

    print(f"✅ NVIDIA_API_KEY is set (starts with: {api_key[:10]}...)")
    return True

def test_embeddings():
    """Test NVIDIA embeddings model"""
    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        print("\n🧪 Testing NVIDIAEmbeddings...")
        embedder = NVIDIAEmbeddings(model="NV-Embed-QA")

        # Test with a simple query
        test_text = "DirectX 12 raytracing error"
        embeddings = embedder.embed_query(test_text)

        print(f"✅ Embeddings working! Vector dimension: {len(embeddings)}")
        return True

    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return False

def test_reranking():
    """Test NVIDIA reranking model"""
    try:
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        from langchain_core.documents import Document

        print("\n🧪 Testing NVIDIARerank...")
        ranker = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3")

        # Test with sample documents
        query = "Why is my shader failing?"
        docs = [
            Document(page_content="ERROR: Shader compilation failed at line 42"),
            Document(page_content="FPS: 120.5"),
            Document(page_content="HLSL syntax error in particle_physics.hlsl"),
        ]

        reranked = ranker.compress_documents(documents=docs, query=query)

        print(f"✅ Reranking working! Reranked {len(reranked)} documents")
        print(f"   Top result: {reranked[0].page_content[:50]}...")
        return True

    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        return False

def list_available_models():
    """List available NVIDIA models"""
    try:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        print("\n📋 Available NVIDIA models:")
        print("\nNote: Call get_available_models() to see full list")
        print("Common models:")
        print("  Embeddings: NV-Embed-QA, nvidia/llama-3.2-nv-embedqa-1b-v2")
        print("  Reranking: nvidia/nv-rerankqa-mistral-4b-v3")

        return True

    except Exception as e:
        print(f"⚠️  Could not list models: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NVIDIA API Test Suite for PlasmaDX")
    print("=" * 60)

    # Step 1: Check API key
    if not check_api_key():
        sys.exit(1)

    # Step 2: Test embeddings
    embeddings_ok = test_embeddings()

    # Step 3: Test reranking
    reranking_ok = test_reranking()

    # Step 4: List models
    list_available_models()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"  API Key:    {'✅ OK' if check_api_key() else '❌ FAILED'}")
    print(f"  Embeddings: {'✅ OK' if embeddings_ok else '❌ FAILED'}")
    print(f"  Reranking:  {'✅ OK' if reranking_ok else '❌ FAILED'}")

    if embeddings_ok and reranking_ok:
        print("\n🎉 All tests passed! Your NVIDIA API is ready for PlasmaDX RAG system.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        sys.exit(1)
