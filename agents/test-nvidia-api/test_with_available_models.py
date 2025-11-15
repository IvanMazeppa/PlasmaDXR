#!/usr/bin/env python3
"""Test NVIDIA API with different available models"""

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
    print("✅ Loaded .env file")

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("❌ No API key found!")
    sys.exit(1)

print(f"✅ API Key: {api_key[:15]}... (length: {len(api_key)})")

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

print("\n" + "="*60)
print("Trying different embedding models...")
print("="*60 + "\n")

# List of models to try (from NVIDIA documentation)
models_to_try = [
    "nvidia/nv-embedqa-e5-v5",  # Latest as of 2024
    "nvidia/nv-embed-v1",
    "nvolveqa_40k",
    "NV-Embed-QA",
]

for model_name in models_to_try:
    try:
        print(f"Testing: {model_name}")
        embedder = NVIDIAEmbeddings(model=model_name)

        test_text = "DirectX shader error"
        embeddings = embedder.embed_query(test_text)

        print(f"✅ SUCCESS with {model_name}!")
        print(f"   Vector dimension: {len(embeddings)}")
        print(f"   First 3 values: {embeddings[:3]}")
        print()

        # Save the working model name
        with open('.working_model', 'w') as f:
            f.write(model_name)

        print(f"🎉 Found working model: {model_name}")
        print(f"   Saved to .working_model file")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Failed: {e}")
        print()
        continue

print("⚠️  None of the models worked. You may need to activate your NGC account.")
print("   Visit: https://org.ngc.nvidia.com/activate")
sys.exit(1)
