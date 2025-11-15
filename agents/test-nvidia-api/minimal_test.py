#!/usr/bin/env python3
"""Minimal NVIDIA API test - just check if API key works"""

import os
import sys

# Check API key
api_key = os.getenv("NVIDIA_API_KEY")
print(f"NVIDIA_API_KEY set: {api_key is not None}")
if api_key:
    print(f"Key starts with: {api_key[:10]}...")
    print(f"Key length: {len(api_key)}")
else:
    print("ERROR: No API key found!")
    sys.exit(1)

# Try basic import
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    print("✅ langchain-nvidia-ai-endpoints imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run: pip install langchain-nvidia-ai-endpoints")
    sys.exit(1)

print("\n🎉 Basic setup looks good! API key is present and package is installed.")
