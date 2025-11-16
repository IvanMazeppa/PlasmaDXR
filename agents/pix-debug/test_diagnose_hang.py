#!/usr/bin/env python3
"""
Quick test of diagnose_gpu_hang tool
"""
import asyncio
import sys
import json

# Import the MCP server module
import mcp_server

async def test_diagnose():
    """Test diagnose_gpu_hang with 10K particles"""
    args = {
        "particle_count": 10000,
        "timeout_seconds": 15,
        "test_threshold": False,
        "capture_logs": True
    }
    
    print("🚀 Testing diagnose_gpu_hang with 10K particles...")
    print(f"Args: {json.dumps(args, indent=2)}\n")
    
    result = await mcp_server.diagnose_gpu_hang(args)
    
    print("\n📊 Result:")
    print(result[0].text)

if __name__ == "__main__":
    asyncio.run(test_diagnose())
