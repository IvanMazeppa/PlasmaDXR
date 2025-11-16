#!/usr/bin/env python3
import asyncio
import json
import subprocess
import mcp_server

async def test():
    print("🧪 Testing process termination...")
    
    # Check if any PlasmaDX processes running before test
    result = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq PlasmaDX-Clean.exe"], 
                          capture_output=True, text=True)
    before_count = result.stdout.count("PlasmaDX-Clean.exe")
    print(f"PlasmaDX processes before test: {before_count}")
    
    # Run test
    args = {
        "particle_count": 100,
        "timeout_seconds": 5,  # Short timeout for quick test
        "test_threshold": False,
        "capture_logs": True
    }
    
    print(f"\n🚀 Running 5-second test...")
    result = await mcp_server.diagnose_gpu_hang(args)
    data = json.loads(result[0].text)
    
    print(f"\n📊 Result: {data['tests'][0]['status']}")
    
    # Check if processes were killed
    await asyncio.sleep(2)  # Wait for kill to complete
    result = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq PlasmaDX-Clean.exe"], 
                          capture_output=True, text=True)
    after_count = result.stdout.count("PlasmaDX-Clean.exe")
    print(f"PlasmaDX processes after test: {after_count}")
    
    if after_count <= before_count:
        print("✅ Process terminated successfully!")
    else:
        print(f"⚠️  Warning: {after_count - before_count} process(es) still running")

asyncio.run(test())
