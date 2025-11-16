#!/usr/bin/env python3
import asyncio
import json
import mcp_server

async def test():
    args = {
        "particle_count": 100,
        "timeout_seconds": 10,
        "test_threshold": False,
        "capture_logs": True
    }
    
    print("🚀 Testing F7 automation with correct window title...")
    result = await mcp_server.diagnose_gpu_hang(args)
    
    data = json.loads(result[0].text)
    
    print("\n📊 Result:")
    print(f"  Status: {data['tests'][0]['status']}")
    print(f"  Runtime: {data['tests'][0]['runtime_seconds']:.1f}s")
    
    # Check for F7 success
    logs = data['tests'][0].get('logs_captured', [])
    f7_logs = [log for log in logs if 'Volumetric ReSTIR' in log and ('ON' in log or 'OFF' in log)]
    
    if f7_logs:
        print(f"  ✅ F7 WORKED! Found:")
        for log in f7_logs[:3]:
            print(f"     {log}")
    else:
        print(f"  ❌ F7 did not work")
        print(f"  Last 3 logs:")
        for log in logs[-3:]:
            print(f"     {log}")

asyncio.run(test())
