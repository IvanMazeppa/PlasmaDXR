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
    
    print("🚀 Testing --restir flag...")
    result = await mcp_server.diagnose_gpu_hang(args)
    
    data = json.loads(result[0].text)
    
    print("\n📊 Result:")
    print(f"  Status: {data['tests'][0]['status']}")
    print(f"  Runtime: {data['tests'][0]['runtime_seconds']:.1f}s")
    
    # Check if --restir worked
    logs = data['tests'][0].get('logs_captured', [])
    restir_enabled = any('Volumetric ReSTIR' in log and 'autonomous testing mode' in log for log in logs)
    
    if restir_enabled:
        print(f"  ✅ --restir FLAG WORKED!")
        restir_logs = [log for log in logs if 'ReSTIR' in log or 'Volumetric' in log][:5]
        for log in restir_logs:
            print(f"     {log}")
    else:
        print(f"  ❌ --restir flag did not work")
        print(f"  Last 5 logs:")
        for log in logs[-5:]:
            print(f"     {log}")

asyncio.run(test())
