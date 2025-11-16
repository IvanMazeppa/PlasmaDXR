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
    
    print("🚀 Testing diagnose_gpu_hang at 100 particles...")
    result = await mcp_server.diagnose_gpu_hang(args)
    
    data = json.loads(result[0].text)
    
    print("\n📊 Test Result:")
    print(f"  Status: {data['tests'][0]['status']}")
    print(f"  Runtime: {data['tests'][0]['runtime_seconds']:.1f}s")
    print(f"  Hang detected: {data['tests'][0]['hang_detected']}")
    
    # Check if F7 worked
    logs = data['tests'][0].get('logs_captured', [])
    f7_worked = any('Volumetric ReSTIR: ON' in log for log in logs)
    print(f"  F7 keypress worked: {'✅ YES' if f7_worked else '❌ NO'}")
    
    if logs:
        print(f"\n📝 Log sample (last 5 lines):")
        for log in logs[-5:]:
            print(f"    {log}")

asyncio.run(test())
