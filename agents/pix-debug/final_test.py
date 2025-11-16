#!/usr/bin/env python3
import asyncio
import json
import mcp_server

async def test():
    print("🧪 Final Test: diagnose_gpu_hang with --restir flag")
    print("=" * 60)
    
    args = {
        "particle_count": 100,
        "timeout_seconds": 10,
        "test_threshold": False,
        "capture_logs": True
    }
    
    print(f"\n📋 Test Configuration:")
    print(f"  Particle count: {args['particle_count']}")
    print(f"  Timeout: {args['timeout_seconds']}s")
    print(f"  Threshold testing: {args['test_threshold']}")
    print(f"  Capture logs: {args['capture_logs']}")
    
    print(f"\n🚀 Launching PlasmaDX...")
    result = await mcp_server.diagnose_gpu_hang(args)
    
    data = json.loads(result[0].text)
    test = data['tests'][0]
    
    print(f"\n✅ Test Complete!")
    print(f"=" * 60)
    print(f"\n📊 Results:")
    print(f"  Status: {test['status']}")
    print(f"  Runtime: {test['runtime_seconds']:.1f}s")
    print(f"  Hang detected: {test['hang_detected']}")
    print(f"  Crash detected: {test['crash_detected']}")
    
    # Check for --restir flag success
    logs = test.get('logs_captured', [])
    restir_logs = [log for log in logs if 'Volumetric ReSTIR' in log and 'autonomous testing mode' in log]
    
    print(f"\n🔍 ReSTIR Activation Check:")
    if restir_logs:
        print(f"  ✅ SUCCESS - --restir flag worked!")
        print(f"  Found: {restir_logs[0]}")
    else:
        print(f"  ❌ FAILED - --restir flag not detected in logs")
    
    # Show initialization logs
    print(f"\n📝 Initialization Logs (first 5):")
    for log in logs[:5]:
        print(f"  {log}")
    
    # Show final logs
    print(f"\n📝 Final Logs (last 5):")
    for log in logs[-5:]:
        print(f"  {log}")
    
    # Check recommendations
    recs = data.get('recommendations', [])
    if recs:
        print(f"\n💡 Recommendations ({len(recs)}):")
        for i, rec in enumerate(recs[:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n{'=' * 60}")
    if restir_logs:
        print("✅ AUTONOMOUS DEBUGGING AGENT: OPERATIONAL")
    else:
        print("❌ ISSUE DETECTED - See logs above")

asyncio.run(test())
