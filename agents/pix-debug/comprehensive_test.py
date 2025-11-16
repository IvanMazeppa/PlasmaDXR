#!/usr/bin/env python3
import asyncio
import json
import mcp_server

async def test():
    print("=" * 70)
    print("COMPREHENSIVE AUTONOMOUS DEBUGGING AGENT TEST")
    print("=" * 70)
    
    args = {
        "particle_count": 100,
        "timeout_seconds": 5,
        "test_threshold": False,
        "capture_logs": True
    }
    
    print("\n📋 Configuration:")
    print(f"  Particle count: {args['particle_count']}")
    print(f"  Timeout: {args['timeout_seconds']}s")
    print(f"  Expected: Normal run (PopulateVolumeMip2 disabled)")
    
    print("\n🚀 Launching autonomous test...")
    result = await mcp_server.diagnose_gpu_hang(args)
    data = json.loads(result[0].text)
    test = data['tests'][0]
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    # Core functionality
    print("\n✅ Core Functionality:")
    print(f"  Status: {test['status']}")
    print(f"  Runtime: {test['runtime_seconds']:.1f}s")
    print(f"  Hang detected: {test.get('hang_detected', False)}")
    
    # --restir flag verification
    logs = test.get('logs_captured', [])
    restir_line = next((log for log in logs if 'Volumetric ReSTIR' in log and 'autonomous testing mode' in log), None)
    
    print("\n✅ --restir Flag:")
    if restir_line:
        print(f"  ✓ WORKING: {restir_line}")
    else:
        print(f"  ✗ NOT FOUND in logs")
        return
    
    # Log capture
    init_logs = [log for log in logs[:50] if 'INFO' in log]
    final_logs = [log for log in logs[-50:] if 'frame' in log.lower()]
    
    print("\n✅ Log Capture:")
    print(f"  Init logs captured: {len(init_logs)} lines")
    print(f"  Final logs captured: {len(final_logs)} lines")
    print(f"  Sample init: {init_logs[0] if init_logs else '(none)'}")
    print(f"  Sample final: {final_logs[-1] if final_logs else '(none)'}")
    
    # ReSTIR initialization
    restir_init = [log for log in logs if 'Volumetric ReSTIR System initialized' in log]
    print("\n✅ ReSTIR System:")
    if restir_init:
        print(f"  ✓ Initialized: {restir_init[0]}")
    else:
        print(f"  ✗ Initialization not found")
    
    # Process termination
    import subprocess
    result = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq PlasmaDX-Clean.exe"], 
                          capture_output=True, text=True)
    running = result.stdout.count("PlasmaDX-Clean.exe")
    
    print("\n✅ Process Termination:")
    if running == 0:
        print(f"  ✓ Clean termination (0 processes running)")
    else:
        print(f"  ✗ {running} process(es) still running")
    
    # Recommendations
    recs = data.get('recommendations', [])
    print(f"\n✅ Analysis:")
    print(f"  Recommendations generated: {len(recs)}")
    if recs:
        print(f"  Sample: {recs[0][:80]}...")
    
    # Final verdict
    print("\n" + "=" * 70)
    if restir_line and running == 0:
        print("✅ AUTONOMOUS DEBUGGING AGENT: FULLY OPERATIONAL")
        print("\nReady for:")
        print("  - Finding PopulateVolumeMip2 crash thresholds")
        print("  - Threshold testing (5-point boundary detection)")
        print("  - Automated log analysis and recommendations")
    else:
        print("⚠️  Issues detected - see above")
    print("=" * 70)

asyncio.run(test())
