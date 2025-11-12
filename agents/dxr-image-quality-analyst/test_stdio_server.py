#!/usr/bin/env python3
"""
Test MCP server stdio communication manually
Simulates what Claude Code does when connecting
"""
import asyncio
import json
import sys

async def test_server():
    """Test server by sending MCP initialize message"""

    # Start the server as a subprocess
    proc = await asyncio.create_subprocess_exec(
        'python', '-m', 'src.agent',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd='/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/agents/rtxdi-quality-analyzer'
    )

    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        }
    }

    print("Sending initialize request...")
    request_json = json.dumps(init_request) + "\n"
    proc.stdin.write(request_json.encode())
    await proc.stdin.drain()

    # Wait for response with timeout
    try:
        response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        response = json.loads(response_line.decode())
        print("✓ Server responded!")
        print(json.dumps(response, indent=2))

        # Send tools/list request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        print("\nSending tools/list request...")
        request_json = json.dumps(list_tools_request) + "\n"
        proc.stdin.write(request_json.encode())
        await proc.stdin.drain()

        response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        response = json.loads(response_line.decode())
        print("✓ Tools list received!")
        print(json.dumps(response, indent=2))

    except asyncio.TimeoutError:
        print("✗ Server did not respond (timeout)")
        stderr = await proc.stderr.read()
        if stderr:
            print("STDERR:", stderr.decode())
    except Exception as e:
        print(f"✗ Error: {e}")
        stderr = await proc.stderr.read()
        if stderr:
            print("STDERR:", stderr.decode())
    finally:
        proc.terminate()
        await proc.wait()

if __name__ == "__main__":
    asyncio.run(test_server())
