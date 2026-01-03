#!/usr/bin/env python3
"""
Direct MCP Server Test - Bypasses Claude Agent SDK

Tests that MCP servers respond correctly to JSON-RPC protocol,
without relying on the inner Claude agent.

This helps diagnose whether the issue is:
1. MCP servers not starting/responding
2. SDK not correctly routing tool calls
3. Inner agent not making tool calls
"""

import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

def send_jsonrpc(process, method: str, params: dict = None, id: int = 1):
    """Send a JSON-RPC request to an MCP server process."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id
    }
    if params:
        request["params"] = params

    message = json.dumps(request)
    # MCP uses Content-Length header
    full_message = f"Content-Length: {len(message)}\r\n\r\n{message}"

    process.stdin.write(full_message)
    process.stdin.flush()

    # Read response
    # First read headers
    headers = {}
    while True:
        line = process.stdout.readline()
        if not line or line == "\r\n" or line == "\n":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    # Read body based on Content-Length
    content_length = int(headers.get("Content-Length", 0))
    if content_length > 0:
        body = process.stdout.read(content_length)
        return json.loads(body)
    return None


def test_script_generator():
    """Test the script-generator MCP server directly."""
    print("=" * 60)
    print("DIRECT MCP SERVER TEST - script-generator")
    print("=" * 60)

    server_dir = PROJECT_ROOT / "agents" / "script-generator"

    # Start the server
    print(f"\n1. Starting script-generator from: {server_dir}")

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        str(server_dir / "venv" / "bin" / "python"),
        str(server_dir / "server.py")
    ]

    print(f"   Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(server_dir),
            env=env,
            bufsize=0
        )

        print("   Server started, PID:", process.pid)

        # Give it a moment to initialize
        import time
        time.sleep(1)

        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"   ERROR: Server exited immediately!")
            print(f"   STDOUT: {stdout[:500] if stdout else 'empty'}")
            print(f"   STDERR: {stderr[:500] if stderr else 'empty'}")
            return False

        print("\n2. Sending initialize request...")

        # Send initialize request (MCP protocol)
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "direct-test",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }

        message = json.dumps(init_request)
        full_message = f"Content-Length: {len(message)}\r\n\r\n{message}"

        print(f"   Sending: {message[:100]}...")
        process.stdin.write(full_message)
        process.stdin.flush()

        # Read response with timeout
        import select

        print("   Waiting for response...")
        ready, _, _ = select.select([process.stdout], [], [], 5.0)

        if not ready:
            print("   ERROR: No response within 5 seconds")
            process.terminate()
            return False

        # Read headers
        response_text = ""
        while True:
            if not select.select([process.stdout], [], [], 1.0)[0]:
                break
            char = process.stdout.read(1)
            if not char:
                break
            response_text += char
            if response_text.endswith("\r\n\r\n"):
                break

        print(f"   Headers: {response_text.strip()}")

        # Parse Content-Length
        content_length = 0
        for line in response_text.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length > 0:
            body = process.stdout.read(content_length)
            print(f"   Response body: {body[:200]}...")

            try:
                response = json.loads(body)
                if "result" in response:
                    print("   ✅ Initialize succeeded!")
                    print(f"   Server info: {response['result'].get('serverInfo', {})}")

                    # Now try to list tools
                    print("\n3. Listing available tools...")

                    list_request = {
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 2
                    }
                    message = json.dumps(list_request)
                    full_message = f"Content-Length: {len(message)}\r\n\r\n{message}"

                    process.stdin.write(full_message)
                    process.stdin.flush()

                    # Read response
                    import time
                    time.sleep(0.5)

                    response_text = ""
                    while select.select([process.stdout], [], [], 1.0)[0]:
                        char = process.stdout.read(1)
                        if not char:
                            break
                        response_text += char
                        if "\r\n\r\n" in response_text:
                            # Get content length
                            for line in response_text.split("\r\n"):
                                if line.lower().startswith("content-length:"):
                                    cl = int(line.split(":")[1].strip())
                                    # Read body
                                    body = process.stdout.read(cl)
                                    tools_response = json.loads(body)
                                    if "result" in tools_response:
                                        tools = tools_response["result"].get("tools", [])
                                        print(f"   ✅ Found {len(tools)} tools:")
                                        for tool in tools[:5]:
                                            print(f"      - {tool.get('name')}")
                                        if len(tools) > 5:
                                            print(f"      ... and {len(tools) - 5} more")

                                        process.terminate()
                                        return True
                                    break
                            break

                elif "error" in response:
                    print(f"   ❌ Initialize error: {response['error']}")

            except json.JSONDecodeError as e:
                print(f"   ERROR: Invalid JSON response: {e}")

        process.terminate()
        return False

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run direct MCP tests."""
    print("\nDirect MCP Server Tests")
    print("This bypasses the Claude Agent SDK to test MCP servers directly.\n")

    if test_script_generator():
        print("\n" + "=" * 60)
        print("✅ SCRIPT-GENERATOR MCP SERVER WORKS!")
        print("=" * 60)
        print("\nThe MCP server responds correctly. The issue is likely:")
        print("1. SDK not correctly configuring the bundled Claude Code CLI")
        print("2. Inner agent not understanding it should use tools")
        print("3. Tool call format mismatch")
    else:
        print("\n" + "=" * 60)
        print("❌ SCRIPT-GENERATOR MCP SERVER FAILED")
        print("=" * 60)
        print("\nThe MCP server is not responding correctly.")


if __name__ == "__main__":
    main()
