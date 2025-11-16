#!/usr/bin/env python3
"""
Test script for PIX Debugging MCP Server
Verifies all tools are properly exposed and functional
"""

import asyncio
import json
import sys

async def test_mcp_server():
    """Test MCP server tool definitions"""
    print("=" * 70)
    print("PIX Debugging MCP Server - Tool Test")
    print("=" * 70)
    
    # Import server
    try:
        import mcp_server
        print("✅ MCP server module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}")
        return False
    
    # Test list_tools
    try:
        tools = await mcp_server.list_tools()
        print(f"\n✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:60]}...")
        
        # Verify expected tools
        expected_tools = [
            "capture_buffers",
            "analyze_restir_reservoirs", 
            "analyze_particle_buffers",
            "pix_capture",
            "pix_list_captures",
            "diagnose_visual_artifact"
        ]
        
        tool_names = [t.name for t in tools]
        missing_tools = [t for t in expected_tools if t not in tool_names]
        
        if missing_tools:
            print(f"\n⚠️  Missing tools: {missing_tools}")
            return False
        else:
            print("\n✅ All expected tools are registered")
            
    except Exception as e:
        print(f"\n❌ Error listing tools: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test diagnose tool (doesn't require file access)
    try:
        print("\n" + "=" * 70)
        print("Testing diagnose_visual_artifact tool...")
        print("=" * 70)
        
        result = await mcp_server.call_tool(
            "diagnose_visual_artifact",
            {"symptom": "black dots at far distances"}
        )
        
        print("✅ Tool executed successfully")
        
        # Parse result
        if result and len(result) > 0:
            result_text = result[0].text
            result_json = json.loads(result_text)
            print(f"\nResult summary: {result_json.get('summary', 'N/A')}")
            
            if result_json.get('recommendations'):
                print(f"Recommendations: {len(result_json['recommendations'])} items")
        
    except Exception as e:
        print(f"❌ Error testing tool: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nMCP server is ready for use in Claude Code!")
    print("\nTo configure in Claude Code, add to ~/.config/claude-code/config.json:")
    print(json.dumps({
        "mcpServers": {
            "pix-debug": {
                "command": "/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/venv/bin/python",
                "args": ["/mnt/d/Users/dilli/AndroidStudioProjects/Agility_SDI_DXR_MCP/pix-debugging-agent-v4/mcp_server.py"]
            }
        }
    }, indent=2))
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
