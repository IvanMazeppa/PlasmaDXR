#!/usr/bin/env python3
"""
PIX Debugging MCP Server for Claude Code
Run this to expose DXR debugging tools directly in your Claude Code session
"""

import asyncio
import json
import os
import subprocess
import struct
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pyautogui
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool
