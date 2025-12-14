#!/bin/bash
cd "$(dirname "$0")"
# Use virtual environment with sentence-transformers for semantic search
exec ./venv/bin/python blender_server.py
