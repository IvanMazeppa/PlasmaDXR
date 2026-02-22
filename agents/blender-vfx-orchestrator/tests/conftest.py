"""Shared test fixtures and path setup for the tests/ package."""
import sys
from pathlib import Path

# Ensure the orchestrator root is on sys.path so tests can import
# models/, utils/, tools/ etc. without package installation.
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)
