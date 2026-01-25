"""
Tracing package for Blender VFX Orchestrator.

Provides custom TracingProcessor implementations for verbose logging
similar to OpenAI dashboard output.
"""

from .verbose_processor import (
    VerboseTraceProcessor,
    enable_verbose_tracing,
    disable_verbose_tracing,
)

__all__ = [
    "VerboseTraceProcessor",
    "enable_verbose_tracing",
    "disable_verbose_tracing",
]
