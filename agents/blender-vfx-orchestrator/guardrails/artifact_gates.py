"""
Artifact Gates: Deterministic validation of execution outputs.

These gates run AFTER Blender execution and BEFORE quality evaluation.
They catch structural failures (empty caches, missing renders) without
involving LLM evaluation — pure mechanical checks.

Key principle: Deterministic first, adaptive second.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ArtifactGateResult:
    """Result of artifact validation gate."""
    passed: bool
    gate_name: str
    reason: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"[{self.gate_name}] {status}: {self.reason}"


@dataclass
class ExecutionArtifactSummary:
    """Summary of execution artifacts for observability."""
    cache_exists: bool
    cache_size_bytes: int
    cache_file_count: int
    render_count: int
    render_paths: List[str]
    vdb_count: int
    vdb_paths: List[str]
    blend_exists: bool
    blend_path: Optional[str]
    log_exists: bool
    log_path: Optional[str]

    @property
    def cache_size_mb(self) -> float:
        return self.cache_size_bytes / (1024 * 1024)


# ============================================================================
# GATE THRESHOLDS (Tunable)
# ============================================================================

# Minimum cache size in bytes (100KB = meaningful simulation data)
# Based on postmortem: empty caches were ~1.3KB, valid low-res (64) caches were ~350KB,
# production-res caches were ~23MB. 100KB catches empty bakes while allowing low-res tests.
MIN_CACHE_SIZE_BYTES = 100_000  # 100KB

# Minimum number of render files
MIN_RENDER_COUNT = 1

# Minimum VDB file size for volumetric data (if VDB expected)
MIN_VDB_SIZE_BYTES = 10_000  # 10KB


# ============================================================================
# ARTIFACT DISCOVERY
# ============================================================================

def discover_execution_artifacts(
    output_dir: Optional[str],
    run_dir: Optional[str] = None,
    script_path: Optional[str] = None,
) -> ExecutionArtifactSummary:
    """
    Discover all artifacts from an execution run.

    Args:
        output_dir: Primary output directory (contains cache/, renders)
        run_dir: CLI runner log directory (may contain additional logs)
        script_path: Path to the executed script (used to find cache in script dir)

    Returns:
        ExecutionArtifactSummary with discovered artifacts
    """
    cache_exists = False
    cache_size_bytes = 0
    cache_file_count = 0
    render_paths: List[str] = []
    vdb_paths: List[str] = []
    blend_path: Optional[str] = None
    log_path: Optional[str] = None

    def _scan_cache_dir(cache_path: Path) -> tuple[bool, int, int]:
        """Helper to scan a cache directory and return (exists, size, count)."""
        if not cache_path.exists() or not cache_path.is_dir():
            return False, 0, 0
        size = 0
        count = 0
        for f in cache_path.rglob("*"):
            if f.is_file():
                count += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        return True, size, count

    # Search output_dir (primary location)
    if output_dir:
        output_path = Path(output_dir)
        if output_path.exists():
            # Check cache subdirectory
            cache_path = output_path / "cache"
            cache_exists, cache_size_bytes, cache_file_count = _scan_cache_dir(cache_path)

            # Find renders
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.exr", "*.tiff"]:
                render_paths.extend(str(p) for p in output_path.rglob(ext))

            # Find VDBs
            vdb_paths.extend(str(p) for p in output_path.rglob("*.vdb"))

            # Find .blend
            blends = list(output_path.rglob("*.blend"))
            if blends:
                blend_path = str(blends[0])

    # Search run_dir for logs and fallback artifacts
    if run_dir:
        run_path = Path(run_dir)
        if run_path.exists():
            logs = list(run_path.glob("*.log")) + list(run_path.glob("*.txt"))
            if logs:
                log_path = str(logs[0])

            # Check for cache in run_dir if not found in output_dir
            if not cache_exists:
                run_cache_path = run_path / "cache"
                exists, size, count = _scan_cache_dir(run_cache_path)
                if exists:
                    cache_exists, cache_size_bytes, cache_file_count = exists, size, count

            # Also check for renders/VDBs in run_dir if not found in output_dir
            if not render_paths:
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.exr", "*.tiff"]:
                    render_paths.extend(str(p) for p in run_path.rglob(ext))

            if not vdb_paths:
                vdb_paths.extend(str(p) for p in run_path.rglob("*.vdb"))

    # Search script directory for cache (R1 fix: scripts may write cache here)
    if script_path and not cache_exists:
        script_dir = Path(script_path).parent
        # Check common Mantaflow cache locations relative to script
        for cache_subpath in ["cache", "cache/FluidDomain", "../cache", "../cache/FluidDomain"]:
            potential_cache = script_dir / cache_subpath
            exists, size, count = _scan_cache_dir(potential_cache)
            if exists and size > 0:
                cache_exists, cache_size_bytes, cache_file_count = exists, size, count
                break

    # Also check /tmp/blender_vfx/ as common default location
    if not cache_exists:
        tmp_vfx = Path("/tmp/blender_vfx")
        if tmp_vfx.exists():
            # Find most recent cache directory
            cache_dirs = list(tmp_vfx.rglob("cache"))
            for cd in sorted(cache_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
                exists, size, count = _scan_cache_dir(cd)
                if exists and size > 0:
                    cache_exists, cache_size_bytes, cache_file_count = exists, size, count
                    break

    return ExecutionArtifactSummary(
        cache_exists=cache_exists,
        cache_size_bytes=cache_size_bytes,
        cache_file_count=cache_file_count,
        render_count=len(render_paths),
        render_paths=sorted(set(render_paths)),
        vdb_count=len(vdb_paths),
        vdb_paths=sorted(set(vdb_paths)),
        blend_exists=blend_path is not None,
        blend_path=blend_path,
        log_exists=log_path is not None,
        log_path=log_path,
    )


# ============================================================================
# ARTIFACT GATES
# ============================================================================

def gate_cache_size(
    summary: ExecutionArtifactSummary,
    min_size_bytes: int = MIN_CACHE_SIZE_BYTES,
    require_cache: bool = True,
) -> ArtifactGateResult:
    """
    Gate: Cache must exist and meet minimum size threshold.

    This catches the "empty bake" failure mode where Blender reports success
    but produces no actual simulation data (e.g., planar emitter without
    use_plane_init=True).

    Args:
        summary: Discovered artifacts
        min_size_bytes: Minimum acceptable cache size
        require_cache: If True, missing cache is a failure. If False, missing
                      cache passes (for non-simulation effects).

    Returns:
        ArtifactGateResult
    """
    if not summary.cache_exists:
        if require_cache:
            return ArtifactGateResult(
                passed=False,
                gate_name="CACHE_SIZE",
                reason="Cache directory does not exist",
                details={"expected_min_bytes": min_size_bytes},
            )
        else:
            return ArtifactGateResult(
                passed=True,
                gate_name="CACHE_SIZE",
                reason="Cache not required for this effect type",
                details={"cache_required": False},
            )

    if summary.cache_size_bytes < min_size_bytes:
        if not require_cache:
            # Cache exists but is tiny — OK for effects that don't need
            # Mantaflow cache (e.g., rigid body uses .blend pointcache)
            return ArtifactGateResult(
                passed=True,
                gate_name="CACHE_SIZE",
                reason=f"Cache small ({summary.cache_size_mb:.2f}MB) but not required for this technique",
                details={"cache_required": False, "cache_size_mb": summary.cache_size_mb},
            )
        return ArtifactGateResult(
            passed=False,
            gate_name="CACHE_SIZE",
            reason=f"Cache too small: {summary.cache_size_mb:.2f}MB < {min_size_bytes / 1_000_000:.2f}MB minimum",
            details={
                "cache_size_bytes": summary.cache_size_bytes,
                "cache_size_mb": summary.cache_size_mb,
                "min_size_bytes": min_size_bytes,
                "file_count": summary.cache_file_count,
            },
        )

    return ArtifactGateResult(
        passed=True,
        gate_name="CACHE_SIZE",
        reason=f"Cache size OK: {summary.cache_size_mb:.2f}MB ({summary.cache_file_count} files)",
        details={
            "cache_size_bytes": summary.cache_size_bytes,
            "cache_size_mb": summary.cache_size_mb,
            "file_count": summary.cache_file_count,
        },
    )


def gate_render_count(
    summary: ExecutionArtifactSummary,
    min_count: int = MIN_RENDER_COUNT,
) -> ArtifactGateResult:
    """
    Gate: At least one render file must exist.

    Args:
        summary: Discovered artifacts
        min_count: Minimum number of renders required

    Returns:
        ArtifactGateResult
    """
    if summary.render_count < min_count:
        return ArtifactGateResult(
            passed=False,
            gate_name="RENDER_COUNT",
            reason=f"Not enough renders: {summary.render_count} < {min_count} required",
            details={
                "render_count": summary.render_count,
                "min_count": min_count,
            },
        )

    return ArtifactGateResult(
        passed=True,
        gate_name="RENDER_COUNT",
        reason=f"Render count OK: {summary.render_count} renders",
        details={
            "render_count": summary.render_count,
            "render_paths": summary.render_paths[:5],  # First 5 for brevity
        },
    )


def gate_vdb_validity(
    summary: ExecutionArtifactSummary,
    require_vdb: bool = False,
    min_size_bytes: int = MIN_VDB_SIZE_BYTES,
) -> ArtifactGateResult:
    """
    Gate: If VDB files expected, they must exist and be non-trivial.

    Args:
        summary: Discovered artifacts
        require_vdb: If True, missing VDB is a failure
        min_size_bytes: Minimum acceptable VDB file size

    Returns:
        ArtifactGateResult
    """
    if not require_vdb and summary.vdb_count == 0:
        return ArtifactGateResult(
            passed=True,
            gate_name="VDB_VALIDITY",
            reason="VDB not required for this effect type",
            details={"vdb_required": False},
        )

    if require_vdb and summary.vdb_count == 0:
        return ArtifactGateResult(
            passed=False,
            gate_name="VDB_VALIDITY",
            reason="No VDB files found but VDB was required",
            details={"vdb_required": True, "vdb_count": 0},
        )

    # Check VDB file sizes
    if summary.vdb_paths:
        small_vdbs = []
        for vdb_path in summary.vdb_paths:
            try:
                size = Path(vdb_path).stat().st_size
                if size < min_size_bytes:
                    small_vdbs.append((vdb_path, size))
            except OSError:
                pass

        if small_vdbs and len(small_vdbs) == len(summary.vdb_paths):
            # All VDBs are too small
            return ArtifactGateResult(
                passed=False,
                gate_name="VDB_VALIDITY",
                reason=f"All VDB files too small (< {min_size_bytes} bytes)",
                details={
                    "small_vdbs": [(p, s) for p, s in small_vdbs[:3]],
                    "min_size_bytes": min_size_bytes,
                },
            )

    return ArtifactGateResult(
        passed=True,
        gate_name="VDB_VALIDITY",
        reason=f"VDB OK: {summary.vdb_count} files",
        details={
            "vdb_count": summary.vdb_count,
            "vdb_paths": summary.vdb_paths[:3],
        },
    )


# ============================================================================
# COMBINED GATE
# ============================================================================

def validate_execution_artifacts(
    output_dir: Optional[str],
    run_dir: Optional[str] = None,
    effect_type: Optional[str] = None,
    min_cache_size_bytes: int = MIN_CACHE_SIZE_BYTES,
    min_render_count: int = MIN_RENDER_COUNT,
    require_vdb: bool = False,
    verbose: bool = True,
    script_path: Optional[str] = None,
    technique: Optional[str] = None,
) -> tuple[bool, List[ArtifactGateResult], ExecutionArtifactSummary]:
    """
    Run all artifact gates and return combined result.

    This is the main entry point for artifact validation. It runs after
    Blender execution succeeds but before quality evaluation begins.

    Args:
        output_dir: Primary output directory
        run_dir: CLI runner log directory
        effect_type: Effect type (used to determine which gates apply)
        min_cache_size_bytes: Minimum cache size threshold
        min_render_count: Minimum render count
        require_vdb: Whether VDB output is required
        verbose: If True, print gate results to stderr
        script_path: Path to executed script (helps find cache in script dir)

    Returns:
        Tuple of (all_passed, gate_results, summary)
    """
    # Discover artifacts (R1 fix: also check script directory for cache)
    summary = discover_execution_artifacts(output_dir, run_dir, script_path)

    if verbose:
        print(f"[ArtifactGates] Discovered artifacts:", file=sys.stderr)
        print(f"  Cache: {summary.cache_size_mb:.2f}MB ({summary.cache_file_count} files)", file=sys.stderr)
        print(f"  Renders: {summary.render_count}", file=sys.stderr)
        print(f"  VDBs: {summary.vdb_count}", file=sys.stderr)

    # Determine if cache is required based on effect type
    # Shader-only effects (procedural, shader_based) don't need Mantaflow cache.
    # Rigid body simulations store state in the .blend pointcache, not as
    # separate cache files — so the cache-size gate is a false positive.
    cache_required = True
    no_cache_keywords = [
        "shader", "procedural", "material",
        "rigid_body", "rigid body", "rigidbody",
        "cell_fracture", "voronoi_fracture",
        "prefracture", "constraint", "fracture",
    ]
    # Check effect type
    if effect_type:
        effect_lower = effect_type.lower()
        if any(kw in effect_lower for kw in no_cache_keywords):
            cache_required = False
    # Check technique (rigid body techniques don't produce Mantaflow cache)
    if technique and not cache_required is False:
        technique_lower = technique.lower()
        if any(kw in technique_lower for kw in no_cache_keywords):
            cache_required = False

    # Run gates
    gate_results: List[ArtifactGateResult] = []

    # Gate 1: Cache size
    cache_gate = gate_cache_size(summary, min_cache_size_bytes, require_cache=cache_required)
    gate_results.append(cache_gate)

    # Gate 2: Render count
    render_gate = gate_render_count(summary, min_render_count)
    gate_results.append(render_gate)

    # Gate 3: VDB validity (optional)
    vdb_gate = gate_vdb_validity(summary, require_vdb=require_vdb)
    gate_results.append(vdb_gate)

    # Report results
    all_passed = all(g.passed for g in gate_results)

    if verbose:
        for gate in gate_results:
            status = "✓" if gate.passed else "✗"
            print(f"[ArtifactGates] {status} {gate}", file=sys.stderr)

        if not all_passed:
            failed = [g for g in gate_results if not g.passed]
            print(f"[ArtifactGates] GATE FAILED: {len(failed)} gate(s) failed", file=sys.stderr)

    return all_passed, gate_results, summary


def format_gate_failure_for_diagnosis(
    gate_results: List[ArtifactGateResult],
    summary: ExecutionArtifactSummary,
) -> str:
    """
    Format gate failures into a diagnosis prompt for the DIAGNOSE state.

    Args:
        gate_results: Results from validate_execution_artifacts
        summary: Artifact summary

    Returns:
        Formatted string describing the failures
    """
    failed_gates = [g for g in gate_results if not g.passed]

    if not failed_gates:
        return "All artifact gates passed."

    lines = [
        "## ARTIFACT GATE FAILURE",
        "",
        "The following deterministic gates failed BEFORE quality evaluation:",
        "",
    ]

    for gate in failed_gates:
        lines.append(f"### {gate.gate_name}: FAILED")
        lines.append(f"Reason: {gate.reason}")
        if gate.details:
            lines.append(f"Details: {gate.details}")
        lines.append("")

    lines.append("## Artifact Summary")
    lines.append(f"- Cache exists: {summary.cache_exists}")
    lines.append(f"- Cache size: {summary.cache_size_mb:.2f}MB ({summary.cache_file_count} files)")
    lines.append(f"- Render count: {summary.render_count}")
    lines.append(f"- VDB count: {summary.vdb_count}")

    # Add likely causes based on failure type
    lines.append("")
    lines.append("## Likely Causes")

    for gate in failed_gates:
        if gate.gate_name == "CACHE_SIZE":
            lines.append("- **Empty cache**: Simulation may not have run or emitter was misconfigured")
            lines.append("  - Check: `use_plane_init=True` for planar liquid emitters")
            lines.append("  - Check: Domain object selected and active during bake")
            lines.append("  - Check: Frame range includes simulation keyframes")
        elif gate.gate_name == "RENDER_COUNT":
            lines.append("- **No renders**: Render loop may have been skipped or output path invalid")
            lines.append("  - Check: Render output directory exists and is writable")
            lines.append("  - Check: Camera exists and is active")
            lines.append("  - Check: Scene has visible objects in render layers")
        elif gate.gate_name == "VDB_VALIDITY":
            lines.append("- **Invalid VDB**: Volume export may have failed")
            lines.append("  - Check: OpenVDB addon is enabled")
            lines.append("  - Check: Domain has valid cache data")

    return "\n".join(lines)
