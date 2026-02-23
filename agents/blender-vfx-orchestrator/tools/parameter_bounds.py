"""Phase 2A-5: Parameter Bounds and Damped Convergence.

Per-effect-type parameter bounds that prevent overcorrection.
All parameter modifications are clamped to safe ranges and
limited to a maximum step size per iteration.

Kill switch: set ENABLE_PARAMETER_BOUNDS=0 in environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


# Kill switch
ENABLE_PARAMETER_BOUNDS = os.environ.get("ENABLE_PARAMETER_BOUNDS", "1") != "0"


@dataclass
class ParameterBound:
    """Defines min/max range and max step size for a single parameter."""

    name: str
    min_value: float
    max_value: float
    step_size: float  # max absolute change per iteration

    def clamp(self, value: float) -> float:
        """Clamp value to [min_value, max_value]."""
        return max(self.min_value, min(self.max_value, value))

    def damped_change(
        self,
        current: float,
        target: float,
        effective_step: Optional[float] = None,
    ) -> float:
        """Limit change magnitude to step_size, then clamp to range.

        Args:
            current: The current parameter value.
            target: The desired target value.
            effective_step: Override step size (e.g. halved for oscillating params).

        Returns:
            Bounded result value.
        """
        step = effective_step if effective_step is not None else self.step_size
        delta = target - current
        if abs(delta) > step:
            delta = step if delta > 0 else -step
        result = current + delta
        return self.clamp(result)


# ---------------------------------------------------------------------------
# Per-effect-type bounds
# ---------------------------------------------------------------------------

PARAMETER_BOUNDS: Dict[str, Dict[str, ParameterBound]] = {
    # Pyro / fire
    "fire": {
        "energy": ParameterBound("energy", 20.0, 200.0, step_size=50.0),
        "density": ParameterBound("density", 1.0, 15.0, step_size=3.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 0.5, 10.0, step_size=2.0),
        "temperature": ParameterBound("temperature", 500.0, 3000.0, step_size=500.0),
        "vorticity": ParameterBound("vorticity", 0.0, 5.0, step_size=1.0),
        "dissolve_speed": ParameterBound("dissolve_speed", 1.0, 15.0, step_size=3.0),
    },
    "explosion": {
        "energy": ParameterBound("energy", 50.0, 500.0, step_size=100.0),
        "density": ParameterBound("density", 1.0, 20.0, step_size=5.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 1.0, 15.0, step_size=3.0),
        "temperature": ParameterBound("temperature", 800.0, 5000.0, step_size=800.0),
        "vorticity": ParameterBound("vorticity", 0.0, 8.0, step_size=2.0),
    },
    "smoke": {
        "energy": ParameterBound("energy", 5.0, 100.0, step_size=20.0),
        "density": ParameterBound("density", 0.5, 10.0, step_size=2.0),
        "temperature": ParameterBound("temperature", 300.0, 1500.0, step_size=300.0),
        "vorticity": ParameterBound("vorticity", 0.0, 3.0, step_size=0.5),
        "dissolve_speed": ParameterBound("dissolve_speed", 1.0, 20.0, step_size=4.0),
    },
    "pyro": {
        "energy": ParameterBound("energy", 20.0, 300.0, step_size=60.0),
        "density": ParameterBound("density", 1.0, 15.0, step_size=3.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 0.5, 12.0, step_size=2.5),
        "temperature": ParameterBound("temperature", 500.0, 4000.0, step_size=600.0),
        "vorticity": ParameterBound("vorticity", 0.0, 6.0, step_size=1.5),
    },
    # Liquid
    "liquid": {
        "energy": ParameterBound("energy", 30.0, 300.0, step_size=80.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 5.0, step_size=1.0),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
    "water": {
        "energy": ParameterBound("energy", 30.0, 300.0, step_size=80.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 5.0, step_size=1.0),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
    "water_splash": {
        "energy": ParameterBound("energy", 50.0, 400.0, step_size=100.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 3.0, step_size=0.5),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
    # Celestial
    "nebula": {
        "density": ParameterBound("density", 0.1, 5.0, step_size=1.0),
        "energy": ParameterBound("energy", 0.5, 50.0, step_size=10.0),
        "temperature": ParameterBound("temperature", 2000.0, 20000.0, step_size=3000.0),
    },
    "sun": {
        "energy": ParameterBound("energy", 50.0, 1000.0, step_size=200.0),
        "temperature": ParameterBound("temperature", 3000.0, 10000.0, step_size=1500.0),
        "density": ParameterBound("density", 1.0, 20.0, step_size=4.0),
    },
}


# Oscillation damping factor: step_size is multiplied by this for oscillating params
OSCILLATION_DAMPING_FACTOR = 0.5


def apply_bounds(
    effect_type: str,
    modifications: Dict[str, Any],
    previous_params: Optional[Dict[str, Any]] = None,
    oscillating_params: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Apply parameter bounds and damped convergence to proposed modifications.

    Args:
        effect_type: The VFX effect type (fire, liquid, explosion, etc.).
        modifications: Dict of {param_name: target_value}.
        previous_params: Previous iteration's parameter values (for damped change).
        oscillating_params: Params flagged as oscillating by PipelineMonitor.

    Returns:
        Bounded modifications dict.
    """
    if not ENABLE_PARAMETER_BOUNDS:
        return modifications

    bounds = PARAMETER_BOUNDS.get(effect_type, {})
    if not bounds:
        return modifications

    result: Dict[str, Any] = {}
    bounded_count = 0

    for param_name, target_value in modifications.items():
        # Only bound numeric values
        if not isinstance(target_value, (int, float)):
            result[param_name] = target_value
            continue

        bound = bounds.get(param_name)
        if bound is None:
            # No bounds defined for this param — pass through
            result[param_name] = target_value
            continue

        original = target_value

        # Determine effective step size
        effective_step = bound.step_size
        if oscillating_params and param_name in oscillating_params:
            effective_step = bound.step_size * OSCILLATION_DAMPING_FACTOR

        # Apply damped change if we have previous value
        if (
            previous_params
            and param_name in previous_params
            and isinstance(previous_params[param_name], (int, float))
        ):
            current = float(previous_params[param_name])
            bounded_value = bound.damped_change(current, float(target_value), effective_step)
        else:
            # No previous value — just clamp
            bounded_value = bound.clamp(float(target_value))

        # Preserve int type if original was int
        if isinstance(target_value, int):
            bounded_value = int(round(bounded_value))

        if bounded_value != original:
            bounded_count += 1
            print(
                f"[ParameterBounds] {param_name}: {original} → {bounded_value} "
                f"(range=[{bound.min_value}, {bound.max_value}], step={effective_step})",
                file=sys.stderr,
            )

        result[param_name] = bounded_value

    if bounded_count > 0:
        print(
            f"[ParameterBounds] {bounded_count}/{len(modifications)} params bounded for {effect_type}",
            file=sys.stderr,
        )

    return result
