"""Tests for Phase 2A-5: Parameter Bounds and Damped Convergence."""

import os
import sys
import pytest

# Ensure the orchestrator package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.parameter_bounds import (
    ParameterBound,
    PARAMETER_BOUNDS,
    OSCILLATION_DAMPING_FACTOR,
    apply_bounds,
)


# ===== ParameterBound dataclass =====


class TestParameterBound:
    """Tests for the ParameterBound dataclass."""

    def test_clamp_above_max(self):
        """Spec test 1: propose energy=5000 with max=200 → clamped to 200."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        assert bound.clamp(5000) == 200.0

    def test_clamp_below_min(self):
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        assert bound.clamp(5) == 20.0

    def test_clamp_within_range(self):
        """Spec test 3: propose energy=150 within [20, 200] → unchanged."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        assert bound.clamp(150) == 150.0

    def test_damped_change_limits_step(self):
        """Spec test 2: current=50, target=200, step_size=50 → result=100."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        result = bound.damped_change(current=50, target=200)
        assert result == 100.0

    def test_damped_change_negative_direction(self):
        """Large decrease is also clamped to step_size."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        result = bound.damped_change(current=200, target=30)
        assert result == 150.0

    def test_damped_change_within_step(self):
        """Small change within step_size passes through unchanged."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        result = bound.damped_change(current=100, target=120)
        assert result == 120.0

    def test_damped_change_respects_clamp(self):
        """Result of damped change is also clamped to range."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        # current=190, target=300 → damped to 240 → clamped to 200
        result = bound.damped_change(current=190, target=300)
        assert result == 200.0

    def test_damped_change_effective_step_override(self):
        """Effective step override works for oscillation damping."""
        bound = ParameterBound("energy", 20.0, 200.0, step_size=50.0)
        result = bound.damped_change(current=50, target=200, effective_step=25.0)
        assert result == 75.0


# ===== apply_bounds function =====


class TestApplyBounds:
    """Tests for the apply_bounds function."""

    def test_clamps_out_of_range(self):
        """Out-of-range values get clamped."""
        mods = {"energy": 5000.0}
        result = apply_bounds("fire", mods)
        assert result["energy"] == 200.0

    def test_effect_type_routing(self):
        """Spec test 4: fire bounds applied for fire, NOT for liquid."""
        mods = {"energy": 5000.0}
        fire_result = apply_bounds("fire", mods)
        liquid_result = apply_bounds("liquid", mods)
        # Fire max is 200, liquid max is 300
        assert fire_result["energy"] == 200.0
        assert liquid_result["energy"] == 300.0

    def test_unknown_effect_type_passes_through(self):
        """Unknown effect types get no bounds applied."""
        mods = {"energy": 99999.0}
        result = apply_bounds("unknown_type", mods)
        assert result["energy"] == 99999.0

    def test_unknown_param_passes_through(self):
        """Params not in bounds dict pass through unchanged."""
        mods = {"custom_weird_param": 99999.0}
        result = apply_bounds("fire", mods)
        assert result["custom_weird_param"] == 99999.0

    def test_non_numeric_passes_through(self):
        """Non-numeric values (strings, bools) pass through."""
        mods = {"technique": "mantaflow", "enabled": True}
        result = apply_bounds("fire", mods)
        assert result["technique"] == "mantaflow"
        assert result["enabled"] is True

    def test_damped_with_previous_params(self):
        """Damped convergence uses previous value when available."""
        mods = {"energy": 200.0}
        prev = {"energy": 50.0}
        # step_size=50, so 50→200 gets damped to 100
        result = apply_bounds("fire", mods, previous_params=prev)
        assert result["energy"] == 100.0

    def test_damped_without_previous_params(self):
        """Without previous params, just clamps."""
        mods = {"energy": 150.0}
        result = apply_bounds("fire", mods, previous_params=None)
        assert result["energy"] == 150.0

    def test_oscillating_param_gets_tighter_bounds(self):
        """Spec test 5: oscillating parameter gets tighter bounds (halved step)."""
        mods = {"energy": 200.0}
        prev = {"energy": 50.0}
        oscillating = {"energy"}

        result = apply_bounds(
            "fire", mods, previous_params=prev, oscillating_params=oscillating
        )
        # Normal step=50, halved=25, so 50→200 damped to 75
        assert result["energy"] == 75.0

    def test_non_oscillating_param_normal_step(self):
        """Non-oscillating params keep normal step size."""
        mods = {"energy": 200.0}
        prev = {"energy": 50.0}
        oscillating = {"density"}  # Different param

        result = apply_bounds(
            "fire", mods, previous_params=prev, oscillating_params=oscillating
        )
        # energy not oscillating, step=50, so 50→200 damped to 100
        assert result["energy"] == 100.0

    def test_preserves_int_type(self):
        """Integer input produces integer output."""
        mods = {"resolution_max": 512}
        result = apply_bounds("liquid", mods)
        assert result["resolution_max"] == 256
        assert isinstance(result["resolution_max"], int)

    def test_mixed_params(self):
        """Mix of bounded and unbounded params."""
        mods = {
            "energy": 5000.0,
            "custom_param": 42.0,
            "label": "test",
        }
        result = apply_bounds("fire", mods)
        assert result["energy"] == 200.0  # clamped
        assert result["custom_param"] == 42.0  # unbounded, pass-through
        assert result["label"] == "test"  # non-numeric, pass-through

    def test_empty_modifications(self):
        """Empty modifications dict returns empty."""
        result = apply_bounds("fire", {})
        assert result == {}


# ===== Kill switch =====


class TestKillSwitch:
    """Tests for ENABLE_PARAMETER_BOUNDS kill switch."""

    def test_disabled_passes_through(self, monkeypatch):
        """When disabled, all values pass through unchanged."""
        import tools.parameter_bounds as module

        monkeypatch.setattr(module, "ENABLE_PARAMETER_BOUNDS", False)
        mods = {"energy": 99999.0}
        result = apply_bounds("fire", mods)
        assert result["energy"] == 99999.0

    def test_enabled_applies_bounds(self, monkeypatch):
        """When enabled, bounds are applied."""
        import tools.parameter_bounds as module

        monkeypatch.setattr(module, "ENABLE_PARAMETER_BOUNDS", True)
        mods = {"energy": 99999.0}
        result = apply_bounds("fire", mods)
        assert result["energy"] == 200.0


# ===== Bounds coverage =====


class TestBoundsCoverage:
    """Verify all effect types have reasonable bounds."""

    @pytest.mark.parametrize(
        "effect_type",
        ["fire", "explosion", "smoke", "pyro", "liquid", "water", "water_splash", "nebula", "sun"],
    )
    def test_effect_type_has_bounds(self, effect_type):
        """Every listed effect type has at least one bounded parameter."""
        assert effect_type in PARAMETER_BOUNDS
        assert len(PARAMETER_BOUNDS[effect_type]) > 0

    @pytest.mark.parametrize("effect_type", list(PARAMETER_BOUNDS.keys()))
    def test_bounds_are_valid(self, effect_type):
        """All bounds have min < max and positive step_size."""
        for param_name, bound in PARAMETER_BOUNDS[effect_type].items():
            assert bound.min_value < bound.max_value, (
                f"{effect_type}.{param_name}: min={bound.min_value} >= max={bound.max_value}"
            )
            assert bound.step_size > 0, (
                f"{effect_type}.{param_name}: step_size={bound.step_size} <= 0"
            )
