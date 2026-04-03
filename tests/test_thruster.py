"""
Unit tests for Thruster actuator hardware model.

Tests cover:
- Saturation (max thrust limit)
- Deadband (min thrust cutoff)
- Noise injection (Gaussian distribution)
- Mass flow rate calculation (Isp * g0)
- Direction preservation
- Statistical properties of noise
"""

import numpy as np
import pytest
from mission_sim.core.physics.components.actuators.thruster import Thruster


class TestThruster:
    """Test suite for Thruster actuator model."""

    @pytest.fixture
    def thruster(self):
        """Default thruster with moderate parameters."""
        return Thruster(
            max_thrust_n=1.0,
            min_thrust_n=0.01,
            noise_std_n=0.005,
            specific_impulse_s=3000.0
        )

    @pytest.fixture
    def zero_noise_thruster(self):
        """Thruster with zero noise for deterministic tests."""
        return Thruster(
            max_thrust_n=1.0,
            min_thrust_n=0.01,
            noise_std_n=0.0,
            specific_impulse_s=3000.0
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------
    def test_execute_zero_command(self, thruster):
        """Test that zero command produces zero force and zero mass flow."""
        commanded = np.array([0.0, 0.0, 0.0])
        force, mdot = thruster.execute(commanded)

        assert np.all(force == 0.0)
        assert mdot == 0.0

    def test_execute_direction_preservation(self, zero_noise_thruster):
        """Test that output force direction matches commanded direction."""
        commanded = np.array([0.5, 0.2, -0.1])
        force, _ = zero_noise_thruster.execute(commanded)

        # Direction should be identical (unit vector)
        commanded_dir = commanded / np.linalg.norm(commanded)
        force_dir = force / np.linalg.norm(force)
        np.testing.assert_allclose(force_dir, commanded_dir, atol=1e-12)

    # -------------------------------------------------------------------------
    # Saturation (max thrust limit)
    # -------------------------------------------------------------------------
    def test_saturation_clipping(self, zero_noise_thruster):
        """Test that force is clipped to max_thrust_n."""
        # Command above max
        commanded = np.array([2.0, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert np.linalg.norm(force) == zero_noise_thruster.max_thrust_n
        assert force[0] == zero_noise_thruster.max_thrust_n

        # Command exactly at max
        commanded = np.array([1.0, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert np.linalg.norm(force) == 1.0

        # Command below max but above min -> unchanged
        commanded = np.array([0.5, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert force[0] == 0.5

    # -------------------------------------------------------------------------
    # Deadband (min thrust cutoff)
    # -------------------------------------------------------------------------
    def test_deadband_cutoff(self, zero_noise_thruster):
        """Test that commands below min_thrust_n produce zero force."""
        # Command below min
        commanded = np.array([0.005, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert np.linalg.norm(force) == 0.0

        # Command exactly at min (should fire)
        commanded = np.array([0.01, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert force[0] == 0.01

        # Command slightly above min
        commanded = np.array([0.011, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert force[0] == 0.011

    # -------------------------------------------------------------------------
    # Mass flow rate calculation
    # -------------------------------------------------------------------------
    def test_mass_flow_rate_formula(self, zero_noise_thruster):
        """Test that mass flow rate = F / (Isp * g0)."""
        commanded = np.array([0.5, 0.0, 0.0])
        force, mdot = zero_noise_thruster.execute(commanded)

        expected_mdot = force[0] / (zero_noise_thruster.isp * zero_noise_thruster.g0)
        assert mdot == pytest.approx(expected_mdot, rel=1e-12)

    def test_mass_flow_rate_zero_when_no_thrust(self, thruster):
        """Test that zero command gives zero mass flow."""
        _, mdot = thruster.execute(np.zeros(3))
        assert mdot == 0.0

        # Deadband also gives zero mass flow
        commanded = np.array([0.005, 0.0, 0.0])  # below min_thrust
        _, mdot = thruster.execute(commanded)
        assert mdot == 0.0

    # -------------------------------------------------------------------------
    # Noise injection
    # -------------------------------------------------------------------------
    def test_noise_statistics(self):
        """Test that output force magnitude noise has the specified standard deviation."""
        np.random.seed(42)
        thruster = Thruster(
            max_thrust_n=1.0,
            min_thrust_n=0.0,      # disable deadband for this test
            noise_std_n=0.01,
            specific_impulse_s=3000.0
        )

        commanded = np.array([0.5, 0.0, 0.0])
        forces = []
        for _ in range(1000):
            force, _ = thruster.execute(commanded)
            forces.append(force[0])

        forces = np.array(forces)
        # Mean should be close to commanded (0.5)
        assert np.mean(forces) == pytest.approx(0.5, abs=0.01)
        # Std should be close to noise_std_n (0.01)
        assert np.std(forces) == pytest.approx(0.01, rel=0.15)

    def test_noise_does_not_affect_direction_significantly(self):
        """Test that noise does not reverse direction."""
        np.random.seed(42)
        thruster = Thruster(
            max_thrust_n=1.0,
            min_thrust_n=0.0,
            noise_std_n=0.05,
            specific_impulse_s=3000.0
        )

        commanded = np.array([0.5, 0.0, 0.0])
        for _ in range(100):
            force, _ = thruster.execute(commanded)
            # Force should remain positive (no sign flip)
            assert force[0] > 0
            assert abs(force[1]) < 0.1
            assert abs(force[2]) < 0.1

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------
    def test_command_with_negative_components(self, zero_noise_thruster):
        """Test that negative thrust commands work correctly."""
        commanded = np.array([-0.5, 0.0, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        assert force[0] == -0.5
        # Mass flow rate should be positive (absolute thrust)
        expected_mdot = 0.5 / (zero_noise_thruster.isp * zero_noise_thruster.g0)
        _, mdot = zero_noise_thruster.execute(commanded)
        assert mdot == expected_mdot

    def test_command_with_all_axes(self, zero_noise_thruster):
        """Test that 3D commands are properly scaled."""
        commanded = np.array([0.3, 0.4, 0.0])
        force, _ = zero_noise_thruster.execute(commanded)
        expected_mag = np.linalg.norm(commanded)
        assert np.linalg.norm(force) == expected_mag
        np.testing.assert_allclose(force, commanded, atol=1e-12)

    def test_saturation_on_diagonal_command(self, zero_noise_thruster):
        """Test saturation when command magnitude exceeds max_thrust."""
        commanded = np.array([0.8, 0.6, 0.0])  # magnitude = 1.0
        force, _ = zero_noise_thruster.execute(commanded)
        # Should be exactly at max thrust (1.0) in the same direction
        expected_mag = zero_noise_thruster.max_thrust_n
        assert np.linalg.norm(force) == expected_mag
        # Direction should be preserved
        np.testing.assert_allclose(force / expected_mag, commanded / 1.0, atol=1e-12)

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def test_repr(self, thruster):
        """Test __repr__ contains key parameters."""
        repr_str = repr(thruster)
        assert "Thruster" in repr_str
        assert "Max=1.0N" in repr_str
        assert "Noise=0.005N" in repr_str
        assert "Isp=3000.0s" in repr_str