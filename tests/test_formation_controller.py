"""
Unit tests for L2 Formation Controller.

Tests cover:
- Mode initialization and state machine transitions
- Control force computation (LQR)
- Measurement processing (spherical to Cartesian conversion)
- No-measurement fallback (STM prediction)
- Telecommand generation
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from mission_sim.core.cyber.ids import FormationMode, ISLNetworkFrame
from mission_sim.core.cyber.platform_gnc.formation_controller import FormationController
from mission_sim.core.cyber.models.cw_dynamics import CWDynamics
from mission_sim.core.physics.ids import MicrowaveISLMeasurement
from mission_sim.core.spacetime.ids import Telecommand, CoordinateFrame


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def cw_dynamics():
    """CW dynamics for a typical LEO orbit (n = 0.001 rad/s ~ 90 min period)."""
    n = 0.001  # rad/s
    return CWDynamics(n)


@pytest.fixture
def controller(cw_dynamics):
    """Basic formation controller with default thresholds."""
    return FormationController(
        deputy_id="DEP_01",
        chief_id="CHIEF",
        dynamics=cw_dynamics,
        generation_threshold_pos=100.0,
        generation_threshold_vel=0.5,
        keeping_threshold_pos=1.0,
        keeping_threshold_vel=0.01,
    )


@pytest.fixture
def sample_measurement():
    """Sample spherical measurement (range 100m, azimuth 0.1 rad, elevation 0.05 rad)."""
    return MicrowaveISLMeasurement(
        phys_timestamp=10.0,
        range_m=100.0,
        azimuth_rad=0.1,
        elevation_rad=0.05,
        signal_strength=0.9
    )


@pytest.fixture
def network_frame(sample_measurement):
    """Network frame encapsulating the measurement."""
    return ISLNetworkFrame(
        payload=sample_measurement,
        source_id="CHIEF",
        dest_id="DEP_01",
        tx_time=9.0,
        rx_time=10.0
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_initial_mode(controller):
    """Test that controller starts in GENERATION mode."""
    assert controller.mode == FormationMode.GENERATION


def test_state_machine_generation_to_keeping(controller, cw_dynamics):
    """Test transition from GENERATION to KEEPING when errors fall below thresholds."""
    controller.last_estimated_state = np.array([50.0, 0.0, 0.0, 0.2, 0.0, 0.0])
    controller.update(current_time=11.0, frames=[], dt=1.0)
    assert controller.mode == FormationMode.KEEPING


def test_state_machine_remains_generation_if_errors_high(controller):
    """Test that mode stays in GENERATION when errors exceed thresholds."""
    controller.last_estimated_state = np.array([200.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    controller.update(current_time=11.0, frames=[], dt=1.0)
    assert controller.mode == FormationMode.GENERATION


def test_control_force_computation(controller, cw_dynamics):
    """Test that control force is computed via LQR and has correct shape."""
    controller.last_estimated_state = np.array([10.0, 5.0, 2.0, 0.1, 0.05, 0.02])
    cmd = controller.update(current_time=11.0, frames=[], dt=1.0)
    assert isinstance(cmd, Telecommand)
    assert cmd.frame == CoordinateFrame.LVLH
    assert cmd.force_vector.shape == (3,)
    assert cmd.duration_s == 1.0
    assert cmd.actuator_id == "DEP_01_thruster"
    assert np.linalg.norm(cmd.force_vector) > 0


def test_measurement_processing(controller, network_frame, cw_dynamics):
    """Test that a valid network frame updates the estimated state."""
    np.testing.assert_array_equal(controller.last_estimated_state, np.zeros(6))
    controller.update(current_time=11.0, frames=[network_frame], dt=1.0)

    pos = controller.last_estimated_state[0:3]
    # Expected Cartesian: (99.5, 9.95, 5.0) * alpha (0.7) = (69.65, 6.965, 3.5)
    assert pos[0] == pytest.approx(69.65, abs=1.0)
    assert pos[1] == pytest.approx(6.965, abs=0.5)
    assert pos[2] == pytest.approx(3.5, abs=0.5)
    # Velocity may have small non-zero values due to STM prediction, so use atol
    np.testing.assert_allclose(controller.last_estimated_state[3:6], [0.0, 0.0, 0.0], atol=1e-3)


def test_stale_frame_ignored(controller, network_frame, cw_dynamics):
    """Test that stale frames (age > max_delay) are ignored."""
    network_frame.payload.phys_timestamp = 10.0
    controller.update(current_time=30.0, frames=[network_frame], dt=1.0)
    np.testing.assert_array_equal(controller.last_estimated_state, np.zeros(6))


def test_frame_not_destined_for_this_deputy(controller, network_frame, cw_dynamics):
    """Test that frames destined for other deputy are ignored."""
    network_frame.dest_id = "OTHER_DEP"
    controller.update(current_time=11.0, frames=[network_frame], dt=1.0)
    np.testing.assert_array_equal(controller.last_estimated_state, np.zeros(6))


def test_no_measurement_prediction(controller, cw_dynamics):
    """Test that when no frames, the controller predicts using STM."""
    initial_state = np.array([10.0, 5.0, 2.0, 0.1, 0.05, 0.02])
    controller.last_estimated_state = initial_state.copy()
    controller.update(current_time=11.0, frames=[], dt=1.0)

    # Compute expected state using the controller's own dynamics and STM
    stm = controller.dynamics.compute_discrete_stm(1.0)
    expected = stm @ initial_state
    np.testing.assert_allclose(controller.last_estimated_state, expected, atol=1e-6)


def test_reset(controller):
    """Test reset method restores initial state."""
    controller.last_estimated_state = np.array([10.0, 5.0, 2.0, 0.1, 0.05, 0.02])
    controller.mode = FormationMode.KEEPING
    controller.reset()
    np.testing.assert_array_equal(controller.last_estimated_state, np.zeros(6))
    assert controller.mode == FormationMode.GENERATION


def test_telecommand_force_frame(controller, cw_dynamics):
    """Test that telecommand frame is always LVLH."""
    cmd = controller.update(current_time=11.0, frames=[], dt=1.0)
    assert cmd.frame == CoordinateFrame.LVLH


def test_control_force_scaling(controller, cw_dynamics):
    """Test that control force magnitude is reasonable (not huge)."""
    controller.last_estimated_state = np.array([1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0])
    cmd = controller.update(current_time=11.0, frames=[], dt=1.0)
    assert np.linalg.norm(cmd.force_vector) < 1e6


def test_spherical_conversion_accuracy(controller, cw_dynamics):
    """Test that spherical to Cartesian conversion in update is accurate."""
    meas = MicrowaveISLMeasurement(
        phys_timestamp=10.0,
        range_m=100.0,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        signal_strength=1.0
    )
    frame = ISLNetworkFrame(payload=meas, source_id="CHIEF", dest_id="DEP_01", tx_time=9.0, rx_time=10.0)
    controller.update(current_time=11.0, frames=[frame], dt=1.0)
    # Expected Cartesian: (100, 0, 0) then scaled by alpha=0.7 => (70, 0, 0)
    # Allow small floating-point errors
    np.testing.assert_allclose(controller.last_estimated_state[0:3], [70.0, 0.0, 0.0], atol=1e-3)