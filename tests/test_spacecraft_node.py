"""
Integration tests for SpacecraftNode (L2 formation node).

Tests cover:
- L1 compatibility: using node as a point mass (delegation to SpacecraftPointMass)
- L2 control: apply_control with thruster model, mass update, ΔV accumulation
- Sensing: ISL antenna measurement generation
- Transmission: network router encapsulation
- Mixed usage: simultaneous L1 thrust and L2 control
"""

import numpy as np
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.spacecraft_node import SpacecraftNode
from mission_sim.core.physics.ids import SpacecraftType, MicrowaveISLMeasurement
from mission_sim.core.physics.components.actuators.thruster import Thruster
from mission_sim.core.physics.components.sensors.isl_antenna import ISLAntenna
from mission_sim.core.cyber.network.isl_router import ISLRouter
from mission_sim.core.cyber.ids import ISLNetworkFrame


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def fixed_rng():
    """Seed random number generators for reproducible noise."""
    np.random.seed(42)


@pytest.fixture
def basic_node(fixed_rng):
    """Create a basic deputy node with default hardware."""
    initial_state = np.array([1.5e11, 0.0, 0.0, 0.0, 30e3, 0.0])  # Halo-like state
    node = SpacecraftNode(
        sc_id="DEP_01",
        initial_state=initial_state,
        frame=CoordinateFrame.SUN_EARTH_ROTATING,
        initial_mass=1000.0,
        sc_type=SpacecraftType.DEPUTY,
    )
    return node


@pytest.fixture
def chief_node(fixed_rng):
    """Create a chief node for relative measurements."""
    initial_state = np.array([1.5e11, 0.0, 0.0, 0.0, 30e3, 0.0])
    node = SpacecraftNode(
        sc_id="CHIEF",
        initial_state=initial_state,
        frame=CoordinateFrame.SUN_EARTH_ROTATING,
        initial_mass=2000.0,
        sc_type=SpacecraftType.CHIEF,
    )
    return node


# ----------------------------------------------------------------------
# L1 compatibility tests
# ----------------------------------------------------------------------
def test_l1_compatibility_apply_thrust(basic_node):
    """Test that apply_thrust works like original point mass."""
    initial_velocity = basic_node.velocity.copy()
    initial_dv = basic_node.accumulated_dv

    force = np.array([100.0, 0.0, 0.0])
    basic_node.apply_thrust(force, basic_node.frame)

    dt = 1.0
    gravity = np.zeros(3)
    deriv = basic_node.get_derivative(gravity, basic_node.frame)
    basic_node.state += deriv * dt
    basic_node.integrate_dv(dt)
    basic_node.clear_thrust()

    expected_velocity_change = np.array([0.1, 0.0, 0.0])
    actual_velocity_change = basic_node.velocity - initial_velocity
    np.testing.assert_allclose(actual_velocity_change, expected_velocity_change, atol=1e-9)

    assert basic_node.accumulated_dv == pytest.approx(initial_dv + 0.1, rel=1e-9)


def test_l1_delegation_properties(basic_node):
    """Test that properties work correctly (self-contained)."""
    assert basic_node.id == "DEP_01"
    assert basic_node.mass == 1000.0
    assert basic_node.accumulated_dv == 0.0
    assert np.array_equal(basic_node.position, basic_node.state[0:3])
    assert np.array_equal(basic_node.velocity, basic_node.state[3:6])


# ----------------------------------------------------------------------
# L2 control path tests
# ----------------------------------------------------------------------
def test_apply_control_thruster_effect(basic_node):
    """Test that apply_control passes command through thruster and updates external_accel."""
    commanded_force = np.array([0.5, 0.0, 0.0])
    initial_external_accel = basic_node.external_accel.copy()
    initial_mass = basic_node.mass

    basic_node.apply_control(commanded_force)

    expected_acc = commanded_force / initial_mass
    actual_acc = basic_node.external_accel - initial_external_accel
    np.testing.assert_allclose(actual_acc, expected_acc, atol=0.02)

    assert hasattr(basic_node, '_current_mass_flow_rate')
    assert basic_node._current_mass_flow_rate > 0


def test_update_mass_consumption(basic_node):
    """Test that mass decreases after calling update_mass."""
    commanded_force = np.array([0.5, 0.0, 0.0])
    dt = 10.0

    basic_node.apply_control(commanded_force)
    initial_mass = basic_node.mass
    basic_node.update_mass(dt)

    expected_mass_flow = 0.5 / (3000.0 * 9.80665)
    expected_mass = initial_mass - expected_mass_flow * dt
    assert basic_node.mass == pytest.approx(expected_mass, rel=1e-6)


def test_integrate_dv_with_control(basic_node):
    """Test that ΔV accumulation works with L2 control path."""
    commanded_force = np.array([0.5, 0.0, 0.0])
    dt = 1.0

    basic_node.apply_control(commanded_force)
    gravity = np.zeros(3)
    deriv = basic_node.get_derivative(gravity, basic_node.frame)
    basic_node.state += deriv * dt
    basic_node.integrate_dv(dt)
    basic_node.clear_thrust()

    expected_dv = (0.5 / basic_node.mass) * dt
    assert basic_node.accumulated_dv == pytest.approx(expected_dv, abs=0.02)


# ----------------------------------------------------------------------
# Sensing and transmission tests
# ----------------------------------------------------------------------
def test_sense_measurement(basic_node, chief_node):
    """Test that sense generates a MicrowaveISLMeasurement."""
    chief_node.state[:3] = basic_node.position + np.array([100.0, 50.0, 20.0])
    current_time = 1000.0

    measurement = basic_node.sense(chief_node, current_time)

    assert isinstance(measurement, MicrowaveISLMeasurement)
    assert measurement.phys_timestamp == current_time
    assert measurement.range_m == pytest.approx(113.8, abs=1.0)
    assert measurement.signal_strength == 1.0


def test_sense_out_of_range(basic_node, chief_node):
    """Test that sense returns None when target is beyond max_range."""
    chief_node.state[:3] = basic_node.position + np.array([250e3, 0.0, 0.0])
    measurement = basic_node.sense(chief_node, 0.0)
    assert measurement is None


def test_transmit_with_router(basic_node):
    """Test that transmit returns a network frame when router is provided."""
    router = ISLRouter(base_latency_s=0.1, jitter_s=0.0, packet_loss_rate=0.0)
    basic_node.router = router

    measurement = MicrowaveISLMeasurement(
        phys_timestamp=100.0,
        range_m=100.0,
        azimuth_rad=0.1,
        elevation_rad=0.05,
        signal_strength=0.8
    )
    frame = basic_node.transmit(measurement, "CHIEF", current_time=100.0)

    assert isinstance(frame, ISLNetworkFrame)
    assert frame.source_id == basic_node.id
    assert frame.dest_id == "CHIEF"
    assert frame.payload is measurement
    assert frame.rx_time == 100.0 + 0.1


def test_transmit_no_router(basic_node):
    """Test that transmit returns None when no router is set."""
    measurement = MicrowaveISLMeasurement(
        phys_timestamp=100.0,
        range_m=100.0,
        azimuth_rad=0.1,
        elevation_rad=0.05,
        signal_strength=0.8
    )
    frame = basic_node.transmit(measurement, "CHIEF", 100.0)
    assert frame is None


# ----------------------------------------------------------------------
# Mixed usage: L1 thrust + L2 control (should not conflict)
# ----------------------------------------------------------------------
def test_mixed_l1_l2_commands(basic_node):
    """Test that L1 apply_thrust and L2 apply_control both affect external_accel."""
    force_l1 = np.array([10.0, 0.0, 0.0])
    basic_node.apply_thrust(force_l1, basic_node.frame)

    force_l2 = np.array([5.0, 0.0, 0.0])
    basic_node.apply_control(force_l2)

    expected_total = (force_l1 + force_l2) / basic_node.mass
    np.testing.assert_allclose(basic_node.external_accel, expected_total, atol=0.01)

    basic_node.clear_thrust()
    assert np.all(basic_node.external_accel == 0.0)


# ----------------------------------------------------------------------
# Full L2 integration with environment and integration steps
# ----------------------------------------------------------------------
def test_full_integration_step_with_control(basic_node):
    """Simulate one full integration step using L2 control."""
    dt = 1.0
    gravity = np.zeros(3)
    commanded_force = np.array([1.0, 0.0, 0.0])

    basic_node.apply_control(commanded_force)
    deriv = basic_node.get_derivative(gravity, basic_node.frame)
    basic_node.state += deriv * dt
    basic_node.integrate_dv(dt)
    basic_node.update_mass(dt)
    basic_node.clear_thrust()

    expected_acc = commanded_force[0] / basic_node.mass
    expected_dv = expected_acc * dt
    assert basic_node.accumulated_dv == pytest.approx(expected_dv, abs=0.01)
    assert abs(basic_node.position[0]) > 0.0


# ----------------------------------------------------------------------
# Type consistency with SpacecraftPointMass (duck typing)
# ----------------------------------------------------------------------
def test_node_can_be_used_as_point_mass_in_function(basic_node):
    """Verify that functions expecting a point mass also work with SpacecraftNode."""
    def fake_integrator(sc, dt):
        sc.state += np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * dt
        return sc.state

    initial_state = basic_node.state.copy()
    fake_integrator(basic_node, 0.1)
    assert basic_node.state[0] == initial_state[0] + 0.1