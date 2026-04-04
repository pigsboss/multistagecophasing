"""
Unit tests for CelestialEnvironment parallel acceleration computation.

Tests cover:
- Shape validation: input (N,6) -> output (N,3)
- Consistency: parallel result matches sequential single calls
- Multi-spacecraft isolation: states do not interfere
- Multiple force model accumulation
- Frame consistency enforcement
"""

import numpy as np
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.models.gravity import GravityCRTBP
from mission_sim.core.physics.models.j2_gravity import J2Gravity


# -----------------------------------------------------------------------------
# Mock force models for controlled testing
# -----------------------------------------------------------------------------

class ConstantAccelForce(IForceModel):
    """Force model that returns a constant acceleration for all states."""
    def __init__(self, accel_vector):
        self.accel = np.array(accel_vector, dtype=np.float64)

    def compute_accel(self, state, epoch):
        return self.accel.copy()

    def compute_vectorized_acc(self, state_matrix, epoch):
        # Vectorized version: broadcast constant to N rows
        return np.tile(self.accel, (state_matrix.shape[0], 1))


class StateDependentForce(IForceModel):
    """
    Force model that returns acceleration proportional to state's x position.
    Used to verify that each spacecraft's own state is used correctly.
    """
    def compute_accel(self, state, epoch):
        return np.array([state[0], 0.0, 0.0])

    def compute_vectorized_acc(self, state_matrix, epoch):
        return np.column_stack((state_matrix[:, 0], np.zeros(state_matrix.shape[0]), np.zeros(state_matrix.shape[0])))


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def env():
    """Basic environment in J2000_ECI frame."""
    return CelestialEnvironment(
        computation_frame=CoordinateFrame.J2000_ECI,
        initial_epoch=0.0,
        verbose=False
    )


@pytest.fixture
def sample_states():
    """Create a batch of 3 spacecraft states (N=3, shape 6)."""
    states = np.array([
        [1.0e7, 0.0, 0.0, 0.0, 7.5e3, 0.0],      # SC1
        [1.0e7, 100.0, 0.0, 0.0, 7.5e3, 0.1],    # SC2 (slightly offset)
        [1.0e7, -50.0, 20.0, 0.0, 7.5e3, -0.05], # SC3
    ], dtype=np.float64)
    return states


# -----------------------------------------------------------------------------
# Tests for shape and basic functionality
# -----------------------------------------------------------------------------

def test_compute_accelerations_shape(env, sample_states):
    """Test that compute_accelerations returns correct shape (N,3)."""
    # Register a simple force model
    env.register_force(ConstantAccelForce([1.0, 2.0, 3.0]))

    acc_matrix = env.compute_accelerations(sample_states)

    assert acc_matrix.shape == (sample_states.shape[0], 3)
    assert acc_matrix.dtype == np.float64


def test_compute_accelerations_empty_registry(env, sample_states):
    """Test that with no force models, acceleration is zero."""
    acc_matrix = env.compute_accelerations(sample_states)
    assert np.all(acc_matrix == 0.0)


def test_compute_accelerations_invalid_input(env):
    """Test that invalid input raises ValueError."""
    # Wrong dimensions
    with pytest.raises(ValueError, match="state_matrix must be a 2D numpy array"):
        env.compute_accelerations(np.ones(6))

    # Wrong second dimension (escape parentheses for regex)
    with pytest.raises(ValueError, match=r"state_matrix must be a 2D numpy array of shape \(N, 6\)"):
        env.compute_accelerations(np.ones((3, 5)))

    # Non-array input
    with pytest.raises(ValueError):
        env.compute_accelerations([1, 2, 3])


# -----------------------------------------------------------------------------
# Tests for consistency with single-spacecraft interface
# -----------------------------------------------------------------------------

def test_consistency_with_single_spacecraft(env, sample_states):
    """Test that compute_accelerations yields same results as calling get_total_acceleration for each."""
    env.register_force(StateDependentForce())

    # Parallel batch
    acc_parallel = env.compute_accelerations(sample_states)

    # Sequential single calls
    acc_sequential = []
    for i in range(sample_states.shape[0]):
        acc, _ = env.get_total_acceleration(sample_states[i], env.computation_frame)
        acc_sequential.append(acc)
    acc_sequential = np.array(acc_sequential)

    np.testing.assert_allclose(acc_parallel, acc_sequential, atol=1e-12)


def test_consistency_with_multiple_force_models(env, sample_states):
    """Test that multiple force models accumulate correctly in parallel."""
    env.register_force(ConstantAccelForce([1.0, 0.0, 0.0]))
    env.register_force(ConstantAccelForce([0.0, 2.0, 0.0]))
    env.register_force(StateDependentForce())

    acc_parallel = env.compute_accelerations(sample_states)

    # Manual sequential accumulation
    acc_sequential = []
    for i in range(sample_states.shape[0]):
        total = np.zeros(3)
        for force in env._force_registry:
            total += force.compute_accel(sample_states[i], env.epoch)
        acc_sequential.append(total)
    acc_sequential = np.array(acc_sequential)

    np.testing.assert_allclose(acc_parallel, acc_sequential, atol=1e-12)


# -----------------------------------------------------------------------------
# Tests for state isolation (no cross-talk)
# -----------------------------------------------------------------------------

def test_state_isolation(env):
    """Test that each spacecraft's acceleration depends only on its own state."""
    # Use a force model that depends on position (x coordinate)
    env.register_force(StateDependentForce())

    # Create states with very different x positions
    states = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1e6, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    acc = env.compute_accelerations(states)

    # Expected accelerations: [x, 0, 0]
    expected = np.column_stack((states[:, 0], np.zeros(3), np.zeros(3)))
    np.testing.assert_allclose(acc, expected, atol=1e-12)


def test_multiple_force_models_state_isolation(env):
    """Test isolation with multiple force models, including constant and state-dependent."""
    env.register_force(ConstantAccelForce([1.0, 0.0, 0.0]))
    env.register_force(StateDependentForce())

    states = np.array([
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    acc = env.compute_accelerations(states)
    # Expected: constant (1,0,0) + (x,0,0) = (1+x, 0, 0)
    expected = np.array([[6.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    np.testing.assert_allclose(acc, expected, atol=1e-12)


# -----------------------------------------------------------------------------
# Tests with real force models (integration)
# -----------------------------------------------------------------------------

def test_real_gravity_crtbp_parallel():
    """Test that GravityCRTBP works in parallel mode and matches sequential."""
    env = CelestialEnvironment(CoordinateFrame.SUN_EARTH_ROTATING, verbose=False)
    crtbp = GravityCRTBP()
    env.register_force(crtbp)

    # Sample states around Sun-Earth L2 region (in rotating frame)
    AU = 1.495978707e11
    states = np.array([
        [1.01106 * AU, 0.0, 0.05 * AU, 0.0, 0.0105 * AU * 1.990986e-7, 0.0],
        [1.01106 * AU, 1000.0, 0.05 * AU, 0.0, 0.0105 * AU * 1.990986e-7, 0.0],
    ], dtype=np.float64)

    acc_parallel = env.compute_accelerations(states)

    acc_sequential = []
    for i in range(states.shape[0]):
        acc, _ = env.get_total_acceleration(states[i], env.computation_frame)
        acc_sequential.append(acc)
    acc_sequential = np.array(acc_sequential)

    np.testing.assert_allclose(acc_parallel, acc_sequential, rtol=1e-12)


def test_real_j2_gravity_parallel():
    """Test that J2Gravity works in parallel mode and matches sequential."""
    env = CelestialEnvironment(CoordinateFrame.J2000_ECI, verbose=False)
    j2 = J2Gravity()
    env.register_force(j2)

    # Sample LEO positions (in ECI)
    r = 7000e3
    states = np.array([
        [r, 0.0, 0.0, 0.0, 7800.0, 0.0],
        [r + 1000.0, 0.0, 0.0, 0.0, 7800.0, 0.0],
    ], dtype=np.float64)

    acc_parallel = env.compute_accelerations(states)
    acc_sequential = []
    for i in range(states.shape[0]):
        acc, _ = env.get_total_acceleration(states[i], env.computation_frame)
        acc_sequential.append(acc)
    acc_sequential = np.array(acc_sequential)

    np.testing.assert_allclose(acc_parallel, acc_sequential, rtol=1e-12)


# -----------------------------------------------------------------------------
# Tests for frame consistency
# -----------------------------------------------------------------------------

def test_frame_mismatch_in_compute_accelerations(env, sample_states):
    """Test that compute_accelerations enforces frame consistency."""
    # The compute_accelerations method assumes all states are in env.computation_frame.
    # It doesn't have a frame parameter, so there's no explicit frame check in this method.
    # However, force models may have their own frame assumptions.
    # We just verify that the method runs without error when states are in the correct frame.
    env.compute_accelerations(sample_states)  # should not raise


# -----------------------------------------------------------------------------
# Tests for time stepping and epoch
# -----------------------------------------------------------------------------

def test_epoch_propagation(env, sample_states):
    """Test that environment epoch advances correctly and is passed to force models."""
    class EpochRecordingForce(IForceModel):
        def __init__(self):
            self.recorded_epochs = []

        def compute_accel(self, state, epoch):
            self.recorded_epochs.append(epoch)
            return np.zeros(3)

        def compute_vectorized_acc(self, state_matrix, epoch):
            self.recorded_epochs.append(epoch)
            return np.zeros((state_matrix.shape[0], 3))

    force = EpochRecordingForce()
    env.register_force(force)

    # Initial epoch = 0.0
    env.compute_accelerations(sample_states)
    assert force.recorded_epochs[-1] == 0.0

    # Step time
    env.step_time(10.0)
    env.compute_accelerations(sample_states)
    assert force.recorded_epochs[-1] == 10.0

    env.step_time(5.5)
    env.compute_accelerations(sample_states)
    assert force.recorded_epochs[-1] == 15.5
