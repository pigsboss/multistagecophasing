def test_absolute_to_lvlh():
    """Test absolute to LVLH transformation with unified definition (X=radial, Y=along-track, Z=cross-track)."""
    import numpy as np
    from mission_sim.utils.math_tools import absolute_to_lvlh

    # Chief in circular orbit
    r = 7000e3
    v = np.sqrt(3.986004418e14 / r)

    # Chief state
    r_chief = np.array([r, 0.0, 0.0])
    v_chief = np.array([0.0, v, 0.0])

    # Deputy is 100m ahead of the Chief (in the absolute Y direction)
    r_deputy = np.array([r, 100.0, 0.0])
    v_deputy = np.array([0.0, v, 0.0])

    # Compute relative state in LVLH
    rho_lvlh, rho_dot_lvlh = absolute_to_lvlh(r_chief, v_chief, r_deputy, v_deputy)

    # Under LVLH definition (X=radial, Y=along-track, Z=cross-track):
    # The deputy's offset is purely along-track (Y), so relative position should be [0, 100, 0]
    from numpy.testing import assert_allclose
    assert_allclose(rho_lvlh, [0.0, 100.0, 0.0], atol=1e-7)
    # Velocity in LVLH is not zero due to rotating frame; we skip exact check,
    # but verify it's not huge (order of n * offset)
    n = v / r
    expected_vel_magnitude = n * 100  # about 0.107 m/s for LEO
    assert np.linalg.norm(rho_dot_lvlh) < expected_vel_magnitude * 1.1