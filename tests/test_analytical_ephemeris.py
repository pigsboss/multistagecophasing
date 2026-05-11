# tests/test_analytical_ephemeris.py
"""
Hey everyone! Here are some tests for the analytical N‑body backup ephemeris.

We check out:
- Interface compliance of NBodyTargetEphemeris
- Two‑body precision against an analytical (Kepler) reference
- Backward integration capability
- Correct state indexing for more than two bodies

We use pytest and the project's own Kepler solver as ground truth.
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from mission_sim.core.spacetime.ephemeris.base import Ephemeris
from mission_sim.core.spacetime.ephemeris.analytical import (
    NBodyPropagator,
    NBodyTargetEphemeris,
)
from mission_sim.core.spacetime.time import utc_smooth_to_tdb
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.utils.solvers.base import kepler_elements_to_cartesian_batch
import mission_sim.core.spacetime.time as timemod


# ---------------------------------------------------------------------------
# Fixture to provide a minimal leap‑second table so time conversion works
# ---------------------------------------------------------------------------
MINIMAL_LEAP_DATES = [
    datetime(1972, 1, 1, tzinfo=timezone.utc),
    datetime(1972, 7, 1, tzinfo=timezone.utc),
    datetime(1973, 1, 1, tzinfo=timezone.utc),
    datetime(1974, 1, 1, tzinfo=timezone.utc),
    datetime(1975, 1, 1, tzinfo=timezone.utc),
    datetime(1976, 1, 1, tzinfo=timezone.utc),
    datetime(1977, 1, 1, tzinfo=timezone.utc),
    datetime(1978, 1, 1, tzinfo=timezone.utc),
    datetime(1979, 1, 1, tzinfo=timezone.utc),
    datetime(1980, 1, 1, tzinfo=timezone.utc),
    datetime(1981, 7, 1, tzinfo=timezone.utc),
    datetime(1982, 7, 1, tzinfo=timezone.utc),
    datetime(1983, 7, 1, tzinfo=timezone.utc),
    datetime(1985, 7, 1, tzinfo=timezone.utc),
    datetime(1988, 1, 1, tzinfo=timezone.utc),
    datetime(1990, 1, 1, tzinfo=timezone.utc),
    datetime(1991, 1, 1, tzinfo=timezone.utc),
    datetime(1992, 7, 1, tzinfo=timezone.utc),
    datetime(1993, 7, 1, tzinfo=timezone.utc),
    datetime(1994, 7, 1, tzinfo=timezone.utc),
    datetime(1996, 1, 1, tzinfo=timezone.utc),
    datetime(1997, 7, 1, tzinfo=timezone.utc),
    datetime(1999, 1, 1, tzinfo=timezone.utc),
    datetime(2006, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 1, 1, tzinfo=timezone.utc),
    datetime(2012, 7, 1, tzinfo=timezone.utc),
    datetime(2015, 7, 1, tzinfo=timezone.utc),
    datetime(2017, 1, 1, tzinfo=timezone.utc),
]


@pytest.fixture(autouse=True)
def _install_test_leap_seconds():
    """Ensure time conversion functions have a valid leap‑second table."""
    saved = timemod._LEAP_SECONDS_DATES
    timemod._LEAP_SECONDS_DATES = sorted(MINIMAL_LEAP_DATES)
    yield
    timemod._LEAP_SECONDS_DATES = saved


# ---------------------------------------------------------------------------
# Helper: analytical two‑body state generator
# ---------------------------------------------------------------------------
def _analytical_two_body(
    times_tdb: np.ndarray,
    m1: float,
    m2: float,
    mu1: float,
    mu2: float,
    r_cm0: np.ndarray,
    v_cm0: np.ndarray,
    r_rel0: np.ndarray,
    v_rel0: np.ndarray,
):
    """
    Compute the inertial states of two bodies (m1, m2) at times *times_tdb*
    using the analytical Kepler solution for the relative motion.

    Parameters
    ----------
    times_tdb : 1D array of TDB seconds since the reference epoch.
    m1, m2 : float
        Masses (in units where G=1, i.e. mu_i = m_i).
    mu1, mu2 : float
        Gravitational parameters (must equal m1, m2 in this test).
    r_cm0, v_cm0 : ndarray (3,)
        Center‑of‑mass position and velocity at the reference epoch.
    r_rel0, v_rel0 : ndarray (3,)
        Relative position and velocity (body2 – body1) at the reference epoch.

    Returns
    -------
    states : ndarray (len(times_tdb), 2, 6)
        States of body 1 and body 2 in the inertial frame.
    """
    # Convert relative state to orbital elements (circular in our test)
    mu_total = mu1 + mu2
    a = np.linalg.norm(r_rel0)
    # For circular orbit:
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    n_motion = np.sqrt(mu_total / a**3)

    # Mean anomaly at reference epoch (set to 0 for simplicity)
    M0 = 0.0

    # Build input arrays for batch solver (one element per time)
    N = len(times_tdb)
    M_arr = M0 + n_motion * times_tdb
    a_arr = np.full(N, a)
    e_arr = np.full(N, e)
    i_arr = np.zeros(N)
    Omega_arr = np.zeros(N)
    omega_arr = np.zeros(N)

    # Relative position and velocity in the orbital plane
    rel_states = kepler_elements_to_cartesian_batch(
        a_arr, e_arr, i_arr, Omega_arr, omega_arr, M_arr, mu_total
    )  # shape (N, 6)

    # Decompose into body‑centered coordinates
    total_mass = m1 + m2
    states = np.empty((N, 2, 6), dtype=np.float64)

    # CM motion (constant)
    for k in range(N):
        dt = times_tdb[k]
        r_cm = r_cm0 + v_cm0 * dt
        v_cm = v_cm0

        r_rel = rel_states[k, :3]
        v_rel = rel_states[k, 3:]

        # Positions: body1 = r_cm - (m2/total_mass) * r_rel
        #            body2 = r_cm + (m1/total_mass) * r_rel
        r1 = r_cm - (m2 / total_mass) * r_rel
        r2 = r_cm + (m1 / total_mass) * r_rel
        v1 = v_cm - (m2 / total_mass) * v_rel
        v2 = v_cm + (m1 / total_mass) * v_rel

        states[k, 0, 0:3] = r1
        states[k, 0, 3:6] = v1
        states[k, 1, 0:3] = r2
        states[k, 1, 3:6] = v2

    return states


# ---------------------------------------------------------------------------
# Fixture: a simple two‑body system
# ---------------------------------------------------------------------------
@pytest.fixture
def two_body_system():
    """
    Create a barycentric circular two‑body system.

    G = 1, so masses equal gravitational parameters.
    m1 = 0.9, m2 = 0.1.
    Relative orbit radius = 1.0.
    Center of mass initially at rest at the origin.
    """
    m1 = 0.9
    m2 = 0.1
    mu1 = m1
    mu2 = m2

    a_rel = 1.0
    total_mass = m1 + m2
    mu_total = mu1 + mu2
    n_motion = np.sqrt(mu_total / a_rel**3)  # should be 1.0

    # CM at origin, zero velocity
    r_cm = np.zeros(3)
    v_cm = np.zeros(3)

    # Relative state: body2 – body1 = (1, 0, 0) with tangential velocity
    r_rel = np.array([a_rel, 0.0, 0.0])
    v_rel = np.array([0.0, n_motion * a_rel, 0.0])  # (0, 1, 0)

    # Individual initial states
    r1 = r_cm - (m2 / total_mass) * r_rel  # (-0.1, 0, 0)
    r2 = r_cm + (m1 / total_mass) * r_rel  # (0.9, 0, 0)
    v1 = v_cm - (m2 / total_mass) * v_rel  # (0, -0.1, 0)
    v2 = v_cm + (m1 / total_mass) * v_rel  # (0, 0.9, 0)

    return {
        "masses": (m1, m2),
        "mu": (mu1, mu2),
        "initial_states": {"body1": np.append(r1, v1), "body2": np.append(r2, v2)},
        "epoch_tdb": 0.0,
        "cm": (r_cm, v_cm),
        "rel": (r_rel, v_rel),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_interface_compliance(two_body_system):
    """
    Check that NBodyTargetEphemeris respects the Ephemeris contract.
    """
    prop = NBodyPropagator(
        bodies=["body1", "body2"],
        mu_list=two_body_system["mu"],
        initial_states_dict=two_body_system["initial_states"],
        epoch_tdb=two_body_system["epoch_tdb"],
    )
    ephem = NBodyTargetEphemeris(prop, "body1")

    # Inheritance
    assert isinstance(ephem, Ephemeris)

    # Coordinate frame
    assert ephem.frame == CoordinateFrame.J2000_ECI

    # get_state returns a (6,) array
    t_utc = 0.0
    state = ephem.get_state(t_utc)
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)
    assert state.dtype == np.float64

    # __call__ is an alias for get_state
    np.testing.assert_array_equal(ephem(t_utc), state)


def test_two_body_precision(two_body_system):
    """
    Compare N‑body integration with analytical Kepler solution
    over a short arc (0.1 of a period).
    """
    m1, m2 = two_body_system["masses"]
    mu1, mu2 = two_body_system["mu"]
    epoch_tdb = two_body_system["epoch_tdb"]
    r_cm0, v_cm0 = two_body_system["cm"]
    r_rel0, v_rel0 = two_body_system["rel"]

    # Sample times (TDB seconds)
    times = np.linspace(0.0, 0.1 * 2.0 * np.pi, 5)  # ~0.63 s

    # Analytical solution
    ana_states = _analytical_two_body(
        times, m1, m2, mu1, mu2, r_cm0, v_cm0, r_rel0, v_rel0
    )

    # N‑body propagator
    prop = NBodyPropagator(
        bodies=["body1", "body2"],
        mu_list=[mu1, mu2],
        initial_states_dict=two_body_system["initial_states"],
        epoch_tdb=epoch_tdb,
    )

    errors = []
    for k, t_tdb in enumerate(times):
        for ib, body_name in enumerate(["body1", "body2"]):
            num_state = prop.get_body_state(body_name, t_tdb)
            ana_state = ana_states[k, ib]
            errors.append(np.abs(num_state - ana_state).max())

    max_err = max(errors)
    # DP5(4) should deliver sub‑millimetre accuracy on such a short arc
    assert max_err < 1e-6, f"Max position error {max_err:.2e} exceeds 1e-6"


def test_backward_propagation(two_body_system):
    """
    Verify that the propagator can integrate backward in time.
    """
    m1, m2 = two_body_system["masses"]
    mu1, mu2 = two_body_system["mu"]
    r_cm0, v_cm0 = two_body_system["cm"]
    r_rel0, v_rel0 = two_body_system["rel"]

    times = np.array([-0.1, 0.0])  # backwards

    ana_states = _analytical_two_body(
        times, m1, m2, mu1, mu2, r_cm0, v_cm0, r_rel0, v_rel0
    )

    prop = NBodyPropagator(
        bodies=["body1", "body2"],
        mu_list=[mu1, mu2],
        initial_states_dict=two_body_system["initial_states"],
        epoch_tdb=0.0,
    )

    for k, t_tdb in enumerate(times):
        for ib, body_name in enumerate(["body1", "body2"]):
            num_state = prop.get_body_state(body_name, t_tdb)
            ana_state = ana_states[k, ib]
            err = np.abs(num_state - ana_state).max()
            assert err < 1e-6, f"Backward error at t={t_tdb} too large: {err:.2e}"


def test_multi_body():
    """
    Ensure that more than two bodies are handled and indexed correctly.
    """
    # Three bodies: zero mass for the third one so it does not perturb the two
    # main ones, but its state advances with constant velocity.
    m1, m2 = 0.9, 0.1
    mu1, mu2 = m1, m2
    mu3 = 0.0
    m3 = 0.0

    r1 = np.array([-0.1, 0.0, 0.0])
    v1 = np.array([0.0, -0.1, 0.0])
    r2 = np.array([0.9, 0.0, 0.0])
    v2 = np.array([0.0, 0.9, 0.0])
    r3 = np.array([5.0, 0.0, 0.0])   # far away
    v3 = np.array([0.2, 0.0, 0.0])

    initial_states = {
        "body1": np.concatenate([r1, v1]),
        "body2": np.concatenate([r2, v2]),
        "body3": np.concatenate([r3, v3]),
    }

    prop = NBodyPropagator(
        bodies=["body1", "body2", "body3"],
        mu_list=[mu1, mu2, mu3],
        initial_states_dict=initial_states,
        epoch_tdb=0.0,
    )

    # Propagate a short time
    t_tdb = 0.01
    state3 = prop.get_body_state("body3", t_tdb)

    # Basic shape check
    assert state3.shape == (6,)
    assert isinstance(state3, np.ndarray)

    # The state should have changed (the body is under gravity)
    assert not np.allclose(state3, initial_states["body3"],
                           rtol=1e-12, atol=1e-12), \
        "Third body state did not change at all"
