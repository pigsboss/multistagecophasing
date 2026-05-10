"""
N-body backup ephemeris (analytical) and helper classes.

Provides a pure‑Numba N‑body propagator and a lightweight
:class:`Ephemeris` wrapper that can be used as a fallback when
SPICE ephemeris data are unavailable.

The core integration relies on the Dormand‑Prince 5(4) integrator
from :mod:`mission_sim.utils.propagators.base`.
"""

import numpy as np
from numba import njit

from mission_sim.core.spacetime.time import utc_smooth_to_tdb
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris.base import Ephemeris
from mission_sim.utils.propagators.base import integrate_dp8


# ---------------------------------------------------------------------------
# N‑body acceleration kernel  (Numba‑compiled)
# ---------------------------------------------------------------------------

@njit
def _nbody_derivs(t, Y, mu_arr, n_body):
    """
    N‑body gravitational acceleration.

    Parameters
    ----------
    t : float
        Time (unused, but required for the integrator interface).
    Y : ndarray, shape (6 * n_body,)
        Concatenated state vector: for each body i,
        (x_i, y_i, z_i, vx_i, vy_i, vz_i).
    mu_arr : ndarray, shape (n_body,)
        Standard gravitational parameter (GM) for each body.
    n_body : int
        Number of bodies.

    Returns
    -------
    dYdt : ndarray, shape (6 * n_body,)
        Time derivative of the state vector.
    """
    dY = np.zeros_like(Y)
    for i in range(n_body):
        base_i = 6 * i
        pos_i = Y[base_i:base_i + 3]

        # position derivative = velocity
        dY[base_i:base_i + 3] = Y[base_i + 3:base_i + 6]

        # acceleration from every other body
        acc = np.zeros(3, dtype=np.float64)
        for j in range(n_body):
            if i == j:
                continue
            base_j = 6 * j
            pos_j = Y[base_j:base_j + 3]
            r_ij = pos_j - pos_i
            r_norm = np.sqrt(np.sum(r_ij * r_ij))
            # avoid division by zero (should never happen for distinct bodies)
            if r_norm > 0.0:
                acc += mu_arr[j] * r_ij / (r_norm * r_norm * r_norm)

        dY[base_i + 3:base_i + 6] = acc

    return dY


# ---------------------------------------------------------------------------
# N‑body propagator  (convenience class)
# ---------------------------------------------------------------------------

class NBodyPropagator:
    """
    N‑body gravitational propagator using direct integration.

    Maintains an integrated state vector of *n* bodies and can be advanced
    to arbitrary TDB times.

    Parameters
    ----------
    bodies : list of str
        Ordered list of body names (used for indexing).
    mu_list : list of float
        Standard gravitational parameter (GM) for each body, same order
        as *bodies*.
    initial_states_dict : dict[str, ndarray]
        Initial state (6,) per body name, in the J2000 equatorial frame.
    epoch_tdb : float
        TDB seconds since J2000.0 corresponding to the initial states.
    rtol, atol : float
        Relative and absolute tolerances for the embedded Runge‑Kutta
        integrator.
    """

    def __init__(self, bodies, mu_list, initial_states_dict, epoch_tdb,
                 rtol=1e-12, atol=1e-12):
        self.bodies = list(bodies)
        self.mu_arr = np.array(mu_list, dtype=np.float64)
        self.n_body = len(self.bodies)

        # Assemble full state vector
        self.Y = np.empty(6 * self.n_body, dtype=np.float64)
        for idx, name in enumerate(self.bodies):
            self.Y[6 * idx:6 * idx + 6] = initial_states_dict[name]

        self.t_current = float(epoch_tdb)
        self.rtol = rtol
        self.atol = atol

    def propagate_to(self, t_tdb: float) -> None:
        """
        Advance the internal state to the requested TDB time.

        If ``t_tdb`` is within 1 ps of the current time, no integration
        is performed.
        """
        if abs(t_tdb - self.t_current) < 1e-16:
            return

        t_arr, Y_arr = integrate_dp8(
            _nbody_derivs,
            self.t_current,
            self.Y,
            (self.t_current, t_tdb),
            rtol=self.rtol,
            atol=self.atol,
            args=(self.mu_arr, self.n_body),
        )
        self.Y[:] = Y_arr[-1]
        self.t_current = t_arr[-1]

    def get_body_state(self, body_name: str, t_tdb: float) -> np.ndarray:
        """
        Return the state of a single body at the specified TDB time.

        Parameters
        ----------
        body_name : str
            Name of the body (must be one of the bodies passed at
            construction).
        t_tdb : float
            Target TDB time (seconds since J2000.0).

        Returns
        -------
        state : ndarray, shape (6,)
            J2000 equatorial state vector (x, y, z, vx, vy, vz).
        """
        self.propagate_to(t_tdb)
        idx = self.bodies.index(body_name)
        return self.Y[6 * idx:6 * idx + 6].copy()


# ---------------------------------------------------------------------------
# Ephemeris wrapper for the N‑body propagator
# ---------------------------------------------------------------------------

class NBodyTargetEphemeris(Ephemeris):
    """
    An :class:`Ephemeris` that wraps an :class:`NBodyPropagator` and
    provides the standard ``get_state(utc_smooth_sec)`` interface.

    The underlying state is **not** pre‑computed at discrete times;
    instead each call triggers a forward integration to the requested
    epoch.  This makes the object CPU‑heavy but memory‑light and
    suitable for scenario‑specific queries.

    Parameters
    ----------
    propagator : NBodyPropagator
        Fully initialised N‑body propagator.
    target_body : str
        Name of the body for which states are queried.
    """

    def __init__(self, propagator: NBodyPropagator, target_body: str):
        super().__init__(None, None, CoordinateFrame.J2000_ECI)
        self._prop = propagator
        self._target = target_body

    def get_state(self, utc_smooth_sec: float) -> np.ndarray:
        """
        Return the J2000 equatorial state of the target body at the given
        smooth‑UTC epoch.

        Parameters
        ----------
        utc_smooth_sec : float
            Smooth UTC seconds since J2000.0 UTC.

        Returns
        -------
        state : ndarray, shape (6,)
        """
        tdb = utc_smooth_to_tdb(utc_smooth_sec)
        return self._prop.get_body_state(self._target, tdb)
