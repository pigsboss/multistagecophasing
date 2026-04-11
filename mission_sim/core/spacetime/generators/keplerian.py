# mission_sim/core/trajectory/generators/keplerian.py
"""Keplerian orbit generator (two-body analytical solution)"""

import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator
from mission_sim.utils.math_tools import elements_to_cartesian


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    Keplerian orbit generator (two-body analytical solution).
    Generates reference ephemeris using classical Kepler formulas, no perturbations.
    
    This is a simplified model for L1-level baseline calibration and ideal nominal
    orbit generation. It does not integrate with high-precision ephemeris.
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        Initialize the generator.

        Args:
            mu: Gravitational parameter of central body (m³/s²), default Earth.
        """
        # Note: KeplerianGenerator is a simplified model and does not use
        # high-precision ephemeris. The ephemeris and use_high_precision parameters
        # from BaseTrajectoryGenerator are intentionally ignored.
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        Generate ephemeris from orbital elements.

        config must contain:
            - elements: [a, e, i, Omega, omega, M0] orbital elements
            - dt: time step (s)
            - sim_time: simulation duration (s)

        Returns:
            Ephemeris object (J2000_ECI frame)
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("KeplerianGenerator requires 6 orbital elements 'elements' in config.")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        times = np.arange(0, sim_time + dt, dt)

        a, e, i, Omega, omega, M0 = elements
        n = np.sqrt(self.mu / a**3)

        states = []
        for t in times:
            # Mean anomaly
            M = M0 + n * t
            # Solve Kepler's equation M = E - e sin(E) (Newton iteration)
            E = self._kepler_solver(M, e)
            # True anomaly
            nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
            # In-plane coordinates
            r = a * (1 - e * np.cos(E))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            vx_orb = -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu)
            vy_orb = np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu))

            # Use full six-element conversion function
            state_eci = elements_to_cartesian(self.mu, a, e, i, Omega, omega, M)
            states.append(state_eci)

        return Ephemeris(times, np.array(states), CoordinateFrame.J2000_ECI)

    def _kepler_solver(self, M: float, e: float, tol: float = 1e-12) -> float:
        """Solve Kepler's equation M = E - e sin(E) (Newton iteration)"""
        E = M if e < 0.8 else np.pi  # Initial guess
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        return E
