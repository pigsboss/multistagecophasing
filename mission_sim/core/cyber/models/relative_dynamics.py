"""
MCPC Cyber Domain: Relative Dynamics Base Class
-----------------------------------------------
Defines the interface for relative motion models used by the Formation Controller.
Supports discrete-time state prediction for delay compensation.
"""

from abc import ABC, abstractmethod
import numpy as np


class RelativeDynamics(ABC):
    """
    Abstract base class for relative motion dynamics.
    Provides discrete-time state transition matrix (STM) for prediction.
    """

    @abstractmethod
    def compute_discrete_stm(self, dt: float) -> np.ndarray:
        """
        Compute the discrete-time state transition matrix (6x6) for a given time step.

        The STM maps the relative state [dr, dv] from time t to t+dt:
            x(t+dt) = Φ(dt) * x(t)

        For time-invariant linear systems (e.g., CW equations for circular orbits),
        the STM can be precomputed. For time-varying systems, it should be
        computed at each call based on the current reference state.

        Args:
            dt: Time step (seconds)

        Returns:
            Φ: 6x6 state transition matrix (float64)
        """
        pass

    @abstractmethod
    def predict_state(self, current_state: np.ndarray, stm: np.ndarray) -> np.ndarray:
        """
        Predict the relative state after applying the state transition matrix.

        Args:
            current_state: Current relative state [dx, dy, dz, dvx, dvy, dvz]
            stm: State transition matrix (6x6)

        Returns:
            Predicted relative state (6,)
        """
        return stm @ current_state