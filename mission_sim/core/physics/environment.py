# mission_sim/core/physics/environment.py
"""
MCPC Core Physics: Celestial Environment & Dispatch Engine
---------------------------------------------------------
Manages force models and provides acceleration computations for spacecraft.
Upgraded to support L2 Multi-Spacecraft Parallelization using NumPy vectorized operations
while maintaining full backward compatibility with L1 legacy interfaces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from mission_sim.core.spacetime.ids import CoordinateFrame


class IForceModel(ABC):
    """
    [ABSTRACT / MCPC UNIVERSAL] Force Model Interface (Strategy Pattern)
    All specific force models (Gravity, SRP, etc.) must implement this interface.
    Ensures the environment can dynamically load any perturbation model.
    """

    @abstractmethod
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        [L1 LEGACY & FALLBACK]
        Compute physical acceleration for a single state and time.

        Args:
            state: Spacecraft state vector [x, y, z, vx, vy, vz] (SI units)
            epoch: Current simulation epoch (s)

        Returns:
            np.ndarray: Acceleration vector [ax, ay, az] (m/s²)
        """
        pass

    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """
        [L2-SPECIFIC / PARALLELIZATION]
        Batch compute accelerations for multiple spacecraft simultaneously.
        
        Subclasses (e.g., CRTBPGravity) should override this with high-performance 
        NumPy vectorized logic. If not overridden, it defaults to a sequential 
        loop calling 'compute_accel'.
        
        Args:
            state_matrix (np.ndarray): 2D array of shape (N, 6).
            epoch (float): Current simulation epoch (s).
            
        Returns:
            np.ndarray: 2D array of shape (N, 3) containing acceleration vectors.
        """
        if state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
            raise ValueError(f"state_matrix must be of shape (N, 6), got {state_matrix.shape}")

        num_sc = state_matrix.shape[0]
        acc_matrix = np.zeros((num_sc, 3), dtype=np.float64)

        # Fallback to sequential loop if vectorized version is not implemented
        for i in range(num_sc):
            acc_matrix[i, :] = self.compute_accel(state_matrix[i, :], epoch)

        return acc_matrix


class CelestialEnvironment:
    """
    [MCPC UNIVERSAL] Celestial Environment Manager
    Acts as a force registry, managing multiple perturbation models and 
    providing unified total acceleration for single or multiple spacecraft.
    """

    def __init__(self, computation_frame: CoordinateFrame, initial_epoch: float = 0.0, verbose: bool = True):
        """
        Initialize the environment engine.

        Args:
            computation_frame: The base CoordinateFrame for physical calculations.
            initial_epoch: Initial simulation epoch (s).
            verbose: Whether to print detailed registration information.
        """
        self.computation_frame = computation_frame
        self.epoch = float(initial_epoch)
        self.verbose = verbose

        # Core force model registry
        self._force_registry: List[IForceModel] = []

    def register_force(self, force_model: IForceModel) -> None:
        """
        Dynamically inject a force model into the registry.

        Args:
            force_model: Instance implementing the IForceModel interface.

        Raises:
            TypeError: If the model does not implement IForceModel.
        """
        if not isinstance(force_model, IForceModel):
            raise TypeError("Registered force models must implement the IForceModel interface.")

        self._force_registry.append(force_model)
        if self.verbose:
            print(f"[Environment] Successfully registered force model: {force_model.__class__.__name__}")

    def step_time(self, dt: float) -> None:
        """
        Advance the environment epoch.

        Args:
            dt: Time step (s).
        """
        self.epoch += dt

    # ---------------------------------------------------------
    # Legacy L1 Support
    # ---------------------------------------------------------

    def get_total_acceleration(
        self, sc_state: np.ndarray, sc_frame: CoordinateFrame
    ) -> Tuple[np.ndarray, CoordinateFrame]:
        """
        [L1 LEGACY] Compute total acceleration for a single spacecraft.
        Enforces coordinate frame consistency checks.

        Args:
            sc_state: Spacecraft state vector [x, y, z, vx, vy, vz].
            sc_frame: The coordinate frame of the input state.

        Returns:
            Tuple[np.ndarray, CoordinateFrame]:
                - Total acceleration vector [ax, ay, az] (m/s²).
                - The computation frame of the environment.

        Raises:
            ValueError: If sc_frame does not match the computation_frame.
        """
        if sc_frame != self.computation_frame:
            raise ValueError(
                f"[Environment Error] Frame mismatch! Environment is running in {self.computation_frame.name}, "
                f"but received state in {sc_frame.name}."
            )

        # Bridge to the vectorized pipeline for logic consistency
        state_matrix = sc_state.reshape(1, 6)
        total_acc_matrix = self.compute_accelerations(state_matrix)
        
        return total_acc_matrix[0], self.computation_frame

    # ---------------------------------------------------------
    # New L2 Vectorized Pipeline
    # ---------------------------------------------------------

    def compute_accelerations(self, state_matrix: np.ndarray) -> np.ndarray:
        """
        [L2-SPECIFIC / PARALLELIZATION]
        Batch compute the sum of accelerations from all registered force models.
        
        Args:
            state_matrix: 2D array of shape (N, 6) representing N spacecraft states.
            
        Returns:
            np.ndarray: 2D array of shape (N, 3) representing [ax, ay, az] for each spacecraft.
        """
        if not isinstance(state_matrix, np.ndarray) or state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
             raise ValueError("state_matrix must be a 2D numpy array of shape (N, 6)")

        num_sc = state_matrix.shape[0]
        total_acc_matrix = np.zeros((num_sc, 3), dtype=np.float64)
        
        # Accumulate contributions from each force model using the vectorized interface
        for force in self._force_registry:
            total_acc_matrix += force.compute_vectorized_acc(state_matrix, self.epoch)
            
        return total_acc_matrix

    def __repr__(self) -> str:
        forces = [f.__class__.__name__ for f in self._force_registry]
        return (
            f"CelestialEnvironment | Frame={self.computation_frame.name} | "
            f"Epoch={self.epoch:.1f}s | ActiveForces={forces}"
        )
