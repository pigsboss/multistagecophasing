"""
MCPC Physical Force Model Base Contract
---------------------------------------
Defines the universal interface for all environmental force models (e.g., Gravity, SRP).
Provides backward compatibility for L1 single-satellite simulation while 
enforcing the vectorized matrix contract required for L2 multi-satellite parallelization.
"""

from abc import ABC, abstractmethod
import numpy as np

class ForceModel(ABC):
    """
    [ABSTRACT / MCPC UNIVERSAL] 
    The fundamental blueprint for any physical force acting on a spacecraft.
    """

    @abstractmethod
    def compute_acceleration(self, state: np.ndarray, timestamp: float) -> np.ndarray:
        """
        [L1 LEGACY & FALLBACK]
        Compute the acceleration for a SINGLE spacecraft.
        
        Args:
            state (np.ndarray): 1D array of shape (6,) representing [x, y, z, vx, vy, vz].
            timestamp (float): Current simulation epoch in seconds.
            
        Returns:
            np.ndarray: 1D array of shape (3,) representing [ax, ay, az].
        """
        pass

    def compute_vectorized_acc(self, state_matrix: np.ndarray, timestamp: float) -> np.ndarray:
        """
        [L2-SPECIFIC / PARALLELIZATION]
        Compute accelerations for MULTIPLE spacecraft simultaneously.
        
        This method is designed to be overridden by subclasses (like GravityCRTBP)
        with high-performance numpy vectorized operations. If not overridden, 
        it falls back to a sequential loop using `compute_acceleration`.
        
        Args:
            state_matrix (np.ndarray): 2D array of shape (N, 6) where N is the number of spacecraft.
            timestamp (float): Current simulation epoch in seconds.
            
        Returns:
            np.ndarray: 2D array of shape (N, 3) containing the acceleration vectors [ax, ay, az] for each spacecraft.
        """
        # Input shape validation to prevent silent broadcasting bugs
        if state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
            raise ValueError(f"state_matrix must be of shape (N, 6), got {state_matrix.shape}")

        num_sc = state_matrix.shape[0]
        acc_matrix = np.zeros((num_sc, 3), dtype=np.float64)

        # Fallback mechanism: Sequential loop over all spacecraft
        # WARNING: Subclasses should override this to avoid performance bottlenecks in L2+
        for i in range(num_sc):
            acc_matrix[i, :] = self.compute_acceleration(state_matrix[i, :], timestamp)

        return acc_matrix
