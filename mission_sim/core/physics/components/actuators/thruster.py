"""
MCPC Core Physics: Thruster Actuator
------------------------------------
Simulates the physical propulsion hardware of a spacecraft.
Acts as the boundary where Cyber domain GNC commands become objective physical forces,
introducing hardware limitations such as saturation (max thrust) and execution noise.
"""

import numpy as np
from typing import Tuple

class Thruster:
    """
    [L2-SPECIFIC / ACTUATOR] Continuous Thrust Propulsion Hardware.
    Processes ideal control commands into realistic physical force vectors,
    accounting for hardware constraints and calculating fuel consumption.
    """

    def __init__(
        self, 
        max_thrust_n: float = 1.0,           # Maximum thrust capacity (Newtons)
        min_thrust_n: float = 0.0,           # Minimum thrust (deadband)
        noise_std_n: float = 0.005,          # 5 mN execution noise (1-sigma)
        specific_impulse_s: float = 3000.0   # Specific Impulse (Isp) for electric propulsion (Seconds)
    ):
        """
        Initialize the physical constraints of the thruster hardware.
        
        Args:
            max_thrust_n: The absolute maximum force the thruster can output (Saturation limit).
            min_thrust_n: The minimum force required to turn the thruster on (Deadband limit).
            noise_std_n: Standard deviation of the Gaussian noise added to the thrust output.
            specific_impulse_s: Efficiency of the thruster, used to calculate mass depletion.
        """
        self.max_thrust_n = float(max_thrust_n)
        self.min_thrust_n = float(min_thrust_n)
        self.noise_std_n = float(noise_std_n)
        self.isp = float(specific_impulse_s)
        self.g0 = 9.80665  # Standard gravity (m/s^2) for Isp calculations

    def execute(self, commanded_force: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        [CYBER -> PHYSICS]
        Transform an ideal GNC commanded force into an actual physical force 
        and calculate the instantaneous fuel consumption rate.
        
        Args:
            commanded_force (np.ndarray): 1D array of shape (3,) representing the 
                                          ideal force requested by the GNC [Fx, Fy, Fz] (N).
                                          
        Returns:
            Tuple[np.ndarray, float]:
                - Actual applied force vector [Fx, Fy, Fz] (N) after saturation and noise.
                - Mass flow rate (kg/s) representing fuel consumed during this step.
        """
        # 1. Compute commanded magnitude and direction
        commanded_mag = np.linalg.norm(commanded_force)
        
        if commanded_mag < 1e-12:
            return np.zeros(3, dtype=np.float64), 0.0
            
        direction = commanded_force / commanded_mag

        # 2. Hardware Saturation (Clipping)
        # If the command exceeds hardware limits, thrust operates at maximum capacity
        if commanded_mag > self.max_thrust_n:
            actual_mag = self.max_thrust_n
        # Deadband limit: If the command is too small, the thruster might not fire
        elif commanded_mag < self.min_thrust_n:
            return np.zeros(3, dtype=np.float64), 0.0
        else:
            actual_mag = commanded_mag

        # 3. Inject Execution Noise (Gaussian)
        # Noise is added to the magnitude, but we ensure thrust never goes negative
        noisy_mag = actual_mag + np.random.normal(0.0, self.noise_std_n)
        noisy_mag = max(0.0, noisy_mag)
        noisy_mag = min(self.max_thrust_n, noisy_mag) # Re-saturate if noise pushes it over

        # 4. Construct the Actual Applied Force Vector
        actual_force = noisy_mag * direction

        # 5. Calculate Mass Flow Rate (dm/dt = F / (Isp * g0))
        mass_flow_rate = noisy_mag / (self.isp * self.g0)

        return actual_force, mass_flow_rate

    def __repr__(self) -> str:
        return (f"Thruster(Max={self.max_thrust_n}N, Noise={self.noise_std_n}N, "
                f"Isp={self.isp}s)")
