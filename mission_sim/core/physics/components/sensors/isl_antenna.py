"""
MCPC Core Physics: ISL Antenna Sensor
-------------------------------------
Simulates the hardware physics of an Inter-Satellite Link (ISL) microwave antenna.
Acts as the boundary between objective physical truth and imperfect sensor perception.
"""

import numpy as np
from typing import Optional
from mission_sim.core.physics.ids import MicrowaveISLMeasurement

class ISLAntenna:
    """
    [L2-SPECIFIC / SENSOR] Microwave Inter-Satellite Link Antenna.
    Converts perfect relative Cartesian vectors into noisy spherical measurements.
    """

    def __init__(
        self, 
        range_noise_std: float = 0.05,       # 5 cm distance noise (1-sigma)
        angle_noise_std: float = 5e-5,       # ~10 arcseconds angle noise (1-sigma)
        max_range_m: float = 200000.0,       # 200 km maximum communication range
        reference_range_m: float = 1000.0    # Range at which signal strength is 1.0 (normalized)
    ):
        """
        Initialize the physical parameters of the ISL Antenna.
        
        Args:
            range_noise_std: Standard deviation of range measurement noise (meters).
            angle_noise_std: Standard deviation of angular measurement noise (radians).
            max_range_m: Maximum effective measurement distance. Beyond this, signal drops.
            reference_range_m: Distance used to normalize signal strength attenuation.
        """
        self.range_noise_std = float(range_noise_std)
        self.angle_noise_std = float(angle_noise_std)
        self.max_range_m = float(max_range_m)
        self.reference_range_m = float(reference_range_m)

    def measure(self, true_rel_pos: np.ndarray, current_time: float) -> Optional[MicrowaveISLMeasurement]:
        """
        [PHYSICS -> MEASUREMENT]
        Generate a physical measurement payload based on the objective true relative position.
        
        Args:
            true_rel_pos (np.ndarray): 1D array of shape (3,) representing [x, y, z] 
                                       true relative position in the sensor's native frame.
            current_time (float): The exact physical epoch of the measurement.
            
        Returns:
            MicrowaveISLMeasurement: The noisy physical payload, or None if the target 
                                     is out of the antenna's effective range.
        """
        # 1. Compute True Spherical Coordinates
        range_true = np.linalg.norm(true_rel_pos)
        
        # Hardware limits: Signal lost if too far, math error if perfectly coincident
        if range_true > self.max_range_m or range_true < 1e-6:
            return None

        # Azimuth in XY plane, Elevation from XY plane
        azimuth_true = np.arctan2(true_rel_pos[1], true_rel_pos[0])
        elevation_true = np.arcsin(true_rel_pos[2] / range_true)

        # 2. Inject Hardware Noise (Gaussian distribution)
        range_meas = range_true + np.random.normal(0.0, self.range_noise_std)
        azimuth_meas = azimuth_true + np.random.normal(0.0, self.angle_noise_std)
        elevation_meas = elevation_true + np.random.normal(0.0, self.angle_noise_std)

        # 3. Compute Signal Attenuation (Inverse-Square Law Approximation)
        # Normalized mapping: [0.0 to 1.0]
        if range_true <= self.reference_range_m:
            signal_strength = 1.0
        else:
            signal_strength = (self.reference_range_m / range_true) ** 2
            
        # 4. Construct the pure physical payload (No network routing info yet)
        return MicrowaveISLMeasurement(
            phys_timestamp=float(current_time),
            range_m=float(range_meas),
            azimuth_rad=float(azimuth_meas),
            elevation_rad=float(elevation_meas),
            signal_strength=float(signal_strength)
        )

    def __repr__(self) -> str:
        return (f"ISLAntenna(RangeStd={self.range_noise_std}m, "
                f"AngleStd={self.angle_noise_std}rad, MaxRange={self.max_range_m}m)")
