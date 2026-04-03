# mission_sim/core/physics/spacecraft.py
"""
MCPC Core Physics: Spacecraft Point Mass Model (L1)
---------------------------------------------------
Fundamental L1 spacecraft model: point mass, ideal thrust, ΔV accumulation.
Strictly enforces coordinate frame contracts.

This is the L1 legacy model. L2 formation simulations should use SpacecraftL2
which composes this model with hardware components.
"""

import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame


class SpacecraftPointMass:
    """
    Spacecraft point mass model for L1 absolute dynamics.
    Maintains position, velocity, mass, and accumulates ΔV.
    Follows physical domain contracts: state + frame.

    Attributes:
        id (str): Spacecraft identifier
        state (np.ndarray): Current state [x, y, z, vx, vy, vz] (SI)
        frame (CoordinateFrame): Frame of the state (strong contract)
        mass (float): Current total mass (kg)
        accumulated_dv (float): Total ΔV consumed (m/s)
        external_accel (np.ndarray): Accumulated thrust acceleration (m/s²)
        consumed_fuel (float): Total fuel mass consumed (kg)
    """

    def __init__(
        self,
        sc_id: str,
        initial_state: np.ndarray,
        frame: CoordinateFrame,
        initial_mass: float = 1000.0
    ):
        """
        Initialize the spacecraft point mass.

        Args:
            sc_id: Unique spacecraft identifier
            initial_state: [x, y, z, vx, vy, vz] (SI units)
            frame: Coordinate frame of initial_state (strong contract)
            initial_mass: Initial mass (kg)
        """
        self.id = sc_id
        self.state = np.array(initial_state, dtype=np.float64)
        self.frame = frame
        self.mass = float(initial_mass)

        # Physical accumulators
        self.external_accel = np.zeros(3, dtype=np.float64)
        self.consumed_fuel = 0.0
        self.accumulated_dv = 0.0

    @property
    def position(self) -> np.ndarray:
        """Current position vector (3,)"""
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity vector (3,)"""
        return self.state[3:6]

    def apply_thrust(self, force_vector: np.ndarray, force_frame: CoordinateFrame) -> None:
        """
        Apply a thrust force (information domain → physical domain).
        Enforces coordinate frame consistency.

        Args:
            force_vector: Thrust force [Fx, Fy, Fz] (N)
            force_frame: Frame in which the force vector is expressed

        Raises:
            ValueError: If force_frame does not match spacecraft frame
        """
        if force_frame != self.frame:
            raise ValueError(
                f"[{self.id} PHYSICS REJECT] Thrust frame mismatch! "
                f"Spacecraft is in {self.frame.name}, "
                f"but GNC thrust is in {force_frame.name}."
            )
        accel = np.array(force_vector, dtype=np.float64) / self.mass
        self.external_accel += accel

    def get_derivative(self, gravity_accel: np.ndarray, gravity_frame: CoordinateFrame) -> np.ndarray:
        """
        Compute state derivative for numerical integration.

        Args:
            gravity_accel: Total environmental acceleration [ax, ay, az] (m/s²)
            gravity_frame: Frame of the environmental acceleration

        Returns:
            State derivative [vx, vy, vz, ax_total, ay_total, az_total]

        Raises:
            ValueError: If gravity_frame does not match spacecraft frame
        """
        if gravity_frame != self.frame:
            raise ValueError(
                f"[{self.id} PHYSICS FATAL] Dynamics frame conflict! "
                f"Spacecraft is in {self.frame.name}, "
                f"environment acceleration in {gravity_frame.name}."
            )
        v = self.velocity
        a_total = np.array(gravity_accel, dtype=np.float64) + self.external_accel
        return np.concatenate([v, a_total])

    def integrate_dv(self, dt: float) -> None:
        """
        Accumulate ΔV from current thrust acceleration.
        Must be called after apply_thrust and before clear_thrust.

        Args:
            dt: Integration time step (s)
        """
        self.accumulated_dv += np.linalg.norm(self.external_accel) * dt

    def clear_thrust(self) -> None:
        """Reset thrust acceleration accumulator after integration."""
        self.external_accel = np.zeros(3, dtype=np.float64)

    def consume_mass(self, m_dot: float, dt: float) -> None:
        """
        Update mass due to propellant consumption.

        Args:
            m_dot: Mass flow rate (kg/s)
            dt: Time step (s)
        """
        dm = m_dot * dt
        self.mass -= dm
        self.consumed_fuel += dm

    def __repr__(self) -> str:
        return (
            f"SpacecraftPointMass[{self.id}] | Frame: {self.frame.name} | "
            f"Mass: {self.mass:.1f}kg | ΔV: {self.accumulated_dv:.4f} m/s"
        )
