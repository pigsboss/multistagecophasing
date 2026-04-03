# mission_sim/core/physics/spacecraft_node.py
"""
MCPC L2 Physics: Spacecraft Node for Formation Simulation
----------------------------------------------------------
A self-contained spacecraft node for formation flying, including:
- State (position, velocity, mass, ΔV accumulation)
- Thruster actuator (saturation, deadband, noise, MIB)
- ISL antenna sensor (range/angle noise, signal attenuation)
- Optional network router for transmitting measurements

This class does NOT depend on SpacecraftPointMass, making it robust
and self-contained for L2 simulations.
"""

import numpy as np
from typing import Optional
from mission_sim.core.spacetime.ids import CoordinateFrame, Telecommand
from mission_sim.core.physics.components.actuators.thruster import Thruster
from mission_sim.core.physics.components.sensors.isl_antenna import ISLAntenna
from mission_sim.core.cyber.network.isl_router import ISLRouter
from mission_sim.core.physics.ids import SpacecraftType, MicrowaveISLMeasurement


class SpacecraftNode:
    """
    L2 integrated spacecraft node for formation flying.
    Self-contained: includes state, actuators, sensors, and router.
    """

    def __init__(
        self,
        sc_id: str,
        initial_state: np.ndarray,
        frame: CoordinateFrame,
        initial_mass: float = 1000.0,
        sc_type: SpacecraftType = SpacecraftType.DEPUTY,
        thruster: Optional[Thruster] = None,
        antenna: Optional[ISLAntenna] = None,
        router: Optional[ISLRouter] = None,
    ):
        """
        Initialize a spacecraft node.

        Args:
            sc_id: Unique identifier
            initial_state: [x, y, z, vx, vy, vz] (SI)
            frame: Coordinate frame of the initial state
            initial_mass: Total initial mass (kg)
            sc_type: CHIEF or DEPUTY
            thruster: Thruster hardware (creates default if None)
            antenna: ISL antenna hardware (creates default if None)
            router: Network router for transmitting measurements
        """
        self.id = sc_id
        self.state = np.array(initial_state, dtype=np.float64)
        self.frame = frame
        self.mass = float(initial_mass)
        self.sc_type = sc_type
        self.router = router

        # Hardware components (use defaults if not provided)
        self.thruster = thruster if thruster is not None else Thruster()
        self.antenna = antenna if antenna is not None else ISLAntenna()

        # L2 specific state
        self.last_control_force = np.zeros(3, dtype=np.float64)
        self._current_mass_flow_rate = 0.0

        # Internal accumulators (for compatibility with L1 integrator)
        self.accumulated_dv = 0.0
        self.external_accel = np.zeros(3, dtype=np.float64)

    # -----------------------------------------------------------------
    # Core state properties (L1 compatible)
    # -----------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def apply_thrust(self, force_vector: np.ndarray, force_frame: CoordinateFrame) -> None:
        """
        Apply thrust force directly (L1 compatibility).
        For L2 control, use apply_control() instead.
        """
        if force_frame != self.frame:
            raise ValueError(
                f"[{self.id}] Thrust frame mismatch: expected {self.frame.name}, got {force_frame.name}"
            )
        accel = force_vector / self.mass
        self.external_accel += accel

    def get_derivative(self, gravity_accel: np.ndarray, gravity_frame: CoordinateFrame) -> np.ndarray:
        """
        Compute state derivative for numerical integration.
        gravity_accel: total environmental acceleration (m/s²)
        gravity_frame: must match self.frame
        """
        if gravity_frame != self.frame:
            raise ValueError(
                f"[{self.id}] Gravity frame mismatch: expected {self.frame.name}, got {gravity_frame.name}"
            )
        total_acc = gravity_accel + self.external_accel
        return np.concatenate([self.velocity, total_acc])

    def integrate_dv(self, dt: float) -> None:
        """Accumulate ΔV from current thrust."""
        self.accumulated_dv += np.linalg.norm(self.external_accel) * dt

    def clear_thrust(self) -> None:
        """Reset thrust accumulator after integration."""
        self.external_accel = np.zeros(3, dtype=np.float64)

    def consume_mass(self, m_dot: float, dt: float) -> None:
        """Update mass due to propellant consumption."""
        dm = m_dot * dt
        self.mass -= dm
        self.mass = max(0.1, self.mass)  # prevent negative mass

    # -----------------------------------------------------------------
    # L2-specific hardware interaction
    # -----------------------------------------------------------------
    def apply_control(self, commanded_force: np.ndarray) -> None:
        """
        Apply a GNC command through the thruster hardware model.
        Updates external_accel and mass flow rate.
        """
        actual_force, mass_flow_rate = self.thruster.execute(commanded_force)
        self.last_control_force = actual_force.copy()
        self.external_accel += actual_force / self.mass
        self._current_mass_flow_rate = mass_flow_rate

    def update_mass(self, dt: float) -> None:
        """Update mass based on fuel consumption from last control step."""
        if self._current_mass_flow_rate != 0.0:
            self.consume_mass(self._current_mass_flow_rate, dt)
            self._current_mass_flow_rate = 0.0

    def sense(self, target_node: 'SpacecraftNode', current_time: float) -> Optional[MicrowaveISLMeasurement]:
        """
        Generate a physical measurement of another node.
        """
        rel_pos = target_node.position - self.position
        return self.antenna.measure(rel_pos, current_time)

    def transmit(self, measurement: MicrowaveISLMeasurement, dest_id: str, current_time: float):
        """
        Transmit a measurement through the network router (if set).
        """
        if self.router is None:
            return None
        return self.router.transmit(measurement, self.id, dest_id, current_time)

    def __repr__(self) -> str:
        return (
            f"SpacecraftNode[{self.id}] | Type={self.sc_type.name} | "
            f"Frame={self.frame.name} | Mass={self.mass:.1f}kg | ΔV={self.accumulated_dv:.4f}m/s"
        )