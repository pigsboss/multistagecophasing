"""
MCPC Framework - Multi-stage Co-Phasing Control
Industrial-grade spacecraft dynamics and control simulation framework.
"""
from .core.spacetime.ids import CoordinateFrame, Telecommand
from .utils.logger import HDF5Logger, SimulationMetadata
from .utils.math_tools import get_lqr_gain, absolute_to_lvlh, elements_to_cartesian

# Simulation controllers
from .simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from .simulation.twobody.leo import LEOL1Simulation
from .simulation.twobody.geo import GEOL1Simulation

__all__ = [
    # Core types
    "CoordinateFrame",
    "Telecommand",
    # Utilities
    "HDF5Logger",
    "SimulationMetadata",
    "get_lqr_gain",
    "absolute_to_lvlh",
    "elements_to_cartesian",
    # Simulations
    "SunEarthL2L1Simulation",
    "LEOL1Simulation",
    "GEOL1Simulation",
]
