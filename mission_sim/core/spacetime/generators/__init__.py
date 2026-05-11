# mission_sim/core/trajectory/generators/__init__.py
"""
Spatiotemporal nominal orbit generator factory.
Supports high-precision ephemeris integration, can generate reference orbits based on precise celestial positions.
"""

from .base import BaseTrajectoryGenerator
from .keplerian import KeplerianGenerator
from .j2_keplerian import J2KeplerianGenerator
from .halo import HaloDifferentialCorrector
from .crtbp import CRTBPOrbitGenerator, CRTBPOrbitType, SymmetryType, CRTBPOrbitConfig, create_crtbp_generator, generate_family
# from .attitude import NumericalAttitudeGenerator, create_numerical_attitude_generator

__all__ = [
    "BaseTrajectoryGenerator",
    "KeplerianGenerator",
    "J2KeplerianGenerator",
    "HaloDifferentialCorrector",
    "CRTBPOrbitGenerator",
    "CRTBPOrbitType",
    "SymmetryType",
    "CRTBPOrbitConfig",
    "create_crtbp_generator",
    "generate_family",
    "create_generator",
    "create_generator_with_ephemeris",
    "create_high_precision_generator",
    # "NumericalAttitudeGenerator",
    # "create_numerical_attitude_generator",
]

def create_generator(orbit_type: str, **kwargs) -> BaseTrajectoryGenerator:
    """Factory function: create corresponding generator instance based on orbit type"""
    orbit_type = orbit_type.lower()
    if orbit_type == "keplerian":
        return KeplerianGenerator(**kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(**kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(**kwargs)
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        # For CRTBP orbit types, use the new CRTBPOrbitGenerator
        # Need to convert string to CRTBPOrbitType enum
        if orbit_type == "crtbp":
            # Default to HALO
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        # Extract system type from kwargs, default to sun_earth
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown orbit type: {orbit_type}")


def create_generator_with_ephemeris(orbit_type: str, ephemeris, **kwargs) -> BaseTrajectoryGenerator:
    """
    Factory function: create generator instance with ephemeris.
    
    Args:
        orbit_type: Orbit type
        ephemeris: Ephemeris instance (HighPrecisionEphemeris or SPICEInterface)
        **kwargs: Generator-specific parameters
        
    Returns:
        BaseTrajectoryGenerator: Generator instance
    """
    orbit_type = orbit_type.lower()
    
    # Keplerian orbit generators don't need high-precision ephemeris
    if orbit_type in ["keplerian", "j2_keplerian"]:
        import warnings
        warnings.warn(
            f"{orbit_type} generator doesn't need high-precision ephemeris, ignoring ephemeris parameter",
            UserWarning
        )
        # Remove ephemeris parameter to avoid passing to generator
        if 'ephemeris' in kwargs:
            del kwargs['ephemeris']
    
    if orbit_type == "keplerian":
        return KeplerianGenerator(**kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(**kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        # For CRTBP orbit types, use the new CRTBPOrbitGenerator
        if orbit_type == "crtbp":
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            ephemeris=ephemeris,
            use_high_precision=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown orbit type: {orbit_type}")


def create_high_precision_generator(orbit_type: str, ephemeris, **kwargs) -> BaseTrajectoryGenerator:
    """
    Factory function: create high-precision ephemeris generator instance.
    
    Args:
        orbit_type: Orbit type
        ephemeris: High-precision ephemeris instance (must support get_state method)
        **kwargs: Generator-specific parameters
        
    Returns:
        BaseTrajectoryGenerator: Generator instance (with high-precision mode enabled)
    """
    orbit_type = orbit_type.lower()
    
    # Keplerian orbits don't support high-precision mode
    if orbit_type in ["keplerian", "j2_keplerian"]:
        raise ValueError(
            f"{orbit_type} generator doesn't support high-precision mode. "
            "Please use create_generator() or consider using numerical integration generator."
        )
    
    # Ensure use_high_precision=True is passed
    kwargs['use_high_precision'] = True
    
    if orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        if orbit_type == "crtbp":
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            ephemeris=ephemeris,
            use_high_precision=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown orbit type: {orbit_type}")
