"""
SPICE Integration Tests for MCPC High-Precision Ephemeris

This module tests the integration between MCPC and NASA NAIF SPICE toolkit.
Tests require spiceypy to be installed and SPICE kernels to be available.

Environment Variables:
    SPICE_KERNELS: Path to SPICE kernels directory (optional)
    
Skipped if:
    - spiceypy is not installed
    - Required kernels (naif0012.tls, de440.bsp, etc.) are not found
"""

import warnings
# Filter out requests library warnings about urllib3/chardet compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import pytest
import numpy as np
from pathlib import Path
import os
import sys

# Check if spiceypy is available
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    spice = None

# Import MCPC modules
from mission_sim.core.spacetime.ephemeris.spice_interface import (
    SPICEInterface, SPICEKernelManager, SPICECalculator, 
    SPICEConfig, SPICEError, KernelNotFoundError, MissionType
)
from mission_sim.core.spacetime.ephemeris.high_precision import (
    HighPrecisionEphemeris, EphemerisMode, EphemerisConfig, CelestialBody
)
from mission_sim.core.spacetime.ids import CoordinateFrame


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def spice_kernels_path():
    """
    Determine SPICE kernels path for testing.
    
    Priority:
    1. Environment variable SPICE_KERNELS
    2. ./spice_kernels (relative to test execution)
    3. ../spice_kernels (relative to test file)
    """
    env_path = os.environ.get('SPICE_KERNELS')
    if env_path:
        return Path(env_path)
    
    # Try common locations
    candidates = [
        Path('./spice_kernels'),
        Path(__file__).parent.parent / 'spice_kernels',
        Path(__file__).parent.parent.parent / 'spice_kernels',
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


@pytest.fixture(scope="module")
def required_kernels_present(spice_kernels_path):
    """Check if minimum required kernels are available."""
    if not spice_kernels_path or not spice_kernels_path.exists():
        return False
    
    # Check for essential kernels
    required_patterns = ['naif*.tls', 'de440.bsp', 'de441.bsp', 'de442.bsp']
    found_any = False
    
    for pattern in required_patterns:
        if list(spice_kernels_path.rglob(pattern)):
            found_any = True
            break
    
    return found_any


@pytest.fixture(scope="function")
def spice_interface(spice_kernels_path, required_kernels_present):
    """
    Create a SPICEInterface instance for testing.
    
    Automatically cleans up (unloads kernels) after each test.
    """
    if not SPICE_AVAILABLE:
        pytest.skip("spiceypy not installed")
    
    if not required_kernels_present:
        pytest.skip(f"Required SPICE kernels not found in {spice_kernels_path}")
    
    config = SPICEConfig(
        mission_type="earth_moon",
        verbose=False,
        use_light_time_correction=True,
        use_stellar_aberration=True
    )
    
    interface = SPICEInterface(spice_kernels_path, config)
    
    # Initialize
    success = interface.initialize()
    if not success:
        pytest.skip("SPICE initialization failed")
    
    yield interface
    
    # Cleanup
    interface.shutdown()


@pytest.fixture(scope="function")
def high_precision_ephemeris(spice_kernels_path, required_kernels_present):
    """Create a HighPrecisionEphemeris instance in SPICE mode."""
    if not SPICE_AVAILABLE:
        pytest.skip("spiceypy not installed")
    
    if not required_kernels_present:
        pytest.skip("Required SPICE kernels not found")
    
    config = EphemerisConfig(
        mode=EphemerisMode.SPICE,
        spice_kernels_path=spice_kernels_path,
        spice_mission_type="earth_moon",
        verbose=False
    )
    
    ephem = HighPrecisionEphemeris(config=config)
    
    if not ephem._spice_initialized:
        pytest.skip("SPICE initialization failed in HighPrecisionEphemeris")
    
    yield ephem
    
    ephem.shutdown()


# ============================================================================
# Basic SPICE Availability Tests
# ============================================================================

class TestSPICEAvailability:
    """Test basic SPICE availability and imports."""
    
    def test_spice_available(self):
        """Test that spiceypy is installed."""
        assert SPICE_AVAILABLE, "spiceypy should be installed for these tests"
    
    def test_spice_interface_import(self):
        """Test that SPICEInterface can be imported."""
        from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
        assert SPICEInterface is not None


# ============================================================================
# SPICEKernelManager Tests
# ============================================================================

class TestSPICEKernelManager:
    """Test kernel management functionality."""
    
    def test_kernel_manager_initialization(self, spice_kernels_path, required_kernels_present):
        """Test kernel manager initialization."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available or kernels missing")
        
        manager = SPICEKernelManager(spice_kernels_path)
        manager.initialize("earth_moon")
        
        assert manager.is_initialized()
        assert len(manager.get_loaded_kernels()) > 0
        
        # Cleanup
        manager.unload_all()
        assert not manager.is_initialized()
    
    def test_kernel_auto_discovery(self, spice_kernels_path, required_kernels_present):
        """Test that kernels are automatically discovered in subdirectories."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available or kernels missing")
        
        manager = SPICEKernelManager(spice_kernels_path, SPICEConfig(verbose=True))
        manager.initialize("earth_moon")
        
        loaded = manager.get_loaded_kernels()
        names = [k.name for k in loaded]
        
        # Check that essential kernels are loaded
        assert any('naif' in name.lower() for name in names), "Leapseconds kernel should be loaded"
        
        manager.unload_all()
    
    def test_invalid_kernel_path(self):
        """Test handling of invalid kernel path."""
        if not SPICE_AVAILABLE:
            pytest.skip("spiceypy not installed")
        
        with pytest.raises(KernelNotFoundError):
            manager = SPICEKernelManager("/nonexistent/path")
            manager.initialize()
    
    def test_invalid_kernel_type(self, spice_kernels_path, required_kernels_present):
        """Test that invalid mission type raises appropriate error."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available or kernels missing")
        
        manager = SPICEKernelManager(spice_kernels_path)
        
        # 改为期望 ValueError 而不是 KernelNotFoundError
        with pytest.raises(ValueError):
            manager.initialize("invalid_mission_type")


# ============================================================================
# SPICECalculator Tests
# ============================================================================

class TestSPICECalculator:
    """Test SPICE calculation functions."""
    
    def test_moon_earth_state(self, spice_interface):
        """Test calculating Moon state relative to Earth."""
        state = spice_interface.get_state(
            target='moon',
            epoch=0.0,  # J2000 epoch
            observer='earth',
            frame=CoordinateFrame.J2000_ECI,
            abcorr='NONE'
        )
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
        
        # Check magnitude (Moon distance ~384,400 km)
        pos_norm = np.linalg.norm(state[:3])
        assert 3.5e8 < pos_norm < 4.5e8, f"Moon distance {pos_norm} out of expected range"
        
        # Check velocity magnitude (~1 km/s)
        vel_norm = np.linalg.norm(state[3:6])
        assert 500 < vel_norm < 1500, f"Moon velocity {vel_norm} out of expected range"
    
    def test_earth_sun_state(self, spice_interface):
        """Test calculating Earth state relative to Sun."""
        state = spice_interface.get_state(
            target='earth',
            epoch=0.0,
            observer='sun',
            frame=CoordinateFrame.J2000_ECI
        )
        
        # Earth orbital radius ~1 AU = 149.6 million km
        pos_norm = np.linalg.norm(state[:3])
        assert 1.4e11 < pos_norm < 1.6e11, f"Earth orbital radius {pos_norm} out of expected range"
    
    def test_light_time_correction(self, spice_interface):
        """Test that light time correction produces slightly different results."""
        # Geometric state
        state_geom = spice_interface.get_geometric_state('moon', 0.0, 'earth')
        
        # Light-time corrected state
        state_lt, lt = spice_interface.get_light_time_corrected_state('moon', 0.0, 'earth')
        
        # Results should be close but not identical
        diff = np.linalg.norm(state_geom[:3] - state_lt[:3])
        assert diff < 1e6  # Difference should be less than 1000 km
        assert diff > 1.0  # But not zero (that would indicate no correction)
        
        # Light time should be about 1.3 seconds (Moon distance / c)
        assert 1.0 < lt < 2.0, f"Light time {lt} out of expected range"
    
    def test_time_conversion(self, spice_interface):
        """Test UTC to ET conversion."""
        utc = "2026-04-10T12:00:00"
        
        # Convert to ET (seconds since J2000)
        et = spice_interface.utc_to_et(utc)
        
        # Should be positive (after year 2000)
        assert et > 0
        
        # Convert back
        utc_back = spice_interface.et_to_utc(et)
        
        # Should be approximately the same (allowing for format differences)
        assert "2026" in utc_back and "04" in utc_back and "10" in utc_back
    
    def test_coordinate_frame_transform(self, spice_interface):
        """Test coordinate frame transformation."""
        # This test assumes SUN_EARTH_ROTATING frame is defined in SPICE kernels
        # If not, it will be skipped
        
        try:
            rot_mat = spice_interface.get_rotation_matrix(
                CoordinateFrame.J2000_ECI,
                CoordinateFrame.SUN_EARTH_ROTATING,
                epoch=0.0
            )
            
            assert rot_mat.shape == (3, 3)
            
            # Check orthogonality (R^T * R = I)
            identity_check = rot_mat.T @ rot_mat
            np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
            
        except SPICEError:
            pytest.skip("SUN_EARTH_ROTATING frame not defined in kernels")
    
    def test_moon_libration(self, spice_interface):
        """Test moon libration matrix retrieval."""
        try:
            rot_mat = spice_interface.get_moon_libration_matrix(epoch=0.0)
            
            assert rot_mat.shape == (3, 3)
            
            # Check orthogonality
            identity_check = rot_mat.T @ rot_mat
            np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
            
        except SPICEError:
            pytest.skip("MOON_PA frame not available (requires moon_pa_de440_200625.bpc)")


# ============================================================================
# HighPrecisionEphemeris Integration Tests
# ============================================================================

class TestHighPrecisionEphemerisSPICE:
    """Test HighPrecisionEphemeris with SPICE mode."""
    
    def test_spice_mode_initialization(self, high_precision_ephemeris):
        """Test that HighPrecisionEphemeris initializes in SPICE mode."""
        ephem = high_precision_ephemeris
        
        assert ephem.config.mode == EphemerisMode.SPICE
        assert ephem._spice_initialized
        assert ephem._spice_interface is not None
        assert ephem._spice_interface.is_available()
    
    def test_get_state_spice(self, high_precision_ephemeris):
        """Test get_state with SPICE backend."""
        ephem = high_precision_ephemeris
        
        state = ephem.get_state(
            target_body=CelestialBody.MOON,
            epoch=0.0,
            observer_body=CelestialBody.EARTH,
            frame=CoordinateFrame.J2000_ECI
        )
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
        
        # Verify reasonable values
        pos_norm = np.linalg.norm(state[:3])
        assert 3.5e8 < pos_norm < 4.5e8  # Moon distance
    
    def test_mode_switching(self, spice_kernels_path, required_kernels_present):
        """Test switching between analytical and SPICE modes."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available")
        
        # Start with analytical
        ephem = HighPrecisionEphemeris(config=EphemerisConfig(mode=EphemerisMode.ANALYTICAL))
        assert ephem.config.mode == EphemerisMode.ANALYTICAL
        
        # Switch to SPICE
        ephem.set_mode(EphemerisMode.SPICE, spice_kernels_path=spice_kernels_path)
        assert ephem.config.mode == EphemerisMode.SPICE
        assert ephem._spice_initialized
        
        # Verify it works
        state = ephem.get_state(CelestialBody.MOON, 0.0)
        assert np.linalg.norm(state[:3]) > 0
        
        ephem.shutdown()
    
    def test_spice_fallback_to_analytical(self):
        """Test fallback to analytical mode when SPICE fails."""
        # Create with invalid path
        config = EphemerisConfig(
            mode=EphemerisMode.SPICE,
            spice_kernels_path="/invalid/path",
            verbose=False
        )
        
        ephem = HighPrecisionEphemeris(config=config)
        
        # Should have fallen back to analytical
        assert not ephem._spice_initialized
        
        # Should still provide results (analytical approximation)
        state = ephem.get_state(CelestialBody.MOON, 0.0)
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
    
    def test_moon_libration_matrix_high_precision(self, high_precision_ephemeris):
        """Test moon libration matrix through HighPrecisionEphemeris."""
        ephem = high_precision_ephemeris
        
        try:
            rot_mat = ephem.get_moon_libration_matrix(epoch=0.0)
            assert rot_mat.shape == (3, 3)
            
            # Verify it's a valid rotation matrix (det = 1)
            det = np.linalg.det(rot_mat)
            np.testing.assert_almost_equal(det, 1.0, decimal=6)
            
        except SPICEError:
            pytest.skip("Moon libration data not available")
    
    def test_context_manager(self, spice_kernels_path, required_kernels_present):
        """Test context manager properly initializes and cleans up."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available")
        
        config = EphemerisConfig(
            mode=EphemerisMode.SPICE,
            spice_kernels_path=spice_kernels_path
        )
        
        with HighPrecisionEphemeris(config=config) as ephem:
            assert ephem._spice_initialized
            state = ephem.get_state(CelestialBody.MOON, 0.0)
            assert np.linalg.norm(state[:3]) > 0
        
        # After exiting context, SPICE should be shut down
        assert not ephem._spice_initialized


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestSPICEErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_body_name(self, spice_interface):
        """Test handling of invalid celestial body name."""
        with pytest.raises((SPICEError, KeyError)):
            spice_interface.get_state('invalid_body', 0.0)
    
    def test_invalid_time(self, spice_interface):
        """Test handling of invalid time (far future/past)."""
        # Very far future (year 3000) - might not be covered by kernels
        try:
            state = spice_interface.get_state('moon', 1e10, 'earth')  # ~300 years in seconds
            # If it succeeds, that's fine (kernels might cover it)
        except SPICEError:
            # Expected if outside kernel coverage
            pass
    
    def test_invalid_kernel_type(self, spice_kernels_path, required_kernels_present):
        """Test that invalid mission type raises appropriate error."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available")
        
        manager = SPICEKernelManager(spice_kernels_path)
        
        # Expect ValueError because mission type validation happens before kernel loading
        with pytest.raises(ValueError):
            manager.initialize("invalid_mission_type")


# ============================================================================
# Accuracy Verification Tests
# ============================================================================

class TestSPICEAccuracy:
    """Verify accuracy of SPICE calculations against known values."""
    
    def test_j2000_epoch_moon_position(self, high_precision_ephemeris):
        """
        Verify Moon position at J2000 epoch against known values.
        
        Reference: JPL Horizons or similar ephemeris
        """
        ephem = high_precision_ephemeris
        
        state = ephem.get_state(CelestialBody.MOON, 0.0, CelestialBody.EARTH)
        
        # Known approximate Moon position at J2000 (from JPL data)
        # These are approximate values for verification
        # Actual values depend on the specific ephemeris (DE440, etc.)
        
        pos = state[:3]
        distance = np.linalg.norm(pos)
        
        # Moon distance should be approximately 384,400 km ± 20,000 km
        assert 360e6 < distance < 410e6, f"Moon distance {distance} out of realistic range"
        
        # Velocity should be approximately 1 km/s
        vel = np.linalg.norm(state[3:6])
        assert 900 < vel < 1100, f"Moon velocity {vel} out of realistic range"
    
    def test_earth_sun_distance_consistency(self, high_precision_ephemeris):
        """Test that Earth-Sun distance varies reasonably over a year."""
        ephem = high_precision_ephemeris
        
        distances = []
        for day in [0, 91, 182, 273]:  # Quarterly samples
            seconds = day * 24 * 3600
            state = ephem.get_state(CelestialBody.EARTH, seconds, CelestialBody.SUN)
            dist = np.linalg.norm(state[:3])
            distances.append(dist)
        
        # Earth's orbital eccentricity means distance varies by ~3%
        max_dist = max(distances)
        min_dist = min(distances)
        variation = (max_dist - min_dist) / np.mean(distances)
        
        # Should be around 0.03 (3%) for Earth's orbital eccentricity
        assert 0.01 < variation < 0.05, f"Orbital variation {variation} unrealistic"


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test helper functions."""
    
    def test_get_spice_state_function(self, spice_kernels_path, required_kernels_present):
        """Test the get_spice_state convenience function."""
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available")
        
        from mission_sim.core.spacetime.ephemeris.spice_interface import get_spice_state
        
        state = get_spice_state(
            spice_kernels_path,
            target='moon',
            epoch=0.0,
            observer='earth'
        )
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
        assert np.linalg.norm(state[:3]) > 0


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Manual test execution
    pytest.main([__file__, "-v", "--tb=short"])
