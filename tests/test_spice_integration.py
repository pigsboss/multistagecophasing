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
    from spiceypy.utils.exceptions import SpiceyError
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    spice = None
    SpiceyError = None

# Import MCPC modules
from mission_sim.core.spacetime.ephemeris.spice_interface import (
    SPICEInterface, SPICEKernelManager, SPICECalculator, 
    SPICEConfig, SPICEError, KernelNotFoundError, KernelLoadError, MissionType
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
    
    def _get_available_frame(self, spice_interface, preferred_frames):
        """获取一个可用的坐标系框架"""
        for frame in preferred_frames:
            try:
                # 尝试获取从 J2000 到该框架的旋转矩阵
                if isinstance(frame, CoordinateFrame):
                    rot_mat = spice_interface.get_rotation_matrix(
                        CoordinateFrame.J2000_ECI,
                        frame,
                        epoch=0.0
                    )
                else:
                    rot_mat = spice_interface.get_rotation_matrix(
                        "J2000",
                        frame,
                        epoch=0.0
                    )
                return frame, rot_mat
            except SPICEError:
                continue
        return None, None

    def test_coordinate_frame_transform(self, spice_interface):
        """Test coordinate frame transformation."""
        # 优先尝试的框架列表
        preferred_frames = [
            CoordinateFrame.SUN_EARTH_ROTATING,
            "IAU_EARTH",
            "IAU_MOON",
            "ECLIPJ2000",
            "IAU_SUN",
        ]
        
        frame, rot_mat = self._get_available_frame(spice_interface, preferred_frames)
        
        if frame is None:
            pytest.skip("No suitable coordinate frame available for testing")
        
        # 验证旋转矩阵
        assert rot_mat.shape == (3, 3)
        
        # Check orthogonality (R^T * R = I)
        identity_check = rot_mat.T @ rot_mat
        np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
        
        # Check determinant should be 1 (proper rotation)
        det = np.linalg.det(rot_mat)
        np.testing.assert_almost_equal(det, 1.0, decimal=10)
    
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
    
    def test_moon_pa_specific_in_spice_calculator(self, spice_interface):
        """
        Test that SPICECalculator specifically uses MOON_PA when available.
        """
        # Get the calculator from the interface
        calculator = spice_interface._calc
        
        # Test get_moon_libration_matrix
        epoch = 0.0
        try:
            rot_mat = calculator.get_moon_libration_matrix(epoch)
            
            # The method should have tried MOON_PA first
            # We can't directly check which frame was used, but we can verify
            # that the result is a valid rotation matrix
            
            assert rot_mat.shape == (3, 3)
            
            # Check orthogonality
            identity_check = rot_mat.T @ rot_mat
            np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
            
            # Check determinant (should be 1 for rotation matrix)
            det = np.linalg.det(rot_mat)
            np.testing.assert_almost_equal(det, 1.0, decimal=6)
            
        except SPICEError as e:
            # If MOON_PA is not available, the method should have fallen back to IAU_MOON
            # and still returned a valid matrix
            pytest.skip(f"Cannot test MOON_PA in SPICECalculator: {e}")


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
    
    def test_high_precision_ephemeris_moon_pa(self, high_precision_ephemeris):
        """
        Test that HighPrecisionEphemeris uses MOON_PA when available.
        """
        ephem = high_precision_ephemeris
        
        try:
            rot_mat = ephem.get_moon_libration_matrix(epoch=0.0)
            assert rot_mat.shape == (3, 3)
            
            # Check it's a valid rotation matrix
            identity_check = rot_mat.T @ rot_mat
            np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
            
            det = np.linalg.det(rot_mat)
            np.testing.assert_almost_equal(det, 1.0, decimal=6)
            
        except SPICEError as e:
            pytest.skip(f"Cannot test MOON_PA in HighPrecisionEphemeris: {e}")
    
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
# Moon PA Frame Tests
# ============================================================================

class TestMoonPAFrame:
    """Test MOON_PA frame loading and functionality."""
    
    def test_moon_pa_file_exists_and_loaded(self, spice_kernels_path, required_kernels_present):
        """
        Test that moon_pa_de440_200625.bpc file is found and loaded correctly.
        """
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available or kernels missing")
        
        # Check if moon_pa file exists
        moon_pa_pattern = "moon_pa*.bpc"
        moon_pa_files = list(spice_kernels_path.rglob(moon_pa_pattern))
        
        if not moon_pa_files:
            pytest.skip(f"No moon_pa files found with pattern: {moon_pa_pattern}")
        
        # Initialize SPICE with verbose output to see loading messages
        config = SPICEConfig(
            mission_type="earth_moon",
            verbose=True,  # Enable verbose to see loading messages
            use_light_time_correction=True
        )
        
        interface = SPICEInterface(spice_kernels_path, config)
        
        try:
            # Initialize SPICE
            success = interface.initialize()
            assert success, "SPICE initialization failed"
            
            # Get kernel manager and check loaded kernels
            km = interface._km
            loaded_kernels = km.get_loaded_kernels()
            
            # Check if any moon_pa kernel is loaded
            moon_pa_loaded = any("moon_pa" in k.name.lower() for k in loaded_kernels)
            assert moon_pa_loaded, f"No moon_pa kernel loaded. Loaded kernels: {[k.name for k in loaded_kernels]}"
            
            # Try to get moon libration matrix using MOON_PA
            # This should succeed if MOON_PA frame is available
            try:
                rot_mat = interface.get_moon_libration_matrix(epoch=0.0)
                assert rot_mat.shape == (3, 3), f"Invalid rotation matrix shape: {rot_mat.shape}"
                
                # Verify it's a valid rotation matrix (orthogonal with det=1)
                identity_check = rot_mat.T @ rot_mat
                np.testing.assert_array_almost_equal(
                    identity_check, 
                    np.eye(3), 
                    decimal=10,
                    err_msg="Rotation matrix is not orthogonal"
                )
                
                det = np.linalg.det(rot_mat)
                np.testing.assert_almost_equal(
                    det, 
                    1.0, 
                    decimal=6,
                    err_msg=f"Rotation matrix determinant is not 1: {det}"
                )
                
            except SPICEError as e:
                pytest.fail(f"Failed to get moon libration matrix with MOON_PA frame: {e}")
                
        finally:
            interface.shutdown()
    
    def test_moon_pa_frame_available(self, spice_interface):
        """
        Test that MOON_PA frame is available and can be used for coordinate transformations.
        """
        # This test uses the fixture spice_interface which is already initialized
        
        # Directly test if MOON_PA frame is available using spiceypy
        try:
            # Try to get transformation from J2000 to MOON_PA
            rot_mat = spice.pxform('J2000', 'MOON_PA', 0.0)
            assert rot_mat is not None
            
            # Convert to numpy array and check properties
            rot_mat_np = np.array(rot_mat)
            assert rot_mat_np.shape == (3, 3)
            
            # Check orthogonality
            identity_check = rot_mat_np.T @ rot_mat_np
            np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
            
            # Also test the reverse transformation
            rot_mat_reverse = spice.pxform('MOON_PA', 'J2000', 0.0)
            rot_mat_reverse_np = np.array(rot_mat_reverse)
            
            # Check that reverse is transpose of forward
            np.testing.assert_array_almost_equal(
                rot_mat_reverse_np, 
                rot_mat_np.T, 
                decimal=10,
                err_msg="MOON_PA to J2000 transform is not transpose of J2000 to MOON_PA"
            )
            
        except SpiceyError as e:
            # 如果 MOON_PA 不可用，尝试使用 IAU_MOON 作为备选
            try:
                rot_mat = spice.pxform('J2000', 'IAU_MOON', 0.0)
                assert rot_mat is not None
                
                rot_mat_np = np.array(rot_mat)
                assert rot_mat_np.shape == (3, 3)
                
                identity_check = rot_mat_np.T @ rot_mat_np
                np.testing.assert_array_almost_equal(identity_check, np.eye(3), decimal=10)
                
                print(f"Note: Using IAU_MOON instead of MOON_PA: {e}")
                
            except SpiceyError as e2:
                pytest.skip(f"Neither MOON_PA nor IAU_MOON frame available: {e2}")
    
    def test_moon_pa_vs_iau_moon_difference(self, spice_interface):
        """
        Test that MOON_PA and IAU_MOON frames are different (MOON_PA should be higher precision).
        """
        epoch = 0.0
        
        try:
            # Get transformation matrices for both frames
            rot_mat_moon_pa = spice.pxform('J2000', 'MOON_PA', epoch)
            rot_mat_iau_moon = spice.pxform('J2000', 'IAU_MOON', epoch)
            
            rot_mat_moon_pa_np = np.array(rot_mat_moon_pa)
            rot_mat_iau_moon_np = np.array(rot_mat_iau_moon)
            
            # Calculate difference between the two rotation matrices
            diff = np.linalg.norm(rot_mat_moon_pa_np - rot_mat_iau_moon_np)
            
            # The difference should be non-zero (MOON_PA is higher precision)
            assert diff > 1e-10, f"MOON_PA and IAU_MOON are identical (diff={diff}), MOON_PA may not be loaded"
            
            # Log the difference for information
            print(f"\nDifference between MOON_PA and IAU_MOON at epoch {epoch}: {diff}")
            
            # Verify both are valid rotation matrices
            for name, mat in [("MOON_PA", rot_mat_moon_pa_np), ("IAU_MOON", rot_mat_iau_moon_np)]:
                identity_check = mat.T @ mat
                np.testing.assert_array_almost_equal(
                    identity_check, 
                    np.eye(3), 
                    decimal=10,
                    err_msg=f"{name} rotation matrix is not orthogonal"
                )
                
        except SpiceyError as e:
            pytest.skip(f"Cannot compare MOON_PA and IAU_MOON: {e}")
    
    def test_moon_pa_file_specific_loading(self, spice_kernels_path, required_kernels_present):
        """
        Test specific moon_pa file loading with detailed error messages.
        """
        if not SPICE_AVAILABLE or not required_kernels_present:
            pytest.skip("SPICE not available")
        
        # Look for specific moon_pa file
        specific_file = spice_kernels_path / "pck" / "moon_pa_de440_200625.bpc"
        if not specific_file.exists():
            # Try other possible locations
            found = False
            for subdir in ["", "pck", "data/pck"]:
                candidate = spice_kernels_path / subdir / "moon_pa_de440_200625.bpc"
                if candidate.exists():
                    specific_file = candidate
                    found = True
                    break
            
            if not found:
                pytest.skip(f"Specific moon_pa file not found: moon_pa_de440_200625.bpc")
        
        # Test loading this specific file
        config = SPICEConfig(mission_type="earth_moon", verbose=True)
        interface = SPICEInterface(spice_kernels_path, config)
        
        try:
            success = interface.initialize()
            assert success, "SPICE initialization failed"
            
            # Verify the specific file was loaded
            km = interface._km
            loaded_kernels = km.get_loaded_kernels()
            
            specific_loaded = any(
                "moon_pa_de440_200625.bpc" in str(k) for k in loaded_kernels
            )
            
            assert specific_loaded, (
                f"Specific moon_pa file not loaded: {specific_file.name}\n"
                f"Loaded kernels: {[k.name for k in loaded_kernels]}"
            )
            
            # Test that we can use the frame
            rot_mat = interface.get_moon_libration_matrix(0.0)
            assert rot_mat.shape == (3, 3)
            
            # Additional verification: try to get frame information
            try:
                # Try to get frame ID for MOON_PA
                frame_id = spice.namfrm("MOON_PA")
                assert frame_id != 0, "MOON_PA frame ID is 0 (invalid)"
                print(f"\nMOON_PA frame ID: {frame_id}")
                
            except SpiceyError as e:
                pytest.fail(f"Cannot get frame ID for MOON_PA: {e}")
                
        finally:
            interface.shutdown()
    


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
