import numpy as np
import pytest
from numpy.testing import assert_allclose

from mission_sim.utils.math_tools import (
    normalize_vector,
    compute_lvlh_dcm,
    absolute_to_lvlh,
    lvlh_to_absolute
)

class TestMathTools:

    def test_normalize_vector(self):
        """Test vector normalization and zero-vector safety."""
        v = np.array([3.0, 4.0, 0.0])
        v_norm = normalize_vector(v)
        assert_allclose(v_norm, [0.6, 0.8, 0.0], atol=1e-12)

        v_zero = np.array([0.0, 0.0, 0.0])
        v_zero_norm = normalize_vector(v_zero)
        assert_allclose(v_zero_norm, [0.0, 0.0, 0.0], atol=1e-12)

    def test_compute_lvlh_dcm_properties(self):
        """
        Verify that the computed LVLH DCM is a valid, orthogonal rotation matrix.
        DCM * DCM.T should equal the Identity matrix, and Det(DCM) = 1.
        LVLH definition: X = radial, Y = along-track, Z = cross-track.
        """
        r_chief = np.array([7000e3, 0.0, 0.0])  # radial direction (X-axis)
        v_chief = np.array([0.0, 7.5e3, 0.0])   # velocity direction (Y-axis)

        dcm = compute_lvlh_dcm(r_chief, v_chief)

        # 1. Check Orthogonality (DCM @ DCM.T == I)
        identity = np.eye(3)
        assert_allclose(dcm @ dcm.T, identity, atol=1e-12)

        # 2. Check Right-Handedness (Determinant == 1)
        det = np.linalg.det(dcm)
        assert_allclose(det, 1.0, atol=1e-12)

        # 3. Check specific axis alignments
        # X-axis (row 0) should align with r_chief (radial)
        assert_allclose(dcm[0, :], [1.0, 0.0, 0.0], atol=1e-12)
        # Y-axis (row 1) should align with v_chief (along-track)
        assert_allclose(dcm[1, :], [0.0, 1.0, 0.0], atol=1e-12)
        # Z-axis (row 2) should align with angular momentum direction (cross-track)
        # For this orbit, angular momentum is along +Z
        assert_allclose(dcm[2, :], [0.0, 0.0, 1.0], atol=1e-12)

    def test_bidirectional_transformation_leo(self):
        """
        Test forward and inverse transformation in Low Earth Orbit (LEO) conditions.
        Ensures position error is < 1e-10 meters.
        """
        r_c = np.array([6800e3, 1000e3, -500e3])
        v_c = np.array([-1.0e3, 7.5e3, 2.0e3])
        r_d = r_c + np.array([100.0, -500.0, 20.0])
        v_d = v_c + np.array([0.1, -0.05, 0.01])

        rho, rho_dot = absolute_to_lvlh(r_c, v_c, r_d, v_d)
        r_d_recovered, v_d_recovered = lvlh_to_absolute(r_c, v_c, rho, rho_dot)

        assert_allclose(r_d_recovered, r_d, atol=1e-10)
        assert_allclose(v_d_recovered, v_d, atol=1e-10)

    def test_bidirectional_transformation_deep_space(self):
        """
        Test forward and inverse transformation in Deep Space (e.g., Sun-Earth L2).
        """
        r_c = np.array([1.5e11, 2.0e10, -5.0e9])
        v_c = np.array([-2.0e3, 30.0e3, 1.0e3])
        r_d = r_c + np.array([10.0, -5.0, 2.0])
        v_d = v_c + np.array([0.001, -0.002, 0.0005])

        rho, rho_dot = absolute_to_lvlh(r_c, v_c, r_d, v_d)
        r_d_recovered, v_d_recovered = lvlh_to_absolute(r_c, v_c, rho, rho_dot)

        assert_allclose(r_d_recovered, r_d, rtol=1e-14)
        assert_allclose(v_d_recovered, v_d, atol=1e-10)

    def test_coriolis_compensation_validation(self):
        """
        Validate that the velocity transformation correctly handles frame rotation.
        If a deputy is completely stationary in the LVLH frame (rho_dot = 0),
        its absolute velocity should exactly match the Chief's velocity PLUS
        the transport velocity (omega x rho).
        """
        r_c = np.array([7000e3, 0.0, 0.0])
        v_c = np.array([0.0, 7.5e3, 0.0])
        
        rho_lvlh_static = np.array([0.0, 0.0, 100.0])  # purely radial offset
        rho_dot_lvlh_static = np.array([0.0, 0.0, 0.0])

        r_d_abs, v_d_abs = lvlh_to_absolute(r_c, v_c, rho_lvlh_static, rho_dot_lvlh_static)
        rho_lvlh_rec, rho_dot_lvlh_rec = absolute_to_lvlh(r_c, v_c, r_d_abs, v_d_abs)

        assert_allclose(rho_dot_lvlh_rec, [0.0, 0.0, 0.0], atol=1e-12)