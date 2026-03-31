import numpy as np
import pytest
from mission_sim.core.cyber.models.threebody.base import CRTBP

def test_crtbp_jacobi_constant_conservation(crtbp_params, halo_initial_state):
    """测试雅可比常数在半个周期内的守恒性"""
    crtbp = CRTBP(crtbp_params["mu"], crtbp_params["L"], crtbp_params["omega"])
    state0_nd = halo_initial_state
    # 积分半周期
    from scipy.integrate import solve_ivp
    T_half = 1.5  # 近似半周期
    sol = solve_ivp(crtbp.dynamics, (0, T_half), state0_nd, method='DOP853', rtol=1e-12, atol=1e-12)
    C0 = crtbp.jacobi_constant(state0_nd)
    Cf = crtbp.jacobi_constant(sol.y[:, -1])
    assert abs(Cf - C0) < 1e-10, f"雅可比常数变化 {abs(Cf - C0)} 超过 1e-10"

def test_crtbp_to_physical_to_nd(crtbp_params):
    """测试无量纲与物理单位互转"""
    crtbp = CRTBP(crtbp_params["mu"], crtbp_params["L"], crtbp_params["omega"])
    state_nd = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.03])
    t_nd = 1.0
    state_phys, t_phys = crtbp.to_physical(state_nd, t_nd)
    state_nd2, t_nd2 = crtbp.to_nd(state_phys, t_phys)
    assert np.allclose(state_nd, state_nd2, rtol=1e-10)
    assert abs(t_nd - t_nd2) < 1e-10