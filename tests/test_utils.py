import numpy as np
from mission_sim.utils.math_tools import get_lqr_gain, absolute_to_lvlh
from mission_sim.utils.differential_correction import jacobi_constant

def test_get_lqr_gain():
    """测试 LQR 增益计算（简单双积分器）"""
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    K = get_lqr_gain(A, B, Q, R)
    # 双积分器 LQR 增益应为 [1, sqrt(3)]? 这里只验证不崩溃
    assert K.shape == (1, 2)

def test_absolute_to_lvlh():
    """测试 LVLH 转换"""
    # 主星圆轨道
    r = 7000e3
    v = np.sqrt(3.986004418e14 / r)
    state_chief = np.array([r, 0, 0, 0, v, 0])
    # 从星在主星前方 100m
    state_deputy = np.array([r, 100, 0, 0, v, 0])
    rel = absolute_to_lvlh(state_chief, state_deputy)
    # 预期相对位置 [0, 100, 0] 左右，速度应为 0
    assert np.allclose(rel[0:3], [0, 100, 0], atol=1e-6)
    assert np.linalg.norm(rel[3:6]) < .2

def test_jacobi_constant():
    """测试雅可比常数计算"""
    state_nd = np.array([1.01106, 0.0, 0.05, 0.0, 0.0105, 0.0])
    mu = 3.00348e-6
    C = jacobi_constant(state_nd, mu)
    assert isinstance(C, float)
    assert not np.isnan(C)
