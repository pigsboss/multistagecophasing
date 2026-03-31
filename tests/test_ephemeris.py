import numpy as np
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris

def test_ephemeris_linear_interpolation():
    """测试线性数据的三次样条插值（应精确）"""
    times = np.array([0.0, 1.0, 2.0])
    states = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ])
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)
    state_mid = eph.get_interpolated_state(0.5)
    expected = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(state_mid, expected, rtol=1e-10)

def test_ephemeris_cubic_spline():
    """测试正弦曲线的三次样条插值精度"""
    times = np.linspace(0, 2*np.pi, 10)
    states = []
    for t in times:
        states.append([np.sin(t), 0.0, 0.0, np.cos(t), 0.0, 0.0])
    eph = Ephemeris(times, np.array(states), CoordinateFrame.J2000_ECI)

    test_times = np.linspace(0, 2*np.pi, 100)
    errors = []
    for t in test_times:
        interp = eph.get_interpolated_state(t)
        exact = np.array([np.sin(t), 0.0, 0.0, np.cos(t), 0.0, 0.0])
        err = np.linalg.norm(interp - exact)
        errors.append(err)
    max_error = max(errors)
    assert max_error < 0.03, f"最大插值误差 {max_error} 超过 0.03"

def test_ephemeris_out_of_range_warning():
    """测试超出时间范围的警告（不崩溃）"""
    times = np.array([0.0, 10.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)
    # 应产生警告但不崩溃
    eph.get_interpolated_state(-1.0)
    eph.get_interpolated_state(11.0)
