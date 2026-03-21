import numpy as np
import pytest
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.core.gnc.propagator import SimplePropagator, KeplerPropagator

def test_gnc_load_reference_trajectory():
    """测试加载参考轨迹"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.load_reference_trajectory(eph)
    assert gnc.ref_ephemeris is eph

def test_gnc_load_reference_trajectory_wrong_frame():
    """测试错误坐标系的轨迹被拒绝"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)
    with pytest.raises(ValueError, match="标称星历坐标系不匹配"):
        gnc.load_reference_trajectory(eph)

def test_gnc_update_navigation():
    """测试导航状态更新"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    obs = np.array([1,2,3,4,5,6])
    gnc.update_navigation(obs, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.array_equal(gnc.current_nav_state, obs)

def test_gnc_update_navigation_wrong_frame():
    """测试错误坐标系观测被拒绝"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    obs = np.zeros(6)
    with pytest.raises(ValueError, match="导航状态坐标系不匹配"):
        gnc.update_navigation(obs, CoordinateFrame.J2000_ECI)

def test_gnc_compute_control_force():
    """测试控制力计算"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.load_reference_trajectory(eph)
    gnc.update_navigation(np.array([1,0,0,0,0,0]), CoordinateFrame.SUN_EARTH_ROTATING)
    K = np.array([[0.01,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]])
    force, frame = gnc.compute_control_force(0.0, K)
    assert np.allclose(force, [-0.01, 0, 0])
    assert frame == CoordinateFrame.SUN_EARTH_ROTATING

def test_gnc_propagator_simple():
    """测试简单外推器"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.set_propagator(SimplePropagator())
    gnc.current_nav_state = np.array([0,0,0,1,0,0])
    gnc.update_navigation(None, CoordinateFrame.SUN_EARTH_ROTATING, dt=10.0)
    assert np.allclose(gnc.current_nav_state, [10,0,0,1,0,0])

def test_gnc_propagator_kepler():
    """测试二体外推器"""
    mu = 3.986004418e14
    propagator = KeplerPropagator(mu)
    gnc = GNC_Subsystem("test", CoordinateFrame.J2000_ECI)
    gnc.set_propagator(propagator)
    # 初始状态：地心 7000km 圆轨道
    r = 7000e3
    v = np.sqrt(mu / r)
    gnc.current_nav_state = np.array([r, 0, 0, 0, v, 0])
    dt = 10.0
    gnc.update_navigation(None, CoordinateFrame.J2000_ECI, dt=dt)
    # 粗略验证位置变化
    new_pos = gnc.current_nav_state[0:3]
    assert abs(new_pos[0] - r) < 1000.0, f"径向距离变化 {abs(new_pos[0]-r)} 超过 1000m"  # 允许 1000m 的变化
    # 验证速度大小应大致保持不变
    new_vel = gnc.current_nav_state[3:6]
    assert abs(np.linalg.norm(new_vel) - v) < 0.1, f"速度变化 {abs(np.linalg.norm(new_vel)-v)} 超过 0.1m/s"
