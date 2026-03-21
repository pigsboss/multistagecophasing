import numpy as np
import pytest
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.physics.models.gravity_crtbp import Gravity_CRTBP

class MockForceModel(IForceModel):
    def compute_accel(self, state, epoch):
        return np.array([1.0, 0.0, 0.0])

def test_celestial_environment_register():
    """测试环境引擎注册力模型"""
    env = CelestialEnvironment(CoordinateFrame.SUN_EARTH_ROTATING)
    mock = MockForceModel()
    env.register_force(mock)
    assert len(env._force_registry) == 1

def test_celestial_environment_coordinate_check():
    """测试坐标系一致性校验"""
    env = CelestialEnvironment(CoordinateFrame.SUN_EARTH_ROTATING)
    state = np.zeros(6)
    with pytest.raises(ValueError, match="坐标系冲突"):
        env.get_total_acceleration(state, CoordinateFrame.J2000_ECI)

def test_spacecraft_apply_thrust():
    """测试航天器施加推力"""
    sc = SpacecraftPointMass("test", np.zeros(6), CoordinateFrame.SUN_EARTH_ROTATING, initial_mass=100.0)
    force = np.array([10.0, 0.0, 0.0])
    sc.apply_thrust(force, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.allclose(sc.external_accel, [0.1, 0.0, 0.0])

def test_spacecraft_apply_thrust_wrong_frame():
    """测试错误坐标系推力被拒绝"""
    sc = SpacecraftPointMass("test", np.zeros(6), CoordinateFrame.SUN_EARTH_ROTATING)
    force = np.array([10.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="推力坐标系不匹配"):
        sc.apply_thrust(force, CoordinateFrame.J2000_ECI)

def test_spacecraft_get_derivative():
    """测试状态导数计算"""
    sc = SpacecraftPointMass("test", np.array([1,2,3,4,5,6]), CoordinateFrame.SUN_EARTH_ROTATING, initial_mass=10.0)
    gravity = np.array([0.1, 0.2, 0.3])
    deriv = sc.get_derivative(gravity, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.allclose(deriv[0:3], [4,5,6])
    assert np.allclose(deriv[3:6], [0.1, 0.2, 0.3])  # 无推力
    # 施加推力后
    sc.apply_thrust(np.array([10.0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)
    deriv = sc.get_derivative(gravity, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.allclose(deriv[3:6], [0.1+1.0, 0.2, 0.3])  # 推力加速度 1.0 m/s²

def test_spacecraft_integrate_dv():
    """测试 ΔV 累计"""
    sc = SpacecraftPointMass("test", np.zeros(6), CoordinateFrame.SUN_EARTH_ROTATING, initial_mass=1.0)
    sc.apply_thrust(np.array([1.0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)
    dt = 10.0
    sc.integrate_dv(dt)
    assert np.allclose(sc.accumulated_dv, 10.0)  # a = 1 m/s², dt = 10 s => ΔV = 10 m/s
