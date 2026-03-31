import numpy as np
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.physics.models.gravity_crtbp import Gravity_CRTBP
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.core.physics.models.srp import Cannonball_SRP
from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag


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
    sc = SpacecraftPointMass("test", np.array([1, 2, 3, 4, 5, 6]), CoordinateFrame.SUN_EARTH_ROTATING, initial_mass=10.0)
    gravity = np.array([0.1, 0.2, 0.3])
    deriv = sc.get_derivative(gravity, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.allclose(deriv[0:3], [4, 5, 6])
    assert np.allclose(deriv[3:6], [0.1, 0.2, 0.3])  # 无推力
    # 施加推力后
    sc.apply_thrust(np.array([10.0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)
    deriv = sc.get_derivative(gravity, CoordinateFrame.SUN_EARTH_ROTATING)
    assert np.allclose(deriv[3:6], [0.1 + 1.0, 0.2, 0.3])  # 推力加速度 1.0 m/s²


def test_spacecraft_integrate_dv():
    """测试 ΔV 累计"""
    sc = SpacecraftPointMass("test", np.zeros(6), CoordinateFrame.SUN_EARTH_ROTATING, initial_mass=1.0)
    sc.apply_thrust(np.array([1.0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)
    dt = 10.0
    sc.integrate_dv(dt)
    assert np.allclose(sc.accumulated_dv, 10.0)  # a = 1 m/s², dt = 10 s => ΔV = 10 m/s


def test_atmospheric_drag_acceleration():
    """测试大气阻力加速度方向与大小符合指数模型"""
    # 设置参数
    area_to_mass = 0.02          # m²/kg
    Cd = 2.2
    rho0 = 1.225                 # kg/m³
    H = 8500.0                   # m
    h0 = 0.0                     # m
    R_earth = 6378137.0          # m

    drag = AtmosphericDrag(area_to_mass=area_to_mass, Cd=Cd, rho0=rho0, H=H, h0=h0, R_earth=R_earth)

    # 场景1：100 km 高度，速度沿 X 正方向
    altitude = 100e3
    r = R_earth + altitude
    pos = np.array([r, 0, 0])
    v_mag = np.sqrt(3.986004418e14 / r)
    vel = np.array([0.0, v_mag, 0.0])   # 典型 LEO 速度
    state = np.concatenate([pos, vel])

    acc = drag.compute_accel(state, 0.0)

    # 验证加速度方向与速度相反
    assert np.dot(acc, vel) < 0
    # 验证加速度主要沿 -Y 方向
    assert acc[1] < 0
    assert abs(acc[0]) < 1e-6
    assert abs(acc[2]) < 1e-6

    # 验证密度计算：高度 100 km 密度应小于海平面密度
    # 根据指数模型，密度 = rho0 * exp(-(h - h0)/H)
    expected_rho = rho0 * np.exp(-(altitude - h0) / H)
    v = np.linalg.norm(vel)
    expected_acc_magnitude = 0.5 * Cd * area_to_mass * expected_rho * v_mag**2.0
    computed_magnitude = np.linalg.norm(acc)
    assert np.allclose(computed_magnitude, expected_acc_magnitude, rtol=1e-6)

    # 场景2：更高高度，密度应更小，加速度更小
    altitude2 = 1000e3
    r2 = R_earth + altitude2
    pos2 = np.array([r2, 0, 0])
    state2 = np.concatenate([pos2, vel])
    acc2 = drag.compute_accel(state2, 0.0)
    assert np.linalg.norm(acc2) < np.linalg.norm(acc)

    # 场景3：速度方向改变，加速度方向也应随之改变
    vel3 = np.array([0, 7700.0, 0])
    state3 = np.concatenate([pos, vel3])
    acc3 = drag.compute_accel(state3, 0.0)
    # 加速度应与速度反向
    assert np.dot(acc3, vel3) < 0
    assert acc3[1] < 0
    assert abs(acc3[0]) < 1e-6
    assert abs(acc3[2]) < 1e-6


def test_atmospheric_drag_constructor():
    """测试大气阻力模型构造函数默认参数"""
    drag_default = AtmosphericDrag(area_to_mass=0.02)
    assert drag_default.area_to_mass == 0.02
    assert drag_default.Cd == 2.2
    assert drag_default.rho0 == 1.225
    assert drag_default.H == 8500.0
    assert drag_default.h0 == 0.0
    assert drag_default.R_earth == 6378137.0

    # 自定义参数
    drag_custom = AtmosphericDrag(area_to_mass=0.01, Cd=2.5, rho0=1.0, H=8000.0, h0=1000.0, R_earth=6371000.0)
    assert drag_custom.area_to_mass == 0.01
    assert drag_custom.Cd == 2.5
    assert drag_custom.rho0 == 1.0
    assert drag_custom.H == 8000.0
    assert drag_custom.h0 == 1000.0
    assert drag_custom.R_earth == 6371000.0


def test_j2_pure_function_consistency():
    """验证 J2 摄动纯函数与类方法计算结果一致"""
    mu_earth = 3.986004418e14
    j2 = 1.08262668e-3
    r_earth = 6378137.0

    j2_model = J2Gravity(mu_earth, j2, r_earth)

    # 随机生成一个状态
    np.random.seed(42)
    pos = np.random.uniform(7000e3, 8000e3, 3)
    vel = np.random.uniform(-8000, 8000, 3)
    state = np.concatenate([pos, vel])

    # 使用类方法计算
    acc_class = j2_model.compute_accel(state, 0.0)

    # 直接调用纯函数（导入内部函数）
    from mission_sim.core.physics.models.j2_gravity import _j2_accel
    acc_pure = _j2_accel(pos, mu_earth, j2, r_earth)

    assert np.allclose(acc_class, acc_pure, rtol=1e-12)


def test_crtbp_pure_function_consistency():
    """验证 CRTBP 纯函数与类方法计算结果一致"""
    model = Gravity_CRTBP()

    # 日地旋转系中某位置（例如 Halo 轨道附近）
    pos = np.array([1.1e11, 0.0, 5e9])
    vel = np.array([0.0, 1e4, 0.0])
    state = np.concatenate([pos, vel])

    acc_class = model.compute_accel(state, 0.0)

    from mission_sim.core.physics.models.gravity_crtbp import _crtbp_accel
    acc_pure = _crtbp_accel(pos, vel, model.GM_SUN, model.GM_EARTH, model.OMEGA, model.pos_sun, model.pos_earth)

    assert np.allclose(acc_class, acc_pure, rtol=1e-12)


def test_srp_pure_function_consistency():
    """验证 SRP 纯函数与类方法计算结果一致"""
    area_to_mass = 0.02
    reflectivity = 1.0
    AU = 1.495978707e11
    mu = 3.00348e-6
    P_solar = 4.56e-6
    sun_pos = np.array([-mu * AU, 0.0, 0.0])

    srp_model = Cannonball_SRP(area_to_mass, reflectivity, sun_pos, AU, mu, P_solar)

    # 日地旋转系中某位置
    pos = np.array([1.0e11, 0.0, 0.0])
    vel = np.zeros(3)
    state = np.concatenate([pos, vel])

    acc_class = srp_model.compute_accel(state, 0.0)

    from mission_sim.core.physics.models.srp import _srp_accel
    acc_pure = _srp_accel(pos, sun_pos, area_to_mass, reflectivity, P_solar, AU)

    assert np.allclose(acc_class, acc_pure, rtol=1e-12)
