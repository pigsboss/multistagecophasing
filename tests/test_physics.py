import numpy as np
import pytest
from numpy.testing import assert_allclose
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.physics.models.gravity import GravityCRTBP
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.core.physics.models.srp import CannonballSRP
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
    """Test coordinate frame consistency check."""
    # Ensure correct import if not already present in the file
    from mission_sim.core.spacetime.ids import CoordinateFrame
    from mission_sim.core.physics.environment import CelestialEnvironment
    import numpy as np
    import pytest

    env = CelestialEnvironment(CoordinateFrame.SUN_EARTH_ROTATING)
    state = np.zeros(6)
    
    # Match the actual exception message from environment.py
    # The error message contains "Frame mismatch!" 
    with pytest.raises(ValueError, match=r"Frame mismatch"):
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
    with pytest.raises(ValueError, match="Thrust frame mismatch"):
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
    """测试新的 CRTBP 实现的基本功能和一致性"""
    from mission_sim.core.physics.models.gravity import GravityCRTBP
    
    # 创建模型实例（适配器会自动使用 Numba 加速）
    model = GravityCRTBP()
    
    # 测试位置（日地系统 L2 附近）
    AU = 1.495978707e11
    pos = np.array([1.1e11, 0.0, 5e9])
    vel = np.array([0.0, 1e4, 0.0])
    state = np.concatenate([pos, vel])
    
    # 测试类方法计算加速度
    acc_class = model.compute_accel(state, 0.0)
    
    # 验证加速度为3维向量
    assert acc_class.shape == (3,), f"Expected shape (3,), got {acc_class.shape}"
    assert not np.any(np.isnan(acc_class)), "Acceleration contains NaN values"
    
    # 验证加速度大小合理（数量级检查）
    acc_magnitude = np.linalg.norm(acc_class)
    # CRTBP加速度的数量级应该在 1e-4 到 1e-2 m/s² 范围内
    assert 1e-4 < acc_magnitude < 1e-2, f"Acceleration magnitude {acc_magnitude} seems unrealistic"
    
    # 验证加速度方向（在L2点附近，加速度应主要指向地球方向）
    # 在x轴正方向的位置，加速度应为负（指向地球）
    assert acc_class[0] < 0, f"X acceleration should be negative, got {acc_class[0]}"
    
    # 测试向量化方法的一致性
    state_matrix = state.reshape(1, 6)
    acc_vectorized = model.compute_vectorized_acc(state_matrix, 0.0)
    
    # 验证单状态与向量化结果一致
    assert_allclose(acc_class, acc_vectorized[0], rtol=1e-12, 
                   err_msg="Vectorized computation doesn't match single state computation")
    
    # 测试多个状态
    N = 5
    states = np.tile(state, (N, 1)) + np.random.randn(N, 6) * 1e6
    acc_matrix = model.compute_vectorized_acc(states, 0.0)
    
    # 验证向量化输出的形状
    assert acc_matrix.shape == (N, 3), f"Expected shape ({N}, 3), got {acc_matrix.shape}"
    
    # 验证每个状态的计算与单状态计算一致
    for i in range(N):
        acc_single = model.compute_accel(states[i], 0.0)
        assert_allclose(acc_single, acc_matrix[i], rtol=1e-12,
                       err_msg=f"State {i}: Vectorized doesn't match single computation")
    
    # 验证常量属性仍然存在（向后兼容）
    assert hasattr(model, 'GM_SUN'), "GM_SUN constant not found"
    assert hasattr(model, 'GM_EARTH'), "GM_EARTH constant not found"
    assert hasattr(model, 'AU'), "AU constant not found"
    assert hasattr(model, 'OMEGA'), "OMEGA constant not found"
    
    # 验证常量值正确
    assert np.isclose(model.GM_SUN, 1.32712440018e20), f"GM_SUN incorrect: {model.GM_SUN}"
    assert np.isclose(model.GM_EARTH, 3.986004418e14), f"GM_EARTH incorrect: {model.GM_EARTH}"
    assert np.isclose(model.AU, 1.495978707e11), f"AU incorrect: {model.AU}"
    assert np.isclose(model.OMEGA, 1.990986e-7), f"OMEGA incorrect: {model.OMEGA}"

def test_srp_pure_function_consistency():
    """验证 SRP 纯函数与类方法计算结果一致"""
    area_to_mass = 0.02
    reflectivity = 1.0
    AU = 1.495978707e11
    mu = 3.00348e-6
    P_solar = 4.56e-6
    sun_pos = np.array([-mu * AU, 0.0, 0.0])

    srp_model = CannonballSRP(area_to_mass, reflectivity, sun_pos, AU, mu, P_solar)

    # 日地旋转系中某位置
    pos = np.array([1.0e11, 0.0, 0.0])
    vel = np.zeros(3)
    state = np.concatenate([pos, vel])

    acc_class = srp_model.compute_accel(state, 0.0)

    from mission_sim.core.physics.models.srp import _srp_accel
    acc_pure = _srp_accel(pos, sun_pos, area_to_mass, reflectivity, P_solar, AU)

    assert np.allclose(acc_class, acc_pure, rtol=1e-12)



"""
Test Suite for Vectorized Physics Models
"""
def test_gravity_crtbp_vectorization():
    """
    Test the vectorized implementation of CRTBP gravity against the 
    single-state Numba implementation.
    Verifies input/output shapes and strict mathematical equivalence.
    """
    # 1. Initialize the force model
    crtbp = GravityCRTBP()
    
    # 2. Generate a synthetic state matrix of shape (N, 6)
    # Simulating a formation of 100 spacecraft near the Sun-Earth L2 point
    N = 100
    x_L2 = crtbp.AU * 1.01  # Approximate L2 position on X-axis
    
    np.random.seed(42)  # Fix seed for deterministic testing
    state_matrix = np.zeros((N, 6), dtype=np.float64)
    
    # Random distribution within +/- 1000 km and +/- 10 m/s relative to L2
    state_matrix[:, 0] = x_L2 + np.random.uniform(-1e6, 1e6, N)
    state_matrix[:, 1] = np.random.uniform(-1e6, 1e6, N)
    state_matrix[:, 2] = np.random.uniform(-1e6, 1e6, N)
    state_matrix[:, 3] = np.random.uniform(-10.0, 10.0, N)
    state_matrix[:, 4] = np.random.uniform(-10.0, 10.0, N)
    state_matrix[:, 5] = np.random.uniform(-10.0, 10.0, N)
    
    epoch = 0.0  # Epoch is invariant for conservative CRTBP

    # 3. Compute vectorized accelerations (The L2 parallel path)
    acc_matrix_vec = crtbp.compute_vectorized_acc(state_matrix, epoch)
    
    # [ASSERT 1] Verify the strict dimensional contract: (N, 6) -> (N, 3)
    assert acc_matrix_vec.shape == (N, 3), \
        f"Contract violated: Expected shape ({N}, 3), got {acc_matrix_vec.shape}"

    # 4. Compute sequential accelerations (The L1 Numba fallback path)
    acc_matrix_seq = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        acc_matrix_seq[i, :] = crtbp.compute_accel(state_matrix[i, :], epoch)
        
    # [ASSERT 2] Element-wise mathematical equivalence
    # We use a very tight tolerance (1e-12). Note that exact equality (==) 
    # is avoided due to potential AVX/SIMD floating-point rounding differences.
    assert_allclose(
        acc_matrix_vec, 
        acc_matrix_seq, 
        rtol=1e-12, 
        atol=1e-12, 
        err_msg="Vectorized CRTBP accelerations do not match sequential Numba calculations!"
    )
