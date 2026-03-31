import numpy as np
import pytest
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.cyber.platform_gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.core.cyber.platform_gnc.propagator import SimplePropagator, KeplerPropagator, CRTBPPropagator
from mission_sim.core.cyber.models.threebody.base import CRTBP


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
    obs = np.array([1, 2, 3, 4, 5, 6])
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
    gnc.update_navigation(np.array([1, 0, 0, 0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)
    K = np.array([[0.01, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    force, frame = gnc.compute_control_force(0.0, K)
    assert np.allclose(force, [-0.01, 0, 0])
    assert frame == CoordinateFrame.SUN_EARTH_ROTATING


def test_gnc_propagator_simple():
    """测试简单外推器"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.set_propagator(SimplePropagator())
    gnc.current_nav_state = np.array([0, 0, 0, 1, 0, 0])
    gnc.update_navigation(None, CoordinateFrame.SUN_EARTH_ROTATING, dt=10.0)
    assert np.allclose(gnc.current_nav_state, [10, 0, 0, 1, 0, 0])


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
    assert abs(new_pos[0] - r) < 1000.0, f"径向距离变化 {abs(new_pos[0]-r)} 超过 1000m"
    # 验证速度大小应大致保持不变
    new_vel = gnc.current_nav_state[3:6]
    assert abs(np.linalg.norm(new_vel) - v) < 0.1, f"速度变化 {abs(np.linalg.norm(new_vel)-v)} 超过 0.1m/s"


def test_gnc_k_matrix_invalid_shape_raises():
    """测试不可修复的 K 矩阵形状导致异常（Fail-Fast）"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.load_reference_trajectory(eph)
    gnc.update_navigation(np.zeros(6), CoordinateFrame.SUN_EARTH_ROTATING)

    # 传入形状为 (4,6) 的 K 矩阵（无法修复）
    K_invalid = np.zeros((4, 6))
    with pytest.raises(ValueError, match=r"无法安全转换为所需的 \(3, 6\)"):
        gnc.compute_control_force(0.0, K_invalid)


def test_gnc_k_matrix_broadcast():
    """测试可修复的 K 矩阵形状（(6,)）成功广播为 (3,6) 并正常执行"""
    gnc = GNC_Subsystem("test", CoordinateFrame.SUN_EARTH_ROTATING)
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.SUN_EARTH_ROTATING)
    gnc.load_reference_trajectory(eph)
    gnc.update_navigation(np.array([1, 0, 0, 0, 0, 0]), CoordinateFrame.SUN_EARTH_ROTATING)

    # 传入形状为 (6,) 的一维数组，应被广播为 (3,6)
    K_1d = np.array([0.01, 0, 0, 0, 0, 0])
    force, frame = gnc.compute_control_force(0.0, K_1d)
    # 广播后，三行相同，误差向量只有 X 分量，因此三个通道都产生力
    assert np.allclose(force, [-0.01, -0.01, -0.01])
    assert frame == CoordinateFrame.SUN_EARTH_ROTATING


def test_crtbp_propagator():
    """测试 CRTBP 外推器精度优于线性外推"""
    # 创建 CRTBP 实例（日地系统标准参数）
    mu = 3.00348e-6
    L = 1.495978707e11
    omega = 1.990986e-7
    crtbp = CRTBP(mu, L, omega)

    # 构建外推器
    crtbp_prop = CRTBPPropagator(crtbp)
    simple_prop = SimplePropagator()

    # 定义初始状态：一个典型的日地 L2 附近状态（无量纲转物理）
    x0_nd = np.array([1.01106, 0.0, 0.05, 0.0, 0.0105, 0.0])
    state0_phys, _ = crtbp.to_physical(x0_nd, 0.0)

    # 外推短时
    dt = 10.0
    state_crtbp = crtbp_prop.propagate(state0_phys, dt)
    state_simple = simple_prop.propagate(state0_phys, dt)

    # 高精度积分：在无量纲域积分，然后转换到物理
    from scipy.integrate import solve_ivp

    def dynamics_nd(t_nd, state_nd):
        return crtbp.dynamics(t_nd, state_nd)

    dt_nd = dt * omega
    sol_nd = solve_ivp(dynamics_nd, (0, dt_nd), x0_nd, method='DOP853', rtol=1e-12, atol=1e-12)
    state_true_nd = sol_nd.y[:, -1]
    state_true, _ = crtbp.to_physical(state_true_nd, dt_nd)

    # 计算误差
    err_crtbp = np.linalg.norm(state_crtbp - state_true)
    err_simple = np.linalg.norm(state_simple - state_true)

    # 验证 CRTBP 外推误差小于线性外推（至少小一个数量级）
    assert err_crtbp < 0.1 * err_simple, \
        f"CRTBP 外推误差 ({err_crtbp:.2e}) 不小于线性外推误差 ({err_simple:.2e}) 的 10%"

    # 绝对精度检查
    assert err_crtbp < 100.0, f"CRTBP 外推绝对误差 {err_crtbp:.2f} 超过 100m"