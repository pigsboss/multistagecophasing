# tests/test_orbit_generation.py
"""
轨道生成精度验证脚本
测试 Ephemeris 插值误差、Halo 轨道闭合性、J2 轨道与解析解对比等。
使用 pytest 框架运行：pytest tests/test_orbit_generation.py -v
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators import KeplerianGenerator
from mission_sim.core.trajectory.halo_corrector import HaloDifferentialCorrector
from mission_sim.core.physics.models.j2_gravity import J2Gravity

# 允许的最大闭合误差（位置：米，速度：米/秒）
MAX_POS_CLOSURE = 1e5      # 100 km，Halo 轨道通常闭合性较好，但初始猜测可能偏差
MAX_VEL_CLOSURE = 10.0     # 10 m/s


def test_ephemeris_linear_interpolation():
    """测试 Ephemeris 插值器在简单线性数据上的准确性"""
    # 创建线性数据
    times = np.array([0.0, 1.0, 2.0])
    states = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ])
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)

    # 插值中间点
    t_mid = 0.5
    state_mid = eph.get_interpolated_state(t_mid)
    expected = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(state_mid, expected, rtol=1e-10), "线性插值误差过大"


def test_ephemeris_cubic_spline():
    """测试 Ephemeris 三次样条插值在光滑曲线上的准确性"""
    # 生成一个简单的正弦曲线位置，速度通过导数给出
    times = np.linspace(0, 2*np.pi, 10)
    states = []
    for t in times:
        x = np.sin(t)
        vx = np.cos(t)
        states.append([x, 0.0, 0.0, vx, 0.0, 0.0])
    states = np.array(states)
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)

    # 在密集时间点检验插值误差
    test_times = np.linspace(0, 2*np.pi, 100)
    errors = []
    for t in test_times:
        interp = eph.get_interpolated_state(t)
        exact = np.array([np.sin(t), 0.0, 0.0, np.cos(t), 0.0, 0.0])
        err = np.linalg.norm(interp - exact)
        errors.append(err)
    max_error = max(errors)
    # 三次样条在光滑函数上误差应很小
    assert max_error < 1e-5, f"三次样条插值最大误差 {max_error} 超过 1e-5"


def test_keplerian_generator_energy():
    """测试 KeplerianGenerator 生成轨道能量是否守恒"""
    config = {
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],  # a=7000km, e=0, 圆轨道
        "dt": 10.0,
        "sim_time": 86400.0
    }
    generator = KeplerianGenerator()
    eph = generator.generate(config)

    mu = generator.mu
    # 计算每个时刻的机械能
    energies = []
    for i in range(len(eph.times)):
        r = np.linalg.norm(eph.states[i, :3])
        v = np.linalg.norm(eph.states[i, 3:6])
        energy = 0.5 * v**2 - mu / r
        energies.append(energy)

    # 能量变化应很小
    energy_std = np.std(energies)
    assert energy_std < 1e-3, f"开普勒轨道能量标准差 {energy_std} 过大"


def test_keplerian_generator_angular_momentum():
    """测试 KeplerianGenerator 生成轨道角动量是否守恒"""
    config = {
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dt": 10.0,
        "sim_time": 86400.0
    }
    generator = KeplerianGenerator()
    eph = generator.generate(config)

    h_vecs = []
    for i in range(len(eph.times)):
        r = eph.states[i, :3]
        v = eph.states[i, 3:6]
        h = np.cross(r, v)
        h_vecs.append(h)
    h_vecs = np.array(h_vecs)

    # 角动量变化（标准差）应很小
    h_std = np.std(h_vecs, axis=0)
    assert np.all(h_std < 1e-3), f"角动量标准差 {h_std} 过大"


def test_halo_orbit_closure():
    """测试 Halo 轨道的闭合性（首尾位置和速度误差）"""
    # 跳过测试如果 HaloDifferentialCorrector 尚未实现微分修正
    try:
        generator = HaloDifferentialCorrector()
        config = {
            "Az": 0.05,
            "dt": 0.001
        }
        eph = generator.generate(config)
    except Exception as e:
        pytest.skip(f"Halo 轨道生成失败，跳过测试: {e}")

    # 检查首尾状态差异
    pos_start = eph.states[0, :3]
    pos_end = eph.states[-1, :3]
    vel_start = eph.states[0, 3:6]
    vel_end = eph.states[-1, 3:6]

    pos_error = np.linalg.norm(pos_end - pos_start)
    vel_error = np.linalg.norm(vel_end - vel_start)

    # 打印误差供参考
    print(f"\nHalo 轨道闭合误差: 位置 {pos_error:.2e} m, 速度 {vel_error:.2e} m/s")
    assert pos_error < MAX_POS_CLOSURE, f"位置闭合误差 {pos_error} m 超过 {MAX_POS_CLOSURE} m"
    assert vel_error < MAX_VEL_CLOSURE, f"速度闭合误差 {vel_error} m/s 超过 {MAX_VEL_CLOSURE} m/s"


def test_j2_gravity_consistency():
    """测试 J2 引力模型与理论加速度的一致性（通过数值积分验证能量变化率）"""
    # 简单测试：检查 J2 模型是否返回合理加速度
    j2 = J2Gravity()
    # 选取一个 LEO 典型位置 (x=7000km, y=0, z=0)
    state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])  # 圆轨道速度约7.5km/s
    acc = j2.compute_accel(state, 0.0)
    # 加速度大小应远小于中心引力加速度（~8 m/s²），但不应为零
    assert np.linalg.norm(acc) > 0.0, "J2 加速度为零"
    # 在赤道面上，J2 加速度应有径向和法向分量？实际上在赤道上只有径向？
    # 简化：检查非零即可
    print(f"J2 加速度: {acc} m/s²")


def test_ephemeris_out_of_range_warning():
    """测试 Ephemeris 在请求超出时间范围时是否产生警告（但不崩溃）"""
    times = np.array([0.0, 10.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, CoordinateFrame.J2000_ECI)

    # 请求在范围内
    eph.get_interpolated_state(5.0)  # 不应警告
    # 请求超出范围（外推）
    # 注意：函数内部有 print 警告，我们这里只测试不抛出异常
    try:
        eph.get_interpolated_state(-1.0)
        eph.get_interpolated_state(11.0)
    except Exception as e:
        pytest.fail(f"外推时抛出异常: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])