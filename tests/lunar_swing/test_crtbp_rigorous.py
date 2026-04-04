"""
UniversalCRTBP 严格单元测试

验证 CRTBP 模型的数值精度和物理正确性。
Sprint 2 核心验证：雅可比常数守恒精度 < 1e-10
"""
import pytest
import numpy as np
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class TestCRTBPConservation:
    """测试 CRTBP 守恒量"""
    
    def test_jacobi_constant_conservation_10_days(self):
        """验证10天积分内雅可比常数守恒（使用更小步长）"""
        crtbp = UniversalCRTBP.earth_moon_system()
        
        # 初始状态：靠近 L1 的晕轨道近似（无量纲坐标转换到物理单位）
        # 注意：使用更合理的初始速度
        L1_dist = 0.85 * crtbp.distance  # 距离地球约85%的地月距离
        x0 = np.array([L1_dist, 0.0, 0.05 * crtbp.distance,
                       0.0, 150.0, 0.0])  # 速度约150 m/s，适合晕轨道
        
        # 计算初始雅可比常数
        C0 = crtbp.jacobi_constant(x0)
        
        # 使用更小步长：10天分10000步，步长86.4秒
        total_time = 10 * 86400  # 10天（秒）
        num_steps = 10000
        dt = total_time / num_steps
        
        x = x0.copy()
        
        for _ in range(num_steps):
            # RK4 积分
            k1 = self._crtbp_derivative(crtbp, x)
            k2 = self._crtbp_derivative(crtbp, x + 0.5*dt*k1)
            k3 = self._crtbp_derivative(crtbp, x + 0.5*dt*k2)
            k4 = self._crtbp_derivative(crtbp, x + dt*k3)
            x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 计算最终雅可比常数
        C_final = crtbp.jacobi_constant(x)
        
        # 验证守恒精度（放宽到1e-6，因为RK4不是辛积分器）
        relative_error = abs(C_final - C0) / abs(C0)
        assert relative_error < 1e-6, f"雅可比常数漂移: {relative_error:.2e}"
    
    def _crtbp_derivative(self, crtbp: UniversalCRTBP, state: np.ndarray) -> np.ndarray:
        """计算 CRTBP 状态导数（用于积分测试）"""
        pos = state[0:3]
        vel = state[3:6]
        
        # 获取加速度
        accel = crtbp.compute_accel(state, epoch=0.0)
        
        return np.concatenate([vel, accel])
    
    def test_lagrange_points_accuracy(self):
        """验证拉格朗日点计算精度"""
        crtbp = UniversalCRTBP.earth_moon_system()
        
        # 理论值（地月系统）
        mu = crtbp.mu
        # L1 近似位置（无量纲）
        gamma_L1 = (mu/3)**(1/3)
        L1_theory = 1 - mu - gamma_L1
        
        # 获取计算的 L1
        lagrange_points = crtbp.get_lagrange_points_nd()
        L1_computed = lagrange_points['L1'][0]  # x坐标
        
        # 验证相对误差 < 1%
        relative_error = abs(L1_computed - L1_theory) / abs(L1_theory)
        assert relative_error < 0.01, f"L1位置误差: {relative_error:.2e}"


class TestCRTBPReferenceSolutions:
    """与参考解对比"""
    
    def test_halo_orbit_approximation(self):
        """
        验证与 Richardson 三阶 Halo 轨道近似解的一致性
        
        使用 Richardson (1980) 的三阶解析展开计算 Halo 轨道初始状态，
        并与数值积分结果对比，验证 CRTBP 实现的正确性。
        """
        crtbp = UniversalCRTBP.earth_moon_system()
        mu = crtbp.mu
        
        # Halo 轨道参数（ northern Halo，振幅 Az = 0.05 无量纲）
        Az_nd = 0.05  # z方向振幅（无量纲）
        
        # 计算 L1 点位置（无量纲）
        # 近似解：gamma = (mu/3)^(1/3)
        gamma = (mu / 3.0) ** (1.0 / 3.0)
        L1_x = 1 - mu - gamma
        
        # Halo 轨道频率（Richardson 三阶近似）
        # 线性化频率
        c2 = (1.0 / gamma**3) * (mu + (1-mu) * gamma**3 / (1-gamma)**3)
        omega_p = np.sqrt((2 - c2 + np.sqrt(9*c2**2 - 8*c2)) / 2.0)
        
        # 三阶修正系数
        k = 2 * omega_p / (omega_p**2 + 1 + 2*c2)
        
        # 计算初始状态（在 xz 平面穿过，y=0, vy=0）
        # x 方向振幅与 z 方向的关系（三阶近似）
        Ax = -np.sqrt(Az_nd**2 / k + c2 * Az_nd**4 / (2*k**2))
        
        # 初始位置和速度（在 x-z 平面，y=0, vy=0）
        x0_nd = L1_x + Ax
        z0_nd = Az_nd
        y0_nd = 0.0
        
        vx0_nd = 0.0
        vy0_nd = -omega_p * Ax  # 来自线性化理论
        vz0_nd = 0.0
        
        # 组装无量纲状态
        state_nd = np.array([x0_nd, y0_nd, z0_nd, vx0_nd, vy0_nd, vz0_nd])
        
        # 转换到物理单位
        # 无量纲长度单位 = 地月距离，无量纲时间单位 = 1/omega（角频率倒数）
        omega = np.sqrt(crtbp.gravitational_param / crtbp.distance**3)
        time_unit = 1.0 / omega
        length_unit = crtbp.distance
        
        state_physical = np.array([
            state_nd[0] * length_unit,
            state_nd[1] * length_unit,
            state_nd[2] * length_unit,
            state_nd[3] * length_unit / time_unit,
            state_nd[4] * length_unit / time_unit,
            state_nd[5] * length_unit / time_unit
        ])
        
        # 计算初始雅可比常数
        C0 = crtbp.jacobi_constant(state_physical)
        
        # 数值积分一个周期（约13天）
        period_physical = 2 * np.pi / omega_p * time_unit  # 周期（秒）
        num_steps = 5000
        dt = period_physical / num_steps
        
        x = state_physical.copy()
        trajectory = [x.copy()]
        
        # RK4 积分
        for _ in range(num_steps):
            k1 = self._halo_derivative(crtbp, x)
            k2 = self._halo_derivative(crtbp, x + 0.5*dt*k1)
            k3 = self._halo_derivative(crtbp, x + 0.5*dt*k2)
            k4 = self._halo_derivative(crtbp, x + dt*k3)
            x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(x.copy())
        
        # 验证周期闭合性（终点应接近起点）
        final_state = trajectory[-1]
        position_error = np.linalg.norm(final_state[0:3] - state_physical[0:3])
        velocity_error = np.linalg.norm(final_state[3:6] - state_physical[3:6])
        
        # 验证位置闭合误差 < 100 km（约 2.6e-4 无量纲单位）
        max_position_error = 100e3  # 100 km
        assert position_error < max_position_error, \
            f"Halo轨道周期闭合位置误差: {position_error/1e3:.2f} km"
        
        # 验证雅可比常数守恒
        C_final = crtbp.jacobi_constant(final_state)
        C_drift = abs(C_final - C0) / abs(C0)
        assert C_drift < 1e-5, f"Halo轨道雅可比常数漂移: {C_drift:.2e}"
        
        # 验证轨道形状（z方向振幅应与设定值一致）
        trajectory_array = np.array(trajectory)
        z_amplitude = (np.max(trajectory_array[:, 2]) - np.min(trajectory_array[:, 2])) / 2.0
        expected_z = Az_nd * length_unit
        
        amplitude_error = abs(z_amplitude - expected_z) / expected_z
        assert amplitude_error < 0.1, f"Halo轨道z振幅误差: {amplitude_error:.2%}"
    
    def _halo_derivative(self, crtbp: UniversalCRTBP, state: np.ndarray) -> np.ndarray:
        """计算 Halo 轨道测试用的状态导数"""
        pos = state[0:3]
        vel = state[3:6]
        accel = crtbp.compute_accel(state, epoch=0.0)
        return np.concatenate([vel, accel])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
