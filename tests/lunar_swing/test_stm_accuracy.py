"""
STM 计算器精度验证测试

验证变分方程积分实现的正确性。
"""
import numpy as np
import pytest
from mission_sim.utils.dynamics.stm_calculator import STMCalculator
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class TestSTMAccuracy:
    """测试状态转移矩阵计算的正确性"""
    
    @pytest.fixture
    def earth_moon_crtbp(self):
        """地月系统 CRTBP 模型"""
        return UniversalCRTBP.earth_moon_system()
    
    @pytest.fixture
    def sample_state(self):
        """测试用初始状态（无量纲单位，接近 L1 点）"""
        # 地月 L1 点大约在 x = 0.8369 (无量纲)
        return np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
    
    def test_identity_initial_condition(self, earth_moon_crtbp, sample_state):
        """测试：t=0 时 STM 应为单位矩阵"""
        calc = STMCalculator()
        
        # 定义完整的动力学函数（返回6维状态导数）
        def full_dynamics(t, x):
            # x = [pos(3), vel(3)]
            # dx/dt = [vel(3), accel(3)]
            accel = earth_moon_crtbp._crtbp_acceleration_nd(x)
            return np.concatenate([x[3:6], accel])
        
        # 零时间传播
        _, stm = calc.propagate_with_stm(
            dynamics=full_dynamics,
            initial_state=sample_state,
            t0=0.0,
            tf=0.0,
            method='rk4',
            num_steps=1
        )
        
        assert np.allclose(stm, np.eye(6), atol=1e-10), \
            "零时间传播应得到单位矩阵"
    
    def test_semigroup_property(self, earth_moon_crtbp, sample_state):
        """测试：STM 的半群性质 Φ(t2,t0) = Φ(t2,t1)Φ(t1,t0)"""
        calc = STMCalculator()
        
        # 定义完整的动力学函数（返回6维状态导数）
        def full_dynamics(t, x):
            accel = earth_moon_crtbp._crtbp_acceleration_nd(x)
            return np.concatenate([x[3:6], accel])
        
        # 地月系统无量纲时间单位约 4.3 天，测试一个短周期
        T = 2.0  # 约 8.6 天
        
        # 分两段计算
        x_mid, stm1 = calc.propagate_with_stm(
            full_dynamics, sample_state, 0.0, T/2, method='rk4', num_steps=100
        )
        _, stm2 = calc.propagate_with_stm(
            full_dynamics, x_mid, T/2, T, method='rk4', num_steps=100
        )
        
        # 直接计算全程
        _, stm_full = calc.propagate_with_stm(
            full_dynamics, sample_state, 0.0, T, method='rk4', num_steps=200
        )
        
        # 验证半群性质
        stm_product = stm2 @ stm1
        error = np.linalg.norm(stm_product - stm_full, ord='fro')
        
        print(f"STM 半群性质误差: {error:.2e}")
        assert error < 1e-4, f"STM 不满足半群性质，误差: {error}"
    
    def test_symplectic_property(self, earth_moon_crtbp, sample_state):
        """测试：保守系统 STM 应近似满足辛性质 (STM^T J STM = J)"""
        calc = STMCalculator()
        
        # 定义完整的动力学函数（返回6维状态导数）
        def full_dynamics(t, x):
            accel = earth_moon_crtbp._crtbp_acceleration_nd(x)
            return np.concatenate([x[3:6], accel])
        
        # 辛矩阵 J
        J = np.block([
            [np.zeros((3,3)), np.eye(3)],
            [-np.eye(3), np.zeros((3,3))]
        ])
        
        _, stm = calc.propagate_with_stm(
            full_dynamics, sample_state, 0.0, 1.0, method='rk4', num_steps=100
        )
        
        # 计算 STM^T J STM
        lhs = stm.T @ J @ stm
        
        # 检查是否接近 J
        error = np.linalg.norm(lhs - J, ord='fro')
        print(f"辛性质误差: {error:.2e}")
        
        # 注意：RK4 不严格保持辛结构，误差应随步长减小而减小
        assert error < 1.0, f"STM 严重违反辛性质，误差: {error}"
    
    def test_state_sensitivity_consistency(self, earth_moon_crtbp, sample_state):
        """测试：STM 预测的偏差应与实际传播偏差一致"""
        calc = STMCalculator()
        
        # 定义完整的动力学函数（返回6维状态导数）
        def full_dynamics(t, x):
            accel = earth_moon_crtbp._crtbp_acceleration_nd(x)
            return np.concatenate([x[3:6], accel])
        
        # 基础传播
        x0 = sample_state.copy()
        xf, stm = calc.propagate_with_stm(
            full_dynamics, x0, 0.0, 1.0, method='rk4', num_steps=100
        )
        
        # 施加微小扰动
        delta_x0 = np.array([1e-6, 0, 0, 0, 0, 0])
        xf_perturbed, _ = calc.propagate_with_stm(
            full_dynamics, x0 + delta_x0, 0.0, 1.0, method='rk4', num_steps=100
        )
        
        # 实际偏差
        delta_xf_actual = xf_perturbed - xf
        
        # STM 预测偏差
        delta_xf_predicted = stm @ delta_x0
        
        # 相对误差
        rel_error = np.linalg.norm(delta_xf_actual - delta_xf_predicted) / \
                   np.linalg.norm(delta_xf_actual)
        
        print(f"敏感度一致性相对误差: {rel_error:.2e}")
        assert rel_error < 1e-3, f"STM 预测与实际偏差不一致: {rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
