"""
地月共振摆动轨道设计器 - 核心算法实现

实现 LunarSwingTargeter 类，提供共振轨道搜索、状态转移矩阵计算和稳定性分析功能。
"""
import numpy as np
from typing import Tuple, Dict, Callable, Union, Optional, List
import numpy.linalg as la

from mission_sim.utils.dynamics.stm_calculator import STMCalculator


class LunarSwingTargeter:
    """地月共振摆动轨道设计器 - 单参数打靶法实现
    
    使用微分修正（Differential Correction）算法搜索周期轨道。
    固定初始位置的 x, y, z 和速度 vx，以 vy, vz 为设计变量，
    通过牛顿迭代使一个周期后的位置残差收敛到零。
    """

    def __init__(self,
                 dynamics_model: Union[Callable, object],
                 mu: float = 0.01215,  # 地月系统质量参数
                 integrator_type: str = 'rk4',
                 num_steps: int = 200,
                 options: dict = None):
        """
        初始化轨道设计器。

        Args:
            dynamics_model: 动力学模型，应支持 f(t, state) -> state_derivative
            mu: 地月系统质量参数 (m2/(m1+m2))
            integrator_type: 积分器类型 ('rk4', 'rkf78')
            num_steps: 积分步数（影响STM计算精度）
            options: 配置选项
        """
        self.dynamics = dynamics_model
        self.mu = mu
        self.integrator_type = integrator_type
        self.num_steps = num_steps
        self.options = options or {}
        
        # 初始化 STM 计算器
        self._stm_calc = STMCalculator()

    def find_resonant_orbit(self,
                           resonance_ratio: Tuple[int, int],
                           initial_guess: np.ndarray,
                           target_period: float = None,
                           tol: float = 1e-6,
                           max_iter: int = 50,
                           damping: float = 0.5,
                           char_period: float = None) -> Dict:
        """
        使用单参数打靶法搜索共振周期轨道。
        
        算法流程：
        1. 固定初始状态中的 x, y, z, vx（位置完全固定，x方向速度固定）
        2. 以 vy, vz 为设计变量（2维）
        3. 积分一个周期，计算末端位置残差 Δr = [Δx, Δy, Δz]
        4. 利用 STM 计算敏感度矩阵 ∂Δr/∂[vy, vz]（3×2矩阵）
        5. 使用最小二乘求解修正量 δv = [δvy, δvz]
        6. 迭代直至位置残差收敛

        Args:
            resonance_ratio: (n, m) 共振比，如 (2, 1) 表示 2:1 共振
            initial_guess: 6 维初始状态猜测 [x, y, z, vx, vy, vz]（无量纲）
            target_period: 目标周期（秒），None 则根据共振比自动计算
            tol: 位置残差收敛容差（无量纲单位）
            max_iter: 最大迭代次数
            damping: 阻尼因子 (0-1)，防止牛顿迭代发散
            char_period: CRTBP特征周期（秒），用于时间单位转换。
                        默认为4.342*86400（地月系统约4.342天）

        Returns:
            字典包含：'state'（周期轨道状态）, 'period', 'convergence_history', 
                     'success', 'final_residual'
        """
        n, m = resonance_ratio
        
        # CRTBP特征周期（地月系统约4.342天）
        if char_period is None:
            char_period = 4.342 * 86400  # 秒

        # 计算目标周期（地月旋转系）
        if target_period is None:
            # 月球轨道周期约 27.321661 天
            T_moon = 27.321661 * 24 * 3600  # 秒
            target_period = (m / n) * T_moon

        # 转换为无量纲时间（CRTBP时间单位）
        target_period_nd = target_period / char_period

        print(f"搜索 {n}:{m} 共振轨道，目标周期: {target_period/86400:.3f} 天 "
              f"({target_period_nd:.3f} 无量纲单位)")
        print(f"初始猜测: {initial_guess}")

        # 初始状态：固定位置部分，仅优化速度中的 vy, vz
        x0 = initial_guess.copy()
        history = []
        
        # 设计变量索引：vy=4, vz=5
        design_indices = [4, 5]
        # 残差维度：位置残差 x, y, z (索引 0, 1, 2)
        residual_indices = [0, 1, 2]

        for i in range(max_iter):
            # 使用STM计算器同时传播状态和计算STM
            # 使用无量纲时间进行积分
            x_final, stm = self._stm_calc.propagate_with_stm(
                dynamics=self._get_dynamics_func(),
                initial_state=x0,
                t0=0.0,
                tf=target_period_nd,  # 使用无量纲时间
                method=self.integrator_type,
                num_steps=self.num_steps
            )
            
            # 检查数值有效性
            if not np.all(np.isfinite(x_final)) or not np.all(np.isfinite(stm)):
                print(f"✗ 第 {i+1} 次迭代出现数值溢出，停止搜索")
                return {
                    'state': x0,
                    'period': target_period,
                    'convergence_history': history,
                    'success': False,
                    'final_residual': np.full(6, np.nan),
                    'error': 'numerical_overflow'
                }
            
            # 计算残差：末端状态与初始状态的差异
            residual = x_final - x0
            # 仅关注位置残差（单参数打靶法通常只要求位置闭合）
            pos_residual = residual[residual_indices]
            res_norm = la.norm(pos_residual)

            # 记录收敛历史
            history.append({
                'iteration': i,
                'residual_norm': res_norm,
                'state': x0.copy(),
                'final_state': x_final.copy(),
                'stm': stm.copy()
            })

            # 检查收敛
            if res_norm < tol:
                print(f"✓ 在第 {i+1} 次迭代收敛，位置残差范数: {res_norm:.2e}")
                return {
                    'state': x0,
                    'period': target_period,
                    'convergence_history': history,
                    'success': True,
                    'final_residual': residual
                }

            # 微分修正：计算敏感度矩阵
            # 残差对设计变量的导数：∂(x_final - x0)/∂(vy0, vz0) = ∂x_final/∂(vy0, vz0)
            # 由于 x0 固定，∂x0/∂(vy0, vz0) = 0
            # 因此敏感度矩阵 = STM[位置, 速度设计变量]
            # STM 结构: [∂x/∂x0, ∂x/∂v0; ∂v/∂x0, ∂v/∂v0]
            # 我们需要 ∂[x,y,z]/∂[vy, vz] -> stm[0:3, 4:6]
            sensitivity = stm[np.ix_(residual_indices, design_indices)]  # 3×2 矩阵
            
            # 检查敏感度矩阵有效性
            if not np.all(np.isfinite(sensitivity)):
                print(f"✗ 第 {i+1} 次迭代敏感度矩阵无效，停止搜索")
                return {
                    'state': x0,
                    'period': target_period,
                    'convergence_history': history,
                    'success': False,
                    'final_residual': residual,
                    'error': 'invalid_sensitivity_matrix'
                }
            
            # 使用最小二乘求解修正量：sensitivity * δv = -pos_residual
            # 超定方程组 (3方程, 2未知数)，使用伪逆求解
            try:
                # 添加正则化以提高数值稳定性
                reg = 1e-10  # 正则化参数
                sensitivity_reg = sensitivity.T @ sensitivity + reg * np.eye(2)
                delta_v_design = la.solve(sensitivity_reg, sensitivity.T @ (-pos_residual))
            except la.LinAlgError:
                print(f"警告: 第 {i+1} 次迭代矩阵奇异，使用梯度下降")
                # 退化到梯度下降
                delta_v_design = -0.01 * pos_residual[0:2] if len(pos_residual) >= 2 else -0.01 * pos_residual[0:1]

            # 应用阻尼
            delta_v_design *= damping
            
            # 修正设计变量
            x0[design_indices] += delta_v_design

            if i % 5 == 0 or res_norm > 1e-2:
                print(f"  迭代 {i+1}: 位置残差 = {res_norm:.3e}, "
                      f"修正量 = [{delta_v_design[0]:.3e}, {delta_v_design[1]:.3e}]")

        print(f"✗ 在 {max_iter} 次迭代后未收敛，最终残差范数: {history[-1]['residual_norm']:.2e}")
        return {
            'state': x0,
            'period': target_period,
            'convergence_history': history,
            'success': False,
            'final_residual': residual
        }

    def _get_dynamics_func(self) -> Callable:
        """获取适用于 STM 计算器的动力学函数 f(t, x) -> [vx, vy, vz, ax, ay, az]"""
        if hasattr(self.dynamics, '_crtbp_acceleration_nd'):
            # 使用 UniversalCRTBP 的维度加速度方法
            # 需要包装成完整的6维状态导数 [vel, accel]
            def dynamics_wrapper(t, x):
                accel = self.dynamics._crtbp_acceleration_nd(x)
                return np.concatenate([x[3:6], accel])
            return dynamics_wrapper
        elif callable(self.dynamics):
            # 如果已经是函数形式 f(t, x)，直接使用
            def dynamics_wrapper(t, x):
                return self.dynamics(t, x)
            return dynamics_wrapper
        elif hasattr(self.dynamics, 'compute_derivative'):
            # 如果是对象，包装其 compute_derivative 方法
            def dynamics_wrapper(t, x):
                return self.dynamics.compute_derivative(x)
            return dynamics_wrapper
        else:
            # 默认使用内置 CRTBP 动力学
            def dynamics_wrapper(t, x):
                return self._simple_crtbp_derivative(x)
            return dynamics_wrapper

    def _simple_crtbp_derivative(self, state: np.ndarray) -> np.ndarray:
        """简化的 CRTBP 动力学导数（无量纲旋转系）"""
        x, y, z, vx, vy, vz = state
        mu = self.mu

        r1_sq = (x + mu)**2 + y**2 + z**2
        r2_sq = (x + mu - 1)**2 + y**2 + z**2
        
        # 避免除以零
        eps = 1e-15
        r1 = np.sqrt(r1_sq + eps)
        r2 = np.sqrt(r2_sq + eps)

        ax = 2*vy + x - (1-mu)*(x+mu)/(r1**3) - mu*(x+mu-1)/(r2**3)
        ay = -2*vx + y - (1-mu)*y/(r1**3) - mu*y/(r2**3)
        az = -(1-mu)*z/(r1**3) - mu*z/(r2**3)

        return np.array([vx, vy, vz, ax, ay, az])

    def compute_stm(self,
                   initial_state: np.ndarray,
                   duration: float) -> np.ndarray:
        """
        计算状态转移矩阵。

        使用变分方程数值积分计算从 initial_state 出发，
        经过 duration 时间后的状态转移矩阵。

        Args:
            initial_state: 初始状态（6维）
            duration: 积分时长（秒）

        Returns:
            6x6 状态转移矩阵 Φ(duration, 0)
        """
        _, stm = self._stm_calc.propagate_with_stm(
            dynamics=self._get_dynamics_func(),
            initial_state=initial_state,
            t0=0.0,
            tf=duration,
            method=self.integrator_type,
            num_steps=self.num_steps
        )
        return stm

    def analyze_stability(self,
                         orbit_state: np.ndarray,
                         period: float) -> Dict:
        """
        分析轨道稳定性（计算单值矩阵特征值）。

        Returns:
            包含特征值、稳定性指标等信息的字典
        """
        stm = self.compute_stm(orbit_state, period)
        eigenvalues = la.eigvals(stm)

        # 稳定性判据：所有特征值模长 <= 1
        max_mag = np.max(np.abs(eigenvalues))
        is_stable = max_mag <= 1.0 + 1e-6

        return {
            'eigenvalues': eigenvalues,
            'max_magnitude': max_mag,
            'stable': is_stable,
            'monodromy_matrix': stm
        }

    def __repr__(self) -> str:
        return (f"LunarSwingTargeter(mu={self.mu}, "
                f"integrator_type='{self.integrator_type}')")
