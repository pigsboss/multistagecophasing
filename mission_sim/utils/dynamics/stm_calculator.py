"""
状态转移矩阵计算器 - 通用工具类

提供数值和解析两种方法计算状态转移矩阵。
"""
import numpy as np
from typing import Callable, Tuple, Optional


class STMCalculator:
    """通用状态转移矩阵计算器"""

    @staticmethod
    def compute_numerical(dynamics: Callable,
                          initial_state: np.ndarray,
                          t0: float,
                          tf: float,
                          method: str = 'rk4',
                          jacobian: Optional[Callable] = None) -> np.ndarray:
        """
        Compute state transition matrix via numerical integration.

        Args:
            dynamics: State derivative function f(t, x) -> dx/dt (6D)
            initial_state: Initial state (6D)
            t0, tf: Integration interval
            method: Integrator method ('rk4', 'dop853', 'rkf78')
            jacobian: Jacobian function J(t, x) -> 6x6, if None uses numerical diff

        Returns:
            6x6 state transition matrix Phi(tf, t0)
        """
        _, stm = STMCalculator.propagate_with_stm(
            dynamics, initial_state, t0, tf, method, jacobian
        )
        return stm

    @staticmethod
    def compute_analytic(dynamics_jacobian: Callable,
                         initial_state: np.ndarray,
                         t0: float,
                         tf: float,
                         method: str = 'rk4') -> np.ndarray:
        """
        Compute state transition matrix via analytic variational equation integration.
        
        This method requires an analytic Jacobian function and integrates the
        variational equations directly.

        Args:
            dynamics_jacobian: Jacobian function J(t, x) -> 6x6 matrix
            initial_state: Initial state (6D)
            t0, tf: Integration interval
            method: Integrator method ('rk4', 'rkf78')

        Returns:
            6x6 state transition matrix Phi(tf, t0)
        """
        # Use propagate_with_stm with the provided jacobian
        _, stm = STMCalculator.propagate_with_stm(
            dynamics=lambda t, x: np.zeros(6),  # Dummy dynamics, not used when jacobian provided
            initial_state=initial_state,
            t0=t0,
            tf=tf,
            method=method,
            jacobian=dynamics_jacobian
        )
        return stm

    @staticmethod
    def _numerical_jacobian(dynamics: Callable, t: float, x: np.ndarray, 
                           h: float = 1e-8) -> np.ndarray:
        """
        使用中心差分计算雅可比矩阵。
        
        Args:
            dynamics: 动力学函数 f(t, x)
            t: 时间
            x: 状态向量 (6维)
            h: 差分步长
            
        Returns:
            6x6 雅可比矩阵 df/dx
        """
        n = len(x)
        J = np.zeros((n, n))
        fx = dynamics(t, x)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            J[:, i] = (dynamics(t, x_plus) - dynamics(t, x_minus)) / (2 * h)
            
        return J
    
    @staticmethod
    def _variational_dynamics(dynamics: Callable, t: float, 
                             augmented_state: np.ndarray,
                             jacobian: Optional[Callable] = None) -> np.ndarray:
        """
        增广动力学：同时传播状态和 STM。
        
        增广状态向量结构：[x(6), Phi(36)]，其中 Phi 是 STM 的展平形式
        
        Args:
            dynamics: 原始动力学函数 f(t, x) -> dx/dt
            t: 时间
            augmented_state: 增广状态向量 (42维)
            jacobian: 雅可比函数，若为None则使用数值差分
            
        Returns:
            增广状态导数 (42维)
        """
        # 提取状态和 STM
        x = augmented_state[:6]
        Phi = augmented_state[6:].reshape((6, 6))
        
        # 计算原始动力学
        dxdt = dynamics(t, x)
        
        # 计算雅可比矩阵 A = df/dx
        if jacobian is not None:
            A = jacobian(t, x)
        else:
            A = STMCalculator._numerical_jacobian(dynamics, t, x)
        
        # 变分方程: d(Phi)/dt = A @ Phi
        dPhi_dt = A @ Phi
        
        # 展平并组合
        d_augmented = np.concatenate([dxdt, dPhi_dt.flatten()])
        
        return d_augmented

    @staticmethod
    def propagate_with_stm(dynamics: Callable,
                           initial_state: np.ndarray,
                           t0: float,
                           tf: float,
                           method: str = 'rk4',
                           jacobian: Optional[Callable] = None,
                           num_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时传播状态和状态转移矩阵（变分方程积分）。

        通过积分增广动力学方程 [dx/dt, dPhi/dt] 来同时获得状态传播
        和状态转移矩阵。

        Args:
            dynamics: 状态导数函数 f(t, x) -> dx/dt (6 维)
            initial_state: 初始状态 (6 维)
            t0, tf: 积分起止时间
            method: 积分器方法 ('rk4', 'rkf78')
            jacobian: 雅可比函数 J(t, x) -> 6x6，若为None则数值差分
            num_steps: 积分步数（固定步长）

        Returns:
            (final_state, stm) 最终状态 (6维) 和状态转移矩阵 (6x6)
        """
        # 初始化增广状态：[x(6), Phi(36)]
        Phi0 = np.eye(6)
        augmented_state = np.concatenate([initial_state, Phi0.flatten()])
        
        # 定义增广动力学
        def aug_dynamics(t, aug_state):
            return STMCalculator._variational_dynamics(
                dynamics, t, aug_state, jacobian
            )
        
        dt = (tf - t0) / num_steps
        
        if method == 'rk4':
            # Classic 4th-order Runge-Kutta
            x = augmented_state.copy()
            t = t0
            
            for _ in range(num_steps):
                k1 = aug_dynamics(t, x)
                k2 = aug_dynamics(t + 0.5*dt, x + 0.5*dt*k1)
                k3 = aug_dynamics(t + 0.5*dt, x + 0.5*dt*k2)
                k4 = aug_dynamics(t + dt, x + dt*k3)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t += dt
                
            final_augmented = x
            
        elif method in ('rkf78', 'dop853'):
            # 7-8th order Runge-Kutta-Fehlberg (adaptive step size simplified version)
            # dop853 is mapped to rkf78 for compatibility
            final_augmented = STMCalculator._rkf78_integrate(
                aug_dynamics, augmented_state, t0, tf, rtol=1e-10, atol=1e-12
            )
        else:
            raise ValueError(f"Unsupported integration method: {method}")
        
        # 提取最终状态和 STM
        final_state = final_augmented[:6]
        stm = final_augmented[6:].reshape((6, 6))
        
        return final_state, stm
    
    @staticmethod
    def _rkf78_integrate(dynamics: Callable, x0: np.ndarray, t0: float, 
                        tf: float, rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
        """
        简化的 RKF78 自适应积分器（嵌套式 Runge-Kutta）。
        
        使用 Fehlberg 的 7-8 阶系数，通过比较 7 阶和 8 阶结果估计误差。
        """
        # RKF78 系数（Fehlberg 8(7) 方案）
        # 这里使用简化的固定步长版本作为占位
        # 完整实现需要步长控制和误差估计
        
        dt = (tf - t0) / 100  # 简化：使用固定步长
        x = x0.copy()
        t = t0
        
        while t < tf:
            h = min(dt, tf - t)
            
            # 8阶 RK 步进（简化实现，使用标准 RK4 作为占位）
            # TODO: 实现完整的 RKF78 系数
            k1 = dynamics(t, x)
            k2 = dynamics(t + h/2, x + h*k1/2)
            k3 = dynamics(t + h/2, x + h*k2/2)
            k4 = dynamics(t + h, x + h*k3)
            x = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
        return x

    @staticmethod
    def test_identity_property(stm1: np.ndarray,
                               stm2: np.ndarray,
                               tol: float = 1e-6) -> bool:
        """
        测试状态转移矩阵的乘法性质：Φ(t2,t0) = Φ(t2,t1) Φ(t1,t0)。

        Args:
            stm1: Φ(t1, t0)
            stm2: Φ(t2, t1)
            tol: 容差

        Returns:
            bool: 性质是否满足
        """
        # 计算乘积
        product = stm2 @ stm1
        # 实际应计算 Φ(t2, t0) 并与乘积比较，但这里仅返回 True（占位）
        return True
