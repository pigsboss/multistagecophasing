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
                           char_period: float = None,
                           adaptive_damping: bool = True) -> Dict:
        """
        使用单参数打靶法搜索共振周期轨道。
        
        增加自适应阻尼和更好的数值稳定性处理。
        
        Args:
            resonance_ratio: (n, m) 共振比，如 (2, 1) 表示 2:1 共振
            initial_guess: 6 维初始状态猜测 [x, y, z, vx, vy, vz]（无量纲）
            target_period: 目标周期（秒），None 则根据共振比自动计算
            tol: 位置残差收敛容差（无量纲单位）
            max_iter: 最大迭代次数
            damping: 初始阻尼因子 (0-1)
            char_period: CRTBP特征周期（秒），用于时间单位转换。
                        默认为4.342*86400（地月系统约4.342天）
            adaptive_damping: 是否使用自适应阻尼，当残差增加时减少阻尼
    
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

        # Convert to nondimensional time (CRTBP time units)
        target_period_nd = target_period / char_period

        print(f"Searching {n}:{m} resonant orbit, target period: {target_period/86400:.3f} days "
              f"({target_period_nd:.3f} nondim units)")
        print(f"Initial guess: {initial_guess}")

        # 初始状态
        x0 = initial_guess.copy()
        history = []
        
        # 确定设计变量和残差索引
        is_planar = (initial_guess[2] == 0.0 and initial_guess[5] == 0.0)
        if is_planar:
            design_indices = [3, 4]   # vx, vy
            residual_indices = [0, 1] # x, y
            num_design = 2
            num_residual = 2
        else:
            design_indices = [4, 5]   # vy, vz
            residual_indices = [0, 1, 2]  # x, y, z
            num_design = 2
            num_residual = 3

        current_damping = damping
        prev_residual_norm = float('inf')
        
        for i in range(max_iter):
            # 传播状态并计算STM
            x_final, stm = self._stm_calc.propagate_with_stm(
                dynamics=self._get_dynamics_func(),
                initial_state=x0,
                t0=0.0,
                tf=target_period_nd,
                method=self.integrator_type,
                num_steps=self.num_steps
            )
            
            # 数值有效性检查
            if not np.all(np.isfinite(x_final)) or not np.all(np.isfinite(stm)):
                print(f"✗ Numerical overflow at iteration {i+1}, reducing damping and restarting from previous state")
                if i > 0:
                    x0 = history[-1]['state'].copy()
                    current_damping *= 0.5
                    print(f"   Reduced damping to {current_damping:.3f}")
                    continue
                else:
                    return {
                        'state': x0,
                        'period': target_period,
                        'convergence_history': history,
                        'success': False,
                        'final_residual': np.full(6, np.nan),
                        'error': 'numerical_overflow'
                    }
            
            # 计算位置残差
            residual = x_final - x0
            pos_residual = residual[residual_indices]
            res_norm = la.norm(pos_residual)
            
            # 记录历史
            history.append({
                'iteration': i,
                'residual_norm': res_norm,
                'state': x0.copy(),
                'final_state': x_final.copy(),
                'stm': stm.copy(),
                'damping': current_damping
            })

            # 检查收敛
            if res_norm < tol:
                print(f"✓ Converged at iteration {i+1}, position residual: {res_norm:.2e}")
                return {
                    'state': x0,
                    'period': target_period,
                    'convergence_history': history,
                    'success': True,
                    'final_residual': residual
                }

            # 自适应阻尼：如果残差增加，减少阻尼
            if adaptive_damping and res_norm > prev_residual_norm * 1.1 and i > 2:
                current_damping *= 0.7
                if current_damping < 0.1:
                    current_damping = 0.1
                print(f"  Residual increased, reducing damping to {current_damping:.3f}")
            
            prev_residual_norm = res_norm
            
            # 计算敏感度矩阵
            sensitivity = stm[np.ix_(residual_indices, design_indices)]  # 3x2 or 2x2 matrix
            
            # 检查敏感度矩阵有效性
            if not np.all(np.isfinite(sensitivity)):
                print(f"  Warning: Invalid sensitivity matrix, using gradient descent")
                # 使用梯度下降法作为备选
                grad_descent_step = 0.01
                delta_v_design = -grad_descent_step * pos_residual[:num_design]
            else:
                # 奇异值分解求伪逆，提高数值稳定性
                U, s, Vt = la.svd(sensitivity, full_matrices=False)
                
                # 条件数检查
                cond_number = s[0] / (s[-1] + 1e-12)
                if cond_number > 1e12:
                    print(f"  Warning: High condition number ({cond_number:.1e}), adding regularization")
                    # 添加Tikhonov正则化
                    reg = 1e-8 * np.eye(num_design)
                    delta_v_design = la.solve(sensitivity.T @ sensitivity + reg, 
                                             sensitivity.T @ (-pos_residual))
                else:
                    # 使用SVD伪逆
                    s_inv = np.zeros_like(s)
                    s_inv[s > 1e-12] = 1.0 / s[s > 1e-12]
                    delta_v_design = Vt.T @ np.diag(s_inv) @ U.T @ (-pos_residual)
            
            # 应用阻尼
            delta_v_design *= current_damping
            
            # 限制修正量大小，防止过大跳跃
            max_correction = 0.5
            correction_norm = la.norm(delta_v_design)
            if correction_norm > max_correction:
                delta_v_design = delta_v_design / correction_norm * max_correction
                print(f"  Limiting correction from {correction_norm:.3f} to {max_correction:.3f}")
            
            # 更新设计变量
            x0[design_indices] += delta_v_design

            # 定期输出进度
            if i % 5 == 0 or res_norm < 1e-2 or res_norm > 10:
                corr_str = ', '.join([f"{dv:.3e}" for dv in delta_v_design])
                print(f"  Iter {i+1}: pos residual = {res_norm:.3e}, "
                      f"correction = [{corr_str}], damping = {current_damping:.2f}")

        print(f"✗ Failed to converge after {max_iter} iterations, final residual: {history[-1]['residual_norm']:.2e}")
        return {
            'state': x0,
            'period': target_period,
            'convergence_history': history,
            'success': False,
            'final_residual': residual
        }

    # ==================================================================
    # LS 专题沉淀（LS-001 G5-①）：对称周期解 + 初值扫描
    # ==================================================================
    def _propagate(self, state0, tf, num_steps=None, jacobian=None):
        return self._stm_calc.propagate_with_stm(
            dynamics=self._get_dynamics_func(),
            initial_state=np.asarray(state0, dtype=np.float64),
            t0=0.0, tf=tf, method=self.integrator_type,
            jacobian=jacobian, num_steps=int(num_steps or self.num_steps))

    def _analytic_jacobian(self):
        """若动力学对象提供解析变分矩阵则返回 J(t,x)；否则 None（数值差分回退）。"""
        if hasattr(self.dynamics, "variational_matrix_nd"):
            return lambda t, x: self.dynamics.variational_matrix_nd(x)
        return None

    def find_symmetric_periodic_orbit(self, resonance_ratio, x0_guess, vy0_guess,
                                      tol=1e-8, max_iter=50, damping=0.5,
                                      adaptive_damping=True, char_period=None,
                                      num_steps=None, verbose=False) -> Dict:
        """
        对称周期轨道求解（垂直截面判据）。

        设计变量 (x0, vy0)；在 t = T/2 处要求以垂直方式穿越 x 轴：
            残差 r = [y(T/2), vx(T/2)]
        由平面 CRTBP 的对称性，半周期垂直穿越 ⇒ 全周期完整闭合（位置+速度）。
        收敛后以全周期积分复核 full_state_residual 与 Jacobi 漂移。

        Returns: dict(success, state, period, period_nd, half_period_residual,
                      full_state_residual, jacobi, jacobi_drift, iterations, history)
        """
        n, m = resonance_ratio
        if char_period is None:
            char_period = 4.342 * 86400.0
        T_moon = 27.321661 * 24 * 3600
        period = (m / n) * T_moon
        T_nd = period / char_period
        half_T_nd = 0.5 * T_nd
        ns = int(num_steps or self.num_steps)
        jac = self._analytic_jacobian()

        x0 = np.zeros(6)
        x0[0] = float(x0_guess)
        x0[4] = float(vy0_guess)

        history = []
        current_damping = float(damping)
        prev_norm = float("inf")
        last_norm = float("nan")

        for i in range(max_iter):
            x_f, stm = self._propagate(x0, half_T_nd, num_steps=ns, jacobian=jac)
            if not (np.all(np.isfinite(x_f)) and np.all(np.isfinite(stm))):
                if history:
                    x0 = history[-1]["state"].copy()
                current_damping *= 0.5
                if current_damping < 0.05:
                    return dict(success=False, state=x0.copy(), period=period, period_nd=T_nd,
                                half_period_residual=last_norm, full_state_residual=np.nan,
                                jacobi=np.nan, jacobi_drift=np.nan, iterations=i, history=history,
                                error="numerical_overflow")
                continue

            res = np.array([x_f[1], x_f[3]])  # [y(T/2), vx(T/2)]
            rn = float(np.linalg.norm(res))
            last_norm = rn
            history.append(dict(iteration=i, residual_norm=rn, state=x0.copy(),
                                damping=current_damping))
            if verbose and (i % 5 == 0 or rn < 1e-6):
                print(f"  [sym] iter {i}: residual={rn:.3e}, damping={current_damping:.2f}")

            if rn < tol:
                x_full, _ = self._propagate(x0, T_nd, num_steps=ns * 2, jacobian=jac)
                full_res = float(np.linalg.norm(x_full - x0))
                if hasattr(self.dynamics, "jacobi_constant"):
                    c0 = float(self.dynamics.jacobi_constant(x0))
                    c1 = float(self.dynamics.jacobi_constant(x_full))
                    drift = abs(c1 - c0) / max(abs(c0), 1e-30)
                else:
                    c0, drift = np.nan, np.nan
                return dict(success=True, state=x0.copy(), period=period, period_nd=T_nd,
                            half_period_residual=rn, full_state_residual=full_res,
                            jacobi=c0, jacobi_drift=float(drift),
                            iterations=i + 1, history=history)

            if adaptive_damping and rn > prev_norm * 1.1 and i > 2:
                current_damping = max(0.1, current_damping * 0.7)
            prev_norm = rn

            J = np.array([[stm[1, 0], stm[1, 4]],
                          [stm[3, 0], stm[3, 4]]])
            U, s, Vt = la.svd(J, full_matrices=False)
            if s[-1] < 1e-12:
                ds = np.array([-0.01, -0.01])
            else:
                ds = Vt.T @ np.diag(1.0 / s) @ U.T @ (-res)
            ds *= current_damping
            nrm = float(la.norm(ds))
            if nrm > 0.2:
                ds = ds / nrm * 0.2
            x0[0] += ds[0]
            x0[4] += ds[1]

        return dict(success=False, state=x0.copy(), period=period, period_nd=T_nd,
                    half_period_residual=last_norm, full_state_residual=np.nan,
                    jacobi=np.nan, jacobi_drift=np.nan, iterations=max_iter, history=history,
                    error="max_iter")

    def scan_resonant_orbits(self, resonance_ratio, x0_grid, vy0_grid,
                             tol=1e-6, max_iter=20, num_steps=3000,
                             polish_tol=1e-9, polish_num_steps=40000,
                             char_period=None, max_polish=None) -> List[Dict]:
        """网格扫描 + 多分辨率精修：粗扫收敛候选 → 高精度 polish；返回收敛解列表。"""
        coarse = []
        for x0g in np.asarray(x0_grid, dtype=np.float64):
            for vy0g in np.asarray(vy0_grid, dtype=np.float64):
                r = self.find_symmetric_periodic_orbit(
                    resonance_ratio, x0g, vy0g, tol=tol, max_iter=max_iter,
                    char_period=char_period, num_steps=num_steps, verbose=False)
                if r.get("success"):
                    coarse.append((float(x0g), float(vy0g)))
        out = []
        for j, (x0g, vy0g) in enumerate(coarse):
            if max_polish is not None and j >= max_polish:
                break
            p = self.find_symmetric_periodic_orbit(
                resonance_ratio, x0g, vy0g, tol=polish_tol, max_iter=40,
                char_period=char_period, num_steps=polish_num_steps, verbose=False)
            if p.get("success"):
                p["seed"] = (x0g, vy0g)
                out.append(p)
        return out

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
        """Simplified CRTBP dynamics derivative (dimensionless rotating frame)"""
        x, y, z, vx, vy, vz = state
        mu = self.mu
        
        # Compute distances
        dx1 = x + mu
        dx2 = x + mu - 1
        
        r1_sq = dx1**2 + y**2 + z**2
        r2_sq = dx2**2 + y**2 + z**2
        
        # Prevent division by zero
        eps = 1e-15
        r1_sq = max(r1_sq, eps)
        r2_sq = max(r2_sq, eps)
        
        r1 = np.sqrt(r1_sq)
        r2 = np.sqrt(r2_sq)
        
        r1_cubed = r1**3
        r2_cubed = r2**3

        ax = 2*vy + x - (1-mu)*dx1/r1_cubed - mu*dx2/r2_cubed
        ay = -2*vx + y - (1-mu)*y/r1_cubed - mu*y/r2_cubed
        az = -(1-mu)*z/r1_cubed - mu*z/r2_cubed

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
        Analyze orbit stability by computing eigenvalues of the monodromy matrix.

        Args:
            orbit_state: Initial state vector (6D)
            period: Orbital period (seconds)

        Returns:
            Dictionary containing eigenvalues, stability indicators, etc.
        """
        stm = self.compute_stm(orbit_state, period)
        
        # Check for invalid STM (NaN/Inf)
        if not np.all(np.isfinite(stm)):
            return {
                'eigenvalues': np.full(6, np.nan),
                'max_magnitude': np.nan,
                'stable': False,
                'monodromy_matrix': stm,
                'error': 'invalid_stm'
            }
        
        eigenvalues = la.eigvals(stm)

        # Stability criterion: all eigenvalue magnitudes <= 1
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
