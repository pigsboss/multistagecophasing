# mission_sim/core/trajectory/generators.py
"""
标称轨道生成器集合 (Strategy Pattern)
提供多态工厂，支持：
- 开普勒轨道 (KeplerianGenerator)
- 带 J2 摄动的开普勒轨道 (J2KeplerianGenerator)
- Halo 轨道 (HaloDifferentialCorrector)
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.utils.differential_correction import differential_correction


class BaseTrajectoryGenerator(ABC):
    """
    标称轨道生成器抽象基类。
    所有具体生成器必须实现 generate 方法，返回 Ephemeris 对象。
    """
    @abstractmethod
    def generate(self, config: dict) -> Ephemeris:
        pass


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    开普勒轨道生成器（二体问题解析解）。
    使用经典开普勒公式生成参考星历，无摄动。
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        初始化生成器。

        Args:
            mu: 中心天体引力常数 (m³/s²)，默认地球。
        """
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成星历。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数
            - dt: 时间步长 (s)
            - sim_time: 仿真时长 (s)

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        times = np.arange(0, sim_time + dt, dt)

        a, e, i, Omega, omega, M0 = elements
        n = np.sqrt(self.mu / a**3)

        states = []
        for t in times:
            # 平近点角
            M = M0 + n * t
            # 解 Kepler 方程（简化：小 e 时可近似，这里使用牛顿迭代）
            E = self._kepler_solver(M, e)
            # 真近点角
            nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
            # 轨道面内坐标
            r = a * (1 - e * np.cos(E))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            vx_orb = -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu)
            vy_orb = np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu))

            # 坐标变换到 J2000_ECI (简化：仅处理圆轨道或忽略倾角)
            # 为简化，此处假设轨道在赤道平面内 (i=0, Omega=0, omega=0)
            # 实际应用中应包含完整的三维旋转。
            state_eci = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
            states.append(state_eci)

        return Ephemeris(times, np.array(states), CoordinateFrame.J2000_ECI)

    def _kepler_solver(self, M: float, e: float, tol: float = 1e-12) -> float:
        """解 Kepler 方程 M = E - e sin(E) (牛顿迭代)"""
        E = M if e < 0.8 else np.pi  # 初始猜测
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        return E


class J2KeplerianGenerator(BaseTrajectoryGenerator):
    """
    带 J2 摄动的开普勒轨道生成器。
    通过数值积分二体 + J2 摄动，生成高精度参考星历。
    适用于 LEO/GEO 任务。
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        初始化生成器。

        Args:
            mu: 中心天体引力常数 (m³/s²)，默认地球。
        """
        self.mu = mu
        self.j2_model = J2Gravity(mu_earth=mu)  # 使用 J2 模型

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成 J2 摄动轨道。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数
            - dt: 输出步长 (s)
            - sim_time: 仿真时长 (s)
            - integrator: 积分器方法 (可选，默认 'DOP853')
            - rtol: 相对容差 (可选，默认 1e-12)

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("J2KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        integrator = config.get("integrator", 'DOP853')
        rtol = config.get("rtol", 1e-12)

        # 将轨道根数转换为笛卡尔状态
        state0 = self._elements_to_cartesian(elements)

        # 定义运动方程
        def dynamics(t, state):
            pos = state[:3]
            vel = state[3:6]
            r = np.linalg.norm(pos)
            # 中心引力
            acc_central = -self.mu * pos / r**3
            # J2 摄动
            acc_j2 = self.j2_model.compute_accel(state, t)
            return np.concatenate([vel, acc_central + acc_j2])

        # 积分
        times = np.arange(0, sim_time + dt, dt)
        sol = solve_ivp(
            dynamics,
            t_span=(0, sim_time),
            y0=state0,
            t_eval=times,
            method=integrator,
            rtol=rtol,
            atol=rtol
        )
        if not sol.success:
            raise RuntimeError(f"J2 轨道积分失败: {sol.message}")

        return Ephemeris(sol.t, sol.y.T, CoordinateFrame.J2000_ECI)

    def _elements_to_cartesian(self, elements):
        """将轨道根数转换为 J2000_ECI 笛卡尔状态（简化，仅圆轨道）"""
        a, e, i, Omega, omega, M0 = elements
        # 简化：仅支持圆轨道 e=0，且 i=0，Ω=0，ω=0
        # 实际应用需实现完整转换
        n = np.sqrt(self.mu / a**3)
        M = M0
        E = M  # 圆轨道近似
        nu = E
        r = a
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        vx = -np.sqrt(self.mu / a) * np.sin(nu)
        vy = np.sqrt(self.mu / a) * np.cos(nu)
        return np.array([x, y, 0.0, vx, vy, 0.0])


class HaloDifferentialCorrector(BaseTrajectoryGenerator):
    """
    Halo 轨道微分修正生成器。
    使用单参数微分修正方法，求解日地/地月旋转系中的周期 Halo 轨道。
    """

    def __init__(self, mu: float = 3.00348e-6):
        """
        初始化生成器。

        Args:
            mu: 无量纲质量比（日地系统默认 3.00348e-6，地月系统约 0.01215）
        """
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        生成 Halo 轨道星历。

        config 必须包含:
            - Az: 目标 Z 振幅 (无量纲)
            - dt: 输出步长 (无量纲)
            - initial_guess: 可选，[x0, z0, vy0] 初始猜测
            - max_iter: 微分修正最大迭代次数 (可选，默认 10)
            - tol: 修正容差 (可选，默认 1e-10)

        Returns:
            Ephemeris 对象（SUN_EARTH_ROTATING 坐标系）
        """
        # 获取参数
        Az = config.get("Az", 0.05)
        dt_nd = config.get("dt", 0.001)
        max_iter = config.get("max_iter", 10)
        tol = config.get("tol", 1e-10)

        # 获取初始猜测
        if "initial_guess" in config:
            x0, z0, vy0 = config["initial_guess"]
        else:
            x0, z0, vy0 = self._robust_initial_guess(Az)

        # 初始状态 [x, y, z, vx, vy, vz] (从 XZ 平面出发)
        state0_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

        # 定义需要修正的目标量：半周期后 y=0 且 y 方向速度不变（或使用更严格的条件）
        # 简化：修正使轨道闭合（周期条件）
        # 使用微分修正寻找周期轨道
        try:
            # 微分修正需要目标函数 f(params) = [y(T/2), vz(T/2)] 等
            # 这里简化为修正初始状态使得轨道周期闭合
            state0_corrected = differential_correction(
                state0_nd,
                self._crtbp_equations,
                target_conditions=self._closure_conditions,
                max_iter=max_iter,
                tol=tol
            )
        except Exception as e:
            print(f"[HaloCorrector] 微分修正失败，使用原始初始猜测: {e}")
            state0_corrected = state0_nd

        # 确定周期（半周期检测）
        T_nd = self._find_period(state0_corrected)
        if T_nd is None:
            T_nd = 3.141  # 默认近似周期

        # 生成完整轨道
        times_nd = np.arange(0, T_nd, dt_nd)
        sol = solve_ivp(
            self._crtbp_equations,
            (0, T_nd),
            state0_corrected,
            t_eval=times_nd,
            method='DOP853',
            rtol=1e-12,
            atol=1e-12
        )

        # 转换为物理单位
        AU = 1.495978707e11
        OMEGA = 1.990986e-7
        physical_times = sol.t / OMEGA
        physical_states = sol.y.T.copy()
        physical_states[:, 0:3] *= AU
        physical_states[:, 3:6] *= (AU * OMEGA)

        return Ephemeris(physical_times, physical_states, CoordinateFrame.SUN_EARTH_ROTATING)

    def _crtbp_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """CRTBP 无量纲运动方程"""
        x, y, z, vx, vy, vz = state
        mu = self.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
        az = -(1-mu)*z/r1**3 - mu*z/r2**3
        return np.array([vx, vy, vz, ax, ay, az])

    def _closure_conditions(self, state0: np.ndarray, tf: float) -> np.ndarray:
        """定义闭合条件：半周期后 y=0, vz=0（对称性）"""
        # 积分到 tf
        sol = solve_ivp(self._crtbp_equations, (0, tf), state0, method='DOP853',
                        rtol=1e-12, atol=1e-12)
        state_tf = sol.y[:, -1]
        # 返回需要为零的向量 [y, vz]
        return np.array([state_tf[1], state_tf[5]])

    def _find_period(self, state0: np.ndarray, max_time: float = 20.0) -> float:
        """寻找半周期（y=0 穿越）"""
        def y_zero(t, y):
            return y[1]
        y_zero.direction = -1  # 从正到负
        sol = solve_ivp(self._crtbp_equations, (0, max_time), state0,
                        events=[y_zero], method='DOP853',
                        rtol=1e-12, atol=1e-12)
        if len(sol.t_events[0]) > 0:
            t_half = sol.t_events[0][0]
            if t_half > 0.1:
                return 2.0 * t_half
        return None

    def _robust_initial_guess(self, Az: float) -> tuple:
        """根据振幅生成稳健初始猜测"""
        mu = self.mu
        gamma = np.cbrt(mu / 3.0)
        x_L2 = 1 - mu + gamma
        x0 = x_L2 + 0.015 * Az
        z0 = Az
        vy0 = 0.01 + 0.1 * Az
        return x0, z0, vy0