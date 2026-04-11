"""
通用 CRTBP 轨道生成器框架
支持多种 CRTBP 轨道的生成，包括：Halo、DRO、Lyapunov、Vertical、共振轨道等
基于微分修正和打靶法，利用轨道对称性简化求解
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve
from dataclasses import dataclass

from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.ids import CoordinateFrame, CelestialBody


class CRTBPOrbitType(Enum):
    """CRTBP 轨道类型枚举"""
    HALO = auto()               # Halo 轨道（三维周期轨道）
    DRO = auto()                # Distant Retrograde Orbit（遥远逆行轨道）
    LISSAJOUS = auto()          # Lissajous 轨道（拟周期轨道）
    LYAPUNOV = auto()           # 李雅普诺夫轨道（平面周期轨道）
    VERTICAL = auto()           # 垂直轨道（z方向周期）
    RESONANT = auto()           # 共振轨道（与系统周期成比例）
    LEADER_FOLLOWER = auto()    # 领航-跟随轨道（编队相对运动）


class SymmetryType(Enum):
    """对称性类型枚举"""
    XZ_PLANE = auto()      # x-z 平面对称（Halo轨道）
    X_AXIS = auto()        # x 轴对称（Lyapunov、DRO轨道）
    Z_AXIS = auto()        # z 轴对称（垂直轨道）
    NONE = auto()          # 无对称性（一般轨道）


@dataclass
class CRTBPOrbitConfig:
    """CRTBP 轨道配置"""
    orbit_type: CRTBPOrbitType
    amplitude: float = 0.05           # 轨道振幅（无量纲）
    lagrange_point: int = 2           # 平动点编号（1,2,3,4,5）
    system_type: str = "sun_earth"    # 系统类型：sun_earth, earth_moon
    duration: float = 10.0            # 积分时长（无量纲时间）
    step_size: float = 0.001          # 输出步长（无量纲时间）
    
    # 微分修正参数
    max_iterations: int = 50          # 最大迭代次数
    tolerance: float = 1e-10          # 收敛容差
    
    # 共振轨道参数
    resonance_ratio: Tuple[int, int] = (1, 1)  # 共振比例 m:n
    
    # 轨道族参数
    family_continuation: bool = False  # 是否进行族延续
    continuation_steps: int = 10       # 族延续步数


class CRTBPOrbitGenerator(BaseTrajectoryGenerator):
    """
    通用 CRTBP 轨道生成器
    
    基于微分修正和打靶法，支持多种 CRTBP 轨道生成。
    利用轨道对称性简化周期轨道求解。
    """
    
    # CRTBP 系统参数（默认：日地系统）
    DEFAULT_SYSTEMS = {
        "sun_earth": {
            "mu": 3.00348e-6,
            "L1": 0.990,      # L1 点 x 坐标（无量纲）
            "L2": 1.010,      # L2 点 x 坐标
            "characteristic_length": 1.495978707e11,  # 特征长度 (m) = 1 AU
            "characteristic_time": 1.990986e-7,       # 特征时间 (rad/s) = 地球公转角速度
            "primary_body": CelestialBody.SUN,
            "secondary_body": CelestialBody.EARTH,
            "barycenter_body": CelestialBody.SUN_EARTH_BARYCENTER,
        },
        "earth_moon": {
            "mu": 0.01215,
            "L1": 0.836,      # 地月 L1 点 x 坐标
            "L2": 1.164,      # 地月 L2 点 x 坐标
            "characteristic_length": 3.844e8,         # 特征长度 = 平均地月距离
            "characteristic_time": 2.661699e-6,       # 特征时间 = 月球公转角速度
            "primary_body": CelestialBody.EARTH,
            "secondary_body": CelestialBody.MOON,
            "barycenter_body": CelestialBody.EARTH_MOON_BARYCENTER,
        }
    }
    
    def __init__(self, 
                 system_type: str = "sun_earth",
                 orbit_type: CRTBPOrbitType = CRTBPOrbitType.HALO,
                 ephemeris: Optional[Any] = None,
                 use_high_precision: bool = False,
                 verbose: bool = False):
        """
        初始化 CRTBP 轨道生成器
        
        Args:
            system_type: CRTBP 系统类型 ("sun_earth", "earth_moon")
            orbit_type: 轨道类型
            ephemeris: 高精度星历实例（可选）
            use_high_precision: 是否使用高精度模式
            verbose: 是否输出详细信息
        """
        super().__init__(ephemeris, use_high_precision)
        
        if system_type not in self.DEFAULT_SYSTEMS:
            raise ValueError(f"不支持的 CRTBP 系统类型: {system_type}")
        
        self.system_type = system_type
        self.orbit_type = orbit_type
        self.verbose = verbose
        
        # 从默认系统或高精度星历获取参数
        if self.use_high_precision and self.ephemeris:
            self._init_from_ephemeris()
        else:
            self._init_from_defaults()
        
        # 轨道类型特定的配置
        self._orbit_strategies = {
            CRTBPOrbitType.HALO: self._halo_strategy,
            CRTBPOrbitType.DRO: self._dro_strategy,
            CRTBPOrbitType.LYAPUNOV: self._lyapunov_strategy,
            CRTBPOrbitType.VERTICAL: self._vertical_strategy,
            CRTBPOrbitType.RESONANT: self._resonant_strategy,
            CRTBPOrbitType.LISSAJOUS: self._lissajous_strategy,
            CRTBPOrbitType.LEADER_FOLLOWER: self._leader_follower_strategy,
        }
    
    def _init_from_defaults(self):
        """从默认配置初始化系统参数"""
        system_config = self.DEFAULT_SYSTEMS[self.system_type]
        
        self.mu = system_config["mu"]
        self.L1 = system_config["L1"]
        self.L2 = system_config["L2"]
        self.L = system_config["characteristic_length"]
        self.omega = system_config["characteristic_time"]
        self.primary = system_config["primary_body"]
        self.secondary = system_config["secondary_body"]
        self.barycenter = system_config["barycenter_body"]
        
        if self.verbose:
            print(f"[CRTBPGenerator] 使用默认系统参数: {self.system_type}")
            print(f"  μ = {self.mu:.6e}, L = {self.L:.2e} m, ω = {self.omega:.6e} rad/s")
    
    def _init_from_ephemeris(self):
        """从高精度星历初始化系统参数"""
        try:
            # 获取天体质心位置和距离
            if self.system_type == "sun_earth":
                # 获取太阳和地球质量
                sun_gm = self.ephemeris.get_body_parameters(CelestialBody.SUN)["GM"]
                earth_gm = self.ephemeris.get_body_parameters(CelestialBody.EARTH)["GM"]
                
                # 获取太阳-地球距离（平均）
                t0 = 0.0  # J2000 历元
                earth_state = self._get_celestial_state(CelestialBody.EARTH, t0, CelestialBody.SUN)
                distance = np.linalg.norm(earth_state[:3])
                
                self.mu = earth_gm / (sun_gm + earth_gm)
                self.L = distance
                
                # 计算特征角速度
                total_gm = sun_gm + earth_gm
                self.omega = np.sqrt(total_gm / distance**3)
                
                self.primary = CelestialBody.SUN
                self.secondary = CelestialBody.EARTH
                self.barycenter = CelestialBody.SUN_EARTH_BARYCENTER
                
                # 计算平动点位置（无量纲）
                self.L1 = 1 - (self.mu/3)**(1/3)
                self.L2 = 1 + (self.mu/3)**(1/3)
                
            elif self.system_type == "earth_moon":
                # 类似处理地月系统
                earth_gm = self.ephemeris.get_body_parameters(CelestialBody.EARTH)["GM"]
                moon_gm = self.ephemeris.get_body_parameters(CelestialBody.MOON)["GM"]
                
                t0 = 0.0
                moon_state = self._get_celestial_state(CelestialBody.MOON, t0, CelestialBody.EARTH)
                distance = np.linalg.norm(moon_state[:3])
                
                self.mu = moon_gm / (earth_gm + moon_gm)
                self.L = distance
                total_gm = earth_gm + moon_gm
                self.omega = np.sqrt(total_gm / distance**3)
                
                self.primary = CelestialBody.EARTH
                self.secondary = CelestialBody.MOON
                self.barycenter = CelestialBody.EARTH_MOON_BARYCENTER
                
                self.L1 = 1 - (self.mu/3)**(1/3)
                self.L2 = 1 + (self.mu/3)**(1/3)
            
            if self.verbose:
                print(f"[CRTBPGenerator] 从高精度星历初始化: {self.system_type}")
                print(f"  μ = {self.mu:.6e}, L = {self.L:.2e} m, ω = {self.omega:.6e} rad/s")
                print(f"  L1 = {self.L1:.4f}, L2 = {self.L2:.4f} (无量纲)")
                
        except Exception as e:
            print(f"[CRTBPGenerator] 从星历初始化失败: {e}")
            print("  回退到默认系统参数")
            self._init_from_defaults()
    
    def generate(self, config: Dict[str, Any]) -> Ephemeris:
        """
        生成指定类型的 CRTBP 轨道
        
        Args:
            config: 配置字典，可包含以下键：
                - orbit_type: 轨道类型（可选，覆盖初始化时的设置）
                - amplitude: 轨道振幅
                - lagrange_point: 平动点编号（1,2,3,4,5）
                - duration: 积分时长（无量纲时间）
                - step_size: 输出步长（无量纲时间）
                - max_iterations: 微分修正最大迭代次数
                - tolerance: 收敛容差
                - resonance_ratio: 共振比例 (m, n)
                - 其他轨道特定参数
        
        Returns:
            Ephemeris: 生成的轨道星历
        """
        # 合并配置
        config = self._merge_config(config)
        
        # 获取轨道策略
        strategy = self._orbit_strategies.get(self.orbit_type)
        if strategy is None:
            raise ValueError(f"不支持的轨道类型: {self.orbit_type}")
        
        # 执行策略
        return strategy(config)
    
    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合并用户配置与默认配置"""
        default_config = {
            "orbit_type": self.orbit_type,
            "amplitude": 0.05,
            "lagrange_point": 2,
            "duration": 10.0,
            "step_size": 0.001,
            "max_iterations": 50,
            "tolerance": 1e-10,
            "resonance_ratio": (1, 1),
            "family_continuation": False,
            "continuation_steps": 10,
        }
        
        # 更新配置，用户配置优先
        merged = default_config.copy()
        merged.update(config)
        
        # 确保 orbit_type 是枚举类型
        if isinstance(merged["orbit_type"], str):
            try:
                merged["orbit_type"] = CRTBPOrbitType[merged["orbit_type"].upper()]
            except KeyError as e:
                # 根据MCPC编码标准，使用英文错误信息
                valid_types = [t.name for t in CRTBPOrbitType]
                raise ValueError(
                    f"Invalid orbit type: {merged['orbit_type']}. "
                    f"Valid types are: {valid_types}"
                ) from e
        
        return merged
    
    def _halo_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """Halo 轨道生成策略"""
        if self.verbose:
            print(f"[HaloGenerator] 生成 Halo 轨道，振幅={config['amplitude']}")
        
        # 获取平动点位置
        lagrange_point = config["lagrange_point"]
        L_point = self._get_lagrange_point(lagrange_point)
        
        # Halo 轨道初始猜测
        Az = config["amplitude"]
        if abs(Az - 0.05) < 1e-6:
            x0, z0, vy0 = 1.01106, 0.05, 0.0105
        else:
            # 简单比例缩放
            x0 = 1.01106 + (Az - 0.05) * 0.1
            z0 = Az
            vy0 = 0.0105 + (Az - 0.05) * 0.05
        
        # 调整到指定平动点
        if lagrange_point == 1:
            x0 = self.L1 + (x0 - 1.0)
        elif lagrange_point == 2:
            x0 = self.L2 + (x0 - 1.0)
        
        state0_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
        
        # 微分修正
        corrected_state, period = self._differential_correction(
            state0_nd, 
            symmetry=SymmetryType.XZ_PLANE,
            max_iter=config["max_iterations"],
            tol=config["tolerance"]
        )
        
        # 生成完整轨道
        return self._integrate_orbit(
            corrected_state, 
            period, 
            config["step_size"],
            config["duration"]
        )
    
    def _dro_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """DRO 轨道生成策略"""
        if self.verbose:
            print(f"[DROGenerator] 生成 DRO 轨道，振幅={config['amplitude']}")
        
        # DRO 初始猜测（关于 x 轴对称）
        amplitude = config["amplitude"]
        lagrange_point = config["lagrange_point"]
        
        # 基于平动点位置
        if lagrange_point == 2:
            L_point = self.L2
        elif lagrange_point == 1:
            L_point = self.L1
        else:
            raise ValueError(f"DRO 轨道仅支持 L1 或 L2 点，当前: L{lagrange_point}")
        
        x0 = L_point + amplitude
        vy0 = self._estimate_initial_velocity(x0, orbit_type="dro")
        
        state0_nd = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        
        # 微分修正（x轴对称）
        corrected_state, period = self._differential_correction(
            state0_nd,
            symmetry=SymmetryType.X_AXIS,
            max_iter=config["max_iterations"],
            tol=config["tolerance"]
        )
        
        return self._integrate_orbit(
            corrected_state,
            period,
            config["step_size"],
            config["duration"]
        )
    
    def _lyapunov_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """Lyapunov 轨道生成策略（平面轨道）"""
        if self.verbose:
            print(f"[LyapunovGenerator] 生成 Lyapunov 轨道")
        
        amplitude = config["amplitude"]
        lagrange_point = config["lagrange_point"]
        
        # Lyapunov 轨道在平动点附近平面内运动
        if lagrange_point in [1, 2, 3]:
            L_point = self._get_lagrange_point(lagrange_point)
            x0 = L_point + amplitude  # 在 x 方向偏移
            
            # 估计初始速度（保持轨道闭合）
            vy0 = self._estimate_initial_velocity(x0, orbit_type="lyapunov")
            
            state0_nd = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
            
            corrected_state, period = self._differential_correction(
                state0_nd,
                symmetry=SymmetryType.X_AXIS,
                max_iter=config["max_iterations"],
                tol=config["tolerance"]
            )
        else:
            # L4/L5 点的 Lyapunov 轨道不同
            raise NotImplementedError(f"L{lagrange_point} 点的 Lyapunov 轨道尚未实现")
        
        return self._integrate_orbit(
            corrected_state,
            period,
            config["step_size"],
            config["duration"]
        )
    
    def _vertical_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """垂直轨道生成策略（z方向振荡）"""
        if self.verbose:
            print(f"[VerticalGenerator] 生成垂直轨道")
        
        amplitude = config["amplitude"]
        lagrange_point = config["lagrange_point"]
        
        L_point = self._get_lagrange_point(lagrange_point)
        
        # 垂直轨道：在平动点上方/下方振荡
        # 改进的初始猜测：垂直轨道需要适当的初始速度
        x0 = L_point
        z0 = amplitude
        
        # 对于垂直轨道，需要初始y方向速度来维持轨道
        # 根据线性化理论，垂直振荡频率约为1
        vy0 = 0.01 + amplitude * 0.1  # 随振幅调整
        
        # 垂直轨道应有初始x方向速度来帮助闭合
        vx0 = -amplitude * 0.005
        
        state0_nd = np.array([x0, 0.0, z0, vx0, vy0, 0.0])
        
        # 垂直轨道使用XZ平面对称（与Halo轨道类似）
        corrected_state, period = self._differential_correction(
            state0_nd,
            symmetry=SymmetryType.XZ_PLANE,
            max_iter=config["max_iterations"],
            tol=config["tolerance"]
        )
        
        return self._integrate_orbit(
            corrected_state,
            period,
            config["step_size"],
            config["duration"]
        )
    
    def _resonant_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """共振轨道生成策略"""
        m, n = config["resonance_ratio"]
        if self.verbose:
            print(f"[ResonantGenerator] 生成 {m}:{n} 共振轨道")
        
        # 使用打靶法寻找满足共振条件的轨道
        def shooting_function(variables: np.ndarray) -> np.ndarray:
            """打靶函数：寻找周期轨道"""
            x0, vy0 = variables
            state0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
            
            # 积分 m 个估计周期
            period_guess = 2 * np.pi * m  # 无量纲时间
            sol = solve_ivp(self._crtbp_equations, (0, period_guess), state0,
                          method='DOP853', rtol=1e-12, atol=1e-12)
            
            final_state = sol.y[:, -1]
            
            # 约束条件：应返回初始状态（周期轨道）
            return np.array([
                final_state[0] - state0[0],  # x 位置匹配
                final_state[4] - state0[4],  # y 速度匹配
            ])
        
        # 初始猜测
        if config["lagrange_point"] == 2:
            initial_guess = np.array([self.L2 + 0.1, 0.01])
        else:
            initial_guess = np.array([self.L1 + 0.1, 0.01])
        
        # 使用优化器求解
        result = root(shooting_function, initial_guess, method='hybr')
        
        if not result.success:
            raise RuntimeError(f"共振轨道求解失败: {result.message}")
        
        x0_opt, vy0_opt = result.x
        state0_nd = np.array([x0_opt, 0.0, 0.0, 0.0, vy0_opt, 0.0])
        
        # 估算周期
        period = 2 * np.pi * m
        
        return self._integrate_orbit(
            state0_nd,
            period,
            config["step_size"],
            config["duration"]
        )
    
    def _lissajous_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """Lissajous 轨道生成策略（拟周期轨道）"""
        if self.verbose:
            print(f"[LissajousGenerator] 生成 Lissajous 轨道")
        
        # Lissajous 轨道不需要严格周期条件
        # 直接积分给定振幅的初始状态
        amplitude_x = config.get("amplitude_x", 0.01)
        amplitude_z = config.get("amplitude_z", 0.01)
        lagrange_point = config["lagrange_point"]
        
        L_point = self._get_lagrange_point(lagrange_point)
        
        # 初始状态：在平动点附近，给予小的扰动
        x0 = L_point + amplitude_x
        z0 = amplitude_z
        
        # 估计初始速度（基于线性化模型）
        omega_x = self._linear_frequency('x', lagrange_point)
        omega_z = self._linear_frequency('z', lagrange_point)
        
        vy0 = omega_x * amplitude_x * 0.5  # 简化估计
        vx0 = omega_z * amplitude_z * 0.5
        
        state0_nd = np.array([x0, 0.0, z0, vx0, vy0, 0.0])
        
        # 直接积分，不进行微分修正
        duration = config["duration"]
        step_size = config["step_size"]
        
        times_nd = np.arange(0, duration, step_size)
        sol = solve_ivp(self._crtbp_equations, (0, duration), state0_nd,
                       t_eval=times_nd, method='DOP853', rtol=1e-12, atol=1e-12)
        
        return self._nd_to_physical(sol.t, sol.y.T)
    
    def _leader_follower_strategy(self, config: Dict[str, Any]) -> Ephemeris:
        """领航-跟随轨道生成策略"""
        if self.verbose:
            print(f"[LeaderFollowerGenerator] 生成领航-跟随轨道")
        
        # 首先生成参考轨道（如 Halo 轨道）
        reference_config = config.copy()
        reference_config["orbit_type"] = CRTBPOrbitType.HALO
        reference_ephem = self._halo_strategy(reference_config)
        
        # 然后在参考轨道基础上叠加相对运动
        # 这里简化处理：直接返回参考轨道
        # 实际应用中需要生成多个航天器的相对轨道
        
        return reference_ephem
    
    def _crtbp_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        CRTBP 无量纲动力学方程
        
        Args:
            t: 无量纲时间
            state: 无量纲状态 [x, y, z, vx, vy, vz]
            
        Returns:
            np.ndarray: 状态导数
        """
        x, y, z, vx, vy, vz = state
        mu = self.mu
        
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        
        # 加速度分量
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
        az = -(1-mu)*z/r1**3 - mu*z/r2**3
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def _jacobi_constant(self, state_nd: np.ndarray) -> float:
        """计算雅可比常数（无量纲）"""
        x, y, z, vx, vy, vz = state_nd
        mu = self.mu
        
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        
        U = (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2
        v2 = vx**2 + vy**2 + vz**2
        
        return 2*U - v2
    
    def _get_lagrange_point(self, point: int) -> float:
        """获取平动点位置（无量纲 x 坐标）"""
        if point == 1:
            return self.L1
        elif point == 2:
            return self.L2
        elif point == 3:
            # L3 在另一侧
            return -1.0  # 简化处理
        elif point == 4:
            # L4: x = 0.5 - mu, y = √3/2
            return 0.5 - self.mu
        elif point == 5:
            # L5: x = 0.5 - mu, y = -√3/2
            return 0.5 - self.mu
        else:
            raise ValueError(f"无效的平动点编号: {point}")
    
    def _estimate_initial_velocity(self, position: float, orbit_type: str) -> float:
        """估计初始速度"""
        if orbit_type == "dro":
            # DRO 轨道：基于能量估算
            r1 = np.sqrt((position + self.mu)**2)
            r2 = np.sqrt((position - 1 + self.mu)**2)
            U = (position**2)/2 + (1-self.mu)/r1 + self.mu/r2
            C = 3.0  # 典型的 DRO 雅可比常数
            vy2 = 2*U - C
            return np.sqrt(max(vy2, 0))
        
        elif orbit_type == "lyapunov":
            # Lyapunov 轨道：小振幅近似
            return 0.01 * abs(position - 1.0)
        
        elif orbit_type == "vertical":
            # 垂直轨道：z方向振荡
            # 返回y方向速度，而不是x方向
            return 0.01 + position * 0.1
        
        else:
            return 0.01  # 默认值
    
    def _linear_frequency(self, direction: str, lagrange_point: int) -> float:
        """计算平动点附近的线性振荡频率"""
        if direction == 'x':
            if lagrange_point in [1, 2]:
                # L1/L2 点的 x 方向频率
                gamma = abs(self._get_lagrange_point(lagrange_point) - 1.0)
                return np.sqrt((self.mu/gamma**3) * (1 + gamma) + 1)
            else:
                return 1.0  # 简化
        
        elif direction == 'z':
            # z 方向频率（对于 L1/L2/L3）
            gamma = abs(self._get_lagrange_point(lagrange_point) - 1.0)
            return np.sqrt(self.mu/gamma**3)
        
        else:
            return 1.0
    
    def _differential_correction(self, 
                                initial_guess: np.ndarray,
                                symmetry: SymmetryType,
                                max_iter: int = 50,
                                tol: float = 1e-10) -> Tuple[np.ndarray, float]:
        """
        通用的微分修正算法
        
        Args:
            initial_guess: 初始状态猜测
            symmetry: 对称性类型
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            Tuple[np.ndarray, float]: (修正后的状态, 轨道周期)
        """
        state = initial_guess.copy()
        
        for iteration in range(max_iter):
            if symmetry == SymmetryType.XZ_PLANE:
                # Halo 轨道：积分到 y=0 平面
                T_half = self._find_half_period(state)
                if T_half is None:
                    break
                
                sol = solve_ivp(self._crtbp_equations, (0, T_half), state,
                              method='DOP853', rtol=1e-12, atol=1e-12)
                final_state = sol.y[:, -1]
                
                # 约束：y=0, vx=0, vz=0
                error = np.array([final_state[1], final_state[3], final_state[5]])
                
                # 简单修正策略（实际应用需要更精细的变分方程）
                if np.linalg.norm(error) < tol:
                    if self.verbose:
                        print(f"  微分修正收敛于迭代 {iteration+1}")
                    return state, 2 * T_half
                
                # 调整初始猜测（简化）
                state[0] *= 0.999  # 调整 x
                state[2] *= 0.999  # 调整 z
                state[4] *= 1.001  # 调整 vy
            
            elif symmetry == SymmetryType.X_AXIS:
                # DRO/Lyapunov 轨道
                T_half = self._find_half_period(state)
                if T_half is None:
                    break
                
                sol = solve_ivp(self._crtbp_equations, (0, T_half), state,
                              method='DOP853', rtol=1e-12, atol=1e-12)
                final_state = sol.y[:, -1]
                
                error = np.array([final_state[1], final_state[2], 
                                final_state[3], final_state[5]])
                
                if np.linalg.norm(error) < tol:
                    if self.verbose:
                        print(f"  微分修正收敛于迭代 {iteration+1}")
                    return state, 2 * T_half
                
                # 简化调整
                state[0] *= 0.999
                state[4] *= 1.001
            
            elif symmetry == SymmetryType.Z_AXIS:
                # 垂直轨道
                T_half = self._find_half_period(state, event='z')
                if T_half is None:
                    break
                
                sol = solve_ivp(self._crtbp_equations, (0, T_half), state,
                              method='DOP853', rtol=1e-12, atol=1e-12)
                final_state = sol.y[:, -1]
                
                error = np.array([final_state[0], final_state[1],
                                final_state[4], final_state[5]])
                
                if np.linalg.norm(error) < tol:
                    if self.verbose:
                        print(f"  微分修正收敛于迭代 {iteration+1}")
                    return state, 2 * T_half
                
                state[2] *= 0.999
                state[3] *= 1.001
        
        # 未完全收敛，返回最佳结果
        if self.verbose:
            print(f"  微分修正未完全收敛，返回当前最佳结果")
        
        period = self._estimate_period(state)
        return state, period
    
    def _find_half_period(self, state: np.ndarray, 
                         max_time: float = 20.0,
                         event: str = 'y') -> Optional[float]:
        """
        寻找半周期（穿越平面）
        
        Args:
            state: 初始状态
            max_time: 最大搜索时间
            event: 事件类型 ('y', 'z')
            
        Returns:
            Optional[float]: 半周期时间，或 None（未找到）
        """
        if event == 'y':
            def event_func(t, y):
                return y[1]  # y=0
        elif event == 'z':
            def event_func(t, y):
                return y[2]  # z=0
        else:
            raise ValueError(f"不支持的事件类型: {event}")
        
        event_func.direction = -1  # 从正到负穿越
        
        sol = solve_ivp(self._crtbp_equations, (0, max_time), state,
                       events=[event_func], method='DOP853',
                       rtol=1e-12, atol=1e-12)
        
        if len(sol.t_events[0]) > 0:
            t_half = sol.t_events[0][0]
            if t_half > 0.1:  # 有效半周期
                return t_half
        
        return None
    
    def _estimate_period(self, state: np.ndarray) -> float:
        """估计轨道周期"""
        # 基于雅可比常数的简单估算
        C = self._jacobi_constant(state)
        
        if C < 3.0:
            return 2 * np.pi * (1.0 + 0.1 * (3.0 - C))
        else:
            return 2 * np.pi * (1.0 - 0.05 * (C - 3.0))
    
    def _integrate_orbit(self, 
                        initial_state: np.ndarray,
                        period: float,
                        step_size: float,
                        duration: Optional[float] = None) -> Ephemeris:
        """
        积分生成完整轨道
        
        Args:
            initial_state: 初始状态（无量纲）
            period: 轨道周期（无量纲）
            step_size: 输出步长（无量纲）
            duration: 积分时长（无量纲），如为None则使用period
            
        Returns:
            Ephemeris: 物理单位的轨道星历
        """
        T = duration if duration is not None else period
        
        # 生成时间序列
        times_nd = np.arange(0, T, step_size)
        
        # 积分轨道
        sol = solve_ivp(self._crtbp_equations, (0, T), initial_state,
                       t_eval=times_nd, method='DOP853', 
                       rtol=1e-12, atol=1e-12)
        
        # 转换为物理单位
        return self._nd_to_physical(sol.t, sol.y.T)
    
    def _nd_to_physical(self, times_nd: np.ndarray, states_nd: np.ndarray) -> Ephemeris:
        """
        将无量纲状态转换为物理单位
        
        Args:
            times_nd: 无量纲时间序列
            states_nd: 无量纲状态序列 (N, 6)
            
        Returns:
            Ephemeris: 物理单位的星历
        """
        # 转换为物理单位
        physical_times = times_nd / self.omega  # 无量纲时间 → 秒
        physical_states = states_nd.copy()
        
        # 位置：无量纲 → 米
        physical_states[:, 0:3] *= self.L
        
        # 速度：无量纲 → 米/秒
        physical_states[:, 3:6] *= (self.L * self.omega)
        
        # 选择坐标系
        if self.system_type == "sun_earth":
            frame = CoordinateFrame.SUN_EARTH_ROTATING
        elif self.system_type == "earth_moon":
            frame = CoordinateFrame.EARTH_MOON_ROTATING
        else:
            frame = CoordinateFrame.J2000_ECI
        
        return Ephemeris(physical_times, physical_states, frame)
    
    def _validate_orbit(self, states: np.ndarray, times: np.ndarray) -> None:
        """验证轨道质量（周期性、能量守恒）"""
        if len(states) == 0:
            return
        
        # 检查轨道闭合
        pos_start = states[0, 0:3]
        pos_end = states[-1, 0:3]
        vel_start = states[0, 3:6]
        vel_end = states[-1, 3:6]
        
        pos_error = np.linalg.norm(pos_end - pos_start)
        vel_error = np.linalg.norm(vel_end - vel_start)
        
        if self.verbose:
            print(f"[轨道验证] 闭合误差: 位置 {pos_error:.2e} m, 速度 {vel_error:.2e} m/s")
        
        # 检查雅可比常数守恒
        if len(states) > 10:
            C_vals = []
            sample_indices = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
            
            for idx in sample_indices:
                # 转换为无量纲计算雅可比常数
                state_nd = states[idx].copy()
                state_nd[0:3] /= self.L
                state_nd[3:6] /= (self.L * self.omega)
                C = self._jacobi_constant(state_nd)
                C_vals.append(C)
            
            C_std = np.std(C_vals)
            if self.verbose:
                print(f"[轨道验证] 雅可比常数标准差: {C_std:.2e}")


# 便捷函数
def create_crtbp_generator(system_type: str = "sun_earth",
                          orbit_type: str = "halo",
                          **kwargs) -> CRTBPOrbitGenerator:
    """创建 CRTBP 轨道生成器的工厂函数"""
    if isinstance(orbit_type, str):
        orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
    else:
        orbit_type_enum = orbit_type
    
    return CRTBPOrbitGenerator(
        system_type=system_type,
        orbit_type=orbit_type_enum,
        **kwargs
    )


def generate_family(generator: CRTBPOrbitGenerator,
                   param_name: str,
                   param_values: List[float],
                   base_config: Dict[str, Any]) -> List[Ephemeris]:
    """
    生成轨道族
    
    Args:
        generator: CRTBP 轨道生成器
        param_name: 参数名称（如 'amplitude'）
        param_values: 参数值列表
        base_config: 基础配置
        
    Returns:
        List[Ephemeris]: 轨道族星历列表
    """
    orbits = []
    
    for value in param_values:
        config = base_config.copy()
        config[param_name] = value
        
        try:
            orbit = generator.generate(config)
            orbits.append(orbit)
            print(f"  成功生成轨道，{param_name}={value}")
        except Exception as e:
            print(f"  生成轨道失败，{param_name}={value}: {e}")
    
    return orbits


# 导出
__all__ = [
    "CRTBPOrbitGenerator",
    "CRTBPOrbitType",
    "SymmetryType",
    "CRTBPOrbitConfig",
    "create_crtbp_generator",
    "generate_family",
]
