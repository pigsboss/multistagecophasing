# mission_sim/simulation/twobody/base.py
"""
二体场景仿真基类
适用于 LEO/GEO/HEO 等以中心引力为主导的轨道任务。
封装 J2 摄动、地面测控、GNC 等通用模块，具体轨道生成与控制律由子类实现。
"""

import numpy as np
from abc import abstractmethod
from typing import Optional

from mission_sim.simulation.base import BaseSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.core.cyber.platform_gnc.ground_station import GroundStation
from mission_sim.core.cyber.platform_gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.core.physics.spacecraft import SpacecraftPointMass


class TwoBodyBaseSimulation(BaseSimulation):
    """
    二体场景仿真基类
    提供二体环境下通用的物理域、信息域初始化，以及参考轨迹加载。
    子类需实现：
        - _generate_nominal_orbit()   : 生成标称轨道星历
        - _design_control_law()       : 设计控制律（计算反馈增益矩阵）
    """

    def __init__(self, config: dict):
        """
        初始化二体场景基类。

        Args:
            config: 仿真配置字典，至少包含以下字段：
                - mission_name: 任务名称
                - simulation_days: 仿真天数
                - time_step: 积分步长 (s)
                - data_dir: 数据输出目录
                - verbose: 是否输出详细信息（默认 True）
                - mu_earth: 地球引力常数 (m³/s²) [可选，默认 3.986004418e14]
                - enable_j2: 是否启用 J2 摄动 [默认 True]
                - spacecraft_mass: 航天器质量 (kg) [默认 1000]
                - injection_error: 初始状态注入误差 [默认 [0,0,0,0,0,0]]
                - elements: 轨道根数 [a, e, i, Omega, omega, M0] (必须提供)
                - pos_noise_std, vel_noise_std, sampling_rate_hz, visibility_windows: 地面站参数
                - propagator_type: 盲区外推器类型 ('simple', 'kepler', None)
                - propagator_mu: 外推器引力常数（若使用 kepler 外推器）
        """
        # 设置默认的二体参数
        default_config = {
            "mu_earth": 3.986004418e14,
            "enable_j2": True,
            "spacecraft_mass": 1000.0,
            "injection_error": np.zeros(6, dtype=np.float64),
        }
        default_config.update(config)
        super().__init__(default_config)

        # 提取常用参数
        self.mu_earth = self.config.get("mu_earth", 3.986004418e14)
        self.enable_j2 = self.config.get("enable_j2", True)

        # 确保轨道根数存在
        self.elements = self.config.get("elements")
        if self.elements is None or len(self.elements) != 6:
            raise ValueError("二体场景必须提供 6 个轨道根数 'elements'。")

    # ====================== 抽象方法（子类必须实现） ======================
    @abstractmethod
    def _generate_nominal_orbit(self) -> bool:
        """
        生成标称轨道星历表。
        子类必须实现此方法，将结果存入 self.ephemeris。
        Returns:
            bool: 是否成功生成（若失败，基类会尝试调用 _generate_fallback_orbit）
        """
        pass

    @abstractmethod
    def _design_control_law(self):
        """
        设计控制律（计算反馈增益矩阵）。
        子类必须实现此方法，将结果存入 self.k_matrix。
        """
        pass

    # ====================== 可重写方法 ======================
    def _initialize_physical_domain(self):
        """
        初始化物理域（环境、航天器）。
        直接使用理论值计算初始状态，避免星历插值误差。
        """
        # 创建环境引擎（地心惯性系）
        self.environment = CelestialEnvironment(
            computation_frame=CoordinateFrame.J2000_ECI,
            initial_epoch=0.0,
            verbose=self.verbose
        )

        # 注册 J2 摄动（若启用）
        if self.enable_j2:
            j2_model = J2Gravity(mu_earth=self.mu_earth)
            self.environment.register_force(j2_model)

        # 直接使用理论值计算初始状态，避免星历插值误差
        from mission_sim.utils.math_tools import elements_to_cartesian
        a, e, i, Omega, omega, M0 = self.elements
        nom0_phys = elements_to_cartesian(self.mu_earth, a, e, i, Omega, omega, M0)
        injection = self.config.get("injection_error", np.zeros(6, dtype=np.float64))
        true0_phys = nom0_phys + injection

        # 创建航天器模型
        self.spacecraft = SpacecraftPointMass(
            sc_id="SC",
            initial_state=true0_phys,
            frame=CoordinateFrame.J2000_ECI,
            initial_mass=self.config.get("spacecraft_mass", 1000.0)
        )

    def _initialize_information_domain(self):
        """
        初始化信息域（测控、GNC）。
        创建地面站、GNC 子系统，加载参考轨迹，并配置盲区外推器（若指定）。
        """
        frame = CoordinateFrame.J2000_ECI

        # 地面站配置
        self.ground_station = GroundStation(
            name="DSN",
            operating_frame=frame,
            pos_noise_std=self.config.get("pos_noise_std", 5.0),
            vel_noise_std=self.config.get("vel_noise_std", 0.005),
            sampling_rate_hz=self.config.get("sampling_rate_hz", 0.1),
            visibility_windows=self.config.get("visibility_windows")
        )

        # GNC 子系统
        self.gnc_system = GNC_Subsystem("SC", operating_frame=frame, verbose=self.verbose)
        self.gnc_system.load_reference_trajectory(self.ephemeris)

        # 用航天器的初始状态初始化导航状态（避免初始巨大误差）
        self.gnc_system.current_nav_state = self.spacecraft.state.copy()

        # 盲区外推器配置
        prop_type = self.config.get("propagator_type")
        if prop_type == "simple":
            from mission_sim.core.cyber.platform_gnc.propagator import SimplePropagator
            self.gnc_system.set_propagator(SimplePropagator())
        elif prop_type == "kepler":
            from mission_sim.core.cyber.platform_gnc.propagator import KeplerPropagator
            mu = self.config.get("propagator_mu", self.mu_earth)
            self.gnc_system.set_propagator(KeplerPropagator(mu))

    # ====================== 辅助方法（供子类使用） ======================
    def _compute_j2_lqr_gain(self, altitude: float) -> np.ndarray:
        """
        基于 J2 摄动线性化模型计算 LQR 反馈增益矩阵。
        适用于圆轨道，考虑 J2 摄动对状态矩阵的影响。
        （简化版本，可被子类调用或重写）

        Args:
            altitude: 轨道高度 (m)

        Returns:
            K: 反馈增益矩阵 (3x6)
        """
        # 轨道半径
        r = altitude + 6378137.0
        omega = np.sqrt(self.mu_earth / r**3)  # 轨道角速度 (rad/s)

        # 状态矩阵 A (6x6) [位置, 速度] 在 LVLH 系下的线性化（简化）
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 0] = 3 * omega**2
        A[4, 1] = 0  # 忽略 J2 影响简化
        A[5, 2] = -omega**2
        A[3, 4] = 2 * omega
        A[4, 3] = -2 * omega

        # 控制矩阵 B (6x3)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.spacecraft.mass

        # 权重矩阵（可配置）
        Q = np.diag([1.0, 1.0, 1.0, 1e4, 1e4, 1e4])
        R = np.diag([1e6, 1e6, 1e6])

        from mission_sim.utils.math_tools import get_lqr_gain
        return get_lqr_gain(A, B, Q, R)

    def _generate_fallback_orbit(self):
        """
        备用轨道生成（当主生成失败时调用）。
        默认实现为使用开普勒解析解生成圆轨道（高度 7000km）。
        子类可重写以提供更合适的备用方案。
        """
        print("   使用备用轨道生成方案（开普勒圆轨道）...")
        from mission_sim.core.spacetime.generators import KeplerianGenerator
        a = 7000e3  # 半径 7000km
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        M0 = 0.0
        elements = [a, e, i, Omega, omega, M0]
        config = {
            "elements": elements,
            "dt": self.config["time_step"],
            "sim_time": self.config["simulation_days"] * 86400
        }
        gen = KeplerianGenerator(mu=self.mu_earth)
        self.ephemeris = gen.generate(config)
        print(f"✅ 备用轨道生成完成，周期: {2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 86400:.2f} 天")