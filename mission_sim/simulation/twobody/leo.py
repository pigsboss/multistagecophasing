# mission_sim/simulation/twobody/leo.py
"""
低地球轨道 (LEO) L1 级仿真场景
适用于高度 200-2000km 的近地轨道任务，考虑 J2 摄动和大气阻力。
"""

import numpy as np
from typing import Optional

from mission_sim.simulation.twobody.base import TwoBodyBaseSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.generators import KeplerianGenerator, J2KeplerianGenerator
from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag
from mission_sim.utils.math_tools import elements_to_cartesian, absolute_to_lvlh, compute_lvlh_dcm


class LEOL1Simulation(TwoBodyBaseSimulation):
    """
    LEO 轨道 L1 级仿真
    继承自 TwoBodyBaseSimulation，实现 LEO 任务的标称轨道生成、环境初始化及控制律设计。
    """

    def __init__(self, config: dict):
        # 设置 LEO 默认配置
        leo_defaults = {
            "mu_earth": 3.986004418e14,
            "enable_j2": True,
            "enable_atmospheric_drag": True,
            "area_to_mass": 0.02,
            "Cd": 2.2,
            "rho0": 1.225,
            "H": 8500.0,
            "h0": 0.0,
            "spacecraft_mass": 1000.0,
            "injection_error": np.zeros(6),
            "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
            "use_j2_generator": True,
            "dt": 10.0,
            "sim_time": None,
        }
        for key, value in leo_defaults.items():
            if key not in config:
                config[key] = value
        super().__init__(config)

        self.enable_atmospheric_drag = self.config.get("enable_atmospheric_drag", True)
        self.area_to_mass = self.config.get("area_to_mass", 0.02)
        self.Cd = self.config.get("Cd", 2.2)
        self.rho0 = self.config.get("rho0", 1.225)
        self.H = self.config.get("H", 8500.0)
        self.h0 = self.config.get("h0", 0.0)

    def _generate_nominal_orbit(self) -> bool:
        try:
            sim_seconds = self.config["simulation_days"] * 86400
            dt = self.config.get("dt", 10.0)
            elements = self.config.get("elements")
            if elements is None or len(elements) != 6:
                raise ValueError("LEO 场景需要提供 6 个轨道根数 'elements'。")

            use_j2 = self.config.get("use_j2_generator", True)
            if use_j2:
                # Try with mu parameter first, fall back without if it fails
                try:
                    generator = J2KeplerianGenerator(mu=self.mu_earth)
                except TypeError:
                    # If J2KeplerianGenerator doesn't accept mu parameter
                    generator = J2KeplerianGenerator()
                gen_config = {
                    "elements": elements,
                    "dt": dt,
                    "sim_time": sim_seconds,
                    "integrator": "DOP853",
                    "rtol": 1e-12
                }
                self.ephemeris = generator.generate(gen_config)
            else:
                generator = KeplerianGenerator(mu=self.mu_earth)
                gen_config = {
                    "elements": elements,
                    "dt": dt,
                    "sim_time": sim_seconds
                }
                self.ephemeris = generator.generate(gen_config)

            # 强制第一个状态为理论值
            a, e, i, Omega, omega, M0 = elements
            state0_theoretical = elements_to_cartesian(self.mu_earth, a, e, i, Omega, omega, M0)
            self.ephemeris.states[0] = state0_theoretical
            self.ephemeris.times[0] = 0.0

            if self.verbose:
                print(f"✅ LEO 标称轨道生成完成，时长 {sim_seconds/86400:.1f} 天，点数 {len(self.ephemeris.times)}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ LEO 标称轨道生成失败: {e}")
            return False

    def _initialize_physical_domain(self):
        super()._initialize_physical_domain()
        if self.enable_atmospheric_drag:
            drag_model = AtmosphericDrag(
                area_to_mass=self.area_to_mass,
                Cd=self.Cd,
                rho0=self.rho0,
                H=self.H,
                h0=self.h0,
                R_earth=6378137.0
            )
            self.environment.register_force(drag_model)
            if self.verbose:
                print(f"✅ 已注册大气阻力模型 (A/m={self.area_to_mass:.4f} m²/kg, Cd={self.Cd:.2f})")

    def _design_control_law(self):
        """设计 LEO 轨道维持控制律（使用 LQR）"""
        if self.ephemeris is not None:
            r0 = np.linalg.norm(self.ephemeris.states[0, 0:3])
            altitude = r0 - 6378137.0
        else:
            altitude = 7000e3 - 6378137.0

        K_base = self._compute_j2_lqr_gain(altitude)
        gain_scale = self.config.get("control_gain_scale", 1.0)
        self.k_matrix = gain_scale * K_base

        # G-05 防调参（DD-001 D-03）：增益超出设计域 [0.1, 10] 必须显式告警（无条件）
        self.gain_out_of_domain = not (0.1 <= gain_scale <= 10.0)
        if self.gain_out_of_domain:
            print(f"⚠️ [G-05] control_gain_scale={gain_scale:.3g} 超出设计域 [0.1, 10]；"
                  f"结果仅供诊断，不得用于验收")

        if self.verbose:
            print(f"✅ LQR 控制律设计完成，增益缩放因子: {gain_scale:.2f}")

    def _compute_control(self, epoch: float, obs_state: Optional[np.ndarray], frame: CoordinateFrame):
        """实现 LVLH 控制（误差与控制力统一经 GNCSubsystem 记录，DD-001 D-02）"""
        self.gnc_system.update_navigation(obs_state, frame, self.config["time_step"])

        self.gnc_system.compute_tracking_error(epoch)
        target_state = self.gnc_system.last_target_state
        nav_state = self.gnc_system.current_nav_state

        # LVLH 误差状态（旋转系相对量，含 ω×Δr 输运项；DD-001 G-03）
        # 修复记录（WP-1.3）：原实现仅对惯性相对速度做旋转、缺失输运项，
        # 导致闭环谱半径 1.0007/步（慢发散，e 折约 4 小时）。
        error_lvlh_pos, error_lvlh_vel = absolute_to_lvlh(
            target_state[0:3], target_state[3:6], nav_state[0:3], nav_state[3:6])
        error_lvlh = np.concatenate([error_lvlh_pos, error_lvlh_vel])

        force_lvlh = -self.k_matrix @ error_lvlh

        # LVLH 力 → 惯性系（与 absolute_to_lvlh 同一 DCM）
        C_I2L = compute_lvlh_dcm(target_state[0:3], target_state[3:6])
        force_inertial = C_I2L.T @ force_lvlh

        # 唯一通道：登记实际下发控制力（供显示/HDF5/报告）
        self.gnc_system.record_control_force(force_inertial)

        return force_inertial, CoordinateFrame.J2000_ECI

    def _generate_fallback_orbit(self):
        print("   使用备用轨道生成方案（开普勒圆轨道，高度 500km）...")
        from mission_sim.core.spacetime.generators import KeplerianGenerator
        a = 6878137.0
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        M0 = 0.0
        elements = [a, e, i, Omega, omega, M0]
        sim_seconds = self.config["simulation_days"] * 86400
        dt = self.config.get("dt", 10.0)
        config = {
            "elements": elements,
            "dt": dt,
            "sim_time": sim_seconds
        }
        gen = KeplerianGenerator(mu=self.mu_earth)
        self.ephemeris = gen.generate(config)
        print(f"✅ 备用轨道生成完成，周期: {2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 86400:.2f} 天")
