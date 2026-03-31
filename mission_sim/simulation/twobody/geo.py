# mission_sim/simulation/twobody/geo.py
"""
地球静止轨道 (GEO) L1 级仿真场景
"""

import numpy as np
from mission_sim.simulation.twobody.base import TwoBodyBaseSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.generators import KeplerianGenerator, J2KeplerianGenerator
from mission_sim.utils.math_tools import elements_to_cartesian


class GEOL1Simulation(TwoBodyBaseSimulation):
    GEO_RADIUS = 42164000.0
    GEO_PERIOD_DAYS = 1.0

    def __init__(self, config: dict):
        geo_defaults = {
            "mu_earth": 3.986004418e14,
            "enable_j2": True,
            "enable_atmospheric_drag": False,
            "spacecraft_mass": 2000.0,
            "injection_error": np.zeros(6),
            "elements": [self.GEO_RADIUS, 0.0, 0.0, 0.0, 0.0, 0.0],
            "use_j2_generator": True,
            "dt": 60.0,
            "sim_time": None,
        }
        for key, value in geo_defaults.items():
            if key not in config:
                config[key] = value
        super().__init__(config)

        self.config["enable_atmospheric_drag"] = False

    def _generate_nominal_orbit(self) -> bool:
        try:
            sim_seconds = self.config["simulation_days"] * 86400
            dt = self.config.get("dt", 60.0)
            elements = self.config.get("elements")
            if elements is None or len(elements) != 6:
                raise ValueError("GEO 场景需要提供 6 个轨道根数 'elements'。")

            use_j2 = self.config.get("use_j2_generator", True)
            if use_j2:
                generator = J2KeplerianGenerator(mu=self.mu_earth)
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

            a, e, i, Omega, omega, M0 = elements
            state0_theoretical = elements_to_cartesian(self.mu_earth, a, e, i, Omega, omega, M0)
            self.ephemeris.states[0] = state0_theoretical
            self.ephemeris.times[0] = 0.0

            if self.verbose:
                print(f"✅ GEO 标称轨道生成完成，时长 {sim_seconds/86400:.1f} 天，点数 {len(self.ephemeris.times)}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"❌ GEO 标称轨道生成失败: {e}")
            return False

    def _design_control_law(self):
        if self.ephemeris is not None:
            r0 = np.linalg.norm(self.ephemeris.states[0, 0:3])
            altitude = r0 - 6378137.0
        else:
            altitude = self.GEO_RADIUS - 6378137.0

        K_base = self._compute_j2_lqr_gain(altitude)
        gain_scale = self.config.get("control_gain_scale", 1.0)
        self.k_matrix = gain_scale * K_base

        if self.verbose:
            print(f"✅ LQR 控制律设计完成，增益缩放因子: {gain_scale:.2f}")

    def _compute_control(self, epoch: float, obs_state, frame):
        self.gnc_system.update_navigation(obs_state, frame, self.config["time_step"])

        target_state = self.ephemeris.get_interpolated_state(epoch)
        nav_state = self.gnc_system.current_nav_state
        error = nav_state - target_state

        r_nom = target_state[0:3]
        v_nom = target_state[3:6]
        norm_r = np.linalg.norm(r_nom)
        i_hat = r_nom / norm_r
        h_vec = np.cross(r_nom, v_nom)
        norm_h = np.linalg.norm(h_vec)
        k_hat = h_vec / norm_h
        j_hat = np.cross(k_hat, i_hat)
        C_I2L = np.vstack([i_hat, j_hat, k_hat])

        error_lvlh = np.zeros(6)
        error_lvlh[0:3] = C_I2L @ error[0:3]
        error_lvlh[3:6] = C_I2L @ error[3:6]

        force_lvlh = -self.k_matrix @ error_lvlh
        force_inertial = C_I2L.T @ force_lvlh

        return force_inertial, CoordinateFrame.J2000_ECI

    def _generate_fallback_orbit(self):
        print("   使用备用轨道生成方案（开普勒圆轨道，标准 GEO）...")
        from mission_sim.core.spacetime.generators import KeplerianGenerator
        a = self.GEO_RADIUS
        e = 0.0
        i = 0.0
        Omega = 0.0
        omega = 0.0
        M0 = 0.0
        elements = [a, e, i, Omega, omega, M0]
        sim_seconds = self.config["simulation_days"] * 86400
        dt = self.config.get("dt", 60.0)
        config = {
            "elements": elements,
            "dt": dt,
            "sim_time": sim_seconds
        }
        gen = KeplerianGenerator(mu=self.mu_earth)
        self.ephemeris = gen.generate(config)
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 86400
        print(f"✅ 备用轨道生成完成，周期: {period:.2f} 天")
