# mission_sim/simulation/threebody/earth_moon.py
"""
MCPC: 地月共振摆动轨道演化专用场景（EarthMoonResonanceSimulation）
==================================================================
LUNAR-SWING 专题沉淀件（LS-001 G5-④）。

自由演化实验场景：在 J2000 ECI 惯性系下积分航天器轨道，力模型可组合：
    中心引力（地球）+ 带谐重力场 J2–J4 + 月球第三体（星历回调/SPICE）+ 简化指数大气（可选）。
输出时间序列 HDF5：epochs / states / altitude_m / moon_distance_m / energy_m2s2。
积分器：scipy solve_ivp DOP853（自适应），采样间隔 time_step。

限制声明：简化指数大气；月球以瞬时真实状态（SPICE）或注入回调给出；不做控制。
"""

import os

import h5py
import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
MU_MOON = 4.9048695e12


def crtbp_state_to_eci(state_nd, moon_r, moon_v, L, omega, mu):
    """CRTBP 无量纲状态（旋转系，质心原点）→ J2000 ECI 物理状态（地心原点）。

    旋转系基向量由月球瞬时状态构造：d̂ = r_m/|r_m|（x 轴），
    ĥ = (r_m×v_m)/|·|（z 轴），ŷ = ĥ×d̂；
    地心位置 r_eci = L·(x_nd·d̂ + y_nd·ŷ) + μ·L·d̂（质心→地心偏移）；
    速度 v_eci = L·ω·(v_nd 变换) + ω⃗×r_eci，ω⃗ = ω·ĥ。
    """
    state_nd = np.asarray(state_nd, dtype=np.float64)
    moon_r = np.asarray(moon_r, dtype=np.float64)
    moon_v = np.asarray(moon_v, dtype=np.float64)
    d_hat = moon_r / np.linalg.norm(moon_r)
    h_hat = np.cross(moon_r, moon_v)
    h_hat = h_hat / np.linalg.norm(h_hat)
    y_hat = np.cross(h_hat, d_hat)
    B = np.vstack([d_hat, y_hat, h_hat])  # 行 = 旋转系基向量（ECI 表示）

    r_eci = L * (B.T @ state_nd[:3]) + (float(mu) * L) * d_hat
    v_frame = (L * omega) * (B.T @ state_nd[3:])
    omega_vec = omega * h_hat
    v_eci = v_frame + np.cross(omega_vec, r_eci)
    return np.concatenate([r_eci, v_eci])


class EarthMoonResonanceSimulation:
    """地月共振演化场景（自由演化，无控制）。"""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.verbose = bool(self.config.get("verbose", True))
        self.data_dir = self.config.get("data_dir", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        mission = self.config.get("mission_name", "EM_resonance")
        self.h5_file = os.path.join(self.data_dir, f"{mission}.h5")
        self._moon_prov = None
        self.summary = {}

    # ------------------------------------------------------------------
    def _moon_provider(self):
        prov = self.config.get("moon_provider")
        if callable(prov):
            return lambda t: np.asarray(prov(float(t)), dtype=np.float64)[:6]
        try:
            from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
            from mission_sim.core.spacetime.ids import CoordinateFrame
            spice = SPICEInterface()
            if not spice.initialize("earth_moon"):
                raise RuntimeError("SPICE initialize returned False")
            frame = CoordinateFrame.J2000_ECI

            def wrapped(t):
                return np.asarray(spice.get_state("moon", float(t), observer="earth",
                                                  frame=frame), dtype=np.float64)
            if self.verbose:
                print("[EarthMoon] 使用 SPICE 真实月球星历（DE 内核）")
            return wrapped
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"未提供 moon_provider 且 SPICE 不可用：{e}") from e

    def _build_accel(self):
        from mission_sim.core.physics.environment import CelestialEnvironment
        from mission_sim.core.spacetime.ids import CoordinateFrame
        from mission_sim.core.physics.models.central_gravity import CentralGravity
        from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
        from mission_sim.core.physics.models.third_body import ThirdBodyGravity
        from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag

        env = CelestialEnvironment(CoordinateFrame.J2000_ECI, verbose=False)
        env.register_force(CentralGravity(mu=MU_EARTH))
        if self.config.get("enable_j2", True):
            env.register_force(HighOrderGeopotential(
                mu_earth=MU_EARTH,
                j2=self.config.get("j2", None),
                j3=self.config.get("j3", None),
                j4=self.config.get("j4", None)))
        if self.config.get("enable_moon", True):
            self._moon_prov = self._moon_provider()
            env.register_force(ThirdBodyGravity(
                mu_third=MU_MOON, position_provider=self._moon_prov, name="Moon"))
        if self.config.get("enable_drag", False):
            env.register_force(AtmosphericDrag(
                area_to_mass=self.config.get("area_to_mass", 0.02),
                Cd=self.config.get("Cd", 2.2),
                rho0=self.config.get("rho0", 1.225),
                H=self.config.get("H", 8500.0),
                h0=0.0, R_earth=R_EARTH))

        frame = CoordinateFrame.J2000_ECI

        def accel(t, state):
            # 时间相关力（月球星历）必须按当前历元求值：环境默认使用 self.epoch
            env.epoch = float(t)
            a, _ = env.get_total_acceleration(np.asarray(state, dtype=np.float64), frame)
            return np.asarray(a, dtype=np.float64)

        return accel

    # ------------------------------------------------------------------
    def run(self) -> bool:
        try:
            accel = self._build_accel()
            state0 = np.asarray(self.config["initial_state"], dtype=np.float64)
            t_end = float(self.config["simulation_days"]) * 86400.0
            dt_out = float(self.config.get("time_step", 600.0))
            t_eval = np.arange(0.0, t_end, dt_out)
            if t_eval.size == 0 or t_eval[-1] < t_end:
                t_eval = np.append(t_eval, t_end)

            sol = solve_ivp(
                lambda t, y: np.concatenate([y[3:6], accel(t, y)]),
                (0.0, t_end), state0, t_eval=t_eval, method="DOP853",
                rtol=float(self.config.get("rtol", 1e-10)),
                atol=float(self.config.get("atol", 1e-6)))
            if not sol.success:
                print(f"❌ [EarthMoon] 积分失败: {sol.message}")
                return False

            self._write(sol.t, sol.y.T)
            self._make_summary(sol.t, sol.y.T)
            if self.verbose:
                self._print_summary()
            return True
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"❌ [EarthMoon] 场景运行失败: {e}")
            return False

    def _write(self, times, states):
        r_norm = np.linalg.norm(states[:, 0:3], axis=1)
        alt = r_norm - R_EARTH
        energy = 0.5 * np.sum(states[:, 3:6] ** 2, axis=1) - MU_EARTH / r_norm
        if self._moon_prov is not None:
            moon_dist = np.array([
                np.linalg.norm(states[i, 0:3] - self._moon_prov(times[i])[:3])
                for i in range(times.size)])
        else:
            moon_dist = np.full(times.shape, np.nan)
        with h5py.File(self.h5_file, "w") as f:
            f.create_dataset("epochs", data=times)
            f.create_dataset("states", data=states)
            f.create_dataset("altitude_m", data=alt)
            f.create_dataset("moon_distance_m", data=moon_dist)
            f.create_dataset("energy_m2s2", data=energy)
            f.attrs["mission_name"] = self.config.get("mission_name", "EM_resonance")
            f.attrs["simulation_days"] = float(self.config["simulation_days"])
            f.attrs["enable_j2"] = bool(self.config.get("enable_j2", True))
            f.attrs["enable_moon"] = bool(self.config.get("enable_moon", True))
            f.attrs["enable_drag"] = bool(self.config.get("enable_drag", False))

    def _make_summary(self, times, states):
        alt = np.linalg.norm(states[:, 0:3], axis=1) - R_EARTH
        days = float(self.config["simulation_days"])
        self.summary = {
            "days": days,
            "alt_min_m": float(alt.min()),
            "alt_max_m": float(alt.max()),
            "alt_final_m": float(alt[-1]),
            "alt_decay_m_per_day": float((alt[0] - alt[-1]) / max(days, 1e-12)),
            "r_max_m": float(np.linalg.norm(states[:, 0:3], axis=1).max()),
            "bounded": bool(np.linalg.norm(states[:, 0:3], axis=1).max() < 2.0e9),
        }

    def _print_summary(self):
        s = self.summary
        print("\n📊 [EarthMoon] 演化实验汇总")
        print("-" * 60)
        print(f"   时长: {s['days']:.1f} 天 | 近地点最低高度: {s['alt_min_m']/1e3:.1f} km")
        print(f"   高度衰减: {s['alt_decay_m_per_day']/1e3:.3f} km/天 | 末期高度: {s['alt_final_m']/1e3:.1f} km")
        print(f"   有界性: {'有界' if s['bounded'] else '⚠️ 无界'} | r_max={s['r_max_m']:.3e} m")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
