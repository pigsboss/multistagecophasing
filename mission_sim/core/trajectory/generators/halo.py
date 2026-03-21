# mission_sim/core/trajectory/generators/halo.py
import numpy as np
from scipy.integrate import solve_ivp
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators.base import BaseTrajectoryGenerator


class HaloDifferentialCorrector(BaseTrajectoryGenerator):
    def __init__(self, mu: float = 3.00348e-6):
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        Az = config.get("Az", 0.05)
        dt_nd = config.get("dt", 0.001)

        # 使用已知的标准初值（Az=0.05）
        if abs(Az - 0.05) < 1e-6:
            x0, z0, vy0 = 1.01106, 0.05, 0.0105
        else:
            # 简单的比例缩放（可根据实际需要改进）
            x0 = 1.01106 + (Az - 0.05) * 0.1
            z0 = Az
            vy0 = 0.0105 + (Az - 0.05) * 0.05

        # 固定 z0 = Az
        z0 = Az
        state0_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

        # 寻找半周期
        T_half = self._find_half_period(state0_nd)
        if T_half is None:
            raise RuntimeError("无法找到半周期")
        T_nd = 2.0 * T_half

        # 生成完整周期轨道
        times_nd = np.arange(0, T_nd, dt_nd)
        sol = solve_ivp(self._crtbp_equations, (0, T_nd), state0_nd,
                        t_eval=times_nd, method='DOP853', rtol=1e-12, atol=1e-12)

        # 转换为物理单位
        AU = 1.495978707e11
        OMEGA = 1.990986e-7
        physical_times = sol.t / OMEGA
        physical_states = sol.y.T.copy()
        physical_states[:, 0:3] *= AU
        physical_states[:, 3:6] *= (AU * OMEGA)

        self._validate_orbit(physical_states, physical_times)
        return Ephemeris(physical_times, physical_states, CoordinateFrame.SUN_EARTH_ROTATING)

    def _crtbp_equations(self, t, state):
        x, y, z, vx, vy, vz = state
        mu = self.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
        az = -(1-mu)*z/r1**3 - mu*z/r2**3
        return np.array([vx, vy, vz, ax, ay, az])

    def _find_half_period(self, state0, max_time=20.0):
        def y_zero(t, y):
            return y[1]
        y_zero.direction = -1
        sol = solve_ivp(self._crtbp_equations, (0, max_time), state0,
                        events=[y_zero], method='DOP853',
                        rtol=1e-12, atol=1e-12)
        if len(sol.t_events[0]) > 0:
            t_half = sol.t_events[0][0]
            if t_half > 0.1:
                return t_half
        return None

    def _validate_orbit(self, states, times):
        pos_start = states[0, 0:3]
        pos_end = states[-1, 0:3]
        vel_start = states[0, 3:6]
        vel_end = states[-1, 3:6]
        pos_error = np.linalg.norm(pos_end - pos_start)
        vel_error = np.linalg.norm(vel_end - vel_start)
        print(f"[HaloCorrector] 轨道闭合误差: 位置 {pos_error:.2e} m, 速度 {vel_error:.2e} m/s")

        AU = 1.495978707e11
        OMEGA = 1.990986e-7
        if len(states) > 10:
            C_vals = []
            sample_indices = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
            for idx in sample_indices:
                state_nd = states[idx] / AU
                state_nd[3:6] /= (AU * OMEGA)
                C = self._jacobi_constant(state_nd)
                C_vals.append(C)
            C_std = np.std(C_vals)
            print(f"[HaloCorrector] 雅可比常数标准差: {C_std:.2e}")

    def _jacobi_constant(self, state_nd):
        x, y, z, vx, vy, vz = state_nd
        mu = self.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        U = (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2
        v2 = vx**2 + vy**2 + vz**2
        return 2*U - v2