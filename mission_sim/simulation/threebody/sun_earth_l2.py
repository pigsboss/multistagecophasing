"""日地 L2 点 L1 级仿真"""
import numpy as np
from scipy.integrate import solve_ivp
from mission_sim.simulation.threebody.base import ThreeBodyBaseSimulation
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.types import CoordinateFrame

class SunEarthL2L1Simulation(ThreeBodyBaseSimulation):
    def __init__(self, config):
        # 默认配置
        default_config = {
            "mu": 3.00348e-6,
            "L": 1.495978707e11,
            "omega": 1.990986e-7,
            "spacecraft_mass": 6200.0,
            "time_step": 60.0,
            "simulation_days": 30,
            "injection_error": np.array([2000,2000,-1000,0.01,-0.01,0.005]),
        }
        default_config.update(config)
        super().__init__(default_config)

    def _generate_nominal_orbit(self):
        # 使用已知初值积分生成 Halo 轨道（不追求严格周期）
        mu = self.config["mu"]
        Az = self.config.get("Az", 0.05)
        # 获取初值（可配置）
        if "initial_state_nd" in self.config:
            state0_nd = np.array(self.config["initial_state_nd"])
        else:
            # 默认初值（Az=0.05）
            x0 = 1.01106
            z0 = Az
            vy0 = 0.0105
            state0_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
        # 积分一个周期（默认 3.1416 无量纲时间）
        T_nd = self.config.get("period_nd", 3.1416)
        dt_nd = self.config.get("dt_nd", 0.001)
        t_nd = np.arange(0, T_nd, dt_nd)
        sol = solve_ivp(self.crtbp.dynamics, (0, T_nd), state0_nd,
                        t_eval=t_nd, method='DOP853', rtol=1e-12, atol=1e-12)
        # 转换为物理单位
        states_phys = sol.y.T.copy()
        states_phys[:, 0:3] *= self.crtbp.L
        states_phys[:, 3:6] *= self.crtbp.vel_scale
        times_phys = sol.t / self.crtbp.T
        self.ephemeris = Ephemeris(times_phys, states_phys,
                                   CoordinateFrame.SUN_EARTH_ROTATING)

    def _design_control_law(self):
        # 基于 CRTBP 线性化的 LQR
        mu = self.crtbp.mu
        gamma = np.cbrt(mu / 3.0)
        omega = self.crtbp.T  # 特征角速度
        A = np.zeros((6,6))
        A[0:3,3:6] = np.eye(3)
        A[3,0] = (2*gamma+1)*omega**2
        A[4,1] = (1-gamma)*omega**2
        A[5,2] = -gamma*omega**2
        A[3,4] = 2*omega
        A[4,3] = -2*omega
        B = np.zeros((6,3))
        B[3:6,0:3] = np.eye(3) / self.spacecraft.mass
        Q = np.diag([1.0,1.0,1.0,1e6,1e6,1e6])
        R = np.diag([10.0,10.0,10.0])
        from mission_sim.utils.math_tools import get_lqr_gain
        self.k_matrix = get_lqr_gain(A, B, Q, R)
        # 推力限幅
        self.k_matrix = self.k_matrix  # 限幅在 GNC 中实现