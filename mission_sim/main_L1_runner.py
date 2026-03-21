# mission_sim/main_L1_runner.py
"""
MCPC 框架 L1 级仿真主程序 - 增强版
支持多轨道类型选择和力学模型自动注册
JWST 标称轨道 30 天全链路仿真
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp

# 核心类型与领域模型
from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators import create_generator
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.physics.models.gravity_crtbp import Gravity_CRTBP
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.core.physics.models.srp import Cannonball_SRP
from mission_sim.core.gnc.ground_station import GroundStation
from mission_sim.core.gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.core.gnc.propagator import SimplePropagator, KeplerPropagator

# 基础设施工具
from mission_sim.utils.logger import HDF5Logger, SimulationMetadata
from mission_sim.utils.math_tools import get_lqr_gain
from mission_sim.utils.visualizer_L1 import L1Visualizer


class L1MissionSimulation:
    """
    L1 级任务仿真控制器 - 增强版
    支持多轨道类型选择、力学模型自动注册、盲区外推器集成、燃料账单生成
    """

    def __init__(self, config: dict = None):
        """
        初始化仿真控制器

        Args:
            config: 仿真配置字典，包含以下可选字段：
                - mission_name: 任务名称
                - simulation_days: 仿真天数
                - time_step: 仿真步长 (秒)
                - orbit_type: 轨道类型 ("halo", "keplerian", "j2_keplerian")
                - orbit_params: 轨道生成器所需参数
                - enable_srp: 是否启用太阳光压
                - srp_area_to_mass: 光压面积质量比 (m²/kg)
                - srp_reflectivity: 光压反射系数
                - enable_j2: 是否启用 J2 摄动（仅对 LEO 场景）
                - enable_drag: 是否启用大气阻力
                - propagator_type: 外推器类型 ("simple", "kepler", None)
                - propagator_mu: 外推器引力常数 (仅 kepler)
                - log_buffer_size: 日志缓冲区大小
                - log_compression: 是否压缩日志
                - enable_visualization: 是否生成可视化
                - data_dir: 数据输出目录
        """
        # 默认配置
        self.default_config = {
            "mission_name": "JWST 30-Day L2 Station Keeping",
            "simulation_days": 30,
            "time_step": 10.0,
            "orbit_type": "halo",
            "orbit_params": {
                "Az": 0.05,
                "dt": 0.001
            },
            "enable_srp": False,
            "srp_area_to_mass": 0.01,
            "srp_reflectivity": 1.0,
            "enable_j2": False,
            "enable_drag": False,
            "propagator_type": None,
            "propagator_mu": 3.986004418e14,
            "log_buffer_size": 500,
            "log_compression": True,
            "progress_interval": 0.05,
            "enable_visualization": True,
            "data_dir": "data",
            "log_level": "INFO"
        }

        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        # 确保 orbit_params 存在
        if "orbit_params" not in self.config:
            self.config["orbit_params"] = {}

        # 初始化状态变量
        self.simulation_start_time = None
        self.simulation_end_time = None
        self.current_step = 0
        self.total_steps = 0

        # 核心组件
        self.environment = None
        self.spacecraft = None
        self.ground_station = None
        self.gnc_system = None
        self.ephemeris = None
        self.logger = None
        self.k_matrix = None

        # 创建输出目录
        os.makedirs(self.config["data_dir"], exist_ok=True)

        # 生成唯一的任务ID
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.h5_file = os.path.join(
            self.config["data_dir"],
            f"L1_{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}.h5"
        )

        print("=" * 80)
        print(f"🚀 MCPC 框架 L1 级: {self.config['mission_name']}")
        print("=" * 80)
        print(f"[配置] 任务ID: {self.mission_id}")
        print(f"[配置] 轨道类型: {self.config['orbit_type']}")
        print(f"[配置] 仿真时长: {self.config['simulation_days']} 天")
        print(f"[配置] 步长: {self.config['time_step']} 秒")
        print(f"[配置] 输出文件: {self.h5_file}")

    def run(self) -> bool:
        """
        执行完整仿真流程

        Returns:
            仿真是否成功
        """
        try:
            self.simulation_start_time = time.time()

            # 1. 生成标称轨道
            if not self._generate_nominal_orbit():
                print("❌ 标称轨道生成失败，使用备用轨道")
                self._generate_fallback_orbit()

            # 2. 初始化物理域
            self._initialize_physical_domain()

            # 3. 初始化信息域
            self._initialize_information_domain()

            # 4. 设计控制律
            self._design_control_law()

            # 5. 初始化数据记录
            self._initialize_data_logging()

            # 6. 执行仿真主循环
            self._execute_simulation_loop()

            # 7. 最终处理
            success = self._finalize_simulation()

            return success

        except KeyboardInterrupt:
            print("\n⏹️ 仿真被用户中断")
            self._emergency_shutdown()
            return False

        except Exception as e:
            print(f"\n❌ 仿真运行失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self._emergency_shutdown()
            return False

    def _generate_nominal_orbit(self) -> bool:
        """
        生成标称轨道（多态工厂调用）

        Returns:
            是否成功生成
        """
        print("\n" + "-" * 60)
        print("[阶段1] 生成标称轨道")
        print("-" * 60)

        try:
            orbit_type = self.config["orbit_type"]
            orbit_params = self.config["orbit_params"].copy()

            # 调用工厂函数
            generator = create_generator(orbit_type, **orbit_params)

            # 生成轨道
            self.ephemeris = generator.generate(orbit_params)

            # 验证轨道质量
            if self._validate_orbit_quality(self.ephemeris):
                print(f"✅ 标称轨道生成成功")
                print(f"   周期: {self.ephemeris.times[-1] / 86400:.2f} 天")
                print(f"   点数: {len(self.ephemeris.times)}")
                return True
            else:
                print("⚠️ 轨道质量验证失败，但继续使用")
                return True

        except Exception as e:
            print(f"❌ 轨道生成失败: {e}")
            return False

    def _generate_fallback_orbit(self) -> None:
        """
        生成备用轨道（当主方法失败时，使用经验初值积分）
        """
        print("   使用备用轨道生成方案...")

        # 使用已知的、经验证的初始状态（Halo 轨道）
        state0_nd = np.array([
            1.01106,    # x: L2点附近
            0.0,        # y: 从XZ平面出发
            0.05,       # z: 目标振幅
            0.0,        # vx: 初始x速度为0
            0.0105,     # vy: 精心调整的切向速度
            0.0         # vz: 初始z速度为0
        ])

        # 积分轨道
        def crtbp_eom(t, state):
            x, y, z, vx, vy, vz = state
            mu = 3.00348e-6

            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

            ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
            ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
            az = -(1-mu)*z/r1**3 - mu*z/r2**3

            return np.array([vx, vy, vz, ax, ay, az])

        # 无量纲周期 (约 π)
        T_nd = 3.141
        dt_nd = self.config["orbit_params"].get("dt", 0.001)
        times_nd = np.arange(0, T_nd, dt_nd)

        sol = solve_ivp(
            fun=crtbp_eom,
            t_span=(0, T_nd),
            y0=state0_nd,
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

        self.ephemeris = Ephemeris(
            times=physical_times,
            states=physical_states,
            frame=CoordinateFrame.SUN_EARTH_ROTATING
        )

        print(f"✅ 备用轨道生成完成，周期: {T_nd / OMEGA / 86400:.2f} 天")

    def _validate_orbit_quality(self, ephemeris: Ephemeris) -> bool:
        """验证轨道质量"""
        pos_start = ephemeris.states[0, 0:3]
        pos_end = ephemeris.states[-1, 0:3]
        vel_start = ephemeris.states[0, 3:6]
        vel_end = ephemeris.states[-1, 3:6]

        pos_error = np.linalg.norm(pos_end - pos_start)
        vel_error = np.linalg.norm(vel_end - vel_start)

        print(f"   轨道闭合性检查:")
        print(f"     位置误差: {pos_error:.2e} m")
        print(f"     速度误差: {vel_error:.2e} m/s")

        # 检查时间序列单调性
        time_diff = np.diff(ephemeris.times)
        if np.any(time_diff <= 0):
            print("⚠️ 时间序列非单调递增")
            return False

        # 检查数据有效性
        if np.any(np.isnan(ephemeris.states)):
            print("❌ 轨道数据包含NaN值")
            return False

        # 对于 Halo 轨道，位置误差小于 1e8 m 即可接受
        return pos_error < 1e8

    def _initialize_physical_domain(self) -> None:
        """
        初始化物理域（环境、航天器），根据轨道类型自动注册力学模型
        """
        print("\n" + "-" * 60)
        print("[阶段2] 初始化物理域")
        print("-" * 60)

        # 1. 天体环境
        frame = self.ephemeris.frame
        self.environment = CelestialEnvironment(
            computation_frame=frame,
            initial_epoch=0.0
        )

        # 2. 根据轨道类型注册力学模型
        orbit_type = self.config["orbit_type"]

        # 注册中心引力模型（根据轨道类型不同，使用不同的力模型）
        if orbit_type == "halo":
            # Halo 轨道使用 CRTBP（日地旋转系）
            self.environment.register_force(Gravity_CRTBP())
            print("[物理域] 已注册 CRTBP 引力模型")
        elif orbit_type in ("keplerian", "j2_keplerian"):
            # 对于开普勒或 J2 轨道，注册 J2 摄动模型（如果启用）
            if self.config.get("enable_j2", False):
                j2_model = J2Gravity()
                self.environment.register_force(j2_model)
                print("[物理域] 已注册 J2 摄动模型")
            # 中心引力由 J2KeplerianGenerator 的动力学方程隐含，或在仿真中通过 IForceModel 注册中心引力。
            # 注意：当前 CelestialEnvironment 只通过 IForceModel 计算加速度，不包含中心引力。
            # 为此，我们可能需要一个简单的中心引力模型（CentralGravity），但为简化，暂不注册。
            # 在 L1 级，对于开普勒轨道，我们依赖星历生成器提供的参考轨迹，实际动力学积分仍需中心引力。
            # 这里需要确保 get_total_acceleration 返回正确的加速度。由于无额外模型，加速度为0，将导致错误。
            # 因此，必须添加中心引力模型。我们将创建一个简单的 CentralGravity 类。
            # 为保持代码简洁，这里直接添加一个临时中心引力模型。
            class CentralGravity:
                def __init__(self, mu):
                    self.mu = mu
                def compute_accel(self, state, epoch):
                    r = state[:3]
                    r_norm = np.linalg.norm(r)
                    if r_norm < 1e-10:
                        return np.zeros(3)
                    return -self.mu * r / r_norm**3
            mu = 3.986004418e14  # 地球引力常数，可根据实际轨道类型调整
            self.environment.register_force(CentralGravity(mu))
            print("[物理域] 已注册中心引力模型")

        # 可选力学模型
        if self.config.get("enable_srp", False):
            srp = Cannonball_SRP(
                area_to_mass=self.config["srp_area_to_mass"],
                reflectivity=self.config["srp_reflectivity"]
            )
            self.environment.register_force(srp)
            print("[物理域] 已注册太阳光压模型")

        if self.config.get("enable_drag", False):
            # 大气阻力模型尚未实现，留作后续
            print("[物理域] 大气阻力模型未实现，跳过")

        # 3. 航天器初始化
        nominal_state_0 = self.ephemeris.get_interpolated_state(0.0)

        # 初始注入误差：位置 ±2km，速度 ±1cm/s
        injection_error = np.array([
            2000.0, 2000.0, -1000.0,  # 位置误差 (m)
            0.01, -0.01, 0.005        # 速度误差 (m/s)
        ])

        true_initial_state = nominal_state_0 + injection_error
        sc_mass = 6200.0  # JWST 发射质量 (kg)

        self.spacecraft = SpacecraftPointMass(
            sc_id="JWST_Shadow",
            initial_state=true_initial_state,
            frame=frame,
            initial_mass=sc_mass
        )

        print("✅ 物理域初始化完成")
        print(f"   航天器质量: {sc_mass} kg")
        print(f"   初始位置偏差: {np.linalg.norm(injection_error[0:3]):.1f} m")
        print(f"   初始速度偏差: {np.linalg.norm(injection_error[3:6]) * 1000:.3f} mm/s")

    def _initialize_information_domain(self) -> None:
        """
        初始化信息域（测控、GNC），可选注入外推器
        """
        print("\n" + "-" * 60)
        print("[阶段3] 初始化信息域")
        print("-" * 60)

        frame = self.ephemeris.frame

        # 1. 地面测控站
        self.ground_station = GroundStation(
            name="DSN_Network",
            operating_frame=frame,
            pos_noise_std=5.0,
            vel_noise_std=0.005,
            sampling_rate_hz=0.1
        )

        # 2. GNC 子系统
        self.gnc_system = GNC_Subsystem(
            sc_id="JWST_Shadow",
            operating_frame=frame
        )
        self.gnc_system.load_reference_trajectory(self.ephemeris)

        # 3. 注入外推器
        propagator_type = self.config.get("propagator_type")
        if propagator_type == "simple":
            propagator = SimplePropagator()
            self.gnc_system.set_propagator(propagator)
            print("✅ 已启用简单线性外推器")
        elif propagator_type == "kepler":
            mu = self.config.get("propagator_mu", 3.986004418e14)
            propagator = KeplerPropagator(mu)
            self.gnc_system.set_propagator(propagator)
            print(f"✅ 已启用二体外推器 (mu={mu:.4e})")
        else:
            print("ℹ️ 未启用盲区外推器")

        print("✅ 信息域初始化完成")
        print(f"   测控噪声: 位置 {self.ground_station.pos_noise_std} m, "
              f"速度 {self.ground_station.vel_noise_std * 1000:.1f} mm/s")

    def _design_control_law(self) -> None:
        """
        设计 LQR 最优控制律（针对日地 L2 点线性化模型）
        """
        print("\n" + "-" * 60)
        print("[阶段4] 设计最优控制律 (LQR)")
        print("-" * 60)

        # 日地 L2 点动力学线性化常数
        mu = 3.00348e-6
        gamma_l = np.cbrt(mu / 3.0)
        omega = 1.990986e-7

        print(f"   计算参数: μ={mu:.6e}, γ={gamma_l:.6e}, ω={omega:.6e} rad/s")

        # 状态矩阵 A (6x6)
        a_mat = np.zeros((6, 6))
        a_mat[0:3, 3:6] = np.eye(3)
        a_mat[3, 0] = (2 * gamma_l + 1) * omega**2
        a_mat[4, 1] = (1 - gamma_l) * omega**2
        a_mat[5, 2] = -gamma_l * omega**2
        a_mat[3, 4] = 2 * omega
        a_mat[4, 3] = -2 * omega

        # 控制矩阵 B (6x3)
        b_mat = np.zeros((6, 3))
        b_mat[3:6, 0:3] = np.eye(3) / self.spacecraft.mass

        # 权重矩阵
        q_mat = np.diag([1.0, 1.0, 1.0, 1e6, 1e6, 1e6])
        r_mat = np.diag([10.0, 10.0, 10.0])

        try:
            self.k_matrix = get_lqr_gain(a_mat, b_mat, q_mat, r_mat)

            # 验证K矩阵形状
            if self.k_matrix.shape != (3, 6):
                print(f"⚠️ 警告: K矩阵形状异常: {self.k_matrix.shape}，期望(3,6)")
                if self.k_matrix.shape == (1, 6) or self.k_matrix.shape == (6,):
                    self.k_matrix = np.tile(self.k_matrix, (3, 1))
                elif self.k_matrix.shape[0] > 3:
                    self.k_matrix = self.k_matrix[:3, :]

        except Exception as e:
            print(f"❌ LQR增益计算失败: {e}")
            self.k_matrix = np.eye(3, 6) * 1e-3
            print("    使用备用增益矩阵")

        print("✅ LQR 控制律设计完成")
        print(f"   增益矩阵形状: {self.k_matrix.shape}")
        print(f"   增益矩阵范数: ||K|| = {np.linalg.norm(self.k_matrix):.2e}")

    def _initialize_data_logging(self) -> None:
        """
        初始化数据记录系统
        """
        print("\n" + "-" * 60)
        print("[阶段5] 初始化数据记录系统")
        print("-" * 60)

        metadata = SimulationMetadata.create_mission_metadata(
            mission_name=self.config["mission_name"],
            config={
                "simulation_days": self.config["simulation_days"],
                "time_step": self.config["time_step"],
                "spacecraft_mass": self.spacecraft.mass,
                "control_type": "LQR",
                "mission_id": self.mission_id,
                "orbit_type": self.config["orbit_type"],
                "ephemeris_period_days": self.ephemeris.times[-1] / 86400
            }
        )

        self.logger = HDF5Logger(
            filepath=self.h5_file,
            buffer_size=self.config["log_buffer_size"],
            compression=self.config["log_compression"],
            auto_flush=True
        )
        self.logger.initialize_file(metadata)

        print("✅ 数据记录系统初始化完成")
        print(f"   数据文件: {self.h5_file}")
        print(f"   缓冲区大小: {self.config['log_buffer_size']} 条")

    def _execute_simulation_loop(self) -> None:
        """
        执行仿真主循环
        """
        print("\n" + "-" * 60)
        print("[阶段6] 执行仿真主循环")
        print("-" * 60)

        dt = self.config["time_step"]
        sim_seconds = self.config["simulation_days"] * 86400
        self.total_steps = int(sim_seconds / dt)

        progress_steps = max(1, int(self.total_steps * self.config["progress_interval"]))

        print(f"   开始 {self.config['simulation_days']} 天闭环仿真...")
        print(f"   仿真步数: {self.total_steps:,}")
        print(f"   仿真步长: {dt} 秒")
        print(f"   进度报告间隔: 每 {progress_steps} 步")
        print("-" * 60)

        for step in range(self.total_steps):
            self.current_step = step
            epoch = step * dt

            # --- 导航感知 ---
            obs_state, frame = self.ground_station.track_spacecraft(
                self.spacecraft.state,
                self.spacecraft.frame,
                epoch
            )
            self.gnc_system.update_navigation(obs_state, frame, dt)

            # --- 控制决策 ---
            force_cmd, force_frame = self.gnc_system.compute_control_force(epoch, self.k_matrix)
            force_cmd = self._ensure_3d_control_force(force_cmd)

            # --- 物理演化 ---
            self.spacecraft.apply_thrust(force_cmd, force_frame)

            # RK4 积分
            k1 = self._get_state_derivative(self.spacecraft.state)
            k2 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k1)
            k3 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k2)
            k4 = self._get_state_derivative(self.spacecraft.state + dt * k3)
            self.spacecraft.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            self.spacecraft.integrate_dv(dt)
            self.spacecraft.clear_thrust()
            self.environment.step_time(dt)

            # --- 数据记录 ---
            if step % 10 == 0:
                nom_state = self.ephemeris.get_interpolated_state(epoch)
                self.logger.log_step(
                    epoch=epoch,
                    nominal_state=nom_state,
                    true_state=self.spacecraft.state,
                    nav_state=self.gnc_system.current_nav_state,
                    tracking_error=self.gnc_system.last_tracking_error,
                    control_force=force_cmd,
                    accumulated_dv=self.spacecraft.accumulated_dv
                )

            # --- 进度报告 ---
            if step % progress_steps == 0 and step > 0:
                self._report_progress(step, epoch, force_cmd)

        print("-" * 60)
        print("✅ 仿真主循环完成")

    def _ensure_3d_control_force(self, force_cmd) -> np.ndarray:
        """确保控制力是3维数组"""
        if isinstance(force_cmd, (int, float, np.number)):
            return np.array([float(force_cmd), 0.0, 0.0], dtype=np.float64)
        elif isinstance(force_cmd, np.ndarray):
            if force_cmd.shape == (1,) or force_cmd.shape == ():
                return np.array([float(force_cmd[0]) if force_cmd.shape == (1,) else float(force_cmd), 0.0, 0.0], dtype=np.float64)
            elif force_cmd.shape == (3,):
                return force_cmd.astype(np.float64)
            elif force_cmd.size > 3:
                return force_cmd[:3].astype(np.float64)
            else:
                try:
                    return force_cmd.reshape(3).astype(np.float64)
                except:
                    return np.zeros(3, dtype=np.float64)
        else:
            return np.zeros(3, dtype=np.float64)

    def _get_state_derivative(self, state: np.ndarray) -> np.ndarray:
        """计算状态导数（供RK4积分使用）"""
        acc_env, acc_frame = self.environment.get_total_acceleration(state, self.spacecraft.frame)
        derivative = np.zeros(6)
        derivative[0:3] = state[3:6]
        derivative[3:6] = acc_env + self.spacecraft.external_accel
        return derivative

    def _report_progress(self, step: int, epoch: float, force_cmd: np.ndarray) -> None:
        """报告仿真进度"""
        progress = (step / self.total_steps) * 100
        days = epoch / 86400

        err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
        err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
        control_norm = np.linalg.norm(force_cmd)

        print(f"  [Day {days:6.1f}] 进度: {progress:5.1f}% | "
              f"位置误差: {err_pos:6.1f}m | "
              f"速度误差: {err_vel:6.2f}mm/s | "
              f"控制力: {control_norm:7.4f}N | "
              f"累计 ΔV: {self.spacecraft.accumulated_dv:8.4f}m/s")

    def _finalize_simulation(self) -> bool:
        """
        仿真最终处理，包括燃料账单生成和可视化
        """
        print("\n" + "=" * 60)
        print("[阶段7] 仿真最终处理")
        print("=" * 60)

        self.simulation_end_time = time.time()
        sim_duration = self.simulation_end_time - self.simulation_start_time

        if self.logger:
            self.logger.close()

        final_err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
        final_err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000

        # 生成燃料账单文件 (T7)
        fuel_bill_path = os.path.join(
            self.config["data_dir"],
            f"fuel_bill_{self.mission_id}.csv"
        )
        try:
            with open(fuel_bill_path, 'w') as f:
                f.write("mission_id,total_dv_mps,avg_dv_per_day_mps,simulation_days,final_position_error_m,final_velocity_error_mms\n")
                f.write(f"{self.mission_id},{self.spacecraft.accumulated_dv:.6f},"
                        f"{self.spacecraft.accumulated_dv / self.config['simulation_days']:.6f},"
                        f"{self.config['simulation_days']},{final_err_pos:.2f},{final_err_vel:.2f}\n")
            print(f"✅ 燃料账单已保存: {fuel_bill_path}")
        except Exception as e:
            print(f"⚠️ 燃料账单保存失败: {e}")

        # 仿真结果汇总
        print("📊 仿真结果汇总")
        print("-" * 60)
        print(f"✅ 仿真完成!")
        print(f"   实际仿真时间: {sim_duration:.1f} 秒")
        print(f"   仿真步数: {self.total_steps:,}")
        print(f"   最终位置误差: {final_err_pos:.2f} m")
        print(f"   最终速度误差: {final_err_vel:.2f} mm/s")
        print(f"   总 ΔV 消耗: {self.spacecraft.accumulated_dv:.4f} m/s")
        print(f"   平均每天 ΔV: {self.spacecraft.accumulated_dv / self.config['simulation_days']:.4f} m/s/天")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")

        if self.logger:
            stats = self.logger.get_statistics()
            if "file_size_mb" in stats:
                print(f"   数据文件大小: {stats['file_size_mb']:.2f} MB")

        if self.config["enable_visualization"]:
            self._generate_visualization()

        print("\n" + "=" * 60)
        print(f"🎉 {self.config['mission_name']} 仿真圆满完成！")
        print("=" * 60)

        return True

    def _generate_visualization(self) -> None:
        """生成可视化报告"""
        print("\n" + "-" * 60)
        print("[阶段8] 生成可视化报告")
        print("-" * 60)

        try:
            if not os.path.exists(self.h5_file):
                print("⚠️ 数据文件不存在，跳过可视化")
                return

            vis = L1Visualizer(
                filepath=self.h5_file,
                mission_name=self.config["mission_name"]
            )

            base_name = os.path.join(
                self.config["data_dir"],
                f"L1_{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}"
            )

            vis.plot_3d_trajectory(save_path=f"{base_name}_trajectory.png")
            vis.plot_tracking_error(save_path=f"{base_name}_errors.png")
            vis.plot_control_effort(save_path=f"{base_name}_control.png")

            print("✅ 可视化报告生成完成")
            print(f"   轨迹图: {base_name}_trajectory.png")
            print(f"   误差图: {base_name}_errors.png")
            print(f"   控制图: {base_name}_control.png")

        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")

    def _emergency_shutdown(self) -> None:
        """紧急关闭处理"""
        print("\n⚠️ 执行紧急关闭...")
        if self.logger:
            try:
                self.logger.close()
                print(f"✅ 已保存数据到: {self.h5_file}")
            except Exception as e:
                print(f"❌ 数据保存失败: {e}")

        if self.current_step > 0:
            completed_days = (self.current_step * self.config["time_step"]) / 86400
            print(f"   已完成 {completed_days:.2f} 天仿真")

    def get_statistics(self) -> dict:
        """获取仿真统计信息"""
        stats = {
            "mission_id": self.mission_id,
            "mission_name": self.config["mission_name"],
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "completed": self.current_step >= self.total_steps,
            "data_file": self.h5_file
        }

        if self.logger:
            stats.update(self.logger.get_statistics())

        if self.simulation_start_time and self.simulation_end_time:
            stats["simulation_duration_seconds"] = self.simulation_end_time - self.simulation_start_time

        if self.spacecraft:
            stats["accumulated_dv"] = self.spacecraft.accumulated_dv

        if self.gnc_system:
            stats["final_position_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
            stats["final_velocity_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[3:6])

        return stats


def run_L1_simulation(custom_config: dict = None) -> bool:
    """
    运行 L1 级仿真的入口函数

    Args:
        custom_config: 自定义配置

    Returns:
        仿真是否成功
    """
    simulation = L1MissionSimulation(custom_config)
    success = simulation.run()
    if success:
        stats = simulation.get_statistics()
        print(f"\n📈 仿真统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    return success


if __name__ == "__main__":
    # 示例配置
    custom_config = {
        "mission_name": "JWST 30-Day L2 站位维持仿真 (多轨道支持)",
        "simulation_days": 30,
        "time_step": 10.0,
        "orbit_type": "halo",                # 可选: halo, keplerian, j2_keplerian
        "orbit_params": {
            "Az": 0.05,
            "dt": 0.001
        },
        "enable_srp": False,                # 启用光压需根据实际需要
        "propagator_type": None,             # None, "simple", "kepler"
        "log_buffer_size": 1000,
        "enable_visualization": True
    }
    success = run_L1_simulation(custom_config)
    sys.exit(0 if success else 1)
