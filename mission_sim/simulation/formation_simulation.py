# mission_sim/simulation/formation_simulation.py
"""
MCPC L2 编队仿真引擎（五步时序接通版）
======================================
按 DD-001 v1.1（R-L2-01~08）与 REF-02 §3.1 五步时序重构主循环：
    ① [物理] RK4 积分（环境力在各子步评估；控制力为上一步 ④ 下发）
    ② [物理] 主星 ISL 天线生成测量（LVLH 相对位置，含硬件噪声）
    ③ [赛博] 路由器打包（延迟/抖动/丢包）
    ④ [赛博] 各从星 GNC（三段式状态机 + STM 时延补偿）→ 执行器（含冲量平滑）
    ⑤ [时钟] 质量更新与时间推进

v2 修订（WP-2.1~2.4）：
- 接通 FormationController.update()（原实现直接使用 K 矩阵，状态机/时延补偿从未生效）；
- 控制误差与目标统一为 LVLH 旋转系口径（compute_lvlh_dcm / absolute_to_lvlh）；
- 环境力注册（中心引力 + J2 或 CRTBP），与 L1 同源动力学；积分器统一 RK4；
- 消费 ISL/推力器/路由器配置；支持 reconfiguration_events 外部重构指令；
- HDF5 formation 组与既有分析工具保持兼容。
"""

import os
import h5py
import numpy as np

from .base import BaseSimulation
from mission_sim.core.physics.spacecraft_node import SpacecraftNode
from mission_sim.core.physics.components.actuators.thruster import Thruster
from mission_sim.core.physics.components.actuators.impulse_smoother import ImpulseSmoother
from mission_sim.core.physics.components.sensors.isl_antenna import ISLAntenna
from mission_sim.core.cyber.network.isl_router import ISLRouter
from mission_sim.core.cyber.platform_gnc.formation_controller import FormationController
from mission_sim.core.cyber.models.cw_dynamics import CWDynamics
from mission_sim.core.cyber.models.crtbp_relative_dynamics import CRTBPRelativeDynamics
from mission_sim.core.cyber.models.threebody.base import CRTBP
from mission_sim.core.physics.ids import SpacecraftType
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.physics.models.central_gravity import CentralGravity
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.utils.math_tools import absolute_to_lvlh, compute_lvlh_dcm
from mission_sim.core.cyber.ids import FormationMode


class FormationSimulation(BaseSimulation):
    """
    多星编队仿真引擎 (L2 级，五步时序接通版)
    """

    def __init__(self, config):
        super().__init__(config)
        self.deputies = []
        self.deputy_controllers = {}
        self.chief = None
        self.crtbp_model = None
        self.formation_targets = dict(config.get("formation_targets", {}) or {})

        # 帧统计（T-L2-005 链路守恒核对）
        self._frames_sent = 0
        self._frames_dropped = 0
        self._frames_delivered = 0

        # 多星模式下屏蔽单星专用变量
        self.spacecraft = None
        self.gnc_system = None

    def _generate_nominal_orbit(self) -> bool:
        """编队模式下不需要标称星历，直接返回 True"""
        self.ephemeris = None
        return True

    # ==================================================================
    # 硬件工厂（消费配置，DD-001 D7）
    # ==================================================================
    def _make_thruster(self) -> Thruster:
        return Thruster(
            max_thrust_n=self.config.get("thruster_max_thrust_n", 1.0),
            min_thrust_n=self.config.get("thruster_min_thrust_n", 0.0),
            noise_std_n=self.config.get("thruster_noise_std_n", 0.005),
            specific_impulse_s=self.config.get("thruster_isp_s", 3000.0),
            random_seed=self.config.get("thruster_seed"),
        )

    def _make_antenna(self) -> ISLAntenna:
        return ISLAntenna(
            range_noise_std=self.config.get("isl_range_noise_std", 0.05),
            angle_noise_std=self.config.get("isl_angle_noise_std", 5e-5),
            max_range_m=self.config.get("isl_max_range_m", 200000.0),
            reference_range_m=self.config.get("isl_ref_range_m", 1000.0),
            random_seed=self.config.get("isl_seed"),
        )

    def _make_router(self) -> ISLRouter:
        return ISLRouter(
            base_latency_s=self.config.get("router_base_latency_s", 0.05),
            jitter_s=self.config.get("router_jitter_s", 0.01),
            packet_loss_rate=self.config.get("router_packet_loss_rate", 0.02),
            random_seed=self.config.get("router_seed"),
        )

    # ==================================================================
    # 物理域
    # ==================================================================
    def _initialize_physical_domain(self):
        """初始化物理域：主星（天线+路由器）、从星（推力器+可选冲量平滑器）、环境力注册。"""
        chief_frame_str = self.config.get("chief_frame", "J2000_ECI")
        if isinstance(chief_frame_str, str):
            frame_map = {
                "J2000_ECI": CoordinateFrame.J2000_ECI,
                "SUN_EARTH_ROTATING": CoordinateFrame.SUN_EARTH_ROTATING,
                "LVLH": CoordinateFrame.LVLH,
            }
            chief_frame = frame_map.get(chief_frame_str, CoordinateFrame.J2000_ECI)
        else:
            chief_frame = chief_frame_str

        # 主星：测量发射端（天线 + 路由器）
        self.chief = SpacecraftNode(
            sc_id="CHIEF",
            initial_state=np.array(self.config["chief_initial_state"], dtype=np.float64),
            frame=chief_frame,
            initial_mass=self.config.get("chief_mass_kg", 2000.0),
            sc_type=SpacecraftType.CHIEF,
            antenna=self._make_antenna(),
            router=self._make_router(),
        )

        # 从星：控制执行端（推力器 + 可选冲量平滑器）
        smoother_mib = self.config.get("impulse_smoother_mib_ns")
        for dep_id, initial_state in self.config["deputy_initial_states"]:
            smoother = None
            if smoother_mib:
                smoother = ImpulseSmoother(
                    mib_ns=float(smoother_mib),
                    max_thrust_n=self.config.get("thruster_max_thrust_n", 1.0),
                )
            dep_node = SpacecraftNode(
                sc_id=dep_id,
                initial_state=np.array(initial_state, dtype=np.float64),
                frame=chief_frame,
                initial_mass=self.config.get("deputy_mass_kg", 500.0),
                sc_type=SpacecraftType.DEPUTY,
                thruster=self._make_thruster(),
                impulse_smoother=smoother,
            )
            self.deputies.append(dep_node)

        # 环境力注册（R-L2-03：注册表非空，与 L1 同源动力学）
        self.environment = CelestialEnvironment(
            computation_frame=chief_frame,
            initial_epoch=0.0,
            verbose=self.verbose,
        )
        if self.config.get("enable_crtbp", False):
            from mission_sim.core.physics.models.gravity import GravityCRTBP
            self.environment.register_force(GravityCRTBP())
        else:
            mu = self.config.get("mu_earth", 3.986004418e14)
            self.environment.register_force(CentralGravity(mu=mu))
            if self.config.get("enable_j2", True):
                self.environment.register_force(J2Gravity(mu_earth=mu))

    # ==================================================================
    # 信息域
    # ==================================================================
    def _initialize_information_domain(self):
        """初始化信息域：相对动力学、控制器（含目标与初始估计注入）。"""
        enable_crtbp = self.config.get("enable_crtbp", False)

        if enable_crtbp:
            mu = self.config.get("mu", 3.00348e-6)
            L = self.config.get("L", 1.495978707e11)
            omega = self.config.get("omega", 1.990986e-7)
            self.crtbp_model = CRTBP(mu, L, omega)

            def chief_traj_callable(t):
                return self.chief.state

            dynamics = CRTBPRelativeDynamics(self.crtbp_model, chief_trajectory=chief_traj_callable)
        else:
            orbit_n = self.config.get("orbit_angular_rate")
            if not orbit_n:
                h = np.cross(self.chief.position, self.chief.velocity)
                orbit_n = float(np.linalg.norm(h) / np.dot(self.chief.position, self.chief.position))
            dynamics = CWDynamics(float(orbit_n))

        K_lqr = self.config.get("lqr_gain")

        for dep in self.deputies:
            controller = FormationController(
                deputy_id=dep.id,
                chief_id=self.chief.id,
                dynamics=dynamics,
                K_lqr=K_lqr,
                generation_threshold_pos=self.config.get("generation_threshold_pos", 100.0),
                generation_threshold_vel=self.config.get("generation_threshold_vel", 0.5),
                keeping_threshold_pos=self.config.get("keeping_threshold_pos", 1.0),
                keeping_threshold_vel=self.config.get("keeping_threshold_vel", 0.01),
            )
            # 初始估计状态：用初始真实的 LVLH 相对状态预热（避免巨大暂态）
            rho, rho_dot = absolute_to_lvlh(
                self.chief.position, self.chief.velocity, dep.position, dep.velocity)
            controller.last_estimated_state = np.concatenate([rho, rho_dot])
            target = self.formation_targets.get(dep.id)
            if target is not None:
                controller.target_state = np.asarray(target, dtype=np.float64)
            self.deputy_controllers[dep.id] = controller

    def _design_control_law(self):
        """此方法在基类 run 中被调用，但编队模式无需额外操作"""
        pass

    # ==================================================================
    # 数据日志（与既有分析工具兼容）
    # ==================================================================
    def _initialize_data_logging(self):
        super()._initialize_data_logging()

        with h5py.File(self.h5_file, 'a') as f:
            formation_grp = f.require_group("formation")
            for dep in self.deputies:
                dep_grp = formation_grp.require_group(f"deputy_{dep.id}")
                for dset in ["time", "rel_position", "rel_velocity", "control_force", "mode"]:
                    if dset in dep_grp:
                        continue
                    if dset == "time":
                        dep_grp.create_dataset(dset, shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True, compression="gzip")
                    elif dset == "mode":
                        dep_grp.create_dataset(dset, shape=(0,), maxshape=(None,), dtype=np.int8, chunks=True, compression="gzip")
                    else:
                        dep_grp.create_dataset(dset, shape=(0, 3), maxshape=(None, 3), dtype=np.float64, chunks=True, compression="gzip")

            # 保存每个从星的初始期望相对状态（如配置中存在；运行中重构不回溯修改）
            if self.formation_targets:
                targets_grp = f.require_group("metadata/targets")
                for dep_id, target_state in self.formation_targets.items():
                    target_array = np.array(target_state, dtype=np.float64)
                    dset_name = f"deputy_{dep_id}"
                    if dset_name in targets_grp:
                        del targets_grp[dset_name]
                    targets_grp.create_dataset(dset_name, data=target_array)

    # ==================================================================
    # 主循环（五步时序）
    # ==================================================================
    def _rk4_step(self, node: SpacecraftNode, dt: float) -> None:
        """单节点 RK4 步进：环境力在各子步评估，控制力（external_accel）保持常数。"""
        def deriv(state):
            acc_env, _ = self.environment.get_total_acceleration(state, node.frame)
            return np.concatenate([state[3:6], acc_env + node.external_accel])

        y = node.state
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        node.state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _execute_simulation_loop(self):
        """核心仿真主循环（REF-02 §3.1 五步时序）。"""
        dt = float(self.config["time_step"])
        total_steps = int(self.config["simulation_days"] * 86400 / dt)
        self.total_steps = total_steps
        all_nodes = [self.chief] + self.deputies
        pending_frames = []
        log_interval = int(self.config.get("log_interval", 10))
        progress_steps = max(1, int(total_steps * self.config.get("progress_interval", 0.05)))

        events = sorted(self.config.get("reconfiguration_events", []) or [], key=lambda e: e[0])
        ev_idx = 0

        for step in range(total_steps):
            epoch = step * dt
            t_now = epoch + dt

            # ① [物理] 积分（环境力子步评估；控制力为上一步 ④ 下发）
            for node in all_nodes:
                self._rk4_step(node, dt)
                node.integrate_dv(dt)
                node.clear_thrust()

            # ② [物理] 主星 ISL 天线生成测量（LVLH 相对位置；含硬件噪声与量程限制）
            for dep in self.deputies:
                rho, _ = absolute_to_lvlh(
                    self.chief.position, self.chief.velocity, dep.position, dep.velocity)
                meas = self.chief.antenna.measure(rho, t_now)
                if meas is None:
                    continue
                # ③ [赛博] 路由器打包（延迟/抖动/丢包）
                self._frames_sent += 1
                frame = self.chief.router.transmit(meas, self.chief.id, dep.id, t_now)
                if frame is None:
                    self._frames_dropped += 1
                else:
                    pending_frames.append(frame)

            delivered = [f for f in pending_frames if f.rx_time <= t_now]
            pending_frames = [f for f in pending_frames if f.rx_time > t_now]
            self._frames_delivered += len(delivered)

            # 外部重构指令（任务序列触发 KEEPING→RECONFIGURATION，RE-001 I-1）
            while ev_idx < len(events) and events[ev_idx][0] <= t_now:
                _, dep_id, new_target = events[ev_idx]
                ev_idx += 1
                tgt = np.asarray(new_target, dtype=np.float64)
                self.formation_targets[dep_id] = tgt
                self.deputy_controllers[dep_id].command_reconfiguration(tgt)

            # ④ [赛博] 各从星 GNC 解算（状态机 + 时延补偿）→ 执行器（含冲量平滑）
            C = compute_lvlh_dcm(self.chief.position, self.chief.velocity)
            for dep in self.deputies:
                controller = self.deputy_controllers[dep.id]
                cmd = controller.update(
                    t_now, delivered, dt,
                    target_state=self.formation_targets.get(dep.id),
                )
                # 控制器输出为 LVLH 加速度指令（B=I 口径）→ 推力 → 惯性系
                accel_lvlh = np.asarray(cmd.force_vector, dtype=np.float64)
                force_inertial = C.T @ (accel_lvlh * dep.mass)
                dep.apply_control(force_inertial, dt)

            # ⑤ [时钟] 质量更新与时间推进
            for node in all_nodes:
                node.update_mass(dt)
            self.environment.step_time(dt)

            # 日志与进度
            if step % log_interval == 0:
                self._log_formation_step(t_now)
            if self.verbose and (step % progress_steps == 0 or step == total_steps - 1):
                self._report_progress_custom(t_now, step, total_steps)

    def _log_formation_step(self, epoch):
        """记录编队数据到 HDF5（LVLH 相对状态 + 实际控制力 + 模式）。"""
        try:
            with h5py.File(self.h5_file, 'a') as f:
                formation_grp = f["formation"]
                for dep in self.deputies:
                    dep_grp = formation_grp[f"deputy_{dep.id}"]
                    rho, rho_dot = absolute_to_lvlh(
                        self.chief.position, self.chief.velocity,
                        dep.position, dep.velocity
                    )
                    control = getattr(dep, "last_control_force", np.zeros(3))
                    mode = self.deputy_controllers[dep.id].mode if dep.id in self.deputy_controllers else FormationMode.GENERATION
                    mode_int = {
                        FormationMode.GENERATION: 0,
                        FormationMode.KEEPING: 1,
                        FormationMode.RECONFIGURATION: 2,
                    }.get(mode, 0)

                    for dset, data in [
                        ("time", epoch),
                        ("rel_position", rho),
                        ("rel_velocity", rho_dot),
                        ("control_force", control),
                        ("mode", mode_int),
                    ]:
                        ds = dep_grp[dset]
                        ds.resize(ds.shape[0] + 1, axis=0)
                        ds[-1] = data
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 编队日志写入失败: {e}")

    def _report_progress_custom(self, epoch, step, total_steps):
        progress = (step / total_steps) * 100
        max_dist = max([np.linalg.norm(d.state[:3] - self.chief.state[:3]) for d in self.deputies])
        print(f"  [Day {epoch/86400:.1f}] 进度: {progress:4.1f}% | 最大相对距离: {max_dist:10.3f}m")

    def _print_summary(self):
        """编队任务汇总打印"""
        import time
        self.simulation_end_time = time.time()
        print("\n📊 编队仿真结果汇总")
        print("-" * 60)
        print(f"✅ 任务完成!")
        print(f"   仿真时长: {self.config['simulation_days']} 天")
        print(f"   ISL 帧统计: 发送 {self._frames_sent} / 丢弃 {self._frames_dropped} / 交付 {self._frames_delivered}")
        print(f"   主星 {self.chief.id} ΔV: {self.chief.accumulated_dv:.6f} m/s")
        for dep in self.deputies:
            ctrl = self.deputy_controllers.get(dep.id)
            mode = ctrl.mode.name if ctrl is not None else "N/A"
            print(f"   从星 {dep.id} ΔV: {dep.accumulated_dv:.6f} m/s | 末态模式: {mode}")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
