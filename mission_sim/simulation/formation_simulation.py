# mission_sim/simulation/formation_simulation.py
import numpy as np
import os
import h5py
from .base import BaseSimulation
from mission_sim.core.physics.spacecraft_node import SpacecraftNode
from mission_sim.core.cyber.platform_gnc.formation_controller import FormationController
from mission_sim.core.cyber.models.cw_dynamics import CWDynamics
from mission_sim.core.cyber.models.crtbp_relative_dynamics import CRTBPRelativeDynamics
from mission_sim.core.cyber.models.threebody.base import CRTBP
from mission_sim.core.physics.ids import SpacecraftType
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.utils.math_tools import absolute_to_lvlh
from mission_sim.core.cyber.ids import FormationMode


class FormationSimulation(BaseSimulation):
    """
    多星编队仿真引擎 (L2 级)
    """

    def __init__(self, config):
        super().__init__(config)
        self.deputies = []
        self.deputy_controllers = {}
        self.chief = None
        self.crtbp_model = None
        
        # 多星模式下屏蔽单星专用变量
        self.spacecraft = None 
        self.gnc_system = None

    def _generate_nominal_orbit(self) -> bool:
        """编队模式下不需要标称星历，直接返回 True"""
        self.ephemeris = None
        return True

    def _initialize_physical_domain(self):
        """初始化物理域：创建主星、从星节点以及物理环境"""
        # 1. 解析坐标系
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

        # 2. 创建主星
        self.chief = SpacecraftNode(
            sc_id="CHIEF",
            initial_state=np.array(self.config["chief_initial_state"], dtype=np.float64),
            frame=chief_frame,
            initial_mass=self.config.get("chief_mass_kg", 2000.0),
            sc_type=SpacecraftType.CHIEF
        )

        # 3. 创建从星
        for dep_id, initial_state in self.config["deputy_initial_states"]:
            dep_node = SpacecraftNode(
                sc_id=dep_id,
                initial_state=np.array(initial_state, dtype=np.float64),
                frame=chief_frame,
                initial_mass=self.config.get("deputy_mass_kg", 500.0),
                sc_type=SpacecraftType.DEPUTY
            )
            self.deputies.append(dep_node)

        # 4. 创建物理环境引擎
        self.environment = CelestialEnvironment(
            computation_frame=chief_frame,
            initial_epoch=0.0,
            verbose=self.verbose
        )

    def _initialize_information_domain(self):
        """初始化信息域：根据配置选择动力学模型并创建控制器"""
        enable_crtbp = self.config.get("enable_crtbp", False)
        
        if enable_crtbp:
            # 使用 CRTBP 相对动力学（需要用户提供 lqr_gain）
            mu = self.config.get("mu", 3.00348e-6)
            L = self.config.get("L", 1.495978707e11)
            omega = self.config.get("omega", 1.990986e-7)
            self.crtbp_model = CRTBP(mu, L, omega)
            def chief_traj_callable(t):
                return self.chief.state
            dynamics = CRTBPRelativeDynamics(self.crtbp_model, chief_trajectory=chief_traj_callable)
        else:
            # 使用 CW 动力学（圆轨道，可自动计算 LQR 增益）
            orbit_n = self.config.get("orbit_angular_rate", 0.001)
            dynamics = CWDynamics(n=orbit_n)

        # 获取 LQR 增益（可能为 None，CW 模型会自行计算）
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
            # 注入初始估计状态（基于绝对状态差）
            init_rel_pos = dep.state[:3] - self.chief.state[:3]
            init_rel_vel = dep.state[3:6] - self.chief.state[3:6]
            controller.last_estimated_state = np.concatenate([init_rel_pos, init_rel_vel])
            self.deputy_controllers[dep.id] = controller

    def _design_control_law(self):
        """此方法在基类 run 中被调用，但编队模式无需额外操作"""
        pass

    def _initialize_data_logging(self):
        """初始化日志：先调用基类创建标准数据集，再创建编队数据集，并保存每个从星的期望相对状态"""
        super()._initialize_data_logging()
        
        with h5py.File(self.h5_file, 'a') as f:
            # 创建编队数据集组
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
                        dep_grp.create_dataset(dset, shape=(0,3), maxshape=(None,3), dtype=np.float64, chunks=True, compression="gzip")
            
            # 保存每个从星的期望相对状态（如果配置中存在）
            formation_targets = self.config.get("formation_targets", {})
            if formation_targets:
                targets_grp = f.require_group("metadata/targets")
                for dep_id, target_state in formation_targets.items():
                    # target_state 应为长度为6的数组 [x, y, z, vx, vy, vz]
                    target_array = np.array(target_state, dtype=np.float64)
                    dset_name = f"deputy_{dep_id}"
                    if dset_name in targets_grp:
                        del targets_grp[dset_name]  # 覆盖已存在的
                    targets_grp.create_dataset(dset_name, data=target_array)

    def _execute_simulation_loop(self):
        """核心仿真主循环"""
        dt = self.config["time_step"]
        total_steps = int(self.config["simulation_days"] * 86400 / dt)
        all_nodes = [self.chief] + self.deputies
        
        target_rel_lvlh = self.config.get("formation_targets", {})
        
        for step in range(total_steps):
            epoch = step * dt
            
            # 1. 环境加速度并行计算
            state_matrix = np.array([node.state for node in all_nodes])
            acc_matrix = self.environment.compute_accelerations(state_matrix)

            # 2. 状态传播（Euler）
            for i, node in enumerate(all_nodes):
                gravity = acc_matrix[i]
                deriv = node.get_derivative(gravity, node.frame)
                node.state += deriv * dt
                node.integrate_dv(dt)
                node.clear_thrust()

            # 3. 控制决策（使用控制器内置的 K 矩阵）
            for dep in self.deputies:
                controller = self.deputy_controllers[dep.id]
                current_rel = np.concatenate([
                    dep.state[:3] - self.chief.state[:3],
                    dep.state[3:6] - self.chief.state[3:6]
                ])
                target_rel = target_rel_lvlh.get(dep.id, np.zeros(6))
                error = current_rel - target_rel
                u_accel = -controller.K @ error
                force_cmd = u_accel * dep.mass
                dep.apply_control(force_cmd)
                dep.last_control_force = force_cmd

            # 4. 质量更新
            for node in all_nodes:
                node.update_mass(dt)
            
            # 5. 推进环境时间
            self.environment.step_time(dt)
            
            # 6. 记录编队数据（每10步）
            if step % 10 == 0:
                self._log_formation_step(epoch)
            
            # 7. 进度报告
            if self.verbose and (step % (max(1, total_steps // 20)) == 0 or step == total_steps-1):
                self._report_progress_custom(epoch, step, total_steps)

    def _log_formation_step(self, epoch):
        """记录编队数据到 HDF5"""
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
                    mode_int = {FormationMode.GENERATION:0, FormationMode.KEEPING:1, FormationMode.RECONFIGURATION:2}.get(mode,0)
                    
                    for dset, data in [
                        ("time", epoch),
                        ("rel_position", rho),
                        ("rel_velocity", rho_dot),
                        ("control_force", control),
                        ("mode", mode_int)
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
        print(f"   主星 {self.chief.id} ΔV: {self.chief.accumulated_dv:.6f} m/s")
        for dep in self.deputies:
            print(f"   从星 {dep.id} ΔV: {dep.accumulated_dv:.6f} m/s")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
