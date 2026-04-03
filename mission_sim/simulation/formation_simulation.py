# mission_sim/simulation/formation_simulation.py
"""
MCPC L2 Formation Simulation: Multi-Satellite Closed-Loop Simulation
---------------------------------------------------------------------
Orchestrates the complete simulation of a chief + N deputy spacecraft formation.
Follows the strict tick order defined in the L2 architecture.
"""

import numpy as np
import h5py, os
from typing import List, Dict
from mission_sim.simulation.base import BaseSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.physics.spacecraft_node import SpacecraftNode
from mission_sim.core.physics.ids import SpacecraftType
from mission_sim.core.cyber.network.isl_router import ISLRouter
from mission_sim.core.cyber.platform_gnc.formation_controller import FormationController
from mission_sim.core.cyber.models.cw_dynamics import CWDynamics
from mission_sim.core.cyber.ids import FormationMode
from mission_sim.utils.math_tools import absolute_to_lvlh


class FormationSimulation(BaseSimulation):
    def __init__(self, config: dict):
        super().__init__(config)

        # Convert chief_frame from string to enum if needed
        chief_frame = config.get("chief_frame", "J2000_ECI")
        if isinstance(chief_frame, str):
            frame_map = {
                "J2000_ECI": CoordinateFrame.J2000_ECI,
                "SUN_EARTH_ROTATING": CoordinateFrame.SUN_EARTH_ROTATING,
                "LVLH": CoordinateFrame.LVLH,
            }
            self.chief_frame = frame_map.get(chief_frame, CoordinateFrame.J2000_ECI)
        else:
            self.chief_frame = chief_frame

        self.chief_mass = config.get("chief_mass_kg", 2000.0)
        self.deputy_mass = config.get("deputy_mass_kg", 500.0)
        self.orbit_n = config.get("orbit_angular_rate", 0.001)

        # Initialize chief node
        chief_initial = np.array(config["chief_initial_state"], dtype=np.float64)
        self.chief = SpacecraftNode(
            sc_id="CHIEF",
            initial_state=chief_initial,
            frame=self.chief_frame,
            initial_mass=self.chief_mass,
            sc_type=SpacecraftType.CHIEF,
        )

        # Initialize deputies
        self.deputies: List[SpacecraftNode] = []
        self.deputy_controllers: Dict[str, FormationController] = {}
        deputy_states = config.get("deputy_initial_states", [])

        for dep_id, init_state in deputy_states:
            dep = SpacecraftNode(
                sc_id=dep_id,
                initial_state=np.array(init_state, dtype=np.float64),
                frame=self.chief_frame,
                initial_mass=self.deputy_mass,
                sc_type=SpacecraftType.DEPUTY,
            )
            self.deputies.append(dep)

            # Create formation controller for this deputy
            dynamics = CWDynamics(n=self.orbit_n)
            controller = FormationController(
                deputy_id=dep_id,
                chief_id="CHIEF",
                dynamics=dynamics,
                generation_threshold_pos=config.get("generation_threshold_pos", 100.0),
                generation_threshold_vel=config.get("generation_threshold_vel", 0.5),
                keeping_threshold_pos=config.get("keeping_threshold_pos", 1.0),
                keeping_threshold_vel=config.get("keeping_threshold_vel", 0.01),
            )
            self.deputy_controllers[dep_id] = controller

        # Assign routers to deputies
        router_seed = config.get("router_seed", 42)
        for dep in self.deputies:
            dep.router = ISLRouter(
                base_latency_s=config.get("router_base_latency_s", 0.05),
                jitter_s=config.get("router_jitter_s", 0.01),
                packet_loss_rate=config.get("router_packet_loss_rate", 0.02),
                random_seed=router_seed,
            )

        # Environment
        self.environment = CelestialEnvironment(
            computation_frame=self.chief_frame,
            initial_epoch=0.0,
            verbose=self.verbose,
        )

        # For base class logging compatibility
        self.gnc_system = None
        self.spacecraft = self.chief

    def _initialize_physical_domain(self):
        if self.config.get("enable_crtbp", False):
            from mission_sim.core.physics.models.gravity_crtbp import GravityCRTBP
            self.environment.register_force(GravityCRTBP())
        if self.config.get("enable_j2", False):
            from mission_sim.core.physics.models.j2_gravity import J2Gravity
            self.environment.register_force(J2Gravity())
        if self.config.get("enable_atmospheric_drag", False):
            from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag
            drag = AtmosphericDrag(area_to_mass=self.config.get("area_to_mass", 0.02))
            self.environment.register_force(drag)
        if self.config.get("enable_srp", False):
            from mission_sim.core.physics.models.srp import CannonballSRP
            srp = CannonballSRP(area_to_mass=self.config.get("area_to_mass", 0.02))
            self.environment.register_force(srp)

    def _initialize_information_domain(self):
        pass

    def _generate_nominal_orbit(self) -> bool:
        if "ephemeris" in self.config:
            self.ephemeris = self.config["ephemeris"]
            return True
        if "chief_elements" in self.config:
            from mission_sim.core.spacetime.generators import KeplerianGenerator
            elements = self.config["chief_elements"]
            gen = KeplerianGenerator(mu=self.config.get("mu_earth", 3.986004418e14))
            sim_seconds = self.config["simulation_days"] * 86400
            dt = self.config["time_step"]
            self.ephemeris = gen.generate({"elements": elements, "dt": dt, "sim_time": sim_seconds})
            return True

        # Fallback: generate circular orbit from chief initial state
        r0 = np.linalg.norm(self.chief.position)
        v0 = np.linalg.norm(self.chief.velocity)
        n = v0 / r0
        sim_seconds = self.config["simulation_days"] * 86400
        dt = self.config["time_step"]
        times = np.arange(0, sim_seconds + dt, dt)
        states = []
        for t in times:
            theta = n * t
            x = r0 * np.cos(theta)
            y = r0 * np.sin(theta)
            vx = -r0 * n * np.sin(theta)
            vy = r0 * n * np.cos(theta)
            states.append([x, y, 0.0, vx, vy, 0.0])
        self.ephemeris = Ephemeris(times, np.array(states), self.chief_frame)
        if self.verbose:
            print(f"自动生成圆轨道星历，半径 {r0/1000:.1f} km，角速度 {n:.6f} rad/s")
        return True

    def _design_control_law(self):
        pass

    def _initialize_data_logging(self):
        super()._initialize_data_logging()
        with h5py.File(self.h5_file, 'a') as f:
            formation_grp = f.require_group("formation")
            for dep in self.deputies:
                dep_grp = formation_grp.require_group(f"deputy_{dep.id}")
                for dset in ["time", "rel_position", "rel_velocity", "control_force", "mode"]:
                    if dset not in dep_grp:
                        if dset == "time":
                            shape, maxshape, dtype = (0,), (None,), np.float64
                        elif dset == "mode":
                            shape, maxshape, dtype = (0,), (None,), np.int8
                        else:
                            shape, maxshape, dtype = (0, 3), (None, 3), np.float64
                        dep_grp.create_dataset(dset, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True, compression="gzip")

    def _execute_simulation_loop(self):
        dt = self.config["time_step"]
        sim_seconds = self.config["simulation_days"] * 86400
        self.total_steps = int(sim_seconds / dt)
        progress_steps = max(1, int(self.total_steps * self.config.get("progress_interval", 0.05)))

        if self.verbose:
            print(f"\n开始多星闭环仿真，总步数: {self.total_steps}, 步长: {dt} 秒")
            print(f"航天器: 1 主星 + {len(self.deputies)} 从星")

        all_nodes = [self.chief] + self.deputies

        for step in range(self.total_steps):
            epoch = step * dt
            self.current_step = step

            # 1. Parallel acceleration computation
            state_matrix = np.array([node.state for node in all_nodes])
            acc_matrix = self.environment.compute_accelerations(state_matrix)

            # 2. State propagation (Euler)
            for i, node in enumerate(all_nodes):
                gravity = acc_matrix[i]
                deriv = node.get_derivative(gravity, node.frame)
                node.state += deriv * dt
                node.integrate_dv(dt)
                node.clear_thrust()

            # 3. Generate physical measurements (deputies -> chief)
            measurements = []
            for dep in self.deputies:
                meas = dep.sense(self.chief, epoch)
                if meas is not None:
                    measurements.append((meas, dep.id, "CHIEF"))

            # 4. Route measurements through routers
            frames_by_deputy = {dep.id: [] for dep in self.deputies}
            for meas, src, dst in measurements:
                # Only process frames destined for deputies (chief doesn't need them)
                if dst not in frames_by_deputy:
                    continue
                for dep in self.deputies:
                    if dep.id == src:
                        frame = dep.transmit(meas, dst, epoch)
                        if frame is not None:
                            frames_by_deputy[dst].append(frame)
                        break

            # 5. Control: each deputy computes force
            for dep in self.deputies:
                controller = self.deputy_controllers[dep.id]
                frames = frames_by_deputy.get(dep.id, [])
                cmd = controller.update(epoch, frames, dt)
                dep.apply_control(cmd.force_vector)

            # 6. Update mass
            for node in all_nodes:
                node.update_mass(dt)

            # 7. Log data every 10 steps
            if step % 10 == 0:
                self._log_formation_step(epoch)

            # 8. Progress report
            if self.verbose and step % progress_steps == 0 and step > 0:
                self._report_formation_progress(step, epoch)

        if self.verbose:
            print("多星仿真主循环完成")

    def _log_formation_step(self, epoch: float):
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
                    mode = self.deputy_controllers[dep.id].mode
                    mode_int = {FormationMode.GENERATION: 0, FormationMode.KEEPING: 1, FormationMode.RECONFIGURATION: 2}.get(mode, 0)
    
                    for dset, data in [("time", epoch), ("rel_position", rho), ("rel_velocity", rho_dot),
                                       ("control_force", control), ("mode", mode_int)]:
                        ds = dep_grp[dset]
                        ds.resize(ds.shape[0] + 1, axis=0)
                        ds[-1] = data
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 编队日志写入失败: {e}")

    def _report_formation_progress(self, step: int, epoch: float):
        progress = (step / self.total_steps) * 100
        days = epoch / 86400
        max_err = 0.0
        for dep in self.deputies:
            err = np.linalg.norm(dep.position - self.chief.position)
            max_err = max(max_err, err)
        print(f"  [Day {days:6.1f}] 进度: {progress:5.1f}% | 最大相对距离: {max_err:6.1f}m | 主星 ΔV: {self.chief.accumulated_dv:6.2f}m/s")

    def _finalize_simulation(self) -> bool:
        result = super()._finalize_simulation()
        if self.verbose:
            print("\n📊 编队仿真汇总")
            print(f"   主星 ΔV: {self.chief.accumulated_dv:.4f} m/s")
            for dep in self.deputies:
                print(f"   从星 {dep.id} ΔV: {dep.accumulated_dv:.4f} m/s")
        return result

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats["num_deputies"] = len(self.deputies)
        stats["chief_dv"] = self.chief.accumulated_dv
        stats["deputy_dv"] = {dep.id: dep.accumulated_dv for dep in self.deputies}
        return stats

    def _print_summary(self):
        """Override to avoid using gnc_system which is not used in formation simulation."""
        print("\n📊 仿真结果汇总")
        print("-" * 60)
        print(f"✅ 仿真完成!")
        print(f"   实际仿真时间: {self.simulation_end_time - self.simulation_start_time:.1f} 秒")
        print(f"   仿真步数: {self.total_steps:,}")
        print(f"   总 ΔV 消耗 (主星): {self.chief.accumulated_dv:.4f} m/s")
        for dep in self.deputies:
            print(f"   从星 {dep.id} ΔV: {dep.accumulated_dv:.4f} m/s")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
        if self.logger:
            stats = self.logger.get_statistics()
            if "file_size_mb" in stats:
                print(f"   数据文件大小: {stats['file_size_mb']:.2f} MB")
