# mission_sim/simulation/base.py
"""
顶层仿真基类 (模板方法模式)
定义 L1-L5 仿真的通用流程骨架，具体实现由子类填充。
"""

import os
import time
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, Dict

from mission_sim.core.types import CoordinateFrame
from mission_sim.utils.logger import HDF5Logger, SimulationMetadata
from mission_sim.utils.visualizer_L1 import L1Visualizer


class BaseSimulation(ABC):
    """
    仿真控制器抽象基类。

    采用模板方法模式，将仿真流程拆分为多个可重写的步骤（hooks）。
    子类只需关注场景特定的部分（轨道生成、环境初始化、控制律等），
    而主循环、日志、后处理等通用流程由基类提供默认实现，但允许子类重写。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基类。

        Args:
            config: 仿真配置字典，至少包含以下字段：
                - mission_name: 任务名称
                - simulation_days: 仿真天数
                - time_step: 积分步长 (s)
                - data_dir: 数据输出目录
                - verbose: 是否输出详细信息（默认 True）
                其他字段由子类定义。
        """
        self.config = config
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_start_time = None
        self.simulation_end_time = None
        self.current_step = 0
        self.total_steps = 0

        # 核心组件（由子类初始化）
        self.ephemeris = None          # 标称轨道星历
        self.environment = None        # 环境引擎（CelestialEnvironment）
        self.spacecraft = None         # 航天器模型（SpacecraftPointMass 或更高层级的模型）
        self.ground_station = None     # 地面站（GroundStation）
        self.gnc_system = None         # GNC 子系统（GNC_Subsystem 或其子类）
        self.logger = None             # 数据记录器（HDF5Logger）
        self.k_matrix = None           # 控制增益矩阵（可选）

        # 输出目录和文件
        self.data_dir = self.config.get("data_dir", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.h5_file = os.path.join(
            self.data_dir,
            f"{self.config.get('mission_name', 'mission').replace(' ', '_')}_{self.mission_id}.h5"
        )

        # 控制输出详细程度
        self.verbose = self.config.get("verbose", True)

        # 打印启动信息（仅当 verbose 开启）
        if self.verbose:
            self._print_startup_info()

    def _print_startup_info(self):
        """打印启动信息（子类可重写以定制输出）"""
        print("=" * 80)
        print(f"🚀 MCPC 仿真: {self.config.get('mission_name', 'Unnamed')}")
        print(f"   任务ID: {self.mission_id}")
        print(f"   仿真时长: {self.config.get('simulation_days', 0)} 天")
        print(f"   积分步长: {self.config.get('time_step', 0)} 秒")
        print(f"   输出文件: {self.h5_file}")
        print("=" * 80)

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
    def _initialize_physical_domain(self):
        """
        初始化物理域（环境、航天器）。
        子类必须实现，创建 self.environment 和 self.spacecraft。
        """
        pass

    @abstractmethod
    def _initialize_information_domain(self):
        """
        初始化信息域（测控、GNC）。
        子类必须实现，创建 self.ground_station 和 self.gnc_system。
        """
        pass

    @abstractmethod
    def _design_control_law(self):
        """
        设计控制律（计算反馈增益矩阵）。
        子类必须实现，将结果存入 self.k_matrix。
        """
        pass

    # ====================== 可重写方法（子类可按需重写） ======================
    def _generate_fallback_orbit(self):
        """
        备用轨道生成（当主生成失败时调用）。
        默认抛出 NotImplementedError，子类可按需实现。
        """
        raise NotImplementedError("子类必须实现备用轨道生成方法")

    def _initialize_data_logging(self):
        """
        初始化数据记录器。
        默认使用 HDF5Logger，子类可重写以使用其他记录方式或扩展元数据。
        """
        # 构建元数据字典
        metadata = {
            "mission_name": self.config["mission_name"],
            "simulation_days": self.config["simulation_days"],
            "time_step": self.config["time_step"],
            "spacecraft_mass": self.spacecraft.mass,
            "control_type": "LQR",
            "mission_id": self.mission_id,
            "ephemeris_period_days": self.ephemeris.times[-1] / 86400
        }
    
        # 使用 HDF5Logger 初始化文件
        self.logger = HDF5Logger(
            filepath=self.h5_file,
            buffer_size=self.config.get("log_buffer_size", 500),
            compression=self.config.get("log_compression", True),
            auto_flush=True
        )
        self.logger.initialize_file(metadata)
    
        if self.verbose:
            print(f"[日志] 数据文件: {self.h5_file}")

    def _execute_simulation_loop(self):
        """
        主循环（默认使用 RK4 积分，子类可重写以使用其他积分器或改变循环结构）。
        此方法调用多个 hook 方法，子类可通过重写这些 hook 来定制行为。
        """
        dt = self.config["time_step"]
        sim_seconds = self.config["simulation_days"] * 86400
        self.total_steps = int(sim_seconds / dt)
        progress_steps = max(1, int(self.total_steps * self.config.get("progress_interval", 0.05)))

        if self.verbose:
            print(f"\n开始闭环仿真，总步数: {self.total_steps}, 步长: {dt} 秒")
        for step in range(self.total_steps):
            self.current_step = step
            epoch = step * dt

            # 1. 导航感知
            obs_state, frame = self._get_observation(epoch)

            # 2. 控制决策
            force_cmd, force_frame = self._compute_control(epoch, obs_state, frame)

            # 3. 状态传播
            self._propagate_state(force_cmd, force_frame, dt)

            # 4. 燃料统计与清理
            self._post_step_processing(dt)

            # 5. 数据记录（降采样）
            if step % 10 == 0:
                self._log_step(epoch)

            # 6. 进度报告（仅当 verbose 开启）
            if self.verbose and step % progress_steps == 0 and step > 0:
                self._report_progress(step, epoch, force_cmd)

        if self.verbose:
            print("仿真主循环完成")

    # ---------- 主循环中的 hook 方法 ----------
    def _get_observation(self, epoch: float):
        """
        获取当前时刻的观测状态。
        默认从地面站获取，子类可重写以实现其他感知方式（如星间链路）。
        Returns:
            (obs_state, frame): 观测状态向量和坐标系
        """
        return self.ground_station.track_spacecraft(
            self.spacecraft.state, self.spacecraft.frame, epoch
        )

    def _compute_control(self, epoch: float, obs_state: Optional[np.ndarray], frame: CoordinateFrame):
        """
        根据观测状态计算控制力。
        默认调用 GNC 子系统，子类可重写以实现自定义控制逻辑。
        Returns:
            (force_cmd, force_frame): 控制力向量和坐标系
        """
        self.gnc_system.update_navigation(obs_state, frame, self.config["time_step"])
        force_cmd, force_frame = self.gnc_system.compute_control_force(epoch, self.k_matrix)
        force_cmd = self._ensure_3d_control_force(force_cmd)
        return force_cmd, force_frame

    def _propagate_state(self, force_cmd: np.ndarray, force_frame: CoordinateFrame, dt: float):
        """
        状态传播（默认 RK4 积分）。
        子类可重写以使用其他积分器或增加更多物理效应。
        """
        self.spacecraft.apply_thrust(force_cmd, force_frame)
        k1 = self._get_state_derivative(self.spacecraft.state)
        k2 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k1)
        k3 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k2)
        k4 = self._get_state_derivative(self.spacecraft.state + dt * k3)
        self.spacecraft.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _get_state_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        计算状态导数（供 RK4 使用）。
        默认使用环境引擎计算加速度，子类可重写以使用更复杂的动力学。
        """
        acc_env, acc_frame = self.environment.get_total_acceleration(state, self.spacecraft.frame)
        deriv = np.zeros(6)
        deriv[0:3] = state[3:6]
        deriv[3:6] = acc_env + self.spacecraft.external_accel
        return deriv

    def _post_step_processing(self, dt: float):
        """
        步后处理：燃料累计、清空推力、环境时间推进。
        子类可重写以增加额外处理。
        """
        self.spacecraft.integrate_dv(dt)
        self.spacecraft.clear_thrust()
        self.environment.step_time(dt)

    def _log_step(self, epoch: float):
        """
        记录单步数据。
        默认记录标称状态、真值、导航状态、跟踪误差、控制力、累计 ΔV。
        子类可重写以增加自定义记录内容。
        """
        nom_state = self.ephemeris.get_interpolated_state(epoch)
        self.logger.log_step(
            epoch=epoch,
            nominal_state=nom_state,
            true_state=self.spacecraft.state,
            nav_state=self.gnc_system.current_nav_state,
            tracking_error=self.gnc_system.last_tracking_error,
            control_force=self.gnc_system.last_control_force,
            accumulated_dv=self.spacecraft.accumulated_dv
        )

    def _report_progress(self, step: int, epoch: float, force_cmd: np.ndarray):
        """
        报告仿真进度。
        子类可重写以定制输出格式。
        """
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

    # ---------- 后处理 hook ----------
    def _finalize_simulation(self) -> bool:
        """
        仿真后处理：关闭日志、生成燃料账单、可视化。
        子类可重写以增加自定义后处理逻辑。
        """
        self.simulation_end_time = time.time()
        if self.logger:
            self.logger.close()

        # 燃料账单 CSV
        fuel_bill_path = os.path.join(self.data_dir, f"fuel_bill_{self.mission_id}.csv")
        try:
            with open(fuel_bill_path, 'w') as f:
                f.write("mission_id,total_dv_mps,avg_dv_per_day_mps,simulation_days,final_position_error_m,final_velocity_error_mms\n")
                f.write(f"{self.mission_id},{self.spacecraft.accumulated_dv:.6f},"
                        f"{self.spacecraft.accumulated_dv / self.config['simulation_days']:.6f},"
                        f"{self.config['simulation_days']},"
                        f"{np.linalg.norm(self.gnc_system.last_tracking_error[0:3]):.2f},"
                        f"{np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000:.2f}\n")
            if self.verbose:
                print(f"✅ 燃料账单已保存: {fuel_bill_path}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 燃料账单保存失败: {e}")

        # 可视化
        if self.config.get("enable_visualization", False):
            self._generate_visualization()

        # 打印汇总信息（仅当 verbose 开启）
        if self.verbose:
            self._print_summary()

        return True

    def _generate_visualization(self):
        """生成可视化报告（默认使用 L1Visualizer，子类可重写）"""
        try:
            vis = L1Visualizer(self.h5_file, self.config["mission_name"])
            base = os.path.join(self.data_dir, f"{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}")
            vis.plot_3d_trajectory(save_path=f"{base}_trajectory.png")
            vis.plot_tracking_error(save_path=f"{base}_errors.png")
            vis.plot_control_effort(save_path=f"{base}_control.png")
            if self.verbose:
                print("✅ 可视化报告生成完成")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 可视化失败: {e}")

    def _print_summary(self):
        """打印仿真结果汇总"""
        print("\n📊 仿真结果汇总")
        print("-" * 60)
        print(f"✅ 仿真完成!")
        print(f"   实际仿真时间: {self.simulation_end_time - self.simulation_start_time:.1f} 秒")
        print(f"   仿真步数: {self.total_steps:,}")
        final_err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
        final_err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
        print(f"   最终位置误差: {final_err_pos:.2f} m")
        print(f"   最终速度误差: {final_err_vel:.2f} mm/s")
        print(f"   总 ΔV 消耗: {self.spacecraft.accumulated_dv:.4f} m/s")
        print(f"   平均每天 ΔV: {self.spacecraft.accumulated_dv / self.config['simulation_days']:.4f} m/s/天")
        print(f"   数据文件: {os.path.abspath(self.h5_file)}")
        if self.logger:
            stats = self.logger.get_statistics()
            if "file_size_mb" in stats:
                print(f"   数据文件大小: {stats['file_size_mb']:.2f} MB")

    # ---------- 辅助方法 ----------
    def _ensure_3d_control_force(self, force) -> np.ndarray:
        """确保控制力是 3 维数组"""
        if isinstance(force, (int, float, np.number)):
            return np.array([float(force), 0.0, 0.0], dtype=np.float64)
        elif isinstance(force, np.ndarray):
            if force.shape == ():
                return np.array([float(force), 0.0, 0.0], dtype=np.float64)
            elif force.shape == (1,):
                return np.array([float(force[0]), 0.0, 0.0], dtype=np.float64)
            elif force.shape == (3,):
                return force.astype(np.float64)
            elif force.size >= 3:
                return force[:3].astype(np.float64)
        return np.zeros(3, dtype=np.float64)

    def _emergency_shutdown(self):
        """紧急关闭处理（在异常或中断时调用）"""
        if self.verbose:
            print("\n⚠️ 执行紧急关闭...")
        if self.logger:
            try:
                self.logger.close()
                if self.verbose:
                    print(f"✅ 已保存数据到: {self.h5_file}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ 数据保存失败: {e}")
        if self.current_step > 0:
            completed_days = (self.current_step * self.config["time_step"]) / 86400
            if self.verbose:
                print(f"   已完成 {completed_days:.2f} 天仿真")

    def get_statistics(self) -> dict:
        """获取仿真统计信息"""
        stats = {
            "mission_id": self.mission_id,
            "mission_name": self.config["mission_name"],
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "completed": self.current_step >= self.total_steps,
            "data_file": self.h5_file,
        }
        if self.spacecraft:
            stats["accumulated_dv"] = self.spacecraft.accumulated_dv
        if self.gnc_system:
            stats["final_position_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
            stats["final_velocity_error"] = np.linalg.norm(self.gnc_system.last_tracking_error[3:6])
        if self.logger:
            stats.update(self.logger.get_statistics())
        if self.simulation_start_time and self.simulation_end_time:
            stats["simulation_duration_seconds"] = self.simulation_end_time - self.simulation_start_time
        return stats

    # ====================== 主流程 ======================
    def run(self) -> bool:
        """执行完整仿真流程（模板方法）"""
        try:
            self.simulation_start_time = time.time()
            if not self._generate_nominal_orbit():
                if self.verbose:
                    print("⚠️ 使用备用轨道")
                self._generate_fallback_orbit()
            self._initialize_physical_domain()
            self._initialize_information_domain()
            self._design_control_law()
            self._initialize_data_logging()
            self._execute_simulation_loop()
            return self._finalize_simulation()
        except KeyboardInterrupt:
            if self.verbose:
                print("\n⏹️ 仿真被用户中断")
            self._emergency_shutdown()
            return False
        except Exception as e:
            if self.verbose:
                print(f"\n❌ 仿真运行失败: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            else:
                # 静默模式也打印错误，但简短一些
                print(f"❌ 仿真运行失败: {e}")
            self._emergency_shutdown()
            return False
