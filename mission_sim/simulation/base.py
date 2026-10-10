# mission_sim/simulation/base.py
"""
顶层仿真基类 (模板方法模式)
定义 L1-L5 仿真的通用流程骨架，并提供对单星与多星编队的健壮性支持。
"""

import os
import time
import uuid
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, Dict

from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.utils.logger import HDF5Logger, SimulationMetadata
from mission_sim.utils.visualizer_L1 import L1Visualizer


class BaseSimulation(ABC):
    """
    仿真控制器抽象基类。
    采用模板方法模式，将仿真流程拆分为多个可重写的步骤（hooks）。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mission_id = config.get("mission_id")
        if self.mission_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            uid = str(uuid.uuid4())[:8]
            self.mission_id = f"{timestamp}_{pid}_{uid}"
        self.simulation_start_time = None
        self.simulation_end_time = None
        self.current_step = 0
        self.total_steps = 0

        # 核心组件（由子类初始化）
        self.ephemeris = None          
        self.environment = None        
        self.spacecraft = None         
        self.ground_station = None     
        self.gnc_system = None         
        self.logger = None             
        self.k_matrix = None           

        # 积分器配置
        self.integrator_type = self.config.get("integrator", "rk4")
        self.integrator_rtol = self.config.get("integrator_rtol", 1e-9)
        self.integrator_atol = self.config.get("integrator_atol", 1e-12)

        # 输出目录和文件
        self.data_dir = self.config.get("data_dir", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.h5_file = os.path.abspath(os.path.join(
            self.data_dir,
            f"{self.config.get('mission_name', 'mission').replace(' ', '_')}_{self.mission_id}.h5"
        ))

        self.verbose = self.config.get("verbose", True)

        if self.verbose:
            self._print_startup_info()

    def _print_startup_info(self):
        print("=" * 80)
        print(f"🚀 MCPC 仿真: {self.config.get('mission_name', 'Unnamed')}")
        print(f"   任务ID: {self.mission_id}")
        print(f"   仿真时长: {self.config.get('simulation_days', 0)} 天")
        print(f"   积分步长: {self.config.get('time_step', 0)} 秒")
        print(f"   积分器: {self.integrator_type}")
        print(f"   输出文件: {self.h5_file}")
        print("=" * 80)

    # ====================== 抽象方法 ======================
    @abstractmethod
    def _generate_nominal_orbit(self) -> bool: pass

    @abstractmethod
    def _initialize_physical_domain(self): pass

    @abstractmethod
    def _initialize_information_domain(self): pass

    @abstractmethod
    def _design_control_law(self): pass

    # ====================== 逻辑修正的方法 ======================

    def _initialize_data_logging(self):
        """修正：安全访问 spacecraft.mass，防止多星模式崩溃"""
        sc_mass = getattr(self.spacecraft, 'mass', 0.0) if self.spacecraft else 0.0
        eph_period = 0.0
        if self.ephemeris and hasattr(self.ephemeris, 'times') and len(self.ephemeris.times) > 0:
            eph_period = self.ephemeris.times[-1] / 86400

        metadata = {
            "mission_name": self.config["mission_name"],
            "simulation_days": self.config["simulation_days"],
            "time_step": self.config["time_step"],
            "spacecraft_mass": sc_mass,
            "control_type": "LQR",
            "mission_id": self.mission_id,
            "ephemeris_period_days": eph_period,
            "integrator": self.integrator_type,
        }

        self.logger = HDF5Logger(
            filepath=self.h5_file,
            buffer_size=self.config.get("log_buffer_size", 500),
            compression=self.config.get("log_compression", True),
            auto_flush=True,
            verbose=self.verbose,
            backup=self.config.get("log_backup", True)
        )
        self.logger.initialize_file(metadata)

    def _report_progress(self, step: int, epoch: float, force_cmd: np.ndarray):
        """修正：安全访问 gnc_system，防止多星模式进度报告报错"""
        progress = (step / self.total_steps) * 100
        days = epoch / 86400
        
        # 基础报告信息
        msg = f"  [Day {days:6.1f}] 进度: {progress:5.1f}%"
        
        # 如果是单星 GNC 模式，则报告误差
        if self.gnc_system and hasattr(self.gnc_system, 'last_tracking_error'):
            err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
            err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
            msg += f" | 位置误差: {err_pos:6.1f}m | 速度误差: {err_vel:6.2f}mm/s"
        
        # 报告控制力与 ΔV (如果存在)
        control_norm = np.linalg.norm(force_cmd)
        msg += f" | 控制力: {control_norm:7.4f}N"
        
        if self.spacecraft and hasattr(self.spacecraft, 'accumulated_dv'):
            msg += f" | 累计 ΔV: {self.spacecraft.accumulated_dv:8.4f}m/s"
            
        print(msg)

    def _print_summary(self):
        """修正：安全统计，防止最后打印总结时抛出 AttributeError"""
        print("\n📊 仿真结果汇总")
        print("-" * 60)
        print(f"✅ 仿真完成!")
        print(f"   实际仿真时间: {self.simulation_end_time - self.simulation_start_time:.1f} 秒")
        print(f"   仿真步数: {self.total_steps:,}")
        
        # 处理单星 ΔV 消耗
        if self.spacecraft and hasattr(self.spacecraft, 'accumulated_dv'):
            print(f"   总 ΔV 消耗: {self.spacecraft.accumulated_dv:.4f} m/s")
            print(f"   平均每天 ΔV: {self.spacecraft.accumulated_dv / self.config['simulation_days']:.4f} m/s/天")

        # 处理单星 GNC 误差
        if self.gnc_system and hasattr(self.gnc_system, 'last_tracking_error'):
            final_err_pos = np.linalg.norm(self.gnc_system.last_tracking_error[0:3])
            final_err_vel = np.linalg.norm(self.gnc_system.last_tracking_error[3:6]) * 1000
            print(f"   最终位置误差: {final_err_pos:.2f} m")
            print(f"   最终速度误差: {final_err_vel:.2f} mm/s")
        elif not hasattr(self, 'spacecraft'): 
            print("   编队维持状态: 详见 HDF5 日志分析")

        print(f"   数据文件: {os.path.abspath(self.h5_file)}")

    # ====================== 以下是保留的原始核心脚手架逻辑 ======================

    def _generate_fallback_orbit(self):
        raise NotImplementedError("子类必须实现备用轨道生成方法")

    def _execute_simulation_loop(self):
        dt = self.config["time_step"]
        sim_seconds = self.config["simulation_days"] * 86400
        self.total_steps = int(sim_seconds / dt)
        progress_steps = max(1, int(self.total_steps * self.config.get("progress_interval", 0.05)))

        if self.verbose: print(f"\n开始闭环仿真，总步数: {self.total_steps}, 步长: {dt} 秒")
        for step in range(self.total_steps):
            self.current_step = step
            epoch = step * dt
            obs_state, frame = self._get_observation(epoch)
            force_cmd, force_frame = self._compute_control(epoch, obs_state, frame)
            self._propagate_state(force_cmd, force_frame, dt)
            self._post_step_processing(dt)
            if step % int(self.config.get("log_interval", 10)) == 0: self._log_step(epoch)
            if self.verbose and step % progress_steps == 0 and step > 0:
                self._report_progress(step, epoch, force_cmd)
        if self.verbose: print("仿真主循环完成")

    def _get_observation(self, epoch: float):
        return self.ground_station.track_spacecraft(self.spacecraft.state, self.spacecraft.frame, epoch)

    def _compute_control(self, epoch: float, obs_state: Optional[np.ndarray], frame: CoordinateFrame):
        self.gnc_system.update_navigation(obs_state, frame, self.config["time_step"])
        force_cmd, force_frame = self.gnc_system.compute_control_force(epoch, self.k_matrix)
        force_cmd = self._ensure_3d_control_force(force_cmd)
        return force_cmd, force_frame

    def _propagate_state(self, force_cmd: np.ndarray, force_frame: CoordinateFrame, dt: float):
        self.spacecraft.apply_thrust(force_cmd, force_frame)
        if self.integrator_type == "rk45":
            from scipy.integrate import solve_ivp
            def dynamics(t, y):
                acc_env, _ = self.environment.get_total_acceleration(y, self.spacecraft.frame)
                return np.concatenate([y[3:6], acc_env + self.spacecraft.external_accel])
            t0 = self.environment.epoch
            sol = solve_ivp(dynamics, (t0, t0 + dt), self.spacecraft.state, method='RK45', 
                            rtol=self.integrator_rtol, atol=self.integrator_atol)
            self.spacecraft.state = sol.y[:, -1] if sol.success else self.spacecraft.state
        else:
            k1 = self._get_state_derivative(self.spacecraft.state)
            k2 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k1)
            k3 = self._get_state_derivative(self.spacecraft.state + 0.5 * dt * k2)
            k4 = self._get_state_derivative(self.spacecraft.state + dt * k3)
            self.spacecraft.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _get_state_derivative(self, state: np.ndarray) -> np.ndarray:
        acc_env, _ = self.environment.get_total_acceleration(state, self.spacecraft.frame)
        return np.concatenate([state[3:6], acc_env + self.spacecraft.external_accel])

    def _post_step_processing(self, dt: float):
        self.spacecraft.integrate_dv(dt)
        self.spacecraft.clear_thrust()
        self.environment.step_time(dt)

    def _log_step(self, epoch: float):
        nom_state = self.ephemeris.get_state(epoch)
        self.logger.log_step(epoch=epoch, nominal_state=nom_state, true_state=self.spacecraft.state,
                             nav_state=self.gnc_system.current_nav_state,
                             tracking_error=self.gnc_system.last_tracking_error,
                             control_force=self.gnc_system.last_control_force,
                             accumulated_dv=self.spacecraft.accumulated_dv)

    def _finalize_simulation(self) -> bool:
        self.simulation_end_time = time.time()
        if self.logger: self.logger.close()
        if self.config.get("enable_visualization", False): self._generate_visualization()
        if self.verbose: self._print_summary()
        return True

    def _generate_visualization(self):
        try:
            vis = L1Visualizer(self.h5_file, self.config["mission_name"])
            base = os.path.join(self.data_dir, f"{self.config['mission_name'].replace(' ', '_')}_{self.mission_id}")
            vis.plot_3d_trajectory(save_path=f"{base}_trajectory.png")
            if self.verbose: print("✅ 可视化报告生成完成")
        except Exception as e:
            if self.verbose: print(f"⚠️ 可视化失败: {e}")

    def _ensure_3d_control_force(self, force) -> np.ndarray:
        if isinstance(force, np.ndarray) and force.shape == (3,): return force
        return np.array([float(force), 0.0, 0.0]) if np.isscalar(force) else np.zeros(3)

    def _emergency_shutdown(self):
        if self.verbose: print("\n⚠️ 执行紧急关闭...")
        if self.logger: self.logger.close()

    def run(self) -> bool:
        try:
            self.simulation_start_time = time.time()
            if not self._generate_nominal_orbit(): self._generate_fallback_orbit()
            self._initialize_physical_domain()
            self._initialize_information_domain()
            self._design_control_law()
            self._initialize_data_logging()
            self._execute_simulation_loop()
            return self._finalize_simulation()
        except Exception as e:
            if self.verbose: print(f"\n❌ 仿真运行失败: {e}"); import traceback; traceback.print_exc()
            self._emergency_shutdown()
            return False
