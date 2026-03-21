# mission_sim/utils/visualizer_L1.py
"""
MCPC L1 级专用可视化工具 (继承自 BaseVisualizer)
职责：从 HDF5 中读取时序数据，绘制各种分析图表，并生成 HTML 报告。
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import welch
from typing import List, Optional, Union, Tuple
import os
import io
import base64
from datetime import datetime

from mission_sim.utils.visualizer import BaseVisualizer


class L1Visualizer(BaseVisualizer):
    """
    MCPC L1 级专用可视化工具 (继承自 BaseVisualizer)
    职责：从 HDF5 中读取时序数据，绘制 3D 绝对轨迹、GNC 追踪误差、控制力、
          误差分布直方图、控制力频谱、状态历史、推力活动时序，以及多仿真对比和 HTML 报告。
    """

    def __init__(self, filepath: str, mission_name: str = None):
        super().__init__(filepath)
        import h5py
        with h5py.File(self.filepath, 'r') as f:
            # 读取任务名称
            if mission_name is None:
                mission_name = f.attrs.get('mission_name', "L1 Baseline Mission")
            self.mission_name = mission_name
            # 读取配置
            self.config = {
                'simulation_days': f.attrs.get('simulation_days', 1.0),
                'time_step': f.attrs.get('time_step', 60.0),
                'spacecraft_mass': f.attrs.get('spacecraft_mass', 6200.0),
            }
            # 可选：读取更多配置
            for key in ['Az', 'mu', 'L', 'omega']:
                if key in f.attrs:
                    self.config[key] = f.attrs[key]

    # ====================== 核心绘图方法 ======================
    def plot_3d_trajectory(self, save_path: str = None, frame: str = 'rotating',
                           ref_point: str = 'auto', draw_ref: bool = True):
        """
        绘制 3D 标称轨道与实际物理轨迹的对比图并保存。

        Args:
            save_path: 保存路径
            frame: 参考系，可选 'rotating'（旋转系）或 'inertial'（惯性系）。
            ref_point: 参考点，用于确定画布中心和范围。可选 'auto'（自动以轨迹中心为中心），
                       'sun', 'earth', 'l2'（日地 L2 点），或自定义坐标 [x, y, z]（单位：米）。
            draw_ref: 是否在图中绘制参考点标记。
        """
        nominal_states = self.load_dataset('nominal_states')
        true_states = self.load_dataset('true_states')
        times = self.load_dataset('epochs')

        # 如果需要惯性系，进行坐标变换
        if frame == 'inertial':
            # 获取旋转角速度 ω（从元数据或使用默认值）
            try:
                with h5py.File(self.filepath, 'r') as f:
                    omega = f.attrs.get('omega', 1.990986e-7)
            except:
                omega = 1.990986e-7

            def to_inertial(pos, t):
                angle = -omega * t
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x = pos[:,0] * cos_a - pos[:,1] * sin_a
                y = pos[:,0] * sin_a + pos[:,1] * cos_a
                z = pos[:,2]
                return np.column_stack((x, y, z))

            nom_pos = to_inertial(nominal_states[:, 0:3], times)
            true_pos = to_inertial(true_states[:, 0:3], times)
            title_suffix = "Inertial Frame"
        else:
            nom_pos = nominal_states[:, 0:3]
            true_pos = true_states[:, 0:3]
            title_suffix = "Rotating Frame"

        # 转换为千米
        M2KM = 1e-3
        nom_pos_km = nom_pos * M2KM
        true_pos_km = true_pos * M2KM

        # 确定参考点坐标（单位：米，后转为千米）和轨迹（若需要）
        ref_km = None
        ref_label = None
        ref_traj_km = None

        if isinstance(ref_point, (list, tuple, np.ndarray)) and len(ref_point) == 3:
            ref_m = np.array(ref_point)
            ref_km = ref_m * M2KM
            ref_label = "Custom"
        elif ref_point == 'sun':
            try:
                with h5py.File(self.filepath, 'r') as f:
                    mu = f.attrs.get('mu', 3.00348e-6)
                    AU = f.attrs.get('AU', 1.495978707e11)
            except:
                mu, AU = 3.00348e-6, 1.495978707e11
            if frame == 'rotating':
                ref_m = np.array([-mu * AU, 0, 0])
                ref_km = ref_m * M2KM
            else:  # inertial
                # 太阳在惯性系中静止（近似）
                ref_m = np.array([-mu * AU, 0, 0])
                ref_km = ref_m * M2KM
            ref_label = "Sun"
        elif ref_point == 'earth':
            try:
                with h5py.File(self.filepath, 'r') as f:
                    mu = f.attrs.get('mu', 3.00348e-6)
                    AU = f.attrs.get('AU', 1.495978707e11)
            except:
                mu, AU = 3.00348e-6, 1.495978707e11
            if frame == 'rotating':
                ref_m = np.array([(1 - mu) * AU, 0, 0])
                ref_km = ref_m * M2KM
            else:
                # 地球在惯性系中运动，计算其轨迹
                earth_pos_rot = np.array([(1 - mu) * AU, 0, 0])  # 旋转系中地球位置
                # 将旋转系中每个时间点的地球位置转换到惯性系
                earth_pos_rot_broadcast = np.tile(earth_pos_rot, (len(times), 1))
                ref_traj = to_inertial(earth_pos_rot_broadcast, times)
                ref_traj_km = ref_traj * M2KM
                # 取第一个点作为参考点标记（也可取平均）
                ref_km = ref_traj_km[0]
            ref_label = "Earth"
        elif ref_point == 'l2':
            try:
                with h5py.File(self.filepath, 'r') as f:
                    AU = f.attrs.get('AU', 1.495978707e11)
            except:
                AU = 1.495978707e11
            if frame == 'rotating':
                ref_m = np.array([1.01106 * AU, 0, 0])
                ref_km = ref_m * M2KM
            else:
                # L2 点在惯性系中也是运动的，简单起见固定位置（近似）
                ref_m = np.array([1.01106 * AU, 0, 0])
                ref_km = ref_m * M2KM
            ref_label = "L2"

        # 计算坐标轴范围
        all_points = np.vstack([nom_pos_km, true_pos_km])
        if ref_km is not None:
            # 包含参考点
            all_points = np.vstack([all_points, ref_km])
            if ref_traj_km is not None:
                all_points = np.vstack([all_points, ref_traj_km])
        # 自动计算范围
        x_min, x_max = all_points[:,0].min(), all_points[:,0].max()
        y_min, y_max = all_points[:,1].min(), all_points[:,1].max()
        z_min, z_max = all_points[:,2].min(), all_points[:,2].max()
        # 添加余量
        margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
        xlim = [x_min - margin, x_max + margin]
        ylim = [y_min - margin, y_max + margin]
        zlim = [z_min - margin, z_max + margin]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(nom_pos_km[:, 0], nom_pos_km[:, 1], nom_pos_km[:, 2],
                color='gray', linestyle='--', linewidth=1.5, label='Nominal Orbit')
        ax.plot(true_pos_km[:, 0], true_pos_km[:, 1], true_pos_km[:, 2],
                color='dodgerblue', linewidth=2, label='True Trajectory')

        ax.scatter(true_pos_km[0, 0], true_pos_km[0, 1], true_pos_km[0, 2],
                   color='green', marker='o', s=50, label='Start')
        ax.scatter(true_pos_km[-1, 0], true_pos_km[-1, 1], true_pos_km[-1, 2],
                   color='red', marker='X', s=50, label='End')

        # 绘制参考点（如果需要）
        if draw_ref and ref_km is not None:
            ax.scatter(ref_km[0], ref_km[1], ref_km[2],
                       color='gold', marker='*', s=100, label=ref_label)
            if ref_traj_km is not None:
                ax.plot(ref_traj_km[:, 0], ref_traj_km[:, 1], ref_traj_km[:, 2],
                        color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'{ref_label} path')

        # 设置坐标轴范围
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # 设置等比例坐标轴
        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        ax.set_title(f"{self.mission_name} - 3D Absolute Trajectory ({title_suffix})",
                     fontsize=14, fontweight='bold')
        ax.legend()

        self.save_plot(fig, save_path or self._default_path(f"trajectory_{frame}"))

    def plot_tracking_error(self, save_path: str = None):
        """绘制 GNC 系统的位置与速度追踪偏差并保存"""
        times = self.load_dataset('epochs')
        errors = self.load_dataset('tracking_errors')
        days = times / 86400.0

        fig, axes = self.create_figure(2, 1, f"{self.mission_name} - Tracking Error")
        ax1, ax2 = axes

        ax1.plot(days, errors[:, 0], label='Error X', alpha=0.8)
        ax1.plot(days, errors[:, 1], label='Error Y', alpha=0.8)
        ax1.plot(days, errors[:, 2], label='Error Z', alpha=0.8)
        ax1.set_ylabel('Position Error [m]', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(days, errors[:, 3] * 1000, label='Error Vx', alpha=0.8)
        ax2.plot(days, errors[:, 4] * 1000, label='Error Vy', alpha=0.8)
        ax2.plot(days, errors[:, 5] * 1000, label='Error Vz', alpha=0.8)
        ax2.set_xlabel('Time [Days]', fontweight='bold')
        ax2.set_ylabel('Velocity Error [mm/s]', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        self.save_plot(fig, save_path or self._default_path("tracking_error"))

    def plot_control_effort(self, save_path: str = None):
        """绘制控制打火推力时序与累计 ΔV 消耗并保存"""
        times = self.load_dataset('epochs')
        forces = self.load_dataset('control_forces')
        accumulated_dvs = self.load_dataset('accumulated_dvs')
        days = times / 86400.0

        fig, axes = self.create_figure(2, 1, f"{self.mission_name} - Control Effort & Fuel")
        ax1, ax2 = axes

        ax1.plot(days, forces[:, 0], label='Force X', alpha=0.7)
        ax1.plot(days, forces[:, 1], label='Force Y', alpha=0.7)
        ax1.plot(days, forces[:, 2], label='Force Z', alpha=0.7)
        ax1.set_ylabel('Control Force [N]', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(days, accumulated_dvs, color='firebrick', linewidth=2)
        ax2.set_xlabel('Time [Days]', fontweight='bold')
        ax2.set_ylabel(r'Accumulated $\Delta V$ [m/s]', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        final_dv = accumulated_dvs[-1]
        ax2.annotate(f'Total: {final_dv:.4f} m/s',
                     xy=(days[-1], final_dv), xytext=(days[-1]*0.8, final_dv*0.8),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=10, fontweight='bold')

        self.save_plot(fig, save_path or self._default_path("control_effort"))

    # ====================== 新增分析图表 ======================
    def plot_error_histogram(self, save_path: str = None, bins: int = 30):
        """绘制位置误差和速度误差的直方图（用于评估稳态精度分布）"""
        errors = self.load_dataset('tracking_errors')
        pos_err = np.linalg.norm(errors[:, 0:3], axis=1)
        vel_err = np.linalg.norm(errors[:, 3:6], axis=1) * 1000  # mm/s

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{self.mission_name} - Error Distribution", fontsize=14, fontweight='bold')

        ax1.hist(pos_err, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Position Error [m]')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Position Error Distribution')
        ax1.axvline(np.median(pos_err), color='red', linestyle='--', label=f'Median: {np.median(pos_err):.2f} m')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(vel_err, bins=bins, alpha=0.7, color='darkorange', edgecolor='black')
        ax2.set_xlabel('Velocity Error [mm/s]')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Velocity Error Distribution')
        ax2.axvline(np.median(vel_err), color='red', linestyle='--', label=f'Median: {np.median(vel_err):.2f} mm/s')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        self.save_plot(fig, save_path or self._default_path("error_histogram"))

    def plot_force_spectrum(self, save_path: str = None, fs: float = None):
        """绘制控制力功率谱密度（分析推力频率成分）"""
        times = self.load_dataset('epochs')
        forces = self.load_dataset('control_forces')
        if len(forces) < 10:
            print("警告：控制力数据点不足，跳过频谱分析")
            return

        if fs is None:
            dt = np.mean(np.diff(times))
            fs = 1.0 / dt

        fig, axes = self.create_figure(3, 1, f"{self.mission_name} - Control Force Spectrum", figsize=(12, 10))
        labels = ['Fx', 'Fy', 'Fz']
        for i, ax in enumerate(axes):
            f, Pxx = welch(forces[:, i], fs=fs, nperseg=min(256, len(forces)//4))
            ax.semilogy(f, Pxx)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'{labels[i]} Spectrum')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, fs/2])

        self.save_plot(fig, save_path or self._default_path("force_spectrum"))

    def plot_state_history(self, save_path: str = None):
        """绘制航天器绝对状态（位置、速度）随时间变化曲线"""
        times = self.load_dataset('epochs')
        true_states = self.load_dataset('true_states')
        days = times / 86400.0

        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle(f"{self.mission_name} - State History", fontsize=14, fontweight='bold')

        labels_pos = ['X [m]', 'Y [m]', 'Z [m]']
        labels_vel = ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]']

        for i in range(3):
            axes[i, 0].plot(days, true_states[:, i], label='True', color='b')
            axes[i, 0].set_ylabel(labels_pos[i])
            axes[i, 0].grid(True, alpha=0.3)

            axes[i, 1].plot(days, true_states[:, i+3], label='True', color='g')
            axes[i, 1].set_ylabel(labels_vel[i])
            axes[i, 1].grid(True, alpha=0.3)

        axes[2, 0].set_xlabel('Time [Days]')
        axes[2, 1].set_xlabel('Time [Days]')

        self.save_plot(fig, save_path or self._default_path("state_history"))

    def plot_thrust_activity(self, save_path: str = None, thrust_threshold: float = 1e-3):
        """绘制推力活动时序，并标注推力脉冲"""
        times = self.load_dataset('epochs')
        forces = self.load_dataset('control_forces')
        days = times / 86400.0

        force_mag = np.linalg.norm(forces, axis=1)
        thrust_active = force_mag > thrust_threshold

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{self.mission_name} - Thrust Activity", fontsize=14, fontweight='bold')

        ax1.plot(days, force_mag, color='purple', alpha=0.7)
        ax1.fill_between(days, 0, force_mag, where=thrust_active, color='red', alpha=0.3, label='Thrusting')
        ax1.set_ylabel('Thrust Magnitude [N]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.stem(days[thrust_active], force_mag[thrust_active], linefmt='r-', markerfmt='ro', basefmt='k-', label='Firing events')
        ax2.set_xlabel('Time [Days]')
        ax2.set_ylabel('Thrust [N]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        n_pulses = np.sum(thrust_active)
        if n_pulses > 0:
            ax2.set_title(f'Total pulses: {n_pulses}, Mean thrust: {np.mean(force_mag[thrust_active]):.3f} N')

        self.save_plot(fig, save_path or self._default_path("thrust_activity"))

    # ====================== 多文件对比 ======================
    def compare_simulations(self, other_files: List[str], labels: List[str], save_path: str = None):
        """比较多个仿真结果"""
        current_data = self._load_all_data()
        all_data = [current_data] + [self._load_all_data_from_file(f) for f in other_files]
        all_labels = [self.mission_name] + labels

        # 对比误差曲线
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"Simulation Comparison: Position Error", fontsize=14, fontweight='bold')

        for data, label in zip(all_data, all_labels):
            times = data['epochs'] / 86400.0
            pos_err = np.linalg.norm(data['tracking_errors'][:, 0:3], axis=1)
            axes[0].plot(times, pos_err, label=label, alpha=0.7)
            vel_err = np.linalg.norm(data['tracking_errors'][:, 3:6], axis=1) * 1000
            axes[1].plot(times, vel_err, label=label, alpha=0.7)

        axes[0].set_ylabel('Position Error [m]')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel('Velocity Error [mm/s]')
        axes[1].set_xlabel('Time [Days]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        self.save_plot(fig, save_path or "comparison_error.png")

        # 对比 ΔV 累计
        fig2, ax = plt.subplots(figsize=(10, 6))
        for data, label in zip(all_data, all_labels):
            times = data['epochs'] / 86400.0
            ax.plot(times, data['accumulated_dvs'], label=label)
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Accumulated ΔV [m/s]')
        ax.set_title('Fuel Consumption Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig2, save_path or "comparison_fuel.png")

    # ====================== 报告生成 ======================
    def generate_report(self, output_dir: str = "reports", report_name: str = None):
        """生成完整的 HTML 报告"""
        import matplotlib
        matplotlib.use('Agg')

        if report_name is None:
            report_name = f"{self.mission_name.replace(' ', '_')}_{self.mission_name}.html"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, report_name)

        figs = {}
        try:
            figs['trajectory'] = self._plot_and_return_fig(self.plot_3d_trajectory, save=False)
        except Exception as e:
            print(f"轨迹图生成失败: {e}")
        try:
            figs['error'] = self._plot_and_return_fig(self.plot_tracking_error, save=False)
        except Exception as e:
            print(f"误差图生成失败: {e}")
        try:
            figs['control'] = self._plot_and_return_fig(self.plot_control_effort, save=False)
        except Exception as e:
            print(f"控制图生成失败: {e}")
        try:
            figs['histogram'] = self._plot_and_return_fig(self.plot_error_histogram, save=False)
        except Exception as e:
            print(f"直方图生成失败: {e}")
        try:
            figs['thrust_activity'] = self._plot_and_return_fig(self.plot_thrust_activity, save=False)
        except Exception as e:
            print(f"推力活动图生成失败: {e}")

        errors = self.load_dataset('tracking_errors')
        pos_err = np.linalg.norm(errors[:, 0:3], axis=1)
        vel_err = np.linalg.norm(errors[:, 3:6], axis=1) * 1000
        dv = self.load_dataset('accumulated_dvs')[-1]
        sim_days = self.config.get('simulation_days', 1)

        with open(report_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{self.mission_name} - L1 Simulation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    h1 {{ color: #2c3e50; }}
                    .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2c3e50; color: white; }}
                    .footer {{ text-align: center; font-size: 0.8em; color: #777; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <h1>{self.mission_name} - L1 级仿真报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="section">
                    <h2>仿真概览</h2>
                    <table>
                        <tr><th>参数</th><th>值</th></tr>
                        <tr><td>仿真时长 (天)</td><td>{sim_days}</td></tr>
                        <tr><td>积分步长 (秒)</td><td>{self.config.get('time_step', 'N/A')}</td></tr>
                        <tr><td>总 ΔV 消耗 (m/s)</td><td>{dv:.4f}</td></tr>
                        <tr><td>平均每天 ΔV (m/s/天)</td><td>{dv / sim_days:.4f}</td></tr>
                        <tr><td>最终位置误差 (m)</td><td>{pos_err[-1]:.2f}</td></tr>
                        <tr><td>最终速度误差 (mm/s)</td><td>{vel_err[-1]:.2f}</td></tr>
                    </table>
                </div>
                <div class="section"><h2>3D 轨迹对比</h2>{self._fig_to_base64(figs.get('trajectory'))}</div>
                <div class="section"><h2>跟踪误差曲线</h2>{self._fig_to_base64(figs.get('error'))}</div>
                <div class="section"><h2>控制力与燃料消耗</h2>{self._fig_to_base64(figs.get('control'))}</div>
                <div class="section"><h2>误差分布直方图</h2>{self._fig_to_base64(figs.get('histogram'))}</div>
                <div class="section"><h2>推力活动时序</h2>{self._fig_to_base64(figs.get('thrust_activity'))}</div>
                <div class="footer">MCPC 框架 L1 级仿真报告 - 自动生成</div>
            </body>
            </html>
            """)
        print(f"报告已保存至: {report_path}")

    # ====================== 辅助方法 ======================
    def _default_path(self, suffix: str) -> str:
        return f"data/{self.mission_name.replace(' ', '_')}_{suffix}.png"

    def _load_all_data(self) -> dict:
        data = {}
        for key in ['epochs', 'tracking_errors', 'accumulated_dvs', 'control_forces', 'true_states', 'nominal_states']:
            try:
                data[key] = self.load_dataset(key)
            except KeyError:
                data[key] = np.array([])
        return data

    def _load_all_data_from_file(self, filepath: str) -> dict:
        data = {}
        try:
            with h5py.File(filepath, 'r') as f:
                for key in ['epochs', 'tracking_errors', 'accumulated_dvs', 'control_forces', 'true_states', 'nominal_states']:
                    if key in f:
                        data[key] = f[key][()]
                    else:
                        data[key] = np.array([])
        except Exception as e:
            print(f"加载文件 {filepath} 失败: {e}")
            data = {k: np.array([]) for k in ['epochs', 'tracking_errors', 'accumulated_dvs', 'control_forces', 'true_states', 'nominal_states']}
        return data

    def _plot_and_return_fig(self, plot_func, save=False, **kwargs):
        original_save = self.save_plot
        def dummy_save(fig, path):
            pass
        self.save_plot = dummy_save
        try:
            plot_func(**kwargs)
            fig = plt.gcf()
            plt.close(fig)
            return fig
        except Exception as e:
            print(f"绘图失败: {e}")
            return None
        finally:
            self.save_plot = original_save

    def _fig_to_base64(self, fig):
        if fig is None:
            return "<p>图表生成失败</p>"
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" />'
