# mission_sim/utils/visualizer.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class L1Visualizer:
    """
    L1 级仿真数据离线可视化引擎
    严格按照 L1 架构契约，提供状态历史、GNC 作动和 3D 动画渲染。
    """
    def __init__(self, filepath: str, sc_id: str):
        self.filepath = filepath
        self.sc_id = sc_id
        
        # 预加载数据，避免重复读取
        self._load_data()

    def _load_data(self):
        """内部方法：从 HDF5 中提取所有可用数据"""
        try:
            with h5py.File(self.filepath, 'r') as f:
                # 数组型 Dataset 直接用 np.array() 包裹会隐式读取，这没问题
                self.states = np.array(f["Spacecraft"]["state"])
                self.thrusts = np.array(f["GNC"]["thrust_cmd"])
                
                try:
                    # 【核心修复】：加上 [()] 来显式提取 Dataset 里的真实数值
                    self.target_state = np.array(f["simulation_info"]["target_state"][()])
                    self.dt = float(f["simulation_info"]["dt"][()])
                except KeyError:
                    self.target_state = np.array([1.511e11, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.dt = 0.1
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到数据文件: {self.filepath}。请确保仿真已成功运行并落盘。")

    def plot_state_history(self, save_path: str):
        """1. 绘制三轴位置误差随时间的收敛曲线"""
        time_axis = np.arange(self.states.shape[0]) * self.dt
        errors = self.states - self.target_state
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"L1 Orbit Maintenance Convergence (LQR)\nSpacecraft: {self.sc_id}", fontsize=14)
        
        labels = ["X Error (Radial)", "Y Error (Along-track)", "Z Error (Cross-track)"]
        colors = ['r', 'g', 'b']
        
        for i in range(3):
            axs[i].plot(time_axis, errors[:, i], color=colors[i], label=labels[i])
            axs[i].set_ylabel("Error [m]")
            axs[i].grid(True, linestyle='--')
            axs[i].legend(loc="upper right")
            
        axs[2].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[{self.sc_id} Visualizer] State history saved to {save_path}")
        plt.close()

    def plot_gnc_activity(self, save_path: str):
        """2. 绘制 GNC 控制律输出的推力历史 (评价 LQR 控制代价)"""
        time_axis = np.arange(self.thrusts.shape[0]) * self.dt
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"GNC Thrust Command Activity\nSpacecraft: {self.sc_id}", fontsize=14)
        
        labels = ["Thrust X [N]", "Thrust Y [N]", "Thrust Z [N]"]
        colors = ['darkred', 'darkgreen', 'darkblue']
        
        for i in range(3):
            axs[i].plot(time_axis, self.thrusts[:, i], color=colors[i], label=labels[i], alpha=0.8)
            axs[i].set_ylabel("Force [N]")
            axs[i].grid(True, linestyle='--')
            axs[i].legend(loc="upper right")
            
        axs[2].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[{self.sc_id} Visualizer] GNC activity saved to {save_path}")
        plt.close()

    def create_animation(self, save_path: str, downsample: int = 100, thrust_scale: float = 1.0):
        """3. 渲染航天器在 L2 点附近的三维收敛轨迹动画"""
        print(f"[{self.sc_id} Visualizer] Rendering 3D animation (this may take a moment)...")
        
        # 降采样以加速渲染 (20000 步全部渲染太慢)
        states_downsampled = self.states[::downsample]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 坐标系预处理 (减去目标位置，显示相对偏差)
        rel_pos = states_downsampled[:, 0:3] - self.target_state[0:3]
        
        # 动态计算坐标轴范围，确保轨迹始终在视野内
        max_range = np.max(np.abs(rel_pos)) * 1.1
        if max_range < 1.0: max_range = 10.0 # 避免完全收敛后坐标轴塌陷
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X Offset [m]')
        ax.set_ylabel('Y Offset [m]')
        ax.set_zlabel('Z Offset [m]')
        ax.set_title(f"L2 Orbit Maintenance 3D Trajectory\n{self.sc_id}")
        
        # 绘制目标点 (L2 理论位置)
        ax.scatter([0], [0], [0], color='black', marker='*', s=100, label="Target (L2)")
        
        # 初始化航天器标记和尾迹
        sc_marker, = ax.plot([], [], [], 'ro', markersize=6, label="Spacecraft")
        trail, = ax.plot([], [], [], 'r-', alpha=0.5, linewidth=1)
        
        ax.legend()

        def update(frame):
            # 更新轨迹尾巴 (从起点画到当前帧)
            trail.set_data(rel_pos[:frame, 0], rel_pos[:frame, 1])
            trail.set_3d_properties(rel_pos[:frame, 2])
            
            # 更新航天器当前位置
            sc_marker.set_data([rel_pos[frame, 0]], [rel_pos[frame, 1]])
            sc_marker.set_3d_properties([rel_pos[frame, 2]])
            return trail, sc_marker

        ani = FuncAnimation(fig, update, frames=len(rel_pos), interval=50, blit=False)
        
        # 默认保存为 gif (无需额外安装 ffmpeg)
        writer = PillowWriter(fps=20)
        ani.save(save_path, writer=writer)
        print(f"[{self.sc_id} Visualizer] Animation successfully saved to {save_path}")
        plt.close()

