# mission_sim/utils/visualizer_L2.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mission_sim.utils.visualizer import BaseVisualizer

class L2Visualizer(BaseVisualizer):
    """
    L2 级编队可视化插件：专注相对运动、基线保持与协同导航评估。
    """
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self._load_formation_data()

    def _load_formation_data(self):
        """从 HDF5 中提取 L2 特有的编队数据集"""
        with h5py.File(self.filepath, 'r') as f:
            # 核心数据集：从星在 LVLH 系下的 6D 相对状态
            self.rel_states = f["Formation/rel_state_lvlh"][:]
            
            # 目标状态提取（逻辑自洽：必须减去目标才是误差）
            if "metadata/targets/rel_deputy" in f:
                self.target_rel = f["metadata/targets/rel_deputy"][:]
            else:
                # 兼容性兜底：假设目标是 Y 轴 100m
                self.target_rel = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
                
        self.time = np.arange(len(self.rel_states)) * self.dt

    def plot_formation_convergence(self, save_path="data/L2_relative_error.png"):
        """1. 绘制 LVLH 系下的相对位置误差收敛曲线 (评价 LQR 阻尼与稳态精度)"""
        # 计算相对位置偏差 (3D)
        pos_errors = self.rel_states[:, 0:3] - self.target_rel[0:3]
        
        fig, axes = self.create_figure(3, 1, "L2 Formation: Relative Position Error (LVLH)")
        labels = ['Radial (X) [m]', 'Along-track (Y) [m]', 'Cross-track (Z) [m]']
        colors = ['#e74c3c', '#2ecc71', '#3498db'] # 严谨的 RGB 对应 XYZ

        for i in range(3):
            axes[i].plot(self.time, pos_errors[:, i], color=colors[i], lw=1.5, label=f'Error {labels[i].split()[0]}')
            axes[i].axhline(0, color='black', lw=1, ls='--', alpha=0.5)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, alpha=0.3, ls=':')
            axes[i].legend(loc='upper right', fontsize='small')
            
        axes[2].set_xlabel("Time [s]")
        self.save_plot(fig, save_path)

    def plot_relative_trajectory_3d(self, save_path="data/L2_3d_view.png"):
        """2. 绘制 3D 相对运动轨迹 (以主星为坐标原点)"""
        rel_pos = self.rel_states[:, 0:3]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制主星 (位于 LVLH 原点)
        ax.scatter(0, 0, 0, color='red', s=100, marker='*', label='Chief (Origin)')
        
        # 绘制从星运动轨迹
        ax.plot(rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2], 
                color='darkblue', lw=2, alpha=0.7, label='Deputy Trajectory')
        
        # 绘制终点目标位置
        ax.scatter(self.target_rel[0], self.target_rel[1], self.target_rel[2], 
                   color='green', marker='X', s=120, label='Target Setpoint')

        # 设置坐标轴标签与风格
        ax.set_xlabel('Radial X [m]')
        ax.set_ylabel('Along-track Y [m]')
        ax.set_zlabel('Cross-track Z [m]')
        ax.set_title("L2 Multi-Spacecraft Formation Geometry\n(Relative to Chief in LVLH Frame)", fontsize=12)
        ax.legend()
        
        # 优化视觉效果：确保比例一致
        max_range = np.array([rel_pos[:,0].max()-rel_pos[:,0].min(), 
                              rel_pos[:,1].max()-rel_pos[:,1].min(), 
                              rel_pos[:,2].max()-rel_pos[:,2].min()]).max() / 2.0
        mid_x = (rel_pos[:,0].max()+rel_pos[:,0].min()) * 0.5
        mid_y = (rel_pos[:,1].max()+rel_pos[:,1].min()) * 0.5
        mid_z = (rel_pos[:,2].max()+rel_pos[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"📊 3D Relative plot saved: {save_path}")
        plt.close()