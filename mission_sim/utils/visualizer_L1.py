# mission_sim/utils/visualizer_L1.py
import h5py
import numpy as np
from mission_sim.utils.visualizer import BaseVisualizer

class L1Visualizer(BaseVisualizer):
    def plot_absolute_convergence(self, sc_id, save_path):
        with h5py.File(self.filepath, 'r') as f:
            states = f[f"{sc_id}/state"][:]
            # 假设目标是 L2 点 (1.511e11, 0, 0)
            target = np.array([1.511e11, 0, 0])
            
        time = np.arange(len(states)) * self.dt
        errors = states[:, 0:3] - target
        
        fig, axes = self.create_figure(3, 1, f"L1: {sc_id} Absolute Orbit Convergence")
        labels = ['X (Radial) [m]', 'Y (Along-track) [m]', 'Z (Cross-track) [m]']
        
        for i in range(3):
            axes[i].plot(time, errors[:, i], color='tab:red')
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, ls='--')
        
        axes[2].set_xlabel("Time [s]")
        self.save_plot(fig, save_path)