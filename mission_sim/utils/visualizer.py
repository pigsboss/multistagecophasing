# mission_sim/utils/visualizer.py
import h5py
import numpy as np
import matplotlib.pyplot as plt

class BaseVisualizer:
    """
    仿真可视化基类：定义通用的数据提取逻辑与绘图风格。
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._load_metadata()

    def _load_metadata(self):
        """从 HDF5 中提取通用的仿真配置参数"""
        try:
            with h5py.File(self.filepath, 'r') as f:
                # 尝试从不同的可能位置获取 dt
                if 'simulation_info' in f:
                    self.dt = float(f["simulation_info"]["dt"][()])
                elif 'metadata/sim_config' in f:
                    self.dt = float(f["metadata/sim_config"].attrs["dt"])
                else:
                    self.dt = 0.1 # 默认兜底
        except (FileNotFoundError, KeyError):
            self.dt = 0.1

    def create_figure(self, rows, cols, title, figsize=(10, 8)):
        """统一的绘图风格工厂"""
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        return fig, axes

    def save_plot(self, plt_obj, save_path):
        """统一保存逻辑"""
        plt_obj.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt_obj.savefig(save_path, dpi=300)
        
        # 修正：plt_obj 是 Figure 对象，应该传给 plt.close()
        plt.close(plt_obj) 
        
        print(f"📊 Visualization saved: {save_path}")