# analysis/control_robustness_analysis.py
"""
控制算法鲁棒性蒙特卡洛分析
通过改变初始偏差、测控噪声、模型误差等参数，统计控制收敛时间、稳态误差、燃料消耗等指标。
输出统计图表和性能报告。
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mission_sim.main_L1_runner import L1MissionSimulation
from mission_sim.utils.logger import HDF5Logger


@dataclass
class RobustnessMetrics:
    """单次仿真性能指标"""
    mission_id: str
    position_error_final: float      # 最终位置误差 (m)
    velocity_error_final: float      # 最终速度误差 (m/s)
    accumulated_dv: float            # 总 ΔV (m/s)
    max_control_force: float         # 最大控制力 (N)
    rms_position_error: float        # 位置误差 RMS (m)
    rms_velocity_error: float        # 速度误差 RMS (m/s)
    convergence_time: float          # 收敛时间 (s) 定义为误差降至阈值以下
    simulation_time: float           # 实际仿真耗时 (s)
    parameters: Dict[str, Any] = field(default_factory=dict)


class ControlRobustnessAnalyzer:
    """
    控制鲁棒性分析器
    运行蒙特卡洛仿真，收集性能指标，生成统计报告和图表。
    """

    # 默认仿真配置
    DEFAULT_CONFIG = {
        "mission_name": "Robustness_Analysis",
        "simulation_days": 30,
        "time_step": 10.0,
        "log_buffer_size": 500,
        "log_compression": True,
        "enable_visualization": False,   # 分析时不生成图表，加快速度
        "data_dir": "data/robustness_analysis",
        "log_level": "WARNING"
    }

    # 默认参数变化范围
    DEFAULT_PARAM_VARY = {
        "initial_pos_error_scale": [0.5, 1.0, 2.0],      # 初始位置误差倍数 (基准 2000m)
        "initial_vel_error_scale": [0.5, 1.0, 2.0],      # 初始速度误差倍数 (基准 0.01m/s)
        "pos_noise_std": [2.5, 5.0, 10.0],               # 位置噪声标准差 (m)
        "vel_noise_std": [0.0025, 0.005, 0.01],          # 速度噪声标准差 (m/s)
        "control_gain_scale": [0.5, 1.0, 2.0]            # 控制增益缩放因子
    }

    def __init__(self,
                 base_config: Optional[Dict] = None,
                 param_vary: Optional[Dict] = None,
                 output_dir: str = "analysis_results",
                 n_runs: int = 10):
        """
        初始化分析器。

        Args:
            base_config: 基础仿真配置 (覆盖默认)
            param_vary: 参数变化字典，键为参数名，值为列表或数组
            output_dir: 输出目录
            n_runs: 每个参数组合的重复运行次数（用于统计稳定性）
        """
        self.base_config = {**self.DEFAULT_CONFIG, **(base_config or {})}
        self.param_vary = param_vary or self.DEFAULT_PARAM_VARY
        self.output_dir = output_dir
        self.n_runs = n_runs

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 存储所有指标
        self.metrics_list: List[RobustnessMetrics] = []

        print(f"[Analyzer] 初始化完成，输出目录: {output_dir}")
        print(f"[Analyzer] 参数组合: {list(self.param_vary.keys())}")
        print(f"[Analyzer] 每个组合运行次数: {n_runs}")

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据参数构建仿真配置。
        """
        config = self.base_config.copy()

        # 复制参数到配置
        for key, value in params.items():
            config[key] = value

        return config

    def _run_single_simulation(self, config: Dict[str, Any]) -> Optional[RobustnessMetrics]:
        """
        运行单次仿真，返回性能指标。
        """
        try:
            sim = L1MissionSimulation(config)
            start_time = time.time()
            success = sim.run()
            elapsed = time.time() - start_time

            if not success:
                print(f"  [警告] 仿真失败，跳过")
                return None

            # 从仿真对象获取最终指标
            stats = sim.get_statistics()
            final_pos_err = stats.get("final_position_error", np.nan)
            final_vel_err = stats.get("final_velocity_error", np.nan)
            total_dv = stats.get("accumulated_dv", np.nan)

            # 从 HDF5 文件加载更多指标
            if os.path.exists(sim.h5_file):
                try:
                    logger = HDF5Logger(sim.h5_file)
                    # 读取所有数据
                    data = logger.load_all_data()
                    # 位置误差 RMS
                    errors = data.get('tracking_errors', np.array([]))
                    if len(errors) > 0:
                        rms_pos = np.sqrt(np.mean(errors[:, 0:3]**2))
                        rms_vel = np.sqrt(np.mean(errors[:, 3:6]**2))
                        # 收敛时间：误差首次降至阈值以下 (位置误差 < 1000m)
                        threshold = 1000.0
                        times = data.get('epochs', np.array([]))
                        idx = np.where(np.linalg.norm(errors[:, 0:3], axis=1) < threshold)[0]
                        conv_time = times[idx[0]] if len(idx) > 0 else np.nan
                    else:
                        rms_pos = rms_vel = conv_time = np.nan
                except Exception as e:
                    print(f"  [警告] 读取 HDF5 数据失败: {e}")
                    rms_pos = rms_vel = conv_time = np.nan
                finally:
                    logger.close()
            else:
                rms_pos = rms_vel = conv_time = np.nan

            # 提取控制力最大值（从 HDF5）
            max_force = np.nan
            if os.path.exists(sim.h5_file):
                try:
                    logger = HDF5Logger(sim.h5_file)
                    forces = logger.load_data('control_forces')
                    if len(forces) > 0:
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                    logger.close()
                except:
                    pass

            metrics = RobustnessMetrics(
                mission_id=sim.mission_id,
                position_error_final=final_pos_err,
                velocity_error_final=final_vel_err,
                accumulated_dv=total_dv,
                max_control_force=max_force,
                rms_position_error=rms_pos,
                rms_velocity_error=rms_vel,
                convergence_time=conv_time,
                simulation_time=elapsed,
                parameters=config
            )
            return metrics

        except Exception as e:
            print(f"  [错误] 仿真异常: {e}")
            return None

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        生成所有参数组合（笛卡尔积）。
        """
        import itertools
        keys = list(self.param_vary.keys())
        values = [self.param_vary[k] for k in keys]
        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            # 对于每个参数组合，重复 n_runs 次
            for _ in range(self.n_runs):
                combinations.append(params.copy())
        return combinations

    def run_monte_carlo(self):
        """
        执行蒙特卡洛仿真。
        """
        param_combos = self._generate_param_combinations()
        total_runs = len(param_combos)
        print(f"\n[Analyzer] 开始蒙特卡洛仿真，总运行次数: {total_runs}")

        # 使用进度条
        for i, params in enumerate(tqdm(param_combos, desc="运行仿真", unit="次")):
            # 构建配置
            config = self._build_config(params)
            # 运行仿真
            metrics = self._run_single_simulation(config)
            if metrics is not None:
                self.metrics_list.append(metrics)
            # 可选：定期保存中间结果
            if (i+1) % 50 == 0:
                self._save_intermediate_results()

        # 保存最终结果
        self._save_intermediate_results(final=True)
        print(f"\n[Analyzer] 仿真完成，成功运行次数: {len(self.metrics_list)}/{total_runs}")

    def _save_intermediate_results(self, final: bool = False):
        """
        保存当前结果到文件。
        """
        import json
        if not self.metrics_list:
            return
        filename = "robustness_results_final.json" if final else "robustness_results_partial.json"
        filepath = os.path.join(self.output_dir, filename)
        data = [asdict(m) for m in self.metrics_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Analyzer] 已保存 {len(self.metrics_list)} 条结果到 {filepath}")

    def plot_results(self):
        """
        生成统计图表。
        """
        if not self.metrics_list:
            print("[Analyzer] 无数据，无法绘图")
            return

        # 提取关键指标
        metrics = self.metrics_list
        # 将参数组合按主要变化维度分组，这里简化：按初始误差缩放因子分组
        pos_err_scales = []
        vel_err_scales = []
        final_pos_err = []
        final_vel_err = []
        total_dv = []
        max_force = []

        for m in metrics:
            # 获取参数
            pos_scale = m.parameters.get("initial_pos_error_scale", 1.0)
            vel_scale = m.parameters.get("initial_vel_error_scale", 1.0)
            pos_err_scales.append(pos_scale)
            vel_err_scales.append(vel_scale)
            final_pos_err.append(m.position_error_final)
            final_vel_err.append(m.velocity_error_final)
            total_dv.append(m.accumulated_dv)
            max_force.append(m.max_control_force)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("控制算法鲁棒性分析", fontsize=14, fontweight='bold')

        # 1. 最终位置误差 vs 初始位置误差缩放因子
        ax = axes[0, 0]
        unique_scales = sorted(set(pos_err_scales))
        pos_err_by_scale = [np.array([final_pos_err[i] for i, s in enumerate(pos_err_scales) if s == sc]) for sc in unique_scales]
        bp = ax.boxplot(pos_err_by_scale, positions=unique_scales, widths=0.6, patch_artist=True)
        ax.set_xlabel('初始位置误差缩放因子')
        ax.set_ylabel('最终位置误差 (m)')
        ax.set_title('最终位置误差 vs 初始位置误差')
        ax.grid(True, alpha=0.3)

        # 2. 总 ΔV vs 初始速度误差缩放因子
        ax = axes[0, 1]
        unique_vel_scales = sorted(set(vel_err_scales))
        dv_by_vel = [np.array([total_dv[i] for i, s in enumerate(vel_err_scales) if s == sc]) for sc in unique_vel_scales]
        bp = ax.boxplot(dv_by_vel, positions=unique_vel_scales, widths=0.6, patch_artist=True)
        ax.set_xlabel('初始速度误差缩放因子')
        ax.set_ylabel('总 ΔV (m/s)')
        ax.set_title('燃料消耗 vs 初始速度误差')
        ax.grid(True, alpha=0.3)

        # 3. 最终位置误差直方图
        ax = axes[1, 0]
        ax.hist(final_pos_err, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('最终位置误差 (m)')
        ax.set_ylabel('频数')
        ax.set_title('最终位置误差分布')
        ax.grid(True, alpha=0.3)

        # 4. 总 ΔV 直方图
        ax = axes[1, 1]
        ax.hist(total_dv, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
        ax.set_xlabel('总 ΔV (m/s)')
        ax.set_ylabel('频数')
        ax.set_title('燃料消耗分布')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "control_robustness_plots.png")
        plt.savefig(save_path, dpi=150)
        print(f"[Analyzer] 图表已保存至 {save_path}")
        plt.show()

    def generate_report(self):
        """
        生成统计报告。
        """
        if not self.metrics_list:
            print("[Analyzer] 无数据，无法生成报告")
            return

        # 提取指标
        final_pos_err = np.array([m.position_error_final for m in self.metrics_list if not np.isnan(m.position_error_final)])
        final_vel_err = np.array([m.velocity_error_final for m in self.metrics_list if not np.isnan(m.velocity_error_final)])
        total_dv = np.array([m.accumulated_dv for m in self.metrics_list if not np.isnan(m.accumulated_dv)])
        max_force = np.array([m.max_control_force for m in self.metrics_list if not np.isnan(m.max_control_force)])
        rms_pos = np.array([m.rms_position_error for m in self.metrics_list if not np.isnan(m.rms_position_error)])
        conv_time = np.array([m.convergence_time for m in self.metrics_list if not np.isnan(m.convergence_time)])

        report = []
        report.append("=" * 60)
        report.append("控制算法鲁棒性分析报告")
        report.append("=" * 60)
        report.append(f"总运行次数: {len(self.metrics_list)}")
        report.append(f"有效数据点: {len(final_pos_err)}")
        report.append("")

        # 统计描述
        report.append("--- 最终位置误差 (m) ---")
        report.append(f"  均值: {np.mean(final_pos_err):.2f}")
        report.append(f"  标准差: {np.std(final_pos_err):.2f}")
        report.append(f"  最小值: {np.min(final_pos_err):.2f}")
        report.append(f"  最大值: {np.max(final_pos_err):.2f}")
        report.append(f"  中位数: {np.median(final_pos_err):.2f}")
        report.append("")

        report.append("--- 最终速度误差 (m/s) ---")
        report.append(f"  均值: {np.mean(final_vel_err):.4f}")
        report.append(f"  标准差: {np.std(final_vel_err):.4f}")
        report.append("")

        report.append("--- 总 ΔV (m/s) ---")
        report.append(f"  均值: {np.mean(total_dv):.4f}")
        report.append(f"  标准差: {np.std(total_dv):.4f}")
        report.append(f"  最小值: {np.min(total_dv):.4f}")
        report.append(f"  最大值: {np.max(total_dv):.4f}")
        report.append("")

        report.append("--- 最大控制力 (N) ---")
        report.append(f"  均值: {np.mean(max_force):.2f}")
        report.append(f"  最大值: {np.max(max_force):.2f}")
        report.append("")

        report.append("--- RMS 位置误差 (m) ---")
        report.append(f"  均值: {np.mean(rms_pos):.2f}")
        report.append(f"  标准差: {np.std(rms_pos):.2f}")
        report.append("")

        report.append("--- 收敛时间 (s) ---")
        report.append(f"  均值: {np.mean(conv_time)/86400:.2f} 天")
        report.append(f"  标准差: {np.std(conv_time)/86400:.2f} 天")
        report.append("")

        report.append("=" * 60)

        report_str = "\n".join(report)
        print(report_str)

        # 保存报告
        report_path = os.path.join(self.output_dir, "robustness_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        print(f"[Analyzer] 报告已保存至 {report_path}")


def main():
    """
    示例运行。
    """
    # 自定义参数变化范围（可修改）
    param_vary = {
        "initial_pos_error_scale": [0.5, 1.0, 2.0],      # 初始位置误差倍数
        "initial_vel_error_scale": [0.5, 1.0, 2.0],      # 初始速度误差倍数
        "pos_noise_std": [2.5, 5.0, 10.0],               # 位置噪声标准差
        "vel_noise_std": [0.0025, 0.005, 0.01],          # 速度噪声标准差
        # "control_gain_scale": [0.5, 1.0, 2.0]           # 可选
    }

    # 基础配置（覆盖默认）
    base_config = {
        "simulation_days": 10,            # 缩短仿真时间加快分析
        "time_step": 10.0,
        "log_buffer_size": 100,
        "enable_visualization": False,
        "data_dir": "data/robustness_analysis"
    }

    # 创建分析器
    analyzer = ControlRobustnessAnalyzer(
        base_config=base_config,
        param_vary=param_vary,
        output_dir="analysis_results",
        n_runs=3          # 每个参数组合运行3次，实际可增加到10以上
    )

    # 运行蒙特卡洛
    analyzer.run_monte_carlo()

    # 生成图表和报告
    analyzer.plot_results()
    analyzer.generate_report()


if __name__ == "__main__":
    main()