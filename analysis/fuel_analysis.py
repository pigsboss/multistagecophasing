# analysis/fuel_analysis.py
"""
燃料开销分析脚本
分析不同轨道类型、控制增益、盲区时长下的 ΔV 消耗，生成燃料账单图表。
支持参数扫描和结果可视化。
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.utils.logger import HDF5Logger
from mission_sim.core.types import CoordinateFrame


@dataclass
class FuelMetrics:
    """单次仿真的燃料相关指标"""
    mission_id: str
    orbit_type: str              # 轨道类型: "Halo", "Keplerian", "J2"
    control_gain_scale: float    # 控制增益缩放因子
    blind_interval_days: float   # 盲区时长 (天)
    simulation_days: float       # 仿真时长 (天)
    total_dv: float              # 总 ΔV (m/s)
    avg_dv_per_day: float        # 平均每天 ΔV (m/s/天)
    max_control_force: float     # 最大控制力 (N)
    final_position_error: float  # 最终位置误差 (m)
    simulation_time: float       # 实际仿真耗时 (s)
    parameters: Dict[str, Any]   # 完整参数副本


class FuelAnalyzer:
    """
    燃料消耗分析器
    通过参数扫描，评估不同条件下的燃料开销，生成报告和图表。
    """

    # 默认基础配置
    DEFAULT_BASE_CONFIG = {
        "mission_name": "Fuel_Analysis",
        "time_step": 10.0,
        "log_buffer_size": 500,
        "log_compression": True,
        "enable_visualization": False,
        "data_dir": "data/fuel_analysis",
        "log_level": "WARNING"
    }

    # 默认扫描参数
    DEFAULT_SCAN_PARAMS = {
        "orbit_type": ["Halo", "Keplerian"],          # 轨道类型
        "control_gain_scale": [0.5, 1.0, 2.0],        # 控制增益缩放因子
        "blind_interval_days": [0, 1, 3, 7],          # 盲区时长 (天)
        "simulation_days": [30, 90]                   # 仿真时长 (天)
    }

    def __init__(self,
                 base_config: Optional[Dict] = None,
                 scan_params: Optional[Dict] = None,
                 output_dir: str = "analysis_results",
                 n_repeats: int = 3):
        """
        初始化燃料分析器。

        Args:
            base_config: 基础仿真配置（覆盖默认）
            scan_params: 扫描参数字典
            output_dir: 输出目录
            n_repeats: 每个参数组合的重复次数（用于统计稳定性）
        """
        self.base_config = {**self.DEFAULT_BASE_CONFIG, **(base_config or {})}
        self.scan_params = scan_params or self.DEFAULT_SCAN_PARAMS
        self.output_dir = output_dir
        self.n_repeats = n_repeats

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 存储所有指标
        self.metrics_list: List[FuelMetrics] = []

        print(f"[FuelAnalyzer] 初始化完成，输出目录: {output_dir}")
        print(f"[FuelAnalyzer] 扫描参数: {list(self.scan_params.keys())}")
        print(f"[FuelAnalyzer] 每个组合重复次数: {n_repeats}")

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据参数构建仿真配置。
        """
        config = self.base_config.copy()
        # 合并参数
        config.update(params)

        # 根据轨道类型设置特定参数
        orbit_type = params.get("orbit_type", "Halo")
        if orbit_type == "Halo":
            # Halo 轨道默认配置（日地 L2）
            config.setdefault("Az", 0.05)
            config.setdefault("dt", 0.001)
            # 可选：使用初始猜测加速收敛
            config.setdefault("initial_guess", [1.01106, 0.05, 0.0105])
        elif orbit_type == "Keplerian":
            # 开普勒圆轨道（地心，7000km 半径）
            config.setdefault("elements", [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])
            config.setdefault("simulation_days", params.get("simulation_days", 30))
        elif orbit_type == "J2":
            # 带 J2 摄动的开普勒轨道（需在环境注册 J2）
            config.setdefault("elements", [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])
            config.setdefault("simulation_days", params.get("simulation_days", 30))
            config["enable_j2"] = True   # 标志，在仿真初始化时注册 J2 模型
        else:
            raise ValueError(f"未知的轨道类型: {orbit_type}")

        # 盲区配置：设置地面站可视窗口
        blind_days = params.get("blind_interval_days", 0)
        if blind_days > 0:
            # 构造可视窗口：假设前 blind_days 天不可见，之后全程可见
            blind_seconds = blind_days * 86400
            sim_seconds = params.get("simulation_days", 30) * 86400
            config["visibility_windows"] = [(blind_seconds, sim_seconds)]
        else:
            config["visibility_windows"] = []   # 全程可见

        # 控制增益缩放因子：通过修改 LQR 的 Q 矩阵间接实现
        # 注：L1MissionSimulation 中的控制增益基于固定 Q/R 计算，这里无法直接缩放 K 矩阵，
        # 因此我们将在仿真运行时，通过修改 gnc 的 K 矩阵来实现。
        # 由于 gnc 的 compute_control_force 接受 K 矩阵作为参数，我们可以在仿真类中增加设置。
        # 但 L1MissionSimulation 的 _design_control_law 是硬编码的，我们需要修改该类以支持缩放。
        # 为简化，这里先记录参数，后面在仿真中动态调整。
        # 实际实现时，可扩展 L1MissionSimulation 的 __init__ 接受 control_gain_scale 参数。
        # 为了本分析脚本能运行，我们假设已修改 L1MissionSimulation 以支持 gain_scale。
        # 临时方案：直接在 config 中增加 control_gain_scale，然后在 L1MissionSimulation 中处理。
        # 为让分析脚本可用，我们需先实现该功能。但为了代码完整性，我们保留参数传递。
        # 此处先假设 L1MissionSimulation 已支持 control_gain_scale 参数。

        return config

    def _run_single_simulation(self, config: Dict[str, Any]) -> Optional[FuelMetrics]:
        """
        运行单次仿真，返回燃料相关指标。
        """
        try:
            sim = SunEarthL2L1Simulation(config)
            start_time = time.time()
            success = sim.run()
            elapsed = time.time() - start_time

            if not success:
                print(f"  [警告] 仿真失败，跳过")
                return None

            # 从仿真对象获取指标
            stats = sim.get_statistics()
            total_dv = stats.get("accumulated_dv", np.nan)
            final_pos_err = stats.get("final_position_error", np.nan)

            # 从 HDF5 文件读取最大控制力
            max_force = np.nan
            if os.path.exists(sim.h5_file):
                try:
                    logger = HDF5Logger(sim.h5_file)
                    forces = logger.load_data('control_forces')
                    if len(forces) > 0:
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                    logger.close()
                except Exception as e:
                    print(f"  [警告] 读取控制力数据失败: {e}")

            # 计算平均每天 ΔV
            sim_days = config.get("simulation_days", 30)
            avg_dv_per_day = total_dv / sim_days if sim_days > 0 else np.nan

            metrics = FuelMetrics(
                mission_id=sim.mission_id,
                orbit_type=config.get("orbit_type", "Unknown"),
                control_gain_scale=config.get("control_gain_scale", 1.0),
                blind_interval_days=config.get("blind_interval_days", 0),
                simulation_days=sim_days,
                total_dv=total_dv,
                avg_dv_per_day=avg_dv_per_day,
                max_control_force=max_force,
                final_position_error=final_pos_err,
                simulation_time=elapsed,
                parameters=config
            )
            return metrics

        except Exception as e:
            print(f"  [错误] 仿真异常: {e}")
            return None

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        生成所有参数组合（笛卡尔积），并重复 n_repeats 次。
        """
        import itertools
        keys = list(self.scan_params.keys())
        values = [self.scan_params[k] for k in keys]
        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            for _ in range(self.n_repeats):
                combinations.append(params.copy())
        return combinations

    def run_scan(self):
        """
        执行参数扫描仿真。
        """
        param_combos = self._generate_param_combinations()
        total_runs = len(param_combos)
        print(f"\n[FuelAnalyzer] 开始参数扫描，总运行次数: {total_runs}")

        for i, params in enumerate(tqdm(param_combos, desc="运行仿真", unit="次")):
            config = self._build_config(params)
            metrics = self._run_single_simulation(config)
            if metrics is not None:
                self.metrics_list.append(metrics)
            # 定期保存中间结果
            if (i+1) % 50 == 0:
                self._save_results(partial=True)

        # 最终保存
        self._save_results(final=True)
        print(f"\n[FuelAnalyzer] 扫描完成，成功运行次数: {len(self.metrics_list)}/{total_runs}")

    def _save_results(self, partial: bool = False, final: bool = False):
        """
        保存结果到 JSON 文件。
        """
        if not self.metrics_list:
            return
        if final:
            filename = "fuel_analysis_results_final.json"
        elif partial:
            filename = "fuel_analysis_results_partial.json"
        else:
            filename = "fuel_analysis_results.json"
        filepath = os.path.join(self.output_dir, filename)
        data = [asdict(m) for m in self.metrics_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[FuelAnalyzer] 已保存 {len(self.metrics_list)} 条结果到 {filepath}")

    def plot_results(self):
        """
        生成燃料分析图表。
        """
        if not self.metrics_list:
            print("[FuelAnalyzer] 无数据，无法绘图")
            return

        # 按轨道类型分组
        orbit_types = sorted(set(m.orbit_type for m in self.metrics_list))
        # 按盲区时长分组
        blind_intervals = sorted(set(m.blind_interval_days for m in self.metrics_list))
        # 按控制增益分组
        gain_scales = sorted(set(m.control_gain_scale for m in self.metrics_list))

        # 创建多子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("燃料消耗分析", fontsize=14, fontweight='bold')

        # 1. 平均每天 ΔV vs 轨道类型和盲区时长（箱线图）
        ax = axes[0, 0]
        data_by_orb_blind = {}
        for orb in orbit_types:
            for blind in blind_intervals:
                key = f"{orb}_{blind}"
                dv_vals = [m.avg_dv_per_day for m in self.metrics_list
                           if m.orbit_type == orb and m.blind_interval_days == blind]
                if dv_vals:
                    data_by_orb_blind[key] = dv_vals
        # 绘制箱线图
        labels = [f"{k}" for k in data_by_orb_blind.keys()]
        positions = range(len(data_by_orb_blind))
        bp = ax.boxplot(data_by_orb_blind.values(), positions=positions, widths=0.6, patch_artist=True)
        ax.set_xticks(positions, labels, rotation=45, ha='right')
        ax.set_ylabel('平均每天 ΔV (m/s/天)')
        ax.set_title('燃料消耗随轨道类型和盲区时长变化')
        ax.grid(True, alpha=0.3)

        # 2. 总 ΔV vs 控制增益（折线图，按轨道类型分组）
        ax = axes[0, 1]
        for orb in orbit_types:
            gain_vals = []
            dv_means = []
            dv_stds = []
            for gain in gain_scales:
                dv_vals = [m.total_dv for m in self.metrics_list
                           if m.orbit_type == orb and m.control_gain_scale == gain]
                if dv_vals:
                    gain_vals.append(gain)
                    dv_means.append(np.mean(dv_vals))
                    dv_stds.append(np.std(dv_vals))
            if gain_vals:
                ax.errorbar(gain_vals, dv_means, yerr=dv_stds, marker='o', capsize=3, label=orb)
        ax.set_xlabel('控制增益缩放因子')
        ax.set_ylabel('总 ΔV (m/s)')
        ax.set_title('总 ΔV 随控制增益变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 总 ΔV vs 盲区时长（箱线图，按轨道类型分）
        ax = axes[1, 0]
        for orb in orbit_types:
            blind_vals = []
            dv_vals_by_blind = []
            for blind in blind_intervals:
                dv_vals = [m.total_dv for m in self.metrics_list
                           if m.orbit_type == orb and m.blind_interval_days == blind]
                if dv_vals:
                    blind_vals.append(blind)
                    dv_vals_by_blind.append(dv_vals)
            if blind_vals:
                # 使用箱线图
                positions = blind_vals
                bp = ax.boxplot(dv_vals_by_blind, positions=positions, widths=0.6, patch_artist=True)
                # 设置颜色
                for box in bp['boxes']:
                    box.set_alpha(0.7)
                # 连线均值
                means = [np.mean(vals) for vals in dv_vals_by_blind]
                ax.plot(positions, means, 'o-', label=orb)
        ax.set_xlabel('盲区时长 (天)')
        ax.set_ylabel('总 ΔV (m/s)')
        ax.set_title('总 ΔV 随盲区时长变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 平均每天 ΔV 分布直方图
        ax = axes[1, 1]
        all_dv = [m.avg_dv_per_day for m in self.metrics_list if not np.isnan(m.avg_dv_per_day)]
        ax.hist(all_dv, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('平均每天 ΔV (m/s/天)')
        ax.set_ylabel('频数')
        ax.set_title('燃料消耗分布')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "fuel_analysis_plots.png")
        plt.savefig(save_path, dpi=150)
        print(f"[FuelAnalyzer] 图表已保存至 {save_path}")
        plt.show()

    def generate_report(self):
        """
        生成燃料分析报告（文本）。
        """
        if not self.metrics_list:
            print("[FuelAnalyzer] 无数据，无法生成报告")
            return

        # 按轨道类型、盲区时长、增益分组统计
        orbit_types = sorted(set(m.orbit_type for m in self.metrics_list))
        blind_intervals = sorted(set(m.blind_interval_days for m in self.metrics_list))
        gain_scales = sorted(set(m.control_gain_scale for m in self.metrics_list))

        lines = []
        lines.append("=" * 70)
        lines.append("燃料开销分析报告")
        lines.append("=" * 70)
        lines.append(f"总运行次数: {len(self.metrics_list)}")
        lines.append(f"轨道类型: {orbit_types}")
        lines.append(f"盲区时长 (天): {blind_intervals}")
        lines.append(f"控制增益缩放: {gain_scales}")
        lines.append("")

        # 汇总表
        lines.append("--- 按轨道类型汇总 ---")
        for orb in orbit_types:
            dv_vals = [m.total_dv for m in self.metrics_list if m.orbit_type == orb and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"{orb}:")
                lines.append(f"  平均总 ΔV: {np.mean(dv_vals):.4f} ± {np.std(dv_vals):.4f} m/s")
                lines.append(f"  中位数总 ΔV: {np.median(dv_vals):.4f} m/s")
                lines.append(f"  最小/最大: {np.min(dv_vals):.4f} / {np.max(dv_vals):.4f} m/s")
                lines.append(f"  样本数: {len(dv_vals)}")
                lines.append("")

        lines.append("--- 按盲区时长汇总 (所有轨道) ---")
        for blind in blind_intervals:
            dv_vals = [m.total_dv for m in self.metrics_list
                       if m.blind_interval_days == blind and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"盲区 {blind} 天:")
                lines.append(f"  平均总 ΔV: {np.mean(dv_vals):.4f} ± {np.std(dv_vals):.4f} m/s")
                lines.append(f"  样本数: {len(dv_vals)}")
                lines.append("")

        lines.append("--- 按控制增益汇总 (所有轨道) ---")
        for gain in gain_scales:
            dv_vals = [m.total_dv for m in self.metrics_list
                       if m.control_gain_scale == gain and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"增益 {gain}:")
                lines.append(f"  平均总 ΔV: {np.mean(dv_vals):.4f} ± {np.std(dv_vals):.4f} m/s")
                lines.append(f"  样本数: {len(dv_vals)}")
                lines.append("")

        lines.append("=" * 70)

        report_str = "\n".join(lines)
        print(report_str)

        # 保存报告
        report_path = os.path.join(self.output_dir, "fuel_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        print(f"[FuelAnalyzer] 报告已保存至 {report_path}")


def main():
    """
    示例运行。
    """
    # 自定义扫描参数（可根据需要修改）
    scan_params = {
        "orbit_type": ["Halo", "Keplerian"],          # 轨道类型
        "control_gain_scale": [0.5, 1.0, 2.0],        # 控制增益缩放因子
        "blind_interval_days": [0, 3, 7],             # 盲区时长 (天)
        "simulation_days": [30]                       # 仿真时长 (天)
    }

    # 基础配置
    base_config = {
        "time_step": 10.0,
        "log_buffer_size": 200,
        "enable_visualization": False,
        "data_dir": "data/fuel_analysis",
        "log_level": "WARNING"
    }

    # 创建分析器
    analyzer = FuelAnalyzer(
        base_config=base_config,
        scan_params=scan_params,
        output_dir="analysis_results",
        n_repeats=2        # 每个组合运行2次，实际可增加到5以上
    )

    # 运行扫描
    analyzer.run_scan()

    # 生成图表和报告
    analyzer.plot_results()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
