#!/usr/bin/env python3
"""
L2 编队仿真性能基准测试

运行多星编队仿真，记录：
- 仿真总耗时 (秒)
- CPU 时间 (用户+系统)
- 内存峰值 (MB)
- 仿真步数
- 各星最终 ΔV

Usage:
    python tests/benchmark_l2.py [--num_deputies N] [--days DAYS] [--output FILE.json]
"""

import os
import sys
import time
import json
import argparse
import tracemalloc
from typing import Dict, Any

import numpy as np

# 确保项目根目录在路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mission_sim.simulation.formation_simulation import FormationSimulation


def run_benchmark(num_deputies: int = 10, simulation_days: float = 1.0, data_dir: str = "data/benchmark") -> Dict[str, Any]:
    """
    运行编队仿真并收集性能数据。

    Args:
        num_deputies: 从星数量
        simulation_days: 仿真时长 (天)
        data_dir: 输出目录

    Returns:
        包含性能指标的字典
    """
    # 生成初始状态 (圆形 LEO 轨道)
    r = 7000e3          # 半径 7000 km
    v = np.sqrt(3.986004418e14 / r)  # 圆轨道速度

    chief_state = [r, 0.0, 0.0, 0.0, v, 0.0]

    # 生成从星状态：在 LVLH 坐标系下随机偏移 (±500 m, ±5 m/s)
    deputy_states = []
    np.random.seed(42)  # 可重复
    for i in range(num_deputies):
        dx = np.random.uniform(-500, 500)
        dy = np.random.uniform(-500, 500)
        dz = np.random.uniform(-500, 500)
        dvx = np.random.uniform(-5, 5)
        dvy = np.random.uniform(-5, 5)
        dvz = np.random.uniform(-5, 5)
        # 在惯性系中简单叠加（近似）
        state = [
            r + dx,
            dy,
            dz,
            dvx,
            v + dvy,
            dvz
        ]
        deputy_states.append((f"DEP_{i+1:03d}", state))

    config = {
        "mission_name": f"Benchmark_{num_deputies}Dep",
        "simulation_days": simulation_days,
        "time_step": 1.0,
        "data_dir": data_dir,
        "verbose": False,
        "chief_initial_state": chief_state,
        "deputy_initial_states": deputy_states,
        "chief_frame": "J2000_ECI",
        "chief_mass_kg": 2000.0,
        "deputy_mass_kg": 500.0,
        "orbit_angular_rate": v / r,  # 角速度 = v / r
        "router_base_latency_s": 0.05,
        "router_jitter_s": 0.01,
        "router_packet_loss_rate": 0.02,
        "generation_threshold_pos": 100.0,
        "generation_threshold_vel": 0.5,
        "keeping_threshold_pos": 1.0,
        "keeping_threshold_vel": 0.01,
        # 禁用额外摄动，保持简单
        "enable_crtbp": False,
        "enable_j2": False,
        "enable_atmospheric_drag": False,
        "enable_srp": False,
    }

    # 启动内存跟踪
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]  # 当前内存分配

    # 计时
    start_wall = time.time()
    start_cpu = time.process_time()

    # 运行仿真
    sim = FormationSimulation(config)
    success = sim.run()

    end_cpu = time.process_time()
    end_wall = time.time()

    # 获取内存峰值
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 收集统计信息
    stats = sim.get_statistics() if success else {}
    deputy_dv = stats.get("deputy_dv", {})

    result = {
        "success": success,
        "num_deputies": num_deputies,
        "simulation_days": simulation_days,
        "total_steps": sim.total_steps,
        "wall_time_s": end_wall - start_wall,
        "cpu_time_s": end_cpu - start_cpu,
        "memory_peak_mb": peak_mem / (1024 * 1024),
        "chief_dv_mps": stats.get("chief_dv", 0.0),
        "deputy_dv_mean_mps": np.mean(list(deputy_dv.values())) if deputy_dv else 0.0,
        "deputy_dv_std_mps": np.std(list(deputy_dv.values())) if deputy_dv else 0.0,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="L2 Formation Simulation Benchmark")
    parser.add_argument("--num_deputies", type=int, default=10, help="Number of deputy spacecraft")
    parser.add_argument("--days", type=float, default=1.0, help="Simulation duration (days)")
    parser.add_argument("--output", type=str, default="benchmark_result.json", help="Output JSON file")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeated runs (for statistics)")
    args = parser.parse_args()

    results = []
    for i in range(args.repeat):
        print(f"Running benchmark iteration {i+1}/{args.repeat}...")
        res = run_benchmark(args.num_deputies, args.days)
        results.append(res)
        if not res["success"]:
            print(f"  ❌ Simulation failed on iteration {i+1}")
        else:
            print(f"  ✅ Wall time: {res['wall_time_s']:.2f}s, CPU: {res['cpu_time_s']:.2f}s, Peak memory: {res['memory_peak_mb']:.1f}MB")

    # 汇总多轮结果
    if args.repeat > 1:
        summary = {
            "num_deputies": args.num_deputies,
            "simulation_days": args.days,
            "repeats": args.repeat,
            "wall_time_mean_s": np.mean([r["wall_time_s"] for r in results]),
            "wall_time_std_s": np.std([r["wall_time_s"] for r in results]),
            "cpu_time_mean_s": np.mean([r["cpu_time_s"] for r in results]),
            "cpu_time_std_s": np.std([r["cpu_time_s"] for r in results]),
            "memory_peak_mean_mb": np.mean([r["memory_peak_mb"] for r in results]),
            "memory_peak_std_mb": np.std([r["memory_peak_mb"] for r in results]),
            "all_results": results,
        }
        output = summary
    else:
        output = results[0]

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n📊 Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()