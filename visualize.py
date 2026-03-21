#!/usr/bin/env python3
# visualize.py
"""
MCPC 仿真数据可视化工具
用于对已有的 HDF5 数据文件进行深度分析和报告生成。

支持单文件分析和多文件对比。
"""

import argparse
import sys
import os
from mission_sim.utils.visualizer_L1 import L1Visualizer


def main():
    parser = argparse.ArgumentParser(
        description="MCPC 仿真数据可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单文件分析，生成所有基础图表:
    python visualize.py data/simulation.h5

  单文件分析，只生成 3D 轨迹（惯性系，参考点太阳）:
    python visualize.py data/simulation.h5 --plot trajectory --frame inertial --ref_point sun

  生成 HTML 报告:
    python visualize.py data/simulation.h5 --report

  多文件对比:
    python visualize.py sim1.h5 sim2.h5 sim3.h5 --labels "Baseline" "Gain=0.5" "Gain=2.0"
        """
    )

    parser.add_argument("files", nargs="+", help="HDF5 文件路径（可多个）")
    parser.add_argument("--plot", choices=["all", "trajectory", "error", "control",
                                           "histogram", "spectrum", "state", "thrust"],
                        default="all", help="选择要绘制的图表类型（仅单文件时有效）")
    parser.add_argument("--report", action="store_true", help="生成 HTML 报告（仅单文件）")
    parser.add_argument("--labels", nargs="+", help="多文件对比时的标签（数量需与文件数一致）")
    parser.add_argument("--output", default=".", help="输出目录")
    parser.add_argument("--mission_name", default=None, help="任务名称（覆盖文件内元数据）")
    parser.add_argument("--frame", choices=['rotating', 'inertial'], default='rotating',
                        help="3D 轨迹的参考系（仅影响轨迹图）")
    parser.add_argument("--ref_point", default='auto',
                        help="参考点，用于确定画布中心和范围。可选 'auto', 'sun', 'earth', 'l2' 或 'x,y,z'（单位：米）")
    parser.add_argument("--draw_ref", action="store_true", default=True,
                        help="是否在 3D 轨迹图中绘制参考点标记（默认 True）")
    parser.add_argument("--no_draw_ref", dest="draw_ref", action="store_false",
                        help="不在 3D 轨迹图中绘制参考点标记")

    args = parser.parse_args()

    # 单文件处理
    if len(args.files) == 1:
        vis = L1Visualizer(args.files[0], mission_name=args.mission_name)
        if args.report:
            vis.generate_report(output_dir=args.output)
        else:
            # 根据 plot 选项调用对应方法
            if args.plot == "all":
                vis.plot_3d_trajectory(
                    save_path=os.path.join(args.output, "trajectory.png"),
                    frame=args.frame,
                    ref_point=args.ref_point,
                    draw_ref=args.draw_ref
                )
                vis.plot_tracking_error(save_path=os.path.join(args.output, "errors.png"))
                vis.plot_control_effort(save_path=os.path.join(args.output, "control.png"))
                vis.plot_error_histogram(save_path=os.path.join(args.output, "error_histogram.png"))
                vis.plot_force_spectrum(save_path=os.path.join(args.output, "force_spectrum.png"))
                vis.plot_state_history(save_path=os.path.join(args.output, "state_history.png"))
                vis.plot_thrust_activity(save_path=os.path.join(args.output, "thrust_activity.png"))
            elif args.plot == "trajectory":
                vis.plot_3d_trajectory(
                    save_path=os.path.join(args.output, "trajectory.png"),
                    frame=args.frame,
                    ref_point=args.ref_point,
                    draw_ref=args.draw_ref
                )
            elif args.plot == "error":
                vis.plot_tracking_error(save_path=os.path.join(args.output, "errors.png"))
            elif args.plot == "control":
                vis.plot_control_effort(save_path=os.path.join(args.output, "control.png"))
            elif args.plot == "histogram":
                vis.plot_error_histogram(save_path=os.path.join(args.output, "error_histogram.png"))
            elif args.plot == "spectrum":
                vis.plot_force_spectrum(save_path=os.path.join(args.output, "force_spectrum.png"))
            elif args.plot == "state":
                vis.plot_state_history(save_path=os.path.join(args.output, "state_history.png"))
            elif args.plot == "thrust":
                vis.plot_thrust_activity(save_path=os.path.join(args.output, "thrust_activity.png"))
    else:
        # 多文件对比
        if not args.labels or len(args.labels) != len(args.files):
            print("错误：多文件对比需要提供 --labels 参数，且数量与文件数相同。")
            sys.exit(1)

        # 以第一个文件为基准，对比其余文件
        vis = L1Visualizer(args.files[0], mission_name=args.mission_name)
        vis.compare_simulations(args.files[1:], args.labels[1:])


if __name__ == "__main__":
    main()