#!/usr/bin/env python3
# run.py
"""
MCPC 框架通用仿真入口

支持通过命令行参数或配置文件选择场景（scene）和层级（level），
动态加载对应的仿真类并运行。

使用示例：
    python run.py --scene sun_earth_l2 --level 1
    python run.py --scene sun_earth_l2 --level 1 --config config.yaml
    python run.py --scene sun_earth_l2 --level 1 --simulation_days 30 --time_step 60
    python run.py --scene sun_earth_l2 --level 1 --quiet                 # 静默模式
"""

import sys
import os
import argparse
import importlib
import yaml
from typing import Dict, Any


# 场景名到模块路径的映射
SCENE_MODULE_MAP = {
    "sun_earth_l2": "mission_sim.simulation.threebody.sun_earth_l2",
    # 未来可扩展：
    # "leo": "mission_sim.simulation.twobody.leo",
    # "geo": "mission_sim.simulation.twobody.geo",
    # "heo": "mission_sim.simulation.twobody.heo",
    # "cislunar": "mission_sim.simulation.threebody.cislunar",
}


def load_config(config_file: str = None) -> Dict[str, Any]:
    """
    加载配置文件（YAML 格式）。

    Args:
        config_file: 配置文件路径，若为 None 则返回空字典

    Returns:
        配置字典
    """
    if config_file is None:
        return {}
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 无法加载配置文件 {config_file}: {e}")
        sys.exit(1)


def get_simulation_class(scene: str, level: int):
    """
    根据场景和层级动态获取仿真类。

    Args:
        scene: 场景标识符，如 "sun_earth_l2"
        level: 层级编号 (1-5)

    Returns:
        仿真类
    """
    if scene not in SCENE_MODULE_MAP:
        raise ValueError(f"未知场景: {scene}，支持: {list(SCENE_MODULE_MAP.keys())}")

    module_path = SCENE_MODULE_MAP[scene]
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"无法导入模块 {module_path}: {e}")

    # 根据场景名生成类名（例如 sun_earth_l2 -> SunEarthL2L1Simulation）
    # 将蛇形转换为驼峰
    parts = scene.split('_')
    class_name = ''.join(p.capitalize() for p in parts) + f"L{level}Simulation"

    if not hasattr(module, class_name):
        raise AttributeError(f"模块 {module_path} 中没有找到类 {class_name}")

    return getattr(module, class_name)


def merge_config(default_config: Dict[str, Any], file_config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置：默认配置 < 文件配置 < 命令行参数
    """
    merged = default_config.copy()
    merged.update(file_config)
    merged.update(cli_args)
    return merged


def main():
    parser = argparse.ArgumentParser(description="MCPC 仿真通用入口")
    parser.add_argument("--scene", required=True, help="仿真场景（如 sun_earth_l2）")
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4, 5], help="仿真层级（L1-L5）")
    parser.add_argument("--config", help="YAML 配置文件路径")
    parser.add_argument("--mission_name", help="任务名称")
    parser.add_argument("--simulation_days", type=float, help="仿真天数")
    parser.add_argument("--time_step", type=float, help="积分步长 (秒)")
    parser.add_argument("--data_dir", help="数据输出目录")
    parser.add_argument("--enable_visualization", action="store_true", help="启用可视化")
    parser.add_argument("--disable_visualization", action="store_false", dest="enable_visualization", help="禁用可视化")
    parser.add_argument("--quiet", action="store_true", help="静默模式，仅输出错误信息（不输出进度等）")
    # 其他参数可通过 --key value 的形式传递，但 argparse 无法直接处理任意键值对。
    # 这里简化，用户可以添加额外参数，我们通过剩余参数收集，然后设置 config[key] = value
    # 为了支持任意参数，我们允许使用 --key value 的任意组合，并存入 config
    parser.add_argument("extra", nargs="*", help="其他参数，格式: --key1 value1 --key2 value2 ...")

    args = parser.parse_args()

    # 解析额外参数（格式：--key value）
    extra_args = {}
    if args.extra:
        # 处理类似 ['--key1', 'value1', '--key2', 'value2'] 的列表
        i = 0
        while i < len(args.extra):
            if args.extra[i].startswith('--'):
                key = args.extra[i][2:]
                if i + 1 < len(args.extra):
                    value = args.extra[i + 1]
                    extra_args[key] = value
                    i += 2
                else:
                    # 没有值，设置为 True
                    extra_args[key] = True
                    i += 1
            else:
                # 忽略非选项
                i += 1

    # 加载配置文件
    file_config = load_config(args.config)

    # 默认配置（不同场景可能有不同默认值，这里仅提供通用默认值）
    default_config = {
        "mission_name": "MCPC Simulation",
        "simulation_days": 1,
        "time_step": 60.0,
        "data_dir": "data",
        "enable_visualization": True,
        "log_buffer_size": 500,
        "log_compression": True,
        "progress_interval": 0.05,
        "verbose": True,   # 默认详细输出
    }

    # 合并配置：命令行参数覆盖文件配置，再覆盖默认
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k not in ["scene", "level", "config", "extra"]}
    cli_args.update(extra_args)  # 额外参数也作为命令行参数覆盖

    # 处理静默模式：覆盖 verbose 配置
    cli_args["verbose"] = not args.quiet

    config = merge_config(default_config, file_config, cli_args)

    try:
        # 动态获取仿真类
        SimClass = get_simulation_class(args.scene, args.level)
        # 实例化并运行
        sim = SimClass(config)
        success = sim.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 仿真启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()