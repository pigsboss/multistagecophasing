#!/usr/bin/env python3
"""
批量基准测试工具
读取任务描述文件，运行基准测试程序，收集结果并生成图表
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics

# 第三方库导入
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plotting will be skipped")


@dataclass
class BenchmarkJob:
    """基准测试任务"""
    name: str
    tool: str
    args: List[str]
    output_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """基准测试结果（简化版，用于批量处理）"""
    job_name: str
    tool: str
    task_name: str
    implementation: str
    scale_params: Dict[str, Any]
    execution_times: List[float]
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    std_time: float
    iterations_per_second: float
    output_file: str


@dataclass
class PlotGroupData:
    """绘图组数据"""
    name: str
    task: str
    x_parameter: str
    x_label: str
    y_parameter: str
    y_label: str
    data: Dict[str, List[Tuple[float, float]]]  # implementation -> [(x, y), ...]


class BatchBenchmarkRunner:
    """批量基准测试运行器"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.jobs: List[BenchmarkJob] = []
        self.results: List[BenchmarkResult] = []
        self.plot_groups: List[PlotGroupData] = []
        
        # 加载配置
        self.load_config()
        
        # 创建输出目录
        self.output_dir = Path(self.config.get('settings', {}).get('output_dir', 'batch_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(self.config.get('settings', {}).get('plots_dir', 'plots'))
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """加载YAML配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            if not self.config:
                raise ValueError("Empty configuration file")
            
            print(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def generate_jobs_from_plot_groups(self):
        """从绘图组配置生成具体的测试任务"""
        plot_groups_config = self.config.get('plot_groups', [])
        jobs_config = self.config.get('jobs', [])
        
        if not plot_groups_config:
            print("No plot groups defined in configuration")
            return
        
        # 创建任务模板映射
        job_templates = {job['name']: job for job in jobs_config}
        
        job_counter = 0
        for plot_group in plot_groups_config:
            group_name = plot_group['name']
            task = plot_group['task']
            scale_type = plot_group['scale_type']
            
            # 找到对应的任务模板
            template = None
            for job_template in jobs_config:
                if job_template.get('scale_type') == scale_type and job_template.get('task') == task:
                    template = job_template
                    break
            
            if not template:
                print(f"Warning: No job template found for {task} with scale_type {scale_type}")
                continue
            
            # 获取变化参数和固定参数
            varied_params = plot_group['varied_params']
            fixed_params = plot_group.get('fixed_params', {})
            
            # 对于每个变化参数组合生成任务
            for param_name, param_values in varied_params.items():
                for param_value in param_values:
                    # 构建任务参数
                    job_name = f"{template['name']}_{param_name}_{param_value}"
                    
                    # 构建命令行参数
                    args = template['base_args'].copy()
                    
                    # 添加规模参数
                    if scale_type == 'trajectory':
                        if param_name == 'steps':
                            args.extend(["--size-traj", f"({param_value},{fixed_params.get('trajectories', 500)})"])
                        elif param_name == 'trajectories':
                            args.extend(["--size-traj", f"({fixed_params.get('steps', 5000)},{param_value})"])
                    
                    elif scale_type == 'monte_carlo':
                        if param_name == 'samples':
                            args.extend(["--size-mc", str(param_value)])
                    
                    elif scale_type == 'nbody':
                        if param_name == 'bodies':
                            args.extend(["--size-nbody", f"({param_value},{fixed_params.get('steps', 20)})"])
                        elif param_name == 'steps':
                            args.extend(["--size-nbody", f"({fixed_params.get('bodies', 500)},{param_value})"])
                    
                    # 输出文件
                    output_file = self.output_dir / f"{job_name}.json"
                    args.extend(["--output", str(output_file)])
                    
                    # 元数据
                    metadata = {
                        'plot_group': group_name,
                        'task': task,
                        'scale_type': scale_type,
                        param_name: param_value,
                        **fixed_params
                    }
                    
                    # 创建任务
                    job = BenchmarkJob(
                        name=job_name,
                        tool=template['tool'],
                        args=args,
                        output_file=str(output_file),
                        metadata=metadata
                    )
                    
                    self.jobs.append(job)
                    job_counter += 1
        
        print(f"Generated {job_counter} jobs from plot groups")
    
    def run_jobs(self):
        """运行所有基准测试任务"""
        sleep_interval = self.config.get('settings', {}).get('sleep_interval', 1.0)
        
        print(f"\nStarting batch benchmark with {len(self.jobs)} jobs...")
        print(f"Sleep interval between jobs: {sleep_interval}s")
        
        for i, job in enumerate(self.jobs, 1):
            print(f"\n{'='*60}")
            print(f"Running job {i}/{len(self.jobs)}: {job.name}")
            print(f"Tool: {job.tool}")
            print(f"Args: {' '.join(job.args)}")
            
            try:
                # 构建命令
                cmd = ["python", job.tool] + job.args
                
                # 运行命令
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                elapsed_time = time.time() - start_time
                
                print(f"Completed in {elapsed_time:.2f}s")
                print(f"Output file: {job.output_file}")
                
                # 加载结果文件
                if Path(job.output_file).exists():
                    self.load_job_result(job)
                
            except subprocess.CalledProcessError as e:
                print(f"Error running job {job.name}: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
            except Exception as e:
                print(f"Unexpected error in job {job.name}: {e}")
            
            # 等待一段时间（除了最后一个任务）
            if i < len(self.jobs):
                print(f"Sleeping for {sleep_interval}s...")
                time.sleep(sleep_interval)
        
        print(f"\n{'='*60}")
        print(f"All jobs completed. Total results collected: {len(self.results)}")
    
    def load_job_result(self, job: BenchmarkJob):
        """加载单个任务的结果文件"""
        try:
            with open(job.output_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 提取基准测试结果
            for benchmark_result in result_data.get('benchmark_results', []):
                # 提取规模参数
                scale_params = job.metadata.copy()
                
                # 创建结果对象
                result = BenchmarkResult(
                    job_name=job.name,
                    tool=job.tool,
                    task_name=benchmark_result['task_name'],
                    implementation=benchmark_result['implementation'],
                    scale_params=scale_params,
                    execution_times=benchmark_result['execution_times'],
                    min_time=benchmark_result['min_time'],
                    max_time=benchmark_result['max_time'],
                    avg_time=benchmark_result['avg_time'],
                    median_time=benchmark_result['median_time'],
                    std_time=benchmark_result['std_time'],
                    iterations_per_second=benchmark_result['iterations_per_second'],
                    output_file=job.output_file
                )
                
                self.results.append(result)
                
        except Exception as e:
            print(f"Error loading result file {job.output_file}: {e}")
    
    def aggregate_results(self):
        """聚合结果并准备绘图数据"""
        plot_groups_config = self.config.get('plot_groups', [])
        
        for plot_group_config in plot_groups_config:
            group_name = plot_group_config['name']
            task = plot_group_config['task']
            x_param = plot_group_config['x_parameter']
            y_param = plot_group_config['y_parameter']
            
            # 收集该组的数据
            group_data: Dict[str, List[Tuple[float, float]]] = {}
            
            # 过滤相关结果
            relevant_results = [
                r for r in self.results 
                if r.task_name == task and 
                r.scale_params.get('plot_group') == group_name
            ]
            
            # 按实现方式分组
            for result in relevant_results:
                impl = result.implementation
                x_value = result.scale_params.get(x_param)
                y_value = getattr(result, y_param, None)
                
                if x_value is not None and y_value is not None:
                    if impl not in group_data:
                        group_data[impl] = []
                    
                    # 转换为浮点数
                    try:
                        x_float = float(x_value)
                        y_float = float(y_value)
                        group_data[impl].append((x_float, y_float))
                    except (ValueError, TypeError):
                        continue
            
            # 对每个实现的数据按x值排序
            for impl in group_data:
                group_data[impl].sort(key=lambda x: x[0])
            
            # 创建绘图组数据
            plot_group = PlotGroupData(
                name=group_name,
                task=task,
                x_parameter=x_param,
                x_label=plot_group_config.get('x_label', x_param),
                y_parameter=y_param,
                y_label=plot_group_config.get('y_label', y_param),
                data=group_data
            )
            
            self.plot_groups.append(plot_group)
    
    def save_unified_results(self):
        """保存统一的结果文件"""
        unified_output = self.config.get('settings', {}).get(
            'unified_output', 'unified_results.json'
        )
        output_path = self.output_dir / unified_output
        
        # 准备输出数据
        output_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config_file': str(self.config_file),
            'total_jobs': len(self.jobs),
            'total_results': len(self.results),
            'results': []
        }
        
        # 添加每个结果
        for result in self.results:
            result_dict = {
                'job_name': result.job_name,
                'tool': result.tool,
                'task_name': result.task_name,
                'implementation': result.implementation,
                'scale_params': result.scale_params,
                'execution_times': result.execution_times,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'avg_time': result.avg_time,
                'median_time': result.median_time,
                'std_time': result.std_time,
                'iterations_per_second': result.iterations_per_second,
                'output_file': result.output_file
            }
            output_data['results'].append(result_dict)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nUnified results saved to: {output_path}")
        return output_path
    
    def generate_plots(self):
        """生成图表"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        print(f"\nGenerating plots in {self.plots_dir}...")
        
        for plot_group in self.plot_groups:
            if not plot_group.data:
                print(f"Warning: No data for plot group '{plot_group.name}'")
                continue
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(plot_group.name, fontsize=16, fontweight='bold')
            
            # 设置颜色循环
            colors = plt.cm.tab10(np.linspace(0, 1, len(plot_group.data)))
            
            # 线图
            for (impl, data_points), color in zip(plot_group.data.items(), colors):
                if len(data_points) < 2:
                    continue
                
                x_vals = [p[0] for p in data_points]
                y_vals = [p[1] for p in data_points]
                
                ax1.plot(x_vals, y_vals, 'o-', label=impl, color=color, linewidth=2, markersize=8)
            
            ax1.set_xlabel(plot_group.x_label, fontsize=12)
            ax1.set_ylabel(plot_group.y_label, fontsize=12)
            ax1.set_title(f"Line Chart - {plot_group.task}", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # 柱状图（使用最后一个x值的数据）
            if plot_group.data:
                last_x_values = {}
                for impl, data_points in plot_group.data.items():
                    if data_points:
                        # 取最后一个x值的数据
                        last_x, last_y = data_points[-1]
                        last_x_values[impl] = last_y
                
                if last_x_values:
                    implementations = list(last_x_values.keys())
                    y_values = [last_x_values[impl] for impl in implementations]
                    
                    bars = ax2.bar(range(len(implementations)), y_values, 
                                  color=colors[:len(implementations)])
                    
                    ax2.set_xlabel('Implementation', fontsize=12)
                    ax2.set_ylabel(plot_group.y_label, fontsize=12)
                    ax2.set_title(f"Bar Chart - Last Scale Point", fontsize=14)
                    ax2.set_xticks(range(len(implementations)))
                    ax2.set_xticklabels(implementations, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # 在柱子上添加数值标签
                    for bar, val in zip(bars, y_values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(y_values),
                                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # 保存图形
            safe_name = plot_group.name.replace(' ', '_').replace('(', '').replace(')', '')
            plot_file = self.plots_dir / f"{safe_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved plot: {plot_file}")
    
    def print_summary_statistics(self):
        """打印汇总统计信息"""
        print(f"\n{'='*80}")
        print("BATCH BENCHMARK SUMMARY")
        print('='*80)
        
        # 按任务和实现方式分组
        task_groups = {}
        for result in self.results:
            task = result.task_name
            impl = result.implementation
            
            if task not in task_groups:
                task_groups[task] = {}
            if impl not in task_groups[task]:
                task_groups[task][impl] = []
            
            task_groups[task][impl].append(result)
        
        # 打印每个任务的统计信息
        for task, impl_groups in task_groups.items():
            print(f"\n{task}:")
            print("-" * 60)
            
            for impl, results in impl_groups.items():
                if not results:
                    continue
                
                avg_times = [r.avg_time for r in results]
                iter_per_sec = [r.iterations_per_second for r in results]
                
                print(f"  {impl}:")
                print(f"    Number of runs: {len(results)}")
                print(f"    Avg time range: {min(avg_times):.4f}s - {max(avg_times):.4f}s")
                print(f"    Median avg time: {statistics.median(avg_times):.4f}s")
                print(f"    Avg iterations/s: {statistics.mean(iter_per_sec):.2f}")
                
                # 显示最佳规模
                if results:
                    best_result = min(results, key=lambda x: x.avg_time)
                    print(f"    Best scale: {best_result.scale_params}")
    
    def run(self):
        """运行完整的批量基准测试流程"""
        print("MCPC Batch Benchmark Tool")
        print("="*80)
        
        # 步骤1: 从绘图组生成任务
        self.generate_jobs_from_plot_groups()
        
        if not self.jobs:
            print("No jobs to run. Exiting.")
            return
        
        # 步骤2: 运行所有任务
        self.run_jobs()
        
        # 步骤3: 聚合结果
        self.aggregate_results()
        
        # 步骤4: 保存统一结果
        unified_file = self.save_unified_results()
        
        # 步骤5: 生成图表
        self.generate_plots()
        
        # 步骤6: 打印汇总统计
        self.print_summary_statistics()
        
        print(f"\n{'='*80}")
        print("Batch benchmark completed successfully!")
        print(f"Results saved in: {self.output_dir}")
        print(f"Plots saved in: {self.plots_dir}")
        print('='*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MCPC Batch Benchmark Tool - Run multiple benchmark jobs and generate analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yml                     # Run with configuration file
  %(prog)s jobs_template.yml --dry-run    # Show jobs without running them
  
Configuration file format (YAML):
  See jobs_template.yml for example
        """
    )
    
    parser.add_argument(
        "config_file",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show jobs to be run without executing them"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # 创建批量运行器
    runner = BatchBenchmarkRunner(str(config_path))
    
    if args.dry_run:
        # 仅显示将要运行的任务
        runner.generate_jobs_from_plot_groups()
        
        print("\nDRY RUN - Jobs to be executed:")
        print("="*80)
        
        for i, job in enumerate(runner.jobs, 1):
            print(f"\nJob {i}: {job.name}")
            print(f"  Tool: {job.tool}")
            print(f"  Args: {' '.join(job.args)}")
            print(f"  Output: {job.output_file}")
            print(f"  Metadata: {json.dumps(job.metadata, indent=2)}")
        
        print(f"\nTotal jobs: {len(runner.jobs)}")
        
    else:
        # 运行完整的批量测试
        runner.run()


if __name__ == "__main__":
    main()
