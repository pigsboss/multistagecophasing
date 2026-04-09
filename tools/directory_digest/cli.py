#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py - 命令行接口
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from .digest import DirectoryDigest


def parse_context_size(size_str: str) -> int:
    """解析上下文大小字符串"""
    size_str = size_str.lower().strip()
    
    if size_str.endswith('k'):
        multiplier = 1000
        size_str = size_str[:-1]
    elif size_str.endswith('tokens'):
        multiplier = 1
        size_str = size_str[:-6].strip()
    else:
        multiplier = 1
    
    try:
        if '.' in size_str:
            base_value = float(size_str)
        else:
            base_value = int(size_str)
        
        return int(base_value * multiplier)
    except ValueError:
        print(f"警告: 无法解析上下文大小 '{size_str}'，使用默认值 128000", file=sys.stderr)
        return 128000


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="directory_digest",
        description="Directory Digest Tool - 目录知识摘要生成器",
        epilog="""
示例:
  # 基础使用 - 默认输出到 stdout（JSON格式）
  %(prog)s /path/to/project
  
  # 使用规则文件
  %(prog)s . --rules .digest_rules.yaml --save report.json
  
  # 自定义上下文大小
  %(prog)s . --context-size 64k --mode full
  
  # 排序模式查看分类
  %(prog)s . --mode sort | less
        """
    )
    
    # 核心选项组
    core_group = parser.add_argument_group("核心选项")
    core_group.add_argument(
        "directory", 
        metavar="PATH",
        help="要分析的目录路径"
    )
    core_group.add_argument(
        "-m", "--mode", 
        choices=["full", "framework", "sort"], 
        default="framework",
        help="工作模式 (默认: %(default)s)"
    )
    
    # 规则和上下文控制
    rule_group = parser.add_argument_group("规则和上下文控制")
    rule_group.add_argument(
        "-r", "--rules", 
        metavar="FILE",
        help="指定规则文件路径（YAML格式）"
    )
    rule_group.add_argument(
        "--context-size", 
        default="128k",
        help="目标LLM上下文大小 (默认: %(default)s)"
    )
    
    # 输出控制组
    output_group = parser.add_argument_group("输出控制")
    output_group.add_argument(
        "-o", "--output", 
        choices=["json", "yaml", "md", "html", "toml", "txt"], 
        default="json",
        help="输出格式 (默认: %(default)s)"
    )
    output_group.add_argument(
        "-s", "--save", 
        metavar="FILE",
        help="输出文件路径（默认输出到stdout）"
    )
    output_group.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="显示详细处理信息"
    )
    
    # 大小限制组
    size_group = parser.add_argument_group("大小限制")
    size_group.add_argument(
        "--max-size", 
        type=int, 
        default=10240,
        help="文件大小阈值(MB) (默认: %(default)s MB = 10 GB)"
    )
    
    # 处理选项组
    proc_group = parser.add_argument_group("处理选项")
    proc_group.add_argument(
        "--ignore", 
        default=".git,__pycache__,*.pyc,*.pyo,node_modules,.venv,venv,*.min.js,*.map",
        help="忽略规则，逗号分隔的glob模式"
    )
    proc_group.add_argument(
        "-p", "--parallel", 
        action="store_true",
        help="启用并行处理"
    )
    proc_group.add_argument(
        "-w", "--workers", 
        type=int, 
        default=0,
        help="并行工作线程数 (0=自动检测)"
    )
    
    return parser.parse_args()


def main() -> Optional[int]:
    """主函数"""
    try:
        args = parse_args()
        
        # 配置转换
        config = {
            'max_file_size': args.max_size * 1024 * 1024,
            'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()],
            'use_parallel': args.parallel,
            'max_workers': args.workers if args.workers > 0 else os.cpu_count() or 4,
            'rules_file': Path(args.rules) if args.rules else None,
            'context_size': parse_context_size(args.context_size),
        }
        
        # 创建目录摘要器
        digest = DirectoryDigest(args.directory, config=config)
        
        if args.verbose:
            print(f"分析目录: {args.directory}", file=sys.stderr)
            print(f"模式: {args.mode}, 格式: {args.output}", file=sys.stderr)
        
        # 生成摘要
        output = digest.create_digest(args.mode)
        
        # 处理输出
        if args.save is None or args.save == '-':
            # 输出到标准输出
            import json
            content = json.dumps(output, indent=2, ensure_ascii=False)
            sys.stdout.write(content)
            if not content.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            # 写入文件
            output_path = Path(args.save)
            digest.save_output(output, args.output, output_path, args.mode)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
