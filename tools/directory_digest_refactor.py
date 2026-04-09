
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory Digest Refactor - Refactored main entry point
Uses tools/_directory_digest module to implement full functionality
"""

import os
import sys
import argparse
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import base modules
from tools._directory_digest import (
    ProcessingStrategy,
    FileType,
    OutputFormats,
    FileMetadata,
    FileDigest,
    DirectoryStructure,
    FileTypeDetector,
    RuleEngine,
    ContextManager,
    FormatConverter,
    DirectoryDigestBase,
    STRATEGY_CONFIGS,
)

# Import processor registry
from tools._directory_digest.processors import create_default_registry


# 尝试导入高级分析器（如果可用）
try:
    from tools._directory_digest.analyzers.semantics.base import (
        HumanReadableSummary,
        SourceCodeAnalysis,
        HumanReadableSummarizer,
        SourceCodeAnalyzer,
        SmartTextProcessor,
    )
    SEMANTICS_AVAILABLE = True
except ImportError:
    SEMANTICS_AVAILABLE = False
    # 定义简化的后备类
    from dataclasses import dataclass, field
    
    @dataclass
    class HumanReadableSummary:
        title: Optional[str] = None
        line_count: int = 0
        word_count: int = 0
        character_count: int = 0
        summary: Optional[str] = None
        first_lines: List[str] = field(default_factory=list)
        
        def to_dict(self) -> Dict:
            return {
                "title": self.title,
                "line_count": self.line_count,
                "word_count": self.word_count,
                "character_count": self.character_count,
                "summary": self.summary,
                "first_lines": self.first_lines
            }
    
    @dataclass
    class SourceCodeAnalysis:
        language: str = "unknown"
        total_lines: int = 0
        code_lines: int = 0
        comment_lines: int = 0
        blank_lines: int = 0
        imports: List[str] = field(default_factory=list)
        functions: List[Dict] = field(default_factory=list)
        classes: List[Dict] = field(default_factory=list)
        
        def to_dict(self) -> Dict:
            return {
                "language": self.language,
                "total_lines": self.total_lines,
                "code_lines": self.code_lines,
                "comment_lines": self.comment_lines,
                "blank_lines": self.blank_lines,
                "imports": self.imports[:20],
                "functions": self.functions[:20],
                "classes": self.classes[:20]
            }


class DirectoryDigest(DirectoryDigestBase):
    """完整的目录摘要生成器 - 继承自基础版并扩展功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化处理器注册表（新的核心组件）
        self.processor_registry = create_default_registry(
            rule_engine=self.rule_engine,
            context_manager=self.context_manager,
            stats=self.stats,
            config=self.config
        )
        
        # 保留高级分析器组件（如果语义分析可用）
        if SEMANTICS_AVAILABLE:
            self.human_summarizer = HumanReadableSummarizer()
            self.source_analyzer = SourceCodeAnalyzer()
        else:
            self.human_summarizer = None
            self.source_analyzer = None
    
    def create_digest(self, mode: str = "framework") -> Dict:
        """
        创建完整目录摘要
        
        Args:
            mode: 输出模式，"framework"、"full" 或 "sort"
        """
        import time
        start_time = time.time()
        
        # 构建目录结构（自动使用基类方法）
        self.structure = self._build_directory_structure(self.root)
        
        # 处理所有文件
        if mode == "sort":
            # sort 模式只需基础元数据，无需深度内容分析
            for file_digest in self._collect_all_files_flat():
                filepath = file_digest.metadata.path
                file_type = self.file_type_detector.detect(filepath)
                file_digest.metadata.file_type = file_type
                
                # 更新统计
                type_stat_key = file_type.value
                if type_stat_key in self.stats:
                    self.stats[type_stat_key] += 1
                
                self._calculate_hashes(file_digest)
            
            # 更新统计信息
            self.stats['processing_time'] = time.time() - start_time
            
            # 关键修复：调用 _generate_sort_output 生成完整报告
            return self._generate_sort_output()  # 使用基类方法
        else:
            # 其他模式使用处理器注册表进行深度处理
            self.processor_registry.process_directory(
                self.structure, 
                mode=mode,
                parallel=self.use_parallel,
                max_workers=self.max_workers
            )
            
            # 更新统计信息
            self.stats['processing_time'] = time.time() - start_time
            
            return self._generate_output(mode)
    
    def _generate_output(self, mode: str) -> Dict:
        """生成完整输出"""
        if not self.structure:
            return {}
        
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": mode,
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary(),
            },
            "structure": self.structure.to_dict(mode)
        }
        
        if self.context_manager.file_records:
            output["context_allocation"] = {
                "file_records": self.context_manager.file_records
            }
        
        return output
    
    




def parse_context_size(size_str: str) -> int:
    """解析上下文大小字符串"""
    size_str = size_str.lower().strip()
    
    if size_str.endswith('k'):
        multiplier = 1000
        size_str = size_str[:-1]
    else:
        multiplier = 1
    
    try:
        base_value = float(size_str) if '.' in size_str else int(size_str)
        return int(base_value * multiplier)
    except ValueError:
        print(f"Warning: Could not parse context size '{size_str}', using default 128000", file=sys.stderr)
        return 128000


def main():
    """Command line entry point"""
    # Custom help formatter to preserve formatting
    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _format_action(self, action):
            return super()._format_action(action)
    
    parser = argparse.ArgumentParser(
        prog="directory_digest_refactor",
        description="""
Directory Digest Tool - Directory Knowledge Digest Generator

Recursively "digests" the filesystem into LLM-understandable context summaries.
Rule-based intelligent classification: Human-readable text | Source code | Config files | Binary files

Three operation modes:
  framework (default)  Generate file structure and metadata summaries
  full                 Include complete file contents (subject to context window limits)
  sort                 List files by type/size with statistics and recommendations
        """.strip(),
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  # Basic usage - output to stdout (JSON format)
  %(prog)s /path/to/project
  
  # Use rules file
  %(prog)s . --rules .digest_rules.yaml --save report.json
  
  # Custom context size
  %(prog)s . --context-size 64k --mode full
  
  # Sort mode for classification overview
  %(prog)s . --mode sort | less
  
  # Full mode with piping
  %(prog)s . --mode full | grep -A 5 "class Controller"
  
  # Analyze large project with parallel processing
  %(prog)s /data --parallel --workers 8 --save report.json
  
  # Strict mode: skip files larger than 100MB
  %(prog)s . --max-size 100
  
  # Custom ignore rules, output YAML to stdout
  %(prog)s . --ignore "*.log,*.tmp,cache,*.min.js" --output yaml
  
  # Generate HTML report
  %(prog)s /code --mode full --output html --save report.html
        """
    )
    
    # Core options group
    core_group = parser.add_argument_group("Core Options", "Specify input directory and operation mode")
    core_group.add_argument(
        "directory",
        metavar="PATH",
        help="Directory path to analyze"
    )
    core_group.add_argument(
        "-m", "--mode",
        choices=["full", "framework", "sort"],
        default="framework",
        metavar="MODE",
        help="Operation mode (default: %(default)s)"
    )
    
    # Rules and context control
    rule_group = parser.add_argument_group(
        "Rules and Context Control",
        "Control file classification and LLM context optimization"
    )
    rule_group.add_argument(
        "-r", "--rules",
        metavar="FILE",
        help="""
        Path to rules file (YAML format).
        Defines file classification and processing strategies.
        If not provided, uses built-in heuristic rules.
        """
    )
    rule_group.add_argument(
        "--context-size",
        type=str,
        default="128k",
        metavar="SIZE",
        help="""
        Target LLM context size (tokens).
        Supports formats: "64k", "128k", "256k" or specific numbers.
        Used for optimizing token allocation.
        (default: %(default)s)
        """
    )
    
    # Output control group
    output_group = parser.add_argument_group(
        "Output Control",
        "Control output format, destination, and content detail"
    )
    output_group.add_argument(
        "-o", "--output",
        choices=["json", "yaml", "md", "html", "toml", "txt"],
        default="json",
        metavar="FORMAT",
        help="Output format (default: %(default)s)"
    )
    output_group.add_argument(
        "-s", "--save",
        metavar="FILE",
        help="""
        Specify output file path. Special cases:
          - Omit this option: output to stdout (suitable for piping)
          - Use "-": Force output to stdout (even if this option is provided)
          - Other paths: Write to specified file (directories created automatically)
        (default: output to stdout)
        """
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed processing information (including per-file status)"
    )
    
    # Size limits group
    size_group = parser.add_argument_group(
        "Size Limits",
        "Control thresholds for file processing"
    )
    size_group.add_argument(
        "--max-size",
        type=int,
        default=10240,
        metavar="MB",
        help="""
        File size threshold (MB). Files exceeding this size will be **completely skipped**:
        No checksum calculation, no content analysis, only path and size metadata retained.
        Useful for excluding oversized logs, VM images, datasets, media files.
        (default: %(default)s MB = 10 GB)
        """
    )
    
    # Processing options group
    proc_group = parser.add_argument_group("Processing Options", "Control parallel processing and file filtering")
    proc_group.add_argument(
        "--ignore",
        default=".git,__pycache__,*.pyc,*.pyo,node_modules,.venv,venv,*.min.js,*.map",
        metavar="PATTERNS",
        help="""
        Ignore patterns, comma-separated glob patterns.
        Default ignores version control, cache, dependency directories and minified files: %(default)s
        """
    )
    proc_group.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Enable parallel processing (recommended for large projects with >1000 files)"
    )
    proc_group.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        metavar="N",
        help="""
        Number of parallel worker threads.
        0 means auto-detect CPU core count (default: %(default)s -> actually uses %(const)s threads)
        """ % {'default': 0, 'const': os.cpu_count() or 4}
    )
    
    args = parser.parse_args()
    
    # Parse context size
    context_size = parse_context_size(args.context_size)
    
    # Configuration conversion
    config = {
        'max_file_size': args.max_size * 1024 * 1024,  # MB to Bytes
        'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()],
        'use_parallel': args.parallel,
        'max_workers': args.workers if args.workers > 0 else os.cpu_count() or 4,
        'rules_file': Path(args.rules) if args.rules else None,
        'context_size': context_size,
    }
    
    # Create digest generator
    digest = DirectoryDigest(args.directory, config)
    
    if args.verbose:
        print(f"Analyzing directory: {args.directory}", file=sys.stderr)
        print(f"Mode: {args.mode}, Format: {args.output}", file=sys.stderr)
        print(f"Skip files larger than: {args.max_size} MB ({args.max_size/1024:.1f} GB)", file=sys.stderr)
        print(f"Context window: {context_size:,} tokens", file=sys.stderr)
        if args.rules:
            print(f"Rules file: {args.rules}", file=sys.stderr)
        if args.parallel:
            print(f"Parallel processing enabled with {config['max_workers']} workers", file=sys.stderr)
    
    # Generate digest
    output = digest.create_digest(args.mode)
    
    # Handle output: default stdout, --save specifies file path, --save - forces stdout
    output_to_stdout = (args.save is None or args.save == '-')
    
    if output_to_stdout:
        # 输出到标准输出（支持管道处理）
        try:
            content = FormatConverter.convert(output, args.output, mode=args.mode)
            sys.stdout.write(content)
            if not content.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()
        except BrokenPipeError:
            # 忽略管道中断错误（如输出被 head/tail 截断）
            pass
            
        # 统计信息输出到 stderr - 与原始代码一致
        if args.verbose or args.mode == "sort":
            stats = output['metadata']['statistics']
            ctx_usage = output['metadata'].get('context_usage')
                
            print(f"\n[Summary] Files: {stats['total_files']}, "
                  f"Critical: {stats.get('critical_docs', 0)}, "
                  f"Reference: {stats.get('reference_docs', 0)}, "
                  f"Source: {stats.get('source_code', 0)}, "
                  f"Text Data: {stats.get('text_data', 0)}, "
                  f"Binary: {stats.get('binary_files', 0)}", 
                  file=sys.stderr)
                
            if stats.get('skipped_large_files', 0) > 0:
                print(f"         Skipped (size): {stats['skipped_large_files']}", file=sys.stderr)
                
            if stats.get('skipped_by_context', 0) > 0:
                print(f"         Skipped (context): {stats['skipped_by_context']}", file=sys.stderr)
                
            if ctx_usage:
                print(f"[Context] Used: {ctx_usage['used_tokens']:,}/{ctx_usage['max_tokens']:,} tokens "
                      f"({ctx_usage['token_utilization']:.1%})", file=sys.stderr)
            else:
                print(f"[Context] Not applicable for sort mode", file=sys.stderr)
                
            if args.mode == "sort" and "recommendations" in output:
                for rec in output["recommendations"]:
                    print(f"[Tip] {rec}", file=sys.stderr)
            
        return None  # stdout 模式返回 None
        
    else:
        # Write to specified file
        output_path = Path(args.save)
        saved_path = digest.save_output(output, args.output, output_path, args.mode)
        
        # Display processing results (to stderr, avoid mixing with file content)
        stats = output['metadata']['statistics']
        ctx_usage = output['metadata'].get('context_usage')
        
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Directory Digest Summary", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)
        print(f"Total files scanned:     {stats['total_files']}", file=sys.stderr)
        print(f"  ├── Critical docs:     {stats.get('critical_docs', 0)}", file=sys.stderr)
        print(f"  ├── Reference docs:    {stats.get('reference_docs', 0)}", file=sys.stderr)
        print(f"  ├── Source code:       {stats.get('source_code', 0)}", file=sys.stderr)
        print(f"  ├── Text data:         {stats.get('text_data', 0)}", file=sys.stderr)
        print(f"  ├── Binary files:      {stats.get('binary_files', 0)}", file=sys.stderr)
        if stats.get('skipped_large_files', 0) > 0:
            print(f"  ├── Skipped (size):    {stats['skipped_large_files']}", file=sys.stderr)
        if stats.get('skipped_by_context', 0) > 0:
            print(f"  └── Skipped (context): {stats['skipped_by_context']}", file=sys.stderr)
        
        if ctx_usage:
            print(f"Context window:          {ctx_usage['max_tokens']:,} tokens", file=sys.stderr)
            print(f"Context used:            {ctx_usage['used_tokens']:,} tokens "
                  f"({ctx_usage['token_utilization']:.1%})", file=sys.stderr)
        else:
            print(f"Context window:          Not applicable for sort mode", file=sys.stderr)
        
        print(f"Processing time:         {stats['processing_time']:.2f} s", file=sys.stderr)
        print(f"Output saved to:         {saved_path}", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)
        
        if args.mode == "sort" and "recommendations" in output:
            print(f"\nRecommendations:", file=sys.stderr)
            for rec in output["recommendations"]:
                print(f"  • {rec}", file=sys.stderr)
        
        return saved_path


if __name__ == "__main__":
    main()
