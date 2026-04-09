
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory Digest Refactor - 重构版主入口
利用 tools/_directory_digest 模块实现完整功能
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 导入基础模块
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
        
        # 初始化高级组件（如果可用）
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
        
        # 构建目录结构
        self.structure = self._build_directory_structure(self.root)
        
        # 处理所有文件
        self._process_directory(self.structure, mode)
        
        # 更新统计信息
        self.stats['processing_time'] = time.time() - start_time
        
        # 根据模式生成输出
        if mode == "sort":
            return self._generate_sort_output()
        else:
            return self._generate_output(mode)
    
    def _process_directory(self, structure: DirectoryStructure, mode: str):
        """处理目录中的所有文件"""
        for file_digest in structure.files:
            self._process_file(file_digest, mode)
        
        # 递归处理子目录
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_file(self, file_digest: FileDigest, mode: str):
        """处理单个文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 1. 获取处理策略
            strategy, force_binary = self.rule_engine.classify_file(filepath)
            
            # 2. 估算token消耗
            estimated_tokens = self.rule_engine.estimate_token_usage(filepath, strategy)
            
            # 3. 检查文件大小限制
            if file_digest.metadata.size > self.max_file_size:
                self._process_as_binary(file_digest)
                self.stats['skipped_large_files'] += 1
                return
            
            # 4. 检查上下文限制
            file_record = {
                "path": str(filepath),
                "strategy": strategy.value,
                "estimated_tokens": estimated_tokens,
                "size": file_digest.metadata.size,
            }
            
            if not self.context_manager.can_allocate(estimated_tokens):
                # 尝试降级策略
                downgraded_strategy = self.context_manager.downgrade_strategy(strategy)
                downgraded_tokens = self.rule_engine.estimate_token_usage(filepath, downgraded_strategy)
                
                if not self.context_manager.can_allocate(downgraded_tokens):
                    self.stats['skipped_by_context'] += 1
                    return
                
                strategy = downgraded_strategy
                estimated_tokens = downgraded_tokens
            
            # 5. 分配token
            if not self.context_manager.allocate(estimated_tokens, file_record):
                self.stats['skipped_by_context'] += 1
                return
            
            # 6. 根据策略处理文件
            if force_binary or strategy == ProcessingStrategy.METADATA_ONLY:
                self._process_as_binary(file_digest)
            elif strategy == ProcessingStrategy.FULL_CONTENT:
                self._process_as_text(file_digest, mode, include_full=True)
            elif strategy == ProcessingStrategy.SUMMARY_ONLY:
                self._process_as_text(file_digest, mode, include_full=False)
            elif strategy == ProcessingStrategy.CODE_SKELETON:
                self._process_as_code(file_digest, mode)
            elif strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
                self._process_as_config(file_digest, mode)
            elif strategy == ProcessingStrategy.HEADER_WITH_STATS:
                self._process_as_data(file_digest, mode)
            else:
                self._process_as_text(file_digest, mode, include_full=False)
            
        except Exception as e:
            print(f"Warning: Error processing file {filepath}: {e}", file=sys.stderr)
    
    def _process_as_binary(self, file_digest: FileDigest):
        """处理为二进制文件"""
        file_digest.metadata.file_type = FileType.BINARY_FILES
        self.stats['binary_files'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_text(self, file_digest: FileDigest, mode: str, include_full: bool = False):
        """处理为文本文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 生成摘要
        if self.human_summarizer and SEMANTICS_AVAILABLE:
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
        else:
            # 简化摘要
            lines = content.split('\n')
            file_digest.human_readable_summary = HumanReadableSummary(
                title=filepath.stem.replace('_', ' ').title(),
                line_count=len(lines),
                word_count=len(content.split()),
                character_count=len(content),
                first_lines=lines[:10],
                summary=f"Text file with {len(lines)} lines"
            )
        
        if include_full and mode == "full":
            file_digest.full_content = content
        
        # 判断文件类型
        filename = filepath.name.lower()
        critical_patterns = ['readme', 'license', 'copying', 'notice', 'changelog', 'changes']
        
        if any(pattern in filename for pattern in critical_patterns):
            file_digest.metadata.file_type = FileType.CRITICAL_DOCS
            self.stats['critical_docs'] += 1
        else:
            file_digest.metadata.file_type = FileType.REFERENCE_DOCS
            self.stats['reference_docs'] += 1
        
        self._calculate_hashes(file_digest)
    
    def _process_as_code(self, file_digest: FileDigest, mode: str):
        """处理为源代码文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 分析源代码
        if self.source_analyzer and SEMANTICS_AVAILABLE:
            analysis = self.source_analyzer.analyze(filepath, content)
            file_digest.source_code_analysis = analysis
        else:
            # 简化分析
            lines = content.split('\n')
            file_digest.source_code_analysis = SourceCodeAnalysis(
                language=filepath.suffix.lstrip('.') or "unknown",
                total_lines=len(lines),
                code_lines=sum(1 for line in lines if line.strip()),
                comment_lines=sum(1 for line in lines if line.strip().startswith(('#', '//'))),
                blank_lines=sum(1 for line in lines if not line.strip())
            )
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.SOURCE_CODE
        self.stats['source_code'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_config(self, file_digest: FileDigest, mode: str):
        """处理为配置文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 生成结构摘要
        lines = content.split('\n')
        keys = []
        for line in lines[:50]:
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', ';')):
                if ':' in stripped or '=' in stripped:
                    key = stripped.split(':', 1)[0].split('=', 1)[0].strip()
                    if key and len(key) < 50:
                        keys.append(key)
        
        file_digest.human_readable_summary = HumanReadableSummary(
            title=filepath.name,
            line_count=len(lines),
            character_count=len(content),
            first_lines=lines[:10],
            summary=f"Config file with keys: {', '.join(keys[:10])}" + ("..." if len(keys) > 10 else "")
        )
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)
    
    def _process_as_data(self, file_digest: FileDigest, mode: str):
        """处理为数据文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        lines = content.split('\n')
        file_digest.human_readable_summary = HumanReadableSummary(
            title=filepath.name,
            line_count=len(lines),
            character_count=len(content),
            first_lines=lines[:20],
            summary=f"Data file with {len(lines)} lines"
        )
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)
    
    def _generate_sort_output(self) -> Dict:
        """生成分类排序输出"""
        all_files = self._collect_all_files_flat()
        
        # 按类型分组
        by_type = {
            FileType.CRITICAL_DOCS.value: [],
            FileType.REFERENCE_DOCS.value: [],
            FileType.SOURCE_CODE.value: [],
            FileType.TEXT_DATA.value: [],
            FileType.BINARY_FILES.value: [],
            FileType.UNKNOWN.value: []
        }
        
        for f in all_files:
            file_type = f.metadata.file_type.value
            file_info = {
                'path': str(f.metadata.path.relative_to(self.root)),
                'size': f.metadata.size,
                'size_formatted': self._format_size(f.metadata.size),
                'modified': f.metadata.modified_time.isoformat(),
                'type': file_type
            }
            
            if file_type in by_type:
                by_type[file_type].append(file_info)
            else:
                by_type[FileType.UNKNOWN.value].append(file_info)
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": "sort",
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary()
            },
            "file_listings": {
                k: sorted(v, key=lambda x: x['path']) 
                for k, v in by_type.items() if v
            }
        }
    
    def _generate_output(self, mode: str) -> Dict:
        """生成完整输出"""
        if not self.structure:
            return {}
        
        # 确保结构有 to_dict 方法
        try:
            structure_dict = self.structure.to_dict(mode)
        except AttributeError:
            # 如果 to_dict 不存在，手动转换
            structure_dict = self._structure_to_dict(self.structure, mode)
        
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": mode,
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary(),
            },
            "structure": structure_dict
        }
        
        if self.context_manager.file_records:
            output["context_allocation"] = {
                "file_records": self.context_manager.file_records
            }
        
        return output
    
    def _structure_to_dict(self, structure: DirectoryStructure, mode: str) -> Dict:
        """手动转换目录结构为字典"""
        return {
            "path": str(structure.path),
            "files": [self._file_digest_to_dict(f, mode) for f in structure.files],
            "subdirectories": {
                name: self._structure_to_dict(subdir, mode)
                for name, subdir in structure.subdirectories.items()
            }
        }
    
    def _file_digest_to_dict(self, file_digest: FileDigest, mode: str) -> Dict:
        """转换 FileDigest 为字典"""
        result = {
            "metadata": {
                "path": str(file_digest.metadata.path),
                "size": file_digest.metadata.size,
                "modified_time": file_digest.metadata.modified_time.isoformat(),
                "created_time": file_digest.metadata.created_time.isoformat(),
                "file_type": file_digest.metadata.file_type.value,
                "mime_type": file_digest.metadata.mime_type,
                "md5_hash": file_digest.metadata.md5_hash,
                "sha256_hash": file_digest.metadata.sha256_hash
            }
        }
        
        if mode == "full" and hasattr(file_digest, 'full_content') and file_digest.full_content:
            result["full_content"] = file_digest.full_content
        
        if hasattr(file_digest, 'human_readable_summary') and file_digest.human_readable_summary:
            result["summary"] = file_digest.human_readable_summary.to_dict()
        
        if hasattr(file_digest, 'source_code_analysis') and file_digest.source_code_analysis:
            result["source_analysis"] = file_digest.source_code_analysis.to_dict()
        
        return result
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        import math
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"


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
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        prog="directory_digest_refactor",
        description="Directory Digest Tool (Refactored) - 目录知识摘要生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "directory",
        metavar="PATH",
        help="要分析的目录路径"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["full", "framework", "sort"],
        default="framework",
        help="工作模式 (默认: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output",
        choices=["json", "yaml", "md", "txt"],
        default="json",
        help="输出格式 (默认: %(default)s)"
    )
    parser.add_argument(
        "-s", "--save",
        metavar="FILE",
        help="保存输出到文件 (默认: stdout)"
    )
    parser.add_argument(
        "-r", "--rules",
        metavar="FILE",
        help="规则文件路径"
    )
    parser.add_argument(
        "--context-size",
        type=str,
        default="128k",
        help="目标LLM上下文大小 (默认: %(default)s)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10240,
        metavar="MB",
        help="文件大小阈值(MB) (默认: %(default)s)"
    )
    parser.add_argument(
        "--ignore",
        default=".git,__pycache__,*.pyc,*.pyo,node_modules,.venv,venv",
        help="忽略规则，逗号分隔"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细处理信息"
    )
    
    args = parser.parse_args()
    
    # 解析上下文大小
    context_size = parse_context_size(args.context_size)
    
    # 配置转换
    config = {
        'max_file_size': args.max_size * 1024 * 1024,
        'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()],
        'rules_file': Path(args.rules) if args.rules else None,
        'context_size': context_size,
    }
    
    # 创建摘要器
    digest = DirectoryDigest(args.directory, config)
    
    if args.verbose:
        print(f"Analyzing directory: {args.directory}", file=sys.stderr)
        print(f"Mode: {args.mode}, Format: {args.output}", file=sys.stderr)
    
    # 生成摘要
    output = digest.create_digest(args.mode)
    
    # 处理输出
    output_to_stdout = (args.save is None or args.save == '-')
    
    if output_to_stdout:
        try:
            content = FormatConverter.convert(output, args.output, mode=args.mode)
            sys.stdout.write(content)
            if not content.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()
        except BrokenPipeError:
            pass
    else:
        output_path = Path(args.save)
        saved_path = digest.save_output(output, args.output, output_path, args.mode)
        
        if args.verbose:
            stats = output['metadata']['statistics']
            print(f"\nSummary: Files={stats['total_files']}, "
                  f"Time={stats['processing_time']:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
