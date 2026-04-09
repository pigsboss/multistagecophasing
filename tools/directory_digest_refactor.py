
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

# Extended FormatConverter for sort mode support
class ExtendedFormatConverter(FormatConverter):
    """扩展格式转换器，支持 sort 模式的完整 ls -l 格式输出"""
    
    @staticmethod
    def convert(digest_data: Dict, format: str, mode: str = None) -> str:
        """转换为指定格式，优先处理 sort 模式"""
        if mode == "sort" or digest_data.get('metadata', {}).get('output_mode') == "sort":
            if format in ["json", "yaml"]:
                # Sort 模式下 JSON/YAML 仍输出结构化数据
                import json
                return json.dumps(digest_data, indent=2, ensure_ascii=False)
            else:
                # 文本格式使用 ls -l 风格
                return ExtendedFormatConverter._to_sort_format(digest_data)
        
        # 其他模式调用父类
        return super().convert(digest_data, format, mode) if hasattr(super(), 'convert') else \
               FormatConverter.convert(digest_data, format)
    
    @staticmethod
    def _to_sort_format(digest_data: Dict) -> str:
        """Generate ls -l style output for sort mode - 与原始代码完全一致"""
        lines = []
        root_dir = digest_data.get('metadata', {}).get('root_directory', '.')
        
        lines.append(f"Directory Digest: {root_dir}")
        lines.append(f"Generated: {digest_data.get('metadata', {}).get('generated_at', 'unknown')}")
        lines.append("")
        
        # 类型映射 (type_key: (display_name, type_char))
        type_names = {
            'critical_docs': ('Critical Docs', 'C'),
            'reference_docs': ('Reference Docs', 'R'),
            'source_code': ('Source Code', 'S'),
            'text_data': ('Text Data', 'T'),
            'binary_files': ('Binary Files', 'B'),
            'unknown': ('Unknown', '?')
        }
        
        listings = digest_data.get('file_listings', {})
        
        for type_key, (type_name, type_char) in type_names.items():
            if type_key not in listings or not listings[type_key]:
                continue
            
            files = listings[type_key]
            total_size = sum(f.get('size', 0) for f in files)
            
            lines.append(f"{type_name} ({len(files)} files, {ExtendedFormatConverter._format_size(total_size)})")
            lines.append("-" * 80)
            
            # 类 ls -l 格式：类型 大小 日期 路径
            for f in files[:100]:  # 限制显示数量
                path = f.get('path', 'unknown')
                size = f.get('size_formatted', '0 B')
                modified = f.get('modified', 'unknown')
                
                # 格式化日期 - 与原始代码一致
                if modified != 'unknown':
                    try:
                        dt = datetime.fromisoformat(modified)
                        date_str = dt.strftime("%b %d %H:%M")
                    except:
                        date_str = modified[:16] if len(modified) > 16 else modified
                else:
                    date_str = "unknown"
                
                # 格式：类型 大小 日期 路径
                lines.append(f"{type_char}  {size:>10}  {date_str:>12}  {path}")
            
            if len(files) > 100:
                lines.append(f"... ({len(files) - 100} more files)")
            
            lines.append("")
        
        # 统计摘要 - 与原始代码一致
        stats = digest_data.get('metadata', {}).get('statistics', {})
        lines.append("Summary:")
        lines.append(f"  Total: {stats.get('total_files', 0)} files, "
                    f"{ExtendedFormatConverter._format_size(stats.get('total_size', 0))}")
        lines.append(f"  Critical Docs: {stats.get('critical_docs', 0)}")
        lines.append(f"  Reference Docs: {stats.get('reference_docs', 0)}")
        lines.append(f"  Source Code: {stats.get('source_code', 0)}")
        lines.append(f"  Text Data: {stats.get('text_data', 0)}")
        lines.append(f"  Binary Files: {stats.get('binary_files', 0)}")
        lines.append(f"  Skipped (>limit): {stats.get('skipped_large_files', 0)}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        import math
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"

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
        
        # Initialize statistics
        self.stats = {
            'total_files': 0,
            'critical_docs': 0,
            'reference_docs': 0,
            'source_code': 0,
            'text_data': 0,
            'binary_files': 0,
            'skipped_large_files': 0,
            'skipped_by_context': 0,
            'total_size': 0,
            'processing_time': 0
        }
        
        # Parallel processing config
        self.use_parallel = self.config.get('use_parallel', False)
        self.max_workers = self.config.get('max_workers', os.cpu_count() or 4)
        
        # Initialize advanced components (if available)
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
        """Process all files in directory"""
        if self.use_parallel and len(structure.files) > 10:
            self._process_directory_parallel(structure, mode)
        else:
            for file_digest in structure.files:
                self._process_file(file_digest, mode)
        
        # Recursively process subdirectories
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_directory_parallel(self, structure: DirectoryStructure, mode: str):
        """Process directory using parallel processing"""
        import concurrent.futures
        
        # Collect all files
        all_files = []
        def collect_files(node: DirectoryStructure):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect_files(subdir)
        
        collect_files(structure)
        
        # Process using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_file_safe, file_digest, mode): file_digest
                for file_digest in all_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_digest = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Warning: Error processing file {file_digest.metadata.path}: {e}", file=sys.stderr)
    
    def _process_file_safe(self, file_digest: FileDigest, mode: str):
        """Safe file processing wrapper for parallel execution"""
        try:
            self._process_file(file_digest, mode)
        except Exception as e:
            print(f"Warning: Error processing file {file_digest.metadata.path}: {e}", file=sys.stderr)
    
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
    
    def _collect_all_files_flat(self) -> List[FileDigest]:
        """扁平化收集所有文件"""
        all_files = []
        
        def collect(node: DirectoryStructure):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect(subdir)
        
        if self.structure:
            collect(self.structure)
        return all_files

    def save_output(self, output: Dict, format: str = "json", output_path: Optional[Path] = None, mode: str = None) -> Path:
        """Save output to file"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.lower()
            if ext == "markdown":
                ext = "md"
            output_path = self.root / f"directory_digest_{timestamp}.{ext}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ExtendedFormatConverter to support sort mode
        content = ExtendedFormatConverter.convert(output, format, mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Summary saved to: {output_path}", file=sys.stderr)
        return output_path

    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件的哈希值"""
        import hashlib
        filepath = file_digest.metadata.path
        
        try:
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            with open(filepath, 'rb') as f:
                # 使用64KB缓冲区流式读取
                for chunk in iter(lambda: f.read(65536), b''):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            file_digest.metadata.md5_hash = md5_hash.hexdigest()
            file_digest.metadata.sha256_hash = sha256_hash.hexdigest()
            
        except (OSError, IOError) as e:
            print(f"Warning: Could not read file for hash calculation: {filepath} - {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "read_error"
            file_digest.metadata.sha256_hash = "read_error"
        except Exception as e:
            print(f"Warning: Hash calculation failed for {filepath}: {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "hash_error"
            file_digest.metadata.sha256_hash = "hash_error"

    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """读取文件内容，处理编码问题"""
        try:
            # 首先尝试UTF-8
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    
                    # 尝试常见编码
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            return raw_content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    # 所有编码都失败，使用latin-1并忽略错误
                    return raw_content.decode('latin-1', errors='ignore')
            except Exception:
                return None

    def _build_directory_structure(self, path: Path) -> DirectoryStructure:
        """递归构建目录结构"""
        structure = DirectoryStructure(path=path)
        
        try:
            for item in path.iterdir():
                # 检查是否应该忽略
                if self._should_ignore(item):
                    continue
                
                if item.is_dir():
                    # 递归处理子目录
                    sub_structure = self._build_directory_structure(item)
                    structure.subdirectories[item.name] = sub_structure
                else:
                    # 文件，创建FileDigest
                    stat_result = item.stat()
                    structure.files.append(FileDigest(
                        metadata=FileMetadata(
                            path=item,
                            size=stat_result.st_size,
                            modified_time=datetime.fromtimestamp(stat_result.st_mtime),
                            created_time=datetime.fromtimestamp(stat_result.st_ctime),
                            file_type=FileType.UNKNOWN,
                            mime_type=mimetypes.guess_type(str(item))[0]
                        )
                    ))
                    self.stats['total_files'] += 1
                    self.stats['total_size'] += stat_result.st_size
                    
        except PermissionError:
            print(f"Warning: Permission denied for directory {path}", file=sys.stderr)
        
        return structure

    def _should_ignore(self, path: Path) -> bool:
        """检查路径是否应该被忽略"""
        import fnmatch
        
        for pattern in self.config.get('ignore_patterns', []):
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if pattern.startswith('*') and path.name.endswith(pattern[1:]):
                return True
        
        return False
    
    def _generate_sort_output(self) -> Dict:
        """Generate sorted classification output (ls -l style) - 与原始代码逻辑完全一致"""
        all_files = self._collect_all_files_flat()
        
        # 按类型分组，同时保留完整元数据 - 与原始代码一致
        by_type = {
            FileType.CRITICAL_DOCS.value: [],
            FileType.REFERENCE_DOCS.value: [],
            FileType.SOURCE_CODE.value: [],
            FileType.TEXT_DATA.value: [],
            FileType.BINARY_FILES.value: [],
            FileType.UNKNOWN.value: []
        }
        
        # 按大小分组
        large_files = []      # > 1MB
        medium_files = []     # 100KB - 1MB
        small_files = []      # < 100KB
        
        for f in all_files:
            file_type = f.metadata.file_type.value
            # 使用 hasattr 检查，与原始代码一致
            file_info = {
                'path': str(f.metadata.path.relative_to(self.root)),
                'size': f.metadata.size,
                'size_formatted': self._format_bytes(f.metadata.size),  # 使用 _format_bytes
                'modified': f.metadata.modified_time.isoformat() if hasattr(f.metadata, 'modified_time') and f.metadata.modified_time else 'unknown',
                'type': file_type,
                'is_binary': file_type == FileType.BINARY_FILES.value
            }
            
            if file_type in by_type:
                by_type[file_type].append(file_info)
            else:
                by_type[FileType.UNKNOWN.value].append(file_info)
            
            # 按大小分组
            size = f.metadata.size
            if size > 1024 * 1024:
                large_files.append(file_info)
            elif size > 100 * 1024:
                medium_files.append(file_info)
            else:
                small_files.append(file_info)
        
        # 构建报告 - 与原始代码结构完全一致
        sort_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": "sort",
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary()
            },
            "classification": {},
            "by_size": {
                "large_files": large_files,
                "medium_files": medium_files,
                "small_files": small_files
            },
            "file_listings": {
                k: sorted(v, key=lambda x: x['path']) 
                for k, v in by_type.items() if v
            }
        }
        
        # 为每种类型生成详细信息 - 与原始代码一致
        for type_name, files in by_type.items():
            if not files:
                continue
                
            # 按扩展名分组
            by_ext = {}
            for f in files:
                path = f['path']
                ext = Path(path).suffix.lower() or "(no extension)"
                if ext not in by_ext:
                    by_ext[ext] = []
                by_ext[ext].append(path)
            
            # 计算总大小
            total_size = sum(f['size'] for f in files)
            
            sort_report["classification"][type_name] = {
                "count": len(files),
                "total_size_bytes": total_size,
                "total_size_formatted": self._format_bytes(total_size),
                "extensions": {
                    ext: {
                        "count": len(paths),
                        "files": sorted(paths)[:10],
                        "truncated": len(paths) > 10,
                        "total_count": len(paths)
                    }
                    for ext, paths in sorted(by_ext.items(), key=lambda x: len(x[1]), reverse=True)
                }
            }
        
        # 添加建议 - 与原始代码一致
        recommendations = []
        if large_files:
            recommendations.append(
                f"Found {len(large_files)} large files (>1MB). "
                f"In 'full' mode, use --max-content-size to limit full content output."
            )

        if by_type.get(FileType.UNKNOWN.value, []):
            count = len(by_type[FileType.UNKNOWN.value])
            if count > 5:
                recommendations.append(
                    f"Found {count} unknown type files. Consider reviewing or adding to ignore patterns."
                )
        
        sort_report["recommendations"] = recommendations
        
        return sort_report
    
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
    
    def _format_bytes(self, size_bytes: int) -> str:
        """格式化字节大小为人类可读 - 与原始代码一致"""
        return self._format_size(size_bytes)


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
            content = ExtendedFormatConverter.convert(output, args.output, mode=args.mode)
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
