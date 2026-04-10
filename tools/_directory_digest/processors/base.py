"""
Directory Digest - 处理器模块
包含文本文件、源代码、配置文件等各类文件的处理策略
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

# 导入基础模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import (
    ProcessingStrategy,
    FileType,
    FileMetadata,
    FileDigest,
    STRATEGY_CONFIGS,
)

# 尝试导入分析器模块
try:
    from analyzers.semantics.base import (
        HumanReadableSummary,
        SourceCodeAnalysis,
        SmartTextProcessor,
    )
    from analyzers.semantics.sheets import ConfigAnalysisResult
    ANALYZERS_AVAILABLE = True
except ImportError:
    ANALYZERS_AVAILABLE = False
    # 定义简单的后备数据类
    @dataclass
    class HumanReadableSummary:
        """后备人类可读摘要"""
        title: Optional[str] = None
        line_count: int = 0
        word_count: int = 0
        character_count: int = 0
        encoding: Optional[str] = None
        first_lines: List[str] = field(default_factory=list)
        last_lines: List[str] = field(default_factory=list)
        summary: Optional[str] = None
        
        def to_dict(self) -> Dict:
            return {
                "title": self.title,
                "line_count": self.line_count,
                "word_count": self.word_count,
                "character_count": self.character_count,
                "encoding": self.encoding,
                "first_lines": self.first_lines,
                "last_lines": self.last_lines,
                "summary": self.summary
            }
    
    @dataclass
    class SourceCodeAnalysis:
        """后备源代码分析"""
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
    
    @dataclass
    class ConfigAnalysisResult:
        """后备配置分析结果"""
        keys: List[str] = field(default_factory=list)
        sections: List[str] = field(default_factory=list)
        structure_summary: Optional[str] = None
        
        def to_dict(self) -> Dict:
            return {
                "keys": self.keys[:20],
                "sections": self.sections[:20],
                "structure_summary": self.structure_summary
            }


# ==================== 处理器基类 ====================

class BaseFileProcessor(ABC):
    """文件处理器基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_full_content_size = self.config.get('max_full_content_size', 1024 * 1024)  # 1MB
    
    @abstractmethod
    def can_handle(self, file_digest: FileDigest) -> bool:
        """判断是否能处理此文件"""
        pass
    
    @abstractmethod
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.SUMMARY_ONLY) -> FileDigest:
        """
        处理文件内容
        
        Args:
            file_digest: 文件摘要对象
            content: 文件内容
            mode: 输出模式 ("full", "framework", "sort")
            strategy: 处理策略
            
        Returns:
            更新后的 FileDigest
        """
        pass
    
    def _should_include_full_content(self, file_digest: FileDigest, mode: str) -> bool:
        """判断是否应该包含完整内容"""
        if mode != "full":
            return False
        if file_digest.metadata.size > self.max_full_content_size:
            return False
        return True


# ==================== 文本文件处理器 ====================

class TextFileProcessor(BaseFileProcessor):
    """人类可读文本文件处理器"""
    
    TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.rst', '.tex', '.html', '.htm', '.cmt'}
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        # 优先使用分类阶段确定的类型/策略
        if file_digest.metadata.file_type in (FileType.CRITICAL_DOCS, FileType.REFERENCE_DOCS):
            return True
        if file_digest.metadata.processing_strategy in (
            ProcessingStrategy.SUMMARY_ONLY, 
            ProcessingStrategy.FULL_CONTENT,
            ProcessingStrategy.HEADER_WITH_STATS  # 文档类也使用此策略
        ):
            return True
        
        # 后备：扩展名检查（用于未经过分类阶段的情况）
        suffix = file_digest.metadata.path.suffix.lower()
        return suffix in self.TEXT_EXTENSIONS
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.SUMMARY_ONLY) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            if strategy == ProcessingStrategy.FULL_CONTENT:
                file_digest.full_content = content
        
        # 生成摘要（仅在非 FULL_CONTENT 策略时嵌入，避免信息冗余和Token浪费）
        if strategy != ProcessingStrategy.FULL_CONTENT:
            summary = self._generate_summary(filepath, content, strategy)
            file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _generate_summary(self, filepath: Path, content: str, 
                         strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成文本摘要"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # 基础统计
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', content)
        word_count = len(words)
        
        # 提取标题
        title = self._extract_title(filepath, lines)
        
        # 提取首尾行
        first_lines = lines[:min(10, len(lines))]
        last_lines = lines[-min(5, len(lines)):] if len(lines) > 5 else []
        
        # 根据策略调整
        if strategy == ProcessingStrategy.HEADER_WITH_STATS:
            # 只保留头部
            first_lines = first_lines[:20]
            last_lines = []
        
        # 生成综合摘要
        summary_text = self._generate_summary_text(filepath, lines, strategy)
        
        return HumanReadableSummary(
            title=title,
            line_count=line_count,
            word_count=word_count,
            character_count=len(content),
            encoding=self._detect_encoding(content),
            first_lines=first_lines,
            last_lines=last_lines,
            summary=summary_text
        )
    
    def _extract_title(self, filepath: Path, lines: List[str]) -> Optional[str]:
        """提取标题"""
        # 1. 从文件名
        filename = filepath.stem
        if filename and len(filename) > 2:
            cleaned = filename.replace('_', ' ').replace('-', ' ').title()
            if 3 <= len(cleaned) <= 100:
                return cleaned
        
        # 2. 从内容
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            
            # Markdown 标题
            md_match = re.match(r'^#+\s+(.+)$', line)
            if md_match:
                return md_match.group(1).strip()
            
            # 其他标题模式
            if len(line) > 3 and len(line) < 100:
                if line[0].isalpha() or line[0] in ('【', '[', '*'):
                    return line
        
        return None
    
    def _detect_encoding(self, content: str) -> str:
        """检测编码"""
        try:
            content.encode('utf-8')
            return 'utf-8'
        except UnicodeEncodeError:
            return 'unknown'
    
    def _generate_summary_text(self, filepath: Path, lines: List[str], 
                               strategy: ProcessingStrategy) -> str:
        """生成摘要文本"""
        parts = []
        
        # 基础信息
        suffix = filepath.suffix.lower()
        parts.append(f"File type: {suffix[1:] if suffix else 'text'}")
        parts.append(f"Total lines: {len(lines)}")
        
        # 根据策略添加内容
        if strategy == ProcessingStrategy.FULL_CONTENT:
            parts.append("\n[FULL CONTENT INCLUDED]")
        elif strategy == ProcessingStrategy.SUMMARY_ONLY:
            # 包含前几行预览
            preview = '\n'.join(lines[:min(20, len(lines))])
            if len(lines) > 20:
                preview += f"\n... [and {len(lines) - 20} more lines]"
            parts.append(f"\nPreview:\n{preview}")
        
        return '\n'.join(parts)


# ==================== 源代码处理器 ====================

class SourceCodeProcessor(BaseFileProcessor):
    """源代码文件处理器"""
    
    CODE_EXTENSIONS = {
        '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', 
        '.jsx', '.tsx', '.go', '.rs', '.rb', '.php', '.swift',
        '.sh', '.bash', '.ps1', '.bat', '.cmd', '.css', '.scss'
    }
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        # 优先使用分类阶段确定的类型/策略
        if file_digest.metadata.file_type == FileType.SOURCE_CODE:
            return True
        if file_digest.metadata.processing_strategy == ProcessingStrategy.CODE_SKELETON:
            return True
        
        # 后备：扩展名检查（用于未经过分类阶段的情况）
        suffix = file_digest.metadata.path.suffix.lower()
        return suffix in self.CODE_EXTENSIONS
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.CODE_SKELETON) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            file_digest.full_content = content
        
        # 分析代码（结构分析对代码文件始终有价值）
        analysis = self._analyze_code(filepath, content, strategy)
        file_digest.source_code_analysis = analysis
        
        # 生成简单摘要（仅在非 FULL_CONTENT 策略时嵌入，避免与全文重复）
        if strategy not in (ProcessingStrategy.METADATA_ONLY, ProcessingStrategy.FULL_CONTENT):
            summary = self._generate_code_summary(filepath, content, analysis)
            file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _analyze_code(self, filepath: Path, content: str, 
                      strategy: ProcessingStrategy) -> SourceCodeAnalysis:
        """分析源代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行类型
        blank_lines = 0
        comment_lines = 0
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif self._is_comment_line(stripped, filepath.suffix.lower()):
                comment_lines += 1
            else:
                code_lines += 1
        
        # 提取导入、函数、类
        imports = self._extract_imports(content, filepath.suffix.lower())
        functions = self._extract_functions(content, filepath.suffix.lower())
        classes = self._extract_classes(content, filepath.suffix.lower())
        
        # 语言识别
        language = self._identify_language(filepath)
        
        return SourceCodeAnalysis(
            language=language,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes
        )
    
    def _is_comment_line(self, line: str, suffix: str) -> bool:
        """判断是否为注释行"""
        if suffix in ('.py', '.sh', '.bash', '.rb'):
            return line.startswith('#')
        elif suffix in ('.java', '.cpp', '.c', '.h', '.js', '.ts'):
            return line.startswith('//') or line.startswith('/*')
        return False
    
    def _extract_imports(self, content: str, suffix: str) -> List[str]:
        """提取导入语句"""
        imports = []
        
        if suffix == '.py':
            # Python 导入
            patterns = [
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            ]
        elif suffix in ('.js', '.ts'):
            # JavaScript/TypeScript 导入
            patterns = [
                r'^import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*=\s+require\([\'"]([^\'"]+)[\'"]\)',
            ]
        elif suffix in ('.java', '.cpp', '.c'):
            # Java/C/C++ 导入/包含
            patterns = [
                r'^import\s+([a-zA-Z0-9_\.]+);',
                r'^#include\s+[<"]([^>"]+)[>"]',
            ]
        else:
            patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
        
        return list(set(imports))[:50]  # 去重并限制数量
    
    def _extract_functions(self, content: str, suffix: str) -> List[Dict]:
        """提取函数定义"""
        functions = []
        
        if suffix == '.py':
            # Python 函数
            func_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        elif suffix in ('.js', '.ts'):
            func_pattern = r'^\s*(?:function|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|\s*=\s*(?:\([^)]*\)\s*=>|function))'
        else:
            func_pattern = r'^\s*(?:[a-zA-Z_][a-zA-Z0-9_:\*&]+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                func_name = match.group(1)
                if func_name not in ('if', 'for', 'while', 'switch', 'return'):
                    functions.append({
                        "name": func_name,
                        "line": i + 1
                    })
        
        return functions[:50]
    
    def _extract_classes(self, content: str, suffix: str) -> List[Dict]:
        """提取类定义"""
        classes = []
        
        class_pattern = r'^\s*(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        return classes[:50]
    
    def _identify_language(self, filepath: Path) -> str:
        """识别编程语言"""
        suffix_map = {
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp_header',
            '.c': 'c', '.h': 'c_header',
            '.js': 'javascript', '.jsx': 'jsx',
            '.ts': 'typescript', '.tsx': 'tsx',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.sh': 'shell', '.bash': 'shell',
            '.ps1': 'powershell',
            '.bat': 'batch', '.cmd': 'batch',
            '.css': 'css', '.scss': 'scss', '.less': 'less',
        }
        return suffix_map.get(filepath.suffix.lower(), 'unknown')
    
    def _generate_code_summary(self, filepath: Path, content: str, 
                               analysis: SourceCodeAnalysis) -> HumanReadableSummary:
        """生成代码摘要"""
        lines = content.split('\n')
        
        summary_parts = [
            f"Language: {analysis.language}",
            f"Total lines: {analysis.total_lines}",
            f"Code lines: {analysis.code_lines}",
            f"Comment lines: {analysis.comment_lines}",
        ]
        
        if analysis.functions:
            summary_parts.append(f"Functions: {len(analysis.functions)}")
        if analysis.classes:
            summary_parts.append(f"Classes: {len(analysis.classes)}")
        if analysis.imports:
            summary_parts.append(f"Imports: {len(analysis.imports)}")
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=analysis.total_lines,
            character_count=len(content),
            first_lines=lines[:10],
            summary='\n'.join(summary_parts)
        )


# ==================== 配置文件处理器 ====================

class ConfigFileProcessor(BaseFileProcessor):
    """配置文件处理器"""
    
    CONFIG_EXTENSIONS = {
        '.yaml', '.yml', '.json', '.xml', '.toml', '.ini', 
        '.cfg', '.conf', '.env', '.properties', '.tf', '.tls'
    }
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        # 优先使用分类阶段确定的策略
        if file_digest.metadata.processing_strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
            return True
        
        # 其次检查文件类型
        if file_digest.metadata.file_type == FileType.TEXT_DATA:
            suffix = file_digest.metadata.path.suffix.lower()
            return suffix in self.CONFIG_EXTENSIONS
        
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.STRUCTURE_EXTRACT) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容
        if self._should_include_full_content(file_digest, mode):
            file_digest.full_content = content
        
        # 分析配置结构
        config_analysis = self._analyze_config(filepath, content, strategy)
        
        # 生成摘要（仅在非 FULL_CONTENT 策略时嵌入，全文已包含完整信息）
        if strategy != ProcessingStrategy.FULL_CONTENT:
            summary = self._generate_config_summary(filepath, content, config_analysis, strategy)
            file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _analyze_config(self, filepath: Path, content: str, 
                       strategy: ProcessingStrategy) -> ConfigAnalysisResult:
        """分析配置文件结构"""
        suffix = filepath.suffix.lower()
        keys = []
        sections = []
        structure_summary = None
        
        if suffix in ('.yaml', '.yml'):
            keys, sections, structure_summary = self._analyze_yaml(content)
        elif suffix == '.json':
            keys, sections, structure_summary = self._analyze_json(content)
        elif suffix in ('.ini', '.cfg', '.conf'):
            keys, sections, structure_summary = self._analyze_ini(content)
        elif suffix == '.toml':
            keys, sections, structure_summary = self._analyze_toml(content)
        elif suffix == '.xml':
            keys, sections, structure_summary = self._analyze_xml(content)
        
        return ConfigAnalysisResult(
            keys=keys,
            sections=sections,
            structure_summary=structure_summary
        )
    
    def _analyze_yaml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 YAML 配置"""
        keys = []
        sections = []
        
        # 简单的键提取（不依赖 PyYAML）
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                key_part = line.split(':', 1)[0].strip()
                if key_part and not key_part.startswith('-'):
                    keys.append(key_part)
        
        structure_summary = f"YAML config with {len(set(keys))} top-level keys"
        return list(set(keys)), sections, structure_summary
    
    def _analyze_json(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 JSON 配置"""
        keys = []
        sections = []
        
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                keys = list(data.keys())
                structure_summary = f"JSON object with {len(keys)} keys"
            else:
                structure_summary = f"JSON {type(data).__name__}"
        except json.JSONDecodeError:
            structure_summary = "Invalid JSON"
        
        return keys, sections, structure_summary
    
    def _analyze_ini(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 INI 配置"""
        keys = []
        sections = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                sections.append(line[1:-1])
            elif line and not line.startswith('#') and '=' in line:
                key = line.split('=', 1)[0].strip()
                if key:
                    keys.append(key)
        
        structure_summary = f"INI config with {len(sections)} sections, {len(set(keys))} keys"
        return list(set(keys)), sections, structure_summary
    
    def _analyze_toml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 TOML 配置"""
        # 类似 INI 分析
        return self._analyze_ini(content)
    
    def _analyze_xml(self, content: str) -> Tuple[List[str], List[str], Optional[str]]:
        """分析 XML 配置"""
        keys = []
        sections = []
        
        # 简单标签提取
        tags = re.findall(r'<(\w+)[^>]*>', content)
        keys = list(set(tags))
        
        structure_summary = f"XML with {len(keys)} unique tags"
        return keys, sections, structure_summary
    
    def _generate_config_summary(self, filepath: Path, content: str,
                                 config_analysis: ConfigAnalysisResult,
                                 strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成配置文件摘要"""
        lines = content.split('\n')
        
        summary_parts = [f"Config type: {filepath.suffix[1:]}"]
        
        if config_analysis.structure_summary:
            summary_parts.append(config_analysis.structure_summary)
        
        if config_analysis.sections:
            summary_parts.append(f"Sections: {', '.join(config_analysis.sections[:10])}")
            if len(config_analysis.sections) > 10:
                summary_parts[-1] += f" (+{len(config_analysis.sections) - 10} more)"
        
        if config_analysis.keys:
            summary_parts.append(f"Keys: {', '.join(config_analysis.keys[:15])}")
            if len(config_analysis.keys) > 15:
                summary_parts[-1] += f" (+{len(config_analysis.keys) - 15} more)"
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=len(lines),
            character_count=len(content),
            first_lines=lines[:15],
            summary='\n'.join(summary_parts)
        )


# ==================== 数据文件处理器 ====================

class DataFileProcessor(BaseFileProcessor):
    """数据文件处理器（CSV、TSV、日志等）"""
    
    DATA_EXTENSIONS = {'.csv', '.tsv', '.log', '.out', '.err', '.dat', '.txt'}
    
    def can_handle(self, file_digest: FileDigest) -> bool:
        # 优先使用分类阶段确定的策略和类型
        if file_digest.metadata.file_type == FileType.TEXT_DATA:
            # 明确排除配置文件（应由ConfigFileProcessor处理）
            if file_digest.metadata.processing_strategy == ProcessingStrategy.HEADER_WITH_STATS:
                suffix = file_digest.metadata.path.suffix.lower()
                return suffix in self.DATA_EXTENSIONS
            # 如果没有明确策略，根据扩展名判断，但排除已明确分类的
            if file_digest.metadata.processing_strategy is None:
                suffix = file_digest.metadata.path.suffix.lower()
                if suffix in self.DATA_EXTENSIONS:
                    # 避免与其他处理器冲突
                    return file_digest.metadata.file_type not in (
                        FileType.CRITICAL_DOCS, FileType.REFERENCE_DOCS, FileType.SOURCE_CODE
                    )
        return False
    
    def process(self, file_digest: FileDigest, content: str, mode: str = "framework", 
                strategy: ProcessingStrategy = ProcessingStrategy.HEADER_WITH_STATS) -> FileDigest:
        
        if not content:
            return file_digest
        
        filepath = file_digest.metadata.path
        
        # 处理完整内容（仅在 FULL_CONTENT 策略且模式为 full 时）
        if mode == "full" and strategy == ProcessingStrategy.FULL_CONTENT:
            file_digest.full_content = content
        
        # 生成数据摘要（仅在非 FULL_CONTENT 策略时嵌入，避免信息冗余）
        if strategy != ProcessingStrategy.FULL_CONTENT:
            summary = self._generate_data_summary(filepath, content, strategy)
            file_digest.human_readable_summary = summary
        
        return file_digest
    
    def _generate_data_summary(self, filepath: Path, content: str,
                               strategy: ProcessingStrategy) -> HumanReadableSummary:
        """生成数据文件摘要"""
        lines = content.split('\n')
        line_count = len(lines)
        
        suffix = filepath.suffix.lower()
        
        # 计算统计信息
        stats = self._calculate_data_stats(content, suffix)
        
        # 提取头部
        header_lines = self._extract_header(lines, suffix)
        
        summary_parts = [
            f"Data type: {suffix[1:] if suffix else 'text'}",
            f"Total lines: {line_count}"
        ]
        
        for key, value in stats.items():
            summary_parts.append(f"{key}: {value}")
        
        return HumanReadableSummary(
            title=filepath.name,
            line_count=line_count,
            character_count=len(content),
            first_lines=header_lines,
            summary='\n'.join(summary_parts)
        )
    
    def _calculate_data_stats(self, content: str, suffix: str) -> Dict[str, Any]:
        """计算数据统计信息"""
        stats = {}
        lines = content.split('\n')
        
        if suffix == '.csv':
            # CSV 统计
            non_empty_lines = [l for l in lines if l.strip()]
            if non_empty_lines:
                stats["Data rows (approx)"] = len(non_empty_lines)
                # 估算列数
                first_line = non_empty_lines[0]
                stats["Columns (approx)"] = first_line.count(',') + 1
        
        elif suffix in ('.log', '.out', '.err'):
            # 日志统计
            error_count = sum(1 for l in lines if 'error' in l.lower() or 'exception' in l.lower())
            warning_count = sum(1 for l in lines if 'warn' in l.lower())
            
            if error_count > 0:
                stats["Errors"] = error_count
            if warning_count > 0:
                stats["Warnings"] = warning_count
        
        # 通用统计
        stats["File size"] = f"{len(content)} chars"
        
        return stats
    
    def _extract_header(self, lines: List[str], suffix: str) -> List[str]:
        """提取数据文件头部"""
        header_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 保留注释行
            if stripped and stripped.startswith(('#', '//', '/*', '*', '!')):
                header_lines.append(line)
                continue
            
            # 保留空行直到遇到数据
            if not stripped and header_lines:
                header_lines.append(line)
                continue
            
            # 数据文件的前几行
            if i < 20:
                header_lines.append(line)
            
            # 如果看起来像数据行了，停止
            if i > 10 and stripped and not stripped.startswith(('#', '//', '/*')):
                # 检查是否是纯数据行
                if len(re.findall(r'[a-zA-Z]', stripped)) / len(stripped) < 0.3:
                    break
        
        return header_lines[:30]


# ==================== 重构后的处理器注册表 ====================

class FileProcessorRegistry:
    """文件处理器注册表 - 整合处理流程协调与并行处理支持"""
    
    def __init__(self, 
                 rule_engine=None, 
                 context_manager=None, 
                 stats=None, 
                 config=None):
        """
        初始化处理器注册表
        
        Args:
            rule_engine: 规则引擎实例，用于文件分类
            context_manager: 上下文管理器实例，用于Token分配
            stats: 统计信息字典，用于更新处理统计
            config: 配置字典，包含大小限制等参数
        """
        self.processors: List[BaseFileProcessor] = []
        self.rule_engine = rule_engine
        self.context_manager = context_manager
        self.stats = stats or {}
        self.config = config or {}
        
        # 并行处理配置
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024 * 1024)  # 默认10GB
        
        # 线程锁，用于并行处理时安全更新统计信息
        self._stats_lock = None
        self._context_lock = None
        
    def register(self, processor: BaseFileProcessor):
        """注册处理器"""
        self.processors.append(processor)
    
    def get_processor(self, file_digest: FileDigest) -> Optional[BaseFileProcessor]:
        """获取适合此文件的处理器"""
        for processor in self.processors:
            if processor.can_handle(file_digest):
                return processor
        return None
    
    def process_file(self, file_digest: FileDigest, mode: str = "framework") -> bool:
        """
        处理单个文件 - 整合原 _process_file 逻辑
        
        Args:
            file_digest: 文件摘要对象
            mode: 输出模式 ("full", "framework", "sort")
            
        Returns:
            bool: 处理是否成功
        """
        import sys
        
        filepath = file_digest.metadata.path
        
        try:
            # 1. 获取处理策略（使用规则引擎）
            if self.rule_engine:
                strategy, force_binary = self.rule_engine.classify_file(filepath)
            else:
                # 无规则引擎时的默认策略
                strategy, force_binary = self._default_classify(file_digest)
            
            # 2. 估算token消耗
            if self.rule_engine:
                estimated_tokens = self.rule_engine.estimate_token_usage(filepath, strategy)
            else:
                estimated_tokens = self._estimate_tokens(file_digest, strategy)
            
            # 3. 检查文件大小限制
            if file_digest.metadata.size > self.max_file_size:
                self._update_stats('skipped_large_files')
                self._process_as_binary(file_digest, mode)
                return True
            
            # 4. 检查上下文限制（Token分配）
            if self.context_manager:
                if not self._check_and_allocate_context(estimated_tokens, file_digest, strategy):
                    self._update_stats('skipped_by_context')
                    return False
            
            # 5. 根据策略处理文件
            if force_binary or strategy == ProcessingStrategy.METADATA_ONLY:
                success = self._process_as_binary(file_digest, mode)
                if success:
                    self._update_stats('binary_files')
            else:
                # 获取合适的处理器并执行处理
                processor = self.get_processor(file_digest)
                if processor:
                    # 读取文件内容
                    content = self._read_file_content(filepath)
                    if content is None:
                        self._process_as_binary(file_digest, mode)
                        self._update_stats('binary_files')
                        return True
                    
                    # 执行处理
                    processor.process(file_digest, content, mode, strategy)
                    
                    # 根据文件类型更新统计
                    self._update_stats_by_processor(file_digest, processor)
                    
                    # 在 full 模式下保存完整内容
                    if mode == "full" and strategy == ProcessingStrategy.FULL_CONTENT:
                        file_digest.full_content = content
                else:
                    # 无匹配处理器，作为二进制处理
                    self._process_as_binary(file_digest, mode)
                    self._update_stats('binary_files')
            
            return True
            
        except Exception as e:
            import sys
            print(f"Warning: Error processing file {filepath}: {e}", file=sys.stderr)
            # 出错时作为二进制文件处理，确保不中断流程
            try:
                self._process_as_binary(file_digest, mode)
                self._update_stats('binary_files')
            except:
                pass
            return False
    
    def _check_and_allocate_context(self, estimated_tokens: int, 
                                   file_digest: FileDigest, 
                                   strategy: ProcessingStrategy) -> bool:
        """检查并分配上下文Token，支持策略降级"""
        # 尝试分配Token
        if not self.context_manager.can_allocate(estimated_tokens):
            # Token不足，尝试降级策略
            downgraded_strategy = self.context_manager.downgrade_strategy(strategy)
            if self.rule_engine:
                downgraded_tokens = self.rule_engine.estimate_token_usage(file_digest.metadata.path, downgraded_strategy)
            else:
                downgraded_tokens = self._estimate_tokens(file_digest, downgraded_strategy)
            
            if not self.context_manager.can_allocate(downgraded_tokens):
                return False
            
            # 使用降级后的策略
            strategy = downgraded_strategy
            estimated_tokens = downgraded_tokens
        
        # 分配Token
        file_record = {
            "path": str(file_digest.metadata.path),
            "strategy": strategy.value,
            "estimated_tokens": estimated_tokens,
            "size": file_digest.metadata.size,
        }
        
        return self.context_manager.allocate(estimated_tokens, file_record)
    
    def _process_as_binary(self, file_digest: FileDigest, mode: str) -> bool:
        """处理为二进制文件 - 仅计算哈希"""
        try:
            self._calculate_hashes(file_digest)
            file_digest.metadata.file_type = FileType.BINARY_FILES
            return True
        except Exception as e:
            import sys
            print(f"Warning: Failed to process binary file {file_digest.metadata.path}: {e}", file=sys.stderr)
            return False
    
    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件哈希值"""
        import hashlib
        filepath = file_digest.metadata.path
        
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        file_digest.metadata.md5_hash = md5_hash.hexdigest()
        file_digest.metadata.sha256_hash = sha256_hash.hexdigest()
    
    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """读取文件内容，处理编码问题"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            return raw_content.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    return raw_content.decode('latin-1', errors='ignore')
            except Exception:
                return None
    
    def _default_classify(self, file_digest: FileDigest) -> Tuple[ProcessingStrategy, bool]:
        """默认文件分类（当没有规则引擎时）"""
        suffix = file_digest.metadata.path.suffix.lower()
        size = file_digest.metadata.size
        
        # 二进制扩展名
        binary_exts = {'.exe', '.dll', '.so', '.dylib', '.zip', '.tar', '.gz', 
                      '.jpg', '.png', '.mp3', '.mp4', '.pdf'}
        if suffix in binary_exts:
            return ProcessingStrategy.METADATA_ONLY, True
        
        # 源代码扩展名
        code_exts = {'.py', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs'}
        if suffix in code_exts:
            return ProcessingStrategy.CODE_SKELETON, False
        
        # 文档扩展名
        doc_exts = {'.md', '.txt', '.rst', '.html'}
        if suffix in doc_exts:
            if size < 500 * 1024:
                return ProcessingStrategy.SUMMARY_ONLY, False
            else:
                return ProcessingStrategy.HEADER_WITH_STATS, False
        
        # 默认
        if size > 1024 * 1024:  # > 1MB
            return ProcessingStrategy.METADATA_ONLY, False
        return ProcessingStrategy.SUMMARY_ONLY, False
    
    def _estimate_tokens(self, file_digest: FileDigest, strategy: ProcessingStrategy) -> int:
        """估算Token消耗（当没有规则引擎时）"""
        config = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY])
        
        if strategy == ProcessingStrategy.METADATA_ONLY:
            return int(config.token_estimate * 100)
        
        file_size = file_digest.metadata.size
        if config.max_size and file_size > config.max_size:
            return STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY].token_estimate * 100
        
        estimated_chars = min(file_size, config.max_size or file_size)
        return int(estimated_chars * config.token_estimate)
    
    def _update_stats(self, key: str):
        """更新统计信息（线程安全）"""
        if key in self.stats:
            # 如果启用了并行处理，需要加锁
            if self._stats_lock:
                import threading
                with self._stats_lock:
                    self.stats[key] += 1
            else:
                self.stats[key] += 1
    
    def _update_stats_by_processor(self, file_digest: FileDigest, processor: BaseFileProcessor):
        """根据处理器类型更新统计"""
        processor_type = type(processor).__name__
        
        # 检查是否为关键文档
        is_critical = False
        if processor_type == 'TextFileProcessor':
            filename = file_digest.metadata.path.name.lower()
            critical_patterns = ['readme', 'license', 'copying', 'notice', 'changelog', 'changes', 
                                'contributing', 'install', 'authors', 'news', 'todo', 'roadmap']
            is_critical = any(pattern in filename for pattern in critical_patterns)
        
        # 统计键映射 - 与 FileType 枚举完全一致
        type_mapping = {
            'SourceCodeProcessor': 'source_code',
            'TextFileProcessor': 'critical_docs' if is_critical else 'reference_docs',
            'ConfigFileProcessor': 'text_data',
            'DataFileProcessor': 'text_data'
        }
        
        stat_key = type_mapping.get(processor_type, 'binary_files')
        self._update_stats(stat_key)
        
        # 同时更新文件类型元数据 - 与 FileType 枚举完全一致
        file_type_mapping = {
            'SourceCodeProcessor': FileType.SOURCE_CODE,
            'TextFileProcessor': FileType.CRITICAL_DOCS if is_critical else FileType.REFERENCE_DOCS,
            'ConfigFileProcessor': FileType.TEXT_DATA,
            'DataFileProcessor': FileType.TEXT_DATA
        }
        file_digest.metadata.file_type = file_type_mapping.get(
            processor_type, FileType.BINARY_FILES
        )
        
        # 如果没有匹配的处理器，设置为 UNKNOWN
        if processor_type not in type_mapping:
            file_digest.metadata.file_type = FileType.UNKNOWN
            self._update_stats('unknown')
    
    def process_directory(self, structure: Any, mode: str = "framework", 
                         parallel: bool = False, max_workers: int = 4):
        """
        处理整个目录结构，支持并行处理
        
        Args:
            structure: DirectoryStructure 对象
            mode: 输出模式
            parallel: 是否启用并行处理
            max_workers: 并行工作线程数
        """
        # 收集所有文件
        all_files = []
        
        def collect_files(node):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect_files(subdir)
        
        collect_files(structure)
        
        if parallel and len(all_files) > 10:
            self._process_parallel(all_files, mode, max_workers)
        else:
            self._process_sequential(all_files, mode)
    
    def _process_sequential(self, files: List[FileDigest], mode: str):
        """顺序处理文件"""
        for file_digest in files:
            self.process_file(file_digest, mode)
    
    def _process_parallel(self, files: List[FileDigest], mode: str, max_workers: int):
        """并行处理文件 - 修正版"""
        import concurrent.futures
        import threading
        import sys
        
        # 初始化线程锁
        self._stats_lock = threading.Lock()
        
        # 预筛选：顺序检查Token和大小限制（避免在worker中处理）
        files_to_process = []
        skipped_count = {'skipped_by_context': 0, 'skipped_large_files': 0}
        
        for file_digest in files:
            filepath = file_digest.metadata.path
            
            # 检查大小限制
            if file_digest.metadata.size > self.max_file_size:
                skipped_count['skipped_large_files'] += 1
                self._process_as_binary(file_digest, mode)
                with self._stats_lock:
                    self.stats['binary_files'] += 1
                continue
            
            # 预检查Token（不实际分配，只检查可行性）
            if self.rule_engine:
                strategy, _ = self.rule_engine.classify_file(filepath)
                estimated = self.rule_engine.estimate_token_usage(filepath, strategy)
            else:
                strategy, _ = self._default_classify(file_digest)
                estimated = self._estimate_tokens(file_digest, strategy)
            
            if self.context_manager and not self.context_manager.can_allocate(estimated):
                skipped_count['skipped_by_context'] += 1
                with self._stats_lock:
                    self.stats['skipped_by_context'] += 1
                continue
            
            files_to_process.append((file_digest, strategy, estimated))
        
        # 并行处理筛选后的文件
        processed_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_file_worker, fd, mode, st): (fd, st, est)
                for fd, st, est in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_digest, strategy, estimated = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        processed_files.append((file_digest, strategy, estimated))
                except Exception as e:
                    print(f"Warning: Error in parallel processing {file_digest.metadata.path}: {e}", 
                          file=sys.stderr)
        
        # 在主线程中统一分配Token和更新最终统计
        for file_digest, strategy, estimated in processed_files:
            if self.context_manager:
                file_record = {
                    "path": str(file_digest.metadata.path),
                    "strategy": strategy.value,
                    "estimated_tokens": estimated,
                    "size": file_digest.metadata.size,
                }
                self.context_manager.allocate(estimated, file_record)
        
        # 清理锁
        self._stats_lock = None
    
    def _process_file_worker(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """并行处理工作函数（处理单个文件内容）"""
        try:
            filepath = file_digest.metadata.path
            
            # 检查大小（虽然已经在主线程检查过，但这里作为二次确认）
            if file_digest.metadata.size > self.max_file_size:
                self._process_as_binary(file_digest, mode)
                self._update_stats('skipped_large_files')
                self._update_stats('binary_files')
                return False
            
            # 获取处理器
            processor = self.get_processor(file_digest)
            
            if processor and strategy != ProcessingStrategy.METADATA_ONLY:
                content = self._read_file_content(filepath)
                if content:
                    processor.process(file_digest, content, mode, strategy)
                    self._update_stats_by_processor(file_digest, processor)
                    return True
                else:
                    self._process_as_binary(file_digest, mode)
                    self._update_stats('binary_files')
                    return False
            else:
                self._process_as_binary(file_digest, mode)
                self._update_stats('binary_files')
                return False
                
        except Exception as e:
            import sys
            print(f"Warning: Worker error for {file_digest.metadata.path}: {e}", file=sys.stderr)
            return False


# ==================== 公共 API ====================

def create_default_registry(rule_engine=None, 
                           context_manager=None, 
                           stats=None, 
                           config=None) -> FileProcessorRegistry:
    """
    创建默认处理器注册表
    
    Args:
        rule_engine: 规则引擎实例
        context_manager: 上下文管理器实例  
        stats: 统计信息字典
        config: 配置字典
        
    Returns:
        FileProcessorRegistry: 配置好的注册表实例
    """
    registry = FileProcessorRegistry(
        rule_engine=rule_engine,
        context_manager=context_manager,
        stats=stats,
        config=config
    )
    
    # 按优先级顺序注册（先注册的优先级高）
    registry.register(TextFileProcessor(config))
    registry.register(SourceCodeProcessor(config))
    registry.register(ConfigFileProcessor(config))
    registry.register(DataFileProcessor(config))
    
    return registry


__all__ = [
    # 基类
    'BaseFileProcessor',
    
    # 具体处理器
    'TextFileProcessor',
    'SourceCodeProcessor',
    'ConfigFileProcessor',
    'DataFileProcessor',
    
    # 注册表
    'FileProcessorRegistry',
    'create_default_registry',
]
