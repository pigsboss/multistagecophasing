#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directory_digest.py - 目录知识摘要器（LLM上下文优化版）

将文件系统递归"消化"为LLM可理解的上下文摘要，针对128K-256K token窗口优化。

核心特性：
1. 规则驱动的智能分类 - 支持YAML规则文件，可显式定义文件处理策略
2. LLM上下文感知 - 动态分配token，优先重要文件，自动降级策略
3. 多策略处理 - 全量嵌入、摘要、代码骨架、结构提取等
4. 命令行友好 - 无GUI依赖，支持管道处理

规则文件（--rules）支持：
- 显式定义文件分类和策略
- 优先级控制
- 大小限制
- 强制二进制标记

策略枚举：
- full_content: 全量嵌入（仅小文件）
- summary_only: 文本摘要
- code_skeleton: 代码骨架提取
- structure_extract: 结构提取（配置文件）
- header_with_stats: 头部+统计（数据文件）
- metadata_only: 仅元数据（二进制文件）

使用示例：
  directory_digest ./project --rules .digest_rules.yaml
  directory_digest . --context-size 64k --mode full
  directory_digest . --mode sort | less
"""

import os
import sys
import json
import hashlib
import mimetypes
import ast  # Python抽象语法树解析
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import re
from enum import Enum
from collections import Counter

# 依赖检测
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    print("警告: chardet 库未安装，将使用简化的编码检测", file=sys.stderr)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("警告: PyYAML 库未安装，YAML文件解析功能受限", file=sys.stderr)


# ==================== 策略枚举和配置 ====================

class ProcessingStrategy(Enum):
    """文件处理策略枚举"""
    # 主要策略
    FULL_CONTENT = "full_content"              # 全量嵌入（仅限小文件）
    SUMMARY_ONLY = "summary_only"              # 生成文本摘要
    CODE_SKELETON = "code_skeleton"           # 代码骨架（函数/类/导入）
    STRUCTURE_EXTRACT = "structure_extract"    # 提取结构（键/节）
    HEADER_WITH_STATS = "header_with_stats"    # 头部+统计信息
    METADATA_ONLY = "metadata_only"           # 仅元数据


@dataclass
class StrategyConfig:
    """策略配置"""
    token_estimate: float                    # 每字符token估算
    max_size: Optional[int] = None           # 最大适用文件大小
    max_lines: Optional[int] = None          # 最大行数限制
    include_metadata: bool = True            # 是否包含元数据
    
    # 特定策略选项
    include_functions: bool = False
    include_classes: bool = False
    max_keys: Optional[int] = None
    include_stats: bool = False


# 策略配置映射
STRATEGY_CONFIGS: Dict[ProcessingStrategy, StrategyConfig] = {
    ProcessingStrategy.FULL_CONTENT: StrategyConfig(
        token_estimate=0.25,
        max_size=100 * 1024,  # 100KB
    ),
    ProcessingStrategy.SUMMARY_ONLY: StrategyConfig(
        token_estimate=0.05,
        max_lines=50,
    ),
    ProcessingStrategy.CODE_SKELETON: StrategyConfig(
        token_estimate=0.02,
        include_functions=True,
        include_classes=True,
    ),
    ProcessingStrategy.STRUCTURE_EXTRACT: StrategyConfig(
        token_estimate=0.03,
        max_keys=20,
    ),
    ProcessingStrategy.HEADER_WITH_STATS: StrategyConfig(
        token_estimate=0.01,
        max_lines=10,
        include_stats=True,
    ),
    ProcessingStrategy.METADATA_ONLY: StrategyConfig(
        token_estimate=0.001,
    ),
}


# ==================== 数据类型定义 ====================

class FileType(Enum):
    """文件类型枚举"""
    CRITICAL_DOCS = "critical_docs"      # 关键文档
    REFERENCE_DOCS = "reference_docs"    # 参考文档
    SOURCE_CODE = "source_code"          # 源代码
    TEXT_DATA = "text_data"              # 文本数据
    BINARY_FILES = "binary_files"        # 二进制文件
    UNKNOWN = "unknown"                  # 未知类型


@dataclass
class FileMetadata:
    """文件元数据基类"""
    path: Path
    size: int
    modified_time: datetime
    created_time: datetime
    file_type: FileType
    mime_type: Optional[str] = None
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "path": str(self.path),
            "size": self.size,
            "modified_time": self.modified_time.isoformat(),
            "created_time": self.created_time.isoformat(),
            "file_type": self.file_type.value,
            "mime_type": self.mime_type,
            "md5_hash": self.md5_hash,
            "sha256_hash": self.sha256_hash
        }


@dataclass
class HumanReadableSummary:
    """人类可读文本摘要"""
    title: Optional[str] = None
    line_count: int = 0
    word_count: int = 0
    character_count: int = 0
    language: Optional[str] = None
    encoding: Optional[str] = None
    first_lines: List[str] = field(default_factory=list)
    last_lines: List[str] = field(default_factory=list)
    key_sections: List[Tuple[str, str]] = field(default_factory=list)
    summary: Optional[str] = None
    
    # 新增字段
    reading_time_minutes: float = 0.0  # 阅读时间（分钟）
    reading_level: Optional[str] = None  # 阅读难度级别
    key_topics: List[str] = field(default_factory=list)  # 关键主题
    sentiment_score: Optional[float] = None  # 情感分析分数
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "line_count": self.line_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "language": self.language,
            "encoding": self.encoding,
            "first_lines": self.first_lines,
            "last_lines": self.last_lines,
            "key_sections": [{"title": t, "content": c[:200]} for t, c in self.key_sections],
            "summary": self.summary,
            "reading_time_minutes": self.reading_time_minutes,
            "reading_level": self.reading_level,
            "key_topics": self.key_topics[:10],
            "sentiment_score": self.sentiment_score
        }


@dataclass
class SourceCodeAnalysis:
    """源代码分析结果"""
    language: str
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    imports: List[str] = field(default_factory=list)
    functions: List[Dict] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # 新增字段
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)  # 复杂度指标
    style_issues: List[Dict] = field(default_factory=list)            # 代码风格问题
    security_issues: List[Dict] = field(default_factory=list)         # 安全问题
    test_coverage: Optional[float] = None                             # 测试覆盖率
    
    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "imports": self.imports,
            "functions": self.functions[:20],  # 限制数量，避免输出过大
            "classes": self.classes[:20],      # 限制数量
            "global_vars": self.global_vars[:20],
            "constants": self.constants[:20],
            "dependencies": self.dependencies[:20],
            "complexity_metrics": self.complexity_metrics,
            "style_issues": self.style_issues[:10],
            "security_issues": self.security_issues[:10],
            "test_coverage": self.test_coverage
        }


@dataclass
class FileDigest:
    """单个文件摘要"""
    metadata: FileMetadata
    full_content: Optional[str] = None
    human_readable_summary: Optional[HumanReadableSummary] = None
    source_code_analysis: Optional[SourceCodeAnalysis] = None
    
    def to_dict(self, mode: str = "framework") -> Dict:
        """转换为字典，根据模式决定输出内容"""
        result = {
            "metadata": self.metadata.to_dict()
        }
        
        if mode == "full" and self.full_content and self.metadata.file_type == FileType.HUMAN_READABLE:
            result["full_content"] = self.full_content
        elif self.human_readable_summary:
            result["summary"] = self.human_readable_summary.to_dict()
        
        if self.source_code_analysis:
            result["source_analysis"] = self.source_code_analysis.to_dict()
        
        return result


@dataclass
class DirectoryStructure:
    """目录结构表示"""
    path: Path
    files: List[FileDigest] = field(default_factory=list)
    subdirectories: Dict[str, 'DirectoryStructure'] = field(default_factory=dict)
    
    def to_dict(self, mode: str = "framework") -> Dict:
        """转换为嵌套字典结构"""
        return {
            "path": str(self.path),
            "files": [f.to_dict(mode) for f in self.files],
            "subdirectories": {name: d.to_dict(mode) for name, d in self.subdirectories.items()}
        }


# ==================== 文件类型检测器 ====================

class FileTypeDetector:
    """智能文件类型检测器"""
    
    # 扩展名到类型的映射（优先级1）
    EXTENSION_MAPPING = {
        # 参考文档
        FileType.REFERENCE_DOCS: [
            '.md', '.markdown', '.rst', '.html', '.htm'
        ],
        # 源代码（包括脚本）
        FileType.SOURCE_CODE: [
            '.py', '.java', '.cpp', '.c', '.h', '.hpp',
            '.js', '.ts', '.jsx', '.tsx',
            '.go', '.rs', '.rb', '.php', '.swift',
            '.sh', '.bash', '.ps1', '.bat', '.cmd',
            '.css', '.scss', '.less'
        ],
        # 文本数据（配置文件、数据文件、日志等）
        FileType.TEXT_DATA: [
            '.txt', '.log', '.csv', '.tsv',
            '.yaml', '.yml', '.json', '.xml', '.toml', 
            '.ini', '.cfg', '.conf', '.env',
            '.tf', '.tls', '.tpc', '.ker', '.cmt'
        ],
        # 二进制文件
        FileType.BINARY_FILES: [
            '.exe', '.dll', '.so', '.dylib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp',
            '.mp3', '.mp4', '.avi', '.mkv',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.db', '.sqlite',
            '.h5', '.hdf5', '.fits'
        ]
    }
    
    @staticmethod
    def detect_by_extension(filepath: Path) -> Optional[FileType]:
        """通过扩展名检测文件类型"""
        suffix = filepath.suffix.lower()
        
        for file_type, extensions in FileTypeDetector.EXTENSION_MAPPING.items():
            if suffix in extensions:
                return file_type
        
        return None
    
    @staticmethod
    def detect_by_content(filepath: Path) -> FileType:
        """通过内容分析检测文件类型"""
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(4096)
                
                # 检测空字节（二进制文件特征）
                if b'\x00' in sample:
                    return FileType.BINARY_FILES
                
                # 检测可打印字符比例
                printable_count = 0
                for byte in sample:
                    if 32 <= byte <= 126 or byte in (9, 10, 13):
                        printable_count += 1
                
                printable_ratio = printable_count / len(sample) if sample else 0
                
                if printable_ratio < 0.7:
                    return FileType.BINARY_FILES
                
                # 检测源代码特征
                try:
                    decoded = sample.decode('utf-8', errors='ignore')
                    if FileTypeDetector._looks_like_source_code(decoded):
                        return FileType.SOURCE_CODE
                except:
                    pass
                
                # 默认作为文本数据
                return FileType.TEXT_DATA
                
        except Exception:
            return FileType.BINARY_FILES
    
    @staticmethod
    def _looks_like_source_code(content: str) -> bool:
        """判断内容是否像源代码"""
        patterns = [
            r'^\s*import\s+',
            r'^\s*package\s+',
            r'^\s*#include\s+',
            r'^\s*def\s+\w+\s*\(',
            r'^\s*function\s+\w+',
            r'^\s*class\s+\w+',
            r'^\s*public\s+',
            r'^\s*private\s+',
            r'^\s*protected\s+',
            r'^\s*static\s+',
            r'^\s*const\s+',
            r'^\s*let\s+\w+\s*=',
            r'^\s*var\s+\w+\s*=',
            r'^\s*console\.log',
            r'^\s*print\(',
            r'^\s*System\.out\.',
            r'^\s*//',
            r'^\s*/\*',
            r'^\s*\*/',
            r'^\s*#\s*',
        ]
        
        lines = content.split('\n')[:50]
        code_pattern_count = 0
        
        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line):
                    code_pattern_count += 1
                    break
        
        return code_pattern_count >= 3
    
    @staticmethod
    def detect(filepath: Path) -> FileType:
        """综合检测文件类型"""
        type_by_ext = FileTypeDetector.detect_by_extension(filepath)
        if type_by_ext:
            return type_by_ext
        
        return FileTypeDetector.detect_by_content(filepath)


# ==================== 规则引擎 ====================

@dataclass
class FileRule:
    """文件规则定义"""
    name: str
    patterns: List[str]  # glob模式列表
    strategy: ProcessingStrategy
    priority: int = 50
    force_binary: bool = False
    max_size: Optional[int] = None
    comment: Optional[str] = None
    
    def matches(self, filepath: Path) -> bool:
        """检查文件是否匹配此规则"""
        # 检查大小限制
        if self.max_size:
            try:
                if filepath.stat().st_size > self.max_size:
                    return False
            except (OSError, IOError):
                return False
                
        # 检查模式匹配
        for pattern in self.patterns:
            if fnmatch.fnmatch(filepath.name, pattern):
                return True
            # 也检查完整路径匹配
            if fnmatch.fnmatch(str(filepath), pattern):
                return True
        return False


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, rules_file: Optional[Path] = None):
        self.rules: List[FileRule] = []
        self.default_strategy = ProcessingStrategy.METADATA_ONLY
        
        if rules_file and rules_file.exists():
            self.load_rules(rules_file)
        else:
            self.load_default_rules()
        
        # 按优先级降序排序
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def load_default_rules(self):
        """加载内置默认规则"""
        default_rules = [
            # 关键文档（最高优先级）
            FileRule("critical_readme", ["README*", "readme*"], 
                    ProcessingStrategy.FULL_CONTENT, priority=100, max_size=256*1024),
            FileRule("critical_license", ["LICENSE*", "COPYING*", "NOTICE*"], 
                    ProcessingStrategy.FULL_CONTENT, priority=100, max_size=128*1024),
            FileRule("critical_changelog", ["CHANGELOG*", "CHANGES*"], 
                    ProcessingStrategy.SUMMARY_ONLY, priority=95, max_size=256*1024),
            FileRule("critical_contrib", ["CONTRIBUTING*", "INSTALL*", "AUTHORS*", "NEWS*", "TODO*", "ROADMAP*"], 
                    ProcessingStrategy.SUMMARY_ONLY, priority=95, max_size=256*1024),
            
            # Zone.Identifier文件（Windows备用数据流）- 添加这个规则
            FileRule("zone_identifier", ["*:Zone.Identifier"], 
                    ProcessingStrategy.STRUCTURE_EXTRACT, priority=95, max_size=1*1024),
            
            # 二进制文件（高优先级，避免误判）
            FileRule("binary_archives", ["*.gz", "*.bz2", "*.xz", "*.7z", "*.rar", "*.zip", "*.tar"], 
                    ProcessingStrategy.METADATA_ONLY, priority=90, force_binary=True),
            FileRule("media_files", ["*.avi", "*.mp4", "*.mov", "*.wav", "*.mp3", "*.jpg", "*.png"],
                    ProcessingStrategy.METADATA_ONLY, priority=90, force_binary=True),
            FileRule("scientific_binary", ["*.fits", "*.h5", "*.hdf5", "*.bsp", "*.bc"],
                    ProcessingStrategy.METADATA_ONLY, priority=90, force_binary=True),
            FileRule("documents_binary", ["*.pdf", "*.doc", "*.docx", "*.ppt", "*.pptx"],
                    ProcessingStrategy.METADATA_ONLY, priority=90, force_binary=True),
            
            # 参考文档
            FileRule("reference_docs", ["*.md", "*.markdown", "*.rst", "*.tex", "*.html", "*.htm"],
                    ProcessingStrategy.SUMMARY_ONLY, priority=80, max_size=512*1024),
            
            # 源代码
            FileRule("main_source_files", ["main.*", "app.*", "index.*", "__main__.*"],
                    ProcessingStrategy.CODE_SKELETON, priority=75),
            FileRule("source_code", ["*.py", "*.c", "*.cpp", "*.h", "*.java", "*.js", "*.ts", "*.go", "*.rs"],
                    ProcessingStrategy.CODE_SKELETON, priority=70),
            FileRule("shell_scripts", ["*.sh", "*.bash", "*.zsh", "*.ps1", "*.bat", "*.cmd"],
                    ProcessingStrategy.CODE_SKELETON, priority=70),
            FileRule("web_code", ["*.css", "*.scss", "*.less", "*.vue", "*.jsx", "*.tsx"],
                    ProcessingStrategy.CODE_SKELETON, priority=70),
            
            # 文本数据
            FileRule("config_files", ["*.yaml", "*.yml", "*.json", "*.toml", "*.conf", "*.ini", "*.cfg"],
                    ProcessingStrategy.STRUCTURE_EXTRACT, priority=60),
            FileRule("data_files", ["*.csv", "*.tsv", "*.xml", "*.jsonl"],
                    ProcessingStrategy.HEADER_WITH_STATS, priority=50),
            FileRule("log_files", ["*.log", "*.out", "*.err"],
                    ProcessingStrategy.HEADER_WITH_STATS, priority=40),
            FileRule("spice_text_kernels", ["*.tf", "*.tls", "*.tpc", "*.ker"],
                    ProcessingStrategy.STRUCTURE_EXTRACT, priority=40),
            FileRule("text_files", ["*.txt", "*.cmt"],
                    ProcessingStrategy.SUMMARY_ONLY, priority=30, max_size=1024*1024),
        ]
        
        self.rules.extend(default_rules)
    
    def load_rules(self, rules_file: Path):
        """从YAML文件加载规则"""
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"警告: 无法加载规则文件 {rules_file}: {e}", file=sys.stderr)
            print("将使用内置默认规则", file=sys.stderr)
            self.load_default_rules()
            return
        
        # 新格式：直接以分类名为键
        if 'file_classifications' not in data:
            # 假设数据是分类名到模式列表的映射
            for category, patterns in data.items():
                if not patterns:
                    continue
                try:
                    # 将分类名映射到ProcessingStrategy
                    strategy_map = {
                        'critical_docs': ProcessingStrategy.FULL_CONTENT,
                        'reference_docs': ProcessingStrategy.SUMMARY_ONLY,
                        'source_code': ProcessingStrategy.CODE_SKELETON,
                        'text_data': ProcessingStrategy.STRUCTURE_EXTRACT,
                        'binary_files': ProcessingStrategy.METADATA_ONLY
                    }
                    strategy = strategy_map.get(category, ProcessingStrategy.METADATA_ONLY)
                    
                    rule = FileRule(
                        name=category,
                        patterns=patterns,
                        strategy=strategy,
                        priority=100 if category == 'critical_docs' else 
                                90 if category == 'binary_files' else 50,
                        force_binary=(category == 'binary_files'),
                        max_size=None,
                        comment=f"From rules file: {category}"
                    )
                    self.rules.append(rule)
                except Exception as e:
                    print(f"警告: 解析规则分类 {category} 时出错: {e}", file=sys.stderr)
        else:
            # 旧格式：包含'file_classifications'列表
            rule_defs = data.get('file_classifications', [])
            if not rule_defs:
                print("警告: 规则文件中未找到 'file_classifications' 部分", file=sys.stderr)
                self.load_default_rules()
                return
            
            for rule_def in rule_defs:
                try:
                    # 将策略字符串转换为枚举
                    strategy_name = rule_def.get('strategy', 'metadata_only')
                    strategy = ProcessingStrategy(strategy_name)
                    
                    # 解析模式
                    patterns = rule_def.get('patterns', [])
                    if not patterns:
                        continue
                    
                    rule = FileRule(
                        name=rule_def.get('name', 'unnamed'),
                        patterns=patterns,
                        strategy=strategy,
                        priority=rule_def.get('priority', 50),
                        force_binary=rule_def.get('force_binary', False),
                        max_size=rule_def.get('max_size_kb', 0) * 1024 if rule_def.get('max_size_kb') else None,
                        comment=rule_def.get('comment')
                    )
                    self.rules.append(rule)
                except Exception as e:
                    print(f"警告: 解析规则时出错: {rule_def.get('name', 'unnamed')} - {e}", file=sys.stderr)
        
        # 按优先级降序排序
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def classify_file(self, filepath: Path) -> Tuple[ProcessingStrategy, bool]:
        """
        分类文件并返回处理策略
        
        Returns:
            (处理策略, 是否强制为二进制)
        """
        try:
            stat_result = filepath.stat()
        except (OSError, IOError):
            return ProcessingStrategy.METADATA_ONLY, True
        
        # 1. 应用显式规则
        for rule in self.rules:
            if rule.matches(filepath):
                return rule.strategy, rule.force_binary
        
        # 2. 启发式后备规则
        file_size = stat_result.st_size
        
        # 大小启发式：超过1MB大概率不是纯文本
        if file_size > 1024 * 1024:  # 1MB
            return ProcessingStrategy.METADATA_ONLY, True
        
        # 扩展名启发式
        suffix = filepath.suffix.lower()
        
        # 已知文本扩展名
        if suffix in ['.txt', '.md', '.rst']:
            if file_size < 500 * 1024:  # 500KB以下
                return ProcessingStrategy.SUMMARY_ONLY, False
            else:
                return ProcessingStrategy.HEADER_WITH_STATS, False
        
        # 默认：仅元数据
        return ProcessingStrategy.METADATA_ONLY, False
    
    def estimate_token_usage(self, filepath: Path, strategy: ProcessingStrategy) -> int:
        """估算文件使用特定策略的token消耗"""
        try:
            file_size = filepath.stat().st_size
        except (OSError, IOError):
            return 0
        
        config = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY])
        
        # 估算字符数（保守估计）
        if strategy == ProcessingStrategy.METADATA_ONLY:
            return int(config.token_estimate * 100)  # 约100字符的元数据
        
        # 对于内容策略，根据文件大小估算
        if config.max_size and file_size > config.max_size:
            # 文件太大，使用元数据策略
            return STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY].token_estimate * 100
        
        # 字符数估算（保守）
        estimated_chars = min(file_size, config.max_size or file_size)
        return int(estimated_chars * config.token_estimate)


# ==================== 上下文管理器 ====================

class ContextManager:
    """LLM上下文管理器"""
    
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.reserved_tokens = 4000  # 系统提示预留
        self.used_tokens = 0
        self.file_records: List[Dict] = []
        
    @property
    def available_tokens(self) -> int:
        """可用token数量"""
        return self.max_tokens - self.reserved_tokens - self.used_tokens
    
    def can_allocate(self, estimated_tokens: int) -> bool:
        """检查是否能分配指定数量的token"""
        return self.used_tokens + estimated_tokens <= self.available_tokens
    
    def allocate(self, estimated_tokens: int, file_record: Dict) -> bool:
        """分配token并记录文件"""
        if not self.can_allocate(estimated_tokens):
            return False
        
        self.used_tokens += estimated_tokens
        self.file_records.append(file_record)
        return True
    
    def downgrade_strategy(self, current_strategy: ProcessingStrategy) -> ProcessingStrategy:
        """策略降级（当token不足时）"""
        strategy_hierarchy = [
            ProcessingStrategy.FULL_CONTENT,
            ProcessingStrategy.SUMMARY_ONLY,
            ProcessingStrategy.CODE_SKELETON,
            ProcessingStrategy.STRUCTURE_EXTRACT,
            ProcessingStrategy.HEADER_WITH_STATS,
            ProcessingStrategy.METADATA_ONLY,
        ]
        
        try:
            current_index = strategy_hierarchy.index(current_strategy)
            # 降级一级
            if current_index + 1 < len(strategy_hierarchy):
                return strategy_hierarchy[current_index + 1]
        except ValueError:
            pass
        
        return ProcessingStrategy.METADATA_ONLY
    
    def get_summary(self) -> Dict[str, Any]:
        """获取上下文使用摘要"""
        return {
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens,
            "used_tokens": self.used_tokens,
            "available_tokens": self.available_tokens,
            "file_count": len(self.file_records),
            "token_utilization": self.used_tokens / (self.max_tokens - self.reserved_tokens),
        }


# ==================== 复杂度分析器 ====================

class ComplexityAnalyzer:
    """代码复杂度分析器"""
    
    @staticmethod
    def analyze_python(content: str) -> Dict[str, Any]:
        """分析Python代码复杂度"""
        try:
            import ast
            
            tree = ast.parse(content)
            
            # 统计各种复杂度指标
            metrics = {
                "cyclomatic_complexity": 0,
                "function_count": 0,
                "class_count": 0,
                "average_function_length": 0,
                "max_nesting_depth": 0,
                "import_count": 0
            }
            
            # 递归分析AST
            def analyze_node(node, depth: int = 0):
                nonlocal metrics
                
                # 更新最大嵌套深度
                metrics["max_nesting_depth"] = max(metrics["max_nesting_depth"], depth)
                
                # 分析节点类型
                if isinstance(node, ast.If):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.While):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.For):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.Try):
                    metrics["cyclomatic_complexity"] += 1
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth + 1)
                elif isinstance(node, ast.FunctionDef):
                    metrics["function_count"] += 1
                    # 分析函数内的语句数量
                    stmt_count = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                    metrics["average_function_length"] += stmt_count
                elif isinstance(node, ast.ClassDef):
                    metrics["class_count"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics["import_count"] += 1
                else:
                    for child in ast.iter_child_nodes(node):
                        analyze_node(child, depth)
            
            # 开始分析
            analyze_node(tree)
            
            # 计算平均函数长度
            if metrics["function_count"] > 0:
                metrics["average_function_length"] = metrics["average_function_length"] / metrics["function_count"]
            
            # 评估复杂度等级
            complexity_score = metrics["cyclomatic_complexity"]
            if complexity_score < 10:
                metrics["complexity_level"] = "简单"
            elif complexity_score < 20:
                metrics["complexity_level"] = "中等"
            elif complexity_score < 30:
                metrics["complexity_level"] = "复杂"
            else:
                metrics["complexity_level"] = "非常复杂"
            
            return metrics
            
        except SyntaxError:
            return {
                "cyclomatic_complexity": 0,
                "function_count": 0,
                "class_count": 0,
                "average_function_length": 0,
                "max_nesting_depth": 0,
                "import_count": 0,
                "complexity_level": "无法分析"
            }
    
    @staticmethod
    def analyze_generic(content: str) -> Dict[str, Any]:
        """通用代码复杂度分析"""
        lines = content.split('\n')
        
        # 简单的复杂度估算
        metrics = {
            "line_count": len(lines),
            "estimated_complexity": 0
        }
        
        # 基于关键词估算复杂度
        complexity_patterns = [
            (r'\bif\b', 1),
            (r'\belse\b', 1),
            (r'\bfor\b', 2),
            (r'\bwhile\b', 2),
            (r'\btry\b', 1),
            (r'\bcatch\b', 1),
            (r'\bswitch\b', 2),
            (r'\bcase\b', 1)
        ]
        
        for line in lines:
            line_lower = line.lower()
            for pattern, weight in complexity_patterns:
                if re.search(pattern, line_lower):
                    metrics["estimated_complexity"] += weight
        
        # 评估复杂度等级
        if metrics["estimated_complexity"] < 10:
            metrics["complexity_level"] = "简单"
        elif metrics["estimated_complexity"] < 30:
            metrics["complexity_level"] = "中等"
        elif metrics["estimated_complexity"] < 50:
            metrics["complexity_level"] = "复杂"
        else:
            metrics["complexity_level"] = "非常复杂"
        
        return metrics

# ==================== 摘要生成器 ====================

class HumanReadableSummarizer:
    """人类可读文本摘要生成器"""
    
    @staticmethod
    def _analyze_text_metrics(content: str) -> Dict[str, Any]:
        """分析文本指标"""
        # 计算阅读时间（平均阅读速度：200字/分钟）
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', content)
        word_count = len(words)
        reading_time_minutes = word_count / 200.0
        
        # 计算阅读难度（Flesch Reading Ease简化版）
        sentences = re.split(r'[.!?。！？]+', content)
        sentences = [s for s in sentences if s.strip()]
        
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        
        # 简单的阅读难度评估
        if avg_sentence_length < 15:
            reading_level = "容易"
        elif avg_sentence_length < 25:
            reading_level = "中等"
        else:
            reading_level = "困难"
        
        # 提取关键主题（基于高频词）
        from collections import Counter
        # 过滤掉常见停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                     'to', 'the', 'and', 'of', 'in', 'a', 'is', 'that', 'it', 'for', 'on', 'with', 'as'}
        
        # 中英文分别处理
        chinese_words = [w for w in words if re.match(r'[\u4e00-\u9fff]', w)]
        english_words = [w.lower() for w in words if re.match(r'[a-zA-Z]', w)]
        
        # 计算高频词
        chinese_freq = Counter([w for w in chinese_words if w not in stop_words]).most_common(5)
        english_freq = Counter([w for w in english_words if w not in stop_words]).most_common(5)
        
        key_topics = [word for word, _ in chinese_freq + english_freq]
        
        # 简单的情感分析（基于积极/消极词词典）
        positive_words = {'好', '优秀', '成功', '喜欢', '爱', '高兴', '快乐', '开心', '满意'}
        negative_words = {'坏', '失败', '讨厌', '恨', '悲伤', '难过', '生气', '失望', '问题'}
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        total_sentiment = positive_count + negative_count
        if total_sentiment > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment
        else:
            sentiment_score = 0.0
        
        return {
            "reading_time_minutes": reading_time_minutes,
            "reading_level": reading_level,
            "key_topics": key_topics,
            "sentiment_score": sentiment_score,
            "word_count": word_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length
        }

    @staticmethod
    def summarize(filepath: Path, content: str, max_lines: int = 10) -> HumanReadableSummary:
        """
        生成人类可读文本摘要
        
        Args:
            filepath: 文件路径
            content: 文件内容
            max_lines: 提取的最大行数
            
        Returns:
            人类可读文本摘要对象
        """
        if not content:
            return HumanReadableSummary(
                line_count=0,
                word_count=0,
                character_count=0
            )
        
        # ===== 新增：智能截断处理 =====
        # 对于结构化数据文件（SPICE内核、CSV等），使用智能截断
        if SmartTextProcessor.is_structured_data_file(filepath):
            # 检查文件是否过大或明显是结构化数据
            lines = content.split('\n')
            is_large = len(lines) > 50
            
            if is_large:
                # 使用智能处理器提取人类可读部分
                human_content = SmartTextProcessor.extract_human_relevant_content(
                    content, filepath, max_human_lines=50
                )
                
                human_lines = human_content.split('\n')
                
                # 检测编码和语言（基于截断后的内容）
                encoding = HumanReadableSummarizer._detect_encoding(human_content)
                language = HumanReadableSummarizer._detect_language(human_content)
                
                # 提取标题（基于原始内容，因为标题通常在开头）
                title = HumanReadableSummarizer._extract_title(filepath, content, lines)
                
                # 提取关键主题（基于截断后的内容）
                text_metrics = HumanReadableSummarizer._analyze_text_metrics(human_content)
                
                return HumanReadableSummary(
                    title=title,
                    line_count=len(lines),  # 原始总行数
                    word_count=text_metrics.get('word_count', len(human_content.split())),
                    character_count=len(human_content),  # 截断后的字符数
                    language=language,
                    encoding=encoding,
                    first_lines=human_lines[:20],
                    last_lines=[],  # 结构化文件不需要末尾
                    key_sections=[],  # 简化处理
                    summary=f"[Intelligently Truncated] File contains structured data. "
                           f"Original: {len(lines)} lines, preserved {len(human_lines)} lines of metadata. "
                           f"Type: {filepath.suffix} kernel/data file.",
                    reading_time_minutes=text_metrics.get('reading_time_minutes', 0),
                    reading_level='technical',
                    key_topics=text_metrics.get('key_topics', []),
                    sentiment_score=None
                )
        # ===== 新增结束 =====
        
        # 原有的小文件处理逻辑继续执行...
        # 检测编码
        encoding = HumanReadableSummarizer._detect_encoding(content)
        
        # 分割行
        lines = content.split('\n')
        line_count = len(lines)
        
        # 统计基本信息
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', content)
        word_count = len(words)
        
        # 提取标题
        title = HumanReadableSummarizer._extract_title(filepath, content, lines)
        
        # 检测语言
        language = HumanReadableSummarizer._detect_language(content)
        
        # 提取关键章节
        key_sections = HumanReadableSummarizer._extract_key_sections(filepath, content, lines)
        
        # 生成文本摘要
        summary = HumanReadableSummarizer._generate_text_summary(
            filepath, content, lines, max_lines
        )
        
        # 分析文本指标
        text_metrics = HumanReadableSummarizer._analyze_text_metrics(content)
        
        # 提取首尾行
        first_lines = lines[:min(5, len(lines))]
        last_lines = lines[-min(3, len(lines)):] if len(lines) > 3 else []
        
        return HumanReadableSummary(
            title=title,
            line_count=line_count,
            word_count=word_count,
            character_count=len(content),
            language=language,
            encoding=encoding,
            first_lines=first_lines,
            last_lines=last_lines,
            key_sections=key_sections[:10],  # 最多10个关键章节
            summary=summary,
            reading_time_minutes=text_metrics["reading_time_minutes"],
            reading_level=text_metrics["reading_level"],
            key_topics=text_metrics["key_topics"],
            sentiment_score=text_metrics["sentiment_score"]
        )
    
    @staticmethod
    def _detect_encoding(content: str) -> str:
        """检测文本编码"""
        try:
            # 尝试UTF-8
            content.encode('utf-8')
            return 'utf-8'
        except UnicodeEncodeError:
            # 如果没有chardet，使用简化检测
            if not CHARDET_AVAILABLE:
                return 'unknown'
            
            try:
                # 使用chardet检测
                if isinstance(content, str):
                    byte_content = content.encode('latin-1', errors='ignore')
                else:
                    byte_content = content
                
                detection = chardet.detect(byte_content)
                return detection.get('encoding', 'unknown')
            except Exception:
                return 'unknown'
    
    @staticmethod
    def _extract_title(filepath: Path, content: str, lines: List[str]) -> Optional[str]:
        """提取标题"""
        # 1. 从文件名提取（去除扩展名）
        filename = filepath.stem
        if filename and filename != filepath.name:
            # 简单的文件名清理
            cleaned = filename.replace('_', ' ').replace('-', ' ').title()
            if len(cleaned) > 3 and len(cleaned) < 50:
                return cleaned
        
        # 2. 从内容中提取标题
        # 检查前几行是否有明显的标题模式
        title_patterns = [
            # Markdown标题
            (r'^#\s+(.+)$', 1),        # # 标题
            (r'^##\s+(.+)$', 1),       # ## 标题
            (r'^###\s+(.+)$', 1),      # ### 标题
            # 下划线标题
            (r'^(.+)\n=+$', 1),        # 标题\n=====
            (r'^(.+)\n-+$', 1),        # 标题\n-----
            # HTML标题
            (r'^<h1[^>]*>(.+?)</h1>', 1),
            (r'^<title[^>]*>(.+?)</title>', 1),
            # 配置文件的标题部分
            (r'^\[(.+)\]$', 1),        # [section]
            # 文档标题行（通常以大写字母开头且较短）
            (r'^([A-Z][A-Za-z\s]{5,40})$', 1),
            # 中文标题行（通常没有特殊符号，长度适中）
            (r'^([\u4e00-\u9fff\s]{4,30})$', 1),
        ]
        
        # 检查前10行
        for i in range(min(10, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            
            for pattern, group_idx in title_patterns:
                match = re.match(pattern, line)
                if match:
                    title_candidate = match.group(group_idx).strip()
                    if 3 <= len(title_candidate) <= 100:
                        return title_candidate
        
        # 3. 如果还没有找到，使用第一行非空行
        for line in lines:
            line = line.strip()
            if line and len(line) < 80:
                return line[:80]
        
        return None
    
    @staticmethod
    def _detect_language(content: str) -> Optional[str]:
        """检测文本语言"""
        # 统计中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
        chinese_count = len(chinese_chars)
        
        # 统计英文字符和单词
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        english_count = len(english_words)
        
        # 计算比例
        total_meaningful = chinese_count + english_count
        if total_meaningful == 0:
            return None
        
        chinese_ratio = chinese_count / total_meaningful
        
        if chinese_ratio > 0.7:
            return "zh"  # 中文为主
        elif chinese_ratio < 0.3:
            return "en"  # 英文为主
        else:
            return "mixed"  # 混合
    
    @staticmethod
    def _extract_key_sections(filepath: Path, content: str, lines: List[str]) -> List[Tuple[str, str]]:
        """提取关键章节"""
        key_sections = []
        suffix = filepath.suffix.lower()
        
        # 根据文件类型使用不同的章节提取策略
        if suffix in ['.md', '.markdown', '.rst']:
            # Markdown/RST文档
            key_sections = HumanReadableSummarizer._extract_markdown_sections(lines)
        elif suffix == '.cmt':
            # .cmt 文件可能是结构化注释文件
            key_sections = HumanReadableSummarizer._extract_cmt_sections(content, lines)
        elif suffix in ['.json', '.yaml', '.yml']:
            # 结构化数据文件
            key_sections = HumanReadableSummarizer._extract_structured_sections(content, suffix)
        elif suffix in ['.ini', '.cfg', '.conf', '.toml', '.properties']:
            # 配置文件
            key_sections = HumanReadableSummarizer._extract_config_sections(lines)
        elif suffix in ['.xml', '.html', '.htm']:
            # XML/HTML文件
            key_sections = HumanReadableSummarizer._extract_xml_sections(content)
        else:
            # 通用文本文件
            key_sections = HumanReadableSummarizer._extract_general_sections(lines)
        
        return key_sections
    
    @staticmethod
    def _extract_cmt_sections(content: str, lines: List[str]) -> List[Tuple[str, str]]:
        """提取 .cmt 文件的章节"""
        sections = []
        
        # .cmt 文件常见的章节模式
        cmt_patterns = [
            # 带星号的注释块
            (r'^\s*\*+\s*$', 'separator'),
            # 可能的关键词：Section, Chapter, Part
            (r'^\s*([Ss]ection|[Cc]hapter|[Pp]art)\s+([A-Za-z0-9\.\-]+)', 'section_header'),
            # 时间戳或日期标记
            (r'^\s*(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2})', 'timestamp'),
        ]
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # 检测章节开始
            section_found = False
            for pattern, section_type in cmt_patterns:
                if re.match(pattern, line_stripped):
                    # 保存前一章节
                    if current_section and current_content:
                        sections.append((current_section, '\n'.join(current_content[:3])))
                    
                    # 开始新章节
                    if section_type == 'separator':
                        current_section = f"Separator (line {i+1})"
                    else:
                        current_section = f"{line_stripped[:50]}... (line {i+1})"
                    
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                # 添加到当前章节内容
                if line_stripped and len(current_content) < 10:  # 限制内容行数
                    current_content.append(line)
        
        # 保存最后一个章节
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content[:3])))
        
        return sections
    
    @staticmethod
    def _extract_markdown_sections(lines: List[str]) -> List[Tuple[str, str]]:
        """提取Markdown文档的章节"""
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            # 检测Markdown标题
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                # 保存前一章节
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content[:5])))
                
                # 开始新章节
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = f"{'#' * level} {title}"
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content[:5])))
        
        return sections
    
    @staticmethod
    def _extract_structured_sections(content: str, suffix: str) -> List[Tuple[str, str]]:
        """提取结构化数据文件的章节"""
        sections = []
        
        try:
            if suffix == '.json':
                # JSON文件：提取顶层键
                data = json.loads(content)
                if isinstance(data, dict):
                    for key in list(data.keys())[:10]:  # 最多10个键
                        value = data[key]
                        if isinstance(value, (dict, list)):
                            sections.append((f"Key: {key}", f"Type: {type(value).__name__}"))
                        else:
                            sections.append((f"Key: {key}", f"Value: {str(value)[:100]}"))
            
            elif suffix in ['.yaml', '.yml']:
                # YAML文件：提取顶层键
                if not YAML_AVAILABLE:
                    # 如果没有yaml库，尝试简单的文本解析
                    lines = content.split('\n')
                    for line in lines[:20]:  # 只检查前20行
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 简单的键值对检测
                            if ':' in line:
                                key = line.split(':', 1)[0].strip()
                                sections.append((f"YAML Key: {key}", "..."))
                    return sections
                
                # 如果有yaml库，使用它解析
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    for key in list(data.keys())[:10]:
                        value = data[key]
                        if isinstance(value, (dict, list)):
                            sections.append((f"Key: {key}", f"Type: {type(value).__name__}"))
                        else:
                            sections.append((f"Key: {key}", f"Value: {str(value)[:100]}"))
        
        except Exception:
            # 解析失败时返回空列表
            pass
        
        return sections
    
    @staticmethod
    def _extract_config_sections(lines: List[str]) -> List[Tuple[str, str]]:
        """提取配置文件的章节"""
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 检测INI格式的章节 [section]
            ini_match = re.match(r'^\[(.+)\]$', line_stripped)
            if ini_match:
                # 保存前一章节
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content[:5])))
                
                # 开始新章节
                current_section = f"[{ini_match.group(1)}]"
                current_content = []
            
            # 检测TOML格式的章节 [section]
            elif line_stripped.startswith('[') and ']' in line_stripped:
                # 保存前一章节
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content[:5])))
                
                # 开始新章节
                current_section = line_stripped
                current_content = []
            
            elif current_section:
                # 添加键值对到当前章节
                if '=' in line and not line.startswith('#') and not line.startswith(';'):
                    current_content.append(line.strip())
        
        # 保存最后一个章节
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content[:5])))
        
        return sections
    
    @staticmethod
    def _extract_xml_sections(content: str) -> List[Tuple[str, str]]:
        """提取XML/HTML文件的章节"""
        sections = []
        
        # 简单正则匹配XML标签
        # 匹配形如 <tag attr="value">content</tag> 的结构
        xml_patterns = [
            r'<(\w+)[^>]*>.*?</\1>',  # 普通标签
            r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*/>',  # 自闭合标签
            r'<!--(.*?)-->',  # 注释
        ]
        
        for pattern in xml_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches[:10]:  # 最多10个匹配
                if isinstance(match, tuple):
                    tag = match[0] if match else "Unknown"
                    sections.append((f"XML Tag: {tag}", "..."))
                else:
                    sections.append((f"XML Element", str(match)[:100]))
        
        return sections
    
    @staticmethod
    def _extract_general_sections(lines: List[str]) -> List[Tuple[str, str]]:
        """提取通用文本文件的章节"""
        sections = []
        
        # 检测常见的章节标题模式
        section_patterns = [
            # 带有编号的章节
            (r'^(第[一二三四五六七八九十]+章|[0-9]+(\.[0-9]+)*\s+[^\s].*)$', 1),
            # 大写字母开头的行（可能是标题）
            (r'^[A-Z][A-Za-z\s]{10,80}$', 0),
            # 中文标题（无特殊符号，长度适中）
            (r'^[\u4e00-\u9fff\s]{4,50}$', 0),
            # 下划线分隔的标题
            (r'^(.+)\n[-=]{3,}$', 1),
        ]
        
        for i in range(min(50, len(lines))):  # 只检查前50行
            line = lines[i].strip()
            if not line or len(line) < 5:
                continue
            
            for pattern, group_idx in section_patterns:
                match = re.match(pattern, line)
                if match:
                    if group_idx == 0:
                        title = line
                    else:
                        title = match.group(group_idx)
                    
                    # 收集接下来的几行作为内容
                    content_lines = []
                    for j in range(i+1, min(i+6, len(lines))):
                        if lines[j].strip():
                            content_lines.append(lines[j].strip()[:100])
                    
                    if content_lines:
                        sections.append((title, ' '.join(content_lines)[:200]))
                    break
        
        return sections
    
    @staticmethod
    def _generate_text_summary(filepath: Path, content: str, lines: List[str], max_lines: int) -> Optional[str]:
        """生成文本摘要"""
        if not content:
            return None
        
        suffix = filepath.suffix.lower()
        line_count = len(lines)
        
        # 生成基础摘要
        summary_parts = []
        
        # 1. 基本信息
        summary_parts.append(f"文件类型: {suffix[1:] if suffix else '文本文件'}")
        summary_parts.append(f"总行数: {line_count}")
        summary_parts.append(f"总字数: {len(re.findall(r'\b[\w\u4e00-\u9fff]+\b', content))}")
        
        # 2. 根据文件类型添加特定信息
        if suffix in ['.md', '.markdown', '.rst']:
            # 文档文件：统计标题数量
            heading_count = sum(1 for line in lines if re.match(r'^#{1,6}\s+', line.strip()))
            summary_parts.append(f"标题数量: {heading_count}")
        
        elif suffix == '.json':
            # JSON文件：结构信息
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    summary_parts.append(f"顶层键数量: {len(data)}")
                    keys = list(data.keys())[:5]
                    summary_parts.append(f"主要键: {', '.join(keys)}")
            except Exception:
                summary_parts.append("JSON结构: 无效或无法解析")
        
        elif suffix in ['.yaml', '.yml']:
            # YAML文件：结构信息
            if not YAML_AVAILABLE:
                summary_parts.append("YAML结构: 未安装PyYAML，跳过详细解析")
            else:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    if isinstance(data, dict):
                        summary_parts.append(f"顶层键数量: {len(data)}")
                        keys = list(data.keys())[:5]
                        summary_parts.append(f"主要键: {', '.join(keys)}")
                except Exception:
                    summary_parts.append("YAML结构: 无效或无法解析")
        
        # 3. 内容摘要（前几行和后几行）
        if line_count > max_lines * 2:
            first_part = '\n'.join(lines[:max_lines])
            last_part = '\n'.join(lines[-max_lines:])
            summary_parts.append(f"\n开头部分（前{max_lines}行）：")
            summary_parts.append(first_part)
            summary_parts.append(f"\n结尾部分（后{max_lines}行）：")
            summary_parts.append(last_part)
        else:
            summary_parts.append("\n完整内容：")
            summary_parts.append(content[:500] + ("..." if len(content) > 500 else ""))
        
        return '\n'.join(summary_parts)


class SourceCodeAnalyzer:
    """源代码分析器"""
    
    @staticmethod
    def analyze(filepath: Path, content: str) -> SourceCodeAnalysis:
        """
        分析源代码文件
        
        Args:
            filepath: 文件路径
            content: 文件内容
            
        Returns:
            源代码分析结果
        """
        if not content:
            return SourceCodeAnalysis(
                language="unknown",
                total_lines=0,
                code_lines=0,
                comment_lines=0,
                blank_lines=0
            )
        
        # 根据文件扩展名选择分析方法
        suffix = filepath.suffix.lower()
        
        if suffix == '.py':
            return SourceCodeAnalyzer._analyze_python(content)
        elif suffix in ['.java', '.cpp', '.c', '.h', '.hpp', '.cc']:
            return SourceCodeAnalyzer._analyze_c_family(content, suffix)
        elif suffix in ['.js', '.ts', '.jsx', '.tsx']:
            return SourceCodeAnalyzer._analyze_javascript(content, suffix)
        elif suffix in ['.go', '.rs', '.rb', '.php', '.sh', '.bash', '.zsh', '.sql']:
            # 对于这些语言，使用通用分析
            return SourceCodeAnalyzer._analyze_generic(content, suffix)
        else:
            # 通用源代码分析
            return SourceCodeAnalyzer._analyze_generic(content, suffix)
    
    @staticmethod
    def _analyze_python(content: str) -> SourceCodeAnalysis:
        """分析Python源代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计空白行和注释行
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
        
        # 使用AST分析Python代码
        imports = []
        functions = []
        classes = []
        global_vars = []
        constants = []
        
        try:
            tree = ast.parse(content)
            
            # 遍历AST节点
            for node in ast.walk(tree):
                # 导入语句
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        if module:
                            imports.append(f"{module}.{alias.name}")
                        else:
                            imports.append(alias.name)
                
                # 函数定义
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": len(node.args.args),
                        "defaults": len(node.args.defaults) if node.args.defaults else 0,
                        "docstring": ast.get_docstring(node),
                        "line": node.lineno,
                        "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')]
                    })
                
                # 类定义
                elif isinstance(node, ast.ClassDef):
                    # 获取基类
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            # 处理 module.Class 形式的基类
                            bases.append(ast.unparse(base))
                    
                    classes.append({
                        "name": node.name,
                        "bases": bases,
                        "docstring": ast.get_docstring(node),
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                
                # 全局变量（模块顶层的赋值）
                elif isinstance(node, ast.Assign):
                    # 检查是否在模块级别（没有父节点或父节点是模块）
                    if not hasattr(node, 'parent') or isinstance(node.parent, ast.Module):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                var_name = target.id
                                # 尝试获取值（简化的常量检测）
                                try:
                                    # 如果是简单的常量（Python 3.8+ 使用 ast.Constant）
                                    if isinstance(node.value, ast.Constant):
                                        constants.append(var_name)
                                    else:
                                        global_vars.append(var_name)
                                except Exception:
                                    global_vars.append(var_name)
        
        except SyntaxError as e:
            # AST解析失败，使用正则表达式进行基本分析
            return SourceCodeAnalyzer._analyze_python_with_regex(content)
        except Exception as e:
            # 其他错误，返回基本分析
            print(f"警告: Python AST分析失败: {e}", file=sys.stderr)
            return SourceCodeAnalyzer._analyze_python_with_regex(content)
        
        # 计算代码行数
        code_lines = total_lines - blank_lines - comment_lines
        
        # 代码复杂度分析
        complexity_metrics = ComplexityAnalyzer.analyze_python(content)
        
        # 代码风格检查（简化版）
        style_issues = []
        
        # 检查行长度
        for i, line in enumerate(lines, 1):
            if len(line) > 100:  # PEP 8建议79字符，这里放宽到100
                style_issues.append({
                    "type": "行过长",
                    "line": i,
                    "message": f"第{i}行超过100个字符",
                    "severity": "警告"
                })
        
        # 检查导入顺序
        import_lines = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('import') or stripped.startswith('from'):
                import_lines.append((i, stripped))
        
        if len(import_lines) > 1:
            # 检查是否有标准库和非标准库混排
            std_imports = []
            third_party_imports = []
            
            for line_num, import_stmt in import_lines:
                if any(pattern in import_stmt for pattern in [
                    'os', 'sys', 'math', 're', 'json', 'datetime'
                ]):
                    std_imports.append((line_num, import_stmt))
                else:
                    third_party_imports.append((line_num, import_stmt))
            
            if std_imports and third_party_imports:
                last_std = std_imports[-1][0]
                first_third = third_party_imports[0][0]
                
                if first_third < last_std:
                    style_issues.append({
                        "type": "导入顺序",
                        "line": first_third,
                        "message": "第三方导入应该在标准库导入之后",
                        "severity": "建议"
                    })
        
        # 更新返回的SourceCodeAnalysis对象，添加新增字段
        return SourceCodeAnalysis(
            language="python",
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes,
            global_vars=global_vars,
            constants=constants,
            dependencies=SourceCodeAnalyzer._extract_python_dependencies(imports),
            complexity_metrics=complexity_metrics,
            style_issues=style_issues,
            security_issues=[],  # 可以在此添加安全检查
            test_coverage=None   # 可以在此添加测试覆盖率分析
        )
    
    @staticmethod
    def _analyze_python_with_regex(content: str) -> SourceCodeAnalysis:
        """使用正则表达式分析Python代码（AST解析失败时的后备方案）"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
        
        # 使用正则表达式提取信息
        imports = []
        functions = []
        classes = []
        
        # 提取导入语句
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        ]
        
        for line in lines:
            stripped = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, stripped)
                if match:
                    imports.append(match.group(1))
                    break
        
        # 提取函数定义
        func_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "args": 0,  # 无法从正则表达式获取参数数量
                    "line": i + 1,
                    "docstring": None
                })
        
        # 提取类定义
        class_pattern = r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        for i, line in enumerate(lines):
            match = re.match(class_pattern, line.strip())
            if match:
                classes.append({
                    "name": match.group(1),
                    "bases": [],
                    "line": i + 1,
                    "methods": 0
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="python",
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes,
            global_vars=[],
            constants=[],
            dependencies=SourceCodeAnalyzer._extract_python_dependencies(imports)
        )
    
    @staticmethod
    def _extract_python_dependencies(imports: List[str]) -> List[str]:
        """从导入语句中提取依赖包名"""
        dependencies = set()
        
        for imp in imports:
            # 移除子模块，只保留顶级包名
            parts = imp.split('.')
            if parts:
                # 过滤掉Python标准库和常见内置模块
                top_level = parts[0]
                if top_level and not top_level.startswith('_'):
                    # 常见标准库，这里只是一个简化的列表
                    stdlib = {
                        'os', 'sys', 'math', 're', 'json', 'datetime', 'time',
                        'collections', 'itertools', 'functools', 'typing',
                        'pathlib', 'hashlib', 'random', 'statistics', 'decimal'
                    }
                    if top_level not in stdlib:
                        dependencies.add(top_level)
        
        return list(dependencies)
    
    @staticmethod
    def _analyze_c_family(content: str, suffix: str) -> SourceCodeAnalysis:
        """分析C/C++/Java家族代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                blank_lines += 1
                continue
            
            # 处理块注释
            if in_block_comment:
                comment_lines += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue
            
            if stripped.startswith('//'):
                comment_lines += 1
            elif stripped.startswith('/*'):
                comment_lines += 1
                in_block_comment = True
                if '*/' in stripped and stripped.index('*/') > stripped.index('/*'):
                    in_block_comment = False
            elif stripped.startswith('#'):
                # C预处理器指令，不算注释
                pass
        
        # 提取导入/包含语句
        imports = []
        if suffix in ['.cpp', '.c', '.h', '.hpp', '.cc']:
            pattern = r'^#include\s+[<"]([^>"]+)[>"]'
        else:  # Java
            pattern = r'^import\s+([a-zA-Z0-9_.]+)'
        
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                imports.append(match.group(1))
        
        # 提取函数/方法定义
        functions = []
        
        # C/C++/Java函数模式
        func_patterns = [
            # 返回类型 函数名(参数)
            r'^\s*(?:[a-zA-Z_][a-zA-Z0-9_:<>]*\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            # 构造函数
            r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{'
        ]
        
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    # 过滤掉常见的关键字和类型名
                    if func_name not in ['if', 'for', 'while', 'switch', 'return', 'int', 'void', 'float', 'double']:
                        functions.append({
                            "name": func_name,
                            "line": i + 1
                        })
                    break
        
        # 提取类定义
        classes = []
        class_pattern = r'^\s*(?:public|private|protected|class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        language_map = {
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c_header',
            '.hpp': 'cpp_header',
            '.cc': 'cpp'
        }
        
        return SourceCodeAnalysis(
            language=language_map.get(suffix, "c_family"),
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes,
            global_vars=[],
            constants=[],
            dependencies=[]
        )
    
    @staticmethod
    def _analyze_javascript(content: str, suffix: str) -> SourceCodeAnalysis:
        """分析JavaScript/TypeScript代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                blank_lines += 1
                continue
            
            # 处理块注释
            if in_block_comment:
                comment_lines += 1
                if '*/' in stripped:
                    in_block_comment = False
                continue
            
            if stripped.startswith('//'):
                comment_lines += 1
            elif stripped.startswith('/*'):
                comment_lines += 1
                in_block_comment = True
                if '*/' in stripped and stripped.index('*/') > stripped.index('/*'):
                    in_block_comment = False
        
        # 提取导入语句
        imports = []
        import_patterns = [
            r'^import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
            r'^const\s+.*=\s+require\([\'"]([^\'"]+)[\'"]\)',
            r'^require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for line in lines:
            for pattern in import_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    imports.append(match.group(1))
                    break
        
        # 提取函数定义
        functions = []
        func_patterns = [
            r'^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^\s*(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:\([^)]*\)\s*=>|function)',
            r'^\s*(?:export\s+)?class\s+[^{]+\s*{[^}]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{'
        ]
        
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.search(pattern, line)
                if match:
                    functions.append({
                        "name": match.group(1),
                        "line": i + 1
                    })
                    break
        
        # 提取类定义
        classes = []
        class_pattern = r'^\s*(?:export\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        language = "javascript"
        if suffix in ['.ts', '.tsx']:
            language = "typescript"
        elif suffix == '.jsx':
            language = "jsx"
        
        return SourceCodeAnalysis(
            language=language,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=classes,
            global_vars=[],
            constants=[],
            dependencies=[]
        )
    
    
    @staticmethod
    def _analyze_generic(content: str, suffix: str) -> SourceCodeAnalysis:
        """通用源代码分析"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        
        # 根据后缀确定注释模式
        comment_patterns = []
        if suffix in ['.py', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.rb', '.pl', '.pm']:
            comment_patterns = ['#']
        elif suffix in ['.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp', '.cc', '.go', '.rs', '.swift', '.kt', '.scala']:
            comment_patterns = ['//', '/*']
        elif suffix in ['.php']:
            comment_patterns = ['//', '#', '/*']
        else:
            comment_patterns = ['//', '#', '/*']
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
                continue
            
            # 检查注释
            is_comment = False
            for pattern in comment_patterns:
                if stripped.startswith(pattern):
                    comment_lines += 1
                    is_comment = True
                    break
            if is_comment:
                continue
            
            # 检查块注释结束
            if '*/' in stripped:
                comment_lines += 1
        
        # 提取可能的函数定义（通用模式）
        functions = []
        func_patterns = [
            r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python风格
            r'^\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # JS/PHP风格
            r'^\s*(?:public|private|protected)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('  # Java/C风格
        ]
        
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    functions.append({
                        "name": match.group(1),
                        "line": i + 1
                    })
                    break
        
        # 提取可能的类定义
        classes = []
        class_patterns = [
            r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^\s*struct\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for i, line in enumerate(lines):
            for pattern in class_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    classes.append({
                        "name": match.group(1),
                        "line": i + 1
                    })
                    break
        
        code_lines = total_lines - blank_lines - comment_lines
        
        # 根据后缀猜测语言
        language_guess = suffix[1:] if suffix else "unknown"
        
        return SourceCodeAnalysis(
            language=language_guess,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=[],
            functions=functions,
            classes=classes,
            global_vars=[],
            constants=[],
            dependencies=[]
        )


    def _extract_config_structure(self, content: str, suffix: str) -> List[Dict]:
        """从配置文件中提取结构信息"""
        keys = []
        try:
            if suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    for i, key in enumerate(list(data.keys())[:20]):
                        keys.append({"name": f"key:{key}", "line": i+1, "type": "config_key"})
            elif suffix == '.json':
                data = json.loads(content)
                if isinstance(data, dict):
                    for i, key in enumerate(list(data.keys())[:20]):
                        keys.append({"name": f"json:{key}", "line": i+1, "type": "config_key"})
            elif suffix in ['.ini', '.cfg', '.conf', '.env', '.rc']:
                for i, line in enumerate(content.split('\n')[:50]):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith(';'):
                        match = re.match(r'^\s*([^=;#\s]+)\s*=', line)
                        if match:
                            keys.append({"name": match.group(1).strip(), "line": i+1, "type": "config_key"})
            elif suffix in ['.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd']:
                # 提取函数定义
                func_pattern = r'^\s*(?:function\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)'
                for i, line in enumerate(content.split('\n')):
                    match = re.match(func_pattern, line)
                    if match:
                        keys.append({"name": match.group(1), "line": i+1, "type": "function"})
        except Exception:
            pass
        return keys

# ==================== 内容分析器 ====================

class ContentAnalyzer:
    """基于内容特征的动态分类器"""
    
    @staticmethod
    def calculate_entropy(content: str) -> float:
        """计算香农熵"""
        if not content:
            return 0.0
        from collections import Counter
        import math
        
        char_counts = Counter(content)
        total = len(content)
        entropy = 0.0
        
        for count in char_counts.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0
        
        return entropy
    
    @staticmethod
    def detect_structure(content: str) -> Dict[str, Any]:
        """
        检测内容结构特征
        返回: {
            'is_tabular': bool,
            'is_natural_language': bool,
            'is_code': bool,
            'tabular_ratio': float,
            'natural_language_ratio': float,
            'structure_type': str,  # 'natural_language', 'tabular_data', 'mixed', 'code', 'config'
            'entropy': float
        }
        """
        lines = content.split('\n')
        total_lines = len(lines)
        if total_lines == 0:
            return {'is_tabular': False, 'is_natural_language': False, 'is_code': False,
                   'structure_type': 'empty', 'entropy': 0}
        
        sample = content[:10000]  # 限制分析范围提高性能
        sample_lines = lines[:min(100, total_lines)]
        
        # 1. Tabular Data 检测
        delimiter_pattern = re.compile(r'[,;\t|]{2,}')
        numeric_pattern = re.compile(r'^[\s\d\.\-\+eE,;\t|]+$')
        delimiter_lines = 0
        numeric_lines = 0
        
        for line in sample_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if delimiter_pattern.search(stripped) or stripped.count(',') > 3:
                delimiter_lines += 1
            if numeric_pattern.match(stripped) and len(stripped) > 5:
                numeric_lines += 1
        
        tabular_ratio = (delimiter_lines + numeric_lines) / max(len(sample_lines), 1)
        is_tabular = tabular_ratio > 0.3
        
        # 2. Natural Language 检测
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sample.lower())
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sample))
        
        word_diversity = 0
        stop_words_found = 0
        if words:
            unique = len(set(words))
            word_diversity = unique / len(words)
            stop_words = {'the', 'and', 'is', 'of', 'to', 'in', 'that', 'have', 'it', 
                         '的', '了', '是', '在', '有', '和', '就', '不', '人'}
            stop_words_found = sum(1 for w in words if w in stop_words)
        
        sentence_endings = sample.count('.') + sample.count('?') + sample.count('!') + \
                          sample.count('。') + sample.count('？') + sample.count('！')
        
        is_natural_language = (
            (sentence_endings > 3 and word_diversity > 0.1 and stop_words_found > 5) or
            (chinese_chars > 50 and sentence_endings > 0)
        )
        nl_ratio = min(1.0, (sentence_endings / max(len(sample_lines), 1)) * 2)
        
        # 3. Code 检测
        code_patterns = [
            r'\b(def|class|function|if|for|while|return|import|from|#include|package|public|private)\b',
            r'[{}\[\]()]+',
            r'^(\s{4}|\t)',  # 缩进
        ]
        code_matches = sum(1 for p in code_patterns if re.search(p, sample[:2000], re.M))
        is_code = code_matches > 3
        
        # 4. Config 检测（键值对结构）
        config_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*.+$',  # key = value
            r'^[^:]+:\s+.+$',  # YAML风格
        ]
        config_lines = sum(1 for line in sample_lines if any(re.match(p, line.strip()) for p in config_patterns))
        is_config = (config_lines / max(len(sample_lines), 1)) > 0.3 and not is_code
        
        # 5. 确定结构类型
        if is_tabular and is_natural_language:
            structure_type = 'mixed'
        elif is_tabular:
            structure_type = 'tabular_data'
        elif is_code:
            structure_type = 'code'
        elif is_config:
            structure_type = 'config'
        elif is_natural_language:
            structure_type = 'natural_language'
        else:
            structure_type = 'unknown'
        
        return {
            'is_tabular': is_tabular,
            'is_natural_language': is_natural_language,
            'is_code': is_code,
            'tabular_ratio': tabular_ratio,
            'natural_language_ratio': nl_ratio,
            'structure_type': structure_type,
            'entropy': ContentAnalyzer.calculate_entropy(content)
        }
    
    @staticmethod
    def classify_file(content: str, filepath: Path) -> Dict[str, Any]:
        """完整文件分类"""
        # 首先尝试解码
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = content.decode('latin-1', errors='ignore')
            except:
                return {'type': 'binary', 'strategy': 'metadata_only', 'reason': 'undecodable'}
        
        analysis = ContentAnalyzer.detect_structure(content)
        
        # 决策映射
        type_mapping = {
            'natural_language': ('human_readable', 'embed_with_limit'),
            'code': ('source_code', 'skeleton_or_full'),
            'config': ('config_script', 'structure_summary'),
            'tabular_data': ('tabular_data', 'header_only'),
            'mixed': ('mixed_document', 'extract_header'),
            'unknown': ('unknown_text', 'first_lines')
        }
        
        file_type, strategy = type_mapping.get(analysis['structure_type'], ('unknown', 'first_lines'))
        
        return {
            'type': file_type,
            'strategy': strategy,
            'structure_type': analysis['structure_type'],
            'entropy': analysis['entropy'],
            'metrics': analysis
        }

# ==================== 智能文本处理器 ====================

class SmartTextProcessor:
    """
    智能文本处理器 - 提取人类关心的内容，截断机器数据结构
    
    用于处理SPICE内核、大型CSV、XML数据等文件：
    - 保留文件头（注释、说明、元数据）
    - 检测数据区开始（结构化列表、表格）
    - 用统计摘要替代具体数据
    """
    
    # 文件扩展名到处理策略的映射（移除 .bsp 和 .bc）
    STRUCTURED_EXTENSIONS = {'.tf', '.tls', '.tpc', '.ker', '.csv', '.dat', '.xml', '.cmt'}
    
    @staticmethod
    def is_structured_data_file(filepath: Path) -> bool:
        """判断是否为结构化数据文件（需要智能截断）"""
        return filepath.suffix.lower() in SmartTextProcessor.STRUCTURED_EXTENSIONS
    
    @staticmethod
    def extract_human_relevant_content(content: str, filepath: Path, max_human_lines: int = 50) -> str:
        """
        从面向机器的文件中提取人类可读部分
        
        Args:
            content: 文件完整内容
            filepath: 文件路径
            max_human_lines: 最大保留行数
            
        Returns:
            截断后的人类可读内容
        """
        lines = content.split('\n')
        suffix = filepath.suffix.lower()
        
        # 特殊处理 SPICE 内核文件
        if suffix in ['.tf', '.tls', '.tpc', '.ker', '.bsp', '.bc']:
            return SmartTextProcessor._extract_spice_content(lines, filepath, content)
        
        # 处理 CSV 文件
        if suffix == '.csv':
            return SmartTextProcessor._extract_csv_content(lines, filepath)
        
        # 处理 .cmt 文件
        if suffix == '.cmt':
            return SmartTextProcessor._extract_cmt_content(lines, filepath)
        
        # 处理其他结构化数据文件
        if suffix in ['.xml', '.dat', '.json']:
            return SmartTextProcessor._extract_generic_structured(lines, suffix)
        
        # 默认：返回前 N 行
        return '\n'.join(lines[:max_human_lines])
    
    @staticmethod
    def _extract_spice_content(lines: List[str], filepath: Path, full_content: str) -> str:
        """
        提取 SPICE 文件的人类可读部分
        
        SPICE 文件结构：
        - 开头：版权声明、文件描述、版本历史（人类可读）
        - \\begintext ... \\begindata 之间：文本注释（人类可读）
        - \\begindata 之后：机器数据结构（应截断）
        """
        header_lines = []
        in_text_block = False
        data_started = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测 SPICE 标记
            if '\\begintext' in stripped:
                in_text_block = True
                header_lines.append(line)  # 保留标记行以便观察
                continue
            elif '\\begindata' in stripped:
                data_started = True
                header_lines.append(line)   # 保留标记行
                break
            
            # 收集注释行（SPICE常用注释格式）
            if (stripped.startswith('C') or 
                stripped.startswith('*') or 
                stripped.startswith('/*') or
                stripped.startswith('#') or
                stripped.startswith('CC') or
                in_text_block):
                header_lines.append(line)
            elif not data_started and not stripped.startswith('\\'):
                # 文件开头的非数据行（描述性文字）
                # 排除纯数字行（可能是数据）
                if not re.match(r'^\s*[\d\.\-\+eE\s]+$', stripped):
                    header_lines.append(line)
        
        # 构建结果
        result_lines = header_lines[:50]  # 限制头部行数
        
        # 添加数据结构摘要
        total_lines = len(lines)
        if data_started or len(header_lines) < total_lines:
            result_lines.append(f"\n[DATA SECTION TRUNCATED]")
            result_lines.append(f"File type: SPICE Kernel ({filepath.suffix})")
            result_lines.append(f"Total lines: {total_lines}")
            result_lines.append(f"Preserved: {len(header_lines)} lines of metadata/comments")
            result_lines.append(f"Truncated: ~{total_lines - len(header_lines)} lines of structured data")
            
            # 尝试提取关键元数据
            # 查找时间范围
            year_pattern = r'(\d{4})[\s/-](\d{1,2})[\s/-](\d{1,2})'
            years = re.findall(r'\b(19|20)\d{2}\b', full_content[:10000])
            if years:
                unique_years = sorted(set(years))
                if len(unique_years) > 1:
                    result_lines.append(f"Time coverage: {unique_years[0]} to {unique_years[-1]}")
            
            # 查找版本信息
            version_match = re.search(r'Version\s*[:=]?\s*([\d\.]+)', full_content[:5000], re.I)
            if version_match:
                result_lines.append(f"Version: {version_match.group(1)}")
            
            # 查找NAIF ID或对象名称
            naif_match = re.search(r'NAIF\s+(\w+)', full_content[:2000], re.I)
            if naif_match:
                result_lines.append(f"NAIF reference: {naif_match.group(0)}")
        
        return '\n'.join(result_lines)
    
    @staticmethod
    def _extract_csv_content(lines: List[str], filepath: Path) -> str:
        """提取CSV文件的头部和统计信息"""
        if not lines:
            return ""
        
        # 保留头部（列名）和前几个数据样本
        header_lines = []
        
        # 第一行通常是列名
        if lines:
            header_lines.append(lines[0])
            header_lines.append("")  # 空行分隔
        
        # 收集前5行数据作为样本
        sample_count = 0
        for line in lines[1:]:
            if line.strip() and sample_count < 5:
                header_lines.append(line)
                sample_count += 1
            elif sample_count >= 5:
                break
        
        # 添加统计信息
        total_rows = len([l for l in lines if l.strip()])
        header_lines.append(f"\n[CSV DATA SUMMARY]")
        header_lines.append(f"Total rows: ~{total_rows}")
        header_lines.append(f"Columns: {lines[0].count(',') + 1 if lines else 'unknown'}")
        header_lines.append(f"Displayed: 5 sample rows")
        header_lines.append(f"Note: Full data truncated to save context window")
        
        return '\n'.join(header_lines)
    
    @staticmethod
    def _extract_cmt_content(lines: List[str], filepath: Path) -> str:
        """提取 .cmt 文件的人类可读部分"""
        header_lines = []
        
        # .cmt 文件通常是注释文件，保留所有注释行和有意义的内容
        for line in lines[:100]:  # 最多检查100行
            stripped = line.strip()
            
            # 保留非空行（包括注释和各种标记）
            if stripped:
                header_lines.append(line)
            
            # 如果检测到可能的数据区域（连续多行无意义的字符串），停止
            # 简单的启发式规则：连续3行超过80字符且无标点符号
            if len(header_lines) > 10:
                recent_lines = header_lines[-3:]
                if all(len(l.strip()) > 80 and 
                       not any(c in l for c in '.,;:!?') 
                       for l in recent_lines if l.strip()):
                    # 可能是数据区域，停止收集
                    header_lines = header_lines[:-3]
                    header_lines.append("[DATA SECTION TRUNCATED]")
                    break
        
        # 如果文件很大，添加统计信息
        if len(lines) > len(header_lines):
            header_lines.append(f"\n[FILE SUMMARY]")
            header_lines.append(f"Total lines: {len(lines)}")
            header_lines.append(f"Preserved: {len(header_lines)} lines")
            header_lines.append(f"Truncated: {len(lines) - len(header_lines)} lines")
            header_lines.append(f"File type: Comment file (.cmt)")
        
        return '\n'.join(header_lines[:100])  # 限制总行数
    
    @staticmethod
    def _extract_generic_structured(lines: List[str], suffix: str) -> str:
        """通用结构化数据提取"""
        header_lines = []
        data_line_count = 0
        
        # 启发式规则：检测数据行（纯数字、重复模式等）
        for i, line in enumerate(lines[:100]):  # 只检查前100行
            stripped = line.strip()
            
            if not stripped:
                header_lines.append(line)
                continue
            
            # 检测是否为纯数据行（无字母或极少字母）
            alpha_ratio = sum(1 for c in stripped if c.isalpha()) / len(stripped) if stripped else 0
            
            if alpha_ratio < 0.1 and i > 5:  # 前5行之后的高比例非文本行视为数据
                data_line_count += 1
                if data_line_count > 3:  # 连续3行数据则截断
                    remaining = len(lines) - i
                    header_lines.append(f"\n... [{remaining} lines of {suffix} data truncated] ...")
                    break
            else:
                data_line_count = 0
                header_lines.append(line)
        
        return '\n'.join(header_lines)


# ==================== 格式转换器 ====================

class OutputFormats(Enum):
    """支持的所有输出格式"""
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "md"
    HTML = "html"
    TOML = "toml"
    PLAINTEXT = "txt"


class FormatConverter:
    """格式转换器，支持多种输出格式"""
    
    @staticmethod
    def convert(digest_data: Dict, format: str, mode: str = None) -> str:
        """转换为指定格式"""
        # 如果是 sort 模式，强制使用专门的 sort 格式
        if mode == "sort" or digest_data.get('metadata', {}).get('output_mode') == "sort":
            return FormatConverter._to_sort_format(digest_data)
        
        if format == "json":
            return json.dumps(digest_data, indent=2, ensure_ascii=False)
        elif format == "markdown" or format == "md":
            return FormatConverter._to_markdown(digest_data)
        elif format == "yaml":
            return FormatConverter._to_yaml(digest_data)
        elif format == "html":
            return FormatConverter._to_html(digest_data)
        elif format == "toml":
            return FormatConverter._to_toml(digest_data)
        elif format == "txt" or format == "text":
            return FormatConverter._to_plaintext(digest_data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @staticmethod
    def _to_markdown(digest_data: Dict) -> str:
        """转换为Markdown格式"""
        metadata = digest_data.get('metadata', {})
        structure = digest_data.get('structure', {})
        
        # 构建Markdown内容
        lines = []
        
        # 1. 标题
        lines.append(f"# 目录摘要报告")
        lines.append("")
        
        # 2. 元数据
        lines.append("## 元数据")
        lines.append("")
        lines.append(f"- **生成时间**: {metadata.get('generated_at', '未知')}")
        lines.append(f"- **根目录**: `{metadata.get('root_directory', '未知')}`")
        lines.append(f"- **输出模式**: {metadata.get('output_mode', '未知')}")
        lines.append("")
        
        # 3. 统计信息
        stats = metadata.get('statistics', {})
        lines.append("## 统计信息")
        lines.append("")
        lines.append(f"- **总文件数**: {stats.get('total_files', 0)}")
        lines.append(f"- **人类可读文件**: {stats.get('human_readable', 0)}")
        lines.append(f"- **源代码文件**: {stats.get('source_code', 0)}")
        lines.append(f"- **二进制文件**: {stats.get('binary', 0)}")
        lines.append(f"- **总大小**: {FormatConverter._format_size(stats.get('total_size', 0))}")
        lines.append(f"- **处理时间**: {stats.get('processing_time', 0):.2f}秒")
        lines.append("")
        
        # 4. 目录结构
        lines.append("## 目录结构")
        lines.append("")
        
        # 递归生成目录树
        def generate_tree(node: Dict, level: int = 0, prefix: str = "") -> List[str]:
            """递归生成目录树"""
            tree_lines = []
            indent = "  " * level
            
            # 当前目录路径（相对根目录）
            rel_path = node.get('path', '')
            root_path = metadata.get('root_directory', '')
            if rel_path.startswith(root_path):
                rel_path = rel_path[len(root_path):].lstrip('/\\')
            
            dir_name = Path(node.get('path', '')).name
            if level == 0:
                dir_name = rel_path or "."
            
            # 目录标题
            if level == 0:
                tree_lines.append(f"### {dir_name}")
                tree_lines.append("")
            else:
                tree_lines.append(f"{prefix}**{dir_name}**/")
            
            # 当前目录下的文件
            files = node.get('files', [])
            for i, file_data in enumerate(files):
                file_meta = file_data.get('metadata', {})
                file_path = Path(file_meta.get('path', ''))
                file_name = file_path.name
                
                # 文件图标和类型标识
                file_type = file_meta.get('file_type', 'unknown')
                type_icon = FormatConverter._get_file_type_icon(file_type)
                
                # 文件大小
                file_size = FormatConverter._format_size(file_meta.get('size', 0))
                
                # 是否是最后一个文件/目录
                is_last_file = i == len(files) - 1 and not node.get('subdirectories')
                
                # 前缀和连接符
                if level > 0:
                    if is_last_file and not node.get('subdirectories'):
                        file_prefix = prefix + "└── "
                    else:
                        file_prefix = prefix + "├── "
                else:
                    file_prefix = "- "
                
                tree_lines.append(f"{file_prefix}{type_icon} `{file_name}` ({file_size})")
                
                # 文件摘要信息
                if file_data.get('summary') or file_data.get('source_analysis'):
                    summary_info = FormatConverter._get_file_summary_markdown(file_data)
                    if summary_info:
                        tree_lines.append(f"{prefix}    {summary_info}")
            
            # 子目录
            subdirs = node.get('subdirectories', {})
            subdir_names = sorted(subdirs.keys())
            
            for j, subdir_name in enumerate(subdir_names):
                subdir_node = subdirs[subdir_name]
                is_last_subdir = j == len(subdir_names) - 1
                
                # 子目录前缀
                if level > 0:
                    if is_last_subdir:
                        subdir_prefix = prefix + "└── "
                        next_prefix = prefix + "    "
                    else:
                        subdir_prefix = prefix + "├── "
                        next_prefix = prefix + "│   "
                else:
                    subdir_prefix = "- "
                    next_prefix = "  "
                
                # 递归处理子目录
                subdir_lines = generate_tree(subdir_node, level + 1, next_prefix)
                
                # 替换第一行的前缀
                if subdir_lines and level == 0:
                    # 根目录下的子目录，使用标题格式
                    tree_lines.append(f"\n### {subdir_name}")
                    tree_lines.extend(subdir_lines[1:])  # 跳过第一行（已经在上面处理了）
                elif subdir_lines:
                    # 非根目录的子目录
                    first_line = subdir_lines[0]
                    if "**" in first_line:
                        # 这是一个目录标题行，替换前缀
                        tree_lines.append(f"{subdir_prefix}{first_line}")
                    else:
                        tree_lines.append(f"{subdir_prefix}{first_line}")
                    tree_lines.extend(subdir_lines[1:])
            
            return tree_lines
        
        # 生成目录树
        tree_lines = generate_tree(structure, 0, "")
        lines.extend(tree_lines)
        lines.append("")
        
        # 5. 文件详情（如果有的话）
        all_files = FormatConverter._collect_all_files(structure)
        if all_files:
            lines.append("## 文件详情")
            lines.append("")
            
            for i, file_data in enumerate(all_files[:50]):  # 限制显示50个文件详情
                file_meta = file_data.get('metadata', {})
                file_path = Path(file_meta.get('path', ''))
                
                # 获取相对于根目录的路径
                root_path = metadata.get('root_directory', '')
                full_path = str(file_path)
                if full_path.startswith(root_path):
                    rel_path = full_path[len(root_path):].lstrip('/\\')
                else:
                    rel_path = full_path
                
                lines.append(f"### {i+1}. `{rel_path}`")
                lines.append("")
                
                # 基本信息
                lines.append("#### 基本信息")
                lines.append("")
                lines.append(f"- **类型**: {file_meta.get('file_type', 'unknown')}")
                lines.append(f"- **大小**: {FormatConverter._format_size(file_meta.get('size', 0))}")
                lines.append(f"- **修改时间**: {file_meta.get('modified_time', '未知')}")
                lines.append(f"- **MD5**: `{file_meta.get('md5_hash', '无')}`")
                lines.append("")
                
                # 摘要信息
                summary = file_data.get('summary')
                if summary:
                    lines.append("#### 摘要")
                    lines.append("")
                    lines.append(f"```text")
                    lines.append(f"{summary.get('summary', '无摘要')[:500]}")  # 限制长度
                    lines.append(f"```")
                    lines.append("")
                
                # 源代码分析
                source_analysis = file_data.get('source_analysis')
                if source_analysis:
                    lines.append("#### 源代码分析")
                    lines.append("")
                    lines.append(f"- **语言**: {source_analysis.get('language', 'unknown')}")
                    lines.append(f"- **总行数**: {source_analysis.get('total_lines', 0)}")
                    lines.append(f"- **代码行**: {source_analysis.get('code_lines', 0)}")
                    lines.append(f"- **注释行**: {source_analysis.get('comment_lines', 0)}")
                    lines.append(f"- **空白行**: {source_analysis.get('blank_lines', 0)}")
                    
                    imports = source_analysis.get('imports', [])
                    if imports:
                        lines.append(f"- **导入项**: {', '.join(imports[:10])}")
                        if len(imports) > 10:
                            lines.append(f"  （还有 {len(imports) - 10} 个）")
                    
                    functions = source_analysis.get('functions', [])
                    if functions:
                        lines.append(f"- **函数**: {len(functions)}个")
                        for func in functions[:5]:
                            func_name = func.get('name', '未知')
                            func_line = func.get('line', '?')
                            lines.append(f"  - `{func_name}` (第{func_line}行)")
                        if len(functions) > 5:
                            lines.append(f"  （还有 {len(functions) - 5} 个函数）")
                    
                    classes = source_analysis.get('classes', [])
                    if classes:
                        lines.append(f"- **类**: {len(classes)}个")
                        for cls in classes[:5]:
                            cls_name = cls.get('name', '未知')
                            cls_line = cls.get('line', '?')
                            lines.append(f"  - `{cls_name}` (第{cls_line}行)")
                        if len(classes) > 5:
                            lines.append(f"  （还有 {len(classes) - 5} 个类）")
                    lines.append("")
                
                # 完整内容（如果是全量模式且内容较小）
                full_content = file_data.get('full_content')
                if full_content and len(full_content) < 5000:  # 只显示小于5000字符的内容
                    lines.append("#### 内容预览")
                    lines.append("")
                    lines.append("```")
                    # 显示前1000个字符
                    preview = full_content[:1000]
                    lines.append(preview)
                    if len(full_content) > 1000:
                        lines.append(f"...\n（完整内容共 {len(full_content)} 字符）")
                    lines.append("```")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # 6. 尾部信息
        lines.append("## 报告信息")
        lines.append("")
        lines.append(f"本报告由 **Directory Digest Tool** 生成。")
        lines.append(f"生成配置：模式=`{metadata.get('output_mode', '未知')}`")
        
        # 添加时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"生成完成时间：{timestamp}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"
    
    @staticmethod
    def _get_file_type_icon(file_type: str) -> str:
        """获取文件类型图标"""
        icons = {
            'critical_docs': '🔑',
            'reference_docs': '📚',
            'source_code': '💻',
            'text_data': '📄',
            'binary_files': '📦',
            'unknown': '❓',
        }
        return icons.get(file_type, '📄')
    
    @staticmethod
    def _get_file_summary_markdown(file_data: Dict) -> str:
        """获取文件的Markdown摘要"""
        summary = file_data.get('summary')
        source_analysis = file_data.get('source_analysis')
        
        if source_analysis:
            language = source_analysis.get('language', '')
            total_lines = source_analysis.get('total_lines', 0)
            functions = len(source_analysis.get('functions', []))
            classes = len(source_analysis.get('classes', []))
            
            info_parts = []
            if language:
                info_parts.append(f"语言: {language}")
            if total_lines:
                info_parts.append(f"行数: {total_lines}")
            if functions:
                info_parts.append(f"函数: {functions}")
            if classes:
                info_parts.append(f"类: {classes}")
            
            if info_parts:
                return f"*({', '.join(info_parts)})*"
        
        elif summary:
            line_count = summary.get('line_count', 0)
            word_count = summary.get('word_count', 0)
            info_parts = []
            
            if line_count:
                info_parts.append(f"行数: {line_count}")
            if word_count:
                info_parts.append(f"字数: {word_count}")
            
            if info_parts:
                return f"*({', '.join(info_parts)})*"
        
        return ""
    
    @staticmethod
    def _collect_all_files(node: Dict) -> List[Dict]:
        """递归收集所有文件"""
        files = []
        
        # 添加当前节点的文件
        files.extend(node.get('files', []))
        
        # 递归处理子目录
        for subdir in node.get('subdirectories', {}).values():
            files.extend(FormatConverter._collect_all_files(subdir))
        
        return files
    
    @staticmethod
    def _to_yaml(digest_data: Dict) -> str:
        """转换为YAML格式"""
        import yaml
        try:
            return yaml.dump(digest_data, allow_unicode=True, default_flow_style=False)
        except Exception:
            # 如果YAML库不可用，返回JSON格式
            return json.dumps(digest_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _to_html(digest_data: Dict) -> str:
        """转换为HTML格式"""
        # 简单实现：将Markdown转换为HTML的基本结构
        md_content = FormatConverter._to_markdown(digest_data)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>目录摘要报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto; }}
    </style>
</head>
<body>
{FormatConverter._markdown_to_html(md_content)}
</body>
</html>"""
        return html_content
    
    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """将Markdown转换为HTML（简单实现）"""
        html = markdown_text
        
        # 简单的Markdown到HTML转换
        html = html.replace('# ', '<h1>').replace('\n#', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n##', '</h2>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n###', '</h3>\n<h3>')
        
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('`', '<code>').replace('`', '</code>')
        html = html.replace('- ', '<li>').replace('\n-', '</li>\n<li>')
        
        # 处理代码块
        import re
        html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
        
        return html
    
    @staticmethod
    def _to_toml(digest_data: Dict) -> str:
        """转换为TOML格式"""
        # 简单实现：返回JSON，因为TOML库可能需要额外安装
        return f"# TOML格式暂未完全实现，以下是JSON表示\n{json.dumps(digest_data, indent=2, ensure_ascii=False)}"
    
    @staticmethod
    def _to_sort_format(digest_data: Dict) -> str:
        """类 ls -l 格式输出"""
        lines = []
        root_dir = digest_data.get('metadata', {}).get('root_directory', '.')
        
        lines.append(f"Directory Digest: {root_dir}")
        lines.append(f"Generated: {digest_data.get('metadata', {}).get('generated_at', 'unknown')}")
        lines.append("")
        
        # 类型映射
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
            
            lines.append(f"{type_name} ({len(files)} files, {FormatConverter._format_size(total_size)})")
            lines.append("-" * 80)
            
            # 类 ls -l 格式：类型-权限 大小 日期 时间 路径
            for f in files[:100]:  # 限制显示数量
                path = f.get('path', 'unknown')
                size = f.get('size_formatted', '0 B')
                modified = f.get('modified', 'unknown')
                
                # 简化格式：- 大小 日期 路径
                # 或者更详细的：-rw-r--r-- 1 user group 4.2K Jan 15 14:32 path/to/file
                if modified != 'unknown':
                    try:
                        dt = datetime.fromisoformat(modified)
                        date_str = dt.strftime("%b %d %H:%M")
                    except:
                        date_str = modified[:16]
                else:
                    date_str = "unknown"
                
                # 格式：类型 大小 日期 路径
                lines.append(f"{type_char}  {size:>10}  {date_str:>12}  {path}")
            
            if len(files) > 100:
                lines.append(f"... ({len(files) - 100} more files)")
            
            lines.append("")
        
        # 统计摘要
        stats = digest_data.get('metadata', {}).get('statistics', {})
        lines.append("Summary:")
        lines.append(f"  Total: {stats.get('total_files', 0)} files, "
                    f"{FormatConverter._format_size(stats.get('total_size', 0))}")
        lines.append(f"  Critical Docs: {stats.get('critical_docs', 0)}")
        lines.append(f"  Reference Docs: {stats.get('reference_docs', 0)}")
        lines.append(f"  Source Code: {stats.get('source_code', 0)}")
        lines.append(f"  Text Data: {stats.get('text_data', 0)}")
        lines.append(f"  Binary Files: {stats.get('binary_files', 0)}")
        lines.append(f"  Skipped (>limit): {stats.get('skipped_large_files', 0)}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _to_plaintext(digest_data: Dict) -> str:
        """转换为纯文本格式"""
        # 使用Markdown但去除格式
        md_content = FormatConverter._to_markdown(digest_data)
        
        # 去除Markdown格式
        import re
        plaintext = md_content
        
        # 去除标题标记
        plaintext = re.sub(r'#+\s+', '', plaintext)
        # 去除粗体标记
        plaintext = plaintext.replace('**', '')
        # 去除代码标记
        plaintext = plaintext.replace('`', '')
        # 去除列表标记
        plaintext = re.sub(r'^[│├└──\s]+', '', plaintext, flags=re.MULTILINE)
        
        return plaintext


# ==================== 主摘要生成器 ====================

class DirectoryDigest:
    """目录摘要生成器"""
    
    def __init__(self, 
                 root_path: Union[str, Path],
                 config: Optional[Dict] = None):
        """
        初始化摘要生成器
        
        Args:
            root_path: 根目录路径
            config: 配置字典
        """
        self.root = Path(root_path).resolve()
        self.config = config or {}
        
        # 默认配置
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024 * 1024)  # 10GB
        self.ignore_patterns = self.config.get('ignore_patterns', [
            '*.pyc', '*.pyo', '*.so', '*.dll', '__pycache__', 
            '.git', '.svn', '.hg', '.DS_Store', '*.swp', '*.swo'
        ])
        self.use_parallel = self.config.get('use_parallel', False)
        self.max_workers = self.config.get('max_workers', os.cpu_count() or 4)
        
        # 新配置项
        self.rules_file = self.config.get('rules_file')
        self.context_size = self.config.get('context_size', 128000)
        self.rule_engine = RuleEngine(self.rules_file)
        self.context_manager = ContextManager(self.context_size)
        
        # 原有分析器
        self.file_type_detector = FileTypeDetector()
        self.human_summarizer = HumanReadableSummarizer()
        self.source_analyzer = SourceCodeAnalyzer()
        
        # 存储结果
        self.structure: Optional[DirectoryStructure] = None
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
    
    def _generate_sort_output(self) -> Dict:
        """生成分类排序输出（增强版，保留文件元数据用于 ls -l 格式）"""
        all_files = self._collect_all_files_flat()
        
        # 按类型分组，同时保留完整元数据
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
            file_info = {
                'path': str(f.metadata.path.relative_to(self.root)),
                'size': f.metadata.size,
                'size_formatted': self._format_bytes(f.metadata.size),
                'modified': f.metadata.modified_time.isoformat() if hasattr(f.metadata, 'modified_time') else 'unknown',
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
        
        # 构建报告
        sort_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": "sort",
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary()  # 添加这一行
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
        
        # 为每种类型生成详细信息
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
                        "files": sorted(paths)[:10],  # 只显示前10个
                        "truncated": len(paths) > 10,
                        "total_count": len(paths)
                    }
                    for ext, paths in sorted(by_ext.items(), key=lambda x: len(x[1]), reverse=True)
                }
            }
        
        # 添加建议
        recommendations = []
        if large_files:
            recommendations.append(
                f"Found {len(large_files)} large files (>1MB). "
                f"In 'full' mode, use --max-content-size to limit full content output."
            )
        # 删除对 CONFIG_SCRIPT 的引用，因为新的分类体系中已不存在这个类型
        # 配置文件现在属于 TEXT_DATA，脚本属于 SOURCE_CODE

        if by_type.get(FileType.UNKNOWN.value, []):
            count = len(by_type[FileType.UNKNOWN.value])
            if count > 5:
                recommendations.append(
                    f"Found {count} unknown type files. Consider reviewing or adding to ignore patterns."
                )
        
        sort_report["recommendations"] = recommendations
        
        return sort_report

    def _format_bytes(self, size_bytes: int) -> str:
        """格式化字节大小为人类可读"""
        if size_bytes == 0:
            return "0 B"
        import math
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"

    def create_digest(self, mode: str = "framework") -> Dict:
        """
        创建目录摘要
        
        Args:
            mode: 输出模式，"framework"（框架）、"full"（全量）或 "sort"（分类排序）
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
                    # 文件，先创建空的FileDigest，稍后填充
                    structure.files.append(FileDigest(
                        metadata=FileMetadata(
                            path=item,
                            size=item.stat().st_size,
                            modified_time=datetime.fromtimestamp(item.stat().st_mtime),
                            created_time=datetime.fromtimestamp(item.stat().st_ctime),
                            file_type=FileType.UNKNOWN,
                            mime_type=mimetypes.guess_type(str(item))[0]
                        )
                    ))
                    self.stats['total_files'] += 1
                    self.stats['total_size'] += item.stat().st_size
                    
        except PermissionError:
            print(f"警告: 无权限访问目录 {path}", file=sys.stderr)
        
        return structure
    
    def _should_ignore(self, path: Path) -> bool:
        """检查路径是否应该被忽略"""
        import fnmatch
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if pattern.startswith('*') and path.name.endswith(pattern[1:]):
                return True
        
        return False
    
    def _process_directory_parallel(self, structure: DirectoryStructure, mode: str):
        """并行处理目录中的所有文件"""
        import concurrent.futures
        
        # 收集所有文件
        all_files = []
        def collect_files(node: DirectoryStructure):
            all_files.extend(node.files)
            for subdir in node.subdirectories.values():
                collect_files(subdir)
        
        collect_files(structure)
        
        # 使用线程池并行处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有处理任务
            future_to_file = {
                executor.submit(self._process_file_safe, file_digest, mode): file_digest
                for file_digest in all_files
            }
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(future_to_file):
                file_digest = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"警告: 并行处理文件 {file_digest.metadata.path} 时出错: {e}", file=sys.stderr)
    
    def _process_file_safe(self, file_digest: FileDigest, mode: str):
        """安全的文件处理（用于并行处理）"""
        try:
            self._process_file(file_digest, mode)
        except Exception as e:
            print(f"警告: 处理文件 {file_digest.metadata.path} 时出错: {e}", file=sys.stderr)
    
    def _process_directory(self, structure: DirectoryStructure, mode: str):
        """处理目录中的所有文件"""
        if self.use_parallel and len(structure.files) > 10:
            # 使用并行处理
            self._process_directory_parallel(structure, mode)
        else:
            # 串行处理
            for file_digest in structure.files:
                self._process_file(file_digest, mode)
        
        # 递归处理子目录
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_file(self, file_digest: FileDigest, mode: str):
        """基于规则和上下文限制处理文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 1. 获取处理策略
            strategy, force_binary = self.rule_engine.classify_file(filepath)
            
            # 2. 估算token消耗
            estimated_tokens = self.rule_engine.estimate_token_usage(filepath, strategy)
            
            # 3. 检查文件大小限制
            if file_digest.metadata.size > self.max_file_size:
                file_digest.metadata.file_type = FileType.BINARY_FILES
                file_digest.human_readable_summary = HumanReadableSummary(
                    summary=f"[SKIPPED] File size ({file_digest.metadata.size / (1024*1024):.2f} MB) exceeds limit",
                    line_count=0
                )
                self.stats['skipped_large_files'] += 1
                self.stats['binary'] += 1
                return
            
            # 4. 检查上下文限制
            file_record = {
                "path": str(filepath),
                "strategy": strategy.value,
                "estimated_tokens": estimated_tokens,
                "size": file_digest.metadata.size,
            }
            
            # 尝试分配token
            if not self.context_manager.can_allocate(estimated_tokens):
                # 尝试降级策略
                downgraded_strategy = self.context_manager.downgrade_strategy(strategy)
                downgraded_tokens = self.rule_engine.estimate_token_usage(filepath, downgraded_strategy)
                
                if not self.context_manager.can_allocate(downgraded_tokens):
                    # 即使降级也无法分配，跳过此文件
                    self.stats['skipped_by_context'] += 1
                    file_digest.metadata.file_type = FileType.UNKNOWN
                    file_digest.human_readable_summary = HumanReadableSummary(
                        summary=f"[SKIPPED] Exceeds context window (estimated {estimated_tokens} tokens)",
                        line_count=0
                    )
                    return
                
                strategy = downgraded_strategy
                estimated_tokens = downgraded_tokens
            
            # 5. 分配token
            file_record["strategy"] = strategy.value
            file_record["estimated_tokens"] = estimated_tokens
            
            if not self.context_manager.allocate(estimated_tokens, file_record):
                self.stats['skipped_by_context'] += 1
                return
            
            # 6. 根据策略处理文件
            if force_binary or strategy == ProcessingStrategy.METADATA_ONLY:
                self._process_as_binary(file_digest)
            elif strategy in [ProcessingStrategy.FULL_CONTENT, ProcessingStrategy.SUMMARY_ONLY]:
                self._process_as_text(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.CODE_SKELETON:
                self._process_as_code(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
                self._process_as_config(file_digest, mode, strategy)
            elif strategy == ProcessingStrategy.HEADER_WITH_STATS:
                self._process_as_data(file_digest, mode, strategy)
            else:
                self._process_as_text(file_digest, mode, ProcessingStrategy.SUMMARY_ONLY)
            
        except Exception as e:
            print(f"Warning: Error processing file {filepath}: {e}", file=sys.stderr)
            file_digest.metadata.file_type = FileType.UNKNOWN
            file_digest.human_readable_summary = HumanReadableSummary(
                summary=f"[ERROR] Processing failed: {str(e)}"
            )

    def _process_as_binary(self, file_digest: FileDigest):
        """处理为二进制文件"""
        file_digest.metadata.file_type = FileType.BINARY_FILES
        self.stats['binary_files'] += 1
        self._calculate_hashes(file_digest)

    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """统一读取文件内容，处理编码问题"""
        try:
            # 首先尝试UTF-8
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    
                    # 如果没有chardet，尝试常见编码
                    if not CHARDET_AVAILABLE:
                        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        for encoding in encodings_to_try:
                            try:
                                return raw_content.decode(encoding)
                            except UnicodeDecodeError:
                                continue
                        # 所有编码都失败，使用latin-1并忽略错误
                        return raw_content.decode('latin-1', errors='ignore')
                    else:
                        # 使用chardet检测编码
                        result = chardet.detect(raw_content)
                        encoding = result['encoding'] if result['encoding'] else 'latin-1'
                        return raw_content.decode(encoding, errors='ignore')
            except Exception:
                return None
    
    def _process_as_text(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为文本文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            # 解码失败，降级为二进制
            self._process_as_binary(file_digest)
            return
        
        # 根据策略处理内容
        if strategy == ProcessingStrategy.FULL_CONTENT:
            file_digest.full_content = content
            file_digest.human_readable_summary = self.human_summarizer.summarize(filepath, content)
        else:  # SUMMARY_ONLY
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
            # 如果是全量模式但使用了摘要策略，存储摘要而非全文
            if mode == "full":
                file_digest.full_content = summary.summary or f"[Summary] {summary.line_count} lines, {summary.word_count} words"
        
        # 判断是否为关键文档
        filename = filepath.name.lower()
        critical_patterns = ['readme', 'license', 'copying', 'notice', 'changelog', 'changes', 
                            'contributing', 'install', 'authors', 'news', 'todo', 'roadmap']
        
        if any(pattern in filename for pattern in critical_patterns):
            file_digest.metadata.file_type = FileType.CRITICAL_DOCS
            self.stats['critical_docs'] += 1
        else:
            file_digest.metadata.file_type = FileType.REFERENCE_DOCS
            self.stats['reference_docs'] += 1
        self._calculate_hashes(file_digest)

    def _process_as_code(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为源代码文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 分析源代码
        analysis = self.source_analyzer.analyze(filepath, content)
        
        # 根据策略调整分析结果
        if strategy == ProcessingStrategy.CODE_SKELETON:
            # 骨架模式：只保留关键信息
            skeleton = {
                "language": analysis.language,
                "total_lines": analysis.total_lines,
                "functions": len(analysis.functions[:10]),  # 最多10个函数
                "classes": len(analysis.classes[:10]),      # 最多10个类
                "imports": analysis.imports[:10],           # 最多10个导入
            }
            # 可以添加骨架化处理...
        
        file_digest.source_code_analysis = analysis
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.SOURCE_CODE
        self.stats['source_code'] += 1
        self._calculate_hashes(file_digest)

    def _process_as_config(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为配置文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 尝试作为源代码分析
        analysis = self.source_analyzer.analyze(filepath, content)
        file_digest.source_code_analysis = analysis
        
        # 生成结构摘要
        if strategy == ProcessingStrategy.STRUCTURE_EXTRACT:
            structure = self.source_analyzer._extract_config_structure(content, filepath.suffix.lower())
            if structure:
                summary_lines = [f"[Config Structure] Type: {filepath.suffix}"]
                for item in structure[:20]:  # 最多20项
                    summary_lines.append(f"  - {item.get('name', 'unknown')}")
                if len(structure) > 20:
                    summary_lines.append(f"  ... and {len(structure) - 20} more")
                
                file_digest.human_readable_summary = HumanReadableSummary(
                    summary='\n'.join(summary_lines),
                    line_count=len(content.split('\n')),
                    character_count=len(content)
                )
        
        if mode == "full":
            file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)

    def _process_as_data(self, file_digest: FileDigest, mode: str, strategy: ProcessingStrategy):
        """处理为数据文件"""
        filepath = file_digest.metadata.path
        
        content = self._read_file_content(filepath)
        if content is None:
            self._process_as_binary(file_digest)
            return
        
        # 使用智能截断处理器
        if SmartTextProcessor.is_structured_data_file(filepath):
            human_content = SmartTextProcessor.extract_human_relevant_content(content, filepath)
            
            if mode == "full":
                file_digest.full_content = human_content
            else:
                file_digest.human_readable_summary = HumanReadableSummary(
                    summary=f"[Structured Data] Type: {filepath.suffix}",
                    first_lines=human_content.split('\n')[:10],
                    line_count=len(content.split('\n')),
                    character_count=len(content)
                )
        else:
            # 普通文本处理
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
            if mode == "full":
                file_digest.full_content = content
        
        file_digest.metadata.file_type = FileType.TEXT_DATA
        self.stats['text_data'] += 1
        self._calculate_hashes(file_digest)
    
    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件的哈希值（流式处理，内存高效）"""
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
            # 权限问题、文件不存在等
            print(f"Warning: Could not read file for hash calculation: {filepath} - {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "read_error"
            file_digest.metadata.sha256_hash = "read_error"
        except Exception as e:
            # 其他意外错误
            print(f"Warning: Hash calculation failed for {filepath}: {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "hash_error"
            file_digest.metadata.sha256_hash = "hash_error"
    
    def _calculate_hashes_from_bytes(self, file_digest: FileDigest, raw_bytes: bytes):
        """从字节数据计算哈希值"""
        try:
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            # 处理大文件时使用分块
            chunk_size = 65536
            for i in range(0, len(raw_bytes), chunk_size):
                chunk = raw_bytes[i:i+chunk_size]
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
            
            file_digest.metadata.md5_hash = md5_hash.hexdigest()
            file_digest.metadata.sha256_hash = sha256_hash.hexdigest()
            
        except Exception as e:
            print(f"Warning: Hash calculation from bytes failed: {e}", file=sys.stderr)
            file_digest.metadata.md5_hash = "hash_error"
            file_digest.metadata.sha256_hash = "hash_error"
    
    def _extract_table_header(self, content: str) -> str:
        """从数据表中提取头部（列名、注释等）"""
        lines = content.split('\n')
        header_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # 空行且已有内容，视为头部结束
            if not stripped and header_lines:
                break
            
            # 注释行保留
            if stripped.startswith(('#', '//', '/*', '*', 'C', 'CC', '!')):
                header_lines.append(line)
                continue
            
            # 列名行（字母开头，含分隔符但非纯数字）
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_\s,;:|]*$', stripped) and len(stripped) < 200:
                header_lines.append(line)
                continue
            
            # 检测到数据行特征则停止
            if re.match(r'^[\d\.,;\s\t|]+$', stripped) and len(stripped) > 10:
                break
            
            header_lines.append(line)
            
            if len(header_lines) >= 50:  # 限制头部行数
                break
        
        return '\n'.join(header_lines)
    
    def _generate_code_skeleton(self, analysis: SourceCodeAnalysis) -> str:
        """生成代码骨架描述"""
        lines = []
        if analysis.classes:
            lines.append(f"Classes: {len(analysis.classes)}")
            for cls in analysis.classes[:5]:
                base = f"({', '.join(cls.get('bases', []))})" if cls.get('bases') else ""
                lines.append(f"  class {cls.get('name', 'Unknown')}{base}")
        if analysis.functions:
            lines.append(f"Functions: {len(analysis.functions)}")
            for func in analysis.functions[:10]:
                lines.append(f"  def {func.get('name', 'unknown')}()")
        if analysis.imports:
            lines.append(f"Imports: {len(analysis.imports)}")
            for imp in analysis.imports[:5]:
                lines.append(f"  import {imp}")
        return '\n'.join(lines)
    
    def _process_config_with_limit(self, file_digest: FileDigest, content: str, mode: str):
        """处理配置文件，应用大小限制"""
        if len(content) > self.max_full_content_size:
            # 截断但保留结构完整性（尝试在键值对边界截断）
            truncated = content[:self.max_full_content_size]
            # 找最后一个完整的行
            last_newline = truncated.rfind('\n')
            if last_newline > 0:
                truncated = truncated[:last_newline]
            content = truncated + "\n\n[CONFIG TRUNCATED]"
            truncated_flag = True
        else:
            truncated_flag = False
        
        file_digest.full_content = content
        analysis = self.source_analyzer.analyze(file_digest.metadata.path, content)
        file_digest.source_code_analysis = analysis
        
        if truncated_flag:
            if not hasattr(file_digest, 'content_metadata'):
                file_digest.content_metadata = {}
            file_digest.content_metadata['content_status'] = 'truncated'
            file_digest.content_metadata['truncated_reason'] = 'size_limit'
            file_digest.content_metadata['available_on_request'] = True
    
    def _process_human_readable(self, file_digest: FileDigest, mode: str):
        """处理人类可读文本文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 读取文件（编码处理逻辑保持不变）
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    if not CHARDET_AVAILABLE:
                        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        for encoding in encodings_to_try:
                            try:
                                content = raw_content.decode(encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            content = raw_content.decode('latin-1', errors='ignore')
                    else:
                        result = chardet.detect(raw_content)
                        encoding = result['encoding'] if result['encoding'] else 'latin-1'
                        content = raw_content.decode(encoding, errors='ignore')
            
            # 全量模式检查：仅当文件大小不超过阈值时才存储全文
            if mode == "full":
                if file_digest.metadata.size <= self.max_full_content_size:
                    file_digest.full_content = content
                else:
                    # 超过阈值，生成摘要说明而非全文
                    lines = content.split('\n')
                    preview_lines = lines[:50]  # 前50行预览
                    preview = '\n'.join(preview_lines)
                    
                    file_digest.full_content = (
                        f"[FILE TOO LARGE - FULL CONTENT OMITTED]\n"
                        f"File size: {file_digest.metadata.size / 1024:.2f} KB\n"
                        f"Size limit: {self.max_full_content_size / 1024:.2f} KB\n"
                        f"Total lines: {len(lines)}\n"
                        f"Characters: {len(content)}\n"
                        f"\n--- PREVIEW (first {len(preview_lines)} lines) ---\n"
                        f"{preview}\n"
                        f"\n--- END PREVIEW ---"
                    )
            
            # 生成摘要（始终生成）
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
        except Exception as e:
            print(f"Warning: Error processing human readable file {filepath}: {e}", file=sys.stderr)
            file_digest.human_readable_summary = HumanReadableSummary(
                line_count=0, word_count=0, character_count=0,
                summary=f"Failed to read file content: {str(e)}"
            )
    
    def _process_config_script(self, file_digest: FileDigest, mode: str):
        """处理配置文件/脚本（既是源代码又是人类可读）"""
        # 首先作为人类可读文件处理（生成摘要和可能的完整内容）
        self._process_human_readable(file_digest, mode)
        
        filepath = file_digest.metadata.path
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取源代码分析（复用现有分析器）
            analysis = self.source_analyzer.analyze(filepath, content)
            
            # 标记为配置文件/脚本类型
            if analysis.language == "unknown":
                analysis.language = f"config/{filepath.suffix.lstrip('.')}"
            else:
                analysis.language = f"config_script/{analysis.language}"
            
            # 修复：通过 source_analyzer 实例调用 _extract_config_structure
            config_structure = self.source_analyzer._extract_config_structure(content, filepath.suffix.lower())
            if config_structure:
                # 合并到函数列表中（或作为单独字段）
                analysis.functions = config_structure + analysis.functions
            
            file_digest.source_code_analysis = analysis
            
        except Exception as e:
            print(f"Warning: Error analyzing config file {filepath}: {e}", file=sys.stderr)

    def _process_source_code(self, file_digest: FileDigest, mode: str):
        """处理源代码文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 尝试以UTF-8读取
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                with open(filepath, 'rb') as f:
                    raw_content = f.read()
                    
                    # 如果没有chardet，尝试常见编码
                    if not CHARDET_AVAILABLE:
                        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        for encoding in encodings_to_try:
                            try:
                                content = raw_content.decode(encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            # 所有编码都失败，使用latin-1并忽略错误
                            content = raw_content.decode('latin-1', errors='ignore')
                    else:
                        # 使用chardet检测编码
                        result = chardet.detect(raw_content)
                        encoding = result['encoding'] if result['encoding'] else 'latin-1'
                        # 尝试解码
                        try:
                            content = raw_content.decode(encoding, errors='ignore')
                        except Exception:
                            # 如果还是失败，使用latin-1并忽略错误
                            content = raw_content.decode('latin-1', errors='ignore')
            
            # 全量模式也存储源代码内容（可选）
            if mode == "full" and file_digest.metadata.size < self.max_file_size:
                file_digest.full_content = content
            
            # 分析源代码
            analysis = self.source_analyzer.analyze(filepath, content)
            file_digest.source_code_analysis = analysis
            
        except Exception as e:
            print(f"警告: 处理源代码文件 {filepath} 时出错: {e}", file=sys.stderr)
            # 创建基本分析结果
            file_digest.source_code_analysis = SourceCodeAnalysis(
                language="unknown",
                total_lines=0,
                code_lines=0,
                comment_lines=0,
                blank_lines=0
            )
    
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

    def _generate_output(self, mode: str) -> Dict:
        """生成最终输出"""
        if not self.structure:
            return {}
        
        # 基础输出结构
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "root_directory": str(self.root),
                "output_mode": mode,
                "context_config": {
                    "max_tokens": self.context_size,
                    "rules_file": str(self.rules_file) if self.rules_file else None,
                },
                "statistics": self.stats,
                "context_usage": self.context_manager.get_summary(),
            },
            "structure": self.structure.to_dict(mode)
        }
        
        # 添加上下文分配信息
        if self.context_manager.file_records:
            output["context_allocation"] = {
                "file_records": self.context_manager.file_records,
                "strategy_distribution": self._analyze_strategy_distribution(),
            }
        
        return output

    def _analyze_strategy_distribution(self) -> Dict[str, Any]:
        """分析策略使用分布"""
        distribution = {}
        for record in self.context_manager.file_records:
            strategy = record.get("strategy", "unknown")
            distribution[strategy] = distribution.get(strategy, 0) + 1
        
        return distribution
    
    def save_output(self, output: Dict, format: str = "json", output_path: Optional[Path] = None, mode: str = None):
        """保存输出到文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.lower()
            if ext == "markdown":
                ext = "md"
            output_path = self.root / f"directory_digest_{timestamp}.{ext}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = FormatConverter.convert(output, format, mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"摘要已保存到: {output_path}")
        return output_path
    
    def save_output_with_mode(self, output: Dict, format: str = "json", output_path: Optional[Path] = None, mode: str = None):
        """兼容方法：保存输出到文件（带模式参数）"""
        return self.save_output(output, format, output_path, mode)


# ==================== 命令行接口 ====================

def main():
    """命令行入口点"""
    import argparse
    
    # 自定义帮助格式化器，保留换行和格式
    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _format_action(self, action):
            # 保持父类的格式化行为，但确保帮助文本正确缩进
            return super()._format_action(action)
    
    parser = argparse.ArgumentParser(
        prog="directory_digest",
        description="""
Directory Digest Tool - 目录知识摘要生成器

将文件系统递归"消化"为LLM可理解的上下文摘要。
基于规则文件智能分类：人类可读文本 | 源代码 | 配置文件/脚本 | 二进制文件

三种工作模式：
  framework (默认)  生成文件结构和元数据摘要
  full              包含完整文件内容（受上下文窗口限制）
  sort              按类型/大小分类列出文件，提供统计建议
        """.strip(),
        formatter_class=CustomHelpFormatter,
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
  
  # 全量模式，管道处理
  %(prog)s . --mode full | grep -A 5 "class Controller"
  
  # 分析超大项目，启用并行处理
  %(prog)s /data --parallel --workers 8 --save report.json
  
  # 严格模式：跳过大于100MB的文件
  %(prog)s . --max-size 100
  
  # 自定义忽略规则，输出YAML到stdout
  %(prog)s . --ignore "*.log,*.tmp,cache,*.min.js" --output yaml
  
  # 生成HTML报告
  %(prog)s /code --mode full --output html --save report.html
        """
    )
    
    # ===== 核心选项组 =====
    core_group = parser.add_argument_group("核心选项", "指定输入目录和工作模式")
    core_group.add_argument(
        "directory", 
        metavar="PATH",
        help="要分析的目录路径"
    )
    core_group.add_argument(
        "-m", "--mode", 
        choices=["full", "framework", "sort"], 
        default="framework",
        metavar="MODE",
        help="工作模式 (默认: %(default)s)"
    )
    
    # ===== 规则和上下文控制 =====
    rule_group = parser.add_argument_group(
        "规则和上下文控制", 
        "控制文件分类和LLM上下文优化"
    )
    rule_group.add_argument(
        "-r", "--rules", 
        metavar="FILE",
        help="""
        指定规则文件路径（YAML格式）。
        规则文件定义文件分类和处理策略。
        如果不提供，使用内置的启发式规则。
        """
    )
    rule_group.add_argument(
        "--context-size", 
        type=str,
        default="128k",
        metavar="SIZE",
        help="""
        目标LLM上下文大小（tokens）。
        支持格式: "64k", "128k", "256k" 或具体数字。
        用于优化token分配。
        (默认: %(default)s)
        """
    )
    
    # ===== 输出控制组 =====
    output_group = parser.add_argument_group(
        "输出控制", 
        "控制输出格式、保存位置和内容详细程度"
    )
    output_group.add_argument(
        "-o", "--output", 
        choices=["json", "yaml", "md", "html", "toml", "txt"], 
        default="json",
        metavar="FORMAT",
        help="输出格式 (默认: %(default)s)"
    )
    output_group.add_argument(
        "-s", "--save", 
        metavar="FILE",
        help="""
        指定输出文件路径。特殊情况：
          - 不提供此选项：输出到标准输出（stdout，适合管道处理）
          - 使用 "-": 强制输出到标准输出（即使提供此选项）
          - 其他路径: 写入指定文件（自动创建目录）
        (默认: 输出到 stdout)
        """
    )
    output_group.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="显示详细处理信息（包括每个文件的处理状态）"
    )
    
    # ===== 大小限制组 =====
    size_group = parser.add_argument_group(
        "大小限制",
        "控制文件处理的阈值"
    )
    size_group.add_argument(
        "--max-size", 
        type=int, 
        default=10240,  # 10GB = 10 * 1024 MB
        metavar="MB",
        help="""
        文件大小阈值(MB)。超过此大小的文件将被**完全跳过**：
        不计算checksum，不分析内容，仅保留路径和大小元数据。
        适用于排除超大日志、虚拟机镜像、数据集、媒体文件。
        (默认: %(default)s MB = 10 GB)
        """
    )
    
    # ===== 处理选项组 =====
    proc_group = parser.add_argument_group("处理选项", "控制并行处理和文件过滤规则")
    proc_group.add_argument(
        "--ignore", 
        default=".git,__pycache__,*.pyc,*.pyo,node_modules,.venv,venv,*.min.js,*.map",
        metavar="PATTERNS",
        help="""
        忽略规则，逗号分隔的glob模式。
        默认忽略版本控制、缓存、依赖目录和压缩文件: %(default)s
        """
    )
    proc_group.add_argument(
        "-p", "--parallel", 
        action="store_true",
        help="启用并行处理（推荐用于包含>1000个文件的大型项目）"
    )
    proc_group.add_argument(
        "-w", "--workers", 
        type=int, 
        default=0,
        metavar="N",
        help="""
        并行工作线程数。
        0 表示自动检测 CPU 核心数 (默认: %(default)s → 实际使用 %(const)s 线程)
        """ % {'default': 0, 'const': os.cpu_count() or 4}
    )
    
    args = parser.parse_args()
    
    # 解析上下文大小
    def parse_context_size(size_str: str) -> int:
        """解析上下文大小字符串"""
        size_str = size_str.lower().strip()
        
        # 移除后缀 'k' 或 'tokens'
        if size_str.endswith('k'):
            multiplier = 1000
            size_str = size_str[:-1]
        elif size_str.endswith('tokens'):
            multiplier = 1
            size_str = size_str[:-6].strip()
        else:
            multiplier = 1
        
        try:
            # 解析数字
            if '.' in size_str:
                base_value = float(size_str)
            else:
                base_value = int(size_str)
            
            return int(base_value * multiplier)
        except ValueError:
            print(f"警告: 无法解析上下文大小 '{args.context_size}'，使用默认值 128000", file=sys.stderr)
            return 128000
    
    context_size = parse_context_size(args.context_size)
    
    # 配置转换
    config = {
        'max_file_size': args.max_size * 1024 * 1024,  # MB 转 Bytes
        'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()],
        'use_parallel': args.parallel,
        'max_workers': args.workers if args.workers > 0 else os.cpu_count() or 4,
        'rules_file': Path(args.rules) if args.rules else None,
        'context_size': context_size,
    }
    
    # 创建摘要器
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
    
    # 生成摘要
    output = digest.create_digest(args.mode)
    
    # 处理输出：默认 stdout，--save 指定文件路径，--save - 强制 stdout
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
        
        # 统计信息输出到 stderr
        if args.verbose or args.mode == "sort":
            stats = output['metadata']['statistics']
            ctx_usage = output['metadata'].get('context_usage')  # 修改这行
            
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
            
            if ctx_usage:  # 添加这个条件判断
                print(f"[Context] Used: {ctx_usage['used_tokens']:,}/{ctx_usage['max_tokens']:,} tokens "
                      f"({ctx_usage['token_utilization']:.1%})", file=sys.stderr)
            else:
                print(f"[Context] Not applicable for sort mode", file=sys.stderr)
            
            if args.mode == "sort" and "recommendations" in output:
                for rec in output["recommendations"]:
                    print(f"[Tip] {rec}", file=sys.stderr)
        
        return None  # stdout 模式返回 None
        
    else:
        # 写入指定文件
        output_path = Path(args.save)
        saved_path = digest.save_output(output, args.output, output_path, args.mode)
        
        # 显示处理结果（到 stderr，避免与文件内容混淆）
        stats = output['metadata']['statistics']
        ctx_usage = output['metadata'].get('context_usage')  # 修改这行
        
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
        
        if ctx_usage:  # 添加这个条件判断
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
