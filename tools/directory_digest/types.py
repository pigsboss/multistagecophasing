# -*- coding: utf-8 -*-
"""
types.py - 数据类定义
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from .constants import FileType, ProcessingStrategy


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
    reading_time_minutes: float = 0.0
    reading_level: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    
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
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    style_issues: List[Dict] = field(default_factory=list)
    security_issues: List[Dict] = field(default_factory=list)
    test_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "imports": self.imports,
            "functions": self.functions[:20],
            "classes": self.classes[:20],
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
        
        if mode == "full" and self.full_content and self.metadata.file_type == FileType.CRITICAL_DOCS:
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
        import fnmatch
        
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
