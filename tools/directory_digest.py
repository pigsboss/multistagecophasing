#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directory_digest.py - 目录知识摘要器

将文件系统递归"消化"为LLM可理解的上下文摘要。

三种文件分类策略：
1. 人类可读文本文件 (HumanReadable) - .txt, .md, .rst, 配置文件等
2. 源代码文件 (SourceCode) - .py, .java, .cpp 等编程语言文件  
3. 二进制文件 (Binary) - 图像、压缩包、可执行文件等

两种输出模式：
- 全量模式 (full): 包含人类可读文本文件的完整内容
- 框架模式 (framework): 所有文件都只输出摘要/元信息
"""

import os
import sys
import json
import hashlib
import mimetypes
import ast  # Python抽象语法树解析
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import re
from enum import Enum

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


# ==================== 数据类型定义 ====================

class FileType(Enum):
    """文件类型枚举"""
    HUMAN_READABLE = "human_readable"    # 人类可读文本
    SOURCE_CODE = "source_code"          # 源代码
    BINARY = "binary"                    # 二进制文件
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
            "summary": self.summary
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
            "dependencies": self.dependencies[:20]
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
        # 人类可读文本
        FileType.HUMAN_READABLE: [
            '.txt', '.md', '.markdown', '.rst', '.tex', '.latex',
            '.json', '.yaml', '.yml', '.xml', '.html', '.htm', '.csv',
            '.ini', '.cfg', '.conf', '.toml', '.properties',
            '.log', '.out', '.err'
        ],
        # 源代码
        FileType.SOURCE_CODE: [
            '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.cc',
            '.js', '.ts', '.jsx', '.tsx', '.vue',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.m', '.mm', '.cs', '.fs', '.vb',
            '.sh', '.bash', '.zsh', '.fish', '.ps1',
            '.sql', '.pl', '.pm', '.r', '.lua', '.dart'
        ],
        # 二进制文件（部分常见）
        FileType.BINARY: [
            '.exe', '.dll', '.so', '.dylib', '.a', '.lib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico',
            '.mp3', '.mp4', '.avi', '.mkv', '.mov', '.wav',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.bin', '.dat', '.db', '.sqlite', '.sqlite3'
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
                    return FileType.BINARY
                
                # 检测可打印字符比例
                printable_count = 0
                for byte in sample:
                    if 32 <= byte <= 126 or byte in (9, 10, 13):
                        printable_count += 1
                
                printable_ratio = printable_count / len(sample) if sample else 0
                
                if printable_ratio < 0.7:
                    return FileType.BINARY
                
                # 检测源代码特征
                try:
                    decoded = sample.decode('utf-8', errors='ignore')
                    if FileTypeDetector._looks_like_source_code(decoded):
                        return FileType.SOURCE_CODE
                except:
                    pass
                
                return FileType.HUMAN_READABLE
                
        except Exception:
            return FileType.BINARY
    
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


# ==================== 摘要生成器 ====================

class HumanReadableSummarizer:
    """人类可读文本摘要生成器"""
    
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
            summary=summary
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
        elif suffix in ['.go']:
            return SourceCodeAnalyzer._analyze_go(content)
        elif suffix in ['.rs']:
            return SourceCodeAnalyzer._analyze_rust(content)
        elif suffix in ['.rb']:
            return SourceCodeAnalyzer._analyze_ruby(content)
        elif suffix in ['.php']:
            return SourceCodeAnalyzer._analyze_php(content)
        elif suffix in ['.sh', '.bash', '.zsh']:
            return SourceCodeAnalyzer._analyze_shell(content)
        elif suffix in ['.sql']:
            return SourceCodeAnalyzer._analyze_sql(content)
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
                                    # 如果是简单的常量
                                    if isinstance(node.value, (ast.Constant, ast.Num, ast.Str)):
                                        constants.append(var_name)
                                    else:
                                        global_vars.append(var_name)
                                except:
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
            dependencies=SourceCodeAnalyzer._extract_python_dependencies(imports)
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
    def _analyze_go(content: str) -> SourceCodeAnalysis:
        """分析Go代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('//'):
                comment_lines += 1
        
        # 提取导入语句
        imports = []
        import_section = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import'):
                import_section = True
                if '(' in stripped:
                    continue  # 多行导入开始
                else:
                    # 单行导入
                    match = re.search(r'import\s+"([^"]+)"', stripped)
                    if match:
                        imports.append(match.group(1))
            elif import_section:
                if stripped == ')':
                    import_section = False
                else:
                    match = re.search(r'"([^"]+)"', stripped)
                    if match:
                        imports.append(match.group(1))
        
        # 提取函数定义
        functions = []
        func_pattern = r'^func\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*\('
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="go",
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=imports,
            functions=functions,
            classes=[],  # Go没有类，只有结构体
            global_vars=[],
            constants=[],
            dependencies=[]
        )
    
    @staticmethod
    def _analyze_rust(content: str) -> SourceCodeAnalysis:
        """分析Rust代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('//'):
                comment_lines += 1
        
        # 提取导入语句
        imports = []
        use_pattern = r'^use\s+([a-zA-Z0-9_:]+)'
        
        for line in lines:
            match = re.search(use_pattern, line.strip())
            if match:
                imports.append(match.group(1))
        
        # 提取函数定义
        functions = []
        func_pattern = r'^\s*(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        # 提取结构体定义
        classes = []
        struct_pattern = r'^\s*(?:pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(struct_pattern, line.strip())
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="rust",
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
    def _analyze_ruby(content: str) -> SourceCodeAnalysis:
        """分析Ruby代码"""
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
        
        # 提取require/load语句
        imports = []
        import_patterns = [
            r'^require\s+["\']([^"\']+)["\']',
            r'^load\s+["\']([^"\']+)["\']',
            r'^require_relative\s+["\']([^"\']+)["\']'
        ]
        
        for line in lines:
            for pattern in import_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    imports.append(match.group(1))
                    break
        
        # 提取方法定义
        functions = []
        func_pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\?|!)?)'
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        # 提取类定义
        classes = []
        class_pattern = r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line.strip())
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="ruby",
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
    def _analyze_php(content: str) -> SourceCodeAnalysis:
        """分析PHP代码"""
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
            
            if stripped.startswith('//') or stripped.startswith('#'):
                comment_lines += 1
            elif stripped.startswith('/*'):
                comment_lines += 1
                in_block_comment = True
                if '*/' in stripped and stripped.index('*/') > stripped.index('/*'):
                    in_block_comment = False
        
        # 提取include/require语句
        imports = []
        import_pattern = r'^(?:include|require)(?:_once)?\s+(?:[\'"]([^\'"]+)[\'"]|\([\'"]([^\'"]+)[\'"]\))'
        
        for line in lines:
            match = re.search(import_pattern, line.strip())
            if match:
                imp = match.group(1) or match.group(2)
                if imp:
                    imports.append(imp)
        
        # 提取函数定义
        functions = []
        func_pattern = r'^\s*(?:public|private|protected)?\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        # 提取类定义
        classes = []
        class_pattern = r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line.strip())
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="php",
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
    def _analyze_shell(content: str) -> SourceCodeAnalysis:
        """分析Shell脚本"""
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
        
        # 提取函数定义
        functions = []
        func_pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\)\s*{'
        
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line.strip())
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="shell",
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            imports=[],  # Shell通常没有明确的导入
            functions=functions,
            classes=[],
            global_vars=[],
            constants=[],
            dependencies=[]
        )
    
    @staticmethod
    def _analyze_sql(content: str) -> SourceCodeAnalysis:
        """分析SQL代码"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('--'):
                comment_lines += 1
        
        # 提取表创建语句
        classes = []  # 这里用classes存储表定义
        table_pattern = r'^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(table_pattern, line.strip(), re.IGNORECASE)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i + 1,
                    "type": "table"
                })
        
        # 提取存储过程/函数
        functions = []
        proc_pattern = r'^CREATE\s+(?:PROCEDURE|FUNCTION)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        for i, line in enumerate(lines):
            match = re.search(proc_pattern, line.strip(), re.IGNORECASE)
            if match:
                functions.append({
                    "name": match.group(1),
                    "line": i + 1,
                    "type": "procedure"
                })
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return SourceCodeAnalysis(
            language="sql",
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
    
    @staticmethod
    def _analyze_generic(content: str, suffix: str) -> SourceCodeAnalysis:
        """通用源代码分析"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 统计行数
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('//') or stripped.startswith('#'):
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
    def convert(digest_data: Dict, format: str) -> str:
        """转换为指定格式"""
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
        # 待实现
        return "# 目录摘要\n\n待实现\n"
    
    @staticmethod
    def _to_yaml(digest_data: Dict) -> str:
        """转换为YAML格式"""
        # 待实现
        return "metadata:\n  generated_at: ...\n"
    
    @staticmethod
    def _to_html(digest_data: Dict) -> str:
        """转换为HTML格式"""
        # 待实现
        return "<html><body>待实现</body></html>"
    
    @staticmethod
    def _to_toml(digest_data: Dict) -> str:
        """转换为TOML格式"""
        # 待实现
        return '[metadata]\ngenerated_at = "..."'
    
    @staticmethod
    def _to_plaintext(digest_data: Dict) -> str:
        """转换为纯文本格式"""
        # 待实现
        return "目录摘要 - 待实现"


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
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)
        self.ignore_patterns = self.config.get('ignore_patterns', [
            '*.pyc', '*.pyo', '*.so', '*.dll', '__pycache__', 
            '.git', '.svn', '.hg', '.DS_Store', '*.swp', '*.swo'
        ])
        
        self.file_type_detector = FileTypeDetector()
        self.human_summarizer = HumanReadableSummarizer()
        self.source_analyzer = SourceCodeAnalyzer()
        
        # 存储结果
        self.structure: Optional[DirectoryStructure] = None
        self.stats = {
            'total_files': 0,
            'human_readable': 0,
            'source_code': 0,
            'binary': 0,
            'total_size': 0,
            'processing_time': 0
        }
    
    def create_digest(self, mode: str = "framework") -> Dict:
        """
        创建目录摘要
        
        Args:
            mode: 输出模式，"framework"（框架）或 "full"（全量）
            
        Returns:
            摘要字典
        """
        import time
        start_time = time.time()
        
        # 构建目录结构
        self.structure = self._build_directory_structure(self.root)
        
        # 处理所有文件
        self._process_directory(self.structure, mode)
        
        # 更新统计信息
        self.stats['processing_time'] = time.time() - start_time
        
        # 生成最终输出
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
    
    def _process_directory(self, structure: DirectoryStructure, mode: str):
        """处理目录中的所有文件"""
        # 处理当前目录的文件
        for file_digest in structure.files:
            self._process_file(file_digest, mode)
        
        # 递归处理子目录
        for subdir in structure.subdirectories.values():
            self._process_directory(subdir, mode)
    
    def _process_file(self, file_digest: FileDigest, mode: str):
        """处理单个文件"""
        filepath = file_digest.metadata.path
        
        try:
            # 1. 检测文件类型
            file_type = self.file_type_detector.detect(filepath)
            file_digest.metadata.file_type = file_type
            
            # 更新统计
            if file_type == FileType.HUMAN_READABLE:
                self.stats['human_readable'] += 1
            elif file_type == FileType.SOURCE_CODE:
                self.stats['source_code'] += 1
            elif file_type == FileType.BINARY:
                self.stats['binary'] += 1
            
            # 2. 计算哈希值
            self._calculate_hashes(file_digest)
            
            # 3. 根据文件类型处理内容
            if file_type == FileType.HUMAN_READABLE:
                self._process_human_readable(file_digest, mode)
            elif file_type == FileType.SOURCE_CODE:
                self._process_source_code(file_digest, mode)
            # 二进制文件不需要额外处理
            
        except Exception as e:
            print(f"警告: 处理文件 {filepath} 时出错: {e}", file=sys.stderr)
            file_digest.metadata.file_type = FileType.BINARY
    
    def _calculate_hashes(self, file_digest: FileDigest):
        """计算文件的哈希值"""
        filepath = file_digest.metadata.path
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                
                # MD5
                md5_hash = hashlib.md5(content).hexdigest()
                file_digest.metadata.md5_hash = md5_hash
                
        except Exception:
            pass
    
    def _process_human_readable(self, file_digest: FileDigest, mode: str):
        """处理人类可读文本文件"""
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
            
            # 全量模式存储完整内容
            if mode == "full":
                file_digest.full_content = content
            
            # 生成摘要
            summary = self.human_summarizer.summarize(filepath, content)
            file_digest.human_readable_summary = summary
            
        except Exception as e:
            print(f"警告: 处理人类可读文件 {filepath} 时出错: {e}", file=sys.stderr)
            # 创建基本摘要
            file_digest.human_readable_summary = HumanReadableSummary(
                line_count=0,
                word_count=0,
                character_count=0,
                summary=f"无法读取文件内容: {str(e)}"
            )
    
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
                "statistics": self.stats
            },
            "structure": self.structure.to_dict(mode)
        }
        
        return output
    
    def save_output(self, output: Dict, format: str = "json", output_path: Optional[Path] = None):
        """保存输出到文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.lower()
            if ext == "markdown":
                ext = "md"
            output_path = self.root / f"directory_digest_{timestamp}.{ext}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = FormatConverter.convert(output, format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"摘要已保存到: {output_path}")
        return output_path


# ==================== 命令行接口 ====================

def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="目录知识摘要器 - 将文件系统递归消化为LLM可理解的上下文摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /path/to/directory --mode full --output json
  %(prog)s . --mode framework --output yaml --ignore ".git,*.pyc"
        """
    )
    
    parser.add_argument("directory", help="要分析的目录路径")
    parser.add_argument("--mode", choices=["full", "framework"], default="framework",
                       help="输出模式: full(全量) 或 framework(框架)")
    parser.add_argument("--output", choices=["json", "yaml", "md", "html", "toml", "txt"], 
                       default="json", help="输出格式")
    parser.add_argument("--ignore", default=".git,__pycache__,*.pyc,*.pyo",
                       help="忽略的模式，用逗号分隔")
    parser.add_argument("--max-size", type=int, default=10,
                       help="最大文件大小(MB)，超过此大小的文件只分析元信息")
    parser.add_argument("--save", help="输出文件路径，默认为目录名_digest.json")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'max_file_size': args.max_size * 1024 * 1024,
        'ignore_patterns': [p.strip() for p in args.ignore.split(',') if p.strip()]
    }
    
    # 创建摘要器
    digest = DirectoryDigest(args.directory, config)
    
    # 生成摘要
    if args.verbose:
        print(f"开始分析目录: {args.directory}")
        print(f"模式: {args.mode}, 格式: {args.output}")
    
    output = digest.create_digest(args.mode)
    
    # 保存输出
    output_path = Path(args.save) if args.save else None
    saved_path = digest.save_output(output, args.output, output_path)
    
    # 显示统计信息
    stats = output['metadata']['statistics']
    print(f"\n摘要统计:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  人类可读文本: {stats['human_readable']}")
    print(f"  源代码文件: {stats['source_code']}")
    print(f"  二进制文件: {stats['binary']}")
    print(f"  总大小: {stats['total_size'] / (1024*1024):.2f} MB")
    print(f"  处理时间: {stats['processing_time']:.2f} 秒")
    
    return saved_path


if __name__ == "__main__":
    main()
