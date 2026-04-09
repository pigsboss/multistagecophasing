"""
文本数据和配置文件分析器 - 专门处理结构化文本数据
包括：
- 配置文件分析（key-value 结构提取）
- CSV/TSV 表格数据分析
- 结构化数据文件分析（SPICE, XML, JSON等）
- 表单规模和表头提取
- 数据统计和采样
"""

import re
import json
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter

from .base import (
    BaseDataSheetAnalyzer,
    BaseConfigAnalyzer,
    SemanticAnalysisResult,
    SmartTextProcessor,
    ContentAnalyzer,
)
from ...base import YAML_AVAILABLE, CHARDET_AVAILABLE

if YAML_AVAILABLE:
    import yaml


# ==================== 配置文件分析结果数据类 ====================

@dataclass
class ConfigAnalysisResult:
    """配置文件分析结果"""
    key_count: int = 0
    section_count: int = 0
    top_level_keys: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    is_hierarchical: bool = False
    estimated_size: str = "unknown"
    sample_content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_count": self.key_count,
            "section_count": self.section_count,
            "top_level_keys": self.top_level_keys,
            "sections": self.sections,
            "is_hierarchical": self.is_hierarchical,
            "estimated_size": self.estimated_size,
            "sample_content": self.sample_content[:500]
        }


# ==================== 表格数据分析结果数据类 ====================

@dataclass
class TableAnalysisResult:
    """表格数据文件分析结果"""
    row_count: int = 0
    column_count: int = 0
    headers: List[str] = field(default_factory=list)
    delimiter: Optional[str] = None
    has_header: bool = False
    sample_rows: List[List[str]] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    estimated_total_rows: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "headers": self.headers,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "sample_rows": self.sample_rows[:5],
            "column_types": self.column_types,
            "estimated_total_rows": self.estimated_total_rows
        }


# ==================== 配置文件分析器 ====================

class ConfigFileAnalyzer(BaseConfigAnalyzer):
    """配置文件分析器 - 提取 key-value 结构"""
    
    def __init__(self):
        super().__init__()
    
    def can_handle(self, filepath: Path, content: Optional[str] = None) -> bool:
        """判断是否为配置文件"""
        config_extensions = {
            '.yaml', '.yml', '.json', '.toml',
            '.ini', '.cfg', '.conf', '.env',
            '.properties', '.rc', '.xml'
        }
        return filepath.suffix.lower() in config_extensions
    
    def analyze(self, filepath: Path, content: Optional[str] = None) -> SemanticAnalysisResult:
        """
        分析配置文件
        
        Args:
            filepath: 文件路径
            content: 文件内容
            
        Returns:
            SemanticAnalysisResult: 分析结果
        """
        result = SemanticAnalysisResult(
            content_type="config",
            language="config"
        )
        
        try:
            if content is None:
                content = self._read_file_content(filepath)
            
            if content is None:
                result.success = False
                result.error_message = "Could not read file content"
                return result
            
            # 根据文件类型进行分析
            suffix = filepath.suffix.lower()
            
            if suffix in ['.yaml', '.yml']:
                config_result = self._analyze_yaml(content)
            elif suffix == '.json':
                config_result = self._analyze_json(content)
            elif suffix == '.toml':
                config_result = self._analyze_toml(content)
            elif suffix in ['.ini', '.cfg', '.conf']:
                config_result = self._analyze_ini(content)
            elif suffix in ['.env', '.properties', '.rc']:
                config_result = self._analyze_key_value(content)
            elif suffix == '.xml':
                config_result = self._analyze_xml(content)
            else:
                config_result = self._analyze_generic_config(content)
            
            # 生成摘要
            result.summary = self._generate_config_summary(config_result)
            result.keywords = config_result.top_level_keys[:10]
            result.metadata["config_analysis"] = config_result
            result.metadata["config_sample"] = self._extract_sample_content(content, suffix)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = f"Config analysis failed: {str(e)}"
        
        return result
    
    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """读取文件内容"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
    
    def _analyze_yaml(self, content: str) -> ConfigAnalysisResult:
        """分析 YAML 配置文件"""
        result = ConfigAnalysisResult()
        
        if not YAML_AVAILABLE:
            result.sample_content = "[YAML parser not available]"
            return result
        
        try:
            data = yaml.safe_load(content)
            
            if isinstance(data, dict):
                result.top_level_keys = list(data.keys())[:20]
                result.key_count = len(data)
                result.is_hierarchical = True
                result.section_count = self._count_nested_sections(data)
            
            result.sample_content = content[:500]
            
        except Exception as e:
            result.sample_content = f"[YAML parse error: {str(e)}]"
        
        return result
    
    def _analyze_json(self, content: str) -> ConfigAnalysisResult:
        """分析 JSON 配置文件"""
        result = ConfigAnalysisResult()
        
        try:
            data = json.loads(content)
            
            if isinstance(data, dict):
                result.top_level_keys = list(data.keys())[:20]
                result.key_count = len(data)
                result.is_hierarchical = True
                result.section_count = self._count_nested_sections(data)
            elif isinstance(data, list):
                result.key_count = len(data)
                result.is_hierarchical = False
            
            result.sample_content = content[:500]
            
        except Exception as e:
            result.sample_content = f"[JSON parse error: {str(e)}]"
        
        return result
    
    def _analyze_toml(self, content: str) -> ConfigAnalysisResult:
        """分析 TOML 配置文件"""
        result = ConfigAnalysisResult()
        
        # 简化处理 - 提取键值对和节
        lines = content.split('\n')
        sections = []
        keys = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                sections.append(stripped[1:-1])
            elif '=' in stripped and not stripped.startswith('#'):
                key_part = stripped.split('=', 1)[0].strip()
                keys.append(key_part)
        
        result.top_level_keys = list(set(keys))[:20]
        result.sections = sections
        result.key_count = len(keys)
        result.section_count = len(sections)
        result.sample_content = content[:500]
        
        return result
    
    def _analyze_ini(self, content: str) -> ConfigAnalysisResult:
        """分析 INI 配置文件"""
        result = ConfigAnalysisResult()
        
        lines = content.split('\n')
        sections = []
        keys = []
        current_section = None
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('[') and stripped.endswith(']'):
                current_section = stripped[1:-1]
                sections.append(current_section)
            elif '=' in stripped and not stripped.startswith('#') and not stripped.startswith(';'):
                key_part = stripped.split('=', 1)[0].strip()
                if current_section:
                    keys.append(f"{current_section}.{key_part}")
                else:
                    keys.append(key_part)
        
        result.top_level_keys = list(set(keys))[:20]
        result.sections = sections
        result.key_count = len(keys)
        result.section_count = len(sections)
        result.sample_content = content[:500]
        
        return result
    
    def _analyze_key_value(self, content: str) -> ConfigAnalysisResult:
        """分析简单 key-value 配置文件"""
        result = ConfigAnalysisResult()
        
        lines = content.split('\n')
        keys = []
        
        for line in lines:
            stripped = line.strip()
            if '=' in stripped and not stripped.startswith('#'):
                key_part = stripped.split('=', 1)[0].strip()
                if key_part:
                    keys.append(key_part)
        
        result.top_level_keys = list(set(keys))[:20]
        result.key_count = len(keys)
        result.sample_content = content[:500]
        
        return result
    
    def _analyze_xml(self, content: str) -> ConfigAnalysisResult:
        """分析 XML 配置文件"""
        result = ConfigAnalysisResult()
        
        # 简化处理 - 提取标签名
        tag_pattern = r'<(\w+)[^>]*>'
        tags = re.findall(tag_pattern, content)
        unique_tags = list(set(tags))
        
        result.top_level_keys = unique_tags[:20]
        result.key_count = len(tags)
        result.is_hierarchical = True
        result.sample_content = content[:500]
        
        return result
    
    def _analyze_generic_config(self, content: str) -> ConfigAnalysisResult:
        """分析通用配置文件"""
        result = ConfigAnalysisResult()
        
        # 提取看起来像键值对的行
        lines = content.split('\n')
        keys = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith(';'):
                continue
            
            if ':' in stripped or '=' in stripped:
                delimiter = ':' if ':' in stripped else '='
                key_part = stripped.split(delimiter, 1)[0].strip()
                if key_part:
                    keys.append(key_part)
        
        result.top_level_keys = list(set(keys))[:20]
        result.key_count = len(keys)
        result.sample_content = content[:500]
        
        return result
    
    def _count_nested_sections(self, data: Any, depth: int = 0) -> int:
        """计算嵌套节的数量"""
        if depth > 5:  # 限制深度
            return 0
        
        count = 0
        if isinstance(data, dict):
            count += len(data)
            for value in data.values():
                count += self._count_nested_sections(value, depth + 1)
        elif isinstance(data, list) and depth > 0:  # 只统计嵌套列表
            count += 1
        
        return count
    
    def _extract_sample_content(self, content: str, suffix: str) -> str:
        """提取样例内容"""
        lines = content.split('\n')
        return '\n'.join(lines[:20])  # 前20行
    
    def _generate_config_summary(self, config_result: ConfigAnalysisResult) -> str:
        """生成配置文件摘要"""
        parts = []
        parts.append(f"Configuration file with {config_result.key_count} keys")
        
        if config_result.section_count > 0:
            parts.append(f"  - {config_result.section_count} sections")
        
        if config_result.is_hierarchical:
            parts.append(f"  - Hierarchical structure")
        
        if config_result.top_level_keys:
            parts.append(f"  - Top keys: {', '.join(config_result.top_level_keys[:5])}")
        
        return '\n'.join(parts)


# ==================== 表格数据文件分析器 ====================

class TableDataAnalyzer(BaseDataSheetAnalyzer):
    """表格数据文件分析器 - CSV/TSV 等"""
    
    def __init__(self):
        super().__init__()
    
    def can_handle(self, filepath: Path, content: Optional[str] = None) -> bool:
        """判断是否为表格数据文件"""
        table_extensions = {
            '.csv', '.tsv', '.dat', '.data', '.txt'
        }
        return filepath.suffix.lower() in table_extensions
    
    def analyze(self, filepath: Path, content: Optional[str] = None) -> SemanticAnalysisResult:
        """
        分析表格数据文件
        
        Args:
            filepath: 文件路径
            content: 文件内容
            
        Returns:
            SemanticAnalysisResult: 分析结果
        """
        result = SemanticAnalysisResult(
            content_type="data_sheet",
            language="data"
        )
        
        try:
            if content is None:
                content = self._read_file_content(filepath)
            
            if content is None:
                result.success = False
                result.error_message = "Could not read file content"
                return result
            
            # 检测是否为结构化数据文件，需要智能截断
            if SmartTextProcessor.is_structured_data_file(filepath):
                result = self._analyze_structured_data(filepath, content, result)
            else:
                result = self._analyze_csv_table(filepath, content, result)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = f"Table analysis failed: {str(e)}"
        
        return result
    
    def _read_file_content(self, filepath: Path) -> Optional[str]:
        """读取文件内容"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
    
    def _analyze_csv_table(self, filepath: Path, content: str,
                          result: SemanticAnalysisResult) -> SemanticAnalysisResult:
        """分析 CSV/TSV 表格数据"""
        lines = content.split('\n')
        table_result = TableAnalysisResult()
        
        # 尝试自动检测分隔符
        delimiters = [',', '\t', ';', '|', ' ']
        best_delimiter = None
        max_columns = 0
        
        for delimiter in delimiters:
            column_counts = []
            for line in lines[:20]:
                stripped = line.strip()
                if not stripped or stripped.startswith(('#', '//', '/*', '*', 'C', 'CC', '!')):
                    continue
                columns = [c.strip() for c in stripped.split(delimiter)]
                if len(columns) > 1:
                    column_counts.append(len(columns))
            
            if column_counts:
                avg_columns = sum(column_counts) / len(column_counts)
                if avg_columns > max_columns and avg_columns >= 2:
                    max_columns = avg_columns
                    best_delimiter = delimiter
        
        if best_delimiter:
            table_result.delimiter = best_delimiter
            table_result = self._parse_with_delimiter(content, best_delimiter, table_result)
        else:
            # 没有明确的分隔符，尝试通用分析
            table_result = self._analyze_generic_table(content, table_result)
        
        # 生成摘要
        result.summary = self._generate_table_summary(table_result)
        result.keywords = table_result.headers[:10]
        result.metadata["table_analysis"] = table_result
        result.metadata["table_sample"] = self._extract_table_sample(content, table_result)
        
        return result
    
    def _parse_with_delimiter(self, content: str, delimiter: str,
                             table_result: TableAnalysisResult) -> TableAnalysisResult:
        """使用指定分隔符解析表格"""
        lines = content.split('\n')
        rows = []
        header_candidate = None
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(('#', '//', '/*', '*', 'C', 'CC', '!')):
                continue
            
            columns = [c.strip() for c in stripped.split(delimiter)]
            rows.append(columns)
            
            if len(rows) >= 100:  # 只分析前100行
                break
        
        if rows:
            # 检查第一行是否为表头
            first_row = rows[0]
            table_result.column_count = len(first_row)
            
            # 判断是否有表头：检查第一行是否有较多非数字内容
            non_numeric_count = sum(1 for cell in first_row 
                                   if cell and not cell.replace('.', '').replace('-', '').isdigit())
            
            if non_numeric_count >= len(first_row) * 0.5:
                table_result.headers = first_row
                table_result.has_header = True
                table_result.sample_rows = rows[1:6]
            else:
                table_result.headers = [f"col_{i}" for i in range(len(first_row))]
                table_result.has_header = False
                table_result.sample_rows = rows[:5]
            
            table_result.row_count = len(rows)
            
            # 估算总行数
            total_lines = len([l for l in lines if l.strip()])
            table_result.estimated_total_rows = total_lines
            
            # 分析列类型
            table_result.column_types = self._analyze_column_types(rows, table_result.has_header)
        
        return table_result
    
    def _analyze_column_types(self, rows: List[List[str]], has_header: bool) -> Dict[str, str]:
        """分析列数据类型"""
        column_types = {}
        start_idx = 1 if has_header else 0
        sample_rows = rows[start_idx:start_idx+20]
        
        if not sample_rows:
            return column_types
        
        num_columns = len(sample_rows[0])
        
        for col_idx in range(num_columns):
            type_counts = Counter()
            
            for row in sample_rows:
                if col_idx < len(row):
                    cell = row[col_idx].strip()
                    if not cell:
                        type_counts['empty'] += 1
                    elif cell.replace('.', '').replace('-', '').replace('e', '').isdigit():
                        type_counts['numeric'] += 1
                    else:
                        type_counts['text'] += 1
            
            # 确定主要类型
            if type_counts:
                main_type = type_counts.most_common(1)[0][0]
                column_types[f"col_{col_idx}"] = main_type
        
        return column_types
    
    def _analyze_generic_table(self, content: str, 
                               table_result: TableAnalysisResult) -> TableAnalysisResult:
        """分析通用表格数据（无明确分隔符）"""
        lines = content.split('\n')
        
        # 统计行数
        table_result.row_count = len([l for l in lines if l.strip()])
        table_result.estimated_total_rows = table_result.row_count
        
        # 检查是否为固定宽度格式
        if table_result.row_count > 5:
            sample_lines = lines[:20]
            line_lengths = [len(l) for l in sample_lines if l.strip()]
            if line_lengths and max(line_lengths) - min(line_lengths) < 10:
                table_result.column_count = 1  # 简化处理
        
        table_result.sample_content = content[:500]
        
        return table_result
    
    def _analyze_structured_data(self, filepath: Path, content: str,
                                 result: SemanticAnalysisResult) -> SemanticAnalysisResult:
        """分析结构化数据文件（SPICE 等）"""
        # 使用智能处理器提取人类可读部分
        human_content = SmartTextProcessor.extract_human_relevant_content(
            content, filepath, max_human_lines=50
        )
        
        result.summary = (
            f"[Structured Data File] Type: {filepath.suffix}\n"
            f"Original lines: {len(content.split('\n'))}\n"
            f"Preserved lines: {len(human_content.split('\n'))}"
        )
        
        result.keywords = self._extract_keywords_from_content(human_content)
        result.metadata["structured_content"] = human_content
        result.metadata["is_truncated"] = len(content.split('\n')) > 50
        
        return result
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """从内容中提取关键词"""
        words = re.findall(r'\b[\w\u4e00-\u9fff]{3,}\b', content.lower())
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 
                     '的', '了', '在', '是', '我', '有', '和', '就'}
        filtered = [w for w in words if w not in stop_words]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(10)]
    
    def _extract_table_sample(self, content: str, table_result: TableAnalysisResult) -> str:
        """提取表格样例"""
        lines = content.split('\n')
        sample_lines = []
        
        for line in lines[:20]:  # 前20行
            stripped = line.strip()
            if not stripped or stripped.startswith(('#', '//', '/*', '*', 'C', 'CC', '!')):
                continue
            sample_lines.append(line)
            
            if len(sample_lines) >= 10:  # 保留10行数据
                break
        
        return '\n'.join(sample_lines)
    
    def _generate_table_summary(self, table_result: TableAnalysisResult) -> str:
        """生成表格数据摘要"""
        parts = []
        
        if table_result.estimated_total_rows:
            parts.append(f"Table data with ~{table_result.estimated_total_rows} rows")
        else:
            parts.append(f"Table data with {table_result.row_count} rows")
        
        if table_result.column_count > 0:
            parts.append(f"  - {table_result.column_count} columns")
        
        if table_result.headers:
            parts.append(f"  - Headers: {', '.join(table_result.headers[:5])}")
        
        if table_result.delimiter:
            delimiter_name = {
                ',': 'comma',
                '\t': 'tab',
                ';': 'semicolon',
                '|': 'pipe',
                ' ': 'space'
            }.get(table_result.delimiter, 'custom')
            parts.append(f"  - Delimiter: {delimiter_name}")
        
        return '\n'.join(parts)


# ==================== 组合配置/表格数据文件分析器 ====================

class CompositeSheetAnalyzer(BaseDataSheetAnalyzer):
    """组合配置/表格数据文件分析器"""
    
    def __init__(self):
        super().__init__()
        self.config_analyzer = ConfigFileAnalyzer()
        self.table_analyzer = TableDataAnalyzer()
    
    def can_handle(self, filepath: Path, content: Optional[str] = None) -> bool:
        """只要有一个分析器能处理就返回True"""
        return (self.config_analyzer.can_handle(filepath, content) or
                self.table_analyzer.can_handle(filepath, content))
    
    def analyze(self, filepath: Path, content: Optional[str] = None) -> SemanticAnalysisResult:
        """使用合适的分析器进行分析"""
        if self.config_analyzer.can_handle(filepath, content):
            return self.config_analyzer.analyze(filepath, content)
        elif self.table_analyzer.can_handle(filepath, content):
            return self.table_analyzer.analyze(filepath, content)
        else:
            # 默认使用表格分析器
            return self.table_analyzer.analyze(filepath, content)


# ==================== 公共 API 导出 ====================

__all__ = [
    'ConfigAnalysisResult',
    'TableAnalysisResult',
    'ConfigFileAnalyzer',
    'TableDataAnalyzer',
    'CompositeSheetAnalyzer',
]
