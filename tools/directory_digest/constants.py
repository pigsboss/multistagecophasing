# -*- coding: utf-8 -*-
"""
constants.py - 常量定义
"""

from enum import Enum

# ==================== 策略枚举和配置 ====================

class ProcessingStrategy(Enum):
    """文件处理策略枚举"""
    FULL_CONTENT = "full_content"              # 全量嵌入（仅限小文件）
    SUMMARY_ONLY = "summary_only"              # 生成文本摘要
    CODE_SKELETON = "code_skeleton"           # 代码骨架（函数/类/导入）
    STRUCTURE_EXTRACT = "structure_extract"    # 提取结构（键/节）
    HEADER_WITH_STATS = "header_with_stats"    # 头部+统计信息
    METADATA_ONLY = "metadata_only"           # 仅元数据


class FileType(Enum):
    """文件类型枚举"""
    CRITICAL_DOCS = "critical_docs"      # 关键文档
    REFERENCE_DOCS = "reference_docs"    # 参考文档
    SOURCE_CODE = "source_code"          # 源代码
    TEXT_DATA = "text_data"              # 文本数据
    BINARY_FILES = "binary_files"        # 二进制文件
    UNKNOWN = "unknown"                  # 未知类型


class OutputFormats(Enum):
    """支持的所有输出格式"""
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "md"
    HTML = "html"
    TOML = "toml"
    PLAINTEXT = "txt"


# 策略配置映射
STRATEGY_CONFIGS = {
    ProcessingStrategy.FULL_CONTENT: {
        'token_estimate': 0.25,
        'max_size': 100 * 1024,  # 100KB
    },
    ProcessingStrategy.SUMMARY_ONLY: {
        'token_estimate': 0.05,
        'max_lines': 50,
    },
    ProcessingStrategy.CODE_SKELETON: {
        'token_estimate': 0.02,
        'include_functions': True,
        'include_classes': True,
    },
    ProcessingStrategy.STRUCTURE_EXTRACT: {
        'token_estimate': 0.03,
        'max_keys': 20,
    },
    ProcessingStrategy.HEADER_WITH_STATS: {
        'token_estimate': 0.01,
        'max_lines': 10,
        'include_stats': True,
    },
    ProcessingStrategy.METADATA_ONLY: {
        'token_estimate': 0.001,
    },
}


# 扩展名到类型的映射（优先级1）
EXTENSION_MAPPING = {
    FileType.REFERENCE_DOCS: [
        '.md', '.markdown', '.rst', '.html', '.htm'
    ],
    FileType.SOURCE_CODE: [
        '.py', '.java', '.cpp', '.c', '.h', '.hpp',
        '.js', '.ts', '.jsx', '.tsx',
        '.go', '.rs', '.rb', '.php', '.swift',
        '.sh', '.bash', '.ps1', '.bat', '.cmd',
        '.css', '.scss', '.less'
    ],
    FileType.TEXT_DATA: [
        '.txt', '.log', '.csv', '.tsv',
        '.yaml', '.yml', '.json', '.xml', '.toml', 
        '.ini', '.cfg', '.conf', '.env',
        '.tf', '.tls', '.tpc', '.ker', '.cmt'
    ],
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
