# -*- coding: utf-8 -*-
"""
detector.py - 文件类型检测器
"""

import re
from pathlib import Path
from typing import Optional

from ..constants import FileType, EXTENSION_MAPPING


class FileTypeDetector:
    """智能文件类型检测器"""
    
    @staticmethod
    def detect_by_extension(filepath: Path) -> Optional[FileType]:
        """通过扩展名检测文件类型"""
        suffix = filepath.suffix.lower()
        
        for file_type, extensions in EXTENSION_MAPPING.items():
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
