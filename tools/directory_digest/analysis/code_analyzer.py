# -*- coding: utf-8 -*-
"""
code_analyzer.py - 源代码分析器
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..types import SourceCodeAnalysis


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
        else:
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
                    "args": 0,
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
        if suffix in ['.py', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.rb']:
            comment_patterns = ['#']
        elif suffix in ['.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs']:
            comment_patterns = ['//', '/*']
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
            functions=[],
            classes=[],
            global_vars=[],
            constants=[],
            dependencies=[]
        )
    
    def _extract_config_structure(self, content: str, suffix: str) -> List[Dict]:
        """从配置文件中提取结构信息"""
        keys = []
        try:
            if suffix in ['.ini', '.cfg', '.conf', '.env', '.rc']:
                for i, line in enumerate(content.split('\n')[:50]):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith(';'):
                        match = re.match(r'^\s*([^=;#\s]+)\s*=', line)
                        if match:
                            keys.append({"name": match.group(1).strip(), "line": i+1, "type": "config_key"})
        except Exception:
            pass
        return keys
