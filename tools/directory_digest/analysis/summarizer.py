# -*- coding: utf-8 -*-
"""
summarizer.py - 人类可读文本摘要生成器
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

from ..types import HumanReadableSummary


class HumanReadableSummarizer:
    """人类可读文本摘要生成器"""
    
    @staticmethod
    def summarize(filepath: Path, content: str, max_lines: int = 10) -> HumanReadableSummary:
        """生成人类可读文本摘要"""
        if not content:
            return HumanReadableSummary(
                line_count=0,
                word_count=0,
                character_count=0
            )
        
        # 分割行
        lines = content.split('\n')
        line_count = len(lines)
        
        # 统计基本信息
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', content)
        word_count = len(words)
        
        # 提取标题
        title = HumanReadableSummarizer._extract_title(filepath, content, lines)
        
        # 提取首尾行
        first_lines = lines[:min(5, len(lines))]
        last_lines = lines[-min(3, len(lines)):] if len(lines) > 3 else []
        
        return HumanReadableSummary(
            title=title,
            line_count=line_count,
            word_count=word_count,
            character_count=len(content),
            first_lines=first_lines,
            last_lines=last_lines,
            summary=f"File: {filepath.name}, Lines: {line_count}, Words: {word_count}"
        )
    
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
            (r'^#\s+(.+)$', 1),        # # 标题
            (r'^##\s+(.+)$', 1),       # ## 标题
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
        
        return None
