# -*- coding: utf-8 -*-
"""
text_processor.py - 智能文本处理器
"""

import re
from pathlib import Path
from typing import List


class SmartTextProcessor:
    """
    智能文本处理器 - 提取人类关心的内容，截断机器数据结构
    """
    
    # 文件扩展名到处理策略的映射
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
        if suffix in ['.tf', '.tls', '.tpc', '.ker']:
            return SmartTextProcessor._extract_spice_content(lines, filepath, content)
        
        # 处理 CSV 文件
        if suffix == '.csv':
            return SmartTextProcessor._extract_csv_content(lines, filepath)
        
        # 默认：返回前 N 行
        return '\n'.join(lines[:max_human_lines])
    
    @staticmethod
    def _extract_spice_content(lines: List[str], filepath: Path, full_content: str) -> str:
        """
        提取 SPICE 文件的人类可读部分
        """
        header_lines = []
        in_text_block = False
        data_started = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测 SPICE 标记
            if '\\begintext' in stripped:
                in_text_block = True
                header_lines.append(line)
                continue
            elif '\\begindata' in stripped:
                data_started = True
                header_lines.append(line)
                break
            
            # 收集注释行
            if (stripped.startswith('C') or 
                stripped.startswith('*') or 
                stripped.startswith('/*') or
                stripped.startswith('#') or
                stripped.startswith('CC') or
                in_text_block):
                header_lines.append(line)
            elif not data_started and not stripped.startswith('\\'):
                # 文件开头的非数据行
                if not re.match(r'^\s*[\d\.\-\+eE\s]+$', stripped):
                    header_lines.append(line)
        
        # 构建结果
        result_lines = header_lines[:50]
        
        # 添加数据结构摘要
        total_lines = len(lines)
        if data_started or len(header_lines) < total_lines:
            result_lines.append(f"\n[DATA SECTION TRUNCATED]")
            result_lines.append(f"File type: SPICE Kernel ({filepath.suffix})")
            result_lines.append(f"Total lines: {total_lines}")
            result_lines.append(f"Preserved: {len(header_lines)} lines of metadata/comments")
        
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
            header_lines.append("")
        
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
        
        return '\n'.join(header_lines)
