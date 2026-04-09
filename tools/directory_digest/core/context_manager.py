# -*- coding: utf-8 -*-
"""
context_manager.py - LLM上下文管理器
"""

from typing import Dict, List, Any
from ..constants import ProcessingStrategy


class ContextManager:
    """LLM上下文管理器"""
    
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.reserved_tokens = 4000
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
