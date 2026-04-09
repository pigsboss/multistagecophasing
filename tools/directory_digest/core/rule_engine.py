# -*- coding: utf-8 -*-
"""
rule_engine.py - 规则引擎
"""

import fnmatch
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..constants import ProcessingStrategy
from ..types import FileRule

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("警告: PyYAML 库未安装，YAML文件解析功能受限", file=sys.stderr)


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
            
            # Zone.Identifier文件（Windows备用数据流）
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
        
        # 从 constants.py 导入 STRATEGY_CONFIGS
        from ..constants import STRATEGY_CONFIGS
        
        config = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY])
        
        # 估算字符数（保守估计）
        if strategy == ProcessingStrategy.METADATA_ONLY:
            return int(config['token_estimate'] * 100)  # 约100字符的元数据
        
        # 对于内容策略，根据文件大小估算
        if 'max_size' in config and config['max_size'] and file_size > config['max_size']:
            # 文件太大，使用元数据策略
            return int(STRATEGY_CONFIGS[ProcessingStrategy.METADATA_ONLY]['token_estimate'] * 100)
        
        # 字符数估算（保守）
        max_size = config.get('max_size')
        estimated_chars = min(file_size, max_size or file_size)
        return int(estimated_chars * config['token_estimate'])
