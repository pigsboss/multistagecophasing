# -*- coding: utf-8 -*-
"""
directory_digest - 目录知识摘要器包
"""

from .digest import DirectoryDigest
from .cli import main

__version__ = "1.0.0"
__all__ = ["DirectoryDigest", "main"]
