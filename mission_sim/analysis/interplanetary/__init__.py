# mission_sim/analysis/interplanetary/__init__.py
# -*- coding: utf-8 -*-
"""行星际转移窗口分析（Ceres S4 能力沉淀件）。

* ``two_body``：日心二体平根数星历 + Lambert 转移内核（星历/动力学原语）。
* ``target_scan``：多目标（主带天体）发射—TOF 网格扫描器（含 ``python3 -m`` CLI 与
  ``--self-test``，直接以子模块方式运行；本包 ``__init__`` 不做预导入以免 runpy 重复装载）。
"""
