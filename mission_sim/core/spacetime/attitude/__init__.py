"""Attitude models package（最小收缩重建，DD-001 D-09 / R-L4-06）。

- 本包仅保留通用接口 `AttitudeModel`；
- 天体指向/参考系姿态：使用 `spice_interface`（pxform 等）；
- 航天器姿态动力学（四元数）：见 `mission_sim/core/physics/attitude_dynamics.py`。

历史说明：原 `__init__` 引用的 config/models/interpolated 子模块不存在（断链导入），
按 R-L4-06 决策收缩重建（不复活旧职责）。
"""

from .base import AttitudeModel

__all__ = ["AttitudeModel"]
