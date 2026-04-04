# mission_sim/core/physics/models/gravity_crtbp.py
"""
MCPC Core Physics Models: Dual-Engine Vectorized CRTBP
------------------------------------------------------
注意：此文件已过时，请使用 mission_sim.core.physics.models.gravity.universal_crtbp 中的类。

此文件已废弃，仅用于向后兼容。请导入：
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    然后使用 UniversalCRTBP.sun_earth_system() 创建实例。
"""

import warnings
import numpy as np
from mission_sim.core.physics.environment import IForceModel

# 发出弃用警告
warnings.warn(
    "gravity_crtbp.py is deprecated and will be removed in a future version. "
    "Please use UniversalCRTBP.sun_earth_system() from mission_sim.core.physics.models.gravity.universal_crtbp instead.",
    DeprecationWarning,
    stacklevel=2
)

# 从新的重力模块导入
try:
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    
    # 创建向后兼容的适配器类，保持原有接口
    class GravityCRTBP(IForceModel):
        """
        向后兼容的适配器类，保持原有 GravityCRTBP 的接口。
        实际实现使用 UniversalCRTBP 的 Sun-Earth 系统。
        """
        def __init__(self):
            # 创建 UniversalCRTBP 的 Sun-Earth 系统实例
            self._universal_crtbp = UniversalCRTBP.sun_earth_system(use_numba=True)
        
            # 保持原 GravityCRTBP 的常量名以便兼容
            # 直接从 UniversalCRTBP 实例获取常量值
            self.GM_SUN = self._universal_crtbp.GM_SUN    # m³/s²
            self.GM_EARTH = self._universal_crtbp.GM_EARTH    # m³/s²
            self.AU = self._universal_crtbp.AU          # m
            # 注意：不设置 self.OMEGA，通过 property 访问
        
            # 计算原私有属性
            self.mu = self.GM_EARTH / (self.GM_SUN + self.GM_EARTH)
            self._x1 = -self.mu * self.AU
            self._x2 = (1.0 - self.mu) * self.AU
        
        @property
        def OMEGA(self) -> float:
            """通过 universal_crtbp 的 property 访问"""
            return self._universal_crtbp.OMEGA
        
        @property
        def _omega_sq(self):
            """兼容旧代码，计算 omega 的平方"""
            return self.OMEGA**2
        
        def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
            """[L1 LEGACY] Compute acceleration for a SINGLE spacecraft."""
            # 使用 UniversalCRTBP 的实现
            return self._universal_crtbp.compute_accel(state, epoch)
        
        def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
            """
            [L2-SPECIFIC / PARALLELIZATION] 
            Batch compute accelerations for N spacecraft.
            """
            # 使用 UniversalCRTBP 的向量化实现
            return self._universal_crtbp.compute_vectorized_acc(state_matrix, epoch)
        
        def __repr__(self) -> str:
            # 保持原 GravityCRTBP 的字符串表示
            return f"GravityCRTBP(mu={self.mu:.2e}, OMEGA={self.OMEGA:.2e})"
    
    # 导出类
    __all__ = ['GravityCRTBP']
    
except ImportError as e:
    # 如果无法导入新模块，提供更详细的错误信息
    raise ImportError(
        f"Cannot import UniversalCRTBP from mission_sim.core.physics.models.gravity.universal_crtbp. "
        f"Please check if the new gravity module is properly installed. Error: {e}"
    )
