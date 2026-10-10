# mission_sim/core/physics/payload/platform_payload.py
"""
MCPC L3: 平台-载荷双层模型（简化）
----------------------------------
按 DD-001 R-L3-01 / D-06（不越界）：
- 解耦假设：载荷质量 ≪ 平台质量（比例 ≤ limit）时，机构反作用力矩可忽略，
  机构耦合降维为纯运动学映射（由调用方装配）；
- 平台抖动表示：确定性正弦叠加 + 带限随机相位分量（频段 1–100 Hz，seed 可复现），
  作为 OPD 链路的扰动源。
"""

import numpy as np


class PayloadDecouplingError(RuntimeError):
    """解耦假设越界（载荷质量比超过上限）。"""


class PlatformPayloadModel:
    """平台-载荷双层模型（简化）。

    参数：
        platform_mass_kg / payload_mass_kg: 平台与载荷质量
        decoupling_mass_ratio_limit: 解耦假设允许的最大质量比（默认 0.10）
        seed: 抖动序列随机相位种子（可复现）
    """

    # 确定性频点 (Hz) 与幅值 (rad/轴)：能量集中在 1–40 Hz，高频小量
    _DET_COMPONENTS = [
        (1.0, 1.2e-7), (3.0, 9.0e-8), (8.0, 5.0e-8),
        (20.0, 1.5e-8), (40.0, 8.0e-9), (80.0, 4.0e-9),
    ]
    _RAND_BAND_HZ = (1.0, 100.0)
    _RAND_SIGMA_RAD = 2.0e-8
    _RAND_N_COMPONENTS = 100

    def __init__(self, platform_mass_kg: float = 1000.0, payload_mass_kg: float = 50.0,
                 decoupling_mass_ratio_limit: float = 0.10, seed: int = 20261010):
        self.platform_mass_kg = float(platform_mass_kg)
        self.payload_mass_kg = float(payload_mass_kg)
        self.decoupling_mass_ratio_limit = float(decoupling_mass_ratio_limit)
        self.seed = int(seed)
        self.mass_ratio = self.payload_mass_kg / self.platform_mass_kg

    def validate_decoupling(self) -> bool:
        """校验解耦假设；越界抛 PayloadDecouplingError（检查并报警，不静默降级）。"""
        if self.mass_ratio > self.decoupling_mass_ratio_limit + 1e-12:
            raise PayloadDecouplingError(
                f"载荷/平台质量比 {self.mass_ratio:.3f} 超过解耦假设上限 "
                f"{self.decoupling_mass_ratio_limit:.3f}：机构反作用力矩不可再忽略，"
                f"需升级为全耦合多体模型（见 DD-001 R-L3-01/D-06）。"
            )
        return True

    def jitter_series(self, t_array: np.ndarray) -> np.ndarray:
        """生成平台抖动角序列 (N, 2)（rad；两轴独立相位）。"""
        t = np.asarray(t_array, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        out = np.zeros((t.size, 2), dtype=np.float64)

        for axis in range(2):
            sig = np.zeros(t.size, dtype=np.float64)
            for freq, amp in self._DET_COMPONENTS:
                phase = rng.uniform(0.0, 2.0 * np.pi)
                sig += amp * np.sin(2.0 * np.pi * freq * t + phase)

            freqs = rng.uniform(self._RAND_BAND_HZ[0], self._RAND_BAND_HZ[1], self._RAND_N_COMPONENTS)
            phases = rng.uniform(0.0, 2.0 * np.pi, self._RAND_N_COMPONENTS)
            rand = np.sum(np.sin(2.0 * np.pi * np.outer(t, freqs) + phases), axis=1)
            rand = rand / np.sqrt(self._RAND_N_COMPONENTS / 2.0)  # 归一化到单位 RMS
            sig += self._RAND_SIGMA_RAD * rand
            out[:, axis] = sig
        return out

    def __repr__(self) -> str:
        return (f"PlatformPayloadModel(m_plat={self.platform_mass_kg:.0f} kg, "
                f"m_pl={self.payload_mass_kg:.0f} kg, ratio={self.mass_ratio:.3f})")
