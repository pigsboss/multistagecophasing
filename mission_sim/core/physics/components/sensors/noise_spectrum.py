# mission_sim/core/physics/components/sensors/noise_spectrum.py
"""
MCPC L4: 传感器/环境噪声谱生成（Allan 方差近似）
------------------------------------------------
按 DD-001 R-L4-02：提供工程噪声谱的近似生成（白噪声 / 闪烁 1/f / 随机游走），
配合 Allan 偏差斜率 {−1/2, 0, +1/2} 判定（T-L4-003）与 L4 评估注入。
"""

import numpy as np


class AllanNoiseGenerator:
    """按 Allan 噪声类型生成时间序列（幅度归一化到 sigma）。"""

    KINDS = ("white", "flicker", "random_walk")
    _FLICKER_FLOOR_HZ = 0.02  # 1/f 整形的低频保护（避免发散的慢趋势）

    def __init__(self, kind: str = "white", sigma: float = 1.0, seed: int = 20261010):
        if kind not in self.KINDS:
            raise ValueError(f"未知噪声类型 {kind}（可用：{self.KINDS}）")
        self.kind = kind
        self.sigma = float(sigma)
        self.seed = int(seed)

    def generate(self, n: int, dt: float = 1.0) -> np.ndarray:
        """生成长度 n、步长 dt 的噪声序列。"""
        rng = np.random.default_rng(self.seed)
        if self.kind == "white":
            x = rng.normal(0.0, 1.0, n)
        elif self.kind == "random_walk":
            x = np.cumsum(rng.normal(0.0, 1.0, n))
        else:  # flicker：频域 1/√f 整形（1/f 功率谱）
            w = rng.normal(0.0, 1.0, n)
            W = np.fft.rfft(w)
            f = np.fft.rfftfreq(n, d=float(dt))
            W[0] = 0.0
            W[1:] = W[1:] / np.sqrt(f[1:] + self._FLICKER_FLOOR_HZ)
            x = np.fft.irfft(W, n=n)
        x = x - float(np.mean(x))
        s = float(np.std(x))
        if s > 0.0:
            x = x / s * self.sigma
        return x
