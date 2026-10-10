# mission_sim/utils/dynamics/mlx_screen.py
# -*- coding: utf-8 -*-
"""
MLX/Metal fp32 批量筛选层（可选依赖；缺省自动降级）
====================================================
AS-CR-01 / G5⑦ 沉淀件之二：

  · 仅用于**筛选**（fp32）：批量半周期残差/分类，结果一律以 fp64 复核；
  · GPU 不支持 fp64（实测），故不作精确层；
  · 环境变量 `MCPC_DISABLE_MLX=1` 或缺省未安装 mlx 时 → `screening_available()=False`，
    调用方自动回退 numba/numpy 路径（降级路径，T-AS-009③）。
"""
import os

import numpy as np

try:
    if os.environ.get("MCPC_DISABLE_MLX", "0") == "1":
        raise ImportError("disabled by MCPC_DISABLE_MLX")
    import mlx.core as mx
    _MLX_OK = True
except ImportError:  # pragma: no cover
    mx = None
    _MLX_OK = False

_STEP_CACHE = {}


def screening_available() -> bool:
    """筛选层是否可用（动态读取环境开关）。"""
    if os.environ.get("MCPC_DISABLE_MLX", "0") == "1":
        return False
    return _MLX_OK


def _rhs_mlx(x, mu):
    xa, ya, za, vx, vy, vz = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    muf = mx.array(np.float32(mu))
    r1 = mx.sqrt((xa + muf) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    r2 = mx.sqrt((xa + muf - 1.0) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    ax = 2 * vy + xa - (1 - muf) * (xa + muf) / r1 ** 3 - muf * (xa + muf - 1.0) / r2 ** 3
    ay = -2 * vx + ya - (1 - muf) * ya / r1 ** 3 - muf * ya / r2 ** 3
    az = -(1 - muf) * za / r1 ** 3 - muf * za / r2 ** 3
    return mx.stack([vx, vy, vz, ax, ay, az], axis=1)


def _make_step_fn(chunk, h, mu):
    key = (int(chunk), float(h), float(mu))
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]

    @mx.compile
    def step(x):
        for _ in range(chunk):
            k1 = _rhs_mlx(x, mu)
            k2 = _rhs_mlx(x + 0.5 * h * k1, mu)
            k3 = _rhs_mlx(x + 0.5 * h * k2, mu)
            k4 = _rhs_mlx(x + h * k3, mu)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    _STEP_CACHE[key] = step
    return step


def _chunk_for(n_steps):
    chunk = min(200, n_steps)
    while chunk > 1 and n_steps % chunk != 0:
        chunk -= 1
    return chunk


def batch_half_period_residual_fp32(seeds, T_half, n_steps, mu):
    """批量半周期截面残差 [y, ẋ]（fp32，筛选用）。

    分批块 mx.compile 融合；批量越大 GPU 效率越高（≥65536 饱和，见 AS-CR-01）。
    """
    if not _MLX_OK:
        raise RuntimeError("mlx 不可用（筛选层缺省降级）")
    seeds = np.asarray(seeds, dtype=np.float32)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    n_steps = int(n_steps)
    h = float(T_half) / n_steps
    chunk = _chunk_for(n_steps)
    step = _make_step_fn(chunk, h, float(mu))
    x = mx.array(seeds)
    for _ in range(n_steps // chunk):
        x = step(x)
    mx.eval(x)
    out = np.asarray(x)
    return np.stack([out[:, 1].astype(np.float64), out[:, 3].astype(np.float64)], axis=1)
