#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNAR-SWING 30 天演化实验剖析（报告数据源）
============================================
读取 `run_highfidelity_evolution.py` 输出（HDF5 + summary.json），提取：
  · 高度极值及时间（近/远地点节律）
  · 月球交会最近距离
  · 两体视半长轴 a(t) 统计与 30 天漂移（维持 Δv 估算方法的一阶数据）
  · B/C 组一致性核对

用法: python scripts/lunar_swing/analyze_evolution.py [results_dir]
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from scipy.signal import find_peaks

MU_E = 3.986004418e14
R_E = 6378137.0


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        project_root / "docs/excursion/lunar_swing/results/high_fidelity"
    summ = json.load(open(base / "summary.json", encoding="utf-8"))
    print(f"# {base}")
    print(f"# mode={summ['mode']}  days={summ['days']}  闭合残差={summ['closure_residual']:.3e}")
    print(f"# ECI 初态 = {np.array2string(np.asarray(summ['state_eci0']), precision=3)}")

    for name in ["A_j2only", "B_plus_moon", "C_plus_drag"]:
        f = h5py.File(base / f"LS_{name}.h5", "r")
        t = f["epochs"][:] / 86400.0
        st = f["states"][:]
        alt = f["altitude_m"][:] / 1e3
        md = f["moon_distance_m"][:] / 1e3
        r = np.linalg.norm(st[:, 0:3], axis=1)
        v2 = np.sum(st[:, 3:6] ** 2, axis=1)
        a_osc = 1.0 / (2.0 / r - v2 / MU_E) / 1e3  # 两体视半长轴 km

        pk, _ = find_peaks(alt, distance=144)   # ≥1 天
        tr, _ = find_peaks(-alt, distance=144)
        print(f"\n[{name}]")
        print(f"  alt: init={alt[0]:.0f} min={alt.min():.0f}@{t[alt.argmin()]:.2f}d "
              f"max={alt.max():.0f}@{t[alt.argmax()]:.2f}d final={alt[-1]:.0f} km")
        print(f"  apogees(d): {[round(t[i], 2) for i in pk[:8]]}")
        print(f"  perigees(d): {[round(t[i], 2) for i in tr[:8]]}")
        if np.isfinite(md).any():
            i0 = int(np.nanargmin(md))
            print(f"  moon: min={md[i0]:.0f} km @ {t[i0]:.2f} d (init {md[0]:.0f} km)")
        print(f"  a_osc(km): init={a_osc[0]:.0f} min={a_osc.min():.0f} max={a_osc.max():.0f} "
              f"final={a_osc[-1]:.0f}  Δ(首末)={a_osc[-1]-a_osc[0]:+.0f}")
        f.close()

    # B/C 轨迹一致性（最大分量差）
    with h5py.File(base / "LS_B_plus_moon.h5", "r") as fb, \
         h5py.File(base / "LS_C_plus_drag.h5", "r") as fc:
        d = np.abs(fb["states"][:] - fc["states"][:]).max()
        dm = np.abs(fb["altitude_m"][:] - fc["altitude_m"][:]).max()
    print(f"\nB vs C: max|Δstate|={d:.3e} (alt {dm:.3e} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
