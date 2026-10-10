#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNAR-SWING 高保真演化实验（30 天，三组对照）
==============================================
LS-001 G3 / W7：把严格 2:1 对称周期解搬入高保真耦合模型演化，回答"有界性问题"。

组别（同初态、同积分器）：
    A: J2 仅（无月球/无阻力）
    B: J2 + 月球第三体（SPICE 真实星历，缓存插值加速）
    C: J2 + 月球 + 简化指数大气（面积质量比 0.02 m²/kg）

输出：`docs/excursion/lunar_swing/results/high_fidelity/` 下各组 HDF5 + summary.json。

用法: python scripts/lunar_swing/run_highfidelity_evolution.py [--days 30] [--no-spice]
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.simulation.threebody.earth_moon import (
    EarthMoonResonanceSimulation, crtbp_state_to_eci)

CHAR_PERIOD = 4.342 * 86400.0


def solve_orbit():
    crtbp = UniversalCRTBP.earth_moon_system()
    tg = LunarSwingTargeter(dynamics_model=crtbp, mu=crtbp.mu, num_steps=40000)
    res = tg.find_symmetric_periodic_orbit(
        resonance_ratio=(2, 1), x0_guess=0.85, vy0_guess=0.95,
        tol=1e-10, max_iter=50, char_period=CHAR_PERIOD, verbose=True)
    assert res["success"], "2:1 严格解求解失败"
    return crtbp, res


def make_spice_provider(epoch0_et: float):
    from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
    from mission_sim.core.spacetime.ids import CoordinateFrame
    spice = SPICEInterface()
    if not spice.initialize("earth_moon"):
        raise RuntimeError("SPICE initialize failed")
    frame = CoordinateFrame.J2000_ECI

    def prov(t):
        return np.asarray(spice.get_state("moon", epoch0_et + float(t),
                                          observer="earth", frame=frame), dtype=np.float64)
    return prov


def make_circular_provider():
    n = 2.0 * np.pi / (27.321661 * 86400.0)

    def prov(t):
        th = n * float(t)
        r = 3.844e8 * np.array([np.cos(th), np.sin(th), 0.0])
        v = 3.844e8 * n * np.array([-np.sin(th), np.cos(th), 0.0])
        return np.concatenate([r, v])
    return prov


def make_cached_provider(base_prov, t0: float, t1: float, dt_cache: float = 60.0):
    """把逐点 SPICE 调用缓存为网格插值（大幅加速 RHS）。"""
    grid = np.arange(t0, t1 + dt_cache, dt_cache)
    vals = np.array([np.asarray(base_prov(t), dtype=np.float64) for t in grid])

    def cached(t):
        t = float(np.clip(t, t0, t1))
        out = np.empty(6)
        for j in range(6):
            out[j] = np.interp(t, grid, vals[:, j])
        return out
    return cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--epoch0-et", type=float, default=0.0, help="起始历元（TDB，J2000 起，秒）")
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true", help="SPICE 不可用时回退确定性月球")
    ap.add_argument("--outdir", default="docs/excursion/lunar_swing/results/high_fidelity")
    args = ap.parse_args()

    outdir = Path(outdir_abs := (project_root / args.outdir)) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LUNAR-SWING High-Fidelity Evolution Experiment")
    print("=" * 70)

    crtbp, res = solve_orbit()
    print(f"2:1 严格解: 闭合残差={res['full_state_residual']:.2e}, Jacobi 漂移={res['jacobi_drift']:.2e}")
    print(f"state_nd = {np.array2string(np.asarray(res['state']), precision=10)}")

    mode = "keplerian" if args.no_spice else "spice"
    try:
        base_prov = make_circular_provider() if args.no_spice else make_spice_provider(args.epoch0_et)
        if not args.no_spice:
            print("月球星历: SPICE（真实）")
        else:
            print("月球星历: 确定性圆形（离线）")
    except Exception as e:  # noqa: BLE001
        if not args.allow_fallback:
            print(f"❌ SPICE 不可用：{e}（如需离线回退请加 --allow-fallback 或 --no-spice）")
            return 1
        print(f"⚠️ SPICE 不可用（{e}），回退确定性月球")
        mode = "keplerian"
        base_prov = make_circular_provider()

    moon_prov = make_cached_provider(base_prov, 0.0, args.days * 86400.0, dt_cache=60.0)

    moon0 = np.asarray(base_prov(0.0), dtype=np.float64)
    L, omega = crtbp.distance, crtbp.omega
    state_eci0 = crtbp_state_to_eci(np.asarray(res["state"]), moon0[:3], moon0[3:],
                                    L, omega, crtbp.mu)
    print(f"ECI 初态: {np.array2string(state_eci0, precision=6)}")

    runs = [
        ("A_j2only", dict(enable_j2=True, enable_moon=False, enable_drag=False, j3=0.0, j4=0.0)),
        ("B_plus_moon", dict(enable_j2=True, enable_moon=True, enable_drag=False, j3=0.0, j4=0.0)),
        ("C_plus_drag", dict(enable_j2=True, enable_moon=True, enable_drag=True, j3=0.0, j4=0.0)),
    ]
    summaries = {}
    for name, opts in runs:
        cfg = dict(mission_name=f"LS_{name}", simulation_days=args.days, time_step=600.0,
                   data_dir=str(outdir), initial_state=state_eci0.tolist(),
                   moon_provider=moon_prov, verbose=False,
                   rtol=3e-10, atol=1e-6, **opts)
        sim = EarthMoonResonanceSimulation(cfg)
        ok = sim.run()
        summaries[name] = {"ok": bool(ok), **(sim.summary or {})}
        s = summaries[name]
        print(f"[{name}] ok={ok} bounded={s.get('bounded')} "
              f"alt_min={s.get('alt_min_m', float('nan'))/1e3:.1f} km "
              f"decay={s.get('alt_decay_m_per_day', float('nan'))/1e3:.3f} km/day")

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps({
        "mode": mode,
        "days": args.days,
        "epoch0_et": args.epoch0_et,
        "state_nd": np.asarray(res["state"]).tolist(),
        "closure_residual": res["full_state_residual"],
        "state_eci0": state_eci0.tolist(),
        "summaries": summaries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 完成，结果目录: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
