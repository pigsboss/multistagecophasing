#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS W7 实验：低近地点（~197 km）2:1 惯性共振近失解 30 天演化
================================================================
- IC：对称谷底最优近失解 state_nd = [0.0049602039, 0, 0, 0, 10.6562923, 0]
  （半周期残差 2.5e-4；**非严格周期解**——族域勘测负结果留档见 AS-005）
- 组别：G1=J2+moon（无阻力）；G2=+阻力（ρ×1.0）；G3/G4=ρ×0.5 / ρ×1.5（±50% 敏感度）
- 后处理：近地点事件、月球近距事件、大气阈值穿越；能量账本（ΔE_mech = W_moon+W_J2J4+W_drag）
用法: python3 scripts/atmospheric_swing/run_lowperigee_experiment.py [--days 30]
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np

from mission_sim.analysis.energy_ledger import ledger_closure
from mission_sim.core.cyber.algorithms.low_perigee import CHAR_PERIOD
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.simulation.threebody.earth_moon import (
    EarthMoonResonanceSimulation, crtbp_state_to_eci)
from mission_sim.utils.dynamics import events as ev

STATE_ND = [0.0049602039, 0.0, 0.0, 0.0, 10.6562923, 0.0]
T_SC_DAYS = 13.6608305
MU_E = 3.986004418e14
R_E = 6378137.0
A_M = 0.02  # m²/kg
OUTDIR = "docs/excursion/lunar_swing/results/as_lowperigee"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--epoch0-et", type=float, default=0.0)
    args = ap.parse_args()

    outdir = project_root / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    c = UniversalCRTBP.earth_moon_system()
    L, om, mu_nd = c.distance, c.omega, c.mu

    # SPICE 月球初态 → ECI 初态
    from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
    from mission_sim.core.spacetime.ids import CoordinateFrame
    spice = SPICEInterface()
    assert spice.initialize("earth_moon"), "SPICE 初始化失败"
    frame = CoordinateFrame.J2000_ECI
    moon0 = np.asarray(spice.get_state("moon", args.epoch0_et, observer="earth", frame=frame), dtype=np.float64)
    st_eci = crtbp_state_to_eci(np.asarray(STATE_ND), moon0[:3], moon0[3:], L, om, mu_nd)
    print("ECI 初态:", np.array2string(st_eci, precision=6))

    def moon_prov(t):
        return np.asarray(spice.get_state("moon", args.epoch0_et + float(t), observer="earth",
                                          frame=frame), dtype=np.float64)

    runs = {
        "G1_nodrag": dict(enable_drag=False, rho_scale=1.0),
        "G2_drag1.0": dict(enable_drag=True, rho_scale=1.0),
        "G3_drag0.5": dict(enable_drag=True, rho_scale=0.5),
        "G4_drag1.5": dict(enable_drag=True, rho_scale=1.5),
    }
    summaries = {}
    for name, opt in runs.items():
        cfg = dict(mission_name=f"AS_{name}", simulation_days=args.days, time_step=30.0,
                   data_dir=str(outdir), initial_state=st_eci.tolist(),
                   moon_provider=moon_prov, verbose=False, rtol=3e-10, atol=1e-6,
                   enable_j2=True, j3=0.0, j4=0.0,
                   enable_moon=True, enable_drag=opt["enable_drag"],
                   rho0=1.225 * opt["rho_scale"], area_to_mass=A_M, Cd=2.2, H=8500.0)
        sim = EarthMoonResonanceSimulation(cfg)
        ok = sim.run()
        s = {"ok": bool(ok), **(sim.summary or {})}
        summaries[name] = s
        print(f"[{name}] ok={ok} bounded={s.get('bounded')} "
              f"alt_min={s.get('alt_min_m', np.nan)/1e3:.1f} km "
              f"alt_max={s.get('alt_max_m', np.nan)/1e3:.1f} km "
              f"alt_final={s.get('alt_final_m', np.nan)/1e3:.1f} km", flush=True)

    # ---- 后处理：事件 + 账本（G1/G2） ----
    print("\n== 事件检测 ==")
    for name in ["G1_nodrag", "G2_drag1.0"]:
        f = h5py.File(outdir / f"AS_{name}.h5", "r")
        t = f["epochs"][:]
        st = f["states"][:]
        md = f["moon_distance_m"][:]
        alt = f["altitude_m"][:]
        per = ev.perigee_events(t, st, r_earth=R_E)
        app = ev.approach_events(t, md, 60_000e3)
        atm = ev.atmosphere_events(t, st, h_threshold=300e3, r_earth=R_E)
        print(f"[{name}] 近地点 {len(per)} 次: " + ", ".join(
            f"{p['t']/86400:.2f}d@{p['altitude']/1e3:.0f}km" for p in per[:8]))
        print(f"          月球<6万km 区间 {len(app)} 段; 最低距月 {np.nanmin(md)/1e3:.0f} km"
              f" @ {t[int(np.nanargmin(md))]/86400:.2f} d")
        print(f"          大气 300km 阈值穿越: 进入 {len(atm['entries'])} 次 / 退出 {len(atm['exits'])} 次")
        f.close()

    print("\n== 能量账本（第一周期 ≈ %.2f d；扰动力 = J2–J4 + 月球 + 阻力） ==" % T_SC_DAYS)
    from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
    from mission_sim.core.physics.models.third_body import ThirdBodyGravity
    from mission_sim.core.physics.models.atmospheric_drag import AtmosphericDrag
    geo = HighOrderGeopotential(mu_earth=MU_E, j3=0.0, j4=0.0)
    third = ThirdBodyGravity(mu_third=4.9048695e12, position_provider=moon_prov)
    drag = AtmosphericDrag(area_to_mass=A_M, Cd=2.2, rho0=1.225, H=8500.0, h0=0.0, R_earth=R_E)

    ledgers = {}
    rho_scale = {"G1_nodrag": 1.0, "G2_drag1.0": 1.0, "G3_drag0.5": 0.5, "G4_drag1.5": 1.5}
    for name in ["G1_nodrag", "G2_drag1.0", "G3_drag0.5", "G4_drag1.5"]:
        drag_ledger = AtmosphericDrag(area_to_mass=A_M, Cd=2.2, rho0=1.225 * rho_scale[name],
                                      H=8500.0, h0=0.0, R_earth=R_E)
        f = h5py.File(outdir / f"AS_{name}.h5", "r")
        t = f["epochs"][:]
        st = f["states"][:]
        f.close()
        # 第一周期窗口
        i1 = int(np.searchsorted(t, T_SC_DAYS * 86400.0))
        tw, sw = t[:i1 + 1], st[:i1 + 1]

        def parts(tt, s, _name=name, _drag=drag_ledger):
            out = {"j2j4": geo.compute_accel(s, tt), "moon": third.compute_accel(s, tt)}
            if _name != "G1_nodrag":
                out["drag"] = _drag.compute_accel(s, tt)
            return out

        out = ledger_closure(tw, sw, parts, mu=MU_E)
        ledgers[name] = {k: out[k] for k in ("dE", "work", "work_total", "residual", "residual_rel")}
        print(f"[{name}] ΔE={out['dE']:+.4e} J/kg; W_moon={out['work'].get('moon', 0):+.4e}; "
              f"W_drag={out['work'].get('drag', 0):+.4e}; W_J2J4={out['work'].get('j2j4', 0):+.4e}; "
              f"残差(rel)={out['residual_rel']:.1e}")

    summary = {"state_nd": STATE_ND, "days": args.days, "summaries": summaries,
               "ledger_cycle1": ledgers}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print("\n✅ 完成:", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
