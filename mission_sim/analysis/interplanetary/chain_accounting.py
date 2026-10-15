# mission_sim/analysis/interplanetary/chain_accounting.py
# -*- coding: utf-8 -*-
"""全链对账（Ceres R6-3 能力沉淀件）。

来源：谷神星探测专题 自由探索 R6（``工具/s4_lt4_r6.py`` v0.1 的 ``--chain`` 段）。算法与
报数口径不变（对拍见 ``tests/test_interplanetary_certificate_closure.py``：与岛内机读结果
``任务/拓展_R6_全链对账.json`` 逐位一致）。

本件做一件事
------------
:func:`chain_reconcile`：把 ``去程(火星借力)＋停留＋返程(2099.4 d)`` 的 Δv 分解与 003 的
≈14 km/s 往返总对账，给逐环归因表与时间线修正建议表（A：延后 50–65 d；B：2099.4 d 版）。

口径
----
* 输入＝:mod:`flyby_mars` 的 ``opt1``／``opt2`` 结果 dict（**不重跑**任何轨迹）；
* 推进剂账＝齐奥尔科夫斯基（Xe 等效，Isp＝4 000 s，m0＝4 669 kg）；
* 003 参考＝Castillo-Rogez et al. 2022（PSJ 3:41）Figure 10 图内日期链与 "round-trip ≈14 km/s"。

与源件差异
----------
① 依赖内联：岛内 ``s4_lt_common``／``s4_lt_design`` → 本仓 ``low_thrust``／``sf_design``；
   003 常数与 :func:`flyby_mars.prop_kg` / ``F003`` 复用 :mod:`flyby_mars`（同一份）。
② 不再向岛内 ``任务/拓展_R6_全链对账.json`` 落盘：只返回结果 dict（差异同前批各件）。
③ ``prop_kg`` 从源件的模块级函数**原样搬运**到本件（源件在 R6 模块内定义）。

用法::

  python3 -m mission_sim.analysis.interplanetary.chain_accounting --chain \
      --opt1-json <R6-1 结果.json> --opt2-json <R6-2 结果.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional, Sequence

from mission_sim.analysis.interplanetary import low_thrust as LT  # noqa: E402
from mission_sim.analysis.interplanetary import sf_design as D  # noqa: E402
from mission_sim.analysis.interplanetary.flyby_mars import (  # noqa: E402
    DV_003_ROUNDTRIP,
    DV_NOFLYBY_KMS,
    F003,
    T_DEPART,
    XE_BUDGET_KG,
)

M0 = D.M0_KG
ISP = D.ISP_S


def prop_kg(dv_kms: float, m0: float = M0, isp: float = ISP) -> float:
    """齐奥尔科夫斯基推进剂质量（Xe 等效）[kg]。"""
    return m0 * (1.0 - math.exp(-dv_kms / (isp * LT.G0 / 1000.0)))


def chain_reconcile(
    opt1_res: Dict[str, Any], opt2_res: Dict[str, Any], verbose: bool = True
) -> Dict[str, Any]:
    """R6-3：`去程(借力)＋停留500d＋返程(2099.4d)` 的 Δv 分解 vs 003 ≈14；时间线修正建议。"""
    print("=" * 96)
    print("R6-3 全链对账：本岛 `去程(借力)+500d+返程(2099.4d)` vs 003 ≈14 km/s")
    out: Dict[str, Any] = dict(
        kind="r6_chain_reconcile",
        dv_003_roundtrip=DV_003_ROUNDTRIP,
        dv_no_flyby_ref_kms=DV_NOFLYBY_KMS,
        xe_budget_kg=XE_BUDGET_KG,
        stay_d=F003["stay_d"],
        return_tof_d=F003["tof_ret"],
    )
    # 去程（有借力）——取收敛且自由飞掠可行、推进 Δv 最小者
    cands = [c for c in opt1_res["chains"] if c["resid_km"] < 100.0]
    best_out = min(cands, key=lambda c: c["dv_propulsive_kms"]) if cands else None
    out["outbound"] = best_out
    sf = opt2_res.get("sf_checks", [])
    ret = min(sf, key=lambda s: s["resid_km"]) if sf else None
    out["return"] = ret
    if best_out and ret:
        dv_out = best_out["dv_propulsive_kms"]
        dv_ret = ret["dv_total_kms"]
        tot_sf = dv_out + dv_ret
        out["reconcile"] = dict(
            dv_out_kms=dv_out,
            dv_out_noflyby_kms=DV_NOFLYBY_KMS,
            dv_ret_kms=dv_ret,
            dv_sum_sf_kms=tot_sf,
            dv_003_kms=DV_003_ROUNDTRIP,
            ratio_vs_003=tot_sf / DV_003_ROUNDTRIP,
            diff_kms=tot_sf - DV_003_ROUNDTRIP,
            prop_out_kg=prop_kg(dv_out),
            prop_ret_kg=prop_kg(dv_ret),
            prop_sum_kg=prop_kg(tot_sf),
            prop_003_14_kg=prop_kg(14.0),
            xe_budget_kg=XE_BUDGET_KG,
            dv_at_xe_budget_kms=LT.dv_from_prop(M0, XE_BUDGET_KG, ISP),
        )
        r = out["reconcile"]
        print(
            "\n  【本岛】去程(借力) %.3f ＋ 返程(2099.4 d) %.3f = **%.3f km/s**"
            % (dv_out, dv_ret, tot_sf)
        )
        print(
            "  【003】往返总 %.1f km/s ⇒ 本岛为 003 的 **%.2f×**（差 %+.1f km/s）"
            % (DV_003_ROUNDTRIP, r["ratio_vs_003"], r["diff_kms"])
        )
        print(
            "  推进剂（Xe 等效，Isp 4000 s，m0 %.0f kg）：本岛 %.0f kg vs 预算 %.0f kg；"
            "003 的 14.0 本身即需 %.0f kg"
            % (M0, r["prop_sum_kg"], XE_BUDGET_KG, r["prop_003_14_kg"])
        )
        print(
            "  Xe %.0f kg 预算 ⇔ Δv=%.3f km/s（本岛口径）"
            % (XE_BUDGET_KG, r["dv_at_xe_budget_kms"])
        )
    out["rings"] = [
        dict(
            ring="① 去程（地球→谷神星）",
            island="有借力 SF 推进 %.3f km/s（无借力参考 %.3f）"
            % (
                best_out["dv_propulsive_kms"] if best_out else float("nan"),
                DV_NOFLYBY_KMS,
            ),
            ref="003 未公开分段 Δv（Figure 10 只给日期/质量/v∞）",
            attribution="口径差＋本岛 SF 可行点非 Δv 最优＋Mars 星历近似【待核】",
        ),
        dict(
            ring="② 火星飞掠",
            island=(
                "自由飞掠可行（v∞ %.2f→%.2f，转角 %.1f°≤δ_max %.1f°）"
                % (
                    best_out["flyby"]["vinf_in_kms"],
                    best_out["flyby"]["vinf_out_kms"],
                    best_out["flyby"]["turn_deg"],
                    best_out["flyby"]["dmax_deg_alt300"],
                )
                if best_out
                else "n/a"
            ),
            ref="003: 2031-08-13, alt 300 km, v∞ 4.85",
            attribution="本岛链给出的 v∞ 与 003 量级一致；但**自由飞掠不省 SEP 的 Δv**——"
            "它只改向，模长守恒",
        ),
        dict(
            ring="③ 停留 500 d",
            island="%d d（003 图内 %d d）" % (F003["stay_d"], F003["stay_d"]),
            ref="500 day stay（图注）",
            attribution="一致",
        ),
        dict(
            ring="④ 返程（谷神星→地球）",
            island="SF %.3f km/s（残差 %.2e km；v∞ %.3f km/s）；证书 r_min(%.1f d)=%.4f AU"
            % (
                ret["dv_total_kms"] if ret else float("nan"),
                ret["resid_km"] if ret else float("nan"),
                ret["vinf_kms"] if ret else float("nan"),
                F003["tof_ret"],
                opt2_res["certificate"]["r_min_au"],
            ),
            ref="003: TOF 2099.4 d, v∞ 6.00",
            attribution="半径条件**余量大**（远低于 1.0165）；但 v∞=6.00 在本岛脉冲/守卫下不可达"
            "（见 R5-2：min 6.379）⇒ 差额主要在 v∞ 守卫与 SF 未优化",
        ),
    ]
    out["timeline_options"] = [
        dict(
            option="A｜延后 50–65 d",
            basis="R5 半径必要条件闭合阈值 T*≈1768–1784 d",
            reentry=LT.ymd(T_DEPART + 1768.0) + "…" + LT.ymd(T_DEPART + 1784.0),
            reentry_003_dep=LT.ymd(LT.dfe(2038, 9, 4) + 1768.0)
            + "…"
            + LT.ymd(LT.dfe(2038, 9, 4) + 1784.0),
            delta_d="+50…+65",
            note="改动最小；与 SR-1 其余时间线相容；与'4.7 yr'正文口径更近",
            dv_ref_sf_kms=26.29,
            dv_ref_spiral_kms=12.882,
            prop_sf_kg=prop_kg(26.29),
            prop_spiral_kg=prop_kg(12.882),
            dv_note="返程 SF 可行点 26.29 km/s（TOF 1784，非 Δv 最优）／切向螺旋 12.882 km/s"
            "（近最优上界）；推进剂分别 %.0f／%.0f kg（预算 %.0f）"
            % (prop_kg(26.29), prop_kg(12.882), XE_BUDGET_KG),
        ),
        dict(
            option="B｜2099.4 d 版（003 图口径）",
            basis="003 Figure 10 图内日期链",
            reentry=LT.ymd(T_DEPART + 2099.4),
            delta_d="+380.4",
            reentry_003_dep=LT.ymd(LT.dfe(2038, 9, 4) + 2099.4),
            note="与 003 图一致（003 离场日口径下再入 %s）；但本岛 v∞≤6.0 守卫不可达（min 6.379）"
            % LT.ymd(LT.dfe(2038, 9, 4) + 2099.4),
            dv_ref_sf_kms=(ret["dv_total_kms"] if ret else None),
            prop_sf_kg=(prop_kg(ret["dv_total_kms"]) if ret else None),
            dv_note="返程 SF 抽出解见 §R6-2；证书 r_min=%.4f AU（远低于 1.0165，余量 %.4f AU）"
            % (
                opt2_res["certificate"]["r_min_au"],
                opt2_res["certificate"]["margin_au"],
            ),
        ),
    ]
    print("\n  时间线修正建议：")
    for o in out["timeline_options"]:
        print("    %s：再入 %s（Δ %s）" % (o["option"], o["reentry"], o["delta_d"]))
    return out


# ================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="全链对账（Ceres R6-3）")
    ap.add_argument("--chain", action="store_true", help="执行全链对账")
    ap.add_argument(
        "--opt1-json", default=None, help="R6-1 结果 JSON（flyby_mars.opt1）"
    )
    ap.add_argument(
        "--opt2-json", default=None, help="R6-2 结果 JSON（flyby_mars.opt2）"
    )
    ap.add_argument("--out-json", default=None, help="结果 JSON 路径（空＝不写）")
    return ap


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.chain:
        o1 = _load(a.opt1_json)
        o2 = _load(a.opt2_json)
        if o1 and o2:
            out = chain_reconcile(o1, o2)
            if a.out_json:
                LT.dump_json(a.out_json, out)
        else:
            print("  缺 R6-1/R6-2 机读件，先跑 flyby_mars --opt1／--opt2")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
