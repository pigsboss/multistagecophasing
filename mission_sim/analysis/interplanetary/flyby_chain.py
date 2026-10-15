# mission_sim/analysis/interplanetary/flyby_chain.py
# -*- coding: utf-8 -*-
"""顺访链（谷神星 ＋ 主带小天体飞掠）连续小推力设计（Ceres E1 能力沉淀件）。

来源：谷神星探测专题 E1（``工具/s4_lt_variants.py``）。链型、候选表与窗口敏感口径不变。

口径（与 ``sf_design`` 完全一致）
---------------------------------
* 链型＝两条腿：起点 → 中间天体（**飞越**，只匹配位置）→ 终点（**交会**：去程；或
  **再入**：返程）。
* 时间线沿用 SR-1 参考场景（发射 2030-12-15／抵达 2037-06-17；离场 2038-10-30／
  再入 2043-07-15）。**飞越日期自由**：链内做脉冲（Lambert）预扫描＋连续推力精算。
* 候选：具名目标（``target_scan.TARGETS``）＋巡访池修正口径可行件
  （``low_thrust.POOL_TARGETS``）。
* 窗口敏感：发射（去程）／离场（返程）±30 d、C3 ±5%。
* 与专题岛既有 SF 链 v2′（``E6_N10_SF多起点.md`` §三-6）**口径差异**：v2′ 为
  "谷神星出发 → A 飞越 → B 飞越"（mode B、a＝常数）；本件为"地球注入（C3 给定）→
  A 飞越 → 谷神星交会"（C3 由运载给、a＝功率-推力模型）——**两者不可直接比数**。

用法::

  python3 -m mission_sim.analysis.interplanetary.flyby_chain [--quick]
          [--keys V1_out_lutetia] [--nseg1 20] [--nseg2 20] [--out-prefix 拓展_SF_顺访]
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mission_sim.analysis.interplanetary import low_thrust as LT
from mission_sim.analysis.interplanetary import sf_design as D

OUT_PREFIX = "拓展_SF_顺访"

# 变体定义：kind='out'＝去程顺访（地球→A→谷神星）；kind='ret'＝返程顺访（谷神星→A→地球）
VARIANTS: List[Dict[str, str]] = [
    dict(
        key="V1_out_lutetia",
        kind="out",
        mid="lutetia",
        mid_name="(21) Lutetia",
        note="去程顺访：Lutetia a=2.434 AU（介于地球与谷神星之间）",
    ),
    dict(
        key="V2_out_vesta",
        kind="out",
        mid="vesta",
        mid_name="(4) Vesta",
        note="去程顺访：Vesta a=2.361 AU",
    ),
    dict(
        key="V3_ret_vesta",
        kind="ret",
        mid="vesta",
        mid_name="(4) Vesta",
        note="返程顺访：Ceres 离场 → Vesta 飞越 → 地球再入",
    ),
    dict(
        key="V4_out_venetia487",
        kind="out",
        mid="venetia487",
        mid_name="(487) Venetia",
        note="去程顺访：巡访池可行件（矢量代理飞越真解 0.19 km/s）",
    ),
    dict(
        key="V5_ret_may348",
        kind="ret",
        mid="may348",
        mid_name="(348) May",
        note="返程顺访：巡访池可行件（真解飞越 0.70 km/s）",
    ),
]

SR1 = dict(D.SR1_DATES)


def _t(key: str) -> float:
    """``SR1`` 中的 ISO 日期键 → 相对 J2000 偏日数 [d]。"""
    return LT.jd_from_iso(SR1[key]) - LT.JD0


def make_prob(
    kind: str,
    mid: str,
    t_fly: float,
    nseg1: int,
    nseg2: int,
    model: Any,
    m0: float = D.M0_KG,
    c3: float = D.SR1_C3,
    t0: Optional[float] = None,
    t2: Optional[float] = None,
) -> D.Prob:
    """建链问题。``kind='out'``：earth→mid→ceres；``kind='ret'``：ceres→mid→earth。"""
    if kind == "out":
        bodies = ["earth", mid, "ceres"]
        kinds = ["depart_earth", "flyby", "rendezvous"]
        t0 = _t("launch") if t0 is None else t0
        t2 = _t("arrive") if t2 is None else t2
    else:
        bodies = ["ceres", mid, "earth"]
        kinds = ["start_body", "flyby", "arrive_earth"]
        t0 = _t("depart") if t0 is None else t0
        t2 = _t("back") if t2 is None else t2
    return D.Prob(
        [t0, t_fly, t2],
        [nseg1, nseg2],
        bodies,
        kinds,
        model,
        m0=m0,
        c3=c3 if kind == "out" else None,
        vinf_max=(6.0 if kind == "ret" else None),
        name="%s（顺访 %s）" % (kind, mid),
    )


def scan_flyby(
    kind: str,
    mid: str,
    model: Any,
    t0: Optional[float] = None,
    t2: Optional[float] = None,
    n_grid: int = 25,
    nscan: int = 250,
) -> Any:
    """脉冲（Lambert ≤1 圈）预扫描飞越日期：返回 (按参考 Δv 升序的候选日, (t0, t2))。"""
    if kind == "out":
        t0 = _t("launch") if t0 is None else t0
        t2 = _t("arrive") if t2 is None else t2
    else:
        t0 = _t("depart") if t0 is None else t0
        t2 = _t("back") if t2 is None else t2
    span = t2 - t0
    grid = np.linspace(t0 + 0.25 * span, t0 + 0.75 * span, n_grid)
    rows: List[Dict[str, Any]] = []
    for tf in grid:
        p = make_prob(kind, mid, float(tf), 6, 6, model, t0=t0, t2=t2)
        pb = D.pulse_bound(p, nmax=1, nscan=nscan)
        if pb["ok"]:
            rows.append(
                dict(
                    t_fly=float(tf),
                    date=LT.ymd(float(tf)),
                    dv=pb["total"],
                    parts=pb["parts"],
                )
            )
    rows.sort(key=lambda r: r["dv"])
    return rows, (t0, t2)


def solve_variant(
    v: Dict[str, str],
    model: Any,
    quick: bool = False,
    nseg1: Optional[int] = None,
    nseg2: Optional[int] = None,
    seed: int = D.SEED_DEFAULT,
    windows: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """求解单条顺访变体：飞越日预扫描 → SF 主解 → 窗口敏感。"""
    n1 = nseg1 or (14 if quick else 20)
    n2 = nseg2 or (14 if quick else 20)
    t0_default, t2_default = (
        (_t("launch"), _t("arrive"))
        if v["kind"] == "out"
        else (_t("depart"), _t("back"))
    )
    out: Dict[str, Any] = dict(
        key=v["key"],
        kind=v["kind"],
        mid=v["mid"],
        mid_name=v["mid_name"],
        note=v["note"],
        model=model.as_dict(),
        nseg=[n1, n2],
        seed=seed,
        timeline=dict(SR1),
        mid_elements=LT.BODIES[v["mid"]],
    )
    if verbose:
        print("\n" + "=" * 92)
        print("■ %s — %s" % (v["key"], v["note"]))
        print("=" * 92)
    rows, _ = scan_flyby(v["kind"], v["mid"], model)
    if verbose:
        print("  脉冲预扫描飞越日（前 5）：")
        for r in rows[:5]:
            print(
                "    %s  ΣΔv(脉冲参照)=%7.3f  [leg1 dep %.3f arr %.3f]"
                % (r["date"], r["dv"], r["parts"][0]["dep"], r["parts"][1]["arr"])
            )
    out["flyby_scan"] = rows[:10]
    if not rows:
        out["status"] = "脉冲预扫描无解"
        return out
    tf_best = rows[0]["t_fly"]
    out["flyby_date"] = rows[0]["date"]

    p = make_prob(v["kind"], v["mid"], tf_best, n1, n2, model)
    pb = D.pulse_bound(p)
    out["pulse_bound_fixed"] = pb
    r = D.sf_solve(p, maxiter=80, multistart=2, seed=seed, verbose=verbose)
    out["sf"] = r
    if verbose:
        print(
            "  ★ SF 主解：ΣΔv=%.3f km/s  残差 %.2e km  触界 %d  用时 %.0f s"
            % (r["total"], r["resid_km"], r["counters"]["n_binding"], r["seconds"])
        )
        for e in r["counters"]["enc"]:
            print(
                "    %s @%s：r 失配 %.2f km，相对速度 v∞=%.3f km/s"
                % (e["body"], e["date"], e["r_miss_km"], e["vinf_kms"])
            )

    if windows:
        sens: Dict[str, List[Dict[str, Any]]] = dict(launch_pm30d=[], c3_pm5pct=[])
        for d in (-30.0, 30.0):
            p2 = make_prob(
                v["kind"],
                v["mid"],
                tf_best + d,
                n1,
                n2,
                model,
                t0=t0_default + d,
                t2=t2_default + d,
            )
            r2 = D.sf_solve(p2, maxiter=60, multistart=1, seed=seed)
            sens["launch_pm30d"].append(
                dict(
                    shift_d=d,
                    total=r2["total"],
                    resid_km=r2["resid_km"],
                    vinf=[e["vinf_kms"] for e in r2["counters"]["enc"]],
                )
            )
            if verbose:
                print(
                    "    敏感 %+5.0f d：ΣΔv=%.3f（残差 %.1e km）"
                    % (d, r2["total"], r2["resid_km"])
                )
        for f in (0.95, 1.05):
            p3 = (
                make_prob(v["kind"], v["mid"], tf_best, n1, n2, model, c3=D.SR1_C3 * f)
                if v["kind"] == "out"
                else make_prob(v["kind"], v["mid"], tf_best, n1, n2, model)
            )
            r3 = D.sf_solve(p3, maxiter=60, multistart=1, seed=seed)
            sens["c3_pm5pct"].append(
                dict(
                    c3=D.SR1_C3 * f if v["kind"] == "out" else None,
                    total=r3["total"],
                    resid_km=r3["resid_km"],
                    vinf=[e["vinf_kms"] for e in r3["counters"]["enc"]],
                )
            )
            if verbose and v["kind"] == "out":
                print(
                    "    敏感 C3×%.2f：ΣΔv=%.3f（残差 %.1e km）"
                    % (f, r3["total"], r3["resid_km"])
                )
        out["sensitivity"] = sens
    out["status"] = "ok" if r["resid_km"] < 50.0 else "残差偏大（%s）" % r["resid_km"]
    return out


def run_variants(
    quick: bool = False,
    model: Optional[Any] = None,
    seed: int = D.SEED_DEFAULT,
    keys: Optional[Sequence[str]] = None,
    out_prefix: str = OUT_PREFIX,
) -> List[Dict[str, Any]]:
    """逐条求解 ``VARIANTS``（可用 ``keys`` 过滤），并把每条与汇总落盘。"""
    model = model or D.sep_model()
    t_start = time.time()
    outs: List[Dict[str, Any]] = []
    for v in VARIANTS:
        if keys and v["key"] not in keys:
            continue
        o = solve_variant(v, model, quick=quick, seed=seed)
        outs.append(o)
        LT.dump_json("%s_%s.json" % (out_prefix, v["key"]), o)
    LT.dump_json(
        "%s_汇总.json" % out_prefix,
        dict(
            model=model.as_dict(),
            seed=seed,
            variants=outs,
            seconds=time.time() - t_start,
            note="顺访变体（SF 直接法）；见专题岛 拓展研究/02_连续小推力轨迹设计_仿真.md",
        ),
    )
    print("\n合计 %.0f s" % (time.time() - t_start))
    return outs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="顺访链（谷神星＋主带小天体飞掠）SEP 设计")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keys", default=None, help="逗号分隔的变体键（默认全部）")
    ap.add_argument("--nseg1", type=int, default=None)
    ap.add_argument("--nseg2", type=int, default=None)
    ap.add_argument("--kdeg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=D.SEED_DEFAULT)
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    run_variants(
        quick=a.quick,
        model=D.sep_model(k_deg=a.kdeg),
        seed=a.seed,
        keys=a.keys.split(",") if a.keys else None,
        out_prefix=a.out_prefix,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
