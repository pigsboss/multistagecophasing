# mission_sim/analysis/interplanetary/target_scan.py
# -*- coding: utf-8 -*-
"""多目标（主带天体）到达窗口扫描器（Ceres S4 能力沉淀件）。

来源：谷神星探测专题 S4/CML5 仿真线的 ``工具/s4_target_scan.py``。把单目标
发射—TOF 网格扫描泛化到一组主带天体候选，用**同一张网格**给出可对标的逐年最优与
全局最优 C3 / v∞，供"哪几个目标在同一运载包络内可行、代价几何"的量级判断。

口径（与源件完全一致）
----------------------
* 星历【近似】：二体平根数传播（地球沿用 J2000 平均根数；目标取 JPL SBDB 快照）。
* 腿模型【算】：日心二体 Lambert（通用变量、单圈、脉冲）；无借力 / 无 DSM / 无有限推力。
* 目标根数【文据】：JPL SBDB ``sbdb.api?...&full-prec=true`` 快照，直连日期 2026-10-15，
  epoch = JD 2461200.5。含摄动的真实任务包络另计，两者不可直接对标。

用法
----
  python3 -m mission_sim.analysis.interplanetary.target_scan --self-test
  python3 -m mission_sim.analysis.interplanetary.target_scan --targets ceres,vesta \\
          --t0 2030-01-01 --t1 2045-12-31 --step 15 \\
          --tof-min 600 --tof-max 1600 --tof-step 10 --top-per-year 3 \\
          --txt scan.txt --json scan.json
  # 直连 SBDB 刷新根数（仅打印，不改变内置快照）
  python3 -m mission_sim.analysis.interplanetary.target_scan --fetch
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mission_sim.analysis.interplanetary.two_body import (
    EARTH_J2000,
    SEC_PER_DAY,
    Ephemeris,
    direct_leg,
    hohmann_transfer,
    iso_from_jd,
    iter_julian_days,
    jd_from_iso,
    lambert_on_180,
    tof_grid,
)

# ------------------------------------------------------------------ 目标目录
# 【文据】JPL SBDB full-prec 快照，直连日期 2026-10-15，epoch = JD 2461200.5。
SBDB_SNAPSHOT_EPOCH = 2461200.5
TARGETS: Dict[str, Dict[str, Any]] = {
    "ceres": {
        "name_zh": "谷神星 (1) Ceres（基线）",
        "a_au": 2.765552595034094,
        "e": 0.07969229514816586,
        "i_deg": 10.58802780183462,
        "om_deg": 80.24862682043221,
        "w_deg": 73.29421453021587,
        "ma_deg": 274.4193463761342,
        "n_deg_day": 0.21430445064843,
    },
    "vesta": {
        "name_zh": "灶神星 (4) Vesta",
        "a_au": 2.361365965127599,
        "e": 0.09020374382834395,
        "i_deg": 7.143925545058711,
        "om_deg": 103.701293265032,
        "w_deg": 151.4686478221564,
        "ma_deg": 81.19015607686903,
        "n_deg_day": 0.2716183613599909,
    },
    "pallas": {
        "name_zh": "智神星 (2) Pallas",
        "a_au": 2.769559010737709,
        "e": 0.2307000995648547,
        "i_deg": 34.93279321851542,
        "om_deg": 172.8866193357694,
        "w_deg": 310.9699161652136,
        "ma_deg": 254.2496521742734,
        "n_deg_day": 0.2138396029251949,
    },
    "hygiea": {
        "name_zh": "健神星 (10) Hygiea",
        "a_au": 3.150974033963701,
        "e": 0.1067092741240963,
        "i_deg": 3.829529946447122,
        "om_deg": 283.1198927508594,
        "w_deg": 312.4242387344704,
        "ma_deg": 252.0344242359649,
        "n_deg_day": 0.1762125505792448,
    },
    "psyche": {
        "name_zh": "灵神星 (16) Psyche",
        "a_au": 2.925720466462538,
        "e": 0.1349324738201893,
        "i_deg": 3.098749116151128,
        "om_deg": 149.9753859305033,
        "w_deg": 230.0326782748359,
        "ma_deg": 79.76939505329617,
        "n_deg_day": 0.1969494758105845,
    },
    "lutetia": {
        "name_zh": "司琴星 (21) Lutetia",
        "a_au": 2.434430742712361,
        "e": 0.1647703512177245,
        "i_deg": 3.064452891064415,
        "om_deg": 80.83855721319993,
        "w_deg": 249.8802780888828,
        "ma_deg": 298.7726414605465,
        "n_deg_day": 0.2594824191639628,
    },
}
DEFAULT_ORDER: List[str] = ["ceres", "vesta", "pallas", "hygiea", "psyche", "lutetia"]
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={sstr}&full-prec=true"
SSTR: Dict[str, str] = {
    "ceres": "ceres",
    "vesta": "vesta",
    "pallas": "pallas",
    "hygiea": "hygiea",
    "psyche": "psyche",
    "lutetia": "lutetia",
}


def spec_of(key: str) -> Dict[str, Any]:
    """目标键 → 二体平根数规格（含快照历元）。【文据】"""
    t = TARGETS[key]
    return {
        "name": t["name_zh"],
        "a_au": t["a_au"],
        "e": t["e"],
        "i_deg": t["i_deg"],
        "om_deg": t["om_deg"],
        "w_deg": t["w_deg"],
        "ma_deg": t["ma_deg"],
        "n_deg_day": t["n_deg_day"],
        "epoch_jd": SBDB_SNAPSHOT_EPOCH,
    }


def fetch_sbdb(key: str) -> Dict[str, Any]:
    """直连 JPL SBDB 刷新某目标瞬时根数（full-prec）。失败抛异常。【文据】

    仅作为快照的在线复核通道；默认流程不调用（离线可复算）。
    """
    req = urllib.request.Request(
        SBDB_URL.format(sstr=SSTR[key]),
        headers={"User-Agent": "mcpc-target_scan/1.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    els = {e["name"]: e["value"] for e in data["orbit"]["elements"]}
    return {
        "name": f"{TARGETS[key]['name_zh']}(SBDB 直连)",
        "a_au": float(els["a"]),
        "e": float(els["e"]),
        "i_deg": float(els["i"]),
        "om_deg": float(els["om"]),
        "w_deg": float(els["w"]),
        "ma_deg": float(els["ma"]),
        "n_deg_day": float(els["n"]),
        "epoch_jd": float(data["orbit"]["epoch"]),
    }


# ------------------------------------------------------------------ 扫描
def scan_target(
    key: str,
    t0: float,
    t1: float,
    step_d: float,
    tofs: Sequence[float],
    top_per_year: int,
    earth: Optional[Ephemeris] = None,
    spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """单目标发射—TOF 网格扫描：返回逐年前 top_per_year（按 C3）与全局最优。【算】

    Args:
        key: 目标键（见 ``TARGETS``）。
        t0, t1: 发射日儒略日区间（含两端）。
        step_d: 发射日步长 [d]。
        tofs: 飞行时间列表 [d]。
        top_per_year: 每年保留的条数（按 C3 升序）。
        earth: 地球星历（复用可省去重复构造）。
        spec: 目标根数规格；None 时取内置快照。

    Returns:
        dict(key, name_zh, grid_points, global_best, per_year_top, n_years)。
    """
    earth = earth or Ephemeris(EARTH_J2000)
    spec = spec or spec_of(key)
    obj = Ephemeris(spec)
    per_year: Dict[int, List[Dict[str, float]]] = {}
    npts, best = 0, None
    for lj in iter_julian_days(t0, t1, step_d):
        for tof in tofs:
            try:
                row = direct_leg(lj, tof, earth, obj)
            except ValueError:
                continue
            npts += 1
            per_year.setdefault(int(iso_from_jd(lj)[:4]), []).append(row)
            if best is None or row["c3"] < best["c3"]:
                best = row
    years = sorted(per_year)
    top = {y: sorted(per_year[y], key=lambda r: r["c3"])[:top_per_year] for y in years}
    return {
        "key": key,
        "name_zh": spec["name"],
        "grid_points": npts,
        "global_best": best,
        "per_year_top": top,
        "n_years": len(years),
    }


def hohmann_lower_bound(a_au: float) -> Dict[str, float]:
    """地球↔目标 共面圆轨道 Hohmann 解析下界（对照用）。【算】"""
    h = hohmann_transfer(1.0, a_au)
    return {
        "a_au": a_au,
        "vinf_dep": h["vinf_dep"],
        "c3": h["c3"],
        "vinf_arr": h["vinf_arr"],
        "tof_d": h["tof_d"],
    }


# ------------------------------------------------------------------ 自检
def run_self_test() -> int:
    """逐目标 Hohmann 解析 + Δν=180° 拱点分支 Lambert 复现。【算】

    判据：Lambert 复现的 C3 与解析下界误差 < 0.01%。
    """
    print("=" * 92)
    print("target_scan · self-test（逐目标 Hohmann 解析对照 + Δν=180° 拱点分支复现）")
    print("=" * 92)
    print(
        f"{'目标':<22}{'a(AU)':>9}{'C3解析':>10}{'v∞dep解析':>11}{'v∞arr解析':>11}"
        f"{'TOF(d)':>9}{'C3复现':>10}{'误差%':>8}"
    )
    ok = True
    for key in DEFAULT_ORDER:
        spec = spec_of(key)
        h = hohmann_lower_bound(spec["a_au"])
        _vinf_dep, c3, _vinf_arr = lambert_on_180(
            1.0, spec["a_au"], h["tof_d"] * SEC_PER_DAY, 0.0
        )
        err = (c3 - h["c3"]) / h["c3"] * 100.0
        ok = ok and abs(err) < 0.01
        print(
            f"{spec['name'][:20]:<22}{spec['a_au']:9.4f}{h['c3']:10.2f}{h['vinf_dep']:11.3f}"
            f"{h['vinf_arr']:11.3f}{h['tof_d']:9.1f}{c3:10.3f}{err:+8.4f}"
        )
    print("-" * 92)
    print("判据：复现 C3 与解析误差 <0.01% ⇒", "通过" if ok else "**未通过**")
    print(
        "说明：Pallas i=34.9°、e=0.231 ⇒ 共面圆轨道 Hohmann 仅作**下界**参照；真实直飞 C3 必高于此。"
    )
    return 0 if ok else 1


# ------------------------------------------------------------------ 输出
def _row_str(r: Mapping[str, float]) -> str:
    return (
        f"{iso_from_jd(r['launch_jd'])}  {iso_from_jd(r['arrival_jd'])}  "
        f"{r['tof_d']:7.1f}  {r['c3']:9.2f}  {r['vinf_dep']:8.3f}  {r['vinf_arr']:9.3f}"
    )


def render_txt(
    results: Sequence[Mapping[str, Any]], params: Mapping[str, Any], generated: str
) -> str:
    """扫描结果 → 制表文本（与源件 s4_target_scan 的版式一致）。"""
    L: List[str] = []
    L.append("=" * 100)
    L.append("S4/CML5 仿真线 · 多目标（主带天体）到达窗口快速扫描（论证级）【算】")
    L.append("=" * 100)
    L.append(
        f"生成：{generated}（系统 date）｜工具：mission_sim/analysis/interplanetary/"
        "target_scan.py（复用 two_body.py）"
    )
    L.append(
        "模型：二体平根数星历 + 日心二体 Lambert（通用变量、单圈、脉冲）；无借力/DSM/有限推力。【近似】"
    )
    L.append(
        f"目标根数：JPL SBDB full-prec 快照 2026-10-15，epoch=JD {SBDB_SNAPSHOT_EPOCH}【文据】"
    )
    L.append(
        f"网格：发射 {params['t0']} → {params['t1']} @{params['step']:g} d；"
        f"TOF {params['tof_min']:g}–{params['tof_max']:g} d @{params['tof_step']:g} d"
    )
    L.append("-" * 100)
    L.append("【A】全局最优一览（按 C3 升序；C3 = 出发 v∞²）")
    L.append(
        f"{'目标':<22}{'最优发射':>12}{'最优到达':>12}{'TOF(d)':>8}{'C3':>9}"
        f"{'v∞dep':>9}{'v∞arr':>9}{'相对Ceres ΔC3':>14}"
    )
    base = next((r for r in results if r["key"] == "ceres"), None)
    base_c3 = base["global_best"]["c3"] if base else None
    for r in sorted(results, key=lambda z: z["global_best"]["c3"]):
        g = r["global_best"]
        d = "" if base_c3 is None else f"{g['c3'] - base_c3:+.2f}"
        L.append(
            f"{r['name_zh'][:20]:<22}{iso_from_jd(g['launch_jd']):>12}"
            f"{iso_from_jd(g['arrival_jd']):>12}{g['tof_d']:8.1f}{g['c3']:9.2f}"
            f"{g['vinf_dep']:9.3f}{g['vinf_arr']:9.3f}{d:>14}"
        )
    L.append("")
    L.append("【B】Hohmann 解析下界（共面圆轨道，仅作参照）【算】")
    L.append(
        f"{'目标':<22}{'a(AU)':>9}{'C3下界':>10}{'v∞dep':>9}{'v∞arr':>9}{'TOF(d)':>9}"
    )
    for r in results:
        h = hohmann_lower_bound(TARGETS[r["key"]]["a_au"])
        L.append(
            f"{r['name_zh'][:20]:<22}{h['a_au']:9.4f}{h['c3']:10.2f}{h['vinf_dep']:9.3f}"
            f"{h['vinf_arr']:9.3f}{h['tof_d']:9.1f}"
        )
    L.append("")
    for r in results:
        L.append("-" * 100)
        L.append(
            f"【{r['key']}】{r['name_zh']}｜网格点 {r['grid_points']}｜年数 {r['n_years']}｜"
            f"全局最优发射 {iso_from_jd(r['global_best']['launch_jd'])}"
            f"（C3 {r['global_best']['c3']:.2f}）"
        )
        L.append(
            "     发射日       到达日        TOF(天)   C3(km²/s²)  出发v∞(km/s)  抵达v∞(km/s)"
        )
        for y in sorted(r["per_year_top"]):
            L.append(f"  —— {y} 年 ——")
            for row in r["per_year_top"][y]:
                L.append("     " + _row_str(row))
    L.append("-" * 100)
    L.append(
        "口径提醒：本表为**直飞、无借力**的二体量级；文献中借力＋DSM 方案会显著更低，"
    )
    L.append(
        "         两口径不可直接对标。Pallas 高倾角（i=34.9°）与高偏心（e=0.231）是其 C3 偏高的主因【算】。"
    )
    return "\n".join(L) + "\n"


def build_meta(
    params: Mapping[str, Any], results: Sequence[Mapping[str, Any]], generated: str
) -> Dict[str, Any]:
    """扫描结果 → JSON 载荷（结构与源件一致）。"""
    return {
        "tool": "target_scan.py",
        "generated": generated,
        "model": "二体平根数星历 + 日心二体 Lambert（通用变量、单圈、脉冲）；无借力/DSM/有限推力",
        "elements_source": "JPL SBDB full-prec 快照 2026-10-15, epoch=JD 2461200.5",
        "params": dict(params),
        "targets": {
            r["key"]: {
                "name_zh": r["name_zh"],
                "grid_points": r["grid_points"],
                "global_best": r["global_best"],
                "per_year_top": {str(y): v for y, v in r["per_year_top"].items()},
                "hohmann_lower_bound": hohmann_lower_bound(TARGETS[r["key"]]["a_au"]),
            }
            for r in results
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="多目标（主带天体）到达窗口快速扫描（论证级；二体 Lambert）"
    )
    p.add_argument(
        "--targets",
        default=",".join(DEFAULT_ORDER),
        help="逗号分隔的目标键（默认全部）",
    )
    p.add_argument("--t0", default="2030-01-01")
    p.add_argument("--t1", default="2045-12-31")
    p.add_argument("--step", type=float, default=15.0, help="发射日步长（天）")
    p.add_argument("--tof-min", type=float, default=600.0)
    p.add_argument("--tof-max", type=float, default=1600.0)
    p.add_argument("--tof-step", type=float, default=10.0)
    p.add_argument("--top-per-year", type=int, default=3)
    p.add_argument("--txt", default="", help="制表文本输出路径（空＝不写）")
    p.add_argument("--json", default="", help="JSON 输出路径（空＝不写）")
    p.add_argument(
        "--fetch", action="store_true", help="直连 SBDB 刷新根数并打印（不改内置表）"
    )
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()

    if args.self_test:
        return run_self_test()

    keys = [k.strip() for k in args.targets.split(",") if k.strip()]
    for k in keys:
        if k not in TARGETS:
            print(f"未知目标：{k}（可选：{', '.join(DEFAULT_ORDER)}）")
            return 2

    if args.fetch:
        for k in keys:
            try:
                print("SBDB 直连：", json.dumps(fetch_sbdb(k), ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {k}: 直连失败（{exc.__class__.__name__}: {exc}）——保留内置快照"
                )

    t0, t1 = jd_from_iso(args.t0), jd_from_iso(args.t1)
    tofs = tof_grid(args.tof_min, args.tof_max, args.tof_step)
    earth = Ephemeris(EARTH_J2000)

    results = []
    for k in keys:
        r = scan_target(k, t0, t1, args.step, tofs, args.top_per_year, earth=earth)
        results.append(r)
        g = r["global_best"]
        print(
            f"[{k}] 网格 {r['grid_points']}｜最优 {iso_from_jd(g['launch_jd'])} → "
            f"{iso_from_jd(g['arrival_jd'])}｜TOF {g['tof_d']:.0f} d｜C3 {g['c3']:.2f}｜"
            f"v∞dep {g['vinf_dep']:.3f}｜v∞arr {g['vinf_arr']:.3f}"
        )

    params = {
        "targets": args.targets,
        "t0": args.t0,
        "t1": args.t1,
        "step": args.step,
        "tof_min": args.tof_min,
        "tof_max": args.tof_max,
        "tof_step": args.tof_step,
        "top_per_year": args.top_per_year,
        "fetch": args.fetch,
    }
    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as fh:
            fh.write(render_txt(results, params, generated))
        print(f"TXT 已写入：{args.txt}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                build_meta(params, results, generated), fh, ensure_ascii=False, indent=2
            )
        print(f"JSON 已写入：{args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
